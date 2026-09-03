r"""자(ruler) 배터리 — meta_splice_probe 라벨 위에서 여덟 후보 자의 판별력을 잰다.

무엇을 재나
──────────────────────────────────────────────────────────────────────────────
meta_splice_probe.py 가 만든 라벨(`labels.parquet` + `sites_shard*.parquet`)의
사이트마다, 메타 블록의 유무만 다른 두 문맥에서 후보 «자»들의 스칼라 점수를 계산해
good vs null 판별력(AUC), Δ_ctl 과의 순위상관, 사후보고 슬라이스 오염, 대소문자
불변성, 카나리아(길이·confidence) 초과 여부를 보고한다.

  ctx_close = 챗프롬프트토큰 + prefix(메타 포함) 토큰      = 챗프롬프트 + text[:meta_end]
  ctx_open  = 챗프롬프트토큰 + prefix 에서 메타만 뺀 토큰   = 챗프롬프트 + text[:meta_start]

  R1_pmi1tok  현행 재현. witness/op-swap 미끼를 \boxed{} 로 싸고
              countdown_pmi.divergent_spans 의 발산 슬라이스 토큰 logp 만으로
              (g−d)@close − (g−d)@open.
  R2_full     같은 g/d 로 «식 전체» 토큰당 평균 logp 의 (g−d) shift.
  R3_family   미끼 = witness 의 모든 */ 를 +− 로(없으면 그 반대로) 바꾼 다른 가족 식.
              eval ≠ target 확인, 같으면 (witness)+1 로 회피. 식 전체 shift.
  R4_osdgold  \boxed{witness} 토큰당 평균 logp 의 close − open.
  R5_osdself  원본 롤아웃에서 메타 뒤 최대 200토큰의 평균 logp close − open (현행 OSD 대용).
  R6_behavior 텍스트만. 메타 앞/뒤 300자에서 연산자·숫자쌍 multiset novelty.
  C_len       메타 프로즈 길이 (카나리아).
  C_conf      confidence 값 (카나리아).

logP 채점은 순수 vLLM(verl 금지)의 **prompt_logprobs** 로 한다. vLLM 0.10.2 실측
(2026-08-27, `$PY -c "import vllm, inspect..."`):
  · `SamplingParams(max_tokens=1, prompt_logprobs=0, ...)` 이 유효하고,
  · `LLM.generate` 는 `vllm.inputs.TokensPrompt(prompt_token_ids=[...])` 를 받으며,
  · `RequestOutput.prompt_logprobs` 는 프롬프트 토큰과 같은 길이의 리스트
    (0번째는 None), 각 원소는 `dict[token_id -> Logprob(logprob, rank, decoded_token)]`
    이고 prompt_logprobs=0 이어도 실제 토큰의 logprob 은 항상 들어 있다.
  문맥토큰+대상토큰을 **토큰 id 로 이어** 한 시퀀스로 넣으므로(문자열 재토큰화 없음)
  대상 구간 경계가 병합으로 표류하지 않는다.

학습과 같은 샘플링/템플릿 재현 — run_countdown_arm.sh 실측 (2026-08-27 grep):
  L179  actor_rollout_ref.rollout.temperature=1.0
  L181  actor_rollout_ref.rollout.top_p=1.0
  L185  ++data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-false}
  L112  ENABLE_THINKING="${ENABLE_THINKING:-false}"      → 기본 false
  L187  data.max_response_length=3072
chat template 은 scripts/countdown_gs0_eval.py 와 같은 방식:
apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
enable_thinking=False) + TypeError 폴백. (채점은 teacher-forcing 이라
temperature/top_p 는 결과에 안 들어가지만, 문맥을 만든 분포의 실측값을 기록으로 남긴다.)

입력 스키마 계약 (meta_splice_probe.py 산출물)
──────────────────────────────────────────────────────────────────────────────
labels.parquet ⋈ sites_shard*.parquet (site_id 로 병합; labels 가 전부 들고 있으면
shards 는 없어도 된다) 에서 아래 필드를 별칭 허용으로 찾는다. 못 찾으면 어떤 컬럼이
있는지 나열하며 **즉시 실패**한다(조용한 기본값 없음):
  site_id     [site_id|id|sid]              없으면 행 번호
  nums        [nums]                        필수
  target      [target]                      필수
  witness     [witness|gold]                필수 (_solves 로 재검증, 틀리면 skip+기록)
  text        [text|response|response_text|rollout_text|full_text]  필수 (원본 롤아웃 전체)
  label       [label|cls|class|site_class]  필수. good / null (대소문자 무시)
  delta_ctl   [delta_ctl|delta_ctl_B|d_ctl|dctl|delta]  ρ 용. 없으면 ρ=NaN 로 보고
              (labels.parquet 실물 컬럼명은 delta_ctl_B — stageB K=32 측정 Δ)
  post        [post|post_hoc|posthoc|is_post]  없으면 label=="post" 로 유도, 그마저 없으면 0
메타 스팬은 저장된 오프셋을 믿지 않고 라벨러와 같은 함수
`countdown_rewards.parse_meta(text, "new")` 로 **재파싱**한다 — 두 파일이 갈리면
여기서 갈린 개수가 `n_meta_reparse_fail` 로 드러난다.

통과선 (하드코딩)
──────────────────────────────────────────────────────────────────────────────
  AUC ≥ 0.65  ∧  사후보고 non-FAIL(post 평균 ≤ good 평균)
  ∧  카나리아 초과(AUC − max(C_len, C_conf AUC) > 0)  ∧  불변성 비율 < 0.5

⛔ 이 파일의 GPU 실행(vLLM 로드)은 `score` 서브커맨드가 호출될 때만 일어난다.
   `report` 서브커맨드는 저장된 점수 parquet 만으로 분석을 다시 돌린다(CPU).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.training.countdown_rewards import parse_meta                    # noqa: E402
from src.training.countdown_pmi import boxed, divergent_spans            # noqa: E402
from src.training.countdown_task import (                                # noqa: E402
    build_prompt, eval_countdown, swap_op_decoy,
)
# 검증기는 평가 스크립트의 것을 **import** 한다(복제 금지 — 갈리면 측정이 갈린다).
from scripts.countdown_gs0_eval import _solves                           # noqa: E402

# ── run_countdown_arm.sh 실측값 (위 헤더의 grep 근거) ─────────────────────────
TRAIN_SAMPLING = {
    "temperature": 1.0,          # L179
    "top_p": 1.0,                # L181
    "max_response_length": 3072, # L187
    "enable_thinking": False,    # L112/L185 기본 false
}

RULERS = ["R1_pmi1tok", "R2_full", "R3_family", "R4_osdgold",
          "R5_osdself", "R6_behavior", "C_len", "C_conf"]
CANARIES = ["C_len", "C_conf"]
FORWARD_RULERS = ["R1_pmi1tok", "R2_full", "R3_family", "R4_osdgold", "R5_osdself"]

# 통과선 — 하드코딩 (임무 명세)
PASS_AUC_MIN = 0.65
PASS_INV_MAX = 0.5

SELF_MAX_TOK = 200      # R5: 메타 뒤 최대 200토큰
BEHAV_WINDOW = 300      # R6: 메타 앞/뒤 300자
N_INVARIANCE = 50       # ④: good 50개
BOOTSTRAP_B = 1000


# ══════════════════════════════════════════════════════════════════════════════
# 1. 입력 로딩 — 별칭 해석 + 명시적 실패
# ══════════════════════════════════════════════════════════════════════════════

_ALIASES = {
    "site_id":   ["site_id", "id", "sid"],
    "nums":      ["nums"],
    "target":    ["target"],
    "witness":   ["witness", "gold"],
    "text":      ["text", "response", "response_text", "rollout_text", "full_text"],
    "label":     ["label", "cls", "class", "site_class"],
    "delta_ctl": ["delta_ctl", "delta_ctl_B", "d_ctl", "dctl", "delta"],
    "post":      ["post", "post_hoc", "posthoc", "is_post"],
}
_REQUIRED = ["nums", "target", "witness", "text", "label"]


def _resolve_columns(df) -> dict:
    got = {}
    for key, names in _ALIASES.items():
        for n in names:
            if n in df.columns:
                got[key] = n
                break
    missing = [k for k in _REQUIRED if k not in got]
    if missing:
        raise SystemExit(
            f"[ruler] 입력에서 필수 필드를 못 찾았다: {missing}\n"
            f"  허용 별칭: {({k: _ALIASES[k] for k in missing})}\n"
            f"  실제 컬럼: {sorted(df.columns)}\n"
            "  meta_splice_probe.py 산출 스키마가 바뀌었으면 _ALIASES 를 넓혀라 "
            "(조용한 기본값으로 때우지 말 것).")
    return got


def load_sites(labels_path: str, sites_glob: str, limit: int = 0):
    import pandas as pd
    lab = pd.read_parquet(labels_path)
    shard_paths = sorted(Path(sites_glob).parent.glob(Path(sites_glob).name))
    if shard_paths:
        sh = pd.concat([pd.read_parquet(p) for p in shard_paths], ignore_index=True)
    else:
        sh = None

    cols = {}
    try:
        cols = _resolve_columns(lab)
        df = lab
    except SystemExit:
        if sh is None:
            raise
        # labels 단독으로 부족하면 site_id 로 병합해 다시 찾는다.
        key_l = next((n for n in _ALIASES["site_id"] if n in lab.columns), None)
        key_s = next((n for n in _ALIASES["site_id"] if n in sh.columns), None)
        if key_l is None or key_s is None:
            raise SystemExit(
                "[ruler] labels 만으로 스키마가 부족한데 site_id 가 없어 shards 와 "
                f"병합할 수 없다. labels 컬럼={sorted(lab.columns)} "
                f"shards 컬럼={sorted(sh.columns)}")
        df = lab.merge(sh, left_on=key_l, right_on=key_s,
                       suffixes=("", "_shard"), how="left")
        cols = _resolve_columns(df)

    if "site_id" not in cols:
        df = df.reset_index().rename(columns={"index": "site_id"})
        cols["site_id"] = "site_id"
    if limit:
        df = df.head(limit)

    labels_raw = df[cols["label"]].astype(str).str.lower()
    seen = sorted(labels_raw.unique())
    if "good" not in seen or "null" not in seen:
        raise SystemExit(
            f"[ruler] label 컬럼({cols['label']}) 값에 good/null 이 없다. 실측 값: {seen}")

    if "post" in cols:
        post = df[cols["post"]].fillna(0).astype(int).to_numpy()
    else:
        post = (labels_raw == "post").astype(int).to_numpy()
        print("[ruler] post 컬럼이 없어 label=='post' 로 유도했다 "
              f"(n_post={int(post.sum())})", flush=True)

    if "delta_ctl" in cols:
        delta_ctl = df[cols["delta_ctl"]].astype(float).to_numpy()
    else:
        delta_ctl = np.full(len(df), np.nan)
        print("[ruler] delta_ctl 컬럼이 없다 — Spearman ρ 는 NaN 로 보고된다", flush=True)

    sites = []
    for i, (_, row) in enumerate(df.iterrows()):
        sites.append({
            "site_id": row[cols["site_id"]],
            "nums": [int(v) for v in row[cols["nums"]]],
            "target": int(row[cols["target"]]),
            "witness": str(row[cols["witness"]]),
            "text": str(row[cols["text"]]),
            "label": labels_raw.iloc[i],
            "post": int(post[i]),
            "delta_ctl": float(delta_ctl[i]),
        })
    return sites


# ══════════════════════════════════════════════════════════════════════════════
# 2. 미끼 두 종
# ══════════════════════════════════════════════════════════════════════════════

def make_decoy_swap(witness: str, nums, target, rng) -> Optional[str]:
    """연산자 하나 교체 미끼. countdown_task.swap_op_decoy 재사용(eval≠target 은 그
    함수가 이미 확인) + _solves 로 방어적 재확인."""
    d = swap_op_decoy(witness, nums, target, rng)
    if d is None or _solves(d, nums, target):
        return None
    return d


_FAM_MULDIV = str.maketrans({"*": "+", "/": "-"})
_FAM_ADDSUB = str.maketrans({"+": "*", "-": "/"})


def make_decoy_family(witness: str, target) -> tuple[Optional[str], str]:
    """witness 의 모든 */ 를 +− 로(없으면 그 반대로) 바꾼 «다른 가족» 식.

    eval 값 ≠ target 확인. 같으면 (witness)+1 (항상 target+1 ≠ target)로 회피.
    반환: (미끼, 종류) — 종류 ∈ {"muldiv2addsub", "addsub2muldiv", "plus1", "none"}.
    """
    t = int(target)
    for cand, kind in ((witness.translate(_FAM_MULDIV), "muldiv2addsub"),
                       (witness.translate(_FAM_ADDSUB), "addsub2muldiv")):
        if cand == witness:
            continue
        v = eval_countdown(cand)          # 규칙 위반이면 None → target 과 다름
        if v != t:
            return cand, kind
    cand = f"({witness})+1"
    if eval_countdown(cand) != t:         # 정의상 target+1 이지만 명세대로 확인한다
        return cand, "plus1"
    return None, "none"


# ══════════════════════════════════════════════════════════════════════════════
# 3. 문맥·대상 준비 (CPU) — 스팬은 parse_meta 로 재파싱
# ══════════════════════════════════════════════════════════════════════════════

_CONF_LINE = re.compile(r"^confidence\s*:", re.I)
_DEC_LINE = re.compile(r"^decision\s*:", re.I)


def lower_first_meta_char(text: str, m: dict) -> Optional[str]:
    """메타 프로즈 첫 알파벳이 대문자면 그 한 글자만 소문자로 바꾼 전체 텍스트.

    conf/decision/태그 줄은 건너뛰고 첫 프로즈 줄의 첫 알파벳을 찾는다.
    대문자가 아니거나 프로즈가 없으면 None (그 사이트는 ④ 표본에서 제외).
    """
    raw, s = m.get("raw") or "", m.get("start")
    if not raw or s is None:
        return None
    pos = 0
    for line in raw.splitlines(keepends=True):
        st = line.strip()
        if (st and not st.startswith("<meta") and not st.startswith("</meta")
                and not _CONF_LINE.match(st) and not _DEC_LINE.match(st)):
            for j, ch in enumerate(line):
                if ch.isalpha():
                    if ch.isupper():
                        k = s + pos + j
                        return text[:k] + text[k].lower() + text[k + 1:]
                    return None
            return None
        pos += len(line)
    return None


def behavior_novelty(text: str, m: dict) -> float:
    """R6: 메타 앞 300자/뒤 300자의 연산자·숫자쌍 multiset novelty (forward 불필요).

    쌍 = (a, op, b). 교환법칙이 성립하는 +,* 는 피연산자를 정렬해 정규화한다.
    novelty = |뒤에만 있는 것(multiset 차)| / |뒤 전체|. 뒤에 쌍이 없으면 NaN.
    """
    s, e = m["start"], m["end"]
    before = text[max(0, s - BEHAV_WINDOW):s]
    after = text[e:e + BEHAV_WINDOW]

    def pairs(seg: str) -> Counter:
        c = Counter()
        for mm in re.finditer(r"(\d+)\s*([+\-*/])\s*(\d+)", seg):
            a, op, b = mm.group(1), mm.group(2), mm.group(3)
            if op in "+*" and b < a:
                a, b = b, a
            c[(a, op, b)] += 1
        return c

    cb, ca = pairs(before), pairs(after)
    tot = sum(ca.values())
    if tot == 0:
        return float("nan")
    only_after = sum(max(0, n - cb.get(k, 0)) for k, n in ca.items())
    return only_after / tot


@dataclass
class SiteJob:
    site: dict
    ctx_close: list
    ctx_open: list
    gold_ids: tuple            # \boxed{witness}
    dswap_ids: tuple           # \boxed{op-swap decoy}
    gold_slice: slice          # divergent_spans 의 발산 슬라이스 (R1)
    dswap_slice: slice
    div_path: str              # "div" | "full"
    dfam_ids: tuple            # \boxed{family decoy}
    family_kind: str
    self_ids: tuple            # 메타 뒤 최대 200토큰 (원문에서 추출)
    ctx_lower: Optional[list] = None   # ④ 소문자화 변형의 close 문맥
    meta: dict = field(default_factory=dict)
    r6: float = float("nan")
    c_len: float = float("nan")
    c_conf: float = float("nan")


def build_chat_prefix(tok, inst: dict, meta_format: str) -> str:
    """countdown_gs0_eval.py 와 같은 방식의 chat template (enable_thinking 포함)."""
    msgs = build_prompt(inst, variant=meta_format)
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=TRAIN_SAMPLING["enable_thinking"])
    except TypeError:                     # enable_thinking 없는 토크나이저
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def prepare_jobs(tok, sites: list, meta_format: str, seed: int,
                 n_invariance: int) -> tuple[list, dict]:
    import random
    rng = random.Random(seed)
    skip = Counter()
    jobs: list[SiteJob] = []
    n_lower = 0
    for site in sites:
        text = site["text"]
        m = parse_meta(text, meta_format)
        if not m["emitted"] or m["start"] is None or m["end"] is None:
            skip["n_meta_reparse_fail"] += 1
            continue
        if not _solves(site["witness"], site["nums"], site["target"]):
            skip["n_witness_invalid"] += 1
            continue
        dswap = make_decoy_swap(site["witness"], site["nums"], site["target"], rng)
        if dswap is None:
            skip["n_no_swap_decoy"] += 1
            continue
        pair = divergent_spans(tok, site["witness"], dswap)
        if pair is None:
            skip["n_no_divergent_pair"] += 1
            continue
        dfam, fam_kind = make_decoy_family(site["witness"], site["target"])
        if dfam is None:
            skip["n_no_family_decoy"] += 1
            continue

        chat = build_chat_prefix(tok, site, meta_format)
        # 문맥은 문자열로 이어 한 번에 토큰화한다(close/open 이 같은 방식이라 상쇄).
        # 대상 토큰은 **id 로 이어붙이므로** 경계 병합이 없다.
        ctx_close = tok.encode(chat + text[:m["end"]], add_special_tokens=False)
        ctx_open = tok.encode(chat + text[:m["start"]], add_special_tokens=False)
        self_ids = tuple(tok.encode(text[m["end"]:],
                                    add_special_tokens=False)[:SELF_MAX_TOK])
        dfam_ids = tuple(tok.encode(boxed(dfam), add_special_tokens=False))

        job = SiteJob(
            site=site, ctx_close=ctx_close, ctx_open=ctx_open,
            gold_ids=pair.gold_ids, dswap_ids=pair.decoy_ids,
            gold_slice=pair.gold_slice, dswap_slice=pair.decoy_slice,
            div_path=pair.path, dfam_ids=dfam_ids, family_kind=fam_kind,
            self_ids=self_ids, meta=m,
            r6=behavior_novelty(text, m),
            c_len=float(len(m.get("body") or "")),
            c_conf=(float(m["confidence"]) if m.get("confidence") is not None
                    else float("nan")),
        )
        if site["label"] == "good" and n_lower < n_invariance:
            lowered = lower_first_meta_char(text, m)
            if lowered is not None:
                job.ctx_lower = tok.encode(chat + lowered[:m["end"]],
                                           add_special_tokens=False)
                n_lower += 1
        jobs.append(job)
    skip["n_invariance_sites"] = n_lower
    return jobs, dict(skip)


# ══════════════════════════════════════════════════════════════════════════════
# 4. vLLM prompt_logprobs 채점 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def score_logp(llm, reqs: Sequence[tuple], *, batch_size: int = 128,
               max_model_len: int = 0) -> list:
    """reqs: [(ctx_ids, target_ids), ...] → 대상 구간 per-token logprob 배열 리스트.

    문맥토큰+대상토큰을 이어 한 시퀀스로 넣고 max_tokens=1 + prompt_logprobs=0 으로
    호출, 대상 구간의 실제 토큰 logprob 을 뽑는다 (vLLM 0.10.2 실측 API — 헤더 참조).
    너무 긴 시퀀스는 None 로 표시하고 건너뛴다(조용한 절단 없음).
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    sp = SamplingParams(max_tokens=1, temperature=TRAIN_SAMPLING["temperature"],
                        top_p=TRAIN_SAMPLING["top_p"],
                        prompt_logprobs=0, detokenize=False)
    out: list = [None] * len(reqs)
    todo = []
    for i, (c, t) in enumerate(reqs):
        if not t or (max_model_len and len(c) + len(t) + 2 > max_model_len):
            continue
        todo.append(i)
    for lo in range(0, len(todo), batch_size):
        idxs = todo[lo:lo + batch_size]
        prompts = [TokensPrompt(prompt_token_ids=list(reqs[i][0]) + list(reqs[i][1]))
                   for i in idxs]
        results = llm.generate(prompts, sp, use_tqdm=False)
        for i, r in zip(idxs, results):
            c, t = reqs[i]
            plp = r.prompt_logprobs
            if plp is None or len(plp) != len(c) + len(t):
                raise RuntimeError(
                    f"[ruler] prompt_logprobs 길이 불일치: {None if plp is None else len(plp)}"
                    f" vs {len(c) + len(t)} — vLLM API 가정이 깨졌다.")
            lps = []
            for pos, tid in enumerate(t):
                d = plp[len(c) + pos]
                if d is None or tid not in d:
                    raise RuntimeError(
                        f"[ruler] 위치 {pos} 의 실제 토큰 {tid} 이 prompt_logprobs 에 없다.")
                lps.append(d[tid].logprob)
            out[i] = np.asarray(lps, dtype=np.float64)
        done = min(lo + batch_size, len(todo))
        print(f"[ruler] scored {done}/{len(todo)} sequences", flush=True)
    return out


def score_logp_hf(model, reqs: Sequence[tuple], *, batch_size: int = 4,
                  max_model_len: int = 0) -> list:
    """score_logp 와 동일 계약의 HF transformers 백엔드 (vLLM prompt_logprobs
    segfault 우회). 대상 구간 위치의 로짓만 float32 로 뽑아 log_softmax —
    전체 vocab 로짓을 상주시키지 않아 배치 메모리가 작다."""
    import torch
    out: list = [None] * len(reqs)
    todo = [i for i, (c, t) in enumerate(reqs)
            if t and len(c) >= 1
            and not (max_model_len and len(c) + len(t) + 2 > max_model_len)]
    todo.sort(key=lambda i: len(reqs[i][0]) + len(reqs[i][1]))
    for lo in range(0, len(todo), batch_size):
        idxs = todo[lo:lo + batch_size]
        seqs = [list(reqs[i][0]) + list(reqs[i][1]) for i in idxs]
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, sq in enumerate(seqs):
            ids[j, :len(sq)] = torch.tensor(sq); att[j, :len(sq)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(model.device),
                           attention_mask=att.to(model.device)).logits
        for j, i in enumerate(idxs):
            c, t = reqs[i]
            # t[p] 의 예측 로짓 위치는 len(c)+p−1 (직전 토큰 자리)
            pos = torch.arange(len(c) - 1, len(c) - 1 + len(t), device=logits.device)
            sel = logits[j, pos, :].float()
            lsm = torch.log_softmax(sel, dim=-1)
            tid = torch.tensor(list(t), device=logits.device)
            out[i] = lsm[torch.arange(len(t), device=logits.device), tid
                         ].double().cpu().numpy()
        del logits
        done = min(lo + batch_size, len(todo))
        if done % 64 < batch_size or done == len(todo):
            print(f"[ruler] scored {done}/{len(todo)} sequences (hf)", flush=True)
    return out


def compute_scores(scorer, jobs: list, batch_size: int, max_model_len: int):
    """사이트당 R1~R5 (+④ 소문자 변형의 R1~R5) 를 채운 rows 리스트."""
    reqs, index = [], {}

    def add(key, ctx, tgt):
        index[key] = len(reqs)
        reqs.append((ctx, tgt))

    for j, job in enumerate(jobs):
        for cname, ctx in (("close", job.ctx_close), ("open", job.ctx_open)):
            add((j, cname, "gold"), ctx, job.gold_ids)
            add((j, cname, "dswap"), ctx, job.dswap_ids)
            add((j, cname, "dfam"), ctx, job.dfam_ids)
            if job.self_ids:
                add((j, cname, "self"), ctx, job.self_ids)
        if job.ctx_lower is not None:
            add((j, "lower", "gold"), job.ctx_lower, job.gold_ids)
            add((j, "lower", "dswap"), job.ctx_lower, job.dswap_ids)
            add((j, "lower", "dfam"), job.ctx_lower, job.dfam_ids)
            if job.self_ids:
                add((j, "lower", "self"), job.ctx_lower, job.self_ids)

    lps = scorer(reqs)

    def get(j, cname, tname):
        i = index.get((j, cname, tname))
        return None if i is None else lps[i]

    def rulers_from(j: int, close: str) -> dict:
        """close ∈ {"close","lower"} 문맥과 open 문맥으로 R1~R5 를 계산."""
        job = jobs[j]
        g_c, g_o = get(j, close, "gold"), get(j, "open", "gold")
        s_c, s_o = get(j, close, "dswap"), get(j, "open", "dswap")
        f_c, f_o = get(j, close, "dfam"), get(j, "open", "dfam")
        e_c, e_o = get(j, close, "self"), get(j, "open", "self")
        nan = float("nan")
        if any(x is None for x in (g_c, g_o, s_c, s_o, f_c, f_o)):
            return {k: nan for k in FORWARD_RULERS}
        gs, ds = job.gold_slice, job.dswap_slice
        r1 = ((g_c[gs].sum() - s_c[ds].sum()) - (g_o[gs].sum() - s_o[ds].sum()))
        r2 = ((g_c.mean() - s_c.mean()) - (g_o.mean() - s_o.mean()))
        r3 = ((g_c.mean() - f_c.mean()) - (g_o.mean() - f_o.mean()))
        r4 = g_c.mean() - g_o.mean()
        r5 = (e_c.mean() - e_o.mean()) if (e_c is not None and e_o is not None) else nan
        return {"R1_pmi1tok": float(r1), "R2_full": float(r2), "R3_family": float(r3),
                "R4_osdgold": float(r4), "R5_osdself": float(r5)}

    rows = []
    for j, job in enumerate(jobs):
        row = {
            "site_id": job.site["site_id"], "label": job.site["label"],
            "post": job.site["post"], "delta_ctl": job.site["delta_ctl"],
            "div_path": job.div_path, "family_kind": job.family_kind,
            "n_self_tok": len(job.self_ids),
            "R6_behavior": job.r6, "C_len": job.c_len, "C_conf": job.c_conf,
        }
        row.update(rulers_from(j, "close"))
        if job.ctx_lower is not None:
            base = {k: row[k] for k in FORWARD_RULERS}
            low = rulers_from(j, "lower")
            for k in FORWARD_RULERS:
                row[f"inv_delta_{k}"] = (
                    abs(low[k] - base[k])
                    if (low[k] == low[k] and base[k] == base[k]) else float("nan"))
            # R6·카나리아는 소문자화에 정의상 불변(메타 밖/길이·수치 무변) → Δ=0.
            row["inv_delta_R6_behavior"] = 0.0
            row["inv_delta_C_len"] = 0.0
            row["inv_delta_C_conf"] = 0.0
        rows.append(row)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 5. 분석 — AUC/CI, Spearman, 사후보고, 불변성, 카나리아, 판정
# ══════════════════════════════════════════════════════════════════════════════

def auc_good_null(scores, labels) -> tuple:
    """AUC = P(good > null) (+0.5 tie). NaN 점수는 제외. (auc, n_good, n_null)."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s == s]
    pos = np.array([s for s, l in pairs if l == "good"])
    neg = np.array([s for s, l in pairs if l == "null"])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), len(pos), len(neg)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg))), len(pos), len(neg)


def bootstrap_auc_ci(scores, labels, b: int = BOOTSTRAP_B, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    pairs = [(s, l) for s, l in zip(scores, labels) if s == s and l in ("good", "null")]
    if not pairs:
        return float("nan"), float("nan")
    vals = []
    n = len(pairs)
    for _ in range(b):
        idx = rng.integers(0, n, n)
        a, _, _ = auc_good_null([pairs[i][0] for i in idx], [pairs[i][1] for i in idx])
        if a == a:
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> tuple:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan"), int(len(x))
    rx, ry = _rank(x), _rank(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan"), int(len(x))
    rho = float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))
    return rho, int(len(x))


def analyze(rows: list, skip: dict, meta: dict) -> dict:
    labels = [r["label"] for r in rows]
    report = {"thresholds": {"auc_min": PASS_AUC_MIN, "invariance_max": PASS_INV_MAX,
                             "canary": "auc - max(C_len, C_conf auc) > 0",
                             "post": "mean(post) <= mean(good)"},
              "n_sites_scored": len(rows), "skip": skip, "meta": meta,
              "rulers": {}}

    # 카나리아 AUC 를 먼저 — ⑤ 의 기준선.
    canary_aucs = {}
    for c in CANARIES:
        a, _, _ = auc_good_null([r[c] for r in rows], labels)
        canary_aucs[c] = a
    canary_max = max((v for v in canary_aucs.values() if v == v), default=float("nan"))

    for name in RULERS:
        scores = [r[name] for r in rows]
        a, n_pos, n_neg = auc_good_null(scores, labels)
        lo, hi = bootstrap_auc_ci(scores, labels)
        rho, rho_n = spearman(scores, [r["delta_ctl"] for r in rows])

        good = np.array([r[name] for r in rows
                         if r["label"] == "good" and r[name] == r[name]])
        null = np.array([r[name] for r in rows
                         if r["label"] == "null" and r[name] == r[name]])
        postv = np.array([r[name] for r in rows
                          if r["post"] == 1 and r[name] == r[name]])
        good_mean = float(good.mean()) if len(good) else float("nan")
        null_mean = float(null.mean()) if len(null) else float("nan")
        post_mean = float(postv.mean()) if len(postv) else float("nan")
        # ③ 사후보고가 good 보다 **높으면** FAIL. post 표본이 없으면 non-FAIL.
        post_fail = bool(len(postv) and post_mean == post_mean
                         and good_mean == good_mean and post_mean > good_mean)

        # ④ |Δ점수| 중앙값 / (good−null 점수 간격)
        deltas = [r.get(f"inv_delta_{name}") for r in rows
                  if r.get(f"inv_delta_{name}") is not None]
        deltas = [d for d in deltas if d == d]
        gap = abs(good_mean - null_mean)
        inv_ratio = (float(np.median(deltas)) / gap
                     if deltas and gap == gap and gap > 0 else float("nan"))

        canary_margin = (a - canary_max) if (a == a and canary_max == canary_max) \
            else float("nan")

        is_canary = name in CANARIES
        verdict = bool(
            a == a and a >= PASS_AUC_MIN
            and not post_fail
            and canary_margin == canary_margin and canary_margin > 0
            and (inv_ratio != inv_ratio or inv_ratio < PASS_INV_MAX))
        # ④ 표본이 아예 없으면(불변성 미측정) 통과로 치지 않는다 — forward 자에 한해.
        if name in FORWARD_RULERS and not deltas:
            verdict = False

        report["rulers"][name] = {
            "auc": a, "auc_ci95": [lo, hi], "n_good": n_pos, "n_null": n_neg,
            "spearman_delta_ctl": rho, "spearman_n": rho_n,
            "good_mean": good_mean, "null_mean": null_mean,
            "post_mean": post_mean, "n_post": int(len(postv)),
            "post_fail": post_fail,
            "invariance_ratio": inv_ratio, "invariance_n": len(deltas),
            "canary_margin": canary_margin, "is_canary": is_canary,
            "pass": verdict,
        }
    report["canary_aucs"] = canary_aucs
    return report


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:.{nd}f}"


def print_markdown(report: dict) -> None:
    print("\n## 자 배터리 판별력 보고 "
          f"(n={report['n_sites_scored']}, skip={report['skip']})\n")
    print("| 자 | AUC [95% CI] | ρ(Δ_ctl) | post평균 vs good평균 | 불변성비 | "
          "카나리아margin | 판정 |")
    print("|---|---|---|---|---|---|---|")
    for name in RULERS:
        r = report["rulers"][name]
        post = (f"{_fmt(r['post_mean'])} vs {_fmt(r['good_mean'])}"
                + (" **FAIL**" if r["post_fail"] else ""))
        verdict = "PASS" if r["pass"] else ("(카나리아)" if r["is_canary"] else "FAIL")
        print(f"| {name} | {_fmt(r['auc'])} "
              f"[{_fmt(r['auc_ci95'][0])}, {_fmt(r['auc_ci95'][1])}] "
              f"| {_fmt(r['spearman_delta_ctl'])} (n={r['spearman_n']}) "
              f"| {post} | {_fmt(r['invariance_ratio'])} (n={r['invariance_n']}) "
              f"| {_fmt(r['canary_margin'])} | {verdict} |")
    print(f"\n통과선: AUC≥{PASS_AUC_MIN} ∧ 사후보고 non-FAIL ∧ 카나리아 초과 ∧ "
          f"불변성비<{PASS_INV_MAX}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def cmd_score(args) -> None:
    import pandas as pd
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sites = load_sites(args.labels, args.sites_glob, args.limit)
    print(f"[ruler] {len(sites)} 라벨된 사이트 로드", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    jobs, skip = prepare_jobs(tok, sites, args.meta_format, args.seed,
                              args.n_invariance)
    print(f"[ruler] 채점 대상 {len(jobs)} 사이트 · skip={skip}", flush=True)
    if not jobs:
        raise SystemExit("[ruler] 채점할 사이트가 0개다 — 위 skip 사유를 보라.")

    if args.backend == "hf":
        import torch
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").cuda().eval()
        scorer = (lambda reqs: score_logp_hf(
            model, reqs, batch_size=args.batch_size,
            max_model_len=args.max_model_len))
    else:
        from vllm import LLM
        llm = LLM(model=args.model_path, dtype="bfloat16", seed=args.seed,
                  tensor_parallel_size=1,
                  gpu_memory_utilization=args.gpu_util,
                  max_model_len=args.max_model_len,
                  # gs0_eval 과 같은 이유(공용 서버 CPU 경합에서 torch.compile 사망 회피).
                  enforce_eager=True)
        scorer = (lambda reqs: score_logp(
            llm, reqs, batch_size=args.batch_size,
            max_model_len=args.max_model_len))

    rows = compute_scores(scorer, jobs, args.batch_size, args.max_model_len)
    scores_path = out / "ruler_scores.parquet"
    pd.DataFrame(rows).to_parquet(scores_path)
    print(f"[ruler] wrote {scores_path}", flush=True)

    meta = {"model_path": args.model_path, "labels": args.labels,
            "sites_glob": args.sites_glob, "meta_format": args.meta_format,
            "seed": args.seed, "train_sampling_measured": TRAIN_SAMPLING}
    report = analyze(rows, skip, meta)
    (out / "ruler_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"[ruler] wrote {out / 'ruler_report.json'}", flush=True)
    print_markdown(report)


def cmd_report(args) -> None:
    """저장된 ruler_scores.parquet 만으로 분석을 다시 돌린다 (GPU 불필요)."""
    import pandas as pd
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(args.scores).to_dict("records")
    report = analyze(rows, skip={}, meta={"scores": args.scores, "reanalysis": True})
    (out / "ruler_report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"[ruler] wrote {out / 'ruler_report.json'}", flush=True)
    print_markdown(report)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="vLLM 로드 → 채점 → 분석/보고 (GPU 1장)")
    sc.add_argument("--model_path", required=True)
    sc.add_argument("--labels", required=True, help="labels.parquet")
    sc.add_argument("--sites_glob", default="", help="sites_shard*.parquet 글롭 "
                    "(labels 가 모든 필드를 들고 있으면 생략 가능)")
    sc.add_argument("--out_dir", required=True)
    sc.add_argument("--meta_format", default="new", choices=["new", "old"])
    sc.add_argument("--limit", type=int, default=0, help="0 이면 전부")
    sc.add_argument("--n_invariance", type=int, default=N_INVARIANCE)
    sc.add_argument("--seed", type=int, default=0)
    sc.add_argument("--gpu_util", type=float, default=0.85)
    sc.add_argument("--max_model_len", type=int, default=4864,
                    help="프롬프트+응답(실측 max_response_length=3072)+대상이 들어갈 길이")
    sc.add_argument("--batch_size", type=int, default=128)
    sc.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    sc.set_defaults(func=cmd_score)

    rp = sub.add_parser("report", help="저장된 점수로 분석만 다시 (CPU)")
    rp.add_argument("--scores", required=True, help="ruler_scores.parquet")
    rp.add_argument("--out_dir", required=True)
    rp.set_defaults(func=cmd_report)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "score" and not args.sites_glob:
        args.sites_glob = str(Path(args.labels).parent / "sites_shard*.parquet")
    args.func(args)


if __name__ == "__main__":
    main()
