#!/usr/bin/env python
r"""math_pmi_shift_probe — «자발적 답 수정 구간에서 믿음이 움직이는가, 그것이 내용 때문인가».

이 정책에서 실측된 유일한 진짜 메타인지 행동은 **자발적 답 수정**(spontaneous answer
revision)이다: 롤아웃의 2.9% 에서 뒤의 `\boxed{}` 가 앞의 것과 **수학적으로 다르고**, 같은
문제 안에서 수정한 롤아웃은 첫 답이 틀렸던 비율이 82.6%(수정하지 않은 롤아웃 30.3%)다.
이 스크립트는 그 구간을 사이에 두고 모델의 **믿음**(정답-대-오답 로그오즈 = PMI)이 움직이는지,
그리고 그 움직임이 구간의 **내용** 때문인지(단순 존재·길이가 아니라) 를 잰다. 훈련이 아니라
**신호 시험**이다 — 생성은 없고(teacher forcing), vLLM `prompt_logprobs` 로 점수만 낸다.

한 롤아웃(행)마다 세 문맥에서 같은 이어쓰기(`ANSWER_TMPL`)의 **답 토큰만** 채점한다:
  CTX_OPEN     생성 프롬프트 + 응답을 **첫 박스 끝까지**(닫는 `}` 포함) 자른 것
  CTX_CLOSE    생성 프롬프트 + 응답을 `--close_at` 이 자른 것 — 기본 `before_last_box`:
               **마지막 `\boxed{` 시작 바로 앞**(답 Y 를 아직 안 쓴 상태 — 추론만으로 믿음이
               움직였는지 잰다; Y 를 쓴 뒤 Y 를 편애하는 건 동어반복이라 뺐다). 구 동작
               `after_last_box` 는 마지막 박스 **끝까지**(Y 포함)로 대조용으로만 남겨둔다.
  CTX_PLACEBO  CTX_OPEN + **내용 없는 채움말**(`PLACEBO_SENT` 반복, 실제 구간
               (CTX_CLOSE − CTX_OPEN)의 토큰 수에 ±15% 안으로 맞춤 — 행마다 실제 달성비를 낸다)
수정하지 않은 행도 버리지 않는다 — 그 행들이 **귀무 모집단**이다(박스가 하나뿐이면 CLOSE 는
OPEN 과 같아져 SHIFT_real 이 정의상 0 이다).

앵커 쌍 세 벌(전부 세 문맥에서 잰다, 기본 `--pairs gold,self,goldx`):
  GOLD  (A_pos, A_neg) = (gold, decoy) — decoy = 그 문제 8개 롤아웃의 **시도-1 답**(첫 박스)
        중 가장 흔한 **오답**(동률 → 먼저 나온 것). 없으면 그 행의 GOLD 쌍을 건너뛰고 센다.
  SELF  (A_pos, A_neg) = (M, X) — **gold 를 보지 않는다**. X = 이 행의 **첫** 답,
        M = 그 문제 8개 **최종 답**의 다수(동률 → 먼저 나온 것). M ≡ X 면 건너뛰고 센다.
  GOLDX (A_pos, A_neg) = (gold, X) — X = 이 행의 **첫** 답. X ≡ gold(이미 맞혔다)거나 gold
        가 없으면 건너뛴다.
  PMI_ctx = logp(A_pos | ctx) − logp(A_neg | ctx)
  SHIFT_real = PMI_close − PMI_open ·  SHIFT_placebo = PMI_placebo − PMI_open

★어떤 토큰을 더하는가(`answer_target_ids` 가 단일 진실 원천 — 테스트가 고정한다):
  문맥 = `ctx + ANSWER_HEAD`, 대상 = 그 뒤에 답 문자열이 붙으면서 **달라지는 토큰들**
  (= enc(ctx+HEAD+answer) 와 enc(ctx+HEAD) 의 공통 접두 이후 전부). `ANSWER_TMPL` 의 닫는
  `}` 는 **채점하지 않는다**(답 문자열이 끝나는 자리마다 토큰 경계가 달라 비교가 오염된다).
  토크나이저가 `{` 와 답의 첫 글자를 한 토큰으로 합치면(Qwen3 실측: `-3`·`x^2+1`·`\frac…`)
  그 합친 토큰이 대상에 들어간다. 한 쌍의 두 후보는 `pair_target_ids` 가 자르는 자리를
  **앞쪽으로 맞춰** 주므로 둘의 문맥 토큰열이 바이트 동일하고 채점 문자 구간도 같다
  (한쪽만 `{` 를 공짜로 얻는 비대칭이 없다).

★보상 재사용: `src.training.dcpo_pmi_shift.pmi_shift_reward` 를 **그대로** 부른다(부호 반전
  비대칭 조성을 여기서 다시 구현하지 않는다) — 기본 설정으로 (PMI_open, PMI_close) 에 매겨
  분포와 «실제 결과 변화(last_correct − first_correct)» 와의 상관을 낸다.

★관문(summary.md 의 VERDICT 블록):
  (a) 수정한 행의 SHIFT_real 이 양수이고 CI 가 0 을 제외한다 — **주어진 앵커 쌍 전부**.
  (b) 수정한 행의 (SHIFT_real − SHIFT_placebo) CI 가 0 을 제외한다 — 존재·길이가 아니라 **내용**.
  (c) SHIFT_real 의 **문제 안** AUC(그 행의 최종 정오 예측) ≥ 0.60 — **수정한 행만**으로 잰다
      (문제당 ≥2 개 ∧ 정오가 섞인 문제만 기여; 미수정 행은 SHIFT_real ≡ 0 이라 섞으면 관문이
      구조적으로 깨진다). 전 행 AUC 는 summary.md 에 참고용으로만 남는다.

재개: `rows.jsonl` 에 이미 있는 `roll_id` 는 다시 채점하지 않는다(math_protocol_eval 규약).

사용(예):
  python scripts/local/math_pmi_shift_probe.py --out_name pmishift_s1 --gpu_util 0.45
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("TMPDIR", "/hdd_data/seungpil/tmp")
os.environ.setdefault("HF_HOME", "/hdd_data/seungpil/scratch/hf_home")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402
from src.training.dcpo_pmi_shift import pmi_shift_reward  # noqa: E402
from src.training.math_meta import (  # noqa: E402
    answers_equivalent, answers_equivalent_loose, boxed_spans, first_boxed, grade_math,
    last_boxed, selftest_math_verify,
)
from src.training.trial2 import agreement_state  # noqa: E402

_NAN = float("nan")


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> dict:
    """백분위 부트스트랩 CI(롤아웃 단위 재표본). 유한한 값만 쓴다.

    ★math_cited_site_gate.py 에서 그대로 옮겨왔다(0922 아카이브) — 산술 불변.
    mc.habit._boot 과는 스키마가 다르다(반올림·키 이름), 그래서 합치지 않는다.
    """
    v = [float(x) for x in values if x is not None and isinstance(x, (int, float))
         and math.isfinite(float(x))]
    if not v:
        return {"mean": _NAN, "lo": _NAN, "hi": _NAN, "n": 0}
    rng = random.Random(seed)
    n = len(v)
    bs = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"mean": sum(v) / n, "lo": bs[int(alpha / 2 * n_boot)],
            "hi": bs[min(n_boot - 1, int((1 - alpha / 2) * n_boot))], "n": n}

ROLLOUTS_DEFAULT = "/hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl"
OUT_ROOT = "/hdd_data/seungpil/scratch/eval"
DEFAULT_MODEL = "/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507"

#: 내용 없는 채움말 — 추론도 수식도 답도 없다(길이·존재만 남기는 대조).
PLACEBO_SENT = "\n\nLet me write this out once more, carefully and completely, step by step."
#: 세 문맥에서 공통으로 쓰는 이어쓰기. HEAD + answer + TAIL 로 쪼개 **답 토큰만** 채점한다.
ANSWER_TMPL = "\n\nFinal answer: \\boxed{{{answer}}}"
ANSWER_HEAD = "\n\nFinal answer: \\boxed{"
ANSWER_TAIL = "}"
#: 채움말 길이 허용 오차(실제 구간 토큰 수 대비)
PLACEBO_TOL = 0.15
#: 관문 (c) 의 하한
AUC_MIN = 0.60
#: 채점 시퀀스 상한 — 넘는 요청은 조용히 자르지 않고 **건너뛰고 센다**(too_long).
MAX_MODEL_LEN = 12288

PAIRS = ("gold", "self", "goldx")
#: CTX_CLOSE 경계 — before_last_box(신규 기본): 마지막 \boxed{ 시작 **바로 앞**까지(답 미포함,
#: 추론만). after_last_box(구 동작): 마지막 박스 닫는 `}` 뒤까지(답 포함, 동어반복).
CLOSE_AT_CHOICES = ("before_last_box", "after_last_box")
CTXS = ("open", "close", "placebo")
AGREE_STATES = ("ALL_SAME", "DOMINANT", "SPLIT", "SCATTER", "NOANS")


# ── 토큰·문맥 조립 ──────────────────────────────────────────────────────────────
def _enc(tok, text: str) -> list[int]:
    return list(tok.encode(text or "", add_special_tokens=False))


def answer_continuation(answer: str) -> str:
    """세 문맥에 공통으로 붙는 이어쓰기 = ANSWER_HEAD + answer + ANSWER_TAIL."""
    return ANSWER_TMPL.format(answer=answer)


def common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def answer_target_ids(tok, ctx: str, answer: str, *, cut: int = -1) -> tuple[list[int], list[int]]:
    r"""(문맥 토큰, **채점 대상** 토큰). 모듈 docstring 의 «어떤 토큰을 더하는가» 규약 본체.

    head = enc(ctx + ANSWER_HEAD), full = enc(ctx + ANSWER_HEAD + answer),
    c = 공통 접두 길이 → (full[:c], full[c:]). 닫는 `}` 는 대상에 들어가지 않는다.
    `cut >= 0` 이면 그 자리에서 자른다(`pair_target_ids` 가 두 후보의 자리를 맞출 때 쓴다).

    ★Qwen3 토크나이저 실측: 숫자로 시작하는 답('42','900')은 `{` 뒤에서 깨끗이 갈리지만
      `-3`·`x^2+1`·`\frac…` 같은 답은 `{` 가 답의 첫 토큰에 합쳐진다. 그래서 한 앵커 쌍의
      두 후보를 **따로** 자르면 문맥 길이가 한 토큰 어긋날 수 있다 — `pair_target_ids` 가
      둘의 자리를 앞쪽으로 맞춰(min) 두 후보가 **같은 문맥 토큰열**에서, 같은 문자 구간
      (`{`+답)을 채점하게 한다."""
    head = _enc(tok, ctx + ANSWER_HEAD)
    full = _enc(tok, ctx + ANSWER_HEAD + str(answer or ""))
    c = common_prefix_len(head, full) if cut < 0 else min(int(cut), len(full))
    return full[:c], full[c:]


def pair_target_ids(tok, ctx: str, pos: str, neg: str) -> tuple[tuple, tuple]:
    """한 문맥에서 두 후보 답을 **같은 토큰 자리**에서 채점하기 위한 ((ctx,tgt), (ctx,tgt)).
    자르는 자리 = 두 후보의 공통 접두 길이 중 **작은 쪽** — 두 문맥 토큰열이 바이트 동일해진다."""
    head = _enc(tok, ctx + ANSWER_HEAD)
    cuts = [common_prefix_len(head, _enc(tok, ctx + ANSWER_HEAD + str(a or "")))
            for a in (pos, neg)]
    cut = min(cuts)
    return (answer_target_ids(tok, ctx, pos, cut=cut),
            answer_target_ids(tok, ctx, neg, cut=cut))


def placebo_filler(tok, n_tokens: int, sent: str = PLACEBO_SENT) -> str:
    """`sent` 를 반복해 **정확히** n_tokens 개 토큰이 되도록 자른 채움말(0 이하면 "")."""
    n = int(n_tokens)
    if n <= 0:
        return ""
    unit = _enc(tok, sent)
    if not unit:
        raise RuntimeError("[pmi] PLACEBO_SENT 가 0 토큰이다 — 토크나이저를 확인하라")
    s = sent * (n // len(unit) + 2)
    ids = _enc(tok, s)
    if len(ids) <= n:
        return s
    return tok.decode(ids[:n])


def row_contexts(tok, variant: str, problem: str, text: str,
                 *, close_at: str = "before_last_box") -> dict | None:
    r"""한 행의 세 문맥 + 경계 정보. 박스가 하나도 없으면 None(그 행은 채점하지 않는다).

    CTX_OPEN 의 경계는 **첫 박스의 닫는 `}` 바로 뒤**다(`boxed_spans` 의 end 오프셋).
    CTX_CLOSE 의 경계는 `close_at` 이 정한다:
      before_last_box(기본) — **마지막 `\boxed{` 시작 바로 앞**(`boxed_spans` 의 start
        오프셋 — 정규식이 `\boxed{` 전체에서 매치하므로 start 가 이미 `\boxed` 위치다).
        답 문자열 Y 를 아직 안 쓴 상태 — 추론만으로 믿음이 움직였는지를 잰다.
      after_last_box(구 동작) — 마지막 박스의 닫는 `}` 뒤(Y 포함, 동어반복 — 대조용으로만).
    `revised` = 박스 ≥ 2 ∧ 첫 답 ≢ 마지막 답."""
    sp = boxed_spans(text or "")
    if not sp:
        return None
    prompt = render_generation_prompt(tok, variant, problem)
    first, last = sp[0], sp[-1]
    if len(sp) < 2:
        close_end = first[2]  # 박스 하나뿐 — CTX_CLOSE = CTX_OPEN(수정 구간이 없다)
    elif close_at == "before_last_box":
        close_end = last[1]
    else:
        close_end = last[2]
    ctx_open = prompt + (text or "")[:first[2]]
    ctx_close = prompt + (text or "")[:close_end]
    n_open, n_close = len(_enc(tok, ctx_open)), len(_enc(tok, ctx_close))
    seg = n_close - n_open
    # ★채움말을 **붙인 뒤** 다시 재서 부족하면 그만큼 늘린다 — decode→재encode 왕복에서
    #   경계 토큰이 합쳐져 1~2 토큰 모자라는 일이 있다(짧은 구간에서 ±15% 를 깬다).
    filler = ""
    aim = seg
    for _ in range(3):
        if seg <= 0:
            break
        filler = placebo_filler(tok, aim)
        got = len(_enc(tok, ctx_open + filler)) - n_open
        if got >= seg:
            break
        aim += seg - got
    ctx_placebo = ctx_open + filler
    n_plac = len(_enc(tok, ctx_placebo))
    return {
        "ctx_open": ctx_open, "ctx_close": ctx_close, "ctx_placebo": ctx_placebo,
        "first_answer": first[0], "last_answer": last[0], "n_boxes": len(sp),
        "revised": bool(len(sp) >= 2 and not answers_equivalent_loose(first[0], last[0])),
        "n_tok_open": n_open, "n_tok_close": n_close, "n_tok_placebo": n_plac,
        "seg_tokens": seg, "placebo_tokens": n_plac - n_open,
        "placebo_ratio": ((n_plac - n_open) / seg) if seg > 0 else _NAN,
    }


# ── 앵커 선별 ───────────────────────────────────────────────────────────────────
def _clusters(answers: Sequence[str]) -> list[list]:
    """[대표, 개수, 첫 등장] — 수학적 동치 군집(빈 답은 버린다)."""
    out: list[list] = []
    for i, a in enumerate(answers):
        s = str(a or "").strip()
        if not s:
            continue
        for cl in out:
            if answers_equivalent(cl[0], s):
                cl[1] += 1
                break
        else:
            out.append([s, 1, i])
    return out


def decoy_answer(first_answers: Sequence[str], gold: str) -> str:
    """GOLD 쌍의 A_neg — 시도-1 답들 중 **가장 흔한 오답**(동률 → 먼저 나온 것). 없으면 ""."""
    cl = [c for c in _clusters(first_answers) if not answers_equivalent(gold, c[0])]
    if not cl:
        return ""
    return min(cl, key=lambda c: (-c[1], c[2]))[0]


def majority_answer(final_answers: Sequence[str]) -> str:
    """SELF 쌍의 A_pos — 최종 답들의 다수(동률 → 먼저 나온 것). 없으면 "".
    ★`src.training.trial2.agreement_state(...)["dominant_answer"]` 와 **같은 규칙**이다
    (최대 군집 크기, 동률 → 첫 등장, 빈 답 제외) — plan_rows 는 그쪽을 직접 쓴다.
    이 함수는 하위호환(단독 호출·테스트)을 위해 남긴다."""
    cl = _clusters(final_answers)
    if not cl:
        return ""
    return min(cl, key=lambda c: (-c[1], c[2]))[0]


def pair_answers(plan: dict, first_answer: str,
                 pairs: Sequence[str] = PAIRS) -> tuple[dict, dict]:
    """pair → (A_pos, A_neg) 와 건너뛴 쌍 → 사유.
      gold:  (gold, decoy)     — decoy 없으면 `no_decoy`
      self:  (majority, first) — 첫 답 없으면 `no_first`, M 없으면 `no_majority`,
                                 M ≡ X 면 `majority_equals_first`
      goldx: (gold, first)     — gold 없으면 `no_gold`, 첫 답 없으면 `no_first`,
                                 X ≡ gold 면 `x_equals_gold`(이미 맞혀서 재는 게 무의미)"""
    want: dict = {}
    skip: dict = {}
    x = str(first_answer or "").strip()
    for p in pairs:
        if p == "gold":
            if not plan.get("decoy"):
                skip[p] = "no_decoy"
            else:
                want[p] = (str(plan.get("gold", "")), str(plan["decoy"]))
        elif p == "self":
            m = str(plan.get("majority", "") or "").strip()
            if not x:
                skip[p] = "no_first"
            elif not m:
                skip[p] = "no_majority"
            elif answers_equivalent(m, x):
                skip[p] = "majority_equals_first"
            else:
                want[p] = (m, x)
        elif p == "goldx":
            gold = str(plan.get("gold", "") or "").strip()
            if not gold:
                skip[p] = "no_gold"
            elif not x:
                skip[p] = "no_first"
            elif answers_equivalent(x, gold):
                skip[p] = "x_equals_gold"
            else:
                want[p] = (gold, x)
        else:
            raise SystemExit(f"[pmi] 모르는 앵커 쌍 {p!r} — 가능: {', '.join(PAIRS)}")
    return want, skip


def resolve_pairs(spec: str | None) -> list[str]:
    want = [x.strip() for x in (spec or "").split(",") if x.strip()] or list(PAIRS)
    bad = [x for x in want if x not in PAIRS]
    if bad:
        raise SystemExit(f"[pmi] 모르는 앵커 쌍 {bad} — 가능: {', '.join(PAIRS)}")
    return [p for p in PAIRS if p in set(want)]


# ── PMI 산술 ────────────────────────────────────────────────────────────────────
def _fin(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def pmi_values(lp_pos: dict, lp_neg: dict) -> dict:
    """문맥→logp 두 벌 → {pmi_open, pmi_close, pmi_placebo, shift_real, shift_placebo}.
    한쪽이라도 없으면(채점 실패·길이 초과) 그 PMI 는 NaN 이고 파생 shift 도 NaN 이다."""
    out: dict = {}
    for c in CTXS:
        a, b = lp_pos.get(c), lp_neg.get(c)
        out[f"pmi_{c}"] = (float(a) - float(b)) if (_fin(a) and _fin(b)) else _NAN
    out["shift_real"] = out["pmi_close"] - out["pmi_open"]
    out["shift_placebo"] = out["pmi_placebo"] - out["pmi_open"]
    return out


# ── AUC ─────────────────────────────────────────────────────────────────────────
def auc(scores: Sequence[float], labels: Sequence[int]) -> dict:
    """쌍 세기 ROC-AUC(동점 0.5). 한 부류만 있으면 auc = NaN."""
    pos = [float(s) for s, l in zip(scores, labels) if int(l) and _fin(s)]
    neg = [float(s) for s, l in zip(scores, labels) if not int(l) and _fin(s)]
    if not pos or not neg:
        return {"auc": _NAN, "n_pos": len(pos), "n_neg": len(neg), "n_pairs": 0}
    tot = 0.0
    for p in pos:
        for n in neg:
            tot += 1.0 if p > n else (0.5 if p == n else 0.0)
    npair = len(pos) * len(neg)
    return {"auc": tot / npair, "n_pos": len(pos), "n_neg": len(neg), "n_pairs": npair}


def within_problem_auc(rows: Sequence[dict], score_key: str, label_key: str,
                       group_key: str = "group_id") -> dict:
    """**문제 안** 쌍 AUC 를 두 부류가 다 있는 문제들에 걸쳐 합산한 값
    (= 일치 쌍 가중치 합 / 전체 쌍 수). 쓸 수 있는 문제 수도 같이 낸다."""
    groups: dict = {}
    for r in rows:
        if _fin(r.get(score_key)) and r.get(label_key) is not None:
            groups.setdefault(r.get(group_key), []).append(
                (float(r[score_key]), int(r[label_key])))
    conc = 0.0
    npair = 0
    used = 0
    for _, xs in groups.items():
        a = auc([s for s, _ in xs], [l for _, l in xs])
        if not a["n_pairs"]:
            continue
        used += 1
        conc += a["auc"] * a["n_pairs"]
        npair += a["n_pairs"]
    return {"auc": (conc / npair) if npair else _NAN, "n_problems": used, "n_pairs": npair}


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    v = [(float(a), float(b)) for a, b in zip(xs, ys) if _fin(a) and _fin(b)]
    n = len(v)
    if n < 2:
        return _NAN
    mx = sum(a for a, _ in v) / n
    my = sum(b for _, b in v) / n
    sxy = sum((a - mx) * (b - my) for a, b in v)
    sxx = sum((a - mx) ** 2 for a, _ in v)
    syy = sum((b - my) ** 2 for _, b in v)
    if sxx <= 0 or syy <= 0:
        return _NAN
    return sxy / math.sqrt(sxx * syy)


# ── 관문 ────────────────────────────────────────────────────────────────────────
def ci_positive(ci: dict) -> bool:
    """CI 가 0 을 **위쪽으로** 제외하는가(lo > 0). 비었거나 NaN 이면 False."""
    lo = (ci or {}).get("lo", _NAN)
    return bool(_fin(lo) and float(lo) > 0.0)


def make_verdicts(shift_rev: dict, diff_rev: dict, auc_final: dict, *,
                  auc_min: float = AUC_MIN) -> dict:
    """세 관문의 PASS/FAIL(순수 함수 — 테스트가 진리표로 고정한다).

    shift_rev: pair → 수정 행의 SHIFT_real CI
    diff_rev:  pair → 수정 행의 (SHIFT_real − SHIFT_placebo) CI
    auc_final: pair → SHIFT_real 의 문제 안 AUC(최종 정오)
    각 절은 **주어진 모든 pair** 가 통과해야 PASS 다(pair 가 하나도 없으면 FAIL)."""
    a = {p: ci_positive(c) for p, c in (shift_rev or {}).items()}
    b = {p: ci_positive(c) for p, c in (diff_rev or {}).items()}
    c = {p: bool(_fin(v) and float(v) >= float(auc_min))
         for p, v in (auc_final or {}).items()}
    out = {
        "a_shift_real_positive_on_revised": {"by_pair": a, "pass": bool(a) and all(a.values())},
        "b_content_not_presence": {"by_pair": b, "pass": bool(b) and all(b.values())},
        "c_within_problem_auc_final_correct": {"by_pair": c, "auc_min": float(auc_min),
                                               "pass": bool(c) and all(c.values())},
    }
    out["all_pass"] = all(out[k]["pass"] for k in out)
    return out


# ── 행 계획 · 재개 ──────────────────────────────────────────────────────────────
def group_rollouts(rolls: Sequence[dict], *, limit_problems: int = 0) -> list[tuple]:
    groups: dict = {}
    order: list = []
    for r in rolls:
        gid = r.get("group_id", r.get("problem_id"))
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(r)
    if limit_problems and limit_problems > 0:
        order = order[:limit_problems]
    return [(g, groups[g]) for g in order]


def plan_rows(rolls: Sequence[dict], *, limit_problems: int = 0) -> list[dict]:
    """texts.jsonl 행들 → 채점 계획(문제 순서 보존). `roll_id` = f"{group_id}#{문제 안 순번}".
    문제 단위 재료(decoy / majority / 합의 상태)를 행마다 붙인다."""
    out: list[dict] = []
    for gid, rows in group_rollouts(rolls, limit_problems=limit_problems):
        firsts = [first_boxed(r.get("text", "")) for r in rows]
        finals = [(r.get("final_answer") or last_boxed(r.get("text", ""))) for r in rows]
        gold = str(rows[0].get("gold", ""))
        decoy = decoy_answer(firsts, gold)
        agree = agreement_state(finals, k=len(rows))
        maj, state = agree["dominant_answer"], agree["state"]
        for j, r in enumerate(rows):
            out.append({"roll_id": f"{gid}#{j}", "group_id": gid,
                        "problem_id": r.get("problem_id", gid), "problem": r.get("problem", ""),
                        "gold": gold, "text": r.get("text", ""),
                        "r_corr": int(r.get("r_corr", 0) or 0),
                        "truncated": int(r.get("truncated", 0) or 0),
                        "decoy": decoy, "majority": maj, "agree_state": state})
    return out


def load_done(path: Path) -> dict:
    """rows.jsonl → {roll_id: row}. 없으면 빈 dict(재개 규약은 roll_id 하나다)."""
    done: dict = {}
    p = Path(path)
    if not p.exists():
        return done
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[r["roll_id"]] = r
    return done


def _correct(answer: str, gold: str) -> int:
    """답 문자열의 정오 — `grade_math` 에 박스로 싸서 넘긴다(맨 문자열 파싱 실패 회피)."""
    a = str(answer or "").strip()
    return grade_math("\\boxed{" + a + "}", gold) if a else 0


def score_row(tok, plan: dict, variant: str, pairs: Sequence[str],
              scorer: Callable[[Sequence[tuple]], Sequence], *,
              close_at: str = "before_last_box") -> dict:
    """한 행을 채점해 rows.jsonl 한 줄을 만든다. `scorer(reqs)` 는 [(ctx_ids, target_ids)] 를
    받아 **대상 구간 logp 합**(못 재면 None) 리스트를 돌려준다(테스트는 가짜 채점기를 넣는다)."""
    row: dict = {"roll_id": plan["roll_id"], "group_id": plan["group_id"],
                 "problem_id": plan.get("problem_id"), "gold": plan.get("gold", ""),
                 "r_corr": int(plan.get("r_corr", 0)), "truncated": int(plan.get("truncated", 0)),
                 "agree_state": plan.get("agree_state", "NOANS"),
                 "decoy": plan.get("decoy", ""), "majority": plan.get("majority", "")}
    ctx = row_contexts(tok, variant, plan["problem"], plan.get("text", ""), close_at=close_at)
    if ctx is None:
        row.update({"skip": "no_box", "revised": False, "first_answer": "", "last_answer": "",
                    "first_correct": 0, "last_correct": 0})
        return row
    row.update({k: ctx[k] for k in ("first_answer", "last_answer", "revised", "n_boxes",
                                    "n_tok_open", "n_tok_close", "n_tok_placebo",
                                    "seg_tokens", "placebo_tokens", "placebo_ratio")})
    row["skip"] = ""
    row["first_correct"] = _correct(ctx["first_answer"], plan.get("gold", ""))
    row["last_correct"] = _correct(ctx["last_answer"], plan.get("gold", ""))

    want, skip = pair_answers(plan, ctx["first_answer"], pairs)
    keys: list[tuple] = []
    reqs: list[tuple] = []
    seen: dict = {}
    # ★요청 키는 (문맥, pos, neg, 역할) — 두 후보의 채점 자리가 **쌍 안에서** 맞춰지므로
    #   같은 답이라도 쌍이 다르면 같은 요청이 아니다(쌍이 같으면 중복 제거된다).
    for pos, neg in want.values():
        for c in CTXS:
            if (c, pos, neg) in seen:
                continue
            seen[(c, pos, neg)] = 1
            rp, rn = pair_target_ids(tok, ctx[f"ctx_{c}"], pos, neg)
            keys += [(c, pos, neg, "pos"), (c, pos, neg, "neg")]
            reqs += [rp, rn]
    got = list(scorer(reqs)) if reqs else []
    lp = {k: v for k, v in zip(keys, got)}
    for p in pairs:
        if p in skip:
            row[f"{p}_skip"] = skip[p]
            continue
        pos, neg = want[p]
        vals = pmi_values({c: lp.get((c, pos, neg, "pos")) for c in CTXS},
                          {c: lp.get((c, pos, neg, "neg")) for c in CTXS})
        row[f"{p}_skip"] = "" if all(_fin(vals[f"pmi_{c}"]) for c in CTXS) else "too_long"
        row[f"{p}_pos"], row[f"{p}_neg"] = pos, neg
        for k, v in vals.items():
            row[f"{p}_{k}"] = v
    return row


# ── 요약 ────────────────────────────────────────────────────────────────────────
def with_derived(rows: Sequence[dict], pairs: Sequence[str]) -> list[dict]:
    """행마다 짝 차이 `{pair}_shift_diff` = SHIFT_real − SHIFT_placebo 를 붙인다."""
    out = []
    for r in rows:
        r = dict(r)
        for p in pairs:
            a, b = r.get(f"{p}_shift_real"), r.get(f"{p}_shift_placebo")
            r[f"{p}_shift_diff"] = (float(a) - float(b)) if (_fin(a) and _fin(b)) else _NAN
        # goldx 의 majority_correct/majority_wrong 분할용 — M = 그 문제 최종 답의 다수.
        r["majority_correct"] = _correct(r.get("majority", ""), r.get("gold", ""))
        out.append(r)
    return out


def _cell(r: dict) -> str:
    return f"{int(r.get('first_correct', 0))}->{int(r.get('last_correct', 0))}"


def split_summary(rows: Sequence[dict], key: str, *, seed: int = 0, n_boot: int = 2000) -> dict:
    """한 지표의 분할 요약 — 전체 / 수정 여부 / 합의 상태 / (first_correct, last_correct) 칸."""
    def ci(sub, off):
        return bootstrap_ci([r.get(key) for r in sub], seed=seed + off, n_boot=n_boot)

    out = {"all": ci(rows, 0),
           "revised": ci([r for r in rows if r.get("revised")], 1),
           "not_revised": ci([r for r in rows if not r.get("revised")], 2)}
    out["by_agree_state"] = {
        st: ci([r for r in rows if r.get("agree_state") == st], 3 + i)
        for i, st in enumerate(AGREE_STATES)
        if any(r.get("agree_state") == st for r in rows)}
    cells = sorted({_cell(r) for r in rows})
    out["by_outcome_cell"] = {c: ci([r for r in rows if _cell(r) == c], 20 + i)
                              for i, c in enumerate(cells)}
    return out


def reward_summary(rows: Sequence[dict], pair: str, *, seed: int = 0, n_boot: int = 2000) -> dict:
    """`pmi_shift_reward` 기본 설정의 분포 + 실제 결과 변화와의 상관."""
    vals: list[float] = []
    deltas: list[float] = []
    n_save = n_derail = 0
    for r in rows:
        o, c = r.get(f"{pair}_pmi_open"), r.get(f"{pair}_pmi_close")
        if not (_fin(o) and _fin(c)):
            continue
        vals.append(pmi_shift_reward(float(o), float(c)))
        deltas.append(int(r.get("last_correct", 0)) - int(r.get("first_correct", 0)))
        n_save += int(float(o) < 0.0 < float(c))
        n_derail += int(float(c) < 0.0 < float(o))
    if not vals:
        return {"n": 0}
    sv = sorted(vals)
    by_delta = {}
    for d in (-1, 0, 1):
        xs = [v for v, dd in zip(vals, deltas) if dd == d]
        by_delta[str(d)] = {"n": len(xs), "mean": (sum(xs) / len(xs)) if xs else _NAN}
    return {"n": len(vals), "mean": sum(vals) / len(vals), "median": sv[len(sv) // 2],
            "min": sv[0], "max": sv[-1], "ci": bootstrap_ci(vals, seed=seed, n_boot=n_boot),
            "n_positive": sum(1 for v in vals if v > 0), "n_zero": sum(1 for v in vals if v == 0),
            "n_negative": sum(1 for v in vals if v < 0),
            "n_save": n_save, "n_derail": n_derail,
            "pearson_r_with_outcome_change": pearson(vals, deltas),
            "mean_reward_by_outcome_change": by_delta}


def combo_reward_summary(rows: Sequence[dict], pair: str) -> dict:
    """gold+majority 조합 보상(오프라인 집계 전용 — `pmi_shift_reward` 를 다른 인자로 두 번
    부를 뿐 새 보상 코드는 아니다): 다수결이 **틀린**(majority_wrong) 행은
    `reversal_save=2.0`(반전에 더 큰 상을 준다 — 틀린 다수를 뒤집는 반전을 더 원한다),
    다수결이 **맞은**(majority_correct) 행은 기본 `reversal_save=1.0`. 두 몫의 평균이
    "combo reward"다."""
    def vals(sub, **kw):
        out = []
        for r in sub:
            o, c = r.get(f"{pair}_pmi_open"), r.get(f"{pair}_pmi_close")
            if _fin(o) and _fin(c):
                out.append(pmi_shift_reward(float(o), float(c), **kw))
        return out

    wrong = [r for r in rows if not r.get("majority_correct")]
    correct = [r for r in rows if r.get("majority_correct")]
    v_wrong = vals(wrong, reversal_save=2.0)
    v_correct = vals(correct, reversal_save=1.0)
    v_all = v_wrong + v_correct
    return {
        "n_majority_wrong": len(v_wrong),
        "mean_majority_wrong_reversal_save2": (sum(v_wrong) / len(v_wrong)) if v_wrong else _NAN,
        "n_majority_correct": len(v_correct),
        "mean_majority_correct_reversal_save1": ((sum(v_correct) / len(v_correct))
                                                 if v_correct else _NAN),
        "n_combo": len(v_all),
        "combo_mean": (sum(v_all) / len(v_all)) if v_all else _NAN,
    }


def pair_summary(rows: Sequence[dict], pair: str, *, seed: int = 0, n_boot: int = 2000) -> dict:
    sub = [r for r in rows if not r.get("skip") and not r.get(f"{pair}_skip")
           and _fin(r.get(f"{pair}_shift_real"))]
    reasons: dict = {}
    for r in rows:
        why = r.get("skip") or r.get(f"{pair}_skip") or ""
        if why:
            reasons[why] = reasons.get(why, 0) + 1
    out: dict = {"n_scored": len(sub), "n_revised": sum(1 for r in sub if r.get("revised")),
                 "n_skipped": sum(reasons.values()), "skip_reasons": reasons}
    out["shift_real"] = split_summary(sub, f"{pair}_shift_real", seed=seed, n_boot=n_boot)
    out["shift_placebo"] = split_summary(sub, f"{pair}_shift_placebo", seed=seed + 50,
                                         n_boot=n_boot)
    out["shift_real_minus_placebo"] = split_summary(sub, f"{pair}_shift_diff", seed=seed + 100,
                                                    n_boot=n_boot)
    out["pmi_open"] = bootstrap_ci([r.get(f"{pair}_pmi_open") for r in sub], seed=seed + 150,
                                   n_boot=n_boot)
    out["pmi_close"] = bootstrap_ci([r.get(f"{pair}_pmi_close") for r in sub], seed=seed + 151,
                                    n_boot=n_boot)
    # ★관문 (c) 재정의(0918): 수정하지 않은 행은 SHIFT_real ≡ 0 이라 «최종 정오»와 무관하게
    #   섞여 들어와 AUC 를 구조적으로 깬다. **수정한 행만**(문제당 ≥2 개 ∧ 정오가 섞인 문제)
    #   으로 잰 것이 관문이고, 전 행 AUC 는 참고용으로만 남긴다.
    revised_sub = [r for r in sub if r.get("revised")]
    out["auc_final_correct"] = {
        "within_problem_revised": within_problem_auc(revised_sub, f"{pair}_shift_real",
                                                      "last_correct"),
        "within_problem_all_rows_informational": within_problem_auc(
            sub, f"{pair}_shift_real", "last_correct"),
        "pooled": auc([r.get(f"{pair}_shift_real") for r in sub],
                      [int(r.get("last_correct", 0)) for r in sub])}
    out["auc_revised"] = {
        "within_problem": within_problem_auc(sub, f"{pair}_shift_real", "revised_int"),
        "pooled": auc([r.get(f"{pair}_shift_real") for r in sub],
                      [int(bool(r.get("revised"))) for r in sub])}
    out["reward"] = reward_summary(sub, pair, seed=seed + 200, n_boot=n_boot)
    if pair == "goldx":
        # ★1c: goldx 만 다수결 정오로 SHIFT_real 을 추가로 쪼갠다(gold+majority 조합 축).
        out["shift_real_by_majority"] = {
            "majority_correct": split_summary(
                [r for r in sub if r.get("majority_correct")], f"{pair}_shift_real",
                seed=seed + 300, n_boot=n_boot),
            "majority_wrong": split_summary(
                [r for r in sub if not r.get("majority_correct")], f"{pair}_shift_real",
                seed=seed + 350, n_boot=n_boot),
        }
        out["combo_reward"] = combo_reward_summary(sub, pair)
    return out


def summarize(rows: Sequence[dict], pairs: Sequence[str], *, seed: int = 0,
              n_boot: int = 2000) -> dict:
    rows = with_derived(rows, pairs)
    for r in rows:
        r["revised_int"] = int(bool(r.get("revised")))
    scored = [r for r in rows if not r.get("skip")]
    out: dict = {"counts": {
        "n_rows": len(rows), "n_scored_rows": len(scored),
        "n_revised": sum(1 for r in scored if r.get("revised")),
        "n_rows_no_box": sum(1 for r in rows if r.get("skip") == "no_box"),
        "n_problems": len({r.get("group_id") for r in rows}),
        "placebo_ratio_mean": (sum(float(r["placebo_ratio"]) for r in scored
                                   if _fin(r.get("placebo_ratio")))
                               / max(1, sum(1 for r in scored if _fin(r.get("placebo_ratio"))))),
        "n_placebo_out_of_tol": sum(1 for r in scored if _fin(r.get("placebo_ratio"))
                                    and abs(float(r["placebo_ratio"]) - 1.0) > PLACEBO_TOL),
    }, "pairs": {}}
    for i, p in enumerate(pairs):
        out["pairs"][p] = pair_summary(rows, p, seed=seed + 1000 * (i + 1), n_boot=n_boot)
    out["verdicts"] = make_verdicts(
        {p: out["pairs"][p]["shift_real"]["revised"] for p in pairs},
        {p: out["pairs"][p]["shift_real_minus_placebo"]["revised"] for p in pairs},
        {p: out["pairs"][p]["auc_final_correct"]["within_problem_revised"]["auc"]
         for p in pairs})
    return out


def _f(x, nd: int = 4) -> str:
    try:
        return "nan" if x != x else f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "nan"


def _ci_s(ci: dict) -> str:
    ci = ci or {}
    return f"{_f(ci.get('mean'))} [{_f(ci.get('lo'))}, {_f(ci.get('hi'))}] (n={ci.get('n', 0)})"


def to_markdown(summ: dict) -> str:
    c = summ["counts"]
    close_at = (summ.get("meta") or {}).get("close_at", "before_last_box")
    L = ["# math_pmi_shift_probe — 자발적 수정 구간의 믿음 이동(PMI shift)", "",
         f"- close_at = `{close_at}` — CTX_CLOSE 는 "
         + ("마지막 \\boxed{ **직전**(답 미포함, 추론만)" if close_at == "before_last_box"
            else "마지막 박스 **끝**(답 포함, 구 동작)"),
         f"- 행 {c['n_rows']}개(문제 {c['n_problems']}개) · 채점 {c['n_scored_rows']}개 · "
         f"수정 {c['n_revised']}개 · 박스 없음 {c['n_rows_no_box']}개",
         f"- placebo 길이비 평균 {_f(c['placebo_ratio_mean'], 3)} · ±{PLACEBO_TOL:.0%} 밖 "
         f"{c['n_placebo_out_of_tol']}행", ""]
    for p, s in summ["pairs"].items():
        L += [f"## 앵커 쌍: {p}", "",
              f"- 채점 {s['n_scored']}행(수정 {s['n_revised']}) · 건너뜀 {s['n_skipped']} "
              f"{s['skip_reasons']}",
              f"- PMI_open {_ci_s(s['pmi_open'])} · PMI_close {_ci_s(s['pmi_close'])}", "",
              "| 지표 | all | revised | not_revised |", "|---|---|---|---|"]
        for key, name in (("shift_real", "SHIFT_real"), ("shift_placebo", "SHIFT_placebo"),
                          ("shift_real_minus_placebo", "real − placebo")):
            d = s[key]
            L.append(f"| {name} | {_ci_s(d['all'])} | {_ci_s(d['revised'])} | "
                     f"{_ci_s(d['not_revised'])} |")
        L += ["", "### SHIFT_real — 합의 상태별 / 결과 칸별", "",
              "| 분할 | CI |", "|---|---|"]
        for st, ci in s["shift_real"]["by_agree_state"].items():
            L.append(f"| state {st} | {_ci_s(ci)} |")
        for cell, ci in s["shift_real"]["by_outcome_cell"].items():
            L.append(f"| (first,last) {cell} | {_ci_s(ci)} |")
        a1, a2 = s["auc_final_correct"], s["auc_revised"]
        L += ["", "### AUC (SHIFT_real)", "",
              "| 대상 | 문제 안(관문) | (문제 수) | 문제 안(전 행, 참고) | (문제 수) | pooled |",
              "|---|---|---|---|---|---|",
              f"| 최종 정오 | {_f(a1['within_problem_revised']['auc'], 3)} | "
              f"{a1['within_problem_revised']['n_problems']} | "
              f"{_f(a1['within_problem_all_rows_informational']['auc'], 3)} | "
              f"{a1['within_problem_all_rows_informational']['n_problems']} | "
              f"{_f(a1['pooled']['auc'], 3)} |",
              f"| revised | {_f(a2['within_problem']['auc'], 3)} | "
              f"{a2['within_problem']['n_problems']} | - | - | "
              f"{_f(a2['pooled']['auc'], 3)} |", ""]
        r = s["reward"]
        L += ["### pmi_shift_reward (기본 설정)", "",
              f"- n={r.get('n', 0)} · mean {_f(r.get('mean'))} · median {_f(r.get('median'))} · "
              f"[{_f(r.get('min'))}, {_f(r.get('max'))}]",
              f"- 부호 +{r.get('n_positive', 0)} / 0 {r.get('n_zero', 0)} / "
              f"−{r.get('n_negative', 0)} · save {r.get('n_save', 0)} · "
              f"derail {r.get('n_derail', 0)}",
              f"- 결과 변화(last−first)와의 상관 r = "
              f"{_f(r.get('pearson_r_with_outcome_change'), 3)} · 칸별 평균 보상 "
              f"{ {k: _f(v['mean'], 3) for k, v in r.get('mean_reward_by_outcome_change', {}).items()} }",
              ""]
        if p == "goldx" and "shift_real_by_majority" in s:
            mb = s["shift_real_by_majority"]
            cr = s["combo_reward"]
            L += ["### goldx — 다수결 정오로 SHIFT_real 분할 · combo reward", "",
                  "| 다수결 | SHIFT_real CI |", "|---|---|",
                  f"| majority_correct | {_ci_s(mb['majority_correct']['all'])} |",
                  f"| majority_wrong | {_ci_s(mb['majority_wrong']['all'])} |", "",
                  f"- combo reward(다수 틀림→reversal_save=2.0 · 다수 맞음→기본 1.0): "
                  f"mean {_f(cr['combo_mean'])} (n={cr['n_combo']}) — "
                  f"틀림 {_f(cr['mean_majority_wrong_reversal_save2'])} "
                  f"(n={cr['n_majority_wrong']}) · 맞음 "
                  f"{_f(cr['mean_majority_correct_reversal_save1'])} "
                  f"(n={cr['n_majority_correct']})", ""]
    v = summ["verdicts"]
    L += ["## VERDICT", "", "| 관문 | 판정 | pair별 |", "|---|---|---|"]
    for k, name in (("a_shift_real_positive_on_revised", "(a) 수정 행 SHIFT_real > 0, CI 0 제외"),
                    ("b_content_not_presence", "(b) SHIFT_real − SHIFT_placebo CI 0 제외"),
                    ("c_within_problem_auc_final_correct",
                     f"(c) 수정 행만의 문제 안 AUC(최종 정오) ≥ {AUC_MIN:.2f}")):
        L.append(f"| {name} | {'PASS' if v[k]['pass'] else 'FAIL'} | "
                 f"{ {p: ('PASS' if ok else 'FAIL') for p, ok in v[k]['by_pair'].items()} } |")
    L += ["", f"**ALL: {'PASS' if v['all_pass'] else 'FAIL'}**", ""]
    return "\n".join(L) + "\n"


# ── 채점 백엔드 ─────────────────────────────────────────────────────────────────
def make_vllm_scorer(llm, *, max_model_len: int = MAX_MODEL_LEN,
                     batch_size: int = 512) -> Callable:
    """teacher-forced logp 합 채점기 — 문맥+대상을 한 시퀀스로 넣고 `prompt_logprobs=0`
    으로 대상 구간의 실제 토큰 logprob 을 더한다(scripts/ruler_battery.score_logp 규약).
    너무 긴 시퀀스는 None(조용한 절단 없음)."""
    from vllm import SamplingParams  # noqa: PLC0415
    from vllm.inputs import TokensPrompt  # noqa: PLC0415

    sp = SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, prompt_logprobs=0,
                        detokenize=False)

    def score(reqs: Sequence[tuple]) -> list:
        out: list = [None] * len(reqs)
        todo = [i for i, (c, t) in enumerate(reqs)
                if t and len(c) + len(t) + 2 <= max_model_len]
        for lo in range(0, len(todo), batch_size):
            idxs = todo[lo:lo + batch_size]
            prompts = [TokensPrompt(prompt_token_ids=list(reqs[i][0]) + list(reqs[i][1]))
                       for i in idxs]
            for i, r in zip(idxs, llm.generate(prompts, sp, use_tqdm=False)):
                c, t = reqs[i]
                plp = r.prompt_logprobs
                if plp is None or len(plp) != len(c) + len(t):
                    raise RuntimeError(
                        f"[pmi] prompt_logprobs 길이 불일치: "
                        f"{None if plp is None else len(plp)} vs {len(c) + len(t)}")
                s = 0.0
                for pos, tid in enumerate(t):
                    d = plp[len(c) + pos]
                    if d is None or tid not in d:
                        raise RuntimeError(f"[pmi] 위치 {pos} 의 토큰 {tid} 이 "
                                           "prompt_logprobs 에 없다")
                    s += d[tid].logprob
                out[i] = s
        return out

    return score


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_name", required=True)
    ap.add_argument("--rollouts", default=ROLLOUTS_DEFAULT)
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--variant", default="math_opt")
    ap.add_argument("--gpu_util", type=float, default=0.45)
    ap.add_argument("--limit_problems", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512,
                    help="한 판에 넣을 채점 시퀀스 수 — 판마다 rows.jsonl 에 방출·flush 한다")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--pairs", default="gold,self,goldx")
    ap.add_argument("--close_at", default="before_last_box", choices=list(CLOSE_AT_CHOICES),
                    help="CTX_CLOSE 경계 — before_last_box(기본, 신규): 마지막 \\boxed{ 앞. "
                         "after_last_box(구 동작): 마지막 박스 끝(답 포함, 동어반복).")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()

    pairs = resolve_pairs(a.pairs)
    selftest_math_verify()
    out = Path(OUT_ROOT) / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"

    rolls = [json.loads(l) for l in open(a.rollouts) if l.strip()]
    plans = plan_rows(rolls, limit_problems=a.limit_problems)
    done = load_done(rows_path)
    pending = [p for p in plans if p["roll_id"] not in done]
    print(f"[pmi] 행 {len(plans)}개 · 앵커 {pairs} · 이미 있음 {len(done)}개 · "
          f"채점할 행 {len(pending)}개", flush=True)

    if pending:
        from vllm import LLM  # noqa: PLC0415
        llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
                  gpu_memory_utilization=a.gpu_util, max_model_len=MAX_MODEL_LEN,
                  enforce_eager=True)
        tok = llm.get_tokenizer()
        scorer = make_vllm_scorer(llm, max_model_len=MAX_MODEL_LEN, batch_size=max(1, a.chunk))
        fh = rows_path.open("a")

        def flush(batch: Sequence[dict]) -> None:
            for plan in batch:
                row = score_row(tok, plan, a.variant, pairs, scorer, close_at=a.close_at)
                done[row["roll_id"]] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[pmi] {len(done)}/{len(plans)} 행", flush=True)

        # 한 판 = 대략 `--chunk` 개 채점 시퀀스(행마다 최대 3문맥 × 4답 = 12 요청)
        per_plan = max(1, len(pairs) * 2 * len(CTXS))
        step = max(1, int(a.chunk) // per_plan)
        for lo in range(0, len(pending), step):
            flush(pending[lo:lo + step])
        fh.close()

    rows = [done[p["roll_id"]] for p in plans if p["roll_id"] in done]
    summ = summarize(rows, pairs, seed=a.seed, n_boot=a.n_boot)
    summ["meta"] = {"rollouts": a.rollouts, "model_path": a.model_path, "variant": a.variant,
                    "pairs": pairs, "seed": a.seed, "limit_problems": a.limit_problems,
                    "placebo_sent": PLACEBO_SENT, "answer_tmpl": ANSWER_TMPL,
                    "max_model_len": MAX_MODEL_LEN, "close_at": a.close_at}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    md = to_markdown(summ)
    (out / "summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
