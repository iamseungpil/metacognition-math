"""메타 갈아끼우기 프로브 — «혼잣말이 실제 정답률을 바꿨는가»의 인과 라벨 생성.

목적. 롤아웃 중간의 <meta> 블록(혼잣말)을 (a) 그대로 두고(orig), (b) 지우고(abl),
(c) «다른 문제»의 메타로 통째 갈아끼운 뒤(ctrl) 이어 생성해, 그 혼잣말이 이후
정답률을 실제로 바꿨는지 site 단위 Δ 를 잰다. stageA(넓고 얕게 K=6) → stageB
(상·하위 200 site 만 K=32 재측정, 선별과 측정을 분리) → summarize(클러스터
부트스트랩 CI + good/null 라벨).

순수 vLLM 0.10.2 오프라인. verl 은 import 하지 않는다.

샘플링 파라미터 — $W/run_countdown_arm.sh 실측값 (2026-08-27 grep):
    L179  actor_rollout_ref.rollout.temperature=1.0
    L180  actor_rollout_ref.rollout.top_k=-1
    L181  actor_rollout_ref.rollout.top_p=1.0
    L187  data.max_response_length=3072
    L112  ENABLE_THINKING="${ENABLE_THINKING:-false}"   → enable_thinking=False
    L178  actor_rollout_ref.rollout.n=8                 → 문제당 8롤아웃
chat template 은 scripts/countdown_gs0_eval.py 의 chat() 과 동일하게
apply_chat_template(..., add_generation_prompt=True, enable_thinking=False).

GPU 는 CUDA_VISIBLE_DEVICES 환경변수로만 지정받는다 — 이 스크립트는 set 하지 않는다.
실행: PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=k HF_HUB_OFFLINE=1 $PY scripts/meta_splice_probe.py stageA --shard 0 --nshards 4
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# vLLM 0.10.2 v1 EngineCore 는 기본 fork 로 뜨는데, 부모 프로세스에 CUDA 컨텍스트가
# 이미 잡혀 있으면 자식이 "Cannot re-initialize CUDA in forked subprocess" 로 죽는다
# (2026-08-27 smoke 실측). vllm import 전에 spawn 을 강제해 회피한다.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np

# ── 단일 정의처에서 가져온다 (복제 금지) ─────────────────────────────────────
# ★2026-09-02: REPO 를 실제로 sys.path 에 넣는다 (이전엔 docstring 에만 있어 harvest 3건이 죽었다).
#   site-packages 에 `scripts` 라는 정규 패키지가 있어 `scripts.countdown_gs0_eval` 은 항상 그쪽으로
#   풀린다(정규 패키지 > 네임스페이스). 그래서 scripts/ 디렉터리를 직접 path 에 넣고 모듈명으로 부른다.
_REPO = os.environ.get("REPO", str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, _REPO); sys.path.insert(0, os.path.join(_REPO, "scripts"))
from countdown_gs0_eval import _solves, _RESCUE_EXPR  # noqa: E402
from src.training.countdown_rewards import parse_meta          # noqa: E402

# ── 상수 (지시된 경로) ─────────────────────────────────────────────────────
W = "/home/jovyan/beomi/splee"
MODEL = os.environ.get("PROBE_MODEL",
        "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
        "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
TRAIN = f"{W}/metacognition-math/hf_data/metacot-sdc-data/countdown_train_4num_new.parquet"
OUT = Path(os.environ.get("PROBE_OUT", f"{W}/cd6_work/probe"))
OUT.mkdir(parents=True, exist_ok=True)

# run_countdown_arm.sh 실측값 (모듈 docstring 의 grep 근거)
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = -1
MAX_RESPONSE_LENGTH = 3072
ENABLE_THINKING = False
N_ROLLOUTS = 8

MAX_NEW_TOKENS = 600          # 이어생성 길이
MAX_MODEL_LEN = 5120          # 프롬프트(~0.5k) + prefix(≤3072) + 600 + 여유
N_PROBLEMS = int(os.environ.get("PROBE_N_PROBLEMS", "330"))
# PROBE_DISJOINT=1 → 기존 330문제와 «겹치지 않는» 새 문제만 표집 (표본 확장용)
DISJOINT = os.environ.get("PROBE_DISJOINT", "") == "1"
SITES_PER_PROBLEM = 3
SITES_PER_SHARD = 300
POST_SITES_MAX = 60
POS_LO, POS_HI = 0.05, 0.85
DONOR_RATIO = (0.7, 1.3)

_FOUND_CLAIM = re.compile(r"found|valid solution|correct (?:solution|expression)", re.I)
_ANNOUNCE = re.compile(r"[^.\n]*\b(metacognitive|meta block|<meta>|reflect on my approach|assess (?:my|the) approach|pause for metacognition|stop and (?:reflect|write))[^.\n]*[.:]?\s*$", re.I)


# ══ 결정적 난수/파싱 유틸 ═══════════════════════════════════════════════════
def det_seed(*parts) -> int:
    """정수해시 — 실행 순서·프로세스와 무관하게 (site_id, cond, k) 만으로 결정."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - 1)


def last_boxed(text: str):
    r"""마지막 \boxed{...} 의 내용물(중괄호 균형 파싱). 없으면 None."""
    key = "\\boxed{"
    idx = text.rfind(key)
    while idx != -1:
        i = idx + len(key)
        depth, j = 1, i
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            return text[i:j - 1].strip()
        idx = text.rfind(key, 0, idx)
    return None


def rollout_correct(text: str, nums, target) -> int:
    expr = last_boxed(text)
    return int(expr is not None and _solves(expr, nums, target))


def solved_in_prefix(prefix: str, nums, target) -> int:
    for mm in _RESCUE_EXPR.finditer(prefix):
        if _solves(mm.group(0).strip().rstrip("."), nums, target):
            return 1
    return 0


# ══ 데이터/모델 로딩 ═══════════════════════════════════════════════════════
def load_problems():
    """TRAIN 에서 numpy seed 0 으로 330문제 표집. (표집 리스트 내 위치, 행) 목록."""
    import pandas as pd
    df = pd.read_parquet(TRAIN)
    rng = np.random.RandomState(0)
    base = rng.choice(len(df), size=330, replace=False)      # 기존 실험과 동일한 330문제
    if DISJOINT:
        rest = np.setdiff1d(np.arange(len(df)), base)
        dseed = int(os.environ.get("PROBE_DISJOINT_SEED", "1"))     # 1 = probe_p3 와 동일 · 다른 값 = 새 문제 집합(복제용)
        perm = np.random.RandomState(1).permutation(rest)
        if dseed != 1:                                                # probe_p3 의 330문제도 제외한 뒤 새 seed 로 섞는다
            perm = np.random.RandomState(dseed).permutation(perm[330:])
        sel = perm[:N_PROBLEMS]
    else:
        sel = base[:N_PROBLEMS] if N_PROBLEMS <= 330 else rng.choice(
            len(df), size=N_PROBLEMS, replace=False)
    out = []
    for pos, orig_idx in enumerate(sel):
        row = df.iloc[int(orig_idx)]
        out.append({
            "pos": pos,                       # 표집 리스트 내 위치 (샤딩 기준)
            "prob_idx": int(orig_idx),        # 원본 parquet 인덱스
            "nums": [int(v) for v in row["nums"]],
            "target": int(row["target"]),
            "witness": str(row.get("witness", "")),
            "messages": [{"role": m["role"], "content": m["content"]}
                         for m in row["prompt"]],
        })
    return out


def build_llm(seed: int = 0):
    from vllm import LLM
    llm = LLM(model=MODEL, dtype="bfloat16", seed=seed,
              gpu_memory_utilization=float(os.environ.get("PROBE_GPU_UTIL", "0.85")),
              max_model_len=MAX_MODEL_LEN,
              # countdown_gs0_eval 과 같은 이유(공용 서버 CPU 경합)로 eager.
              enforce_eager=True)
    return llm, llm.get_tokenizer()


def chat_text(tok, msgs) -> str:
    """countdown_gs0_eval.chat() 과 동일 — enable_thinking=False 포함."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=ENABLE_THINKING)
    except TypeError:                          # enable_thinking 없는 토크나이저
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def chat_token_ids(tok, msgs):
    return tok.encode(chat_text(tok, msgs), add_special_tokens=False)


# ══ 이어생성 공통부 ════════════════════════════════════════════════════════
def build_cont_requests(tok, jobs):
    """jobs: [(site_row, cond, prefix_text, k, seed)] → (TokensPrompt, SamplingParams) 리스트.

    문맥은 토큰 id 로 구성한다: chat 프롬프트 토큰 + prefix 토큰 → TokensPrompt.
    (텍스트로 다시 넣으면 chat template 이 이중 적용된다.)
    """
    from vllm import SamplingParams, TokensPrompt
    prompts, params, keep = [], [], []
    ptok_cache = {}
    for job in jobs:
        site, cond, prefix, k, seed = job
        key = site["site_id"].rsplit("_r", 1)[0]
        if key not in ptok_cache:
            ptok_cache[key] = chat_token_ids(tok, json.loads(site["prompt_json"]))
        ids = ptok_cache[key] + tok.encode(prefix, add_special_tokens=False)
        if len(ids) + MAX_NEW_TOKENS >= MAX_MODEL_LEN:
            continue                          # 문맥 초과 — 이 요청은 버린다
        prompts.append(TokensPrompt(prompt_token_ids=ids))
        # ★0902 라벨 수정: BAN_META=1 이면 abl/ctrl(메타를 지운·바꾼 조건)에서 새 <meta> 재발화를 금지한다.
        #   실측: 블록 없는 이어쓰기의 40% 가 550자 뒤에 새 메타를 썼다 → L2 가 «이 메타 vs 새 메타» 로 희석됐다.
        _ban = ({"bad_words": ["<meta", " <meta", "\n<meta", "<meta>", "<META"]}
                if os.environ.get("BAN_META", "") == "1" and cond in ("abl", "ctrl") else {})
        params.append(SamplingParams(
            n=1, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            max_tokens=MAX_NEW_TOKENS, seed=seed, **_ban))
        keep.append(job)
    return prompts, params, keep


def prefix_of(site, cond: str) -> str:
    """조건별 문맥. orig=prefix 그대로 · abl=메타 삭제 · ctrl=donor 메타로 교체."""
    text = site["response_text"]
    s, e = int(site["meta_start"]), int(site["meta_end"])
    if cond == "orig":
        return text[:e]
    if cond == "abl":
        pre = text[:s]
        if os.environ.get("STRIP_ANNOUNCE", "") == "1":
            # ★0902 라벨 수정 3차: «이제 메타를 쓰겠다»류 예고 문장까지 지운다 — 예고가 남으면 토큰을 금지해도
            #   모델이 "(meta)/ meta" 로 우회해 같은 블록을 다시 쓴다(실측 눈검사).
            pre = _ANNOUNCE.sub("", pre).rstrip() + "\n\n"
        return pre
    if cond == "ctrl":
        return text[:s] + site["donor_raw"]
    raise ValueError(cond)


def run_continuations(llm, tok, jobs, desc: str):
    """jobs 를 한 배치로 생성, 행 [(site_id, cond, k, correct, gen_len)] 반환."""
    prompts, params, keep = build_cont_requests(tok, jobs)
    print(f"[{desc}] {len(keep)}/{len(jobs)} 요청 (문맥초과 {len(jobs)-len(keep)}건 제외)",
          flush=True)
    if not keep:
        return []
    outs = llm.generate(prompts, params)      # vLLM 자체 tqdm 진행바
    rows = []
    for (site, cond, _prefix, k, _seed), o in zip(keep, outs):
        gen = o.outputs[0]
        rows.append({
            "site_id": site["site_id"], "cond": cond, "k": int(k),
            "correct": rollout_correct(gen.text, site["nums"], site["target"]),
            "gen_len": len(gen.token_ids),
            "has_meta": int("<meta" in gen.text),                     # 0902: 재발화 감시
            "text": gen.text[:1500] if os.environ.get("SAVE_TEXT", "") == "1" else "",
        })
    return rows


# ══ stageA ═════════════════════════════════════════════════════════════════
def pick_donor(site, pool, ratio=DONOR_RATIO):
    """«다른 문제» 사이트의 메타 블록. 길이비 0.7~1.3 우선, 없으면 최근접 길이.

    결정적: det_seed(site_id,'donor') 로만 뽑는다.
    """
    others = [c for c in pool if c["prob_idx"] != site["prob_idx"]]
    if not others:
        return None
    L = max(1, len(site["meta_raw"]))
    band = [c for c in others if ratio[0] <= len(c["meta_raw"]) / L <= ratio[1]]
    rng = np.random.RandomState(det_seed(site["site_id"], "donor"))
    if band:
        return band[int(rng.randint(len(band)))]
    return min(others, key=lambda c: abs(len(c["meta_raw"]) - L))


def stage_a(args):
    import pandas as pd
    from tqdm import tqdm
    from vllm import SamplingParams
    OUT.mkdir(parents=True, exist_ok=True)

    probs = load_problems()
    mine = [p for p in probs if p["pos"] % args.nshards == args.shard]
    k_cont = 2 if args.smoke else 6
    max_sites = 10 if args.smoke else SITES_PER_SHARD
    if args.smoke:
        mine = mine[:6]
    print(f"[stageA] shard {args.shard}/{args.nshards}: 문제 {len(mine)}개", flush=True)

    llm, tok = build_llm(seed=0)

    # ── 1) 문제당 8롤아웃 ──────────────────────────────────────────────────
    # ★0902 프롬프트 축 실험: PROBE_SYS_FILE 이 있으면 시스템 메시지를 그 파일 내용으로 교체 (유저 메시지 동일)
    _sysf = os.environ.get("PROBE_SYS_FILE")
    if _sysf:
        _sys = open(_sysf).read()
        for p in mine:
            assert p["messages"][0]["role"] == "system"
            p["messages"] = [{"role": "system", "content": _sys}] + [dict(m) for m in p["messages"][1:]]
        print(f"[stageA] 시스템 프롬프트 교체: {_sysf} ({len(_sys)}자)", flush=True)
    prompts = [chat_text(tok, p["messages"]) for p in mine]
    params = [SamplingParams(n=N_ROLLOUTS, temperature=TEMPERATURE, top_p=TOP_P,
                             top_k=TOP_K, max_tokens=MAX_RESPONSE_LENGTH,
                             seed=det_seed("rollout", p["prob_idx"]))
              for p in mine]
    outs = llm.generate(prompts, params)

    # ── 2) 성공수 1..7 문제만 유지 + 사이트 수확 ────────────────────────────
    harvested = []
    for p, o in tqdm(list(zip(mine, outs)), desc="harvest"):
        texts = [x.text for x in o.outputs]
        corr = [rollout_correct(t, p["nums"], p["target"]) for t in texts]
        nc = sum(corr)
        if not (1 <= nc <= N_ROLLOUTS - 1):   # 0/8, 8/8 은 정보 없음
            continue
        for ri, t in enumerate(texts):
            m = parse_meta(t, "new")
            if not m["emitted"]:
                continue
            s, e = m["start"], m["end"]       # 메타 블록 문자 구간 (open..close_end)
            prefix = t[:e]
            body = m["body"] or ""
            harvested.append({
                "site_id": f"p{p['prob_idx']}_r{ri}",
                "prob_idx": p["prob_idx"], "rollout_idx": ri,
                "nums": p["nums"], "target": p["target"], "witness": p["witness"],
                "n_correct_of8": nc,
                "meta_raw": m["raw"], "meta_body": body,
                "confidence": m["confidence"], "decision": m["decision"],
                "meta_start": s, "meta_end": e,
                "pos": e / max(1, len(t)), "meta_len": len(m["raw"]),
                "solved_in_prefix": solved_in_prefix(prefix, p["nums"], p["target"]),
                "found_claim": int(bool(_FOUND_CLAIM.search(body))),
                "response_text": t,
                "prompt_json": json.dumps(p["messages"], ensure_ascii=False),
                "shard": args.shard,
            })
    print(f"[stageA] 수확 사이트 {len(harvested)}개", flush=True)

    # ── 3) 선별 ───────────────────────────────────────────────────────────
    #  post=1 (prefix 에 이미 정답 or '찾았다' 주장) 은 사후보고 슬라이스용으로
    #  «같이» 담되, stageB 의 극단 선별에서 제외된다 (stage_b 가 post==0 필터).
    def is_post(c):
        return c["solved_in_prefix"] or c["found_claim"]

    def in_band(c):
        return POS_LO <= c["pos"] <= POS_HI

    def select(pool, per_prob, cap):
        by_prob = {}
        for c in pool:
            by_prob.setdefault(c["prob_idx"], []).append(c)
        chosen = []
        for _, cs in sorted(by_prob.items()):
            cs.sort(key=lambda c: (not in_band(c), abs(c["pos"] - 0.45),
                                   c["rollout_idx"]))
            chosen += cs[:per_prob]
        chosen.sort(key=lambda c: (not in_band(c), c["prob_idx"], c["rollout_idx"]))
        return chosen[:cap]

    clean = select([c for c in harvested if not is_post(c)],
                   SITES_PER_PROBLEM, max_sites)
    post = select([c for c in harvested if is_post(c)],
                  SITES_PER_PROBLEM, POST_SITES_MAX if not args.smoke else 4)
    for c in clean:
        c["post"] = 0
    for c in post:
        c["post"] = 1
    sites = clean + post
    print(f"[stageA] 선별: clean {len(clean)} + post {len(post)}", flush=True)

    # ── 4) donor 배정 (ctrl 용, 다른 문제 · 길이비 0.7~1.3 · 결정적) ─────────
    for c in sites:
        d = pick_donor(c, harvested)
        c["donor_site_id"] = d["site_id"] if d else None
        c["donor_raw"] = d["meta_raw"] if d else None

    # ── 5) 갈아끼우기 훑기: 3조건 × K 이어생성 ─────────────────────────────
    jobs = []
    for c in sites:
        for cond in ("orig", "abl", "ctrl"):
            if cond == "ctrl" and c["donor_raw"] is None:
                continue
            pfx = prefix_of(c, cond)
            for k in range(k_cont):
                jobs.append((c, cond, pfx, k, det_seed(c["site_id"], cond, k)))
    rows = run_continuations(llm, tok, jobs, f"stageA-splice shard{args.shard}")

    # ── 6) 출력 ───────────────────────────────────────────────────────────
    site_cols = ["site_id", "prob_idx", "rollout_idx", "nums", "target", "witness",
                 "n_correct_of8", "meta_raw", "meta_body", "confidence", "decision",
                 "meta_start", "meta_end", "pos", "meta_len", "solved_in_prefix",
                 "found_claim", "post", "donor_site_id", "donor_raw",
                 "response_text", "prompt_json", "shard"]
    pd.DataFrame([{k: c[k] for k in site_cols} for c in sites]).to_parquet(
        OUT / f"sites_shard{args.shard}.parquet")
    pd.DataFrame(rows).to_parquet(OUT / f"stageA_shard{args.shard}.parquet")
    print(f"[stageA] wrote {OUT}/sites_shard{args.shard}.parquet "
          f"({len(sites)} sites) · stageA_shard{args.shard}.parquet ({len(rows)} rows)",
          flush=True)


# ══ stageB ═════════════════════════════════════════════════════════════════
def _load_all(pattern):
    import pandas as pd
    files = sorted(glob.glob(str(OUT / pattern)))
    if not files:
        sys.exit(f"[err] {OUT}/{pattern} 이 없다 — 이전 단계를 먼저 돌려라.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _cond_p(stage_df):
    """site_id × cond → p(correct). wide DataFrame (p_orig, p_abl, p_ctrl, n_*).

    조건 컬럼은 세 개를 항상 보장한다(smoke 등에서 한 조건이 통째로 비면 unstack 이
    그 컬럼을 안 만들고, summarize 의 `p_orig - p_abl` 이 TypeError 로 죽는다)."""
    conds = ["orig", "abl", "ctrl"]
    g = stage_df.groupby(["site_id", "cond"])["correct"].agg(["mean", "count"])
    wide = g["mean"].unstack("cond").reindex(columns=conds).add_prefix("p_")
    n = g["count"].unstack("cond").reindex(columns=conds).add_prefix("n_")
    return wide.join(n)


def rank_sites(n_top=100, n_bot=100):
    """stageA 병합 → post 제외 → Δ̂=p_orig−p_ctrl 정렬 → 상위/하위 선정.

    반환: (선정 사이트 DataFrame[site_id, delta_hat_A, label, rank], sites_df 전체)
    """
    sites = _load_all("sites_shard*.parquet").drop_duplicates("site_id")
    a = _cond_p(_load_all("stageA_shard*.parquet"))
    m = sites.merge(a, on="site_id", how="inner")
    m = m[(m["post"] == 0) & m["p_orig"].notna() & m["p_ctrl"].notna()].copy()
    m["delta_hat_A"] = m["p_orig"] - m["p_ctrl"]
    m = m.sort_values(["delta_hat_A", "site_id"],
                      ascending=[False, True]).reset_index(drop=True)
    n_top = min(n_top, len(m) // 2)
    n_bot = min(n_bot, len(m) - n_top)
    top = m.head(n_top).copy()
    top["label"] = "good"
    bot = m.tail(n_bot).copy()
    bot["label"] = "null"
    sel = __import__("pandas").concat([top, bot], ignore_index=True)
    sel["rank"] = range(len(sel))
    return sel, sites


def stage_b(args):
    import pandas as pd
    if args.sites_file:
        # ── clean-보강 경로: 위생 통과 사이트 전체를 K회 재측정 (라벨은 summarize에서) ──
        want = [l.strip() for l in open(args.sites_file) if l.strip()]
        sites = _load_all("sites_shard*.parquet").drop_duplicates("site_id")
        a = _cond_p(_load_all("stageA_shard*.parquet"))
        sel = sites.merge(a, on="site_id", how="inner")
        sel = sel[sel["site_id"].isin(want)].copy()
        sel["delta_hat_A"] = sel["p_orig"] - sel["p_ctrl"]
        sel["label"] = "clean"
        sel = sel.sort_values("site_id").reset_index(drop=True)
        sel["rank"] = range(len(sel))
    else:
        sel, _sites = rank_sites()
    mine = sel[sel["rank"] % args.nshards == args.shard]
    print(f"[stageB] shard {args.shard}/{args.nshards}: {len(mine)}/{len(sel)} 사이트 "
          f"· K={args.k}", flush=True)

    llm, tok = build_llm(seed=0)
    jobs = []
    for _, c in mine.iterrows():
        site = c.to_dict()
        for cond in [c.strip() for c in args.conds.split(",") if c.strip()]:
            pfx = prefix_of(site, cond)
            for k in range(args.k_offset, args.k_offset + args.k):
                # ★공통 난수: 같은 (site,k) 에 두 조건이 «동일한» seed 를 쓴다.
                #   조건 간 차이에서 샘플링 잡음을 상쇄하기 위함 (paired CRN).
                jobs.append((site, cond, pfx, k, det_seed(site["site_id"], "B", k)))
    tag = ("stageB_clean" if args.sites_file else "stageB") + os.environ.get("PROBE_TAG", "")
    if args.conds.replace(" ", "") != "orig,ctrl":
        tag += "_" + args.conds.replace(",", "").replace(" ", "")
    if args.k_offset:
        tag += f"_o{args.k_offset}"
    rows = run_continuations(llm, tok, jobs, f"{tag} shard{args.shard}")
    pd.DataFrame(rows).to_parquet(OUT / f"{tag}_shard{args.shard}.parquet")
    print(f"[stageB] wrote {OUT}/{tag}_shard{args.shard}.parquet ({len(rows)} rows)",
          flush=True)


# ══ summarize ══════════════════════════════════════════════════════════════
def cluster_boot_ci(df, value_col, n_boot=10000, seed=0):
    """문제(prob_idx) 단위 클러스터 부트스트랩 — 사이트가 문제 안에서 상관되므로
    문제를 통째로 재표집한다. 반환 (mean, lo95, hi95, n_sites, n_problems)."""
    df = df.dropna(subset=[value_col])
    if not len(df):
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"),
                    n_sites=0, n_problems=0)
    g = df.groupby("prob_idx")[value_col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(g), size=(n_boot, len(g)))
    means = s[idx].sum(1) / np.maximum(1.0, n[idx].sum(1))
    return dict(mean=float(df[value_col].mean()),
                lo=float(np.percentile(means, 2.5)),
                hi=float(np.percentile(means, 97.5)),
                n_sites=int(len(df)), n_problems=int(len(g)))


def summarize(_args):
    import pandas as pd
    sites = _load_all("sites_shard*.parquet").drop_duplicates("site_id")
    a = _cond_p(_load_all("stageA_shard*.parquet"))
    sa = sites.merge(a, on="site_id", how="inner")
    sa["delta_abl_A"] = sa["p_orig"] - sa["p_abl"]
    sa["delta_ctl_A"] = sa["p_orig"] - sa["p_ctrl"]

    b = _cond_p(_load_all("stageB_shard*.parquet"))
    b["delta_ctl_B"] = b["p_orig"] - b["p_ctrl"]
    sb = sites.merge(b[["delta_ctl_B", "p_orig", "p_ctrl"]], on="site_id",
                     how="inner")

    sel, _ = rank_sites()              # stageA 랭킹 재현 (라벨의 정의처)
    lab = sel[["site_id", "label", "delta_hat_A", "rank"]]
    sb = sb.merge(lab, on="site_id", how="left")

    med = float(sb["pos"].median())
    summary = {
        # ⚠선별-측정 분리: 상위군은 stageA(K=6) 의 Δ̂ 로 «선별»했고, 아래 CI 는
        #   stageB(K=32, 독립 이어생성) 로 «측정»했다. 같은 표본으로 뽑고 재면
        #   winner's curse 로 Δ 가 부풀므로 반드시 이 분리를 유지한다.
        "delta_ctl_B_all": cluster_boot_ci(sb, "delta_ctl_B"),
        "delta_ctl_B_topA": cluster_boot_ci(sb[sb["label"] == "good"], "delta_ctl_B"),
        "delta_ctl_B_botA": cluster_boot_ci(sb[sb["label"] == "null"], "delta_ctl_B"),
        "delta_ctl_A_post1": cluster_boot_ci(sa[sa["post"] == 1], "delta_ctl_A"),
        "delta_abl_A_all": cluster_boot_ci(sa[sa["post"] == 0], "delta_abl_A"),
        "delta_ctl_B_pos_early": cluster_boot_ci(sb[sb["pos"] <= med], "delta_ctl_B"),
        "delta_ctl_B_pos_late": cluster_boot_ci(sb[sb["pos"] > med], "delta_ctl_B"),
        "pos_median": med,
        "n_sites_stageA": int(len(sa)), "n_sites_stageB": int(len(sb)),
        "note": ("selection(stageA K=6) 과 measurement(stageB K=32) 분리. "
                 "CI 는 문제 단위 클러스터 부트스트랩 10000회 95%."),
    }

    labels = sel[["site_id", "label", "delta_hat_A", "rank"]].merge(
        sb[["site_id", "delta_ctl_B", "p_orig", "p_ctrl", "prob_idx", "pos",
            "post", "confidence", "decision"]], on="site_id", how="left")
    # post=1 사이트도 label="post" 행으로 같이 싣는다 — ruler_battery 의 ③
    # (사후보고 슬라이스 오염 검사)가 이 행들 없이는 표본 0 으로 공허해진다.
    # good/null 극단 선별(rank_sites)은 post==0 만 쓰므로 site_id 중복은 없다.
    post_rows = sa[sa["post"] == 1][["site_id", "delta_ctl_A", "prob_idx", "pos",
                                     "post", "confidence", "decision"]].copy()
    post_rows["label"] = "post"
    labels = pd.concat([labels, post_rows], ignore_index=True)
    labels.to_parquet(OUT / "labels.parquet")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    print(f"\n{'slice':<24}{'meanΔ':>9}{'lo95':>9}{'hi95':>9}{'n_site':>8}{'n_prob':>8}")
    for k, v in summary.items():
        if isinstance(v, dict) and "mean" in v:
            print(f"{k:<24}{v['mean']:>9.4f}{v['lo']:>9.4f}{v['hi']:>9.4f}"
                  f"{v['n_sites']:>8}{v['n_problems']:>8}")
    print(f"\n[summarize] wrote {OUT}/labels.parquet ({len(labels)} rows) · "
          f"{OUT}/summary.json", flush=True)


# ══ main ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("stageA", help="8롤아웃 → 사이트 수확 → 3조건×K=6 훑기")
    a.add_argument("--shard", type=int, required=True)
    a.add_argument("--nshards", type=int, required=True)
    a.add_argument("--smoke", action="store_true",
                   help="문제 6 · 사이트 ≤10 · K=2 로 축소")
    a.set_defaults(fn=stage_a)

    b = sub.add_parser("stageB", help="상·하위 200 사이트 orig/ctrl K=32 재측정")
    b.add_argument("--shard", type=int, required=True)
    b.add_argument("--nshards", type=int, required=True)
    b.add_argument("--k", type=int, default=32)
    b.add_argument("--conds", default="orig,ctrl",
                   help="측정할 조건 쉼표목록 (orig/abl/ctrl). 기본 orig,ctrl")
    b.add_argument("--k_offset", type=int, default=0,
                   help="k 인덱스 시작값. K증강 시 기존과 다른 난수를 보장 (det_seed가 k에 의존)")
    b.add_argument("--sites_file", default=None,
                   help="site_id 목록 파일(줄당 1개). 지정 시 랭킹 대신 이 사이트들을 K회 재측정 "
                        "(clean-보강 실행용). 출력은 stageB_clean_shard{I}.parquet")
    b.set_defaults(fn=stage_b)

    s = sub.add_parser("summarize", help="병합·부트스트랩 CI·good/null 라벨")
    s.set_defaults(fn=summarize)

    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
