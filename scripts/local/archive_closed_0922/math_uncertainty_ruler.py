#!/usr/bin/env python
r"""math_uncertainty_ruler — 롤아웃 **하나**의 토큰 불확실성이 «이 시도가 맞았나» 를 가르는가.

왜 재는가 (cd9, 0914 — `docs/HYPOTHESIS_LEDGER_cd9.md` §C1 재계산 직후):
이 4B 비-thinking 정책에서 메타인지 관문은 전부 떨어졌다. §C1 이 그 이유를 하나로 모았다 —
우리가 가진 내부 신호(끝 은닉 프로브)가 읽는 것은 **문제 난이도**이지 **이 시도의 정오**가
아니다(풀링 AUC .825 · 난이도 ρ +.508 · 그러나 문제 **안** .577, CI 가 .5 를 포함). 적힌
확신도는 문제 안 .568, 형제 답 점유율만 .754 로 산다 — 그런데 그건 K 개 형제를 필요로 하는
self-consistency 이지 롤아웃 하나의 자기지식이 아니다. GRPO 는 정의상 문제 간 성분을 지우므로
**보상이 볼 수 있는 것은 문제 안 성분뿐**이고, 거기 남은 단일-롤아웃 신호를 우리는 아직
한 번도 안 재 봤다: **토큰 수준 불확실성**(logprob·엔트로피·DeepConf 식 최저 구간 확신).

설계:
  1. `--score` (GPU) — 이미 있는 롤아웃을 **교사강제**한다. vLLM `LLM.generate` 에
     prompt = render_generation_prompt(variant, problem) + 롤아웃 텍스트, `max_tokens=1`,
     `prompt_logprobs=K` 로 넣어 위치별 top-K 분포를 받는다. 생성이 아니라 재채점이므로
     샘플링 난수·온도는 결과에 안 들어간다(vllm 의 prompt logprobs 는 `logprobs_mode`
     기본값 "raw_logprobs" — 원 logits 의 log_softmax 다). **응답 위치만** 집계한다.
  2. `--analyze` (CPU) — §C1 과 **같은 세 숫자**를 낸다: 풀링 AUC · 문제 안 AUC(혼합문제
     평균, 문제 단위 부트스트랩 CI) · 난이도 ρ(문제별 평균 신호 vs pass rate). 여기에
     분산 분해(문제 간 비중), **비-잘림 한정** 문제 안 AUC, **길이 잔차화** 문제 안 AUC 를
     덧붙인다. 대조군으로 형제 답 점유율(LOO)·응답 길이·비-잘림을 c1.py 와 같은 관례로
     같은 표에 올린다.

판정: **없다 — 이것은 자(ruler)이지 관문이 아니다.** 이 스크립트는 통과/실패를 찍지 않고
모집단마다 `BEST-WITHIN: <metric> <auc> [ci]` 한 줄만 찍는다. 그 줄이 .5 근처면 «단일 롤아웃
불확실성으로는 문제 안을 못 가른다» 가 되고, 그러면 내부신호 보상은 형제 축(self-consistency)
이나 다른 자리로 가야 한다. .65 를 넘는 자가 나오면 그 자가 다음 보상의 표적 후보다.
★어느 쪽이든 **여기서 고른 자는 다시 인과 관문을 통과해야 한다** — 상관은 조향이 아니다.

★해석 주의 두 가지(둘 다 코드가 같이 찍는다):
  · **엔트로피는 top-K 절단 지지집합 위의 엔트로피**다. 전체 어휘 엔트로피가 아니라 상위 K개를
    다시 정규화한 값이라 항상 과소추정이며, 상한은 log(K)=3.00 nat(K=20). 절대값을 문헌의
    «엔트로피» 와 직접 비교하지 말 것. 순위 비교(AUC)에만 쓴다.
  · **잘림(truncated) 교란**. 잘린 롤아웃은 거의 틀리고 동시에 길다 — 길이·엔트로피 자가
    «정오» 가 아니라 «잘렸나» 를 읽을 수 있다. 그래서 비-잘림 한정 열과 길이 잔차화 열을
    반드시 같이 읽는다(§C1 에서 응답 길이의 .670 이 발화행에선 .396 으로 뒤집힌 전례).

사용(예):
  # GPU (qwen35 env, vllm 0.29)
  python scripts/local/math_uncertainty_ruler.py --score \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path /hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507 \
      --variant math_opt \
      --out_dir /hdd_data/seungpil/scratch/eval/uncertainty_ruler_L5
  # CPU
  python scripts/local/math_uncertainty_ruler.py --analyze \
      --out_dir /hdd_data/seungpil/scratch/eval/uncertainty_ruler_L5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_NAN = float("nan")

TOPK = 20                 # ★vllm max_logprobs 기본 상한과 같다(20). 넘기려면 엔진 인자도 올려야 한다.
FORK_ENTROPY = 1.0        # nat — "forking token" 판정선
WINDOW = 128              # DeepConf 식 최저 구간 폭(토큰)
TAIL = 64                 # 앞/뒤 구간 폭(토큰)
N_BOOT = 2000
BOOT_SEED = 11

# ★부호: 값이 클수록 «맞았을 것» 이 되도록 곱하는 수. 아래 주석이 그 이유다.
METRIC_SIGNS: dict[str, int] = {
    "mean_logp": +1,            # 확신이 높을수록 맞는다는 가설
    "min_logp": +1,             # 최악의 한 토큰이 덜 나쁠수록
    "mean_entropy": -1,         # 퍼진 분포 = 헤맴
    "max_entropy": -1,
    "frac_high_entropy": -1,    # 갈림길 토큰이 많을수록 헤맴
    "mean_logp_at_forks": +1,   # 갈림길에서도 확신이 높을수록
    "entropy_first64": -1,
    "entropy_last64": -1,
    "entropy_slope": -1,        # ★부호 미확정 가설: 풀수록 엔트로피가 **오르면** 무너지는 중이라고 본다
    "answer_logp": +1,          # 마지막 \boxed 스팬의 확신
    "tail_logp_64": +1,
    "lowest_window_logp": +1,   # DeepConf "lowest group confidence"
    "n_resp_tok": -1,           # 대조 — 길수록 틀린다(§C1 .670, 단 잘림 교란)
    "not_truncated": +1,        # 대조
    "sibling_share": +1,        # 대조 — §C1 의 유일한 생존자(.754). 단일 롤아웃 자가 아니다
}

# ── 순수 함수(테스트 대상) ────────────────────────────────────────────────────


def topk_entropy(logprobs: Sequence[float], k: int = TOPK) -> float:
    r"""상위 k 개 logprob 을 **다시 정규화한** 분포의 엔트로피(nat).

    ★이것은 절단 지지집합(truncated-support) 엔트로피다 — 전체 어휘가 아니라 상위 k 개만
    보고, 그 k 개의 확률합이 1이 되도록 나눈 뒤 −Σ p log p 를 잰다. 따라서 참 엔트로피의
    하한이고 상한은 log(k). vllm 은 위치마다 top-k **에 더해 실제 토큰**을 끼워 주므로
    (k+1 개가 올 수 있다) 여기서 상위 k 개만 골라 «자리마다 같은 지지집합 크기» 를 지킨다.
    """
    v = [float(x) for x in logprobs if x is not None and math.isfinite(float(x))]
    if not v:
        return _NAN
    v.sort(reverse=True)
    v = v[:k]
    m = v[0]
    w = [math.exp(x - m) for x in v]
    z = sum(w)
    if z <= 0:
        return _NAN
    h = 0.0
    for wi in w:
        p = wi / z
        if p > 0:
            h -= p * math.log(p)
    return h


def frac_above(values: Sequence[float], thresh: float) -> float:
    """유한한 값 중 thresh 를 **넘는** 비율. 유한한 값이 없으면 nan."""
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not v:
        return _NAN
    return sum(1 for x in v if x > thresh) / len(v)


def ols_slope(y: Sequence[float]) -> float:
    """정규화 위치(0..1)에 대한 y 의 OLS 기울기. 점이 2개 미만이면 nan."""
    v = [float(x) for x in y if x is not None and math.isfinite(float(x))]
    n = len(v)
    if n < 2:
        return _NAN
    x = [i / (n - 1) for i in range(n)]
    mx, my = sum(x) / n, sum(v) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx <= 0:
        return _NAN
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, v)) / sxx


def lowest_window_mean(values: Sequence[float], window: int = WINDOW) -> float:
    """폭 `window` 슬라이딩 창 평균의 **최솟값**(DeepConf lowest group confidence).
    길이가 window 보다 짧으면 전체 평균 하나가 유일한 창이다."""
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return _NAN
    w = min(window, v.size)
    cs = np.concatenate(([0.0], np.cumsum(v)))
    means = (cs[w:] - cs[:-w]) / w
    return float(means.min())


def locate_boxed_token_span(tok, full_text: str, start: int, end: int) -> tuple[int, int]:
    r"""문자 구간 [start,end) 를 덮는 토큰 인덱스 [i0,i1) 을 `full_text` 의 토큰화에서 찾는다.

    fast tokenizer 면 offset mapping 을 쓴다(토큰 [o0,o1) 이 구간과 겹치면 포함).
    offset 이 없으면 «앞 접두사 길이» 로 센다: len(encode(full[:start])) .. len(encode(full[:end])).
    후자는 토큰 경계가 구간 경계와 안 맞으면 ±1 토큰 흔들릴 수 있다(근사임을 명시).
    """
    if start >= end:
        return (0, 0)
    enc = None
    try:
        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
    except Exception:
        offs = None
    if offs:
        idx = [i for i, (a, b) in enumerate(offs) if b > start and a < end]
        if idx:
            return (idx[0], idx[-1] + 1)
        return (0, 0)
    i0 = len(tok.encode(full_text[:start], add_special_tokens=False))
    i1 = len(tok.encode(full_text[:end], add_special_tokens=False))
    return (i0, max(i1, i0 + 1))


def auc(scores: Sequence[float], labels: Sequence[float]) -> float:
    """Mann-Whitney AUC. 한쪽 라벨만 있거나 유한값이 없으면 nan. 상수 점수는 .5 가 아니라
    ★nan 을 돌려준다 — «가르지 못함» 과 «잴 수 없음» 을 섞지 않기 위해서다."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(y)
    s, y = s[ok], y[ok]
    if s.size == 0 or len(set(y.tolist())) < 2:
        return _NAN
    if np.nanmax(s) == np.nanmin(s):
        return _NAN
    from scipy.stats import rankdata  # noqa: PLC0415
    r = rankdata(s)
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def per_problem_aucs(scores: np.ndarray, labels: np.ndarray,
                     gidx: dict, mixed: Sequence) -> list[float]:
    """혼합문제(0<pass<1)별 AUC 목록. 유한한 것만."""
    out = []
    for g in mixed:
        idx = [i for i in gidx[g] if np.isfinite(scores[i])]
        if len(idx) < 2:
            continue
        a = auc(scores[idx], labels[idx])
        if np.isfinite(a):
            out.append(a)
    return out


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = N_BOOT,
                      seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """문제 단위 재표본 평균의 95% 백분위 CI. 3개 미만이면 (mean, nan, nan)."""
    v = np.asarray([float(x) for x in values], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (_NAN, _NAN, _NAN)
    if v.size < 3:
        return (float(v.mean()), _NAN, _NAN)
    rs = np.random.RandomState(seed)
    bs = v[rs.randint(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def within_problem_residual(scores: np.ndarray, covar: np.ndarray, gidx: dict) -> np.ndarray:
    """문제 안에서 `scores` 를 `covar`(길이) 에 회귀한 **잔차**. 문제 안 분산이 0이면 중심화만."""
    out = np.full(scores.shape, _NAN, dtype=np.float64)
    for _g, idx in gidx.items():
        ii = [i for i in idx if np.isfinite(scores[i]) and np.isfinite(covar[i])]
        if len(ii) < 2:
            continue
        s = scores[ii]
        x = covar[ii]
        xm = x.mean()
        sxx = float(((x - xm) ** 2).sum())
        if sxx <= 0:
            out[ii] = s - s.mean()
            continue
        b = float(((x - xm) * (s - s.mean())).sum()) / sxx
        res = s - (s.mean() + b * (x - xm))
        # ★회귀가 전부 설명하면 잔차는 부동소수 잡음만 남는다 — 그 잡음이 순위를 만들어
        #   «잔차도 완벽히 가른다» 는 허깨비를 찍지 않도록 0 으로 스냅한다.
        scale = float(np.max(np.abs(s))) if s.size else 0.0
        if float(np.max(np.abs(res))) <= 1e-12 * max(1.0, scale):
            res = np.zeros_like(res)
        out[ii] = res
    return out


def between_problem_var_frac(scores: np.ndarray, gidx: dict) -> float:
    """신호 분산 중 **문제 간** 비중(c1.py §C1c 와 같은 관례: 그룹평균의 분산 vs 그룹 내 잔차의
    평균제곱). GRPO 그룹 중심화가 지우는 성분이 바로 이 비중이다."""
    gm, resid = [], []
    for _g, idx in gidx.items():
        v = np.asarray([scores[i] for i in idx if np.isfinite(scores[i])], dtype=np.float64)
        if v.size < 2:
            continue
        gm.append(v.mean())
        resid.append(v - v.mean())
    if not gm:
        return _NAN
    vb = float(np.var(gm))
    vw = float(np.mean(np.concatenate(resid) ** 2))
    if vb + vw <= 0:
        return _NAN
    return vb / (vb + vw)


def sibling_share_loo(rows: Sequence[dict], gidx: dict) -> np.ndarray:
    """형제 답 점유율(leave-one-out) — c1.py 와 **글자 그대로 같은 관례**: 자기를 뺀 형제 중
    final_answer 가 자기와 **문자열로 같은** 비율. 자기 답이 None 이거나 형제가 없으면 nan."""
    out = np.full(len(rows), _NAN, dtype=np.float64)
    for _g, idx in gidx.items():
        ans = [rows[i].get("final_answer") for i in idx]
        for i in idx:
            others = [a for j, a in zip(idx, ans) if j != i and a is not None]
            a = rows[i].get("final_answer")
            if a is None or not others:
                continue
            out[i] = sum(1 for o in others if o == a) / len(others)
    return out


# ── GPU 단계 ─────────────────────────────────────────────────────────────────


def _load_rollouts(paths: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        src = Path(p)
        tag = src.parent.name
        for line in src.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["_source"] = tag
            # ★group_id 는 디렉터리마다 독립이다 — 모집단을 섞어도 그룹이 안 겹치게 접두사를 붙인다.
            r["_gkey"] = f"{tag}::{r['group_id']}"
            rows.append(r)
    return rows


def _pos_logprobs(plp, i: int, token_id: int) -> tuple[float, list[float]]:
    """위치 i 의 (선택 토큰 logprob, 그 자리의 모든 logprob 목록).

    ★vllm 0.29: `RequestOutput.prompt_logprobs` 는 `PromptLogprobs`
    = `FlatLogprobs | list[dict[int, Logprob] | None]`. 어느 쪽이든 `plp[i]` 가
    `dict[token_id -> Logprob]` 를 준다(FlatLogprobs 는 Sequence API 를 흉내낸다).
    **위치 0 은 조건이 없어 None(FlatLogprobs 에선 빈 dict)** 이므로 반드시 걸러야 한다.
    자리마다 top-K **에 더해** 실제 토큰이 들어와 K+1 개가 올 수 있다(gpu_model_runner 가
    `num_prompt_logprobs + 1` 랭크를 뽑는다) — 선택 토큰은 항상 들어 있다.
    """
    d = plp[i] if i < len(plp) else None
    if not d:
        return (_NAN, [])
    vals = [float(lp.logprob) for lp in d.values()]
    got = d.get(token_id)
    chosen = float(got.logprob) if got is not None else _NAN
    return (chosen, vals)


def score_stage(a) -> None:
    from transformers import AutoTokenizer  # noqa: PLC0415

    from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: PLC0415
    from src.training.math_meta import boxed_spans  # noqa: PLC0415

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rollouts(a.rollouts)
    if a.max_rows:
        rows = rows[: a.max_rows]
    print(f"[ruler] rows={len(rows)} sources={sorted(set(r['_source'] for r in rows))}")

    tok = AutoTokenizer.from_pretrained(a.model_path)

    # ── 프롬프트 복원 + 라운드트립 검사 ────────────────────────────────────
    prepared: list[dict] = []
    n_rt_fail = 0
    prompt_cache: dict[str, str] = {}
    for r in rows:
        prob = r["problem"]
        if prob not in prompt_cache:
            prompt_cache[prob] = render_generation_prompt(tok, a.variant, prob)
        prompt = prompt_cache[prob]
        text = r.get("text") or ""
        ids_p = tok(prompt, add_special_tokens=False)["input_ids"]
        full = prompt + text
        ids_f = tok(full, add_special_tokens=False)["input_ids"]
        if len(ids_f) <= len(ids_p) or ids_f[: len(ids_p)] != ids_p:
            # ★프롬프트 토큰이 prompt+text 토큰의 접두가 아니면 «응답 위치» 정의가 깨진다 → 버린다.
            n_rt_fail += 1
            continue
        prepared.append({"row": r, "full": full, "ids": ids_f, "n_prompt": len(ids_p),
                         "boxed": boxed_spans(text), "prompt_chars": len(prompt)})
    print(f"[ruler] roundtrip fail(skipped) {n_rt_fail} / {len(rows)}")
    if not prepared:
        raise SystemExit("[ruler] 라운드트립을 통과한 행이 없다 — variant 가 맞는지 확인하라.")

    max_len = max(len(p["ids"]) for p in prepared) + 16
    print(f"[ruler] max_model_len={max_len}  total tokens={sum(len(p['ids']) for p in prepared):,}")

    from vllm import LLM, SamplingParams  # noqa: PLC0415

    llm = LLM(model=a.model_path, dtype="bfloat16", gpu_memory_utilization=a.gpu_util,
              max_model_len=max_len, enforce_eager=True, max_logprobs=a.topk,
              seed=a.seed)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=a.topk, logprobs=None)

    f_scores = (out_dir / "scores.jsonl").open("w")
    ent_blobs: list[np.ndarray] = []
    ent_offsets: list[int] = [0]
    ent_keys: list[str] = []
    n_written = 0
    for c0 in range(0, len(prepared), a.chunk):
        batch = prepared[c0: c0 + a.chunk]
        outs = llm.generate([b["full"] for b in batch], sp)
        for b, o in zip(batch, outs):
            rec, ent = _metrics_for(b, o, tok, a.topk)
            f_scores.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ent_blobs.append(ent.astype(np.float16))
            ent_offsets.append(ent_offsets[-1] + ent.size)
            ent_keys.append(rec["row_key"])
            n_written += 1
        f_scores.flush()
        print(f"[ruler] {n_written}/{len(prepared)} scored", flush=True)
    f_scores.close()

    np.savez_compressed(
        out_dir / "entropy_series.npz",
        entropy=np.concatenate(ent_blobs) if ent_blobs else np.zeros(0, np.float16),
        offsets=np.asarray(ent_offsets, dtype=np.int64),
        row_key=np.asarray(ent_keys, dtype=object),
    )
    meta = {"rollouts": list(a.rollouts), "model_path": a.model_path, "variant": a.variant,
            "topk": a.topk, "n_rows_in": len(rows), "n_scored": n_written,
            "n_roundtrip_fail": n_rt_fail, "max_model_len": max_len,
            "fork_entropy": FORK_ENTROPY, "window": WINDOW, "tail": TAIL}
    (out_dir / "score_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[ruler] wrote {out_dir/'scores.jsonl'} ({n_written} rows)")


def _metrics_for(b: dict, out, tok, topk: int) -> tuple[dict, np.ndarray]:
    """한 롤아웃의 응답 위치 지표. (scores.jsonl 한 줄, 엔트로피 시리즈)."""
    r = b["row"]
    ids = b["ids"]
    n_prompt = b["n_prompt"]
    plp = out.prompt_logprobs
    chosen: list[float] = []
    ent: list[float] = []
    if plp is None:
        # ★prompt_logprobs 가 통째로 None 이면 엔진이 안 켜준 것 — 조용히 0 으로 채우지 말고 터뜨린다.
        raise RuntimeError("vllm 이 prompt_logprobs 를 안 돌려줬다 (SamplingParams/max_logprobs 확인)")
    for i in range(n_prompt, len(ids)):
        cp, vals = _pos_logprobs(plp, i, ids[i])
        chosen.append(cp)
        ent.append(topk_entropy(vals, topk))
    cp = np.asarray(chosen, dtype=np.float64)
    ep = np.asarray(ent, dtype=np.float64)
    fin_c = cp[np.isfinite(cp)]
    fin_e = ep[np.isfinite(ep)]
    fork = ep > FORK_ENTROPY

    # 마지막 \boxed 스팬의 토큰 구간
    ans_lp = _NAN
    if b["boxed"]:
        _content, s, e = b["boxed"][-1]
        i0, i1 = locate_boxed_token_span(tok, b["full"], b["prompt_chars"] + s,
                                         b["prompt_chars"] + e)
        i0 = max(i0 - n_prompt, 0)
        i1 = max(i1 - n_prompt, 0)
        seg = cp[i0:i1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            ans_lp = float(seg.mean())

    rec = {
        "row_key": f"{r['_gkey']}#{r.get('problem_id', '')}#{len(ids)}",
        "source": r["_source"],
        "group_id": r["_gkey"],
        "r_corr": float(r["r_corr"]),
        "truncated": float(r.get("truncated", 0)),
        "final_answer": r.get("final_answer"),
        "n_tok": float(r.get("n_tok", len(ids) - n_prompt)),
        "n_resp_tok": int(len(ids) - n_prompt),
        "mean_logp": float(fin_c.mean()) if fin_c.size else _NAN,
        "min_logp": float(fin_c.min()) if fin_c.size else _NAN,
        "mean_entropy": float(fin_e.mean()) if fin_e.size else _NAN,
        "max_entropy": float(fin_e.max()) if fin_e.size else _NAN,
        "frac_high_entropy": frac_above(ep, FORK_ENTROPY),
        "mean_logp_at_forks": float(np.nanmean(cp[fork])) if fork.any() and
        np.isfinite(cp[fork]).any() else _NAN,
        "entropy_first64": float(np.nanmean(ep[:TAIL])) if fin_e.size else _NAN,
        "entropy_last64": float(np.nanmean(ep[-TAIL:])) if fin_e.size else _NAN,
        "entropy_slope": ols_slope(ep.tolist()),
        "answer_logp": ans_lp,
        "tail_logp_64": float(np.nanmean(cp[-TAIL:])) if fin_c.size else _NAN,
        "lowest_window_logp": lowest_window_mean(cp, WINDOW),
    }
    return rec, ep


# ── CPU 단계 ─────────────────────────────────────────────────────────────────

BASE_METRICS = ["mean_logp", "min_logp", "mean_entropy", "max_entropy", "frac_high_entropy",
                "mean_logp_at_forks", "entropy_first64", "entropy_last64", "entropy_slope",
                "answer_logp", "tail_logp_64", "lowest_window_logp"]
REF_METRICS = ["n_resp_tok", "not_truncated", "sibling_share"]


def _analyze_population(rows: list[dict]) -> dict:
    from scipy.stats import spearmanr  # noqa: PLC0415

    n = len(rows)
    gidx: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        gidx[r["group_id"]].append(i)
    y = np.asarray([r["r_corr"] for r in rows], dtype=np.float64)
    trunc = np.asarray([r.get("truncated", 0.0) for r in rows], dtype=np.float64)
    nresp = np.asarray([float(r.get("n_resp_tok") or np.nan) for r in rows], dtype=np.float64)
    prate = {g: float(np.mean([y[i] for i in idx])) for g, idx in gidx.items()}
    mixed = [g for g, p in prate.items() if 0.0 < p < 1.0]

    raw: dict[str, np.ndarray] = {}
    for m in BASE_METRICS:
        raw[m] = np.asarray([float(r.get(m) if r.get(m) is not None else np.nan)
                             for r in rows], dtype=np.float64)
    raw["n_resp_tok"] = nresp
    raw["not_truncated"] = 1.0 - trunc
    raw["sibling_share"] = sibling_share_loo(rows, gidx)

    non_trunc = trunc < 0.5
    table = []
    for m in BASE_METRICS + REF_METRICS:
        s = raw[m] * METRIC_SIGNS[m]
        pooled = auc(s, y)
        per = per_problem_aucs(s, y, gidx, mixed)
        mean_w, lo, hi = bootstrap_mean_ci(per)
        gs, ps = [], []
        for g, idx in gidx.items():
            v = [s[i] for i in idx if np.isfinite(s[i])]
            if not v:
                continue
            gs.append(float(np.mean(v)))
            ps.append(prate[g])
        rho = float(spearmanr(gs, ps).statistic) if len(gs) > 5 and len(set(gs)) > 1 else _NAN

        s_nt = s.copy()
        s_nt[~non_trunc] = np.nan
        per_nt = per_problem_aucs(s_nt, y, gidx, mixed)
        mean_nt, lo_nt, hi_nt = bootstrap_mean_ci(per_nt)

        s_res = within_problem_residual(s, nresp, gidx)
        per_res = per_problem_aucs(s_res, y, gidx, mixed)
        mean_res, lo_res, hi_res = bootstrap_mean_ci(per_res)

        table.append({
            "metric": m, "sign": "+" if METRIC_SIGNS[m] > 0 else "−(부호 반전)",
            "n_rows": int(np.isfinite(s).sum()),
            "auc_pooled": pooled,
            "auc_within": mean_w, "ci_lo": lo, "ci_hi": hi, "n_mixed_used": len(per),
            "difficulty_rho": rho,
            "between_var_frac": between_problem_var_frac(s, gidx),
            "auc_within_nontrunc": mean_nt, "ci_lo_nt": lo_nt, "ci_hi_nt": hi_nt,
            "n_mixed_nontrunc": len(per_nt),
            "auc_within_lenresid": mean_res, "ci_lo_res": lo_res, "ci_hi_res": hi_res,
        })

    cand = [t for t in table if t["metric"] in BASE_METRICS and np.isfinite(t["auc_within"])]
    best = max(cand, key=lambda t: t["auc_within"]) if cand else None
    return {"n_rows": n, "n_problems": len(gidx), "n_mixed": len(mixed),
            "pass_at_1": float(y.mean()),
            "trunc_rate": float(trunc.mean()),
            "table": table, "best_within": best}


def _fmt(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def _fmt_signed(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def analyze_stage(a) -> None:
    out_dir = Path(a.out_dir)
    path = Path(a.scores) if a.scores else out_dir / "scores.jsonl"
    rows = [json.loads(l) for l in path.open() if l.strip()]
    if not rows:
        raise SystemExit(f"[ruler] {path} 가 비었다.")
    pops: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        pops[r.get("source", "all")].append(r)
    if len(pops) > 1:
        pops["ALL(합침)"] = rows

    results = {}
    md = ["# math_uncertainty_ruler — 단일 롤아웃 토큰 불확실성의 자",
          "",
          "★엔트로피는 **top-K 절단 지지집합** 엔트로피(K=%d, 상한 log K = %.2f nat)다 — 절대값을"
          " 문헌과 비교하지 말고 순위(AUC)로만 읽는다." % (TOPK, math.log(TOPK)),
          "★부호는 «클수록 맞았을 것» 이 되도록 맞췄다. 표의 `sign` 열이 원 지표에 −1 을 곱했는지 알려준다.",
          "★판정 없음 — 이것은 자다. §C1 비교선: 은닉 프로브 문제 안 .577, 적힌 확신도 .568, 형제 답 점유율 .754.",
          ""]
    for name, prows in pops.items():
        res = _analyze_population(prows)
        results[name] = res
        md += [f"## {name}",
               "",
               f"행 {res['n_rows']} · 문제 {res['n_problems']} · 혼합문제 {res['n_mixed']} · "
               f"pass@1 {res['pass_at_1']:.3f} · 잘림률 {res['trunc_rate']:.3f}",
               "",
               "| 지표 | 부호 | n행 | AUC 풀링 | **AUC 문제 안** | 95% CI | 혼합문제 | 난이도 ρ | "
               "문제 간 분산 비중 | 문제 안(비-잘림) | 문제 안(길이 잔차) |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in res["table"]:
            md.append("| {m} | {sg} | {n} | {p} | **{w}** | [{lo}, {hi}] | {k} | {rho} | {bv} | "
                      "{nt} | {rs} |".format(
                          m=t["metric"], sg=t["sign"], n=t["n_rows"],
                          p=_fmt(t["auc_pooled"]), w=_fmt(t["auc_within"]),
                          lo=_fmt(t["ci_lo"]), hi=_fmt(t["ci_hi"]), k=t["n_mixed_used"],
                          rho=_fmt_signed(t["difficulty_rho"]), bv=_fmt(t["between_var_frac"]),
                          nt=_fmt(t["auc_within_nontrunc"]), rs=_fmt(t["auc_within_lenresid"])))
        b = res["best_within"]
        line = ("BEST-WITHIN: %s %s [%s, %s]" % (b["metric"], _fmt(b["auc_within"]),
                                                 _fmt(b["ci_lo"]), _fmt(b["ci_hi"]))
                if b else "BEST-WITHIN: none nan [nan, nan]")
        md += ["", f"**{line}**  ({name})", ""]
        print(f"[{name}] {line}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ruler_summary.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "ruler_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=lambda o: None
                   if isinstance(o, float) and not np.isfinite(o) else o))
    print(f"[ruler] wrote {out_dir/'ruler_summary.md'}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--score", action="store_true", help="GPU 단계: 교사강제 재채점")
    p.add_argument("--analyze", action="store_true", help="CPU 단계: C1 식 표")
    p.add_argument("--rollouts", action="append", default=[],
                   help="texts.jsonl (여러 번 줄 수 있다 — 모집단마다 표가 따로 나온다)")
    p.add_argument("--model_path", default="/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507")
    p.add_argument("--variant", default="math_opt")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scores", default=None, help="--analyze 가 읽을 scores.jsonl (기본 out_dir)")
    p.add_argument("--gpu_util", type=float, default=0.45)
    p.add_argument("--topk", type=int, default=TOPK)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--max_rows", type=int, default=0)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    if not (a.score or a.analyze):
        raise SystemExit("--score 또는 --analyze 중 하나는 줘야 한다.")
    if a.score:
        if not a.rollouts:
            raise SystemExit("--score 에는 --rollouts 가 필요하다.")
        score_stage(a)
    if a.analyze:
        analyze_stage(a)


if __name__ == "__main__":
    main()
