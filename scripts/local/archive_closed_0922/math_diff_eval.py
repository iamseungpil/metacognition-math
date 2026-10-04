#!/usr/bin/env python
r"""math_diff_eval — M_DIFF(난이도 판단 팔)의 held-out 판정.

무엇을 재는가. M_DIFF 의 주장은 두 겹이다: ①**모니터링** — 풀기 전에 말한 난이도가 실제
성공 확률을 담는가(캘리브레이션) ②**통제** — 그 말을 근거로 표본을 **재배분**하면 **같은
계산 예산**에서 균일 self-consistency 를 이기는가. ②가 이 팔의 진짜 관문이다 — 캘리브레이션은
수단이지 목적이 아니다(CLAUDE.md 의 «메타인지는 정확도를 끌어올리는 수단»).

한 체크포인트에 대해:
  (a) 문제마다 math_diff 프롬프트로 N=8 표집(temperature 1.0, 8192 토큰).
      **첫 표본**의 버킷이 그 문제의 판단이다(배포 시 우리가 보게 될 한 번의 발화).
  (b) 캘리브레이션 — Spearman(말한 버킷, 실현 pass@8) · 버킷별 실현 pass rate.
  (c) **같은 예산에서의 배분** — 평균 예산 B ∈ {1, 1.5, 2, 3}:
        acc_alloc(stated)  hard 라고 말한 문제에 k_max, 나머지에 1 (k_max 는 평균이 B 가
                           되도록 정한다 — hard 비율 f 에서 k_max = 1 + (B−1)/f)
        acc_uniform(B)     전 문제에 같은 예산(B 가 정수가 아니면 ⌈B⌉ 를 받는 문제 비율을
                           B−⌊B⌋ 로 맞춘다 — 평균이 정확히 B)
        acc_alloc(oracle)  «hard» 를 말 대신 **실제** 그룹 pass rate 하위로 고른 천장
                           (stated 와 **같은 개수**를 고른다 — 예산이 같아야 비교가 된다)
      표는 전부 **고정 롤아웃 순서**의 앞 k 개를 다수결한다(재표집 없음 — 같은 표본을 어떻게
      나눠 쓰느냐만 다르다). 문제 단위 부트스트랩 CI 를 함께 낸다.
  (d) 세금 — acc_first(메타를 쓴 acc@1) vs `--baseline_acc`(plain 프롬프트 같은 체크포인트의
      acc@1; 미지정이면 NaN). 음수면 난이도를 말하는 비용이다.
  (e) 형식 — 발화·meta_first·버킷 파싱·why·블록 다수·메타 안 \boxed.

생성기는 주입 가능하다(단위 테스트는 모의 생성기로 돈다 — GPU 불필요).

사용법:
  math_diff_eval.py --dataset math500 --model_path $WORK/merged/cd9_M_DIFF_s2/step_30 \
      --num_samples 8 --max_tokens 8192 --baseline_acc 0.61 \
      --out_dir $WORK/eval/cd9_M_DIFF_s2/step_30/math500_diff_8k
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.metacot.math_meta_prompt import build_math_prompt  # noqa: E402
from src.training import math_dis as MD  # noqa: E402  (plurality_answer 를 공유한다)
from src.training import math_diff as MDF  # noqa: E402
from src.training import math_meta as MM  # noqa: E402

VARIANT = "math_diff"
_NAN = float("nan")
# ★배분 표의 평균 예산들. 1 은 «배분 없음»(sanity: 세 곡선이 같아야 한다).
BUDGETS = (1.0, 1.5, 2.0, 3.0)
# 요약 키(런처·게이트가 읽는 이름). ★acc_first 가 gate_judgment.py 의 --acc-key 다.
_SUMMARY_KEYS = ("acc_first", "acc_mean", "acc_alloc_b2", "acc_uniform_b2", "alloc_gain_b2",
                 "alloc_gain_b2_ci_lo", "alloc_gain_b2_ci_hi", "acc_alloc_oracle_b2",
                 "acc_tax_vs_baseline", "spearman_stated_vs_pass8",
                 "pass_easy", "pass_medium", "pass_hard", "stated_hard_frac",
                 "diff_emit_rate", "meta_first_rate", "bucket_parsed", "has_why_rate",
                 "bucket_entropy", "bucket_max_share", "multi_block", "boxed_in_meta",
                 "trunc_rate", "tokens_per_problem")

# 생성기: chat 메시지 리스트들 -> 프롬프트마다 [(text, truncated, n_tok), ...] N 개.
Generator = Callable[[Sequence[list[dict]]], Sequence[Sequence[tuple[str, int, int]]]]


# ── (a) 표집 + 파싱 ─────────────────────────────────────────────────────────────
def sample_rows(problems: Sequence[Mapping], generate: Generator) -> list[list[dict]]:
    """문제마다 N 개의 파싱된 행(학습과 **같은 파서** math_diff.parse_diff_row). 순서는 생성
    순서 그대로다 — 배분 시뮬레이션이 «앞 k 개»를 쓰므로 순서가 곧 예산이다."""
    prompts = [build_math_prompt(p["problem"], VARIANT) for p in problems]
    outs = generate(prompts)
    if len(outs) != len(problems):
        raise RuntimeError(f"[diff-eval] 생성기가 {len(outs)} 개를 돌려줬다(문제 {len(problems)})")
    groups = []
    for gi, (src, samples) in enumerate(zip(problems, outs)):
        gold = str(src["gold"])
        rows = []
        for si, (text, trunc, n_tok) in enumerate(samples):
            r = MDF.parse_diff_row(text, gold, src["problem"], truncated=trunc)
            r.update(MDF.diff_row_flags(r))
            r.update({"group_id": f"g{gi}", "problem_id": gi, "sample_idx": si,
                      "uid": f"g{gi}", "n_tok": int(n_tok)})
            rows.append(r)
        # ★크레딧·라벨은 held-out 에서도 **학습과 같은 함수**로 잰다(그룹 = 이 문제의 N 개 표본).
        for li, r in enumerate(rows):
            credit, defined = MDF.diff_row_credit(r, rows, self_idx=li)
            r["loo_agreement"] = MDF.loo_agreement(li, rows)
            r["agree_bucket"] = MDF.agree_bucket(r["loo_agreement"])
            r["loo_pass_rate"] = MDF.diff_row_metrics(r, rows, self_idx=li)["loo_pass_rate"]
            r["diff_credit"] = float(credit)
            r["diff_defined"] = int(defined)
            r["meta_defined"] = int(defined)
        groups.append(rows)
    return groups


# ── (c) 배분 시뮬레이션 ─────────────────────────────────────────────────────────
def majority_vote(rows: Sequence[Mapping], k: int) -> str | None:
    r"""**앞 k 개** 표본의 다수답(math_dis.plurality_answer — 수학 동치로 군집). 동률·무답이면
    첫 표본의 답으로 떨어진다(None 이 아니다) — 배분 표는 «표를 어떻게 나눠 쓰느냐»의 비교라
    한쪽만 «기권»으로 빠지면 예산 비교가 깨진다."""
    ans = [str(r.get("final_answer") or "").strip() for r in list(rows)[:max(1, int(k))]]
    ans = [a for a in ans if a]
    if not ans:
        return None
    plur = MD.plurality_answer(ans)
    return plur if plur is not None else ans[0]


def _acc_from_ks(groups: Sequence[Sequence[Mapping]], golds: Sequence[str],
                 ks: Sequence[int]) -> list[float]:
    """문제별 정오(0/1) — 문제 i 에 예산 ks[i] 를 주고 다수결한 결과."""
    out = []
    for rows, gold, k in zip(groups, golds, ks):
        a = majority_vote(rows, k)
        out.append(0.0 if a is None else float(MM.grade_math(f"\\boxed{{{a}}}", str(gold))))
    return out


def uniform_ks(n: int, budget: float, n_max: int) -> list[int]:
    """평균이 정확히 `budget` 인 균일 배분 — ⌊B⌋ 를 기본으로 주고 앞에서부터 round(n·(B−⌊B⌋))
    문제에 한 개를 더 준다(결정적; 문제 순서는 데이터 순서라 난이도와 무관하다)."""
    b = max(1.0, float(budget))
    lo = int(b)
    extra = int(round(n * (b - lo)))
    return [min(n_max, lo + (1 if i < extra else 0)) for i in range(n)]


def alloc_ks(hard_flags: Sequence[bool], budget: float, n_max: int) -> tuple[list[int], int]:
    r"""«hard 표시 문제에 k_max, 나머지에 1» — 평균이 `budget` 이 되도록 k_max 를 정한다.
        f = hard 비율,  k_max = 1 + (B − 1)/f   (f=0 이면 배분이 불가능 → 전부 1)
    k_max 는 [1, n_max] 로 자른다(표본이 N 개뿐이라 그 위로는 못 간다) — 잘리면 실제 평균
    예산이 B 보다 작아지므로 호출자가 `realized_budget` 을 **함께 보고**한다."""
    n = len(hard_flags)
    n_hard = sum(1 for h in hard_flags if h)
    if n == 0 or n_hard == 0 or budget <= 1.0:
        return [1] * n, 1
    f = n_hard / n
    k_max = int(round(1.0 + (float(budget) - 1.0) / f))
    k_max = max(1, min(int(n_max), k_max))
    return [k_max if h else 1 for h in hard_flags], k_max


def _mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs if x is not None and MM._cdr._finite(x)]
    return sum(xs) / len(xs) if xs else _NAN


def bootstrap_ci(diffs: Sequence[float], *, n_boot: int = 2000, seed: int = 7,
                 alpha: float = 0.05) -> tuple[float, float]:
    """문제 단위 부트스트랩(차이의 평균). 표본이 3 미만이면 (NaN, NaN)."""
    xs = [float(d) for d in diffs]
    if len(xs) < 3:
        return _NAN, _NAN
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(int(n_boot)):
        means.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(alpha / 2 * len(means))], means[int((1 - alpha / 2) * len(means)) - 1])


def allocation_table(groups: Sequence[Sequence[Mapping]], golds: Sequence[str],
                     *, budgets: Sequence[float] = BUDGETS, seed: int = 7) -> dict:
    r"""예산별 세 줄(stated / uniform / oracle)과 그 차이의 부트스트랩 CI.

    ★oracle 은 **같은 개수**의 문제를 고른다 — 말 대신 실제 그룹 pass rate 가 낮은 순서로
      stated_hard 와 같은 수만큼. 예산이 같아야 «판단이 좋았나»가 비교된다(오라클이 더 많이
      쓰면 그건 더 큰 예산의 효과다).
    ★stated 판단은 **첫 표본**의 버킷이다(배포 시 한 번만 말한다).
    """
    n = len(groups)
    n_max = min((len(g) for g in groups), default=0)
    stated = [(g[0].get("bucket") if g else None) for g in groups]
    hard = [b == "hard" for b in stated]
    p_group = [_mean([float(r.get("r_corr", 0)) for r in g]) for g in groups]
    n_hard = sum(1 for h in hard if h)
    # 오라클: pass rate 가 낮은 순서로 n_hard 개(동률은 인덱스 순 — 결정적)
    order = sorted(range(n), key=lambda i: (p_group[i] if MM._cdr._finite(p_group[i]) else 1.0, i))
    oracle_hard = [False] * n
    for i in order[:n_hard]:
        oracle_hard[i] = True
    out: dict = {"stated_hard_frac": (n_hard / n) if n else _NAN, "n_max_samples": n_max,
                 "rows": []}
    for b in budgets:
        ks_s, kmax_s = alloc_ks(hard, b, n_max)
        ks_o, _ = alloc_ks(oracle_hard, b, n_max)
        ks_u = uniform_ks(n, b, n_max)
        acc_s = _acc_from_ks(groups, golds, ks_s)
        acc_o = _acc_from_ks(groups, golds, ks_o)
        acc_u = _acc_from_ks(groups, golds, ks_u)
        lo, hi = bootstrap_ci([a - c for a, c in zip(acc_s, acc_u)], seed=seed)
        out["rows"].append({
            "budget": float(b), "k_max": kmax_s,
            "realized_budget_stated": _mean([float(k) for k in ks_s]),
            "realized_budget_uniform": _mean([float(k) for k in ks_u]),
            "acc_alloc": _mean(acc_s), "acc_uniform": _mean(acc_u), "acc_alloc_oracle": _mean(acc_o),
            "gain": _mean(acc_s) - _mean(acc_u),
            "gain_ci_lo": lo, "gain_ci_hi": hi,
        })
    return out


# ── (b)(d)(e) 요약 ─────────────────────────────────────────────────────────────
def evaluate(problems: Sequence[Mapping], groups: Sequence[Sequence[Mapping]], *,
             baseline_acc: float | None = None, seed: int = 7) -> tuple[list[dict], dict]:
    """행(평탄화)과 요약. `groups` 는 sample_rows 의 출력."""
    if len(groups) != len(problems):
        raise RuntimeError(f"[diff-eval] 그룹 {len(groups)} != 문제 {len(problems)}")
    rows = [r for g in groups for r in g]
    golds = [str(p["gold"]) for p in problems]
    n = max(1, len(rows))
    ng = max(1, len(problems))
    tel = MM.diff_telemetry(rows)
    stated = [(g[0].get("bucket") if g else None) for g in groups]
    pass8 = [_mean([float(r.get("r_corr", 0)) for r in g]) for g in groups]
    # ★캘리브레이션: 말한 버킷(easy0/medium1/hard2) vs (1 − 실현 pass@8). 양수면 «어렵다고
    #   말한 문제가 실제로 덜 풀린다».
    xs = [float(MDF.BUCKETS.index(b)) for b in stated if b in MDF.BUCKETS]
    ys = [1.0 - p for b, p in zip(stated, pass8) if b in MDF.BUCKETS]
    alloc = allocation_table(groups, golds, seed=seed)
    b2 = next((r for r in alloc["rows"] if abs(r["budget"] - 2.0) < 1e-9), {})
    acc_first = _mean([float(g[0].get("r_corr", 0)) for g in groups if g])
    tel.update({
        "variant": VARIANT, "n_rows": len(rows), "n_groups": len(problems),
        # acc_first = **첫 표본**의 정오(= 배포 시의 acc@1). gate_judgment.py 의 acc-key.
        "acc_first": acc_first,
        "acc_mean": sum(int(r.get("r_corr", 0)) for r in rows) / n,
        "spearman_stated_vs_pass8": MDF.spearman(xs, ys),
        "pass_easy": _mean([p for b, p in zip(stated, pass8) if b == "easy"]),
        "pass_medium": _mean([p for b, p in zip(stated, pass8) if b == "medium"]),
        "pass_hard": _mean([p for b, p in zip(stated, pass8) if b == "hard"]),
        "stated_hard_frac": alloc["stated_hard_frac"],
        # ★세금: 메타를 쓴 acc@1 − plain 기준선(--baseline_acc). 기준선이 없으면 NaN(«못 쟀다»).
        "acc_tax_vs_baseline": (acc_first - float(baseline_acc)) if baseline_acc is not None else _NAN,
        "baseline_acc": float(baseline_acc) if baseline_acc is not None else _NAN,
        "multi_block": sum(int(r.get("multi_block", 0)) for r in rows) / n,
        "boxed_in_meta": sum(int(r.get("boxed_in_meta", 0)) for r in rows) / n,
        "trunc_rate": sum(int(r.get("truncated", 0)) for r in rows) / n,
        "tokens_per_problem": sum(int(r.get("n_tok", 0)) for r in rows) / ng,
        "allocation": alloc,
        "acc_alloc_b2": b2.get("acc_alloc", _NAN),
        "acc_uniform_b2": b2.get("acc_uniform", _NAN),
        "acc_alloc_oracle_b2": b2.get("acc_alloc_oracle", _NAN),
        "alloc_gain_b2": b2.get("gain", _NAN),
        "alloc_gain_b2_ci_lo": b2.get("gain_ci_lo", _NAN),
        "alloc_gain_b2_ci_hi": b2.get("gain_ci_hi", _NAN),
    })
    return rows, tel


def format_summary(tel: Mapping) -> str:
    def _f(x):
        try:
            return f"{float(x):.3f}"
        except (TypeError, ValueError):
            return "nan"
    lines = [f"  {k:28s} {_f(tel.get(k))}" for k in _SUMMARY_KEYS]
    lines.append("  allocation (평균 예산 B: stated / uniform / oracle, k_max):")
    for r in (tel.get("allocation") or {}).get("rows", []):
        lines.append(f"    B={r['budget']:<4} k_max={r['k_max']:<3} "
                     f"{_f(r['acc_alloc'])} / {_f(r['acc_uniform'])} / {_f(r['acc_alloc_oracle'])} "
                     f"gain={_f(r['gain'])} [{_f(r['gain_ci_lo'])}, {_f(r['gain_ci_hi'])}]")
    return "\n".join(lines)


# ── 생성기(vLLM) ────────────────────────────────────────────────────────────────
def vllm_generator(model_path: str, *, num_samples: int, max_tokens: int, temperature: float,
                   top_p: float, seed: int, gpu_util: float):
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    from src.metacot.math_meta_prompt import render_chat_messages  # noqa: PLC0415
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_util,
              max_model_len=max_tokens + 2048, enforce_eager=True)
    tok = llm.get_tokenizer()

    def g(prompts):
        outs = llm.generate([render_chat_messages(tok, m) for m in prompts], SamplingParams(
            n=num_samples, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed))
        return [[(x.text, int(x.finish_reason == "length"), len(x.token_ids)) for x in o.outputs]
                for o in outs]

    return g, tok


def main(argv=None) -> int:
    from math_retry_eval import load_problems  # noqa: PLC0415  (데이터 로딩 정의를 한 곳에)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="math500",
                    help="math500(기본) 등 math_rollout.DATASETS 키, 또는 parquet:<path>[:<level>]")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default=VARIANT, choices=[VARIANT],
                    help="math_diff 고정 — 난이도 판단 구조에서만 이 지표들이 정의된다")
    ap.add_argument("--num_samples", type=int, default=8, help="문제당 표본 N(배분 상한이기도 하다)")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--baseline_acc", type=float, default=None,
                    help="plain 프롬프트 같은 체크포인트의 acc@1 — 세금(acc_tax_vs_baseline) 계산용")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)

    MM.selftest_math_verify()
    problems = load_problems(a.dataset, limit=a.limit, shuffle_seed=a.shuffle_seed)
    print(f"[diff-eval] {a.dataset}: {len(problems)} 문제 · 표본 {a.num_samples}", flush=True)
    gen, _tok = vllm_generator(a.model_path, num_samples=a.num_samples, max_tokens=a.max_tokens,
                               temperature=a.temperature, top_p=a.top_p, seed=a.seed,
                               gpu_util=a.gpu_util)
    groups = sample_rows(problems, gen)
    rows, tel = evaluate(problems, groups, baseline_acc=a.baseline_acc)
    tel.update({"dataset": a.dataset, "model_path": a.model_path, "max_tokens": a.max_tokens,
                "num_samples": a.num_samples})
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "texts.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "telemetry.json").write_text(json.dumps(tel, ensure_ascii=False, indent=2, default=float))
    print(format_summary(tel))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
