#!/usr/bin/env python
"""math_retry_eval — M_RETRY/M_AGREE 의 **사전등록 세 곡선**(수정 3/0914b) 을 held-out 에서 잰다.

문제(MATH-500 / AIME25 등 math_rollout.DATASETS, 또는 parquet:<path>[:<level>]) × K 롤아웃을
`--variant`(math_retry 또는 math_agree) 프롬프트로 생성하고, 행마다
`src.training.math_meta.split_attempts` 로 첫 답 / 판단 블록 / 두 번째 시도 / 최종 답을
분해해 채점한다(학습 텔레메트리와 **같은 함수** `retry_telemetry`/`agree_core_metrics` —
학습 중 숫자와 held-out 숫자의 정의가 갈리면 비교가 무의미).

요약(공통, 두 변형 모두):
    first_acc / final_acc                    첫 답·최종 답 정확도
    gain_from_retry = final_acc − first_acc  ← 이것이 «메타 행동 → 정확도» 의 직접 측정
    redirect_rate | first wrong / | first right   판단의 선택성(둘 다 오르면 «자주 말하기» 재현 = 실패)
    judgment_acc                             틀렸으면 redirect·맞았으면 verify 를 맞힌 비율
    second_attempt_rate, trunc_rate, mean tokens

--variant math_agree 전용 추가 지표:
    agree_pred_mean / agree_true_mean / agree_mae   예측 대 실측 형제 동의율, 평균 절대오차
    agree_auc_wrong / agree_auc_wrong_mixed         (1−agree_pred) 가 첫 답 오답을 가르는 AUC
                                                     (전체 / 혼합 문제로만 제한)
    minority_rate, redirect_rate_given_minority/majority   소수·다수 일치일 때 redirect 선택성
    agree_calibration_bins                          agree_pred 5-bin 보정 곡선(pred vs true)

★변형은 math_retry|math_agree 로 고정한다 — 둘 다 첫 답 뒤 판단 블록·재시도 구조가 있어야
세 곡선이 정의된다. agree_true 는 held-out 에서 **같은 문제의 K 샘플**(group_id)로 사후
계산한다(math_meta.compute_agree_true) — 학습 배치의 uid 그룹과 같은 개념, 다른 키.
생성기는 주입 가능(`Generator`)하다 — 단위 테스트는 모의 생성기로 돈다(GPU 불필요).

사용법:
  math_retry_eval.py --dataset math500 --model_path $WORK/merged/cd9_M_RETRY_s1_r6144/step_30 \\
      --num_samples 8 --max_tokens 6144 --out_dir $WORK/eval/cd9_M_RETRY_s1_r6144/step_30/math500_retry
  math_retry_eval.py --dataset math500 --variant math_agree \\
      --model_path $WORK/merged/cd9_M_AGREE_s1_r6144/step_30 --num_samples 8 --max_tokens 6144 \\
      --out_dir $WORK/eval/cd9_M_AGREE_s1_r6144/step_30/math500_retry
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.metacot.math_meta_prompt import (  # noqa: E402
    MATH_PROMPT_VARIANTS, build_math_prompt, render_chat_messages,
)
from src.training import math_meta as MM  # noqa: E402
from src.training import retry_metrics as RMET  # noqa: E402

VARIANT = "math_retry"
_EVAL_VARIANTS = ("math_retry", "math_agree")
assert VARIANT in MATH_PROMPT_VARIANTS and all(v in MATH_PROMPT_VARIANTS for v in _EVAL_VARIANTS)

# 생성기 시그니처: prompts(chat 메시지 리스트들) -> 프롬프트마다 [(text, truncated, n_tok), ...] K 개.
Generator = Callable[[Sequence[list[dict]]], Sequence[Sequence[tuple[str, int, int]]]]


def evaluate(problems: Sequence[Mapping], generate: Generator, *, num_samples: int,
            variant: str = VARIANT) -> tuple[list[dict], dict]:
    """행(롤아웃)과 요약. problems = [{problem, gold}, ...].

    ★variant="math_agree" 이면 행은 `MM.parse_agree_row`(agree_pred/agree_line 포함)로 뽑고,
    같은 problem(group_id) 안의 K 샘플로 `MM.compute_agree_true` 를 사후 계산해 붙인다 —
    학습 배치의 uid 그룹과 같은 개념이되 held-out 은 그룹 크기가 num_samples 그대로다."""
    if variant not in _EVAL_VARIANTS:
        raise ValueError(f"[retry-eval] variant={variant!r} 는 {_EVAL_VARIANTS} 에 없다")
    prompts = [build_math_prompt(p["problem"], variant) for p in problems]
    outs = generate(prompts)
    if len(outs) != len(problems):
        raise RuntimeError(f"[retry-eval] 생성기가 {len(outs)} 개를 돌려줬다(문제 {len(problems)})")
    rows = []
    for gi, (src, samples) in enumerate(zip(problems, outs)):
        for text, trunc, n_tok in samples:
            if variant == "math_agree":
                r = MM.parse_agree_row(text, str(src["gold"]), src["problem"], truncated=int(trunc))
            else:
                r = MM.parse_retry_row(text, str(src["gold"]), src["problem"], truncated=int(trunc))
            r.update({"group_id": f"g{gi}", "problem_id": gi, "n_tok": int(n_tok),
                      "r_corr": int(r["final_correct"]),
                      "redirected": int(r["decision"] == "redirect" and bool(r["has_second_attempt"]))})
            rows.append(r)
    if variant == "math_agree":
        keys = [r["group_id"] for r in rows]
        for r, at in zip(rows, MM.compute_agree_true(rows, keys)):
            r["agree_true"] = at
    tel = MM.retry_telemetry(rows)
    n = max(1, len(rows))
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)
    tel.update({
        "variant": variant, "num_samples": num_samples,
        "n_rows": len(rows), "n_groups": len(groups),
        "gain_from_retry": tel["final_acc"] - tel["first_acc"],
        "emit_rate": sum(int(r["emitted"]) for r in rows) / n,
        "multi_block_rate": sum(1 for r in rows if int(r["n_blocks"]) > 1) / n,
        "boxed_in_meta": sum(int(r["boxed_in_meta"]) for r in rows) / n,
        "pass_at_n_final": sum(int(any(x["final_correct"] for x in g)) for g in groups.values()) / max(1, len(groups)),
        "pass_at_n_first": sum(int(any(x["first_correct"] for x in g)) for g in groups.values()) / max(1, len(groups)),
        "mean_tokens": sum(int(r["n_tok"]) for r in rows) / n,
        # 재시도한 행만 놓고: 첫 답 오답 → 최종 정답으로 «구제»된 비율 / 정답 → 오답으로 «탈선»한 비율
        "rescue_rate_given_redirected": _cond(rows, lambda r: r["redirected"],
                                              lambda r: not r["first_correct"] and r["final_correct"]),
        "derail_rate_given_redirected": _cond(rows, lambda r: r["redirected"],
                                              lambda r: r["first_correct"] and not r["final_correct"]),
        # ★수정 3b 와 같은 «변경» 정의(수학 동치, math_meta.answers_equivalent): 재시도한 행 중 답을
        #   실제로 바꾼 비율. 표기만 바꾼 재시도(7→7.0)는 변경이 아니다 — 학습 항과 held-out 이 같은 자.
        "changed_rate_given_redirected": _cond(rows, lambda r: r["redirected"], lambda r: r["answer_changed"]),
    })
    # ★within-problem 선택성(retry_metrics) — K 샘플이 실제로 있는 group_id 로 배치-로컬
    #   first_pass_rate 를 계산해 mixed 문제(0<rate<1)로만 제한한다. redirect_rate_given_wrong/right
    #   (전 문제 통합)와 달리 «난이도»와 «문제 안 선택성»을 가른다.
    tel.update(RMET.all_metrics(rows, group_key="group_id"))
    if variant == "math_agree":
        # ★스펙 지시: agree_mae/calibration bins/agree_auc_wrong(혼합 제한)/redirect|minority/majority.
        #   agree_core_metrics 의 agree_auc_wrong(전체)·minority_rate 등은 학습 텔레메트리와 같은
        #   정의 — agree_auc_wrong_mixed/calibration bins 만 held-out 전용으로 추가한다.
        tel.update(MM.agree_core_metrics(rows))
        tel["agree_auc_wrong_mixed"] = MM.agree_auc_wrong_mixed(rows, group_key="group_id")
        tel["agree_calibration_bins"] = MM.agree_calibration_bins(rows)
    return rows, tel


def _cond(rows, cond, pred) -> float:
    xs = [r for r in rows if cond(r)]
    return (sum(1 for r in xs if pred(r)) / len(xs)) if xs else float("nan")


_AGREE_SUMMARY_KEYS = ("agree_pred_mean", "agree_true_mean", "agree_mae", "agree_auc_wrong",
                      "agree_auc_wrong_mixed", "minority_rate", "redirect_rate_given_minority",
                      "redirect_rate_given_majority", "agree_line_rate")


def format_summary(tel: Mapping) -> str:
    def _f(x):
        try:
            return f"{float(x):.3f}"
        except (TypeError, ValueError):
            return "nan"
    keys = ("first_acc", "final_acc", "gain_from_retry", "redirect_rate", "redirect_rate_given_wrong",
            "redirect_rate_given_right", "judgment_acc", "second_attempt_rate", "trunc_rate",
            "rescue_rate_given_redirected", "derail_rate_given_redirected", "changed_rate_given_redirected",
            "redirect_rate_all_rows", "emit_rate",
            "multi_block_rate", "boxed_in_meta", "pass_at_n_first", "pass_at_n_final", "mean_tokens",
            # ★within-problem 선택성(retry_metrics) — 난이도와 «문제 안 선택성»을 가른다.
            "n_mixed_problems", "redirect_rate_given_wrong_mixed", "redirect_rate_given_right_mixed",
            "selectivity_mixed", "mean_within_problem_auc", "frac_problems_auc_gt_half",
            "judgment_acc_mixed", "retry_lift_p0", "retry_lift_p0_50", "retry_lift_p50_100",
            "retry_lift_p100")
    # ★agree_pred_mean 이 있으면(--variant math_agree) 형제 동의 지표를 뒤에 이어 붙인다.
    if "agree_pred_mean" in tel:
        keys = keys + _AGREE_SUMMARY_KEYS
    lines = [f"  {k:30s} {_f(tel.get(k))}" for k in keys]
    bins = tel.get("agree_calibration_bins")
    if bins:
        lines.append("  agree_calibration_bins (pred/true):")
        for b in bins:
            lines.append(f"    {b['bin']:>12s} n={b['n']:4d} pred={_f(b['pred_mean'])} true={_f(b['true_mean'])}")
    return "\n".join(lines)


def vllm_generator(model_path: str, *, num_samples: int, max_tokens: int, temperature: float,
                   top_p: float, seed: int, gpu_util: float) -> Generator:
    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=model_path, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_util,
              max_model_len=max_tokens + 2048, enforce_eager=True)
    tok = llm.get_tokenizer()

    def gen(prompts):
        outs = llm.generate([render_chat_messages(tok, m) for m in prompts], SamplingParams(
            n=num_samples, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed))
        return [[(x.text, int(x.finish_reason == "length"), len(x.token_ids)) for x in o.outputs]
                for o in outs]
    return gen


def load_problems(dataset: str, *, limit: int = 0, shuffle_seed: int = 0) -> list[dict]:
    from math_rollout import DATASETS  # noqa: PLC0415
    if dataset.startswith("parquet:"):
        import pandas as pd  # noqa: PLC0415
        parts = dataset.split(":", 2)
        df = pd.read_parquet(parts[1])
        lvl = parts[2] if len(parts) > 2 else None
        rows = [{"problem": r["problem"], "gold": str(r["gold"])}
                for _, r in df.iterrows() if not lvl or (r["extra_info"] or {}).get("level") == lvl]
    elif dataset in DATASETS:
        from datasets import load_dataset  # noqa: PLC0415
        ds_id, cfg, split, qcol, acol = DATASETS[dataset]
        ds = load_dataset(ds_id, cfg, split=split) if cfg else load_dataset(ds_id, split=split)
        rows = [{"problem": r[qcol], "gold": str(r[acol])} for r in ds]
    else:
        raise SystemExit(f"unknown dataset {dataset!r} (math_rollout.DATASETS 또는 parquet:<path>)")
    if shuffle_seed:
        import random as _rnd  # noqa: PLC0415
        _rnd.Random(shuffle_seed).shuffle(rows)
    return rows[:limit] if limit else rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default=VARIANT, choices=list(_EVAL_VARIANTS),
                    help="math_retry|math_agree — 세 곡선(+ math_agree 전용 지표)은 이 구조에서만 정의된다")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)

    MM.selftest_math_verify()
    problems = load_problems(a.dataset, limit=a.limit, shuffle_seed=a.shuffle_seed)
    print(f"[retry-eval] {a.dataset}: {len(problems)} 문제 x {a.num_samples} · variant={a.variant}", flush=True)
    gen = vllm_generator(a.model_path, num_samples=a.num_samples, max_tokens=a.max_tokens,
                         temperature=a.temperature, top_p=a.top_p, seed=a.seed, gpu_util=a.gpu_util)
    rows, tel = evaluate(problems, gen, num_samples=a.num_samples, variant=a.variant)
    tel.update({"dataset": a.dataset, "model_path": a.model_path, "max_tokens": a.max_tokens})
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
