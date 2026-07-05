#!/usr/bin/env python3
"""vLLM-based eval on the 1030 benchmark (GSM8K 500 + MATH-500 + AIME2024 30).

Single-shot evaluation with configurable `max_tokens`. Output is compatible with
`scripts/analyze_entropy_meta.py` (parquet with `completion`, `is_correct`, `question`).

Reuses helpers from `src/eval/eval_hf.py` (loaders + correctness) and
`src/metacot/prompt.py` (meta block parsing). No changes to those files.

Example:
  python scripts/eval_vllm_1030.py \\
    --model_path checkpoints/verl_base_matched_0410_merged \\
    --model_name base_grpo_step300_16k \\
    --output_dir results/eval_1030_base_grpo_step300_16k/ \\
    --max_tokens 16384
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.eval_hf import check_correctness, extract_answer, load_benchmarks
from src.metacot.prompt import META_END, META_START, parse_meta_blocks
from src.training.tokenizer_utils import ensure_meta_tokens_not_special


def render_chat_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return (
            f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        )


def summarize(results: list[dict]) -> dict:
    df = pd.DataFrame(results)
    per_bench: dict[str, dict] = {}
    for bench in sorted(df["benchmark"].unique()):
        bdf = df[df["benchmark"] == bench]
        confs = [c for row in bdf["meta_confidences"] for c in (row or [])]
        per_bench[bench] = {
            "n": int(len(bdf)),
            "accuracy": float(bdf["is_correct"].mean()),
            "meta_emission_rate": float((bdf["num_meta_blocks"] > 0).mean()),
            "avg_num_meta_blocks": float(bdf["num_meta_blocks"].mean()),
            "avg_confidence": float(sum(confs) / len(confs)) if confs else None,
            "avg_completion_length_tokens": float(
                bdf["completion_length_tokens"].mean()
            ),
            "num_truncated": (
                int((bdf["finish_reason"] == "length").sum())
                if "finish_reason" in bdf.columns
                else None
            ),
        }
    return {
        "total": int(len(df)),
        "benchmarks": per_bench,
        "overall_accuracy": float(df["is_correct"].mean()),
        "overall_meta_emission_rate": float((df["num_meta_blocks"] > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--benchmarks", nargs="+", default=["gsm8k", "math500", "aime2024"]
    )
    parser.add_argument("--max_problems", type=int, default=500)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=20480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1)
    # anti-degeneration decode knobs (defaults = no-op; preserve baseline behavior).
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start_ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    print(f"Loading benchmarks: {args.benchmarks}")
    problems = load_benchmarks(args.benchmarks, args.max_problems)
    print(f"Total problems: {len(problems)}")
    if not problems:
        raise SystemExit("No problems loaded; aborting.")

    print(f"Loading vLLM: {args.model_path} (tp={args.tp_size})")
    from vllm import LLM, SamplingParams
    import torch
    import vllm as _vllm_pkg

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=args.seed,
    )
    tokenizer = llm.get_tokenizer()
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [render_chat_prompt(tokenizer, p["question"]) for p in problems]
    prompt_token_lens = [
        len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts
    ]

    sp_kwargs = dict(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        skip_special_tokens=False,
    )
    # anti-degeneration knobs: only attach when non-default so baseline is byte-identical.
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        sp_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.presence_penalty:
        sp_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p and args.min_p > 0.0:
        sp_kwargs["min_p"] = args.min_p
    if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
        sp_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    try:
        sampling = SamplingParams(**sp_kwargs)
    except TypeError as exc:
        # older vLLM may not accept no_repeat_ngram_size — drop it and proceed.
        dropped = sp_kwargs.pop("no_repeat_ngram_size", None)
        print(f"[decode-sweep] SamplingParams fallback, dropped no_repeat_ngram_size={dropped}: {exc}")
        sampling = SamplingParams(**sp_kwargs)
    print(f"[decode-sweep] sampling kwargs (non-default): "
          f"{ {k: v for k, v in sp_kwargs.items() if k not in ('n','temperature','top_p','max_tokens','seed','skip_special_tokens')} }")

    print(
        f"Generating {len(prompts)} prompts x n={args.num_samples} "
        f"(max_tokens={args.max_tokens})"
    )
    outputs = llm.generate(prompts, sampling)

    results: list[dict] = []
    for prob, prompt_tok_len, out in zip(problems, prompt_token_lens, outputs):
        for s_idx, sample in enumerate(out.outputs):
            completion = sample.text
            completion_tok = len(sample.token_ids)
            parsed = parse_meta_blocks(completion)
            confs = parsed["confidences"]
            ans = extract_answer(completion)
            is_correct = check_correctness(completion, prob["gold_answer"])
            results.append(
                {
                    "benchmark": prob["benchmark"],
                    "question": prob["question"],
                    "gold_answer": prob["gold_answer"],
                    "completion": completion,
                    "answer_extracted": ans,
                    "is_correct": bool(is_correct),
                    "num_meta_blocks": int(parsed["num_blocks"]),
                    "meta_confidences": list(confs),
                    "avg_confidence": (
                        float(sum(confs) / len(confs)) if confs else None
                    ),
                    "completion_length_tokens": int(completion_tok),
                    "completion_length_chars": int(len(completion)),
                    "prompt_length_tokens": int(prompt_tok_len),
                    "sample_idx": int(s_idx),
                    "finish_reason": str(sample.finish_reason),
                    "stop_reason": (
                        str(sample.stop_reason)
                        if getattr(sample, "stop_reason", None) is not None
                        else None
                    ),
                }
            )

    end_ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    summary = summarize(results)
    summary["model"] = args.model_name

    parquet_path = os.path.join(args.output_dir, f"{args.model_name}.parquet")
    json_path = os.path.join(args.output_dir, f"{args.model_name}.json")
    meta_path = os.path.join(args.output_dir, f"{args.model_name}.metadata.json")

    pd.DataFrame(results).to_parquet(parquet_path, index=False)
    with open(json_path, "w") as f:
        json.dump(
            {"summary": summary, "results": results}, f, indent=2, ensure_ascii=False
        )

    metadata = {
        "model": args.model_name,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "benchmarks": args.benchmarks,
        "max_problems": args.max_problems,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "tp_size": args.tp_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "total_problems": len(problems),
        "total_rows": len(results),
        "hostname": socket.gethostname(),
        "start_utc": start_ts,
        "end_utc": end_ts,
        "vllm_version": getattr(_vllm_pkg, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "meta_token_ids": {
            META_START: tokenizer.convert_tokens_to_ids(META_START),
            META_END: tokenizer.convert_tokens_to_ids(META_END),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved parquet: {parquet_path}")
    print(f"Saved json:    {json_path}")
    print(f"Saved meta:    {meta_path}")
    print(f"\nOverall accuracy: {summary['overall_accuracy']*100:.1f}%")
    for bench, stats in summary["benchmarks"].items():
        print(
            f"  {bench:<10} n={stats['n']:<4} "
            f"acc={stats['accuracy']*100:5.1f}% "
            f"avg_len={stats['avg_completion_length_tokens']:.0f} "
            f"meta_rate={stats['meta_emission_rate']*100:.1f}%"
        )


if __name__ == "__main__":
    main()
