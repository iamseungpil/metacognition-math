#!/usr/bin/env python3
"""Run on-policy fixed-K repair via vLLM batched inference.

This replaces the previous HF-generate implementation (10% GPU util) with
batched vLLM (>80% GPU util, 60-100x faster).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill.online import (
    load_online_problems,
    load_retriever,
    run_online_fixed_k_repair_rollouts,
    run_online_sdpo_rollouts,
    write_online_sdpo_outputs,
)
from src.training.self_distill import SUPPORTED_SELF_DISTILL_MODES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--question", default=None)
    parser.add_argument("--gold_answer", default=None)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--max_problems", type=int, default=500)
    parser.add_argument("--example_bank", nargs="*", default=None)
    parser.add_argument("--rag_top_k", type=int, default=1)
    parser.add_argument("--mode", default="fixed_k_repair", choices=["sdpo_regen", "fixed_k_repair"])
    parser.add_argument("--dataset_mode", default="auto", choices=["auto", *SUPPORTED_SELF_DISTILL_MODES])
    parser.add_argument(
        "--claim_bearing",
        "--claim-bearing",
        dest="claim_bearing",
        action="store_true",
    )
    parser.add_argument("--repair_candidates", type=int, default=4)
    parser.add_argument(
        "--retrieval_query_mode",
        default="question_only",
        choices=["none", "question_only", "analysis_or_question", "triggered"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    # vLLM-specific
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    print(f"Loading vLLM: {args.model_path} (tp={args.tp_size})")
    from vllm import LLM

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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    problems = load_online_problems(
        question=args.question,
        gold_answer=args.gold_answer,
        input_path=args.input_path,
        benchmark_names=args.benchmarks,
        max_problems=args.max_problems,
    )
    print(f"Loaded {len(problems)} problems")

    retriever = load_retriever(args.example_bank)
    if retriever is not None:
        print("Retriever loaded with example bank")
    elif args.rag_top_k > 0 and args.retrieval_query_mode != "none":
        print(
            "[warn] retrieval was requested, but no example bank was loaded; continuing with retrieval disabled.",
            file=sys.stderr,
        )

    if args.mode == "fixed_k_repair":
        rows = run_online_fixed_k_repair_rollouts(
            llm=llm,
            tokenizer=tokenizer,
            problems=problems,
            output_dir=args.output_dir,
            retriever=retriever,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            rag_top_k=args.rag_top_k,
            repair_candidates=args.repair_candidates,
            retrieval_query_mode=args.retrieval_query_mode,
            chunk_size=args.chunk_size,
            resume=not args.no_resume,
        )
        source_tag = "online_fixed_k_repair"
        dataset_mode = args.dataset_mode if args.dataset_mode != "auto" else "epistemic"
    else:
        rows = run_online_sdpo_rollouts()  # raises NotImplementedError
        source_tag = "online_sdpo_regen"
        dataset_mode = args.dataset_mode if args.dataset_mode != "auto" else "sdpo_regen"

    try:
        payload = write_online_sdpo_outputs(
            rows=rows,
            output_dir=args.output_dir,
            source_tag=source_tag,
            mode=dataset_mode,
            claim_bearing=args.claim_bearing,
            traces_already_written=True,  # online_*_rollouts already streamed traces
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            f"[info] traces preserved at {Path(args.output_dir) / 'online_sdpo_traces.jsonl'}",
            file=sys.stderr,
        )
        sys.exit(2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
