#!/usr/bin/env python3
"""Run on-policy SDPO-style regeneration traces and emit distill artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--is_lora", action="store_true")
    parser.add_argument("--question", default=None)
    parser.add_argument("--gold_answer", default=None)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--benchmarks", nargs="*", default=None)
    parser.add_argument("--max_problems", type=int, default=30)
    parser.add_argument("--example_bank", nargs="*", default=None)
    parser.add_argument("--rag_top_k", type=int, default=1)
    parser.add_argument("--mode", default="sdpo_regen", choices=["sdpo_regen", "fixed_k_repair"])
    parser.add_argument("--dataset_mode", default="auto", choices=["auto", *SUPPORTED_SELF_DISTILL_MODES])
    parser.add_argument("--claim-bearing", action="store_true")
    parser.add_argument("--repair_candidates", type=int, default=4)
    parser.add_argument("--retrieval_query_mode", default="question_only", choices=["none", "question_only", "analysis_or_question", "triggered"])
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="single", choices=["single", "auto"])
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    load_dtype = torch.bfloat16 if use_cuda else torch.float32
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    model_load_kwargs = {
        "torch_dtype": load_dtype,
        "trust_remote_code": True,
    }
    if use_cuda and args.device_map == "auto":
        model_load_kwargs["device_map"] = "auto"

    if args.is_lora:
        from peft import PeftModel

        base_path = args.base_model or "checkpoints/qwen3_meta_sft"
        model = AutoModelForCausalLM.from_pretrained(base_path, **model_load_kwargs)
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_load_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not (use_cuda and args.device_map == "auto"):
        model = model.to("cuda" if use_cuda else "cpu")
    model = model.eval()

    problems = load_online_problems(
        question=args.question,
        gold_answer=args.gold_answer,
        input_path=args.input_path,
        benchmark_names=args.benchmarks,
        max_problems=args.max_problems,
    )
    retriever = load_retriever(args.example_bank)
    if args.rag_top_k > 0 and args.retrieval_query_mode != "none" and retriever is None:
        print(
            "[warn] retrieval was requested, but no example bank was loaded; continuing with retrieval disabled.",
            file=sys.stderr,
        )

    if args.mode == "fixed_k_repair":
        rows = run_online_fixed_k_repair_rollouts(
            model=model,
            tokenizer=tokenizer,
            problems=problems,
            retriever=retriever,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            rag_top_k=args.rag_top_k,
            repair_candidates=args.repair_candidates,
            retrieval_query_mode=args.retrieval_query_mode,
        )
        dataset_mode = "epistemic"
        source_tag = "online_fixed_k_repair"
    else:
        rows = run_online_sdpo_rollouts(
            model=model,
            tokenizer=tokenizer,
            problems=problems,
            retriever=retriever,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            rag_top_k=args.rag_top_k,
        )
        dataset_mode = "sdpo_regen"
        source_tag = "online_sdpo_regen"
    if args.dataset_mode != "auto":
        dataset_mode = args.dataset_mode
    payload = write_online_sdpo_outputs(
        rows=rows,
        output_dir=args.output_dir,
        source_tag=source_tag,
        mode=dataset_mode,
        claim_bearing=args.claim_bearing,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
