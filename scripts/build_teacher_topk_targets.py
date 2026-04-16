#!/usr/bin/env python3
"""Build teacher top-k targets from a self-distill parquet.

Mainline use:
  - `question_only_best_of_n` claim-bearing meta lane before `meta_only` KL

Side-evidence use:
  - `fixed_k_repair`
  - `sdpo_regen`
"""
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

from src.training.self_distill.teacher_query import (
    build_teacher_query_dataset,
    query_teacher_topk_for_messages,
)
from src.training.self_distill.runtime_tokenizer import prepare_runtime_tokenizer_dir
from src.training.self_distill.trace import read_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher_model_path", required=True)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device_map", default="single", choices=["single", "auto"])
    parser.add_argument("--summary_json", default=None)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    load_dtype = torch.bfloat16 if use_cuda else torch.float32

    model_load_kwargs = {
        "torch_dtype": load_dtype,
        "trust_remote_code": True,
    }
    if use_cuda and args.device_map == "auto":
        model_load_kwargs["device_map"] = "auto"

    tokenizer_path, _ = prepare_runtime_tokenizer_dir(args.teacher_model_path, Path(args.output).parent)
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, **model_load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or args.teacher_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not (use_cuda and args.device_map == "auto"):
        model = model.to("cuda" if use_cuda else "cpu")
    model = model.eval()

    def _query(messages):
        return query_teacher_topk_for_messages(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            top_k=args.top_k,
            max_length=args.max_length,
        )

    input_rows = len(read_table(args.input))
    df = build_teacher_query_dataset(
        args.input,
        query_fn=_query,
        top_k=args.top_k,
        source_tag="teacher_topk_query",
    )
    if df.empty:
        raise ValueError("Teacher top-k query produced 0 rows")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    summary = {
        "input_rows": int(input_rows),
        "rows": int(len(df)),
        "skipped_rows": int(max(0, int(input_rows) - int(len(df)))),
        "teacher_query_top_k": int(args.top_k),
        "avg_teacher_num_positions": float(df["teacher_num_positions"].mean()),
        "avg_teacher_completion_len_tokens": float(df["teacher_completion_len_tokens"].mean()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
