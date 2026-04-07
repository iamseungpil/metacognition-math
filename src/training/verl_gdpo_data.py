"""Data preparation for veRL GDPO training.

Converts our existing math datasets (GSM8K, MATH, filtered parquets) into
veRL's expected parquet format with the following schema:

    {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": "..."}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": "72"},
    }

Usage:
    python src/training/verl_gdpo_data.py --output data/verl_train.parquet
    python src/training/verl_gdpo_data.py --output data/verl_val.parquet --split val
"""

from __future__ import annotations

import argparse
import random
import re

import pandas as pd


def build_mixed_train(gsm_n: int = 500, math_n: int = 500) -> list[dict]:
    """Build mixed training data from GSM8K + hendrycks_math train splits.

    Mirrors load_mixed_train() from grpo_v2.py but outputs veRL format.
    """
    from datasets import load_dataset as hf_load

    records = []

    # GSM8K train split
    ds = hf_load("openai/gsm8k", "main", split="train")
    for row in ds:
        if len(records) >= gsm_n:
            break
        ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        records.append({
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": row["question"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ans},
        })

    gsm_count = len(records)

    # hendrycks_math train splits
    math_rows = []
    math_configs = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    for cfg in math_configs:
        try:
            ds = hf_load("EleutherAI/hendrycks_math", cfg, split="train")
        except Exception as e:
            print(f"  [WARN] Could not load hendrycks_math/{cfg}: {e}")
            continue
        for row in ds:
            gt = _extract_math_answer(row)
            if not gt:
                continue
            math_rows.append({
                "data_source": f"hendrycks_math/{cfg}",
                "prompt": [{"role": "user", "content": row["problem"]}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gt},
            })

    random.shuffle(math_rows)
    records.extend(math_rows[:math_n])
    random.shuffle(records)

    print(f"Mixed train: {gsm_count} GSM8K + {len(records) - gsm_count} MATH = {len(records)}")
    return records


def build_val(gsm_n: int = 100, math_n: int = 100) -> list[dict]:
    """Build validation data from GSM8K test + MATH-500 test."""
    from datasets import load_dataset as hf_load

    records = []

    # GSM8K test
    ds = hf_load("openai/gsm8k", "main", split="test")
    for row in ds:
        if len(records) >= gsm_n:
            break
        ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        records.append({
            "data_source": "openai/gsm8k",
            "prompt": [{"role": "user", "content": row["question"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ans},
        })

    # MATH-500 test
    try:
        ds = hf_load("HuggingFaceH4/MATH-500", split="test")
        count = 0
        for row in ds:
            if count >= math_n:
                break
            records.append({
                "data_source": "MATH-500",
                "prompt": [{"role": "user", "content": row["problem"]}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": row["answer"]},
            })
            count += 1
    except Exception as e:
        print(f"  [WARN] MATH-500 load failed: {e}")

    print(f"Validation: {len(records)} total")
    return records


def _extract_math_answer(row: dict) -> str:
    """Extract answer from hendrycks_math format.

    Mirrors grpo_v2._extract_math_answer: prefers 'answer' field,
    falls back to nested-brace-aware boxed extraction from 'solution'.
    """
    answer = row.get("answer")
    if answer:
        return str(answer)

    solution = str(row.get("solution", ""))
    # Nested-brace-aware regex for \\boxed{...}
    boxed = re.findall(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', solution)
    if boxed:
        return boxed[-1].strip()
    return solution


def records_to_parquet(records: list[dict], output_path: str) -> None:
    """Convert records to veRL-compatible parquet file.

    veRL's RLHFDataset reads the parquet into a pandas DataFrame and accesses
    columns directly as Python objects:
      - row['prompt'] -> list of dicts, accessed as chat[0]['content']
      - row['reward_model'] -> dict, accessed as gt = item['reward_model']['ground_truth']
      - row['data_source'] -> string

    We store complex types as Python objects in the DataFrame and use
    pyarrow to handle the nested structures in parquet.
    """
    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

    # Verify round-trip
    df_check = pd.read_parquet(output_path)
    sample = df_check.iloc[0]
    prompt = sample['prompt']
    # pandas/pyarrow may deserialize as list or ndarray; verify access pattern
    assert prompt[0]['content'], f"Round-trip check failed: prompt={prompt}"
    print(f"  Round-trip check OK: prompt[0]['content']={prompt[0]['content'][:60]}...")


def main():
    parser = argparse.ArgumentParser(description="Prepare veRL-format parquet data")
    parser.add_argument("--output", default="data/verl_train.parquet")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--gsm_n", type=int, default=500)
    parser.add_argument("--math_n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.split == "train":
        records = build_mixed_train(gsm_n=args.gsm_n, math_n=args.math_n)
    else:
        records = build_val(gsm_n=args.gsm_n, math_n=args.math_n)

    records_to_parquet(records, args.output)


if __name__ == "__main__":
    main()
