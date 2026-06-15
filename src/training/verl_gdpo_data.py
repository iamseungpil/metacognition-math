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
import json
import random
import re
from pathlib import Path

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


def _load_message_list(raw_messages) -> list[dict]:
    if isinstance(raw_messages, str):
        return json.loads(raw_messages)
    if isinstance(raw_messages, list):
        return raw_messages
    raise TypeError(f"Unsupported messages payload type: {type(raw_messages)!r}")


def _extract_prompt_and_gt_from_messages(raw_messages) -> tuple[str, str]:
    messages = _load_message_list(raw_messages)
    user_text = ""
    assistant_text = ""
    for message in messages:
        role = message.get("role", "")
        content = str(message.get("content", ""))
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant":
            assistant_text = content
    gt = _extract_math_answer({"solution": assistant_text})
    if not user_text or not gt:
        raise ValueError("Could not recover prompt/ground_truth from messages")
    return user_text, gt


def build_v8_redirect_subset(
    meta_path: str,
    base_path: str,
    *,
    allowed_difficulties: tuple[str, ...] = ("medium", "hard"),
    scenario: str = "redirect",
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Build paired redirect-focused RL subsets from V8 meta/base parquets.

    This uses the existing V8 SFT corpora as the authoritative source for
    trigger-conditioned redirect cases. The base-matched parquet is filtered
    by the same row indices so ablations use the exact same question slice.
    """
    meta_df = pd.read_parquet(meta_path)
    base_df = pd.read_parquet(base_path)

    if len(meta_df) != len(base_df):
        raise ValueError(
            f"Meta/base parquet length mismatch: {len(meta_df)} vs {len(base_df)}"
        )

    selector = (
        meta_df["scenario"].eq(scenario)
        & meta_df["difficulty"].isin(list(allowed_difficulties))
    )
    selected_idx = meta_df.index[selector].tolist()
    if not selected_idx:
        raise ValueError("Redirect subset selection produced zero rows")

    rng = random.Random(seed)
    rng.shuffle(selected_idx)
    n_val = max(1, int(round(len(selected_idx) * val_ratio)))
    val_idx = set(selected_idx[:n_val])

    outputs = {
        "meta_train": [],
        "meta_val": [],
        "base_train": [],
        "base_val": [],
    }

    for idx in selected_idx:
        meta_row = meta_df.loc[idx]
        base_row = base_df.loc[idx]
        prompt_text, gt = _extract_prompt_and_gt_from_messages(meta_row["messages"])
        data_source = str(meta_row.get("source", "v8_redirect"))

        record = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_text}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "split_tags": {
                "scenario": str(meta_row.get("scenario", "")),
                "difficulty": str(meta_row.get("difficulty", "")),
                "trigger": str(meta_row.get("trigger", "")),
                "row_index": int(idx),
            },
        }

        target_prefix = "meta_val" if idx in val_idx else "meta_train"
        outputs[target_prefix].append(record)

        base_prompt, base_gt = _extract_prompt_and_gt_from_messages(base_row["messages"])
        if base_prompt != prompt_text or base_gt != gt:
            raise ValueError(f"Base-matched row mismatch at index {idx}")

        base_record = {
            **record,
            "data_source": f"{data_source}::base_matched",
        }
        target_prefix = "base_val" if idx in val_idx else "base_train"
        outputs[target_prefix].append(base_record)

    return outputs


def _gold_is_rule_gradable(gt: str) -> bool:
    """True iff gold answer is rule-gradable (numeric / boxed-able), NOT prose.
    Drops omni-math prose golds (~26%) that rule-based grading scores 0 even when
    the model is right (spec 2026-06-15 §3.6)."""
    if gt is None:
        return False
    s = str(gt).strip()
    if not s:
        return False
    # Reject obvious prose: \text{...} wrappers or >2 alphabetic words.
    if "\\text{" in s:
        return False
    words = re.findall(r"[A-Za-z]{2,}", s)
    has_math = any(t in s for t in ("\\frac", "\\sqrt", "\\pi", "(", "="))
    # Reject prose: multiple alphabetic words with no math structure,
    # e.g. "Player 0 wins" — a stray digit does not make it gradable.
    if len(words) >= 2 and not has_math:
        return False
    # Accept if it contains a digit, a fraction/sqrt/expression token, or is short symbolic.
    if re.search(r"[0-9]", s) or has_math:
        return True
    return len(s) <= 8  # short symbolic like 'x', 'a+b'


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
    parser.add_argument(
        "--mode",
        choices=["mixed", "v8_redirect"],
        default="mixed",
        help="mixed: legacy GSM8K+MATH builder, v8_redirect: paired redirect-focused subset from V8 parquets",
    )
    parser.add_argument("--gsm_n", type=int, default=500)
    parser.add_argument("--math_n", type=int, default=500)
    parser.add_argument("--meta_path", default="data/v8_meta_inside_think.parquet")
    parser.add_argument("--base_path", default="data/v8_base_matched_clean.parquet")
    parser.add_argument("--out_train_meta", default="data/verl_train_redirect.parquet")
    parser.add_argument("--out_val_meta", default="data/verl_val_redirect.parquet")
    parser.add_argument("--out_train_base", default="data/verl_train_redirect_base.parquet")
    parser.add_argument("--out_val_base", default="data/verl_val_redirect_base.parquet")
    parser.add_argument("--scenario", default="redirect")
    parser.add_argument("--difficulties", nargs="*", default=["medium", "hard"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.mode == "mixed":
        if args.split == "train":
            records = build_mixed_train(gsm_n=args.gsm_n, math_n=args.math_n)
        else:
            records = build_val(gsm_n=args.gsm_n, math_n=args.math_n)
        records_to_parquet(records, args.output)
        return

    outputs = build_v8_redirect_subset(
        meta_path=args.meta_path,
        base_path=args.base_path,
        allowed_difficulties=tuple(args.difficulties),
        scenario=args.scenario,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    for key, out_path in [
        ("meta_train", args.out_train_meta),
        ("meta_val", args.out_val_meta),
        ("base_train", args.out_train_base),
        ("base_val", args.out_val_base),
    ]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        records_to_parquet(outputs[key], out_path)
        print(f"[{key}] {len(outputs[key])} rows -> {out_path}")


if __name__ == "__main__":
    main()
