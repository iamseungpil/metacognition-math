#!/usr/bin/env python3
"""Create Verify-Only SFT data: keep only the final verification meta block.

Hypothesis H3: the pre-solve and mid-solve meta blocks add token overhead
without improving accuracy. By keeping ONLY the post-solution verification
block (the one near \\boxed{}), we get calibrated confidence with minimal
overhead.

Logic:
  1. Load V2 Meta-CoT SFT data (from HuggingFace or local)
  2. For each assistant response, find ALL <|meta|>...<|/meta|> blocks
  3. Remove all blocks EXCEPT the LAST one before \\boxed{}
     - The "last" block is typically the verification/confidence assessment
  4. Save as data/verifyonly_sft.parquet

Input:  HuggingFace: datasets/iamseungpil/metacot (metacot_v2_trapi.parquet)
Output: data/verifyonly_sft.parquet

Usage:
    python scripts/create_verifyonly_sft.py
    python scripts/create_verifyonly_sft.py --input /path/to/v2.parquet
    python scripts/create_verifyonly_sft.py --output /path/to/output.parquet
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
META_START = "<|meta|>"
META_END = "<|/meta|>"
META_BLOCK_RE = re.compile(
    r"<\|meta\|>.*?<\|/meta\|>",
    re.DOTALL,
)
HF_REPO_ID = "iamseungpil/metacot"
DEFAULT_OUTPUT = "data/verifyonly_sft.parquet"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_meta_blocks(text: str) -> list:
    """Find all <|meta|>...<|/meta|> blocks with their spans.

    Returns list of (start, end, block_text) tuples.
    """
    blocks = []
    for m in META_BLOCK_RE.finditer(text):
        blocks.append((m.start(), m.end(), m.group()))
    return blocks


def identify_verification_block(text: str, blocks: list) -> int:
    """Identify which block is the verification/post-solution block.

    Heuristic: the verification block is the LAST block that appears
    BEFORE \\boxed{} in the text. If \\boxed{} appears before all blocks,
    or if there's only one block, keep the last block.

    Returns the index of the block to keep, or -1 if no blocks.
    """
    if not blocks:
        return -1

    # Find the position of \\boxed{}
    boxed_match = re.search(r'\\boxed\{', text)
    boxed_pos = boxed_match.start() if boxed_match else len(text)

    # Find the last block before \\boxed{}
    # This is typically the verification block ("Is this correct?", confidence)
    last_before_boxed = -1
    for i, (start, end, _) in enumerate(blocks):
        if start < boxed_pos:
            last_before_boxed = i

    # If no block appears before boxed, keep the last block overall
    if last_before_boxed == -1:
        return len(blocks) - 1

    # Also check: is there a block AFTER boxed? (post-reflection)
    # If so, prefer that one as it's the "what did I learn" block
    # Actually, for verification-only, we want the confidence check
    # which is typically right before boxed. So keep last_before_boxed.
    return last_before_boxed


def strip_to_verifyonly(text: str) -> str:
    """Remove all meta blocks except the verification block.

    Keeps the last meta block before \\boxed{} (the verification check).
    Removes all pre-solve and mid-solve meta blocks.
    """
    blocks = find_meta_blocks(text)

    if len(blocks) <= 1:
        # 0 or 1 blocks: nothing to strip
        return text

    keep_idx = identify_verification_block(text, blocks)

    # Remove all blocks except the one we want to keep
    # Process from end to start to preserve indices
    result = text
    for i in range(len(blocks) - 1, -1, -1):
        if i == keep_idx:
            continue
        start, end, _ = blocks[i]
        result = result[:start] + result[end:]

    # Clean up: collapse runs of 3+ newlines into 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()
    return result


def process_row(messages_str: str) -> str:
    """Process one row: keep only verification meta block in assistant response."""
    messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str

    processed = []
    for msg in messages:
        if msg["role"] == "assistant":
            msg = dict(msg)
            msg["content"] = strip_to_verifyonly(msg["content"])
        processed.append(msg)

    return json.dumps(processed, ensure_ascii=False)


def validate_output(df_in: pd.DataFrame, df_out: pd.DataFrame) -> dict:
    """Validate the verify-only conversion and return statistics."""
    assert len(df_in) == len(df_out), f"Row count mismatch: {len(df_in)} -> {len(df_out)}"

    stats = {
        "total_rows": len(df_out),
        "rows_with_meta": 0,
        "rows_without_meta": 0,
        "avg_blocks_before": 0,
        "avg_blocks_after": 0,
        "avg_len_before": 0,
        "avg_len_after": 0,
        "empty_assistant": 0,
        "missing_boxed": 0,
    }

    blocks_before_list = []
    blocks_after_list = []
    len_before_list = []
    len_after_list = []

    for i in range(len(df_in)):
        msgs_in = json.loads(df_in.iloc[i]["messages"])
        msgs_out = json.loads(df_out.iloc[i]["messages"])

        text_in = msgs_in[-1]["content"]
        text_out = msgs_out[-1]["content"]

        blocks_in = len(META_BLOCK_RE.findall(text_in))
        blocks_out = len(META_BLOCK_RE.findall(text_out))

        blocks_before_list.append(blocks_in)
        blocks_after_list.append(blocks_out)
        len_before_list.append(len(text_in))
        len_after_list.append(len(text_out))

        if blocks_out > 0:
            stats["rows_with_meta"] += 1
        else:
            stats["rows_without_meta"] += 1

        if len(text_out.strip()) == 0:
            stats["empty_assistant"] += 1

        if "\\boxed" not in text_out and "boxed{" not in text_out:
            stats["missing_boxed"] += 1

    stats["avg_blocks_before"] = sum(blocks_before_list) / len(blocks_before_list)
    stats["avg_blocks_after"] = sum(blocks_after_list) / len(blocks_after_list)
    stats["avg_len_before"] = sum(len_before_list) / len(len_before_list)
    stats["avg_len_after"] = sum(len_after_list) / len(len_after_list)

    # Print
    print(f"\n{'='*50}")
    print(f"  VERIFY-ONLY SFT DATA STATISTICS")
    print(f"{'='*50}")
    print(f"  Total rows:           {stats['total_rows']}")
    print(f"  Rows with meta:       {stats['rows_with_meta']}")
    print(f"  Rows without meta:    {stats['rows_without_meta']}")
    print(f"  Avg blocks (before):  {stats['avg_blocks_before']:.1f}")
    print(f"  Avg blocks (after):   {stats['avg_blocks_after']:.1f}")
    print(f"  Avg length (before):  {stats['avg_len_before']:.0f} chars")
    print(f"  Avg length (after):   {stats['avg_len_after']:.0f} chars")
    print(f"  Length reduction:     {(1 - stats['avg_len_after'] / stats['avg_len_before']) * 100:.1f}%")
    print(f"  Empty assistants:     {stats['empty_assistant']}  (should be 0)")
    print(f"  Missing \\boxed:      {stats['missing_boxed']}")

    if stats["empty_assistant"] > 0:
        raise ValueError(f"{stats['empty_assistant']} rows have empty assistant content!")

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create Verify-Only SFT data (keep only verification meta block)"
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to V2 TRAPI parquet (default: download from HuggingFace)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output parquet path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)

    # Load input data
    if args.input and Path(args.input).exists():
        input_path = Path(args.input)
    else:
        # Try local paths first
        local_candidates = [
            project_root / "sft_data" / "metacot_v2_trapi.parquet",
            Path("/tmp/metacot_v2_trapi.parquet"),
            Path("/scratch/metacognition/sft_data/metacot_v2_trapi.parquet"),
        ]
        input_path = None
        for candidate in local_candidates:
            if candidate.exists():
                input_path = candidate
                break

        if input_path is None:
            print("Local file not found, downloading from HuggingFace...")
            from huggingface_hub import hf_hub_download
            input_path = Path(hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="metacot_v2_trapi.parquet",
                repo_type="dataset",
            ))

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows")

    # Show sample before processing
    sample_msgs = json.loads(df.iloc[0]["messages"])
    sample_before = sample_msgs[-1]["content"]
    n_blocks_before = len(META_BLOCK_RE.findall(sample_before))
    print(f"\nSample row 0 before: {len(sample_before)} chars, {n_blocks_before} meta blocks")

    # Process all rows
    new_messages = []
    for _, row in df.iterrows():
        new_messages.append(process_row(row["messages"]))

    df_out = pd.DataFrame({
        "messages": new_messages,
        "source": "verifyonly_sft",
    })

    # Show sample after processing
    sample_msgs_after = json.loads(df_out.iloc[0]["messages"])
    sample_after = sample_msgs_after[-1]["content"]
    n_blocks_after = len(META_BLOCK_RE.findall(sample_after))
    print(f"Sample row 0 after:  {len(sample_after)} chars, {n_blocks_after} meta blocks")

    # Show the kept block
    kept_blocks = META_BLOCK_RE.findall(sample_after)
    if kept_blocks:
        print(f"\n--- Kept verification block ---")
        print(kept_blocks[0][:300])

    # Validate
    stats = validate_output(df, df_out)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {output_path}")

    # Save stats alongside
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
