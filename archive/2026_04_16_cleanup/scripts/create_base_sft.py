#!/usr/bin/env python3
"""Create Base SFT data by removing <|meta|> blocks from V2 Meta-CoT data.

Same problems, same solutions, no meta -- for fair comparison.
The base SFT model trains on identical math content without metacognitive
annotations, serving as the E2 ablation baseline.

Input:  /tmp/metacot_v2_trapi.parquet  (4996 rows, columns: messages, source)
Output: sft_data/base_sft.parquet      (same format, meta blocks stripped)

Usage:
    python scripts/create_base_sft.py
    python scripts/create_base_sft.py --input /path/to/v2.parquet --output sft_data/base_sft.parquet
    python scripts/create_base_sft.py --upload   # also upload to HF
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
# Paired meta blocks: <|meta|> ... <|/meta|>
META_BLOCK_PAIRED = re.compile(
    r"<\|meta\|>.*?<\|/meta\|>",
    re.DOTALL,
)
# Unpaired opening tag: <|meta|> followed by content until end-of-string
# (used as fallback after paired blocks are removed)
META_BLOCK_ORPHAN_OPEN = re.compile(
    r"<\|meta\|>[^\n]*(?:\n(?!.*\\boxed).*)*",
    re.DOTALL,
)

DEFAULT_INPUT = "/tmp/metacot_v2_trapi.parquet"
DEFAULT_OUTPUT = "sft_data/base_sft.parquet"
HF_REPO_ID = "iamseungpil/metacot"


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def strip_meta_blocks(text: str) -> str:
    """Remove all <|meta|>...<|/meta|> blocks from text.

    Handles both paired blocks and unpaired orphan <|meta|> tags
    (where the model forgot to close the tag). Preserves the math
    solution and \\boxed{} answer intact.
    """
    # Step 1: Remove paired blocks
    cleaned = META_BLOCK_PAIRED.sub("", text)

    # Step 2: Remove any remaining orphan <|meta|> lines
    # These are lines that contain <|meta|> but no matching </meta>
    if "<|meta|>" in cleaned:
        lines = cleaned.split("\n")
        filtered = []
        in_orphan_block = False
        for line in lines:
            if "<|meta|>" in line:
                in_orphan_block = True
                continue
            if in_orphan_block:
                # End orphan block at next blank line or line with math content
                stripped = line.strip()
                if not stripped or "\\boxed" in line or re.match(r"^[A-Z\[]", stripped):
                    in_orphan_block = False
                    if stripped:
                        filtered.append(line)
                # else: skip lines inside orphan block
                continue
            filtered.append(line)
        cleaned = "\n".join(filtered)

    # Step 3: Remove any remaining orphan <|/meta|> tags
    cleaned = cleaned.replace("<|/meta|>", "")
    # Also remove any bare <|meta|> that slipped through
    cleaned = cleaned.replace("<|meta|>", "")

    # Collapse runs of 3+ newlines into 2 (one blank line)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()
    return cleaned


def process_row(messages_str: str) -> str:
    """Process one row: parse messages, strip meta from assistant, return JSON."""
    messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str

    processed = []
    for msg in messages:
        if msg["role"] == "assistant":
            msg = dict(msg)  # shallow copy to avoid mutating original
            msg["content"] = strip_meta_blocks(msg["content"])
        processed.append(msg)

    return json.dumps(processed, ensure_ascii=False)


def validate_output(df_in: pd.DataFrame, df_out: pd.DataFrame) -> None:
    """Validate that stripping was successful."""
    assert len(df_in) == len(df_out), (
        f"Row count mismatch: {len(df_in)} -> {len(df_out)}"
    )

    meta_remaining = 0
    boxed_missing = 0
    empty_assistant = 0

    for _, row in df_out.iterrows():
        msgs = json.loads(row["messages"])
        assistant = msgs[-1]["content"]

        if "<|meta|>" in assistant or "<|/meta|>" in assistant:
            meta_remaining += 1
        if "\\boxed" not in assistant and "boxed{" not in assistant:
            boxed_missing += 1
        if len(assistant.strip()) == 0:
            empty_assistant += 1

    print(f"\n=== Validation ===")
    print(f"Total rows:         {len(df_out)}")
    print(f"Meta blocks left:   {meta_remaining}  (should be 0)")
    print(f"Missing \\boxed:     {boxed_missing}")
    print(f"Empty assistants:   {empty_assistant}  (should be 0)")

    if meta_remaining > 0:
        raise ValueError(f"{meta_remaining} rows still contain meta blocks!")
    if empty_assistant > 0:
        raise ValueError(f"{empty_assistant} rows have empty assistant content!")


def upload_to_hf(file_path: str, repo_id: str = HF_REPO_ID) -> None:
    """Upload the parquet file to HuggingFace."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN", "${HF_TOKEN}")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="base_sft.parquet",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} -> hf://{repo_id}/base_sft.parquet")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create Base SFT data (no meta blocks)")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to V2 TRAPI parquet")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output parquet path")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after creation")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    output_path = project_root / args.output if not Path(args.output).is_absolute() else Path(args.output)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    # Load V2 data
    if not input_path.exists():
        # Fallback: try HuggingFace
        print(f"File not found at {input_path}, trying HuggingFace...")
        from huggingface_hub import hf_hub_download
        input_path = Path(hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="metacot_v2_trapi.parquet",
            repo_type="dataset",
        ))

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    # Show sample before
    sample_msgs = json.loads(df.iloc[0]["messages"])
    sample_before = sample_msgs[-1]["content"]
    meta_count_before = len(META_BLOCK_PAIRED.findall(sample_before))
    print(f"\nSample row 0 before: {len(sample_before)} chars, {meta_count_before} meta blocks")

    # Process all rows
    new_messages = []
    for _, row in df.iterrows():
        new_messages.append(process_row(row["messages"]))

    df_out = pd.DataFrame({
        "messages": new_messages,
        "source": "base_sft",  # Distinguish from metacot_v2_trapi
    })

    # Show sample after
    sample_after_msgs = json.loads(df_out.iloc[0]["messages"])
    sample_after = sample_after_msgs[-1]["content"]
    print(f"Sample row 0 after:  {len(sample_after)} chars, "
          f"{len(META_BLOCK_PAIRED.findall(sample_after))} meta blocks")
    print(f"\n--- Before ---\n{sample_before[:300]}")
    print(f"\n--- After ---\n{sample_after[:300]}")

    # Validate
    validate_output(df, df_out)

    # Compute statistics
    len_before = []
    len_after = []
    for i in range(len(df)):
        msgs_b = json.loads(df.iloc[i]["messages"])
        msgs_a = json.loads(df_out.iloc[i]["messages"])
        len_before.append(len(msgs_b[-1]["content"]))
        len_after.append(len(msgs_a[-1]["content"]))

    avg_before = sum(len_before) / len(len_before)
    avg_after = sum(len_after) / len(len_after)
    print(f"\n=== Statistics ===")
    print(f"Avg assistant length: {avg_before:.0f} -> {avg_after:.0f} chars "
          f"({(1 - avg_after / avg_before) * 100:.1f}% reduction)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)
    print(f"\nSaved {len(df_out)} rows to {output_path}")

    # Upload
    if args.upload:
        upload_to_hf(str(output_path))


if __name__ == "__main__":
    main()
