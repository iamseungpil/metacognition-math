"""Merge V6 redirect + verify + straight data into clean 10K dataset.

Usage:
    python scripts/merge_v6_clean_10k.py \
        --redirect data/v6_10k_redirect_full.parquet \
        --verify-straight data/v6_10k_verify_straight.parquet \
        --output data/v6_clean_10k_merged.parquet \
        --audit
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd


def compute_pre_meta_length(messages_json: str) -> int:
    """Length of assistant response before first <|meta|> block."""
    msgs = json.loads(messages_json)
    assistant = msgs[-1]["content"]
    idx = assistant.find("<|meta|>")
    return len(assistant[:idx]) if idx >= 0 else len(assistant)


def count_meta_blocks(messages_json: str) -> int:
    msgs = json.loads(messages_json)
    return msgs[-1]["content"].count("<|meta|>")


def audit_dataframe(df: pd.DataFrame, label: str):
    """Print quality audit for a subset."""
    print(f"\n{'='*60}")
    print(f"  AUDIT: {label} ({len(df)} rows)")
    print(f"{'='*60}")

    # Pre-meta lengths (theatrical check)
    pre_lens = df["messages"].apply(compute_pre_meta_length)
    print(f"  Pre-meta length: mean={pre_lens.mean():.0f}, median={pre_lens.median():.0f}, "
          f"min={pre_lens.min()}, max={pre_lens.max()}")
    print(f"  Pre-meta >= 200 chars: {100*(pre_lens >= 200).mean():.1f}%")
    print(f"  Pre-meta == 0 (theatrical): {100*(pre_lens == 0).mean():.1f}%")

    # Meta block counts
    meta_counts = df["messages"].apply(count_meta_blocks)
    print(f"  Meta blocks: mean={meta_counts.mean():.1f}, 0={100*(meta_counts==0).mean():.1f}%, "
          f"1={100*(meta_counts==1).mean():.1f}%, 2+={100*(meta_counts>=2).mean():.1f}%")

    # Scenario distribution
    if "scenario" in df.columns:
        print(f"  Scenarios: {df['scenario'].value_counts().to_dict()}")

    # Difficulty distribution
    if "difficulty" in df.columns:
        print(f"  Difficulty: {df['difficulty'].value_counts().to_dict()}")

    # Behavioral markers
    markers = ["has_verify", "has_switch", "has_conf_drop", "has_overconfidence",
               "has_diagnosis", "has_decomposition"]
    present = [m for m in markers if m in df.columns]
    if present:
        rates = {m: f"{100*df[m].mean():.1f}%" for m in present}
        print(f"  Behaviors: {rates}")

    # Response length
    def resp_len(row):
        msgs = json.loads(row)
        return len(msgs[-1]["content"])
    lengths = df["messages"].apply(resp_len)
    print(f"  Response length: mean={lengths.mean():.0f}, median={lengths.median():.0f}")

    return pre_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redirect", required=True, help="Redirect parquet file")
    parser.add_argument("--verify-straight", required=True, help="Verify+straight parquet file")
    parser.add_argument("--output", required=True, help="Output merged parquet path")
    parser.add_argument("--audit", action="store_true", help="Print quality audit")
    parser.add_argument("--filter-theatrical", action="store_true",
                        help="Remove redirect rows with pre-meta < 50 chars")
    args = parser.parse_args()

    print("Loading data...")
    df_redirect = pd.read_parquet(args.redirect)
    df_vs = pd.read_parquet(args.verify_straight)

    print(f"  Redirect: {len(df_redirect)} rows")
    print(f"  Verify+Straight: {len(df_vs)} rows")

    if args.audit:
        pre_redirect = audit_dataframe(df_redirect, "Redirect (gpt-5.4 full)")
        audit_dataframe(df_vs, "Verify+Straight (gpt-5.4-mini)")

        if args.filter_theatrical:
            theatrical_mask = pre_redirect < 50
            n_theatrical = theatrical_mask.sum()
            if n_theatrical > 0:
                print(f"\n  Filtering {n_theatrical} theatrical redirect rows (pre-meta < 50)")
                df_redirect = df_redirect[~theatrical_mask].reset_index(drop=True)

    # Ensure consistent columns
    common_cols = sorted(set(df_redirect.columns) & set(df_vs.columns))
    df_merged = pd.concat([df_redirect[common_cols], df_vs[common_cols]], ignore_index=True)

    # Shuffle with fixed seed
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Merged: {len(df_merged)} rows")

    if args.audit:
        audit_dataframe(df_merged, "MERGED TOTAL")

    df_merged.to_parquet(args.output, index=False)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
