#!/usr/bin/env python3
"""Per-token entropy analysis around <|meta|> blocks in Meta-CoT completions.

Loads a Qwen3-8B checkpoint and eval parquet, performs HuggingFace forward
passes to compute full-vocabulary Shannon entropy and next-token surprisal,
then measures before/on/after entropy changes around <|meta|>...<|/meta|>
markers.  Results are split by correctness to test whether metacognitive
blocks resolve uncertainty (entropy drop) or not.

Adapted from:
  - behavior-uncertainty/analysis/analyze_token_distribution.py  (HF forward pass)
  - behavior-uncertainty/scripts/analyze_deep_epistemic.py       (window analysis)

Usage:
    python scripts/analyze_entropy_meta.py \
        --model_path checkpoints/v6_clean_10k_E19 \
        --eval_parquet results/eval_v6_E19/eval_v6_clean_10k_E19.parquet \
        --output_dir results/entropy_analysis/ \
        --max_samples 200 \
        --window 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MetaBlockMeasurement:
    """Entropy measurements around a single <|meta|>...<|/meta|> block."""
    sample_id: int
    block_idx: int
    is_correct: bool
    before_entropy: float
    meta_entropy: float
    after_entropy: float
    delta: float  # after - before
    before_surprisal: float
    meta_surprisal: float
    after_surprisal: float
    delta_surprisal: float  # after - before
    meta_start_tok: int
    meta_end_tok: int
    meta_length_tokens: int


@dataclass
class SampleMeasurement:
    """Aggregated measurement for one eval sample."""
    sample_id: int
    is_correct: bool
    num_meta_blocks: int
    avg_before_entropy: float
    avg_meta_entropy: float
    avg_after_entropy: float
    avg_delta: float
    avg_before_surprisal: float
    avg_meta_surprisal: float
    avg_after_surprisal: float
    avg_delta_surprisal: float
    full_sequence_entropy: float
    full_sequence_surprisal: float
    completion_length_tokens: int


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entropy analysis around <|meta|> blocks in Meta-CoT completions.",
    )
    p.add_argument(
        "--model_path", type=str, required=True,
        help="Path to HuggingFace checkpoint (e.g. checkpoints/v6_clean_10k_E19)",
    )
    p.add_argument(
        "--eval_parquet", type=str, required=True,
        help="Path to eval parquet with 'completion' and 'is_correct' columns",
    )
    p.add_argument(
        "--output_dir", type=str, default="results/entropy_analysis/",
        help="Directory for output JSON and CSV files",
    )
    p.add_argument(
        "--max_samples", type=int, default=200,
        help="Maximum number of samples to process (meta-containing only)",
    )
    p.add_argument(
        "--window", type=int, default=8,
        help="Number of tokens before/after meta markers to measure",
    )
    p.add_argument(
        "--max_seq_len", type=int, default=8192,
        help="Maximum sequence length; longer sequences are skipped",
    )
    p.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype for memory efficiency",
    )
    p.add_argument(
        "--marker_mode", type=str, default="meta",
        choices=["meta", "confidence"],
        help=(
            "Which marker to locate per completion. "
            "'meta' finds <|meta|>...<|/meta|> spans. "
            "'confidence' finds 'confidence: 0.XX' spans as plain text "
            "(used for RL models that emit unwrapped confidence text)."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_path: str,
    dtype: str = "bfloat16",
) -> tuple:
    """Load tokenizer and model with specified dtype and device_map='auto'."""
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[dtype]

    print(f"Loading model from {model_path} (dtype={dtype}, device_map=auto) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt_text(tokenizer, question: str) -> str:
    """Build the chat-formatted prompt that was used during eval.

    The eval pipeline (src/eval/eval_hf.py) uses:
        messages = [{"role": "user", "content": question}]
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    We replicate that exactly.
    """
    messages = [{"role": "user", "content": question}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback: bare text
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


# ---------------------------------------------------------------------------
# Forward pass: entropy and surprisal
# ---------------------------------------------------------------------------

def compute_entropy_and_surprisal(
    tokenizer,
    model,
    prompt_text: str,
    completion_text: str,
    max_seq_len: int = 8192,
) -> Optional[dict]:
    """Run a single forward pass and return per-token entropy + surprisal.

    Returns None if the sequence exceeds max_seq_len.

    Returns dict with keys:
        - token_ids: list[int]  (shifted target IDs, length = seq_len - 1)
        - entropy: np.ndarray   (full-vocab Shannon entropy per position)
        - surprisal: np.ndarray (-log_prob of actual next token per position)
        - prompt_token_len: int (number of tokens in the prompt portion)
    """
    # Tokenize prompt alone to find boundary
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    )["input_ids"]
    prompt_token_len = prompt_ids.shape[1]

    # Tokenize full sequence
    full_text = prompt_text + completion_text
    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]

    seq_len = input_ids.shape[1]
    if seq_len > max_seq_len:
        return None

    # Move to model's device (handles multi-GPU with device_map)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Compute in float32 for numerical stability
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = torch.exp(log_probs)

    # Shift: position t predicts token t+1
    target_ids = input_ids[:, 1:]         # (1, seq_len-1)
    log_probs = log_probs[:, :-1]         # (1, seq_len-1, vocab)
    probs = probs[:, :-1]                 # (1, seq_len-1, vocab)

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)  # (seq_len-1,)

    # Surprisal: -log_prob of actual next token
    token_log_probs = log_probs.gather(
        dim=-1, index=target_ids.unsqueeze(-1),
    ).squeeze(-1).squeeze(0)  # (seq_len-1,)
    surprisal = -token_log_probs  # nats

    return {
        "token_ids": target_ids.squeeze(0).cpu().tolist(),
        "entropy": entropy.cpu().float().numpy(),
        "surprisal": surprisal.cpu().float().numpy(),
        "prompt_token_len": prompt_token_len,
    }


# ---------------------------------------------------------------------------
# Meta marker detection
# ---------------------------------------------------------------------------

def find_meta_token_spans(
    tokenizer,
    token_ids: list[int],
    prompt_token_len: int,
) -> list[tuple[int, int]]:
    """Find (start, end) token index pairs for <|meta|>...<|/meta|> blocks.

    Returns positions in the shifted token_ids array (i.e., already offset
    by the shift in compute_entropy_and_surprisal).  Only considers tokens
    after the prompt portion.

    start = index of the first token INSIDE the meta block (after <|meta|>)
    end   = index of the last token INSIDE the meta block (before <|/meta|>)
    """
    # Resolve marker token IDs
    meta_open_ids = tokenizer.encode("<|meta|>", add_special_tokens=False)
    meta_close_ids = tokenizer.encode("<|/meta|>", add_special_tokens=False)

    # After the shift, prompt tokens occupy indices [0, prompt_token_len-2]
    # The first completion token is at index prompt_token_len - 1
    answer_start = max(prompt_token_len - 1, 0)

    def find_subsequence(seq: list[int], subseq: list[int], start_from: int = 0) -> list[int]:
        """Return all starting indices where subseq appears in seq."""
        positions = []
        sub_len = len(subseq)
        for i in range(start_from, len(seq) - sub_len + 1):
            if seq[i:i + sub_len] == subseq:
                positions.append(i)
        return positions

    open_positions = find_subsequence(token_ids, meta_open_ids, answer_start)
    close_positions = find_subsequence(token_ids, meta_close_ids, answer_start)

    # Pair opens with the nearest subsequent close
    spans = []
    used_closes = set()
    for op in open_positions:
        content_start = op + len(meta_open_ids)
        for cp in close_positions:
            if cp >= content_start and cp not in used_closes:
                content_end = cp - 1  # last content token before </meta>
                if content_end >= content_start:
                    spans.append((content_start, content_end))
                used_closes.add(cp)
                break

    return spans


# ---------------------------------------------------------------------------
# Confidence marker detection (plain text, no wrapping)
# ---------------------------------------------------------------------------

def find_confidence_token_spans(
    tokenizer,
    token_ids: list[int],
    prompt_token_len: int,
) -> list[tuple[int, int]]:
    """Find (start, end) token index pairs for `confidence: 0.XX` spans.

    Decodes the completion portion once and regex-matches the textual pattern,
    then maps character offsets back to token indices via tokenizer re-encode.
    Span covers the marker text itself (`confidence: 0.XX`), matching the
    previously published `rl_meta_confidence` output where `conf_length_tokens`
    is ~7 tokens.

    Returns positions in the shifted token_ids array (already offset by the
    shift in compute_entropy_and_surprisal).  Only considers tokens after the
    prompt portion.
    """
    import re

    answer_start = max(prompt_token_len - 1, 0)
    if answer_start >= len(token_ids):
        return []

    # Decode completion portion only, so character offsets start at 0.
    completion_ids = token_ids[answer_start:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)

    # Walk tokens, tracking cumulative char offset per token.
    char_offsets: list[int] = [0]
    acc = ""
    for t in completion_ids:
        acc += tokenizer.decode([t], skip_special_tokens=False)
        char_offsets.append(len(acc))
    # char_offsets[i] is the char offset at the START of token i within completion_text

    def char_to_token(char_pos: int) -> int:
        """Return the index of the first token whose start is >= char_pos."""
        # Binary-ish linear scan (lists are small)
        for i, off in enumerate(char_offsets):
            if off >= char_pos:
                return i
        return len(char_offsets) - 1

    pat = re.compile(r"confidence\s*:\s*\d+(?:\.\d+)?", re.IGNORECASE)
    spans: list[tuple[int, int]] = []
    for m in pat.finditer(completion_text):
        c_start, c_end = m.start(), m.end()
        # Map character offsets back to token indices relative to completion.
        tok_start_rel = char_to_token(c_start)
        tok_end_rel = char_to_token(c_end) - 1
        # Translate to absolute positions in token_ids array (shifted).
        tok_start = answer_start + tok_start_rel
        tok_end = answer_start + tok_end_rel
        if tok_end >= tok_start:
            spans.append((tok_start, tok_end))
    return spans


# ---------------------------------------------------------------------------
# Window extraction
# ---------------------------------------------------------------------------

def extract_window(
    values: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """Safely extract values[start:end], clamping to array bounds."""
    start = max(0, start)
    end = min(len(values), end)
    if start >= end:
        return np.array([], dtype=np.float32)
    return values[start:end]


def measure_meta_block(
    entropy: np.ndarray,
    surprisal: np.ndarray,
    span: tuple[int, int],
    window: int,
    sample_id: int,
    block_idx: int,
    is_correct: bool,
) -> Optional[MetaBlockMeasurement]:
    """Compute before/on/after measurements for a single meta block."""
    content_start, content_end = span
    content_end_exclusive = content_end + 1

    before_ent = extract_window(entropy, content_start - window, content_start)
    meta_ent = extract_window(entropy, content_start, content_end_exclusive)
    after_ent = extract_window(entropy, content_end_exclusive, content_end_exclusive + window)

    before_surp = extract_window(surprisal, content_start - window, content_start)
    meta_surp = extract_window(surprisal, content_start, content_end_exclusive)
    after_surp = extract_window(surprisal, content_end_exclusive, content_end_exclusive + window)

    # Need at least 1 token in each window to be meaningful
    if len(before_ent) == 0 or len(meta_ent) == 0 or len(after_ent) == 0:
        return None

    be = float(np.mean(before_ent))
    me = float(np.mean(meta_ent))
    ae = float(np.mean(after_ent))
    bs = float(np.mean(before_surp))
    ms = float(np.mean(meta_surp))
    as_ = float(np.mean(after_surp))

    return MetaBlockMeasurement(
        sample_id=sample_id,
        block_idx=block_idx,
        is_correct=is_correct,
        before_entropy=be,
        meta_entropy=me,
        after_entropy=ae,
        delta=ae - be,
        before_surprisal=bs,
        meta_surprisal=ms,
        after_surprisal=as_,
        delta_surprisal=as_ - bs,
        meta_start_tok=content_start,
        meta_end_tok=content_end,
        meta_length_tokens=content_end_exclusive - content_start,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def safe_mean(values: list[float]) -> float:
    """Mean with empty-list protection."""
    return float(np.mean(values)) if values else 0.0


def aggregate_statistics(
    block_measurements: list[MetaBlockMeasurement],
    sample_measurements: list[SampleMeasurement],
) -> dict:
    """Build aggregate statistics dict for JSON output."""
    correct_blocks = [m for m in block_measurements if m.is_correct]
    incorrect_blocks = [m for m in block_measurements if not m.is_correct]

    correct_samples = [s for s in sample_measurements if s.is_correct]
    incorrect_samples = [s for s in sample_measurements if not s.is_correct]

    def block_stats(blocks: list[MetaBlockMeasurement]) -> dict:
        if not blocks:
            return {"n": 0}
        return {
            "n": len(blocks),
            "before_entropy_mean": safe_mean([b.before_entropy for b in blocks]),
            "meta_entropy_mean": safe_mean([b.meta_entropy for b in blocks]),
            "after_entropy_mean": safe_mean([b.after_entropy for b in blocks]),
            "delta_entropy_mean": safe_mean([b.delta for b in blocks]),
            "delta_entropy_std": float(np.std([b.delta for b in blocks])) if blocks else 0.0,
            "before_surprisal_mean": safe_mean([b.before_surprisal for b in blocks]),
            "meta_surprisal_mean": safe_mean([b.meta_surprisal for b in blocks]),
            "after_surprisal_mean": safe_mean([b.after_surprisal for b in blocks]),
            "delta_surprisal_mean": safe_mean([b.delta_surprisal for b in blocks]),
            "delta_surprisal_std": float(np.std([b.delta_surprisal for b in blocks])) if blocks else 0.0,
            "meta_length_tokens_mean": safe_mean([b.meta_length_tokens for b in blocks]),
        }

    def sample_stats(samples: list[SampleMeasurement]) -> dict:
        if not samples:
            return {"n": 0}
        return {
            "n": len(samples),
            "avg_before_entropy": safe_mean([s.avg_before_entropy for s in samples]),
            "avg_meta_entropy": safe_mean([s.avg_meta_entropy for s in samples]),
            "avg_after_entropy": safe_mean([s.avg_after_entropy for s in samples]),
            "avg_delta": safe_mean([s.avg_delta for s in samples]),
            "avg_before_surprisal": safe_mean([s.avg_before_surprisal for s in samples]),
            "avg_meta_surprisal": safe_mean([s.avg_meta_surprisal for s in samples]),
            "avg_after_surprisal": safe_mean([s.avg_after_surprisal for s in samples]),
            "avg_delta_surprisal": safe_mean([s.avg_delta_surprisal for s in samples]),
            "full_sequence_entropy": safe_mean([s.full_sequence_entropy for s in samples]),
            "full_sequence_surprisal": safe_mean([s.full_sequence_surprisal for s in samples]),
        }

    return {
        "total_samples": len(sample_measurements),
        "total_blocks": len(block_measurements),
        "correct_samples": len(correct_samples),
        "incorrect_samples": len(incorrect_samples),
        "all_blocks": block_stats(block_measurements),
        "correct_blocks": block_stats(correct_blocks),
        "incorrect_blocks": block_stats(incorrect_blocks),
        "all_samples": sample_stats(sample_measurements),
        "correct_sample_stats": sample_stats(correct_samples),
        "incorrect_sample_stats": sample_stats(incorrect_samples),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_per_sample_csv(
    sample_measurements: list[SampleMeasurement],
    output_path: str,
) -> None:
    """Save per-sample summary to CSV."""
    fieldnames = [
        "sample_id", "is_correct", "num_meta_blocks",
        "before_entropy", "meta_entropy", "after_entropy", "delta",
        "before_surprisal", "meta_surprisal", "after_surprisal", "delta_surprisal",
        "full_sequence_entropy", "full_sequence_surprisal",
        "completion_length_tokens",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sample_measurements:
            writer.writerow({
                "sample_id": s.sample_id,
                "is_correct": s.is_correct,
                "num_meta_blocks": s.num_meta_blocks,
                "before_entropy": f"{s.avg_before_entropy:.6f}",
                "meta_entropy": f"{s.avg_meta_entropy:.6f}",
                "after_entropy": f"{s.avg_after_entropy:.6f}",
                "delta": f"{s.avg_delta:.6f}",
                "before_surprisal": f"{s.avg_before_surprisal:.6f}",
                "meta_surprisal": f"{s.avg_meta_surprisal:.6f}",
                "after_surprisal": f"{s.avg_after_surprisal:.6f}",
                "delta_surprisal": f"{s.avg_delta_surprisal:.6f}",
                "full_sequence_entropy": f"{s.full_sequence_entropy:.6f}",
                "full_sequence_surprisal": f"{s.full_sequence_surprisal:.6f}",
                "completion_length_tokens": s.completion_length_tokens,
            })
    print(f"Saved per-sample CSV: {output_path}")


def save_per_block_csv(
    block_measurements: list[MetaBlockMeasurement],
    output_path: str,
) -> None:
    """Save per-block detail to CSV."""
    fieldnames = [
        "sample_id", "block_idx", "is_correct",
        "before_entropy", "meta_entropy", "after_entropy", "delta",
        "before_surprisal", "meta_surprisal", "after_surprisal", "delta_surprisal",
        "meta_start_tok", "meta_end_tok", "meta_length_tokens",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in block_measurements:
            writer.writerow({
                "sample_id": b.sample_id,
                "block_idx": b.block_idx,
                "is_correct": b.is_correct,
                "before_entropy": f"{b.before_entropy:.6f}",
                "meta_entropy": f"{b.meta_entropy:.6f}",
                "after_entropy": f"{b.after_entropy:.6f}",
                "delta": f"{b.delta:.6f}",
                "before_surprisal": f"{b.before_surprisal:.6f}",
                "meta_surprisal": f"{b.meta_surprisal:.6f}",
                "after_surprisal": f"{b.after_surprisal:.6f}",
                "delta_surprisal": f"{b.delta_surprisal:.6f}",
                "meta_start_tok": b.meta_start_tok,
                "meta_end_tok": b.meta_end_tok,
                "meta_length_tokens": b.meta_length_tokens,
            })
    print(f"Saved per-block CSV: {output_path}")


def print_summary_table(stats: dict) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 80)
    print("  META-COT ENTROPY ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"  Total samples analyzed: {stats['total_samples']}")
    print(f"  Total meta blocks:      {stats['total_blocks']}")
    print(f"  Correct samples:        {stats['correct_samples']}")
    print(f"  Incorrect samples:      {stats['incorrect_samples']}")

    print("\n  --- Per-Block Entropy (nats) ---")
    header = f"  {'Group':<20} {'N':>5} {'Before':>10} {'Meta':>10} {'After':>10} {'Delta':>10} {'Delta_std':>10}"
    print(header)
    print("  " + "-" * 75)
    for label, key in [("All", "all_blocks"), ("Correct", "correct_blocks"), ("Incorrect", "incorrect_blocks")]:
        s = stats[key]
        if s["n"] == 0:
            print(f"  {label:<20} {0:>5}     --         --         --         --         --")
            continue
        print(
            f"  {label:<20} {s['n']:>5} "
            f"{s['before_entropy_mean']:>10.4f} "
            f"{s['meta_entropy_mean']:>10.4f} "
            f"{s['after_entropy_mean']:>10.4f} "
            f"{s['delta_entropy_mean']:>+10.4f} "
            f"{s['delta_entropy_std']:>10.4f}"
        )

    print("\n  --- Per-Block Surprisal (nats) ---")
    print(header.replace("Entropy", "Surprisal"))
    print("  " + "-" * 75)
    for label, key in [("All", "all_blocks"), ("Correct", "correct_blocks"), ("Incorrect", "incorrect_blocks")]:
        s = stats[key]
        if s["n"] == 0:
            print(f"  {label:<20} {0:>5}     --         --         --         --         --")
            continue
        print(
            f"  {label:<20} {s['n']:>5} "
            f"{s['before_surprisal_mean']:>10.4f} "
            f"{s['meta_surprisal_mean']:>10.4f} "
            f"{s['after_surprisal_mean']:>10.4f} "
            f"{s['delta_surprisal_mean']:>+10.4f} "
            f"{s['delta_surprisal_std']:>10.4f}"
        )

    print("\n  --- Interpretation ---")
    all_b = stats["all_blocks"]
    cor_b = stats["correct_blocks"]
    inc_b = stats["incorrect_blocks"]

    if all_b["n"] > 0:
        d = all_b["delta_entropy_mean"]
        direction = "DECREASES" if d < 0 else "INCREASES"
        print(f"  Overall: entropy {direction} by {abs(d):.4f} nats after meta blocks")

    if cor_b["n"] > 0 and inc_b["n"] > 0:
        cd = cor_b["delta_entropy_mean"]
        id_ = inc_b["delta_entropy_mean"]
        print(f"  Correct:   delta = {cd:+.4f} nats  ({'resolved' if cd < 0 else 'unresolved'})")
        print(f"  Incorrect: delta = {id_:+.4f} nats  ({'resolved' if id_ < 0 else 'unresolved'})")
        if cd < id_:
            print("  --> Correct samples show MORE entropy reduction (meta helps)")
        else:
            print("  --> Incorrect samples show MORE entropy reduction (unexpected)")

    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load eval data ----
    print(f"Loading eval parquet: {args.eval_parquet}")
    df = pd.read_parquet(args.eval_parquet)
    required_cols = {"completion", "is_correct"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: parquet missing columns: {missing}")
        sys.exit(1)

    # Filter to samples that contain the marker of interest
    if args.marker_mode == "meta":
        marker_label = "<|meta|>"
        mask = df["completion"].str.contains(r"<\|meta\|>", regex=True, na=False)
    else:  # confidence
        marker_label = "confidence: 0.XX"
        mask = df["completion"].str.contains(
            r"confidence\s*:\s*\d+(?:\.\d+)?", regex=True, na=False, case=False,
        )
    df_meta = df[mask].reset_index(drop=True)
    print(f"Samples with {marker_label}: {len(df_meta)} / {len(df)}")

    if len(df_meta) == 0:
        print(f"ERROR: No samples contain {marker_label}. Nothing to analyze.")
        sys.exit(1)

    if args.max_samples > 0 and len(df_meta) > args.max_samples:
        df_meta = df_meta.iloc[:args.max_samples]
        print(f"Capped to {args.max_samples} samples")

    # ---- Load model ----
    tokenizer, model = load_model_and_tokenizer(args.model_path, args.dtype)

    # Sanity-print marker tokenization
    if args.marker_mode == "meta":
        meta_open_ids = tokenizer.encode("<|meta|>", add_special_tokens=False)
        meta_close_ids = tokenizer.encode("<|/meta|>", add_special_tokens=False)
        print(f"<|meta|>  encodes to: {meta_open_ids} -> {[tokenizer.decode([t]) for t in meta_open_ids]}")
        print(f"<|/meta|> encodes to: {meta_close_ids} -> {[tokenizer.decode([t]) for t in meta_close_ids]}")
    else:
        probe = tokenizer.encode("confidence: 0.95", add_special_tokens=False)
        print(f"'confidence: 0.95' encodes to: {probe} ({len(probe)} tokens)")

    # ---- Process samples ----
    block_measurements: list[MetaBlockMeasurement] = []
    sample_measurements: list[SampleMeasurement] = []
    skipped = 0
    no_spans = 0
    t0 = time.time()

    for idx in tqdm(range(len(df_meta)), desc="Forward pass"):
        row = df_meta.iloc[idx]
        question = row.get("full_question", row.get("question", ""))
        completion = row["completion"]
        is_correct = bool(row["is_correct"])

        # Build prompt the same way eval did
        prompt_text = build_prompt_text(tokenizer, question)

        # Forward pass
        result = compute_entropy_and_surprisal(
            tokenizer, model, prompt_text, completion,
            max_seq_len=args.max_seq_len,
        )
        if result is None:
            skipped += 1
            continue

        token_ids = result["token_ids"]
        entropy = result["entropy"]
        surprisal = result["surprisal"]
        prompt_token_len = result["prompt_token_len"]

        # Find marker spans (meta tags or confidence text)
        if args.marker_mode == "meta":
            spans = find_meta_token_spans(tokenizer, token_ids, prompt_token_len)
        else:
            spans = find_confidence_token_spans(tokenizer, token_ids, prompt_token_len)
        if not spans:
            no_spans += 1
            continue

        # Measure each meta block
        sample_blocks: list[MetaBlockMeasurement] = []
        for block_idx, span in enumerate(spans):
            measurement = measure_meta_block(
                entropy, surprisal, span, args.window,
                sample_id=idx, block_idx=block_idx, is_correct=is_correct,
            )
            if measurement is not None:
                sample_blocks.append(measurement)
                block_measurements.append(measurement)

        if not sample_blocks:
            continue

        # Compute completion-only entropy (skip prompt tokens)
        answer_start = max(prompt_token_len - 1, 0)
        completion_entropy = entropy[answer_start:]
        completion_surprisal = surprisal[answer_start:]

        sample_measurements.append(SampleMeasurement(
            sample_id=idx,
            is_correct=is_correct,
            num_meta_blocks=len(sample_blocks),
            avg_before_entropy=safe_mean([b.before_entropy for b in sample_blocks]),
            avg_meta_entropy=safe_mean([b.meta_entropy for b in sample_blocks]),
            avg_after_entropy=safe_mean([b.after_entropy for b in sample_blocks]),
            avg_delta=safe_mean([b.delta for b in sample_blocks]),
            avg_before_surprisal=safe_mean([b.before_surprisal for b in sample_blocks]),
            avg_meta_surprisal=safe_mean([b.meta_surprisal for b in sample_blocks]),
            avg_after_surprisal=safe_mean([b.after_surprisal for b in sample_blocks]),
            avg_delta_surprisal=safe_mean([b.delta_surprisal for b in sample_blocks]),
            full_sequence_entropy=float(np.mean(completion_entropy)) if len(completion_entropy) > 0 else 0.0,
            full_sequence_surprisal=float(np.mean(completion_surprisal)) if len(completion_surprisal) > 0 else 0.0,
            completion_length_tokens=len(completion_entropy),
        ))

        # Periodic progress
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{idx+1}/{len(df_meta)}] "
                f"blocks={len(block_measurements)}, "
                f"samples={len(sample_measurements)}, "
                f"skipped={skipped}, "
                f"no_spans={no_spans}, "
                f"elapsed={elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    print(f"\nProcessing complete: {elapsed:.1f}s")
    print(f"  Measured samples: {len(sample_measurements)}")
    print(f"  Measured blocks:  {len(block_measurements)}")
    print(f"  Skipped (too long): {skipped}")
    print(f"  No spans found:    {no_spans}")

    if not block_measurements:
        print("ERROR: No valid measurements. Cannot produce output.")
        sys.exit(1)

    # ---- Aggregate and output ----
    stats = aggregate_statistics(block_measurements, sample_measurements)
    stats["config"] = {
        "model_path": args.model_path,
        "eval_parquet": args.eval_parquet,
        "max_samples": args.max_samples,
        "window": args.window,
        "max_seq_len": args.max_seq_len,
        "dtype": args.dtype,
        "marker_mode": args.marker_mode,
        "elapsed_seconds": round(elapsed, 1),
    }

    # Save JSON
    json_path = output_dir / "entropy_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved aggregate JSON: {json_path}")

    # Save CSVs
    save_per_sample_csv(
        sample_measurements,
        str(output_dir / "entropy_per_sample.csv"),
    )
    save_per_block_csv(
        block_measurements,
        str(output_dir / "entropy_per_block.csv"),
    )

    # Print summary
    print_summary_table(stats)


if __name__ == "__main__":
    main()
