#!/usr/bin/env python3
"""Generate improved Meta-CoT v2 SFT data using TRAPI GPT-5.4.

Key improvements over v1:
- Confidence calibrated to rollout_pass_rate (not always 0.95+)
- Error→correction patterns in 30%+ of chains
- Mandatory final verification meta
- Short meta blocks (<50 tokens each)
- Diverse difficulty-aware prompting

Usage:
    python gen_metacot_v2.py --max-chains 1000 --workers 12
    python gen_metacot_v2.py --max-chains 5000 --workers 15
"""
import argparse
import ast
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, "/scratch/metacognition")

from src.metacot.prompt_v2 import (
    META_COT_V2_SYSTEM_PROMPT,
    META_START,
    META_END,
    build_metacot_v2_prompt,
)
from src.metacot.generator import get_trapi_client


def parse_meta_blocks_v2(text: str) -> dict:
    """Parse <|meta|> blocks from v2 model output.

    V2 changes from v1:
    - Checks for error→fix pattern
    - Checks for final verification
    - More lenient confidence extraction
    """
    blocks = re.findall(
        rf'{re.escape(META_START)}(.*?){re.escape(META_END)}',
        text, re.DOTALL,
    )

    result = {
        "num_blocks": len(blocks),
        "confidences": [],
        "has_pre_assessment": False,
        "has_error_fix": False,
        "has_final_verification": False,
        "has_boxed": "\\boxed" in text,
        "valid": False,
    }

    for i, block in enumerate(blocks):
        block_lower = block.lower()

        # Extract confidence values
        conf_matches = re.findall(
            r'(?:probability|confidence|prob\.?)[:\s]*(\d+\.\d+)',
            block, re.IGNORECASE,
        )
        for m in conf_matches:
            val = float(m)
            if val > 1.0:
                val /= 100.0
            val = min(1.0, max(0.0, val))
            if val > 0.001:
                result["confidences"].append(val)

        # Classify block
        if i == 0 and any(kw in block_lower for kw in [
            "can i solve", "probability", "risk", "approach",
        ]):
            result["has_pre_assessment"] = True

        if any(kw in block_lower for kw in [
            "wait", "wrong", "error", "mistake", "fix", "correction",
            "actually", "let me fix", "that's wrong",
        ]):
            result["has_error_fix"] = True

        if any(kw in block_lower for kw in [
            "final check", "final verification", "verified",
            "final", "answer",
        ]) and i == len(blocks) - 1:
            result["has_final_verification"] = True

    result["valid"] = (
        result["num_blocks"] >= 2
        and result["has_boxed"]
        and len(result["confidences"]) >= 1
    )

    return result


def generate_single_v2(
    client,
    question: str,
    ground_truth: str,
    rollout_pass_rate: float,
    model_name: str = "gpt-5.4_2026-03-05",
    max_retries: int = 15,
) -> dict:
    """Generate a single Meta-CoT v2 chain via TRAPI."""
    user_prompt = build_metacot_v2_prompt(
        question=question,
        rollout_pass_rate=rollout_pass_rate,
    )

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=model_name,
                instructions=META_COT_V2_SYSTEM_PROMPT,
                input=user_prompt,
            )
            chain_text = response.output_text
            parsed = parse_meta_blocks_v2(chain_text)

            usage_data = None
            if hasattr(response, 'usage') and response.usage:
                usage_data = {
                    "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                    "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                }

            if parsed["valid"]:
                return {
                    "chain": chain_text,
                    "parsed": parsed,
                    "usage": usage_data,
                    "attempts": attempt + 1,
                }
            elif attempt < max_retries - 1:
                time.sleep(1)
                continue
            else:
                return {
                    "chain": chain_text,
                    "parsed": parsed,
                    "usage": usage_data,
                    "attempts": attempt + 1,
                }

        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1:
                # Exponential backoff with jitter
                base_wait = min(5 * (2 ** attempt), 120)
                jitter = random.uniform(0, base_wait * 0.5)
                wait = base_wait + jitter
                print(
                    f"  [retry {attempt+1}/{max_retries}] {error_str[:100]}... "
                    f"waiting {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
            else:
                return {
                    "chain": "",
                    "parsed": {"valid": False, "num_blocks": 0, "confidences": []},
                    "error": error_str,
                    "usage": None,
                    "attempts": attempt + 1,
                }


def load_problems_and_pass_rates():
    """Load math problems from verl_train.parquet and compute pass rates from rollouts."""
    # Load training problems
    verl = pd.read_parquet("/scratch/metacognition/verl_train.parquet")
    print(f"Loaded verl_train: {len(verl)} problems")

    # Extract questions and ground truths
    problems = []
    for idx, row in verl.iterrows():
        # Parse prompt (numpy array or list of dicts)
        prompt = row["prompt"]
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        elif isinstance(prompt, str):
            prompt = ast.literal_eval(prompt)

        question = None
        for msg in prompt:
            if msg["role"] == "user":
                question = msg["content"]
                break

        if question is None:
            continue

        ground_truth = ""
        rm = row.get("reward_model", {})
        if isinstance(rm, dict):
            ground_truth = rm.get("ground_truth", "")

        problems.append({
            "idx": idx,
            "question": question,
            "ground_truth": ground_truth,
            "data_source": row.get("data_source", ""),
        })

    problems_df = pd.DataFrame(problems)
    print(f"Extracted {len(problems_df)} questions")

    # Load rollouts for pass rate computation
    rollouts_path = "/scratch/metacognition/rollouts/rollouts_final.parquet"
    if os.path.exists(rollouts_path):
        rollouts = pd.read_parquet(rollouts_path)
        # Compute per-problem pass rates
        pass_rates = rollouts.groupby("problem_id")["is_correct"].mean().to_dict()
        print(f"Computed pass rates for {len(pass_rates)} problems from rollouts")

        # Try to match by question text
        # Build question -> pass_rate mapping from rollouts
        q_to_pr = {}
        for pid, rate in pass_rates.items():
            q_rows = rollouts[rollouts["problem_id"] == pid]["question"]
            if len(q_rows) > 0:
                q_to_pr[q_rows.iloc[0]] = rate

        # Assign pass rates
        problems_df["rollout_pass_rate"] = problems_df["question"].map(q_to_pr)
        matched = problems_df["rollout_pass_rate"].notna().sum()
        print(f"Matched pass rates: {matched}/{len(problems_df)}")

        # Fill missing with 0.5 (default medium difficulty)
        problems_df["rollout_pass_rate"] = problems_df["rollout_pass_rate"].fillna(0.5)
    else:
        print("No rollouts found, using default pass rate 0.5")
        problems_df["rollout_pass_rate"] = 0.5

    return problems_df


def run_generation(
    max_chains: int = 1000,
    workers: int = 12,
    output_dir: str = "/scratch/metacognition/sft_data",
    checkpoint_interval: int = 200,
):
    """Main generation loop."""
    print("=" * 60)
    print(f"Meta-CoT v2 SFT Data Generation")
    print(f"  Target: {max_chains} chains")
    print(f"  Workers: {workers}")
    print(f"  Output: {output_dir}")
    print("=" * 60)

    # Load data
    problems_df = load_problems_and_pass_rates()

    # Sample problems
    if len(problems_df) > max_chains:
        # Stratified sampling by difficulty
        easy = problems_df[problems_df["rollout_pass_rate"] > 0.8]
        medium = problems_df[
            (problems_df["rollout_pass_rate"] > 0.4)
            & (problems_df["rollout_pass_rate"] <= 0.8)
        ]
        hard = problems_df[problems_df["rollout_pass_rate"] <= 0.4]

        n_easy = min(len(easy), int(max_chains * 0.3))
        n_medium = min(len(medium), int(max_chains * 0.35))
        n_hard = min(len(hard), max_chains - n_easy - n_medium)

        # Adjust if not enough in any category
        total_needed = max_chains
        selected_parts = []
        if len(easy) > 0:
            selected_parts.append(easy.sample(n=min(n_easy, len(easy)), random_state=42))
        if len(medium) > 0:
            selected_parts.append(medium.sample(n=min(n_medium, len(medium)), random_state=42))
        if len(hard) > 0:
            selected_parts.append(hard.sample(n=min(n_hard, len(hard)), random_state=42))

        selected = pd.concat(selected_parts).sample(frac=1, random_state=42)

        # If still not enough, sample more from the full set
        if len(selected) < max_chains:
            remaining = problems_df[~problems_df.index.isin(selected.index)]
            extra = remaining.sample(
                n=min(max_chains - len(selected), len(remaining)),
                random_state=42,
            )
            selected = pd.concat([selected, extra])
    else:
        selected = problems_df.sample(frac=1, random_state=42)

    print(f"\nSelected {len(selected)} problems for generation")
    print(f"  Easy (>0.8): {(selected['rollout_pass_rate'] > 0.8).sum()}")
    print(f"  Medium (0.4-0.8): {((selected['rollout_pass_rate'] > 0.4) & (selected['rollout_pass_rate'] <= 0.8)).sum()}")
    print(f"  Hard (<=0.4): {(selected['rollout_pass_rate'] <= 0.4).sum()}")

    # Initialize TRAPI client
    print("\nInitializing TRAPI client...")
    client = get_trapi_client()
    print("TRAPI client ready")

    # Generate chains
    os.makedirs(output_dir, exist_ok=True)
    results = []
    total_tokens = 0
    start_time = time.time()

    def _process(row_tuple):
        _, row = row_tuple
        result = generate_single_v2(
            client=client,
            question=row["question"],
            ground_truth=row["ground_truth"],
            rollout_pass_rate=row["rollout_pass_rate"],
        )
        return row, result

    print(f"\nStarting generation with {workers} concurrent workers...\n", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process, row_tuple): i
            for i, row_tuple in enumerate(selected.iterrows())
        }

        for future in as_completed(futures):
            try:
                row, result = future.result()
            except Exception as exc:
                print(f"  Worker exception: {exc}", flush=True)
                continue

            chain_text = result.get("chain", "")
            parsed = result.get("parsed", {})

            results.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "rollout_pass_rate": row["rollout_pass_rate"],
                "metacot_chain": chain_text,
                "num_meta_blocks": parsed.get("num_blocks", 0),
                "confidences": json.dumps(parsed.get("confidences", [])),
                "has_error_fix": parsed.get("has_error_fix", False),
                "has_final_verification": parsed.get("has_final_verification", False),
                "chain_valid": parsed.get("valid", False),
                "attempts": result.get("attempts", 0),
                "error": result.get("error", ""),
            })

            if result.get("usage"):
                u = result["usage"]
                total_tokens += (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0))

            done = len(results)
            if done % 20 == 0:
                elapsed = time.time() - start_time
                valid = sum(1 for r in results if r["chain_valid"])
                rate = done / elapsed * 3600 if elapsed > 0 else 0
                print(
                    f"  [{done}/{len(selected)}] "
                    f"valid={valid}/{done} ({valid/done:.1%}) "
                    f"tokens={total_tokens:,} "
                    f"rate={rate:.0f}/hr "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

            if done % checkpoint_interval == 0:
                _save_checkpoint(results, output_dir, done)

    # Final save
    elapsed = time.time() - start_time
    _save_checkpoint(results, output_dir, "final")

    # Convert to SFT format
    sft_path = _build_sft_format(results, output_dir)

    # Print summary
    valid = sum(1 for r in results if r["chain_valid"])
    error_fix = sum(1 for r in results if r.get("has_error_fix"))
    final_verif = sum(1 for r in results if r.get("has_final_verification"))

    all_confs = []
    for r in results:
        if r["chain_valid"]:
            confs = json.loads(r["confidences"]) if isinstance(r["confidences"], str) else r["confidences"]
            all_confs.extend(confs)

    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total generated: {len(results)}")
    print(f"Valid chains: {valid} ({valid/len(results):.1%})")
    print(f"Error→fix pattern: {error_fix} ({error_fix/len(results):.1%})")
    print(f"Final verification: {final_verif} ({final_verif/len(results):.1%})")
    print(f"Total tokens used: {total_tokens:,}")
    print(f"Time elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    if all_confs:
        confs_arr = np.array(all_confs)
        print(f"\nConfidence distribution (n={len(confs_arr)}):")
        print(f"  Mean: {confs_arr.mean():.3f}")
        print(f"  Std: {confs_arr.std():.3f}")
        print(f"  [0.0, 0.2): {(confs_arr < 0.2).sum()} ({(confs_arr < 0.2).mean():.1%})")
        print(f"  [0.2, 0.4): {((confs_arr >= 0.2) & (confs_arr < 0.4)).sum()} ({((confs_arr >= 0.2) & (confs_arr < 0.4)).mean():.1%})")
        print(f"  [0.4, 0.6): {((confs_arr >= 0.4) & (confs_arr < 0.6)).sum()} ({((confs_arr >= 0.4) & (confs_arr < 0.6)).mean():.1%})")
        print(f"  [0.6, 0.8): {((confs_arr >= 0.6) & (confs_arr < 0.8)).sum()} ({((confs_arr >= 0.6) & (confs_arr < 0.8)).mean():.1%})")
        print(f"  [0.8, 0.9): {((confs_arr >= 0.8) & (confs_arr < 0.9)).sum()} ({((confs_arr >= 0.8) & (confs_arr < 0.9)).mean():.1%})")
        print(f"  [0.9, 0.95): {((confs_arr >= 0.9) & (confs_arr < 0.95)).sum()} ({((confs_arr >= 0.9) & (confs_arr < 0.95)).mean():.1%})")
        print(f"  [0.95, 1.0]: {(confs_arr >= 0.95).sum()} ({(confs_arr >= 0.95).mean():.1%})")

    print(f"\nSFT data saved to: {sft_path}")
    return sft_path


def _save_checkpoint(results, output_dir, tag):
    """Save intermediate checkpoint."""
    df = pd.DataFrame(results)
    path = os.path.join(output_dir, f"metacot_v2_raw_{tag}.parquet")
    df.to_parquet(path, index=False)
    valid = sum(1 for r in results if r["chain_valid"])
    print(f"  >> Checkpoint saved: {path} ({valid}/{len(results)} valid)", flush=True)


def _build_sft_format(results, output_dir):
    """Convert raw chains to SFT training format (messages)."""
    sft_rows = []
    for r in results:
        if not r["chain_valid"]:
            continue

        messages = [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["metacot_chain"]},
        ]

        # Create a problem_id from question hash
        problem_id = f"v2_{hash(r['question']) % (10**8):08d}"

        sft_rows.append({
            "messages": json.dumps(messages),
            "problem_id": problem_id,
            "source": "metacot_v2_improved",
            "rollout_pass_rate": r["rollout_pass_rate"],
            "has_error_fix": r.get("has_error_fix", False),
            "has_final_verification": r.get("has_final_verification", False),
            "num_meta_blocks": r.get("num_meta_blocks", 0),
        })

    sft_df = pd.DataFrame(sft_rows)
    sft_path = os.path.join(output_dir, "metacot_v2_sft.parquet")
    sft_df.to_parquet(sft_path, index=False)
    print(f"\nSFT dataset: {len(sft_df)} valid chains saved to {sft_path}")
    return sft_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Meta-CoT v2 SFT data")
    parser.add_argument("--max-chains", type=int, default=1000,
                        help="Number of chains to generate (default: 1000 for testing)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Number of concurrent TRAPI requests (default: 12)")
    parser.add_argument("--output-dir", default="/scratch/metacognition/sft_data",
                        help="Output directory")
    parser.add_argument("--checkpoint-interval", type=int, default=200,
                        help="Save checkpoint every N chains")
    args = parser.parse_args()

    run_generation(
        max_chains=args.max_chains,
        workers=args.workers,
        output_dir=args.output_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
