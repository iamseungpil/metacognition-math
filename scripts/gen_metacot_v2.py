"""Generate Meta-CoT v2 data via TRAPI (GPT-5.4).

Generates diverse confidence, error→fix patterns, final verification.
Runs concurrently with retry+jitter.

Usage: python scripts/gen_metacot_v2.py --n 5000 --concurrent 10
"""
import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metacot.prompt_v2 import META_COT_V2_SYSTEM_PROMPT, build_metacot_v2_prompt, META_START, META_END


def get_trapi_client():
    """Create TRAPI client (matching skilldiscovery/gpt_agent.py pattern)."""
    from openai import AzureOpenAI
    endpoint = "https://trapi.research.microsoft.com/gcr/shared"
    api_version = "2025-04-01-preview"
    trapi_scope = "api://trapi/.default"

    # Try env var token first
    token = os.environ.get("TRAPI_TOKEN")
    if token:
        return AzureOpenAI(azure_endpoint=endpoint, api_key=token, api_version=api_version)

    # Azure CLI credential with TRAPI scope (works from host VM)
    from azure.identity import AzureCliCredential, get_bearer_token_provider
    provider = get_bearer_token_provider(AzureCliCredential(), trapi_scope)
    token = provider()
    return AzureOpenAI(azure_endpoint=endpoint, api_key=token, api_version=api_version)


def generate_one(client, question, pass_rate, model="gpt-5.4-mini_2026-03-17", max_retries=10):
    """Generate one meta-CoT chain with retry."""
    system = META_COT_V2_SYSTEM_PROMPT
    user = build_metacot_v2_prompt(question, pass_rate)

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
            )
            text = resp.output_text
            if not text:
                continue

            # Validate: has meta blocks and boxed answer
            has_meta = META_START in text
            has_boxed = "\\boxed" in text or "boxed{" in text
            if not has_meta or not has_boxed:
                continue

            return text

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(120, 5 * (2 ** attempt) + random.uniform(0, 3))
                time.sleep(wait)
            elif "500" in err or "502" in err or "503" in err:
                time.sleep(10 + random.uniform(0, 5))
            else:
                print(f"  Error: {err[:100]}")
                time.sleep(5)

    return None


def validate_chain(text):
    """Validate a meta-CoT chain. Returns dict with quality metrics."""
    # Count meta blocks
    meta_count = text.count(META_START)

    # Extract confidences
    confs = re.findall(
        r'(?:probability|confidence)[:\s\w]*?(\d+\.\d+|\d+)\s*%?',
        text, re.IGNORECASE
    )
    confidences = []
    for m in confs:
        v = float(m)
        if v > 1:
            v /= 100
        confidences.append(max(0.0, min(1.0, v)))

    # Check for error-correction
    has_error_fix = bool(re.search(
        r'\b(wait|wrong|fix|actually|mistake|let me re|hold on|incorrect)\b',
        text, re.IGNORECASE
    ))

    # Check for final verification
    has_final_check = bool(re.search(
        r'(final|verify|check|confirm).*(?:confidence|probability)',
        text[-500:], re.IGNORECASE
    ))

    return {
        "meta_count": meta_count,
        "confidences": confidences,
        "has_error_fix": has_error_fix,
        "has_final_check": has_final_check,
        "mean_conf": sum(confidences) / len(confidences) if confidences else 0,
        "valid": meta_count >= 2 and len(confidences) >= 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--data_path", default="/scratch/metacognition/verl_train.parquet")
    parser.add_argument("--output_path", default="/scratch/metacognition/sft_data/metacot_v2_sft.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} problems")

    # Sample N problems (prioritize diverse pass rates)
    indices = list(range(len(df)))
    random.shuffle(indices)
    indices = indices[:args.n]

    client = get_trapi_client()
    print(f"TRAPI client ready. Generating {len(indices)} chains with {args.concurrent} concurrent...")

    results = []
    failed = 0

    def process_one(idx):
        row = df.iloc[idx]
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        question = prompt[0]["content"] if isinstance(prompt, list) else str(prompt)

        # Get pass rate from reward_model if available
        rm = row.get("reward_model", {})
        if isinstance(rm, str):
            rm = json.loads(rm)
        pass_rate = rm.get("pass_rate", 0.5) if isinstance(rm, dict) else 0.5

        text = generate_one(client, question, pass_rate)
        if text is None:
            return None

        validation = validate_chain(text)
        if not validation["valid"]:
            return None

        # Build SFT messages format
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": text},
        ]

        return {
            "messages": json.dumps(messages),
            "problem_id": str(idx),
            "source": "metacot_v2",
            "meta_count": validation["meta_count"],
            "mean_conf": validation["mean_conf"],
            "has_error_fix": validation["has_error_fix"],
            "has_final_check": validation["has_final_check"],
        }

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {executor.submit(process_one, idx): idx for idx in indices}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                traceback.print_exc()

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(indices)}: {len(results)} valid, {failed} failed")

    # Save
    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_df.to_parquet(args.output_path)

    # Stats
    print(f"\n=== Generation Complete ===")
    print(f"Total: {len(results)} valid chains ({failed} failed)")
    print(f"Meta blocks avg: {out_df['meta_count'].mean():.1f}")
    print(f"Confidence avg: {out_df['mean_conf'].mean():.3f}")
    print(f"Error-fix rate: {out_df['has_error_fix'].mean():.1%}")
    print(f"Final check rate: {out_df['has_final_check'].mean():.1%}")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
