"""Meta-CoT chain generation using TRAPI (GPT-5.4)."""
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from src.metacot.prompt import (
    META_COT_SYSTEM_PROMPT,
    build_metacot_user_prompt,
    parse_metacot_stages,
)


def get_trapi_client():
    """Create TRAPI Azure OpenAI client with proper auth.

    Auth strategy (per TRAPI memory):
    1. SDK with azure_ad_token_provider (auto-refreshes, never cache tokens)
    2. ChainedTokenCredential: AzureCli → ManagedIdentity
    3. Fallback: TRAPI_TOKEN env var for environments without azure.identity
    """
    import os
    from openai import AzureOpenAI

    endpoint = "https://trapi.research.microsoft.com/gcr/shared"
    api_version = "2025-04-01-preview"

    # Fallback: if TRAPI_TOKEN env is set (e.g. Docker or pre-auth'd environment)
    trapi_token = os.environ.get("TRAPI_TOKEN")
    if trapi_token:
        print("Using TRAPI_TOKEN env var (pre-authenticated)")
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token=trapi_token,
            api_version=api_version,
        )
        return client

    # Primary: SDK with token provider (auto-refresh per request)
    from azure.identity import (
        ChainedTokenCredential,
        AzureCliCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )

    scope = "api://trapi/.default"
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        scope,
    )

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=credential,  # auto-refreshes, never cache!
        api_version=api_version,
    )
    return client


def generate_single_chain(
    client,
    profile: dict,
    question: str,
    model_answer: str,
    correct_answer: str,
    is_correct: bool,
    data_pool_summary: str = "",
    model_name: str = "gpt-5.4_2026-03-05",
    max_retries: int = 3,
) -> dict:
    """Generate a single Meta-CoT chain via TRAPI."""
    user_prompt = build_metacot_user_prompt(
        profile=profile,
        question=question,
        model_answer=model_answer,
        correct_answer=correct_answer,
        is_correct=is_correct,
        data_pool_summary=data_pool_summary,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": META_COT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=2048,
                temperature=0.7,
            )
            chain_text = response.choices[0].message.content
            parsed = parse_metacot_stages(chain_text)

            if parsed["valid"]:
                return {
                    "chain": chain_text,
                    "parsed": parsed,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                }
            elif attempt < max_retries - 1:
                continue
            else:
                return {"chain": chain_text, "parsed": parsed, "usage": None}

        except Exception as e:
            if attempt < max_retries - 1:
                # TRAPI memory: 429 needs 30s→60s→120s backoff
                wait = 30 * (2 ** attempt)
                print(f"Retry {attempt+1} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                return {"chain": "", "parsed": {"valid": False}, "error": str(e), "usage": None}


def generate_metacot_dataset(config_path: str):
    """Generate Meta-CoT chains for a batch of rollouts."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    rollouts_path = config["rollouts_path"]
    profile_path = config["profile_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    max_chains = config.get("max_chains", 10000)
    correct_ratio = config.get("correct_ratio", 0.5)
    model_name = config.get("trapi_model", "gpt-5.4_2026-03-05")

    # Load data
    df = pd.read_parquet(rollouts_path)
    with open(profile_path) as f:
        profile = json.load(f)

    # Balance correct/incorrect
    correct = df[df["is_correct"]].drop_duplicates("problem_id")
    incorrect = df[~df["is_correct"]].drop_duplicates("problem_id")

    n_correct = int(max_chains * correct_ratio)
    n_incorrect = max_chains - n_correct

    if len(correct) > n_correct:
        correct = correct.sample(n=n_correct, random_state=42)
    if len(incorrect) > n_incorrect:
        incorrect = incorrect.sample(n=n_incorrect, random_state=42)

    selected = pd.concat([correct, incorrect]).sample(frac=1, random_state=42)
    print(f"Generating {len(selected)} Meta-CoT chains ({len(correct)} correct, {len(incorrect)} incorrect)")

    client = get_trapi_client()
    concurrent = config.get("concurrent_requests", 20)
    results = []
    total_tokens = 0
    rows_list = list(selected.iterrows())

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _process_row(idx_row):
        i, row = idx_row
        result = generate_single_chain(
            client=client,
            profile=profile,
            question=row["question"],
            model_answer=row["completion"],
            correct_answer=row["gold_answer"],
            is_correct=row["is_correct"],
            model_name=model_name,
        )
        return i, row, result

    print(f"Using {concurrent} concurrent workers", flush=True)

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {executor.submit(_process_row, (i, row)): i for i, (_, row) in enumerate(rows_list)}
        done_count = 0

        for future in as_completed(futures):
            i, row, result = future.result()
            results.append({
                "problem_id": row["problem_id"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "model_answer": row["completion"],
                "is_correct": row["is_correct"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "metacot_chain": result["chain"],
                "confidence": result["parsed"].get("confidence"),
                "problem_count": result["parsed"].get("problem_count"),
                "has_l1l2l3": result["parsed"].get("has_l1l2l3", False),
                "chain_valid": result["parsed"].get("valid", False),
            })

            if result.get("usage"):
                total_tokens += result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]

            done_count += 1
            if done_count % 50 == 0:
                valid = sum(1 for r in results if r["chain_valid"])
                print(
                    f"  [{done_count}/{len(selected)}] valid={valid}/{len(results)} "
                    f"tokens_used={total_tokens:,}",
                    flush=True,
                )

            if done_count % 500 == 0:
                _save_results(results, output_dir, f"checkpoint_{done_count}")

    _save_results(results, output_dir, "final")
    valid_count = sum(1 for r in results if r["chain_valid"])
    print(f"Done! {valid_count}/{len(results)} valid chains. Total tokens: {total_tokens:,}")
    return results


def _save_results(results: list, output_dir: Path, tag: str):
    df = pd.DataFrame(results)
    df.to_parquet(output_dir / f"metacot_{tag}.parquet", index=False)


def build_sft_dataset(metacot_path: str, output_path: str):
    """Convert Meta-CoT chains to SFT training format.

    Each training example: user asks the problem, assistant outputs the
    full 5-stage Meta-CoT chain (without the correct answer revealed).
    """
    df = pd.read_parquet(metacot_path)
    # Use existing chain_valid from generation, no re-validation
    df = df[df["chain_valid"]]
    print(f"Valid chains: {len(df)}")

    sft_data = []
    for _, row in df.iterrows():
        # SFT format: solution with \boxed{answer} FIRST, then Meta-CoT analysis
        # This ensures model can still produce extractable answers for eval
        from src.data.dataset_loader import extract_boxed_answer
        gold_boxed = extract_boxed_answer(row["gold_answer"])
        if gold_boxed:
            answer_line = f"\\boxed{{{gold_boxed}}}"
        else:
            answer_line = f"The answer is {row['gold_answer']}"

        # Combine: model's solution attempt + answer + metacognitive analysis
        assistant_content = (
            f"{row['model_answer']}\n\n"
            f"Final Answer: {answer_line}\n\n"
            f"--- Metacognitive Analysis ---\n"
            f"{row['metacot_chain']}"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a math problem solver with metacognitive awareness. "
                    "For each problem: (1) solve it step by step and put your final "
                    "answer in \\boxed{{}}, then (2) analyze your solution quality, "
                    "diagnose errors, plan what to study next, select practice "
                    "problems, and predict your improvement."
                ),
            },
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": assistant_content},
        ]
        sft_data.append({
            "messages": json.dumps(messages),
            "problem_id": row["problem_id"],
            "is_correct": row["is_correct"],
            "category": row["category"],
        })

    out_df = pd.DataFrame(sft_data)
    out_df.to_parquet(output_path, index=False)
    print(f"SFT dataset: {len(out_df)} examples saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--build-sft", action="store_true")
    parser.add_argument("--metacot-path", default=None)
    parser.add_argument("--sft-output", default=None)
    args = parser.parse_args()

    if args.build_sft:
        if not args.metacot_path or not args.sft_output:
            parser.error("--build-sft requires --metacot-path and --sft-output")
        build_sft_dataset(args.metacot_path, args.sft_output)
    else:
        if not args.config:
            parser.error("--config is required for Meta-CoT generation")
        generate_metacot_dataset(args.config)
