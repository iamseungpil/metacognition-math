"""vLLM-based rollout generation for math problems."""
import json
import time
from pathlib import Path

import pandas as pd
import torch
import wandb
import yaml

from src.data.dataset_loader import extract_boxed_answer, extract_numeric_answer


MATH_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def build_chat_messages(question: str, system_prompt: str = MATH_SYSTEM_PROMPT):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def check_correctness(model_answer: str, gold_answer: str, source: str = "") -> bool:
    """Check if model answer matches gold answer."""
    model_boxed = extract_boxed_answer(model_answer)
    if model_boxed is None:
        return False

    gold_boxed = extract_boxed_answer(gold_answer)
    if gold_boxed is not None:
        gold_compare = gold_boxed.strip()
    else:
        gold_compare = gold_answer.strip()

    model_compare = model_boxed.strip()

    # Direct string match
    if model_compare == gold_compare:
        return True

    # Numeric comparison
    try:
        if abs(float(model_compare) - float(gold_compare)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False


def generate_rollouts(config_path: str):
    """Generate rollouts for all training problems using vLLM."""
    from vllm import LLM, SamplingParams

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_id = config["model"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=config.get("wandb_project", "metacot-math"),
        name="phase0-rollout",
        config=config,
        reinit=True,
    )

    rollouts_per_problem = config.get("rollouts_per_problem", 8)
    temperature = config.get("sampling", {}).get("temperature", 0.7)
    top_p = config.get("sampling", {}).get("top_p", 0.95)
    max_tokens = config.get("sampling", {}).get("max_tokens", 2048)
    tp_size = config.get("vllm", {}).get("tensor_parallel_size", 4)
    gpu_util = config.get("vllm", {}).get("gpu_memory_utilization", 0.90)
    max_model_len = config.get("vllm", {}).get("max_model_len", 4096)
    batch_size = config.get("batch_size", 256)

    # Load datasets
    from src.data.dataset_loader import load_all_train
    dataset = load_all_train(config.get("data", {}))

    print(f"Loaded {len(dataset)} problems")
    print(f"Generating {rollouts_per_problem} rollouts per problem")
    print(f"Total rollouts: {len(dataset) * rollouts_per_problem}")

    # Init vLLM
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_util,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=rollouts_per_problem,
    )

    # Process in batches
    all_results = []
    total = len(dataset)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = dataset.select(range(batch_start, batch_end))

        # Build prompts
        prompts = []
        for row in batch:
            messages = build_chat_messages(row["question"])
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        # Generate
        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        elapsed = time.time() - t0

        # Parse results
        for i, output in enumerate(outputs):
            row = batch[i]
            for j, completion in enumerate(output.outputs):
                response_text = completion.text
                is_correct = check_correctness(
                    response_text, row["answer"], row["source"]
                )
                all_results.append({
                    "problem_id": f"{row['source']}_{batch_start + i}",
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "category": row["category"],
                    "difficulty": row["difficulty"],
                    "source": row["source"],
                    "rollout_idx": j,
                    "completion": response_text,
                    "final_answer": extract_boxed_answer(response_text) or "",
                    "is_correct": is_correct,
                    "num_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                })

        # Log to wandb
        batch_correct = sum(r["is_correct"] for r in all_results[-len(outputs) * rollouts_per_problem:])
        batch_total = len(outputs) * rollouts_per_problem
        wandb.log({
            "phase0/batch": batch_count,
            "phase0/rollouts_total": len(all_results),
            "phase0/batch_accuracy": batch_correct / max(batch_total, 1),
            "phase0/cumulative_accuracy": sum(r["is_correct"] for r in all_results) / max(len(all_results), 1),
            "phase0/throughput_per_sec": batch_total / max(elapsed, 1),
        })
        print(
            f"Batch {batch_start}-{batch_end}/{total} done in {elapsed:.1f}s "
            f"({len(all_results)} rollouts so far)"
        )

        # Periodic save every 4 batches
        batch_count = (batch_start // batch_size) + 1
        if batch_count % 4 == 0:
            _save_checkpoint(all_results, output_dir, f"partial_{batch_count}")

    # Final save
    _save_checkpoint(all_results, output_dir, "final")
    wandb.finish()
    print(f"Done! {len(all_results)} rollouts saved to {output_dir}")
    return all_results


def _save_checkpoint(results: list, output_dir: Path, tag: str):
    df = pd.DataFrame(results)
    path = output_dir / f"rollouts_{tag}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rollouts to {path}")

    # Print summary stats
    correct = df["is_correct"].sum()
    total = len(df)
    if total > 0:
        print(f"  Accuracy: {correct}/{total} = {correct/total:.3f}")
        for src, grp in df.groupby("source")["is_correct"]:
            print(f"  {src}: {grp.sum()}/{len(grp)} = {grp.mean():.3f}")


def build_profile(rollouts_path: str, output_path: str):
    """Build capability profile from rollout results."""
    df = pd.read_parquet(rollouts_path)

    # Group by problem, compute majority vote (pass@K)
    problem_groups = df.groupby("problem_id").agg(
        category=("category", "first"),
        difficulty=("difficulty", "first"),
        source=("source", "first"),
        num_rollouts=("is_correct", "count"),
        num_correct=("is_correct", "sum"),
    ).reset_index()

    problem_groups["pass_rate"] = problem_groups["num_correct"] / problem_groups["num_rollouts"]
    problem_groups["majority_correct"] = problem_groups["pass_rate"] >= 0.5

    # Build profile
    profile = {
        "total_problems": len(problem_groups),
        "total_rollouts": len(df),
        "overall_pass_at_1": df["is_correct"].mean(),
        "overall_pass_at_majority": problem_groups["majority_correct"].mean(),
        "category_accuracy": {},
        "difficulty_accuracy": {},
        "weak_categories": [],
    }

    # By category × difficulty
    for cat in problem_groups["category"].unique():
        cat_data = problem_groups[problem_groups["category"] == cat]
        cat_profile = {}
        for diff in ["easy", "medium", "hard"]:
            diff_data = cat_data[cat_data["difficulty"] == diff]
            if len(diff_data) > 0:
                cat_profile[diff] = float(diff_data["majority_correct"].mean())
        profile["category_accuracy"][cat] = cat_profile

    # Identify weak categories (overall < 0.5)
    for cat, acc_dict in profile["category_accuracy"].items():
        avg_acc = sum(acc_dict.values()) / max(len(acc_dict), 1)
        if avg_acc < 0.5:
            profile["weak_categories"].append(cat)

    # By difficulty only
    for diff in ["easy", "medium", "hard"]:
        diff_data = problem_groups[problem_groups["difficulty"] == diff]
        if len(diff_data) > 0:
            profile["difficulty_accuracy"][diff] = float(diff_data["majority_correct"].mean())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"Profile saved to {output_path}")
    print(json.dumps(profile, indent=2))
    return profile


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--rollouts-path", default=None)
    parser.add_argument("--profile-output", default=None)
    args = parser.parse_args()

    if args.profile_only:
        build_profile(args.rollouts_path, args.profile_output)
    else:
        generate_rollouts(args.config)
