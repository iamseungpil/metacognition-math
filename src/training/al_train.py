"""Agent Lightning training for metacognitive math reasoning.

Uses @agl.rollout decorator with verl GRPO backend.
Step-level credit assignment at <|meta|> boundaries.
3 rewards: R_correct(2.0), R_calib, R_penalty.

Usage (on compute, verl conda env):
    python -m src.training.al_train \
        --train-file rollouts/rollouts_final.parquet \
        --model checkpoints/meta_sft
"""
import argparse
import asyncio
import os
import sys
from typing import Any, Dict, TypedDict

sys.path.insert(0, os.getcwd())

import agentlightning as agl

from src.metacot.prompt import META_START, META_END, parse_meta_blocks
from src.training.rewards import compute_r_correct, compute_r_calib, compute_r_penalty
from src.rollout.vllm_rollout import check_correctness


class MathTask(TypedDict):
    question: str
    gold_answer: str
    problem_id: str
    category: str


SYSTEM_PROMPT = (
    "You are a math problem solver with metacognitive awareness. "
    f"Use {META_START} and {META_END} tags to reflect on your reasoning. "
    "Before solving, assess difficulty and your probability of solving. "
    "During solving, verify uncertain steps. "
    "After solving, reflect on what you learned. "
    "Put your final answer in \\boxed{}."
)


@agl.rollout
async def metacot_agent(task: MathTask, llm: agl.LLM) -> None:
    """Metacognitive math agent with <|meta|> self-reflection.

    The agent sends the problem to the LLM (served by vLLM via Agent Lightning),
    parses the response for <|meta|> blocks, and emits step-level rewards.
    """
    import httpx

    question = task["question"]
    gold_answer = task["gold_answer"]

    # Call LLM via the endpoint provided by Agent Lightning
    prompt = f"{SYSTEM_PROMPT}\n\nProblem: {question}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{llm.endpoint}/v1/chat/completions",
            json={
                "model": llm.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.95,
            },
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'token-abc123')}"},
        )
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]

    # Parse <|meta|> blocks
    parsed = parse_meta_blocks(response_text)
    num_meta = parsed["num_blocks"]
    confidences = parsed["confidences"]

    # Check correctness
    is_correct = check_correctness(response_text, gold_answer)

    # Compute rewards
    r_correct = compute_r_correct(is_correct)  # +2.0 if correct
    r_penalty = compute_r_penalty(num_meta)    # -0.5 if no meta

    # R_calib: reward for stating confidence (probe integration later)
    r_calib = 0.0
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        if is_correct and avg_conf > 0.5:
            r_calib = avg_conf
        elif not is_correct and avg_conf < 0.5:
            r_calib = 1.0 - avg_conf
        else:
            r_calib = 0.1

    total_reward = r_correct + r_calib + r_penalty

    # Emit multi-dimensional reward
    agl.emit_reward({
        "task_reward": total_reward,
        "r_correct": r_correct,
        "r_calib": r_calib,
        "r_penalty": r_penalty,
    })

    print(
        f"Q: {question[:50]}... "
        f"correct={is_correct} meta={num_meta} "
        f"reward={total_reward:.2f}",
        flush=True,
    )


def verl_config(model_path: str, n_gpus: int = 4) -> Dict[str, Any]:
    return {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": 32,
            "max_prompt_length": 2048,
            "max_response_length": 2048,
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 4,
                "name": "vllm",
                "gpu_memory_utilization": 0.7,
            },
            "actor": {
                "ppo_mini_batch_size": 16,
                "ppo_micro_batch_size_per_gpu": 2,
                "optim": {"lr": 5e-6},
                "use_kl_loss": False,
                "entropy_coeff": 0,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 4,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": n_gpus,
            "val_before_train": True,
            "critic_warmup": 0,
            "logger": ["console", "wandb"],
            "project_name": "metacot-math",
            "experiment_name": "metacot-grpo-agentlightning",
            "nnodes": 1,
            "save_freq": 200,
            "test_freq": 100,
            "total_epochs": 3,
            "total_training_steps": 1000,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--model", default="checkpoints/meta_sft")
    parser.add_argument("--n-gpus", type=int, default=4)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.train_file)
    problems = df.drop_duplicates("problem_id")
    train_data = [
        MathTask(
            question=row["question"],
            gold_answer=row["gold_answer"],
            problem_id=row["problem_id"],
            category=row["category"],
        )
        for _, row in problems.iterrows()
    ]
    val_data = train_data[:100]

    print(f"Train: {len(train_data)} problems, Val: {len(val_data)}", flush=True)

    config = verl_config(args.model, n_gpus=args.n_gpus)
    algorithm = agl.VERL(config)

    tracer = agl.OtelTracer()
    adapter = agl.LlmProxyTraceToTriplet()

    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=2,
        tracer=tracer,
        adapter=adapter,
    )

    # Register runner and datasets
    trainer.fit()


if __name__ == "__main__":
    main()
