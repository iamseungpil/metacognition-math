"""Stepwise GRPO training with <|meta|> boundaries and Gnosis probe."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.metacot.prompt import META_START, META_END
from src.training.stepwise import (
    find_meta_token_positions,
    get_hidden_states_at_meta,
    compute_gnosis_scores,
    compute_stepwise_rewards,
    compute_step_level_loss,
)
from src.training.rewards import compute_grpo_advantages
from src.rollout.vllm_rollout import check_correctness
from src.probes.simple_probe import SimpleCorrectnessProbe


class MathProblemDataset(Dataset):
    def __init__(self, problems_path: str):
        self.df = pd.read_parquet(problems_path)
        self.problems = self.df.drop_duplicates("problem_id").reset_index(drop=True)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        row = self.problems.iloc[idx]
        return {
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "category": row["category"],
            "problem_id": row["problem_id"],
        }


def _load_probe(probe_path: str, hidden_dim: int = 3584, device: str = "cuda"):
    """Load trained Gnosis/SimpleProbe."""
    if not probe_path or not Path(probe_path).exists():
        print("No probe found, using p_hat=0.5 for all steps")
        return None

    probe = SimpleCorrectnessProbe(hidden_dim=hidden_dim)
    state = torch.load(Path(probe_path) / "best_probe.pt", map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.to(device).to(torch.bfloat16).eval()  # Match model dtype
    print(f"Probe loaded from {probe_path}")
    return probe


def run_grpo(config_path: str):
    """Run stepwise GRPO with <|meta|> boundaries."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config["model_path"]
    problems_path = config["problems_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    group_size = config.get("group_size", 4)
    max_steps = config.get("max_steps", 200)
    lr = config.get("learning_rate", 5e-6)
    max_tokens = config.get("max_tokens", 512)
    lambda1 = config.get("lambda1", 0.5)
    lambda2 = config.get("lambda2", 0.3)

    wandb.init(
        project=config.get("wandb_project", "metacot-math"),
        name=config.get("run_name", "metacot-stepwise-grpo"),
        config=config,
        reinit=True,
    )

    # Load model + tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure <|meta|> tokens are in tokenizer
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [META_START, META_END]
    })

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    ).cuda()

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    # Load probe
    probe = _load_probe(
        config.get("probe_path"),
        hidden_dim=config.get("hidden_dim", 3584),
        device="cuda",
    )

    # Load problems
    dataset = MathProblemDataset(problems_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    step = 0
    epoch = 0

    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break

            question = batch["question"][0]
            gold_answer = batch["gold_answer"][0]

            # Use chat template to match SFT training format
            messages = [{"role": "user", "content": question}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).cuda()
            prompt_len = prompt_ids.shape[1]

            # Generate G rollouts
            rollout_texts = []
            rollout_full_ids = []
            rollout_meta_positions = []
            rollout_step_rewards = []

            model.eval()
            with torch.no_grad():
                for g in range(group_size):
                    output = model.generate(
                        prompt_ids,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                    full_ids = output[0]
                    gen_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=False)
                    clean_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
                    rollout_texts.append(clean_text)
                    rollout_full_ids.append(full_ids)

                    # Find <|meta|> positions
                    meta_positions = find_meta_token_positions(full_ids, tokenizer)
                    rollout_meta_positions.append(meta_positions)
                    del output

                # Get Gnosis scores separately (outside generate loop to save memory)
                for g in range(group_size):
                    full_ids = rollout_full_ids[g]
                    meta_positions = rollout_meta_positions[g]
                    gen_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=False)
                    clean_text = rollout_texts[g]

                    if probe is not None and meta_positions:
                        hidden_states = get_hidden_states_at_meta(
                            model, full_ids.unsqueeze(0),
                            torch.ones_like(full_ids).unsqueeze(0),
                            meta_positions,
                        )
                        gnosis_scores = compute_gnosis_scores(probe, hidden_states)
                        del hidden_states
                    else:
                        gnosis_scores = [0.5] * max(len(meta_positions), 1)

                    is_correct = check_correctness(clean_text, gold_answer)
                    srs = compute_stepwise_rewards(
                        gen_text, is_correct, gnosis_scores,
                        lambda1=lambda1, lambda2=lambda2,
                    )
                    rollout_step_rewards.append(srs)

            # Compute per-rollout total reward for GRPO advantage
            rollout_totals = [
                sum(sr["total"] for sr in srs)
                for srs in rollout_step_rewards
            ]
            advantages = compute_grpo_advantages(rollout_totals, group_size)

            # Policy gradient with step-level rewards
            model.train()
            optimizer.zero_grad()
            total_loss_val = 0.0
            n_contributing = sum(1 for a in advantages if abs(a) >= 1e-8)

            for g in range(group_size):
                adv = advantages[g]
                if abs(adv) < 1e-8:
                    continue

                g_full_ids = rollout_full_ids[g].unsqueeze(0)
                g_attention_mask = torch.ones_like(g_full_ids)
                g_meta_positions = rollout_meta_positions[g]

                # Scale step rewards by GRPO advantage
                scaled_rewards = [
                    {**sr, "total": sr["total"] * adv}
                    for sr in rollout_step_rewards[g]
                ]

                loss = compute_step_level_loss(
                    model, g_full_ids, g_attention_mask,
                    g_meta_positions, scaled_rewards, prompt_len,
                )
                loss = loss / max(n_contributing, 1)
                loss.backward()
                total_loss_val += loss.item()

                del g_full_ids, g_attention_mask

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Memory cleanup
            del rollout_full_ids, rollout_meta_positions, rollout_step_rewards
            torch.cuda.empty_cache()

            step += 1

            # Metrics
            avg_reward = np.mean(rollout_totals)
            avg_correct = np.mean([
                1.0 if check_correctness(t, gold_answer) else 0.0
                for t in rollout_texts
            ])
            avg_meta_blocks = np.mean([
                len(pos) for pos in rollout_meta_positions
            ])

            # Average step-level metrics
            all_calibs = [sr["r_calib"] for srs in rollout_step_rewards for sr in srs]
            all_progress = [sr["r_progress"] for srs in rollout_step_rewards for sr in srs]

            if step % 10 == 0:
                metrics = {
                    "grpo/step": step,
                    "grpo/loss": total_loss_val,
                    "grpo/avg_reward": avg_reward,
                    "grpo/avg_correct": avg_correct,
                    "grpo/avg_meta_blocks": avg_meta_blocks,
                    "grpo/avg_r_calib": np.mean(all_calibs) if all_calibs else 0,
                    "grpo/avg_r_progress": np.mean(all_progress) if all_progress else 0,
                    "grpo/lambda1": lambda1,
                    "grpo/lambda2": lambda2,
                    "grpo/lr": scheduler.get_last_lr()[0],
                }
                wandb.log(metrics, step=step)
                print(
                    f"Step {step}: loss={total_loss_val:.4f} "
                    f"reward={avg_reward:.3f} correct={avg_correct:.3f} "
                    f"meta_blocks={avg_meta_blocks:.1f} "
                    f"r_calib={np.mean(all_calibs):.3f}",
                    flush=True,
                )

            # Save checkpoint
            if step % config.get("save_every", 500) == 0:
                ckpt_dir = output_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    # Save final model
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    wandb.finish()
    print(f"GRPO training done. Model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_grpo(args.config)
