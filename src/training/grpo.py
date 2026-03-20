"""GRPO training with R_meta metacognitive reward."""
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
import wandb
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.training.rewards import (
    compute_r_meta,
    extract_confidence_from_chain,
    extract_strategy_target,
    compute_grpo_advantages,
    compute_gnosis_temporal_difference,
)
from src.rollout.vllm_rollout import check_correctness, build_chat_messages
from src.data.dataset_loader import extract_boxed_answer


class MathProblemDataset(Dataset):
    def __init__(self, problems_path: str):
        self.df = pd.read_parquet(problems_path)
        self.problems = self.df.drop_duplicates("problem_id").reset_index(drop=True)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        row = self.problems.iloc[idx]
        return {
            "problem_id": row["problem_id"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "category": row["category"],
            "difficulty": row["difficulty"],
        }


def run_grpo(config_path: str):
    """Run GRPO training with metacognitive reward."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config["model_path"]  # SFT checkpoint
    problems_path = config["problems_path"]
    profile_path = config["profile_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    group_size = config.get("group_size", 8)
    max_steps = config.get("max_steps", 2000)
    lr = config.get("learning_rate", 5e-6)
    lambda1_start = config.get("lambda1_start", 0.5)
    lambda1_end = config.get("lambda1_end", 1.0)
    lambda2_start = config.get("lambda2_start", 0.0)
    lambda2_end = config.get("lambda2_end", 0.5)
    lambda_warmup_steps = config.get("lambda_warmup_steps", 1000)
    gnosis_retrain_interval = config.get("gnosis_retrain_interval", 500)
    max_tokens = config.get("max_tokens", 2048)
    wandb_project = config.get("wandb_project", "metacot-math")

    with open(profile_path) as f:
        profile = json.load(f)
    weak_categories = profile.get("weak_categories", [])

    # Init wandb
    wandb.init(
        project=wandb_project,
        name=config.get("run_name", "metacot-grpo"),
        config=config,
    )

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    # Load problems
    dataset = MathProblemDataset(problems_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Load probe for R_calib (try simple_probe first, then gnosis)
    probe_path = config.get("gnosis_model_path")
    if probe_path and not Path(probe_path).exists():
        # Fallback to simple_probe checkpoint
        simple_path = str(Path(probe_path).parent / "simple_probe")
        if Path(simple_path).exists():
            probe_path = simple_path
    gnosis_model = _load_gnosis_if_available(probe_path)

    step = 0
    epoch = 0
    running_reward = 0.0
    running_correct = 0.0

    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break

            question = batch["question"][0]
            gold_answer = batch["gold_answer"][0]
            category = batch["category"][0]

            # Lambda scheduling
            progress = min(step / lambda_warmup_steps, 1.0)
            lambda1 = lambda1_start + (lambda1_end - lambda1_start) * progress
            lambda2 = lambda2_start + (lambda2_end - lambda2_start) * progress

            # Generate G rollouts
            messages = build_chat_messages(question)
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()

            rollout_texts = []
            rollout_log_probs = []
            rewards = []
            reward_components = []

            model.eval()
            with torch.no_grad():
                for g in range(group_size):
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                    gen_ids = output.sequences[0, input_ids.shape[1]:]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    rollout_texts.append(gen_text)

                    # Compute log probs
                    scores = torch.stack(output.scores, dim=0)  # (T, 1, V)
                    log_probs = F.log_softmax(scores, dim=-1)
                    token_log_probs = log_probs[
                        range(len(gen_ids)), 0, gen_ids
                    ]
                    rollout_log_probs.append(token_log_probs)

                    # Compute reward
                    is_correct = check_correctness(gen_text, gold_answer)
                    c_text = extract_confidence_from_chain(gen_text)
                    p_hat = _get_gnosis_score(gnosis_model, model, tokenizer, prompt + gen_text)
                    strategy_cat = extract_strategy_target(gen_text)

                    r_dict = compute_r_meta(
                        is_correct=is_correct,
                        c_text=c_text,
                        p_hat=p_hat,
                        strategy_category=strategy_cat,
                        weak_categories=weak_categories,
                        lambda1=lambda1,
                        lambda2=lambda2,
                    )
                    rewards.append(r_dict["total"])
                    reward_components.append(r_dict)

            # GRPO: compute advantages
            advantages = compute_grpo_advantages(rewards, group_size)

            # Policy gradient update with per-rollout gradient accumulation
            model.train()
            optimizer.zero_grad()
            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
            total_loss_val = 0.0

            for g in range(group_size):
                adv = advantages[g]
                if abs(adv) < 1e-8:
                    continue

                # Recompute forward pass for gradients
                full_text = prompt + rollout_texts[g]
                enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
                input_ids_full = enc["input_ids"].cuda()
                attention_mask = enc["attention_mask"].cuda()

                outputs = model(
                    input_ids=input_ids_full,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits

                # Only compute loss on generated tokens
                gen_logits = logits[:, prompt_len - 1:-1]
                gen_targets = input_ids_full[:, prompt_len:]

                log_probs_g = F.log_softmax(gen_logits, dim=-1)
                token_log_probs = log_probs_g.gather(
                    -1, gen_targets.unsqueeze(-1)
                ).squeeze(-1)

                # Policy gradient: -advantage * log_prob, backward per rollout
                loss = -(adv * token_log_probs.mean()) / group_size
                loss.backward()
                total_loss_val += loss.item()

                del outputs, logits, input_ids_full, attention_mask

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Metrics
            avg_reward = np.mean(rewards)
            avg_correct = np.mean([
                check_correctness(t, gold_answer) for t in rollout_texts
            ])
            running_reward = 0.95 * running_reward + 0.05 * avg_reward
            running_correct = 0.95 * running_correct + 0.05 * avg_correct

            # Compute answer entropy (diversity of answers)
            answers = [extract_boxed_answer(t) or "NONE" for t in rollout_texts]
            unique_answers = len(set(answers))
            answer_entropy = np.log2(max(unique_answers, 1))

            step += 1

            if step % 10 == 0:
                # Aggregate reward components
                avg_r_correct = np.mean([rc["r_correct"] for rc in reward_components])
                avg_r_calib = np.mean([rc["r_calib"] for rc in reward_components])
                avg_r_strat = np.mean([rc["r_strat"] for rc in reward_components])

                metrics = {
                    "grpo/step": step,
                    "grpo/loss": total_loss_val,
                    "grpo/avg_reward": avg_reward,
                    "grpo/running_reward": running_reward,
                    "grpo/avg_correct": avg_correct,
                    "grpo/running_correct": running_correct,
                    "grpo/r_correct": avg_r_correct,
                    "grpo/r_calib": avg_r_calib,
                    "grpo/r_strat": avg_r_strat,
                    "grpo/lambda1": lambda1,
                    "grpo/lambda2": lambda2,
                    "grpo/lr": scheduler.get_last_lr()[0],
                    "grpo/answer_entropy": answer_entropy,
                    "grpo/unique_answers": unique_answers,
                    "grpo/reward_std": np.std(rewards),
                    "grpo/advantage_max": max(advantages),
                    "grpo/advantage_min": min(advantages),
                }
                wandb.log(metrics, step=step)
                print(
                    f"Step {step}: loss={total_loss_val:.4f} "
                    f"reward={avg_reward:.3f} correct={avg_correct:.3f} "
                    f"entropy={answer_entropy:.2f}"
                )

            # Save checkpoint
            if step % config.get("save_every", 500) == 0:
                ckpt_dir = output_dir / f"checkpoint-{step}"
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"Saved checkpoint to {ckpt_dir}")

    # Final save
    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")
    wandb.finish()
    print(f"GRPO training done. Model saved to {output_dir / 'final'}")


def _load_gnosis_if_available(path: Optional[str]):
    """Load probe model if path provided, else return None.

    Tries Gnosis (custom transformers) first; falls back to SimpleCorrectnessProbe.
    """
    if path is None or not Path(path).exists():
        return None

    # Try loading as SimpleCorrectnessProbe (our baseline)
    probe_pt = Path(path) / "best_probe.pt" if Path(path).is_dir() else Path(path)
    if probe_pt.exists() and probe_pt.suffix == ".pt":
        try:
            from src.probes.simple_probe import SimpleCorrectnessProbe
            probe = SimpleCorrectnessProbe(hidden_dim=3584)
            probe.load_state_dict(torch.load(probe_pt, map_location="cpu", weights_only=True))
            probe = probe.cuda().eval()
            print(f"Loaded SimpleCorrectnessProbe from {probe_pt}")
            return {"type": "simple", "model": probe}
        except Exception as e:
            print(f"Warning: Could not load simple probe: {e}")

    # Try loading as Gnosis (custom transformers fork)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()
        if hasattr(model, "_should_stop"):
            print(f"Loaded Gnosis model from {path}")
            return {"type": "gnosis", "model": model}
        else:
            print(f"Warning: Model at {path} has no _should_stop, skipping")
            del model
            return None
    except Exception as e:
        print(f"Warning: Could not load Gnosis model: {e}")
        return None


def _get_gnosis_score(
    probe_info, policy_model, tokenizer, text: str
) -> float:
    """Get correctness probability from probe. Falls back to 0.5 if unavailable."""
    if probe_info is None:
        return 0.5

    try:
        if probe_info["type"] == "simple":
            # Use policy model to get hidden states, then probe
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            input_ids = enc["input_ids"].cuda()
            attention_mask = enc["attention_mask"].cuda()

            with torch.no_grad():
                out = policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1]
                prob = probe_info["model"](hidden, attention_mask)
                return float(prob.item())

        elif probe_info["type"] == "gnosis":
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
            input_ids = enc["input_ids"].cuda()
            attention_mask = enc["attention_mask"].cuda()

            with torch.no_grad():
                out = probe_info["model"].model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False,
                )
                hidden = out.last_hidden_state
                score = probe_info["model"]._should_stop(
                    last_hidden=hidden,
                    attn_stack=None,
                    token_probs=None,
                )
                return float(score.squeeze().clamp(1e-6, 1 - 1e-6).item())
    except Exception:
        pass
    return 0.5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_grpo(args.config)
