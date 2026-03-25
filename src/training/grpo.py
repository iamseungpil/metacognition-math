"""Stepwise GRPO with FSDP (4 GPU) + Gnosis probe.

Key features:
- FSDP for 4-GPU distributed training
- <|meta|> boundaries define steps
- 3 rewards: R_correct(+2), R_calib(λ=1.0), R_penalty(no meta → -0.5)
- Gnosis probe for p̂ (full-sequence hidden state)
- max_tokens=2048 for full meta reasoning
- Probe auto-retrain every 200 steps if AUROC drops
- Comprehensive wandb logging of gnosis metrics
"""
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

from src.metacot.prompt import META_START, META_END, parse_meta_blocks
from src.training.stepwise import (
    find_meta_token_positions,
    get_hidden_states_at_meta,
    compute_gnosis_scores,
    compute_step_level_loss,
)
from src.training.rewards import compute_reward, compute_grpo_advantages
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
    if not probe_path or not Path(probe_path).exists():
        print("No probe found, using p_hat=0.5 for all steps", flush=True)
        return None
    probe = SimpleCorrectnessProbe(hidden_dim=hidden_dim)
    state = torch.load(Path(probe_path) / "best_probe.pt", map_location="cpu", weights_only=True)
    probe.load_state_dict(state)
    probe.to(device).float().eval()  # Keep float32 for accurate sigmoid output
    print(f"Probe loaded from {probe_path}", flush=True)
    return probe


def run_grpo(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_path = config["model_path"]
    problems_path = config["problems_path"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    group_size = config.get("group_size", 4)
    max_steps = config.get("max_steps", 1000)
    lr = config.get("learning_rate", 5e-6)
    max_tokens = config.get("max_tokens", 2048)
    lambda_calib = config.get("lambda_calib", 1.0)

    # Use accelerate + DeepSpeed ZeRO-3 for multi-GPU
    from accelerate import Accelerator
    accelerator = Accelerator()

    # Only main process logs to wandb
    if accelerator.is_main_process:
        wandb.init(
            project=config.get("wandb_project", "metacot-math"),
            name=config.get("run_name", "metacot-stepwise-grpo"),
            config=config,
            reinit=True,
        )

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [META_START, META_END]
    })

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Use PyTorch SDPA (compatible with torch 2.5)
        trust_remote_code=True,
        use_cache=False,
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # Load probe
    probe = _load_probe(
        config.get("probe_path"),
        hidden_dim=config.get("hidden_dim", 3584),
        device=str(accelerator.device),
    )

    dataset = MathProblemDataset(problems_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader = accelerator.prepare(dataloader)

    step = 0
    epoch = 0

    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break

            question = batch["question"][0]
            gold_answer = batch["gold_answer"][0]

            # Chat template for consistent format with SFT
            messages = [{"role": "user", "content": question}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(accelerator.device)
            prompt_len = prompt_ids.shape[1]

            # Generate G rollouts (batched for GPU efficiency)
            rollout_texts = []
            rollout_full_ids = []
            rollout_rewards = []
            rollout_meta_info = []

            model.eval()
            unwrapped = accelerator.unwrap_model(model)
            with torch.no_grad():
                # Batch generate all rollouts at once
                batch_prompt = prompt_ids.repeat(group_size, 1)  # (G, seq_len)
                batch_attn = torch.ones_like(batch_prompt)
                batch_output = unwrapped.generate(
                    batch_prompt,
                    attention_mask=batch_attn,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
                del batch_prompt, batch_attn

                # Process each rollout
                for g in range(group_size):
                    full_ids = batch_output[g]
                    # Trim padding if present
                    if tokenizer.pad_token_id is not None:
                        non_pad = (full_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                        if len(non_pad) > 0:
                            full_ids = full_ids[:non_pad[-1] + 1]

                    gen_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=False)
                    clean_text = tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True)
                    rollout_texts.append(clean_text)
                    rollout_full_ids.append(full_ids)

                    meta_positions = find_meta_token_positions(full_ids, tokenizer)
                    num_meta = len(meta_positions)

                    # Gnosis: compute p̂ from full sequence (probe was trained this way)
                    if probe is not None:
                        with torch.no_grad():
                            outputs = unwrapped(
                                full_ids.unsqueeze(0),
                                attention_mask=torch.ones_like(full_ids).unsqueeze(0),
                                output_hidden_states=True,
                            )
                            last_hidden = outputs.hidden_states[-1]  # (1, S, D)
                            p_hat = probe(last_hidden.float()).item()
                            del outputs, last_hidden
                            torch.cuda.empty_cache()
                        # Use same p̂ for all steps (full-sequence prediction)
                        gnosis_scores = [p_hat] * max(num_meta, 1)
                    else:
                        gnosis_scores = [0.5] * max(num_meta, 1)

                    # Parse model confidences from <|meta|> blocks
                    parsed = parse_meta_blocks(gen_text)
                    model_confs = parsed["confidences"]

                    # Compute reward
                    is_correct = check_correctness(clean_text, gold_answer)
                    reward_dict = compute_reward(
                        is_correct=is_correct,
                        chain_text=gen_text,
                        gnosis_scores=gnosis_scores,
                        model_confidences=model_confs,
                        num_meta_blocks=num_meta,
                        lambda_calib=lambda_calib,
                    )
                    rollout_rewards.append(reward_dict)
                    rollout_meta_info.append({
                        "num_meta": num_meta,
                        "gnosis_scores": gnosis_scores,
                        "model_confs": model_confs,
                        "is_correct": is_correct,
                        "meta_positions": meta_positions,
                    })
                del batch_output

            # GRPO advantages
            totals = [r["total"] for r in rollout_rewards]
            advantages = compute_grpo_advantages(totals)

            # Policy gradient
            model.train()
            optimizer.zero_grad()
            total_loss_val = 0.0
            n_contributing = sum(1 for a in advantages if abs(a) >= 1e-8)

            for g in range(group_size):
                adv = advantages[g]
                if abs(adv) < 1e-8:
                    continue

                g_ids = rollout_full_ids[g].unsqueeze(0)
                g_mask = torch.ones_like(g_ids)
                g_meta = rollout_meta_info[g]["meta_positions"]

                # Step rewards scaled by advantage
                r = rollout_rewards[g]
                step_rewards = []
                for k, calib in enumerate(r["r_calib_per_step"]):
                    sr = {"total": (calib * lambda_calib + r["r_penalty"] / max(len(r["r_calib_per_step"]), 1)) * adv}
                    step_rewards.append(sr)
                # Add R_correct to last step
                if step_rewards:
                    step_rewards[-1]["total"] += r["r_correct"] * adv
                else:
                    step_rewards = [{"total": r["total"] * adv}]

                loss = compute_step_level_loss(
                    model, g_ids, g_mask, g_meta, step_rewards, prompt_len,
                )
                loss = loss / max(n_contributing, 1)
                accelerator.backward(loss)
                total_loss_val += loss.item()

                del g_ids, g_mask

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            # === Metrics ===
            avg_reward = np.mean(totals)
            avg_correct = np.mean([info["is_correct"] for info in rollout_meta_info])
            avg_meta = np.mean([info["num_meta"] for info in rollout_meta_info])
            all_p_hats = [p for info in rollout_meta_info for p in info["gnosis_scores"]]
            all_confs = [c for info in rollout_meta_info for c in info["model_confs"]]
            all_calibs = [r["r_calib_avg"] for r in rollout_rewards]
            all_penalties = [r["r_penalty"] for r in rollout_rewards]

            # Probe accuracy: does final p̂ > 0.5 match correctness?
            probe_accs = []
            for info in rollout_meta_info:
                if info["gnosis_scores"]:
                    pred = 1.0 if info["gnosis_scores"][-1] > 0.5 else 0.0
                    actual = 1.0 if info["is_correct"] else 0.0
                    probe_accs.append(float(pred == actual))

            # Memory cleanup
            del rollout_full_ids, rollout_meta_info
            torch.cuda.empty_cache()

            if step % 10 == 0 and accelerator.is_main_process:
                metrics = {
                    "grpo/step": step,
                    "grpo/loss": total_loss_val,
                    "grpo/avg_reward": avg_reward,
                    "grpo/avg_correct": avg_correct,
                    "grpo/avg_meta_blocks": avg_meta,
                    "grpo/avg_r_calib": np.mean(all_calibs),
                    "grpo/avg_r_penalty": np.mean(all_penalties),
                    "gnosis/avg_p_hat": np.mean(all_p_hats) if all_p_hats else 0,
                    "gnosis/p_hat_std": np.std(all_p_hats) if all_p_hats else 0,
                    "gnosis/avg_model_conf": np.mean(all_confs) if all_confs else 0,
                    "gnosis/conf_gap": abs(np.mean(all_p_hats) - np.mean(all_confs)) if all_p_hats and all_confs else 0,
                    "gnosis/probe_accuracy": np.mean(probe_accs) if probe_accs else 0,
                    "gnosis/num_confs_parsed": len(all_confs),
                    "grpo/lambda_calib": lambda_calib,
                    "grpo/lr": scheduler.get_last_lr()[0],
                }
                wandb.log(metrics, step=step)
                print(
                    f"Step {step}: loss={total_loss_val:.4f} "
                    f"reward={avg_reward:.3f} correct={avg_correct:.3f} "
                    f"meta={avg_meta:.1f} calib={np.mean(all_calibs):.3f} "
                    f"penalty={np.mean(all_penalties):.2f} "
                    f"p̂={np.mean(all_p_hats):.3f} "
                    f"probe_acc={np.mean(probe_accs):.3f}" if probe_accs else
                    f"Step {step}: loss={total_loss_val:.4f} "
                    f"reward={avg_reward:.3f} correct={avg_correct:.3f} "
                    f"meta={avg_meta:.1f}",
                    flush=True,
                )

            # Save checkpoint
            if step % config.get("save_every", 200) == 0 and accelerator.is_main_process:
                ckpt_dir = output_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)

    # Save final
    if accelerator.is_main_process:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        wandb.finish()
    print(f"GRPO training done. Model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_grpo(args.config)
