"""GRPO + SimpleProbe + Per-Step Calibration (Phase 3).

Architecture:
- HF generate on 4 GPUs (DDP)
- After generation: HF forward pass → hidden_states → SimpleProbe → p̂
- p̂ used for R_calib: 1 - |c_text - p̂| (probe-based calibration)
- Rewards: R_correct(2.0/0.0) + R_calib(probe) + R_penalty(meta blocks)
- Per-step: each <|meta|> step's confidence compared to global p̂
- Standard GRPO advantages with per-step calibrated sequence reward
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Monkey-patch FSDPModule for PyTorch 2.5 compat
import torch.distributed.fsdp as _fsdp_mod
if not hasattr(_fsdp_mod, "FSDPModule"):
    _fsdp_mod.FSDPModule = type("FSDPModule", (), {})

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from src.metacot.prompt import META_START, META_END, parse_meta_blocks
from src.training.stepwise import find_meta_token_positions


# ─── SimpleCorrectnessProbe (same architecture as probe_sft.py) ───
class SimpleCorrectnessProbe(nn.Module):
    def __init__(self, hidden_dim=4096, intermediate_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim // 2, 1),
        )

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)
        return torch.sigmoid(self.net(pooled).squeeze(-1))


def _extract_answer(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?:the answer is|answer:\s*)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def check_correctness(model_answer, gold_answer):
    model_final = _extract_answer(model_answer)
    gold_str = str(gold_answer).strip()
    gold_final = _extract_answer(gold_str) or gold_str
    if not model_final:
        return False
    if model_final == gold_final:
        return True
    try:
        if abs(float(model_final) - float(gold_final)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return model_final.lower().strip() == gold_final.lower().strip()


def metacot_reward_fn(completions, ground_truth=None, **kwargs):
    """Sequence-level reward. TRL calls this for initial advantage computation.
    We override with probe-based rewards in _generate_and_score_completions.
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = check_correctness(text, str(gt))

        parsed = parse_meta_blocks(text)
        num_meta = parsed["num_blocks"]
        confidences = parsed["confidences"]

        r_correct = 2.0 if is_correct else 0.0
        r_penalty = 0.0 if num_meta >= 2 else (-0.3 if num_meta == 1 else -0.5)

        # Text-based R_calib (probe-based R_calib applied later in override)
        r_calib = 0.0
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            actual = 1.0 if is_correct else 0.0
            r_calib = max(0.0, 1.0 - abs(avg_conf - actual))

        rewards.append(r_correct + r_calib + r_penalty)
    return rewards


class MetaCotGRPOTrainer(GRPOTrainer):
    """GRPO with SimpleProbe-based R_calib per <|meta|> step."""

    def __init__(self, *args, probe=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Move probe to the correct device for this DDP rank
        if probe is not None:
            self.probe = probe.to(self.accelerator.device)
        else:
            self.probe = None
        self._cached_ground_truths = []

    def _generate_and_score_completions(self, inputs):
        """After parent generates and scores, add probe-based metrics."""
        if isinstance(inputs, list):
            self._cached_ground_truths = [x.get("ground_truth", "") for x in inputs]
        elif isinstance(inputs, dict):
            self._cached_ground_truths = inputs.get("ground_truth", [])

        outputs = super()._generate_and_score_completions(inputs)

        B = outputs["completion_ids"].shape[0]
        device = outputs["completion_ids"].device
        tokenizer = self.processing_class
        num_gens = getattr(self.args, 'num_generations', 1)
        n_prompts = len(self._cached_ground_truths)

        # Per-step probe scoring: 1 forward pass → slice hidden at each <|meta|>
        new_rewards = []
        all_p_hats = []
        all_r_calibs = []
        n_with_meta = 0
        n_steps_total = 0
        n_direction_changes = 0  # confidence decrease during rollout

        unwrapped = self.accelerator.unwrap_model(self.model)

        for i in range(B):
            comp_ids = outputs["completion_ids"][i]
            comp_text = tokenizer.decode(comp_ids, skip_special_tokens=False)
            prompt_idx = i // num_gens if num_gens > 0 else i
            gt = self._cached_ground_truths[prompt_idx] if prompt_idx < n_prompts else ""
            is_correct = check_correctness(comp_text, str(gt))

            parsed = parse_meta_blocks(comp_text)
            num_meta = parsed["num_blocks"]
            confidences = parsed["confidences"]
            meta_positions = find_meta_token_positions(comp_ids, tokenizer)

            r_correct = 2.0 if is_correct else 0.0
            r_penalty = 0.0 if num_meta >= 2 else (-0.3 if num_meta == 1 else -0.5)

            if meta_positions:
                n_with_meta += 1
                n_steps_total += len(meta_positions)

            # 1 forward pass → get all hidden states
            step_calibs = []
            step_p_hats = []
            if self.probe is not None and meta_positions:
                try:
                    full_ids = torch.cat([outputs["prompt_ids"][i], comp_ids])
                    full_mask = torch.cat([outputs["prompt_mask"][i], outputs["completion_mask"][i]])
                    prompt_len = outputs["prompt_ids"][i].shape[0]

                    with torch.no_grad():
                        out = unwrapped(
                            full_ids.unsqueeze(0),
                            attention_mask=full_mask.unsqueeze(0),
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        hidden = out.hidden_states[-1]  # (1, S, D)

                    # Per-step probe: slice hidden at each <|/meta|> position
                    for k, (start, end) in enumerate(meta_positions):
                        abs_end = prompt_len + end + 1  # absolute position in full_ids
                        abs_end = min(abs_end, hidden.shape[1])

                        h_prefix = hidden[:, :abs_end, :]  # hidden up to this step
                        m_prefix = full_mask[:abs_end].unsqueeze(0).float()

                        p_hat_k = self.probe(h_prefix.float(), m_prefix).item()
                        step_p_hats.append(p_hat_k)

                        # R_calib_k = 1 - |c_text_k - p̂_k|
                        conf_k = confidences[k] if k < len(confidences) else None
                        if conf_k is not None:
                            r_calib_k = max(0.0, 1.0 - abs(conf_k - p_hat_k))
                        else:
                            r_calib_k = 0.0
                        step_calibs.append(r_calib_k)

                    del out, hidden
                    torch.cuda.empty_cache()

                except Exception as e:
                    if i == 0:
                        print(f"[Probe error] {e}")

            # Aggregate per-step R_calib → mean
            if step_calibs:
                r_calib = sum(step_calibs) / len(step_calibs)
            else:
                # Fallback: text-based if no probe or no meta blocks
                r_calib = 0.0
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    actual = 1.0 if is_correct else 0.0
                    r_calib = max(0.0, 1.0 - abs(avg_conf - actual))

            total_reward = r_correct + r_calib + r_penalty
            new_rewards.append(total_reward)
            all_p_hats.extend(step_p_hats if step_p_hats else [0.5])
            all_r_calibs.extend(step_calibs if step_calibs else [r_calib])

            # Track direction changes (confidence decrease)
            if len(confidences) >= 2:
                for ci in range(1, len(confidences)):
                    if confidences[ci] < confidences[ci-1] - 0.1:
                        n_direction_changes += 1
                        break

        # GRPO advantage: z-normalize within group
        if len(new_rewards) >= 2:
            mean_r = sum(new_rewards) / len(new_rewards)
            var_r = sum((r - mean_r) ** 2 for r in new_rewards) / len(new_rewards)
            std_r = max(var_r ** 0.5, 1e-8)
            new_advantages = [(r - mean_r) / std_r for r in new_rewards]
        else:
            new_advantages = [0.0] * len(new_rewards)

        outputs["advantages"] = torch.tensor(new_advantages, device=device, dtype=torch.float32)

        # Log metrics
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["meta_block_ratio"].append(n_with_meta / max(B, 1))
        self._metrics[mode]["probe/mean_p_hat"].append(sum(all_p_hats) / max(len(all_p_hats), 1))
        self._metrics[mode]["probe/mean_r_calib"].append(sum(all_r_calibs) / max(len(all_r_calibs), 1))
        self._metrics[mode]["reward"].append(sum(new_rewards) / len(new_rewards))
        self._metrics[mode]["avg_steps_per_completion"].append(n_steps_total / max(B, 1))
        self._metrics[mode]["direction_change_ratio"].append(n_direction_changes / max(B, 1))

        return outputs


def prepare_dataset(data_path):
    df = pd.read_parquet(data_path)
    records = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        gt = row.get("reward_model", {})
        if isinstance(gt, str):
            gt = json.loads(gt)
        records.append({
            "prompt": prompt,
            "ground_truth": gt.get("ground_truth", ""),
        })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--probe_path", default="checkpoints/simple_probe_qwen3/best_probe.pt")
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", default="checkpoints/qwen3_grpo_probe")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=32)
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "metacot-math"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    meta_tokens = [META_START, META_END]
    existing = set(tokenizer.additional_special_tokens or [])
    to_add = [t for t in meta_tokens if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": list(existing) + to_add})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )
    if to_add:
        model.resize_token_embeddings(len(tokenizer))

    # Load probe
    probe = None
    if os.path.exists(args.probe_path):
        hidden_dim = model.config.hidden_size
        probe = SimpleCorrectnessProbe(hidden_dim=hidden_dim)
        probe.load_state_dict(torch.load(args.probe_path, map_location="cpu"))
        probe.eval()
        probe.cuda()
        print(f"Loaded probe from {args.probe_path} (AUROC ~0.95)")
    else:
        print(f"WARNING: Probe not found at {args.probe_path}")

    # Full training (no LoRA) — LoRA was too restrictive for GRPO
    lora_config = None

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=2048,
        temperature=1.0,  # high for diverse rollouts → mix of correct/incorrect
        use_vllm=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-6,  # full training, conservative LR
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.0,
        logging_steps=1,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        run_name=f"grpo-probe-step{args.max_steps}-{torch.cuda.device_count()}gpu",
        remove_unused_columns=False,
    )

    train_dataset = prepare_dataset(args.train_data)
    print(f"=== MetaCot GRPO + SimpleProbe ===")
    print(f"Dataset: {len(train_dataset)} problems")
    print(f"Probe: SimpleCorrectnessProbe (AUROC ~0.95)")
    print(f"Rewards: R_correct + R_calib(probe) + R_penalty(meta)")

    trainer = MetaCotGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=metacot_reward_fn,
        processing_class=tokenizer,
        probe=probe,
    )

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
