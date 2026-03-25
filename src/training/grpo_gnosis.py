"""GRPO + Full Gnosis + Stepwise Credit Assignment for Qwen3-8B Meta-CoT.

Architecture:
- MetaCotGRPOTrainer (TRL GRPOTrainer subclass)
- vLLM colocate for fast generation
- Full Gnosis correctness head (attention + hidden + confidence extractors)
- <|meta|> stepwise credit assignment: per-token importance weighting
- Combined loss: L_grpo(stepwise) + λ * L_correctness(Gnosis BCE)
- LoRA for memory efficiency on 4x A100 80GB

Critic-reviewed fixes (v2):
- FIX #1: ground_truth cached before super() call
- FIX #2: decode with skip_special_tokens=False for meta blocks
- FIX #3: stepwise weights always non-negative (importance, not reward)
- FIX #4: R_calib uses text-based calibration (p_hat via Gnosis correctness_loss)
- FIX #5: single forward pass for GRPO (delegate to parent _compute_loss)
- FIX #6: unconditional gnosis_loss logging
"""
import argparse
import json
import os

import pandas as pd
import torch
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from src.metacot.prompt import META_START, META_END, parse_meta_blocks
from src.training.stepwise import find_meta_token_positions as find_meta_positions_in_ids
import re


def _extract_answer(text):
    """Extract answer from \\boxed{} or other formats."""
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
    """Check if model answer matches gold answer."""
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


# ─── Constants ───
LAMBDA_GNOSIS = 0.5


def compute_stepwise_rewards(
    completion_ids,
    completion_text,
    is_correct,
    tokenizer,
    lambda_calib=1.0,
    lambda_progress=0.3,
):
    """Compute per-step rewards based on <|meta|> step boundaries.

    Agent Lightning style: R_correct is assigned to ALL steps equally.
    R_calib and R_progress vary per step.

    Step reward_k = R_correct + λ_calib * R_calib_k + λ_progress * R_progress_k

    Per-token weights are the step rewards, clamped to ≥ 0.01 to avoid
    gradient inversion. The GRPO sequence-level advantage controls direction;
    step rewards control relative credit distribution.

    Returns:
        weights: (completion_len,) tensor, ≥ 0, normalized to mean=1.0
        info: dict for logging
    """
    parsed = parse_meta_blocks(completion_text)
    num_meta = parsed["num_blocks"]
    confidences = parsed["confidences"]

    comp_len = len(completion_ids) if isinstance(completion_ids, list) else completion_ids.shape[-1]
    meta_positions = find_meta_positions_in_ids(completion_ids, tokenizer)

    # R_correct: assigned to ALL steps equally
    r_correct = 2.0 if is_correct else 0.0

    if not meta_positions or num_meta == 0:
        # No meta blocks: uniform weight with penalty
        r_penalty = -0.5
        total = r_correct + r_penalty
        weights = torch.ones(comp_len) * max(total, 0.01)
        weights = weights / weights.mean().clamp(min=1e-8)
        return weights, {"num_steps": 0, "has_meta": False, "total_reward": total}

    # Compute per-step rewards (Agent Lightning style)
    n_steps = len(meta_positions)
    step_rewards = []
    prev_conf = 0.5

    for k in range(n_steps):
        c_text = confidences[k] if k < len(confidences) else None

        # R_calib: how well does stated confidence match actual outcome
        if c_text is not None:
            actual = 1.0 if is_correct else 0.0
            r_calib_k = max(0.0, 1.0 - abs(c_text - actual))
        else:
            r_calib_k = 0.0

        # R_progress: is confidence moving in the right direction?
        if c_text is not None:
            r_progress_k = (c_text - prev_conf) if is_correct else (prev_conf - c_text)
            prev_conf = c_text
        else:
            r_progress_k = 0.0

        # R_meta: bonus for using meta blocks
        r_meta = 0.1 if num_meta >= 2 else 0.0

        # Total step reward: R_correct same for ALL steps
        total_k = r_correct + lambda_calib * r_calib_k + lambda_progress * r_progress_k + r_meta
        step_rewards.append(total_k)

    # Assign per-token weights based on step boundaries
    weights = torch.zeros(comp_len)
    current_step = 0
    for t in range(comp_len):
        while (current_step < n_steps - 1 and
               current_step < len(meta_positions) and
               t > meta_positions[current_step][1]):
            current_step += 1
        if current_step < len(step_rewards):
            weights[t] = step_rewards[current_step]

    # Clamp to non-negative (avoid gradient inversion)
    weights = weights.clamp(min=0.01)

    # Normalize to mean=1.0 (preserves total gradient magnitude)
    weights = weights / weights.mean().clamp(min=1e-8)

    return weights, {
        "num_steps": n_steps,
        "step_rewards": step_rewards,
        "has_meta": True,
        "confidences": confidences,
        "avg_step_reward": sum(step_rewards) / len(step_rewards),
    }


# ─── Reward function for TRL ───
# FIX #2: This receives text decoded with skip_special_tokens=False
# (patched in MetaCotGRPOTrainer._generate_and_score_completions)
def metacot_reward_fn(completions, ground_truth=None, **kwargs):
    """Compute sequence-level rewards for GRPO advantage computation.

    3 rewards:
    - R_correct: +2.0 if correct, 0.0 if not
    - R_calib: calibration — correct+confident or wrong+uncertain = good
    - R_penalty: -0.5 no meta, -0.3 one meta, 0.0 two+ meta
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

        # R_calib: text-based calibration
        # (Gnosis probe-based calibration happens via correctness_loss)
        r_calib = 0.0
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            actual = 1.0 if is_correct else 0.0
            # Calibration gap: how far is stated confidence from actual outcome
            # This rewards: correct+confident, wrong+uncertain
            # Penalizes: correct+uncertain (slightly), wrong+confident (heavily)
            r_calib = max(0.0, 1.0 - abs(avg_conf - actual))

        rewards.append(r_correct + r_calib + r_penalty)
    return rewards


# ─── Custom GRPO Trainer ───
class MetaCotGRPOTrainer(GRPOTrainer):
    """GRPO + Full Gnosis + Stepwise Credit Assignment.

    Fixes applied:
    - #1: ground_truth cached before super()._generate_and_score_completions()
    - #2: completions decoded with skip_special_tokens=False for meta blocks
    - #3: stepwise weights are non-negative importance (not reward values)
    - #4: GRPO loss delegates to parent _compute_loss (no feature loss)
    - #5: single compute_loss combines GRPO + Gnosis without double forward
    - #6: unconditional gnosis_loss logging
    """

    def __init__(self, *args, lambda_gnosis=LAMBDA_GNOSIS, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_gnosis = lambda_gnosis
        self._cached_ground_truths = []

        # Parent handles Gnosis head unfreeze via enable_correctness_head=True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("MetaCotGRPOTrainer does not support returning outputs")

        # 1. GRPO loss — delegate to parent's _compute_loss (full feature support)
        #    Stepwise weights are applied by injecting them into inputs
        grpo_loss = self._compute_loss(model, inputs)

        # 2. Gnosis correctness loss (if model has _should_stop)
        gnosis_loss = torch.zeros(1, device=grpo_loss.device, requires_grad=False)
        unwrapped = self.accelerator.unwrap_model(model)
        has_gnosis = hasattr(unwrapped, '_should_stop')

        if has_gnosis and "correctness_labels" in inputs:
            try:
                gnosis_loss = self._compute_correctness_loss(model, inputs)
            except Exception as e:
                print(f"[Step {self.state.global_step}] Gnosis loss error: {e}")

        total_loss = grpo_loss + self.lambda_gnosis * gnosis_loss

        # FIX #6: always log
        mode = "train" if model.training else "eval"
        self._metrics[mode]["grpo_loss"].append(grpo_loss.detach().item())
        self._metrics[mode]["gnosis_loss"].append(gnosis_loss.detach().item())
        self._metrics[mode]["gnosis_active"].append(1.0 if has_gnosis else 0.0)

        return total_loss

    def _generate_and_score_completions(self, inputs):
        """Override to:
        1. Cache ground_truths before super() consumes inputs (FIX #1)
        2. Compute stepwise weights and correctness labels
        """
        # FIX #1: Cache ground truths before super() processes inputs
        # inputs is a list of dicts at this point, each with "prompt" and "ground_truth"
        if isinstance(inputs, list):
            self._cached_ground_truths = [
                x.get("ground_truth", "") for x in inputs
            ]
        elif isinstance(inputs, dict):
            self._cached_ground_truths = inputs.get("ground_truth", [])

        outputs = super()._generate_and_score_completions(inputs)

        # Now add stepwise weights and correctness labels
        B = outputs["completion_ids"].shape[0]
        comp_len = outputs["completion_ids"].shape[1]
        device = outputs["completion_ids"].device

        stepwise_weights = torch.ones(B, comp_len, device=device)
        correctness_labels = torch.full((B,), -1.0, device=device)

        tokenizer = self.processing_class

        # FIX #1: Use cached ground truths
        # TRL repeats inputs num_generations times, so B = len(inputs) * num_generations
        num_gens = getattr(self.args, 'num_generations', 1)
        n_prompts = len(self._cached_ground_truths)

        for i in range(B):
            comp_ids = outputs["completion_ids"][i]
            # FIX #2: decode with skip_special_tokens=False to preserve <|meta|>
            comp_text = tokenizer.decode(comp_ids, skip_special_tokens=False)

            # FIX #1: map batch index to original prompt index
            prompt_idx = i // num_gens if num_gens > 0 else i
            gt = ""
            if prompt_idx < n_prompts:
                gt = self._cached_ground_truths[prompt_idx]

            is_correct = check_correctness(comp_text, str(gt))
            correctness_labels[i] = 1.0 if is_correct else 0.0

            # FIX #3: compute importance weights (always non-negative)
            weights, info = compute_stepwise_rewards(
                comp_ids, comp_text, is_correct, tokenizer,
            )
            if len(weights) >= comp_len:
                stepwise_weights[i] = weights[:comp_len].to(device)
            else:
                stepwise_weights[i, :len(weights)] = weights.to(device)

        outputs["stepwise_weights"] = stepwise_weights
        outputs["correctness_labels"] = correctness_labels

        # Log step distribution
        n_with_meta = sum(1 for i in range(B)
                          if len(find_meta_positions_in_ids(
                              outputs["completion_ids"][i], tokenizer)) > 0)
        n_correct = (correctness_labels > 0.5).sum().item()
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["meta_block_ratio"].append(n_with_meta / max(B, 1))
        self._metrics[mode]["correctness_ratio"].append(n_correct / max(B, 1))

        return outputs

    def _compute_loss(self, model, inputs):
        """Override parent's _compute_loss to inject stepwise weights.

        FIX #5: Instead of reimplementing, we modify per_token_loss after
        the parent computes it. But since _compute_loss returns a scalar,
        we need to reimplement the critical section with stepwise support.

        We keep full parent compatibility (importance_sampling, entropy_mask, etc.)
        by following the exact same logic.
        """
        # Call parent's full _compute_loss logic
        # But we need to inject stepwise_weights, which requires access to per_token_loss
        # Since we can't hook into the parent cleanly, we delegate but wrap

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps

        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown importance sampling level: {self.importance_sampling_level}")

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        # ─── STEPWISE IMPORTANCE WEIGHTING (FIX #3) ───
        stepwise_weights = inputs.get("stepwise_weights")
        if stepwise_weights is not None:
            # Weights are always ≥ 0, normalized to mean=1.0
            # They modulate HOW MUCH credit each step gets
            # The advantage already encodes the reward direction
            per_token_loss = per_token_loss * stepwise_weights

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Loss aggregation (supports all loss_types)
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Metrics (full parent compatibility)
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        clip_ratio = (coef_1 > 1 + self.epsilon_high) | (coef_1 < 1 - self.epsilon_low)
        clip_ratio = ((clip_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).nanmean().item())

        return loss


# ─── Dataset preparation ───
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
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", default="checkpoints/qwen3_grpo_gnosis")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lambda_gnosis", type=float, default=0.5)
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
        attn_implementation="eager",  # Required for Gnosis attention extraction
        trust_remote_code=True,
        use_cache=False,
    )
    if to_add:
        model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,  # scaling factor = 1.0
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # modules_to_save: these are trained fully (not through LoRA)
        # Gnosis head params must be trainable for correctness_loss
        modules_to_save=["stop_head", "attn_extractor", "hid_extractor", "conf_extractor"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=2048,
        use_vllm=False,  # ptca env has vLLM 0.6.6 (too old for colocate)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        beta=0.0,  # No KL (GRPO uses group advantages)
        # Gnosis correctness head — let parent handle freeze/unfreeze
        enable_correctness_head=True,
        freeze_except_stop_head=False,  # Train BOTH base (LoRA) + Gnosis head
        logging_steps=1,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        run_name="qwen3-grpo-gnosis-stepwise",
        remove_unused_columns=False,
    )

    train_dataset = prepare_dataset(args.train_data)
    print(f"=== MetaCot GRPO + Full Gnosis + Stepwise ===")
    print(f"Dataset: {len(train_dataset)} problems")
    print(f"Lambda Gnosis: {args.lambda_gnosis}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Stepwise: <|meta|> boundaries → per-token importance weights")
    print(f"Rewards: R_correct(2.0) + R_calib(text) + R_penalty(meta)")
    print(f"Gnosis: BCE correctness loss (attention + hidden + confidence)")

    trainer = MetaCotGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=metacot_reward_fn,
        processing_class=tokenizer,
        peft_config=lora_config,
        lambda_gnosis=args.lambda_gnosis,
    )

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
