"""Meta-CoT GRPO v2: full FT + GDPO + decomposed control rewards.

Key design:
  - Full fine-tuning (NO LoRA)
  - GDPO monkey-patch: per-reward normalization before summing
  - Reward ablations via --mode: E1-E10, including E9/E9b/E9c decompositions
  - Calibration and confidence-revision rewards remain active before full control rewards
  - Probe-aligned local-calibration experiments are isolated in E6/E7
  - Explicit control behaviors are decomposed into verify-only / redirect-only /
    diagnosis-only runs before the full controller E10
  - Response samples and reward logs are saved for later qualitative analysis

Usage:
  accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_v2.py --mode E10 --max_steps 1000 \
    --model_path checkpoints/qwen3_metacot_control_v5_all_sft --data mixed_train
"""
import argparse
from functools import partial
import importlib.util
import importlib.machinery
import json
import os
import re
import sys
import types

import numpy as np
import pandas as pd
import torch

# Prevent FSDP import error
import torch.distributed.fsdp as _fsdp_mod
if not hasattr(_fsdp_mod, "FSDPModule"):
    _fsdp_mod.FSDPModule = type("FSDPModule", (), {})


def _ensure_vllm_stub():
    """Provide a minimal vLLM stub for TRL import when vLLM is unavailable.

    Newer TRL versions import `trl.extras.vllm_client` at module import time even
    when `use_vllm=False`. Our runtime does not install vLLM, so we create the
    exact modules imported by TRL to keep the non-vLLM path usable.
    """
    if "vllm" in sys.modules:
        return

    vllm_mod = types.ModuleType("vllm")
    distributed_mod = types.ModuleType("vllm.distributed")
    device_comms_mod = types.ModuleType("vllm.distributed.device_communicators")
    pynccl_mod = types.ModuleType("vllm.distributed.device_communicators.pynccl")
    utils_mod = types.ModuleType("vllm.distributed.utils")
    sampling_params_mod = types.ModuleType("vllm.sampling_params")
    vllm_ascend_mod = types.ModuleType("vllm_ascend")
    vllm_ascend_distributed_mod = types.ModuleType("vllm_ascend.distributed")
    vllm_ascend_device_mod = types.ModuleType("vllm_ascend.distributed.device_communicators")
    vllm_ascend_pyhccl_mod = types.ModuleType("vllm_ascend.distributed.device_communicators.pyhccl")

    def _package_spec(name: str):
        spec = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        if spec is not None and spec.submodule_search_locations is None:
            spec.submodule_search_locations = []
        return spec

    def _module_spec(name: str):
        return importlib.machinery.ModuleSpec(name, loader=None)

    vllm_mod.__spec__ = _package_spec("vllm")
    vllm_mod.__path__ = []
    vllm_mod.__package__ = "vllm"
    distributed_mod.__spec__ = _package_spec("vllm.distributed")
    distributed_mod.__path__ = []
    distributed_mod.__package__ = "vllm.distributed"
    device_comms_mod.__spec__ = _package_spec("vllm.distributed.device_communicators")
    device_comms_mod.__path__ = []
    device_comms_mod.__package__ = "vllm.distributed.device_communicators"
    pynccl_mod.__spec__ = _module_spec("vllm.distributed.device_communicators.pynccl")
    pynccl_mod.__package__ = "vllm.distributed.device_communicators"
    utils_mod.__spec__ = _module_spec("vllm.distributed.utils")
    utils_mod.__package__ = "vllm.distributed"
    sampling_params_mod.__spec__ = _module_spec("vllm.sampling_params")
    sampling_params_mod.__package__ = "vllm"
    vllm_ascend_mod.__spec__ = _package_spec("vllm_ascend")
    vllm_ascend_mod.__path__ = []
    vllm_ascend_mod.__package__ = "vllm_ascend"
    vllm_ascend_distributed_mod.__spec__ = _package_spec("vllm_ascend.distributed")
    vllm_ascend_distributed_mod.__path__ = []
    vllm_ascend_distributed_mod.__package__ = "vllm_ascend.distributed"
    vllm_ascend_device_mod.__spec__ = _package_spec("vllm_ascend.distributed.device_communicators")
    vllm_ascend_device_mod.__path__ = []
    vllm_ascend_device_mod.__package__ = "vllm_ascend.distributed.device_communicators"
    vllm_ascend_pyhccl_mod.__spec__ = _module_spec("vllm_ascend.distributed.device_communicators.pyhccl")
    vllm_ascend_pyhccl_mod.__package__ = "vllm_ascend.distributed.device_communicators"

    class _DummyPyNcclCommunicator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("vLLM communicator is unavailable in this runtime")

    class _DummyStatelessProcessGroup:
        @classmethod
        def create(cls, *args, **kwargs):
            raise RuntimeError("vLLM process group is unavailable in this runtime")

    class _DummyLLM:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("vLLM is unavailable in this runtime")

    class _DummySamplingParams:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("vLLM is unavailable in this runtime")

    class _DummyGuidedDecodingParams:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("vLLM guided decoding is unavailable in this runtime")

    pynccl_mod.PyNcclCommunicator = _DummyPyNcclCommunicator
    utils_mod.StatelessProcessGroup = _DummyStatelessProcessGroup
    vllm_ascend_pyhccl_mod.PyHcclCommunicator = _DummyPyNcclCommunicator
    vllm_mod.LLM = _DummyLLM
    vllm_mod.SamplingParams = _DummySamplingParams
    vllm_mod.sampling_params = sampling_params_mod
    sampling_params_mod.GuidedDecodingParams = _DummyGuidedDecodingParams
    sampling_params_mod.SamplingParams = _DummySamplingParams

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.distributed"] = distributed_mod
    sys.modules["vllm.distributed.device_communicators"] = device_comms_mod
    sys.modules["vllm.distributed.device_communicators.pynccl"] = pynccl_mod
    sys.modules["vllm.distributed.utils"] = utils_mod
    sys.modules["vllm.sampling_params"] = sampling_params_mod
    sys.modules["vllm_ascend"] = vllm_ascend_mod
    sys.modules["vllm_ascend.distributed"] = vllm_ascend_distributed_mod
    sys.modules["vllm_ascend.distributed.device_communicators"] = vllm_ascend_device_mod
    sys.modules["vllm_ascend.distributed.device_communicators.pyhccl"] = vllm_ascend_pyhccl_mod


_ensure_vllm_stub()


def _ensure_mergekit_stub():
    """Provide a minimal mergekit stub for TRL import when mergekit is unavailable."""
    if "mergekit" in sys.modules:
        return

    mergekit_mod = types.ModuleType("mergekit")
    config_mod = types.ModuleType("mergekit.config")
    merge_mod = types.ModuleType("mergekit.merge")

    mergekit_mod.__spec__ = importlib.machinery.ModuleSpec("mergekit", loader=None)
    config_mod.__spec__ = importlib.machinery.ModuleSpec("mergekit.config", loader=None)
    merge_mod.__spec__ = importlib.machinery.ModuleSpec("mergekit.merge", loader=None)

    class _DummyMergeConfiguration:
        @classmethod
        def model_validate(cls, *args, **kwargs):
            raise RuntimeError("mergekit is unavailable in this runtime")

    class _DummyMergeOptions:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("mergekit is unavailable in this runtime")

    def _dummy_run_merge(*args, **kwargs):
        raise RuntimeError("mergekit is unavailable in this runtime")

    config_mod.MergeConfiguration = _DummyMergeConfiguration
    merge_mod.MergeOptions = _DummyMergeOptions
    merge_mod.run_merge = _dummy_run_merge

    sys.modules["mergekit"] = mergekit_mod
    sys.modules["mergekit.config"] = config_mod
    sys.modules["mergekit.merge"] = merge_mod


_ensure_mergekit_stub()


def _ensure_llm_blender_stub():
    """Provide a minimal llm_blender stub for TRL import when unavailable."""
    if "llm_blender" in sys.modules:
        return

    blender_mod = types.ModuleType("llm_blender")
    blender_mod.__spec__ = importlib.machinery.ModuleSpec("llm_blender", loader=None)

    class _DummyBlender:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("llm_blender is unavailable in this runtime")

    blender_mod.Blender = _DummyBlender
    sys.modules["llm_blender"] = blender_mod


_ensure_llm_blender_stub()

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from src.training.rewards import (
    correctness_reward, format_reward, meta_quality_reward,
    calibration_reward, uncertainty_meta_reward,
    stepwise_trajectory_reward, probe_calibration_reward,
    stepwise_probe_reward, length_penalty_reward, correct_meta_reward,
    overconfidence_penalty_reward,
    confidence_revision_reward, effective_verification_reward,
    effective_redirection_reward, diagnosis_reward, decomposition_reward,
    anomaly_notice_reward, repeated_intervention_reward, overconfidence_verify_reward,
    # V2 rewards (2026-04-04): address verify-repetition, redirect-execution, coverage-escape
    same_route_repetition_penalty, route_switch_evidence_reward, confidence_omission_floor,
    # V6.1 rewards (2026-04-05): structural switch + Brier calibration + verify outcome
    structural_switch_reward, brier_calibration_reward, verify_outcome_reward, efficiency_bonus_reward,
    confidence_trajectory_reward,
    # V8 E21 rewards: soft switch v2 + verify v2
    structural_switch_reward_v2, verify_outcome_v2,
    confidence_revision_reward_v2, redirect_execution_reward_v2, verify_execution_reward_v2,
)
from src.training.tokenizer_utils import ensure_meta_tokens_not_special


# ─── GDPO Monkey-Patch ───

def _apply_gdpo_patch():
    """Patch GRPOTrainer to use GDPO advantage computation.

    GRPO: sum(rewards) → group_normalize  (collapses distinct reward combos)
    GDPO: group_normalize(each_reward) → sum → batch_normalize  (preserves signal)

    Reference: arXiv:2601.05242 (NVIDIA NVlabs)
    """
    import trl.trainer.grpo_trainer as grpo_module

    original_method = GRPOTrainer._generate_and_score_completions

    def patched_method(self, inputs):
        # Call original to get all data
        result = original_method(self, inputs)

        # Only patch if we have multiple reward functions
        if not hasattr(self, '_gdpo_enabled') or not self._gdpo_enabled:
            return result

        # Re-compute advantages with GDPO
        # Access rewards_per_func from the stored attribute
        if hasattr(self, '_last_rewards_per_func') and self._last_rewards_per_func is not None:
            rewards_per_func = self._last_rewards_per_func
            device = rewards_per_func.device
            num_gen = self.num_generations

            all_adv = []
            for i in range(rewards_per_func.shape[1]):
                r_i = torch.nan_to_num(rewards_per_func[:, i])
                mean_i = r_i.view(-1, num_gen).mean(dim=1)
                std_i = r_i.view(-1, num_gen).std(dim=1)
                mean_i = mean_i.repeat_interleave(num_gen, dim=0)
                std_i = std_i.repeat_interleave(num_gen, dim=0)
                adv_i = (r_i - mean_i) / (std_i + 1e-4)
                all_adv.append(adv_i)

            combined = torch.stack(all_adv, dim=1)
            weights = self.reward_weights.to(device).unsqueeze(0)
            pre_bn = (combined * weights).nansum(dim=1)
            advantages = (pre_bn - pre_bn.mean()) / (pre_bn.std() + 1e-4)

            # _last_rewards_per_func is LOCAL (per-process) from _calculate_rewards,
            # so advantages computed from it is already local-sized. No slicing needed.
            result["advantages"] = advantages

        return result

    # Also patch _calculate_rewards to store rewards_per_func
    original_calc = GRPOTrainer._calculate_rewards

    def patched_calc(self, inputs, prompts, completions, completion_ids_list):
        result = original_calc(self, inputs, prompts, completions, completion_ids_list)
        self._last_rewards_per_func = result.clone()
        return result

    GRPOTrainer._generate_and_score_completions = patched_method
    GRPOTrainer._calculate_rewards = patched_calc


# ─── Data Loading ───

def load_filtered(path):
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        prompt = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"]
        gt = json.loads(row["reward_model"]) if isinstance(row.get("reward_model"), str) else row.get("reward_model", {})
        records.append({"prompt": prompt, "ground_truth": gt.get("ground_truth", "")})
    return Dataset.from_list(records)


def load_gsm8k(max_n=500):
    from datasets import load_dataset as hf_load
    ds = hf_load("openai/gsm8k", "main", split="train")
    records = []
    for row in ds:
        if len(records) >= max_n:
            break
        ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        records.append({"prompt": [{"role": "user", "content": row["question"]}], "ground_truth": ans})
    return Dataset.from_list(records)


def load_mixed(gsm_n=500, math_n=500):
    """Load GSM8K + MATH-500 for diverse difficulty GRPO training."""
    from datasets import load_dataset as hf_load
    records = []
    # GSM8K (easy-medium)
    ds = hf_load("openai/gsm8k", "main", split="train")
    for row in ds:
        if len(records) >= gsm_n:
            break
        ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        records.append({"prompt": [{"role": "user", "content": row["question"]}], "ground_truth": ans})
    gsm_count = len(records)
    # MATH (medium-hard)
    try:
        ds = hf_load("HuggingFaceH4/MATH-500", split="test")
        for row in ds:
            if len(records) - gsm_count >= math_n:
                break
            records.append({"prompt": [{"role": "user", "content": row["problem"]}], "ground_truth": row["answer"]})
    except Exception as e:
        print(f"MATH-500 load failed: {e}, using GSM8K only")
    print(f"Mixed dataset: {gsm_count} GSM8K + {len(records)-gsm_count} MATH = {len(records)} total")
    import random
    random.shuffle(records)
    return Dataset.from_list(records)


def _extract_math_answer(row):
    """Prefer a clean final answer rather than a full worked solution."""
    answer = row.get("answer")
    if answer:
        return str(answer)

    solution = str(row.get("solution", ""))
    boxed = re.findall(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', solution)
    if boxed:
        return boxed[-1].strip()
    return solution


def load_mixed_train(gsm_n=500, math_n=500):
    """Load train-only math data for RL to avoid benchmark test leakage."""
    from datasets import load_dataset as hf_load

    records = []
    ds = hf_load("openai/gsm8k", "main", split="train")
    for row in ds:
        if len(records) >= gsm_n:
            break
        ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        records.append({"prompt": [{"role": "user", "content": row["question"]}], "ground_truth": ans})
    gsm_count = len(records)

    math_rows = []
    math_configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    for cfg in math_configs:
        ds = hf_load("EleutherAI/hendrycks_math", cfg, split="train")
        for row in ds:
            gt = _extract_math_answer(row)
            if not gt:
                continue
            math_rows.append({"prompt": [{"role": "user", "content": row["problem"]}], "ground_truth": gt})

    import random
    random.shuffle(math_rows)
    records.extend(math_rows[:math_n])
    print(
        f"Mixed train dataset: {gsm_count} GSM8K train + "
        f"{len(records)-gsm_count} hendrycks_math train = {len(records)} total"
    )
    random.shuffle(records)
    return Dataset.from_list(records)


# ─── Sample Saving Callback ───

class SampleSaver:
    """Save completion samples every N steps for qualitative analysis."""

    def __init__(self, output_dir, every_n=50):
        self.output_dir = output_dir
        self.every_n = every_n
        self.samples = []
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

    def maybe_save(self, step, completions, prompts, rewards):
        if step % self.every_n != 0 or step == 0:
            return
        samples = []
        for i in range(min(5, len(completions))):
            text = completions[i][0]["content"] if isinstance(completions[i], list) else str(completions[i])
            samples.append({
                "step": step,
                "prompt": str(prompts[i])[:200] if i < len(prompts) else "",
                "completion": text[:1000],
                "reward": float(rewards[i]) if i < len(rewards) else None,
            })
        path = os.path.join(self.output_dir, "samples", f"step_{step:04d}.json")
        with open(path, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E9b", "E9c", "E10", "E9v2", "E9bv2", "E10v2", "E12", "E13", "E14", "E21", "E21R"], default="E1")
    parser.add_argument("--model_path", default="checkpoints/qwen3_meta_sft")
    parser.add_argument("--data", choices=["gsm8k", "filtered", "mixed", "mixed_train"], default="mixed")
    parser.add_argument("--data_path", default="verl_train_filtered.parquet")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--probe_path", default=os.environ.get("METACOG_PROBE_PATH", "checkpoints/simple_probe_qwen3/best_probe.pt"))
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"checkpoints/grpo_v2_{args.mode}"

    os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "metacot-math")

    # Reward functions that need model/tokenizer context are wired after model load.
    reward_configs = {
        "E1": ([correctness_reward, format_reward], [1.0, 0.5]),
        "E2": ([correctness_reward, format_reward, meta_quality_reward], [1.0, 0.5, 1.0]),
        "E3": ([correctness_reward, format_reward, meta_quality_reward, calibration_reward], [1.0, 0.5, 1.0, 0.5]),
        "E4": ([correctness_reward, format_reward, meta_quality_reward, calibration_reward, uncertainty_meta_reward],
               [1.0, 0.5, 1.0, 0.5, 0.5]),
        # E5: E3 + confidence revision only.
        # Tests whether the model can lower confidence appropriately around
        # anomaly/conflict signals before explicit behavior rewards are added.
        "E5": ([correctness_reward, format_reward, meta_quality_reward,
                calibration_reward, confidence_revision_reward],
               [1.0, 0.3, 0.4, 0.5, 0.9]),
        # E6: E3 + probe calibration only.
        "E6": ([correctness_reward, format_reward, meta_quality_reward,
                calibration_reward, probe_calibration_reward],
               [1.0, 0.3, 0.4, 0.5, 1.1]),
        # E7: E6 + blockwise stepwise scoring.
        "E7": ([correctness_reward, format_reward, meta_quality_reward,
                calibration_reward, probe_calibration_reward, stepwise_probe_reward],
               [1.0, 0.3, 0.4, 0.5, 0.9, 1.1]),
        # E8: stronger calibration / overconfidence shaping.
        # E5 + stronger anti-overconfidence shaping, still no explicit behavior rewards.
        "E8": ([correctness_reward, format_reward, correct_meta_reward,
                calibration_reward, confidence_revision_reward, overconfidence_penalty_reward,
                length_penalty_reward],
               [3.0, 0.2, 0.4, 0.5, 0.8, 1.0, 1.0]),
        # E9: verify-only behavior reward on top of E8.
        "E9": ([correctness_reward, format_reward, correct_meta_reward,
                calibration_reward, confidence_revision_reward, overconfidence_penalty_reward,
                length_penalty_reward, effective_verification_reward, overconfidence_verify_reward],
               [3.0, 0.2, 0.3, 0.4, 0.6, 1.0, 1.0, 0.9, 0.9]),
        # E9b: redirect-only behavior reward on top of E8.
        "E9b": ([correctness_reward, format_reward, correct_meta_reward,
                calibration_reward, confidence_revision_reward, overconfidence_penalty_reward,
                length_penalty_reward, effective_redirection_reward],
               [3.0, 0.2, 0.3, 0.4, 0.6, 1.0, 1.0, 1.0]),
        # E9c: diagnosis/decomposition-only behavior reward on top of E8.
        "E9c": ([correctness_reward, format_reward, correct_meta_reward,
                calibration_reward, confidence_revision_reward, overconfidence_penalty_reward,
                length_penalty_reward, diagnosis_reward, decomposition_reward],
               [3.0, 0.2, 0.3, 0.4, 0.6, 1.0, 1.0, 0.8, 0.8]),
        # E10: full combined controller.
        # Adds verify, redirect, diagnosis, anomaly, and repeated intervention on top of E8.
        "E10": ([correctness_reward, format_reward, correct_meta_reward,
                 calibration_reward, confidence_revision_reward, overconfidence_penalty_reward, length_penalty_reward,
                 effective_verification_reward, effective_redirection_reward,
                 diagnosis_reward, decomposition_reward,
                 anomaly_notice_reward, repeated_intervention_reward, overconfidence_verify_reward],
                [3.0, 0.2, 0.3, 0.4, 0.6, 1.0, 1.0, 0.8, 1.0, 0.6, 0.6, 0.4, 0.5, 1.0]),
        # ── V2 experiments (2026-04-04): address verify-repetition, redirect-execution, coverage-escape ──
        # E9v2: E9 + same-route repetition penalty + coverage floor.
        # Intent: verify must use independent method, not repeat same calculation.
        "E9v2": ([correctness_reward, format_reward, correct_meta_reward,
                  calibration_reward, confidence_revision_reward, overconfidence_penalty_reward,
                  length_penalty_reward, effective_verification_reward, overconfidence_verify_reward,
                  same_route_repetition_penalty, confidence_omission_floor],
                 [3.0, 0.2, 0.3, 0.4, 0.6, 1.0, 1.0, 0.9, 0.9, 0.5, 0.5]),
        # E9bv2: E9b + route-switch evidence + coverage floor.
        # Intent: redirect must show structural method difference in solve tail.
        "E9bv2": ([correctness_reward, format_reward, correct_meta_reward,
                   calibration_reward, confidence_revision_reward, overconfidence_penalty_reward,
                   length_penalty_reward, effective_redirection_reward,
                   route_switch_evidence_reward, confidence_omission_floor],
                  [3.0, 0.2, 0.3, 0.4, 0.6, 1.0, 1.0, 1.0, 0.6, 0.5]),
        # E10v2: full controller + all V2 fixes.
        # Intent: combined verify quality + redirect execution + mandatory meta emission.
        "E10v2": ([correctness_reward, format_reward, correct_meta_reward,
                   calibration_reward, confidence_revision_reward, overconfidence_penalty_reward,
                   length_penalty_reward,
                   effective_verification_reward, effective_redirection_reward,
                   diagnosis_reward, decomposition_reward,
                   anomaly_notice_reward, repeated_intervention_reward, overconfidence_verify_reward,
                   same_route_repetition_penalty, route_switch_evidence_reward, confidence_omission_floor],
                  [3.0, 0.2, 0.3, 0.4, 0.6, 1.0, 1.0,
                   0.8, 1.0, 0.6, 0.6, 0.4, 0.5, 1.0,
                   0.5, 0.6, 0.5]),
        # V6.1 E12: correctness + structural switch (2 rewards only)
        "E12": ([correctness_reward, structural_switch_reward],
                [1.0, 0.3]),
        # V6.2 E13: correctness + switch + confidence trajectory + verify
        "E13": ([correctness_reward, structural_switch_reward,
                 confidence_trajectory_reward, verify_outcome_reward],
                [1.0, 0.3, 0.3, 0.2]),
        # V6.3 E14: Enhanced E9/E13 — verify-first + meta floor + dampened calibration
        # Grounded in: E9 (verify=best signal), E5/E6/E8 (calibration suppresses meta),
        # E9v2 (confidence_omission_floor prevents meta collapse)
        "E14": ([correctness_reward, structural_switch_reward,
                 verify_outcome_reward, confidence_trajectory_reward,
                 confidence_omission_floor],
                [1.0, 0.3, 0.3, 0.15, 0.5]),
        # V8 E21: soft switch v2 + verify v2 + meta floor
        # Grounded in: E20a V8 SFT success (meta 99.7%), switch 0.4% needs RL
        # switch_v2: soft score, gated on meta, verify soft mult
        # verify_v2: template penalty, computation bonus
        "E21": ([correctness_reward, structural_switch_reward_v2,
                 verify_outcome_v2, confidence_trajectory_reward,
                 confidence_omission_floor],
                [1.0, 0.15, 0.3, 0.15, 0.5]),
        "E21R": ([correctness_reward, confidence_revision_reward_v2,
                  redirect_execution_reward_v2, verify_execution_reward_v2,
                  confidence_omission_floor],
                 [1.0, 0.35, 0.30, 0.15, 0.5]),
    }
    # ─── Model (Full FT, NO LoRA) ───
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CRITICAL FIX: Ensure <|meta|> tokens are NOT marked as special tokens.
    # TRL's skip_special_tokens=True strips special tokens before reward functions.
    # If SFT checkpoint saved them as special tokens, we must demote them to regular.
    from src.metacot.prompt import META_START, META_END
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True, use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    reward_funcs, reward_weights = reward_configs[args.mode]
    if args.mode in ("E6", "E7"):
        from src.training.rewards import _load_probe_head
        _probe_hidden_dim = model.config.hidden_size
        _probe_device = next(model.parameters()).device
        _probe_head = _load_probe_head(_probe_hidden_dim, _probe_device, args.probe_path)
        if _probe_head is None:
            raise RuntimeError(
                f"E6/E7 requires a valid probe checkpoint at {args.probe_path} "
                f"(hidden_dim={_probe_hidden_dim})"
            )
        print(f"Probe loaded: hidden_dim={_probe_hidden_dim}, device={_probe_device}")

        contextual_reward_funcs = []
        for fn in reward_funcs:
            if fn in (probe_calibration_reward, stepwise_probe_reward):
                wrapped = partial(
                    fn,
                    model=model,
                    tokenizer=tokenizer,
                    probe_head=_probe_head,
                )
                wrapped.__name__ = fn.__name__
                contextual_reward_funcs.append(wrapped)
            else:
                contextual_reward_funcs.append(fn)
        reward_funcs = contextual_reward_funcs

    use_gdpo = args.mode in ("E3", "E4", "E5", "E6", "E7", "E8", "E9", "E9b", "E9c", "E10", "E9v2", "E9bv2", "E10v2", "E12", "E13", "E14", "E21", "E21R")  # GDPO when 2+ rewards

    if use_gdpo:
        _apply_gdpo_patch()
        print("GDPO patch applied (per-reward normalization)")

    # ─── Data ───
    if args.data == "gsm8k":
        dataset = load_gsm8k()
    elif args.data == "mixed_train":
        dataset = load_mixed_train()
    elif args.data == "mixed":
        dataset = load_mixed()
    else:
        dataset = load_filtered(args.data_path)

    # ─── Config ───
    run_name = os.environ.get("METACOG_RUN_NAME") or os.environ.get("WANDB_NAME") or f"grpo-v2-{args.mode}-{args.max_steps}s"
    print(f"=== GRPO v2: {args.mode} ===")
    print(f"Rewards: {[f.__name__ for f in reward_funcs]} × {reward_weights}")
    print(f"GDPO: {use_gdpo}")
    print(f"Full FT (no LoRA)")
    print(f"Dataset: {len(dataset)} problems")
    if args.mode in ("E6", "E7"):
        print(f"Probe path: {args.probe_path} (pre-loaded)")

    # Config based on Open-R1 patterns, adapted for 4xA100 80GB
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=0.9,
        # HF generate (no vLLM, no veRL — just TRL)
        use_vllm=False,
        # Batch: 4 GPU × 1 batch × 4 accum = 16, 16/4 gen = 4 unique prompts
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Loss: dr_grpo (Open-R1 default, no length bias)
        loss_type="dr_grpo",
        beta=0.04,
        scale_rewards=False,
        num_iterations=2,  # >1 for non-zero loss display
        # Logging
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        reward_weights=reward_weights,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        # NO peft_config → Full FT
    )

    # Enable GDPO flag
    if use_gdpo:
        trainer._gdpo_enabled = True
        trainer._last_rewards_per_func = None

    # Response logging callback
    from transformers import TrainerCallback
    class ResponseLogger(TrainerCallback):
        def __init__(self, output_dir):
            self.output_dir = output_dir
            os.makedirs(os.path.join(output_dir, "responses"), exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            step = state.global_step
            if step % 10 == 0 and logs:
                log_path = os.path.join(self.output_dir, "responses", f"step_{step:04d}.json")
                with open(log_path, "w") as f:
                    json.dump({"step": step, "logs": {k: str(v) for k, v in logs.items()}}, f, indent=2)
                reward = logs.get('reward', '?')
                corr = logs.get('rewards/correctness_reward/mean', '?')
                length = logs.get('completions/mean_length', '?')
                print(f"  [Step {step}] reward={reward}, corr={corr}, len={length}, loss={logs.get('loss', '?')}")

    trainer.add_callback(ResponseLogger(args.output_dir))

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Done. Saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
