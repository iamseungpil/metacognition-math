"""veRL-based GDPO trainer for Meta-CoT.

This is the main entry point that replaces TRL's GRPOTrainer with veRL's
RayPPOTrainer, using GDPO advantage computation for multi-reward training.

Architecture:
  - MetaCotRewardManager: adapts veRL's DataProto to our existing reward
    function signatures (completions, ground_truth, **kwargs) -> List[float]
  - Returns per-reward tensors so that compute_gdpo_outcome_advantage can
    normalize each independently before combining.
  - Integrates with veRL's RayPPOTrainer via the reward_fn callback.

Usage:
  python src/training/verl_gdpo.py --config-name verl_gdpo_e13

Design:
  - ZERO modifications to src/training/rewards.py
  - ZERO modifications to veRL core files
  - New reward manager + new advantage function only
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Callable, List, Optional

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, compute_advantage
from verl.trainer.ppo import core_algos

# Import our reward functions (unchanged)
from src.training.rewards import (
    correctness_reward,
    structural_switch_reward,
    confidence_trajectory_reward,
    verify_outcome_reward,
)
from src.training.verl_gdpo_algos import compute_gdpo_outcome_advantage


# ---------------------------------------------------------------------------
# Reward configuration registry (mirrors grpo_v2.py but only for veRL modes)
# ---------------------------------------------------------------------------

REWARD_CONFIGS = {
    "E12": {
        "funcs": [correctness_reward, structural_switch_reward],
        "weights": [1.0, 0.3],
    },
    "E13": {
        "funcs": [correctness_reward, structural_switch_reward,
                  confidence_trajectory_reward, verify_outcome_reward],
        "weights": [1.0, 0.3, 0.3, 0.2],
    },
    # E20+: future experiments can be added here without touching any other file
}


# ---------------------------------------------------------------------------
# Reward adapter: veRL DataProto -> our reward function signature -> tensor
# ---------------------------------------------------------------------------

def _decode_response(tokenizer, prompt_ids, response_ids, attention_mask, prompt_length: int) -> str:
    """Decode a single response from veRL tensor format to text string."""
    valid_prompt_length = attention_mask[:prompt_length].sum().item()
    valid_prompt_ids = prompt_ids[-int(valid_prompt_length):]

    valid_response_length = attention_mask[prompt_length:].sum().item()
    valid_response_ids = response_ids[:int(valid_response_length)]

    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
    return tokenizer.decode(sequences, skip_special_tokens=False)


class MetaCotRewardManager:
    """Reward manager that calls our existing reward functions per-sample.

    Returns per-reward tensors (stored in batch) for GDPO advantage computation.
    The combined token_level_scores returned to veRL is the weighted sum
    (for logging and KL penalty), while per-reward tensors are stored
    separately for GDPO's per-reward normalization.

    Args:
        tokenizer: HuggingFace tokenizer for decoding responses.
        reward_funcs: List of reward functions with signature
            (completions, ground_truth=None, **kwargs) -> List[float].
        reward_weights: List of float weights for each reward function.
        num_examine: Number of samples to print for debugging.
    """

    def __init__(
        self,
        tokenizer,
        reward_funcs: List[Callable],
        reward_weights: List[float],
        num_examine: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights
        self.num_examine = num_examine

        # Validate
        assert len(reward_funcs) == len(reward_weights), (
            f"Mismatch: {len(reward_funcs)} funcs vs {len(reward_weights)} weights"
        )
        self._func_names = [fn.__name__ for fn in reward_funcs]
        print(f"[MetaCotRewardManager] {len(reward_funcs)} rewards: "
              f"{self._func_names} x {reward_weights}")

    def __call__(self, data: DataProto) -> torch.Tensor:
        """Compute rewards for a batch of rollout data.

        Args:
            data: veRL DataProto containing batch['prompts'], batch['responses'],
                  batch['attention_mask'], and non_tensor_batch with ground_truth.

        Returns:
            reward_tensor: (bs, response_length) with weighted-sum reward at EOS.
            Also stores data.batch['per_reward_token_level_scores'] as a list
            of (bs, response_length) tensors for GDPO.
        """
        # If rm_scores already exist (from a separate reward model), use them
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        bs = len(data)
        response_length = data.batch['responses'].shape[-1]
        prompt_length = data.batch['prompts'].shape[-1]

        # Decode all responses and collect ground truths
        decoded_responses: list[str] = []
        ground_truths: list[str] = []

        for i in range(bs):
            data_item = data[i]
            text = _decode_response(
                self.tokenizer,
                data_item.batch['prompts'],
                data_item.batch['responses'],
                data_item.batch['attention_mask'],
                prompt_length,
            )
            decoded_responses.append(text)

            # Ground truth from non_tensor_batch
            gt = data_item.non_tensor_batch.get('reward_model', {})
            if isinstance(gt, dict):
                gt = gt.get('ground_truth', '')
            ground_truths.append(str(gt))

        # Debug printing
        for i in range(min(self.num_examine, bs)):
            print(f"\n[Sample {i}] GT={ground_truths[i][:50]}")
            print(f"  Response: {decoded_responses[i][:200]}...")

        # Format as our reward functions expect: list of dicts with 'content' key
        completions = [[{"content": text}] for text in decoded_responses]

        # Call each reward function and collect per-reward scores
        per_reward_tensors: list[torch.Tensor] = []
        per_reward_scores_debug: dict[str, list[float]] = {}

        for func_idx, reward_fn in enumerate(self.reward_funcs):
            func_name = self._func_names[func_idx]
            try:
                scores = reward_fn(
                    completions=completions,
                    ground_truth=ground_truths,
                )
            except Exception as e:
                print(f"[WARNING] Reward function {func_name} failed: {e}")
                traceback.print_exc()
                scores = [0.0] * bs

            # Validate output length
            if len(scores) != bs:
                print(f"[WARNING] {func_name} returned {len(scores)} scores for {bs} samples")
                scores = (scores + [0.0] * bs)[:bs]

            # Place score at EOS token position (matching veRL convention)
            reward_tensor = torch.zeros(bs, response_length, dtype=torch.float32)
            for i in range(bs):
                valid_response_length = data[i].batch['attention_mask'][prompt_length:].sum().item()
                eos_pos = max(0, int(valid_response_length) - 1)
                reward_tensor[i, eos_pos] = float(scores[i])

            per_reward_tensors.append(reward_tensor)
            per_reward_scores_debug[func_name] = [float(s) for s in scores]

        # Log per-reward means
        for name, scores in per_reward_scores_debug.items():
            mean_score = np.mean(scores) if scores else 0.0
            print(f"  [{name}] mean={mean_score:.4f}")

        # Store per-reward tensors for GDPO advantage computation
        # We attach them to DataProto's meta_info (dict of arbitrary data)
        data.meta_info['per_reward_token_level_scores'] = per_reward_tensors
        data.meta_info['reward_weights'] = self.reward_weights

        # Combined reward tensor (weighted sum at EOS, for veRL's token_level_scores)
        weights = torch.tensor(self.reward_weights, dtype=torch.float32)
        combined = torch.zeros(bs, response_length, dtype=torch.float32)
        for func_idx, r_tensor in enumerate(per_reward_tensors):
            combined += r_tensor * weights[func_idx]

        return combined


# ---------------------------------------------------------------------------
# GDPO-aware advantage computation (drop-in replacement for compute_advantage)
# ---------------------------------------------------------------------------

def compute_advantage_gdpo(data: DataProto, adv_estimator: str, gamma=1.0, lam=1.0, num_repeat=1):
    """Extended advantage computation that supports 'gdpo' in addition to 'gae' and 'grpo'.

    For 'gdpo', uses per-reward tensors stored by MetaCotRewardManager
    to compute GDPO advantages. Falls back to veRL's original compute_advantage
    for 'gae' and 'grpo'.
    """
    if adv_estimator == 'gdpo':
        per_reward_tensors = data.meta_info.get('per_reward_token_level_scores')
        reward_weights = data.meta_info.get('reward_weights')

        if per_reward_tensors is None or reward_weights is None:
            raise ValueError(
                "GDPO requires per_reward_token_level_scores and reward_weights "
                "in data.meta_info. Ensure MetaCotRewardManager is used."
            )

        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        index = data.non_tensor_batch['uid']

        advantages, returns = compute_gdpo_outcome_advantage(
            per_reward_token_level_rewards=per_reward_tensors,
            reward_weights=reward_weights,
            eos_mask=response_mask,
            index=index,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

    elif adv_estimator in ('gae', 'grpo'):
        # Delegate to veRL's original implementation
        compute_advantage(data, adv_estimator=adv_estimator, gamma=gamma, lam=lam, num_repeat=num_repeat)

    else:
        raise NotImplementedError(f"Unknown adv_estimator: {adv_estimator}")

    return data


# ---------------------------------------------------------------------------
# Monkey-patch compute_advantage in ray_trainer to support 'gdpo'
# ---------------------------------------------------------------------------

def _patch_verl_for_gdpo():
    """Patch veRL's ray_trainer to support 'gdpo' advantage estimator.

    Two minimal patches:
    1. compute_advantage: adds 'gdpo' branch alongside 'gae' and 'grpo'.
    2. init_workers: allows 'gdpo' to skip critic creation (same as 'grpo').

    The original functions are preserved for 'gae' and 'grpo' modes.
    """
    import verl.trainer.ppo.ray_trainer as ray_trainer_module

    # Patch 1: compute_advantage
    ray_trainer_module.compute_advantage = compute_advantage_gdpo

    # Patch 2: init_workers critic check
    # The original init_workers raises NotImplementedError for unknown adv_estimators.
    # We wrap it to treat 'gdpo' like 'grpo' (no critic).
    original_init_workers = RayPPOTrainer.init_workers

    def patched_init_workers(self):
        # Temporarily set adv_estimator to 'grpo' for init_workers
        # so the critic creation check passes, then restore.
        actual_estimator = self.config.algorithm.adv_estimator
        if actual_estimator == 'gdpo':
            from omegaconf import open_dict
            with open_dict(self.config):
                self.config.algorithm.adv_estimator = 'grpo'
            original_init_workers(self)
            with open_dict(self.config):
                self.config.algorithm.adv_estimator = 'gdpo'
        else:
            original_init_workers(self)

    RayPPOTrainer.init_workers = patched_init_workers
    print("[verl_gdpo] Patched compute_advantage and init_workers for 'gdpo' estimator")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

import ray
import hydra


@hydra.main(config_path='../../configs', config_name='verl_gdpo_e13', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
            }
        })

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.single_controller.ray import RayWorkerGroup, RayClassWithInitArgs
    from pprint import pprint
    from omegaconf import OmegaConf

    # Resolve and print config
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Patch veRL to support 'gdpo' advantage estimator
    if config.algorithm.adv_estimator == 'gdpo':
        _patch_verl_for_gdpo()

    # Download model checkpoint (handles HDFS paths transparently)
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # Tokenizer setup with <|meta|> tokens
    tokenizer = hf_tokenizer(local_path)
    from src.metacot.prompt import META_START, META_END
    from src.training.tokenizer_utils import ensure_meta_tokens_not_special
    original_vocab_size = len(tokenizer)
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])
    if len(tokenizer) != original_vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab expanded from {original_vocab_size} to {len(tokenizer)}. "
            f"Meta tokens not found in checkpoint vocab. Use a checkpoint that already "
            f"has <|meta|> and <|/meta|> tokens (e.g., from SFT), or implement "
            f"embedding resize for veRL FSDP workers."
        )
    print(f"[verl_gdpo] Tokenizer vocab size: {len(tokenizer)} (unchanged)")

    # Resolve reward configuration from mode
    mode = config.get('mode', 'E13')
    if mode not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown mode '{mode}'. Available: {list(REWARD_CONFIGS.keys())}"
        )
    reward_cfg = REWARD_CONFIGS[mode]
    print(f"[verl_gdpo] Mode={mode}, Rewards={[f.__name__ for f in reward_cfg['funcs']]}")

    # Build reward manager
    reward_fn = MetaCotRewardManager(
        tokenizer=tokenizer,
        reward_funcs=reward_cfg['funcs'],
        reward_weights=reward_cfg['weights'],
        num_examine=config.get('num_examine', 0),
    )
    val_reward_fn = MetaCotRewardManager(
        tokenizer=tokenizer,
        reward_funcs=reward_cfg['funcs'],
        reward_weights=reward_cfg['weights'],
        num_examine=1,
    )

    # Worker classes (FSDP backend)
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")

    # Role -> worker mapping
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    # Resource pool (all GPUs in one pool for hybrid engine)
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # Optional reward model (disabled by default for rule-based rewards)
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )

    # Build trainer
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    trainer.fit()
    print("[verl_gdpo] Training complete.")


if __name__ == '__main__':
    main()
