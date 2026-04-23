"""veRL-native Shared-preserve SDC trainer (verl 0.7.1 compatible).

Preserves the original SDC intent:
  - scalar/group advantage via GDPO reward heads (correctness, outcome_calibration,
    meta_structure, meta_commit_shape, postmeta_closure)
  - token-wise credit shaped by teacher T+ / T- log-probs on meta/postmeta regions
  - free-text `confidence:` fallback detection (see feedback_reward_fallback)

Refactor notes (2026-04-20):
  verl 0.7.1 removed the `reward_fn`/`val_reward_fn` kwargs from
  `RayPPOTrainer.__init__`.  Reward is now routed through either the
  `RewardLoopManager` (async workers) or `config.reward.custom_reward_function`.
  To keep the SDC-specific reward+side-effect pipeline intact (meta masks,
  reward_extra_infos, teacher signals), we use a thin subclass
  `SDCRayPPOTrainer` that (1) accepts the legacy kwargs and (2) overrides
  `_compute_reward_colocate` to call our in-process reward manager.  This is
  the minimum change that preserves the intent while adopting the 0.7.1
  initialization contract (processor, train_dataset, val_dataset, collate_fn,
  train_sampler).
"""
from __future__ import annotations

import os
import traceback
from typing import Callable, List

import numpy as np
import ray
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

from src.training.rewards import (
    correctness_reward,
    meta_commit_shape_reward,
    meta_structure_reward,
    outcome_calibration_reward,
)
from src.training._decoy_utils import _rule_based_decoy
from src.training.verl_sdc_utils import (
    build_sdc_region_masks,
    compute_sdc_gdpo_advantage,
    postmeta_closure_reward,
)


REWARD_CONFIGS = {
    "SDC_SHARED": {
        "funcs": [
            correctness_reward,
            outcome_calibration_reward,
            meta_structure_reward,
            meta_commit_shape_reward,
            postmeta_closure_reward,
        ],
        "weights": [1.0, 0.7, 0.25, 0.35, 0.45],
        "keys": [
            "correctness",
            "outcome_calibration",
            "meta_structure",
            "meta_commit_shape",
            "postmeta_closure",
        ],
    },
}

_ACTIVE_SDC_CONTEXT = {"trainer": None, "tokenizer": None}


def reward_loop_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Fallback scalar score for veRL agent-loop reward workers.

    veRL 0.7.1's async agent loop tries to stream a per-sample reward during
    generation whenever reward_loop workers exist. SDC training does not use
    that path for optimization; the actual training reward is computed later by
    `MetaCotSDCRewardManager` inside `_compute_reward_colocate`.

    We still provide a lightweight score here so the async generation path does
    not fall back to `default_compute_score`, which does not recognize our
    mixed `hendrycks_math/*` sources and crashes before PPO can start.
    """
    completion = [[{"content": solution_str}]]
    gt = [ground_truth]

    try:
        correctness = float(correctness_reward(completion, gt)[0])
    except Exception:
        correctness = 0.0

    try:
        calibration = float(outcome_calibration_reward(completion, gt)[0])
    except Exception:
        calibration = 0.0

    try:
        meta_structure = float(meta_structure_reward(completion, gt)[0])
    except Exception:
        meta_structure = 0.0

    try:
        meta_commit = float(meta_commit_shape_reward(completion, gt)[0])
    except Exception:
        meta_commit = 0.0

    try:
        closure = float(postmeta_closure_reward(completion, gt)[0])
    except Exception:
        closure = 0.0

    score = (
        correctness * 1.0
        + calibration * 0.7
        + meta_structure * 0.25
        + meta_commit * 0.35
        + closure * 0.45
    )
    return {
        "score": float(score),
        "correctness": correctness,
        "outcome_calibration": calibration,
        "meta_structure": meta_structure,
        "meta_commit_shape": meta_commit,
        "postmeta_closure": closure,
        "data_source": data_source or "",
    }


def _is_gdpo_estimator(adv_estimator) -> bool:
    try:
        from verl.trainer.ppo.core_algos import AdvantageEstimator
    except Exception:
        AdvantageEstimator = None

    if adv_estimator == "gdpo":
        return True
    if AdvantageEstimator is not None and adv_estimator == AdvantageEstimator.GDPO:
        return True
    return False


def _decode_response(tokenizer, prompt_ids, response_ids, attention_mask, prompt_length: int) -> tuple[str, torch.Tensor]:
    # Decode ONLY the response tokens — never the prompt.
    # Why: reward heads pattern-match on \boxed{}, <|meta|>, "the answer is", etc.
    # If the prompt contains any such substring (few-shot example, retrieved
    # problem text, template boilerplate), returning prompt+response here leaks
    # that content into every reward and silently inflates/deflates signals.
    valid_response_length = attention_mask[prompt_length:].sum().item()
    valid_response_ids = response_ids[: int(valid_response_length)]
    text = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
    return text, valid_response_ids


def _decode_prompt_only(tokenizer, prompt_ids, attention_mask, prompt_length: int) -> str:
    valid_prompt_length = attention_mask[:prompt_length].sum().item()
    valid_prompt_ids = prompt_ids[-int(valid_prompt_length):]
    return tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)


def _build_teacher_logprob_batch(
    *,
    tokenizer,
    prompt_texts: list[str],
    answer_texts: list[str],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
):
    prompt_ids_list = []
    seq_lens = []
    for prompt_text, answer_text in zip(prompt_texts, answer_texts):
        # Align teacher conditioning with what the actor actually sees:
        # prompt_text is already the chat-templated prompt (ending in the
        # assistant role marker), so we append the gold/decoy answer directly
        # instead of injecting a synthetic " Answer: " separator the actor
        # never produces. This keeps teacher log-prob on the same conditional
        # distribution the policy is optimizing against.
        teacher_prompt = f"{prompt_text}{answer_text}"
        ids = tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"]
        prompt_ids_list.append(torch.tensor(ids, dtype=torch.long))
        seq_lens.append(len(ids))

    max_prompt_len = max(seq_lens) if seq_lens else 0
    response_len = responses.size(1)
    batch_size = responses.size(0)
    total_len = max_prompt_len + response_len

    input_ids = torch.zeros(batch_size, total_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long)
    position_ids = torch.zeros(batch_size, total_len, dtype=torch.long)
    response_mask_full = torch.zeros(batch_size, total_len, dtype=torch.long)

    for i in range(batch_size):
        p = prompt_ids_list[i]
        p_len = p.numel()
        r_mask = response_mask[i].long()
        r_ids = responses[i].long()
        input_ids[i, :p_len] = p
        attention_mask[i, :p_len] = 1
        valid_r = int(r_mask.sum().item())
        if valid_r > 0:
            input_ids[i, p_len : p_len + response_len] = r_ids
            attention_mask[i, p_len : p_len + valid_r] = 1
            response_mask_full[i, p_len : p_len + response_len] = r_mask
        position_ids[i] = torch.arange(total_len, dtype=torch.long)

    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask_full,
            "position_ids": position_ids,
        }
    )


def _attach_teacher_signals(data: DataProto):
    trainer = _ACTIVE_SDC_CONTEXT.get("trainer")
    tokenizer = _ACTIVE_SDC_CONTEXT.get("tokenizer")
    if trainer is None or tokenizer is None:
        raise RuntimeError("SDC teacher context is not initialized")
    if "sdc_teacher_pos_log_probs" in data.batch.keys():
        return data

    prompt_tensor = data.batch["prompts"]
    response_tensor = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]
    response_mask = data.batch["response_mask"]
    prompt_length = prompt_tensor.size(1)

    prompt_texts: list[str] = []
    gold_answers: list[str] = []
    decoy_answers: list[str] = []

    for i in range(response_tensor.size(0)):
        prompt_text = _decode_prompt_only(
            tokenizer,
            prompt_tensor[i],
            attention_mask[i],
            prompt_length,
        )
        prompt_texts.append(prompt_text)
        gt = data.non_tensor_batch.get("reward_model", [])[i]
        if isinstance(gt, dict):
            gt = gt.get("ground_truth", "")
        gold = str(gt)
        gold_answers.append(gold)
        decoy_answers.append(_rule_based_decoy(gold, seed=42))

    pos_batch = _build_teacher_logprob_batch(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        answer_texts=gold_answers,
        responses=response_tensor,
        response_mask=response_mask,
    )
    neg_batch = _build_teacher_logprob_batch(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        answer_texts=decoy_answers,
        responses=response_tensor,
        response_mask=response_mask,
    )
    pos_out = trainer._compute_ref_log_prob(pos_batch)
    neg_out = trainer._compute_ref_log_prob(neg_batch)
    target_device = response_tensor.device
    data.batch["sdc_teacher_pos_log_probs"] = pos_out.batch["ref_log_prob"].to(target_device)
    data.batch["sdc_teacher_neg_log_probs"] = neg_out.batch["ref_log_prob"].to(target_device)
    return data


class MetaCotSDCRewardManager:
    """SDC_SHARED reward aggregator.

    On each `__call__(batch)`:
      1. Computes SDC region masks (meta / postmeta_shared / postmeta_diff / body)
         for every response and writes them into `batch.batch`.  These masks
         are consumed downstream by `compute_sdc_gdpo_advantage`.
      2. Runs every reward head on decoded completions vs ground_truth, writes
         per-key scalar scores to `batch.non_tensor_batch[key]`, and accumulates
         a token-level reward tensor placed at the EOS position.
      3. Returns a DataProto carrying `rm_scores` + `reward_extra_keys` so that
         `RayPPOTrainer._compute_reward_colocate`'s output contract is honored
         (verl 0.7.1 fit() union's this back into the main batch and then
         `extract_reward(batch)` reads `batch.batch["rm_scores"]`).
    """

    def __init__(
        self,
        tokenizer,
        reward_funcs: List[Callable],
        reward_weights: List[float],
        reward_keys: List[str],
        num_examine: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights
        self.reward_keys = reward_keys
        self.num_examine = num_examine
        assert len(reward_funcs) == len(reward_weights) == len(reward_keys)

    def __call__(self, data: DataProto) -> DataProto:
        if "rm_scores" in data.batch.keys():
            # Already computed (e.g., agent_reward_loop path); no-op with pass-through.
            rm_td = TensorDict({"rm_scores": data.batch["rm_scores"]}, batch_size=len(data))
            return DataProto(batch=rm_td, non_tensor_batch={}, meta_info={"reward_extra_keys": []})

        bs = len(data)
        response_length = data.batch["responses"].shape[-1]
        prompt_length = data.batch["prompts"].shape[-1]

        decoded_responses: list[str] = []
        ground_truths: list[str] = []
        meta_masks = []
        post_shared_masks = []
        post_diff_masks = []
        body_masks = []
        fallback_flags = []

        for i in range(bs):
            item = data[i]
            text, response_ids = _decode_response(
                self.tokenizer,
                item.batch["prompts"],
                item.batch["responses"],
                item.batch["attention_mask"],
                prompt_length,
            )
            decoded_responses.append(text)
            gt = item.non_tensor_batch.get("reward_model", {})
            if isinstance(gt, dict):
                gt = gt.get("ground_truth", "")
            ground_truths.append(str(gt))

            masks = build_sdc_region_masks(
                self.tokenizer,
                response_ids.tolist(),
                self.tokenizer.decode(response_ids, skip_special_tokens=False),
            )

            def _pad(mask: torch.Tensor) -> torch.Tensor:
                out = torch.zeros(response_length, dtype=torch.float32)
                usable = min(response_length, mask.numel())
                out[:usable] = mask[:usable]
                return out

            meta_masks.append(_pad(masks["meta_mask"]))
            post_shared_masks.append(_pad(masks["postmeta_shared_mask"]))
            post_diff_masks.append(_pad(masks["postmeta_diff_mask"]))
            body_masks.append(_pad(masks["body_mask"]))
            fallback_flags.append(masks["fallback_triggered"])

        data.batch["sdc_meta_mask"] = torch.stack(meta_masks, dim=0)
        data.batch["sdc_postmeta_shared_mask"] = torch.stack(post_shared_masks, dim=0)
        data.batch["sdc_postmeta_diff_mask"] = torch.stack(post_diff_masks, dim=0)
        data.batch["sdc_body_mask"] = torch.stack(body_masks, dim=0)
        data.non_tensor_batch["sdc_fallback_triggered"] = np.asarray(fallback_flags, dtype=np.float32)

        completions = [[{"content": text}] for text in decoded_responses]
        combined = torch.zeros(bs, response_length, dtype=torch.float32)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1

        for func_idx, reward_fn in enumerate(self.reward_funcs):
            key = self.reward_keys[func_idx]
            try:
                scores = reward_fn(completions=completions, ground_truth=ground_truths)
            except Exception as exc:
                print(f"[verl_sdc] reward {key} failed: {exc}")
                traceback.print_exc()
                scores = [0.0] * bs
            if len(scores) != bs:
                scores = (list(scores) + [0.0] * bs)[:bs]
            data.non_tensor_batch[key] = np.asarray(scores, dtype=np.float32)

            reward_tensor = torch.zeros(bs, response_length, dtype=torch.float32)
            for i in range(bs):
                eos_pos = max(0, int(valid_response_length[i].item()))
                reward_tensor[i, eos_pos] = float(scores[i]) * float(self.reward_weights[func_idx])
            combined += reward_tensor

        # Emit rm_scores + reward_extra_keys for verl 0.7.1 fit()/extract_reward contract.
        rm_td = TensorDict({"rm_scores": combined}, batch_size=bs)
        extra_keys = list(self.reward_keys) + ["sdc_fallback_triggered"]
        non_tensor = {k: data.non_tensor_batch[k] for k in extra_keys if k in data.non_tensor_batch}
        return DataProto(
            batch=rm_td,
            non_tensor_batch=non_tensor,
            meta_info={"reward_extra_keys": list(non_tensor.keys())},
        )


class SDCRayPPOTrainer(RayPPOTrainer):
    """Thin verl 0.7.1 trainer wrapper that injects an in-process reward manager.

    Why subclass: verl 0.7.1 removed `reward_fn`/`val_reward_fn` kwargs from
    `RayPPOTrainer.__init__`.  Reward now flows through `RewardLoopManager`.
    The SDC pipeline needs the reward call to SIDE-EFFECT the batch (meta
    masks, per-key scores, fallback flag) so that the downstream
    `compute_sdc_gdpo_advantage` can read them.  Routing SDC through the
    async reward_loop_workers would break those side effects.  Overriding
    `_compute_reward_colocate` keeps the contract: fit() still calls it,
    we just service it in-process without the reward_loop_manager.
    """

    def __init__(self, *args, reward_fn=None, val_reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sdc_reward_fn = reward_fn
        self._sdc_val_reward_fn = val_reward_fn if val_reward_fn is not None else reward_fn

    def _compute_reward_colocate(self, batch: DataProto) -> DataProto:
        fn = self._sdc_reward_fn
        if fn is None:
            return super()._compute_reward_colocate(batch)
        return fn(batch)


def _patch_verl_for_sdc():
    import verl.trainer.ppo.ray_trainer as ray_trainer_module
    from verl.single_controller.ray import RayWorkerGroup
    original_compute_advantage = ray_trainer_module.compute_advantage

    def patched_compute_advantage(
        data: DataProto,
        adv_estimator,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=None,
    ):
        if _is_gdpo_estimator(adv_estimator) and config is not None and config.get("sdc_enabled", False):
            if "response_mask" not in data.batch.keys():
                data.batch["response_mask"] = ray_trainer_module.compute_response_mask(data)
            data = _attach_teacher_signals(data)
            advantages, returns = compute_sdc_gdpo_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
                batch=data.batch,
                non_tensor_batch=data.non_tensor_batch,
                config=config,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
            return data
        return original_compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    ray_trainer_module.compute_advantage = patched_compute_advantage

    if not getattr(RayWorkerGroup, "_sdc_checkpoint_wrappers_applied", False):
        def _wg_update_weights(self, global_steps=None):
            return self.execute_all_async("update_weights", global_steps=global_steps)

        def _wg_execute_checkpoint_engine(self, methods, *args, **kwargs):
            return self.execute_all_async("execute_checkpoint_engine", methods, *args, **kwargs)

        RayWorkerGroup.update_weights = _wg_update_weights
        RayWorkerGroup.execute_checkpoint_engine = _wg_execute_checkpoint_engine
        RayWorkerGroup._sdc_checkpoint_wrappers_applied = True
        print("[SDC] patched RayWorkerGroup checkpoint wrappers for veRL 0.7.1")

    try:
        import verl.workers.rollout.vllm_rollout.vllm_async_server as vllm_async_server
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.v1.engine.async_llm import AsyncLLM as V1AsyncLLM

        if not getattr(vllm_async_server, "_sdc_asyncllm_patch_applied", False):
            class _CompatAsyncLLM:
                @staticmethod
                def from_vllm_config(*args, **kwargs):
                    try:
                        return V1AsyncLLM.from_vllm_config(*args, **kwargs)
                    except ValueError as exc:
                        if "VLLM_USE_V1=False" not in str(exc):
                            raise
                        return AsyncLLMEngine.from_vllm_config(*args, **kwargs)

            vllm_async_server.AsyncLLM = _CompatAsyncLLM
            vllm_async_server._sdc_asyncllm_patch_applied = True
            print("[SDC] patched vLLM AsyncLLM compatibility for vllm>=0.8 fallback")
    except Exception as exc:
        print(f"[SDC] skipped vLLM AsyncLLM patch: {type(exc).__name__}: {exc}")


import hydra


@hydra.main(config_path="../../configs", config_name="verl_sdc_e21r_shared", version_base=None)
def main(config):
    if not ray.is_initialized():
        # AMLT single-node jobs can expose a non-loopback pod IP that makes
        # Ray's default head bootstrap path hang while waiting for GCS.
        # For this veRL workload we only need a local head on the same node, so
        # pin Ray bootstrap to loopback and skip the dashboard to reduce
        # startup fragility.
        ray.init(
            include_dashboard=False,
            _node_ip_address="127.0.0.1",
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
        )
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from omegaconf import OmegaConf, open_dict
    from pprint import pprint
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.experimental.reward_loop import migrate_legacy_reward_impl

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Migrate any legacy reward_model.* keys into the new reward.* layout so that
    # RayPPOTrainer internals (need_reward_model, reward_loop_manager) see a
    # consistent config tree.
    try:
        config = migrate_legacy_reward_impl(config)
    except Exception:
        # Migration is best-effort; config may already be in the new layout.
        pass

    logger_cfg = list(config.trainer.get("logger", []))
    has_wandb_key = bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY"))
    if "wandb" in logger_cfg and not has_wandb_key:
        filtered = [name for name in logger_cfg if name != "wandb"] or ["console"]
        with open_dict(config.trainer):
            config.trainer.logger = filtered
        print("[SDC] WANDB key absent; forcing trainer.logger=%s" % filtered)

    reward_fn_cfg = config.reward.get("custom_reward_function", None)
    if reward_fn_cfg is not None and not reward_fn_cfg.get("path"):
        with open_dict(config.reward.custom_reward_function):
            config.reward.custom_reward_function.path = os.path.abspath(__file__)
            config.reward.custom_reward_function.name = "reward_loop_score"
        print("[SDC] configured custom reward_loop fallback:", config.reward.custom_reward_function.path)

    _patch_verl_for_sdc()

    trust_remote_code = config.data.get("trust_remote_code", False)
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    mode = config.get("mode", "SDC_SHARED")
    reward_cfg = REWARD_CONFIGS[mode]
    # Single source of truth: prefer config.algorithm.gdpo_reward_weights / gdpo_reward_keys.
    # REWARD_CONFIGS supplies the functions (which cannot live in YAML) and the
    # default weights/keys used when the YAML omits them.
    alg_cfg = config.get("algorithm", {}) or {}
    yaml_weights = alg_cfg.get("gdpo_reward_weights", None)
    yaml_keys = alg_cfg.get("gdpo_reward_keys", None)
    if yaml_weights is not None:
        resolved_weights = list(yaml_weights)
    else:
        resolved_weights = list(reward_cfg["weights"])
    if yaml_keys is not None:
        resolved_keys = list(yaml_keys)
    else:
        resolved_keys = list(reward_cfg["keys"])
    if len(resolved_weights) != len(reward_cfg["funcs"]):
        raise ValueError(
            f"gdpo_reward_weights length ({len(resolved_weights)}) does not match "
            f"number of reward funcs ({len(reward_cfg['funcs'])}) in mode={mode}"
        )
    if len(resolved_keys) != len(reward_cfg["funcs"]):
        raise ValueError(
            f"gdpo_reward_keys length ({len(resolved_keys)}) does not match "
            f"number of reward funcs ({len(reward_cfg['funcs'])}) in mode={mode}"
        )
    print(f"[SDC] reward weights: {resolved_keys} = {resolved_weights} (source={'yaml' if yaml_weights is not None else 'default'})")
    reward_fn = MetaCotSDCRewardManager(
        tokenizer=tokenizer,
        reward_funcs=reward_cfg["funcs"],
        reward_weights=resolved_weights,
        reward_keys=resolved_keys,
        num_examine=config.get("num_examine", 0),
    )
    val_reward_fn = MetaCotSDCRewardManager(
        tokenizer=tokenizer,
        reward_funcs=reward_cfg["funcs"],
        reward_weights=resolved_weights,
        reward_keys=resolved_keys,
        num_examine=1,
    )

    if config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
        raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")
    # veRL 0.7.1 colocated checkpoint sync expects the actor/ref worker group to
    # expose async `update_weights()` / `execute_checkpoint_engine()` methods.
    # Those live on engine_workers.ActorRolloutRefWorker; the fsdp_workers base
    # class only provides them on a separate Async* subclass, which the current
    # RayPPOTrainer path does not instantiate here.
    from verl.workers.engine_workers import ActorRolloutRefWorker
    from verl.workers.fsdp_workers import CriticWorker
    ray_worker_group_cls = RayWorkerGroup

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        is_train=True,
        max_samples=config.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=config.data.get("val_max_samples", -1),
    )
    train_sampler = create_rl_sampler(config.data, train_dataset)

    trainer = SDCRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    _ACTIVE_SDC_CONTEXT["trainer"] = trainer
    _ACTIVE_SDC_CONTEXT["tokenizer"] = tokenizer
    trainer.init_workers()
    # verl 0.7.1 fit() only calls _compute_reward_colocate when use_rm=True.
    # We keep config.reward.reward_model.enable=False so init_workers does NOT
    # allocate an actual reward-model worker (we compute reward in-process), but
    # we flip use_rm AFTER init so the reward branch routes through our
    # SDCRayPPOTrainer._compute_reward_colocate override. Without this flip,
    # `extract_reward(batch)` raises KeyError for "rm_scores" since nothing
    # populates it.
    trainer.use_rm = True
    trainer.fit()


if __name__ == "__main__":
    main()
