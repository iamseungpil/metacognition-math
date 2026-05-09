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

import hashlib
import os
import traceback
from typing import Callable, List

import numpy as np
import ray
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

from src.metacot.prompt import META_START
from src.training.rewards import (
    correctness_reward,
    meta_commit_shape_reward,
    meta_penalty_reward,
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
    # E21R-v2 + SDC contrastive: only correctness as the GDPO reward head.
    # Meta-format heads (saturated to ~95% by step 100, providing reward floor)
    # are removed because they push the policy toward "clean wrong" on hard tasks.
    # SDC teacher / anti-teacher contrastive (lambda_meta/shared/diff) remain as
    # auxiliary actor losses — they replace outcome_calibration as the second
    # signal source.
    "SDC_CORR_ONLY": {
        "funcs": [correctness_reward],
        "weights": [1.0],
        "keys": ["correctness"],
    },
    # Meta-only SDC RLSD: correctness + asymmetric meta penalty.
    # Penalty-only meta head (0 if meta present, -0.20 if missing) prevents
    # meta-scaffold collapse during RL without creating the +0.9 saturation
    # floor that the symmetric ±0.10 head caused. Pair with
    # sdc_lambda_shared/diff = 0 in the YAML to restrict contrastive teacher
    # to meta tokens only.
    "SDC_CORR_META_PEN": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ── RLSD ablation modes (arxiv 2604.03128) ───────────────────────────────
    # R0: vanilla GRPO baseline — no SDC teacher signal at all. Used to isolate
    # the contribution of meta-region teacher guidance over plain RLVR.
    "VANILLA_GRPO": {
        "funcs": [correctness_reward],
        "weights": [1.0],
        "keys": ["correctness"],
    },
    # R1: RLSD with single (gold-conditioned) teacher T+ on meta region only.
    # Matches the paper's RLSD formulation: factor = exp(sign × (T+ − student))
    # clipped to [1−ε, 1+ε], applied as multiplicative magnitude evaluator.
    # Decoy forward pass is SKIPPED at runtime (saves the 1 of ~3-4 forwards
    # used by the contrastive variants — roughly 25-33% wall-time saving).
    "RLSD_META_ATTR": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # R2: RLSD extended with contrastive component. Combined log-ratio
    #   combined = α × (T+ − student) + β × (T+ − T−)
    # On meta region, factor = clip(exp(sign × combined), 1−ε, 1+ε).
    # α (sdc_alpha_attr, default 0.5) weights attractive component.
    # β (sdc_beta_contrast, default 0.5) weights gold-vs-decoy contrast.
    # Both T+ and T− forward passes required.
    "RLSD_META_CONTRAST": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # R3: OPSD baseline — full distribution distillation KL(T+ ‖ student) on
    # meta tokens. Distillation loss path is NOT YET IMPLEMENTED (deferred);
    # this mode currently behaves like RLSD_META_ATTR with a marker for the
    # advantage path. Trainer must add an auxiliary KL loss term to fully
    # realize OPSD; tracked as phase-2 work.
    "OPSD_META": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # R5: RLSD with FORCED <|meta|> token on both student rollout and teacher
    # conditioning. Resolves the "meta empty 95%" pathology by guaranteeing
    # meta region presence (paper 2603.24472 epistemic suppression mitigation).
    # Verified S3 (inspect_forced_meta.py): V0_prefix + gold + force <|meta|>
    # → 71.4% gold commit, 33.3% AIME accuracy, valid meta + body + boxed.
    "RLSD_FORCED_META": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ROD-PT: R5 RLSD framework + position teacher amplify (Plan v5.17 FINAL).
    # Decoy T- replaced by T_position which measures log_prob(META | prompt+gold+response[:p])
    # at first META_START emit position p. Multiplicative on R5's w_meta:
    #   w_combined = w_attr * w_position (RLSD invariant 보존, sign 절대 안 바꿈).
    # Natural emit (forced_meta=False, V0_prefix unused).
    "ROD_PT": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
}

# Modes that do NOT compute teacher forward (env reward only).
_VANILLA_MODES = {"VANILLA_GRPO"}
# Modes that compute T+ forward only (single-teacher RLSD).
# ROD_PT: R5 + position teacher (decoy off, natural emit, multiplicative w_position)
_SINGLE_TEACHER_MODES = {"RLSD_META_ATTR", "OPSD_META", "ROD_PT"}
# Modes that compute T+ AND T− forward (contrastive RLSD).
_CONTRASTIVE_MODES = {"SDC_SHARED", "SDC_CORR_ONLY", "SDC_CORR_META_PEN", "RLSD_META_CONTRAST"}
# Modes that prepend V0 student prefix + forced <|meta|> to teacher conditioning,
# AND require student rollout to start inside meta (via custom agent loop).
# These also do BOTH T+ and T− forwards (same as contrastive). See verl_sdc_utils
# build_sdc_region_masks for the started_inside_meta plumbing this enables.
_FORCED_META_MODES = {"RLSD_FORCED_META"}

_ACTIVE_SDC_CONTEXT = {"trainer": None, "tokenizer": None, "mode": "SDC_SHARED"}


def reward_loop_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Fallback scalar score for veRL agent-loop reward workers.

    Mode-aware: emits the GDPO reward keys appropriate to the active mode.
      • VANILLA_GRPO, SDC_CORR_ONLY  → correctness only
      • RLSD_META_ATTR / RLSD_META_CONTRAST / OPSD_META / SDC_CORR_META_PEN
                                     → correctness + meta_penalty
      • SDC_SHARED                   → all 5 legacy heads
                                       (correctness, outcome_calibration,
                                        meta_structure, meta_commit_shape,
                                        postmeta_closure)

    Always populates `correctness` and `meta_penalty` for backward compat with
    callers that read those keys directly; mode-specific extra heads are added
    for SDC_SHARED to avoid breaking the legacy 5-head config when async
    rollout is enabled.
    """
    completion = [[{"content": solution_str}]]
    gt = [ground_truth]

    def _safe_call(fn, with_gt=True):
        try:
            return float(fn(completion, gt)[0]) if with_gt else float(fn(completion)[0])
        except Exception:
            return 0.0

    correctness = _safe_call(correctness_reward, with_gt=True)
    meta_pen = _safe_call(meta_penalty_reward, with_gt=False)

    out = {
        "score": float(correctness + meta_pen),
        "correctness": correctness,
        "meta_penalty": meta_pen,
        "data_source": data_source or "",
    }

    mode = _ACTIVE_SDC_CONTEXT.get("mode", "SDC_SHARED")
    if mode == "SDC_SHARED":
        # Restore the 5-head legacy contract so multi_turn / async rollout
        # paths don't crash on missing GDPO reward keys.
        out["outcome_calibration"] = _safe_call(outcome_calibration_reward, with_gt=True)
        out["meta_structure"] = _safe_call(meta_structure_reward, with_gt=False)
        out["meta_commit_shape"] = _safe_call(meta_commit_shape_reward, with_gt=False)
        from src.training.verl_sdc_utils import postmeta_closure_reward as _pcr
        out["postmeta_closure"] = _safe_call(_pcr, with_gt=False)

    return out


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


# ─── V0 prefix cache (RLSD_FORCED_META) ─────────────────────────────────────
# Why a module-level cache: every batch in a step calls _attach_teacher_signals
# once, but each unique prompt may appear under multiple uids. Caching by
# (global_steps, prompt_hash) means we generate V0 prefixes at most once per
# unique prompt per step. Cache is cleared whenever ``global_steps`` advances
# so we never serve a stale prefix from a previous policy.
_V0_PREFIX_CACHE: dict[tuple[int, str], str] = {}
_V0_PREFIX_LAST_STEP: list[int] = [-1]  # mutable holder, avoids `global` decl


def _generate_v0_prefixes(
    trainer,
    tokenizer,
    prompt_texts: list[str],
    global_steps: int,
    max_new_tokens: int = 256,
    max_chars: int = 1500,
) -> dict[str, str]:
    """Generate (or retrieve cached) V0 student-prefix strings for forced-meta mode.

    For RLSD_FORCED_META the teacher conditioning is augmented with the V0
    student's first attempt followed by the gold answer + ``<|meta|>``. This
    grounds the teacher distribution on a contextualized "rethink-after-attempt"
    rather than a synthetic continuation from prompt+answer alone.

    TODO(v0): The actual V0 generation path requires invoking the rollout
        manager mid-step (e.g., ``trainer.async_rollout_manager.generate_sequences``
        with ``meta_info["validate"]=True``). That is non-trivial because the
        rollout workers may be busy with the current generate call. To keep
        this plug-and-play and avoid deadlocks, this initial implementation
        falls back to a constant ``"(no prior attempt)"`` prefix for every
        prompt — which still validates the core hypothesis (force <|meta|> +
        gold context) without V0. Wire real V0 generation later as a follow-up
        after smoke-testing the forced <|meta|> + gold path end-to-end.

    Args:
        trainer: SDCRayPPOTrainer instance (used by future V0 generation).
        tokenizer: Tokenizer (currently unused; reserved for future stripping).
        prompt_texts: List of unique prompt strings to generate prefixes for.
        global_steps: Current trainer step. Cache is invalidated on step change
            so we don't reuse last-step prefixes after the policy has shifted.
        max_new_tokens: Future V0 generation cap (unused in fallback path).
        max_chars: Future cap on stripped prefix length (unused in fallback path).

    Returns:
        dict mapping each input prompt_text → its V0 prefix string. Always
        returns one entry per input.
    """
    # Step transition: drop stale entries to bound memory + avoid reusing
    # prefixes that no longer reflect the current policy's V0 behavior.
    if _V0_PREFIX_LAST_STEP[0] != global_steps:
        _V0_PREFIX_CACHE.clear()
        _V0_PREFIX_LAST_STEP[0] = global_steps

    out: dict[str, str] = {}
    missing: list[str] = []
    for pt in prompt_texts:
        # Hash by prompt prefix-16 of MD5 — collisions are vanishingly unlikely
        # within a single training step's unique prompt set.
        h = hashlib.md5(pt.encode("utf-8", errors="replace")).hexdigest()[:16]
        cache_key = (global_steps, h)
        if cache_key in _V0_PREFIX_CACHE:
            out[pt] = _V0_PREFIX_CACHE[cache_key]
        else:
            missing.append(pt)

    # ── Fallback path (TODO(v0)): no real generation; populate placeholder. ──
    # See module docstring above for rationale. This is intentional and safe:
    # the meta_info "(no prior attempt)" is non-empty, contains no <|meta|>
    # token, and is short — so downstream stripping/capping logic is a no-op
    # for it but still exercises the same code path that real V0 prefixes will.
    fallback_str = "(no prior attempt)"
    for pt in missing:
        h = hashlib.md5(pt.encode("utf-8", errors="replace")).hexdigest()[:16]
        cache_key = (global_steps, h)
        prefix = fallback_str

        # Apply the same hygiene we'll need once real V0 is wired up:
        # 1. Strip chat-template markers (in case generation includes them).
        for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            if marker in prefix:
                prefix = prefix.split(marker, 1)[0]
        # 2. Slice at first <|meta|> if present (real V0 may emit one).
        if META_START in prefix:
            prefix = prefix.split(META_START, 1)[0]
        # 3. Cap length from the END (most recent reasoning is most informative).
        if len(prefix) > max_chars:
            prefix = prefix[-max_chars:]
        prefix = prefix.strip()
        if not prefix:
            prefix = fallback_str

        _V0_PREFIX_CACHE[cache_key] = prefix
        out[pt] = prefix

    return out


def _build_teacher_logprob_batch(
    *,
    tokenizer,
    prompt_texts: list[str],
    answer_texts: list[str],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    v0_prefixes: dict[str, str] | None = None,
    forced_meta: bool = False,
):
    prompt_ids_list = []
    seq_lens = []
    for prompt_text, answer_text in zip(prompt_texts, answer_texts):
        if forced_meta:
            # RLSD_FORCED_META teacher conditioning: prepend the V0 student
            # prefix (when available) and a "(The correct answer is X.)" hint,
            # then force a <|meta|> opening so teacher log-prob is conditioned
            # on the same "start inside meta" state the student rollout sees.
            # Layout: <prompt> <V0_prefix>?\n(The correct answer is <gold>.)\n<|meta|>
            #
            # CRITICAL: prompt_text comes from _decode_prompt_only() of the
            # rollout's input_ids. The agent loop (ForcedMetaAgentLoop) has
            # already appended <|meta|> token id to prompt_ids → prompt_text
            # ends with META_START. Strip it before re-appending the suffix
            # so the final teacher prompt has EXACTLY ONE <|meta|> at the end
            # (matches inspect_forced_meta.py S3 verified format byte-for-byte).
            base = prompt_text
            if base.endswith(META_START):
                base = base[: -len(META_START)]
            v0 = (v0_prefixes.get(prompt_text, "") if v0_prefixes else "").rstrip()
            sep = "\n" if v0 else ""
            teacher_prompt = (
                f"{base}{v0}{sep}(The correct answer is {answer_text}.)\n{META_START}"
            )
        else:
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
    # prompts/responses split required by verl 0.7.1 left_right_2_no_padding.
    prompts_padded = torch.zeros(batch_size, max_prompt_len, dtype=torch.long)
    prompts_attn = torch.zeros(batch_size, max_prompt_len, dtype=torch.long)

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
        # left-pad each prompt to max_prompt_len so verl's prompt-side accessors work
        prompts_padded[i, max_prompt_len - p_len : max_prompt_len] = p
        prompts_attn[i, max_prompt_len - p_len : max_prompt_len] = 1

    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask_full,
            "position_ids": position_ids,
            "prompts": prompts_padded,
            "responses": responses.long(),
        }
    )


def _attach_teacher_signals(data: DataProto):
    trainer = _ACTIVE_SDC_CONTEXT.get("trainer")
    tokenizer = _ACTIVE_SDC_CONTEXT.get("tokenizer")
    mode = _ACTIVE_SDC_CONTEXT.get("mode", "SDC_SHARED")
    if trainer is None or tokenizer is None:
        raise RuntimeError("SDC teacher context is not initialized")
    # R0 (VANILLA_GRPO): no teacher signal at all. Skip all forward passes
    # and return data unmodified — base GDPO advantage path takes over.
    if mode in _VANILLA_MODES:
        return data
    # Both keys must exist for downstream compute_sdc_gdpo_advantage; an
    # interrupted attach (only one key set) must be recomputed, not cached.
    if (
        "sdc_teacher_pos_log_probs" in data.batch.keys()
        and "sdc_teacher_neg_log_probs" in data.batch.keys()
    ):
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

    # ── R5 (RLSD_FORCED_META) prep ─────────────────────────────────────────
    # Generate (or fetch cached) V0 student prefixes for unique prompts in this
    # batch. The forced-meta path threads these through to both T+ and T-
    # teacher conditioning so the teacher distribution is grounded on the same
    # "rethink-after-attempt" context the student sees post-rollout.
    v0_prefixes: dict[str, str] | None = None
    forced_meta_flag = mode in _FORCED_META_MODES
    if forced_meta_flag:
        global_steps = int(getattr(trainer, "global_steps", 0) or 0)
        unique_prompts = sorted(set(prompt_texts))
        v0_prefixes = _generate_v0_prefixes(
            trainer, tokenizer, unique_prompts, global_steps
        )

    pos_batch = _build_teacher_logprob_batch(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        answer_texts=gold_answers,
        responses=response_tensor,
        response_mask=response_mask,
        v0_prefixes=v0_prefixes,
        forced_meta=forced_meta_flag,
    )
    # verl 0.7.1 engine_workers infer_batch reads micro_batch["temperature"];
    # the trainer's main fit() loop sets it on the rollout output, but our
    # freshly-built teacher batches don't inherit meta_info, so re-attach.
    rollout_temp = float(trainer.config.actor_rollout_ref.rollout.temperature)
    pos_batch.meta_info["temperature"] = rollout_temp
    pos_out = trainer._compute_ref_log_prob(pos_batch)
    target_device = response_tensor.device
    data.batch["sdc_teacher_pos_log_probs"] = pos_out.batch["ref_log_prob"].to(target_device)

    # R1 (RLSD_META_ATTR), OPSD_META, ROD_PT: skip decoy forward — saves the one
    # teacher pass dedicated to T- (roughly 25-33% wall-time vs full SDC).
    # Set teacher_neg = teacher_pos so any downstream contrastive computation
    # (delta = T+ − T−, w_shared) becomes a no-op (delta=0 → w_diff=1,
    # w_shared=w_attr). With λ_shared=λ_diff=0 in the yaml this is fully
    # equivalent to "no decoy".
    if mode in _SINGLE_TEACHER_MODES:
        data.batch["sdc_teacher_neg_log_probs"] = data.batch["sdc_teacher_pos_log_probs"].clone()

        # ROD_PT (Plan v5.17 FINAL): position teacher forward.
        # T_position input = prompt + gold + response[:p] where p = first META_START position.
        # Returns log_prob(META | prompt + gold + response[:p]) = position factor signal.
        # We reuse R5 _build_teacher_logprob_batch with truncated response_mask (valid only up to p).
        if mode == "ROD_PT":
            try:
                meta_start_id = int(tokenizer.convert_tokens_to_ids("<|meta|>"))
            except Exception:
                meta_start_id = -1
            target_device = response_tensor.device
            full_log_prob_meta = torch.zeros(response_tensor.size(0), device=target_device)

            if meta_start_id > 0:
                # Find first META_START position p per rollout
                rollout_ps: list[tuple[int, int]] = []
                for b in range(response_tensor.size(0)):
                    valid = (response_tensor[b] == meta_start_id) & response_mask[b].bool()
                    nz = valid.nonzero(as_tuple=True)[0]
                    if nz.numel() > 0:
                        rollout_ps.append((b, int(nz[0].item())))

                if rollout_ps:
                    N = len(rollout_ps)
                    T_resp = response_tensor.size(1)
                    # Build subset batch with truncated mask (valid only up to position p inclusive)
                    truncated_mask_subset = torch.zeros(
                        (N, T_resp), dtype=response_mask.dtype, device=response_mask.device
                    )
                    truncated_responses_subset = []
                    prompt_texts_subset = []
                    gold_subset = []
                    for i, (b, p) in enumerate(rollout_ps):
                        truncated_mask_subset[i, : p + 1] = 1.0
                        truncated_responses_subset.append(response_tensor[b])
                        prompt_texts_subset.append(prompt_texts[b])
                        gold_subset.append(gold_answers[b])
                    truncated_responses = torch.stack(truncated_responses_subset, dim=0)

                    position_batch = _build_teacher_logprob_batch(
                        tokenizer=tokenizer,
                        prompt_texts=prompt_texts_subset,
                        answer_texts=gold_subset,
                        responses=truncated_responses,
                        response_mask=truncated_mask_subset,
                        v0_prefixes=None,
                        forced_meta=False,
                    )
                    position_batch.meta_info["temperature"] = rollout_temp
                    pos_position_out = trainer._compute_ref_log_prob(position_batch)
                    # ref_log_prob[i, t] = log_prob of responses[i, t] given preceding context
                    # → ref_log_prob[i, p] = log_prob(META | prompt + gold + response[:p])
                    ref_log_probs_position = pos_position_out.batch["ref_log_prob"].to(target_device)

                    for i, (b, p) in enumerate(rollout_ps):
                        # Bound check (in case T_resp_dim mismatch from padding)
                        if p < ref_log_probs_position.size(1):
                            full_log_prob_meta[b] = ref_log_probs_position[i, p]

            data.batch["sdc_position_log_prob_meta"] = full_log_prob_meta
    else:
        # Both contrastive (R2/SDC_*) and forced-meta (R5) modes run T- here.
        # Forced-meta passes the same v0_prefixes + forced_meta=True so the
        # decoy teacher conditioning matches the gold teacher format (only the
        # answer slot differs), preserving the contrast-on-answer-only
        # invariant the SDC contrastive math depends on.
        neg_batch = _build_teacher_logprob_batch(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            answer_texts=decoy_answers,
            responses=response_tensor,
            response_mask=response_mask,
            v0_prefixes=v0_prefixes,
            forced_meta=forced_meta_flag,
        )
        neg_batch.meta_info["temperature"] = rollout_temp
        neg_out = trainer._compute_ref_log_prob(neg_batch)
        data.batch["sdc_teacher_neg_log_probs"] = neg_out.batch["ref_log_prob"].to(target_device)

    # When agent_reward_loop pre-populates rm_scores asynchronously, the SDC
    # reward manager early-returns before computing region masks. compute_advantage
    # downstream still expects sdc_meta_mask / sdc_postmeta_*_mask / sdc_body_mask,
    # so populate them here from the response tokens we already have on hand.
    if "sdc_meta_mask" not in data.batch.keys():
        bs = response_tensor.size(0)
        response_length = response_tensor.size(1)
        meta_masks, post_shared, post_diff, body, fb = [], [], [], [], []
        for i in range(bs):
            r_ids = response_tensor[i].tolist()
            masks = build_sdc_region_masks(
                tokenizer,
                r_ids,
                tokenizer.decode(r_ids, skip_special_tokens=False),
            )
            def _pad(m: torch.Tensor) -> torch.Tensor:
                out = torch.zeros(response_length, dtype=torch.float32)
                usable = min(response_length, m.numel())
                out[:usable] = m[:usable]
                return out
            meta_masks.append(_pad(masks["meta_mask"]))
            post_shared.append(_pad(masks["postmeta_shared_mask"]))
            post_diff.append(_pad(masks["postmeta_diff_mask"]))
            body.append(_pad(masks["body_mask"]))
            fb.append(masks["fallback_triggered"])
        data.batch["sdc_meta_mask"] = torch.stack(meta_masks, dim=0).to(target_device)
        data.batch["sdc_postmeta_shared_mask"] = torch.stack(post_shared, dim=0).to(target_device)
        data.batch["sdc_postmeta_diff_mask"] = torch.stack(post_diff, dim=0).to(target_device)
        data.batch["sdc_body_mask"] = torch.stack(body, dim=0).to(target_device)
        data.non_tensor_batch["sdc_fallback_triggered"] = np.asarray(fb, dtype=np.float32)
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
    if mode not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown mode='{mode}'. Available: {sorted(REWARD_CONFIGS.keys())}"
        )
    if mode == "OPSD_META":
        # KL distillation auxiliary loss is NOT implemented yet. The advantage
        # path falls back to RLSD_META_ATTR (T+ only, attractive). Refusing to
        # silently run a half-implemented mode prevents accidental wasted runs.
        raise NotImplementedError(
            "mode=OPSD_META requires KL distillation loss in the actor — "
            "phase-2 work, not yet wired. Use RLSD_META_ATTR or "
            "RLSD_META_CONTRAST for now."
        )
    reward_cfg = REWARD_CONFIGS[mode]
    # Make mode visible to runtime hooks (advantage compute & teacher attach).
    _ACTIVE_SDC_CONTEXT["mode"] = mode
    # Mirror mode into algorithm config so verl_sdc_utils.compute_sdc_gdpo_advantage
    # (which only receives algorithm config) can dispatch on it. Disable struct
    # mode locally so we can write a key that may not exist in legacy yamls.
    if "algorithm" in config:
        try:
            from omegaconf import OmegaConf as _OC
            _was_struct = _OC.is_struct(config.algorithm)
            _OC.set_struct(config.algorithm, False)
            config.algorithm.sdc_mode = mode
            if _was_struct:
                _OC.set_struct(config.algorithm, True)
        except Exception:
            # Last-resort dict-style assignment.
            try:
                config["algorithm"]["sdc_mode"] = mode
            except Exception:
                pass  # cannot inject; compute_sdc_gdpo_advantage will use default
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
