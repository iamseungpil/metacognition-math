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
import pathlib
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
    compute_degeneration_penalty,
    correctness_reward,
    degeneration_penalty_reward,
    meta_commit_shape_reward,
    meta_penalty_reward,
    meta_structure_reward,
    outcome_calibration_reward,
    # ── C1/C2 next-wave arms (deliverables #1/#2): ADDITIVE imports. None of
    # the above existing imports change; these are only referenced by the NEW
    # REWARD_CONFIGS entries (ROD_PT2_E21CTRL, STABLE_GFN_C2FIX). The C2-fix
    # reward and the E21Rv2 control heads are reused from rewards.py verbatim.
    confidence_omission_floor,
    confidence_revision_reward,
    meta_count_bonus,
    meta_penalty_adaptive_reward,
    redirect_execution_reward,
    verify_execution_reward,
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
    # ROD_PT_DEGEN (R16, 2026-05-12): ROD_PT + sample-level degeneration_penalty
    # via dedicated GDPO key, weight 0.3 vs correctness.
    # Stabilizes hard-regime generation (AIME tail collapse 57% → target <30%)
    # so R17 (control-field RLSD + follow-through) can be measured cleanly.
    # Spec: codex-locked V4 (4 rounds review, 1030-trace replay validated).
    # See src/training/rewards.py compute_degeneration_penalty for details.
    "ROD_PT_DEGEN": {
        "funcs": [correctness_reward, meta_penalty_reward, degeneration_penalty_reward],
        "weights": [1.0, 1.0, 0.3],
        "keys": ["correctness", "meta_penalty", "degeneration_penalty"],
    },
    # ROD_MQ (R18a, Plan v7.2.2): meta-quality verifiable signal — single
    # teacher T+ on EXTENDED meta region (meta block + post K=10 tokens).
    # Sign-preserving multiplicative factor:
    #   q_attr = mean over extended meta of clip(T+ − student, [-10,10])
    #   q_centered = q_attr − batch_median(q_attr)   (centered, RLVR invariant)
    #   w_meta_quality = clip(exp(sign × q_centered / τ), 1−ε, 1+ε)   per sequence
    #   w_meta = w_attr × w_meta_quality            (PRODUCT, sign preserved)
    # NO rubric, NO judge — verifiable signal only (RLVR + T+ logit).
    "ROD_MQ": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ROD_MQ_CONTRAST (R18b, Plan v7.2.2): R18a + decoy-contrast term.
    #   q_contrast = mean over extended meta of clip(T+ − T−, [-10,10])
    #   q_meta = α × q_attr + β × q_contrast        (α default 1.0, β default 0.0)
    # Same q_centered / w_meta_quality / w_meta product as R18a.
    "ROD_MQ_CONTRAST": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ROD_MQ_CONTRAST_INJECT (CTSD Phase C, Plan v5): identical reward heads to
    # ROD_MQ_CONTRAST (R18b) — the ONLY difference is rollout-time force-inject of
    # <|meta|> at the max-entropy pre-answer position (algorithm.sdc_force_inject).
    # R18b failed (70.9%) because contrastive reward had no good/bad variance to
    # act on in the model's decorative natural meta; force-inject creates the meta
    # region for the reward to shape. Inject core = src/training/meta_inject.py
    # (unit-tested); two-phase rollout wiring = SDCRayPPOTrainer (node-smoke-req).
    # Gated by A.3 PASS (force-inject shown causally helpful) before launch.
    "ROD_MQ_CONTRAST_INJECT": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # GFN_OPSD_CONTRAST (R18c, Plan v7.2.7 codex r12-r14 LOCK): GFN distribution
    # matching on meta token region. Listwise KL (target=softmax(logR/τ),
    # student=softmax(logP_S/τ)) as primary aux loss; pairwise cTB as diagnostic.
    # Verifiable signal only — T+/T- logit, no rubric/judge.
    # See `compute_sdc_gfn_actor_loss` below and `_patch_actor_loss_for_gfn`
    # for the ppo_loss hook injection.
    "GFN_OPSD_CONTRAST": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # RLSD_FAITHFUL_META (R20, Plan iter-3 SURVEY-GROUNDED LOCK, direction B):
    # "RLSD-faithful meta-token credit". Correctness-ONLY reward head — NO
    # meta_penalty: the asymmetric presence-only meta_penalty injects a
    # teacher/presence SIGN term into base_advantages (token_level_rewards),
    # which BREAKS the RLSD sign/magnitude separation on meta tokens (the
    # diagnosed −14pt cause vs E21Rv2). Here base_advantages sign = pure env
    # correctness; the teacher affects ONLY within-trajectory MAGNITUDE via an
    # UN-clipped sign-preserving w_meta in the advantage path (see
    # verl_sdc_utils sdc_mode=="RLSD_FAITHFUL_META" branch). Single teacher
    # (T+ only). Differentiation vs InT/RLSD/OPSD/Stable-GFN: the 4-part
    # invariant scoped to metacognitive control tokens specifically.
    "RLSD_FAITHFUL_META": {
        "funcs": [correctness_reward],
        "weights": [1.0],
        "keys": ["correctness"],
    },
    # STABLE_GFN (R21, Plan TWO-DIRECTION-SPLIT direction A — "Stable-GFlowNet"
    # signal DELIVERY; codex D1). NEW mode, additive, zero-touch to the 5
    # in-flight modes + RLSD_FAITHFUL_META. Reward head MIRRORS
    # GFN_OPSD_CONTRAST (correctness + meta_penalty) so the ONLY single-variable
    # delta vs the GFN baseline is the *delivery* of the teacher signal:
    #   • Advantage plane: sdc_lambda_meta/shared/diff = 0 in the YAML ⇒ the
    #     multiplicative w_meta throttle (the clipped ±20% no-op, diagnosed
    #     cause C1) is fully removed; the meta region receives the PURE env
    #     correctness advantage sign (RLSD invariant intact).
    #   • The teacher signal is delivered ENTIRELY through a Z-free pairwise
    #     contrastive Trajectory Balance aux loss on the actor-loss plane
    #     (sdc_gfn_objective=pairwise_ctb) + frozen_ref baseline +
    #     reward-temperature (target = logR / T_R, student NOT /τ).
    # Hypothesis HA: un-throttled delivery raises meta-conditioned accuracy IFF
    # the delivered signal is already correct (else it AMPLIFIES C3) — hence B
    # gates A; this mode is the A-only / A∘B delivery vehicle.
    "STABLE_GFN": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ── C1/C2 NEXT-WAVE ARMS (2026-05-18, EXPERIMENT_PLAN_ARMS.md "CODEX
    # PLAN-REVIEW CONVERGED"). All three entries below are NEW and ADDITIVE:
    # no entry above this comment is touched; existing §8/RLSD modes keep their
    # exact funcs/weights/keys (regression-asserted).
    #
    # ROD_PT2_E21CTRL  (Arm 2 = "Recipe X" additive primary)
    #   Intent     : combine the best self-distill ingredient (R10/ROD_PT
    #                content×position 2-teacher) with the best verifiable-RL
    #                ingredient (E21Rv2 control reward heads), un-throttled
    #                (C1 fix in verl_sdc_utils) + C2-fixed (the adaptive meta
    #                cost below replaces the presence-forced meta_penalty).
    #   Hypothesis : un-clip + C2-fix + E21Rv2 control + 2-teacher => accuracy
    #                >= B* (Arm 1) with adaptive (not forced ~100%) meta.
    #   Validation : 1-step numeric reward sanity + the
    #                ROD_PT2_E21CTRL w_meta branch (verl_sdc_utils) un-clipped
    #                to [1/w_max, w_max] (test_c1c2_arms_smoke.py).
    #   Reward head : correctness + the FOUR E21Rv2 control heads
    #                 (confidence_revision, redirect_execution,
    #                  verify_execution, meta_floor) + meta_count_bonus
    #                 + the C2-fix adaptive meta cost (meta_penalty_adaptive).
    #                 Weights mirror compute_score_confidence_centered's
    #                 E21Rv2 control ratios (corr 1.0, conf_rev 0.35,
    #                 redirect 0.30, verify 0.15, floor 0.5, count 1.0); the
    #                 C2-fix head carries weight 1.0 like the OLD meta_penalty
    #                 head it REPLACES (same magnitude envelope, adaptive sign).
    # 2026-05-22 ROOT-CAUSE FIX (replaces the 2026-05-20 emergency 2-key
    # collapse): the original 7-key design crashed with `AssertionError:
    # GDPO reward key 'confidence_revision' not found in non_tensor_batch`
    # because the agent_reward_loop path pre-fills `rm_scores` and the
    # MetaCotSDCRewardManager.__call__ early-return then dropped the
    # per-key emit (lines 1061-1064). Fix lives in __call__ now; reward
    # design restored to the documented E21Rv2 control × C2-fix product.
    "ROD_PT2_E21CTRL": {
        "funcs": [
            correctness_reward,
            confidence_revision_reward,
            redirect_execution_reward,
            verify_execution_reward,
            confidence_omission_floor,
            meta_count_bonus,
            meta_penalty_adaptive_reward,
        ],
        "weights": [1.0, 0.35, 0.30, 0.15, 0.5, 1.0, 1.0],
        "keys": [
            "correctness",
            "confidence_revision",
            "redirect_execution",
            "verify_execution",
            "meta_floor",
            "meta_count_bonus",
            "meta_penalty_adaptive",
        ],
    },
    # STABLE_GFN_C2FIX  (Arm 3 = STABLE_GFN PRIMARY, un-clip + C2-fix)
    #   Intent     : the SHIP'd STABLE_GFN mechanism (Z-free pairwise cTB,
    #                pairwise_ctb code byte-identical) but paper-grade primary:
    #                drop the presence-forced meta_penalty (C2) and apply the
    #                un-clip (C1) on the (λ=0) advantage plane for parity with
    #                Arm 2's magnitude bound.
    #   Hypothesis : cTB delivers the SD teacher without the C1 multiplicative
    #                throttle and, with C2-fix, reaches accuracy >= B* with
    #                adaptive meta; stability observational vs Arm 2.
    #   Validation : config-only diff vs STABLE_GFN (mechanism code
    #                byte-parity-proven in test_c1c2_arms_smoke.py); reward
    #                head = correctness + C2-fix adaptive (single-variable vs
    #                STABLE_GFN's correctness+meta_penalty: ONLY the meta cost
    #                head swaps presence-forced -> adaptive).
    "STABLE_GFN_C2FIX": {
        "funcs": [correctness_reward, meta_penalty_adaptive_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty_adaptive"],
    },
    # MATCHED_E21RV2  (Arm 1 = B* matched RLVR baseline — mandatory comparator)
    #   Intent     : a same-infra E21Rv2 re-run so any SD-vs-RLVR delta is
    #                attributable to METHOD, not nuisance (resolves C6 reward-
    #                path / C7 baseline-parity). It runs under the IDENTICAL
    #                SDC infra (verl_sdc_e21r_shared base, same SFT init / RL
    #                parquet / reward-path / decoding / eval) as Arms 2 & 3.
    #   Hypothesis : E21Rv2's reward, re-run here, yields baseline B* — MEASURE
    #                it; do NOT assume the stale 81.7. B* is THE comparator;
    #                all Arm-2/3 success is RELATIVE to B*.
    #   Validation : 1030-panel 16k overall+per-bench+meta-emission; this is
    #                the pre-registered non-inferiority anchor (parity fields in
    #                configs/verl_matched_e21rv2_arm1_h200_4x4k.yaml).
    #   Reward head : EXACTLY the E21Rv2 confidence-centered control set
    #                 (compute_score_confidence_centered: correctness +
    #                  confidence_revision + redirect_execution +
    #                  verify_execution + meta_floor + meta_count_bonus),
    #                 SAME funcs/weights as Arm-2's control block — but with
    #                 NO self-distill teacher (this mode is in _VANILLA_MODES:
    #                 _attach_teacher_signals returns early, NO T+/T-/position
    #                 forward) and NO C2-fix head: pure matched RLVR. The ONLY
    #                 reward-head difference vs Arm 2 is the absence of the
    #                 C2-fix adaptive head (Arm 2 = this + 2-teacher SD + C2-fix
    #                 + un-clip), so the bridge ablation is clean.
    # 2026-05-22 ROOT-CAUSE FIX (replaces 2026-05-20 emergency 2-key):
    # see ROD_PT2_E21CTRL note above. Arm 1 is the matched-RLVR comparator
    # using the E21Rv2 6-control reward set, no teacher forward.
    "MATCHED_E21RV2": {
        "funcs": [
            correctness_reward,
            confidence_revision_reward,
            redirect_execution_reward,
            verify_execution_reward,
            confidence_omission_floor,
            meta_count_bonus,
        ],
        "weights": [1.0, 0.35, 0.30, 0.15, 0.5, 1.0],
        "keys": [
            "correctness",
            "confidence_revision",
            "redirect_execution",
            "verify_execution",
            "meta_floor",
            "meta_count_bonus",
        ],
    },
}

# Modes that do NOT compute teacher forward (env reward only).
# MATCHED_E21RV2 (Arm 1, ADDITIVE): a teacher-free matched-RLVR baseline —
# joins the no-teacher-forward set so _attach_teacher_signals returns early
# (no T+/T-/position forward), exactly like VANILLA_GRPO. The advantage path
# early-return in verl_sdc_utils was extended with a matching OR-clause.
# VANILLA_GRPO membership/behaviour is unchanged (set still contains it).
_VANILLA_MODES = {"VANILLA_GRPO", "MATCHED_E21RV2"}
# Modes that compute T+ forward only (single-teacher RLSD).
# ROD_PT: R5 + position teacher (decoy off, natural emit, multiplicative w_position)
# ROD_PT_DEGEN: ROD_PT + degeneration_penalty reward head (R16, 2026-05-12)
# ROD_MQ: meta-quality factor on extended meta region, T+ − student only (R18a)
# RLSD_FAITHFUL_META: R20 direction B. T+ only (gold-blind teacher → MAGNITUDE
#   only). NO meta_penalty head (see REWARD_CONFIGS). Restores the RLSD
#   sign/magnitude invariant on meta tokens that ROD_* break via clip+penalty.
# ROD_PT2_E21CTRL: Arm-2 additive primary. Same T+ + position-teacher forward
#   pair as ROD_PT (content×position 2-teacher), decoy OFF (single-teacher), so
#   it joins this set AND the ROD_PT position-teacher branch below. ADDITIVE:
#   does not change membership semantics of any pre-existing mode.
_SINGLE_TEACHER_MODES = {"RLSD_META_ATTR", "OPSD_META", "ROD_PT", "ROD_PT_DEGEN", "ROD_MQ", "RLSD_FAITHFUL_META", "ROD_PT2_E21CTRL"}
# Modes that compute T+ AND T− forward (contrastive RLSD).
# ROD_MQ_CONTRAST: R18a + T+ − T− contrast term mixed via α/β (R18b)
_CONTRASTIVE_MODES = {
    "SDC_SHARED",
    "SDC_CORR_ONLY",
    "SDC_CORR_META_PEN",
    "RLSD_META_CONTRAST",
    "ROD_MQ_CONTRAST",
    # ROD_MQ_CONTRAST_INJECT (CTSD Phase C): == ROD_MQ_CONTRAST advantage math,
    # so it also needs T+ AND T- forward for the q_contrast term.
    "ROD_MQ_CONTRAST_INJECT",
    # GFN_OPSD_CONTRAST (R18c, Plan v7.2.7 codex r12-r14 LOCK):
    # needs T+ AND T- forward to compute logR = α(T+−P_S.detach()) + β(T+−T-).
    "GFN_OPSD_CONTRAST",
    # STABLE_GFN (R21, direction A): same logR = α(T+−P_S.detach()) + β(T+−T-)
    # over the meta region as GFN_OPSD_CONTRAST → needs T+ AND T- forward.
    "STABLE_GFN",
    # STABLE_GFN_C2FIX (Arm 3, 2026-05-18): identical teacher-forward topology
    # to STABLE_GFN (same logR; needs T+ AND T-). The ONLY delta is the reward
    # head (C2-fix adaptive vs presence-forced meta_penalty) + the un-clip on
    # the λ=0 advantage plane — neither affects which teacher forwards run.
    "STABLE_GFN_C2FIX",
}
# Modes that prepend V0 student prefix + forced <|meta|> to teacher conditioning,
# AND require student rollout to start inside meta (via custom agent loop).
# These also do BOTH T+ and T− forwards (same as contrastive). See verl_sdc_utils
# build_sdc_region_masks for the started_inside_meta plumbing this enables.
_FORCED_META_MODES = {"RLSD_FORCED_META"}

_ACTIVE_SDC_CONTEXT = {"trainer": None, "tokenizer": None, "mode": "SDC_SHARED"}
# One-shot guard so the pairwise_ctb "0 usable uid groups" warning prints once
# per process instead of every degenerate microbatch (codex review pt.4).
_CTB_INACTIVE_WARNED = {"done": False}

# ── Arm-2 parameterized teacher-prompt slot (deliverable #2) ───────────────
# The gold-conditioned TEACHER prompt is the SD quality lever (codex R4 #2:
# prompt-strengthening, NOT a gameable reward head). It is a PARAMETERIZED
# slot keyed by `algorithm.sdc_teacher_prompt_set`, defaulting to
# "r10v2_baseline". The G2 probe (scripts/prompt_probe_g2.py) decides the
# FROZEN value later; until then the default is the live R10v2 system prompt
# (zero behavioural change vs ROD_PT — the baseline teacher prompt is the
# identity wrapper, so ROD_PT2_E21CTRL@baseline reproduces ROD_PT teacher
# conditioning exactly aside from the documented reward-head/un-clip deltas).
#
# Single source of truth = the G2 prompt JSONs under scripts/prompts/ (so a
# strengthened set cannot silently diverge between the probe and training).
# The slot ONLY ever applies to mode=ROD_PT2_E21CTRL; every pre-existing mode
# never reads it (guarded in _build_teacher_logprob_batch), so existing
# teacher conditioning is byte-identical.
#
# ── G2 PER-TEACHER SLOTS (deliverable #2, 2026-05-19) ──────────────────────
# The single shared slot above is REPLACED (additively) by TWO independent
# slots so the POSITION teacher and the CONTENT (T+) teacher can carry
# different gold-conditioned prompts (EXPERIMENT_PLAN_ARMS.md "G2 PER-TEACHER
# PROMPT SPEC — CODEX CONVERGED ... 2026-05-19"):
#   * algorithm.sdc_position_teacher_prompt_set  -> position-teacher logP(<|meta|>)
#   * algorithm.sdc_content_teacher_prompt_set   -> content-teacher T+ region
# Each accepts: "r10v2_baseline" (default -> "" identity prefix, byte-identical
# to the SHIP'd ROD_PT teacher conditioning) | "pos_teacher_v1" |
# "content_teacher_v1" (the frozen G2 spec prompts) | the legacy
# "strengthened_v*" sets (back-compat). PRECEDENCE: a per-teacher key, when
# explicitly set, overrides the legacy shared `sdc_teacher_prompt_set`; if a
# per-teacher key is left unset it INHERITS the legacy shared value (so the
# pre-existing single-slot configs keep working byte-identically). Both
# defaulting to "r10v2_baseline" => "" prefix for BOTH teachers => identical
# to current Arm-2 behaviour. Slots ONLY ever apply to mode=ROD_PT2_E21CTRL.
_TEACHER_PROMPT_SETS = (
    "r10v2_baseline",
    "strengthened_v1",
    "strengthened_v2",
    "pos_teacher_v1",
    "content_teacher_v1",
)


def _resolve_teacher_prompt_prefix(prompt_set: str) -> str:
    """Return the teacher-instruction PREFIX text for a prompt set.

    r10v2_baseline -> "" (identity: teacher conditioning unchanged vs ROD_PT;
                         the baseline prompt is the live system prompt the
                         policy already uses, NOT an extra teacher wrapper).
    strengthened_v* -> the literal `system_prompt` from the matching FROZEN
                       G2 prompt JSON, prepended (TRAINING-ONLY teacher
                       conditioning; never enters policy inference inputs —
                       guarded by being applied only inside the teacher
                       log-prob batch builder).
    """
    if prompt_set == "r10v2_baseline":
        return ""
    if prompt_set not in _TEACHER_PROMPT_SETS:
        raise ValueError(
            f"sdc_teacher_prompt_set={prompt_set!r} not in {_TEACHER_PROMPT_SETS}"
        )
    import json as _json

    pj = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "prompts"
        / f"{prompt_set}.json"
    )
    if not pj.exists():
        raise FileNotFoundError(
            f"teacher prompt set {prompt_set!r} -> {pj} missing (the G2 "
            "frozen prompt JSON is the single source of truth)"
        )
    spec = _json.loads(pj.read_text(encoding="utf-8"))
    sp = spec.get("system_prompt")
    if not isinstance(sp, str) or not sp.strip():
        raise ValueError(f"{pj.name}: empty/non-literal system_prompt")
    # Prepended as a teacher-only instruction block, clearly delimited.
    return f"{sp}\n\n"


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
    # R16 fix: Ray RewardLoopWorker actors do NOT inherit module-level
    # `_ACTIVE_SDC_CONTEXT["mode"]` from the trainer process. Module state is
    # per-actor. So the mode-conditional emit pattern below would silently skip
    # degeneration_penalty in async-rollout workers, which then fails the
    # GDPO assertion that demands the key exist in non_tensor_batch.
    # Fix: ALWAYS emit degeneration_penalty as a sample-level scalar. The
    # GDPO weight (configured in YAML) is 0 for non-R16 modes so this is a
    # safe no-op, and 0.3 for ROD_PT_DEGEN where it provides the intended signal.
    try:
        tok = _ACTIVE_SDC_CONTEXT.get("tokenizer")
        if tok is not None:
            try:
                length = len(tok.encode(solution_str, add_special_tokens=False))
            except Exception:
                length = len(solution_str.split())
        else:
            length = len(solution_str.split())
        from src.training.rewards import _extract_answer_fallback as _ext
        ans = _ext(solution_str)
        degen, _ = compute_degeneration_penalty(solution_str, length, ans)
        out["degeneration_penalty"] = float(degen)
        if mode == "ROD_PT_DEGEN":
            out["score"] = float(out["score"] + degen)
    except Exception:
        out["degeneration_penalty"] = 0.0

    if mode == "SDC_SHARED":
        # Restore the 5-head legacy contract so multi_turn / async rollout
        # paths don't crash on missing GDPO reward keys.
        out["outcome_calibration"] = _safe_call(outcome_calibration_reward, with_gt=True)
        out["meta_structure"] = _safe_call(meta_structure_reward, with_gt=False)
        out["meta_commit_shape"] = _safe_call(meta_commit_shape_reward, with_gt=False)
        from src.training.verl_sdc_utils import postmeta_closure_reward as _pcr
        out["postmeta_closure"] = _safe_call(_pcr, with_gt=False)

    # 2026-05-22: SAME R16 pattern (line 530-537) applied to the Arm-2
    # ROD_PT2_E21CTRL stabilizer set. Ray RewardLoopWorker actors do not
    # inherit `_ACTIVE_SDC_CONTEXT["mode"]` from the trainer process; the
    # in-actor mode defaults to "SDC_SHARED", so a mode-conditional emit for
    # ROD_PT2_E21CTRL would silently skip the 5 stabilizer keys (and
    # `meta_penalty_adaptive`) — that is the exact mechanism behind the
    # 2026-05-20 Arm-2 GDPO `AssertionError: GDPO reward key
    # 'confidence_revision' not found in non_tensor_batch` and the
    # downstream delimiter-spam collapse (50.9% / 108 empty meta blocks).
    # ALWAYS emit so async-rollout RewardLoopWorker actors honour the
    # Arm-2 reward-key contract regardless of in-actor module state. For
    # non-Arm-2 modes the GDPO weight is 0 (key not in gdpo_reward_keys),
    # so this is a safe no-op everywhere else.
    out["confidence_revision"] = _safe_call(confidence_revision_reward, with_gt=False)
    out["redirect_execution"] = _safe_call(redirect_execution_reward, with_gt=False)
    out["verify_execution"] = _safe_call(verify_execution_reward, with_gt=False)
    out["meta_floor"] = _safe_call(confidence_omission_floor, with_gt=False)
    out["meta_count_bonus"] = _safe_call(meta_count_bonus, with_gt=False)
    out["meta_penalty_adaptive"] = _safe_call(meta_penalty_adaptive_reward, with_gt=True)

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
    teacher_role: str = "content",
):
    # Arm-2 ONLY: a TRAINING-ONLY gold-conditioned teacher-prompt prefix
    # resolved from the PER-TEACHER parameterized slots (deliverable #2,
    # 2026-05-19). `teacher_role` selects which resolved prefix applies:
    #   "content"  -> the content-teacher (T+) slot
    #                 (_ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_prefix"])
    #   "position" -> the position-teacher logP(<|meta|>) slot
    #                 (_ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_prefix"])
    # For EVERY pre-existing mode this is the empty string (the context keys
    # are absent / "r10v2_baseline" -> ""), so the teacher prompt below is
    # byte-identical to the SHIP'd code path. Unknown roles fall back to "".
    _tp_prefix = ""
    if _ACTIVE_SDC_CONTEXT.get("mode") == "ROD_PT2_E21CTRL":
        if teacher_role == "position":
            _tp_prefix = str(
                _ACTIVE_SDC_CONTEXT.get(
                    "sdc_position_teacher_prompt_prefix", ""
                )
            )
        else:  # "content" (default; also covers forced-meta T+ path)
            _tp_prefix = str(
                _ACTIVE_SDC_CONTEXT.get(
                    "sdc_content_teacher_prompt_prefix", ""
                )
            )

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
            #
            # Arm-2 (ROD_PT2_E21CTRL) ONLY: prepend the resolved strengthened
            # teacher-prompt prefix. `_tp_prefix == ""` for every other mode
            # AND for the r10v2_baseline set, so this is the IDENTITY
            # `f"{prompt_text}{answer_text}"` everywhere except a strengthened
            # Arm-2 run — preserving byte-parity for all existing modes.
            teacher_prompt = f"{_tp_prefix}{prompt_text}{answer_text}"
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
        teacher_role="content",  # gold-conditioned T+ (content teacher)
    )
    # verl 0.7.1 engine_workers infer_batch reads micro_batch["temperature"];
    # the trainer's main fit() loop sets it on the rollout output, but our
    # freshly-built teacher batches don't inherit meta_info, so re-attach.
    # Tolerant of a config layout change / a test double without .config.
    try:
        rollout_temp = float(trainer.config.actor_rollout_ref.rollout.temperature)
    except Exception:
        rollout_temp = 1.0
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
        if mode in ("ROD_PT", "ROD_PT_DEGEN", "ROD_PT2_E21CTRL"):  # F1 codex r2 fix: ROD_PT_DEGEN needs position forward too (utils:308 expects it); ROD_PT2_E21CTRL (Arm 2) reuses the SAME content×position 2-teacher
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
                    real_N = len(rollout_ps)
                    # veRL dispatch requires N divisible by:
                    #   1. dp_size (chunk_tensordict in tensordict_utils.py:315)
                    #   2. force_group_size * micro_batch_size_per_gpu (ref_compute_ref_log_prob)
                    # dp_size is the data-parallel world size = nnodes * n_gpus_per_node
                    # (codex Round 3: previously used n_gpus_per_node only, broke on multi-node).
                    try:
                        nnodes = int(trainer.config.trainer.nnodes)
                    except Exception:
                        nnodes = 1
                    try:
                        n_gpus_per_node = int(trainer.config.trainer.n_gpus_per_node)
                    except Exception:
                        n_gpus_per_node = 4
                    dp_size = nnodes * n_gpus_per_node
                    try:
                        micro_bs = int(trainer.config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu)
                    except Exception:
                        try:
                            micro_bs = int(trainer.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu)
                        except Exception:
                            micro_bs = 4
                    pad_unit = dp_size * micro_bs  # safe LCM upper bound
                    pad_n = (-real_N) % pad_unit
                    rollout_ps_padded = list(rollout_ps)
                    for _ in range(pad_n):
                        rollout_ps_padded.append(rollout_ps[0])  # duplicate for padding
                    N = len(rollout_ps_padded)
                    T_resp = response_tensor.size(1)
                    # Build subset batch with truncated mask (valid only up to position p inclusive)
                    truncated_mask_subset = torch.zeros(
                        (N, T_resp), dtype=response_mask.dtype, device=response_mask.device
                    )
                    truncated_responses_subset = []
                    prompt_texts_subset = []
                    gold_subset = []
                    for i, (b, p) in enumerate(rollout_ps_padded):
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
                        teacher_role="position",  # logP(<|meta|>) position teacher
                    )
                    position_batch.meta_info["temperature"] = rollout_temp
                    pos_position_out = trainer._compute_ref_log_prob(position_batch)
                    # ref_log_prob[i, t] = log_prob of responses[i, t] given preceding context
                    # → ref_log_prob[i, p] = log_prob(META | prompt + gold + response[:p])
                    ref_log_probs_position = pos_position_out.batch["ref_log_prob"].to(target_device)

                    # IMPORTANT: only iterate up to real_N (skip padded duplicates)
                    for i, (b, p) in enumerate(rollout_ps[:real_N]):
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
        # T- decoy teacher. This branch is the `else` of
        # `if mode in _SINGLE_TEACHER_MODES` — ROD_PT2_E21CTRL is a
        # single-teacher mode so it NEVER reaches here (T-=T+ clone above).
        # Leaving teacher_role at its "content" default is byte-identical for
        # every mode that DOES run this (contrastive / forced-meta), because
        # the per-teacher prefix guard only fires for ROD_PT2_E21CTRL, which
        # by construction cannot enter this branch -> _tp_prefix stays "".
        neg_batch = _build_teacher_logprob_batch(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            answer_texts=decoy_answers,
            responses=response_tensor,
            response_mask=response_mask,
            v0_prefixes=v0_prefixes,
            forced_meta=forced_meta_flag,
            teacher_role="content",
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
            # Already computed (e.g., agent_reward_loop path). 2026-05-22 fix:
            # the old pass-through returned `non_tensor_batch={}`, which dropped
            # the SDC-specific reward keys (confidence_revision,
            # redirect_execution, verify_execution, meta_floor, meta_count_bonus)
            # that compute_sdc_gdpo_advantage downstream reads. Result was
            # `AssertionError: GDPO reward key 'confidence_revision' not found
            # in non_tensor_batch` at training step 1 for Arm2 (ROD_PT2_E21CTRL).
            # Fix: emit per-key reward scores on this path too, by running the
            # configured reward funcs on decoded completions. rm_scores itself
            # is left untouched (pre-filled value preserved as the env reward).
            # See tests/test_arm2_reward_emit.py for the contract.
            bs = len(data)
            response_length = data.batch["responses"].shape[-1]
            prompt_length = data.batch["prompts"].shape[-1]
            decoded_responses, ground_truths = [], []
            for i in range(bs):
                item = data[i]
                text, _ids = _decode_response(
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
            completions = [[{"content": t}] for t in decoded_responses]
            valid_response_length = (
                data.batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
            )
            completion_lengths_list = [
                int(valid_response_length[i].item()) + 1 for i in range(bs)
            ]
            from src.training.rewards import (
                _extract_answer_fallback as _extract_ans_for_degen,
            )
            answer_extracted_list = [
                _extract_ans_for_degen(t) for t in decoded_responses
            ]
            for func_idx, reward_fn in enumerate(self.reward_funcs):
                key = self.reward_keys[func_idx]
                try:
                    scores = reward_fn(
                        completions=completions,
                        ground_truth=ground_truths,
                        completion_lengths=completion_lengths_list,
                        answer_extracted=answer_extracted_list,
                    )
                except Exception as exc:
                    print(f"[verl_sdc] reward {key} failed (pre-filled path): {exc}")
                    scores = [0.0] * bs
                if len(scores) != bs:
                    scores = (list(scores) + [0.0] * bs)[:bs]
                data.non_tensor_batch[key] = np.asarray(scores, dtype=np.float32)
            rm_td = TensorDict(
                {"rm_scores": data.batch["rm_scores"]}, batch_size=bs
            )
            non_tensor = {
                k: data.non_tensor_batch[k]
                for k in self.reward_keys
                if k in data.non_tensor_batch
            }
            return DataProto(
                batch=rm_td,
                non_tensor_batch=non_tensor,
                meta_info={"reward_extra_keys": list(self.reward_keys)},
            )

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

        # Plumb completion_lengths + answer_extracted for the degeneration head
        # (codex round-5 review): without these, degeneration_penalty_reward
        # falls back to word-count and treats every short response as missing
        # an answer, falsely triggering the short-truncation penalty.
        # Other reward funcs accept **kwargs so the extras are no-ops for them.
        completion_lengths_list = [int(valid_response_length[i].item()) + 1 for i in range(bs)]
        from src.training.rewards import _extract_answer_fallback as _extract_ans_for_degen
        answer_extracted_list = [_extract_ans_for_degen(t) for t in decoded_responses]

        for func_idx, reward_fn in enumerate(self.reward_funcs):
            key = self.reward_keys[func_idx]
            try:
                scores = reward_fn(
                    completions=completions,
                    ground_truth=ground_truths,
                    completion_lengths=completion_lengths_list,
                    answer_extracted=answer_extracted_list,
                )
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
        # Fail-fast: sdc_force_inject is requested but the two-phase rollout repack
        # (_force_inject_rollout) is NODE-SMOKE-REQUIRED and not yet wired. Refuse
        # to launch rather than silently run a NON-inject experiment mislabeled as
        # inject (codex P1). Remove this guard once the repack is implemented +
        # 1-step smoke-tested on the node.
        _algo = getattr(self.config, "algorithm", {})
        _force_inject = bool(getattr(_algo, "sdc_force_inject", False))
        _sdc_mode = str(getattr(_algo, "sdc_mode", ""))
        # Consistency: the INJECT mode is meaningless without force-inject — refuse
        # to run it as a mislabeled non-inject R18b (codex follow-up #2).
        if _sdc_mode == "ROD_MQ_CONTRAST_INJECT" and not _force_inject:
            raise ValueError(
                "sdc_mode=ROD_MQ_CONTRAST_INJECT requires algorithm.sdc_force_inject=true "
                "(else it is just R18b mislabeled — use ROD_MQ_CONTRAST instead)."
            )
        if _force_inject:
            raise NotImplementedError(
                "sdc_force_inject=true but _force_inject_rollout repack is not yet "
                "wired/node-smoke-tested. Implement the DataProto repack against the "
                "live verl runtime (1-step smoke) before launching ROD_MQ_CONTRAST_INJECT."
            )

    def _compute_reward_colocate(self, batch: DataProto) -> DataProto:
        fn = self._sdc_reward_fn
        if fn is None:
            return super()._compute_reward_colocate(batch)
        return fn(batch)

    # ─── CTSD force-inject (Plan v5 §3 H2, mode ROD_MQ_CONTRAST_INJECT) ────────
    # Two-phase rollout: generate → entropy → inject <|meta|> at the max-entropy
    # pre-answer position → regenerate. The DECISION logic is the unit-tested
    # src/training/meta_inject.plan_inject_prefixes; only the DataProto repack +
    # second generate_sequences call is verl-version-specific and lives here.
    #
    # OFF unless config.algorithm.sdc_force_inject is true → every existing mode
    # is byte-identical (Karpathy: surgical). Gated behind A.3 PASS before launch.
    #
    # NODE-SMOKE-REQUIRED: the repack below is written against verl 0.7.1's
    # DataProto/generate_sequences API but MUST be 1-step smoke-tested on the node
    # (per repo convention, task #123 pattern) before any real run — verl internals
    # cannot be exercised in the local CPU env where the core was unit-tested.
    def _force_inject_rollout(self, gen_batch, gen_output):
        """Return a regenerated gen_output with <|meta|> force-injected, or the
        original gen_output unchanged when force-inject is disabled."""
        algo = getattr(self.config, "algorithm", {})
        if not bool(getattr(algo, "sdc_force_inject", False)):
            return gen_output  # default path: no-op, identical to all other modes

        from .meta_inject import plan_inject_prefixes
        tok = self.tokenizer
        meta_open = tok.convert_tokens_to_ids("<|meta|>")
        meta_close = tok.convert_tokens_to_ids("<|/meta|>")

        # (1) extract per-sample prompt ids, response ids, per-token entropy from
        #     phase-1 gen_output; (2) plan_inject_prefixes(...) → phase-2 prompts;
        #     (3) pack non-None prefixes into a DataProto and call
        #     self.actor_rollout_wg.generate_sequences(...) again; (4) merge the
        #     regenerated samples back, leaving None (no-valid-position) samples
        #     as their phase-1 rollout. See plan_inject_prefixes docstring.
        raise NotImplementedError(
            "force-inject repack is node-smoke-required — wire DataProto pack/"
            "unpack against the live verl runtime before enabling sdc_force_inject."
        )


# ─── GFN_OPSD_CONTRAST (R18c, Plan v7.2.7 codex r12-r14 LOCK) ───────────────
#
# Listwise KL distribution matching as PRIMARY aux loss; pairwise cTB stays
# diagnostic only. Verifiable signal — T+ / T- logit, no rubric / judge.
#
# Math (per sequence i in a uid-coherent group of size g = rollout.n):
#   logR_token_i = α (T+ − P_S.detach()) + β (T+ − T-)               (codex r5 + r12 detach)
#   logR_meta_i   = (logR_token_i × meta_content_mask).sum / mask.sum  (length-normalized; codex r13 D14)
#   logP_S_meta_i = (current_log_prob × meta_content_mask).sum / mask.sum
# Group softmax:
#   target_dist  = softmax(logR_meta.detach() / τ)                    (target stops gradient)
#   student_dist = softmax(logP_S_meta / τ)
# Listwise KL (forward, mode-covering — codex r12+r13 confirmed):
#   L_listwise = Σ target × (log target − log student)
# Ref-floor hinge on body (complement_mask = response_mask × (1 − meta_content_mask)):
#   L_hinge    = mean(max(0, current_log_prob − p_ref)^2)             (codex r13 D3)
# Final auxiliary loss:
#   aux        = λ × L_listwise + γ × L_hinge
def compute_sdc_gfn_actor_loss(
    current_log_prob: torch.Tensor,
    model_inputs: dict,
    config=None,
) -> torch.Tensor:
    """R18c GFN_OPSD_CONTRAST: listwise KL primary + ref-floor hinge diagnostic.

    Args
    ----
    current_log_prob : Tensor [B, T]
        Per-token log-prob from veRL's micro-batch forward — this is the only
        tensor that carries a gradient into the actor.
    model_inputs : dict
        Carries the SDC tensors (see `verl_sdc_utils.compute_sdc_gdpo_advantage`)
        plus the GFN hyperparams transported via `data.batch` (codex r13 D13):
          • sdc_teacher_pos_log_probs [B, T]   (no grad)
          • sdc_teacher_neg_log_probs [B, T]   (no grad)
          • sdc_meta_mask             [B, T]   (tag-inclusive; see note)
          • old_log_probs             [B, T]   (P_S used for attractive gain)
          • response_mask             [B, T]
          • uid                       [B]      np.ndarray / list of group ids
          • sdc_alpha_attr            float    default 0.5
          • sdc_beta_contrast         float    default 0.5
          • sdc_gfn_tau               float    default 1.0
          • sdc_gfn_lambda            float    default 0.1  (production fixed)
          • sdc_gfn_fluency_gamma     float    default 0.01 (ref-floor hinge)
          • sdc_log_ratio_clamp       float    default 10.0
          • sdc_ref_log_probs         [B, T]   optional; needed for hinge term

    Returns
    -------
    Tensor (0-dim) — scalar aux loss to add to `pg_loss` in the ppo_loss hook.

    Notes
    -----
    - `meta_content_mask` here uses `sdc_meta_mask` directly (tag-inclusive),
      matching the R18a/R18b fallback. The codex r7 exclusion of tag positions
      lives in `verl_sdc_utils._meta_mask_from_token_ids` and is consistent
      across both code paths.
    - All non-`current_log_prob` tensors are `.detach()`-ed inside `logR` per
      codex r12 — gradient flows ONLY through `current_log_prob` via
      `log_P_S_meta` and the body hinge.
    - On any non-finite aux, we fall back to a zero scalar so a single bad
      microbatch never poisons the optimizer.
    """
    # Tolerate either dict or TensorDict for `model_inputs`. TensorDicts do
    # not store arbitrary Python objects, so uid is typically transported in
    # `non_tensor_batch` and copied here for the test path. When uid is not
    # available at all, we fall back to "single group across the microbatch",
    # which is a no-op for L_listwise (one group = degenerate softmax).
    teacher_pos = model_inputs["sdc_teacher_pos_log_probs"]
    teacher_neg = model_inputs["sdc_teacher_neg_log_probs"]
    meta_mask = model_inputs["sdc_meta_mask"]
    old_log_probs = model_inputs["old_log_probs"]
    response_mask = model_inputs["response_mask"]
    try:
        uid = model_inputs["uid"]
    except KeyError:
        uid = ["__single_group__"] * int(current_log_prob.size(0))

    alpha = float(model_inputs.get("sdc_alpha_attr", 0.5))
    beta = float(model_inputs.get("sdc_beta_contrast", 0.5))
    tau = float(model_inputs.get("sdc_gfn_tau", 1.0))
    lambda_listwise = float(model_inputs.get("sdc_gfn_lambda", 0.1))
    gamma_hinge = float(model_inputs.get("sdc_gfn_fluency_gamma", 0.01))
    clamp = float(model_inputs.get("sdc_log_ratio_clamp", 10.0))

    # ── Objective dispatch (R21 direction A — additive, zero-touch) ────────
    # Transport mirrors the proven `mode` path: read the per-batch marker first
    # (test path supplies it via the dict model_inputs), then the module-level
    # `_ACTIVE_SDC_CONTEXT` cache populated deterministically in main_task
    # (the same mechanism `_sdc_mode` uses), defaulting to "listwise_kl".
    #   • "listwise_kl"  → GFN_OPSD_CONTRAST (R18c) — EXACT prior behavior,
    #                       numerically byte-identical (R18c never sets the key).
    #   • "pairwise_ctb" → STABLE_GFN (R21, A) — Z-free pairwise contrastive
    #                       Trajectory Balance + frozen_ref baseline +
    #                       reward-temperature.
    def _ctx(key, default):
        try:
            v = model_inputs.get(key, None)
        except Exception:
            v = None
        if v is None:
            try:
                v = _ACTIVE_SDC_CONTEXT.get(key, None)
            except Exception:
                v = None
        return default if v is None else v

    gfn_objective = str(_ctx("sdc_gfn_objective", "listwise_kl"))
    gfn_reward_baseline = str(_ctx("sdc_gfn_reward_baseline", "none"))
    # Reward-temperature T_R for pairwise cTB (target = logR / T_R; student is
    # NOT divided — per Plan direction-A spec). Reuses sdc_gfn_tau ONLY for the
    # listwise softmax; cTB uses its own T_R so the two objectives stay
    # independent and single-variable.
    reward_temp = float(_ctx("sdc_reward_temperature", 1.0))

    device = current_log_prob.device
    dtype = current_log_prob.dtype

    if tau <= 0:
        raise ValueError(f"sdc_gfn_tau must be > 0, got {tau}")
    if gfn_objective == "pairwise_ctb" and reward_temp <= 0:
        raise ValueError(
            f"sdc_reward_temperature must be > 0 for pairwise_ctb, got {reward_temp}"
        )

    # logR with both detach-able terms detached (codex r12 D2 + r5 sign).
    teacher_pos_d = teacher_pos.detach().to(device=device, dtype=dtype)
    teacher_neg_d = teacher_neg.detach().to(device=device, dtype=dtype)
    old_log_probs_d = old_log_probs.detach().to(device=device, dtype=dtype)

    gain_attr = (teacher_pos_d - old_log_probs_d).clamp(-clamp, clamp)
    gain_contrast = (teacher_pos_d - teacher_neg_d).clamp(-clamp, clamp)
    logR_token = alpha * gain_attr + beta * gain_contrast  # [B, T]

    # meta_content_mask: accept the tag-inclusive mask (see notes above).
    meta_content_mask = meta_mask.to(device=device, dtype=dtype)
    denom = meta_content_mask.sum(-1).clamp_min(1.0)

    # Length-normalized scores per rollout (codex r13 D14).
    logR_meta = (logR_token * meta_content_mask).sum(-1) / denom  # [B]
    log_P_S_meta = (current_log_prob * meta_content_mask).sum(-1) / denom  # [B], grad flows

    # ── Frozen-ref logprob (shared: cTB baseline + body hinge) ────────────
    # Fetched ONCE here. Try sdc-prefixed key first (legacy/test path), then
    # veRL native `ref_log_prob` (codex r2 #5). Listwise numerics are unchanged
    # — the hinge below reuses this exact tensor, same as before the move.
    p_ref = model_inputs.get("sdc_ref_log_probs", None)
    if p_ref is None:
        p_ref = model_inputs.get("ref_log_prob", None)
    if p_ref is not None:
        p_ref_t = p_ref.detach().to(device=device, dtype=dtype)
        log_P_ref_meta = (p_ref_t * meta_content_mask).sum(-1) / denom  # [B], no grad
    else:
        p_ref_t = None
        log_P_ref_meta = torch.zeros_like(log_P_S_meta)

    # ── Shared uid-group construction (codex r13 D13 + r12 D2) ────────────
    # Microbatch is uid-coherent only if rollout.n == ppo_micro_batch_size_per_gpu
    # AND balance_batch=False (codex r12 D2). YAML must enforce both.
    if hasattr(uid, "tolist"):
        uid_list = uid.tolist()
    else:
        uid_list = list(uid)

    groups: dict = {}
    for i, u in enumerate(uid_list):
        groups.setdefault(u, []).append(i)

    if gfn_objective == "pairwise_ctb":
        # ── R21 direction A: Z-free pairwise contrastive Trajectory Balance ──
        # GFlowNet TB over the meta region: logZ + logP_S(τ) = logR(τ). logZ is
        # intractable; for two rollouts i,j sharing a uid (⇒ same prompt ⇒ same
        # logZ) it CANCELS in the pairwise difference:
        #     (s_i − s_j) == (r_i − r_j)/T_R
        #   s_i = log_P_S_meta_i − [frozen_ref baseline]   (grad via current_lp)
        #   r_i = logR_meta_i                              (detached; reward-temp
        #                                                   applied to the target
        #                                                   difference only)
        # frozen_ref baseline = −log_P_ref_meta (detached). codex review
        # (gpt-5.5 NEEDS_WORK pt.1): this is NOT a pure variance-reduction
        # control variate — because (logP_ref_i − logP_ref_j) does NOT cancel
        # in the pairwise difference, it CHANGES the TB fixed point. That is
        # the INTENDED Stable-GFN relative-TB form: the constraint becomes
        #   (logP_S_i − logP_S_j) − (logP_ref_i − logP_ref_j) == (r_i−r_j)/T_R
        # i.e. the policy's *improvement over the frozen ref* (not its raw
        # log-flow) is matched to the reward difference — anchoring the
        # absolute scale to the SFT ref. With baseline="none" this reduces to
        # plain raw-policy pairwise TB. Student is NOT divided by T_R
        # (Plan direction-A spec: target = logR/T_R, student raw).
        use_ref_baseline = (gfn_reward_baseline == "frozen_ref") and (p_ref_t is not None)
        s_all = log_P_S_meta - (log_P_ref_meta if use_ref_baseline else 0.0)  # [B], grad
        r_all = (logR_meta.detach() / reward_temp)  # [B], no grad

        ctb_terms: list = []
        for u, idx in groups.items():
            if len(idx) < 2:
                continue  # need ≥ 2 rollouts for a pair
            # Same robust mask as listwise: drop empty-meta rollouts so the
            # pairwise residual is never formed on a zero-length meta region.
            idx_filtered = [i for i in idx if meta_content_mask[i].sum() > 0]
            if len(idx_filtered) < 2:
                continue
            idx_t = torch.tensor(idx_filtered, device=device, dtype=torch.long)
            s_g = s_all.index_select(0, idx_t)  # [g], grad flows
            r_g = r_all.index_select(0, idx_t)  # [g], no grad
            # Upper-triangular (i<j) pairwise residuals → mean squared error.
            ds = s_g.unsqueeze(1) - s_g.unsqueeze(0)  # [g,g]
            dr = r_g.unsqueeze(1) - r_g.unsqueeze(0)  # [g,g], no grad
            resid = ds - dr  # [g,g]
            g = resid.size(0)
            tri = torch.triu(torch.ones(g, g, device=device, dtype=torch.bool), diagonal=1)
            ctb_terms.append((resid[tri] ** 2).mean())

        if not ctb_terms:
            L_primary = torch.zeros((), device=device, dtype=dtype)
        else:
            L_primary = torch.stack(ctb_terms).mean()
        L_listwise = torch.zeros((), device=device, dtype=dtype)  # diag-only when cTB
        L_ctb = L_primary
        groups_used = len(ctb_terms)
    else:
        # ── listwise_kl: GFN_OPSD_CONTRAST (R18c) — UNCHANGED, byte-identical ─
        kl_terms: list = []
        for u, idx in groups.items():
            if len(idx) < 2:
                continue  # need ≥ 2 candidates for a listwise KL term
            # codex r2 #3: drop rollouts with empty meta from THIS listwise group
            # (previous version included them as score 0, contaminating target softmax).
            idx_filtered = [i for i in idx if meta_content_mask[i].sum() > 0]
            if len(idx_filtered) < 2:
                continue
            idx_t = torch.tensor(idx_filtered, device=device, dtype=torch.long)
            logR_group = logR_meta.index_select(0, idx_t)  # [g], no grad (logR is built from detached terms)
            log_P_S_group = log_P_S_meta.index_select(0, idx_t)  # [g], grad flows

            target_dist = torch.softmax(logR_group.detach() / tau, dim=-1)  # explicit .detach() safety
            student_log_dist = torch.log_softmax(log_P_S_group / tau, dim=-1)
            # Forward KL(target ‖ student) = Σ target × (log target − log student)
            kl = (target_dist * (target_dist.clamp_min(1e-9).log() - student_log_dist)).sum()
            kl_terms.append(kl)

        if not kl_terms:
            L_primary = torch.zeros((), device=device, dtype=dtype)
        else:
            L_primary = torch.stack(kl_terms).mean()
        L_listwise = L_primary
        L_ctb = torch.zeros((), device=device, dtype=dtype)  # diag-only when listwise
        groups_used = len(kl_terms)

    # ── Ref-floor hinge on body (codex r13 D3) — reuses the shared p_ref_t ──
    if p_ref_t is not None and gamma_hinge > 0:
        complement_mask = (response_mask.to(device=device, dtype=dtype)
                           * (1.0 - meta_content_mask))
        hinge_token = torch.clamp_max(current_log_prob - p_ref_t, 0.0) ** 2
        denom_body = complement_mask.sum().clamp_min(1.0)
        L_hinge = (hinge_token * complement_mask).sum() / denom_body
    else:
        L_hinge = torch.zeros((), device=device, dtype=dtype)

    # lambda_listwise (sdc_gfn_lambda) is the coefficient on the PRIMARY aux
    # loss for BOTH objectives (listwise KL or pairwise cTB) — single shared
    # delivery-strength knob so the objective swap stays single-variable.
    aux_loss = lambda_listwise * L_primary + gamma_hinge * L_hinge

    # codex review (gpt-5.5 NEEDS_WORK pt.4): a pairwise_ctb microbatch with
    # NO usable uid group (incomplete / singleton groups from a wrong
    # rollout.n / balance_batch) silently trains ZERO cTB signal while the run
    # still "works". Surface it loudly: a per-batch wandb flag + a one-time
    # process warning. (The startup config guard in main_task fail-fasts the
    # actual misconfig; this catches residual per-batch degeneracy.)
    ctb_inactive = bool(gfn_objective == "pairwise_ctb" and groups_used == 0)
    if ctb_inactive and not _CTB_INACTIVE_WARNED["done"]:
        _CTB_INACTIVE_WARNED["done"] = True
        print(
            "[SDC][GFN][WARN] pairwise_ctb produced 0 usable uid groups in a "
            "microbatch — cTB delivery is INACTIVE. Check rollout.n == "
            "ppo_micro_batch_size_per_gpu and trainer.balance_batch=False. "
            "(this warning prints once; watch wandb sdc_gfn_ctb_inactive)"
        )

    # codex r2 #7: surface primary / hinge / logR diagnostics for wandb.
    diag = {
        "sdc_gfn_objective": gfn_objective,
        "sdc_gfn_kl_listwise": float(L_listwise.detach()),
        "sdc_gfn_ctb_loss": float(L_ctb.detach()),
        "sdc_gfn_ctb_inactive": ctb_inactive,
        "sdc_gfn_ref_hinge": float(L_hinge.detach()),
        "sdc_gfn_logR_mean": float(logR_meta.detach().mean()) if logR_meta.numel() else 0.0,
        "sdc_gfn_logR_std": float(logR_meta.detach().std()) if logR_meta.numel() > 1 else 0.0,
        "sdc_gfn_groups_used": int(groups_used),
    }

    if not torch.isfinite(aux_loss):
        # Single bad microbatch must never poison the optimizer. Log via wandb
        # if available; otherwise silently zero-out (codex r13 finite check).
        return torch.zeros((), device=device, dtype=dtype), diag

    return aux_loss, diag


def _patch_actor_loss_for_gfn():
    """codex r13 D13: hook `ppo_loss` in both losses.py AND engine_workers.py.

    Active-path discovery: when `trainer.use_legacy_worker_impl: disable`, the
    new `engine_workers.py` imports `ppo_loss` once at module load. Patching
    `losses_mod.ppo_loss` alone does not retroactively rebind the symbol the
    engine workers already captured — we must patch both.

    The mode dispatch is read from `data.batch['_sdc_mode']` (transported via
    DataProto), NOT from actor config. This keeps the hook plug-and-play with
    every existing config and survives veRL's actor-config validators.
    """
    try:
        import verl.workers.utils.losses as losses_mod  # type: ignore
    except (ImportError, AttributeError) as exc:
        print(f"[SDC][GFN] skipped ppo_loss hook (losses module unavailable): "
              f"{type(exc).__name__}: {exc}")
        return

    try:
        import verl.workers.engine_workers as engine_mod  # type: ignore
    except (ImportError, AttributeError):
        engine_mod = None

    if getattr(losses_mod.ppo_loss, "_sdc_gfn_hook", False):
        return  # already patched

    original_ppo_loss = losses_mod.ppo_loss

    def sdc_gfn_ppo_loss(config, model_output, data, dp_group):
        policy_loss, metrics = original_ppo_loss(config, model_output, data, dp_group)
        # Mode dispatch (codex r13 D13): prefer the explicit per-batch marker
        # `_sdc_mode` if veRL forwarded one; fall back to the module-level
        # `_ACTIVE_SDC_CONTEXT["mode"]` cache populated in main_task. This
        # tolerates TensorDicts that disallow string entries while keeping
        # the plug-and-play marker semantics from the spec.
        sdc_mode = ""
        try:
            if hasattr(data, "batch") and "_sdc_mode" in getattr(data, "batch", {}):
                sdc_mode = str(data.batch["_sdc_mode"])
        except Exception:
            sdc_mode = ""
        if not sdc_mode:
            try:
                sdc_mode = _ACTIVE_SDC_CONTEXT.get("mode", "")
            except Exception:
                sdc_mode = ""
        # R21 direction A is additive: STABLE_GFN routes through the SAME
        # compute_sdc_gfn_actor_loss, which dispatches internally on
        # sdc_gfn_objective. GFN_OPSD_CONTRAST (R18c) runtime path is
        # byte-identical (objective defaults to "listwise_kl").
        if sdc_mode in ("GFN_OPSD_CONTRAST", "STABLE_GFN", "STABLE_GFN_C2FIX"):
            try:
                # codex r2 #4 fix: veRL 0.7.1 ppo_loss signature passes
                # `data` as TensorDict (NOT DataProto). The .batch attribute
                # does not exist; use `data` directly for subscript access.
                # Both TensorDict and DataProto support .get()/__getitem__.
                aux_data = data.batch if hasattr(data, "batch") else data
                # codex r2 #4 fix: pad model_output["log_probs"] to [B, T] form
                # to match how original_ppo_loss processes log_prob.
                try:
                    from verl.workers.utils.losses import no_padding_2_padding
                    current_lp = no_padding_2_padding(model_output["log_probs"], data)
                except (ImportError, AttributeError, KeyError):
                    current_lp = model_output["log_probs"]  # fallback for tests
                result = compute_sdc_gfn_actor_loss(
                    current_log_prob=current_lp,
                    model_inputs=aux_data,
                    config=config,
                )
                # codex r2 #7 fix: accept either tuple (aux, diag) or scalar aux.
                if isinstance(result, tuple):
                    aux, gfn_diag = result
                else:
                    aux, gfn_diag = result, {}
                policy_loss = policy_loss + aux
                metrics["sdc_gfn_aux_loss"] = float(aux.detach())
                for k, v in gfn_diag.items():
                    metrics[k] = v
            except Exception as e:
                # Never crash training on aux-loss failure; surface the diag.
                metrics["sdc_gfn_aux_loss_error"] = f"{type(e).__name__}: {e}"
        return policy_loss, metrics

    sdc_gfn_ppo_loss._sdc_gfn_hook = True  # type: ignore[attr-defined]
    losses_mod.ppo_loss = sdc_gfn_ppo_loss
    if engine_mod is not None and hasattr(engine_mod, "ppo_loss"):
        engine_mod.ppo_loss = sdc_gfn_ppo_loss
        # codex r13 strict assertion: refuse to proceed if either path is unhooked.
        assert getattr(engine_mod.ppo_loss, "_sdc_gfn_hook", False), (
            "GFN hook injection failed in engine_workers — refusing to start"
        )
    print("[SDC][GFN] ppo_loss hooked for GFN_OPSD_CONTRAST (R18c, Plan v7.2.7)")


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

    # GFN_OPSD_CONTRAST (R18c) ppo_loss hook — installed once per process.
    # Idempotent: subsequent calls early-return on the `_sdc_gfn_hook` marker.
    _patch_actor_loss_for_gfn()

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
    # R21 direction A (STABLE_GFN): stash the GFN-objective hyperparams into
    # the module-level context using the SAME deterministic transport `mode`
    # uses (the GFN aux path runs in a remote worker that does not receive the
    # hydra config; reading via _ACTIVE_SDC_CONTEXT avoids the murky batch
    # transport). Existing modes never set these keys in YAML → the defaults
    # ("listwise_kl"/"none"/1.0) preserve GFN_OPSD_CONTRAST byte-identically.
    _ACTIVE_SDC_CONTEXT["sdc_gfn_objective"] = str(
        alg_cfg.get("sdc_gfn_objective", "listwise_kl")
    )
    _ACTIVE_SDC_CONTEXT["sdc_gfn_reward_baseline"] = str(
        alg_cfg.get("sdc_gfn_reward_baseline", "none")
    )
    _ACTIVE_SDC_CONTEXT["sdc_reward_temperature"] = float(
        alg_cfg.get("sdc_reward_temperature", 1.0)
    )
    # Arm-2 parameterized teacher-prompt slot (deliverable #2). Resolved ONCE
    # at launch and stashed via the SAME deterministic transport. Default
    # "r10v2_baseline" -> "" prefix => byte-identical teacher conditioning for
    # every existing mode (none of which read this key) and for an Arm-2 run
    # that has not yet picked a strengthened set from the FROZEN G2 decision.
    _tp_set = str(alg_cfg.get("sdc_teacher_prompt_set", "r10v2_baseline"))
    _ACTIVE_SDC_CONTEXT["sdc_teacher_prompt_set"] = _tp_set
    # ── G2 PER-TEACHER SLOTS (deliverable #2, 2026-05-19) ──────────────────
    # PRECEDENCE (codex-converged): an explicitly-set per-teacher key wins;
    # otherwise it INHERITS the legacy shared `sdc_teacher_prompt_set` (so the
    # pre-existing single-slot config + every prior run is byte-identical).
    # `None` sentinel distinguishes "key absent -> inherit legacy" from "key
    # explicitly set to r10v2_baseline -> use baseline" (both yield "" prefix,
    # but the distinction keeps the back-compat semantics explicit & logged).
    _pos_set_raw = alg_cfg.get("sdc_position_teacher_prompt_set", None)
    _con_set_raw = alg_cfg.get("sdc_content_teacher_prompt_set", None)
    _pos_set = str(_pos_set_raw) if _pos_set_raw is not None else _tp_set
    _con_set = str(_con_set_raw) if _con_set_raw is not None else _tp_set
    _ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_set"] = _pos_set
    _ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_set"] = _con_set
    if mode == "ROD_PT2_E21CTRL":
        # Legacy shared prefix kept resolved for back-compat / logging.
        _ACTIVE_SDC_CONTEXT["sdc_teacher_prompt_prefix"] = (
            _resolve_teacher_prompt_prefix(_tp_set)
        )
        _ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_prefix"] = (
            _resolve_teacher_prompt_prefix(_pos_set)
        )
        _ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_prefix"] = (
            _resolve_teacher_prompt_prefix(_con_set)
        )
        print(
            "[SDC][Arm2] per-teacher prompt slots: position=%s "
            "(prefix_len=%d) content=%s (prefix_len=%d) "
            "[legacy shared=%s, precedence: explicit per-teacher > shared]"
            % (
                _pos_set,
                len(_ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_prefix"]),
                _con_set,
                len(_ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_prefix"]),
                _tp_set,
            )
        )
    else:
        _ACTIVE_SDC_CONTEXT["sdc_teacher_prompt_prefix"] = ""
        _ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_prefix"] = ""
        _ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_prefix"] = ""
    print(
        "[SDC][GFN] objective=%s reward_baseline=%s reward_temperature=%s"
        % (
            _ACTIVE_SDC_CONTEXT["sdc_gfn_objective"],
            _ACTIVE_SDC_CONTEXT["sdc_gfn_reward_baseline"],
            _ACTIVE_SDC_CONTEXT["sdc_reward_temperature"],
        )
    )
    # codex review (gpt-5.5 NEEDS_WORK pt.4): grouped pairwise cTB is only
    # well-posed when each microbatch is uid-coherent. FAIL FAST at launch on
    # the config invariant rather than silently training a degenerate (0-group)
    # objective for 300 steps. Mirrors the R18c listwise constraint (codex r12
    # D2) but now hard-enforced for the cTB delivery path.
    if mode in ("STABLE_GFN", "STABLE_GFN_C2FIX") and _ACTIVE_SDC_CONTEXT["sdc_gfn_objective"] == "pairwise_ctb":
        _arr = config.actor_rollout_ref
        _n = int(_arr.rollout.n)
        _micro = int(_arr.actor.ppo_micro_batch_size_per_gpu)
        _bal = bool(config.trainer.get("balance_batch", True))
        if _n != _micro or _bal:
            raise ValueError(
                "STABLE_GFN+pairwise_ctb requires uid-coherent microbatches: "
                f"rollout.n ({_n}) must equal "
                f"actor.ppo_micro_batch_size_per_gpu ({_micro}) AND "
                f"trainer.balance_batch must be False (got {_bal}). "
                "Otherwise grouped cTB silently degenerates to 0 usable groups."
            )
        print(
            "[SDC][GFN] STABLE_GFN+pairwise_ctb uid-coherence OK "
            f"(n={_n}==micro={_micro}, balance_batch=False)"
        )
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
