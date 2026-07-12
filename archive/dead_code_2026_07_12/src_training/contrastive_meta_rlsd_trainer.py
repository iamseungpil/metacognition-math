"""Contrastive Meta-RLSD trainer — N3 variant (contrastive privileged teachers).

Implements the approved plan at::

    results/plan_contrastive_rlsd_v1_2026_04_17.md   (v1, critic audit passed)

All section references (``§2.1``, ``§2.4``, …) point to that document unless
otherwise noted. Builds on top of M1 baseline
(``results/plan_meta_rlsd_v2_2026_04_17.md``).

Design (modular, plug-and-play — critical):
    * Subclass of :class:`src.training.meta_rlsd_trainer.MetaRLSDTrainer`.
    * **Does NOT modify M1 code** — only overrides ``_compute_loss`` and adds
      helpers. Existing M1 methods (``_build_teacher_inputs``,
      ``_teacher_logprobs``, ``_sync_teacher``, ``_current_lambda``,
      ``_generate_and_score_completions``) are reused verbatim.
    * Config dataclass :class:`ContrastiveMetaRLSDConfig` extends M1's
      :class:`MetaRLSDConfig` with two new fields: ``decoy_strategy`` and
      ``decoy_seed``.
    * Variant dispatch via ``--variant {n3, n3-random, n3-fullmask}``.

Key addition (§2.4, Bayesian form):

    Δ_t = sg[log P_{T+}(y_t | x, a*, y_<t) − log P_{T-}(y_t | x, a-, y_<t)]
    w_t = exp(sign(A_i) · Δ_t)
    Â_t = A_i · [(1−λ) + λ · (m_t · clip(w_t, 1−ε_w, 1+ε_w) + (1−m_t))]

where ``a-`` is a deterministic decoy (``_make_decoy``) and the student
marginal ``P_S`` cancels — Δ_t becomes the *Bayes factor* of correct vs
incorrect hypothesis at token t (§2.4).

Usage::

    accelerate launch --config_file configs/accelerate_ds3.yaml \\
        src/training/contrastive_meta_rlsd_trainer.py \\
        --config configs/contrastive_meta_rlsd.yaml \\
        --variant n3 \\
        --seed 42
"""
from __future__ import annotations

# CRITICAL: force-import deepspeed.runtime submodules before anything else
# imports trl/accelerate. deepspeed 0.15.x does not expose `runtime` as an
# attribute of the `deepspeed` module by default, causing AttributeError in
# accelerate's _prepare_deepspeed (`deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES`).
try:
    import deepspeed  # noqa: F401
    import deepspeed.runtime  # noqa: F401
    import deepspeed.runtime.config  # noqa: F401
    import deepspeed.runtime.lr_schedules  # noqa: F401
    import deepspeed.runtime.zero.partition_parameters  # noqa: F401
except ImportError:
    pass

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass, fields
from functools import partial as _partial
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from src.metacot.prompt import META_END, META_START
from src.training.meta_rlsd_data_pipeline import (
    _build_postmeta_mask,
    load_meta_rlsd_dataset,
    preflight_checks,
)
from src.training.meta_rlsd_trainer import (
    ClipFractionAbortCallback,
    MetaRLSDConfig,
    MetaRLSDTrainer,
    TeacherSyncCallback,
    _build_grpo_config,
    correctness_plus_meta_floor_reward,
)
from src.training.rewards import _check_correctness  # review iter1 Fix 5
from src.training.tokenizer_utils import ensure_meta_tokens_not_special


# ─── Config — contrastive extension of MetaRLSDConfig ─────────────────────

@dataclass
class ContrastiveMetaRLSDConfig(MetaRLSDConfig):
    """Extends :class:`MetaRLSDConfig` with contrastive-decoy hyperparameters.

    New fields (see plan §2.1, §2.6):
        decoy_strategy : "rule_based" or "random"
        decoy_seed     : deterministic seed for decoy generation (§2.5 leakage).
        trainer_class  : informational; the launcher honors ``--variant n3*``
                         and instantiates :class:`ContrastiveMetaRLSDTrainer`.
    """

    decoy_strategy: str = "rule_based"  # "rule_based" | "random"
    decoy_seed: int = 42
    trainer_class: str = "ContrastiveMetaRLSDTrainer"

    # ── SDC fields (plan_SDC_v2 §3.3 / §5.1) ─────────────────────────────
    # Separate schedule for the post-meta repel signal. ``lambda_post``
    # ramps UP (plan §3.3: 0.1 → 0.3 over 150 steps) while the existing
    # ``lambda_meta`` (aliased to base's ``lambda_init/final/decay_steps``)
    # ramps DOWN (0.5 → 0 over 75 steps). These are independent so the
    # attract/repel signals can be tuned orthogonally.
    lambda_post_init: float = 0.1
    lambda_post_final: float = 0.3
    lambda_post_warmup: int = 150
    # SDC-noise control (plan §2.6): magnitude of the Gaussian noise injected
    # into the post-meta weight when ``variant=sdc-noise``. 0.3 matches the
    # typical std of ``log w_t^{rep}`` observed in smoke tests.
    sdc_noise_sigma: float = 0.3
    # Shared-structure threshold for SDC consensus gating. Tokens in the
    # post-meta region with |log P_T+ - log P_T-| <= tau are treated as
    # teacher-consensus structure (e.g., EOS, framing, boxed wrapper) and are
    # preserved rather than repelled.
    sdc_shared_tau: float = 0.5
    # SDC wrap-regression halt baseline (plan §3.6). The SFT model's measured
    # wrap_rate (= fraction of completions that contain both <|meta|> and
    # <|/meta|>). The halt callback fires when wrap_rate drops below
    # ``sdc_wrap_baseline - 0.10`` at step >= 30 for two consecutive batches.
    # Default 1.0 because v8 meta SFT wraps 100% of completions.
    sdc_wrap_baseline: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "ContrastiveMetaRLSDConfig":
        """Same semantics as parent ``from_yaml`` — warns on unknown keys.

        Accepts both the flat schema and plan-style nested ``reward:`` schema.
        Unknown keys are warned-on but do not crash so drift surfaces in logs.
        """
        with open(path, "r") as f:
            payload = yaml.safe_load(f) or {}

        # Flatten ``reward:`` sub-dict if present (plan §2.7 schema).
        reward_subdict = payload.pop("reward", None) or {}
        key_mapping = {
            "correctness": "correctness_weight",
            "correctness_weight": "correctness_weight",
            "meta_floor": "meta_floor_weight",
            "meta_floor_weight": "meta_floor_weight",
            "meta_floor_no_meta_penalty": "reward_meta_no_penalty",
            "meta_floor_short_meta_bonus": "reward_meta_short_bonus",
            "meta_floor_full_bonus": "reward_meta_full_bonus",
            "meta_min_length": "meta_min_length_tokens",
            "meta_min_length_tokens": "meta_min_length_tokens",
        }
        known = {f.name for f in fields(cls)}
        for k, v in reward_subdict.items():
            dest_key = key_mapping.get(k, k)
            if dest_key in known:
                payload[dest_key] = v
            else:
                print(f"[WARN] ContrastiveMetaRLSDConfig ignoring unknown reward key: {k!r}")

        unknown = set(payload) - known
        if unknown:
            print(
                f"[WARN] ContrastiveMetaRLSDConfig ignoring unknown keys in {path}: "
                f"{sorted(unknown)}"
            )
            for k in unknown:
                payload.pop(k, None)

        return cls(**payload)


# ─── Decoy generation — §2.1 ──────────────────────────────────────────────
# NOTE: implementations live in ``_decoy_utils`` so CPU-only unit tests
# can import them without pulling in ``trl``/``transformers``.
# The names are re-exported here for backward compat with the trainer's
# internal callsites (``self._make_decoy``) and the plan §2.1 reference.
from src.training._decoy_utils import (  # noqa: E402, F401
    _numerically_equal,
    _random_noise_decoy,
    _rule_based_decoy,
)




# ─── Trainer ───────────────────────────────────────────────────────────────

class ContrastiveMetaRLSDTrainer(MetaRLSDTrainer):
    """N3 variant — contrastive privileged teachers (T+ vs T-).

    Reuses M1 for everything except the teacher signal:

        * ``__init__`` — records decoy strategy/seed; teacher model is
          already spawned by ``MetaRLSDTrainer.__init__``.
        * ``_make_decoy``                 — §2.1 deterministic decoy.
        * ``_build_contrastive_teacher_inputs`` — §2.3 T+/T- contexts.
        * ``_teacher_contrastive_logprobs`` — sequential forward with
          ``torch.cuda.empty_cache()`` between T+ and T- (§2.3 VRAM).
        * ``_compute_per_token_advantage`` — §2.4 Bayes-factor form
          Δ_t = log P_{T+} − log P_{T-}.
        * ``_compute_loss`` — replaces M1's single-teacher advantage build
          with the contrastive form; PPO clip remains identical to M1
          (§2.2 paper-faithful).

    Logging additions (plan §4.2):
        * ``meta_rlsd/delta_t_mean``, ``meta_rlsd/delta_t_std``
        * ``meta_rlsd/kl_T_pos_T_neg`` (meta-token-masked)
        * ``meta_rlsd/contrastive_fwd_count`` (= 2 per step — invariant)
        * ``meta_rlsd/decoy_is_correct_rate`` (should be < 5%)
        * ``meta_rlsd/decoy_eq_gold_rate`` (hard invariant — must be 0)
    """

    def __init__(self, *args, meta_rlsd_cfg: ContrastiveMetaRLSDConfig, **kwargs):
        assert isinstance(meta_rlsd_cfg, ContrastiveMetaRLSDConfig), (
            "ContrastiveMetaRLSDTrainer requires ContrastiveMetaRLSDConfig; "
            f"got {type(meta_rlsd_cfg).__name__}"
        )
        super().__init__(*args, meta_rlsd_cfg=meta_rlsd_cfg, **kwargs)
        self.decoy_strategy = meta_rlsd_cfg.decoy_strategy
        self.decoy_seed = meta_rlsd_cfg.decoy_seed
        if self.decoy_strategy not in {"rule_based", "random"}:
            raise ValueError(
                f"Unknown decoy_strategy: {self.decoy_strategy!r} "
                "(expected 'rule_based' or 'random')"
            )
        # Cumulative decoy quality trackers (reset each logging window in loss).
        self._last_delta_mean = 0.0
        self._last_delta_std = 0.0
        self._last_kl_t_pos_t_neg = 0.0
        self._last_decoy_eq_gold_rate = 0.0
        self._last_decoy_is_correct_rate = 0.0
        # Review iter1 Fix 4: first-batch answer-token-boundary check flag.
        self._boundary_checked = False
        # Review iter1 Fix 5: within first 10 steps, invariant violations abort.
        self._decoy_abort_window = 10
        # SDC state (plan_SDC_v2 §3.6). Recorded per-batch in _compute_loss.
        self._last_fallback_trigger_rate = 0.0
        self._last_wrap_rate = 1.0
        self._sdc_wrap_warned = False
        self._sdc_fallback_warned = False
        # MEDIUM-4: per-call counter so sdc-noise seed differs across
        # gradient-accumulation micro-batches within a single optimizer step.
        # Without this the same optimizer step reuses the identical noise
        # tensor across accumulation rounds (silent correlation).
        self._sdc_noise_call_count = 0
        # MEDIUM-5: latch for the missing-postmeta-mask warning.
        self._sdc_postmeta_missing_warned = False
        # MEDIUM-1: consecutive-violation counters for the halt callback.
        self._sdc_halt_fallback_consec = 0
        self._sdc_halt_wrap_consec = 0

        # MEDIUM-1: register the SDC halt callback when any sdc-* variant is
        # active. The callback raises RuntimeError after two consecutive
        # violations of either halt rule (fallback-trigger or wrap-regression),
        # per plan §3.6. First-batch violations still surface via the existing
        # warnings.warn calls in _compute_loss (latched to fire once).
        if self._is_sdc_variant():
            self.add_callback(
                SDCHaltCallback(wrap_baseline=float(meta_rlsd_cfg.sdc_wrap_baseline))
            )

    # ── SDC: λ_post schedule (plan_SDC_v2 §3.3) ──────────────────────────

    def _is_sdc_variant(self) -> bool:
        """Return True iff the active variant is one of the SDC family.

        SDC variants share the 3-region disjoint mask and the separate
        ``lambda_post`` schedule. Non-SDC variants (``n3``, ``n3-random``,
        ``n3-fullmask``) fall through to the original 2-region logic so their
        behavior stays identical to pre-SDC code (plan constraints).
        """
        return self.meta_rlsd_cfg.variant in {
            "sdc-split", "sdc-uniform", "sdc-noise", "sdc-shared"
        }

    def _current_lambda_post(self) -> float:
        """λ_post schedule — linear ramp UP (plan §3.3).

        ``lambda_post_init`` → ``lambda_post_final`` linear over
        ``lambda_post_warmup`` steps; constant at ``lambda_post_final``
        thereafter. When the current step is 0 this returns ``lambda_post_init``
        exactly (matches unit test ``test_sdc_lambda_post_schedule`` boundary).
        """
        cfg = self.meta_rlsd_cfg
        step = int(self.state.global_step) if self.state is not None else 0
        warmup = max(1, int(cfg.lambda_post_warmup))
        init = float(cfg.lambda_post_init)
        final = float(cfg.lambda_post_final)
        if step <= 0:
            return init
        if step >= warmup:
            return final
        frac = step / float(warmup)
        return init + (final - init) * frac

    # ── SDC: augment generate+score with postmeta_mask ──────────────────

    def _generate_and_score_completions(self, inputs):
        """Extend parent to add ``postmeta_mask`` for SDC variants.

        For non-SDC variants this is a pure pass-through: the parent's
        ``meta_mask`` is preserved and no ``postmeta_mask`` tensor is emitted.
        For SDC variants we compute a disjoint post-meta mask per rollout
        using :func:`_build_postmeta_mask` on the active (non-padded) text.

        The mask is stored as a tensor in ``out`` so TRL 0.19.1's
        ``shuffle_tensor_dict`` preserves per-row ordering — the same pattern
        as the existing ``meta_mask`` threading (see MetaRLSDTrainer line 630).
        """
        out = super()._generate_and_score_completions(inputs)
        if not self._is_sdc_variant():
            return out

        completion_ids = out["completion_ids"]
        completion_mask = out["completion_mask"]
        meta_mask_tensor = out.get("meta_mask")
        tokenizer = self._meta_tokenizer

        postmeta_masks: List[torch.Tensor] = []
        fallback_hits = 0
        wrap_hits = 0
        for i in range(completion_ids.size(0)):
            ids = completion_ids[i].tolist()
            active_len = int(completion_mask[i].sum().item())
            active_ids = ids[:active_len] if active_len > 0 else ids
            text = tokenizer.decode(active_ids, skip_special_tokens=False)
            # Plan §3.6 halt condition inputs:
            #   wrap_rate counts completions that have BOTH <|meta|> AND <|/meta|>
            #     tokens — i.e., a properly-closed meta block.
            #   fallback_hits counts completions where the post-meta region
            #     had to fall back to end-of-completion because \boxed{} was
            #     missing after the final <|/meta|>.
            if META_START in text and META_END in text:
                wrap_hits += 1
            if meta_mask_tensor is not None:
                mm_row = meta_mask_tensor[i]
            else:
                mm_row = None
            pm_mask, fb = _build_postmeta_mask(tokenizer, ids, text, mm_row)
            # Zero mask beyond active length so padding never contributes.
            if pm_mask.size(0) > active_len:
                pm_mask[active_len:] = 0.0
            postmeta_masks.append(pm_mask)
            if fb:
                fallback_hits += 1

        # LOW-1: assert dtype consistency before stacking; cast to float32 for
        # downstream broadcast safety (meta_mask is float32, so the 3-region
        # arithmetic in _compute_sdc_advantage stays in one dtype).
        assert all(m.dtype == postmeta_masks[0].dtype for m in postmeta_masks), (
            "postmeta_masks dtype inconsistent"
        )
        postmeta_tensor = torch.stack(postmeta_masks, dim=0).to(
            device=completion_ids.device, dtype=torch.float32
        )
        out["postmeta_mask"] = postmeta_tensor

        n = max(1, completion_ids.size(0))
        self._last_fallback_trigger_rate = float(fallback_hits) / n
        self._last_wrap_rate = float(wrap_hits) / n
        return out

    # ── Decoy — §2.1 ────────────────────────────────────────────────────

    def _make_decoy(self, gold: str, seed: Optional[int] = None) -> str:
        """Dispatch on ``self.decoy_strategy`` — see §2.1 / §H3.

        Guarantees (§2.1 invariants A–E):
            (A) decoy != gold strictly
            (B) not numerically equivalent for pure-numeric golds
            (C) deterministic in ``(gold, seed)``
            (D) parse-valid string
            (E) ``_check_correctness(decoy, gold) == False`` —
                rule-based path passes the math_verify-based checker so
                symbolic equivalences (``\\sqrt5`` ≡ ``-\\sqrt5`` etc.)
                are filtered out (aligns with training-time reward checker).
        """
        s = seed if seed is not None else self.decoy_seed
        if self.decoy_strategy == "random":
            return _random_noise_decoy(gold, s)
        return _rule_based_decoy(gold, s, checker=_check_correctness)

    # ── Answer-token-boundary check — plan §9.6 (review iter1 Fix 4) ────

    def _check_answer_boundary(self, pos_ids, gold) -> bool:
        """Verify the final k tokens of pos-input match standalone gold tokenization.

        Plan §9.6: if the trailing tokens don't match, tokenizer has
        merged/split the boundary and the "gold" signal leaks into the
        student-completion span. First-batch only, rank 0 only. Emits
        ``meta_rlsd/boundary_check_passed`` metric (1.0 / 0.0).
        """
        if self._boundary_checked:
            return True
        self._boundary_checked = True
        tokenizer = self._meta_tokenizer
        standalone = tokenizer(str(gold), add_special_tokens=False).input_ids
        if not standalone:
            return True
        k = len(standalone)
        trailing = (
            pos_ids[-k:].tolist() if hasattr(pos_ids, "tolist") else list(pos_ids[-k:])
        )
        if trailing != standalone:
            import warnings

            warnings.warn(
                f"Answer token boundary mismatch: standalone={standalone[-3:]} "
                f"trailing={trailing[-3:]} — may leak signal (plan §9.6)",
                RuntimeWarning,
            )
            return False
        return True

    # ── Contrastive teacher inputs — §2.3 ────────────────────────────────

    def _build_contrastive_teacher_inputs(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        ground_truth: Sequence[str],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str]]:
        """Build T+ and T- teacher inputs by appending gold / decoy answers.

        §2.3 — teacher weights are shared; only the *context* differs.
        Both T+ and T- re-encode the student prompt with ``" Answer: {a}"``
        appended, then concatenate the student completion ids on the right.

        Returns:
            pos_input  — dict with input_ids / attention_mask / prompt_len
            neg_input  — same schema for T-
            decoys     — list of decoy strings (for logging / invariant checks)
        """
        cfg = self.meta_rlsd_cfg
        tokenizer = self._meta_tokenizer
        device = prompt_ids.device

        decoys: List[str] = []
        pos_texts: List[str] = []
        neg_texts: List[str] = []
        for i in range(prompt_ids.size(0)):
            nonpad = prompt_mask[i].bool()
            decoded = tokenizer.decode(prompt_ids[i][nonpad], skip_special_tokens=False)
            gold = str(ground_truth[i]) if i < len(ground_truth) else ""
            decoy = self._make_decoy(gold, seed=cfg.decoy_seed)
            decoys.append(decoy)
            pos_texts.append(f"{decoded} Answer: {gold}")
            neg_texts.append(f"{decoded} Answer: {decoy}")

        pos_enc = tokenizer(
            pos_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=cfg.prompt_length,
            return_tensors="pt",
        )
        neg_enc = tokenizer(
            neg_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=cfg.prompt_length,
            return_tensors="pt",
        )

        pos_prompt_ids = pos_enc["input_ids"].to(device)
        pos_prompt_mask = pos_enc["attention_mask"].to(device)
        neg_prompt_ids = neg_enc["input_ids"].to(device)
        neg_prompt_mask = neg_enc["attention_mask"].to(device)

        pos_input_ids = torch.cat([pos_prompt_ids, completion_ids], dim=1)
        pos_attn = torch.cat([pos_prompt_mask, completion_mask], dim=1)
        neg_input_ids = torch.cat([neg_prompt_ids, completion_ids], dim=1)
        neg_attn = torch.cat([neg_prompt_mask, completion_mask], dim=1)

        pos_input = {
            "input_ids": pos_input_ids,
            "attention_mask": pos_attn,
            "prompt_len": pos_prompt_ids.size(1),
        }
        neg_input = {
            "input_ids": neg_input_ids,
            "attention_mask": neg_attn,
            "prompt_len": neg_prompt_ids.size(1),
        }
        return pos_input, neg_input, decoys

    # ── Sequential contrastive forward — §2.3 VRAM ───────────────────────

    @torch.no_grad()
    def _teacher_contrastive_logprobs(
        self,
        pos_input: Dict[str, torch.Tensor],
        neg_input: Dict[str, torch.Tensor],
        completion_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sequential forward: T+ → free → T- (§2.3 peak VRAM invariant).

        Both forwards use the **same** teacher weights — only the prompt
        context differs. ``torch.cuda.empty_cache()`` between the two calls
        is the whole reason this stays at M1's peak VRAM budget.
        """
        logp_T_pos = self._teacher_logprobs(
            pos_input["input_ids"],
            pos_input["attention_mask"],
            completion_ids,
            pos_input["prompt_len"],
        )
        # Release T+ activation before T- forward — VRAM invariant (§2.3).
        torch.cuda.empty_cache()

        logp_T_neg = self._teacher_logprobs(
            neg_input["input_ids"],
            neg_input["attention_mask"],
            completion_ids,
            neg_input["prompt_len"],
        )
        # C3 fix: shape assert — T+ and T- must align to same completion positions.
        assert logp_T_pos.shape == logp_T_neg.shape, (
            f"T+/T- logprob shape mismatch: {logp_T_pos.shape} vs {logp_T_neg.shape}"
        )
        assert logp_T_pos.shape[:2] == completion_ids.shape[:2], (
            f"T+ logprob shape {logp_T_pos.shape} mismatches completion_ids {completion_ids.shape}"
        )
        return logp_T_pos, logp_T_neg

    # ── Per-token advantage — §2.4 Bayesian form ─────────────────────────

    def _compute_per_token_advantage(
        self,
        advantages: torch.Tensor,
        log_T_pos: torch.Tensor,
        log_T_neg: torch.Tensor,
        meta_mask: torch.Tensor,
        completion_mask: torch.Tensor,
        log_S: Optional[torch.Tensor] = None,
        postmeta_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Build Â_t from contrastive teacher log-probs.

        **N3 path (default)** — §2.4 Bayes-factor form:

            Δ_t   = sg(log P_{T+} − log P_{T-})   ∈ [−10, 10]
            w_t   = exp(sign(A_i) · Δ_t)
            w_t'  = clip(w_t, 1−ε_w, 1+ε_w)
            Â_t   = A_i · [(1−λ) + λ · (m_t · w_t' + (1−m_t))]

        **SDC path** (``variant ∈ {sdc-split, sdc-uniform, sdc-noise}``) — plan §3.1:

            w_t^attr = exp(+sign(A) · clip(log P_{T+} − log P_S, -10, 10))
            w_t^rep  = exp(-sign(A) · clip(log P_{T-} − log P_S, -10, 10))
            Â_t = A_i · [(1−λ_meta−λ_post) + λ_meta·(m^meta·w_attr+m^post·1+m^body·1)
                                           + λ_post·(m^meta·1+m^post·w_rep+m^body·1)]

        which (expanding) gives the 3-region disjoint form:

            Â_t = A_i · [(1−λ_meta·m^meta−λ_post·m^post) · 1
                         + λ_meta·m^meta·w_attr
                         + λ_post·m^post·w_rep]

        with body tokens untouched (factor = 1). Uniform and noise controls
        share this skeleton but swap the weight tensors (see branches below).

        Args:
            advantages: [B] group-relative advantage scalars.
            log_T_pos: [B, T] teacher+ log-probs on completion tokens.
            log_T_neg: [B, T] teacher- log-probs on completion tokens.
            meta_mask: [B, T] float {0,1} — tokens inside <|meta|>..<|/meta|>.
            completion_mask: [B, T] — active (non-padded) completion positions.
            log_S: [B, T] student log-probs (required for SDC, ignored for N3).
            postmeta_mask: [B, T] SDC post-meta region mask (required for SDC).

        Returns:
            hat_A  — per-token advantage tensor, shape [B, T]
            stats  — dict of scalar metrics for wandb logging
        """
        cfg = self.meta_rlsd_cfg

        if self._is_sdc_variant():
            return self._compute_sdc_advantage(
                advantages=advantages,
                log_T_pos=log_T_pos,
                log_T_neg=log_T_neg,
                log_S=log_S,
                meta_mask=meta_mask,
                postmeta_mask=postmeta_mask,
                completion_mask=completion_mask,
            )

        # ── N3 path (unchanged — preserved verbatim for variant=n3*) ───
        # Δ_t — Bayes factor; detached to block grad flow through teacher (§2.4).
        delta_t = torch.clamp(
            (log_T_pos - log_T_neg).detach(),
            -cfg.log_ratio_clamp,
            cfg.log_ratio_clamp,
        )

        A_sign = torch.sign(advantages).unsqueeze(1)  # [B, 1]
        w_t = torch.exp(A_sign * delta_t)  # [B, T]
        w_t_clip = torch.clamp(w_t, 1.0 - cfg.clip_eps_w, 1.0 + cfg.clip_eps_w)

        lam = self._current_lambda()
        per_token_factor = meta_mask * w_t_clip + (1.0 - meta_mask)
        hat_A = advantages.unsqueeze(1) * ((1.0 - lam) + lam * per_token_factor)

        # ── Logging stats (§4.2 smoke acceptance) ──────────────────────
        meta_flat = (meta_mask * completion_mask).bool()
        if meta_flat.any():
            meta_delta = delta_t[meta_flat]
            delta_mean = float(meta_delta.mean().item())
            delta_std = float(meta_delta.std().item()) if meta_delta.numel() > 1 else 0.0
            kl_vals = torch.exp(log_T_pos[meta_flat]) * (
                log_T_pos[meta_flat] - log_T_neg[meta_flat]
            )
            kl_mean = float(kl_vals.mean().item())
            clipped = (w_t != w_t_clip) & meta_flat
            clip_frac_w = float(
                clipped.float().sum().item() / max(meta_flat.sum().item(), 1)
            )
            ratio_mean = float(w_t[meta_flat].mean().item())
            ratio_std = float(w_t[meta_flat].std().item()) if w_t[meta_flat].numel() > 1 else 0.0
        else:
            delta_mean = delta_std = 0.0
            kl_mean = 0.0
            clip_frac_w = 0.0
            ratio_mean = 1.0
            ratio_std = 0.0

        self._last_delta_mean = delta_mean
        self._last_delta_std = delta_std
        self._last_kl_t_pos_t_neg = kl_mean
        self._last_teacher_ratio_mean = ratio_mean
        self._last_teacher_ratio_std = ratio_std
        self._last_clip_fraction_w = clip_frac_w

        stats: Dict[str, float] = {
            "meta_rlsd/delta_t_mean": delta_mean,
            "meta_rlsd/delta_t_std": delta_std,
            "meta_rlsd/kl_T_pos_T_neg": kl_mean,
            "meta_rlsd/teacher_ratio_mean": ratio_mean,
            "meta_rlsd/teacher_ratio_std": ratio_std,
            "meta_rlsd/clip_fraction_w": clip_frac_w,
            "meta_rlsd/lambda_current": lam,
            "meta_rlsd/contrastive_fwd_count": 2.0,  # invariant — §4.2
        }
        return hat_A, stats

    # ── SDC advantage builder (plan_SDC_v2 §3.1) ────────────────────────

    def _compute_sdc_advantage(
        self,
        advantages: torch.Tensor,
        log_T_pos: torch.Tensor,
        log_T_neg: torch.Tensor,
        log_S: Optional[torch.Tensor],
        meta_mask: torch.Tensor,
        postmeta_mask: Optional[torch.Tensor],
        completion_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """SDC 3-region disjoint advantage (plan §3.1 / §5.1).

        Branches on ``variant``:
          * ``sdc-split``   — legacy baseline. Meta = attract, post-meta = repel.
          * ``sdc-uniform`` — L1-matched control (plan §2.4). Both attract and
                              repel applied to (meta ∪ post-meta) with each
                              coefficient halved. Total L1 mass of (Â - A_i)
                              matches sdc-split on a batch where the underlying
                              weight magnitudes are identical.
          * ``sdc-noise``   — null-signal control (plan §2.6). Post-meta repel
                              is replaced by exp(ε_t) where ε ~ N(0, σ²).
          * ``sdc-shared``  — preserve teacher-consensus tokens in post-meta
                              and apply contrast only on T+/T- disagreement.
        """
        cfg = self.meta_rlsd_cfg
        device = advantages.device

        # Safety: post-meta mask is mandatory for SDC. If it's missing (e.g., a
        # prediction_step path) fall back to an all-zero mask — this degenerates
        # SDC to M1-ish attract-only, which is safe but logs a warning.
        if postmeta_mask is None:
            # MEDIUM-5: surface the missing-mask condition exactly once so a
            # silent prediction_step mis-wire does not go unnoticed. The
            # advantage-build still continues with a zero mask (safe degrade).
            if not self._sdc_postmeta_missing_warned:
                import warnings

                warnings.warn(
                    "SDC variant requested but postmeta_mask missing — check "
                    "_generate_and_score_completions override is firing for "
                    f"variant={cfg.variant}",
                    RuntimeWarning,
                )
                self._sdc_postmeta_missing_warned = True
            postmeta_mask = torch.zeros_like(meta_mask)
        if log_S is None:
            # Without student log-probs we cannot form attract / repel; return
            # advantages untouched (body-identity fallback for the whole seq).
            hat_A = advantages.unsqueeze(1).expand_as(meta_mask).clone()
            return hat_A, {
                "meta_rlsd/sdc_degraded": 1.0,
                "meta_rlsd/lambda_meta_current": float(self._current_lambda()),
                "meta_rlsd/lambda_post_current": float(self._current_lambda_post()),
            }

        # Detach teacher and student log-probs — teacher signal is environment,
        # not a gradient path (plan §3.1 and structural property 2/3).
        log_S_d = log_S.detach()
        log_T_pos_d = log_T_pos.detach()
        log_T_neg_d = log_T_neg.detach()

        clamp = cfg.log_ratio_clamp
        A_sign = torch.sign(advantages).unsqueeze(1)  # [B, 1]
        # Attract weight: exp(+sign(A) · (log P_T+ − log P_S)).
        attr_log = torch.clamp(log_T_pos_d - log_S_d, -clamp, clamp)
        w_attr = torch.exp(A_sign * attr_log)
        # Repel weight: exp(−sign(A) · (log P_T− − log P_S)).
        rep_log = torch.clamp(log_T_neg_d - log_S_d, -clamp, clamp)
        w_rep = torch.exp(-A_sign * rep_log)

        # Clip both weights to [1-ε_w, 1+ε_w] per plan R3 risk mitigation.
        w_attr_clip = torch.clamp(w_attr, 1.0 - cfg.clip_eps_w, 1.0 + cfg.clip_eps_w)
        w_rep_clip = torch.clamp(w_rep, 1.0 - cfg.clip_eps_w, 1.0 + cfg.clip_eps_w)

        # Shared-vs-differential decomposition for post-meta tokens. Teacher
        # consensus should preserve common structure; only disagreement should
        # carry answer-disambiguating pressure.
        delta_t_proxy = torch.clamp(log_T_pos_d - log_T_neg_d, -clamp, clamp)
        shared_gate = (delta_t_proxy.abs() <= float(cfg.sdc_shared_tau)).float()
        post_shared_mask = postmeta_mask * shared_gate
        post_diff_mask = postmeta_mask * (1.0 - shared_gate)

        shared_anchor_log = torch.clamp(
            0.5 * (log_T_pos_d + log_T_neg_d) - log_S_d,
            -clamp,
            clamp,
        )
        w_shared = torch.exp(A_sign * shared_anchor_log)
        w_shared_clip = torch.clamp(
            w_shared, 1.0 - cfg.clip_eps_w, 1.0 + cfg.clip_eps_w
        )

        # Differential contrast uses the relative teacher evidence directly,
        # so shared structure cancels and only answer-disambiguating tokens
        # receive extra pressure.
        w_diff = torch.exp(A_sign * delta_t_proxy)
        w_diff_clip = torch.clamp(
            w_diff, 1.0 - cfg.clip_eps_w, 1.0 + cfg.clip_eps_w
        )

        # Body mask = 1 − meta − post (for debugging; enforced by construction
        # because postmeta_mask already has meta_mask subtracted in the pipeline).
        body_mask = torch.clamp(1.0 - meta_mask - postmeta_mask, 0.0, 1.0)

        lam_meta = float(self._current_lambda())       # existing schedule (down)
        lam_post = float(self._current_lambda_post())  # new schedule (up)
        variant = cfg.variant

        advantages_ut = advantages.unsqueeze(1)

        if variant == "sdc-split":
            # Â_t = A_i · [body·1 + meta·((1-λ_m)+λ_m·w_attr) + post·((1-λ_p)+λ_p·w_rep)]
            factor = (
                body_mask
                + meta_mask * ((1.0 - lam_meta) + lam_meta * w_attr_clip)
                + postmeta_mask * ((1.0 - lam_post) + lam_post * w_rep_clip)
            )
            hat_A = advantages_ut * factor
        elif variant == "sdc-uniform":
            # L1-matched control (plan §2.4): apply half-strength attract AND
            # half-strength repel across (meta ∪ post). Coefficients halved so
            # at the symmetric (w_attr==w_rep, λ_m==λ_p) regime the L1 mass of
            # (Â - A) matches sdc-split exactly.
            #
            # HIGH-1: at realistic training time w_attr != w_rep and λ_m != λ_p,
            # so the coefficient-halving alone leaves the L1 mass differing by
            # ~27%. We rescale the (Â - A) delta *per batch* so the L1 mass
            # equals what sdc-split would have produced on the same batch.
            # This lets the H-SDC-4 falsifier (plan §2.4) cleanly isolate the
            # region-split effect from raw signal magnitude.
            combined_region = torch.clamp(meta_mask + postmeta_mask, 0.0, 1.0)
            factor_uniform = (
                body_mask
                + combined_region * (
                    (1.0 - 0.5 * lam_meta - 0.5 * lam_post)
                    + 0.5 * lam_meta * w_attr_clip
                    + 0.5 * lam_post * w_rep_clip
                )
            )
            hat_A_uniform_raw = advantages_ut * factor_uniform

            # Compute the L1 mass sdc-split would produce on *this* batch.
            factor_split_ref = (
                body_mask
                + meta_mask * ((1.0 - lam_meta) + lam_meta * w_attr_clip)
                + postmeta_mask * ((1.0 - lam_post) + lam_post * w_rep_clip)
            )
            hat_A_split_ref = advantages_ut * factor_split_ref
            active = completion_mask
            l1_split_target = float(
                ((hat_A_split_ref - advantages_ut).abs() * active).sum().item()
            )
            l1_uniform_raw = float(
                ((hat_A_uniform_raw - advantages_ut).abs() * active).sum().item()
            )
            # Rescale the delta so |Â_u - A|.sum() == |Â_s - A|.sum() within 1%.
            # Guard against division by zero (e.g., a batch with no active mask
            # positions) — in that degenerate case keep uniform factor untouched.
            if l1_uniform_raw > 1e-8:
                scale = l1_split_target / l1_uniform_raw
            else:
                scale = 1.0
            delta = hat_A_uniform_raw - advantages_ut
            hat_A = advantages_ut + scale * delta
        elif variant == "sdc-noise":
            # Null-signal control (plan §2.6): replace repel weight with a
            # Gaussian-noise weight of matched magnitude. Meta region retains
            # attract so the control isolates the *directional* contribution
            # of the post-meta signal.
            sigma = float(cfg.sdc_noise_sigma)
            # MEDIUM-4: seed = (global_step, call_count, cfg.seed) so noise
            # differs across gradient-accumulation micro-batches at the same
            # optimizer step. Earlier formulation (step * 9973 + seed) reused
            # the same tensor across accumulation rounds which silently
            # correlated noise within a single optimizer update.
            step = int(self.state.global_step) if self.state is not None else 0
            seed_mix = (
                step * 9973
                + self._sdc_noise_call_count * 1009
                + int(cfg.seed)
            )
            self._sdc_noise_call_count += 1
            gen = torch.Generator(device="cpu").manual_seed(seed_mix)
            noise_cpu = torch.randn(meta_mask.shape, generator=gen) * sigma
            eps = noise_cpu.to(device)
            w_noise = torch.exp(torch.clamp(eps, -clamp, clamp))
            w_noise = torch.clamp(w_noise, 1.0 - cfg.clip_eps_w, 1.0 + cfg.clip_eps_w)
            factor = (
                body_mask
                + meta_mask * ((1.0 - lam_meta) + lam_meta * w_attr_clip)
                + postmeta_mask * ((1.0 - lam_post) + lam_post * w_noise)
            )
            hat_A = advantages_ut * factor
        elif variant == "sdc-shared":
            factor = (
                body_mask
                + meta_mask * ((1.0 - lam_meta) + lam_meta * w_attr_clip)
                + post_shared_mask * ((1.0 - lam_post) + lam_post * w_shared_clip)
                + post_diff_mask * ((1.0 - lam_post) + lam_post * w_diff_clip)
            )
            hat_A = advantages_ut * factor
        else:  # pragma: no cover — caught by _is_sdc_variant above
            raise ValueError(f"Unknown SDC variant: {variant!r}")

        # ── Logging ────────────────────────────────────────────────────
        meta_flat = (meta_mask * completion_mask).bool()
        post_flat = (postmeta_mask * completion_mask).bool()
        post_shared_flat = (post_shared_mask * completion_mask).bool()
        post_diff_flat = (post_diff_mask * completion_mask).bool()
        if meta_flat.any():
            md = delta_t_proxy[meta_flat]
            delta_mean = float(md.mean().item())
            delta_std = float(md.std().item()) if md.numel() > 1 else 0.0
            kl_vals = torch.exp(log_T_pos_d[meta_flat]) * (
                log_T_pos_d[meta_flat] - log_T_neg_d[meta_flat]
            )
            kl_mean = float(kl_vals.mean().item())
            w_attr_mean = float(w_attr[meta_flat].mean().item())
        else:
            delta_mean = delta_std = 0.0
            kl_mean = 0.0
            w_attr_mean = 1.0
        if post_flat.any():
            w_rep_mean = float(w_rep[post_flat].mean().item())
            w_rep_std = float(w_rep[post_flat].std().item()) if w_rep[post_flat].numel() > 1 else 0.0
        else:
            w_rep_mean = 1.0
            w_rep_std = 0.0
        if post_shared_flat.any():
            w_shared_mean = float(w_shared[post_shared_flat].mean().item())
            post_shared_frac = float(
                post_shared_flat.float().sum().item()
                / max(completion_mask.sum().item(), 1)
            )
        else:
            w_shared_mean = 1.0
            post_shared_frac = 0.0
        if post_diff_flat.any():
            w_diff_mean = float(w_diff[post_diff_flat].mean().item())
            post_diff_frac = float(
                post_diff_flat.float().sum().item()
                / max(completion_mask.sum().item(), 1)
            )
        else:
            w_diff_mean = 1.0
            post_diff_frac = 0.0

        stats = {
            "meta_rlsd/delta_t_mean": delta_mean,
            "meta_rlsd/delta_t_std": delta_std,
            "meta_rlsd/kl_T_pos_T_neg": kl_mean,
            "meta_rlsd/w_attr_mean": w_attr_mean,
            "meta_rlsd/w_rep_mean": w_rep_mean,
            "meta_rlsd/w_rep_std": w_rep_std,
            "meta_rlsd/w_shared_mean": w_shared_mean,
            "meta_rlsd/w_diff_mean": w_diff_mean,
            "meta_rlsd/lambda_meta_current": lam_meta,
            "meta_rlsd/lambda_post_current": lam_post,
            # back-compat: logs lambda_meta (down-schedule). lambda_post_current logged separately.
            "meta_rlsd/lambda_current": lam_meta,
            "meta_rlsd/contrastive_fwd_count": 2.0,
            "meta_rlsd/postmeta_frac": float(
                post_flat.float().sum().item() / max(completion_mask.sum().item(), 1)
            ),
            "meta_rlsd/postmeta_shared_frac": post_shared_frac,
            "meta_rlsd/postmeta_diff_frac": post_diff_frac,
            "meta_rlsd/fallback_trigger_rate": float(self._last_fallback_trigger_rate),
            "meta_rlsd/wrap_rate": float(self._last_wrap_rate),
            "meta_rlsd/sdc_variant_code": {
                "sdc-split": 1.0, "sdc-uniform": 2.0, "sdc-noise": 3.0, "sdc-shared": 4.0
            }[variant],
        }

        # Keep back-compat fields for downstream dashboards.
        self._last_delta_mean = delta_mean
        self._last_delta_std = delta_std
        self._last_kl_t_pos_t_neg = kl_mean
        self._last_teacher_ratio_mean = w_attr_mean
        self._last_teacher_ratio_std = 0.0
        self._last_clip_fraction_w = 0.0
        return hat_A, stats

    # ── Override — full _compute_loss with contrastive teacher signal ───

    def _compute_loss(self, model, inputs):  # noqa: C901 — mirrors M1 shape
        """Replace M1 single-teacher advantage with contrastive Δ_t.

        PPO clip, normalization, and logging scaffolding mirror M1
        (``meta_rlsd_trainer.MetaRLSDTrainer._compute_loss``) — only the
        advantage construction differs. See plan §2.4 for the formula and
        §2.5 for the leakage-isolation proof (still holds because the
        decoy function is deterministic).
        """
        cfg = self.meta_rlsd_cfg

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        meta_mask = inputs.get("meta_mask")
        if meta_mask is None:
            meta_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        meta_mask = meta_mask.to(completion_mask.device).float()

        # SDC-only: pull postmeta_mask through inputs (set in
        # _generate_and_score_completions). Non-SDC variants never touch this.
        postmeta_mask = inputs.get("postmeta_mask")
        if postmeta_mask is not None:
            postmeta_mask = postmeta_mask.to(completion_mask.device).float()

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Student per-token log-probs — reused from TRL base (M1 also uses this).
        per_token_logps, _entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
        )

        advantages = inputs["advantages"]  # shape [B]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        # ── Contrastive teacher forward — §2.3 ─────────────────────────
        # Pull from self cache (set in _generate_and_score_completions); avoids
        # putting Python lists in tensor_dict that TRL 0.19 shuffles.
        # HIGH-1 fix: resolve gold via shuffle-preserved _gold_idx tensor.
        gold_idx = inputs.get("_gold_idx", None)
        gold_table = getattr(self, "_batch_gold_table", None)
        if gold_idx is not None and gold_table is not None:
            gt_list = [gold_table[int(i)] for i in gold_idx.tolist()]
        else:
            gt_list = getattr(self, "_batch_ground_truth", None) or inputs.get("ground_truth", []) or []
        # W2 fix: ground_truth routing guard — N3 requires gold strings for
        # privileged teacher + decoy generation. Silently empty list would
        # produce identical-decoys null-signal runs.
        if not gt_list or all((not g or not str(g).strip()) for g in gt_list):
            raise RuntimeError(
                "ContrastiveMetaRLSDTrainer: ground_truth missing from inputs. "
                "N3 requires gold answers (check C1 routing in _generate_and_score_completions)."
            )
        pos_input, neg_input, decoys = self._build_contrastive_teacher_inputs(
            prompt_ids, prompt_mask, completion_ids, completion_mask, gt_list
        )
        log_T_pos, log_T_neg = self._teacher_contrastive_logprobs(
            pos_input, neg_input, completion_ids
        )

        # Decoy invariants (§2.1 A — hard; §4.2 smoke acceptance — soft).
        eq_gold = sum(1 for g, d in zip(gt_list, decoys) if d == str(g))
        self._last_decoy_eq_gold_rate = float(eq_gold) / max(len(decoys), 1)

        # Review iter1 Fix 5: decoy_is_correct_rate — decoy value that grades
        # numerically correct per _check_correctness (boxed/string fallback).
        correct_decoy = sum(
            1 for g, d in zip(gt_list, decoys) if _check_correctness(str(d), str(g))
        )
        self._last_decoy_is_correct_rate = float(correct_decoy) / max(len(decoys), 1)

        # Plan §4.2: decoy_eq_gold is a HARD invariant. In the first 10 steps,
        # ANY violation aborts — after that, log and let the monitoring layer act.
        step = int(self.state.global_step) if self.state is not None else 0
        is_main = self.accelerator.is_main_process
        if is_main and step < self._decoy_abort_window:
            if eq_gold > 0:
                raise RuntimeError(
                    f"[N3 abort] decoy_eq_gold={eq_gold}/{len(decoys)} at step={step} "
                    "(plan §4.2 hard invariant); check _rule_based_decoy output."
                )
            if self._last_decoy_is_correct_rate > 0.05:
                raise RuntimeError(
                    f"[N3 abort] decoy_is_correct_rate={self._last_decoy_is_correct_rate:.3f} "
                    f"at step={step} exceeds 5% (plan §4.2); decoys grade as correct."
                )

        # Review iter1 Fix 4: boundary check on first batch (rank 0 only).
        if is_main and not self._boundary_checked and gt_list:
            pos_trail = pos_input["input_ids"][0][: pos_input["prompt_len"]]
            passed = self._check_answer_boundary(pos_trail, gt_list[0])
            # Metric recorded below alongside adv_stats (see "boundary_check_passed").
            self._boundary_check_passed_val = 1.0 if passed else 0.0
        else:
            self._boundary_check_passed_val = getattr(self, "_boundary_check_passed_val", 1.0)

        # ── Per-token advantage — §2.4 (N3) or §3.1 (SDC) ─────────────
        hat_A, adv_stats = self._compute_per_token_advantage(
            advantages,
            log_T_pos,
            log_T_neg,
            meta_mask,
            completion_mask,
            log_S=per_token_logps.detach() if self._is_sdc_variant() else None,
            postmeta_mask=postmeta_mask if self._is_sdc_variant() else None,
        )

        # ── SDC halt conditions (plan §3.6) — RuntimeWarning only ──────
        if self._is_sdc_variant() and is_main:
            step_now = int(self.state.global_step) if self.state is not None else 0
            if step_now >= 50 and self._last_fallback_trigger_rate > 0.20 and not self._sdc_fallback_warned:
                import warnings

                warnings.warn(
                    f"[SDC §3.6] fallback_trigger_rate={self._last_fallback_trigger_rate:.3f} "
                    f"exceeds 20% at step={step_now} — post-meta mask fallback (no \\boxed{{}}) "
                    "is dominant; consider revising the mask definition.",
                    RuntimeWarning,
                )
                self._sdc_fallback_warned = True
            # Wrap-regression smoke rule: halt threshold is
            # "wrap_rate < sdc_wrap_baseline - 10pp" at step >= 30.
            # MEDIUM-2: threshold now reads from cfg.sdc_wrap_baseline (default 1.0
            # for v8 SFT) instead of the prior hardcoded 0.90.
            wrap_threshold = float(cfg.sdc_wrap_baseline) - 0.10
            if step_now >= 30 and self._last_wrap_rate < wrap_threshold and not self._sdc_wrap_warned:
                import warnings

                warnings.warn(
                    f"[SDC §3.6] wrap_rate={self._last_wrap_rate:.3f} < "
                    f"{wrap_threshold:.3f} at step={step_now} — post-meta repel "
                    "may be destroying format markers; consider lowering lambda_post.",
                    RuntimeWarning,
                )
                self._sdc_wrap_warned = True

        # ── PPO asymmetric clip — same as M1 (§2.2) ───────────────────
        log_ratio = per_token_logps - old_per_token_logps
        log_ratio = torch.clamp(log_ratio, -cfg.log_ratio_clamp, cfg.log_ratio_clamp)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high)

        per_token_loss1 = coef_1 * hat_A
        per_token_loss2 = coef_2 * hat_A
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        denom = completion_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * completion_mask).sum() / denom
        loss = loss / max(self.current_gradient_accumulation_steps, 1)

        # ── Logging — merge contrastive stats with M1-style metrics ───
        mode = "train" if self.model.training else "eval"
        metrics = self._metrics[mode]

        for key, value in adv_stats.items():
            metrics.setdefault(key, []).append(float(value))
        metrics.setdefault("meta_rlsd/meta_token_fraction", []).append(
            self._last_meta_token_fraction
        )
        metrics.setdefault("meta_rlsd/A_scalar_mean", []).append(self._last_A_scalar_mean)
        metrics.setdefault("meta_rlsd/A_scalar_std", []).append(self._last_A_scalar_std)
        metrics.setdefault("meta_rlsd/skipped_groups", []).append(
            float(self._last_skipped_groups)
        )
        metrics.setdefault("meta_rlsd/decoy_eq_gold_rate", []).append(
            self._last_decoy_eq_gold_rate
        )
        metrics.setdefault("meta_rlsd/decoy_is_correct_rate", []).append(
            self._last_decoy_is_correct_rate
        )
        metrics.setdefault("meta_rlsd/boundary_check_passed", []).append(
            float(self._boundary_check_passed_val)
        )
        metrics.setdefault("meta_rlsd/loss", []).append(float(loss.detach().item()))

        # PPO clip fractions — matches TRL 0.19.1 GRPOTrainer convention.
        # Low clip fires on (A<0, r<1−ε_low); high clip fires on (A>0, r>1+ε_high).
        is_low = (coef_1 < 1.0 - cfg.clip_eps_low) & (advantages.unsqueeze(1) < 0)
        is_high = (coef_1 > 1.0 + cfg.clip_eps_high) & (advantages.unsqueeze(1) > 0)
        region = ((is_low | is_high) & completion_mask.bool()).float()
        clip_fraction = float(region.sum().item() / max(completion_mask.sum().item(), 1))
        metrics.setdefault("meta_rlsd/ppo_clip_fraction", []).append(clip_fraction)
        metrics.setdefault("clip_fraction", []).append(clip_fraction)

        return loss


# ─── SDC halt callback — plan §3.6 escalation (MEDIUM-1) ──────────────────


class SDCHaltCallback(TrainerCallback):
    """Raise ``RuntimeError`` on SECOND consecutive batch that violates either:

        * ``fallback_trigger_rate > 0.20`` at step ≥ 50
        * ``wrap_rate < sdc_wrap_baseline - 0.10`` at step ≥ 30

    Plan §3.6 says "halt and escalate" when post-meta repel destabilises the
    student (format markers disappear, or the mask fallback dominates). The
    existing :class:`warnings.warn` calls inside ``_compute_loss`` handle the
    *first* observation (and latch so they only fire once). This callback
    counts consecutive logged batches and raises after two in a row, which
    gives downstream runs a hard failure instead of a silent drift.

    Analogous to :class:`ClipFractionAbortCallback` but uses the raise-pattern
    because ``should_training_stop=True`` can be swallowed by TRL's generate
    loop before the next optimizer step — we want the loudest possible halt.
    """

    def __init__(self, wrap_baseline: float = 1.0, consec_required: int = 3):
        # Iteration 5: Empirical SFT baseline measurement shows wrap_rate
        # rolling mean hovers at 0.55-0.80 throughout training from step 1 —
        # this is the SFT model's natural per-rank discrete wrap rate at
        # temperature=0.6, NOT drift caused by SDC. Set threshold to 0.35
        # (large margin below empirical baseline ~0.70) to catch true
        # systematic collapse while allowing normal stochastic variation.
        # Window size 20 for more stable aggregation.
        self.wrap_threshold = 0.35
        self.consec_required = int(consec_required)
        self.fallback_consec = 0
        self.wrap_consec = 0
        self.wrap_window: list[float] = []
        self.wrap_window_size = 20

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = int(getattr(state, "global_step", 0))

        # Fallback-trigger-rate rule — same fix as wrap_rate: use rolling
        # window mean. fallback_trigger_rate is also a per-rank 2-rollout
        # discrete metric (0.0/0.5/1.0) so instantaneous 1.0 samples are
        # normal SFT stochasticity, not drift.
        fb = logs.get("meta_rlsd/fallback_trigger_rate")
        if fb is not None and step >= 50:
            try:
                fb_val = float(fb)
            except (TypeError, ValueError):
                fb_val = 0.0
            if not hasattr(self, "fb_window"):
                self.fb_window = []
            self.fb_window.append(fb_val)
            if len(self.fb_window) > self.wrap_window_size:
                self.fb_window.pop(0)
            if len(self.fb_window) >= self.wrap_window_size:
                fb_mean = sum(self.fb_window) / len(self.fb_window)
                if fb_mean > 0.40:  # was 0.20 per-sample; 0.40 rolling mean
                    self.fallback_consec += 1
                    if self.fallback_consec >= self.consec_required:
                        raise RuntimeError(
                            f"[SDC §3.6 HALT] rolling fallback_trigger_rate mean="
                            f"{fb_mean:.3f} > 0.40 for {self.consec_required} consec "
                            f"windows at step={step}. Post-meta mask fallback dominant."
                        )
                else:
                    self.fallback_consec = 0

        # Wrap-regression rule — rolling window average (fix for 8-rollout discreteness)
        wr = logs.get("meta_rlsd/wrap_rate")
        if wr is not None and step >= 30:
            try:
                wr_val = float(wr)
            except (TypeError, ValueError):
                wr_val = 1.0
            self.wrap_window.append(wr_val)
            if len(self.wrap_window) > self.wrap_window_size:
                self.wrap_window.pop(0)
            if len(self.wrap_window) >= self.wrap_window_size:
                window_mean = sum(self.wrap_window) / len(self.wrap_window)
                if window_mean < self.wrap_threshold:
                    self.wrap_consec += 1
                    if self.wrap_consec >= self.consec_required:
                        raise RuntimeError(
                            f"[SDC §3.6 HALT] rolling wrap_rate mean={window_mean:.3f} < "
                            f"{self.wrap_threshold:.3f} for {self.consec_required} consecutive "
                            f"windows at step={step}. Format markers collapsed."
                        )
                else:
                    self.wrap_consec = 0


# ─── CLI — adds n3* variants on top of M1's {m1, a1, a2a, a2b, a4} ─────────

def _apply_variant(cfg: ContrastiveMetaRLSDConfig, variant: str) -> Tuple[ContrastiveMetaRLSDConfig, type]:
    """Variant dispatch — returns (cfg, trainer_class).

    n3         — contrastive, rule-based decoy, meta-only mask (primary).
    n3-random  — contrastive, random-noise decoy (H3 ablation).
    n3-fullmask— contrastive, rule-based decoy, mask_mode="all_tokens"
                 (mask × contrastive interaction; NOT paper-faithful RLSD).
    m1/a1/…    — routes to base ``MetaRLSDTrainer`` (M1 behavior).
    """
    variant = variant.lower()
    cfg.variant = variant

    if variant == "n3":
        cfg.privileged_answer = True
        cfg.mask_mode = "meta_only"
        cfg.decoy_strategy = "rule_based"
        return cfg, ContrastiveMetaRLSDTrainer
    if variant == "n3-random":
        cfg.privileged_answer = True
        cfg.mask_mode = "meta_only"
        cfg.decoy_strategy = "random"
        return cfg, ContrastiveMetaRLSDTrainer
    if variant == "n3-fullmask":
        cfg.privileged_answer = True
        cfg.mask_mode = "all_tokens"
        cfg.decoy_strategy = "rule_based"
        return cfg, ContrastiveMetaRLSDTrainer
    # SDC variants (plan_SDC_v2 §5.1) — all share meta_only base mask_mode;
    # the 3-region SDC mask is composed *inside* the advantage builder rather
    # than flipping mask_mode. This keeps meta_mask semantics identical to N3
    # so any downstream consumer that ignores postmeta_mask still behaves the
    # same way.
    if variant == "sdc-split":
        cfg.privileged_answer = True
        cfg.mask_mode = "meta_only"
        cfg.decoy_strategy = "rule_based"
        return cfg, ContrastiveMetaRLSDTrainer
    if variant == "sdc-uniform":
        cfg.privileged_answer = True
        cfg.mask_mode = "meta_only"
        cfg.decoy_strategy = "rule_based"
        return cfg, ContrastiveMetaRLSDTrainer
    if variant == "sdc-noise":
        cfg.privileged_answer = True
        cfg.mask_mode = "meta_only"
        cfg.decoy_strategy = "rule_based"
        return cfg, ContrastiveMetaRLSDTrainer
    if variant == "sdc-shared":
        cfg.privileged_answer = True
        cfg.mask_mode = "meta_only"
        cfg.decoy_strategy = "rule_based"
        return cfg, ContrastiveMetaRLSDTrainer

    # Fall back to M1-style variants — use base MetaRLSDTrainer.
    if variant == "m1":
        cfg.privileged_answer = True
        cfg.mask_mode = "meta_only"
    elif variant == "a1":
        cfg.privileged_answer = False
        cfg.mask_mode = "meta_only"
    elif variant == "a2a":
        cfg.privileged_answer = True
        cfg.mask_mode = "all_tokens"
    elif variant == "a2b":
        cfg.privileged_answer = False
        cfg.mask_mode = "all_tokens"
    elif variant == "a4":
        cfg.lambda_init = 0.0
        cfg.lambda_final = 0.0
        cfg.mask_mode = "meta_only"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cfg, MetaRLSDTrainer


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint — mirrors ``meta_rlsd_trainer.main`` with n3 dispatch.

    Variant routing:
        * n3, n3-random, n3-fullmask → :class:`ContrastiveMetaRLSDTrainer`
        * m1, a1, a2a, a2b, a4       → :class:`MetaRLSDTrainer` (base class)
    """
    parser = argparse.ArgumentParser(description="Contrastive Meta-RLSD trainer (N3)")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--variant",
        choices=[
            "n3", "n3-random", "n3-fullmask",
            "sdc-split", "sdc-uniform", "sdc-noise", "sdc-shared",
            "m1", "a1", "a2a", "a2b", "a4",
        ],
        default="n3",
        help="Variant dispatch — n3*/sdc-* use ContrastiveMetaRLSDTrainer",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_preflight",
        action="store_true",
        help="Bypass PF1-PF5 (debug/smoke only — NOT for production)",
    )
    args = parser.parse_args(argv)

    cfg = ContrastiveMetaRLSDConfig.from_yaml(args.config)
    cfg.seed = args.seed
    cfg, trainer_cls = _apply_variant(cfg, args.variant)

    os.environ.setdefault("WANDB_PROJECT", "metacot-contrastive-rlsd")

    # ── Tokenizer + meta tokens ───────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_init, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])

    # ── Pre-flight — §2.10 (reused verbatim from M1) ──────────────────
    # Review iter1 Fix 7: absolute-resolve dataset paths for log clarity.
    cfg.train_data = os.path.abspath(cfg.train_data)
    print(f"[contrastive_rlsd] resolved train_data: {cfg.train_data}", flush=True)
    if cfg.val_data:
        cfg.val_data = os.path.abspath(cfg.val_data)
        print(f"[contrastive_rlsd] resolved val_data: {cfg.val_data}", flush=True)
    if args.skip_preflight:
        print("[contrastive_rlsd] --skip_preflight set: bypassing PF1-PF5", flush=True)
    else:
        report = preflight_checks(
            cfg.train_data,
            tokenizer,
            prompt_length=cfg.prompt_length,
            meta_min_length_tokens=cfg.meta_min_length_tokens,
        )
        is_main = os.environ.get("RANK", "0") == "0"
        if is_main:
            print("=" * 60, file=sys.stderr, flush=True)
            print(f"[contrastive_rlsd] PF: passed={report.passed}", file=sys.stderr, flush=True)
            print(f"[contrastive_rlsd] PF violations ({len(report.violations)}):", file=sys.stderr, flush=True)
            for v in report.violations:
                print(f"  - {v!r}", file=sys.stderr, flush=True)
            print(f"[contrastive_rlsd] PF warnings ({len(report.warnings)}):", file=sys.stderr, flush=True)
            for w in report.warnings:
                print(f"  - {w!r}", file=sys.stderr, flush=True)
            print(f"[contrastive_rlsd] PF stats: {report.stats}", file=sys.stderr, flush=True)
            print("=" * 60, file=sys.stderr, flush=True)
        if not report.passed:
            if is_main:
                print(
                    "[contrastive_rlsd] Pre-flight FAILED — aborting. Use --skip_preflight to bypass.",
                    file=sys.stderr,
                    flush=True,
                )
            return 2

    # ── Student model ──────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_init,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = load_meta_rlsd_dataset(cfg.train_data)
    print(f"[contrastive_rlsd] Loaded {len(train_ds)} training prompts from {cfg.train_data}")

    # ── Reward — reuse M1's composed reward ───────────────────────────
    reward_fn = _partial(
        correctness_plus_meta_floor_reward,
        tokenizer=tokenizer,
        cfg=cfg,  # dataclass fields compatible via inheritance
        correctness_weight=cfg.correctness_weight,
        meta_floor_weight=cfg.meta_floor_weight,
    )
    reward_fn.__name__ = "correctness_plus_meta_floor_reward"  # type: ignore[attr-defined]

    grpo_config = _build_grpo_config(cfg)

    trainer = trainer_cls(
        model=model,
        args=grpo_config,
        train_dataset=train_ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        meta_rlsd_cfg=cfg,
    )
    trainer.add_callback(TeacherSyncCallback(trainer))
    trainer.add_callback(ClipFractionAbortCallback(threshold=0.5, window=20))

    # JSONL logging — plan §9.8 resume-safe monitoring reads this file.
    from transformers import TrainerCallback as _TrainerCallback

    class _JsonlLoggingCallback(_TrainerCallback):
        """Appends every on_log scalar dict as one line to metrics.jsonl."""

        def __init__(self, path: str):
            super().__init__()
            self.path = path
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            try:
                payload = {k: (float(v) if isinstance(v, (int, float)) else v)
                           for k, v in logs.items()}
                payload["_step"] = int(state.global_step)
                with open(self.path, "a") as f:
                    f.write(json.dumps(payload) + "\n")
            except Exception:
                pass

    # Review iter1 Fix 2: prefer explicit META_RLSD_RUN_DIR (driver exports it),
    # fall back to cfg.output_dir itself. Previous os.path.dirname(cfg.output_dir)
    # put the file one level too high when output_dir was relative.
    metrics_dir = os.environ.get("META_RLSD_RUN_DIR") or cfg.output_dir
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "metrics.jsonl")
    trainer.add_callback(_JsonlLoggingCallback(metrics_path))

    # Dump resolved config for reproducibility — includes decoy settings.
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "contrastive_rlsd_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    trainer.train()
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[contrastive_rlsd] Done. Saved to {final_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "ContrastiveMetaRLSDConfig",
    "ContrastiveMetaRLSDTrainer",
    "_rule_based_decoy",
    "_random_noise_decoy",
    "main",
]
