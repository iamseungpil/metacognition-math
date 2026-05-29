"""Utilities for veRL-native SDC (Shared-preserve Directional Credit).

This module keeps the SDC-specific logic lightweight and testable:
  - build response-region masks from decoded completions
  - shape token-wise advantages with teacher-guided factors

It does not depend on TRL. It is designed for the veRL driver path.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo import core_algos
from verl.utils import torch_functional as verl_F

from src.metacot.prompt import META_END
from src.training.meta_quality import assistant_offsets, score_meta_commit_quality
from src.training.meta_rlsd_data_pipeline import (
    _build_meta_mask,
    _build_postmeta_mask,
    _manual_offset_scan,
)


def _offsets_with_degenerate_guard(
    tokenizer,
    completion_ids,
    completion_text: str,
    raw_offsets,
):
    # Mirror the (0,0)-outlier guard used by _build_meta_mask / _build_postmeta_mask
    # so shared/diff masks stay aligned with meta/postmeta masks. Without this,
    # added-vocab tokenizers (e.g. some chat-template forks that added <|meta|>
    # after training) return offset (0,0) for those tokens, which shifts
    # shared/diff onto the wrong characters while meta/postmeta — which do run
    # this guard — land on the right ones.
    if not raw_offsets:
        return raw_offsets
    suspicious = sum(
        1 for i, (s, e) in enumerate(raw_offsets) if i > 0 and s == 0 and e == 0
    )
    if suspicious > max(1, int(len(raw_offsets) * 0.05)):
        return _manual_offset_scan(tokenizer, completion_ids, completion_text)
    return raw_offsets


_ANSWER_PHRASE_RE = re.compile(
    r"(?i)\b(the answer is|answer\s*:|thus[, ]|therefore[, ]|so[, ])"
)
_BOXED_CAPTURE_RE = re.compile(
    r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
)
_PUNCT_ONLY_RE = re.compile(r"^[\s\.,:;!\?\-\+\*/=\\\(\)\[\]\{\}]+$")


def postmeta_closure_reward(completions, ground_truth=None, **kwargs):
    """Closure-focused reward for shared post-meta structure.

    This isolates the part of the behavior we want to preserve:
      - a boxed commit after meta
      - short clean tail into the commit
      - no trailing drift / decoherence after boxing
    """
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) and c else str(c)
        q = score_meta_commit_quality(text)
        score = (
            0.24 * float(q.get("boxed_after_meta", 0.0))
            + 0.18 * float(q.get("post_meta_budget_efficiency", 0.0))
            - 0.24 * float(q.get("no_boxed_penalty", 0.0))
            - 0.16 * float(q.get("post_boxed_text_penalty", 0.0))
            - 0.18 * float(q.get("decoherence_penalty", 0.0))
        )
        rewards.append(max(-0.40, min(0.40, score)))
    return rewards


def _normalize_key_weights(config, reward_keys: list[str]) -> dict[str, float]:
    weights = config.get("gdpo_reward_weights", None)
    if weights is None:
        return {k: 1.0 for k in reward_keys}
    out: dict[str, float] = {}
    for key, weight in zip(reward_keys, list(weights)):
        out[key] = float(weight)
    for key in reward_keys:
        out.setdefault(key, 1.0)
    return out


def _group_keys(config, group_name: str) -> list[str]:
    groups = config.get("sdc_reward_groups", None)
    if groups is None:
        return []
    if OmegaConf.is_config(groups):
        groups = OmegaConf.to_container(groups, resolve=True)
    vals = groups.get(group_name, [])
    return [str(x) for x in vals]


def _token_piece(text: str, span: tuple[int, int]) -> str:
    s, e = span
    if e <= s:
        return ""
    return text[s:e]


def _shared_char_spans(text: str, post_start: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _ANSWER_PHRASE_RE.finditer(text, post_start):
        spans.append((m.start(), m.end()))
    boxed = _BOXED_CAPTURE_RE.search(text, post_start)
    if boxed is not None:
        prefix_len = len("\\boxed{")
        spans.append((boxed.start(), min(boxed.start() + prefix_len, boxed.end())))
        spans.append((boxed.end() - 1, boxed.end()))
    return spans


def build_sdc_region_masks(tokenizer, completion_ids, completion_text: str):
    """Build meta/shared/diff/body masks for a decoded completion."""
    # Lazy import to avoid circular import: verl_sdc imports from this module
    # at top-level. Reading the active mode here (set by verl_sdc.main_task)
    # lets us flag ``started_inside_meta`` for forced-meta modes so the
    # response's leading tokens count as meta even though the <|meta|> opener
    # is in the prompt, not the response.
    try:
        from src.training.verl_sdc import _ACTIVE_SDC_CONTEXT, _FORCED_META_MODES
        mode = _ACTIVE_SDC_CONTEXT.get("mode", "SDC_SHARED")
        started_inside_meta = mode in _FORCED_META_MODES
    except Exception:
        # Fall back to legacy behavior if context module isn't importable
        # (e.g., unit tests that import verl_sdc_utils in isolation).
        started_inside_meta = False
    meta_mask = _build_meta_mask(
        tokenizer, completion_ids, completion_text, started_inside_meta=started_inside_meta
    )
    postmeta_mask, fallback = _build_postmeta_mask(
        tokenizer, completion_ids, completion_text, meta_mask
    )
    expected_len = len(completion_ids)
    if expected_len == 0:
        zeros = torch.zeros(0, dtype=torch.float32)
        return {
            "meta_mask": zeros,
            "postmeta_mask": zeros,
            "postmeta_shared_mask": zeros,
            "postmeta_diff_mask": zeros,
            "body_mask": zeros,
            "fallback_triggered": 0.0,
        }

    _, raw_offsets = assistant_offsets(tokenizer, completion_text)
    offsets = _offsets_with_degenerate_guard(
        tokenizer, completion_ids, completion_text, raw_offsets
    )
    offsets = list(offsets)
    if len(offsets) < expected_len:
        offsets = offsets + [(0, 0)] * (expected_len - len(offsets))
    offsets = offsets[:expected_len]

    meta_end_positions = [m.end() for m in re.finditer(re.escape(META_END), completion_text)]
    post_start = meta_end_positions[-1] if meta_end_positions else -1
    shared_spans = _shared_char_spans(completion_text, post_start) if post_start >= 0 else []

    shared_mask = torch.zeros_like(postmeta_mask)
    for idx, span in enumerate(offsets):
        if idx >= shared_mask.numel():
            break
        if postmeta_mask[idx].item() <= 0.0:
            continue
        piece = _token_piece(completion_text, span)
        if any(not (span[1] <= s or span[0] >= e) for s, e in shared_spans):
            shared_mask[idx] = 1.0
            continue
        stripped = piece.strip()
        if not stripped:
            shared_mask[idx] = 1.0
            continue
        if _PUNCT_ONLY_RE.fullmatch(piece):
            shared_mask[idx] = 1.0
            continue

    diff_mask = torch.clamp(postmeta_mask - shared_mask, 0.0, 1.0)
    body_mask = torch.clamp(1.0 - meta_mask - postmeta_mask, 0.0, 1.0)

    return {
        "meta_mask": meta_mask.float(),
        "postmeta_mask": postmeta_mask.float(),
        "postmeta_shared_mask": shared_mask.float(),
        "postmeta_diff_mask": diff_mask.float(),
        "body_mask": body_mask.float(),
        "fallback_triggered": float(bool(fallback)),
    }


def _dilate_forward(mask: torch.Tensor, post_k: int) -> torch.Tensor:
    """Forward-dilate a binary mask by ``post_k`` tokens (extended meta region).

    For every True position ``t`` in ``mask`` (last dim = sequence), positions
    ``t, t+1, ..., t+post_k`` become True in the output. Positions before ``t``
    are NOT affected — this is a one-sided (forward only) dilation, matching
    the Plan v7.2.2 R18 specification "meta block + post K tokens".

    Implementation: repeated forward-roll + logical OR. ``torch.roll`` wraps
    around the boundary so we zero out the front column on each shift.

    Args:
        mask: tensor of shape ``[..., T]`` with 0/1 entries (any float/int dtype).
        post_k: non-negative number of tokens to extend forward. ``post_k=0``
            returns ``mask`` unchanged.

    Returns:
        Binary mask of same dtype/shape as input.
    """
    if post_k <= 0:
        return mask.clone()
    out = mask.clone()
    cur = mask
    for _ in range(post_k):
        shifted = torch.roll(cur, shifts=1, dims=-1)
        shifted[..., 0] = 0  # roll wraps; zero out front to prevent end→front leakage
        out = out + shifted
        cur = shifted
    return (out > 0).to(mask.dtype)


def compute_sdc_gdpo_advantage(
    *,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    batch: dict,
    non_tensor_batch: dict,
    config,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """veRL-native SDC: build a scalar GDPO advantage then shape it per token.

    Mode dispatch (algorithm.sdc_mode, mirrored from top-level mode at init):
      VANILLA_GRPO         — skip SDC factor; return base GDPO advantage directly.
      RLSD_META_ATTR       — meta region uses w_attr = exp(sign × (T+ − student));
                             shared/diff regions controlled by their lambdas.
      RLSD_META_CONTRAST   — meta region uses w_meta = exp(sign × (α × Δ+ + β × δ))
                             where Δ+ = T+ − student, δ = T+ − T−.
      RLSD_FORCED_META     — same combined formula as RLSD_META_CONTRAST; rollout
                             prompt is augmented to start inside <|meta|>, so
                             the meta_mask is guaranteed non-empty (resolves the
                             "meta empty 95%" pathology of meta-SFT under gold
                             conditioning, paper 2603.24472).
      OPSD_META            — same advantage path as RLSD_META_ATTR; the auxiliary
                             KL distillation loss is applied in the actor (TBD).
      SDC_SHARED / SDC_CORR_ONLY / SDC_CORR_META_PEN — legacy behavior (w_attr on
                             meta, w_shared on shared, w_diff on diff).
      ROD_MQ               — meta-quality factor on EXTENDED meta region
                             (meta block + post K tokens):
                               q_attr     = mean clip(T+ − student) over ext-meta
                               q_centered = q_attr − batch_median(q_attr)
                               w_meta_quality = clip(exp(sign × q_centered / τ), …)
                               w_meta     = w_attr × w_meta_quality   (PRODUCT)
      ROD_MQ_CONTRAST      — R18a + α/β contrast term: q_meta = α·q_attr + β·q_contrast
                             where q_contrast = mean clip(T+ − T−) over ext-meta.
      ROD_PT2_E21CTRL      — Arm 2 (deliverable #2): ROD_PT's w_attr × w_position
                             2-teacher PRODUCT but UN-CLIPPED — each factor uses
                             the log-symmetric bound [1/w_max, w_max] (C1 fix).
      STABLE_GFN_C2FIX     — Arm 3 (deliverable #3): same advantage-plane
                             dispatch as STABLE_GFN (the else: w_attr branch,
                             λ_meta=0 in the config so the factor is identically
                             1); the C1 un-clip is irrelevant on the λ=0 plane
                             and the C2 fix is a reward-head swap, so no new
                             w_meta branch is needed here (config-only arm).
    """
    reward_keys = [str(k) for k in config.get("gdpo_reward_keys", [])]
    assert reward_keys, "sdc_gdpo requires algorithm.gdpo_reward_keys"
    device = response_mask.device
    base_advantages, _ = core_algos.compute_gdpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )

    sdc_mode = str(config.get("sdc_mode", "SDC_SHARED"))

    # R0: vanilla GRPO. Bypass all SDC factor computation — just whiten the
    # base GDPO advantage, exactly like a non-SDC trainer would.
    #
    # MATCHED_E21RV2 (Arm 1, ADDITIVE): a NO-teacher matched-RLVR baseline.
    # It takes the SAME teacher-free advantage path as VANILLA_GRPO (no SDC
    # factor, just whiten the multi-head GDPO advantage). This OR-clause only
    # ADDS a new mode to the early-return; the `== "VANILLA_GRPO"` predicate is
    # still independently true for VANILLA_GRPO, so VANILLA_GRPO behaviour is
    # byte-identical. No new mode was given the SDC factor path.
    if sdc_mode == "VANILLA_GRPO" or sdc_mode == "MATCHED_E21RV2":
        advantages = base_advantages * response_mask
        advantages = verl_F.masked_whiten(advantages, response_mask) * response_mask
        return advantages, advantages

    teacher_pos = batch["sdc_teacher_pos_log_probs"].to(device)
    teacher_neg = batch["sdc_teacher_neg_log_probs"].to(device)
    student_logp = batch["old_log_probs"].to(device)

    meta_mask = batch["sdc_meta_mask"].to(device)
    shared_mask = batch["sdc_postmeta_shared_mask"].to(device)
    diff_mask = batch["sdc_postmeta_diff_mask"].to(device)
    body_mask = batch["sdc_body_mask"].to(device)

    seq_adv = verl_F.masked_mean(base_advantages, response_mask, axis=-1)
    seq_adv = seq_adv.unsqueeze(1)
    sign = torch.sign(seq_adv)

    clamp = float(config.get("sdc_log_ratio_clamp", 10.0))
    clip_eps = float(config.get("sdc_clip_eps_w", 0.2))
    shared_tau = float(config.get("sdc_shared_tau", 0.5))

    attr_log = torch.clamp(teacher_pos - student_logp, -clamp, clamp)
    delta = torch.clamp(teacher_pos - teacher_neg, -clamp, clamp)
    shared_anchor = torch.clamp(0.5 * (teacher_pos + teacher_neg) - student_logp, -clamp, clamp)

    w_attr = torch.clamp(torch.exp(sign * attr_log), 1.0 - clip_eps, 1.0 + clip_eps)
    w_shared = torch.clamp(torch.exp(sign * shared_anchor), 1.0 - clip_eps, 1.0 + clip_eps)
    w_diff = torch.clamp(torch.exp(sign * delta), 1.0 - clip_eps, 1.0 + clip_eps)

    # Meta-region weight: mode-specific.
    # R2 combined form keeps RLSD's multiplicative-magnitude principle intact —
    # we never flip the sign of advantage, only modulate magnitude. The teacher
    # signal mixes attractive (Δ+) and contrastive (δ = T+ − T−) terms.
    # R5 (RLSD_FORCED_META) reuses the same combined α·Δ+ + β·δ formula —
    # the only difference vs R2 is that the rollout starts inside <|meta|> so
    # the meta_mask is non-empty for every sample (resolves the meta-empty 95%
    # pathology); the advantage shaping math itself is identical.
    if sdc_mode in {"RLSD_META_CONTRAST", "RLSD_FORCED_META"}:
        alpha_attr = float(config.get("sdc_alpha_attr", 0.5))
        beta_contrast = float(config.get("sdc_beta_contrast", 0.5))
        if alpha_attr < 0 or beta_contrast < 0:
            raise ValueError(
                f"sdc_alpha_attr={alpha_attr}, sdc_beta_contrast={beta_contrast}: "
                "both must be non-negative"
            )
        if (alpha_attr + beta_contrast) <= 0:
            raise ValueError(
                f"sdc_alpha_attr + sdc_beta_contrast = {alpha_attr + beta_contrast}: "
                "must be > 0 (else combined log-ratio is identically zero, "
                "use RLSD_META_ATTR with sdc_lambda_meta=0 instead)"
            )
        combined_log = torch.clamp(
            alpha_attr * attr_log + beta_contrast * delta,
            -clamp,
            clamp,
        )
        w_meta = torch.clamp(
            torch.exp(sign * combined_log), 1.0 - clip_eps, 1.0 + clip_eps
        )
    elif sdc_mode in ("ROD_PT", "ROD_PT_DEGEN"):
        # ROD-PT (Plan v5.17 FINAL): w_meta = w_attr × w_position  (PRODUCT).
        # ROD_PT_DEGEN (R16): identical w_meta logic — degeneration_penalty is
        # composed at the GDPO reward-head plane via its own weight (0.3),
        # NOT inside the meta-region multiplicative factor.
        #
        # log_prob_meta = P_T(<|meta|> | prefix before the student's first meta
        # position p), from the frozen ref teacher → it is ≤ 0 always, so the
        # position factor is NOT a symmetric amplifier:
        #   w_position = clip(exp(sign · log_prob_meta), 1−ε, 1+ε)
        #   • sign = +1 (correct rollout):  w_position ∈ [1−ε, 1]  — a meta-
        #       position the teacher rarely opens at gets its (positive) meta-token
        #       advantage gently DAMPENED toward 1−ε; never amplified above 1.
        #   • sign = −1 (wrong rollout):    w_position ∈ [1, 1+ε]  — a meta-
        #       position the teacher *is* likely to open at gets its (negative)
        #       meta-token advantage pushed HARDER toward 1+ε.
        # RLSD invariant: `sign` is preserved exactly; only the magnitude moves.
        log_prob_meta = batch.get(
            "sdc_position_log_prob_meta",
            torch.zeros(student_logp.size(0), device=device),
        ).to(device)
        if log_prob_meta.dim() == 1:
            log_prob_meta = log_prob_meta.unsqueeze(1)  # [B, 1]
        w_position = torch.clamp(
            torch.exp(sign * log_prob_meta), 1.0 - clip_eps, 1.0 + clip_eps
        )  # [B, 1]
        w_meta = w_attr * w_position  # [B, T] × [B, 1] → [B, T] PRODUCT
    elif sdc_mode in ("ROD_MQ", "ROD_MQ_CONTRAST", "ROD_MQ_CONTRAST_INJECT"):
        # R18a/R18b (Plan v7.2.2 codex round 5 LOCK): meta-quality verifiable
        # signal on EXTENDED meta region (meta block + post K=10 tokens).
        #
        # ROD_MQ      : q_meta = α × q_attr                                  (T+ only)
        # ROD_MQ_CONTRAST : q_meta = α × q_attr + β × q_contrast             (T+ AND T-)
        #   q_attr     = mean over extended meta of clip(T+ − student, ±10)
        #   q_contrast = mean over extended meta of clip(T+ − T−,        ±10)
        #
        # Sign-preserving multiplicative factor (RLSD invariant: sign never flips):
        #   q_centered     = q_meta − batch_median(q_meta)
        #   w_meta_quality = clip(exp(sign × q_centered / τ), 1−ε, 1+ε)   [B, 1]
        #   w_meta         = w_attr × w_meta_quality   PRODUCT — same family as ROD_PT
        post_k = int(config.get("sdc_meta_post_k", 10))
        extended_meta_mask = _dilate_forward(meta_mask, post_k=post_k)

        # codex round 7 fix: _meta_mask_from_token_ids covers BOTH the tags
        # themselves AND the inner reasoning (see meta_rlsd_data_pipeline:101).
        # Plan §9.2 requires content-only mean — explicitly subtract tag positions.
        # Build tag mask from `responses` (token ids in batch) using known
        # meta tag ids (looked up once via _ACTIVE_SDC_CONTEXT tokenizer).
        meta_tag_mask = torch.zeros_like(extended_meta_mask)
        try:
            from src.training.verl_sdc import _ACTIVE_SDC_CONTEXT
            from src.training.meta_rlsd_data_pipeline import _meta_token_ids_safe
            tok = _ACTIVE_SDC_CONTEXT.get("tokenizer")
            if tok is not None and "responses" in batch.keys():
                response_ids = batch["responses"].to(device)
                start_ids, end_ids = _meta_token_ids_safe(tok)
                tag_id_set = set()
                if start_ids:
                    tag_id_set |= set(start_ids)
                if end_ids:
                    tag_id_set |= set(end_ids)
                if tag_id_set:
                    tag_id_tensor = torch.tensor(
                        sorted(tag_id_set), device=device, dtype=response_ids.dtype
                    )
                    # broadcast: response_ids [B, T] vs tag_id_tensor [N] -> [B, T]
                    meta_tag_mask = torch.isin(response_ids, tag_id_tensor).to(
                        extended_meta_mask.dtype
                    )
        except Exception:
            # Best-effort: tests with mock batch / missing tokenizer fall back
            # to pre-codex-r7 behavior (tag-inclusive). Documented limitation.
            pass

        # Content-only extended mask: tags excluded per Plan §9.2.
        meta_content_mask = (
            extended_meta_mask * response_mask * (1.0 - meta_tag_mask)
        ).float()

        alpha_attr = float(config.get("sdc_alpha_attr", 1.0))
        beta_contrast = float(config.get("sdc_beta_contrast", 0.0))

        gain_attr = torch.clamp(teacher_pos - student_logp, -clamp, clamp)
        denom = meta_content_mask.sum(-1).clamp_min(1.0)
        q_attr = (gain_attr * meta_content_mask).sum(-1) / denom  # [B]

        if sdc_mode in ("ROD_MQ_CONTRAST", "ROD_MQ_CONTRAST_INJECT"):
            gain_contrast = torch.clamp(teacher_pos - teacher_neg, -clamp, clamp)
            q_contrast = (gain_contrast * meta_content_mask).sum(-1) / denom  # [B]
            q_meta = alpha_attr * q_attr + beta_contrast * q_contrast  # codex r5: + not −
        else:
            q_meta = alpha_attr * q_attr

        # Batch-median centered (detached: no gradient through the centering term).
        q_centered = q_meta - q_meta.median().detach()

        tau = float(config.get("sdc_meta_quality_tau", 1.0))
        if tau <= 0:  # codex r7 fix: tau must be positive (exp scaling denominator)
            raise ValueError(f"sdc_meta_quality_tau must be > 0, got {tau}")
        # sign is [B, 1] from earlier; flatten to [B] for per-sequence factor.
        sign_seq = sign.squeeze(-1) if sign.dim() > 1 else sign
        w_meta_quality_seq = torch.clamp(
            torch.exp(sign_seq * q_centered / tau),
            1.0 - clip_eps,
            1.0 + clip_eps,
        )  # [B]
        w_meta_quality = w_meta_quality_seq.unsqueeze(1)  # [B, 1] broadcast vs w_attr
        w_meta = w_attr * w_meta_quality  # PRODUCT, RLSD invariant preserved
    elif sdc_mode == "RLSD_FAITHFUL_META":
        # R20 direction B core (GFN_SGFN_IMPROVEMENT_PLAN iter-3 SURVEY-GROUNDED
        # LOCK; project_gfn_sgfn_plan_v3_lock). "RLSD-faithful meta-token credit":
        # restore the RLSD sign/magnitude separation that ROD_*/clip break.
        #
        #   SIGN     ← env reward ONLY. `sign = torch.sign(seq_adv)`; seq_adv is
        #              the mean of base_advantages, which here is the
        #              correctness-ONLY GDPO advantage — RLSD_FAITHFUL_META's
        #              REWARD_CONFIGS has NO meta_penalty head, so the asymmetric
        #              presence-only meta_penalty (diagnosed cause C2) does NOT
        #              inject any teacher/presence sign onto the meta region.
        #   MAGNITUDE ← teacher ONLY, via attr_log = clamp(T+ − student, ±clamp).
        #
        # vs w_attr (the throttled ROD path): w_attr clamps exp(sign·attr_log)
        # to [1−ε, 1+ε] = [0.8, 1.2] (clip_eps=0.2). Since |attr_log| is
        # typically ≫ 0.2, that clip SATURATES nearly every meta token to the
        # ±18% rail — it destroys the teacher's *relative magnitude ordering*
        # (the diagnosed cause C1, the −14pt-vs-E21Rv2 throttle). The
        # RLSD-faithful weight removes that order-changing clip and instead
        # applies a single LOG-SYMMETRIC bound w_meta ∈ [1/w_max, w_max]
        # (default w_max=4.0): ~20× wider than [0.8,1.2], so the teacher
        # magnitude ordering is preserved across the realistic operating range
        # and only the extreme tails saturate (numerical-stability bound, NOT
        # an order-changing throttle). Sign is preserved EXACTLY: exp(·)>0 ⇒
        # the clamped w_meta>0 ⇒ factor = (1−λ)+λ·w_meta > 0 ⇒ advantage sign
        # never flips (RLSD invariant intact). Teacher magnitude is detached in
        # effect because veRL advantages carry no policy gradient downstream.
        w_max = float(config.get("sdc_faithful_w_max", 4.0))
        if w_max <= 1.0:
            raise ValueError(
                f"sdc_faithful_w_max must be > 1.0 (log-symmetric magnitude "
                f"bound), got {w_max}"
            )
        w_meta = torch.clamp(torch.exp(sign * attr_log), 1.0 / w_max, w_max)
    elif sdc_mode == "ROD_PT2_E21CTRL":
        # Arm 2 (deliverable #2; EXPERIMENT_PLAN_ARMS.md "Recipe X"). The
        # R10/ROD_PT TWO-teacher structure  w_attr(content) × w_position  but
        # UN-CLIPPED — the SINGLE structural fix for diagnosed cause C1.
        #
        #   ROD_PT (the throttled path, lines above):
        #     w_attr     = clamp(exp(sign·attr_log),     1−ε, 1+ε)  = [0.8,1.2]
        #     w_position = clamp(exp(sign·log_prob_meta), 1−ε, 1+ε) = [0.8,1.2]
        #     w_meta     = w_attr × w_position  ∈ [0.64, 1.44]   (clipped ±20%
        #                  no-op = C1: the teacher's relative magnitude ordering
        #                  is destroyed because |attr_log| ≫ 0.2 saturates the
        #                  rail on nearly every meta token).
        #
        #   ROD_PT2_E21CTRL (here): identical 2-teacher PRODUCT but each factor
        #     uses the RLSD_FAITHFUL_META log-symmetric magnitude bound
        #     [1/w_max, w_max] (default w_max=4.0, ~20× wider) instead of the
        #     order-changing ±ε clip. Sign is preserved EXACTLY (exp(·)>0 ⇒
        #     each clamped factor >0 ⇒ product >0 ⇒ advantage sign never flips,
        #     RLSD invariant intact). The teacher's relative magnitude ordering
        #     survives across the realistic operating range; only the extreme
        #     tails saturate (numerical-stability bound, NOT a throttle).
        #
        # Re-uses the SAME sdc_faithful_w_max key as RLSD_FAITHFUL_META so the
        # un-clip bound is a single shared knob across the C1-fix arms.
        w_max = float(config.get("sdc_faithful_w_max", 4.0))
        if w_max <= 1.0:
            raise ValueError(
                f"sdc_faithful_w_max must be > 1.0 (log-symmetric magnitude "
                f"bound), got {w_max}"
            )
        log_prob_meta = batch.get(
            "sdc_position_log_prob_meta",
            torch.zeros(student_logp.size(0), device=device),
        ).to(device)
        if log_prob_meta.dim() == 1:
            log_prob_meta = log_prob_meta.unsqueeze(1)  # [B, 1]
        # UN-CLIPPED content teacher (vs the clipped w_attr above).
        w_attr_unclipped = torch.clamp(
            torch.exp(sign * attr_log), 1.0 / w_max, w_max
        )
        # UN-CLIPPED position teacher (vs the clipped w_position in ROD_PT).
        w_position_unclipped = torch.clamp(
            torch.exp(sign * log_prob_meta), 1.0 / w_max, w_max
        )  # [B, 1]
        w_meta = w_attr_unclipped * w_position_unclipped  # [B,T]×[B,1] PRODUCT
    else:
        # Existing modes (SDC_SHARED, SDC_CORR_ONLY, SDC_CORR_META_PEN,
        # RLSD_META_ATTR, OPSD_META): meta uses pure attractive.
        w_meta = w_attr

    orig_shared_mask = shared_mask
    teacher_shared_gate = (delta.abs() <= shared_tau).float()
    shared_mask = torch.clamp(orig_shared_mask * teacher_shared_gate, 0.0, 1.0)
    diff_mask = torch.clamp(diff_mask + orig_shared_mask * (1.0 - teacher_shared_gate), 0.0, 1.0)

    lam_meta = float(config.get("sdc_lambda_meta", 0.5))
    lam_shared = float(config.get("sdc_lambda_shared", 0.25))
    lam_diff = float(config.get("sdc_lambda_diff", 0.30))

    factor = (
        meta_mask * ((1.0 - lam_meta) + lam_meta * w_meta)
        + shared_mask * ((1.0 - lam_shared) + lam_shared * w_shared)
        + diff_mask * ((1.0 - lam_diff) + lam_diff * w_diff)
        + body_mask
    )
    factor = torch.where((meta_mask + shared_mask + diff_mask + body_mask) > 0, factor, torch.ones_like(factor))
    advantages = seq_adv * factor * response_mask
    # Audit A fix (codex r13 LOCK, Option A): post-factor whiten removed.
    # base advantages from core_algos.compute_gdpo_outcome_advantage() are already whitened.
    # Re-applying masked_whiten here subtracted mean → could flip token-level sign,
    # violating RLSD invariant ("sign × magnitude only"). Option A: trust base whitening.

    # Plan v7.2.7 D17 intent verification metrics (codex r13 LOCK).
    # Gated by `wandb.run` — never log if wandb not initialized (tests, offline runs).
    # Four categories: (1) advantage/factor invariant — Audit A re-occurrence guard,
    # (2) mask/parser health (mode-agnostic), (3) teacher signal health (skip
    # VANILLA_GRPO — no teacher), (4) clip-rate watchdogs. Use NaN for absent
    # metrics rather than 0 (codex r13 explicit) so wandb dashboards do not
    # falsely show "0" for "not computed in this mode".
    try:
        import wandb
        if wandb.run is not None:
            with torch.no_grad():
                # Lazy import here to avoid module-load circularity in tests.
                try:
                    from src.training.verl_sdc import _CONTRASTIVE_MODES as _CM
                except Exception:
                    _CM = {
                        "SDC_SHARED",
                        "SDC_CORR_ONLY",
                        "SDC_CORR_META_PEN",
                        "RLSD_META_CONTRAST",
                        "ROD_MQ_CONTRAST",
                        "ROD_MQ_CONTRAST_INJECT",
                    }

                # ---- (1) Advantage / factor invariant (Audit A re-occurrence guard) ----
                seq_signs = torch.sign(seq_adv).expand_as(advantages)
                adv_signs = torch.sign(advantages)
                mask_nonzero = (seq_adv.expand_as(advantages) != 0) & (response_mask > 0)
                denom_nonzero = mask_nonzero.float().sum().clamp_min(1)
                sign_flip = ((seq_signs != adv_signs) & mask_nonzero).float().sum() / denom_nonzero

                resp_any = (response_mask > 0).any()
                resp_gt1 = (response_mask > 0).sum() > 1
                adv_shaped_mean = (
                    float(advantages[response_mask > 0].mean()) if resp_any else float("nan")
                )
                adv_shaped_std = (
                    float(advantages[response_mask > 0].std()) if resp_gt1 else float("nan")
                )
                factor_mean = (
                    float(factor[response_mask > 0].mean()) if resp_any else float("nan")
                )
                factor_min = (
                    float(factor[response_mask > 0].min()) if resp_any else float("nan")
                )
                factor_max = (
                    float(factor[response_mask > 0].max()) if resp_any else float("nan")
                )

                wandb.log({
                    "intent/adv_sign_flip_rate": float(sign_flip),
                    "intent/adv_base_sign_pos_rate": float((seq_adv > 0).float().mean()),
                    "intent/adv_shaped_mean": adv_shaped_mean,
                    "intent/adv_shaped_std": adv_shaped_std,
                    "intent/factor_mean": factor_mean,
                    "intent/factor_min": factor_min,
                    "intent/factor_max": factor_max,
                    "intent/factor_clip_low_rate": float(
                        (factor <= 1.0 - clip_eps + 1e-6).float().mean()
                    ),
                    "intent/factor_clip_high_rate": float(
                        (factor >= 1.0 + clip_eps - 1e-6).float().mean()
                    ),
                })

                # ---- (2) Mask / parser health (mode-agnostic) ----
                partition_sum = meta_mask + shared_mask + diff_mask + body_mask
                wandb.log({
                    "intent/meta_token_frac": float(meta_mask.float().mean()),
                    "intent/no_meta_rate": float(
                        (meta_mask.sum(-1) == 0).float().mean()
                    ),
                    "intent/mask_partition_coverage_rate": float(
                        (partition_sum > 0).float().mean()
                    ),
                })

                # ---- (3) Teacher signal health (skip VANILLA_GRPO — no teacher) ----
                # NOTE: sdc_mode == VANILLA_GRPO already returned early above,
                # so we are guaranteed to have teacher_pos / student_logp here.
                # The check below is defensive for future modes.
                if sdc_mode != "VANILLA_GRPO":
                    meta_sum = meta_mask.sum().clamp_min(1)
                    gain_attr = (teacher_pos - student_logp).clamp(-clamp, clamp)
                    meta_pos = (meta_mask > 0)
                    if meta_pos.any():
                        flat = gain_attr[meta_pos].flatten()
                        q05 = float(torch.quantile(flat, 0.05))
                        q50 = float(torch.quantile(flat, 0.50))
                        q95 = float(torch.quantile(flat, 0.95))
                    else:
                        q05 = q50 = q95 = float("nan")

                    teacher_neg_meta_mean = (
                        float((teacher_neg * meta_mask).sum() / meta_sum)
                        if sdc_mode in _CM
                        else float("nan")
                    )

                    wandb.log({
                        "intent/teacher_pos_logp_meta_mean": float(
                            (teacher_pos * meta_mask).sum() / meta_sum
                        ),
                        "intent/teacher_neg_logp_meta_mean": teacher_neg_meta_mean,
                        "intent/student_logp_meta_mean": float(
                            (student_logp * meta_mask).sum() / meta_sum
                        ),
                        "intent/teacher_gain_attr_p05": q05,
                        "intent/teacher_gain_attr_p50": q50,
                        "intent/teacher_gain_attr_p95": q95,
                        "intent/teacher_gain_attr_clip_rate": float(
                            (gain_attr.abs() >= clamp - 1e-6).float().mean()
                        ),
                    })
    except Exception:
        # wandb optional — never crash training on a metric-logging failure.
        pass

    return advantages, advantages
