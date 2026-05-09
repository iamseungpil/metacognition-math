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
    if sdc_mode == "VANILLA_GRPO":
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
    elif sdc_mode == "ROD_PT":
        # ROD-PT (Plan v5.17 FINAL): R5 attribution × position factor (PRODUCT).
        # Position factor multiplicatively amplifies w_attr based on teacher's prob
        # of META at first emit position p:
        #   w_position = clip(exp(sign × log_prob_meta), 1-ε, 1+ε)
        #   w_combined = w_attr × w_position
        # RLSD invariant 보존: sign는 sign 변수로 보존, magnitude만 modulate.
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
    advantages = verl_F.masked_whiten(advantages, response_mask) * response_mask
    return advantages, advantages
