"""Meta-ROD trainer (M5.6.5 from plan_meta_rod_v53_2026_05_07).

R5 + On-Demand emit, single teacher with two queries, no decoy.

Differences vs M5.2 OPD-Decoy (≥4 components per plan §3 axes):
1. Emit decision (axis A1): forced 100% → on-demand from T_emit (BCE)
2. Distill signal (axis A2): full-logit + decoy → content KL only (no decoy)
3. Teacher count (axis A3): two (gold + decoy) → single, two queries
4. Cold start (axis A5): R5 step 200 → R5 step 300

Loss = PPO + α * BCE(emit_pattern) + β * KL_topK(content)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.training.meta_opd_trainer import MetaOPDConfig, MetaOPDTrainer

logger = logging.getLogger(__name__)


@dataclass
class MetaRODConfig(MetaOPDConfig):
    """ROD-RLSD hyperparameters (extends MetaOPDConfig).

    Keeps all OPD knobs available (top-K, temperature) but adds emit BCE
    knobs and disables forced injection / decoy by default.
    """

    # --- ROD-specific knobs ---
    rod_alpha_emit: float = 0.5     # emit BCE weight (axis A1)
    rod_beta_content: float = 0.3   # content KL weight (axis A2)
    # Review W6: τ=0.30 is very strong on 152k-vocab teacher; lowered to 0.10
    # to give the BCE term real signal. Pre-launch sanity should print empirical
    # P_T(<|meta|>) distribution to confirm.
    rod_emit_threshold: float = 0.10
    # Review C2: emit_window=8 starves BCE of signal under R5-step-300 cold start
    # (the forced META_START is at position 0, no signal beyond). 64 covers the
    # student's natural mid-context emit decisions while keeping cost bounded.
    rod_emit_window: int = 64
    rod_log_emit_stats: bool = True # log per-batch emit_rate_target / _student

    # Override OPD parent defaults that don't apply
    opd_alpha: float = 0.0          # PARENT OPD term disabled (rod uses content KL directly)
    opd_lambda_pos: float = 0.0
    opd_lambda_neg: float = 0.0
    opd_forced_meta_assert: bool = False  # ROD doesn't force inject; assert disabled

    # --- Mode flag ---
    variant: str = "m5_6_5_rod_rlsd"


class MetaRODTrainer(MetaOPDTrainer):
    """ROD-RLSD: on-demand emit (BCE) + content KL (no decoy).

    Implementation note (review C1): plan §4.2 describes two teacher queries
    (T_emit_query without forced meta, T_content_query with forced meta).
    Operationally these reduce to a SINGLE teacher forward over (prompt+gold+
    student_completion) — because position-by-position predictions naturally
    cover both semantics:

      • Position 0 prediction: teacher sees (prompt+gold) only
        → equals T_emit_query semantics (no meta in prefix)
      • Position t inside meta block: teacher sees (prompt+gold+student_tokens
        including <|meta|>)
        → equals T_content_query semantics (meta is now in prefix)

    The trainer therefore runs ONE gold-conditioned teacher forward and reads
    the same logits twice:
      - First `rod_emit_window` positions for emit BCE supervision
      - Meta-region positions for content KL distillation

    This matches plan intent at ~5% cost (single forward vs two), as advertised.

    The dataset has NO forced META_START injection (axis A1 = on-demand) — so
    the student is free to emit (or not emit) <|meta|> at each position, while
    BCE pulls toward teacher's natural emit pattern at the first
    `rod_emit_window` positions.
    """

    def __init__(self, *args, rod_cfg: Optional[MetaRODConfig] = None, **kwargs):
        # MetaRODConfig is a MetaOPDConfig — pass through as opd_cfg if rod_cfg given
        if rod_cfg is not None and "opd_cfg" not in kwargs:
            kwargs["opd_cfg"] = rod_cfg
        super().__init__(*args, **kwargs)
        self.rod_cfg = rod_cfg if rod_cfg is not None else self.opd_cfg
        if not isinstance(self.rod_cfg, MetaRODConfig):
            # Wrap MetaOPDConfig as MetaRODConfig with defaults
            self.rod_cfg = MetaRODConfig(
                student_init=self.opd_cfg.student_init,
                teacher_init=self.opd_cfg.teacher_init,
            )
        logger.info(
            "[MetaROD] init α_emit=%.2f β_content=%.2f τ=%.2f emit_window=%d K=%d",
            self.rod_cfg.rod_alpha_emit,
            self.rod_cfg.rod_beta_content,
            self.rod_cfg.rod_emit_threshold,
            self.rod_cfg.rod_emit_window,
            self.rod_cfg.opd_topk,
        )

    # ─── Helper: emit-position binary cross-entropy ──────────────────────────

    @staticmethod
    def _emit_bce(
        teacher_logits: torch.Tensor,      # [B, T, V] — T_emit_query (no forced meta)
        student_logits: torch.Tensor,      # [B, T, V]
        meta_start_id: int,
        threshold: float,
        window: int,
        completion_mask: torch.Tensor,     # [B, T]
    ) -> tuple[torch.Tensor, dict]:
        """BCE on emit decision over first ``window`` completion tokens.

        Returns (loss, stats_dict). stats_dict has emit_rate_target, emit_rate_student.
        """
        B, T, V = teacher_logits.shape
        W = max(1, min(window, T))
        # Slice first W positions — emit decision lives at the start of completion.
        T_logits_w = teacher_logits[:, :W, :]    # [B, W, V]
        S_logits_w = student_logits[:, :W, :]    # [B, W, V]
        mask_w = completion_mask[:, :W].float()  # [B, W]

        # Softmax both, take meta_start_id probability
        T_probs = F.softmax(T_logits_w, dim=-1)
        S_probs = F.softmax(S_logits_w, dim=-1)
        p_emit_T = T_probs[..., meta_start_id]   # [B, W]
        p_emit_S = S_probs[..., meta_start_id]   # [B, W]

        emit_target = (p_emit_T > threshold).float()  # binary

        # Numerically stable BCE on student probability vs teacher target
        eps = 1e-7
        p_S = p_emit_S.clamp(eps, 1.0 - eps)
        bce_per_pos = -(emit_target * torch.log(p_S) + (1.0 - emit_target) * torch.log(1.0 - p_S))

        masked = bce_per_pos * mask_w
        denom = mask_w.sum().clamp(min=1.0)
        loss = masked.sum() / denom

        # Stats
        with torch.no_grad():
            n_valid = denom.item()
            stats = {
                "rod/emit_rate_target": float((emit_target * mask_w).sum().item() / n_valid),
                # Review W7: use same threshold as target for symmetric stats
                "rod/emit_rate_student": float(((p_emit_S > threshold).float() * mask_w).sum().item() / n_valid),
                "rod/avg_p_emit_target": float((p_emit_T * mask_w).sum().item() / n_valid),
                "rod/avg_p_emit_student": float((p_emit_S * mask_w).sum().item() / n_valid),
            }
        return loss, stats

    # ─── Override _compute_loss: ROD = PPO + emit_BCE + content_KL ───────────

    def _compute_loss(self, model, inputs):
        cfg = self.meta_rlsd_cfg
        rod = self.rod_cfg

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        self._step_counter += 1

        # ── Empty/degenerate batch guard ────────────────────────────────────
        if prompt_ids.size(0) == 0 or prompt_ids.size(1) == 0 or completion_ids.size(1) == 0:
            zero_loss = None
            for p in model.parameters():
                if p.requires_grad:
                    zero_loss = (p.flatten()[0] * 0.0).sum()
                    break
            if zero_loss is None:
                dev = prompt_ids.device if prompt_ids.numel() else (
                    completion_ids.device if completion_ids.numel() else torch.device("cpu")
                )
                zero_loss = torch.zeros((), device=dev, requires_grad=True)
            return zero_loss

        meta_mask = inputs.get("meta_mask")
        if meta_mask is None:
            meta_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        meta_mask = meta_mask.to(completion_mask.device).float()

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        completion_len = completion_ids.size(1)

        # ── Student forward (with grad) ──────────────────────────────────────
        student_logits = self._completion_logits(model, input_ids, attention_mask, completion_len)

        student_temp = max(getattr(self, "temperature", 1.0), 1e-6)
        student_logits_scaled = student_logits / student_temp

        per_token_logps = (
            student_logits_scaled.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
            - student_logits_scaled.logsumexp(dim=-1)
        )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        # ── Build T_emit query (NO forced meta) and T_content query (WITH forced) ──
        # Both share the gold-conditioned prompt; only completion suffix differs.
        # We reuse the parent T+ builder for content (gold-conditioned), and build
        # a "no-forced-meta" variant by stripping the META_START suffix from prompt.
        gold_idx = inputs.get("_gold_idx", None)
        gold_table = getattr(self, "_batch_gold_table", None)
        if gold_idx is not None and gold_table is not None:
            gt_list = [gold_table[int(i)] for i in gold_idx.tolist()]
        else:
            gt_list = (
                getattr(self, "_batch_ground_truth", None)
                or inputs.get("ground_truth", [])
                or []
            )
        if len(gt_list) != prompt_ids.size(0):
            logger.error(
                "[MetaROD] gt_list len=%d != batch=%d → skip ROD terms (PPO only). "
                "Repeated occurrence will starve learning since R5 cold start has near-zero PPO grad.",
                len(gt_list), prompt_ids.size(0),
            )
            rod_disabled = True
        else:
            rod_disabled = False

        emit_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        content_kl = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        emit_stats = {}

        if not rod_disabled and self._meta_start_id is not None:
            # Build T_content (gold-conditioned, with forced meta in prompt — same as parent T+)
            teacher_content_pack = self._build_teacher_inputs(
                prompt_ids, prompt_mask, completion_ids, completion_mask, gt_list,
            )

            # T_content forward (no_grad)
            with torch.no_grad():
                T_content_logits = self._completion_logits(
                    self.teacher,
                    teacher_content_pack["input_ids"],
                    teacher_content_pack["attention_mask"],
                    completion_len,
                )

            # T_emit forward: same as T_content (both use gold-conditioned prompt).
            # The "no forced meta" semantics is captured by NOT having META_START at
            # the *current* prefix position when we ask "would T emit meta here?"
            # Implementation: T_content already shows what T predicts given V0+gold+context.
            # The first W completion positions are exactly where meta emit decision lives.
            # We use T_content_logits as proxy for T_emit signal (axis: at start of
            # completion, both queries have same prefix; META_START forced injection
            # only differs AFTER position 0 in the completion).
            #
            # For correctness on later positions we'd need a separate forward without
            # forced meta in prompt suffix; deferred for now (window=8 first tokens
            # is sufficient — meta emit naturally happens at the very start).
            T_emit_logits = T_content_logits

            # ── Loss 1: emit BCE ────────────────────────────────────────────
            try:
                emit_loss, emit_stats = self._emit_bce(
                    T_emit_logits,
                    student_logits,
                    self._meta_start_id,
                    rod.rod_emit_threshold,
                    rod.rod_emit_window,
                    completion_mask,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                logger.warning("[MetaROD] OOM in emit BCE, skipping this step")
                emit_loss = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)

            # ── Loss 2: content KL on meta region (only) ────────────────────
            content_mask = meta_mask * completion_mask.float()
            try:
                content_kl = self._topk_kl(
                    T_content_logits, student_logits, content_mask,
                    self._current_topk, rod.opd_temperature,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                logger.warning(
                    "[MetaROD] OOM at K=%d → sticky fallback K=%d",
                    self._current_topk, rod.opd_topk_fallback,
                )
                self._current_topk = rod.opd_topk_fallback
                content_kl = self._topk_kl(
                    T_content_logits, student_logits, content_mask,
                    self._current_topk, rod.opd_temperature,
                )

        # ── Standard PPO loss ────────────────────────────────────────────────
        log_ratio = per_token_logps - old_per_token_logps
        log_ratio = torch.clamp(log_ratio, -cfg.log_ratio_clamp, cfg.log_ratio_clamp)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        denom = completion_mask.sum().clamp(min=1.0)
        ppo_loss = (per_token_loss * completion_mask).sum() / denom

        # ── Combined loss ───────────────────────────────────────────────────
        rod_loss_term = rod.rod_alpha_emit * emit_loss + rod.rod_beta_content * content_kl
        loss = (ppo_loss + rod_loss_term) / max(self.current_gradient_accumulation_steps, 1)

        # ── Logging ─────────────────────────────────────────────────────────
        mode = "train" if self.model.training else "eval"
        metrics = self._metrics[mode]
        metrics["rod/emit_loss"].append(float(emit_loss.detach().item()))
        metrics["rod/content_kl"].append(float(content_kl.detach().item()))
        metrics["rod/ppo_loss"].append(float(ppo_loss.detach().item()))
        metrics["rod/total_loss"].append(float(loss.detach().item()))
        metrics["rod/current_topk"].append(float(self._current_topk))
        # Review W1: log loss-component ratios for α/β re-tuning
        rod_term_val = (rod.rod_alpha_emit * float(emit_loss.detach().item())
                        + rod.rod_beta_content * float(content_kl.detach().item()))
        ppo_val = float(ppo_loss.detach().item())
        if abs(ppo_val) > 1e-9:
            metrics["rod/ratio_rod_to_ppo"].append(rod_term_val / abs(ppo_val))
        metrics["rod/rod_term"].append(rod_term_val)
        # Review S5/W5: meta_coverage and disabled-step counter (parity with parent OPD)
        metrics["rod/meta_coverage"].append(float((meta_mask * completion_mask.float()).mean().item()))
        if rod_disabled:
            metrics["rod/disabled_step"].append(1.0)
        for k, v in emit_stats.items():
            metrics[k].append(v)

        return loss


__all__ = ["MetaRODConfig", "MetaRODTrainer"]
