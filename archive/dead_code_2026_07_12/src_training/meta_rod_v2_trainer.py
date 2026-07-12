"""Meta-ROD v2 trainer (M5.6.5 v2 from plan_meta_rod_v53 §9 v5.4 update).

Post-codex-review design. Differences vs M5.6.5 v1 (`meta_rod_trainer.py`):

  v1                                  | v2
  ------------------------------------|--------------------------------------------
  Cold start: R5 step 300 (forced)    | Cold start: base SFT v8 (no force)
  Hard BCE (p_T > τ)                  | Soft Bernoulli BCE (target = continuous p_T)
  emit_window = 64 (first positions)  | 16 random non-meta body positions per rollout
  Single teacher forward (collapsed)  | Two real forwards: T_emit (no force) + T_content (META injected at sampled positions)
  Meta floor reward active            | Meta floor reward disabled, replaced by:
                                      |   emit_rate_penalty = γ * (EMA(p_T) - actual_rate)²

Loss: PPO + α·soft_BCE + β·content_KL + γ·emit_rate_penalty
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.training.meta_opd_trainer import MetaOPDConfig, MetaOPDTrainer

logger = logging.getLogger(__name__)


@dataclass
class MetaRODv2Config(MetaOPDConfig):
    """ROD v2 hyperparameters."""

    # --- v2 emit BCE (soft Bernoulli distillation) ---
    rod_alpha_emit: float = 0.5
    rod_beta_content: float = 0.3
    rod_gamma_rate: float = 0.1

    # --- v2 position sampling ---
    rod_v2_n_sampled: int = 16        # sampled non-meta body positions per rollout
    rod_v2_min_pos: int = 8           # exclude first N completion tokens (likely inside meta)
    rod_v2_seed: int = 42

    # --- emit-rate target (EMA of teacher emit prob over training) ---
    rod_v2_target_rate_ema: float = 0.05  # initial; EMA-updated each step
    rod_v2_target_rate_alpha: float = 0.95  # EMA smoothing factor

    # Disable parent OPD term + forced injection asserts
    opd_alpha: float = 0.0
    opd_lambda_pos: float = 0.0
    opd_lambda_neg: float = 0.0
    opd_forced_meta_assert: bool = False

    variant: str = "m5_6_5_rod_v2"


class MetaRODv2Trainer(MetaOPDTrainer):
    """ROD v2: soft BCE + sampled positions + two real teacher forwards.

    For each student rollout:
      1. Identify non-meta body positions (mask out inside-meta).
      2. Randomly sample N positions from those body positions.
      3. Run T_emit forward: gold-conditioned prefix only — gives p_emit_T at each
         sampled position.
      4. For each sampled position p: build T_content input by injecting META_START
         at position p in the completion → run T_content forward → KL on the next
         few tokens (content distillation).
      5. soft_BCE = -[p_T·log(p_S) + (1-p_T)·log(1-p_S)] over sampled positions.
      6. emit_rate_penalty = γ·(EMA(p_T) - actual_rate)².
    """

    def __init__(self, *args, rod_v2_cfg: Optional[MetaRODv2Config] = None, **kwargs):
        if rod_v2_cfg is not None and "opd_cfg" not in kwargs:
            kwargs["opd_cfg"] = rod_v2_cfg
        super().__init__(*args, **kwargs)
        self.rod_v2_cfg = rod_v2_cfg if rod_v2_cfg is not None else self.opd_cfg
        if not isinstance(self.rod_v2_cfg, MetaRODv2Config):
            self.rod_v2_cfg = MetaRODv2Config(
                student_init=self.opd_cfg.student_init,
                teacher_init=self.opd_cfg.teacher_init,
            )
        self._rod_rng = random.Random(self.rod_v2_cfg.rod_v2_seed)
        self._target_rate_ema = self.rod_v2_cfg.rod_v2_target_rate_ema
        logger.info(
            "[MetaROD v2] init α=%.2f β=%.2f γ=%.2f n_sampled=%d min_pos=%d "
            "target_rate_init=%.3f ema_α=%.2f K=%d",
            self.rod_v2_cfg.rod_alpha_emit,
            self.rod_v2_cfg.rod_beta_content,
            self.rod_v2_cfg.rod_gamma_rate,
            self.rod_v2_cfg.rod_v2_n_sampled,
            self.rod_v2_cfg.rod_v2_min_pos,
            self.rod_v2_cfg.rod_v2_target_rate_ema,
            self.rod_v2_cfg.rod_v2_target_rate_alpha,
            self.rod_v2_cfg.opd_topk,
        )

    # ─── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sample_body_positions(
        completion_mask: torch.Tensor,   # [B, T]
        meta_mask: torch.Tensor,         # [B, T] — 1.0 inside meta blocks
        n_sampled: int,
        min_pos: int,
        rng: random.Random,
    ) -> List[List[int]]:
        """Per-batch list of sampled non-meta body positions.

        Returns list of length B; each entry is a list of up to ``n_sampled`` indices
        into completion positions where: completion_mask[b, t] == 1, meta_mask[b, t] == 0,
        and t >= min_pos.
        """
        B = completion_mask.size(0)
        out: List[List[int]] = []
        body = (completion_mask * (1.0 - meta_mask)).cpu().numpy()
        T = completion_mask.size(1)
        for b in range(B):
            valid_idx = [t for t in range(min_pos, T) if body[b, t] > 0]
            if not valid_idx:
                out.append([])
                continue
            k = min(n_sampled, len(valid_idx))
            sampled = rng.sample(valid_idx, k)
            out.append(sorted(sampled))
        return out

    @staticmethod
    def _soft_bce_at_positions(
        student_logits: torch.Tensor,    # [B, T, V] — UNSCALED student logits
        teacher_logits: torch.Tensor,    # [B, T, V] — T_emit logits
        meta_start_id: int,
        positions_per_b: List[List[int]],
    ) -> Tuple[torch.Tensor, dict]:
        """Soft Bernoulli BCE: target = p_T(meta), continuous in [0,1].

        Returns (loss, stats).
        """
        device = student_logits.device
        loss_terms: List[torch.Tensor] = []
        target_probs: List[float] = []
        student_probs: List[float] = []

        for b, positions in enumerate(positions_per_b):
            if not positions:
                continue
            for t in positions:
                T_logit = teacher_logits[b, t]      # [V]
                S_logit = student_logits[b, t]      # [V]
                p_T = F.softmax(T_logit, dim=-1)[meta_start_id]
                p_S = F.softmax(S_logit, dim=-1)[meta_start_id]
                p_S_clamp = p_S.clamp(1e-7, 1.0 - 1e-7)
                # Soft Bernoulli BCE — target = continuous p_T (not thresholded)
                bce = -(p_T.detach() * torch.log(p_S_clamp)
                        + (1.0 - p_T.detach()) * torch.log(1.0 - p_S_clamp))
                loss_terms.append(bce)
                target_probs.append(float(p_T.item()))
                student_probs.append(float(p_S.item()))

        if not loss_terms:
            return torch.zeros((), device=device, dtype=student_logits.dtype), {
                "rod_v2/n_sampled_total": 0.0,
                "rod_v2/avg_p_T": 0.0,
                "rod_v2/avg_p_S": 0.0,
            }

        loss = torch.stack(loss_terms).mean()
        stats = {
            "rod_v2/n_sampled_total": float(len(loss_terms)),
            "rod_v2/avg_p_T": float(sum(target_probs) / len(target_probs)),
            "rod_v2/avg_p_S": float(sum(student_probs) / len(student_probs)),
        }
        return loss, stats

    def _content_kl_at_emitted_meta(
        self,
        student_logits: torch.Tensor,    # [B, T_completion, V]
        teacher_logits: torch.Tensor,    # [B, T_completion, V] — gold-conditioned
        meta_mask: torch.Tensor,         # [B, T_completion] — 1.0 inside emitted meta blocks
        completion_mask: torch.Tensor,   # [B, T_completion]
    ) -> Tuple[torch.Tensor, int]:
        """ALT-2 (post-codex-consensus): KL on STUDENT-EMITTED META region.

        Cold-start emit rates already 87-97% (measured on v8_meta_inside_strict_sft),
        so student-emitted META spans are dense from step 0. No counterfactual META
        injection — just gold-conditioned teacher predicts where the student already
        emitted META. Avoids BCE/KL tug-of-war (BCE pulls S(p)→META; KL only fires
        AFTER student already emitted META, so no conflict).

        Reuses parent _topk_kl with `meta_mask * completion_mask`.
        Cost: 0 extra forward (teacher logits already computed for emit BCE).

        Returns (kl_scalar, n_meta_positions_supervised).
        """
        device = student_logits.device
        K = self._current_topk
        kl_mask = meta_mask * completion_mask.float()
        n_pos = int(kl_mask.sum().item())
        if n_pos == 0:
            return torch.zeros((), device=device, dtype=student_logits.dtype), 0
        kl = self._topk_kl(teacher_logits, student_logits, kl_mask, K, temperature=1.0)
        return kl, n_pos

    # ─── _compute_loss (override) ───────────────────────────────────────────

    def _compute_loss(self, model, inputs):
        cfg = self.meta_rlsd_cfg
        v2 = self.rod_v2_cfg

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        self._step_counter += 1

        # Empty/degenerate batch guard
        if prompt_ids.size(0) == 0 or prompt_ids.size(1) == 0 or completion_ids.size(1) == 0:
            for p in model.parameters():
                if p.requires_grad:
                    return (p.flatten()[0] * 0.0).sum()
            return torch.zeros((), device=prompt_ids.device, requires_grad=True)

        meta_mask = inputs.get("meta_mask")
        if meta_mask is None:
            meta_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        meta_mask = meta_mask.to(completion_mask.device).float()

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        completion_len = completion_ids.size(1)

        # Student forward
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

        # Build T_emit (gold-conditioned, no META injected — same as parent T+)
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

        soft_bce = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        content_kl = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        rate_penalty = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
        sampling_stats: dict = {}

        if len(gt_list) == prompt_ids.size(0) and self._meta_start_id is not None:
            teacher_pack = self._build_teacher_inputs(
                prompt_ids, prompt_mask, completion_ids, completion_mask, gt_list,
            )

            with torch.no_grad():
                T_emit_logits = self._completion_logits(
                    self.teacher,
                    teacher_pack["input_ids"],
                    teacher_pack["attention_mask"],
                    completion_len,
                )

            # Sample non-meta body positions
            sampled = self._sample_body_positions(
                completion_mask, meta_mask, v2.rod_v2_n_sampled, v2.rod_v2_min_pos, self._rod_rng,
            )

            # Soft BCE
            soft_bce, sampling_stats = self._soft_bce_at_positions(
                student_logits, T_emit_logits, self._meta_start_id, sampled,
            )

            # Content KL on student-emitted META region (ALT-2 post-codex consensus).
            # Cold start emit rates 87-97% → dense supervision from step 0, no
            # counterfactual injection needed; avoids BCE/KL tug-of-war.
            content_kl, n_kl = self._content_kl_at_emitted_meta(
                student_logits, T_emit_logits, meta_mask, completion_mask,
            )

            # Update target_rate EMA from teacher emit probs
            avg_p_T = sampling_stats.get("rod_v2/avg_p_T", 0.0)
            self._target_rate_ema = (
                v2.rod_v2_target_rate_alpha * self._target_rate_ema
                + (1.0 - v2.rod_v2_target_rate_alpha) * avg_p_T
            )
            sampling_stats["rod_v2/target_rate_ema"] = self._target_rate_ema

            # Actual emit rate (per-rollout average of student p(meta) at sampled positions)
            actual_rate = sampling_stats.get("rod_v2/avg_p_S", 0.0)
            rate_penalty = torch.tensor(
                (self._target_rate_ema - actual_rate) ** 2,
                device=student_logits.device, dtype=student_logits.dtype,
            )
            sampling_stats["rod_v2/actual_rate"] = actual_rate

        # PPO loss (standard)
        log_ratio = per_token_logps - old_per_token_logps
        log_ratio = torch.clamp(log_ratio, -cfg.log_ratio_clamp, cfg.log_ratio_clamp)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        denom = completion_mask.sum().clamp(min=1.0)
        ppo_loss = (per_token_loss * completion_mask).sum() / denom

        # Combined
        rod_term = (v2.rod_alpha_emit * soft_bce
                    + v2.rod_beta_content * content_kl
                    + v2.rod_gamma_rate * rate_penalty)
        loss = (ppo_loss + rod_term) / max(self.current_gradient_accumulation_steps, 1)

        # Logging
        mode = "train" if self.model.training else "eval"
        m = self._metrics[mode]
        m["rod_v2/soft_bce"].append(float(soft_bce.detach().item()))
        m["rod_v2/content_kl"].append(float(content_kl.detach().item()))
        m["rod_v2/rate_penalty"].append(float(rate_penalty.detach().item()))
        m["rod_v2/ppo_loss"].append(float(ppo_loss.detach().item()))
        m["rod_v2/total_loss"].append(float(loss.detach().item()))
        m["rod_v2/current_topk"].append(float(self._current_topk))
        for k, v in sampling_stats.items():
            m[k].append(v)
        return loss


__all__ = ["MetaRODv2Config", "MetaRODv2Trainer"]
