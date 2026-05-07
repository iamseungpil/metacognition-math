"""Meta-ROD-PT trainer (Plan v5.7).

ROD-PT = R5 RLSD framework with decoy replaced by position teacher.

Design (post-codex Round 2 consensus, user-corrected forced-META semantic):
  - Student: natural emit (no forced injection at generation)
  - T_content: gold + V0 + completion (with student's emitted META)
                → R5 SDC factor on student's actual META block tokens
  - T_position: gold + V0 + completion[:meta_start_pos]
                → top-K check at META_START emit position
                → advantage shift if META_START not in top_K

Differences vs M5.2 OPD-Decoy:
  - No decoy teacher T-
  - Position teacher T_position replaces T-
  - Reward type: scalar SDC factor (R5 reward amplify) instead of top-K KL contrast
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.training.meta_opd_trainer import MetaOPDConfig, MetaOPDTrainer

logger = logging.getLogger(__name__)


@dataclass
class MetaRODPTConfig(MetaOPDConfig):
    """ROD-PT hyperparameters."""

    # Position teacher
    pt_position_top_k: int = 16
    pt_position_penalty: float = -1.0   # rollout-level advantage shift
    pt_factor_clip_low: float = 0.2
    pt_factor_clip_high: float = 5.0
    pt_disable_decoy: bool = True       # no T- forward

    # Disable parent OPD aux KL terms (we use R5 SDC factor)
    opd_alpha: float = 0.0
    opd_lambda_pos: float = 0.0
    opd_lambda_neg: float = 0.0
    opd_forced_meta_assert: bool = False

    variant: str = "rod_pt"


class MetaRODPTTrainer(MetaOPDTrainer):
    """ROD-PT: R5 SDC factor on META content + position penalty (no decoy)."""

    def __init__(self, *args, rod_pt_cfg: Optional[MetaRODPTConfig] = None, **kwargs):
        if rod_pt_cfg is not None and "opd_cfg" not in kwargs:
            kwargs["opd_cfg"] = rod_pt_cfg
        super().__init__(*args, **kwargs)
        self.rod_pt_cfg = rod_pt_cfg if rod_pt_cfg is not None else self.opd_cfg
        if not isinstance(self.rod_pt_cfg, MetaRODPTConfig):
            self.rod_pt_cfg = MetaRODPTConfig(
                student_init=self.opd_cfg.student_init,
                teacher_init=self.opd_cfg.teacher_init,
            )
        logger.info(
            "[ROD-PT] init top_K=%d position_penalty=%.2f clip=[%.2f, %.2f]",
            self.rod_pt_cfg.pt_position_top_k,
            self.rod_pt_cfg.pt_position_penalty,
            self.rod_pt_cfg.pt_factor_clip_low,
            self.rod_pt_cfg.pt_factor_clip_high,
        )

    @staticmethod
    def _compute_sdc_factor(
        teacher_logits: torch.Tensor,    # [B, T, V]
        student_logits: torch.Tensor,    # [B, T, V]
        token_ids: torch.Tensor,         # [B, T] — actual completion tokens
        sign_advantage: torch.Tensor,    # [B] — sign(advantage)
        clip_low: float,
        clip_high: float,
    ) -> torch.Tensor:
        """R5 RLSD SDC factor at each token: clip(exp(sign(A) * (logp_T - logp_S))).

        Returns [B, T] factor tensor (DETACHED — used as a reward/weight, not a
        gradient-bearing term). Codex Round 1 fix: detach to prevent the factor
        backpropagating into the policy through logp_S; PPO ratio is the only
        gradient-bearing quantity in RLSD.
        """
        with torch.no_grad():
            # Device fix: teacher_logits may live on CPU (frozen teacher).
            # Align all tensors to student_logits' device before gather.
            target_device = student_logits.device
            T_logits = teacher_logits.detach()
            if T_logits.device != target_device:
                T_logits = T_logits.to(target_device)
            S_logits = student_logits.detach()
            tok = token_ids
            if tok.device != target_device:
                tok = tok.to(target_device)
            sign_A = sign_advantage
            if sign_A.device != target_device:
                sign_A = sign_A.to(target_device)

            logp_T = (
                T_logits.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
                - T_logits.logsumexp(dim=-1)
            )
            logp_S = (
                S_logits.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
                - S_logits.logsumexp(dim=-1)
            )
            sign_A_per_token = sign_A.unsqueeze(1).expand_as(logp_T)
            log_factor = sign_A_per_token * (logp_T - logp_S)
            factor = torch.exp(log_factor).clamp(min=clip_low, max=clip_high)
        return factor

    def _compute_position_penalty(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        meta_mask: torch.Tensor,         # [B, T_completion] — 1.0 inside emitted meta
        gt_list: List[str],
    ) -> Tuple[torch.Tensor, dict]:
        """Per-rollout position penalty.

        For each rollout where student emitted META:
          1. Find first META_START position p (first 1 in meta_mask along T axis).
          2. Build T_position input = prompt + V0 + gold + completion[:p].
          3. Forward teacher, get last-position next-token logits.
          4. Check META_START_ID in top_K.
          5. If NOT in top_K → penalty (rollout-level).

        Returns (penalty_per_rollout [B], stats).
        """
        cfg = self.rod_pt_cfg
        B = prompt_ids.size(0)
        device = prompt_ids.device

        penalty = torch.zeros(B, device=device, dtype=torch.float32)
        n_rollouts_with_meta = 0
        n_penalized = 0
        n_top_k_hit = 0

        if self._meta_start_id is None:
            return penalty, {
                "rod_pt/n_rollouts_with_meta": 0,
                "rod_pt/penalty_rate": 0.0,
            }

        # Find ACTUAL META_START_ID position per rollout (Codex Round 1 fix:
        # don't trust meta_mask first-nonzero; meta_mask may include tags+contents).
        meta_first_pos: List[Optional[int]] = []
        for b in range(B):
            valid = (completion_ids[b] == self._meta_start_id) & completion_mask[b].bool()
            nz = valid.nonzero(as_tuple=True)[0]
            if nz.numel() > 0:
                meta_first_pos.append(int(nz[0].item()))
            else:
                meta_first_pos.append(None)

        # Build T_position inputs for rollouts with META emit
        # Codex Round 1 fix: p==0 is also valid (empty truncated completion;
        # teacher predicts from gold+V0 only).
        teacher_text_inputs: List[str] = []
        rollout_indices: List[int] = []
        tokenizer = self._meta_tokenizer
        cfg_meta_rlsd = self.meta_rlsd_cfg
        for b, p in enumerate(meta_first_pos):
            if p is None:
                continue
            nonpad_prompt = prompt_mask[b].bool()
            prompt_text = tokenizer.decode(prompt_ids[b][nonpad_prompt], skip_special_tokens=False)
            if p == 0:
                comp_text = ""
            else:
                comp_truncated_ids = completion_ids[b, :p]
                comp_mask_truncated = completion_mask[b, :p].bool()
                comp_truncated_ids = comp_truncated_ids[comp_mask_truncated]
                comp_text = tokenizer.decode(comp_truncated_ids, skip_special_tokens=False)
            gold = str(gt_list[b]) if b < len(gt_list) else ""
            full_text = f"{prompt_text} Answer: {gold}{comp_text}"
            teacher_text_inputs.append(full_text)
            rollout_indices.append(b)
            n_rollouts_with_meta += 1

        if not teacher_text_inputs:
            return penalty, {
                "rod_pt/n_rollouts_with_meta": 0,
                "rod_pt/penalty_rate": 0.0,
            }

        # Tokenize and forward teacher
        enc = tokenizer(
            teacher_text_inputs,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=cfg_meta_rlsd.prompt_length + completion_ids.size(1),
            return_tensors="pt",
        )
        teacher_input_ids = enc["input_ids"].to(device)
        teacher_attn = enc["attention_mask"].to(device)

        with torch.no_grad():
            try:
                model_device = next(self.teacher.parameters()).device
            except StopIteration:
                model_device = device
            T_inp = teacher_input_ids.to(model_device)
            T_attn = teacher_attn.to(model_device)
            T_out = self.teacher(input_ids=T_inp, attention_mask=T_attn)
            T_logits_full = T_out.logits.to(device)  # [N, max_len, V]

        # Extract last non-pad position logits (next-token prediction at META_START position)
        K = cfg.pt_position_top_k
        for i, b_orig in enumerate(rollout_indices):
            last_real_idx = int(teacher_attn[i].sum().item()) - 1
            if last_real_idx < 0:
                continue
            last_logits = T_logits_full[i, last_real_idx]  # [V]
            top_K_idx = last_logits.topk(K, dim=-1).indices  # [K]
            in_top_K = (top_K_idx == self._meta_start_id).any().item()
            if in_top_K:
                n_top_k_hit += 1
            else:
                penalty[b_orig] = cfg.pt_position_penalty
                n_penalized += 1

        stats = {
            "rod_pt/n_rollouts_with_meta": float(n_rollouts_with_meta),
            "rod_pt/n_penalized": float(n_penalized),
            "rod_pt/n_top_k_hit": float(n_top_k_hit),
            "rod_pt/penalty_rate": float(n_penalized / max(n_rollouts_with_meta, 1)),
        }
        return penalty, stats

    def _compute_loss(self, model, inputs):
        cfg = self.meta_rlsd_cfg
        pt = self.rod_pt_cfg

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

        # Student forward (with grad)
        student_logits = self._completion_logits(model, input_ids, attention_mask, completion_len)
        student_temp = max(getattr(self, "temperature", 1.0), 1e-6)
        student_logits_scaled = student_logits / student_temp
        per_token_logps = (
            student_logits_scaled.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
            - student_logits_scaled.logsumexp(dim=-1)
        )

        advantages = inputs["advantages"]   # [B]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        # Get gold list
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

        sdc_factor = torch.ones_like(per_token_logps)
        position_penalty = torch.zeros_like(advantages)
        position_stats: dict = {}

        if len(gt_list) == prompt_ids.size(0) and self._meta_start_id is not None:
            # T_content forward — student's actual completion + gold conditioning
            teacher_pack = self._build_teacher_inputs(
                prompt_ids, prompt_mask, completion_ids, completion_mask, gt_list,
            )
            with torch.no_grad():
                T_content_logits = self._completion_logits(
                    self.teacher,
                    teacher_pack["input_ids"],
                    teacher_pack["attention_mask"],
                    completion_len,
                )

            # SDC factor on completion tokens (R5 form)
            sign_A = torch.sign(advantages).clamp(min=-1.0, max=1.0)  # [B]
            sdc_factor_full = self._compute_sdc_factor(
                T_content_logits, student_logits, completion_ids, sign_A,
                pt.pt_factor_clip_low, pt.pt_factor_clip_high,
            )
            # Apply only on meta region (META content tokens). Outside meta = factor 1.0.
            sdc_factor = sdc_factor_full * meta_mask + 1.0 * (1.0 - meta_mask)

            # Position penalty (T_position forward)
            position_penalty, position_stats = self._compute_position_penalty(
                prompt_ids, prompt_mask, completion_ids, completion_mask, meta_mask, gt_list,
            )

        # Modified advantages with position penalty (rollout-level)
        modified_advantages = advantages + position_penalty

        # Standard PPO with SDC factor multiply per-token
        log_ratio = per_token_logps - old_per_token_logps
        log_ratio = torch.clamp(log_ratio, -cfg.log_ratio_clamp, cfg.log_ratio_clamp)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high)
        per_token_loss1 = coef_1 * modified_advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * modified_advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # SDC factor amplifies per-token loss on meta region
        per_token_loss = per_token_loss * sdc_factor

        denom = completion_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * completion_mask).sum() / denom
        loss = loss / max(self.current_gradient_accumulation_steps, 1)

        # Logging
        mode = "train" if self.model.training else "eval"
        m = self._metrics[mode]
        m["rod_pt/sdc_factor_mean"].append(float(sdc_factor.mean().item()))
        m["rod_pt/sdc_factor_max"].append(float(sdc_factor.max().item()))
        m["rod_pt/position_penalty_total"].append(float(position_penalty.sum().item()))
        m["rod_pt/total_loss"].append(float(loss.detach().item()))
        m["rod_pt/meta_coverage"].append(float(meta_mask.mean().item()))
        for k, v in position_stats.items():
            m[k].append(v)
        return loss


__all__ = ["MetaRODPTConfig", "MetaRODPTTrainer"]
