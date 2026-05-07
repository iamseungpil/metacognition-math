"""Meta-OPD-Decoy trainer (M5.2 from plan_meta_opd_2026_05_03 v5 §10.5).

Extends MetaRLSDTrainer with full-logit KL distillation on the meta region
(replacing scalar Δt = log P_T - log P_S), plus decoy contrast.

Plan §10.5 M5.2 specs:
- Loss = PPO_loss + α × (λ_pos × KL(T+ || S) − λ_neg × KL(T- || S))
- KL on top-K subset (K=64 default; K=32 fallback on OOM)
- Mask to meta region only via meta_mask
- Cold start: M5.1 R5 step 300 best ckpt

Differences vs paper RLSD (≥5 components):
1. Distill signal: scalar Δt → full-logit KL on top-K
2. Region: response-wide → meta only
3. Contrastive: none → T+/T- KL difference
4. Conditioning: gold → V0_prefix + gold + forced <|meta|> (inherited from R5)
5. Cold start: base SFT → R5 step 300 best ckpt
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.training._decoy_utils import _rule_based_decoy
from src.training.meta_rlsd_trainer import MetaRLSDConfig, MetaRLSDTrainer

logger = logging.getLogger(__name__)


@dataclass
class MetaOPDConfig(MetaRLSDConfig):
    """OPD-Decoy hyperparameters (extends MetaRLSDConfig).

    Defaults follow plan §10.5 M5.2.
    """

    # --- OPD-specific knobs ---
    opd_alpha: float = 1.0          # overall OPD loss weight (added to PPO loss)
    opd_lambda_pos: float = 0.5     # KL(T+ || S) weight
    opd_lambda_neg: float = 0.3     # KL(T- || S) weight (decoy contrast)
    opd_topk: int = 64              # top-K subset size for KL computation
    opd_topk_fallback: int = 32     # fallback K on OOM
    opd_temperature: float = 1.0    # softmax temperature for KL
    opd_meta_only: bool = True      # apply OPD only on meta region
    opd_decoy_seed: int = 42        # decoy generation seed (matches MetaRLSD parent)
    opd_kl_neg_cap: float = 5.0     # clamp KL(T-||S) max (review W4: avoid unbounded push)
    opd_forced_meta_assert: bool = True  # round 3 / I1: assert prompt_ids ends with META_START
    opd_forced_meta_assert_freq: int = 100  # check every N steps (perf-friendly)

    # --- Mode flag ---
    variant: str = "m5_2_opd_decoy"


class MetaOPDTrainer(MetaRLSDTrainer):
    """OPD-Decoy: full-logit KL on top-K subset, decoy contrast, meta region only.

    Inherits MetaRLSDTrainer for: data pipeline, teacher building (T+), meta_mask
    construction, PPO clipping. Overrides _compute_loss to replace scalar Δt with
    full-distribution KL on top-K teacher-supported tokens, and adds T- decoy.
    """

    def __init__(self, *args, opd_cfg: Optional[MetaOPDConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if opd_cfg is None:
            opd_cfg = MetaOPDConfig(
                student_init=self.meta_rlsd_cfg.student_init,
                teacher_init=self.meta_rlsd_cfg.teacher_init,
            )
        self.opd_cfg = opd_cfg
        self._current_topk = opd_cfg.opd_topk
        self._step_counter = 0  # for periodic forced-injection assertion (round 3 I1)
        # Cache META_START_ID for forced-injection assert
        try:
            from src.metacot.prompt import META_START
            self._meta_start_id = self._meta_tokenizer.convert_tokens_to_ids(META_START)
        except Exception as e:
            logger.warning("[MetaOPD] could not resolve META_START_ID: %s", e)
            self._meta_start_id = None
        # Review W3: enforce left-padding for teacher input building
        if hasattr(self._meta_tokenizer, "padding_side") and self._meta_tokenizer.padding_side != "left":
            logger.warning(
                "[MetaOPD] tokenizer.padding_side=%s — forcing 'left' to prevent PAD-in-context bug",
                self._meta_tokenizer.padding_side,
            )
            self._meta_tokenizer.padding_side = "left"
        # Review W7: cache logits_to_keep support detection
        import inspect
        try:
            self._supports_logits_to_keep = (
                "logits_to_keep" in inspect.signature(self.model.forward).parameters
            )
        except (TypeError, ValueError):
            self._supports_logits_to_keep = True  # assume yes (parent's pattern)
        logger.info(
            "[MetaOPD] init opd_alpha=%.2f λ_pos=%.2f λ_neg=%.2f K=%d meta_only=%s "
            "META_START_ID=%s supports_logits_to_keep=%s",
            opd_cfg.opd_alpha,
            opd_cfg.opd_lambda_pos,
            opd_cfg.opd_lambda_neg,
            opd_cfg.opd_topk,
            opd_cfg.opd_meta_only,
            self._meta_start_id,
            self._supports_logits_to_keep,
        )

    # ─── Helper: top-K KL on a (T_logits, S_logits) pair ─────────────────────

    @staticmethod
    def _topk_kl(
        teacher_logits: torch.Tensor,  # [B, T, V] full logits
        student_logits: torch.Tensor,  # [B, T, V] full logits
        mask: torch.Tensor,            # [B, T] float — 1.0 where to include
        K: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """KL(T || S) on top-K tokens of T's distribution per position.

        Returns scalar — masked-mean over positions where ``mask > 0``. If mask
        is all zeros, returns 0.0 connected to student_logits graph (so backward
        still propagates a no-op gradient through this layer).

        References:
        - Top-K teacher support matching: Revisiting OPD §F3 (2603.25562)
        - Local-support KL (vs full-vocab): Rethinking OPD §3 (2604.13016)

        Note (review C5): K is clamped to vocab size to prevent topk RuntimeError.
        Note (review W5): For T ≠ 1 the K-element support is unchanged
            (topk monotone under temperature scaling) but renormalization
            differs from full-vocab marginal — this is intentional local-support KL.
        """
        V = teacher_logits.size(-1)
        K = max(1, min(int(K), V))

        T_topk_logits, topk_idx = teacher_logits.topk(K, dim=-1)
        S_topk_logits = student_logits.gather(-1, topk_idx)

        # Apply temperature; log-softmax / softmax on the local top-K subset.
        T_log_probs = F.log_softmax(T_topk_logits / temperature, dim=-1)
        T_probs = T_log_probs.exp()
        S_log_probs = F.log_softmax(S_topk_logits / temperature, dim=-1)

        # KL(T || S) per position = Σ_k T(k) × (log T(k) − log S(k))
        kl_per_token = (T_probs * (T_log_probs - S_log_probs)).sum(dim=-1)  # [B, T]

        mask = mask.to(kl_per_token.dtype)
        masked_kl = kl_per_token * mask
        denom = mask.sum().clamp(min=1.0)
        return masked_kl.sum() / denom

    # ─── Decoy teacher input builder (T-) ────────────────────────────────────

    def _build_teacher_inputs_decoy(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        ground_truth: Sequence[str],
    ):
        """Build T- teacher inputs: same as parent T+ but with decoy answer instead of gold.

        Mirrors parent ``_build_teacher_inputs`` but replaces ``gold`` with
        ``_rule_based_decoy(gold, seed)`` per batch element.
        """
        cfg = self.meta_rlsd_cfg
        opd = self.opd_cfg
        tokenizer = self._meta_tokenizer

        if not cfg.privileged_answer:
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_len": prompt_ids.size(1),
            }

        teacher_texts: List[str] = []
        for i in range(prompt_ids.size(0)):
            nonpad = prompt_mask[i].bool()
            decoded = tokenizer.decode(prompt_ids[i][nonpad], skip_special_tokens=False)
            gold = str(ground_truth[i]) if i < len(ground_truth) else ""
            decoy = _rule_based_decoy(gold, seed=opd.opd_decoy_seed + i)
            teacher_texts.append(f"{decoded} Answer: {decoy}")

        enc = tokenizer(
            teacher_texts,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=cfg.prompt_length,
            return_tensors="pt",
        )
        teacher_prompt_ids = enc["input_ids"].to(prompt_ids.device)
        teacher_prompt_mask = enc["attention_mask"].to(prompt_ids.device)

        input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_len": teacher_prompt_ids.size(1),
        }

    # ─── Forward: get logits aligned to completion tokens (off-by-one safe) ──

    def _completion_logits(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_len: int,
    ) -> torch.Tensor:
        """Return logits[:, prompt_len-1 : prompt_len-1+completion_len, :].

        These are the *predictions* for completion tokens (standard HF off-by-one
        convention). Pattern matches parent class _get_per_token_logps_and_entropies
        shim (line 418-421): call with logits_to_keep+1, slice [:, :-1, :].

        Review W7: gracefully degrade if model.forward doesn't support
        logits_to_keep (e.g., older models or non-CausalLM heads).
        """
        if getattr(self, "_supports_logits_to_keep", True):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=completion_len + 1,
            )
            return out.logits[:, :-1, :]  # [B, completion_len, V]
        # Fallback: full forward, manual slice
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # logits[:, prompt_len-1 : prompt_len-1+completion_len, :]
        prompt_len = input_ids.size(1) - completion_len
        return out.logits[:, prompt_len - 1 : prompt_len - 1 + completion_len, :]

    # ─── Override: _compute_loss with OPD term ───────────────────────────────

    def _compute_loss(self, model, inputs):
        cfg = self.meta_rlsd_cfg
        opd = self.opd_cfg

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        # ── Forced-meta injection assertion (round 3 I1) ────────────────────
        # Verify dataset preprocessing has appended <|meta|> token to prompts.
        # Periodic check (every N steps) to avoid per-batch overhead.
        self._step_counter += 1
        if (
            opd.opd_forced_meta_assert
            and self._meta_start_id is not None
            and prompt_ids.size(0) > 0
            and self._step_counter % opd.opd_forced_meta_assert_freq == 1
        ):
            # Find the last non-pad token of each prompt (left-pad assumed).
            with torch.no_grad():
                last_real = prompt_mask.sum(dim=1) - 1  # idx of last real token
                last_tokens = prompt_ids[
                    torch.arange(prompt_ids.size(0), device=prompt_ids.device),
                    last_real.clamp(min=0),
                ]
                forced_count = (last_tokens == self._meta_start_id).sum().item()
                forced_rate = forced_count / max(prompt_ids.size(0), 1)
            self._metrics["train"]["opd/forced_inject_rate"].append(float(forced_rate))
            if forced_rate < 0.5:
                logger.warning(
                    "[MetaOPD] step=%d forced-meta injection rate %.1f%% < 50%% — "
                    "dataset preprocessing may be missing META_START suffix",
                    self._step_counter, 100.0 * forced_rate,
                )

        # ── Empty/degenerate batch guard ────────────────────────────────────
        # Review W1: connect to ONE param, not all (avoid ZeRO-3 all-gather cliff).
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

        # Review C3: temperature scaling on student logits — must match parent's
        # _get_per_token_logps_and_entropies shim (line 423: logits / self.temperature)
        # so PPO ratio is consistent with vLLM-populated old_per_token_logps.
        student_temp = max(getattr(self, "temperature", 1.0), 1e-6)
        student_logits_scaled = student_logits / student_temp

        # Review C1: avoid materializing full-vocab log_softmax (~80GB at B=64,T=4096,V=152k bf16).
        # Use gather + logsumexp pattern for per-token logp, O(B*T) memory.
        per_token_logps = (
            student_logits_scaled.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
            - student_logits_scaled.logsumexp(dim=-1)
        )  # [B, T]

        advantages = inputs["advantages"]  # [B]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        # ── Build T+/T- inputs (gold + decoy) ────────────────────────────────
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
        # Review W2: silent decoy degeneration when gt_list is empty.
        # Skip OPD term entirely if gold list missing — log warning, return PPO-only.
        if len(gt_list) != prompt_ids.size(0):
            logger.warning(
                "[MetaOPD] gt_list len=%d != batch=%d → skip OPD term (PPO only)",
                len(gt_list), prompt_ids.size(0),
            )
            opd_disabled = True
        else:
            opd_disabled = False

        if opd_disabled:
            # Skip teacher forwards + KL terms entirely. PPO-only fallback.
            kl_pos = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
            kl_neg = torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
            opd_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        else:
            teacher_pos_pack = self._build_teacher_inputs(
                prompt_ids, prompt_mask, completion_ids, completion_mask, gt_list,
            )
            teacher_neg_pack = self._build_teacher_inputs_decoy(
                prompt_ids, prompt_mask, completion_ids, completion_mask, gt_list,
            )

            # T+ forward (no_grad), aligned slice
            with torch.no_grad():
                T_pos_logits = self._completion_logits(
                    self.teacher,
                    teacher_pos_pack["input_ids"],
                    teacher_pos_pack["attention_mask"],
                    completion_len,
                )

            # T- forward (no_grad), aligned slice
            with torch.no_grad():
                T_neg_logits = self._completion_logits(
                    self.teacher,
                    teacher_neg_pack["input_ids"],
                    teacher_neg_pack["attention_mask"],
                    completion_len,
                )

            # KL terms on top-K (OPD signal)
            if opd.opd_meta_only:
                opd_mask = meta_mask * completion_mask.float()
            else:
                opd_mask = completion_mask.float()

            # Review C4: read self._current_topk (sticky after OOM).
            # Review W6: catch RuntimeError with "out of memory" too.
            try:
                kl_pos = self._topk_kl(
                    T_pos_logits, student_logits, opd_mask, self._current_topk, opd.opd_temperature,
                )
                kl_neg = self._topk_kl(
                    T_neg_logits, student_logits, opd_mask, self._current_topk, opd.opd_temperature,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                logger.warning(
                    "[MetaOPD] OOM at K=%d → sticky fallback K=%d",
                    self._current_topk, opd.opd_topk_fallback,
                )
                self._current_topk = opd.opd_topk_fallback
                kl_pos = self._topk_kl(
                    T_pos_logits, student_logits, opd_mask, self._current_topk, opd.opd_temperature,
                )
                kl_neg = self._topk_kl(
                    T_neg_logits, student_logits, opd_mask, self._current_topk, opd.opd_temperature,
                )

            # Review W4: cap KL_neg to avoid unbounded "push away from T-" gradient
            kl_neg = kl_neg.clamp(max=opd.opd_kl_neg_cap)

        opd_loss = opd.opd_lambda_pos * kl_pos - opd.opd_lambda_neg * kl_neg

        # ── Standard PPO loss (no scalar SDC factor — OPD replaces it) ──────
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
        # Review C2: divide TOTAL loss by gradient_accumulation_steps once,
        # so OPD term and PPO term scale identically. Previously only ppo_loss
        # was divided, which made opd_loss effectively GA× over-weighted.
        loss = (ppo_loss + opd.opd_alpha * opd_loss) / max(self.current_gradient_accumulation_steps, 1)

        # ── Logging ─────────────────────────────────────────────────────────
        mode = "train" if self.model.training else "eval"
        metrics = self._metrics[mode]
        metrics["opd/kl_pos"].append(float(kl_pos.detach().item()))
        metrics["opd/kl_neg"].append(float(kl_neg.detach().item()))
        metrics["opd/opd_loss"].append(float(opd_loss.detach().item()))
        metrics["opd/ppo_loss"].append(float(ppo_loss.detach().item()))
        metrics["opd/total_loss"].append(float(loss.detach().item()))
        metrics["opd/current_topk"].append(float(self._current_topk))
        # H5.3 sanity: meta-region coverage in this batch (review S5: always log)
        metrics["opd/meta_coverage"].append(float(opd_mask.mean().item()))

        return loss


__all__ = ["MetaOPDConfig", "MetaOPDTrainer", "apply_forced_meta_to_dataset"]


# ─── Forced injection — Plan v5.1 §11.1 option 1 (dataset-level) ─────────────

def apply_forced_meta_to_dataset(dataset, tokenizer=None):
    """Append `<|meta|>` to each prompt's final user message.

    Plan v5.1 §11.1 option 1 (dataset-level injection): ensures every student
    rollout begins inside a meta block, matching R5 veRL `forced_meta_agent_loop`
    behavior — but applied at dataset prep time rather than generation time.

    This is the option-1 fix for the gap flagged in Round 3 review I1
    (forced injection not present in TRL pipeline). Applies before
    `tokenizer.apply_chat_template` so the special token survives templating.

    Args:
        dataset: HF Dataset with column `prompt` (list[dict[str,str]]).
        tokenizer: optional — used to verify META_START is a known special token.

    Returns:
        Modified dataset with forced META_START suffix on the last user turn.
    """
    from src.metacot.prompt import META_START

    if tokenizer is not None:
        meta_id = tokenizer.convert_tokens_to_ids(META_START)
        if meta_id == tokenizer.unk_token_id or meta_id is None:
            logger.warning(
                "[forced_meta] META_START='%s' not in tokenizer vocab — "
                "training-inference mismatch likely. Skipping injection.",
                META_START,
            )
            return dataset

    def _inject(example):
        prompt = example.get("prompt", [])
        if not isinstance(prompt, list) or not prompt:
            return example
        last_user_idx = None
        for i in range(len(prompt) - 1, -1, -1):
            if prompt[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return example
        # Append META_START as text — chat template will re-tokenize properly.
        # We don't append a newline before META_START because some chat templates
        # add their own \n separator after the user content.
        original = prompt[last_user_idx].get("content", "")
        if META_START not in original[-len(META_START) - 5 :]:  # idempotent guard
            new_prompt = list(prompt)
            new_prompt[last_user_idx] = dict(prompt[last_user_idx])
            new_prompt[last_user_idx]["content"] = f"{original}\n{META_START}"
            example["prompt"] = new_prompt
        return example

    n_before = len(dataset)
    dataset = dataset.map(_inject)
    logger.info(
        "[forced_meta] applied META_START suffix to %d examples (option 1 dataset-level)",
        n_before,
    )
    return dataset
