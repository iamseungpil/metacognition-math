"""Meta-RLSD trainer — canonical RLSD with meta-only mask.

Implements the approved plan at::

    results/plan_meta_rlsd_v2_2026_04_17.md

All section references (``§2.1``, ``§5.1``, …) point to that document unless
otherwise noted.

Design (modular, plug-and-play):
    * Subclasses ``trl.GRPOTrainer`` — new surface only (3 overrides + helpers).
    * No existing file is modified. Configuration is a sibling dataclass
      :class:`MetaRLSDConfig` loadable from YAML.
    * Variant dispatch at CLI time flips dataclass fields — zero code changes
      to add another variant.
    * Pre-flight validation and meta-mask construction live in the companion
      module ``meta_rlsd_data_pipeline`` to keep this file focused on RL logic.

Usage::

    accelerate launch --config_file configs/zero3_no_offload.yaml \\
        src/training/meta_rlsd_trainer.py \\
        --config configs/meta_rlsd_m1.yaml \\
        --variant m1 \\
        --seed 42
"""
from __future__ import annotations

# ─── Critical force-imports FIRST, before any trl/accelerate pulls ─────────
# deepspeed 0.15.x does not auto-expose `runtime` as an attribute of the
# `deepspeed` module. accelerate/transformers access `deepspeed.runtime.X`
# via attribute lookup and fail with AttributeError unless we explicitly
# import the submodules here — BEFORE trl/accelerate import deepspeed.
try:
    import deepspeed  # noqa: F401
    import deepspeed.runtime  # noqa: F401
    import deepspeed.runtime.config  # noqa: F401
    import deepspeed.runtime.lr_schedules  # noqa: F401
    import deepspeed.runtime.zero.partition_parameters  # noqa: F401
except ImportError:
    pass  # deepspeed optional; needed only when accelerate config uses ZeRO

# ─── Top-of-file stubs (mirrors grpo_v2.py) ────────────────────────────────
# TRL imports several optional deps at module import time. Reuse the stubs
# from grpo_v2 so we never duplicate stub logic — keeps behavior consistent.
#
# When vLLM IS installed, import it first so `_ensure_vllm_stub` detects it in
# sys.modules and skips the dummy registration. This enables use_vllm=True path.
try:
    import vllm  # noqa: F401  — real vLLM installed, use it
except Exception:
    pass  # vLLM not installed, grpo_v2 will register dummy stubs

from src.training.grpo_v2 import (  # noqa: F401  — side effects register stubs
    _ensure_vllm_stub,
    _ensure_mergekit_stub,
    _ensure_llm_blender_stub,
)

_ensure_vllm_stub()
_ensure_mergekit_stub()
_ensure_llm_blender_stub()

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from src.metacot.prompt import META_END, META_START, parse_meta_blocks
from src.training.meta_rlsd_data_pipeline import (
    PFReport,
    _build_meta_mask,
    load_meta_rlsd_dataset,
    preflight_checks,
)
from src.training.rewards import (
    _check_correctness,  # reused for composability; see §2.5
    _get_text,
    correctness_reward,  # imported to keep the base signal consistent
)
from src.training.tokenizer_utils import ensure_meta_tokens_not_special


# ─── Config — §2.7 mirror ──────────────────────────────────────────────────

@dataclass
class MetaRLSDConfig:
    """Hyperparameters for Meta-RLSD — identical field order to plan §2.7.

    Load from YAML via :meth:`from_yaml`. Unknown YAML keys raise on purpose
    so that config drift from the plan is caught at startup, not at step 300.
    """

    student_init: str
    teacher_init: str
    privileged_answer: bool = True
    mask_mode: str = "meta_only"  # or "all_tokens" for A2a / A2b
    clip_eps_w: float = 0.2
    clip_eps_low: float = 0.2
    clip_eps_high: float = 0.28
    lambda_init: float = 0.5
    lambda_final: float = 0.0
    lambda_decay_steps: int = 75
    teacher_sync_freq: int = 10
    warmup_steps: int = 10
    lr: float = 1.0e-6
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    num_rollouts: int = 8
    temperature: float = 1.0
    # Generation filter params — must match SFT generation_config.json to avoid
    # full-entropy divergence from SFT's narrow distribution (observed iter 8
    # symptom: clipped_ratio=1.0, wrap_rate=0, reward_std=0). Defaults are
    # open (top_p=1.0, top_k=0) to preserve backward compatibility for M1 runs
    # that intentionally want wide sampling; YAML must override for SFT-narrow.
    top_p: float = 1.0
    top_k: int = 0
    max_response_length: int = 4096
    prompt_length: int = 2048
    batch_size: int = 64
    total_steps: int = 300
    seed: int = 42
    train_data: str = "data/verl_train_redirect.parquet"
    val_data: str = ""
    output_dir: str = "checkpoints/meta_rlsd_m1"
    # Stronger default (-0.30, "soft") prevents E21R-style wrap-rate collapse
    # where the model drops meta tokens as overhead. Override to -0.35 ("hard")
    # for extra defense or -0.15 (legacy) via YAML when running weak-penalty ablations.
    reward_meta_no_penalty: float = -0.30  # per-completion penalty when no <|meta|> block present
    reward_meta_no_penalty_strength: str = "soft"  # "off" (0), "soft" (uses no_penalty), "hard" (2× no_penalty)
    # Rationale (E21R post-mortem §3.3 of consolidated report):
    #   E21R step 300 lost 12% of <|meta|> wrapping because its reward was neutral
    #   on wrap-removal. Meta-RLSD strengthens: default -0.30 soft, tunable to
    #   -0.60 hard for stronger defense against wrap-rate collapse.
    reward_meta_short_bonus: float = 0.0
    reward_meta_full_bonus: float = 0.20
    meta_min_length_tokens: int = 20
    log_grad_norm_clip: float = 1.0
    log_ratio_clamp: float = 10.0
    eval_interval: int = 50
    save_interval: int = 100

    # Optional extensions — off by default, enable via YAML.
    variant: str = "m1"
    report_to: str = "wandb"
    run_name: Optional[str] = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    correctness_weight: float = 1.0
    meta_floor_weight: float = 0.2
    # If True, fall back to file-based teacher sync if GatheredParameters OOMs.
    teacher_sync_file_fallback: bool = True
    teacher_sync_tmp_dir: str = "/tmp/_meta_rlsd_teacher_snap"

    # vLLM backend for rollout generation (OOM fix for long Qwen3 completions).
    # Without vLLM, TRL GRPOTrainer uses HF generate which OOMs at 4096+ tokens
    # on Qwen3-8B due to full-vocab logits (152K × batch × length in fp32).
    use_vllm: bool = False
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.5

    @classmethod
    def from_yaml(cls, path: str) -> "MetaRLSDConfig":
        """Load config, flattening nested ``reward:`` sub-dicts and warning on unknown keys.

        Accepts both the flat schema (``reward_meta_no_penalty: -0.30``) and the
        plan-style nested schema (``reward: {meta_floor_no_meta_penalty: -0.30}``).
        Unknown keys are warned-on but do not crash — drift from the plan surfaces
        in logs instead of at step 300.
        """
        with open(path, "r") as f:
            payload = yaml.safe_load(f) or {}

        # Flatten ``reward:`` sub-dict if present (plan §2.7 schema)
        reward_subdict = payload.pop("reward", None) or {}
        # Map nested plan keys to flat dataclass fields.
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
                print(f"[WARN] MetaRLSDConfig ignoring unknown reward key: {k!r}")

        # Filter unknown top-level keys with warning (do not crash)
        unknown = set(payload) - known
        if unknown:
            print(
                f"[WARN] MetaRLSDConfig ignoring unknown keys in {path}: "
                f"{sorted(unknown)}"
            )
            for k in unknown:
                payload.pop(k, None)

        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Reward composition — §2.5 ─────────────────────────────────────────────

def _count_meta_tokens(tokenizer, block_text: str) -> int:
    try:
        return len(tokenizer(block_text, add_special_tokens=False)["input_ids"])
    except Exception:
        return len(tokenizer.encode(block_text, add_special_tokens=False))


def _compute_meta_floor(
    completion_text: str,
    tokenizer,
    *,
    no_penalty: float,
    short_bonus: float,
    full_bonus: float,
    min_len_tokens: int,
    strength: str = "soft",
) -> float:
    """3-level meta floor — token-length gated (§2.5).

    * no meta block           → ``no_penalty`` (negative; **strengthened** by ``strength``)
    * meta present, short     → ``short_bonus`` (0 by default)
    * meta present, long      → ``full_bonus``

    ``strength`` scales the *no-meta* penalty only. E21R (RL step 300) collapsed
    because its meta-floor was effectively neutral — student learned to drop
    ``<|meta|>`` wrapping without reward cost. Meta-RLSD strengthens this:

    * ``"off"``  → 0.0 (disable penalty — NOT recommended)
    * ``"soft"`` → uses ``no_penalty`` as-is (default: -0.30)
    * ``"hard"`` → 2× penalty (e.g., -0.60)

    Short (< min_len_tokens) and full bonuses are unaffected by strength.
    """
    blocks = parse_meta_blocks(completion_text, allow_free_text_fallback=False)
    num_blocks = blocks.get("num_blocks", 0) if isinstance(blocks, dict) else len(blocks)

    if num_blocks == 0:
        if strength == "off":
            return 0.0
        elif strength == "hard":
            return float(no_penalty) * 2.0
        else:  # "soft" or any unrecognized value
            return float(no_penalty)

    # Reconstruct block text from <|meta|>…<|/meta|> regex — parse_meta_blocks
    # returns only metadata; we need raw block content to re-tokenize.
    pattern = re.compile(
        rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
        re.DOTALL | re.IGNORECASE,
    )
    max_tokens = 0
    for match in pattern.finditer(completion_text):
        block_text = match.group(1)
        tok_len = _count_meta_tokens(tokenizer, block_text)
        if tok_len > max_tokens:
            max_tokens = tok_len

    if max_tokens >= min_len_tokens:
        return float(full_bonus)
    return float(short_bonus)


def correctness_plus_meta_floor_reward(
    completions,
    ground_truth=None,
    *,
    tokenizer,
    cfg: MetaRLSDConfig,
    correctness_weight: float = 1.0,
    meta_floor_weight: float = 0.2,
    continuous_weight: float = 0.05,
    **_kwargs,
) -> List[float]:
    """Reward stack with continuous variance floor — §2.5 (revised).

        R = correctness_weight · r_correct
          + meta_floor_weight · r_meta_floor
          + continuous_weight · r_continuous

    Where ``r_continuous`` is a small continuous term in [0, 1] derived from
    three weakly-correlated continuous signals to prevent GRPO group-level
    variance collapse when all rollouts share the same correctness and floor
    outcomes (observed iter 8 deadlock: reward_std = 0 → advantage = 0 →
    grad_norm = 0 on narrow SFT × coarse discrete reward).

    Continuous signals (sum / 3):
        1. meta_count_bonus: ``min(n_meta_blocks, 3) / 3`` — rewards multiple
           thoughtful meta blocks without requiring correctness.
        2. meta_length_frac: average meta-block token length / 200, clipped to
           [0, 1] — rewards longer, more substantive meta reasoning.
        3. boxed_plus_wrap: 0.5 · ``has_boxed`` + 0.5 · ``has_wrap`` —
           format compliance in [0, 1].

    These are intentionally small (weight 0.05 vs correctness 1.0) so
    correctness remains the dominant signal. Paper claim is about method
    (region-specialized contrastive), not reward design — continuous floor is
    a fidelity fix for RL variance, not a reward engineering contribution.
    """
    import re as _re
    from src.metacot.prompt import META_START as _MS, META_END as _ME
    try:
        from src.metacot.prompt import parse_meta_blocks as _pmb
    except ImportError:
        _pmb = None

    rewards: List[float] = []
    for i, comp in enumerate(completions):
        text = _get_text(comp)
        gt = ground_truth[i] if ground_truth is not None else ""
        r_correct = 1.0 if _check_correctness(text, gt) else 0.0
        r_meta = _compute_meta_floor(
            text,
            tokenizer,
            no_penalty=cfg.reward_meta_no_penalty,
            short_bonus=cfg.reward_meta_short_bonus,
            full_bonus=cfg.reward_meta_full_bonus,
            min_len_tokens=cfg.meta_min_length_tokens,
            strength=cfg.reward_meta_no_penalty_strength,
        )

        # ── Continuous variance-floor terms (bounded [0, 1]) ──────────
        has_wrap = float(_MS in text and _ME in text)
        has_boxed = float(bool(_re.search(r"\\boxed\{", text)))
        format_component = 0.5 * has_boxed + 0.5 * has_wrap  # [0, 1]

        if _pmb is not None:
            try:
                blocks = _pmb(text, allow_free_text_fallback=False)
            except TypeError:
                blocks = _pmb(text)
            n_blocks = len(blocks) if blocks else 0
            if blocks:
                # Average block length in tokens (approximate via len(tokenizer(...)))
                try:
                    lengths = [
                        len(tokenizer(b.get("text", "") if isinstance(b, dict) else "",
                                       add_special_tokens=False).input_ids)
                        for b in blocks
                    ]
                    avg_len = sum(lengths) / max(len(lengths), 1)
                except Exception:
                    avg_len = 0.0
            else:
                avg_len = 0.0
        else:
            n_blocks, avg_len = 0, 0.0

        block_component = min(n_blocks, 3) / 3.0        # [0, 1]
        length_component = min(avg_len / 200.0, 1.0)     # [0, 1]

        r_continuous = (block_component + length_component + format_component) / 3.0

        rewards.append(
            correctness_weight * r_correct
            + meta_floor_weight * r_meta
            + continuous_weight * r_continuous
        )
    return rewards


# ─── Trainer ───────────────────────────────────────────────────────────────

class MetaRLSDTrainer(GRPOTrainer):
    """GRPOTrainer + per-token teacher-ratio advantage (§2.1, §2.2).

    Overrides:
        * :meth:`__init__` — spawn frozen teacher snapshot, record λ schedule.
        * :meth:`_generate_and_score_completions` — augment super's output
          with ``meta_mask`` and a group-relative scalar ``A_scalar``.
        * :meth:`_compute_loss` — rewrite advantage broadcast with the
          per-token formula Â_t from §2.1 + PPO asymmetric clip.

    Helpers:
        * :meth:`_sync_teacher` — deepspeed-safe weight copy (§2.4).
        * :meth:`_build_teacher_inputs` — privileged-answer prompt (§2.3).
    """

    def __init__(self, *args, meta_rlsd_cfg: MetaRLSDConfig, **kwargs):
        self.meta_rlsd_cfg = meta_rlsd_cfg
        super().__init__(*args, **kwargs)
        self._meta_tokenizer = self.processing_class

        # TRL 0.19.1 compat: only has `_get_per_token_logps`; newer versions have
        # `_get_per_token_logps_and_entropies`. Provide a shim that computes entropy
        # alongside logps when the newer method is missing.
        if not hasattr(self, "_get_per_token_logps_and_entropies"):
            import types
            def _logps_and_ent(self, model, input_ids, attention_mask, logits_to_keep,
                               compute_entropy: bool = True, batch_size=None):
                batch_size = batch_size or input_ids.size(0)
                all_logps = []
                all_ent = []
                for i in range(0, input_ids.size(0), batch_size):
                    ib = input_ids[i : i + batch_size]
                    ab = attention_mask[i : i + batch_size]
                    logits = model(
                        input_ids=ib, attention_mask=ab, logits_to_keep=logits_to_keep + 1
                    ).logits
                    logits = logits[:, :-1, :]
                    tgt = ib[:, -logits_to_keep:]
                    logits = logits / self.temperature
                    import torch.nn.functional as _F
                    log_probs = _F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    logps = log_probs.gather(dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
                    all_logps.append(logps)
                    if compute_entropy:
                        ent = -(probs * log_probs).sum(dim=-1)
                        all_ent.append(ent)
                logps_cat = torch.cat(all_logps, dim=0)
                if compute_entropy and all_ent:
                    ent_cat = torch.cat(all_ent, dim=0)
                    return logps_cat, ent_cat
                return logps_cat, None
            self._get_per_token_logps_and_entropies = types.MethodType(_logps_and_ent, self)

        # Teacher = frozen bf16 model loaded FRESH from ``teacher_init`` path — §2.4.
        # Rationale (C6): under DeepSpeed ZeRO-3, ``unwrap_model`` returns a module
        # whose parameters are partitioned across ranks, so ``copy.deepcopy`` would
        # yield an incomplete shard — not a full model. Loading a second model from
        # disk sidesteps ZeRO-3 entirely. The teacher lives on CPU until first forward
        # (see ``_teacher_logprobs``) and is kept per-rank (identical weights on each
        # rank because every rank loads the same checkpoint).
        teacher_path = meta_rlsd_cfg.teacher_init
        # Load teacher with ZeRO-3 init disabled — otherwise the `zero3_init_flag`
        # active in the accelerate config would partition the teacher's parameters
        # across ranks, leading to hidden_dim mismatch in forward (0 vs 4096).
        # Two-layer guard against ZeRO-3 auto-init during teacher load:
        # 1) DSInit(enabled=False) — explicitly disabled context manager
        # 2) monkey-patch is_deepspeed_zero3_enabled → False during load —
        #    necessary because transformers >=4.52 calls its own
        #    deepspeed.zero.Init(config_dict_or_path=deepspeed_config())
        #    inside from_pretrained, which fails on ptca's deepspeed where
        #    `deepspeed.runtime` attribute is not auto-imported.
        try:
            from deepspeed.runtime.zero.partition_parameters import Init as _DSInit
            _load_ctx = _DSInit(enabled=False)
        except Exception:
            import contextlib
            _load_ctx = contextlib.nullcontext()

        # Patch: bypass transformers' auto ZeRO-3 init for this call only
        import transformers.modeling_utils as _mu_mod
        _orig_z3_enabled = getattr(_mu_mod, "is_deepspeed_zero3_enabled", None)
        if _orig_z3_enabled is not None:
            _mu_mod.is_deepspeed_zero3_enabled = lambda: False
        try:
            with _load_ctx:
                self.teacher = AutoModelForCausalLM.from_pretrained(
                    teacher_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,  # full load per rank; H200 has VRAM budget
                )
        finally:
            if _orig_z3_enabled is not None:
                _mu_mod.is_deepspeed_zero3_enabled = _orig_z3_enabled
        # Student may have had its vocab resized for new meta tokens — mirror it.
        try:
            target_vocab = len(self._meta_tokenizer)
            cur_vocab = self.teacher.get_input_embeddings().weight.size(0)
            if cur_vocab != target_vocab:
                self.teacher.resize_token_embeddings(target_vocab)
        except Exception as exc:  # pragma: no cover — defensive
            print(f"[meta_rlsd] teacher vocab resize skipped: {exc}")
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self._teacher_device: Optional[torch.device] = None

        # λ schedule state
        self._last_skipped_groups = 0
        self._last_meta_token_fraction = 0.0
        self._last_A_scalar_mean = 0.0
        self._last_A_scalar_std = 0.0
        self._last_teacher_ratio_mean = 0.0
        self._last_teacher_ratio_std = 0.0
        self._last_clip_fraction_w = 0.0

    # ── λ schedule (§2.1 / §2.6) ───────────────────────────────────────

    def _current_lambda(self) -> float:
        """λ schedule — plan §2.1:

            λ_t = max(λ_final, λ_init · max(0, 1 − step/decay_steps))

        The inner ``max(0, frac)`` hardens the zero floor: once ``step``
        exceeds ``lambda_decay_steps``, the schedule yields exactly 0 instead
        of a negative tail. ``lambda_final`` remains as an optional override
        floor (default 0) for ablations that want a non-zero asymptote.
        Review iter1 Fix 3.
        """
        cfg = self.meta_rlsd_cfg
        step = int(self.state.global_step) if self.state is not None else 0
        frac = 1.0 - step / max(1, cfg.lambda_decay_steps)
        val = cfg.lambda_init * max(0.0, frac)
        # lambda_final acts as a floor override (default 0 — plan §2.1)
        return float(max(cfg.lambda_final, val))

    # ── Teacher input construction (§2.3) ──────────────────────────────

    def _build_teacher_inputs(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        ground_truth: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        """Compose ``[teacher_prompt + completion]`` for log-prob scoring.

        * ``privileged_answer=True`` (M1): user content is re-encoded with
          ``" Answer: {gold_answer}"`` appended before the assistant turn.
        * ``privileged_answer=False`` (A1 / A2b): teacher sees student's plain
          prompt — degenerates to GRPO-style self-distillation.
        """
        cfg = self.meta_rlsd_cfg
        tokenizer = self._meta_tokenizer

        if not cfg.privileged_answer:
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "prompt_len": prompt_ids.size(1),
            }

        # Decode student prompts, inject answer, re-encode per batch element.
        teacher_texts: List[str] = []
        for i in range(prompt_ids.size(0)):
            nonpad = prompt_mask[i].bool()
            decoded = tokenizer.decode(prompt_ids[i][nonpad], skip_special_tokens=False)
            gold = str(ground_truth[i]) if i < len(ground_truth) else ""
            # The user turn is the tail of decoded; appending after the
            # decoded prompt preserves the chat template — the completion
            # will be appended on top.
            teacher_texts.append(f"{decoded} Answer: {gold}")

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

    # ── Teacher forward — per-token log P_T (§2.1) ────────────────────

    @torch.no_grad()
    def _teacher_logprobs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        """Return per-token log probs of ``completion_ids`` under teacher."""
        if self._teacher_device is None:
            self._teacher_device = input_ids.device
            self.teacher.to(self._teacher_device)

        logits_to_keep = completion_ids.size(1)
        outputs = self.teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits
        # Shift: logits at position t predict token t+1; we want predictions
        # for completion tokens, which begin at index prompt_len.
        shifted = logits[:, prompt_len - 1 : prompt_len - 1 + logits_to_keep, :]
        shifted = shifted / max(self.meta_rlsd_cfg.temperature, 1e-6)
        log_probs = torch.log_softmax(shifted.float(), dim=-1)
        per_token_logps = log_probs.gather(
            dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)
        return per_token_logps

    # ── Teacher sync (§2.4) ────────────────────────────────────────────

    def _sync_teacher(self) -> None:
        """Copy current student weights into teacher — ZeRO-3 safe (C6).

        Strategy: ``accelerator.get_state_dict`` gathers the full (un-partitioned)
        state on every rank under ZeRO-3. Rank 0 writes a snapshot to disk, all
        ranks barrier, then every rank reloads the same snapshot into its local
        teacher. This avoids rank-local partial state_dicts and keeps teachers
        identical across ranks.
        """
        cfg = self.meta_rlsd_cfg
        snap_dir = os.path.join(cfg.output_dir, "_teacher_snap")
        snap_path = os.path.join(snap_dir, "pytorch_model.bin")

        if self.accelerator.is_main_process:
            os.makedirs(snap_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

        # Gather full state_dict — under ZeRO-3 this all-gathers partitions.
        try:
            state_dict = self.accelerator.get_state_dict(self.model)
        except Exception as exc:  # pragma: no cover — depends on runtime
            if not cfg.teacher_sync_file_fallback:
                raise
            print(f"[meta_rlsd] get_state_dict failed: {exc} — save_pretrained fallback")
            if self.accelerator.is_main_process:
                student = self.accelerator.unwrap_model(self.model)
                student.save_pretrained(
                    cfg.teacher_sync_tmp_dir, safe_serialization=True
                )
            self.accelerator.wait_for_everyone()
            reloaded = AutoModelForCausalLM.from_pretrained(
                cfg.teacher_sync_tmp_dir,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self._teacher_device or torch.device("cpu"))
            reloaded.requires_grad_(False)
            reloaded.eval()
            self.teacher = reloaded
            torch.cuda.empty_cache()
            return

        # Rank 0 persists the gathered state; non-main ranks wait.
        if self.accelerator.is_main_process and state_dict is not None:
            torch.save(state_dict, snap_path)
        self.accelerator.wait_for_everyone()

        # Every rank reloads into its local teacher (identical weights on all ranks).
        new_state = torch.load(snap_path, map_location="cpu")
        new_state_bf16 = {
            k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in new_state.items()
        }
        missing, unexpected = self.teacher.load_state_dict(new_state_bf16, strict=False)
        if (missing or unexpected) and self.accelerator.is_main_process:
            print(
                f"[meta_rlsd] teacher reload: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys"
            )
        if self._teacher_device is not None:
            self.teacher.to(self._teacher_device)
        torch.cuda.empty_cache()

    # ── Override 1: augment generate+score with meta mask + A_scalar ───

    def _generate_and_score_completions(self, inputs):
        """Extend super() with ``meta_mask`` and skipped-group bookkeeping.

        C1 fix: TRL v0.23 passes ``inputs`` as ``list[dict]`` (one per prompt).
        Previously ``inputs.get("ground_truth", ...)`` returned ``[]`` for lists,
        so the privileged-answer path in ``_build_teacher_inputs`` always saw
        empty gold. We now extract the per-prompt gold list BEFORE super() mutates
        ``inputs``, then replicate across the G generations (TRL layout:
        prompt_0 × G, prompt_1 × G, …).
        """
        # C1: pull ground_truth out of the prompt-level list before super runs.
        # Review iter1 Fix 8: also handle the dict-wrapped layout (some TRL code
        # paths and the prediction step wrap the whole batch as a single dict).
        if isinstance(inputs, list):
            gt_list_prompt_level: List[str] = [
                str(x.get("ground_truth", "")) for x in inputs
            ]
        elif isinstance(inputs, dict):
            raw_gt = inputs.get("ground_truth", [])
            if isinstance(raw_gt, (list, tuple)):
                gt_list_prompt_level = [str(x) for x in raw_gt]
            elif raw_gt:  # scalar / tensor — promote to single-element list
                gt_list_prompt_level = [str(raw_gt)]
            else:
                gt_list_prompt_level = []
        else:
            raise TypeError(
                f"_generate_and_score_completions: expected list or dict inputs, "
                f"got {type(inputs).__name__}"
            )

        out = super()._generate_and_score_completions(inputs)

        completion_ids = out["completion_ids"]
        completion_mask = out["completion_mask"]
        tokenizer = self._meta_tokenizer
        cfg = self.meta_rlsd_cfg

        # Build meta mask per completion (§2.1 m_t).
        meta_masks = []
        total_tokens = 0
        total_meta = 0
        for i in range(completion_ids.size(0)):
            ids = completion_ids[i].tolist()
            # Truncate to active tokens for text decoding — padding noise confuses the offset map.
            active_len = int(completion_mask[i].sum().item())
            active_ids = ids[:active_len] if active_len > 0 else ids
            text = tokenizer.decode(active_ids, skip_special_tokens=False)
            mask = _build_meta_mask(tokenizer, ids, text)  # full length (incl. pad slots)
            # Zero mask beyond active length so padding never contributes.
            if mask.size(0) > active_len:
                mask[active_len:] = 0.0
            meta_masks.append(mask)
            total_tokens += active_len
            total_meta += int(mask[:active_len].sum().item())

        meta_mask_tensor = torch.stack(meta_masks, dim=0).to(completion_ids.device)

        if cfg.mask_mode == "all_tokens":
            # A2a / A2b ablation — mask everywhere inside active completion.
            all_mask = completion_mask.float()
            meta_mask_tensor = all_mask

        self._last_meta_token_fraction = (
            (total_meta / max(total_tokens, 1)) if cfg.mask_mode == "meta_only" else 1.0
        )

        out["meta_mask"] = meta_mask_tensor

        # CRITICAL (code review HIGH-1): gold routing must survive TRL's
        # shuffle_tensor_dict. Earlier design cached gold as a Python list on
        # self._batch_ground_truth (pre-shuffle order) while tensors (including
        # completion_ids) were permuted by TRL after this hook returned. That
        # made gt_list[i] correspond to a DIFFERENT problem than
        # completion_ids[i] inside _compute_loss — breaking T+/T- contrastive
        # (privileged "Answer: {wrong_gold}" context).
        #
        # Fix: stash per-rollout gold in a table on self, and put gold's TABLE
        # INDEX into `out` as an int tensor. TRL's shuffle_tensor_dict
        # permutes int tensors in lockstep with completion_ids, preserving
        # alignment. _compute_loss reads inputs["_gold_idx"] to resolve.
        num_gen = self.num_generations
        total_rollouts = completion_ids.size(0)
        if gt_list_prompt_level and total_rollouts == num_gen * len(gt_list_prompt_level):
            per_rollout_gold = [
                gt_list_prompt_level[i // num_gen] for i in range(total_rollouts)
            ]
        elif gt_list_prompt_level and total_rollouts == len(gt_list_prompt_level):
            per_rollout_gold = [str(g) for g in gt_list_prompt_level]
        else:
            per_rollout_gold = [""] * total_rollouts
            if self.accelerator.is_main_process:
                print(
                    "[WARN] meta_rlsd ground_truth routing fallback: "
                    f"total_rollouts={total_rollouts}, "
                    f"gt_len={len(gt_list_prompt_level)}, num_gen={num_gen}"
                )

        # Table lives on self; indices travel through shuffle_tensor_dict.
        self._batch_gold_table = per_rollout_gold
        out["_gold_idx"] = torch.arange(
            total_rollouts, device=completion_ids.device, dtype=torch.long
        )
        # Back-compat: keep _batch_ground_truth set (pre-shuffle order) so any
        # code path that reads it directly still works, but this is no longer
        # authoritative. Authoritative lookup is:
        #   gt_list = [self._batch_gold_table[int(i)] for i in inputs["_gold_idx"].tolist()]
        self._batch_ground_truth = list(per_rollout_gold)

        # Re-compute group-relative A_scalar with degenerate-group skip.
        # TRL already places advantages in ``out['advantages']`` — we want to
        # *preserve* its group normalization (GRPO), but mark degenerate groups
        # (σ<1e-6) so downstream loss can zero them.
        advantages = out.get("advantages")
        if advantages is not None:
            num_gen = self.num_generations
            # Guard: reshape to (num_prompts, num_gen) only if divisible.
            # TRL sometimes passes flat [num_prompts] (pre-expansion) and sometimes
            # [num_prompts * num_gen] (post-expansion) depending on hook stage.
            total = int(advantages.numel())
            if total > 0 and total % num_gen == 0 and total >= num_gen:
                adv_view = advantages.view(-1, num_gen)
                std_view = adv_view.std(dim=1)
                degenerate = std_view < 1e-6
                self._last_skipped_groups = int(degenerate.sum().item())
                if degenerate.any():
                    adv_view[degenerate] = 0.0
                    advantages = adv_view.view(-1)
                    out["advantages"] = advantages
            else:
                # Pre-expansion or small batch: skip degenerate-group check this step.
                self._last_skipped_groups = 0
            self._last_A_scalar_mean = float(advantages.mean().item())
            self._last_A_scalar_std = float(advantages.std().item())

        return out

    # ── Override 2: per-token advantage + PPO clip (§2.1 / §2.2) ───────

    def _compute_loss(self, model, inputs):
        cfg = self.meta_rlsd_cfg

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        # Guard against empty batches (can happen when TRL splits unevenly across ranks).
        if prompt_ids.size(0) == 0 or prompt_ids.size(1) == 0 or completion_ids.size(1) == 0:
            # Return a zero loss so training can continue without this degenerate micro-batch.
            dev = prompt_ids.device if prompt_ids.numel() else (
                completion_ids.device if completion_ids.numel() else torch.device("cpu")
            )
            return torch.zeros((), device=dev, requires_grad=True)
        meta_mask = inputs.get("meta_mask")
        if meta_mask is None:
            # Fallback — build on the fly (e.g., during prediction_step).
            meta_mask = torch.zeros_like(completion_mask, dtype=torch.float32)
        meta_mask = meta_mask.to(completion_mask.device).float()

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
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

        # ── Teacher forward — log P_T for each completion token (§2.1) ──
        # Pull from self cache (set in _generate_and_score_completions); avoids
        # putting Python lists in the shuffled tensor_dict which TRL now indexes.
        # Authoritative gold routing — resolve from _gold_idx tensor so we
        # pick up TRL's shuffle_tensor_dict permutation (HIGH-1 fix). Only
        # falls back to pre-shuffle list if the indexed path is missing
        # (e.g., eval / prediction_step paths that don't go through the hook).
        gold_idx = inputs.get("_gold_idx", None)
        gold_table = getattr(self, "_batch_gold_table", None)
        if gold_idx is not None and gold_table is not None:
            gt_list = [gold_table[int(i)] for i in gold_idx.tolist()]
        else:
            gt_list = getattr(self, "_batch_ground_truth", None) or inputs.get("ground_truth", []) or []
        teacher_pack = self._build_teacher_inputs(
            prompt_ids, prompt_mask, completion_ids, completion_mask, gt_list
        )
        teacher_logps = self._teacher_logprobs(
            teacher_pack["input_ids"],
            teacher_pack["attention_mask"],
            completion_ids,
            teacher_pack["prompt_len"],
        )

        # Δ_t = clamp(sg(log P_T − log P_S), −10, 10)
        log_S = per_token_logps.detach()
        delta_t = torch.clamp(
            teacher_logps - log_S, -cfg.log_ratio_clamp, cfg.log_ratio_clamp
        )
        A_sign = torch.sign(advantages).unsqueeze(1)  # [B,1]
        w_t = torch.exp(A_sign * delta_t)  # [B,T]
        w_t_clip = torch.clamp(w_t, 1.0 - cfg.clip_eps_w, 1.0 + cfg.clip_eps_w)

        lam = self._current_lambda()
        per_token_factor = meta_mask * w_t_clip + (1.0 - meta_mask)
        hat_A = advantages.unsqueeze(1) * ((1.0 - lam) + lam * per_token_factor)

        # ── PPO asymmetric clip (§2.2) ────────────────────────────────
        # W6: clamp log_ratio before exp to avoid NaN/Inf when student and
        # reference diverge (e.g., after a teacher sync or under fp16 underflow).
        log_ratio = per_token_logps - old_per_token_logps
        log_ratio = torch.clamp(log_ratio, -cfg.log_ratio_clamp, cfg.log_ratio_clamp)
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1.0 - cfg.clip_eps_low, 1.0 + cfg.clip_eps_high)

        per_token_loss1 = coef_1 * hat_A
        per_token_loss2 = coef_2 * hat_A
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Token-level normalization (paper equivalence + dr_grpo-style stable)
        denom = completion_mask.sum().clamp(min=1.0)
        loss = (per_token_loss * completion_mask).sum() / denom
        loss = loss / max(self.current_gradient_accumulation_steps, 1)

        # ── Logging (§2.G) ────────────────────────────────────────────
        mode = "train" if self.model.training else "eval"
        metrics = self._metrics[mode]

        # Teacher ratio stats — masked to meta tokens
        meta_flat = (meta_mask * completion_mask).bool()
        if meta_flat.any():
            ratio_vals = w_t[meta_flat]
            self._last_teacher_ratio_mean = float(ratio_vals.mean().item())
            self._last_teacher_ratio_std = float(ratio_vals.std().item())
            clipped = (w_t != w_t_clip) & meta_flat
            self._last_clip_fraction_w = float(clipped.float().sum().item() / max(meta_flat.sum().item(), 1))
        metrics.setdefault("meta_rlsd/teacher_ratio_mean", []).append(self._last_teacher_ratio_mean)
        metrics.setdefault("meta_rlsd/teacher_ratio_std", []).append(self._last_teacher_ratio_std)
        metrics.setdefault("meta_rlsd/clip_fraction_w", []).append(self._last_clip_fraction_w)
        metrics.setdefault("meta_rlsd/meta_token_fraction", []).append(self._last_meta_token_fraction)
        metrics.setdefault("meta_rlsd/lambda_current", []).append(lam)
        metrics.setdefault("meta_rlsd/A_scalar_mean", []).append(self._last_A_scalar_mean)
        metrics.setdefault("meta_rlsd/A_scalar_std", []).append(self._last_A_scalar_std)
        metrics.setdefault("meta_rlsd/skipped_groups", []).append(float(self._last_skipped_groups))
        metrics.setdefault("meta_rlsd/loss", []).append(float(loss.detach().item()))

        # Standard PPO clip fractions — matches TRL 0.19.1 GRPOTrainer convention.
        # Clip fires when min(coef_1·A, clip(coef_1)·A) picks the clip branch:
        #   * A<0 with r<1−ε_low : min picks (1−ε_low)·A (low clip active).
        #   * A>0 with r>1+ε_high: min picks (1+ε_high)·A (high clip active).
        is_low = (coef_1 < 1.0 - cfg.clip_eps_low) & (advantages.unsqueeze(1) < 0)
        is_high = (coef_1 > 1.0 + cfg.clip_eps_high) & (advantages.unsqueeze(1) > 0)
        region = ((is_low | is_high) & completion_mask.bool()).float()
        clip_fraction = float(region.sum().item() / max(completion_mask.sum().item(), 1))
        metrics.setdefault("meta_rlsd/ppo_clip_fraction", []).append(clip_fraction)
        # Also surface as top-level 'clip_fraction' for the abort callback (W5).
        metrics.setdefault("clip_fraction", []).append(clip_fraction)

        return loss


# ─── Teacher-sync callback — fires every ``teacher_sync_freq`` steps ───────

class TeacherSyncCallback(TrainerCallback):
    def __init__(self, trainer: MetaRLSDTrainer):
        self._trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        freq = self._trainer.meta_rlsd_cfg.teacher_sync_freq
        if freq <= 0:
            return
        if state.global_step > 0 and state.global_step % freq == 0:
            self._trainer._sync_teacher()


class ClipFractionAbortCallback(TrainerCallback):
    """Abort training if PPO ``clip_fraction`` stays above ``threshold`` for
    ``window`` consecutive logged steps (plan §4.1).

    A sustained high clip fraction means the policy has drifted far from its
    rollout distribution and PPO is mostly clipping — continuing to train
    typically burns compute without improving the policy.
    """

    def __init__(self, threshold: float = 0.5, window: int = 20):
        self.threshold = float(threshold)
        self.window = int(window)
        self.consecutive = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "clip_fraction" not in logs:
            return
        try:
            value = float(logs["clip_fraction"])
        except (TypeError, ValueError):
            return
        if value > self.threshold:
            self.consecutive += 1
            if self.consecutive >= self.window:
                print(
                    f"[ABORT] clip_fraction > {self.threshold} for "
                    f"{self.window} consecutive steps — stopping training."
                )
                control.should_training_stop = True
        else:
            self.consecutive = 0


# ─── CLI ───────────────────────────────────────────────────────────────────

def _apply_variant(cfg: MetaRLSDConfig, variant: str) -> MetaRLSDConfig:
    """Variant dispatch — see §F of the implementation plan."""
    variant = variant.lower()
    cfg.variant = variant
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
        # Pure GRPO baseline — lambda is effectively irrelevant but we keep the
        # meta-only mask so the loss matches vanilla TRL (meta factor = 1).
        cfg.lambda_init = 0.0
        cfg.lambda_final = 0.0
        cfg.mask_mode = "meta_only"
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return cfg


def _build_grpo_config(cfg: MetaRLSDConfig) -> GRPOConfig:
    return GRPOConfig(
        output_dir=cfg.output_dir,
        max_steps=cfg.total_steps,
        num_generations=cfg.num_rollouts,
        max_completion_length=cfg.max_response_length,
        max_prompt_length=cfg.prompt_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        use_vllm=cfg.use_vllm,
        vllm_mode="colocate" if cfg.use_vllm else None,
        vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size if cfg.use_vllm else 1,
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization if cfg.use_vllm else 0.9,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.lr,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type="dr_grpo",
        beta=cfg.kl_coef,
        scale_rewards=False,
        num_iterations=1,
        logging_steps=1,
        save_steps=cfg.save_interval,
        save_total_limit=3,
        report_to=cfg.report_to,
        run_name=cfg.run_name or f"meta_rlsd_{cfg.variant}_s{cfg.seed}",
        remove_unused_columns=False,
        reward_weights=[1.0],
        log_completions=True,
        max_grad_norm=cfg.log_grad_norm_clip,
        seed=cfg.seed,
        epsilon=cfg.clip_eps_low,
        epsilon_high=cfg.clip_eps_high,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Meta-RLSD trainer")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument(
        "--variant",
        choices=["m1", "a1", "a2a", "a2b", "a4"],
        default="m1",
        help="Variant dispatch — flips privileged_answer / mask_mode / λ",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_preflight", action="store_true",
                        help="Bypass PF1-PF5 (debug/smoke only — NOT for production)")
    args = parser.parse_args(argv)

    cfg = MetaRLSDConfig.from_yaml(args.config)
    cfg.seed = args.seed
    cfg = _apply_variant(cfg, args.variant)

    os.environ.setdefault("WANDB_PROJECT", "metacot-meta-rlsd")

    # ── Tokenizer + meta token normalization ──────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_init, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])

    # ── Pre-flight — §2.10 ────────────────────────────────────────────
    # Review iter1 Fix 7: absolute-resolve dataset paths before preflight so the
    # logs print the actual on-disk path, not a relative fragment depending on cwd.
    cfg.train_data = os.path.abspath(cfg.train_data)
    print(f"[meta_rlsd] resolved train_data: {cfg.train_data}", flush=True)
    if cfg.val_data:
        cfg.val_data = os.path.abspath(cfg.val_data)
        print(f"[meta_rlsd] resolved val_data: {cfg.val_data}", flush=True)
    if args.skip_preflight:
        print("[meta_rlsd] --skip_preflight set: bypassing PF1-PF5", flush=True)
    else:
        report = preflight_checks(
            cfg.train_data,
            tokenizer,
            prompt_length=cfg.prompt_length,
            meta_min_length_tokens=cfg.meta_min_length_tokens,
        )
        # Verbose dump — main process only, to stderr to avoid multi-rank interleaving
        is_main = os.environ.get("RANK", "0") == "0"
        if is_main:
            print("=" * 60, file=sys.stderr, flush=True)
            print(f"[meta_rlsd] PF: passed={report.passed}", file=sys.stderr, flush=True)
            print(f"[meta_rlsd] PF violations ({len(report.violations)}):", file=sys.stderr, flush=True)
            for v in report.violations:
                print(f"  - {v!r}", file=sys.stderr, flush=True)
            print(f"[meta_rlsd] PF warnings ({len(report.warnings)}):", file=sys.stderr, flush=True)
            for w in report.warnings:
                print(f"  - {w!r}", file=sys.stderr, flush=True)
            print(f"[meta_rlsd] PF stats: {report.stats}", file=sys.stderr, flush=True)
            print("=" * 60, file=sys.stderr, flush=True)
        if not report.passed:
            if is_main:
                print("[meta_rlsd] Pre-flight checks FAILED — aborting. Use --skip_preflight to bypass.",
                      file=sys.stderr, flush=True)
            return 2

    # ── Model (student) ───────────────────────────────────────────────
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
    print(f"[meta_rlsd] Loaded {len(train_ds)} training prompts from {cfg.train_data}")

    # ── Reward closure — tokenizer + cfg bound via partial-style wrapper ─
    from functools import partial as _partial

    reward_fn = _partial(
        correctness_plus_meta_floor_reward,
        tokenizer=tokenizer,
        cfg=cfg,
        correctness_weight=cfg.correctness_weight,
        meta_floor_weight=cfg.meta_floor_weight,
    )
    reward_fn.__name__ = "correctness_plus_meta_floor_reward"  # type: ignore[attr-defined]

    grpo_config = _build_grpo_config(cfg)

    trainer = MetaRLSDTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_ds,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        meta_rlsd_cfg=cfg,
    )
    trainer.add_callback(TeacherSyncCallback(trainer))
    trainer.add_callback(ClipFractionAbortCallback(threshold=0.5, window=20))

    # Dump resolved config for reproducibility
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "meta_rlsd_config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    trainer.train()
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[meta_rlsd] Done. Saved to {final_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "MetaRLSDConfig",
    "MetaRLSDTrainer",
    "TeacherSyncCallback",
    "ClipFractionAbortCallback",
    "correctness_plus_meta_floor_reward",
    "main",
]
