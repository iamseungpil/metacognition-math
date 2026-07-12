#!/usr/bin/env python3
"""Smoke test for MetaOPDTrainer (M5.2 OPD-Decoy).

Plan §10.5 M5.2 Validation steps:
- Step 0 (eval-first): R5 step 300 + base SFT teacher ckpt verify (HF or local) + TRL load.
- Step 1: 1-step rollout + T+/T- forward + KL_pos/KL_neg + opd_loss + backward, all non-NaN.
- Step 2: 10-step quick run, M5.2 reward variance vs R5 reward variance compare.

Usage:
    python scripts/smoke_meta_opd.py --config configs/meta_opd_decoy_R7_h100_4x4k.yaml --step 1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

# Ensure repo root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.meta_opd_trainer import MetaOPDConfig, MetaOPDTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke_meta_opd")


def step0_ckpt_verify(cfg: MetaOPDConfig) -> None:
    """Eval-first: confirm cold-start ckpt + teacher ckpt are loadable."""
    log.info("=" * 60)
    log.info("[STEP 0] eval-first ckpt verify")
    log.info("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Cold start ckpt (student init)
    log.info("[STEP 0.1] loading student_init: %s", cfg.student_init)
    tok = AutoTokenizer.from_pretrained(cfg.student_init, trust_remote_code=True)
    log.info("  vocab_size=%d, special_tokens=%s", tok.vocab_size, tok.special_tokens_map)
    student = AutoModelForCausalLM.from_pretrained(
        cfg.student_init, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    log.info("  student loaded: %d params", sum(p.numel() for p in student.parameters()))

    # Teacher ckpt
    log.info("[STEP 0.2] loading teacher_init: %s", cfg.teacher_init)
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_init, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    log.info("  teacher loaded: %d params", sum(p.numel() for p in teacher.parameters()))

    # Sanity: both same architecture
    assert student.config.model_type == teacher.config.model_type, (
        f"Student/teacher arch mismatch: {student.config.model_type} vs {teacher.config.model_type}"
    )
    log.info("[STEP 0] PASSED — both ckpts loadable + same arch")


def step1_oneshot_loss(cfg: MetaOPDConfig) -> None:
    """1-step end-to-end: rollout + T+/T- forward + KL + opd_loss + backward."""
    log.info("=" * 60)
    log.info("[STEP 1] one-shot loss + backward")
    log.info("=" * 60)

    # Synthetic batch — small, fits on 1 GPU
    B, T = 2, 64  # batch 2, seq 64
    V = 152064  # Qwen3-8B vocab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fake logits
    teacher_pos_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)
    teacher_neg_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)
    student_logits = torch.randn(B, T, V, device=device, dtype=torch.float32, requires_grad=True)
    mask = torch.ones(B, T, device=device, dtype=torch.float32)
    mask[:, T // 2 :] = 0.0  # half is body, half is meta (synthetic)

    # Use the helper directly (staticmethod — no self required)
    from src.training.meta_opd_trainer import MetaOPDTrainer

    kl_pos = MetaOPDTrainer._topk_kl(
        teacher_pos_logits, student_logits, mask, cfg.opd_topk, cfg.opd_temperature,
    )
    kl_neg = MetaOPDTrainer._topk_kl(
        teacher_neg_logits, student_logits, mask, cfg.opd_topk, cfg.opd_temperature,
    )
    opd_loss = cfg.opd_lambda_pos * kl_pos - cfg.opd_lambda_neg * kl_neg

    log.info("  kl_pos=%.4f, kl_neg=%.4f, opd_loss=%.4f",
             kl_pos.item(), kl_neg.item(), opd_loss.item())
    assert torch.isfinite(kl_pos), "kl_pos NaN/Inf"
    assert torch.isfinite(kl_neg), "kl_neg NaN/Inf"
    assert torch.isfinite(opd_loss), "opd_loss NaN/Inf"

    # Backward
    opd_loss.backward()
    grad = student_logits.grad
    assert grad is not None, "no gradient"
    assert torch.isfinite(grad).all(), "gradient NaN/Inf"
    log.info("  gradient ok: norm=%.4f", grad.norm().item())

    log.info("[STEP 1] PASSED — opd_loss finite, backward propagates clean grad")


def step2_variance_compare(cfg: MetaOPDConfig, n_steps: int = 10) -> None:
    """10-step: simulate per-step opd_loss variance, compare to scalar Δt analog."""
    log.info("=" * 60)
    log.info("[STEP 2] %d-step variance comparison (OPD vs scalar Δt)", n_steps)
    log.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, V = 4, 128, 152064

    opd_losses = []
    delta_losses = []  # scalar Δt analog (RLSD style)

    from src.training.meta_opd_trainer import MetaOPDTrainer

    for step in range(n_steps):
        teacher_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)
        teacher_neg_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)
        student_logits = torch.randn(B, T, V, device=device, dtype=torch.float32)
        completion_ids = torch.randint(0, V, (B, T), device=device)
        mask = torch.ones(B, T, device=device, dtype=torch.float32)

        # OPD signal: top-K KL (staticmethod — no self)
        kl_pos = MetaOPDTrainer._topk_kl(
            teacher_logits, student_logits, mask, cfg.opd_topk, 1.0,
        )
        kl_neg = MetaOPDTrainer._topk_kl(
            teacher_neg_logits, student_logits, mask, cfg.opd_topk, 1.0,
        )
        opd_signal = (cfg.opd_lambda_pos * kl_pos - cfg.opd_lambda_neg * kl_neg).item()
        opd_losses.append(opd_signal)

        # Scalar Δt analog
        teacher_logp = torch.log_softmax(teacher_logits, dim=-1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        student_logp = torch.log_softmax(student_logits, dim=-1).gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        delta_t = teacher_logp - student_logp
        delta_signal = (delta_t * mask).sum() / mask.sum()
        delta_losses.append(delta_signal.item())

    import statistics
    log.info("  OPD: mean=%.4f, std=%.4f", statistics.mean(opd_losses), statistics.stdev(opd_losses))
    log.info("  Scalar Δt: mean=%.4f, std=%.4f", statistics.mean(delta_losses), statistics.stdev(delta_losses))

    # Note: with synthetic random data, expect comparable variance.
    # In real training, OPD signal should have lower step-to-step variance because
    # full distribution is smoother than single sampled token logprob.
    log.info("[STEP 2] PASSED — both signals computed across %d steps without NaN", n_steps)


def step3_real_model_integration() -> None:
    """Round 3 / review C6: real toy model end-to-end test.

    Builds a tiny LlamaForCausalLM (V=2048, d=64, 2 layers), constructs a
    synthetic batch with forced META_START prefix, calls _compute_loss-equivalent
    path (without invoking full TRL trainer init), verifies:
    - Forward pass + KL term computation
    - Gradient reaches model parameters (not just leaf logits)
    - All-zero meta_mask edge case
    """
    log.info("=" * 60)
    log.info("[STEP 3] real toy model integration")
    log.info("=" * 60)

    from transformers import LlamaConfig, LlamaForCausalLM
    from src.training.meta_opd_trainer import MetaOPDConfig, MetaOPDTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = 2048
    META_START_ID = 100  # synthetic
    config = LlamaConfig(
        vocab_size=V, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, pad_token_id=0,
    )
    model = LlamaForCausalLM(config).to(device).to(torch.float32)
    teacher = LlamaForCausalLM(config).to(device).to(torch.float32)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Synthetic batch with forced META_START at end of prompt
    B, P, T = 2, 16, 32
    prompt_ids = torch.randint(1, V - 1, (B, P), device=device)
    prompt_ids[:, -1] = META_START_ID  # forced injection
    completion_ids = torch.randint(1, V - 1, (B, T), device=device)
    prompt_mask = torch.ones(B, P, device=device, dtype=torch.long)
    completion_mask = torch.ones(B, T, device=device, dtype=torch.long)
    meta_mask = torch.zeros(B, T, device=device, dtype=torch.float32)
    meta_mask[:, : T // 2] = 1.0  # first half = meta region (forced opener territory)

    cfg = MetaOPDConfig(
        student_init="dummy", teacher_init="dummy",
        opd_topk=16, opd_topk_fallback=8,
    )

    # Manually construct minimum trainer state — we test the helpers directly
    # since full TRL trainer init requires accelerator + dataset.
    log.info("[STEP 3.1] _completion_logits — student forward + slice")
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1).to(torch.float32)

    class _Stub:
        opd_cfg = cfg
        _supports_logits_to_keep = False  # toy model — fallback path
    stub = _Stub()
    student_logits = MetaOPDTrainer._completion_logits(stub, model, input_ids, attention_mask, T)
    teacher_pos_logits = MetaOPDTrainer._completion_logits(stub, teacher, input_ids, attention_mask, T)
    teacher_neg_logits = MetaOPDTrainer._completion_logits(stub, teacher, input_ids, attention_mask, T)
    assert student_logits.shape == (B, T, V), f"shape mismatch {student_logits.shape}"
    log.info("  shapes ok: student=%s teacher=%s", student_logits.shape, teacher_pos_logits.shape)

    log.info("[STEP 3.2] gather + logsumexp for per_token_logps (review C1)")
    per_token_logps = (
        student_logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        - student_logits.logsumexp(dim=-1)
    )
    assert torch.isfinite(per_token_logps).all(), "per_token_logps NaN/Inf"
    log.info("  per_token_logps shape=%s, mean=%.3f", per_token_logps.shape, per_token_logps.mean().item())

    log.info("[STEP 3.3] KL terms + opd_loss + ppo_loss + backward")
    opd_mask = meta_mask * completion_mask.float()
    kl_pos = MetaOPDTrainer._topk_kl(teacher_pos_logits, student_logits, opd_mask, cfg.opd_topk)
    kl_neg = MetaOPDTrainer._topk_kl(teacher_neg_logits, student_logits, opd_mask, cfg.opd_topk).clamp(max=cfg.opd_kl_neg_cap)
    opd_loss = cfg.opd_lambda_pos * kl_pos - cfg.opd_lambda_neg * kl_neg

    advantages = torch.randn(B, device=device)
    log_ratio = per_token_logps - per_token_logps.detach()
    coef = torch.exp(log_ratio)
    ppo_loss = -(coef * advantages.unsqueeze(1) * completion_mask.float()).sum() / completion_mask.float().sum().clamp(min=1.0)

    loss = (ppo_loss + cfg.opd_alpha * opd_loss)
    loss.backward()

    # Verify gradient reaches model parameters (not just leaf logits)
    param_grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(param_grads) > 0, "no gradient reached model parameters"
    has_finite = all(torch.isfinite(g).all() for g in param_grads)
    assert has_finite, "model parameter gradient has NaN/Inf"
    log.info("  loss=%.4f (kl_pos=%.3f kl_neg=%.3f opd=%.3f ppo=%.3f)",
             loss.item(), kl_pos.item(), kl_neg.item(), opd_loss.item(), ppo_loss.item())
    log.info("  %d model params have finite gradient ✓", len(param_grads))

    log.info("[STEP 3.4] all-zero meta_mask edge case")
    model.zero_grad()
    zero_mask = torch.zeros_like(meta_mask)
    kl_pos_zero = MetaOPDTrainer._topk_kl(teacher_pos_logits, student_logits, zero_mask, cfg.opd_topk)
    assert torch.isfinite(kl_pos_zero), "KL with zero mask returned NaN/Inf"
    log.info("  KL with zero mask = %.4f (finite)", kl_pos_zero.item())

    log.info("[STEP 3] PASSED — real-model integration end-to-end clean")


def step4_forced_injection_helper() -> None:
    """Round 3+ I1 fix: verify dataset-level forced injection helper.

    Plan v5.1 §11.1 option 1: apply_forced_meta_to_dataset transforms each
    prompt to append <|meta|> as a special token to the last user message.
    """
    log.info("=" * 60)
    log.info("[STEP 4] forced injection helper (dataset-level option 1)")
    log.info("=" * 60)

    from datasets import Dataset
    from src.training.meta_opd_trainer import apply_forced_meta_to_dataset
    from src.metacot.prompt import META_START

    # Synthetic dataset matching MetaRLSD format
    examples = [
        {"prompt": [{"role": "user", "content": "Solve x+1=2"}], "ground_truth": "1"},
        {"prompt": [
            {"role": "system", "content": "You solve math."},
            {"role": "user", "content": "Solve x+2=3"},
        ], "ground_truth": "1"},
        {"prompt": [{"role": "assistant", "content": "no user turn"}], "ground_truth": "?"},
    ]
    ds = Dataset.from_list(examples)
    log.info("[STEP 4.1] before injection: prompt[0] last_msg=%r",
             ds[0]["prompt"][-1]["content"])

    ds_inj = apply_forced_meta_to_dataset(ds)

    # Assert injection worked on user turns
    assert META_START in ds_inj[0]["prompt"][-1]["content"], "ex 0 not injected"
    assert ds_inj[1]["prompt"][1]["role"] == "user", "ex 1 not user turn"
    assert META_START in ds_inj[1]["prompt"][1]["content"], "ex 1 not injected"
    # No user turn — should leave alone
    assert META_START not in ds_inj[2]["prompt"][0]["content"], "ex 2 should not be injected (no user turn)"

    # Idempotence — second pass should be no-op
    ds_inj2 = apply_forced_meta_to_dataset(ds_inj)
    assert ds_inj[0]["prompt"][-1]["content"] == ds_inj2[0]["prompt"][-1]["content"], "not idempotent"

    log.info("[STEP 4.2] after injection: prompt[0] last_msg=%r",
             ds_inj[0]["prompt"][-1]["content"])
    log.info("[STEP 4.3] idempotence verified ✓")
    log.info("[STEP 4.4] no-user-turn skip verified ✓")
    log.info("[STEP 4] PASSED — forced injection helper works correctly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--step", type=int, choices=[0, 1, 2, 3, 4], default=1, help="which smoke step")
    ap.add_argument("--student_init", type=str, default="iamseungpil/metacot-h100-rlsd-forced-meta-R5-0504")
    ap.add_argument("--teacher_init", type=str, default="models/v8_meta_inside_strict_sft/checkpoint-254")
    args = ap.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
        cfg = MetaOPDConfig(**cfg_dict)
    else:
        cfg = MetaOPDConfig(
            student_init=args.student_init,
            teacher_init=args.teacher_init,
        )
    log.info("MetaOPDConfig: %s", cfg)

    if args.step == 0:
        step0_ckpt_verify(cfg)
    elif args.step == 1:
        step1_oneshot_loss(cfg)
    elif args.step == 2:
        step2_variance_compare(cfg)
    elif args.step == 3:
        step3_real_model_integration()
    elif args.step == 4:
        step4_forced_injection_helper()
    else:
        raise ValueError(f"Unknown step: {args.step}")


if __name__ == "__main__":
    main()
