#!/usr/bin/env python3
"""Smoke test for ROD-PT (Plan v5.7).

Step 1: SDC factor compute (standalone math, trl-free)
Step 2: Position penalty top-K logic
Step 3: backward gradient finite
Step 4: real toy LlamaForCausalLM end-to-end
Step 5: trainer module import (skipped if trl not installed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sdc_factor_standalone(
    teacher_logits, student_logits, token_ids, sign_advantage, clip_low, clip_high,
):
    """Mirror MetaRODPTTrainer._compute_sdc_factor."""
    logp_T = (
        teacher_logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        - teacher_logits.logsumexp(dim=-1)
    )
    logp_S = (
        student_logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        - student_logits.logsumexp(dim=-1)
    )
    sign_A_per_token = sign_advantage.unsqueeze(1).expand_as(logp_T)
    log_factor = sign_A_per_token * (logp_T - logp_S)
    return torch.exp(log_factor).clamp(min=clip_low, max=clip_high)


def smoke_step1_sdc_factor():
    print("\n=== STEP 1: SDC factor compute ===")
    B, T, V = 2, 8, 100
    torch.manual_seed(42)
    teacher_logits = torch.randn(B, T, V) * 2.0
    student_logits = torch.randn(B, T, V) * 2.0
    token_ids = torch.randint(0, V, (B, T))
    sign_A = torch.tensor([1.0, -1.0])  # one positive, one negative

    factor = sdc_factor_standalone(
        teacher_logits, student_logits, token_ids, sign_A, 0.2, 5.0,
    )
    print(f"  factor shape={tuple(factor.shape)}, min={factor.min():.4f}, max={factor.max():.4f}, mean={factor.mean():.4f}")
    assert factor.shape == (B, T)
    assert (factor >= 0.2 - 1e-6).all() and (factor <= 5.0 + 1e-6).all(), "factor out of clip range"
    assert torch.isfinite(factor).all()
    print("  PASS")


def smoke_step2_position_top_k():
    print("\n=== STEP 2: Position top-K logic ===")
    META_ID = 7
    K = 16
    V = 100

    # Case 1: META in top-K (high logit at META_ID)
    logits_in = torch.randn(V) * 0.5
    logits_in[META_ID] += 10.0
    top_K_in = logits_in.topk(K).indices
    in_top_K_1 = (top_K_in == META_ID).any().item()
    assert in_top_K_1, "META should be in top-K"

    # Case 2: META NOT in top-K (low logit)
    logits_out = torch.randn(V) * 0.5
    logits_out[META_ID] -= 10.0
    top_K_out = logits_out.topk(K).indices
    in_top_K_2 = (top_K_out == META_ID).any().item()
    assert not in_top_K_2, "META should NOT be in top-K"

    print(f"  Case 1 (META in top-K): {in_top_K_1}, Case 2 (META not in top-K): {in_top_K_2}")
    print("  PASS")


def smoke_step3_backward():
    print("\n=== STEP 3: backward gradient ===")
    B, T, V = 2, 8, 100
    teacher_logits = torch.randn(B, T, V) * 2.0
    student_logits = torch.randn(B, T, V, requires_grad=True)
    token_ids = torch.randint(0, V, (B, T))
    sign_A = torch.tensor([1.0, -1.0])

    factor = sdc_factor_standalone(
        teacher_logits, student_logits, token_ids, sign_A, 0.2, 5.0,
    )
    # Per-token logp (must broadcast as [B, T] not [B])
    logp = (
        student_logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        - student_logits.logsumexp(dim=-1)
    )  # [B, T]
    advantages = torch.tensor([0.5, -0.3])
    coef = torch.exp(logp.clamp(-10, 10))  # [B, T]
    per_token_loss = -coef * advantages.unsqueeze(1) * factor  # [B,T] * [B,1] * [B,T] OK
    meta_mask = torch.zeros(B, T)
    meta_mask[:, 2:5] = 1.0
    loss = (per_token_loss * meta_mask).sum() / meta_mask.sum().clamp(min=1.0)
    loss.backward()

    grad = student_logits.grad
    print(f"  loss={loss.item():.4f}, grad max-abs={grad.abs().max().item():.6f}")
    assert torch.isfinite(loss), "loss not finite"
    assert torch.isfinite(grad).all(), "grad has nan/inf"
    assert grad.abs().max() > 1e-9, "grad degenerate"
    print("  PASS")


def smoke_step4_real_model():
    print("\n=== STEP 4: real toy LlamaForCausalLM ===")
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=2048, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, intermediate_size=128,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(cfg)
    teacher = LlamaForCausalLM(cfg)

    B, prompt_len, comp_len = 2, 16, 24
    prompt_ids = torch.randint(0, cfg.vocab_size, (B, prompt_len))
    completion_ids = torch.randint(0, cfg.vocab_size, (B, comp_len))
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn)
    student_logits = out.logits[:, prompt_len - 1: prompt_len - 1 + comp_len, :]

    with torch.no_grad():
        T_out = teacher(input_ids=input_ids, attention_mask=attn)
        teacher_logits = T_out.logits[:, prompt_len - 1: prompt_len - 1 + comp_len, :]

    advantages = torch.tensor([0.5, -0.3])
    sign_A = torch.sign(advantages)
    factor = sdc_factor_standalone(
        teacher_logits, student_logits, completion_ids, sign_A, 0.2, 5.0,
    )

    meta_mask = torch.zeros(B, comp_len)
    meta_mask[:, 5:12] = 1.0
    factor_meta = factor * meta_mask + 1.0 * (1 - meta_mask)

    log_ratio = (
        student_logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        - student_logits.logsumexp(dim=-1)
    )
    coef = torch.exp(log_ratio.clamp(-10, 10))
    advantage_per_token = advantages.unsqueeze(1).expand_as(coef)
    per_token_loss = -coef * advantage_per_token * factor_meta
    completion_mask = torch.ones(B, comp_len)
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    loss.backward()

    has_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0) for p in model.parameters()
    )
    print(f"  loss={loss.item():.4f} factor_meta_mean={factor_meta.mean():.4f}")
    print(f"  any model param has grad: {has_grad}")
    assert has_grad, "model received no grad"
    print("  PASS")


def smoke_step5_imports():
    print("\n=== STEP 5: trainer import (requires trl) ===")
    try:
        import trl  # noqa
    except ImportError:
        print("  SKIP — trl not installed (expected; runs on node)")
        return
    import yaml
    cfg_path = ROOT / "configs/meta_rod_pt_R10_h100_4x4k.yaml"
    if not cfg_path.exists():
        print(f"  config {cfg_path} not yet created — skip")
        return
    cfg_dict = yaml.safe_load(open(cfg_path))
    from src.training.meta_rod_pt_trainer import MetaRODPTConfig
    cfg = MetaRODPTConfig(**cfg_dict)
    print(f"  MetaRODPTConfig OK: variant={cfg.variant} K={cfg.pt_position_top_k} penalty={cfg.pt_position_penalty}")
    print("  PASS")


def main():
    print("ROD-PT smoke tests starting")
    smoke_step1_sdc_factor()
    smoke_step2_position_top_k()
    smoke_step3_backward()
    smoke_step4_real_model()
    smoke_step5_imports()
    print("\n=== ALL ROD-PT SMOKE STEPS PASS ===")


if __name__ == "__main__":
    main()
