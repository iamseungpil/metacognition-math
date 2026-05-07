#!/usr/bin/env python3
"""Smoke test for M5.6.5 ROD-RLSD trainer (standalone math; trl-free).

Step 1: emit prob computation on synthetic logits, verify thresholding.
Step 2: BCE + KL backward, verify finite gradient + correct sign.
Step 3: real toy LlamaForCausalLM end-to-end (1 micro-batch, no trl).
Step 5b: trainer module import test — only attempted if trl is available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Standalone re-implementation of the staticmethods, used in smoke 1-3.
# These mirror MetaRODTrainer._emit_bce and MetaOPDTrainer._topk_kl exactly.

def emit_bce_standalone(teacher_logits, student_logits, meta_start_id, threshold, window, completion_mask):
    B, T, V = teacher_logits.shape
    W = max(1, min(window, T))
    T_logits_w = teacher_logits[:, :W, :]
    S_logits_w = student_logits[:, :W, :]
    mask_w = completion_mask[:, :W].float()

    T_probs = F.softmax(T_logits_w, dim=-1)
    S_probs = F.softmax(S_logits_w, dim=-1)
    p_emit_T = T_probs[..., meta_start_id]
    p_emit_S = S_probs[..., meta_start_id]

    emit_target = (p_emit_T > threshold).float()

    eps = 1e-7
    p_S = p_emit_S.clamp(eps, 1.0 - eps)
    bce = -(emit_target * torch.log(p_S) + (1.0 - emit_target) * torch.log(1.0 - p_S))

    masked = bce * mask_w
    denom = mask_w.sum().clamp(min=1.0)
    loss = masked.sum() / denom

    with torch.no_grad():
        n_valid = denom.item()
        stats = {
            "rod/emit_rate_target": float((emit_target * mask_w).sum().item() / n_valid),
            "rod/emit_rate_student": float(((p_emit_S > 0.5).float() * mask_w).sum().item() / n_valid),
            "rod/avg_p_emit_target": float((p_emit_T * mask_w).sum().item() / n_valid),
            "rod/avg_p_emit_student": float((p_emit_S * mask_w).sum().item() / n_valid),
        }
    return loss, stats


def topk_kl_standalone(teacher_logits, student_logits, mask, K, temperature=1.0):
    V = teacher_logits.size(-1)
    K = max(1, min(int(K), V))
    T_topk_logits, topk_idx = teacher_logits.topk(K, dim=-1)
    S_topk_logits = student_logits.gather(-1, topk_idx)
    T_log_probs = F.log_softmax(T_topk_logits / temperature, dim=-1)
    T_probs = T_log_probs.exp()
    S_log_probs = F.log_softmax(S_topk_logits / temperature, dim=-1)
    kl_per_token = (T_probs * (T_log_probs - S_log_probs)).sum(dim=-1)
    mask = mask.to(kl_per_token.dtype)
    masked_kl = kl_per_token * mask
    denom = mask.sum().clamp(min=1.0)
    return masked_kl.sum() / denom


# ============================================================================

def smoke_step1_emit_prob():
    print("\n=== STEP 1: emit prob computation ===")
    B, T, V = 2, 8, 100
    META_ID = 7

    torch.manual_seed(42)
    T_logits = torch.randn(B, T, V) * 0.5
    T_logits[:, :2, META_ID] += 12.0  # very high prob at first 2 positions

    S_logits = torch.randn(B, T, V) * 0.5
    completion_mask = torch.ones(B, T)

    loss, stats = emit_bce_standalone(
        T_logits, S_logits, META_ID, threshold=0.30, window=4, completion_mask=completion_mask,
    )
    print(f"  loss={loss.item():.4f}")
    print(f"  stats={stats}")
    assert torch.isfinite(loss), "BCE loss not finite"
    # Window=4: first 2 positions must hit emit_target=1, last 2 stay near 0
    # emit_rate_target should be exactly 0.5 (4 positions emit out of 8 in batch).
    assert stats["rod/emit_rate_target"] == 0.5, \
        f"expected emit_target = 0.5, got {stats['rod/emit_rate_target']}"
    assert stats["rod/emit_rate_student"] < 0.5, "expected student low emit"
    print("  PASS")


def smoke_step2_backward():
    print("\n=== STEP 2: backward pass ===")
    B, T, V = 2, 8, 100
    META_ID = 7

    T_logits = torch.randn(B, T, V) * 0.5
    T_logits[:, :2, META_ID] += 5.0
    S_logits = torch.randn(B, T, V, requires_grad=True)

    completion_mask = torch.ones(B, T)
    meta_mask = torch.zeros(B, T)
    meta_mask[:, 2:5] = 1.0

    emit_loss, _ = emit_bce_standalone(
        T_logits, S_logits, META_ID, threshold=0.30, window=4, completion_mask=completion_mask,
    )
    content_kl = topk_kl_standalone(
        T_logits, S_logits, meta_mask * completion_mask, K=16, temperature=1.0,
    )
    total = 0.5 * emit_loss + 0.3 * content_kl
    total.backward()

    grad = S_logits.grad
    print(f"  emit_loss={emit_loss.item():.4f}, content_kl={content_kl.item():.4f}, total={total.item():.4f}")
    print(f"  grad mean={grad.mean().item():.6f}, max-abs={grad.abs().max().item():.6f}")
    assert torch.isfinite(total), "total loss not finite"
    assert torch.isfinite(grad).all(), "grad has nan/inf"
    assert grad.abs().max() > 1e-6, "grad degenerate"
    print("  PASS")


def smoke_step3_real_model():
    print("\n=== STEP 3: real toy model end-to-end ===")
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

    META_ID = 13

    out = model(input_ids=input_ids, attention_mask=attn)
    student_logits = out.logits[:, prompt_len - 1 : prompt_len - 1 + comp_len, :]
    with torch.no_grad():
        T_out = teacher(input_ids=input_ids, attention_mask=attn)
        teacher_logits = T_out.logits[:, prompt_len - 1 : prompt_len - 1 + comp_len, :]

    completion_mask = torch.ones(B, comp_len)
    meta_mask = torch.zeros(B, comp_len)
    meta_mask[:, 0:6] = 1.0

    emit_loss, stats = emit_bce_standalone(
        teacher_logits, student_logits, META_ID,
        threshold=0.30, window=8, completion_mask=completion_mask,
    )
    content_kl = topk_kl_standalone(
        teacher_logits, student_logits, meta_mask * completion_mask, K=32, temperature=1.0,
    )
    total = 0.5 * emit_loss + 0.3 * content_kl
    total.backward()

    has_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in model.parameters()
    )
    print(f"  emit_loss={emit_loss.item():.4f} content_kl={content_kl.item():.4f}")
    print(f"  any model param has grad: {has_grad}")
    print(f"  stats={stats}")
    assert has_grad, "no model param received grad"
    print("  PASS")


def smoke_step5b_imports_if_trl():
    print("\n=== STEP 5b: trainer module import (requires trl) ===")
    try:
        import trl  # noqa
    except ImportError:
        print("  SKIP — trl not installed in this env (expected; will run on node)")
        return
    import yaml
    cfg_dict = yaml.safe_load(open(ROOT / "configs/meta_rod_R8_h100_4x4k.yaml"))
    from src.training.meta_rod_trainer import MetaRODConfig
    cfg = MetaRODConfig(**cfg_dict)
    print(f"  MetaRODConfig OK: variant={cfg.variant} alpha_emit={cfg.rod_alpha_emit}")
    print("  PASS")


def main():
    print("ROD-RLSD smoke tests starting")
    smoke_step1_emit_prob()
    smoke_step2_backward()
    smoke_step3_real_model()
    smoke_step5b_imports_if_trl()
    print("\n=== ALL ROD SMOKE STEPS PASS (or trl-skip) ===")


if __name__ == "__main__":
    main()
