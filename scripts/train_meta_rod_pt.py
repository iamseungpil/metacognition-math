#!/usr/bin/env python3
"""Launch ROD-PT training (Plan v5.7).

R5 RLSD framework + position teacher (decoy off).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("train_meta_rod_pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_preflight", action="store_true")
    args = ap.parse_args()

    log.info("Loading config: %s", args.config)
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    from src.metacot.prompt import META_END, META_START
    from src.training.meta_rod_pt_trainer import MetaRODPTConfig, MetaRODPTTrainer
    from src.training.meta_rlsd_data_pipeline import load_meta_rlsd_dataset, preflight_checks
    from src.training.meta_rlsd_trainer import (
        ClipFractionAbortCallback,
        _build_grpo_config,
        correctness_plus_meta_floor_reward,
    )
    from src.training.tokenizer_utils import ensure_meta_tokens_not_special

    cfg = MetaRODPTConfig(**cfg_dict)
    cfg.seed = args.seed
    cfg.train_data = os.path.abspath(cfg.train_data)
    log.info("[ROD-PT] cfg variant=%s top_K=%d penalty=%.2f cold=%s",
             cfg.variant, cfg.pt_position_top_k, cfg.pt_position_penalty, cfg.student_init)

    os.environ.setdefault("WANDB_PROJECT", "metacot-meta-rlsd")

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_init, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])

    if not args.skip_preflight:
        report = preflight_checks(
            cfg.train_data, tokenizer,
            prompt_length=cfg.prompt_length,
            meta_min_length_tokens=cfg.meta_min_length_tokens,
        )
        if os.environ.get("RANK", "0") == "0":
            print(f"[ROD-PT] PF: passed={report.passed}", file=sys.stderr, flush=True)
            for v in report.violations:
                print(f"  PF violation: {v!r}", file=sys.stderr)
        if not report.passed:
            return 2

    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_init, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True, use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    train_ds = load_meta_rlsd_dataset(cfg.train_data)
    log.info("[ROD-PT] Loaded %d training prompts (no forced injection)", len(train_ds))

    reward_fn = partial(
        correctness_plus_meta_floor_reward,
        tokenizer=tokenizer, cfg=cfg,
        correctness_weight=cfg.correctness_weight,
        meta_floor_weight=0.0,  # ROD-PT relies on position penalty, not floor reward
        continuous_weight=cfg.continuous_weight,
    )
    reward_fn.__name__ = "correctness_only_reward"

    grpo_config = _build_grpo_config(cfg)
    trainer = MetaRODPTTrainer(
        model=model, reward_funcs=[reward_fn], args=grpo_config,
        train_dataset=train_ds, processing_class=tokenizer,
        meta_rlsd_cfg=cfg, rod_pt_cfg=cfg,
    )

    trainer.add_callback(ClipFractionAbortCallback(threshold=0.5, window=20))

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "rod_pt_config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    log.info("[ROD-PT] Starting training: max_steps=%d", grpo_config.max_steps)
    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info("[ROD-PT] Training complete - final ckpt at %s", final_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
