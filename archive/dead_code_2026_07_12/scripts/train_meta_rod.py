#!/usr/bin/env python3
"""Launch M5.6.5 ROD-RLSD training (plan_meta_rod_v53_2026_05_07).

On-demand emit + content KL distillation, single teacher (two queries), no decoy.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("train_meta_rod")


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
    from src.training.meta_rod_trainer import MetaRODConfig, MetaRODTrainer
    from src.training.meta_rlsd_data_pipeline import (
        load_meta_rlsd_dataset,
        preflight_checks,
    )
    from src.training.meta_rlsd_trainer import (
        ClipFractionAbortCallback,
        _build_grpo_config,
        correctness_plus_meta_floor_reward,
    )
    from src.training.tokenizer_utils import ensure_meta_tokens_not_special

    cfg = MetaRODConfig(**cfg_dict)
    cfg.seed = args.seed
    cfg.train_data = os.path.abspath(cfg.train_data)
    log.info(
        "[ROD] cfg variant=%s total_steps=%d alpha_emit=%.2f beta_content=%.2f "
        "tau=%.2f window=%d cold=%s",
        cfg.variant, cfg.total_steps, cfg.rod_alpha_emit, cfg.rod_beta_content,
        cfg.rod_emit_threshold, cfg.rod_emit_window, cfg.student_init,
    )

    os.environ.setdefault("WANDB_PROJECT", "metacot-meta-rlsd")

    tokenizer = AutoTokenizer.from_pretrained(cfg.student_init, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])

    if not args.skip_preflight:
        report = preflight_checks(
            cfg.train_data,
            tokenizer,
            prompt_length=cfg.prompt_length,
            meta_min_length_tokens=cfg.meta_min_length_tokens,
        )
        is_main = os.environ.get("RANK", "0") == "0"
        if is_main:
            print(f"[ROD] PF: passed={report.passed}", file=sys.stderr, flush=True)
            for v in report.violations:
                print(f"  PF violation: {v!r}", file=sys.stderr)
        if not report.passed:
            print("[ROD] Pre-flight FAILED — abort. Use --skip_preflight to bypass.",
                  file=sys.stderr, flush=True)
            return 2

    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_init,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    train_ds = load_meta_rlsd_dataset(cfg.train_data)
    log.info("[ROD] Loaded %d training prompts", len(train_ds))
    log.info("[ROD] No forced injection (on-demand emit)")

    reward_fn = partial(
        correctness_plus_meta_floor_reward,
        tokenizer=tokenizer,
        cfg=cfg,
        correctness_weight=cfg.correctness_weight,
        meta_floor_weight=cfg.meta_floor_weight,
        continuous_weight=cfg.continuous_weight,
    )
    reward_fn.__name__ = "correctness_plus_meta_floor_reward"

    grpo_config = _build_grpo_config(cfg)

    log.info("[ROD] Initializing MetaRODTrainer")
    trainer = MetaRODTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        meta_rlsd_cfg=cfg,
        rod_cfg=cfg,
    )

    trainer.add_callback(ClipFractionAbortCallback(threshold=0.5, window=20))
    log.info("[ROD] callbacks: ClipFractionAbortCallback")

    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "meta_rod_config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    log.info("[ROD] Starting training: max_steps=%d", grpo_config.max_steps)
    trainer.train()

    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info("[ROD] Training complete - final ckpt at %s", final_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
