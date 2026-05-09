#!/usr/bin/env python3
"""Launch M5.2 OPD-Decoy training (plan v5.1 §10.5 M5.2).

Mirrors `meta_rlsd_trainer.main()` but uses MetaOPDTrainer (full-logit KL +
decoy contrast on top-K subset) and applies forced META_START injection at
dataset prep time (plan v5.1 §11.1 option 1).

Usage:
    python scripts/train_meta_opd.py --config configs/meta_opd_decoy_R7_h100_4x4k.yaml
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("train_meta_opd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip_forced_injection", action="store_true",
        help="Skip dataset-level META_START injection (plan v5.1 §11.1 option 1).",
    )
    ap.add_argument(
        "--skip_preflight", action="store_true",
        help="Bypass pre-flight checks (debug only).",
    )
    args = ap.parse_args()

    log.info("Loading config: %s", args.config)
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    # Defer heavy imports until after config validation
    from src.metacot.prompt import META_END, META_START
    from src.training._decoy_utils import _rule_based_decoy  # noqa: F401
    from src.training.meta_opd_trainer import (
        MetaOPDConfig,
        MetaOPDTrainer,
        apply_forced_meta_to_dataset,
    )
    from src.training.meta_rlsd_data_pipeline import (
        load_meta_rlsd_dataset,
        preflight_checks,
    )
    from src.training.meta_rlsd_trainer import (
        _build_grpo_config,
        correctness_plus_meta_floor_reward,
    )
    from src.training.tokenizer_utils import ensure_meta_tokens_not_special

    cfg = MetaOPDConfig(**cfg_dict)
    cfg.seed = args.seed
    cfg.train_data = os.path.abspath(cfg.train_data)
    log.info("[OPD] cfg variant=%s total_steps=%d alpha=%.2f K=%d cold=%s",
             cfg.variant, cfg.total_steps, cfg.opd_alpha, cfg.opd_topk, cfg.student_init)

    os.environ.setdefault("WANDB_PROJECT", "metacot-meta-rlsd")

    # Tokenizer + meta token normalization
    tokenizer = AutoTokenizer.from_pretrained(cfg.student_init, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])

    # Pre-flight checks (skip-able for smoke)
    if not args.skip_preflight:
        report = preflight_checks(
            cfg.train_data,
            tokenizer,
            prompt_length=cfg.prompt_length,
            meta_min_length_tokens=cfg.meta_min_length_tokens,
        )
        is_main = os.environ.get("RANK", "0") == "0"
        if is_main:
            print(f"[OPD] PF: passed={report.passed}", file=sys.stderr, flush=True)
            for v in report.violations:
                print(f"  PF violation: {v!r}", file=sys.stderr)
        if not report.passed:
            print("[OPD] Pre-flight FAILED — abort. Use --skip_preflight to bypass.",
                  file=sys.stderr, flush=True)
            return 2

    # Student model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_init,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Dataset
    train_ds = load_meta_rlsd_dataset(cfg.train_data)
    log.info("[OPD] Loaded %d training prompts", len(train_ds))

    # Forced META_START injection (plan v5.1 §11.1 option 1)
    if not args.skip_forced_injection:
        train_ds = apply_forced_meta_to_dataset(train_ds, tokenizer=tokenizer)
    else:
        log.warning("[OPD] forced injection SKIPPED — distribution mismatch with R5 cold start")

    # Reward closure (parent's helper)
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

    # Initialize trainer
    log.info("[OPD] Initializing MetaOPDTrainer")
    trainer = MetaOPDTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=train_ds,
        processing_class=tokenizer,
        meta_rlsd_cfg=cfg,
        opd_cfg=cfg,
    )

    # Add safety callbacks (review C4 fix)
    from src.training.meta_rlsd_trainer import ClipFractionAbortCallback
    trainer.add_callback(ClipFractionAbortCallback(threshold=0.5, window=20))
    log.info("[OPD] callbacks registered: ClipFractionAbortCallback (no TeacherSyncCallback for frozen teacher)")

    # Dump resolved config for reproducibility (review C4 fix)
    import dataclasses, json
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "meta_opd_config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2, default=str)

    log.info("[OPD] Starting training: max_steps=%d", grpo_config.max_steps)
    resume_path = getattr(cfg, "resume_from_checkpoint", None)
    if resume_path and os.path.isdir(resume_path):
        log.info("[OPD] Resuming from checkpoint: %s", resume_path)
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        trainer.train()

    # Save final ckpt + tokenizer (review C4 fix)
    final_dir = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    log.info("[OPD] Training complete — final ckpt at %s", final_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
