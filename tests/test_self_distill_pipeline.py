from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill.pipeline import build_generated_sft_config


def test_build_generated_sft_config_plain_sft():
    template = {
        "model_name_or_path": "old-model",
        "dataset_path": "old.parquet",
        "output_dir": "old-output",
        "run_name": "old-run",
    }
    cfg = build_generated_sft_config(
        template,
        model_name_or_path="checkpoints/v8_meta_inside_strict_sft",
        dataset_path="results/rq3_online_sdpo_regen/online_sdpo_regen.parquet",
        output_dir="checkpoints/rq3_sdpo_regen_sft",
        run_name="rq3-sdpo-regen-sft",
    )
    assert cfg["model_name_or_path"] == "checkpoints/v8_meta_inside_strict_sft"
    assert cfg["dataset_path"].endswith("online_sdpo_regen.parquet")
    assert "teacher_kl" not in cfg


def test_build_generated_sft_config_enables_teacher_kl():
    template = {
        "model_name_or_path": "old-model",
        "dataset_path": "old.parquet",
        "output_dir": "old-output",
        "run_name": "old-run",
    }
    cfg = build_generated_sft_config(
        template,
        model_name_or_path="checkpoints/v8_meta_inside_strict_sft",
        dataset_path="results/rq3_online_sdpo_regen/teacher_topk_targets.parquet",
        output_dir="checkpoints/rq3_sdpo_regen_meta_kl",
        run_name="rq3-sdpo-regen-meta-kl",
        enable_teacher_kl=True,
        teacher_kl_coef=0.2,
        teacher_kl_mask_mode="meta_only",
    )
    assert cfg["teacher_kl"]["enabled"] is True
    assert cfg["teacher_kl"]["coef"] == 0.2
    assert cfg["teacher_kl"]["mask_mode"] == "meta_only"


def test_build_generated_sft_config_disables_teacher_kl_when_not_requested():
    template = {
        "model_name_or_path": "old-model",
        "dataset_path": "old.parquet",
        "output_dir": "old-output",
        "run_name": "old-run",
        "teacher_kl": {
            "enabled": True,
            "coef": 0.2,
            "require_targets": True,
            "mask_mode": "meta_only",
        },
    }
    cfg = build_generated_sft_config(
        template,
        model_name_or_path="checkpoints/v8_meta_inside_strict_sft",
        dataset_path="results/self_distill/meta_qonly_epistemic/online_sdpo_regen.parquet",
        output_dir="checkpoints/plain_sft",
        run_name="plain-sft",
        enable_teacher_kl=False,
    )
    assert cfg["teacher_kl"]["enabled"] is False
    assert cfg["teacher_kl"]["require_targets"] is False
