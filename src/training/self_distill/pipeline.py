"""Helpers for generating clean SFT configs for self-distill pipelines."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_generated_sft_config(
    template: dict[str, Any],
    *,
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    run_name: str,
    enable_teacher_kl: bool = False,
    teacher_kl_coef: float = 0.15,
    teacher_kl_mask_mode: str = "meta_only",
) -> dict[str, Any]:
    config = deepcopy(template)
    config["model_name_or_path"] = model_name_or_path
    config["dataset_path"] = dataset_path
    config["output_dir"] = output_dir
    config["run_name"] = run_name

    if enable_teacher_kl:
        teacher = dict(config.get("teacher_kl", {}) or {})
        teacher.update({
            "enabled": True,
            "coef": float(teacher_kl_coef),
            "require_targets": True,
            "mask_mode": teacher_kl_mask_mode,
        })
        config["teacher_kl"] = teacher
    elif "teacher_kl" in config:
        teacher = dict(config.get("teacher_kl", {}) or {})
        teacher.update({
            "enabled": False,
            "require_targets": False,
        })
        config["teacher_kl"] = teacher
    return config


def write_generated_sft_config(
    *,
    template_path: str | Path,
    output_config_path: str | Path,
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    run_name: str,
    enable_teacher_kl: bool = False,
    teacher_kl_coef: float = 0.15,
    teacher_kl_mask_mode: str = "meta_only",
) -> dict[str, Any]:
    template = load_yaml_config(template_path)
    config = build_generated_sft_config(
        template,
        model_name_or_path=model_name_or_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        run_name=run_name,
        enable_teacher_kl=enable_teacher_kl,
        teacher_kl_coef=teacher_kl_coef,
        teacher_kl_mask_mode=teacher_kl_mask_mode,
    )
    out = Path(output_config_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    return config


__all__ = [
    "build_generated_sft_config",
    "load_yaml_config",
    "write_generated_sft_config",
]
