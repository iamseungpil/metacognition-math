#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill.pipeline import write_generated_sft_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_config", required=True)
    parser.add_argument("--output_config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_output_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--enable_teacher_kl", action="store_true")
    parser.add_argument("--teacher_kl_coef", type=float, default=0.15)
    parser.add_argument("--teacher_kl_mask_mode", default="meta_only")
    args = parser.parse_args()

    write_generated_sft_config(
        template_path=args.template_config,
        output_config_path=args.output_config,
        model_name_or_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.train_output_dir,
        run_name=args.run_name,
        enable_teacher_kl=args.enable_teacher_kl,
        teacher_kl_coef=args.teacher_kl_coef,
        teacher_kl_mask_mode=args.teacher_kl_mask_mode,
    )
    print(args.output_config)


if __name__ == "__main__":
    main()
