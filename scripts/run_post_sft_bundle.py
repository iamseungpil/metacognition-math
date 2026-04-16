#!/usr/bin/env python3
"""Run the mainline post-SFT bundle for strict paired checkpoints.

Stages:
1. deterministic paired eval
2. confidence report
3. behavior uncertainty extraction + smoke + critic
4. qualitative AIME case extraction
5. entropy analysis for the strict meta checkpoint
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    print("\n[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise SystemExit(f"missing {what}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_model", default="checkpoints/v8_meta_inside_strict_sft")
    parser.add_argument("--base_model", default="checkpoints/v8_base_matched_strict_sft")
    parser.add_argument("--output_root", default="results/post_sft_bundle")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500", "aime2024"])
    parser.add_argument("--max_problems", type=int, default=30)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--entropy_max_samples", type=int, default=120)
    parser.add_argument("--entropy_window", type=int, default=8)
    parser.add_argument("--skip_entropy", action="store_true")
    parser.add_argument("--device_map", default="auto", choices=["single", "auto"])
    args = parser.parse_args()

    meta_model = ROOT / args.meta_model
    base_model = ROOT / args.base_model
    ensure_exists(meta_model, "meta model")
    ensure_exists(base_model, "base model")

    output_root = ROOT / args.output_root
    eval_dir = output_root / "eval"
    confidence_dir = output_root / "confidence"
    behavior_dir = output_root / "behavior"
    entropy_dir = output_root / "entropy_meta"
    aime_dir = output_root / "aime_cases"
    for path in [eval_dir, confidence_dir, behavior_dir, entropy_dir, aime_dir]:
        path.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "src/eval/eval_hf.py",
            "--model_path",
            str(meta_model),
            "--model_name",
            "strict_meta_sft",
            "--benchmarks",
            *args.benchmarks,
            "--max_problems",
            str(args.max_problems),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--temperature",
            "0",
            "--top_p",
            "1.0",
            "--seed",
            "42",
            "--device_map",
            args.device_map,
            "--output_dir",
            str(eval_dir),
        ]
    )
    run(
        [
            sys.executable,
            "src/eval/eval_hf.py",
            "--model_path",
            str(base_model),
            "--model_name",
            "strict_base_sft",
            "--benchmarks",
            *args.benchmarks,
            "--max_problems",
            str(args.max_problems),
            "--max_new_tokens",
            str(args.max_new_tokens),
            "--temperature",
            "0",
            "--top_p",
            "1.0",
            "--seed",
            "42",
            "--device_map",
            args.device_map,
            "--output_dir",
            str(eval_dir),
        ]
    )

    ensure_exists(eval_dir / "eval_strict_meta_sft.json", "strict meta eval json")
    ensure_exists(eval_dir / "eval_strict_base_sft.json", "strict base eval json")

    run(
        [
            sys.executable,
            "scripts/analyze_confidence_distribution.py",
            "--results_dir",
            str(eval_dir),
            "--output",
            str(confidence_dir / "confidence_report.txt"),
        ]
    )

    targets = {
        "models": [
            {"name": "strict_meta_sft", "eval_json": str(eval_dir / "eval_strict_meta_sft.json")},
            {"name": "strict_base_sft", "eval_json": str(eval_dir / "eval_strict_base_sft.json")},
        ]
    }
    targets_path = behavior_dir / "targets.json"
    targets_path.write_text(json.dumps(targets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    run(
        [
            sys.executable,
            "analysis/behavior_uncertainty_lab/scripts/run_smoke.py",
            "--targets",
            str(targets_path),
        ]
    )
    run(
        [
            sys.executable,
            "analysis/behavior_uncertainty_lab/scripts/extract_behavior_uncertainty.py",
            "--targets",
            str(targets_path),
            "--outdir",
            str(behavior_dir),
        ]
    )
    run(
        [
            sys.executable,
            "analysis/behavior_uncertainty_lab/scripts/run_critic.py",
            "--summary",
            str(behavior_dir / "behavior_uncertainty_summary.csv"),
        ]
    )

    run(
        [
            sys.executable,
            "scripts/extract_aime_qualitative.py",
            "--eval_json",
            str(eval_dir / "eval_strict_meta_sft.json"),
            "--output",
            str(aime_dir / "strict_meta_sft_aime_cases.md"),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/extract_aime_qualitative.py",
            "--eval_json",
            str(eval_dir / "eval_strict_base_sft.json"),
            "--output",
            str(aime_dir / "strict_base_sft_aime_cases.md"),
        ]
    )

    if not args.skip_entropy:
        ensure_exists(eval_dir / "eval_strict_meta_sft.parquet", "strict meta eval parquet")
        run(
            [
                sys.executable,
                "scripts/analyze_entropy_meta.py",
                "--model_path",
                str(meta_model),
                "--eval_parquet",
                str(eval_dir / "eval_strict_meta_sft.parquet"),
                "--output_dir",
                str(entropy_dir),
                "--max_samples",
                str(args.entropy_max_samples),
                "--window",
                str(args.entropy_window),
            ]
        )

    manifest = {
        "meta_model": str(meta_model),
        "base_model": str(base_model),
        "output_root": str(output_root),
        "benchmarks": args.benchmarks,
        "max_problems": args.max_problems,
        "max_new_tokens": args.max_new_tokens,
        "eval_mode": {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
        },
        "entropy_enabled": not args.skip_entropy,
    }
    (output_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\ncompleted bundle -> {output_root}")


if __name__ == "__main__":
    main()
