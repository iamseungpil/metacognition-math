#!/usr/bin/env python3
"""Wait for and materialize a model artifact from the HF dataset repo."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def _copy_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copytree(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--repo-id", default="iamseungpil/metacot")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="/scratch/metacognition/hf_cache")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config_path = output_dir / "config.json"
    if config_path.exists():
        print(f"already_present:{output_dir}")
        return 0

    allow_patterns = [f"models/{args.model_name}/*"]
    deadline = time.time() + args.timeout_seconds

    while True:
        try:
            local_dir = Path(snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                allow_patterns=allow_patterns,
                local_dir=args.cache_dir,
                local_dir_use_symlinks=False,
            ))
            src = local_dir / "models" / args.model_name
            if not src.exists():
                raise FileNotFoundError(src)
            _copy_tree(src, output_dir)
            print(f"downloaded:{output_dir}")
            return 0
        except Exception as exc:  # pragma: no cover - network dependent
            if not args.wait or time.time() >= deadline:
                print(f"failed:{type(exc).__name__}:{exc}")
                return 1
            print(f"waiting_for_hf:{args.model_name}:{type(exc).__name__}")
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
