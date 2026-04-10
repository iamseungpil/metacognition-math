#!/usr/bin/env python
"""Upload the latest checkpoint or completed root model to a HF repo root."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi


MODEL_FILE_NAMES = {
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
}


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _root_complete(path: Path) -> bool:
    return (path / "config.json").exists() and (
        (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    )


def _pick_upload_dir(root: Path) -> tuple[Path, str]:
    checkpoints = sorted(
        [p for p in root.iterdir() if p.is_dir() and _checkpoint_step(p) >= 0],
        key=_checkpoint_step,
    )
    if checkpoints:
        latest = checkpoints[-1]
        return latest, latest.name
    if _root_complete(root):
        return root, "root"
    raise FileNotFoundError(f"No uploadable model artifacts found under {root}")


def _upload(api: HfApi, folder: Path, repo_id: str, repo_type: str, commit_message: str) -> None:
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="",
        allow_patterns=sorted(MODEL_FILE_NAMES | {"*.md", "*.txt"}),
        commit_message=commit_message,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--interval-sec", type=int, default=0)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        raise RuntimeError("HF_TOKEN is required")

    root = Path(args.local_dir).resolve()
    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True, private=args.private)

    last_label: str | None = None
    while True:
        folder, label = _pick_upload_dir(root)
        if label != last_label:
            print(f"[hf-sync] uploading {folder} -> {args.repo_id}", flush=True)
            _upload(api, folder, args.repo_id, args.repo_type, f"Update from {root.name}: {label}")
            last_label = label
        else:
            print(f"[hf-sync] no newer checkpoint under {root}", flush=True)

        if args.interval_sec <= 0:
            return 0
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    sys.exit(main())
