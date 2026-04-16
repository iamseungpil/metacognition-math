#!/usr/bin/env python3
"""Upload only the latest checkpoint (or final folder) to the HF dataset repo.

This keeps a single mutable remote path up to date instead of stacking
checkpoint-* directories indefinitely.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi


def _latest_checkpoint(root: Path) -> Path:
    ckpts = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", child.name)
        if match:
            ckpts.append((int(match.group(1)), child))
    if ckpts:
        ckpts.sort(key=lambda x: x[0])
        return ckpts[-1][1]
    return root


def _is_complete_checkpoint(path: Path) -> bool:
    has_weights = any(
        candidate.exists()
        for candidate in (path / "model.safetensors", path / "pytorch_model.bin")
    )
    has_metadata = all(
        candidate.exists()
        for candidate in (path / "config.json", path / "tokenizer_config.json")
    )
    return has_weights and has_metadata


def _delete_remote_prefix(api: HfApi, repo_id: str, repo_type: str, prefix: str) -> None:
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type, files_metadata=False)
    paths = []
    prefix = prefix.rstrip("/")
    for sibling in info.siblings:
        path = sibling.rfilename
        if path == prefix or path.startswith(prefix + "/"):
            paths.append(path)
    if paths:
        api.delete_files(
            repo_id=repo_id,
            repo_type=repo_type,
            delete_patterns=paths,
            commit_message=f"Clear previous latest sync at {prefix}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    selected = _latest_checkpoint(source_dir)
    if not _is_complete_checkpoint(selected):
        raise RuntimeError(f"Incomplete checkpoint candidate: {selected}")
    api = HfApi(token=args.token)

    _delete_remote_prefix(api, args.repo_id, args.repo_type, args.path_in_repo)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(selected),
        path_in_repo=args.path_in_repo.rstrip("/"),
        revision=args.revision,
        commit_message=f"Sync latest from {selected.name}",
    )
    manifest = {
        "source_dir": str(source_dir),
        "selected": str(selected),
        "selected_name": selected.name,
    }
    api.upload_file(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_or_fileobj=json.dumps(manifest, indent=2).encode(),
        path_in_repo=args.path_in_repo.rstrip("/") + "/sync_manifest.json",
        revision=args.revision,
        commit_message=f"Update sync manifest for {selected.name}",
    )
    print(f"synced {selected} -> {args.repo_id}:{args.path_in_repo}")


if __name__ == "__main__":
    main()
