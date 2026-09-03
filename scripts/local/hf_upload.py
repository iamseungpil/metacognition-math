#!/usr/bin/env python3
"""scripts/local/hf_upload.py — upload ONE merged bf16 checkpoint dir to HF.

Uploads ``<local-dir>`` (expected to contain config.json + safetensors, i.e. the
output of ``verl.model_merger merge``) to the HF model repo
``iamseungpil/metacot-countdown-local`` at:

    runs/<LINEAGE>/step_<N>/   (append-only historical record)
    runs/<LINEAGE>/latest/     (overwritten every call for this lineage)

Never deletes anything outside ``runs/<LINEAGE>/latest/`` — the ``delete_patterns``
passed to ``upload_folder`` is scoped to that one prefix, and only on the second
(``latest``) upload call.

No HF_TOKEN -> prints a clear message and exits 0 (does not fail the caller;
scripts/local/run_arm.sh treats upload as best-effort).

Usage:
    python scripts/local/hf_upload.py --lineage cd7_B_p3_s0 --step 50 \\
        --local-dir /hdd_data/seungpil/scratch/merged/cd7_B_p3_s0/step_50
    python scripts/local/hf_upload.py --dry-run --help
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ID_DEFAULT = "iamseungpil/metacot-countdown-local"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=REPO_ID_DEFAULT,
                     help=f"HF model repo (default: {REPO_ID_DEFAULT})")
    ap.add_argument("--lineage", help="e.g. cd7_B_p3_s0 (required unless --dry-run without other args)")
    ap.add_argument("--step", type=int, help="judgment step number, e.g. 50")
    ap.add_argument("--local-dir", help="merged checkpoint dir (config.json + safetensors)")
    ap.add_argument("--dry-run", action="store_true",
                     help="print planned actions, make no network calls")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.dry_run and (args.lineage is None or args.local_dir is None or args.step is None):
        print("[hf_upload] --dry-run with no --lineage/--step/--local-dir given: "
              "nothing to plan, exiting 0 (use --help for usage).")
        return 0

    missing = [n for n, v in (("--lineage", args.lineage), ("--step", args.step),
                               ("--local-dir", args.local_dir)) if v is None]
    if missing:
        print(f"[hf_upload] missing required args: {', '.join(missing)}", file=sys.stderr)
        return 2

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[hf_upload] no HF_TOKEN, skipping upload")
        return 0

    if not os.path.isdir(args.local_dir):
        print(f"[hf_upload] FATAL: local dir does not exist: {args.local_dir}", file=sys.stderr)
        return 1

    has_config = os.path.isfile(os.path.join(args.local_dir, "config.json"))
    has_weights = any(fn.endswith((".safetensors", ".safetensors.index.json"))
                       for fn in os.listdir(args.local_dir))
    if not (has_config and has_weights):
        print(f"[hf_upload] WARNING: {args.local_dir} does not look like a merged "
              f"HF checkpoint (config.json={has_config}, safetensors={has_weights}) — "
              "uploading anyway, but check the merge step.", file=sys.stderr)

    step_path = f"runs/{args.lineage}/step_{args.step}"
    latest_path = f"runs/{args.lineage}/latest"

    if args.dry_run:
        print(f"[hf_upload] DRY RUN would create/use repo_id={args.repo_id} (private=False)")
        print(f"[hf_upload] DRY RUN would upload_folder({args.local_dir!r} -> {step_path!r}, "
              "no delete_patterns)")
        print(f"[hf_upload] DRY RUN would upload_folder({args.local_dir!r} -> {latest_path!r}, "
              f"delete_patterns=[{latest_path}/*])")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=False, exist_ok=True)

    print(f"[hf_upload] uploading {args.local_dir} -> {args.repo_id}:{step_path}")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.local_dir,
        path_in_repo=step_path,
        commit_message=f"{args.lineage} step {args.step}: merged bf16 checkpoint",
    )

    print(f"[hf_upload] uploading {args.local_dir} -> {args.repo_id}:{latest_path} "
          f"(overwriting latest/, delete_patterns scoped to {latest_path}/*)")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.local_dir,
        path_in_repo=latest_path,
        # Scoped so ONLY files under runs/<lineage>/latest/ that are absent from
        # this upload get deleted. Never touches runs/<lineage>/step_*/ or any
        # other lineage's files.
        delete_patterns=[f"{latest_path}/*"],
        commit_message=f"{args.lineage}: update latest/ -> step {args.step}",
    )

    print("[hf_upload] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
