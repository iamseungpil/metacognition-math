"""Watch a checkpoint directory and push new `global_step_N/` dirs to HuggingFace.

Designed to run as a nohup daemon alongside SDC veRL training.  Deduplicates via
`/scratch/.pushed_<config>.json` so preemption → resume doesn't re-upload.

Usage:
    python scripts/push_ckpts_to_hf.py \
        --ckpt_dir /scratch/checkpoints/verl_sdc_e21r_shared_40g8 \
        --repo_id iamseungpil/metacot-sdc-verl-shared \
        --token "$HF_TOKEN" \
        --interval 600 \
        --config_name verl_sdc_e21r_shared_40g8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--config_name", default="sdc")
    ap.add_argument("--include_wandb", action="store_true", default=True)
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)
    try:
        create_repo(repo_id=args.repo_id, token=args.token, repo_type="model", exist_ok=True)
    except Exception as exc:
        print(f"[push] repo create skipped: {exc}")

    marker = Path(f"/scratch/.pushed_{args.config_name}.json")
    done: set[str] = set()
    if marker.exists():
        try:
            done = set(json.loads(marker.read_text()))
        except Exception:
            done = set()

    ckpt_dir = Path(args.ckpt_dir)
    wandb_run_dir = Path("/scratch/metacognition/wandb")

    print(f"[push] daemon start ckpt_dir={ckpt_dir} repo={args.repo_id} interval={args.interval}s done={len(done)}")

    while True:
        try:
            if ckpt_dir.exists():
                for step_dir in sorted(ckpt_dir.glob("global_step_*")):
                    if not step_dir.is_dir() or step_dir.name in done:
                        continue
                    # Skip if still being written (no recent write for >30s)
                    latest_mtime = max((p.stat().st_mtime for p in step_dir.rglob("*") if p.is_file()), default=0)
                    if latest_mtime == 0 or time.time() - latest_mtime < 30:
                        continue

                    path_in_repo = f"checkpoints/{args.config_name}/{step_dir.name}"
                    print(f"[push] uploading {step_dir.name} → {args.repo_id}:{path_in_repo}")
                    api.upload_folder(
                        folder_path=str(step_dir),
                        repo_id=args.repo_id,
                        path_in_repo=path_in_repo,
                        repo_type="model",
                        commit_message=f"ckpt {step_dir.name} ({args.config_name})",
                        ignore_patterns=["*.tmp", "*.lock"],
                    )
                    done.add(step_dir.name)
                    marker.write_text(json.dumps(sorted(done)))
                    print(f"[push] done {step_dir.name}")

            # Push wandb run files (best-effort, overwrites)
            if args.include_wandb and wandb_run_dir.exists():
                latest = sorted(wandb_run_dir.glob("run-*"), key=lambda p: p.stat().st_mtime)
                if latest:
                    target = latest[-1]
                    try:
                        api.upload_folder(
                            folder_path=str(target / "files"),
                            repo_id=args.repo_id,
                            path_in_repo=f"wandb/{args.config_name}",
                            repo_type="model",
                            commit_message=f"wandb {target.name}",
                            allow_patterns=["*.log", "*.yaml", "wandb-summary.json"],
                        )
                    except Exception as exc:
                        print(f"[push] wandb upload skip: {exc}")

        except Exception as exc:
            print(f"[push] err: {exc}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
