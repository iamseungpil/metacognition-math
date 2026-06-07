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


def _step_num(name: str) -> int:
    """Extract the integer step from a ``global_step_N`` folder name.

    Returns -1 for names that don't end in an int so they sort oldest and are
    never confused with the just-uploaded latest.
    """
    try:
        return int(name.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def _prune_old_verl_ckpts(api, repo_id: str, config_name: str, keep: int, latest_name: str) -> None:
    """Delete older ``checkpoints/<config>/global_step_*`` folders on the HF repo.

    Keeps only the most-recent ``keep`` checkpoints by step number. The
    just-uploaded ``latest_name`` is always preserved. Best-effort: any failure
    is logged and swallowed so the daemon loop never aborts on a prune error.
    Only veRL ``global_step_*`` dirs are considered; TRL ``checkpoint-*`` dirs
    are left untouched.
    """
    if keep <= 0:
        return
    base = f"checkpoints/{config_name}"
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=base,
            recursive=False,
        )
    except Exception as exc:  # repo/path may not exist yet
        print(f"[push] prune list skip: {exc}")
        return

    # Collect immediate subfolder names that look like global_step_*.
    step_names: list[str] = []
    for ent in entries:
        path = getattr(ent, "path", None)
        if path is None:
            continue
        name = path.rsplit("/", 1)[-1]
        is_dir = getattr(ent, "tree_id", None) is not None or type(ent).__name__ == "RepoFolder"
        if name.startswith("global_step_") and is_dir:
            step_names.append(name)

    # Sort newest-first by step number; keep the top `keep`, delete the rest.
    step_names.sort(key=_step_num, reverse=True)
    to_delete = step_names[keep:]
    for name in to_delete:
        if name == latest_name:
            continue  # never delete the just-uploaded latest
        try:
            api.delete_folder(
                path_in_repo=f"{base}/{name}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"prune {name} (keep latest {keep})",
            )
            print(f"[push] pruned old ckpt {name}")
        except Exception as exc:
            print(f"[push] prune {name} skip: {exc}")


def _squash_history(api, repo_id: str) -> None:
    """Collapse the repo's commit history into a single commit (history 정리).

    Each checkpoint upload adds a ~16GB commit; with keep=1 the old refs are
    pruned but the LFS blobs linger in history, bloating usedStorage. super_squash
    rewrites history to one commit referencing only current files. Best-effort:
    any failure is logged and swallowed so the daemon never aborts.
    """
    try:
        api.super_squash_history(repo_id=repo_id, repo_type="model")
        print(f"[push] squashed history for {repo_id}")
    except Exception as exc:
        print(f"[push] squash skip: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--config_name", default="sdc")
    ap.add_argument("--include_wandb", action="store_true", default=True)
    ap.add_argument(
        "--keep",
        type=int,
        default=1,
        help="Keep only the most-recent N veRL global_step_* checkpoints on the "
        "HF repo; older ones are deleted after a successful upload (default 1). "
        "Set <=0 to disable pruning (keep all). TRL checkpoint-N/ dirs are never pruned.",
    )
    ap.add_argument(
        "--squash_every",
        type=int,
        default=20,
        help="After every N successful uploads, super_squash_history to collapse the "
        "bloated commit history (each ~16GB checkpoint commit otherwise lingers in LFS "
        "history). Keeps current files intact. Best-effort. Set <=0 to disable.",
    )
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

    uploads_since_squash = 0
    while True:
        try:
            if ckpt_dir.exists():
                # Match BOTH veRL (global_step_N/) AND TRL (checkpoint-N/) save formats.
                _all = list(ckpt_dir.glob("global_step_*")) + list(ckpt_dir.glob("checkpoint-*"))
                _gs = sorted(
                    [d for d in _all if d.is_dir() and d.name.startswith("global_step_")],
                    key=lambda x: _step_num(x.name),
                )
                _trl = [d for d in _all if d.is_dir() and d.name.startswith("checkpoint-")]
                # DURABILITY (E.8/E.9 cross-node preempt): upload ONLY the newest veRL
                # global_step_* dir (skip the backlog) so HF always holds a recent
                # checkpoint for resume — a ~16GB sequential upload can NEVER keep up
                # with save_freq under preemption, leaving HF stuck far behind. Mark all
                # older global_step dirs done so they are never uploaded. TRL
                # checkpoint-*/ keeps full per-dir upload (its resume layout needs all).
                if _gs:
                    for _old in _gs[:-1]:
                        done.add(_old.name)
                    ckpt_candidates = ([_gs[-1]] if _gs[-1].name not in done else []) + _trl
                else:
                    ckpt_candidates = _trl
                for step_dir in ckpt_candidates:
                    if not step_dir.is_dir() or step_dir.name in done:
                        continue
                    # Skip if still being written (no recent write for >5s).
                    # Lowered from 30s to 5s — TRL atomic save means mtime stable instantly,
                    # and BSC preempt may cut us before 30s.
                    latest_mtime = max((p.stat().st_mtime for p in step_dir.rglob("*") if p.is_file()), default=0)
                    if latest_mtime == 0 or time.time() - latest_mtime < 5:
                        continue

                    # TRL ckpts → checkpoint-N/ at repo root (resume yaml expects this).
                    # veRL ckpts → checkpoints/<config>/global_step_N/ (legacy SDC).
                    if step_dir.name.startswith("checkpoint-"):
                        path_in_repo = step_dir.name
                    else:
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

                    # History 정리: periodically collapse bloated LFS commit history.
                    uploads_since_squash += 1
                    if args.squash_every > 0 and uploads_since_squash >= args.squash_every:
                        _squash_history(api, args.repo_id)
                        uploads_since_squash = 0

                    # Keep-only-latest pruning: only for veRL global_step_* dirs
                    # (TRL checkpoint-N/ resume layout must retain all). Never
                    # touches the just-uploaded latest. Best-effort.
                    if step_dir.name.startswith("global_step_"):
                        _prune_old_verl_ckpts(
                            api,
                            repo_id=args.repo_id,
                            config_name=args.config_name,
                            keep=args.keep,
                            latest_name=step_dir.name,
                        )

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
