"""Pull the latest veRL ``global_step_N`` checkpoint from HF into the local
checkpoint dir so ``trainer.resume_mode=auto`` can resume after a cross-node
preemption wiped ``/scratch``.

Why this exists (durability gap): veRL ``resume_mode=auto`` only inspects the
local ``default_local_dir``. On a fresh node after preemption that dir is empty,
so training restarts from step 0 even though HF holds a recent checkpoint. This
script bridges that gap: it finds the highest ``global_step_N`` on the HF repo,
downloads it into ``<local_dir>/global_step_N/``, and writes the
``latest_checkpointed_iteration.txt`` pointer veRL reads. Idempotent and
best-effort — if HF has nothing (first launch) it is a no-op and training cold
starts as before.

Usage (run BEFORE the verl launch, after the base model is staged):
    python scripts/pull_resume_ckpt.py \
        --repo iamseungpil/metacot-h200-e9-bci-inject \
        --config_name verl_e9_bci_rlvr \
        --local_dir /scratch/checkpoints/verl_e9_bci_rlvr
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


def _max_step(repo: str, config_name: str, token: str):
    """Return (max_step:int, files:list[str]) for checkpoints/<config>/global_step_*."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:
        print(f"[resume] repo list failed ({exc}); cold start")
        return None, []
    prefix = f"checkpoints/{config_name}/global_step_"
    steps = set()
    for f in files:
        m = re.search(rf"{re.escape(prefix)}(\d+)/", f)
        if m:
            steps.add(int(m.group(1)))
    if not steps:
        print(f"[resume] no global_step_* under checkpoints/{config_name}/ on {repo}; cold start")
        return None, []
    return max(steps), files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--local_dir", required=True, help="veRL default_local_dir (parent of global_step_N)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    step, _ = _max_step(args.repo, args.config_name, args.token)
    if step is None:
        return

    local_dir = Path(args.local_dir)
    target = local_dir / f"global_step_{step}"
    # If a complete-looking local copy already exists, don't re-download.
    if (target / "actor").exists() and any(target.glob("actor/model_world_size_*")):
        print(f"[resume] local global_step_{step} already present; skip download")
    else:
        local_dir.mkdir(parents=True, exist_ok=True)
        # snapshot_download lays files at <download_root>/checkpoints/<cfg>/global_step_N/...
        # so download into a staging root then move the step dir into local_dir.
        staging = local_dir / ".resume_dl"
        snapshot_download(
            repo_id=args.repo,
            repo_type="model",
            token=args.token,
            allow_patterns=[f"checkpoints/{args.config_name}/global_step_{step}/**"],
            local_dir=str(staging),
        )
        src = staging / "checkpoints" / args.config_name / f"global_step_{step}"
        if not src.exists():
            print(f"[resume] download did not yield {src}; cold start")
            return
        import shutil

        shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(src), str(target))
        shutil.rmtree(staging, ignore_errors=True)
        print(f"[resume] downloaded global_step_{step} -> {target}")

    # veRL reads this pointer to pick the iteration to resume from.
    pointer = local_dir / "latest_checkpointed_iteration.txt"
    pointer.write_text(str(step))
    print(f"[resume] wrote {pointer} = {step}; resume_mode=auto will resume from step {step}")


if __name__ == "__main__":
    main()
