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
    """Return (max COMPLETE step:int, files:list[str]) for checkpoints/<config>/global_step_*.

    Complete = at least 4 ``actor/model_world_size_*_rank_*.pt`` shards on the
    repo. The pusher uploads per-file (durability under preemption), so a
    partially-uploaded step CAN be visible on HF; resuming from one would crash
    or corrupt training. Only complete steps are resume candidates.
    """
    from collections import Counter

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:
        print(f"[resume] repo list failed ({exc}); cold start")
        return None, []
    pat = re.compile(
        rf"checkpoints/{re.escape(config_name)}/global_step_(\d+)/actor/model_world_size_\d+_rank_\d+\.pt$"
    )
    shard_counts: Counter = Counter()
    for f in files:
        m = pat.match(f)
        if m:
            shard_counts[int(m.group(1))] += 1
    complete = {s for s, n in shard_counts.items() if n >= 4}
    partial = set(shard_counts) - complete
    if partial:
        print(f"[resume] ignoring PARTIAL steps on HF: {sorted(partial)}")
    if not complete:
        print(f"[resume] no COMPLETE global_step_* under checkpoints/{config_name}/ on {repo}; cold start")
        return None, []
    return max(complete), files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config_name", required=True)
    ap.add_argument("--local_dir", required=True, help="veRL default_local_dir (parent of global_step_N)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    step, _ = _max_step(args.repo, args.config_name, args.token)

    # NEVER move the resume point backwards (reviewer-confirmed): on a
    # same-node requeue /scratch survives, so the local dir may hold a COMPLETE
    # checkpoint NEWER than anything on HF (its upload was cut by the kill).
    # Overwriting the pointer with the older HF step would re-train a whole
    # window and let the pusher upload a dir verl is concurrently rewriting.
    local_dir = Path(args.local_dir)
    local_best = None
    for d in local_dir.glob("global_step_*"):
        try:
            n = int(d.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        models = len(list(d.glob("actor/model_world_size_*_rank_*.pt")))
        extras = len(list(d.glob("actor/extra_state_world_size_*_rank_*.pt")))
        if models >= 4 and extras >= 4 and (local_best is None or n > local_best):
            local_best = n
    if local_best is not None and (step is None or step <= local_best):
        pointer = local_dir / "latest_checkpointed_iteration.txt"
        pointer.write_text(str(local_best))
        print(
            f"[resume] local global_step_{local_best} is >= HF ({step}); keeping local resume point"
        )
        return
    if step is None:
        return

    target = local_dir / f"global_step_{step}"
    # If a complete local copy already exists, don't re-download. One stray
    # shard must NOT count as present — require the full resume set.
    if (
        len(list(target.glob("actor/model_world_size_*_rank_*.pt"))) >= 4
        and len(list(target.glob("actor/extra_state_world_size_*_rank_*.pt"))) >= 4
    ):
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
