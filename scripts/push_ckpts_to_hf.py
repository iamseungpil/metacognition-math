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

    COMPLETENESS-AWARE: per-file uploads mean PARTIAL step dirs can exist on the
    repo. Counting those toward ``keep`` could evict the only complete
    checkpoint (e.g. [gs260 partial, gs255 partial, gs250 complete] with keep=2
    would delete gs250 — the sole resumable state). So: keep the newest ``keep``
    COMPLETE checkpoints, plus any step NEWER than the newest kept one (a
    possibly in-flight upload); delete older completes AND stale partials.
    The just-uploaded ``latest_name`` is always preserved. Best-effort: any
    failure is logged and swallowed so the daemon loop never aborts on a prune
    error. Only veRL ``global_step_*`` dirs are considered; TRL ``checkpoint-*``
    dirs are left untouched.
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

    complete = _remote_complete_steps(api, repo_id, config_name)
    complete_sorted = sorted((n for n in step_names if n in complete), key=_step_num, reverse=True)
    kept_complete = set(complete_sorted[:keep])
    newest_kept = max((_step_num(n) for n in kept_complete), default=-1)
    to_delete = [
        n
        for n in step_names
        if n not in kept_complete and _step_num(n) <= newest_kept
    ]
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


def _list_repo_files_retry(api, repo_id: str, attempts: int = 3):
    """list_repo_files with retries; returns None (not []) when all attempts fail
    so callers can distinguish 'HF unreachable' from 'repo empty'."""
    for i in range(1, attempts + 1):
        try:
            return api.list_repo_files(repo_id, repo_type="model")
        except Exception as exc:
            print(f"[push] repo list attempt {i}/{attempts} failed: {exc}")
            if i < attempts:
                time.sleep(30)
    return None


def _remote_complete_steps(api, repo_id: str, config_name: str, files=None) -> set[str]:
    """Return names of ``global_step_*`` dirs already COMPLETE on the HF repo.

    Complete = >=4 shards of EACH of ``actor/model_world_size_*_rank_*.pt``,
    ``actor/extra_state_world_size_*_rank_*.pt`` AND
    ``actor/optim_world_size_*_rank_*.pt``. verl's load_contents mirrors
    save_contents=[model, optimizer, extra]
    (configs/verl_sdc_e21r_shared.yaml:42-44), so a step missing ANY of the
    three shard sets crashes resume — counting only model shards would let the
    5s quiescence gate freeze an unresumable step as the authoritative resume
    point (reviewer-confirmed failure mode). Real incident from the earlier
    optim-blind version: rq3_b2 gs150 was marked done with optim 3/4 shards on
    HF, permanently freezing an unresumable step and putting prune at risk of
    deleting the last actually-complete checkpoint. Used (a) to seed the
    done-set at startup so a resumed node never re-uploads the checkpoint it
    just pulled, and (b) to verify an upload before marking it done — done now
    means "verified on HF", never "skipped".
    """
    import re
    from collections import Counter

    if files is None:
        files = _list_repo_files_retry(api, repo_id)
    if files is None:
        return set()
    pat = re.compile(
        rf"checkpoints/{re.escape(config_name)}/(global_step_\d+)/actor/"
        rf"(model|extra_state|optim)_world_size_\d+_rank_\d+\.pt$"
    )
    model_counts: Counter = Counter()
    extra_counts: Counter = Counter()
    optim_counts: Counter = Counter()
    for f in files:
        m = pat.match(f)
        if m:
            {"model": model_counts, "extra_state": extra_counts, "optim": optim_counts}[
                m.group(2)
            ][m.group(1)] += 1
    return {
        name
        for name, n in model_counts.items()
        if n >= 4 and extra_counts[name] >= 4 and optim_counts[name] >= 4
    }


def _local_ckpt_files(step_dir: Path) -> list:
    """Enumerate a checkpoint dir's files smallest-first (skip tmp/lock)."""
    return sorted(
        (p for p in step_dir.rglob("*") if p.is_file() and not p.name.endswith((".tmp", ".lock"))),
        key=lambda p: p.stat().st_size,
    )


def _upload_step_dir(api, step_dir: Path, repo_id: str, path_in_repo: str, first_attempt: bool) -> bool:
    """Upload one veRL checkpoint dir file-by-file, durably and verifiably.

    A single 16GB upload_folder call is all-or-nothing: one network hiccup hours
    in restarts the whole transfer, and under a ~6h preemption window nothing
    ever lands (observed 07-05: HF frozen at gs245 across three fragments that
    each reached gs291-294; wandb commit gaps show single upload attempts in
    flight for 2.5-2.8h before failing). Per-file commits make progress durable
    at ~4GB granularity. Hardening (reviewer-confirmed failure modes):
    - On the FIRST attempt this session, any remote partial of this step is
      deleted before uploading — shards from a previous fragment's cut-off
      upload must never interleave with this run's shards into a
      mixed-lineage dir that passes the completeness count.
    - Files already present remotely (from THIS session's earlier retry) are
      skipped, so a retry doesn't re-hash 4GB shards.
    - After uploading, the local dir is re-enumerated and any files written
      AFTER the initial snapshot (verl saves are multi-file and non-atomic)
      are uploaded too; success additionally requires the remote file set to
      be a SUPERSET of the final local set.
    """
    files = _local_ckpt_files(step_dir)
    if not files:
        return False

    remote_all = _list_repo_files_retry(api, repo_id)
    if remote_all is None:
        return False
    prefix = f"{path_in_repo}/"
    remote_rels = {f[len(prefix):] for f in remote_all if f.startswith(prefix)}
    if first_attempt and remote_rels and step_dir.name not in _remote_complete_steps(
        api, repo_id, path_in_repo.rsplit("/", 2)[-2], files=remote_all
    ):
        try:
            api.delete_folder(
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"clear foreign partial {step_dir.name}",
            )
            remote_rels = set()
            print(f"[push] cleared foreign partial {path_in_repo}", flush=True)
        except Exception as exc:
            print(f"[push] partial clear failed ({exc}); aborting this cycle", flush=True)
            return False

    for _pass in range(2):  # pass 2 catches files written after the first snapshot
        for p in files:
            rel = str(p.relative_to(step_dir))
            if rel in remote_rels:
                continue
            size_gb = p.stat().st_size / 1e9
            for attempt in range(1, 4):
                t0 = time.time()
                try:
                    api.upload_file(
                        path_or_fileobj=str(p),
                        path_in_repo=f"{path_in_repo}/{rel}",
                        repo_id=repo_id,
                        repo_type="model",
                        commit_message=f"ckpt {step_dir.name} {rel}",
                    )
                    remote_rels.add(rel)
                    print(f"[push]   {rel} ({size_gb:.1f}GB) ok in {time.time() - t0:.0f}s", flush=True)
                    break
                except Exception as exc:
                    print(
                        f"[push]   {rel} attempt {attempt}/3 failed after {time.time() - t0:.0f}s: {exc}",
                        flush=True,
                    )
                    time.sleep(30)
            else:
                return False
        files = _local_ckpt_files(step_dir)
        missing = [p for p in files if str(p.relative_to(step_dir)) not in remote_rels]
        if not missing:
            break

    local_rels = {str(p.relative_to(step_dir)) for p in _local_ckpt_files(step_dir)}
    if not local_rels <= remote_rels:
        print(f"[push] superset check failed, missing: {sorted(local_rels - remote_rels)[:5]}", flush=True)
        return False
    return True


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
    # CONTRACT 0717: launchers no longer pass --token (SECURITY 0716 — set -x would
    # leak the expanded value into std_log). Fall back to the HF_TOKEN env var;
    # token=None additionally lets huggingface_hub auto-detect HUGGING_FACE_HUB_TOKEN.
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
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

    # hf_transfer (rust multi-stream uploader) is a big speedup for the ~4GB
    # shards when present in the env; harmless opt-in, never a hard dependency.
    try:
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass

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
    # Markers written by the OLD daemon done-marked skipped (never-uploaded)
    # veRL ckpts — trusting them on a same-node requeue reproduces the gs245
    # freeze. veRL done-ness is re-derived from HF below; only TRL
    # checkpoint-* entries (uploaded atomically, never skipped) are kept.
    done = {n for n in done if not n.startswith("global_step_")}

    # Seed done from HF: ckpts already complete on the repo (e.g. the one this
    # node just pulled for resume) must never be re-uploaded.
    remote_done = _remote_complete_steps(api, args.repo_id, args.config_name)
    done |= remote_done

    ckpt_dir = Path(args.ckpt_dir)
    wandb_run_dir = Path("/scratch/metacognition/wandb")

    print(
        f"[push] daemon start ckpt_dir={ckpt_dir} repo={args.repo_id} "
        f"interval={args.interval}s done={len(done)} (remote-seeded {len(remote_done)})"
    )

    uploads_since_squash = 0
    attempted: set = set()  # steps whose upload THIS session already started
    loop_i = 0
    while True:
        loop_i += 1
        try:
            if ckpt_dir.exists():
                # Match BOTH veRL (global_step_N/) AND TRL (checkpoint-N/) save formats.
                _all = list(ckpt_dir.glob("global_step_*")) + list(ckpt_dir.glob("checkpoint-*"))
                _gs = sorted(
                    [d for d in _all if d.is_dir() and d.name.startswith("global_step_")],
                    key=lambda x: _step_num(x.name),
                )
                _trl = [d for d in _all if d.is_dir() and d.name.startswith("checkpoint-")]
                # DURABILITY: only the newest veRL global_step_* dir matters for
                # resume, so older ones are simply never attempted — but they are
                # NOT marked done. (The old logic done-marked the skipped backlog,
                # which combined with failing uploads froze HF at gs245 while
                # training reached gs295.) done now strictly means "verified on HF".
                if _gs:
                    newest = _gs[-1]
                    ckpt_candidates = ([newest] if newest.name not in done else []) + _trl
                else:
                    ckpt_candidates = _trl
                for step_dir in ckpt_candidates:
                    if not step_dir.is_dir() or step_dir.name in done:
                        continue
                    # Skip if still being written. veRL saves are MULTI-FILE and
                    # non-atomic over tens of seconds (rank-parallel shards, then
                    # extra_state/data.pt) — a short gate can snapshot mid-save
                    # and freeze an unresumable step (reviewer-confirmed), so
                    # global_step_* requires 60s of quiescence. TRL saves are
                    # atomic; 5s suffices there.
                    quiet = 60 if step_dir.name.startswith("global_step_") else 5
                    latest_mtime = max((p.stat().st_mtime for p in step_dir.rglob("*") if p.is_file()), default=0)
                    if latest_mtime == 0 or time.time() - latest_mtime < quiet:
                        continue

                    # TRL ckpts → checkpoint-N/ at repo root (resume yaml expects this).
                    # veRL ckpts → checkpoints/<config>/global_step_N/ (legacy SDC).
                    if step_dir.name.startswith("checkpoint-"):
                        path_in_repo = step_dir.name
                        print(f"[push] uploading {step_dir.name} → {args.repo_id}:{path_in_repo}")
                        api.upload_folder(
                            folder_path=str(step_dir),
                            repo_id=args.repo_id,
                            path_in_repo=path_in_repo,
                            repo_type="model",
                            commit_message=f"ckpt {step_dir.name} ({args.config_name})",
                            ignore_patterns=["*.tmp", "*.lock"],
                        )
                    else:
                        path_in_repo = f"checkpoints/{args.config_name}/{step_dir.name}"
                        print(f"[push] uploading {step_dir.name} → {args.repo_id}:{path_in_repo}")
                        first = step_dir.name not in attempted
                        attempted.add(step_dir.name)
                        ok = _upload_step_dir(api, step_dir, args.repo_id, path_in_repo, first)
                        if not ok or step_dir.name not in _remote_complete_steps(
                            api, args.repo_id, args.config_name
                        ):
                            print(f"[push] {step_dir.name} INCOMPLETE — will retry next cycle")
                            continue
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

            # Push wandb run files (best-effort, overwrites). Throttled to every
            # 10th cycle — at interval 90s an unconditional push stacks ~40
            # commits/hour on top of ckpt commits and risks HF 429 throttling.
            if args.include_wandb and wandb_run_dir.exists() and loop_i % 10 == 1:
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
