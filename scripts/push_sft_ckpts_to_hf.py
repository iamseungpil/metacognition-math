"""Watch an HF-Trainer output dir and stream new `checkpoint-N/` dirs to HuggingFace.

WHY THIS EXISTS (0727).  The SFT launchers push models to HF only AFTER
`accelerate launch` returns (a100g1_sft_b0p2_rvfull.yaml:129,133), so a node
death mid-training leaves NOTHING durable — one sidecar SFT2 run reached ~40/309
steps and vanished with its node, leaving models/b2p2_rvfull_sft with 0 files.
The RL launchers do not have this hole because they run
`nohup push_ckpts_to_hf.py --interval 90 &` CONCURRENTLY with training, which is
why rq3v2_b2p keeps landing gs120/gs140 on HF even while amlt reads "failed".
This daemon is that same structural fix for the SFT arms.

WHY NOT REUSE scripts/push_ckpts_to_hf.py.  That script's `checkpoint-*` branch
is a 10-line afterthought bolted onto a veRL uploader, and both of its
load-bearing assumptions are false for a DeepSpeed ZeRO-3 SFT:
  * "TRL saves are atomic; 5s suffices" (push_ckpts_to_hf.py:366-367) — here a
    save is a ZeRO-3 all-gather plus four ~4GB safetensors shards plus (unless
    save_only_model) a ~98GB CPU-offloaded Adam dump.  Pauses far exceed 5s, so
    a mid-save snapshot is near-certain.
  * "TRL checkpoint-N/ dirs are never pruned" (its --keep help text) — 1212
    steps / save_steps 25 = 48 checkpoints x ~16GB = ~790GB of uploads.
It also hardcodes repo_type="model" everywhere, uploads EVERY unseen
checkpoint-* dir rather than the newest, seeds `done` from HF only for veRL, and
passes no ignore_patterns, so it would try to push the ~98GB `global_step*/`
DeepSpeed optimizer subdir.  Rather than rewrite that branch (and risk the live
RL pusher), this is a standalone daemon.

WHAT IT GUARANTEES
  * Never uploads a half-written checkpoint — see `is_complete`.
  * Never aborts the loop on an exception; every failure is logged and retried
    on the next cycle.
  * Uploads the NEWEST complete checkpoint only, one per cycle, and marks a dir
    `done` only after HF has actually accepted the commit.  A skipped dir is
    never marked done — that exact bug froze HF at gs245 while training ran to
    gs295 (push_ckpts_to_hf.py:349-353).
  * Prunes older remote checkpoints down to --keep, never touching the one just
    uploaded.  Prune failures are swallowed.

RESUME SEMANTICS (read before choosing --save_only_model on the training side).
Under DeepSpeed there is NO weights-only resume: transformers 4.57.6
trainer.py:2501-2505 routes `resume_from_checkpoint` to
`deepspeed_load_checkpoint`, which globs `{ckpt}/global_step*` and raises
"Can't find a valid checkpoint" when absent (integrations/deepspeed.py:492,505).
Since the optimizer state is ~98GB (fp32 master + exp_avg + exp_avg_sq over
8.2B params), streaming it to HF every save is not physically viable.  So what
this daemon buys is a WARM RESTART tier: the ~16GB bf16 weights of the newest
checkpoint, from which a relaunch can set `model_name_or_path` to the downloaded
dir and continue with a fresh optimizer/LR schedule.  Bit-exact resume remains
possible only on a same-node restart, from the on-disk checkpoint.

Usage:
    python scripts/push_sft_ckpts_to_hf.py \
        --ckpt_dir /scratch/checkpoints/b0p2_rvfull_sft \
        --repo_id iamseungpil/metacot-sft2-b0p2-rvfull \
        --interval 60 --config_name b0p2_rvfull_sft --keep 2

The token comes from the HF_TOKEN environment variable ONLY.  There is
deliberately no --token flag: SECURITY 0716 — a launcher running under `set -x`
expands the argument into std_log, and the value is visible in `ps` output for
the daemon's whole lifetime.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

# `global_step*/` is the DeepSpeed optimizer dump (~98GB under ZeRO-3 + CPU
# offload).  It is excluded UNCONDITIONALLY — even when the training config sets
# save_only_model (so it should not exist), because the cost of being wrong is a
# 115GB push instead of a 16GB one.  The rng/optimizer/scheduler files are
# likewise resume-only state that a warm restart cannot consume.
UPLOAD_IGNORE_PATTERNS = [
    "global_step*",
    "global_step*/*",
    "latest",
    "zero_to_fp32.py",
    "rng_state*.pth",
    "optimizer.pt",
    "optimizer_*",
    "scheduler.pt",
    "scaler.pt",
    "*.tmp",
    "*.lock",
]

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def ckpt_step(name: str) -> int:
    """Step number of a ``checkpoint-N`` dir name, or -1 if it isn't one.

    -1 sorts oldest, so a stray directory can never be mistaken for the newest
    checkpoint.  (push_ckpts_to_hf.py:30 uses ``rsplit("_", 1)`` which returns
    -1 for EVERY ``checkpoint-N`` name — one reason its TRL branch never sorts.)
    """
    m = _CKPT_RE.match(name)
    return int(m.group(1)) if m else -1


def is_complete(ckpt: Path, min_quiet_s: float = 30.0, now: float | None = None) -> bool:
    """True iff `ckpt` is a fully-written HF-Trainer checkpoint, safe to upload.

    The anchor is `trainer_state.json`.  `Trainer._save_checkpoint`
    (transformers 4.57.6 trainer.py:3312-3370) writes it STRICTLY LAST — after
    `save_model` (3325) and after the optional optimizer/scaler/rng block
    (3343-3348) — so its presence means every weight-shard file handle is
    closed.  That is a structural ordering guarantee, not a timing heuristic,
    which is what makes it correct where the 5s quiescence gate is not.

    But `state.save_to_json` is a plain `f.write` with no atomic rename
    (trainer_callback.py:144-148), so a node that dies mid-write can leave a
    torn JSON.  Hence we PARSE it and cross-check `global_step` against the dir
    name, then verify every shard named by `model.safetensors.index.json`
    actually exists and is non-empty (the index is itself written after all
    shards, modeling_utils.py:4183-4192).

    `min_quiet_s` is a cheap backstop against exotic filesystem reordering, NOT
    the primary gate; pass 0 to disable it (tests, or a filesystem where mtimes
    are unreliable).
    """
    try:
        if not ckpt.is_dir():
            return False
        step = ckpt_step(ckpt.name)
        if step < 0:
            return False

        state_path = ckpt / "trainer_state.json"
        if not state_path.is_file():
            return False
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            return False  # torn write mid-save
        if not isinstance(state, dict):
            return False
        try:
            if int(state.get("global_step", -1)) != step:
                return False
        except (TypeError, ValueError):
            return False

        if not (ckpt / "config.json").is_file():
            return False

        index = ckpt / "model.safetensors.index.json"
        if index.is_file():
            try:
                weight_map = json.loads(index.read_text())["weight_map"]
                shards = set(weight_map.values())
            except Exception:
                return False
            if not shards:
                return False
            for shard in shards:
                p = ckpt / shard
                if not p.is_file() or p.stat().st_size == 0:
                    return False
        elif not (ckpt / "model.safetensors").is_file():
            return False  # neither sharded nor single-file weights landed

        if min_quiet_s > 0:
            newest = max(
                (p.stat().st_mtime for p in ckpt.rglob("*") if p.is_file()), default=0.0
            )
            if newest <= 0:
                return False
            if (now if now is not None else time.time()) - newest < min_quiet_s:
                return False
        return True
    except OSError:
        # Racing with the trainer's own writes (file vanished under rglob, etc.)
        # is a "not yet" answer, never a crash.
        return False


def repo_path(path_prefix: str, name: str) -> str:
    """Where a checkpoint dir lands in the repo. Empty prefix → repo root."""
    prefix = (path_prefix or "").strip("/")
    return f"{prefix}/{name}" if prefix else name


def remote_ckpt_names(files, path_prefix: str = "") -> set:
    """`checkpoint-N` dir names present under `path_prefix` in a repo file list.

    Used (a) to seed the done-set at startup, so a node that just downloaded a
    checkpoint for a warm restart never re-uploads those 16GB, and (b) to decide
    what to prune.  push_ckpts_to_hf.py has no equivalent for checkpoint-* dirs
    (its `_remote_complete_steps` matches only the veRL layout), which is why a
    restaged node there re-uploads what it just pulled.
    """
    prefix = (path_prefix or "").strip("/")
    pat = re.compile(
        (re.escape(prefix + "/") if prefix else "") + r"(checkpoint-\d+)/"
    )
    out = set()
    for f in files or []:
        m = pat.match(f)
        if m:
            out.add(m.group(1))
    return out


def select_prune_targets(remote_names, keep: int, latest: str | None = None) -> list:
    """Remote `checkpoint-N` names to delete, keeping the newest `keep`.

    `keep <= 0` disables pruning entirely (returns []), matching the --keep
    contract of the veRL pusher.  `latest` — the checkpoint just uploaded — is
    ALWAYS retained even if it somehow falls outside the newest `keep`, so a
    clock/parse anomaly can never delete the only durable copy of the current
    state.  Non-checkpoint names are ignored rather than deleted: this daemon
    only ever removes what it created.

    Returned oldest-first so a partial prune run always removes the least
    valuable checkpoints first.
    """
    if keep <= 0:
        return []
    names = sorted({n for n in remote_names if ckpt_step(n) >= 0}, key=ckpt_step, reverse=True)
    kept = set(names[:keep])
    if latest is not None:
        kept.add(latest)
    return sorted((n for n in names if n not in kept), key=ckpt_step)


def _prune_remote(api, repo_id: str, repo_type: str, path_prefix: str, keep: int, latest: str) -> None:
    """Delete old remote checkpoints. Best-effort: never raises to the caller.

    NOTE this performs HF deletes at runtime.  It is required — 48 checkpoints x
    ~16GB is ~790GB otherwise — and it is safe only because the target is a
    DEDICATED PER-RUN repo holding nothing but this run's checkpoints.  Do not
    point this daemon at a shared repo such as iamseungpil/metacot.
    """
    if keep <= 0:
        return
    try:
        files = api.list_repo_files(repo_id, repo_type=repo_type)
    except Exception as exc:
        print(f"[push-sft] prune list skip: {exc}", flush=True)
        return
    for name in select_prune_targets(remote_ckpt_names(files, path_prefix), keep, latest):
        try:
            api.delete_folder(
                path_in_repo=repo_path(path_prefix, name),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=f"prune {name} (keep newest {keep})",
            )
            print(f"[push-sft] pruned {name}", flush=True)
        except Exception as exc:
            print(f"[push-sft] prune {name} skip: {exc}", flush=True)


def _squash(api, repo_id: str, repo_type: str) -> None:
    """Collapse commit history so pruned 16GB LFS blobs stop billing storage.

    delete_folder only moves the tip; the blobs live on in history until a
    super-squash (0724 HF cleanup protocol). Best-effort.
    """
    try:
        api.super_squash_history(repo_id=repo_id, repo_type=repo_type)
        print(f"[push-sft] squashed history for {repo_id}", flush=True)
    except Exception as exc:
        print(f"[push-sft] squash skip: {exc}", flush=True)


def _load_marker(marker: Path) -> set:
    try:
        if marker.is_file():
            return set(json.loads(marker.read_text()))
    except Exception as exc:
        print(f"[push-sft] marker read skip: {exc}", flush=True)
    return set()


def _save_marker(marker: Path, done: set) -> None:
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(sorted(done)))
    except Exception as exc:
        print(f"[push-sft] marker write skip: {exc}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt_dir", required=True,
                    help="HF-Trainer output_dir holding checkpoint-N/ subdirs")
    ap.add_argument("--repo_id", required=True,
                    help="Dedicated PER-RUN HF repo, e.g. iamseungpil/metacot-sft2-b0p2-rvfull")
    ap.add_argument("--repo_type", default="model", choices=["model", "dataset"])
    ap.add_argument("--path_prefix", default="",
                    help="Path inside the repo to place checkpoint-N/ under (default: repo root)")
    ap.add_argument("--interval", type=int, default=60, help="Seconds between scans")
    ap.add_argument("--config_name", default="sft",
                    help="Names the /scratch/.pushed_sft_<config>.json dedup marker")
    ap.add_argument("--keep", type=int, default=2,
                    help="Keep only the newest N checkpoint-N/ dirs on the repo; "
                         "older ones are deleted after a successful upload. <=0 disables pruning.")
    ap.add_argument("--squash_every", type=int, default=10,
                    help="super_squash_history after every N uploads (<=0 disables)")
    ap.add_argument("--min_quiet_s", type=float, default=30.0,
                    help="Backstop: require this many seconds since the newest file mtime")
    ap.add_argument("--marker_dir", default="/scratch",
                    help="Where to keep the same-node dedup marker")
    ap.add_argument("--once", action="store_true",
                    help="Run a single scan and exit (smoke-testing the wiring)")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    # Env-only token (see module docstring). token=None additionally lets
    # huggingface_hub fall back to HUGGING_FACE_HUB_TOKEN / the cached login.
    token = os.environ.get("HF_TOKEN")

    try:
        import hf_transfer  # noqa: F401

        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    except ImportError:
        pass

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    try:
        create_repo(repo_id=args.repo_id, token=token, repo_type=args.repo_type, exist_ok=True)
    except Exception as exc:
        print(f"[push-sft] repo create skipped: {exc}", flush=True)

    marker = Path(args.marker_dir) / f".pushed_sft_{args.config_name}.json"
    done = _load_marker(marker)
    try:
        remote_seed = remote_ckpt_names(
            api.list_repo_files(args.repo_id, repo_type=args.repo_type), args.path_prefix
        )
    except Exception as exc:
        print(f"[push-sft] remote seed skip: {exc}", flush=True)
        remote_seed = set()
    done |= remote_seed

    ckpt_dir = Path(args.ckpt_dir)
    print(
        f"[push-sft] daemon start ckpt_dir={ckpt_dir} repo={args.repo_id}"
        f"({args.repo_type}) prefix={args.path_prefix or '<root>'} "
        f"interval={args.interval}s keep={args.keep} done={len(done)} "
        f"(remote-seeded {len(remote_seed)})",
        flush=True,
    )

    uploads = 0
    while True:
        try:
            cands = sorted(
                (d for d in ckpt_dir.glob("checkpoint-*") if d.is_dir() and ckpt_step(d.name) >= 0),
                key=lambda d: ckpt_step(d.name),
                reverse=True,
            )
            for d in cands:  # newest first
                if d.name in done:
                    break  # newest is already on HF → nothing to do this cycle
                if not is_complete(d, min_quiet_s=args.min_quiet_s):
                    # Mid-save, or torn. Fall through to the next-newest so a
                    # single bad dir never blocks durability. Crucially we do NOT
                    # mark it done — done means "verified committed to HF".
                    print(f"[push-sft] {d.name} not complete yet", flush=True)
                    continue
                dest = repo_path(args.path_prefix, d.name)
                print(f"[push-sft] uploading {d.name} → {args.repo_id}:{dest}", flush=True)
                t0 = time.time()
                # upload_folder is a single create_commit in huggingface_hub
                # 0.36.2, so the remote state is atomic all-or-nothing: an
                # interrupted upload leaves nothing and simply retries next
                # cycle (already-hashed LFS blobs dedup, so the retry is cheap).
                api.upload_folder(
                    folder_path=str(d),
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    path_in_repo=dest,
                    commit_message=f"sft ckpt {d.name} ({args.config_name})",
                    ignore_patterns=UPLOAD_IGNORE_PATTERNS,
                )
                done.add(d.name)
                _save_marker(marker, done)
                uploads += 1
                print(f"[push-sft] done {d.name} in {time.time() - t0:.0f}s", flush=True)

                _prune_remote(api, args.repo_id, args.repo_type, args.path_prefix,
                              args.keep, d.name)
                if args.squash_every > 0 and uploads % args.squash_every == 0:
                    _squash(api, args.repo_id, args.repo_type)
                break  # one upload per cycle; re-scan for anything newer
        except Exception as exc:
            # The whole point of the daemon is that it outlives every transient
            # HF/network/filesystem failure. Nothing here may propagate.
            print(f"[push-sft] err: {exc}", flush=True)

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
