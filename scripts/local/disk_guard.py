#!/usr/bin/env python3
"""scripts/local/disk_guard.py prune [--apply]

For every lineage that has at least one job in /hdd_data/seungpil/queue/done/
(i.e. it "finished" per the queue, whether or not every arm step succeeded),
prune $WORK/checkpoints/<lineage>/global_step_*/ down to:
    - the 2 newest global_step_N dirs, PLUS
    - every global_step_N whose merged output already exists at
      $WORK/merged/<lineage>/step_N/ (config.json present) — those are safe to
      delete the raw FSDP shards from, but we still keep the DIRECTORY (with
      the shards already pruned by run_arm.sh) rather than removing the whole
      global_step_N/ tree, since extra_state may still be useful for exact
      resume bookkeeping. What this script actually deletes on --apply is
      whole global_step_N/ directories that are BOTH old (not in the newest-2)
      AND already merged.

Dry-run by default (prints what WOULD be deleted); --apply actually deletes.

"lineage" here = the basename of a $WORK/checkpoints/<lineage> directory. We
derive the set of finished lineages from queue/done/*.json's "name" field
matching a leading "cd7_..." job name convention used by run_arm.sh-launched
jobs (gpu_queue.py submit --name <name>): a job is considered to name a
lineage if its "name" field equals a directory name under $WORK/checkpoints.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

QUEUE_ROOT = Path("/hdd_data/seungpil/queue")
WORK = Path("/hdd_data/seungpil/scratch")
CKPT_ROOT = WORK / "checkpoints"
MERGED_ROOT = WORK / "merged"

STEP_RE = re.compile(r"^global_step_(\d+)$")


def _finished_lineages() -> set[str]:
    done_names = set()
    done_dir = QUEUE_ROOT / "done"
    if not done_dir.exists():
        return done_names
    for f in done_dir.glob("*.json"):
        try:
            job = json.loads(f.read_text())
        except Exception:
            continue
        name = job.get("name")
        if name:
            done_names.add(name)

    # A lineage counts as finished if ITS name (or a job named after it)
    # appears in done/, OR if the checkpoint dir simply exists and has no
    # sibling job still pending/running (best-effort; queue naming is by
    # convention, not enforced elsewhere in this codebase).
    lineages = set()
    if not CKPT_ROOT.exists():
        return lineages
    for d in CKPT_ROOT.iterdir():
        if not d.is_dir():
            continue
        if d.name in done_names:
            lineages.add(d.name)
    return lineages


def _step_dirs(lineage_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for d in lineage_dir.iterdir():
        if not d.is_dir():
            continue
        m = STEP_RE.match(d.name)
        if m:
            out.append((int(m.group(1)), d))
    out.sort(key=lambda t: t[0])
    return out


def _is_merged(lineage: str, step: int) -> bool:
    return (MERGED_ROOT / lineage / f"step_{step}" / "config.json").is_file()


def cmd_prune(args: argparse.Namespace) -> int:
    lineages = _finished_lineages()
    if not lineages:
        print("[disk_guard] no finished lineages found in queue/done/ matching a "
              f"checkpoints/ dir under {CKPT_ROOT} — nothing to do.")
        return 0

    total_to_delete = []
    for lineage in sorted(lineages):
        lineage_dir = CKPT_ROOT / lineage
        steps = _step_dirs(lineage_dir)
        if len(steps) <= 2:
            print(f"[disk_guard] {lineage}: {len(steps)} step dir(s), <= 2, keeping all")
            continue
        newest2 = {s for s, _ in steps[-2:]}
        for step, path in steps:
            if step in newest2:
                continue
            if _is_merged(lineage, step):
                total_to_delete.append((lineage, step, path))
            else:
                print(f"[disk_guard] {lineage}: step {step} is old but NOT merged yet — keeping "
                      f"(no merged/{lineage}/step_{step}/config.json)")

    if not total_to_delete:
        print("[disk_guard] nothing eligible for pruning.")
        return 0

    print(f"[disk_guard] {'DELETING' if args.apply else 'WOULD DELETE'} "
          f"{len(total_to_delete)} global_step_* dir(s):")
    for lineage, step, path in total_to_delete:
        size_note = ""
        try:
            size_note = f" (~{sum(f.stat().st_size for f in path.rglob('*') if f.is_file())/1e9:.1f}GB)"
        except OSError:
            pass
        print(f"  {lineage} step={step}: {path}{size_note}")
        if args.apply:
            shutil.rmtree(path)

    if not args.apply:
        print("[disk_guard] dry-run only — pass --apply to actually delete.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sp = sub.add_parser("prune")
    sp.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    sp.set_defaults(func=cmd_prune)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
