#!/usr/bin/env python3
r"""Freeze everything about a matched-arm run BEFORE its results are looked at.

WHY THIS EXISTS
---------------
Asked what would most likely fool us here, both external reviews gave the same
answer: a change that reaches ONE ARM ONLY. It does not announce itself. Every
reward curve still looks plausible, the job still boots, and the result -
positive or negative - is about the drift rather than about the mechanism.

That is not hypothetical in this repo. Within a single day:
  * the ladder was asymmetric (two supervised stages on the meta side, one on
    the control side) for long enough that it could have produced a headline;
  * an OOM fix diagnosed on one arm was never ported to its twin, and the twin
    died of that exact OOM for nine days;
  * a durability fix landed in launchers that were retired hours later, leaving
    the surviving path unprotected;
  * a save cadence differed between arms with nothing declaring it.

The second recorded failure mode is choosing the judgement rule after seeing the
numbers. The grader is not neutral here: on the same eval, requiring \boxed{}
flips one axis from +8.2pp to -0.97pp. Which grader counts has to be a
pre-launch commitment.

WHAT THIS DOES
--------------
Collects, into one JSON, the things that must be identical (or explicitly
different) across arms, and the things that must be fixed before results exist:

  data      row-count and content hash of each arm's corpus, plus the token
            length distribution the tokenizer actually produces - because equal
            batch size does NOT imply equal token count when one arm's corpus
            carries meta blocks the other does not.
  config    the resolved training config per arm, and a diff of the two.
  code      git commit, dirty flag, and content hashes of the files that decide
            the objective.
  grader    module hash + the declared judgement mode, so the rule cannot be
            swapped after the fact.
  eval      the exact item ids the comparison will be made over.

It does NOT decide whether the run is valid. It records what was true, so that
`--compare` against a later manifest can show what moved.

USAGE
    python scripts/freeze_run_manifest.py --stage sft2 \
        --arm control:configs/sft_b0p2_rvfull.yaml \
        --arm meta:configs/sft_b2p2_rvfull.yaml \
        --grader-mode format_fair \
        --out docs/manifests/sft2_$(date -u +%Y%m%dT%H%M).json

    python scripts/freeze_run_manifest.py --compare a.json b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files whose content decides the objective. A change here changes what is being
# optimised, so their hashes belong in the manifest even when no config moved.
OBJECTIVE_FILES = [
    "src/training/sft.py",
    "src/training/dcpo_region.py",
    "src/training/dcpo_pmi_shift.py",
    "src/training/rewards.py",
    "src/training/verl_sdc.py",
    "experiments/analysis/analysis_common.py",
]

GRADER_MODES = {
    # The fallback in analysis_common.Grader.grade: when a completion carries no
    # \boxed{}, grade the runtime-extracted answer through the same math_verify
    # path. Uniform across arms; exists because arms box at different rates.
    "format_fair": "boxed if present, else the runtime answer_extracted, both via robust_grade",
    # No fallback: a completion without \boxed{} is wrong. Stricter, and it moves
    # the priming axis from +8.2pp to -0.97pp on the same data.
    "strict_boxed": "requires \\boxed{}; no fallback",
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def load_yaml(path: Path):
    import yaml
    return yaml.safe_load(path.read_text())


def corpus_fingerprint(parquet: Path, tokenizer_dir: str | None):
    """Row count, content hash, and - if a tokenizer is reachable - the token
    length distribution. The distribution is the part that matters: it is how a
    'same batch size' claim gets falsified."""
    out: dict = {"path": str(parquet), "exists": parquet.exists()}
    if not parquet.exists():
        return out
    import pandas as pd

    df = pd.read_parquet(parquet)
    out["rows"] = int(len(df))
    out["columns"] = sorted(map(str, df.columns))
    # Hash the message content only, so an index or dtype change does not look
    # like a data change.
    col = "messages" if "messages" in df.columns else df.columns[0]
    h = hashlib.sha256()
    for v in df[col].astype(str):
        h.update(v.encode("utf-8", "replace"))
    out["content_sha256"] = h.hexdigest()
    if "scenario" in df.columns:
        out["scenario_counts"] = {str(k): int(v) for k, v in df["scenario"].value_counts().items()}

    if tokenizer_dir:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
            lens = []
            for v in df[col].astype(str):
                lens.append(len(tok(v, add_special_tokens=False)["input_ids"]))
            lens.sort()
            n = len(lens)
            out["token_len"] = {
                "mean": sum(lens) / n,
                "p50": lens[n // 2],
                "p90": lens[int(n * 0.9)],
                "max": lens[-1],
                "total": sum(lens),
            }
        except Exception as exc:  # tokenizer not staged locally is the normal case
            out["token_len_error"] = repr(exc)[:200]
    return out


def freeze(args) -> dict:
    manifest: dict = {
        "stage": args.stage,
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "objective_files": {
            f: sha256_file(ROOT / f) for f in OBJECTIVE_FILES
        },
        "grader": {
            "mode": args.grader_mode,
            "meaning": GRADER_MODES[args.grader_mode],
            "module_sha256": sha256_file(ROOT / "experiments/analysis/analysis_common.py"),
        },
        "arms": {},
    }

    for spec in args.arm:
        name, _, cfg_path = spec.partition(":")
        cfg_file = ROOT / cfg_path
        cfg = load_yaml(cfg_file) if cfg_file.exists() else None
        arm: dict = {"config_path": cfg_path, "config_sha256": sha256_file(cfg_file), "config": cfg}
        if cfg:
            data = cfg.get("dataset_path")
            if data:
                arm["corpus"] = corpus_fingerprint(ROOT / data, args.tokenizer)
            bs = cfg.get("per_device_train_batch_size")
            ga = cfg.get("gradient_accumulation_steps")
            if bs and ga:
                arm["effective_batch_per_process"] = bs * ga
        manifest["arms"][name] = arm

    if args.eval_items:
        p = ROOT / args.eval_items
        manifest["eval_items"] = {
            "path": args.eval_items,
            "sha256": sha256_file(p),
            "count": sum(1 for _ in p.open()) if p.exists() else None,
        }
    return manifest


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(d, list):
        out[prefix] = json.dumps(d, sort_keys=True)
    else:
        out[prefix] = d
    return out


def arm_symmetry(manifest: dict) -> list[str]:
    """Report every config key that differs between arms. Differences are not
    errors - the treatment IS a difference - but each one has to be intended."""
    arms = manifest.get("arms", {})
    if len(arms) != 2:
        return [f"expected exactly 2 arms, found {len(arms)}"]
    (n1, a1), (n2, a2) = arms.items()
    f1, f2 = _flatten(a1.get("config") or {}), _flatten(a2.get("config") or {})
    lines = []
    for k in sorted(set(f1) | set(f2)):
        if f1.get(k) != f2.get(k):
            lines.append(f"{k}: {n1}={f1.get(k)!r}  {n2}={f2.get(k)!r}")
    return lines


def compare(a_path: str, b_path: str) -> int:
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    fa, fb = _flatten(a), _flatten(b)
    moved = [k for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    if not moved:
        print("identical")
        return 0
    print(f"{len(moved)} field(s) moved:")
    for k in moved:
        print(f"  {k}\n    before: {fa.get(k)!r}\n    after : {fb.get(k)!r}")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="sft2")
    ap.add_argument("--arm", action="append", default=[],
                    help="name:config_path, e.g. control:configs/sft_b0p2_rvfull.yaml")
    ap.add_argument("--grader-mode", choices=sorted(GRADER_MODES), default="format_fair")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer dir; without it the token-length distribution is skipped")
    ap.add_argument("--eval-items", default=None, help="file listing the item ids to be compared")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        sys.exit(compare(*args.compare))

    if not args.arm:
        ap.error("--arm is required (give it twice, once per arm)")

    manifest = freeze(args)
    diffs = arm_symmetry(manifest)
    manifest["arm_config_differences"] = diffs

    text = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        print(f"wrote {out}")
    else:
        print(text)

    print(f"\ngrader frozen as: {args.grader_mode} — {GRADER_MODES[args.grader_mode]}")
    print(f"config keys differing between arms: {len(diffs)}")
    for d in diffs:
        print(f"  {d}")
    print("\nEvery line above must be an INTENDED difference. One that is not is the"
          "\nfailure this file exists to catch.")


if __name__ == "__main__":
    main()
