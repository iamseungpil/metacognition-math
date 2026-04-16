#!/usr/bin/env python3
"""Meta-CoT Autoresearch Loop: iterate hypotheses until Meta-CoT >= Base SFT.

This orchestrator runs hypotheses in priority order, evaluating after each one.
It stops as soon as Meta-CoT accuracy meets or exceeds Base SFT, or when all
hypotheses are exhausted.

Hypotheses (priority order):
  H1: max_tokens=4096 in eval (truncation was 31% of errors)
  H2: Difficulty-adaptive meta (fewer meta blocks for easy problems)
  H3: Verification-only meta (only post-solution verification, not pre/mid)
  H4: Better training data from GPT-5.4 (currently GPT-5.4-mini)
  H5: Longer GRPO training (1000 steps instead of 500)
  H6: Stepwise verification reward (Agent Lightning style)

Usage:
  python scripts/autoresearch_loop.py
  python scripts/autoresearch_loop.py --dry-run
  python scripts/autoresearch_loop.py --hypotheses H1 H3 H5
  python scripts/autoresearch_loop.py --max-iterations 3
  python scripts/autoresearch_loop.py --base-accuracy 0.767  # override known baseline
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Known baseline from previous eval: Base SFT on MATH = 76.7%
# This can be overridden by --base-accuracy or by running a fresh baseline eval
DEFAULT_BASE_ACCURACY = 0.767

RESULTS_DIR = PROJECT_ROOT / "results" / "autoresearch"
LOG_PATH = RESULTS_DIR / "autoresearch_log.json"


@dataclass
class HypothesisConfig:
    """Configuration for a single hypothesis experiment."""
    id: str
    name: str
    description: str
    command: List[str]  # Shell command to run
    requires_training: bool  # Whether this hypothesis requires GPU training
    estimated_time_hours: float  # Rough estimate for planning
    prerequisites: List[str] = field(default_factory=list)  # Other hypothesis IDs


HYPOTHESES = [
    HypothesisConfig(
        id="H1",
        name="Fix Truncation (max_tokens=4096)",
        description=(
            "Previous eval used max_tokens=2048, causing 31% of errors from truncation. "
            "eval_hf.py already defaults to max_tokens=4096. This is a verification run."
        ),
        command=["bash", "scripts/autoresearch.sh", "H1"],
        requires_training=False,
        estimated_time_hours=2.0,
    ),
    HypothesisConfig(
        id="H3",
        name="Verification-Only Meta",
        description=(
            "Strip pre-solve and mid-solve meta blocks, keeping only the final "
            "verification/confidence check before \\boxed{}. Reduces meta token "
            "overhead from ~56% to ~15% while preserving calibration."
        ),
        command=["bash", "scripts/autoresearch.sh", "H3"],
        requires_training=True,
        estimated_time_hours=12.0,
    ),
    HypothesisConfig(
        id="H5",
        name="Extended GRPO Training (1000 steps)",
        description=(
            "Continue GRPO training from checkpoint-200 for 1000 total steps. "
            "More RL training may help the model learn to use meta blocks "
            "without sacrificing accuracy."
        ),
        command=["bash", "scripts/autoresearch.sh", "H5", "--grpo-steps", "1000"],
        requires_training=True,
        estimated_time_hours=8.0,
        prerequisites=["H1"],  # Want to verify H1 first (cheap check)
    ),
]


@dataclass
class HypothesisResult:
    """Result from running a single hypothesis."""
    hypothesis_id: str
    hypothesis_name: str
    status: str  # "pass", "fail", "error", "skipped"
    meta_accuracy: Optional[float] = None
    base_accuracy: Optional[float] = None
    delta: Optional[float] = None
    error_message: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    eval_file: Optional[str] = None
    log_file: Optional[str] = None


@dataclass
class AutoresearchLog:
    """Full log of the autoresearch loop."""
    start_time: str
    end_time: Optional[str] = None
    target_metric: str = "Meta-CoT accuracy >= Base SFT accuracy"
    base_accuracy: Optional[float] = None
    success: bool = False
    winning_hypothesis: Optional[str] = None
    results: List[dict] = field(default_factory=list)
    total_iterations: int = 0
    total_duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def find_eval_results(results_dir: Path, pattern: str) -> Optional[Path]:
    """Find the most recent eval result file matching a pattern."""
    candidates = sorted(results_dir.glob(f"eval_{pattern}*.json"), reverse=True)
    return candidates[0] if candidates else None


def extract_accuracy_from_eval(eval_path: Path) -> Optional[float]:
    """Extract overall accuracy from an eval JSON file."""
    if not eval_path.exists():
        return None
    with open(eval_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return None
    correct = sum(1 for r in results if r.get("is_correct"))
    return correct / len(results) if results else None


def extract_accuracy_per_benchmark(eval_path: Path) -> dict:
    """Extract per-benchmark accuracy from an eval JSON file."""
    if not eval_path.exists():
        return {}
    with open(eval_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        return {}

    benchmarks = {}
    for r in results:
        bench = r.get("benchmark", "unknown")
        if bench not in benchmarks:
            benchmarks[bench] = {"correct": 0, "total": 0}
        benchmarks[bench]["total"] += 1
        if r.get("is_correct"):
            benchmarks[bench]["correct"] += 1

    return {
        bench: vals["correct"] / vals["total"] if vals["total"] > 0 else 0.0
        for bench, vals in benchmarks.items()
    }


def run_hypothesis(
    hypothesis: HypothesisConfig,
    dry_run: bool = False,
    base_accuracy: float = DEFAULT_BASE_ACCURACY,
) -> HypothesisResult:
    """Run a single hypothesis experiment."""

    result = HypothesisResult(
        hypothesis_id=hypothesis.id,
        hypothesis_name=hypothesis.name,
        status="running",
        start_time=datetime.now().isoformat(),
    )

    print(f"\n{'='*60}")
    print(f"  Running: {hypothesis.id} — {hypothesis.name}")
    print(f"  Description: {hypothesis.description}")
    print(f"  Estimated time: {hypothesis.estimated_time_hours:.1f}h")
    print(f"  Requires training: {hypothesis.requires_training}")
    print(f"{'='*60}\n")

    if dry_run:
        print(f"  [DRY-RUN] Would execute: {' '.join(hypothesis.command)}")
        result.status = "skipped"
        result.end_time = datetime.now().isoformat()
        result.duration_seconds = 0.0
        return result

    start_time = time.time()

    try:
        # Build command with dry-run stripped (we handle it at this level)
        cmd = hypothesis.command.copy()

        # Run the experiment
        log_file = RESULTS_DIR / f"run_{hypothesis.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        result.log_file = str(log_file)

        print(f"  Command: {' '.join(cmd)}")
        print(f"  Log: {log_file}")

        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=hypothesis.estimated_time_hours * 3600 * 1.5,  # 1.5x estimated as timeout
        )

        # Write log
        with open(log_file, "w") as f:
            f.write(f"=== {hypothesis.id}: {hypothesis.name} ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Exit code: {proc.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(proc.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(proc.stderr)

        # Print output tail
        stdout_lines = proc.stdout.strip().split("\n")
        print("\n  --- Output (last 20 lines) ---")
        for line in stdout_lines[-20:]:
            print(f"  {line}")

        if proc.returncode == 0:
            result.status = "pass"
        else:
            result.status = "fail"

    except subprocess.TimeoutExpired:
        result.status = "error"
        result.error_message = f"Timeout after {hypothesis.estimated_time_hours * 1.5:.1f}h"
        print(f"  ERROR: {result.error_message}")
    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
        print(f"  ERROR: {result.error_message}")

    end_time = time.time()
    result.end_time = datetime.now().isoformat()
    result.duration_seconds = end_time - start_time

    # Try to extract accuracy from the most recent eval files
    eval_files = sorted(RESULTS_DIR.glob(f"eval_{hypothesis.id.lower()}*.json"), reverse=True)
    if eval_files:
        meta_acc = extract_accuracy_from_eval(eval_files[0])
        result.meta_accuracy = meta_acc
        result.eval_file = str(eval_files[0])

        # Look for corresponding base eval
        base_files = sorted(
            RESULTS_DIR.glob(f"eval_{hypothesis.id.lower()}_base*.json"), reverse=True
        )
        if base_files:
            base_acc = extract_accuracy_from_eval(base_files[0])
            result.base_accuracy = base_acc
        else:
            result.base_accuracy = base_accuracy

        if result.meta_accuracy is not None and result.base_accuracy is not None:
            result.delta = result.meta_accuracy - result.base_accuracy
            if result.meta_accuracy >= result.base_accuracy:
                result.status = "pass"
            else:
                result.status = "fail"

    duration_str = format_duration(result.duration_seconds)
    print(f"\n  Result: {result.status.upper()}")
    if result.meta_accuracy is not None:
        print(f"  Meta-CoT accuracy: {result.meta_accuracy:.1%}")
    if result.base_accuracy is not None:
        print(f"  Base SFT accuracy: {result.base_accuracy:.1%}")
    if result.delta is not None:
        sign = "+" if result.delta >= 0 else ""
        print(f"  Delta: {sign}{result.delta:.1%}")
    print(f"  Duration: {duration_str}")

    return result


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def save_log(log: AutoresearchLog):
    """Save autoresearch log to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(asdict(log), f, indent=2, ensure_ascii=False, default=str)
    print(f"\nLog saved to {LOG_PATH}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Meta-CoT Autoresearch Loop: iterate hypotheses until Meta-CoT >= Base SFT"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without executing"
    )
    parser.add_argument(
        "--hypotheses", nargs="+", default=None,
        help="Specific hypothesis IDs to run (default: all in priority order)"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Maximum number of hypotheses to try"
    )
    parser.add_argument(
        "--base-accuracy", type=float, default=DEFAULT_BASE_ACCURACY,
        help=f"Base SFT accuracy to beat (default: {DEFAULT_BASE_ACCURACY})"
    )
    parser.add_argument(
        "--continue-on-pass", action="store_true",
        help="Continue testing even after a hypothesis passes (for ablation)"
    )
    args = parser.parse_args()

    # Filter hypotheses
    if args.hypotheses:
        hypothesis_ids = [h.upper() for h in args.hypotheses]
        hypotheses = [h for h in HYPOTHESES if h.id in hypothesis_ids]
        if not hypotheses:
            print(f"ERROR: No matching hypotheses for {args.hypotheses}")
            print(f"Available: {[h.id for h in HYPOTHESES]}")
            sys.exit(1)
    else:
        hypotheses = HYPOTHESES

    if args.max_iterations:
        hypotheses = hypotheses[:args.max_iterations]

    # Initialize log
    log = AutoresearchLog(
        start_time=datetime.now().isoformat(),
        base_accuracy=args.base_accuracy,
    )

    print("=" * 60)
    print("  META-COT AUTORESEARCH LOOP")
    print("=" * 60)
    print(f"  Target: Meta-CoT accuracy >= {args.base_accuracy:.1%} (Base SFT)")
    print(f"  Hypotheses to test: {[h.id for h in hypotheses]}")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Continue on pass: {args.continue_on_pass}")
    total_hours = sum(h.estimated_time_hours for h in hypotheses)
    print(f"  Estimated total time: {total_hours:.1f}h (worst case)")
    print(f"  Results dir: {RESULTS_DIR}")
    print("=" * 60)

    if args.dry_run:
        print("\n--- DRY RUN MODE ---\n")
        for i, h in enumerate(hypotheses, 1):
            print(f"  [{i}] {h.id}: {h.name}")
            print(f"      {h.description}")
            print(f"      Command: {' '.join(h.command)}")
            print(f"      Training: {'Yes' if h.requires_training else 'No'}")
            print(f"      Est. time: {h.estimated_time_hours:.1f}h")
            if h.prerequisites:
                print(f"      Prerequisites: {h.prerequisites}")
            print()

        log.end_time = datetime.now().isoformat()
        log.results = [
            asdict(HypothesisResult(
                hypothesis_id=h.id,
                hypothesis_name=h.name,
                status="skipped (dry-run)",
            ))
            for h in hypotheses
        ]
        log.total_iterations = len(hypotheses)
        save_log(log)
        print("Dry run complete. No experiments were executed.")
        return

    # Run hypotheses in order
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loop_start = time.time()

    for i, hypothesis in enumerate(hypotheses, 1):
        print(f"\n{'#' * 60}")
        print(f"  ITERATION {i}/{len(hypotheses)}: {hypothesis.id}")
        print(f"{'#' * 60}")

        # Check prerequisites
        completed_ids = {r["hypothesis_id"] for r in log.results if r["status"] in ("pass", "fail")}
        unmet = [p for p in hypothesis.prerequisites if p not in completed_ids]
        if unmet:
            print(f"  Skipping {hypothesis.id}: unmet prerequisites {unmet}")
            result = HypothesisResult(
                hypothesis_id=hypothesis.id,
                hypothesis_name=hypothesis.name,
                status="skipped",
                error_message=f"Unmet prerequisites: {unmet}",
            )
            log.results.append(asdict(result))
            continue

        # Run
        result = run_hypothesis(
            hypothesis,
            dry_run=args.dry_run,
            base_accuracy=args.base_accuracy,
        )
        log.results.append(asdict(result))
        log.total_iterations = i

        # Check success
        if result.status == "pass" and not args.continue_on_pass:
            print(f"\n  SUCCESS! {hypothesis.id} passed. Stopping loop.")
            log.success = True
            log.winning_hypothesis = hypothesis.id
            break

        if result.status == "pass" and args.continue_on_pass:
            print(f"\n  {hypothesis.id} passed, but --continue-on-pass is set. Continuing...")
            if not log.success:  # Record first success
                log.success = True
                log.winning_hypothesis = hypothesis.id

        # Save intermediate log
        save_log(log)

    # Finalize
    loop_end = time.time()
    log.end_time = datetime.now().isoformat()
    log.total_duration_seconds = loop_end - loop_start

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  AUTORESEARCH LOOP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total iterations: {log.total_iterations}")
    print(f"  Total duration: {format_duration(log.total_duration_seconds)}")
    print(f"  Success: {log.success}")
    if log.winning_hypothesis:
        print(f"  Winning hypothesis: {log.winning_hypothesis}")

    print(f"\n  Results summary:")
    for r in log.results:
        status_icon = {
            "pass": "PASS",
            "fail": "FAIL",
            "error": "ERR ",
            "skipped": "SKIP",
        }.get(r["status"], "????")
        acc_str = f"{r['meta_accuracy']:.1%}" if r.get("meta_accuracy") is not None else "N/A"
        delta_str = ""
        if r.get("delta") is not None:
            sign = "+" if r["delta"] >= 0 else ""
            delta_str = f" ({sign}{r['delta']:.1%})"
        dur_str = format_duration(r.get("duration_seconds") or 0)
        print(f"    [{status_icon}] {r['hypothesis_id']}: {r['hypothesis_name']} "
              f"— acc={acc_str}{delta_str} ({dur_str})")

    save_log(log)

    if not log.success:
        print(f"\n  All hypotheses exhausted. Meta-CoT has not yet surpassed Base SFT.")
        print(f"  Consider: H2 (difficulty-adaptive), H4 (GPT-5.4 data), H6 (stepwise reward)")
        sys.exit(1)


if __name__ == "__main__":
    main()
