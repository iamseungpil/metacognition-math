#!/usr/bin/env python3
"""Smoke acceptance checker for N3 (plan §4.2 + §9.3).

Reads stdout.log / metrics.jsonl from a smoke RUN_DIR, checks invariants,
writes ``smoke_acceptance.json`` with pass/fail + per-check details.

Invariants checked (subset; full list in plan §4.2):
    A1. training exit code == 0
    A2. metrics.jsonl has ≥ 10 step entries
    A3. delta_t_mean is finite, |mean| < 5
    A4. delta_t_std > 0.01 (signal present)
    A5. clip_fraction_w ∈ [0.00, 0.40]
    A6. decoy_eq_gold_rate == 0 (hard)
    A7. decoy_is_correct_rate < 0.05 (soft)
    A8. contrastive_fwd_count == 2 per step (invariant)
    A9. meta_wrap_rate ≥ 0.70 (controller preservation)

Usage:
    python scripts/smoke_n3_acceptance.py <RUN_DIR> <EXIT_CODE>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _load_metrics(run_dir: Path) -> list[dict]:
    """Load metrics.jsonl (one JSON per step) from the run dir."""
    for candidate in (
        run_dir / "metrics.jsonl",
        run_dir / "checkpoints" / "metrics.jsonl",
    ):
        if candidate.exists():
            with candidate.open() as f:
                return [json.loads(line) for line in f if line.strip()]
    return []


def _scalar(rows: list[dict], key: str) -> list[float]:
    """Extract a scalar time series."""
    return [float(r[key]) for r in rows if key in r and isinstance(r[key], (int, float))]


def main():
    if len(sys.argv) < 3:
        print("usage: smoke_n3_acceptance.py <RUN_DIR> <EXIT_CODE>", file=sys.stderr)
        sys.exit(64)

    run_dir = Path(sys.argv[1])
    exit_code = int(sys.argv[2])

    checks: dict[str, dict] = {}
    def record(key: str, ok: bool, detail: str):
        checks[key] = {"pass": ok, "detail": detail}

    # A1
    record("A1_exit_code_zero", exit_code == 0, f"exit_code={exit_code}")

    rows = _load_metrics(run_dir)
    # A2
    record("A2_metrics_rows_gte_10", len(rows) >= 10, f"n_rows={len(rows)}")

    # A3-A4: delta_t statistics
    dt_means = _scalar(rows, "meta_rlsd/delta_t_mean")
    dt_stds = _scalar(rows, "meta_rlsd/delta_t_std")
    dt_mean_avg = sum(dt_means) / len(dt_means) if dt_means else float("nan")
    dt_std_avg = sum(dt_stds) / len(dt_stds) if dt_stds else 0.0
    record(
        "A3_delta_t_mean_finite",
        len(dt_means) > 0 and math.isfinite(dt_mean_avg) and abs(dt_mean_avg) < 5.0,
        f"mean(delta_t_mean)={dt_mean_avg:.4f}",
    )
    record(
        "A4_delta_t_std_signal",
        dt_std_avg > 0.01,
        f"mean(delta_t_std)={dt_std_avg:.4f} (need > 0.01)",
    )

    # A5: clip fraction
    clip_fracs = _scalar(rows, "meta_rlsd/clip_fraction_w")
    clip_avg = sum(clip_fracs) / len(clip_fracs) if clip_fracs else 0.0
    record(
        "A5_clip_fraction_in_range",
        0.0 <= clip_avg <= 0.40,
        f"mean(clip_frac_w)={clip_avg:.4f} (need ∈ [0, 0.40])",
    )

    # A6, A7: decoy-quality
    eq_rates = _scalar(rows, "meta_rlsd/decoy_eq_gold_rate")
    correct_rates = _scalar(rows, "meta_rlsd/decoy_is_correct_rate")
    eq_max = max(eq_rates) if eq_rates else 0.0
    correct_max = max(correct_rates) if correct_rates else 0.0
    record(
        "A6_decoy_eq_gold_zero",
        eq_max == 0.0,
        f"max(decoy_eq_gold_rate)={eq_max:.4f} (must be 0)",
    )
    record(
        "A7_decoy_is_correct_rate_low",
        correct_max < 0.05,
        f"max(decoy_is_correct_rate)={correct_max:.4f} (need < 0.05)",
    )

    # A8: contrastive fwd count invariant
    fwd_counts = _scalar(rows, "meta_rlsd/contrastive_fwd_count")
    fwd_ok = all(abs(v - 2.0) < 1e-6 for v in fwd_counts) if fwd_counts else False
    record(
        "A8_contrastive_fwd_count_eq_2",
        fwd_ok,
        f"rows={len(fwd_counts)}, all == 2.0: {fwd_ok}",
    )

    # A9: meta wrap rate preservation
    wrap_rates = _scalar(rows, "meta_rlsd/wrap_rate") + _scalar(rows, "wrap_rate")
    wrap_avg = sum(wrap_rates) / len(wrap_rates) if wrap_rates else 0.0
    record(
        "A9_wrap_rate_preserved",
        wrap_avg >= 0.70,
        f"mean(wrap_rate)={wrap_avg:.4f} (need ≥ 0.70)",
    )

    all_pass = all(c["pass"] for c in checks.values())

    summary = {
        "run_dir": str(run_dir),
        "exit_code": exit_code,
        "n_metrics_rows": len(rows),
        "all_pass": all_pass,
        "checks": checks,
    }

    out = run_dir / "smoke_acceptance.json"
    out.write_text(json.dumps(summary, indent=2))

    for k, v in checks.items():
        marker = "PASS" if v["pass"] else "FAIL"
        print(f"[{marker}] {k}: {v['detail']}")

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    print(f"Written: {out}")

    sys.exit(0 if all_pass else 4)


if __name__ == "__main__":
    main()
