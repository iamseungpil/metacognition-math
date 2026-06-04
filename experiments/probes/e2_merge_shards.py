"""Merge e2_contrastive_steering shard JSONLs (4-GPU data-parallel run) into one
report + a powered aggregate summary.

Each shard wrote disjoint per-problem records (same schema, different problems). We
concatenate them and recompute the per-arm accuracy / calibration metrics + the paired
Δ-vs-self with a bootstrap 95% CI and a paired permutation p-value. With the full 1030
the n is large enough that the MDE power-gate that made the 110-subset INCONCLUSIVE is
no longer the bottleneck — a paired CI is the clean read.

Usage:
  python e2_merge_shards.py --glob '/scratch/reports/e2_steering_v8_strict_conf1030_shard*.jsonl' \
      --out /scratch/reports/e2_steering_v8_strict_conf1030_merged.json
"""
from __future__ import annotations
import argparse, glob, json, random, statistics as st
from collections import defaultdict, Counter


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def _paired(steer, base, key):
    """Paired (problem_id-matched) deltas of metrics[key], steer - base."""
    bmap = {r["problem_id"]: r for r in base}
    d = []
    for r in steer:
        b = bmap.get(r["problem_id"])
        if not b:
            continue
        a, c = (r.get("metrics") or {}).get(key), (b.get("metrics") or {}).get(key)
        if a is not None and c is not None:
            d.append(a - c)
    return d


def _bootstrap_ci(d, rng, iters=10000):
    if not d:
        return (None, None, None)
    n = len(d)
    means = []
    for _ in range(iters):
        s = [d[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return (st.mean(d), means[int(0.025 * iters)], means[int(0.975 * iters)])


def _perm_p(d, rng, iters=10000):
    """Two-sided paired permutation p-value (sign-flip null)."""
    if not d:
        return None
    obs = abs(sum(d) / len(d))
    ge = 0
    for _ in range(iters):
        s = sum(x if rng.random() < 0.5 else -x for x in d) / len(d)
        if abs(s) >= obs:
            ge += 1
    return (ge + 1) / (iters + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for shard JSONLs")
    ap.add_argument("--out", required=True, help="merged summary JSON path")
    ap.add_argument("--merged_jsonl", default=None, help="optional concatenated JSONL path")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"no shard files match {args.glob!r}")
    recs = []
    for f in files:
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    print(f"[merge] {len(files)} shards -> {len(recs)} records")

    if args.merged_jsonl:
        with open(args.merged_jsonl, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")

    byarm = defaultdict(list)
    for r in recs:
        byarm[r["arm"]].append(r)
    arms = sorted(byarm)
    rng = random.Random(args.seed)

    per_arm = {}
    for arm in arms:
        rs = byarm[arm]
        per_arm[arm] = {
            "n": len(rs),
            "accuracy": _mean((r.get("metrics") or {}).get("accuracy") for r in rs),
            "agree_with_gold": _mean((r.get("metrics") or {}).get("agree_with_gold") for r in rs),
            "calibration_gap": _mean((r.get("metrics") or {}).get("calibration_gap") for r in rs),
            "verbalized_conf": _mean((r.get("metrics") or {}).get("verbalized_conf") for r in rs),
            "pass_at_k_cont": _mean(r.get("pass_at_k_cont") for r in rs),
            "per_benchmark": dict(Counter(r.get("benchmark") for r in rs)),
        }

    base_arm = "self"
    deltas = {}
    if base_arm in byarm:
        for arm in arms:
            if arm == base_arm:
                continue
            entry = {}
            for key in ("accuracy", "calibration_gap", "verbalized_conf"):
                d = _paired(byarm[arm], byarm[base_arm], key)
                mean, lo, hi = _bootstrap_ci(d, rng)
                entry[key] = {
                    "delta": mean, "ci95": [lo, hi], "p_perm": _perm_p(d, rng), "n_paired": len(d),
                }
            deltas[f"{arm}_minus_self"] = entry

    summary = {
        "source_glob": args.glob, "n_shards": len(files), "n_records": len(recs),
        "arms": arms, "per_arm": per_arm, "paired_vs_self": deltas,
    }
    json.dump(summary, open(args.out, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[merge] wrote {args.out}")


if __name__ == "__main__":
    main()
