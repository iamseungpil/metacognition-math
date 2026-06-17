"""Summarize the difficulty-stratified counterfactual eval (acc_with vs acc_without).

Reads the JSONL emitted by eval_counterfactual_difficulty.py and prints, for each
stratum, n / acc_with / acc_without / Δ, plus the McNemar flip counts:
  saved  = meta wrong->right (without wrong, with right)  -> meta HELPED
  broke  = meta right->wrong (without right, with wrong)   -> meta HURT
Δ>0 with saved>broke on hard/redirect = direct evidence meta causes accuracy.
"""
import argparse
import json
from collections import defaultdict


def _acc(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def _block(title, groups):
    print(f"\n=== {title} ===")
    print(f"{'stratum':<22} {'n':>4} {'acc_with':>9} {'acc_wo':>9} {'Δ':>8} {'saved':>6} {'broke':>6}")
    for name in sorted(groups):
        rows = groups[name]
        aw, ao = _acc(rows, "correct_with"), _acc(rows, "correct_without")
        saved = sum(1 for r in rows if r["correct_with"] and not r["correct_without"])
        broke = sum(1 for r in rows if r["correct_without"] and not r["correct_with"])
        print(f"{str(name):<22} {len(rows):>4} {aw:>9.3f} {ao:>9.3f} {aw-ao:>+8.3f} {saved:>6} {broke:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    print(f"total {len(rows)} problems")
    aw, ao = _acc(rows, "correct_with"), _acc(rows, "correct_without")
    saved = sum(1 for r in rows if r["correct_with"] and not r["correct_without"])
    broke = sum(1 for r in rows if r["correct_without"] and not r["correct_with"])
    print(f"OVERALL acc_with {aw:.3f} | acc_without {ao:.3f} | Δ {aw-ao:+.3f} | saved {saved} broke {broke}")
    emit = sum(1 for r in rows if r["emitted_meta_with"]) / len(rows) if rows else 0
    print(f"meta-emission rate (with arm): {emit:.3f}")

    by_diff = defaultdict(list)
    by_scen = defaultdict(list)
    by_src = defaultdict(list)
    by_cross = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
        by_scen[r["scenario"]].append(r)
        by_src[r["data_source"]].append(r)
        by_cross[(r["difficulty"], r["scenario"])].append(r)
    # order difficulty easy<medium<hard
    order = {"easy": 0, "medium": 1, "hard": 2}
    diff_sorted = {f"{order.get(k,9)}.{k}": v for k, v in by_diff.items()}
    _block("by DIFFICULTY", diff_sorted)
    _block("by SCENARIO", by_scen)
    _block("by DIFFICULTY x SCENARIO", {f"{order.get(d,9)}.{d}/{s}": v for (d, s), v in by_cross.items()})
    _block("by DATA_SOURCE", by_src)


if __name__ == "__main__":
    main()
