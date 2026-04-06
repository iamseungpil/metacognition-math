"""E11 Pilot Eval Analysis — 5 dimensions of metacognitive behavior.

Analyzes:
1. Verification effectiveness (FP rate, verify→confidence change)
2. Calibration quality (ECE, Brier, reliability diagram)
3. Approach change (structural switch rate, switch success)
4. Information Checkpointing (confidence trajectory patterns)
5. Accuracy preservation (vs E9 baseline)

Usage:
  PYTHONPATH=. python scripts/analyze_e11_pilot.py \
    --e11_path results/eval_v6_E11/eval_v6_E11.json \
    --e9_path results/eval_1030_v5/eval_1030_E9.json \
    --base_path results/eval_1030_v5/eval_1030_base_sft.json
"""
import argparse
import json
import re
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.rewards import _check_correctness

META_RE = re.compile(r"<\|meta\|>(.*?)<\|/meta\|>", re.DOTALL | re.IGNORECASE)
CONF_RE = re.compile(r"(?:confidence|conf)[:\s]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
VERIFY_RE = re.compile(
    r"\b(verify|check|confirm|double.check|re.?check|substitut\w*\s+back|plug\w*\s+(back|in))\b",
    re.IGNORECASE,
)

METHOD_FAMILIES = {
    "substitution": r"\bsubstitut",
    "factoring": r"\bfactor",
    "quadratic_formula": r"\bquadratic\s+formula",
    "induction": r"\binduction",
    "coordinate": r"\bcoordinate",
    "trigonometry": r"\b(sin|cos|tan)\b",
    "combinatorics": r"\b(choose|binom|permut)",
    "modular": r"\b(mod\s+\d|modular)",
    "casework": r"\bcasework",
    "recursion": r"\brecur",
}


def detect_methods(text):
    methods = set()
    for fam, pat in METHOD_FAMILIES.items():
        if re.search(pat, text.lower()):
            methods.add(fam)
    return methods


def load_and_rescore(path):
    with open(path) as f:
        results = json.load(f)["results"]
    for r in results:
        r["cv2"] = _check_correctness(
            r.get("completion", ""),
            r.get("full_gold_answer", r.get("gold_answer", "")),
        )
    return results


def extract_confidences(completion):
    blocks = META_RE.findall(completion)
    confs = []
    for block in blocks:
        for m in CONF_RE.finditer(block):
            v = float(m.group(1))
            if v > 1.0:
                v /= 100.0
            confs.append(max(0.0, min(1.0, v)))
    return confs


def analyze(e11_results, e9_results, base_results):
    print("=" * 80)
    print("E11 Pilot Eval — 5-Dimension Metacognitive Analysis")
    print("=" * 80)

    n = len(e11_results)
    acc = sum(1 for r in e11_results if r["cv2"]) / n * 100

    by_bench = {}
    for r in e11_results:
        by_bench.setdefault(r["benchmark"], []).append(r)

    print(f"\nOverall: {acc:.1f}% ({n} problems)")
    for bench in ["gsm8k", "math500", "aime2024"]:
        rows = by_bench.get(bench, [])
        if rows:
            ba = sum(1 for r in rows if r["cv2"]) / len(rows) * 100
            print(f"  {bench}: {ba:.1f}%")

    # ─── 1. Verification Effectiveness ───
    print("\n" + "=" * 60)
    print("1. VERIFICATION EFFECTIVENESS")
    print("=" * 60)

    with_meta = [r for r in e11_results if r["num_meta_blocks"] > 0]
    has_conf = [r for r in with_meta if r.get("avg_confidence") is not None]
    high_conf = [r for r in has_conf if r["avg_confidence"] >= 0.7]
    hc_correct = sum(1 for r in high_conf if r["cv2"])
    hc_fp = len(high_conf) - hc_correct

    verify_present = [r for r in e11_results if VERIFY_RE.search(r.get("completion", ""))]
    v_correct = sum(1 for r in verify_present if r["cv2"])
    v_wrong = len(verify_present) - v_correct

    # Verify + high-conf + wrong = 형식적 검산 실패
    verify_hc_wrong = sum(
        1 for r in verify_present
        if not r["cv2"] and r.get("avg_confidence") and r["avg_confidence"] >= 0.7
    )

    print(f"  High-conf(>=0.7): {len(high_conf)} → correct {hc_correct}, wrong {hc_fp}")
    print(f"  FP rate: {hc_fp / max(len(high_conf), 1) * 100:.1f}%")
    print(f"  Verify present: {len(verify_present)} → correct {v_correct}, wrong {v_wrong}")
    print(f"  Verify + high-conf + wrong: {verify_hc_wrong} (형식적 검산 실패)")

    # Compare with E9
    e9_hc = [r for r in e9_results if r.get("avg_confidence") and r["avg_confidence"] >= 0.7]
    e9_fp = sum(1 for r in e9_hc if not r["cv2"]) / max(len(e9_hc), 1) * 100
    print(f"  E9 FP rate: {e9_fp:.1f}% → E11: {hc_fp / max(len(high_conf), 1) * 100:.1f}%")

    # ─── 2. Calibration Quality ───
    print("\n" + "=" * 60)
    print("2. CALIBRATION QUALITY")
    print("=" * 60)

    if has_conf:
        confs = [r["avg_confidence"] for r in has_conf]
        actuals = [1.0 if r["cv2"] else 0.0 for r in has_conf]

        # ECE
        n_bins = 15
        ece = 0.0
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            mask = [lo <= c < hi for c in confs]
            if sum(mask) > 0:
                bin_acc = np.mean([a for a, m in zip(actuals, mask) if m])
                bin_conf = np.mean([c for c, m in zip(confs, mask) if m])
                ece += sum(mask) / len(confs) * abs(bin_acc - bin_conf)

        # Brier
        brier = np.mean([(c - a) ** 2 for c, a in zip(confs, actuals)])

        print(f"  Confidence coverage: {len(has_conf)}/{n} ({len(has_conf) / n * 100:.1f}%)")
        print(f"  ECE: {ece:.3f}")
        print(f"  Brier score: {brier:.3f}")
        print(f"  Conf mean: {np.mean(confs):.3f}, std: {np.std(confs):.3f}")

        # Reliability diagram
        print(f"\n  Reliability diagram:")
        for lo_pct in [0, 20, 40, 60, 80]:
            hi_pct = lo_pct + 20
            lo_f, hi_f = lo_pct / 100, hi_pct / 100
            bucket = [(c, a) for c, a in zip(confs, actuals) if lo_f <= c < hi_f]
            if bucket:
                mean_c = np.mean([c for c, _ in bucket])
                mean_a = np.mean([a for _, a in bucket])
                print(f"    [{lo_pct}-{hi_pct}%]: n={len(bucket):3d}, mean_conf={mean_c:.2f}, actual_acc={mean_a:.2f}, gap={abs(mean_c - mean_a):.2f}")
    else:
        print("  No confidence data available")

    # ─── 3. Approach Change ───
    print("\n" + "=" * 60)
    print("3. APPROACH CHANGE (STRUCTURAL SWITCH)")
    print("=" * 60)

    structural_switch = 0
    partial_switch = 0
    switch_correct = 0
    switch_examples = []

    for r in with_meta:
        comp = r.get("completion", "")
        ms = comp.find("<|meta|>")
        me = comp.rfind("<|/meta|>")
        if ms < 0 or me < 0:
            continue
        pre = comp[:ms]
        post = comp[me + len("<|/meta|>"):]
        if len(pre) < 30 or len(post) < 30:
            continue

        pre_m = detect_methods(pre)
        post_m = detect_methods(post)
        only_pre = pre_m - post_m
        only_post = post_m - pre_m

        if only_pre and only_post:
            structural_switch += 1
            if r["cv2"]:
                switch_correct += 1
            switch_examples.append({
                "q": r["full_question"][:60],
                "bench": r["benchmark"],
                "correct": r["cv2"],
                "pre": list(pre_m),
                "post": list(post_m),
            })
        elif only_pre or only_post:
            partial_switch += 1

    total_meta = len(with_meta)
    print(f"  Meta present: {total_meta}/{n}")
    print(f"  Structural switch: {structural_switch} ({structural_switch / max(total_meta, 1) * 100:.1f}%)")
    print(f"  Switch → correct: {switch_correct}/{structural_switch}")
    print(f"  Partial switch: {partial_switch}")

    # Compare with E9
    print(f"  E9 structural switch: 17/960 (1.8%)")
    print(f"  E11 structural switch: {structural_switch}/{total_meta} ({structural_switch / max(total_meta, 1) * 100:.1f}%)")

    if switch_examples:
        print(f"\n  Switch examples (first 3):")
        for ex in switch_examples[:3]:
            print(f"    {ex['bench']}: {ex['q']}... correct={ex['correct']}")
            print(f"      pre: {ex['pre']} → post: {ex['post']}")

    # ─── 4. Information Checkpointing ───
    print("\n" + "=" * 60)
    print("4. INFORMATION CHECKPOINTING (CONFIDENCE TRAJECTORY)")
    print("=" * 60)

    patterns = Counter()
    pattern_correct = Counter()

    for r in e11_results:
        comp = r.get("completion", "")
        confs = extract_confidences(comp)

        if len(confs) == 0:
            pat = "no_conf"
        elif len(confs) == 1:
            pat = "single"
        elif confs[-1] > confs[0] + 0.1:
            pat = "rise"
        elif confs[-1] < confs[0] - 0.1:
            pat = "drop"
        elif min(confs) < confs[0] - 0.1:
            pat = "dip_recover"
        else:
            pat = "flat"

        patterns[pat] += 1
        if r["cv2"]:
            pattern_correct[pat] += 1

    print(f"  Confidence trajectory patterns:")
    for pat in ["rise", "drop", "dip_recover", "flat", "single", "no_conf"]:
        n_pat = patterns[pat]
        c_pat = pattern_correct[pat]
        if n_pat > 0:
            print(f"    {pat:12s}: {n_pat:4d} ({c_pat}/{n_pat} = {c_pat / n_pat * 100:.1f}% correct)")

    multi_meta = sum(1 for r in e11_results if r["num_meta_blocks"] >= 2)
    print(f"\n  Multi-meta (>=2 blocks): {multi_meta}/{n} ({multi_meta / n * 100:.1f}%)")
    print(f"  E9 multi-meta: ~25/1030 (2.4%)")

    # ─── 5. Accuracy Preservation ───
    print("\n" + "=" * 60)
    print("5. ACCURACY PRESERVATION (vs E9)")
    print("=" * 60)

    e9_acc = sum(1 for r in e9_results if r["cv2"]) / len(e9_results) * 100
    base_acc = sum(1 for r in base_results if r["cv2"]) / len(base_results) * 100

    print(f"  base_sft: {base_acc:.1f}%")
    print(f"  E9:       {e9_acc:.1f}%")
    print(f"  E11:      {acc:.1f}%")
    print(f"  E11 vs E9: {acc - e9_acc:+.1f}pp")
    print(f"  E11 vs base: {acc - base_acc:+.1f}pp")

    # Regression analysis
    e9_by_q = {r["full_question"]: r for r in e9_results}
    regressed = 0
    improved = 0
    for r in e11_results:
        q = r["full_question"]
        if q in e9_by_q:
            e9_r = e9_by_q[q]
            if e9_r["cv2"] and not r["cv2"]:
                regressed += 1
            elif not e9_r["cv2"] and r["cv2"]:
                improved += 1

    print(f"\n  vs E9: +{improved} improved, -{regressed} regressed (net {improved - regressed:+d})")

    # ─── Summary Decision ───
    print("\n" + "=" * 60)
    print("PILOT DECISION")
    print("=" * 60)

    switch_rate = structural_switch / max(total_meta, 1) * 100

    if switch_rate >= 5 and acc >= 60:
        print(f"  → Scenario A: switch {switch_rate:.1f}% >= 5%, acc {acc:.1f}% >= 60%")
        print(f"  → PROCEED to E12/E13 RL")
    elif switch_rate >= 1 and acc >= 58:
        print(f"  → Scenario B: switch {switch_rate:.1f}% (1-5%), acc {acc:.1f}%")
        print(f"  → Seed expansion needed, or try RL with Information Checkpointing reward")
    elif switch_rate < 1:
        print(f"  → Scenario C: switch {switch_rate:.1f}% ≈ 0%")
        print(f"  → E9 base too rigid. Restart from base_sft with 10K clean data")
    else:
        print(f"  → Scenario D: acc {acc:.1f}% < 58% — catastrophic forgetting")
        print(f"  → SFT data or LR issue. Check E9 base integrity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e11_path", required=True)
    parser.add_argument("--e9_path", default="results/eval_1030_v5/eval_1030_E9.json")
    parser.add_argument("--base_path", default="results/eval_1030_v5/eval_1030_base_sft.json")
    args = parser.parse_args()

    e11 = load_and_rescore(args.e11_path)
    e9 = load_and_rescore(args.e9_path)
    base = load_and_rescore(args.base_path)

    analyze(e11, e9, base)
