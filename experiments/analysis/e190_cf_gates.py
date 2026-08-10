"""E-190 wiring checks + primary statistic for the offline counterfactual eval.

Reads the JSONL emitted by src/eval/eval_counterfactual_difficulty.py and prints
the pre-registered gates (ledger E-190 §②) so the node log carries the verdict
inputs, not just the stratified accuracy table _summarize.py prints.

Every gate here was fixed BEFORE any result was seen. Failing a gate means the
run is unreadable, not that the effect is zero — the two are different findings
and the prereg forbids collapsing them.

  W-1a tag leak     : "<|meta|>" appears in arm B          gate < 0.05
  W-1b PROSE leak   : arm B carries meta CONTENT without   gate < 0.05
                      the tag ("confidence:" / "assessment:" / "action:").
                      The nine prior runs on metacot-rv report 0/5,346 for
                      W-1a and NEVER measured this one; our own
                      signature_suppression_ids docstring records that v3l
                      discarded ~3/4 of CFs because meta leaked as prose.
  W-4 truncation    : either arm hit the token cap          gate < 0.10
  A-vs-A            : arm A decoded twice, same params      gate |U_AA| < 0.005
  PRIMARY U         : (saved - broke) / eligible, where eligible = arm A
                      actually emitted meta. Reported with its REALISED MDE,
                      because the prior runs' median |effect| (0.0118) sits
                      below this design's MDE (~0.027) and a null here is a
                      resolution statement, not an effect statement.
  ABSOLUTE          : acc_A - acc_B over ALL problems. If the conditional
                      statistic and this one disagree in sign, this one is the
                      headline (0731: conditional metrics cannot see the cost
                      they push elsewhere).

Usage:
  python -m experiments.analysis.e190_cf_gates --jsonl cf_b2p_gs300.jsonl
"""
import argparse
import json
import math


def _mcnemar_exact(b, c):
    """Two-sided exact binomial p for the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


def _paired_se(b, c, n):
    """SE of (b - c)/n for a paired {-1,0,+1} contrast."""
    if n == 0:
        return float("nan")
    return math.sqrt(max(b + c - (b - c) ** 2 / n, 0.0)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
    n = len(rows)
    if n == 0:
        print("[E-190] EMPTY jsonl — no verdict")
        return
    cnt = lambda k: sum(1 for r in rows if r.get(k))

    print(f"[E-190] n = {n}")
    tag, prose_b, prose_a = cnt("emitted_meta_without"), cnt("meta_signature_without"), cnt("meta_signature_with")
    emit = cnt("emitted_meta_with")
    ok = lambda good: "PASS" if good else "**FAIL**"
    print(f"  W-1a tag leak    {tag:>5}/{n} = {tag/n:.4f}   gate < 0.05   {ok(tag/n < 0.05)}")
    print(f"  W-1b PROSE leak  {prose_b:>5}/{n} = {prose_b/n:.4f}   gate < 0.05   {ok(prose_b/n < 0.05)}"
          f"   (arm A prose {prose_a/n:.4f})")
    print(f"  emission arm A   {emit:>5}/{n} = {emit/n:.4f}")

    if "finish_with" in rows[0]:
        tr = sum(1 for r in rows if r.get("finish_with") != "stop" or r.get("finish_without") != "stop")
        print(f"  W-4 truncation   {tr:>5}/{n} = {tr/n:.4f}   gate < 0.10   {ok(tr/n < 0.10)}")

    if "correct_with2" in rows[0]:
        b = sum(1 for r in rows if r["correct_with"] and not r["correct_with2"])
        c = sum(1 for r in rows if r["correct_with2"] and not r["correct_with"])
        u_aa = (b - c) / n
        idn = cnt("identical_a2")
        print(f"  A-vs-A U_AA      {u_aa:+.5f} +- {1.96*_paired_se(b,c,n):.5f}   gate |U_AA| < 0.005   "
              f"{ok(abs(u_aa) < 0.005)}   identical text {idn}/{n} = {idn/n:.4f}")
    else:
        print("  A-vs-A           NOT RUN (--repeat_arm_a absent) — noise floor unmeasured")

    elig = [r for r in rows if r.get("emitted_meta_with")]
    m = len(elig)
    print(f"\n[E-190] PRIMARY — eligible (arm A emitted meta) n = {m}")
    if m == 0:
        print("  no eligible rows: nothing to measure. NOT a null.")
    else:
        sv = sum(1 for r in elig if r["correct_with"] and not r["correct_without"])
        bk = sum(1 for r in elig if r["correct_without"] and not r["correct_with"])
        U, f = (sv - bk) / m, (sv + bk) / m
        se = _paired_se(sv, bk, m)
        mde = 2.8016 * math.sqrt(f) / math.sqrt(m) if f > 0 else float("nan")
        print(f"  U = {U:+.4f}   95%CI [{U-1.96*se:+.4f}, {U+1.96*se:+.4f}]   McNemar p = {_mcnemar_exact(sv,bk):.4f}")
        print(f"  saved {sv}  broke {bk}   f = {f:.4f}   realised MDE = {mde:.4f}")
        if f > 0:
            print(f"  selectivity  sel+ {sv/m:.4f}  sel- {bk/m:.4f}  ratio {sv/(sv+bk):.4f}  (chance 0.5)"
                  + ("" if f >= 0.05 else "   [f<0.05: reported, NOT judged]"))
        # Pre-registered band. delta = 0.05 equivalence margin.
        lo, hi, d = U - 1.96 * se, U + 1.96 * se, 0.05
        if lo > 0:
            band = "META CHANGES THE OUTCOME (positive)"
        elif hi < 0:
            band = "META IS HARMFUL"
        elif lo > -d and hi < d:
            band = "DECORATIVE (equivalent within +-0.05)"
        else:
            band = "**UNDERPOWERED — no verdict** (CI spans 0 AND exceeds +-0.05; NOT 'decorative')"
        print(f"  BAND: {band}")
        # A significant result that is SMALLER than the realised MDE is the exact
        # regime where winner's curse inflates the estimate: the design could only
        # reliably detect effects of size MDE, so anything it does call significant
        # at |U| < MDE is preferentially an upward fluctuation. Say so on the line,
        # not in a footnote - the nine prior runs live entirely in this regime.
        if lo > 0 or hi < 0:
            if abs(U) < mde:
                print(f"  ⚠ |U| {abs(U):.4f} < realised MDE {mde:.4f}: significant but UNDER-POWERED. "
                      f"Treat the magnitude as an upper bound (winner's curse), and do not report "
                      f"this checkpoint alone — prior runs flipped sign between adjacent steps.")

    aA = sum(1 for r in rows if r["correct_with"]) / n
    aB = sum(1 for r in rows if r["correct_without"]) / n
    print(f"\n[E-190] ABSOLUTE (all {n} problems, emission-agnostic)")
    print(f"  acc_A {aA:.4f} - acc_B {aB:.4f} = {aA-aB:+.4f}")
    if m and aA - aB <= 0:
        print("  NOTE: absolute delta is <= 0. Per prereg, the absolute number is the headline.")


if __name__ == "__main__":
    main()
