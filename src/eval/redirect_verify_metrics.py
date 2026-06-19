"""INTENT metrics: does self-emitted metacognition work *as intended*.

North-star (CLAUDE.md): the model SELF-EMITS a confidence score and DECIDES
redirect-vs-verify from it (redirect when low-conf / on a wrong path, verify
when high-conf to confirm, nothing otherwise). Calibration is a *sub-signal* of
that decision, not the goal. These metrics score the decision, not just the
number.

All functions are pure (GPU-free) and operate on already-graded per-problem
records. A record is a plain dict with (subset of) keys:
    action            'redirect' | 'verify' | 'none'   (what meta emitted)
    confidence        float in [0,1]                    (self-emitted conf)
    recoverable_wrong bool   wrong-but-fixable, gold-derived (causality only,
                             never used to *measure* confidence -> no leak)
    flipped_to_right  bool   redirect arm: did it flip wrong->right
    verify_confirmed  bool   verify arm: did it confirm/correct

Each metric returns a structured dict; rates that are undefined (no samples)
are returned as ``None`` rather than a misleading 0.0, so a downstream report
can distinguish "never fired" from "fired and always wrong".
"""
from __future__ import annotations

from typing import Optional

# SHARED build-time thresholds: single source of truth lives in the data layer
# (src/data/confidence_label.py). We re-export them under SHARED_* names so the
# eval references the SAME numbers the build buckets on, instead of redefining
# its own divergent low=0.5/high=0.7.
from src.data.confidence_label import CONF_LO as SHARED_CONF_LO
from src.data.confidence_label import CONF_HI as SHARED_CONF_HI

# DELIBERATELY STRICTER held-out eval bars (NOT a silent divergence from the
# shared build thresholds above):
#   * HELDOUT_LOW_CONF = 0.5 > SHARED_CONF_LO (0.30): at eval we demand the
#     model was *clearly* low-confidence (conf < 0.5) before we credit a
#     redirect as "decided from low confidence". The build can bucket a problem
#     as redirect at the looser 0.30 self-consistency floor, but to PASS the
#     held-out action<->confidence test the emitted confidence must clear the
#     stricter 0.5 bar — a held-out model that merely squeaks under 0.30 is not
#     given free credit.
#   * HELDOUT_HIGH_CONF = SHARED_CONF_HI (0.70): the high bar matches the shared
#     verify threshold exactly (no reason to be stricter on the high side).
# These are intentionally >= the shared bars; an assertion below guards that we
# never accidentally make the eval LOOSER than the build.
HELDOUT_LOW_CONF = 0.5
HELDOUT_HIGH_CONF = SHARED_CONF_HI

assert HELDOUT_LOW_CONF >= SHARED_CONF_LO, "held-out low bar must not be looser than shared"
assert HELDOUT_HIGH_CONF >= SHARED_CONF_HI, "held-out high bar must not be looser than shared"


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# ----------------------------------------------------------------------------
# (1) confidence calibration -- does emitted conf track actual correctness
# ----------------------------------------------------------------------------
def confidence_calibration(emitted_confs, correct_flags, n_bins: int = 10) -> dict:
    """Expected Calibration Error + signed calibration gap.

    ECE = sum_b (|bin| / N) * |acc_b - conf_b|   (always >= 0).
    signed_gap = mean_conf - mean_acc  ( >0 over-confident, <0 under-confident ).

    Returns {"ece", "signed_gap", "mean_conf", "mean_acc", "n", "bins"}.
    Empty input -> numeric fields None (n=0). Confidences are clamped to [0,1].
    """
    if len(emitted_confs) != len(correct_flags):
        raise ValueError(
            f"length mismatch: {len(emitted_confs)} confs vs "
            f"{len(correct_flags)} flags"
        )
    n = len(emitted_confs)
    if n == 0:
        return {"ece": None, "signed_gap": None, "mean_conf": None,
                "mean_acc": None, "n": 0, "bins": []}

    confs = [_clamp01(float(c)) for c in emitted_confs]
    flags = [1.0 if f else 0.0 for f in correct_flags]

    mean_conf = sum(confs) / n
    mean_acc = sum(flags) / n

    # bin edges: [0, 1/nb), ..., [(nb-1)/nb, 1]  (last bin closed on the right)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            members = [i for i in range(n) if lo <= confs[i] <= hi]
        else:
            members = [i for i in range(n) if lo <= confs[i] < hi]
        if not members:
            bins.append({"lo": lo, "hi": hi, "count": 0,
                         "conf": None, "acc": None})
            continue
        cnt = len(members)
        bconf = sum(confs[i] for i in members) / cnt
        bacc = sum(flags[i] for i in members) / cnt
        ece += (cnt / n) * abs(bacc - bconf)
        bins.append({"lo": lo, "hi": hi, "count": cnt,
                     "conf": bconf, "acc": bacc})

    return {
        "ece": ece,
        "signed_gap": mean_conf - mean_acc,
        "mean_conf": mean_conf,
        "mean_acc": mean_acc,
        "n": n,
        "bins": bins,
    }


# ----------------------------------------------------------------------------
# (2) action appropriateness -- right action for the situation, vs misfire
# ----------------------------------------------------------------------------
def action_appropriateness(records, high_conf: float = HELDOUT_HIGH_CONF,
                           low_conf: float = HELDOUT_LOW_CONF) -> dict:
    """Rate that the emitted action matched the situation.

    NORTH-STAR (CLAUDE.md): the action must be DECIDED FROM the self-emitted
    confidence. So appropriateness is conditioned on BOTH the situation (gold)
    AND the emitted confidence -- a model whose confidence is decoupled from its
    action must NOT score appropriate just because gold happened to agree.

    A REDIRECT is appropriate iff the case was actually recoverable-wrong
    (a genuinely fixable wrong path, gold-derived) AND the model was actually
    low-confidence (emitted confidence < low_conf) -- i.e. it redirected
    *because* it was unsure. A high-confidence redirect on a recoverable-wrong
    case is a MISFIRE (decoupled action), not free credit: this is symmetric to
    verify, which has always been confidence-gated.

    A VERIFY is appropriate iff confidence is high (>= high_conf) -- verify is
    the "confirm what I think is right" action. Firing verify on a low-conf
    (< low_conf) case is a misfire (it should have redirected).

    Defaults come from the held-out bars (`HELDOUT_LOW_CONF`/`HELDOUT_HIGH_CONF`),
    which are deliberately stricter than the build-time `SHARED_CONF_LO`/
    `SHARED_CONF_HI` (see module docstring), not a silent divergence.

    Also reports an action<->confidence CONSISTENCY rate (over ALL fired
    redirect+verify actions): the fraction whose emitted confidence sits on the
    side its action implies (redirect => conf < low_conf, verify =>
    conf >= high_conf), measured independently of the gold situation.

    Returns appropriate/misfire rates per action plus counts. Undefined rates
    (action never fired) are None.
    """
    redirects = [r for r in records if r.get("action") == "redirect"]
    verifies = [r for r in records if r.get("action") == "verify"]

    n_red = len(redirects)
    n_ver = len(verifies)

    # redirect now requires BOTH recoverable_wrong (situation) AND low-conf
    # (decided-from-confidence) -- symmetric to verify's confidence gate.
    red_appropriate = sum(
        1 for r in redirects
        if r.get("recoverable_wrong")
        and float(r.get("confidence", 0.0)) < low_conf
    )
    ver_appropriate = sum(
        1 for r in verifies if float(r.get("confidence", 0.0)) >= high_conf
    )
    # verify misfire is specifically: fired on a clearly LOW-conf case
    ver_misfire = sum(
        1 for r in verifies if float(r.get("confidence", 0.0)) < low_conf
    )

    # action<->confidence consistency: does the emitted confidence sit on the
    # side the action implies, regardless of gold? (redirect=>low, verify=>high)
    n_actions = n_red + n_ver
    n_consistent = (
        sum(1 for r in redirects if float(r.get("confidence", 0.0)) < low_conf)
        + sum(1 for r in verifies if float(r.get("confidence", 0.0)) >= high_conf)
    )

    return {
        "n_redirect": n_red,
        "n_verify": n_ver,
        "redirect_appropriate_rate": (red_appropriate / n_red) if n_red else None,
        "redirect_misfire_rate": ((n_red - red_appropriate) / n_red) if n_red else None,
        "verify_appropriate_rate": (ver_appropriate / n_ver) if n_ver else None,
        "verify_misfire_rate": (ver_misfire / n_ver) if n_ver else None,
        "n_actions": n_actions,
        "n_consistent": n_consistent,
        "action_confidence_consistency_rate": (
            n_consistent / n_actions) if n_actions else None,
    }


# ----------------------------------------------------------------------------
# (3) redirect causal rate -- of emitted redirects, fraction that flipped w->r
# ----------------------------------------------------------------------------
def redirect_causal_rate(records) -> dict:
    """Of all emitted redirects, the fraction that causally flipped the answer
    from wrong to right. This is the load-bearing "is the redirect content
    actually useful" signal. None when no redirect fired.
    """
    redirects = [r for r in records if r.get("action") == "redirect"]
    n_red = len(redirects)
    n_flipped = sum(1 for r in redirects if r.get("flipped_to_right"))
    return {
        "n_redirect": n_red,
        "n_flipped": n_flipped,
        "causal_rate": (n_flipped / n_red) if n_red else None,
    }


# ----------------------------------------------------------------------------
# (4) accuracy delta -- meta on/off and vs baseline
# ----------------------------------------------------------------------------
def accuracy_delta(meta_on_acc: float, meta_off_acc: float,
                   baseline: Optional[float] = None) -> dict:
    """The headline: does turning meta on raise accuracy, and does it beat the
    base SFT baseline. meta_helps = (on > off). beats_baseline compares the
    meta-on accuracy to the base SFT (None when no baseline given).
    """
    delta_on_off = meta_on_acc - meta_off_acc
    if baseline is None:
        delta_vs_baseline = None
        beats_baseline = None
    else:
        delta_vs_baseline = meta_on_acc - baseline
        beats_baseline = meta_on_acc >= baseline
    return {
        "meta_on_acc": meta_on_acc,
        "meta_off_acc": meta_off_acc,
        "baseline": baseline,
        "delta_on_off": delta_on_off,
        "delta_vs_baseline": delta_vs_baseline,
        "meta_helps": delta_on_off > 0,
        "beats_baseline": beats_baseline,
    }


# ----------------------------------------------------------------------------
# (5) meta survival -- did meta survive RL (no forming-collapse)
# ----------------------------------------------------------------------------
def meta_survival(steps, collapse_threshold: float = 0.5,
                  drop_frac: float = 0.5) -> dict:
    """Did well-formed meta emission survive RL training (no mode collapse).

    `steps` = list of {"step": int, "wellformed_rate": float}. Collapse is
    flagged if the FINAL well-formed rate falls below an absolute floor
    (collapse_threshold) OR it dropped to less than (1 - drop_frac) of the peak
    seen during training (a relative crash, e.g. v3l 0.5 -> 0). survived is the
    negation.
    """
    steps = list(steps)
    if not steps:
        return {"survived": None, "collapsed": None, "final_rate": None,
                "min_rate": None, "peak_rate": None, "n_steps": 0}

    ordered = sorted(steps, key=lambda s: s.get("step", 0))
    rates = [float(s["wellformed_rate"]) for s in ordered]
    final = rates[-1]
    peak = max(rates)
    lo = min(rates)

    absolute_collapse = final < collapse_threshold
    relative_collapse = peak > 0 and final < (1.0 - drop_frac) * peak
    collapsed = bool(absolute_collapse or relative_collapse)

    return {
        "survived": not collapsed,
        "collapsed": collapsed,
        "final_rate": final,
        "min_rate": lo,
        "peak_rate": peak,
        "n_steps": len(rates),
    }


# ----------------------------------------------------------------------------
# composite report
# ----------------------------------------------------------------------------
def intent_report(emitted_confs, correct_flags, records,
                  meta_on_acc: float, meta_off_acc: float,
                  baseline: Optional[float] = None,
                  survival_steps=None,
                  high_conf: float = HELDOUT_HIGH_CONF,
                  low_conf: float = HELDOUT_LOW_CONF,
                  n_bins: int = 10) -> dict:
    """One structured verdict bundling all five intent metrics."""
    return {
        "calibration": confidence_calibration(emitted_confs, correct_flags,
                                              n_bins=n_bins),
        "appropriateness": action_appropriateness(records, high_conf=high_conf,
                                                  low_conf=low_conf),
        "redirect_causal": redirect_causal_rate(records),
        "accuracy": accuracy_delta(meta_on_acc, meta_off_acc, baseline=baseline),
        "survival": meta_survival(survival_steps or []),
    }
