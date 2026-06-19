"""Stage-A counterfactual redirect harvest (spec 2026-06-18 REV-6 §4.A).

Mine failed rollouts: splice a `<|switch|>` redirect at the point a wrong trace
went bad, regenerate ANSWER-BLIND with k samples in 3 arms, and keep only pairs
whose redirect CAUSALLY flips wrong->right (Arm R beats both controls by a
margin). The kept traces prime the behavior (Stage B) so RL has redirect to
grade; the counterfactual filter guarantees the redirect is functional (so PMI /
the CF reward is positive by construction, not hollow).

Arms (prefix-forced continuation from the same wrong prefix, k samples each):
  R  = `<|switch|>` redirect block then continue
  N' = semantically-null meta (confidence restatement, NO switch) then continue
  Nc = compute-matched plain continuation, no meta

Generation is delegated to the existing rollout infra (run_online_sdpo_regen);
this module holds the GPU-free, unit-tested DECISION logic + a main() that wires
generation. Grading uses rewards._check_correctness (answer-blind: gold only
filters acceptance, never shown to the model).
"""
import re

# REV-6 §4.A.4 / §4.A.5 defaults.
SPLICE_LO, SPLICE_HI = 0.30, 0.70
ACCEPT_MARGIN = 0.50          # R - max(N', Nc) >= margin
CONF_MAX = 0.5                # redirect must lower confidence
_SWITCH = "<|switch|>"
_CONF_RE = re.compile(r"confidence:\s*([0-9.]+)", re.IGNORECASE)
_META_OPEN, _META_CLOSE = "<|meta|>", "<|/meta|>"


def well_formed_redirect(text: str) -> bool:
    """A single well-formed redirect meta block: contains <|switch|>, exactly one
    meta block, and lowered confidence (< CONF_MAX)."""
    if text is None:
        return False
    if text.count(_META_OPEN) != 1 or text.count(_META_CLOSE) != 1:
        return False
    if _SWITCH not in text:
        return False
    m = _CONF_RE.search(text)
    if not m:
        return False
    try:
        return float(m.group(1)) < CONF_MAX
    except ValueError:
        return False


def splice_index(n_tokens: int, frac: float) -> int:
    """Cut index for the wrong prefix at `frac` of the trace, clamped into the
    [SPLICE_LO, SPLICE_HI] band and never zero-length."""
    frac = max(SPLICE_LO, min(SPLICE_HI, frac))
    return max(1, int(round(n_tokens * frac)))


def arm_rate(grades) -> float:
    """Fraction correct over k samples (grades = list of 0/1)."""
    return (sum(grades) / len(grades)) if grades else 0.0


import math

MIN_K = 4  # too-few samples => INSUFFICIENT, never accept (intent-check C)


def lower_ci_diff(r_grades, ctrl_grades, z: float = 1.645) -> float:
    """One-sided (95%) lower confidence bound on (rate_R - rate_ctrl), normal
    approx. Used instead of a raw point estimate so a noisy small-k gap cannot
    be accepted (intent-check wbitrlry0 C; spec §4.A.5 'lower-CI-bound ≥ margin')."""
    nr, nc = len(r_grades), len(ctrl_grades)
    if nr == 0 or nc == 0:
        return float("-inf")
    pr, pc = arm_rate(r_grades), arm_rate(ctrl_grades)
    se = math.sqrt(pr * (1 - pr) / nr + pc * (1 - pc) / nc)
    return (pr - pc) - z * se


def accept_redirect(r_grades, nprime_grades, nc_grades, bprime_grades=None,
                    margin: float = ACCEPT_MARGIN, min_k: int = MIN_K) -> bool:
    """Counterfactual acceptance on the lower-CI bound of (R - best control).
    Controls now INCLUDE B' (plain-prose backtracking, <|switch|> masked) so the
    harvest estimand matches the eval's R-B' — crediting redirect CONTENT, not a
    free second attempt (intent-check wbitrlry0 D). N'=null-meta, Nc=plain. Tiny
    k => reject (INSUFFICIENT, intent-check C)."""
    if len(r_grades) < min_k:
        return False
    controls = [nprime_grades, nc_grades]
    if bprime_grades is not None:
        controls.append(bprime_grades)
    # worst case over controls = the smallest lower-CI gap must still clear margin
    return min(lower_ci_diff(r_grades, ctrl) for ctrl in controls) >= margin


def expected_yield(emission_rate: float, in_band_frac: float, accept_prob: float, pool_size: int) -> int:
    """PG0 projected accepted-redirect count (spec §0 PG0)."""
    return int(emission_rate * in_band_frac * accept_prob * pool_size)


def _pct(xs, q):
    """q-quantile of xs by nearest-rank (xs non-empty)."""
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def raw_yield_stats(grade_triples, margins=(0.5, 0.3, 0.2, 0.1, 0.0)):
    """PRE-GATE diagnostics for the PG0 accept stage (no new generation — pure
    arithmetic on grade lists already computed in the pilot loop).

    ``grade_triples`` = list of ``(r_grades, nprime_grades, nc_grades)``, each a
    list of 0/1 over the k answer-blind samples of one spliced wrong rollout.

    The official verdict only keeps ``accepted`` = #(splices clearing the strict
    margin=0.5 lower-CI gate), so a 0 is ambiguous. These stats separate:
      * MODEL-CANNOT-REDIRECT (warmup needed): mean_gap_rc ~ 0, saves_rc_frac ~
        0.5 (chance), lci_rc_max ~ 0  -> redirect changes nothing.
      * GATE-TOO-STRICT: mean_gap_rc > 0, saves_rc_frac > 0.5, accepts appear at
        margin 0.2-0.3  -> redirect helps, the 0.5 bar just clipped it.
      * ANY-INJECTION (not redirect CONTENT): mean_gap_rc > 0 but mean_gap_rn ~ 0
        -> R only matches the null-meta arm, the redirect text isn't what helped.

    gap_rc = p_R - p_Nc (redirect vs plain continuation);
    gap_rn = p_R - p_N'  (redirect vs null-meta injection);
    lci_rc = one-sided 95% lower bound on (p_R - p_Nc).
    ``accept_at_margin`` re-runs the real accept gate at each margin (margin 0.0
    is degeneracy-prone: both-all-wrong gives SE=0 -> lci=0 -> counts; read 0.1+).
    """
    n = len(grade_triples)
    if n == 0:
        return {"n": 0}
    pr = [arm_rate(r) for r, _, _ in grade_triples]
    pn = [arm_rate(nn) for _, nn, _ in grade_triples]
    pc = [arm_rate(c) for _, _, c in grade_triples]
    gap_rc = [a - b for a, b in zip(pr, pc)]
    gap_rn = [a - b for a, b in zip(pr, pn)]
    lci_rc = [lower_ci_diff(r, c) for r, _, c in grade_triples]
    saves_rc = sum(1 for g in gap_rc if g > 0)
    saves_rc_strong = sum(1 for g in gap_rc if g >= 0.25)
    accept_at = {
        str(m): sum(
            1 for r, nn, c in grade_triples
            if accept_redirect(r, nn, c, bprime_grades=c, margin=m)
        )
        for m in margins
    }
    return {
        "n": n,
        "mean_r_rate": sum(pr) / n,
        "mean_nprime_rate": sum(pn) / n,
        "mean_nc_rate": sum(pc) / n,
        "mean_gap_rc": sum(gap_rc) / n,
        "mean_gap_rn": sum(gap_rn) / n,
        "saves_rc": saves_rc,
        "saves_rc_frac": saves_rc / n,
        "saves_rc_strong": saves_rc_strong,
        "gap_rc_p50": _pct(gap_rc, 0.5),
        "gap_rc_p90": _pct(gap_rc, 0.9),
        "gap_rc_max": max(gap_rc),
        "lci_rc_p90": _pct(lci_rc, 0.9),
        "lci_rc_max": max(lci_rc),
        "accept_at_margin": accept_at,
    }


def main():  # pragma: no cover - wires GPU generation; logic above is unit-tested
    raise SystemExit(
        "Wire generation via scripts/run_online_sdpo_regen.py: (1) roll out the SFT "
        "init on the train pool, keep wrong rollouts in pass-rate band [0.125,0.5]; "
        "(2) for each, splice at splice_index(); (3) prefix-forced regenerate arms "
        "R/N'/Nc (k=4-8) answer-blind; (4) grade with _check_correctness; "
        "(5) accept_redirect(); fresh-holdout re-confirm; (6) <=2/problem; record "
        "source ids for eval-disjointness. Run scripts/pg0_yield_pilot.py first."
    )


if __name__ == "__main__":
    main()
