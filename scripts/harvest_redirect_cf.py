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
