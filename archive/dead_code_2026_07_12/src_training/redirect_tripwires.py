"""LIVE Stage-C RL safety tripwires for the redirect-priming reward (spec
2026-06-18-redirect-priming-counterfactual-design §3 "LIVE RL tripwires";
intent-check w4udybnbv C7).

The redirect-priming reward (`redirect_cf.redirect_cf_rmeta`) credits a redirect
by the per-row counterfactual `c_with - c_without`. That reward is gameable in
three ways that the per-row gate logic CANNOT see — they are only visible over a
WINDOW of recent steps or against pre-registered bounds:

  1. GAP-GAMING (C2 confound): instead of raising `c_with` (real backtracking),
     the model DEGRADES the suppressed `c_without` arm to widen the gap and farm
     `rmeta_pos`. Visible as frozen-hard-band `acc_without` FALLING while
     `rmeta_pos_rate` RISES over a window.
  2. COLLAPSE (I1, the dominant prior failure): band (`<|switch|>`-row) emission
     rate decays toward 0. Visible as emission sustained below ~1/3 of the
     post-prime step-0 rate over a window.
  3. MASK-FAILURE (route-around): the model emits redirect BEHAVIOR in plain
     prose in the `c_without` arm despite the `<|switch|>` token ban, so the
     counterfactual is contaminated. Visible as the c_without-arm behavior rate
     exceeding PG1's pre-registered false-negative bound.

These are PURE functions operating on lists of per-step scalars: NO torch, NO
I/O, NO global state. The Stage-C loop calls them every ~10 steps and HALTs
training if any fires (see module docstring of redirect_cf for why the reward
must NOT be wired before these exist).
"""

from typing import List, Tuple


def gap_gaming_halt(
    acc_without_window: List[float],
    rmeta_pos_window: List[float],
    min_steps: int = 3,
    drop_eps: float = 0.02,
    rise_eps: float = 0.0,
) -> bool:
    """C2 GAP-GAMING guard.

    Halt iff, over the recent window, the frozen-hard-band ``acc_without`` is
    FALLING while ``rmeta_pos_rate`` is RISING — the signature of the model
    degrading the suppressed arm to farm reward instead of raising ``c_with``.
    A real skill RAISES ``c_with`` WITHOUT lowering ``acc_without``.

    Returns True (halt) iff ALL of:
      - len(window) >= min_steps (need enough steps to see a trend), AND
      - acc_without[-1] - acc_without[0] <= -drop_eps   (sustained FALL), AND
      - rmeta_pos_window[-1] - rmeta_pos_window[0] >= rise_eps  (RISE).
    """
    n = min(len(acc_without_window), len(rmeta_pos_window))
    if n < min_steps:
        return False
    acc_delta = float(acc_without_window[n - 1]) - float(acc_without_window[0])
    rm_delta = float(rmeta_pos_window[n - 1]) - float(rmeta_pos_window[0])
    return (acc_delta <= -drop_eps) and (rm_delta >= rise_eps)


def band_emission_halt(
    band_emit_window: List[float],
    step0_band_rate: float,
    frac: float = 1.0 / 3.0,
    min_steps: int = 3,
) -> bool:
    """I1 PRIMARY collapse guard.

    Halt iff the band (`<|switch|>`-row) emission rate has fallen below
    ``frac * step0_band_rate`` (e.g. below 1/3 of the post-prime step-0 rate) and
    STAYED there — every recent point in the window is below the floor. The
    dominant prior failure mode is emission decaying to ~0, so we require a
    SUSTAINED breach (not a transient dip).

    If ``step0_band_rate <= 0`` there is no baseline to fall below -> never halt.
    """
    if len(band_emit_window) < min_steps:
        return False
    floor = float(frac) * float(step0_band_rate)
    if floor <= 0.0:
        return False
    return all(float(x) < floor for x in band_emit_window)


def behavior_routearound_halt(cf_behavior_rate: float, pg1_bound: float) -> bool:
    """MASK-FAILURE (route-around) guard.

    Halt iff the redirect-BEHAVIOR rate in the suppressed (`c_without`) arm
    exceeds PG1's pre-registered false-negative bound — i.e. the model is routing
    around the `<|switch|>` token ban in plain prose during training, so
    ``c_without`` is contaminated and the counterfactual is invalid.
    """
    return float(cf_behavior_rate) > float(pg1_bound)


def any_halt(
    *,
    gap_gaming: Tuple[List[float], List[float]],
    band: Tuple[List[float], float],
    behavior: Tuple[float, float],
) -> Tuple[bool, str]:
    """Run all three tripwires; return (halt, reason).

    ``reason`` names the FIRST firing tripwire in priority order
    (gap_gaming -> band_emission -> behavior_routearound), or "" if none fire.

    Inputs (kwargs):
      gap_gaming = (acc_without_window, rmeta_pos_window)
      band       = (band_emit_window, step0_band_rate)
      behavior   = (cf_behavior_rate, pg1_bound)
    """
    acc_without_window, rmeta_pos_window = gap_gaming
    band_emit_window, step0_band_rate = band
    cf_behavior_rate, pg1_bound = behavior

    if gap_gaming_halt(acc_without_window, rmeta_pos_window):
        return True, "gap_gaming"
    if band_emission_halt(band_emit_window, step0_band_rate):
        return True, "band_emission"
    if behavior_routearound_halt(cf_behavior_rate, pg1_bound):
        return True, "behavior_routearound"
    return False, ""
