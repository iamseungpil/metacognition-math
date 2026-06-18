"""Tests for the LIVE Stage-C RL safety tripwires (spec 2026-06-18 §3).

These pure, torch-free guards run every ~10 steps in the Stage-C loop and HALT
training if the redirect-priming reward is being gamed:
  1. gap_gaming_halt      (C2 confound) — degrading the suppressed arm
  2. band_emission_halt   (I1 collapse) — <|switch|>-row emission -> ~0
  3. behavior_routearound_halt (mask failure) — routing around the ban in prose
  4. any_halt             — runs all three, names the first firing tripwire

TDD: written before src/training/redirect_tripwires.py exists.
"""
import pytest

from src.training.redirect_tripwires import (
    gap_gaming_halt,
    band_emission_halt,
    behavior_routearound_halt,
    any_halt,
)


# --------------------------------------------------------------------------- #
# 1. gap_gaming_halt — acc_without FALLING while rmeta_pos RISING => halt       #
# --------------------------------------------------------------------------- #

def test_gap_gaming_fires_when_acc_without_drops_and_rmeta_rises():
    # acc_without falls 0.40 -> 0.30 (drop 0.10), rmeta_pos rises 0.10 -> 0.30.
    acc = [0.40, 0.35, 0.30]
    rm = [0.10, 0.20, 0.30]
    assert gap_gaming_halt(acc, rm) is True


def test_gap_gaming_does_not_fire_when_both_rise_real_skill():
    # Real skill: acc_without NON-DECREASING while rmeta_pos rises -> no halt.
    acc = [0.40, 0.45, 0.50]
    rm = [0.10, 0.20, 0.30]
    assert gap_gaming_halt(acc, rm) is False


def test_gap_gaming_does_not_fire_when_acc_drops_but_rmeta_also_drops():
    # rmeta_pos NOT rising -> not the gaming signature -> no halt.
    acc = [0.40, 0.35, 0.30]
    rm = [0.30, 0.20, 0.10]
    assert gap_gaming_halt(acc, rm) is False


def test_gap_gaming_respects_drop_eps_threshold():
    # acc_without falls only 0.005 (< drop_eps=0.02) -> below threshold -> no halt.
    acc = [0.400, 0.398, 0.395]
    rm = [0.10, 0.20, 0.30]
    assert gap_gaming_halt(acc, rm, drop_eps=0.02) is False
    # ... but a tiny drop_eps catches it.
    assert gap_gaming_halt(acc, rm, drop_eps=0.001) is True


def test_gap_gaming_rise_eps_gate():
    # rmeta_pos rises only 0.001; with rise_eps=0.05 that is NOT a rise -> no halt.
    acc = [0.40, 0.30]
    rm = [0.100, 0.101]
    assert gap_gaming_halt(acc, rm, min_steps=2, rise_eps=0.05) is False
    # default rise_eps=0.0 -> any non-negative rise counts -> halt.
    assert gap_gaming_halt(acc, rm, min_steps=2) is True


def test_gap_gaming_window_too_short_no_halt():
    assert gap_gaming_halt([0.4, 0.3], [0.1, 0.3], min_steps=3) is False
    assert gap_gaming_halt([], [], min_steps=3) is False
    assert gap_gaming_halt([0.4], [0.1], min_steps=3) is False


def test_gap_gaming_flat_acc_no_halt():
    # acc_without flat (drop == 0, not <= -drop_eps) -> no halt even if rmeta rises.
    acc = [0.40, 0.40, 0.40]
    rm = [0.10, 0.20, 0.30]
    assert gap_gaming_halt(acc, rm) is False


# --------------------------------------------------------------------------- #
# 2. band_emission_halt — sustained emission below frac * step0 rate => halt   #
# --------------------------------------------------------------------------- #

def test_band_emission_fires_when_sustained_below_third_floor():
    # step0 rate 0.6 -> floor = 0.2. All recent points below 0.2 -> collapse.
    win = [0.15, 0.10, 0.05]
    assert band_emission_halt(win, step0_band_rate=0.6) is True


def test_band_emission_does_not_fire_on_transient_dip():
    # step0 rate 0.6 -> floor 0.2. One point dips below, others above -> not sustained.
    win = [0.15, 0.30, 0.40]
    assert band_emission_halt(win, step0_band_rate=0.6) is False


def test_band_emission_does_not_fire_above_floor():
    win = [0.30, 0.25, 0.22]  # all >= 0.2 floor
    assert band_emission_halt(win, step0_band_rate=0.6) is False


def test_band_emission_custom_frac():
    # step0 0.9, frac=0.5 -> floor 0.45. window all below.
    win = [0.40, 0.30, 0.20]
    assert band_emission_halt(win, step0_band_rate=0.9, frac=0.5) is True
    # frac smaller -> floor lower -> some points clear it -> no halt.
    assert band_emission_halt(win, step0_band_rate=0.9, frac=1.0 / 3.0) is False


def test_band_emission_window_too_short_no_halt():
    assert band_emission_halt([0.05, 0.05], step0_band_rate=0.6, min_steps=3) is False
    assert band_emission_halt([], step0_band_rate=0.6, min_steps=3) is False


def test_band_emission_zero_step0_rate_no_halt():
    # No post-prime baseline emission -> floor 0 -> nothing can be below it -> no halt.
    assert band_emission_halt([0.0, 0.0, 0.0], step0_band_rate=0.0) is False


# --------------------------------------------------------------------------- #
# 3. behavior_routearound_halt — c_without behavior rate exceeds PG1 bound      #
# --------------------------------------------------------------------------- #

def test_behavior_routearound_fires_above_bound():
    assert behavior_routearound_halt(cf_behavior_rate=0.25, pg1_bound=0.2) is True


def test_behavior_routearound_no_halt_at_or_below_bound():
    assert behavior_routearound_halt(cf_behavior_rate=0.20, pg1_bound=0.2) is False
    assert behavior_routearound_halt(cf_behavior_rate=0.05, pg1_bound=0.2) is False


# --------------------------------------------------------------------------- #
# 4. any_halt — runs all three, returns (halt, reason of FIRST firing)         #
# --------------------------------------------------------------------------- #

def test_any_halt_none_fires_returns_false_empty_reason():
    halt, reason = any_halt(
        gap_gaming=([0.40, 0.45, 0.50], [0.10, 0.20, 0.30]),
        band=([0.30, 0.30, 0.30], 0.6),
        behavior=(0.05, 0.2),
    )
    assert halt is False
    assert reason == ""


def test_any_halt_gap_gaming_reason():
    halt, reason = any_halt(
        gap_gaming=([0.40, 0.35, 0.30], [0.10, 0.20, 0.30]),
        band=([0.30, 0.30, 0.30], 0.6),
        behavior=(0.05, 0.2),
    )
    assert halt is True
    assert reason == "gap_gaming"


def test_any_halt_band_reason_when_gap_ok():
    halt, reason = any_halt(
        gap_gaming=([0.40, 0.45, 0.50], [0.10, 0.20, 0.30]),
        band=([0.10, 0.05, 0.05], 0.6),
        behavior=(0.05, 0.2),
    )
    assert halt is True
    assert reason == "band_emission"


def test_any_halt_behavior_reason_when_gap_and_band_ok():
    halt, reason = any_halt(
        gap_gaming=([0.40, 0.45, 0.50], [0.10, 0.20, 0.30]),
        band=([0.30, 0.30, 0.30], 0.6),
        behavior=(0.30, 0.2),
    )
    assert halt is True
    assert reason == "behavior_routearound"


def test_any_halt_reports_first_in_priority_order():
    # All three fire; reason must be the first checked (gap_gaming).
    halt, reason = any_halt(
        gap_gaming=([0.40, 0.35, 0.30], [0.10, 0.20, 0.30]),
        band=([0.10, 0.05, 0.05], 0.6),
        behavior=(0.30, 0.2),
    )
    assert halt is True
    assert reason == "gap_gaming"


def test_any_halt_short_windows_do_not_fire():
    # Empty/short windows -> no halt from gap or band (min_steps), behavior ok.
    halt, reason = any_halt(
        gap_gaming=([], []),
        band=([], 0.6),
        behavior=(0.05, 0.2),
    )
    assert halt is False
    assert reason == ""
