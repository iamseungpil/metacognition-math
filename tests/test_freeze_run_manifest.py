"""Tests for the pre-launch manifest, focused on the trained-region measurement.

The manifest's job is to make a one-arm drift visible before results are looked
at. Two of its numbers carry that weight and are easy to get subtly wrong:

  * the TRAINED region, which starts at ``prefix_split_char`` because ``sft.py``
    masks the prompt and the flawed prefix. Measuring the whole assistant turn
    instead reports 9.7% where the real figure is 29.5% - a difference large
    enough to change whether anyone looks twice.
  * the set of config keys that differ between arms, which is the actual
    tripwire: four differences are the design, a fifth is a bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_run_manifest import (  # noqa: E402
    GRADER_MODES,
    _length_stats,
    arm_symmetry,
    assistant_and_trained_text,
    token_exposure,
)


def _row(assistant: str, cut: int, as_json: bool = True):
    import json

    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": assistant}]
    return {"messages": json.dumps(msgs) if as_json else msgs, "prefix_split_char": cut}


def test_trained_text_starts_at_the_split():
    a, t = assistant_and_trained_text(_row("MASKEDxxxTRAINED", 9))
    assert a == "MASKEDxxxTRAINED"
    assert t == "TRAINED"


def test_zero_split_trains_the_whole_turn():
    """VERIFY rows carry an empty wrong_prefix, so nothing is masked past the prompt."""
    a, t = assistant_and_trained_text(_row("all of it", 0))
    assert t == a


def test_messages_may_arrive_as_a_list_not_a_json_string():
    a, t = assistant_and_trained_text(_row("MASKEDkeep", 6, as_json=False))
    assert (a, t) == ("MASKEDkeep", "keep")


def test_missing_split_column_is_treated_as_no_mask():
    import json

    row = {"messages": json.dumps([{"role": "assistant", "content": "text"}])}
    assert assistant_and_trained_text(row)[1] == "text"


def test_length_stats_totals_and_order():
    s = _length_stats([5, 1, 3])
    assert s["total"] == 9 and s["max"] == 5 and s["p50"] == 3


def test_token_exposure_reports_the_ratio_between_arms():
    manifest = {
        "arms": {
            "control": {"corpus": {"token_len_trained": {"total": 292353, "mean": 165.8}}},
            "meta": {"corpus": {"token_len_trained": {"total": 378614, "mean": 214.8}}},
        }
    }
    te = token_exposure(manifest)
    assert te["available"] is True
    # the 0727 measurement, pinned: the trained region differs by ~29.5%
    assert abs(te["ratio"]["meta/control"] - 1.2951) < 1e-3
    assert abs(te["mean_delta_tokens"] - 49.0) < 0.5


def test_token_exposure_says_why_it_could_not_compute():
    """Without --tokenizer there are no trained-token stats, and silence there
    would look identical to 'the arms match'."""
    te = token_exposure({"arms": {"control": {"corpus": {}}, "meta": {"corpus": {}}}})
    assert te["available"] is False and "0/2" in te["reason"]


def test_arm_symmetry_lists_every_differing_key():
    manifest = {
        "arms": {
            "control": {"config": {"lr": 1e-5, "data": "twin.parquet", "epochs": 3}},
            "meta": {"config": {"lr": 1e-5, "data": "meta.parquet", "epochs": 2}},
        }
    }
    diffs = arm_symmetry(manifest)
    assert len(diffs) == 2
    assert any(d.startswith("data:") for d in diffs)
    assert any(d.startswith("epochs:") for d in diffs)  # the one that would be a bug


def test_arm_symmetry_refuses_anything_but_a_pair():
    assert "expected exactly 2 arms" in arm_symmetry({"arms": {"only": {}}})[0]


def test_both_grader_modes_are_declared():
    """A5 of the pre-registration requires reporting under both, so both must be
    nameable at freeze time rather than invented afterwards."""
    assert set(GRADER_MODES) == {"format_fair", "strict_boxed"}
