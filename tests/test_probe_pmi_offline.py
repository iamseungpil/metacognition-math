"""probe_pmi_offline — parsing + report assembly with STUBBED logprobs (no GPU).

Covers the tokenizer-free half of the probe: rollout parsing (prefix/meta/C
split, malformed skips, entangled flag), data loading (problem grouping,
mixed-first truncation), shuffle-partner selection, the numpy stats helpers,
and assemble_report's kill-criteria verdict logic (spec §3 a-e). The GPU
scoring path is exercised by `--smoke` on the real checkpoint, not here.
"""

import json
import math

import numpy as np
import pytest

from scripts.probe_pmi_offline import (
    PLACEBO_META,
    PMI_AGG_METHODS,
    assemble_report,
    format_report_text,
    load_rollouts,
    paired_t,
    parse_rollout,
    pick_shuffle_partners,
    rank_auc,
)
from src.metacot.prompt import META_END, META_START


# ── fixtures ─────────────────────────────────────────────────────────────────
def _record(meta_inner="\nconfidence: 0.5\nassessment: looks fine\n",
            continuation="\nSo the total is 18.\n</think>\n\n\\boxed{18}",
            correct=True, sample_idx=0, **extra):
    completion = f"<think>\nLet me compute 6 * 3.{META_START}{meta_inner}{META_END}{continuation}"
    rec = {
        "benchmark": "gsm8k",
        "question": "What is 6 * 3?",
        "gold_answer": "18",
        "completion": completion,
        "answer_extracted": "18",
        "is_correct": correct,
        "sample_idx": sample_idx,
    }
    rec.update(extra)
    return rec


# ── parse_rollout ────────────────────────────────────────────────────────────
def test_parse_rollout_splits_prefix_meta_continuation():
    row = parse_rollout(_record(), problem_id=7)
    assert row["completion_prefix"] == "<think>\nLet me compute 6 * 3."
    assert row["meta_text"].startswith(META_START) and row["meta_text"].endswith(META_END)
    assert "confidence: 0.5" in row["meta_text"]
    assert row["continuation_text"] == "\nSo the total is 18.\n</think>\n\n\\boxed{18}"
    assert row["problem_id"] == 7 and row["correct"] is True
    assert row["boxed_answer"] == "18" and row["benchmark"] == "gsm8k"


def test_parse_rollout_entangled_flag_uses_signature_regex():
    # field-label lines (confidence:/assessment:/action:) -> entangled (spec I5)
    assert parse_rollout(_record(), 0)["entangled"] is True
    prose = _record(meta_inner="\nI should double-check the multiplication here.\n")
    assert parse_rollout(prose, 0)["entangled"] is False


def test_parse_rollout_skips_malformed():
    no_meta = _record()
    no_meta["completion"] = "<think>\nplain reasoning, no tags\n</think>\n\\boxed{18}"
    assert parse_rollout(no_meta, 0) is None
    truncated = _record()  # 16k-cutoff population: open tag, no close tag
    truncated["completion"] = f"<think>\nwork {META_START}\nconfidence: 0."
    assert parse_rollout(truncated, 0) is None
    assert parse_rollout(_record(continuation="  \n"), 0) is None  # nothing to score
    assert parse_rollout({"completion": None}, 0) is None


# ── load_rollouts ────────────────────────────────────────────────────────────
def _write_data(tmp_path, records):
    p = tmp_path / "eval.json"
    p.write_text(json.dumps({"summary": {}, "results": records}))
    return str(p)


def test_load_rollouts_groups_problems_and_skips_malformed(tmp_path):
    bad = _record(sample_idx=1)
    bad["completion"] = "no meta here"
    records = [
        _record(sample_idx=0), _record(sample_idx=1),           # problem 0
        _record(sample_idx=0), bad,                             # problem 1 (1 bad)
        _record(sample_idx=0, correct=False),                   # problem 2
    ]
    rows = load_rollouts(_write_data(tmp_path, records))
    assert [r["problem_id"] for r in rows] == [0, 0, 1, 2]
    assert [r["correct"] for r in rows] == [True, True, True, False]


def test_load_rollouts_max_rows_prefers_mixed_problems(tmp_path):
    records = [
        _record(sample_idx=0), _record(sample_idx=1),                  # pid 0 all-correct
        _record(sample_idx=0), _record(sample_idx=1, correct=False),  # pid 1 MIXED
    ]
    rows = load_rollouts(_write_data(tmp_path, records), max_rows=2)
    # head-slice would return pid 0 (one-class); mixed-first keeps AUC computable
    assert [r["problem_id"] for r in rows] == [1, 1]
    assert sorted(r["correct"] for r in rows) == [False, True]


# ── pick_shuffle_partners ────────────────────────────────────────────────────
def test_shuffle_partner_cyclic_within_problem():
    rows = [
        {"problem_id": 0, "continuation_text": "a"},
        {"problem_id": 0, "continuation_text": "b"},
        {"problem_id": 0, "continuation_text": "c"},
    ]
    assert pick_shuffle_partners(rows) == [1, 2, 0]


def test_shuffle_partner_falls_back_to_other_problem():
    rows = [
        {"problem_id": 0, "continuation_text": "a"},   # singleton -> fallback
        {"problem_id": 1, "continuation_text": "x"},
        {"problem_id": 1, "continuation_text": "x"},   # identical sibling -> fallback
    ]
    partners = pick_shuffle_partners(rows, seed=0)
    assert rows[partners[0]]["problem_id"] != 0
    assert rows[partners[1]]["problem_id"] != 1
    assert rows[partners[2]]["problem_id"] != 1


# ── stats helpers ────────────────────────────────────────────────────────────
def test_rank_auc_separation_and_ties():
    assert rank_auc([1.0, 2.0, -1.0, -2.0], [True, True, False, False]) == 1.0
    assert rank_auc([-1.0, -2.0, 1.0, 2.0], [True, True, False, False]) == 0.0
    assert rank_auc([1.0, 1.0, 1.0, 1.0], [True, True, False, False]) == 0.5
    assert math.isnan(rank_auc([1.0, 2.0], [True, True]))  # one-class


def test_paired_t_directions():
    t, p = paired_t(np.full(50, 0.5) + np.linspace(-0.1, 0.1, 50))
    assert t > 5 and p < 1e-4
    t0, p0 = paired_t(np.linspace(-1, 1, 50))
    assert abs(t0) < 1e-9 and p0 == pytest.approx(0.5)
    assert math.isnan(paired_t([1.0])[0])  # n < 2


# ── assemble_report (spec §3 a-e) ────────────────────────────────────────────
def _stub_row(delta, correct, entangled, T=4):
    """Span-aligned arm logprobs with a constant per-token delta."""
    lw = np.full(T, -1.0)
    return {
        "meta_text": "thinking once more about the structure of this problem",
        "continuation_text": "therefore the final count follows from the recurrence",
        "correct": correct, "entangled": entangled,
        "logp_with": lw, "logp_without": lw - delta,
    }


def _passing_passes(n=60, n_correct=40):
    """Correct rows get helpful metas (+1.5/tok), wrong rows harmful (-0.5/tok);
    placebo and shuffle deltas are 0 -> all three kill criteria pass."""
    real, placebo, shuffle = [], [], []
    for i in range(n):
        correct, entangled = i < n_correct, i % 2 == 0
        real.append(_stub_row(1.5 if correct else -0.5, correct, entangled))
        placebo.append(_stub_row(0.0, correct, entangled))
        shuffle.append(_stub_row(0.0, correct, entangled))
    return real, placebo, shuffle


def test_assemble_report_pass_scenario_all_sections():
    real, placebo, shuffle = _passing_passes()
    report = assemble_report(real, placebo, shuffle)
    # (a) distribution per method
    for m in PMI_AGG_METHODS:
        assert report["delta_stats"][m]["n"] == 60
    assert report["delta_stats"]["mean"]["mean"] == pytest.approx(
        (40 * 1.5 - 20 * 0.5) / 60)
    # (b) perfect separation, on the entangled split too (>= MIN_SPLIT_N rows)
    for m in PMI_AGG_METHODS:
        assert report["auc"][m]["overall"] == 1.0
        assert report["auc"][m]["entangled"] == 1.0
        assert report["auc"][m]["n_entangled"] == 30
    # (c) real beats placebo (paired, one-sided)
    plc = report["placebo"][report["verdict"]["method"]]
    assert plc["mean_diff"] > 0 and plc["p_one_sided"] < 0.05
    # (d) shuffle collapses
    assert report["shuffle"][report["verdict"]["method"]]["collapse_ratio"] < 0.25
    # (e) recommendation + verdict
    assert report["recommendation"]["method"] in PMI_AGG_METHODS
    assert report["recommendation"]["clip_c"] > 0
    assert report["verdict"]["auc_split_used"] == "entangled"
    assert report["verdict"]["overall"] == "PASS"


def test_assemble_report_fails_when_real_equals_placebo():
    real, _, shuffle = _passing_passes()
    placebo = [dict(r) for r in real]  # placebo identical -> KILL 1 (spec C1)
    report = assemble_report(real, placebo, shuffle)
    assert report["verdict"]["placebo_pass"] is False
    assert report["verdict"]["overall"] == "FAIL"


def test_assemble_report_fails_when_shuffle_does_not_collapse():
    real, placebo, _ = _passing_passes()
    shuffle = [dict(r) for r in real]  # shuffled C scores like real -> KILL 2
    report = assemble_report(real, placebo, shuffle)
    assert report["verdict"]["shuffle_pass"] is False
    assert report["verdict"]["overall"] == "FAIL"


def test_assemble_report_auc_kill_on_entangled_split():
    # entangled rows carry NO correct/wrong signal (delta +1 regardless) while
    # clean rows separate perfectly -> overall AUC high but KILL 3 fires
    real, placebo, shuffle = [], [], []
    for i in range(60):
        correct, entangled = i % 2 == 0, i < 30
        delta = 1.0 if (entangled or correct) else -1.0
        real.append(_stub_row(delta, correct, entangled))
        placebo.append(_stub_row(0.0, correct, entangled))
        shuffle.append(_stub_row(0.0, correct, entangled))
    report = assemble_report(real, placebo, shuffle)
    m = report["verdict"]["method"]
    assert report["auc"][m]["entangled"] == 0.5
    assert report["verdict"]["auc_entangled_pass"] is False
    assert report["verdict"]["overall"] == "FAIL"


def test_assemble_report_tolerates_alignment_failures_and_falls_back():
    real, placebo, shuffle = _passing_passes(n=10, n_correct=5)
    real[0] = {**real[0], "alignment_failed": True,
               "logp_with": None, "logp_without": None}
    report = assemble_report(real, placebo, shuffle)
    assert report["alignment_failures"]["real"] == 1
    assert report["delta_stats"]["mean"]["n"] == 9  # NaN row excluded
    # < MIN_SPLIT_N entangled rows -> verdict falls back to the overall AUC
    assert report["verdict"]["auc_split_used"] == "overall"


def test_format_report_text_has_sections_and_verdict():
    real, placebo, shuffle = _passing_passes()
    text = format_report_text(assemble_report(real, placebo, shuffle))
    for marker in ("(a)", "(b)", "(c)", "(d)", "(e)", "VERDICT: PASS"):
        assert marker in text
    smoke = format_report_text(assemble_report(real, placebo, shuffle, smoke=True))
    assert "[SMOKE]" in smoke and "VERDICT:" in smoke


def test_placebo_meta_is_tag_wrapped_and_contentless():
    assert PLACEBO_META.startswith(META_START) and PLACEBO_META.endswith(META_END)
    assert "Let me continue." in PLACEBO_META
    # contentless: must NOT match the entangled-signature field labels
    from src.training.dcpo_region import _has_meta_signature
    assert _has_meta_signature(PLACEBO_META) is False
