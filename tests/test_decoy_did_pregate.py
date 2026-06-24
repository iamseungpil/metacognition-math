"""CPU unit tests for the decoupled decoy-DiD pre-gate pure helpers."""
import math

from src.eval.decoy_did_pregate import (
    parse_body_meta, make_placebo, build_continuation, did, rank_auc,
    META_OPEN, META_CLOSE, PLACEBO_INNER,
)


def test_parse_body_meta_extracts_inner_and_through_close():
    txt = f"reasoning here {META_OPEN}confidence: 0.88\ndecision: verify{META_CLOSE} tail answer"
    body, inner = parse_body_meta(txt)
    assert body.endswith(META_CLOSE)
    assert "tail answer" not in body  # post-meta continuation dropped
    assert inner == "confidence: 0.88\ndecision: verify"


def test_parse_body_meta_none_without_complete_block():
    assert parse_body_meta("no meta at all") is None
    assert parse_body_meta(f"open only {META_OPEN}dangling") is None


def test_make_placebo_replaces_inner_keeps_tags():
    txt = f"body {META_OPEN}SECRET GUIDANCE{META_CLOSE}"
    body, inner = parse_body_meta(txt)
    plac = make_placebo(body, inner)
    assert "SECRET GUIDANCE" not in plac
    assert META_OPEN in plac and META_CLOSE in plac
    assert PLACEBO_INNER in plac
    assert plac.startswith("body ")


def test_build_continuation_differs_only_in_answer():
    g = build_continuation("15")
    d = build_continuation("16")
    assert "15" in g and "16" in d
    # identical structure apart from the answer token
    assert g.replace("15", "X") == d.replace("16", "X")


def test_did_boilerplate_is_zero():
    # meta lifts gold and decoy EQUALLY (and same as placebo) -> DiD 0
    assert did(-2.0, -2.0, -3.0, -3.0) == 0.0
    # meta favours gold over decoy MORE than placebo does -> DiD > 0
    assert did(-1.0, -3.0, -2.0, -2.5) > 0
    # meta favours decoy (confidently wrong) -> DiD < 0
    assert did(-3.0, -1.0, -2.0, -2.0) < 0


def test_rank_auc_perfect_and_chance():
    # perfectly separable: positives all higher
    assert rank_auc([0.1, 0.2, 0.9, 1.0], [0, 0, 1, 1]) == 1.0
    # reversed
    assert rank_auc([0.9, 1.0, 0.1, 0.2], [0, 0, 1, 1]) == 0.0
    # one class missing -> None
    assert rank_auc([0.1, 0.2], [1, 1]) is None


def test_rank_auc_ties_average_to_half():
    # all identical scores -> AUC 0.5 (tie-averaged)
    assert abs(rank_auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) - 0.5) < 1e-9


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
