"""Smoke tests for scripts/analyze_ev_signature_meta.py.

Exercises the four EV signature metrics on synthetic SampleTrace objects with
known marker positions. No model / GPU required. Also asserts that the
HFForwardExtractor class imports successfully.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_ev_signature_meta as mod  # noqa: E402


def _make_trace(
    sample_id: int,
    is_correct: bool,
    T: int = 40,
    span=(10, 14),
    vocab_size: int = 1024,
    difficulty: float = 0.5,
    seed: int = 0,
) -> mod.SampleTrace:
    rng = np.random.RandomState(seed)
    # Baseline entropy ~2.0 nats, drop to ~0.5 after the marker to simulate
    # the "resolving uncertainty" pattern the metrics look for.
    entropy = rng.uniform(1.5, 2.5, size=T).astype(np.float64)
    if span[1] + 6 < T:
        entropy[span[1] + 1 : span[1] + 6] = rng.uniform(0.3, 0.7, size=5)
    top1 = rng.uniform(0.2, 0.6, size=T).astype(np.float64)
    top2 = top1 * rng.uniform(0.1, 0.5, size=T)
    return mod.SampleTrace(
        sample_id=sample_id,
        is_correct=is_correct,
        benchmark="synthetic",
        entropy=entropy,
        top1_prob=top1,
        top2_prob=top2,
        difficulty_proxy=difficulty,
        spans=[mod.MarkerSpan(start=span[0], end=span[1])],
        vocab_size=vocab_size,
    )


def test_imports_module_and_extractor_class():
    # HFForwardExtractor class itself imports; we don't instantiate (needs GPU).
    assert hasattr(mod, "HFForwardExtractor")
    assert hasattr(mod, "compute_delta_h_window")
    assert hasattr(mod, "compute_mahalanobis_d")
    assert hasattr(mod, "compute_conditional_mi")
    assert hasattr(mod, "compute_c_t_single")


def test_find_meta_and_confidence_spans():
    t1 = "abc <|meta|>hello<|/meta|> def <|meta|>bar<|/meta|> end"
    spans1 = mod.find_meta_spans_in_text(t1)
    assert len(spans1) == 2
    t2 = "foo confidence: 0.7 bar Confidence: 0.95 baz"
    spans2 = mod.find_confidence_spans_in_text(t2)
    assert len(spans2) == 2


def test_delta_h_is_finite_and_respects_direction():
    trace = _make_trace(0, is_correct=True, seed=1)
    d = mod.compute_delta_h_window(trace.entropy, trace.spans[0])
    assert d is not None
    assert math.isfinite(d)
    # By construction post-marker entropy is lower -> delta_H should be negative.
    assert d < 0


def test_aggregate_delta_h_splits_correctness():
    traces = [
        _make_trace(0, True, seed=1),
        _make_trace(1, True, seed=2),
        _make_trace(2, False, seed=3),
    ]
    stats = mod.aggregate_delta_h(traces)
    assert stats["n_all"] == 3
    assert stats["n_correct"] == 2
    assert stats["n_incorrect"] == 1
    assert math.isfinite(stats["mean_all"])
    assert math.isfinite(stats["mean_correct"])
    assert math.isfinite(stats["mean_incorrect"])


def test_mahalanobis_d_nonnegative_with_ci():
    traces = [_make_trace(i, is_correct=(i % 2 == 0), seed=i + 1) for i in range(6)]
    out = mod.compute_mahalanobis_d(traces, rng_seed=7, n_bootstrap=50)
    assert out["d_m"] >= 0.0
    assert out["ci_low"] >= 0.0
    assert out["ci_high"] >= out["ci_low"]
    assert out["n_ev"] > 0
    assert out["n_neutral"] > 0


def test_mi_conditional_nonneg_and_bounded():
    traces = []
    for i in range(9):
        # Mix of correctness and difficulty.
        traces.append(_make_trace(
            i, is_correct=(i % 3 != 0), difficulty=(i / 9.0), seed=i + 10,
        ))
    out = mod.compute_conditional_mi(traces)
    assert out["i_mi"] >= 0.0
    # Binary x binary -> MI <= log(2) nats ~ 0.6931.
    assert out["i_mi"] <= math.log(2) + 1e-6
    assert out["n_samples"] == 9
    assert set(out["per_tercile"].keys()) == set(mod.DIFFICULTY_TERCILE_LABELS)


def test_c_t_finite_and_cohen_d_defined():
    traces = [
        _make_trace(0, True, seed=1),
        _make_trace(1, True, seed=2),
        _make_trace(2, False, seed=3),
        _make_trace(3, False, seed=4),
    ]
    stats = mod.aggregate_c_t(traces)
    assert stats["n_all"] == 4
    assert math.isfinite(stats["mean_all"])
    assert math.isfinite(stats["mean_correct"])
    assert math.isfinite(stats["mean_incorrect"])
    d = mod.cohen_d(stats["correct_values"], stats["incorrect_values"])
    assert math.isfinite(d)


def test_compute_all_metrics_end_to_end():
    traces = [_make_trace(i, is_correct=(i % 2 == 0), seed=i + 20) for i in range(6)]
    report = mod.compute_all_metrics(traces)
    assert report["n_samples_total"] == 6
    assert math.isfinite(report["metric_1_delta_h"]["mean_all"])
    assert report["metric_2_mahalanobis_d"]["d_m"] >= 0.0
    assert report["metric_3_mi_m_y_given_d"]["i_mi"] >= 0.0
    assert math.isfinite(report["metric_4_c_t"]["mean_all"])
    assert math.isfinite(report["metric_4_cohen_d_correct_vs_incorrect"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
