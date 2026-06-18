"""CPU unit tests for the PG0 pure verdict helper (spec 2026-06-18 §0 PG0).

Only pg0_verdict is tested here — main() wires vLLM (H100-only). The verdict
math reuses scripts.harvest_redirect_cf.expected_yield.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pg0_yield_pilot import pg0_verdict, _load_pool


def test_high_accept_is_go():
    # emission .5 * in_band .5 * accept .8 * 20000 = 4000 >= target 1500.
    r = pg0_verdict(emission_rate=0.5, in_band_frac=0.5, accept_prob=0.8,
                    full_pool=20000, target=1500)
    assert r["verdict"] == "GO"
    assert r["projected_accepted"] == 4000


def test_low_accept_is_stop():
    # accept .02 * .5 * .5 * 20000 = 100 < 1500 -> STOP.
    r = pg0_verdict(emission_rate=0.5, in_band_frac=0.5, accept_prob=0.02,
                    full_pool=20000, target=1500)
    assert r["verdict"] == "STOP"
    assert r["projected_accepted"] == 100


def test_small_pool_is_stop():
    # plenty of accept-prob but a tiny pool cannot reach the target.
    r = pg0_verdict(emission_rate=0.9, in_band_frac=0.9, accept_prob=0.9,
                    full_pool=500, target=1500)
    assert r["verdict"] == "STOP"
    assert r["projected_accepted"] < 1500


def test_boundary_exact_target_is_go():
    # exactly hitting the target counts as GO (>= comparison).
    # 1.0 * 1.0 * 1.0 * 1500 = 1500.
    r = pg0_verdict(emission_rate=1.0, in_band_frac=1.0, accept_prob=1.0,
                    full_pool=1500, target=1500)
    assert r["verdict"] == "GO"
    assert r["projected_accepted"] == 1500


def test_zero_emission_is_stop():
    r = pg0_verdict(emission_rate=0.0, in_band_frac=0.5, accept_prob=0.9,
                    full_pool=100000, target=1500)
    assert r["verdict"] == "STOP"
    assert r["projected_accepted"] == 0


def test_payload_shape():
    r = pg0_verdict(0.3, 0.4, 0.5, 10000, 1500)
    for key in ("emission_rate", "in_band_frac", "accept_prob", "full_pool",
                "projected_accepted", "target", "verdict"):
        assert key in r
    assert r["verdict"] in ("GO", "STOP")


def test_load_pool_handles_ndarray_prompt(tmp_path):
    """Regression: pandas reads the verl ``prompt`` list-of-dict column back as a
    numpy ndarray, which the old ``isinstance(prompt, (list, tuple))`` missed ->
    the pilot pool loaded 0 problems and PG0 STOPped before any GPU work. The
    loader must normalise ndarray -> list and still extract (question, gold)."""
    import pandas as pd

    df = pd.DataFrame([
        {"prompt": [{"role": "user", "content": "What is 2+2?"}],
         "reward_model": {"style": "rule", "ground_truth": "4"}},
        {"prompt": [{"role": "user", "content": "What is 3+5?"}],
         "reward_model": {"style": "rule", "ground_truth": "8"}},
    ])
    p = tmp_path / "mini.parquet"
    df.to_parquet(p)
    # round-tripping through parquet is what turns prompt into an ndarray
    pool = _load_pool(str(p), pool_size=10)
    assert len(pool) == 2
    assert pool[0]["question"] == "What is 2+2?"
    assert pool[0]["gold"] == "4"
