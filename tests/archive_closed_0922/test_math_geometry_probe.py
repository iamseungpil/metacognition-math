"""math_geometry_probe 회귀 시험 — CPU 전용, 모델 없음(합성 features.npz 위에서 돈다)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_geometry_probe as G  # noqa: E402


# ── 순수 헬퍼 ────────────────────────────────────────────────────────────────────
def test_leave_self_out_singleton_group_is_nan():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 3.0]])
    d = G.leave_self_out_distances(X, ["g0", "g0", "g1"])
    assert math.isnan(d["cos"][2]) and math.isnan(d["l2"][2])
    assert np.isfinite(d["cos"][0]) and np.isfinite(d["l2"][0])
    # 두 행 그룹: 자기 제외 중심 = 상대 행 → L2 = 두 행 거리
    assert abs(d["l2"][0] - math.sqrt(2)) < 1e-9


def test_centroid_distance_auc_is_one_when_wrong_sample_planted_far():
    """★핵심 고정: 틀린 행만 그룹 중심에서 멀리 심으면 문제별 AUC = 1.0."""
    rng = np.random.RandomState(0)
    n_groups, k, d = 12, 6, 8
    X, y, groups = [], [], []
    for g in range(n_groups):
        base = rng.randn(d)
        for j in range(k):
            wrong = j == 0                      # 그룹마다 정확히 하나만 오답
            v = base + 0.01 * rng.randn(d)
            if wrong:
                v = v + 50.0 * rng.randn(d)     # 멀리 심는다
            X.append(v)
            y.append(0 if wrong else 1)
            groups.append(f"g{g}")
    X = np.array(X)
    y = np.array(y, dtype=float)
    groups = np.array(groups)
    pass_rates = {f"g{g}": (k - 1) / k for g in range(n_groups)}   # 전부 mixed
    rows, eig = G.evaluate_geometry(X, y, groups, pass_rates, layer=0, position="metaend")
    by = {r["ruler"]: r for r in rows}
    assert by["cos_lso"]["auc_per_problem"] == 1.0
    assert by["l2_lso"]["auc_per_problem"] == 1.0
    assert by["l2_lso"]["auc_pooled"] > 0.9
    assert by["cos_lso"]["n"] == n_groups * k
    assert by["cos_lso"]["n_problems"] == n_groups
    assert eig["n_groups_all(k>=2)"] == n_groups


def test_centroid_distance_auc_is_chance_when_random():
    rng = np.random.RandomState(1)
    n_groups, k, d = 40, 6, 8
    X = rng.randn(n_groups * k, d)
    y = (rng.rand(n_groups * k) < 0.5).astype(float)
    groups = np.array([f"g{i // k}" for i in range(n_groups * k)])
    pass_rates = {}
    for g in set(groups.tolist()):
        m = groups == g
        pass_rates[g] = float(y[m].mean())
    # 전부 0 이나 1 인 그룹은 mixed 가 아니므로 자동으로 빠진다.
    rows, _ = G.evaluate_geometry(X, y, groups, pass_rates, layer=0, position="last")
    by = {r["ruler"]: r for r in rows}
    for name in ("cos_lso", "l2_lso"):
        assert abs(by[name]["auc_per_problem"] - 0.5) < 0.12, (name, by[name])
        assert abs(by[name]["auc_pooled"] - 0.5) < 0.12, (name, by[name])


def test_residualize_removes_the_difficulty_direction():
    rng = np.random.RandomState(2)
    n, d = 200, 5
    p = rng.rand(n)
    direction = rng.randn(d)
    X = np.outer(p, direction) + 0.01 * rng.randn(n, d)
    R = G.residualize_on_scalar(X, p)
    # 잔차는 p 와 거의 무상관
    for j in range(d):
        assert abs(G.pearson(R[:, j], p)) < 0.1
    # 분산 0 인 스칼라면 평균만 뺀다
    R0 = G.residualize_on_scalar(X, np.full(n, 0.3))
    assert np.allclose(R0, X - X.mean(axis=0, keepdims=True))


def test_eigenscore_grows_with_spread_and_needs_two_rows():
    rng = np.random.RandomState(3)
    tight = rng.randn(6, 10) * 0.01
    wide = rng.randn(6, 10) * 10.0
    assert G.eigenscore(wide) > G.eigenscore(tight)
    assert math.isnan(G.eigenscore(np.zeros((1, 4))))


def test_spearman_monotone_and_pearson():
    a = [1, 2, 3, 4, 5]
    b = [1, 4, 9, 16, 25]
    assert abs(G.spearman(a, b) - 1.0) < 1e-9
    assert G.pearson(a, b) < 1.0
    assert abs(G.spearman(a, [5, 4, 3, 2, 1]) + 1.0) < 1e-9


# ── main() 엔드투엔드 ────────────────────────────────────────────────────────────
def test_main_end_to_end_with_synthetic_npz(tmp_path):
    rng = np.random.RandomState(4)
    n_groups, k, d = 16, 5, 6
    X_meta, X_last, y, groups = [], [], [], []
    for g in range(n_groups):
        base = rng.randn(d)
        n_wrong = 1 + (g % 2)
        for j in range(k):
            wrong = j < n_wrong
            v = base + 0.05 * rng.randn(d)
            if wrong:
                v = v + 20.0 * rng.randn(d)
            X_meta.append(v)
            X_last.append(rng.randn(d))          # 신호 없음(대조)
            y.append(0.0 if wrong else 1.0)
            groups.append(f"g{g}")
    npz_path = tmp_path / "features.npz"
    np.savez(npz_path, target=np.array(y), group_id=np.array(groups),
             confidence=np.full(len(y), np.nan), entropy_before=np.full(len(y), np.nan),
             layers=np.array([0]),
             hidden_metaend_L0=np.array(X_meta), hidden_last_L0=np.array(X_last))

    out_dir = tmp_path / "out"
    argv = sys.argv
    sys.argv = ["math_geometry_probe.py", "--features", str(npz_path), "--out_dir", str(out_dir)]
    try:
        assert G.main() == 0
    finally:
        sys.argv = argv

    tab = json.loads((out_dir / "geometry_table.json").read_text())
    by = {(r["position"], r["ruler"]): r for r in tab["rows"]}
    assert len(tab["rows"]) == 2 * 3          # {metaend,last} x {cos,l2,resid}
    # ★그룹의 절반은 오답이 둘이라 자기 제외 중심이 다른 오답에 끌린다 — 그래도 .75 는 넘는다.
    assert by[("metaend", "cos_lso")]["auc_per_problem"] > 0.75
    assert by[("metaend", "l2_lso")]["auc_per_problem"] > 0.75
    assert by[("last", "l2_lso")]["auc_per_problem"] < by[("metaend", "l2_lso")]["auc_per_problem"]
    assert tab["base"]["n_rows_mixed"] == n_groups * k
    assert tab["base"]["n_groups_mixed"] == n_groups
    assert tab["base"]["pass_rate_source"].startswith("npz target")
    assert len(tab["eigenscore"]) == 2
    md = (out_dir / "geometry_table.md").read_text()
    assert "auc_per_problem" in md and "eigenscore" in md
