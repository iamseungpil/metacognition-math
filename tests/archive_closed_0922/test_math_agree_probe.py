"""math_agree_probe 회귀 시험 — CPU 전용, 모델 없음(저장된 features.npz 를 흉내낸 합성
특징 위에서 돈다). 실제 forward pass 는 절대 하지 않는다(스크립트 자체가 그렇다)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_agree_probe as A  # noqa: E402
import math_end_probe as E  # noqa: E402
import math_ruler_pivot as P  # noqa: E402


def _meta(conf="0.6"):
    return f"<meta>\nconfidence: {conf}\ncheck signs\n</meta>"


# ── compute_disagree ─────────────────────────────────────────────────────────────
def test_compute_disagree_majority_and_tie():
    rows = [
        {"group_id": "g0", "final_answer": "1"},
        {"group_id": "g0", "final_answer": "1"},
        {"group_id": "g0", "final_answer": "2"},   # 소수 -> disagree
        {"group_id": "g1", "final_answer": "a"},   # 완전 동률(1-1) -> 둘 다 다수 집합
        {"group_id": "g1", "final_answer": "b"},
    ]
    d = A.compute_disagree(rows)
    assert d == [0, 0, 1, 0, 0]


def test_compute_disagree_none_answer_treated_as_a_value():
    rows = [
        {"group_id": "g0", "final_answer": None},
        {"group_id": "g0", "final_answer": None},
        {"group_id": "g0", "final_answer": "9"},
    ]
    assert A.compute_disagree(rows) == [0, 0, 1]


# ── per_problem_auc (outcome_fixed_auc 재사용 래퍼) ───────────────────────────────
def test_per_problem_auc_matches_outcome_fixed_auc():
    scores = [0.1, 0.9, 0.2, 0.8, 0.5, 0.6]
    labels = [0, 1, 0, 1, 0, 1]
    groups = np.array(["g0", "g0", "g1", "g1", "g2", "g2"])
    got = A.per_problem_auc(scores, labels, groups)
    want = P.outcome_fixed_auc(scores, labels, groups)
    assert got == want
    assert got["auc"] > 0.5          # 구성상 그룹마다 낮은점수=0, 높은점수=1


# ── reconstruct_selected: npz 와 정합/불일치 검출 ─────────────────────────────────
def _fixture(n_groups=20, k=4, seed=0):
    rng = random.Random(seed)
    rows = []
    for gi in range(n_groups):
        gid = f"g{gi}"
        p = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
        # 답이 갈리게: 그룹마다 정답 answer는 "gi", 오답은 "wrong-gi"
        for _ in range(k):
            r_corr = 1 if rng.random() < p else 0
            ans = str(gi) if r_corr else f"wrong-{gi}"
            conf = round(rng.random(), 2)
            text = (f"work \\boxed{{{ans}}}\n" + _meta(conf=str(conf)) + "\nmore words here")
            rows.append({"group_id": gid, "problem_id": gi, "problem": f"prob {gi}",
                        "gold": str(gi), "text": text, "r_corr": r_corr,
                        "final_answer": ans, "truncated": 0, "n_tok": 30})
    return rows


def test_reconstruct_selected_ok_and_mismatch(tmp_path):
    rows = _fixture()
    selected = E.select_rows(rows, "math_opt")
    got = A.reconstruct_selected(rows, "math_opt", len(selected))
    assert [g["group_id"] for g in got] == [g["group_id"] for g in selected]

    try:
        A.reconstruct_selected(rows, "math_opt", len(selected) + 1)
        assert False, "길이가 다르면 assert 로 죽어야 한다"
    except AssertionError:
        pass


# ── main() 엔드투엔드: 합성 npz + rollouts, forward pass 없음 ─────────────────────
def test_main_end_to_end_with_synthetic_features(tmp_path):
    rows = _fixture(n_groups=24, k=4, seed=1)
    selected = E.select_rows(rows, "math_opt")
    n = len(selected)
    assert n > 0

    rng = np.random.RandomState(0)
    y_corr = np.array([s["target"] for s in selected], dtype=float)
    groups = np.array([s["group_id"] for s in selected])
    disagree_all = A.compute_disagree(rows)
    y_dis = np.array([disagree_all[s["idx"]] for s in selected], dtype=float)

    d = 4
    # metaend@L0: r_corr 와 disagree 둘 다 신호를 실어 분리 가능하게 만든 합성 특징.
    hidden_metaend_L0 = (rng.randn(n, d) * 0.3
                         + np.outer(2 * y_corr - 1, np.ones(d))
                         + np.outer(2 * y_dis - 1, np.eye(1, d, 1).ravel()))
    hidden_last_L0 = rng.randn(n, d) * 1.0   # 신호 없음(대조)

    rollouts_path = tmp_path / "texts.jsonl"
    with open(rollouts_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    npz_path = tmp_path / "features.npz"
    np.savez(npz_path, target=y_corr, group_id=groups, confidence=np.full(n, np.nan),
             entropy_before=np.full(n, np.nan), layers=np.array([0]),
             hidden_metaend_L0=hidden_metaend_L0, hidden_last_L0=hidden_last_L0)

    out_dir = tmp_path / "out"
    argv_backup = sys.argv
    sys.argv = ["math_agree_probe.py", "--features", str(npz_path),
               "--rollouts", str(rollouts_path), "--out_dir", str(out_dir)]
    try:
        rc = A.main()
    finally:
        sys.argv = argv_backup
    assert rc == 0

    table = json.loads((out_dir / "agree_probe_table.json").read_text())
    rows_out = table["rows"]
    assert len(rows_out) == 4    # {metaend,last} x {r_corr,disagree}, layer 0 만

    by_key = {(r["position"], r["target"]): r for r in rows_out}
    # metaend 는 두 표적 다 신호가 있으니 last(신호 없음)보다 auc_overall 이 높아야 한다.
    assert by_key[("metaend", "r_corr")]["auc_overall"] > by_key[("last", "r_corr")]["auc_overall"]
    assert by_key[("metaend", "disagree")]["auc_overall"] > by_key[("last", "disagree")]["auc_overall"]

    br = table["base_rates"]
    assert br["n_selected"] == n
    assert 0.0 <= br["accuracy_overall(r_corr mean)"] <= 1.0
    assert 0.0 <= br["disagree_rate_overall"] <= 1.0
    md_text = (out_dir / "agree_probe_table.md").read_text()
    assert "per_problem_auc" in md_text
    assert "base rates" in md_text
