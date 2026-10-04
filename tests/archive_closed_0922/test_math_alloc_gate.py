"""cd9 0914 — math_alloc_gate 회귀 시험(CPU, 합성 모집단).

다수결 규칙 / 배분 예산이 (B,k_max) 마다 정확히 맞는가 / oracle 이 «표본을 늘리면 구제되는»
합성 모집단에서 uniform 을 이기는가 / random 이 uniform 과 비슷한가 / probe 결측이 중앙값으로
채워지고 세어지는가 / 정오 판정표(nan 포함) → FAIL.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_alloc_gate as AG  # noqa: E402


def _rows(answers, corr_of=None, n_tok=100):
    """answers: 최종답 문자열 리스트(고정 롤아웃 순서). corr_of: answer -> r_corr(기본: gold='A')."""
    corr_of = corr_of or (lambda a: 1 if a == "A" else 0)
    return [{"final_answer": a, "r_corr": corr_of(a), "n_tok": n_tok} for a in answers]


# ── 1. 다수결 규칙 ──────────────────────────────────────────────────────────────
def test_vote_correct_majority_and_tie_goes_first():
    rows = _rows(["A", "A", "B"])
    assert AG.vote_correct(rows, 3) == 1.0          # A 가 다수
    assert AG.vote_correct(_rows(["B", "A"]), 2) == 0.0
    tie = _rows(["A", "B"])                          # 동률 → 첫 표본("A") → 맞음
    assert AG.vote_correct(tie, 2) == 1.0
    tie2 = _rows(["B", "A"])                          # 동률 → 첫 표본("B") → 틀림
    assert AG.vote_correct(tie2, 2) == 0.0
    assert AG.vote_correct([], 1) == 0.0


def test_vote_correct_only_counts_the_first_k():
    """예산 k 는 «앞 k 개만 본다» — 뒤에 정답이 있어도 안 쓴다."""
    rows = _rows(["B", "B", "A", "A", "A"])           # 앞 2개만 보면 B 가 다수(틀림)
    assert AG.vote_correct(rows, 2) == 0.0
    assert AG.vote_correct(rows, 5) == 1.0             # 5개 다 보면 A 가 다수(맞음)


# ── 2. 배분 예산이 (B,k_max) 마다 정확히 맞는가 ─────────────────────────────────
def test_alloc_ks_mean_matches_budget_for_every_b_and_kmax():
    order = [f"g{i}" for i in range(40)]
    for b in AG.BUDGETS:
        for kmax in AG.KMAXES:
            ks = AG.alloc_ks(order, b, kmax)
            assert set(ks) == set(order)
            assert set(ks.values()) <= {1, kmax}
            # 반올림 오차 상한: n_hard 반올림이 최대 0.5 어긋나고, 그 한 단위가 예산을
            # (k_max-1)/n 만큼 움직인다.
            if b > kmax:
                assert math.isclose(AG.realized_budget(ks), kmax)
            else:
                tol = 0.5 * (kmax - 1) / len(order) + 1e-9
                assert abs(AG.realized_budget(ks) - b) <= tol, (b, kmax)


def test_alloc_ks_budget_one_is_all_ones():
    order = ["g0", "g1", "g2"]
    for kmax in AG.KMAXES:
        assert AG.alloc_ks(order, 1.0, kmax) == {g: 1 for g in order}


def test_alloc_ks_hardest_first_gets_the_extra_budget():
    order = ["hard0", "hard1", "easy0", "easy1"]        # 이미 hardest-first 로 정렬됐다고 가정
    ks = AG.alloc_ks(order, 2.0, 4)                      # n_hard = round(4*1/3)=1
    assert ks["hard0"] == 4
    assert ks["hard1"] == ks["easy0"] == ks["easy1"] == 1


def test_order_hardest_first_ascending_and_deterministic_ties():
    j = {"b": 0.5, "a": 0.5, "c": 0.1}
    assert AG.order_hardest_first(j) == ["c", "a", "b"]   # 0.1 이 가장 어렵다, 동률은 이름순


# ── 3. oracle 이 «표본을 늘리면 구제되는» 합성 모집단에서 uniform 을 이긴다 ─────────
def _rescue_population(n_easy=20, n_hard=20, seed=0):
    """easy 문제: 첫 표본부터 다수결로 맞는다. hard 문제: 첫 표본은 오답, 표본 8개 중 5개는
    정답이라 k_max>=4 를 주면 다수결로 구제된다(«더 뽑으면 산다» 시나리오)."""
    rng = random.Random(seed)
    groups, oracle = {}, {}
    for i in range(n_easy):
        g = f"easy{i}"
        groups[g] = _rows(["A"] * 8)
        oracle[g] = 1.0
    for i in range(n_hard):
        g = f"hard{i}"
        # 첫 표본은 오답(B), 나머지 7개는 5x A(정답) 2x B → LOO(첫 표본 뺀) pass rate = 5/7
        groups[g] = _rows(["B", "A", "A", "A", "A", "A", "B", "B"])
        oracle[g] = 5.0 / 7.0
    order = list(groups)
    rng.shuffle(order)
    return groups, oracle, order


def test_oracle_beats_uniform_when_extra_samples_rescue_hard_problems():
    groups, oracle, order = _rescue_population()
    n_max = 8
    uni = AG.uniform_ks_map(order, 2.0, n_max)
    uni_acc = sum(AG.acc_for_ks(groups, uni).values()) / len(order)
    best = AG.best_alloc(oracle, groups, 2.0, n_max)
    assert best["acc"] > uni_acc + 0.05, (best["acc"], uni_acc)
    assert best["k_max"] >= 4, "hard 문제를 구제하려면 4표 이상 줘야 한다"


def _stochastic_population(n=300, seed=2):
    """난이도가 부드럽게 퍼진 모집단 — p_group ∈ {.2,.35,.5,.65,.8}, 표본 8개는 그 확률의
    독립 베르누이. `_rescue_population`(4표에서만 뒤집히는 계단 함수)과 달리 여기서는
    «누구에게 여분 표를 주든» 기대 이득이 거의 상쇄되므로 random ≈ uniform 을 깨끗이 잰다."""
    rng = random.Random(seed)
    ps = [0.2, 0.35, 0.5, 0.65, 0.8]
    groups = {}
    for i in range(n):
        p = ps[i % len(ps)]
        answers = ["A" if rng.random() < p else "B" for _ in range(8)]
        groups[f"g{i}"] = _rows(answers)
    order = list(groups)
    rng.shuffle(order)
    return groups, order


def test_random_is_close_to_uniform():
    groups, order = _stochastic_population()
    n_max = 8
    uni = AG.uniform_ks_map(order, 1.5, n_max)
    uni_acc = sum(AG.acc_for_ks(groups, uni).values()) / len(order)
    rng = random.Random(3)
    rnd_judgment = {g: rng.random() for g in order}
    # ★단일 k_max 로 잰다 — best_alloc(3개 중 max)은 random 판단원에도 «최댓값 편향»을
    #   준다(math_plan_gate.py 의 max-of-3 함정과 같다). «random ≈ uniform» 은 고정
    #   k_max 로만 확인해야 편향 없는 비교가 된다.
    order_r = AG.order_hardest_first(rnd_judgment)
    ks = AG.alloc_ks(order_r, 1.5, 2)
    rnd_acc = sum(AG.acc_for_ks(groups, ks).values()) / len(order)
    assert abs(rnd_acc - uni_acc) < 0.08, (rnd_acc, uni_acc)


# ── 4. probe 결측 → 모집단 중앙값으로 채워지고 세어진다 ─────────────────────────
def test_missing_probe_rows_get_median_and_are_counted(tmp_path, monkeypatch):
    texts = tmp_path / "texts.jsonl"
    import json as _json
    rows = []
    # p0: emitted(meta 있음) 표본이 첫 자리에 있다. p1: 아무 표본도 meta 가 없다(=probe 결측).
    for gi, has_meta in enumerate([True, False]):
        for si in range(3):
            body = "<meta>\nconfidence: 0.7\ndecision: verify\n</meta>\n" if (has_meta and si == 0) else ""
            rows.append({"group_id": f"g{gi}", "final_answer": "A", "r_corr": 1, "n_tok": 50,
                        "text": body + "work \\boxed{A}"})
    with texts.open("w") as fh:
        for r in rows:
            fh.write(_json.dumps(r) + "\n")

    def _fake_load_npz(npz_path, rows_):
        sel = AG.emitted_indices(rows_)
        assert len(sel) == 1                 # p0 의 si=0 행만 emitted
        return {sel[0]: 0.8}, {sel[0]: 0.7}, "fake"

    monkeypatch.setattr(AG, "load_npz_signals", _fake_load_npz)
    rows_all, groups, order = AG.load_groups(str(texts))
    judgments, meta, _note = AG.build_judgments(rows_all, groups, order, "unused.npz")
    assert meta["n_missing_probe"] == 1 and meta["n_missing_stated"] == 1
    # g1(결측)은 g0 의 probe/stated 값으로 채워진(모집단이 둘뿐이라 중앙값=g0의 값) 상수를 받는다.
    assert math.isclose(judgments["probe"]["g1"], judgments["probe"]["g0"])
    assert math.isclose(judgments["stated"]["g1"], judgments["stated"]["g0"])


def test_load_npz_signals_missing_file_returns_none(tmp_path):
    p, c, note = AG.load_npz_signals(str(tmp_path / "nope.npz"), [])
    assert p is None and c is None and "없음" in note


# ── 5. 정오 판정표(진리표, nan 포함) → FAIL ─────────────────────────────────────
def test_row_correct_truth_table_including_nan():
    assert AG._row_correct({"r_corr": 1}) is True
    assert AG._row_correct({"r_corr": 1.0}) is True
    assert AG._row_correct({"r_corr": 0}) is False
    assert AG._row_correct({"r_corr": 0.0}) is False
    assert AG._row_correct({"r_corr": float("nan")}) is False, "nan 은 bool()이 True 라 함정"
    assert AG._row_correct({"r_corr": None}) is False
    assert AG._row_correct({}) is False
    assert AG._row_correct({"r_corr": "not-a-number"}) is False


def test_vote_correct_fails_when_matching_answer_row_has_nan_r_corr():
    rows = [{"final_answer": "A", "r_corr": float("nan")}, {"final_answer": "A", "r_corr": 0}]
    assert AG.vote_correct(rows, 2) == 0.0


# ── 6. 게이트 판정 배선 ──────────────────────────────────────────────────────────
def test_gate_verdict_reads_b2_and_computes_stated_gap():
    report = {"budgets": {2.0: {"sources": {
        "probe": {"gain_ci_lo": 0.03, "gain_ci_hi": 0.09, "gain_vs_uniform": 0.06, "acc": 0.7,
                 "k_max": 4},
        "oracle": {"gain_vs_uniform": 0.05, "acc": 0.75, "k_max": 4},
        "stated": {"acc": 0.62, "gain_vs_uniform": 0.01, "k_max": 4},
        "length": {"acc": 0.6, "gain_vs_uniform": 0.0, "k_max": 2},
        "random": {"acc": 0.6, "gain_vs_uniform": 0.0, "k_max": 2},
    }}}}
    v = AG.gate_verdict(report)
    assert v["pass"] is True
    assert math.isclose(v["stated_gap_b2"], 0.7 - 0.62)


def test_gate_verdict_fails_when_ci_includes_zero():
    report = {"budgets": {2.0: {"sources": {
        "probe": {"gain_ci_lo": -0.01, "gain_ci_hi": 0.05, "gain_vs_uniform": 0.03, "acc": 0.65,
                 "k_max": 4},
        "oracle": {"gain_vs_uniform": 0.05, "acc": 0.75, "k_max": 4},
        "stated": {"acc": 0.6, "gain_vs_uniform": 0.0, "k_max": 2},
        "length": {"acc": 0.6, "gain_vs_uniform": 0.0, "k_max": 2},
        "random": {"acc": 0.6, "gain_vs_uniform": 0.0, "k_max": 2},
    }}}}
    assert AG.gate_verdict(report)["pass"] is False
