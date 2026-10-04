"""cd9 math_end_probe 회귀 시험 — CPU 전용, 모델 없음(forward 는 난수 mock)."""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_end_probe as E  # noqa: E402
import math_ruler_pivot as P  # noqa: E402


def _meta(conf="0.6", body="Check the sign.", decision=None):
    d = f"decision: {decision}\n" if decision else ""
    return f"<meta>\nconfidence: {conf}\n{body}\n{d}</meta>"


# ── select_rows: 행 선별 + 표적 ──────────────────────────────────────────────────
def test_select_rows_math_opt_keeps_only_parsed_meta_and_uses_r_corr():
    rows = [
        {"group_id": "g0", "problem": "p0", "gold": "1", "r_corr": 1,
         "text": "work " + _meta() + " \\boxed{1}"},
        {"group_id": "g1", "problem": "p1", "gold": "2", "r_corr": 0,
         "text": "no meta here, just \\boxed{9}"},
        {"group_id": "g2", "problem": "p2", "gold": "3", "r_corr": 0,
         "text": "broken <meta>\nno confidence line\n</meta> \\boxed{5}"},
    ]
    sel = E.select_rows(rows, "math_opt")
    assert [s["group_id"] for s in sel] == ["g0"]
    assert sel[0]["target"] == 1
    assert sel[0]["confidence"] == 0.6
    assert sel[0]["meta_start"] is not None and sel[0]["meta_end"] is not None


def test_select_rows_math_retry_uses_first_answer_correctness():
    text = "work \\boxed{7}\n" + _meta(conf="0.3", decision="redirect") + \
           "\nSecond attempt: other way \\boxed{9}"
    rows = [{"group_id": "g0", "problem": "p0", "gold": "9", "r_corr": 1, "text": text}]

    def fake_grade(pred, gold):
        return int(str(pred).strip() == str(gold).strip())

    sel = E.select_rows(rows, "math_retry", grade_fn=fake_grade)
    assert len(sel) == 1
    assert sel[0]["target"] == 0            # 첫 답 "7" != gold "9" → 첫 답은 오답
    assert sel[0]["confidence"] == 0.3

    # 첫 답이 없으면(박스 하나도 없음) 버린다.
    rows2 = [{"group_id": "g1", "problem": "p1", "gold": "9", "r_corr": 0,
             "text": "no boxed anywhere " + _meta()}]
    assert E.select_rows(rows2, "math_retry", grade_fn=fake_grade) == []


def test_group_pass_rates():
    rows = [{"group_id": "g0", "r_corr": 1}, {"group_id": "g0", "r_corr": 0},
            {"group_id": "g1", "r_corr": 1}, {"group_id": "g1", "r_corr": 1}]
    pr = E.group_pass_rates(rows)
    assert pr == {"g0": 0.5, "g1": 1.0}


# ── 층화 AUC 헬퍼 ────────────────────────────────────────────────────────────────
def test_nontrivial_auc_drops_pure_groups():
    groups = ["a", "a", "b", "b", "c", "c"]
    pass_rates = {"a": 0.5, "b": 0.0, "c": 1.0}   # b,c 는 0<p<1 밖
    y = [1, 0, 0, 0, 1, 1]
    s = [0.9, 0.1, 0.2, 0.3, 0.8, 0.7]
    got = E.nontrivial_auc(s, y, groups, pass_rates)
    assert got == P.auc([1, 0], [0.9, 0.1])


def test_group_pass_rate_ruler_is_near_chance_stratified():
    """자 == 그룹 관측 정답률 그 자체면, p̂ 3분위 층 안 AUC 와 0<p<1-only AUC 는
    개별 행 정답 여부에 대해 거의 정보가 없어야 한다(그 층 안에서는 거의 상수)."""
    rng = random.Random(0)
    groups, y, pass_rates = [], [], {}
    for gi in range(60):
        gid = f"g{gi}"
        p = rng.uniform(0.05, 0.95)
        pass_rates[gid] = p
        for _ in range(6):
            groups.append(gid)
            y.append(1 if rng.random() < p else 0)
    scores = [pass_rates[g] for g in groups]
    tert = E.tertile_auc(scores, y, groups, pass_rates)["auc"]
    nontriv = E.nontrivial_auc(scores, y, groups, pass_rates)
    # ★층화(tertile)만 「문제가 쉬운가」 되읽기를 잡는다 — 0<p<1 필터는 그저 순수(trivial)
    #   그룹을 빼는 것일 뿐 되읽기 방지가 아니므로 pooled 와 마찬가지로 실제 상관을 보인다.
    assert abs(tert - 0.5) < 0.15
    assert nontriv > 0.6
    assert P.auc(y, scores) > 0.6


def test_tertile_auc_returns_nan_when_bimodal_pass_rates_collapse_strata(capsys):
    """★RESULTS_cd9.md §3 결함 재현: math500 처럼 정답률이 0/1 극단으로 쏠리면 np.quantile
    ([1/3, 2/3]) 의 q1==q2==max(v) 로 무너져 모든 행이 층 0 하나에 들어간다 — 그 결과
    outcome_fixed_auc 가 auc_overall 과 바이트 동일한 값을 «층화된 것처럼» 돌려줬다(실측
    probes/end_q3i2507_math500/probe_table.json). 수리 후에는 층화가 실제로 안 됐을 때 NaN
    을 돌리고 경계·층별 n 을 찍는다."""
    rng = random.Random(0)
    groups, y, pass_rates = [], [], {}
    for gi in range(100):
        gid = f"g{gi}"
        # ★70% 가 정확히 max(=1.0) 이면 1/3·2/3 백분위 둘 다 이미 «전부 1.0» 구간 안이라
        #   np.quantile 이 q1==q2==1.0 으로 무너진다(실측 math500 분포와 같은 기제).
        p = 1.0 if gi < 70 else (0.0 if gi < 90 else rng.uniform(0.1, 0.9))
        pass_rates[gid] = p
        for _ in range(8):
            groups.append(gid)
            y.append(1 if rng.random() < p else 0)
    scores = [rng.random() for _ in groups]
    result = E.tertile_auc(scores, y, groups, pass_rates)
    assert math.isnan(result["auc"])
    assert result["degenerate"] is True
    out = capsys.readouterr().out
    assert "[math_end_probe][tertile]" in out and "q1=" in out and "n0=" in out


def test_tertile_auc_prints_boundaries_and_counts_on_nondegenerate_input(capsys):
    rng = random.Random(1)
    groups, y, pass_rates = [], [], {}
    for gi in range(60):
        gid = f"g{gi}"
        p = rng.uniform(0.05, 0.95)
        pass_rates[gid] = p
        for _ in range(6):
            groups.append(gid)
            y.append(1 if rng.random() < p else 0)
    scores = [pass_rates[g] for g in groups]
    result = E.tertile_auc(scores, y, groups, pass_rates)
    assert result["degenerate"] is False
    out = capsys.readouterr().out
    assert "n_nonempty_strata=3" in out or "n_nonempty_strata=2" in out


def test_pass_rule():
    assert E.pass_rule(0.75, 0.65) == "PASS"
    assert E.pass_rule(0.69, 0.65) == "FAIL"
    assert E.pass_rule(0.75, 0.55) == "FAIL"
    assert E.pass_rule(float("nan"), 0.65) == "FAIL"


# ── weights json round-trip ──────────────────────────────────────────────────────
def test_probe_json_round_trip_scores_match():
    from src.rulers.hidden_probe import fit_probe, probe_score
    rng = np.random.RandomState(0)
    X = rng.randn(80, 5)
    y = (X[:, 0] > 0).astype(int)
    groups = [f"g{i // 4}" for i in range(80)]
    probe = fit_probe(X, y, groups)
    d = E.probe_to_json(probe)
    json.loads(json.dumps(d))          # 직렬화 가능
    back = E.probe_from_json(d)
    Xt = rng.randn(10, 5)
    np.testing.assert_allclose(probe_score(probe, Xt), probe_score(back, Xt), rtol=1e-10)


# ── 파이프라인(mock forward, math_ruler_pivot 부속물 재사용) ────────────────────────────
def _fixture(n_groups=24, k=4, seed=0):
    rng = random.Random(seed)
    rows = []
    for gi in range(n_groups):
        gid = f"g{gi}"
        p = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])
        for _ in range(k):
            r_corr = 1 if rng.random() < p else 0
            conf = round(rng.random(), 2)
            text = (f"work \\boxed{{{gi}}}\n" + _meta(conf=str(conf)) + "\nmore words here")
            rows.append({"group_id": gid, "problem_id": gi, "problem": f"prob {gi}",
                        "gold": str(gi), "text": text, "r_corr": r_corr,
                        "final_answer": str(gi), "truncated": 0, "n_tok": 30})
    return rows


def test_build_jobs_and_pipeline_with_mock_forward():
    rows = _fixture()
    pass_rates = E.group_pass_rates(rows)
    selected = E.select_rows(rows, "math_opt")
    assert len(selected) == len(rows)         # 전부 완결된 메타를 심었다
    tok = P.MockTok()
    built = E.build_jobs(tok, "math_opt", selected)
    assert len(built["jobs"]) == 2 * len(selected)
    for it in built["index"]:
        j_meta = built["jobs"][it["job_meta"]]
        j_last = built["jobs"][it["job_last"]]
        assert j_meta.hidden_at == len(j_meta.ids) - 1
        assert j_last.hidden_at == len(j_last.ids) - 1
        assert len(j_last.ids) >= len(j_meta.ids)
        assert 0 < len(j_meta.ent_positions) <= E.DROP_WINDOW

    out = E.run(selected, P.mock_forward_factory(dim=6), tok, layers=[3, 6],
               variant="math_opt", pass_rates=pass_rates)
    names = {r["ruler"] for r in out["rows"]}
    for want in ("probe_metaend@L3", "probe_metaend@L6", "probe_last@L3", "probe_last@L6",
                 "stated_conf", "entropy_mean", "group_pass_rate(diagnostic)"):
        assert want in names, want
    for r in out["rows"]:
        assert r["verdict"] in ("PASS", "FAIL")
        assert r["n"] > 0
    md = E.to_markdown(out["rows"])
    assert "| ruler |" in md and "pass rule" in md
    json.dumps({"rows": out["rows"]}, default=float)   # 직렬화 가능
    # 난수 특징이라 실제 신호는 없음 — group_pass_rate(diagnostic) 만 pooled 신호가 있을 수 있으나
    # 최소한 파이프라인이 죽지 않고 유한/NaN 값을 낸다.
    for r in out["rows"]:
        assert isinstance(r["auc_overall"], float)


def test_run_with_empty_selection_returns_empty_rows():
    out = E.run([], P.mock_forward_factory(), P.MockTok(), layers=[1], variant="math_opt",
               pass_rates={})
    assert out["rows"] == [] and out["n_selected"] == 0
