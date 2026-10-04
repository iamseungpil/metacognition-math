"""cd9 math_ruler_pivot 회귀 시험 — CPU 전용, 모델 없음(forward 는 난수 mock)."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_ruler_pivot as P  # noqa: E402


# ── 순수 헬퍼 ────────────────────────────────────────────────────────────────────
def test_group_of_splits_on_last_at():
    assert P.group_of("math500-17@812") == "math500-17"
    assert P.group_of("a@b@33") == "a@b"


def test_extract_meta_block_stops_at_close():
    cont = "confidence: 0.4\nThis looks off.\ndecision: redirect\n</meta>\nLet me retry.\n</meta> junk"
    blk = P.extract_meta_block(cont)
    assert blk == "<meta>\nconfidence: 0.4\nThis looks off.\ndecision: redirect\n</meta>"
    assert blk.count("</meta>") == 1
    assert P.extract_meta_block("never closed confidence: 0.9") is None
    from src.training.countdown_rewards import parse_meta
    pm = parse_meta(blk, form="math")
    assert pm["emitted"] == 1 and pm["confidence"] == 0.4 and pm["decision"] == "redirect"


def test_outcome_fixed_auc_of_rcorr_itself_is_half():
    # ★자가 r_corr 그 자체면 r_corr 층 안에서 상수 → AUC .5 (또는 층이 한 클래스뿐이면 제외).
    rng = random.Random(0)
    r_corr = [rng.randint(0, 1) for _ in range(400)]
    # delta>0 표적은 r_corr 과 상관되지만 층 안에선 섞여 있다.
    y = [1 if (rc and rng.random() < .8) or (not rc and rng.random() < .3) else 0 for rc in r_corr]
    got = P.outcome_fixed_auc(scores=r_corr, labels=y, strata=r_corr)
    assert got["n_strata"] == 2
    assert abs(got["auc"] - 0.5) < 1e-9
    # 반면 pooled AUC 는 .5 를 크게 넘는다(결과 되읽기 자의 전형).
    assert P.auc(y, r_corr) > 0.6


def test_outcome_fixed_auc_keeps_real_signal():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 300)
    strata = rng.randint(0, 2, 300)
    s = y + 0.5 * rng.randn(300)
    got = P.outcome_fixed_auc(s, y, strata)
    assert got["auc"] > 0.8
    assert P.outcome_fixed_auc([1, 2, 3], [1, 1, 1], [0, 0, 0])["n_strata"] == 0
    assert math.isnan(P.outcome_fixed_auc([1, 2, 3], [1, 1, 1], [0, 0, 0])["auc"])


def test_pass_rule():
    assert P.pass_rule(0.70, 0.60, True) == "PASS"
    assert P.pass_rule(0.64, 0.60, True) == "FAIL"
    assert P.pass_rule(0.70, 0.54, True) == "FAIL"
    assert P.pass_rule(0.70, 0.60, False) == "FAIL"
    assert P.pass_rule(float("nan"), 0.60, True) == "FAIL"
    assert P.pass_rule(0.70, 0.60, None) == "PASS(donor n/a)"


def test_tertile_strata_three_levels():
    st = P.tertile_strata([i / 10 for i in range(30)])
    assert set(st) == {0, 1, 2}


def test_grouped_oof_probe_no_group_leak_and_donor_scored():
    rng = np.random.RandomState(0)
    n = 120
    X = rng.randn(n, 4)
    y = (X[:, 0] > 0).astype(int)
    groups = [f"g{i // 2}" for i in range(n)]
    Xo = rng.randn(10, 4)
    go = [f"g{i}" for i in range(10)]
    pr = P.grouped_oof_probe(X, y, groups, X_other=Xo, groups_other=go, seed=0)
    assert pr["n_folds_used"] == 5
    assert np.isfinite(pr["oof"]).all() and np.isfinite(pr["other"]).all()
    assert P.auc(y, pr["oof"]) > 0.9
    # 학습 그룹에 없는 donor 그룹은 NaN
    pr2 = P.grouped_oof_probe(X, y, groups, X_other=Xo[:1], groups_other=["nope"], seed=0)
    assert math.isnan(pr2["other"][0])


def test_parse_layers():
    assert P.parse_layers("-1,mid", 36) == [18, 36]
    assert P.parse_layers("4", 36) == [4]


# ── 파이프라인(mock forward) ────────────────────────────────────────────────────────
def _fixture(n_problems=24, cuts=2, k=3, seed=0):
    rng = random.Random(seed)
    sites, conts, decoys = [], [], {}
    for p in range(n_problems):
        gid = f"prob{p}"
        gold = str(p * 7)
        decoys[gid] = str(p * 7 + 1)
        for c in range(cuts):
            sid = f"{gid}@{100 + 50 * c}"
            p0 = rng.choice([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0])
            pm = min(1.0, max(0.0, p0 + rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5])))
            d = pm - p0
            lab = "SAVE" if (p0 <= .25 and d >= .25) else ("DERAIL" if (p0 >= .75 and d <= -.25) else "NEUTRAL")
            pv, pr = rng.random(), rng.random()
            bd = "redirect" if pr - pv >= .25 else ("verify" if pv - pr >= .25 else "tie")
            sites.append({"site_id": sid, "problem": f"What is {p} plus {p}?", "gold": gold,
                          "prefix": " ".join(f"step{i}" for i in range(40 + c)) + "\n",
                          "donor_meta": "<meta>\nconfidence: 0.7\nOther problem's meta.\ndecision: verify\n</meta>",
                          "p_nometa": p0, "p_meta": pm, "p_donor": p0, "delta": d, "delta_donor": 0.0,
                          "p_verify": pv, "p_redirect": pr, "best_decision": bd, "label": lab,
                          "movable": int(0 < p0 < 1)})
            for mode in ("nometa", "meta", "donor"):
                for _ in range(k):
                    if mode == "meta":
                        dec = rng.choice(["verify", "redirect"])
                        cont = (f"confidence: {rng.random():.2f}\nHmm the sum.\ndecision: {dec}\n</meta>\n"
                                + " ".join(f"tok{i}" for i in range(50)) + " \\boxed{1}")
                    else:
                        cont = " ".join(f"tok{i}" for i in range(50)) + " \\boxed{1}"
                    conts.append({"site_id": sid, "mode": mode, "r_corr": rng.randint(0, 1),
                                  "cont": cont, "truncated": 0})
    return sites, conts, decoys


def test_build_jobs_wiring():
    sites, conts, decoys = _fixture()
    tok = P.MockTok()
    built = P.build_jobs(tok, sites, conts, decoys, max_conts_per_site=2)
    assert len(built["site_jobs"]) == len(sites)
    metas = built["meta_rows"]
    assert len(metas) == len(sites) * 2 * 2          # meta 2 + donor 2 per site
    own = [m for m in metas if m["mode"] == "meta"]
    assert all(m["block"].endswith("</meta>") and m["block"].count("</meta>") == 1 for m in own)
    assert all(m["decision"] in ("verify", "redirect") and m["conf"] is not None for m in own)
    assert all(m["job_pmi_meta"] is not None for m in metas)       # 디코이 전부 존재
    # 디코이 없는 문제는 PMI 건너뜀
    built2 = P.build_jobs(tok, sites, conts, {}, max_conts_per_site=1)
    assert all(m["job_pmi_meta"] is None for m in built2["meta_rows"])
    # 갈림 창: 자리 job 의 엔트로피 위치는 마지막 64 토큰 이내
    j = built["jobs"][built["site_jobs"][sites[0]["site_id"]]]
    assert 0 < len(j.ent_positions) <= P.FORK_WINDOW and j.hidden_at == len(j.ids) - 1


def test_pipeline_with_mock_forward_produces_table():
    sites, conts, decoys = _fixture()
    out = P.run_pivot(sites, conts, decoys, P.mock_forward_factory(dim=6), P.MockTok(),
                      layers=[3, 6], max_conts_per_site=2)
    names = {r["ruler"] for r in out["rows"]}
    for want in ("probe_cut@L3", "probe_cut@L6", "probe_cut_dec@L3", "fork_entropy_mean",
                 "fork_entropy_max", "entropy_drop", "stated_conf", "judgment_match", "pmi_shift",
                 "probe_metaend@L3", "probe_metaend_dec@L6"):
        assert want in names, want
    for r in out["rows"]:
        assert r["verdict"] in ("PASS", "FAIL", "PASS(donor n/a)")
        assert r["n"] > 0
    # 난수 특징이라 pooled AUC 는 .5 근처 → 전부 FAIL 이어야 정상(양성 오판 없음)
    pm = next(r for r in out["rows"] if r["ruler"] == "pmi_shift")
    assert "auc_rcorr" in pm and pm["donor_lower"] is not None
    md = P.to_markdown(out["rows"])
    assert "| ruler |" in md and "pass rule" in md
    import json
    json.dumps(out, default=float)   # 직렬화 가능


def test_judgment_match_ruler_is_not_outcome_readout():
    # judgment_match 를 정답 판단과 일치하게 심으면 SAVE/DERAIL 과 무관해도 정의는 유지된다.
    sites, conts, decoys = _fixture(n_problems=10)
    feats = {"site": {}, "meta": []}
    by = {s["site_id"]: s for s in sites}
    for s in sites:
        feats["site"][s["site_id"]] = {"hidden": {0: np.zeros(2)}, "fork_entropy_mean": 0.0,
                                       "fork_entropy_max": 0.0}
        bd = s["best_decision"]
        for mode in ("meta", "donor"):
            feats["meta"].append({"site_id": s["site_id"], "mode": mode, "r_corr": 1, "conf": 0.5,
                                  "decision": bd if (mode == "meta" and bd != "tie") else "verify",
                                  "hidden": {0: np.zeros(2)}, "entropy_drop": 0.0, "pmi_shift": 0.0})
    rows = P.evaluate_rulers(sites, feats, layers=[0])
    jm = next(r for r in rows if r["ruler"] == "judgment_match")
    assert jm["own_mean"] == 1.0 and jm["donor_mean"] < 1.0 and jm["donor_lower"] is True
