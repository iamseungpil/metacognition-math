"""math_dist_effect 회귀 시험 — CPU 전용, 모델 없음(은닉벡터를 심는 mock forward).

★심은 기하: nometa 의 정답 이어쓰기는 +e0 쪽, 오답은 −e0 쪽에 둔다. 그러면 «정답 방향»
û ≈ +e0. 모드 meta(=+3e0) 는 정답 쪽으로, donor(=−3e0) 는 반대쪽으로 심어 toward_correct
의 부호가 맞는지 고정한다.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_dist_effect as D  # noqa: E402
import math_ruler_pivot as P  # noqa: E402

DIM = 4
# 모드 이름은 math_sites.build_fed 가 아는 것이어야 한다(문맥 조립을 그대로 쓴다).
SHIFT_MAIN = {"nometa": 0.0, "meta": 3.0, "donor": -3.0}      # meta=정답 쪽, donor=반대쪽
SHIFT_PAIR = {"nometa": 0.0, "own": 2.0, "donor": 0.5}        # own 이 donor 보다 정답 쪽


def _fixture(n_sites=8, k=4, modes=("nometa", "meta", "donor")):
    sites, conts = [], []
    for i in range(n_sites):
        sid = f"g{i}@100"
        sites.append({"site_id": sid, "problem": f"What is {i} plus {i}?", "gold": str(2 * i),
                      "prefix": "step one\nstep two\n", "site_source": "cut",
                      "donor_meta": "<meta>\nconfidence: 0.7\nother problem\ndecision: verify\n</meta>",
                      "own_meta": "<meta>\nconfidence: 0.5\nmine\ndecision: verify\n</meta>"})
        for m in modes:
            for j in range(k):
                # nometa 는 절반 정답/절반 오답(방향 û 가 정의되도록), 나머지 모드는 섞어서.
                rc = 1 if (j % 2 == 0) else 0
                ans = str(2 * i) if rc else str(2 * i + 1)
                conts.append({"site_id": sid, "mode": m, "r_corr": rc,
                              "cont": f"more work here \\boxed{{{ans}}}", "truncated": 0})
    return sites, conts


def _planting_forward(index_ref, shift=None):
    """Job 순서 = build_jobs 의 index 순서 — 그 index 를 보고 벡터를 심는다."""
    def forward(jobs, layers):
        out = []
        for n, j in enumerate(jobs):
            it = index_ref["index"][n]
            e0 = np.zeros(DIM)
            e0[0] = 1.0
            base = (shift or SHIFT_MAIN)[it["mode"]] * e0
            if it["mode"] == "nometa":
                base = base + (1.0 if it["r_corr"] else -1.0) * e0
            v = base + 0.01 * np.arange(DIM)
            rec = {"hidden": {L: v + 0.0 * L for L in layers}, "entropy": [], "lp": []}
            if j.hidden_span is not None:
                rec["hidden_mean"] = {L: v * 0.5 for L in layers}
            out.append(rec)
        return out
    return forward


# ── 순수 헬퍼 ────────────────────────────────────────────────────────────────────
def test_select_continuations_caps_per_site_mode():
    sites, conts = _fixture(n_sites=3, k=6)
    sel = D.select_continuations(conts, 2, {s["site_id"] for s in sites})
    cnt = {}
    for c in sel:
        cnt[(c["site_id"], c["mode"])] = cnt.get((c["site_id"], c["mode"]), 0) + 1
    assert set(cnt.values()) == {2}
    assert len(sel) == 3 * 3 * 2
    # 빈 이어쓰기는 버린다
    assert D.select_continuations([{"site_id": "a", "mode": "nometa", "r_corr": 1, "cont": "  "}],
                                  4) == []


def test_answer_clusters_merge_math_equivalent_and_entropy():
    lab = D.answer_clusters(["42", "42.0", "7", "", "\\frac{1}{2}", "0.5"])
    assert lab[0] == lab[1]              # 42 ≡ 42.0
    assert lab[4] == lab[5]              # 1/2 ≡ 0.5
    assert lab[2] != lab[0] and lab[3] not in (lab[0], lab[2])
    assert abs(D.entropy_of([0, 0, 1, 1]) - math.log(2)) < 1e-9
    assert D.entropy_of([3, 3, 3]) == 0.0
    assert math.isnan(D.entropy_of([]))


def test_site_text_metrics_entropy_and_success_delta():
    conts = {"nometa": ["\\boxed{1}", "\\boxed{1}", "\\boxed{1}", "\\boxed{1}"],
             "spread": ["\\boxed{1}", "\\boxed{2}", "\\boxed{3}", "\\boxed{4}"]}
    rc = {"nometa": [1, 1, 0, 0], "spread": [1, 1, 1, 0]}
    got = D.site_text_metrics(conts, rc)
    assert abs(got["spread"]["ans_entropy_delta"] - math.log(4)) < 1e-9   # 0 → log4
    assert abs(got["spread"]["success_delta"] - 0.25) < 1e-12


def test_site_geometry_direction_sign_and_skip_when_no_correct():
    e0 = np.zeros(DIM)
    e0[0] = 1.0
    fk = ("last", 0)
    # ★중심이 원점이면 코사인 거리가 정의 안 되므로 +쪽으로 치우쳐 심는다(정답 3e0, 오답 1e0).
    base = np.array([3 * e0, 3 * e0, 1 * e0, 1 * e0])
    vecs = {(D.BASE_MODE, fk): base,
            ("good", fk): np.array([9 * e0, 9 * e0 + 0.1, 9 * e0, 9 * e0]),
            ("bad", fk): np.array([-5 * e0, -5 * e0, -5 * e0, -5 * e0 - 0.1])}
    rc = {D.BASE_MODE: [1, 1, 0, 0], "good": [1, 1, 1, 1], "bad": [0, 0, 0, 0]}
    got = D.site_geometry(vecs, rc, [fk])
    assert got[("good", fk)]["toward_correct"] > 0
    assert got[("bad", fk)]["toward_correct"] < 0
    assert got[("good", fk)]["centroid_shift"] >= 0
    # nometa 가 전부 정답이면 방향이 정의 안 됨 → NaN
    rc2 = dict(rc)
    rc2[D.BASE_MODE] = [1, 1, 1, 1]
    got2 = D.site_geometry(vecs, rc2, [fk])
    assert math.isnan(got2[("good", fk)]["toward_correct"])
    assert np.isfinite(got2[("good", fk)]["centroid_shift"])


def test_bootstrap_ci_brackets_mean_and_ignores_nan():
    ci = D.bootstrap_ci([1.0, 1.0, 1.0, float("nan")], n_boot=200, seed=0)
    assert ci["n"] == 3 and ci["mean"] == 1.0 and ci["lo"] == 1.0 and ci["hi"] == 1.0
    ci2 = D.bootstrap_ci(list(np.linspace(0, 1, 50)), n_boot=500, seed=0)
    assert ci2["lo"] < ci2["mean"] < ci2["hi"]
    assert D.bootstrap_ci([])["n"] == 0


def test_pairing_uses_only_sites_where_both_modes_exist():
    recs = []
    for i in range(6):
        for mode, val in (("own", 1.0), ("donor", 0.25)):
            if mode == "donor" and i >= 4:       # 뒤 두 자리엔 donor 가 없다
                continue
            recs.append({"site_id": f"s{i}", "mode": mode, "layer": 0, "position": "last",
                         "centroid_shift": val, "toward_correct": val, "spread_delta": val,
                         "ans_entropy_delta": val, "success_delta": val})
    agg = D.aggregate(recs, n_boot=200, seed=0)
    assert agg["pair_available"] and agg["pair_target"] == "own"
    pr = agg["pairs"][0]
    assert pr["n_sites"] == 4 and abs(pr["toward_correct"] - 0.75) < 1e-9
    own = next(r for r in agg["rows"] if r["mode"] == "own")
    assert own["n_sites"] == 6


# ── 파이프라인(mock forward) ──────────────────────────────────────────────────────
def test_run_end_to_end_with_planted_mock():
    sites, conts = _fixture(n_sites=8, k=4)
    tok = P.MockTok()
    ref = {}
    sel = D.select_continuations(conts, 4, {s["site_id"] for s in sites})
    built = D.build_jobs(tok, sites, sel, "math_plain")
    ref["index"] = built["index"]
    out = D.run(sites, conts, _planting_forward(ref), tok, [0], "math_plain",
                k_per_mode=4, n_boot=200, seed=0)
    rows = {(r["position"], r["mode"]): r for r in out["rows"]}
    for pos in ("last", "mean"):
        assert rows[(pos, "meta")]["toward_correct"] > 0
        assert rows[(pos, "donor")]["toward_correct"] < 0
        assert rows[(pos, "meta")]["n_sites"] == 8
        assert np.isfinite(rows[(pos, "meta")]["ans_entropy_delta"])
        assert abs(rows[(pos, "meta")]["success_delta"]) < 1e-9   # 구성상 정답률이 같다
    assert out["base"]["n_forwards"] == 8 * 3 * 4
    assert out["pair_available"] and out["pair_target"] == "meta"
    json.dumps(out, default=float)


def test_run_pairs_when_donor_present_and_markdown():
    sites, conts = _fixture(n_sites=6, k=4, modes=("nometa", "own", "donor"))
    tok = P.MockTok()
    ref = {}
    sel = D.select_continuations(conts, 4, {s["site_id"] for s in sites})
    ref["index"] = D.build_jobs(tok, sites, sel, "math_plain")["index"]
    out = D.run(sites, conts, _planting_forward(ref, SHIFT_PAIR), tok, [0], "math_plain",
                k_per_mode=4, n_boot=200, seed=0)
    assert out["pair_available"] and out["pair_target"] == "own"
    pr = [p for p in out["pairs"] if p["position"] == "last"][0]
    assert pr["n_sites"] == 6
    assert pr["toward_correct"] > 0        # own(+2) 이 donor(+0.5) 보다 정답 쪽
    md = D.to_markdown(out, out["base"])
    assert "per-mode" in md and "paired" in md and "toward_correct" in md


def test_build_jobs_context_is_generation_prompt_plus_fed_plus_cont():
    sites, conts = _fixture(n_sites=1, k=1, modes=("nometa", "own"))
    tok = P.MockTok()
    built = D.build_jobs(tok, sites, D.select_continuations(conts, 1), "math_plain")
    from math_sites import build_fed
    from src.metacot.math_meta_prompt import render_generation_prompt
    s = sites[0]
    for it in built["index"]:
        fed = build_fed(it["mode"], s["prefix"], s["donor_meta"], s["own_meta"])
        want_head = P._enc(tok, render_generation_prompt(tok, "math_plain", s["problem"]) + fed)
        j = built["jobs"][it["job"]]
        assert it["n_head"] == len(want_head)
        assert j.ids[:len(want_head)] == want_head
        assert j.hidden_span == (len(want_head), len(j.ids))
        assert j.hidden_at == len(j.ids) - 1
