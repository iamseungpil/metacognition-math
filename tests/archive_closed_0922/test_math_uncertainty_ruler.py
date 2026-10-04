"""math_uncertainty_ruler 의 CPU 단위 테스트 — 자(ruler) 자체가 거짓말하지 않는지 본다.

GPU 없이 도는 것만 건드린다: 절단 엔트로피, forking 비율, \boxed 토큰 구간 찾기(실제
토크나이저가 있으면 그걸로, 없으면 offset 있는 mock), 기울기 부호, 최저 창, AUC 헬퍼(상수
→ nan), 그리고 «문제 안에서만 사는 자» 와 «문제 간에서만 사는 자» 를 표가 실제로 갈라내는지.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))

import math_uncertainty_ruler as R  # noqa: E402

MODEL = Path("/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507")


# ── 절단 엔트로피 ─────────────────────────────────────────────────────────────

def test_topk_entropy_uniform_is_log_k():
    lp = [math.log(1 / 8)] * 8
    assert R.topk_entropy(lp, k=8) == pytest.approx(math.log(8), abs=1e-9)


def test_topk_entropy_is_renormalized_not_raw():
    """★합이 1이 아닌 top-K 를 줘도 **다시 정규화**한다 — 균등한 8개면 확률합이 0.5여도 log 8."""
    lp = [math.log(0.5 / 8)] * 8
    assert R.topk_entropy(lp, k=8) == pytest.approx(math.log(8), abs=1e-9)


def test_topk_entropy_peaked_is_near_zero():
    lp = [math.log(0.999)] + [math.log(0.001 / 19)] * 19
    assert R.topk_entropy(lp, k=20) < 0.05


def test_topk_entropy_truncates_to_k_highest():
    """K+1 개가 와도(vllm 이 선택 토큰을 끼워 준다) 상위 K 개만 써서 지지집합 크기를 고정한다."""
    lp = [math.log(1 / 4)] * 4 + [math.log(1e-9)]
    assert R.topk_entropy(lp, k=4) == pytest.approx(math.log(4), abs=1e-9)


def test_topk_entropy_empty_is_nan():
    assert math.isnan(R.topk_entropy([], k=20))
    assert math.isnan(R.topk_entropy([float("nan")], k=20))


# ── forking 비율 ─────────────────────────────────────────────────────────────

def test_frac_above_counts_strictly_greater():
    assert R.frac_above([0.5, 1.0, 1.5, 2.0], 1.0) == pytest.approx(0.5)


def test_frac_above_ignores_nan_and_empty():
    assert R.frac_above([float("nan"), 2.0], 1.0) == pytest.approx(1.0)
    assert math.isnan(R.frac_above([float("nan")], 1.0))


# ── 기울기 ───────────────────────────────────────────────────────────────────

def test_ols_slope_sign_and_scale():
    # 0..1 로 정규화된 위치에 대한 기울기이므로, 0→1 로 오르는 수열은 기울기 +1
    assert R.ols_slope([0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(1.0)
    assert R.ols_slope([1.0, 0.75, 0.5, 0.25, 0.0]) == pytest.approx(-1.0)
    assert R.ols_slope([3.0, 3.0, 3.0]) == pytest.approx(0.0)
    assert math.isnan(R.ols_slope([1.0]))


# ── 최저 창 ──────────────────────────────────────────────────────────────────

def test_lowest_window_picks_worst_run():
    v = [0.0] * 10 + [-5.0] * 4 + [0.0] * 10
    assert R.lowest_window_mean(v, window=4) == pytest.approx(-5.0)
    # 창이 수열보다 길면 전체 평균 하나뿐
    assert R.lowest_window_mean([1.0, 3.0], window=10) == pytest.approx(2.0)
    assert math.isnan(R.lowest_window_mean([], window=4))


def test_lowest_window_is_at_most_mean():
    rng = np.random.RandomState(0)
    v = rng.randn(500).tolist()
    assert R.lowest_window_mean(v, 128) <= float(np.mean(v)) + 1e-9


# ── \boxed 토큰 구간 ─────────────────────────────────────────────────────────

class _MockTok:
    """공백 분리 토크나이저 — offset mapping 을 제공한다."""

    def _spans(self, text):
        out, i = [], 0
        for part in text.split(" "):
            if part:
                out.append((i, i + len(part)))
            i += len(part) + 1
        return out

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        sp = self._spans(text)
        d = {"input_ids": list(range(len(sp)))}
        if return_offsets_mapping:
            d["offset_mapping"] = sp
        return d

    def encode(self, text, add_special_tokens=False):
        return list(range(len(self._spans(text))))


class _NoOffsetTok(_MockTok):
    """offset 을 못 주는 토크나이저 — 접두사 길이 경로를 탄다."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if return_offsets_mapping:
            raise NotImplementedError("no offsets")
        return {"input_ids": list(range(len(self._spans(text))))}


def test_boxed_span_offsets_path_mock():
    text = "aa bb \\boxed{42} cc"
    s = text.index("\\boxed")
    e = text.index("}") + 1
    i0, i1 = R.locate_boxed_token_span(_MockTok(), text, s, e)
    assert (i0, i1) == (2, 3)


def test_boxed_span_fallback_path_mock():
    text = "aa bb \\boxed{42} cc"
    s = text.index("\\boxed")
    e = text.index("}") + 1
    i0, i1 = R.locate_boxed_token_span(_NoOffsetTok(), text, s, e)
    assert i0 == 2 and i1 > i0


def test_boxed_span_empty_range():
    assert R.locate_boxed_token_span(_MockTok(), "aa bb", 3, 3) == (0, 0)


@pytest.mark.skipif(not MODEL.exists(), reason="정책 모델이 없다")
def test_boxed_span_on_real_tokenizer():
    from transformers import AutoTokenizer
    from src.training.math_meta import boxed_spans

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    text = "We simplify and conclude. The answer is \\boxed{\\frac{3}{4}} indeed."
    _content, s, e = boxed_spans(text)[-1]
    i0, i1 = R.locate_boxed_token_span(tok, text, s, e)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    assert 0 <= i0 < i1 <= len(ids)
    # 그 토큰들을 되돌리면 박스 스팬을 덮는다
    span_text = tok.decode(ids[i0:i1])
    assert "boxed" in span_text and "frac" in span_text
    # 접두사 경로도 같은 시작을 준다(±1 허용)
    j0, _j1 = len(tok.encode(text[:s], add_special_tokens=False)), None
    assert abs(j0 - i0) <= 1


# ── AUC 헬퍼 ─────────────────────────────────────────────────────────────────

def test_auc_perfect_and_inverted():
    s = [0.0, 1.0, 2.0, 3.0]
    y = [0, 0, 1, 1]
    assert R.auc(s, y) == pytest.approx(1.0)
    assert R.auc([-x for x in s], y) == pytest.approx(0.0)


def test_auc_constant_is_nan_not_half():
    """★상수 지표는 «가르지 못함(.5)» 이 아니라 «잴 수 없음(nan)» 이다 — 둘을 섞으면 평균이 부푼다."""
    assert math.isnan(R.auc([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1]))


def test_auc_single_class_is_nan():
    assert math.isnan(R.auc([0.0, 1.0], [1, 1]))


def test_auc_ignores_nan_rows():
    assert R.auc([float("nan"), 0.0, 1.0], [1, 0, 1]) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_mean():
    v = [0.6, 0.7, 0.8, 0.55, 0.75, 0.65, 0.9, 0.5]
    m, lo, hi = R.bootstrap_mean_ci(v)
    assert lo <= m <= hi
    m2, lo2, _ = R.bootstrap_mean_ci(v)
    assert (m2, lo2) == (m, lo)          # seed 고정 → 재현


def test_bootstrap_ci_tiny_sample_no_ci():
    m, lo, hi = R.bootstrap_mean_ci([0.5, 0.6])
    assert m == pytest.approx(0.55) and math.isnan(lo) and math.isnan(hi)


def test_between_var_frac_extremes():
    gidx = {"a": [0, 1], "b": [2, 3]}
    only_between = np.array([1.0, 1.0, 5.0, 5.0])
    only_within = np.array([1.0, 5.0, 1.0, 5.0])
    assert R.between_problem_var_frac(only_between, gidx) == pytest.approx(1.0)
    assert R.between_problem_var_frac(only_within, gidx) == pytest.approx(0.0)


def test_within_problem_residual_removes_length():
    gidx = {"a": [0, 1, 2, 3]}
    n = np.array([100.0, 200.0, 300.0, 400.0])
    s = 2.0 * n + 7.0                   # 길이의 순수 함수
    res = R.within_problem_residual(s, n, gidx)
    assert np.allclose(res, 0.0, atol=1e-8)


def test_sibling_share_loo_matches_c1_convention():
    rows = [{"final_answer": "4"}, {"final_answer": "4"}, {"final_answer": "5"},
            {"final_answer": None}]
    gidx = {"g": [0, 1, 2, 3]}
    sh = R.sibling_share_loo(rows, gidx)
    assert sh[0] == pytest.approx(1 / 2)   # 형제 {4,5} 중 하나가 일치
    assert sh[2] == pytest.approx(0.0)
    assert math.isnan(sh[3])               # 답 없음 → nan


# ── analyze 단계: 문제 안 자 vs 문제 간 자 ──────────────────────────────────

def _synth_scores(n_problems=40, k=8, seed=3):
    """★두 지표를 심는다.
    mean_logp  = 문제 **안에서** 정오와 함께 움직인다(문제 간 성분 없음).
    answer_logp= 문제 난이도만 읽고 문제 안에선 잡음(= §C1 의 은닉 프로브 흉내).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for g in range(n_problems):
        p = rng.uniform(0.2, 0.8)
        ys = (rng.rand(k) < p).astype(float)
        if ys.sum() in (0, k):           # 혼합문제로 강제
            ys[0], ys[-1] = 1.0, 0.0
        for j, yv in enumerate(ys):
            rows.append({
                "source": "synth",
                "group_id": f"g{g}",
                "r_corr": float(yv),
                "truncated": 0.0,
                "final_answer": "A" if yv else "B",
                "n_tok": 1000.0,
                "n_resp_tok": 1000 + int(rng.randn() * 5),
                "mean_logp": -1.0 + 1.5 * yv + 0.10 * rng.randn(),
                "min_logp": rng.randn(),
                "mean_entropy": 0.5 + 0.05 * rng.randn(),
                "max_entropy": 2.0,                       # ★상수 → nan 이어야 한다
                "frac_high_entropy": 0.1 + 0.01 * rng.randn(),
                "mean_logp_at_forks": rng.randn(),
                "entropy_first64": 0.4 + 0.05 * rng.randn(),
                "entropy_last64": 0.4 + 0.05 * rng.randn(),
                "entropy_slope": 0.01 * rng.randn(),
                "answer_logp": 3.0 * p + 0.05 * rng.randn(),   # 난이도만
                "tail_logp_64": rng.randn(),
                "lowest_window_logp": rng.randn(),
            })
    return rows


def test_analyze_separates_within_from_between():
    res = R._analyze_population(_synth_scores())
    by = {t["metric"]: t for t in res["table"]}
    within_metric, between_metric = by["mean_logp"], by["answer_logp"]
    # 문제 안 자는 문제 안 AUC 가 높고
    assert within_metric["auc_within"] > 0.9
    # 난이도 자는 풀링은 높지만 문제 안에서는 우연 수준이고
    assert between_metric["auc_pooled"] > 0.6
    assert 0.35 < between_metric["auc_within"] < 0.65
    # 난이도 ρ 는 양쪽 다 높다(문제 안 자의 그룹 평균도 pass rate 를 따라간다) — 둘을 가르는
    # 것은 ρ 가 아니라 **분산 분해**다.
    assert between_metric["difficulty_rho"] > 0.5
    assert between_metric["between_var_frac"] > 0.9
    assert within_metric["between_var_frac"] < 0.5
    # BEST-WITHIN 은 문제 안 자를 고른다
    assert res["best_within"]["metric"] == "mean_logp"


def test_analyze_constant_metric_yields_nan_within():
    res = R._analyze_population(_synth_scores())
    by = {t["metric"]: t for t in res["table"]}
    assert math.isnan(by["max_entropy"]["auc_within"])
    assert math.isnan(by["max_entropy"]["auc_pooled"])


def test_analyze_length_residual_kills_pure_length_signal():
    """n_resp_tok 을 «길이만» 으로 만들면 길이 잔차화 열이 그 자를 nan/우연으로 무너뜨린다."""
    rows = _synth_scores()
    for r in rows:                       # mean_logp 를 길이의 함수로 바꾼다
        r["n_resp_tok"] = 1000 - int(300 * r["r_corr"])
        r["mean_logp"] = -0.001 * r["n_resp_tok"]
    res = R._analyze_population(rows)
    by = {t["metric"]: t for t in res["table"]}
    assert by["mean_logp"]["auc_within"] > 0.9
    assert math.isnan(by["mean_logp"]["auc_within_lenresid"]) or \
        abs(by["mean_logp"]["auc_within_lenresid"] - 0.5) < 0.2


def test_analyze_stage_end_to_end(tmp_path):
    p = tmp_path / "scores.jsonl"
    with p.open("w") as f:
        for r in _synth_scores(n_problems=12):
            f.write(json.dumps(r) + "\n")
    R.analyze_stage(R.build_parser().parse_args(
        ["--analyze", "--out_dir", str(tmp_path)]))
    md = (tmp_path / "ruler_summary.md").read_text()
    assert "BEST-WITHIN:" in md and "AUC 문제 안" in md
    js = json.loads((tmp_path / "ruler_summary.json").read_text())
    assert "synth" in js and js["synth"]["best_within"]["metric"] == "mean_logp"


# ── 위치별 logprob 추출(vllm 구조 mock) ─────────────────────────────────────

class _LP:
    def __init__(self, logprob):
        self.logprob = logprob


def test_pos_logprobs_handles_none_at_position_zero():
    """★vllm 은 프롬프트 첫 자리에 조건이 없어 None(FlatLogprobs 에선 빈 dict)을 준다."""
    plp = [None, {7: _LP(-0.5), 9: _LP(-2.0)}, {}]
    assert all(math.isnan(x) for x in [R._pos_logprobs(plp, 0, 7)[0],
                                       R._pos_logprobs(plp, 2, 7)[0]])
    chosen, vals = R._pos_logprobs(plp, 1, 7)
    assert chosen == pytest.approx(-0.5) and sorted(vals) == [-2.0, -0.5]


def test_pos_logprobs_missing_chosen_token_is_nan():
    plp = [None, {9: _LP(-2.0)}]
    chosen, vals = R._pos_logprobs(plp, 1, 7)
    assert math.isnan(chosen) and vals == [-2.0]


def test_pos_logprobs_out_of_range():
    assert math.isnan(R._pos_logprobs([None], 5, 7)[0])


def test_metric_signs_cover_all_reported_metrics():
    for m in R.BASE_METRICS + R.REF_METRICS:
        assert m in R.METRIC_SIGNS
