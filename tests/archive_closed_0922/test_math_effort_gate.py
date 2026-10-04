"""math_effort_gate 회귀 시험 (CPU, 모의 생성기 — 모델 없음).

1. 다섯 조건의 프롬프트 조립 — blind/blind_matched 는 `render_generation_prompt` 와 바이트
   동일, wait_forced_* 는 `wait_free` 와 바이트 동일(다른 것은 표본 파라미터뿐),
   pad_forced_* 는 cue 문자열만 다르다.
2. 강제 루프 — min_tokens 경로(한 번에 끝, rounds=1, N **뒤에는** 더 써도 된다)와 rounds
   대체 경로(라운드 회계, end-of-turn 제거 + FORCE_CUE 주입, 남은 예산, 3라운드에서 멈춤,
   이미 N 을 넘은/절단된 행은 건드리지 않음).
3. 채점은 생성 부분만 — 여러 라운드를 이어 붙인 텍스트에서도 접두의 원 답을 보지 않는다.
4. reproduction 산술(수학적 동치 경로 포함) + 새 박스 없음 규칙.
5. blind_matched 의 2패스 예산.
6. 짝지은 Δ 산술.
7. 통과 규칙 진리표(★E1 수정) — 앵커는 blind(전체 예산), nan/flip_wrong>.10/trunc>.20/
   degenerate>.20 → FAIL, 후보가 여럿이면 구제율(rescue) 최고 팔을 고른다(첫 팔이 아니다) —
   그래서 effort_pass 와 control_is_effort 가 항상 같은 팔을 가리킨다. blind_matched 절단
   경고([WEAK-ANCHOR])도 여기.
8. ★E2 퇴화(degenerate) 판정 — 두 저비용 지표(반복 조각 커버리지·즉시-반복 런) + generate()
   가 실제로 플래그를 붙이는지 + [DEGENERATE] 경고.
9. ★E3 강제가 산 것 — frac_stopped_at_floor·answer_changed_after_floor 산술과 강제 팔에만
   붙는다는 것.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_effort_gate as E  # noqa: E402
import math_ruler_pivot as P  # noqa: E402
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402

TOK = P.MockTok()
PROBLEM = "What is 2 plus 2?"
SOL = "Step 1: add them.\nStep 2: I get five.\nThus \\boxed{5}   \n"
FT = [1024, 3072]


class MockSampler:
    """호출을 기록하는 모의 생성기. `fn(prompt, call_idx, n)` 가 생성 목록을 만든다."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def __call__(self, prompts, *, n, max_tokens, min_tokens=0):
        self.calls.append({"prompts": list(prompts), "n": n, "max_tokens": max_tokens,
                           "min_tokens": min_tokens})
        return [self.fn(p, len(self.calls) - 1, n) for p in prompts]


def _g(text="x", n_tokens=10, finish="stop"):
    return {"text": text, "n_tokens": n_tokens, "finish_reason": finish}


# ── 1. 프롬프트 조립 ────────────────────────────────────────────────────────────
def test_blind_and_blind_matched_are_byte_identical_to_generation_prompt():
    anchor = render_generation_prompt(TOK, "math_opt", PROBLEM)
    for cond in ("blind", "blind_matched_1024", "blind_matched_3072"):
        assert E.build_prompt(TOK, "math_opt", cond, PROBLEM, SOL) == anchor


def test_wait_forced_is_byte_identical_to_wait_free():
    free = E.build_prompt(TOK, "math_opt", "wait_free", PROBLEM, SOL)
    assert free == render_generation_prompt(TOK, "math_opt", PROBLEM) + SOL + E.WAIT_CUE
    for n in FT:
        assert E.build_prompt(TOK, "math_opt", f"wait_forced_{n}", PROBLEM, SOL) == free


def test_pad_forced_differs_from_wait_only_in_the_cue():
    wait = E.build_prompt(TOK, "math_opt", "wait_forced_1024", PROBLEM, SOL)
    pad = E.build_prompt(TOK, "math_opt", "pad_forced_1024", PROBLEM, SOL)
    assert wait.endswith(E.WAIT_CUE) and pad.endswith(E.PAD_CUE)
    assert wait[: -len(E.WAIT_CUE)] == pad[: -len(E.PAD_CUE)]
    assert E.WAIT_CUE not in pad and "double-check" not in pad


def test_unknown_condition_is_loud():
    try:
        E.build_prompt(TOK, "math_opt", "nonsense", PROBLEM, SOL)
    except ValueError:
        return
    raise AssertionError("모르는 조건은 ValueError 여야 한다")


def test_cond_names_and_force_floor():
    p1, p2, cc = E.cond_names(FT)
    assert p1 == ["blind", "wait_free", "wait_forced_1024", "wait_forced_3072",
                  "pad_forced_1024", "pad_forced_3072"]
    assert p2 == ["blind_matched_1024", "blind_matched_3072"]
    assert cc == ["wait_forced_1024", "wait_forced_3072", "pad_forced_1024", "pad_forced_3072"]
    assert E.force_floor("wait_forced_3072") == 3072 and E.force_floor("pad_forced_1024") == 1024
    assert E.force_floor("blind") == 0 and E.force_floor("blind_matched_1024") == 0
    assert E.is_continuation("wait_free") and E.is_continuation("pad_forced_1024")
    assert not E.is_continuation("blind") and not E.is_continuation("blind_matched_1024")


# ── 2. 강제 루프 ────────────────────────────────────────────────────────────────
def test_min_tokens_path_is_one_round_and_passes_the_floor():
    s = MockSampler(lambda p, i, n: [_g("done", 1500) for _ in range(n)])
    out = E.generate(s, ["A", "B"], k=2, max_tokens=8192, force_tokens=1024,
                     use_min_tokens=True)
    assert len(s.calls) == 1 and s.calls[0]["min_tokens"] == 1024
    assert s.calls[0]["max_tokens"] == 8192          # 천장은 그대로 — 바닥만 깐다
    assert all(g["rounds"] == 1 for row in out for g in row)
    # N **뒤에는** 더 써도 된다(1,500 > 1,024) — 강제는 바닥이지 천장이 아니다.
    assert all(g["n_gen_tokens"] == 1500 for row in out for g in row)


def test_unforced_arms_pass_min_tokens_zero_and_never_loop():
    s = MockSampler(lambda p, i, n: [_g("fresh", 4000) for _ in range(n)])
    out = E.generate(s, ["A"], k=3, max_tokens=8192, force_tokens=0, use_min_tokens=True)
    assert len(s.calls) == 1 and s.calls[0]["min_tokens"] == 0
    assert [g["rounds"] for g in out[0]] == [1, 1, 1]


def test_rounds_fallback_injects_cue_and_counts_rounds():
    def fn(prompt, i, n):
        return [_g(f"seg{i}<|im_end|>\n", 10) for _ in range(n)]

    s = MockSampler(fn)
    out = E.generate(s, ["A"], k=1, max_tokens=100, force_tokens=25, use_min_tokens=False,
                     max_rounds=3)
    assert len(s.calls) == 3                                   # 1 + (max_rounds-1)
    assert all(c["min_tokens"] == 0 for c in s.calls)           # 대체 경로는 바닥을 안 쓴다
    g = out[0][0]
    assert g["rounds"] == 3 and g["n_gen_tokens"] == 30         # 모델 토큰만(주입 cue 는 제외)
    # 꼬리의 end-of-turn 은 떼고 FORCE_CUE 를 주입해 이어 붙인다.
    assert g["text"] == ("seg0" + E.FORCE_CUE + "seg1" + E.FORCE_CUE + "seg2<|im_end|>\n")
    # 2라운드 프롬프트 = 원 프롬프트 + 그때까지의 텍스트, 예산은 **남은 것**.
    assert s.calls[1]["prompts"] == ["A" + "seg0" + E.FORCE_CUE]
    assert s.calls[1]["max_tokens"] == [90] and s.calls[1]["n"] == 1
    assert s.calls[2]["max_tokens"] == [80]


def test_rounds_fallback_stops_as_soon_as_floor_is_reached():
    s = MockSampler(lambda p, i, n: [_g(f"seg{i}", 40) for _ in range(n)])
    out = E.generate(s, ["A"], k=1, max_tokens=100, force_tokens=25, use_min_tokens=False)
    assert len(s.calls) == 1                     # 첫 라운드에 이미 40 ≥ 25
    assert out[0][0]["rounds"] == 1 and out[0][0]["text"] == "seg0"


def test_rounds_fallback_leaves_truncated_rows_alone():
    s = MockSampler(lambda p, i, n: [_g("cut", 5, "length") for _ in range(n)])
    out = E.generate(s, ["A"], k=1, max_tokens=100, force_tokens=50, use_min_tokens=False)
    assert len(s.calls) == 1 and out[0][0]["truncated"] == 1 and out[0][0]["rounds"] == 1


def test_rounds_fallback_only_resamples_the_short_sequences():
    def fn(prompt, i, n):
        return [_g("short", 5), _g("long", 900)] if i == 0 else [_g("more", 900)]

    s = MockSampler(fn)
    out = E.generate(s, ["A"], k=2, max_tokens=1000, force_tokens=100, use_min_tokens=False)
    assert len(s.calls) == 2 and len(s.calls[1]["prompts"]) == 1
    assert [g["rounds"] for g in out[0]] == [2, 1]
    assert out[0][1]["text"] == "long"           # 이미 긴 행은 손대지 않는다


def test_strip_end_of_turn_removes_stacked_markers_and_space():
    assert E.strip_end_of_turn("abc <|im_end|>\n") == "abc"
    assert E.strip_end_of_turn("abc<|im_end|> <|endoftext|>  ") == "abc"
    assert E.strip_end_of_turn("abc") == "abc" and E.strip_end_of_turn("") == ""


# ── 3. 채점은 생성 부분만 ───────────────────────────────────────────────────────
def test_grading_uses_generated_text_only_across_rounds():
    # 접두의 오답 \boxed{5} 는 안 본다. 2라운드에서 낸 \boxed{4} 가 최종 답이다.
    gen = "recheck \\boxed{9}" + E.FORCE_CUE + " actually \\boxed{4}"
    assert E.effective_r_corr(gen, "4", orig_r_corr=0, cont=True) == 1
    assert E.effective_r_corr("\\boxed{9}", "4", orig_r_corr=0, cont=True) == 0


def test_no_new_box_inherits_the_prefix_verdict():
    # 오답 행: 구제 0(기존 규약과 동일). 정답 행: 뒤집힘 0(activation gate 의 과대 계상 수리).
    assert E.effective_r_corr("no box at all", "4", orig_r_corr=0, cont=True) == 0
    assert E.effective_r_corr("no box at all", "4", orig_r_corr=1, cont=True) == 1
    # blind 계열은 이어쓰기가 아니므로 물려받지 않는다.
    assert E.effective_r_corr("no box at all", "4", orig_r_corr=1, cont=False) == 0


# ── 4. reproduction 산술 ────────────────────────────────────────────────────────
def test_reproduction_uses_math_equivalence_not_string_equality():
    assert E.answer_stats("so \\boxed{0.5}", "\\frac{1}{2}", cont=True) == (1, 0)
    assert E.answer_stats("so \\boxed{5}", "5", cont=True) == (1, 0)
    assert E.answer_stats("so \\boxed{4}", "5", cont=True) == (0, 1)


def test_reproduction_when_no_new_box():
    assert E.answer_stats("nothing", "5", cont=True) == (1, 0)      # 원 답 그대로 = 재생산
    assert E.answer_stats("nothing", "5", cont=False) == (0, 0)     # 새 풀이가 답을 안 냈다


def test_reproduction_rate_is_the_mean_over_rollouts():
    recs = [{"p_c": 0.0, "flip_c": 1.0, "repro_c": 1.0}, {"p_c": 0.0, "flip_c": 1.0,
                                                          "repro_c": 0.5}]
    s = E.summarize(recs, ["c"], k=8, seed=0, n_boot=200)
    assert abs(s["reproduction_rate"]["c"]["mean"] - 0.75) < 1e-9


# ── 5. 2패스 예산 ───────────────────────────────────────────────────────────────
def test_matched_budget_is_the_mean_forced_total_clipped():
    recs = [{"ntok_wait_forced_1024": 1200.0}, {"ntok_wait_forced_1024": 1800.0}]
    assert E.matched_budget(recs, "wait_forced_1024", max_tokens=8192) == 1500
    assert E.matched_budget(recs, "wait_forced_1024", max_tokens=1000) == 1000    # 천장으로 자름
    assert E.matched_budget([], "wait_forced_1024", max_tokens=8192) == 8192      # 안 깎는다
    nan_recs = [{"ntok_wait_forced_1024": float("nan")}]
    assert E.matched_budget(nan_recs, "wait_forced_1024", max_tokens=8192) == 8192


def test_second_pass_uses_that_budget_and_first_pass_does_not():
    rows = [{"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": SOL}]
    s = MockSampler(lambda p, i, n: [_g("out \\boxed{4}", 700) for _ in range(n)])
    kw = dict(variant="math_opt", k=2, max_tokens=8192, lim=10_000)
    got1, _ = E.run_pass(s, TOK, rows, ["wait_forced_1024"], **kw)
    w1, _ = E.rows_from(got1, rows, ["wait_forced_1024"], population="wrong")
    budget = E.matched_budget(w1, "wait_forced_1024", max_tokens=8192)
    assert budget == 700
    E.run_pass(s, TOK, rows, ["blind_matched_1024"], budgets={"blind_matched_1024": budget}, **kw)
    assert s.calls[0]["max_tokens"] == 8192 and s.calls[0]["min_tokens"] == 1024
    assert s.calls[-1]["max_tokens"] == 700 and s.calls[-1]["min_tokens"] == 0


# ── 6. 짝지은 Δ 산술 ───────────────────────────────────────────────────────────
def _recs(vals, n=40):
    """vals: cond → p. 모든 조건에 대해 요약이 요구하는 열을 채운다."""
    row = {}
    for c, p in vals.items():
        row[f"p_{c}"], row[f"flip_{c}"], row[f"repro_{c}"] = p, 1 - p, 1 - p
        row[f"changed_{c}"], row[f"trunc_{c}"] = p, 0.0
        row[f"ntok_{c}"], row[f"rounds_{c}"] = 1000.0, 1.0
    return [dict(row) for _ in range(n)]


def test_paired_deltas_cover_all_four_contrasts():
    vals = {"blind": 0.58, "wait_free": 0.23, "wait_forced_1024": 0.60,
            "pad_forced_1024": 0.40, "blind_matched_1024": 0.30}
    s = E.summarize(_recs(vals), list(vals), force_tokens=[1024], k=8, seed=1, n_boot=200)
    pr = s["paired"]
    assert abs(pr["wait_forced_1024_minus_wait_free"]["mean"] - 0.37) < 1e-9
    assert abs(pr["wait_forced_1024_minus_blind"]["mean"] - 0.02) < 1e-9
    assert abs(pr["wait_forced_1024_minus_blind_matched_1024"]["mean"] - 0.30) < 1e-9
    assert abs(pr["wait_forced_1024_minus_pad_forced_1024"]["mean"] - 0.20) < 1e-9
    assert set(s["sign_p"]) == set(pr)
    assert E.contrasts([1024, 3072]) == [
        ("wait_forced_1024", "wait_free"), ("wait_forced_1024", "blind"),
        ("wait_forced_1024", "blind_matched_1024"), ("wait_forced_1024", "pad_forced_1024"),
        ("wait_forced_3072", "wait_free"), ("wait_forced_3072", "blind"),
        ("wait_forced_3072", "blind_matched_3072"), ("wait_forced_3072", "pad_forced_3072")]


def test_missing_condition_drops_its_contrast_instead_of_crashing():
    vals = {"blind": 0.5, "wait_free": 0.2, "wait_forced_1024": 0.6}
    s = E.summarize(_recs(vals), list(vals), force_tokens=[1024], k=8, seed=1, n_boot=100)
    assert "wait_forced_1024_minus_blind" in s["paired"]
    assert "wait_forced_1024_minus_blind_matched_1024" not in s["paired"]


# ── 7. 통과 규칙 진리표(★E1 수정: 앵커 = blind) ─────────────────────────────────
GOOD = {"lo": 0.05, "hi": 0.20, "mean": 0.12}


def _w(delta=None, trunc=0.05, degen=0.05, rescue_f=0.60, rescue_pad=0.40, pad_delta=None):
    n = 1024
    return {"paired": {f"wait_forced_{n}_minus_blind": delta or GOOD,
                       f"wait_forced_{n}_minus_pad_forced_{n}": pad_delta or GOOD},
            "sign_p": {f"wait_forced_{n}_minus_pad_forced_{n}": 0.01},
            "trunc_rate": {f"wait_forced_{n}": trunc},
            "degenerate_rate": {f"wait_forced_{n}": degen},
            "rescue": {f"wait_forced_{n}": {"mean": rescue_f},
                       f"pad_forced_{n}": {"mean": rescue_pad}}}


def _c(flip=0.04):
    return {"flip_wrong_rate": {"wait_forced_1024": {"mean": flip},
                                "pad_forced_1024": {"mean": 0.02}}}


def test_effort_pass_truth_table():
    assert E.effort_pass(_w(), _c(), [1024]) == (True, "wait_forced_1024")
    weak = {"lo": 0.001, "hi": 0.02, "mean": 0.02}               # 유의하지만 +0.03 미달
    crosses = {"lo": -0.10, "hi": 0.10, "mean": 0.05}            # 평균은 충분하나 CI 가 0 포함
    assert E.effort_pass(_w(delta=weak), _c(), [1024])[0] is False
    assert E.effort_pass(_w(delta=crosses), _c(), [1024])[0] is False
    assert E.effort_pass(_w(), _c(flip=0.11), [1024])[0] is False        # 거짓 경보 과다
    assert E.effort_pass(_w(), _c(flip=0.10), [1024])[0] is True         # 경계는 통과
    assert E.effort_pass(_w(trunc=0.21), _c(), [1024])[0] is False       # 절단 과다(그 팔 자신)
    assert E.effort_pass(_w(trunc=0.20), _c(), [1024])[0] is True        # 경계는 통과
    assert E.effort_pass(_w(degen=0.21), _c(), [1024])[0] is False       # ★E2: 퇴화 과다
    assert E.effort_pass(_w(degen=0.20), _c(), [1024])[0] is True        # 경계는 통과
    assert E.effort_pass(_w(delta={"lo": 0.05, "hi": 0.2, "mean": 0.03}), _c(), [1024])[0] is True


def test_pass_rule_nan_and_missing_are_fail():
    nan_ci = {"lo": float("nan"), "hi": float("nan"), "mean": float("nan")}
    assert E.effort_pass(_w(delta=nan_ci), _c(), [1024]) == (False, None)
    assert E.effort_pass(_w(), _c(flip=float("nan")), [1024]) == (False, None)
    assert E.effort_pass(_w(trunc=float("nan")), _c(), [1024]) == (False, None)
    assert E.effort_pass(_w(degen=float("nan")), _c(), [1024]) == (False, None)    # ★E2
    assert E.effort_pass({}, {}, [1024]) == (False, None)
    assert E.effort_pass(_w(), _c(), []) == (False, None)


def test_any_forced_arm_may_carry_the_pass():
    w = _w()
    w["paired"]["wait_forced_3072_minus_blind"] = GOOD
    w["trunc_rate"]["wait_forced_3072"] = 0.05
    w["degenerate_rate"]["wait_forced_3072"] = 0.05
    w["paired"]["wait_forced_1024_minus_blind"] = {"lo": -0.1, "hi": 0.1, "mean": 0.0}
    c = _c()
    c["flip_wrong_rate"]["wait_forced_3072"] = {"mean": 0.03}
    assert E.effort_pass(w, c, [1024, 3072]) == (True, "wait_forced_3072")


def test_effort_pass_selects_best_rescue_not_first_when_multiple_pass():
    # 둘 다 통과 후보다. 1024 가 리스트에서 **뒤**에 오지만 구제율이 더 높다 — «첫 번째»
    # 규칙이면 3072 를 고를 것이고, 「최고 구제율」 규칙이면 1024 를 고른다.
    w = _w(rescue_f=0.90)
    w["paired"]["wait_forced_3072_minus_blind"] = GOOD
    w["trunc_rate"]["wait_forced_3072"] = 0.05
    w["degenerate_rate"]["wait_forced_3072"] = 0.05
    w["rescue"]["wait_forced_3072"] = {"mean": 0.40}
    c = _c()
    c["flip_wrong_rate"]["wait_forced_3072"] = {"mean": 0.03}
    assert E.effort_pass(w, c, [3072, 1024]) == (True, "wait_forced_1024")


def test_effort_pass_and_control_is_effort_name_the_same_arm():
    w = _w(rescue_f=0.90)
    w["paired"]["wait_forced_3072_minus_blind"] = GOOD
    w["trunc_rate"]["wait_forced_3072"] = 0.05
    w["degenerate_rate"]["wait_forced_3072"] = 0.05
    w["rescue"]["wait_forced_3072"] = {"mean": 0.40}
    c = _c()
    c["flip_wrong_rate"]["wait_forced_3072"] = {"mean": 0.03}
    ok, arm = E.effort_pass(w, c, [1024, 3072])
    cie = E.control_is_effort(w, [1024, 3072])
    assert ok and arm == cie["best_arm"] == "wait_forced_1024"


def test_control_is_effort_picks_the_best_rescue_arm_and_reports_its_ci():
    w = _w()
    w["rescue"]["wait_forced_3072"] = {"mean": 0.90}
    w["paired"]["wait_forced_3072_minus_pad_forced_3072"] = {"lo": -0.02, "hi": 0.02, "mean": 0.0}
    w["sign_p"]["wait_forced_3072_minus_pad_forced_3072"] = 0.9
    cie = E.control_is_effort(w, [1024, 3072])
    assert cie["best_arm"] == "wait_forced_3072" and cie["pad_arm"] == "pad_forced_3072"
    assert cie["delta"]["mean"] == 0.0 and cie["sign_p"] == 0.9     # ≈0 → «더 쓰기» 해석
    empty = E.control_is_effort({}, [1024])
    assert empty["best_arm"] is None and math.isnan(empty["delta"]["mean"])


# ── E1: 약한 앵커 경고([WEAK-ANCHOR]) ───────────────────────────────────────────
def test_weak_anchor_warning_fires_when_blind_matched_truncates_a_lot():
    w = {"trunc_rate": {"blind_matched_1024": 0.25}}
    warn = E.weak_anchor_warnings(w, [1024])
    assert len(warn) == 1
    assert "[WEAK-ANCHOR]" in warn[0] and "blind_matched_1024" in warn[0] and "0.25" in warn[0]


def test_weak_anchor_warning_silent_at_or_below_threshold_and_when_missing():
    assert E.weak_anchor_warnings({"trunc_rate": {"blind_matched_1024": 0.20}}, [1024]) == []
    assert E.weak_anchor_warnings({}, [1024]) == []             # 누락 = 경고 없음(크래시도 없음)


# ── 8. ★E2 퇴화(degenerate) 판정 ────────────────────────────────────────────────
def test_repeated_shingle_coverage_flags_a_looping_string():
    looping = "abcdefghijklmnopqrst" * 20          # 20자 조각이 그대로 20번 반복
    assert E.repeated_shingle_coverage(looping) > E.DEGEN_COVERAGE
    assert E.is_degenerate(looping) is True


def test_repeated_shingle_coverage_is_low_for_clean_text():
    clean = ("Step 1: analyze the equation carefully. Step 2: substitute the known "
             "values and simplify. Step 3: solve for x and double-check the algebra.")
    assert E.repeated_shingle_coverage(clean) < E.DEGEN_COVERAGE
    assert E.has_consecutive_repeat_run(clean) is False
    assert E.is_degenerate(clean) is False


def test_consecutive_repeat_run_flags_immediate_repetition():
    unit = "X" * E.RUN_LEN                       # 조각 경계와 맞도록 정확히 RUN_LEN 자
    looped = unit * E.RUN_COUNT
    assert E.has_consecutive_repeat_run(looped) is True
    assert E.is_degenerate(looped) is True


def test_consecutive_repeat_run_false_for_non_repeating_text():
    assert E.has_consecutive_repeat_run("this text does not repeat itself at all, ever.") is False


def test_generate_flags_degenerate_and_clean_outputs():
    loop_text = "abcdefghijklmnopqrst" * 20
    s_loop = MockSampler(lambda p, i, n: [_g(loop_text, 500) for _ in range(n)])
    out_loop = E.generate(s_loop, ["A"], k=1, max_tokens=8192, force_tokens=0, use_min_tokens=True)
    assert out_loop[0][0]["degenerate"] == 1

    s_clean = MockSampler(lambda p, i, n: [_g("a clean, non-repeating solution.", 50)
                                          for _ in range(n)])
    out_clean = E.generate(s_clean, ["A"], k=1, max_tokens=8192, force_tokens=0,
                           use_min_tokens=True)
    assert out_clean[0][0]["degenerate"] == 0


def test_degenerate_rate_is_the_mean_over_rollouts():
    recs = [{"p_c": 0.0, "flip_c": 1.0, "repro_c": 1.0, "degen_c": 1.0},
            {"p_c": 0.0, "flip_c": 1.0, "repro_c": 1.0, "degen_c": 0.0}]
    s = E.summarize(recs, ["c"], k=8, seed=0, n_boot=200)
    assert abs(s["degenerate_rate"]["c"] - 0.5) < 1e-9


def test_degenerate_warning_fires_above_threshold_and_not_at_boundary():
    summ = {"conds": ["wait_forced_1024"], "degenerate_rate": {"wait_forced_1024": 0.25}}
    assert E.degenerate_warnings(summ) == ["[DEGENERATE] wait_forced_1024 0.250"]
    summ["degenerate_rate"]["wait_forced_1024"] = 0.20
    assert E.degenerate_warnings(summ) == []


# ── 9. ★E3 강제가 산 것 ─────────────────────────────────────────────────────────
def test_frac_stopped_at_floor_and_answer_changed_after_floor():
    rows = [{"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": SOL}]
    got = {(0, "wait_forced_1024"): [
        E._rec(_g("kept \\boxed{5}", 1024)),      # 바닥에서 정확히 멈췄다 + 답 그대로
        E._rec(_g("new \\boxed{9}", 1200)),       # 바닥보다 훨씬 더 썼다 + 답이 바뀌었다
    ]}
    recs, _ = E.rows_from(got, rows, ["wait_forced_1024"], population="wrong")
    r = recs[0]
    assert r["stopfloor_wait_forced_1024"] == 0.5      # 1024 만 바닥 ±5% 안
    assert r["aftfloor_wait_forced_1024"] == 0.5        # \\boxed{9} 만 원래 오답과 다르다


def test_frac_stopped_and_after_floor_absent_for_unforced_conditions():
    rows = [{"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": SOL}]
    got = {(0, "blind"): [E._rec(_g("\\boxed{4}", 10))]}
    recs, _ = E.rows_from(got, rows, ["blind"], population="wrong")
    r = recs[0]
    assert "stopfloor_blind" not in r and "aftfloor_blind" not in r
    assert "degen_blind" in r                            # ★E2 는 모든 조건에 붙는다


# ── 10. 끝에서 끝까지(모의 생성기) ──────────────────────────────────────────────
def test_end_to_end_rows_and_markdown_with_mock_sampler():
    rows = [{"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": SOL}]
    conds, p2, ccond = E.cond_names([1024])

    def fn(prompt, i, n):                       # 이어쓰기는 원 오답 재생산, blind 는 정답
        cont = prompt.endswith((E.WAIT_CUE, E.PAD_CUE))
        return [_g("re-check \\boxed{5}" if cont else "fresh \\boxed{4}", 900) for _ in range(n)]

    s = MockSampler(fn)
    kw = dict(variant="math_opt", k=4, max_tokens=8192, lim=10_000)
    got, drop = E.run_pass(s, TOK, rows, conds + p2, **kw)
    assert drop == 0
    recs, gens = E.rows_from(got, rows, conds + p2, population="wrong")
    assert len(recs) == 1 and len(gens) == 4 * len(conds + p2)
    assert set(g["cond"] for g in gens) == set(conds + p2)
    r = recs[0]
    assert r["p_blind"] == 1.0 and r["p_wait_free"] == 0.0
    assert r["repro_wait_free"] == 1.0 and r["repro_blind"] == 0.0
    assert r["ntok_wait_forced_1024"] == 900.0 and r["rounds_wait_forced_1024"] == 1.0
    assert r["short_wait_forced_1024"] == 1.0          # 900 < 1024 (모의라 바닥이 안 걸린다)
    assert "short_blind" not in r
    w = E.summarize(recs, conds + p2, force_tokens=[1024], k=4, seed=11, n_boot=100)
    c = E.summarize([], ccond, k=4, seed=11, n_boot=100)
    md = E.to_markdown(w, c, [1024], {"force_mode": "min_tokens", "matched_budgets": {}})
    assert "EFFORT FAIL" in md and "CONTROL-IS-EFFORT" in md
    assert "wait_forced_1024_minus_blind_matched_1024" in md


def test_row_missing_a_condition_is_excluded_from_per_row_records():
    rows = [{"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": SOL}]
    got = {(0, "blind"): [E._rec(_g("\\boxed{4}", 10))]}   # rows_from 은 _rec 기록을 받는다
    recs, gens = E.rows_from(got, rows, ["blind", "wait_free"], population="wrong")
    assert recs == [] and len(gens) == 1


# ── 11. g17-lite ──────────────────────────────────────────────────────────────
def test_skip_matched_flag_defaults_off_and_parses():
    ap = E.argparse.ArgumentParser()
    ap.add_argument("--skip_matched", action="store_true")
    assert ap.parse_args([]).skip_matched is False
    assert ap.parse_args(["--skip_matched"]).skip_matched is True


def test_single_force_token_restricts_conditions_to_the_four_g17lite_arms():
    """g17-lite: --force_tokens 3072 하나만 주면 1패스가 정확히 {blind, wait_free,
    wait_forced_3072, pad_forced_3072} 로 좁혀진다(별도 조건-제한 플래그 불필요)."""
    p1, p2, ccond = E.cond_names([3072])
    assert p1 == ["blind", "wait_free", "wait_forced_3072", "pad_forced_3072"]
    assert p2 == ["blind_matched_3072"]
    assert ccond == ["wait_forced_3072", "pad_forced_3072"]  # 정답 행도 그 팔로만 좁혀진다


def test_rows_from_without_p2_conditions_still_yields_records_and_contrasts_are_absent():
    """--skip_matched 경로: p2 조건을 아예 안 넘기면 recs 는 만들어지고, blind_matched
    대조는 (파괴적 KeyError 없이) summarize 에서 그냥 빠진다."""
    rows = [{"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": SOL}]
    conds, _, _ = E.cond_names([1024])

    def fn(prompt, i, n):
        cont = prompt.endswith((E.WAIT_CUE, E.PAD_CUE))
        return [_g("re-check \\boxed{5}" if cont else "fresh \\boxed{4}", 900) for _ in range(n)]

    s = MockSampler(fn)
    kw = dict(variant="math_opt", k=4, max_tokens=8192, lim=10_000)
    got, drop = E.run_pass(s, TOK, rows, conds, **kw)          # p2 없이 1패스만
    recs, _ = E.rows_from(got, rows, conds, population="wrong")
    assert len(recs) == 1
    w = E.summarize(recs, conds, force_tokens=[1024], k=4, seed=11, n_boot=100)
    assert "wait_forced_1024_minus_blind_matched_1024" not in w["paired"]


# ── 10. --resummarize: gens.jsonl + 재채점된 texts.jsonl 로 GPU 없이 다시 요약 ──
def _eff_texts(tmp_path, a1_fix: int):
    """줄번호가 roll_id 의 `#i` 다. w_fix(0번 줄)는 a1_fix=1 이면 재채점으로 정답이 된 단위."""
    import json as _json
    rows = [{"group_id": "gf", "problem": PROBLEM, "gold": "4", "text": "\\boxed{5}",
             "r_corr": a1_fix},
            {"group_id": "g0", "problem": PROBLEM, "gold": "4", "text": "\\boxed{5}",
             "r_corr": 0},
            {"group_id": "g1", "problem": PROBLEM, "gold": "4", "text": "\\boxed{7}",
             "r_corr": 0},
            {"group_id": "c0", "problem": PROBLEM, "gold": "4", "text": "\\boxed{4}",
             "r_corr": 1}]
    p = tmp_path / "texts.jsonl"
    p.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    return p


def _eff_gens(tmp_path, name: str, gids):
    import json as _json
    conds = ["blind", "wait_free", "wait_forced_1024", "pad_forced_1024", "blind_matched_1024"]
    rows = []
    for i, gid in (("gf", 0), ("g0", 1), ("g1", 2)):
        if i not in gids:
            continue
        for c in conds:
            for k in range(2):
                rows.append({"cond": c, "group_id": i, "roll_id": f"{i}#{gid}",
                             "population": "wrong",
                             "text": ("\\boxed{4}" if (c.startswith("wait_forced") or k == 0)
                                      else "\\boxed{9}"),
                             "r_corr": 0, "truncated": 0, "n_gen_tokens": 1200, "rounds": 1,
                             "degenerate": 0})
    for c in ("wait_forced_1024", "pad_forced_1024"):
        for k in range(2):
            rows.append({"cond": c, "group_id": "c0", "roll_id": "c0#3", "population": "correct",
                         "text": "\\boxed{4}", "r_corr": 1, "truncated": 0,
                         "n_gen_tokens": 1200, "rounds": 1, "degenerate": 0})
    p = tmp_path / name
    p.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    return p


def test_effort_resummarize_drops_units_whose_attempt1_is_now_correct(tmp_path):
    import json as _json
    tex = _eff_texts(tmp_path, a1_fix=1)
    with_fix = _eff_gens(tmp_path, "with.jsonl", {"gf", "g0", "g1"})
    without = _eff_gens(tmp_path, "without.jsonl", {"g0", "g1"})
    oa, ob = tmp_path / "oa", tmp_path / "ob"
    assert E.resummarize_effort(str(with_fix), str(tex), out_dir=str(oa), seed=1, n_boot=100) == 0
    assert E.resummarize_effort(str(without), str(tex), out_dir=str(ob), seed=1, n_boot=100) == 0
    sa = _json.loads((oa / "gate_summary.json").read_text())
    sb = _json.loads((ob / "gate_summary.json").read_text())
    assert sa["meta"]["n_units_dropped_a1_correct"] == 1
    assert sb["meta"]["n_units_dropped_a1_correct"] == 0
    assert sa["wrong"]["n_rollouts"] == sb["wrong"]["n_rollouts"] == 2
    assert sa["wrong"] == sb["wrong"] and sa["correct"] == sb["correct"]


def test_effort_resummarize_unchanged_when_no_regrade_change(tmp_path):
    import json as _json
    tex = _eff_texts(tmp_path, a1_fix=0)
    g = _eff_gens(tmp_path, "gens.jsonl", {"gf", "g0", "g1"})
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    assert E.resummarize_effort(str(g), str(tex), out_dir=str(o1), seed=1, n_boot=100) == 0
    assert E.resummarize_effort(str(g), str(tex), out_dir=str(o2), seed=1, n_boot=100) == 0
    s1 = _json.loads((o1 / "gate_summary.json").read_text())
    s2 = _json.loads((o2 / "gate_summary.json").read_text())
    assert s1 == s2
    assert s1["meta"]["n_units_dropped_a1_correct"] == 0 and s1["wrong"]["n_rollouts"] == 3
    # 채점은 생성 텍스트로 다시 난다 — wait_forced 는 둘 다 맞아 rescue 1.0
    assert abs(s1["wrong"]["rescue"]["wait_forced_1024"]["mean"] - 1.0) < 1e-9
    assert abs(s1["wrong"]["rescue"]["blind"]["mean"] - 0.5) < 1e-9


# ── 11. ★O1 다리(bridge) 사전시험 ───────────────────────────────────────────────
def test_bridge_prompts_are_prefix_plus_exact_sft_bridge_string():
    anchor = render_generation_prompt(TOK, "math_opt", PROBLEM)
    free = E.build_prompt(TOK, "math_opt", "bridge_free", PROBLEM, SOL)
    assert free == anchor + SOL + E.bridge_text("5")          # X = 시도-1 의 최종 \boxed 답
    assert free == anchor + SOL + E.BRIDGE_TMPL.format(
        open="<|meta|>", close="<|/meta|>", x="5")
    pad = E.build_prompt(TOK, "math_opt", "bridge_pad_free", PROBLEM, SOL)
    assert pad == anchor + SOL + E.PLACEBO_BRIDGE
    assert free[: -len(E.bridge_text("5"))] == pad[: -len(E.PLACEBO_BRIDGE)]   # 접두는 같다
    # 같은 렌더 경로(열린 assistant 턴 이어쓰기) — wait_free 와 접두가 바이트 동일하다.
    wait = E.build_prompt(TOK, "math_opt", "wait_free", PROBLEM, SOL)
    assert wait[: -len(E.WAIT_CUE)] == pad[: -len(E.PLACEBO_BRIDGE)]


def test_placebo_bridge_is_length_matched_and_carries_no_negation_and_no_x():
    real = E.bridge_text("12345")
    lo, hi = 0.85, 1.15
    try:                                     # 토크나이저가 있으면 토큰 수로 잰다
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(
            "/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507")
        a, b = len(tk.encode(E.PLACEBO_BRIDGE)), len(tk.encode(real))
    except Exception:                        # noqa: BLE001 — 없으면 글자 수로 ±15%
        a, b = len(E.PLACEBO_BRIDGE), len(real)
    assert lo <= a / b <= hi, (a, b)
    for bad in ("12345", "not ", "wrong", "Discarding", "\\boxed"):
        assert bad not in E.PLACEBO_BRIDGE


def test_bridge_does_not_inherit_prefix_answer_and_counts_no_new_box():
    rows = [{"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": SOL}]
    conds = ["blind", "wait_free", *E.BRIDGE_CONDS]

    def fn(prompt, i, n):                      # 어느 조건도 새 \boxed 를 내지 않는다
        return [_g("I will think about it.", 50) for _ in range(n)]

    got, _ = E.run_pass(MockSampler(fn), TOK, rows, conds, variant="math_opt", k=2,
                        max_tokens=8192, lim=10_000)
    r = E.rows_from(got, rows, conds, population="wrong")[0][0]
    assert r["p_wait_free"] == 0.0 and r["repro_wait_free"] == 1.0     # 물려받는다(기존)
    for c in E.BRIDGE_CONDS:                                          # 물려받지 않는다(★O1)
        assert r[f"p_{c}"] == 0.0 and r[f"repro_{c}"] == 0.0
        assert r[f"nobox_{c}"] == 1.0 and r[f"nnobox_{c}"] == 2 and r[f"reemit_{c}"] == 0.0
    s = E.summarize([r], conds, k=2, seed=1, n_boot=50)
    assert s["n_no_new_box"]["bridge_free"] == 2 and s["n_no_new_box"]["blind"] == 2
    assert math.isclose(s["no_new_box_rate"]["bridge_free"], 1.0)


def test_bridge_reemit_x_and_rescue_arithmetic():
    rows = [{"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": SOL}]
    conds = ["bridge_free", "bridge_pad_free"]

    def fn(prompt, i, n):     # bridge_free 는 정답, 위약은 원 오답 X=5 를 다시 낸다
        real = prompt.endswith(E.bridge_text("5"))
        return [_g("so \\boxed{4}" if real else "again \\boxed{5.0}", 77) for _ in range(n)]

    got, _ = E.run_pass(MockSampler(fn), TOK, rows, conds, variant="math_opt", k=2,
                        max_tokens=8192, lim=10_000)
    r = E.rows_from(got, rows, conds, population="wrong")[0][0]
    assert r["p_bridge_free"] == 1.0 and r["p_bridge_pad_free"] == 0.0
    assert r["reemit_bridge_free"] == 0.0 and r["reemit_bridge_pad_free"] == 1.0  # 5.0 ≡ 5
    assert r["nobox_bridge_free"] == 0.0 and r["ntok_bridge_free"] == 77.0


def test_bridge_contrasts_are_reported_with_cis():
    vals = {"blind": 0.58, "wait_free": 0.23, "bridge_free": 0.50, "bridge_pad_free": 0.20}
    s = E.summarize(_recs(vals), list(vals), force_tokens=[], k=8, seed=1, n_boot=200)
    pr = s["paired"]
    assert abs(pr["bridge_free_minus_blind"]["mean"] + 0.08) < 1e-9
    assert abs(pr["bridge_free_minus_bridge_pad_free"]["mean"] - 0.30) < 1e-9
    assert abs(pr["bridge_free_minus_wait_free"]["mean"] - 0.27) < 1e-9
    assert E.BRIDGE_CONTRASTS == [("bridge_free", "blind"),
                                  ("bridge_free", "bridge_pad_free"),
                                  ("bridge_free", "wait_free")]


def _bsumm(rb, r0, lo, hi=1.0):
    return {"rescue": {"bridge_free": {"mean": rb}, "blind": {"mean": r0}},
            "paired": {"bridge_free_minus_bridge_pad_free": {"mean": (lo + hi) / 2,
                                                             "lo": lo, "hi": hi}}}


def test_bridge_gate_truth_table():
    ok, info = E.bridge_pass(_bsumm(0.40, 0.60, 0.05))      # .40 ≥ .6×.60 ∧ 하한>0
    assert ok and info["pass_bridge"] == 1 and info["vs_blind_ok"]
    assert not E.bridge_pass(_bsumm(0.30, 0.60, 0.05))[0]   # .30 < .36 → FAIL
    assert not E.bridge_pass(_bsumm(0.40, 0.60, -0.01))[0]  # 위약 대비 CI 가 0 을 문다
    assert not E.bridge_pass(_bsumm(float("nan"), 0.60, 0.05))[0]
    assert not E.bridge_pass({})[0]                          # 조건이 아예 없다
    md = E.to_markdown(_bsumm(0.40, 0.60, 0.05), {}, [], {"force_mode": "x",
                                                          "matched_budgets": {}})
    assert "**BRIDGE PASS**" in md
    assert "**BRIDGE FAIL**" in E.to_markdown(_bsumm(0.1, 0.6, -0.2), {}, [],
                                              {"force_mode": "x", "matched_budgets": {}})
