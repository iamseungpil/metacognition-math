"""math_meta_content_gate 회귀 시험 (CPU, 모델 없음 — MockTok).

1. prompt 모드 프롬프트 조립 — `plain` 은 render_generation_prompt 와 **바이트 동일**,
   나머지는 그 접미 하나만 다르다(사용자 턴 끝에만 붙는다). {CAND} 행위는 cand= 로 채운다.
2. continue 모드 이어쓰기 — cue 로 끝나고 그 뒤에 턴 종료 표식이 없다, `blind` 는 바이트 동일.
3. 채점 — 이어쓴 부분만 본다(접두의 원 오답 박스는 안 본다), 새 박스가 없으면 접두 답 유지.
4. 준수 정규식 — 행위마다 양성/음성 한 쌍, 대조군은 nan, {CAND} 행위는 cand= 필수.
5. verification_first 후보 선정 — 다수결도 gold 도 아닌 오답만 고른다, 무작위 대조는 gold 와
   절대 겹치지 않는다.
6. 조건별 지표 산술(합성 모집단).
7. 통과 규칙 진리표(레거시/무단계) — nan / complied<.5 / trunc>.20 은 전부 FAIL.
8. screen/confirm 두 단계 — 문제 분할이 겹치지 않고 재현 가능하다, screen 은 판정을 안 낸다,
   confirm 은 Holm 보정으로 판정하고 보정 전엔 통과했을 것이 보정 후 FAIL 로 뒤집힐 수 있다.
9. META_ACTS 가 조건 이름의 단일 진실 원천 — 행위를 더하면 조건·요약·순위에 전파된다.
10. pad_length(F1) — 길이-매치 대조: 준수 정규식(3문장 상한 추정), (act − pad_length) 가
    (act − filler) 옆에 보고되는지, pad_length 가 없을 땐 그 열이 아예 안 나오는지.
11. compliance 는 절대값이 아니라 REF[mode] 대비 **차분**(F2) — complied_rate_ref/_delta 산술과
    통과 규칙의 차분 조건, [TOKEN-CONFOUND?] (filler 는 이기고 pad_length 는 못 이기는 행위).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_meta_content_gate as C  # noqa: E402
import math_ruler_pivot as P  # noqa: E402
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402

TOK = P.MockTok()
PROBLEM = "What is 2 plus 2?"
SOL = "Step 1: add them.\nStep 2: I get five.\nThus \\boxed{5}   \n"
VAR = "math_opt"

CAND_ACTS = ("verification_first", "verification_first_random")


# ── 1. prompt 모드 프롬프트 조립 ────────────────────────────────────────────────
def test_plain_prompt_is_byte_identical_to_generation_prompt():
    assert (C.act_prompt(TOK, VAR, PROBLEM, "plain")
            == render_generation_prompt(TOK, VAR, PROBLEM))
    assert C.META_ACTS["plain"].suffix == ""


def test_every_other_prompt_differs_only_by_its_suffix():
    base = render_generation_prompt(TOK, VAR, PROBLEM)
    for name in C.prompt_conds():
        if name in CAND_ACTS:
            continue                     # {CAND} 행위는 아래 전용 시험에서 다룬다
        q = C.act_prompt(TOK, VAR, PROBLEM, name)
        suf = C.META_ACTS[name].suffix
        if not suf:
            assert q == base
            continue
        # 접미를 지우면 기준 프롬프트와 바이트 동일해야 한다 — 다른 데가 갈리면 Δ 가
        # «행위의 효과»가 아니라 «프롬프트가 달라진 효과»를 섞는다.
        assert suf in q
        assert q.replace(suf, "", 1) == base
        assert q != base


def test_prompt_conds_are_exactly_the_thirteen_shipped_conditions():
    assert C.prompt_conds() == ("plain", "filler", "substitute", "magnitude", "special_case",
                                "constraint", "verification_first", "verification_first_random",
                                "backward_mask", "plan", "stepcheck", "recompute", "pad_length")
    assert len(C.prompt_conds()) == 13
    assert "units" not in C.META_ACTS


def test_continue_conds_include_blind_and_the_content_free_control():
    cc = C.continue_conds()
    assert cc[0] == "blind" and "filler_continue" in cc
    assert set(cc) == {"blind", "filler_continue", "substitute", "magnitude", "special_case",
                       "constraint", "recompute"}
    # 새 문헌-근거 행위 셋은 continue 모드에는 없다(META_ACTS 의 cue=None).
    for name in (*CAND_ACTS, "backward_mask"):
        assert name not in cc


def test_prompt_mode_condition_raises_on_continue_only_name():
    import pytest  # noqa: PLC0415
    with pytest.raises(KeyError):
        C.act_prompt(TOK, VAR, PROBLEM, "blind")
    with pytest.raises(KeyError):
        C.act_continuation(TOK, VAR, PROBLEM, SOL, "backward_mask")


def test_cand_acts_need_a_candidate_value_and_substitute_literally():
    for name in CAND_ACTS:
        import pytest  # noqa: PLC0415
        with pytest.raises(KeyError):
            C.act_prompt(TOK, VAR, PROBLEM, name)             # cand 없으면 fail-loud
        q = C.act_prompt(TOK, VAR, PROBLEM, name, cand="17")
        assert "{CAND}" not in q
        assert "A possible answer to this problem is 17." in q
        assert "\\boxed{}" in q


def test_backward_mask_is_prompt_only_static_suffix():
    q = C.act_prompt(TOK, VAR, PROBLEM, "backward_mask")
    assert "work backwards" in q
    assert C.META_ACTS["backward_mask"].cue is None


# ── 2. continue 모드 이어쓰기 ───────────────────────────────────────────────────
def test_blind_continuation_is_byte_identical_fresh_solve():
    assert (C.act_continuation(TOK, VAR, PROBLEM, SOL, "blind")
            == render_generation_prompt(TOK, VAR, PROBLEM))


def test_continuation_ends_with_cue_and_has_no_end_of_turn_after_it():
    head = render_generation_prompt(TOK, VAR, PROBLEM)
    for name in C.continue_conds():
        cue = C.META_ACTS[name].cue
        if not cue:
            continue
        q = C.act_continuation(TOK, VAR, PROBLEM, SOL, name)
        assert q == head + SOL + cue
        assert q.endswith(cue)
        # MockTok 의 턴 종료/개시 표식은 "<assistant> " — 열린 턴 하나뿐이어야 하고
        # cue 뒤에 다시 나오면 안 된다(그러면 이어쓰기가 아니라 새 턴이다).
        assert q.count("<assistant> ") == 1
        assert not q[len(head):].rstrip().endswith("<assistant>")


def test_continuation_matches_hf_continue_final_message_on_real_tokenizer():
    """실 토크나이저가 있으면 이어쓰기 == apply_chat_template(continue_final_message=True).
    없으면 스킵(CPU 시험 환경에 모델 캐시가 없을 수 있다)."""
    import pytest  # noqa: PLC0415
    model_dir = "/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507"
    if not Path(model_dir).exists():
        pytest.skip("실 토크나이저 없음")
    from transformers import AutoTokenizer  # noqa: PLC0415

    from src.metacot.math_meta_prompt import build_math_prompt  # noqa: PLC0415
    tok = AutoTokenizer.from_pretrained(model_dir)
    cue = C.META_ACTS["substitute"].cue
    a = C.act_continuation(tok, VAR, PROBLEM, SOL, "substitute")
    msgs = build_math_prompt(PROBLEM, VAR) + [{"role": "assistant", "content": SOL + cue}]
    b = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                continue_final_message=True, enable_thinking=False)
    assert a == b


# ── 3. 채점 ────────────────────────────────────────────────────────────────────
def test_grade_continuation_uses_generated_part_only():
    # 접두는 \boxed{5}(오답), 이어쓴 부분이 \boxed{4}(정답) → 정답.
    corr, nnb = C.grade_continuation("I redo it: \\boxed{4}", SOL, "4")
    assert corr == 1 and nnb == 0


def test_grade_continuation_no_new_boxed_keeps_prefix_answer():
    # 이어쓴 부분에 새 박스가 없으면 답은 접두 그대로(= 원래 오답) — no_new_boxed=1.
    corr, nnb = C.grade_continuation("Looks fine to me.", SOL, "4")
    assert corr == 0 and nnb == 1
    # 접두가 정답이었다면 그대로 정답이 유지된다(규칙이 접두를 본다는 것을 고정).
    corr2, nnb2 = C.grade_continuation("Looks fine to me.", "so \\boxed{4}", "4")
    assert corr2 == 1 and nnb2 == 1


def test_grade_continuation_ignores_prefix_when_generation_has_a_box():
    # 접두에 정답이 있어도 이어쓴 부분의 마지막 박스가 오답이면 오답이다.
    corr, nnb = C.grade_continuation("hmm, actually \\boxed{9}", "so \\boxed{4}", "4")
    assert corr == 0 and nnb == 0


# ── 4. 준수 정규식 ──────────────────────────────────────────────────────────────
POSITIVE = {
    "substitute": "Substituting x = 3 back into the original equation gives 12 = 12.",
    "magnitude": "The sign should be positive and the magnitude is around 100.",
    "special_case": "Let me check the degenerate case where n = 1.",
    "constraint": "The problem asks for the sum of all positive divisors.",
    "plan": "My approach is to use the Chinese remainder theorem.",
    "stepcheck": "Let me double-check that arithmetic: 7 times 6 is 42.",
    "recompute": "Let me try a different method and compare the two answers.",
    "backward_mask": ("Working backwards, I recover the original perimeter and it matches the "
                       "stated value."),
    "pad_length": ("Careful work matters because small errors compound. Checking assumptions "
                   "builds confidence in the result. Diligence prevents costly mistakes down "
                   "the line. Now let's solve: \\boxed{4}"),
}
NEGATIVE = "We add the two numbers together and write the result down."


def test_compliance_regexes_fire_on_their_act_and_not_on_a_bare_solution():
    for name, txt in POSITIVE.items():
        assert C.complied(name, txt) == 1.0, name
        assert C.complied(name, NEGATIVE) == 0.0, name


def test_compliance_is_nan_for_controls():
    for name in C.CONTROLS:
        assert math.isnan(C.complied(name, POSITIVE["substitute"]))
    assert C.CONTROLS == frozenset({"plain", "blind", "filler", "filler_continue"})


def test_pad_length_compliance_needs_at_least_three_sentences_before_math_starts():
    """F1: 단순한 상한 추정 — 숫자/역슬래시(수식 마커) 전에 문장부호가 3번 있어야 한다."""
    one = "Careful work matters a lot. \\boxed{4}"
    two = "Careful work matters a lot. Checking helps too. \\boxed{4}"
    three = "Careful work matters a lot. Checking helps too. Rigor pays off. \\boxed{4}"
    assert C.complied("pad_length", one) == 0.0
    assert C.complied("pad_length", two) == 0.0
    assert C.complied("pad_length", three) == 1.0
    # 첫 문장이 나오기도 전에 숫자가 나오면(수학이 먼저 시작되면) 실패한다.
    assert C.complied("pad_length", "5 is the answer. Sentence two. Sentence three.") == 0.0


def test_pad_length_is_prompt_only_and_not_a_control():
    assert C.META_ACTS["pad_length"].cue is None
    assert "pad_length" in C.prompt_conds()
    assert "pad_length" not in C.continue_conds()
    assert "pad_length" not in C.CONTROLS          # rx 있음(준수율은 잰다) — CONTROLS 는 아니다
    assert "pad_length" in C.NOT_WINNERS           # 그래도 승자 후보에서는 뺀다(진단용 대조)


def test_plan_compliance_only_reads_the_head_of_the_response():
    tail = ("x" * (C.PLAN_RX_HEAD_CHARS + 50)) + " another approach would also work"
    assert C.complied("plan", tail) == 0.0
    assert C.complied("plan", "My plan: " + tail) == 1.0


def test_cand_act_compliance_requires_candidate_and_fires_on_verdict_or_named_value():
    import pytest  # noqa: PLC0415
    for name in CAND_ACTS:
        with pytest.raises(KeyError):
            C.complied(name, "The candidate answer is wrong.")         # cand 없으면 fail-loud
        verdict = "The candidate answer is wrong, so let me solve it myself."
        assert C.complied(name, verdict, cand="17") == 1.0
        named = "Checking 17: plugging it back in gives a contradiction, so 17 is not correct."
        assert C.complied(name, named, cand="17") == 1.0
        assert C.complied(name, NEGATIVE, cand="17") == 0.0
        # 후보값이 다르면 같은 텍스트라도 «이름 불러 재계산»으로는 안 읽힌다(그 값이 아니므로).
        assert C.complied(name, named, cand="99") == 0.0


# ── 5. verification_first 후보 선정 ─────────────────────────────────────────────
def test_pick_non_plurality_wrong_excludes_gold_and_prefers_larger_cluster():
    # 8 형제: 5개가 "10"(오답, 다수결), 2개가 "7"(오답, 소수), 1개가 "4"(정답=gold).
    answers = ["10", "10", "10", "10", "10", "7", "7", "4"]
    assert C.pick_non_plurality_wrong(answers, gold="4") == "7"


def test_pick_non_plurality_wrong_skips_when_only_minority_is_correct():
    # 다수(6개)가 오답 "10"에 합의, 소수(2개)가 정답 "4" — 다수결 아닌 군집이 gold 뿐 → 스킵.
    answers = ["10"] * 6 + ["4", "4"]
    assert C.pick_non_plurality_wrong(answers, gold="4") is None


def test_pick_non_plurality_wrong_skips_when_everyone_agrees_or_no_answers():
    assert C.pick_non_plurality_wrong(["4"] * 8, gold="4") is None
    assert C.pick_non_plurality_wrong([], gold="4") is None
    assert C.pick_non_plurality_wrong(["", "", ""], gold="4") is None


def test_pick_non_plurality_wrong_treats_numeric_equivalence_not_just_string_equality():
    # "0.5" 와 "1/2" 는 math_verify 동치 — 같은 군집으로 묶여야 다수결 판단이 안 깨진다.
    answers = ["0.5", "1/2", "0.5", "1/2", "9"]
    # {0.5,1/2} 군집(4개)이 다수결, "9"(1개)가 유일한 비다수결 — gold 가 "9" 가 아니면 "9".
    assert C.pick_non_plurality_wrong(answers, gold="0.5") == "9"


def test_pick_random_wrong_cand_never_equals_gold():
    import random  # noqa: PLC0415
    rng = random.Random(0)
    for _ in range(200):
        c = C.pick_random_wrong_cand("4", rng)
        assert c is not None
        assert c != "4"
        assert C.CAND_LO <= int(c) <= C.CAND_HI


# ── 6. 지표 산술(합성 모집단) ───────────────────────────────────────────────────
def _rec(accs: dict, *, tag="t", trunc=0.0, tokens=100.0, comp=1.0, comp_ref=0.0, unit="u"):
    # ★comp_ref 기본 0.0 — (comp − comp_ref) 델타가 항상 MIN_COMPLIED_DELTA 를 가볍게 넘어,
    #   compliance-delta 를 안 건드리는 기존 산술 시험들이 그대로 통과한다.
    return {"unit_id": unit, "tag": tag, "group_id": unit, "gold": "4",
            "cond": {c: {"acc": accs[c], "trunc": trunc, "tokens": tokens,
                         "complied": (comp if C.META_ACTS[c].rx else float("nan")),
                         "complied_ref": (comp_ref if C.META_ACTS[c].rx else float("nan")),
                         "no_new_boxed": 0.0}
                     for c in accs}}


def _pop(n=40, **over):
    accs = {c: 0.30 for c in C.prompt_conds()}
    accs["plain"] = 0.30
    accs["filler"] = 0.35
    accs.update(over)
    return [_rec(accs, unit=f"u{i}") for i in range(n)]


def test_per_condition_metric_arithmetic():
    s = C.summarize(_pop(substitute=0.50, recompute=0.20), C.prompt_conds(),
                    ref="plain", control="filler", k=8, seed=0, n_boot=200)
    assert abs(s["acc"]["substitute"]["mean"] - 0.50) < 1e-9
    assert abs(s["delta_vs_ref"]["substitute"]["mean"] - 0.20) < 1e-9      # 0.50 − 0.30
    assert abs(s["delta_vs_control"]["substitute"]["mean"] - 0.15) < 1e-9  # 0.50 − 0.35
    assert abs(s["delta_vs_control"]["recompute"]["mean"] + 0.15) < 1e-9   # 0.20 − 0.35
    assert abs(s["delta_vs_control"]["filler"]["mean"]) < 1e-9
    # 순위는 (act − filler) 내림차순, 대조군은 순위에 안 들어간다.
    assert s["acts_ranked"][0] == "substitute" and s["acts_ranked"][-1] == "recompute"
    assert "filler" not in s["acts_ranked"] and "plain" not in s["acts_ranked"]
    assert math.isnan(s["complied_rate"]["filler"]) and s["complied_rate"]["substitute"] == 1.0


def test_token_cost_and_compute_bought_flag():
    recs = _pop(substitute=0.50)
    for r in recs:                              # substitute 만 토큰 2배
        r["cond"]["substitute"]["tokens"] = 200.0
    s = C.summarize(recs, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200)
    assert abs(s["token_cost_vs_ref"]["substitute"] - 100.0) < 1e-9
    assert abs(s["token_ratio_vs_ref"]["substitute"] - 2.0) < 1e-9
    assert s["winners"] == ["substitute"] and s["compute_bought"] == ["substitute"]


def test_budget_flag_fires_above_threshold():
    recs = _pop()
    for r in recs:
        r["cond"]["recompute"]["trunc"] = 0.25
    s = C.summarize(recs, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200)
    assert s["budget_flag"] == ["recompute"]
    assert "[BUDGET?]" in C.to_markdown(s, mode="prompt")


# ── 7. 통과 규칙 진리표(레거시/무단계) ───────────────────────────────────────────
def _summ(delta=None, comp=0.9, comp_ref=0.30, trunc=0.0):
    # ★comp_ref 기본 0.30 — comp 기본 0.9 와의 델타가 0.60 ≥ MIN_COMPLIED_DELTA(0.20) 라
    # 기존 진리표(경계값 comp=0.50 포함, 델타 0.20 = 경계에서도 통과)가 그대로 성립한다.
    good = {"lo": 0.05, "hi": 0.25, "mean": 0.15}
    return {"acts_ranked": ["substitute"],
            "delta_vs_control": {"substitute": delta or good},
            "complied_rate": {"substitute": comp},
            "complied_rate_ref": {"substitute": comp_ref},
            "complied_rate_delta": {"substitute": (comp - comp_ref if math.isfinite(comp)
                                                    and math.isfinite(comp_ref) else float("nan"))},
            "trunc_rate": {"substitute": trunc}}


def test_pass_rule_truth_table():
    assert C.content_effect_pass(_summ())[0] is True
    weak = {"lo": 0.001, "hi": 0.02, "mean": 0.02}        # 유의하지만 +0.03 미달
    assert C.content_effect_pass(_summ(delta=weak))[0] is False
    boundary = {"lo": 0.001, "hi": 0.06, "mean": 0.03}    # 정확히 +0.03 → 통과(≥)
    assert C.content_effect_pass(_summ(delta=boundary))[0] is True
    crosses = {"lo": -0.05, "hi": 0.30, "mean": 0.15}
    assert C.content_effect_pass(_summ(delta=crosses))[0] is False
    assert C.content_effect_pass(_summ(comp=0.49))[0] is False     # 준수 미달 → 아무것도 안 쟀다
    assert C.content_effect_pass(_summ(comp=0.50))[0] is True
    assert C.content_effect_pass(_summ(trunc=0.21))[0] is False    # 절단 과다
    assert C.content_effect_pass(_summ(trunc=0.20))[0] is True


def test_pass_rule_requires_compliance_delta_over_reference_not_just_absolute():
    """F2(0915): 절대 complied_rate ≥ .5 만으로는 약하다 — REF[mode] 대비 +.20 도 요구한다.
    같은 절대 준수율(.60)이라도 기준이 높으면(자연발생) 실패하고, 낮으면 통과한다."""
    assert C.content_effect_pass(_summ(comp=0.60, comp_ref=0.55))[0] is False   # Δ=.05 < .20
    assert C.content_effect_pass(_summ(comp=0.60, comp_ref=0.30))[0] is True    # Δ=.30 ≥ .20
    # 경계: 정확히 .20 이면 통과(≥, 다른 게이트들과 같은 관례). 0.9−0.7 은 부동소수점으로도
    # .20000000000000007 이라 확실히 ≥.20 — 뺄셈이 딱 .2 를 안 주는 조합(.6−.4 등)은 피한다.
    assert C.content_effect_pass(_summ(comp=0.90, comp_ref=0.70))[0] is True
    assert C.content_effect_pass(_summ(comp=0.60, comp_ref=0.41))[0] is False


def test_pass_rule_nan_is_fail():
    nan = float("nan")
    assert C.content_effect_pass(_summ(delta={"lo": nan, "hi": nan, "mean": nan}))[0] is False
    assert C.content_effect_pass(_summ(comp=nan))[0] is False
    assert C.content_effect_pass(_summ(trunc=nan))[0] is False
    assert C.content_effect_pass({})[0] is False


def test_controls_can_never_win_even_when_they_beat_the_filler():
    # plain 이 filler 보다 크게 높아도 승자가 될 수 없다(행위를 이름 부르지 않으므로).
    s = C.summarize(_pop(**{"plain": 0.60}), C.prompt_conds(), ref="plain", control="filler",
                    k=8, seed=0, n_boot=200)
    assert s["winners"] == [] and s["pass_content_effect"] == 0


# ── 8. screen/confirm 두 단계 ────────────────────────────────────────────────────
def test_split_bucket_is_disjoint_stable_and_roughly_balanced():
    gids = [f"g{i}" for i in range(600)]
    b1 = {g: C.split_bucket(g, 11) for g in gids}
    b2 = {g: C.split_bucket(g, 11) for g in gids}
    assert b1 == b2                                   # 재현 가능(같은 입력 → 같은 절반)
    assert set(b1.values()) <= {"screen", "confirm"}
    screen = {g for g, v in b1.items() if v == "screen"}
    confirm = {g for g, v in b1.items() if v == "confirm"}
    assert screen.isdisjoint(confirm)
    assert screen | confirm == set(gids)
    assert 200 < len(screen) < 400                    # 치우치지 않는다(대략 절반)
    # 다른 split_seed 는 다른(독립적인) 분할을 낸다.
    b3 = {g: C.split_bucket(g, 12) for g in gids}
    assert b3 != b1


def test_screen_stage_emits_ranking_but_no_pass_verdict_key():
    s = C.summarize(_pop(substitute=0.50), C.prompt_conds(), ref="plain", control="filler",
                    k=4, seed=0, n_boot=200, stage="screen")
    assert s["stage"] == "screen"
    assert s["acts_ranked"][0] == "substitute"          # 순위는 낸다
    # ★통과 판정 키 자체가 없다 — False 로도 안 남긴다.
    assert "pass_content_effect" not in s
    assert "winners" not in s
    assert "compute_bought" not in s
    md = C.to_markdown(s, mode="prompt")
    assert "SCREEN — 순위만, 판정 없음" in md
    # ★설명 문장 안에 "PASS/FAIL" 이라는 낱말이 나오는 것과(단계를 설명하는 산문) **판정 자체를
    #   찍는 것**은 다르다 — 실제 판정 줄("**CONTENT-EFFECT PASS/FAIL**")이 없는지만 고정한다.
    assert "CONTENT-EFFECT" not in md


def test_confirm_stage_applies_holm_and_can_flip_a_marginal_pass_to_fail():
    good = {"lo": 0.05, "hi": 0.25, "mean": 0.15}
    summ = {
        "acts_ranked": ["a", "b", "c"],
        "delta_vs_control": {"a": good, "b": good, "c": good},
        "complied_rate": {"a": 0.9, "b": 0.9, "c": 0.9},
        "complied_rate_delta": {"a": 0.4, "b": 0.4, "c": 0.4},   # comp − ref ≥ .20, 다 통과
        "trunc_rate": {"a": 0.0, "b": 0.0, "c": 0.0},
        "sign_p_vs_control": {"a": 0.001, "b": 0.04, "c": 0.60},
    }
    ok, winners, adj = C.content_effect_pass_confirm(summ, ["a", "b", "c"])
    # Holm(m=3): a → 3*.001=.003(생존) · b → 2*.04=.08(누적 최댓값도 .08, 보정 전 .04<.05 로
    # 통과했을 것이 보정 후 FAIL 로 뒤집힌다) · c → 1*.60=.60(생존 못함).
    assert adj["a"] < 0.05 and "a" in winners
    assert adj["b"] > 0.05 and "b" not in winners
    assert adj["c"] > 0.05 and "c" not in winners
    assert ok is True and winners == ["a"]


def test_holm_adjust_is_monotone_and_never_exceeds_one():
    adj = C.holm_adjust({"a": 0.5, "b": 0.5, "c": 0.5})
    assert all(0.0 <= p <= 1.0 for p in adj.values())
    assert adj["a"] == adj["b"] == adj["c"] == 1.0      # 3 * 0.5 는 이미 1 을 넘는다 → 클립


def test_summarize_confirm_wires_stage_holm_and_confirm_acts_into_output_and_markdown():
    recs = _pop(substitute=0.50, magnitude=0.50, recompute=0.10)
    s = C.summarize(recs, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200, stage="confirm", confirm_acts=["substitute", "magnitude",
                                                                "recompute"])
    assert s["stage"] == "confirm"
    assert s["confirm_acts"] == ["substitute", "magnitude", "recompute"]
    assert set(s["holm_adjusted_p"]) == {"substitute", "magnitude", "recompute"}
    assert "pass_content_effect" in s and "winners" in s
    md = C.to_markdown(s, mode="prompt")
    assert "Holm" in md
    assert "PASS" in md or "FAIL" in md


# ── 9. META_ACTS 가 조건 이름의 단일 진실 원천 ──────────────────────────────────
def test_adding_an_act_to_the_dict_propagates_to_conds_and_summary():
    name = "zz_probe"
    C.META_ACTS[name] = C.Act(suffix="\n\nZZ.", cue="\n\nLet me zz.", rx=r"\bzz\b")
    try:
        assert name in C.prompt_conds() and name in C.continue_conds()
        q = C.act_prompt(TOK, VAR, PROBLEM, name)
        assert q.replace("\n\nZZ.", "", 1) == render_generation_prompt(TOK, VAR, PROBLEM)
        assert C.act_continuation(TOK, VAR, PROBLEM, SOL, name).endswith("\n\nLet me zz.")
        assert C.complied(name, "we zz it") == 1.0
        s = C.summarize(_pop(**{name: 0.55}), C.prompt_conds(), ref="plain", control="filler",
                        k=8, seed=0, n_boot=200)
        assert name in s["conds"] and name in s["acts_ranked"]
        assert s["acts_ranked"][0] == name          # 가장 큰 (act − filler)
        assert name in C.to_markdown(s, mode="prompt")
        assert s["winners"] == [name]
    finally:
        C.META_ACTS.pop(name)
        C._RX_CACHE.pop(name, None)


def test_summary_condition_list_follows_the_dict_order():
    s = C.summarize(_pop(), C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200)
    assert s["conds"] == list(C.prompt_conds())
    assert s["n_units"] == 40 and s["ref"] == "plain" and s["control"] == "filler"


def test_source_tag_namespaces_two_corpora_with_colliding_group_ids():
    a = C.source_tag("/x/eval/mathL5_q3i2507_opt_b8k/texts.jsonl")
    b = C.source_tag("/x/eval/math500_q3i2507_opt_b8k/texts.jsonl")
    assert a == "mathL5_q3i2507_opt_b8k" and b == "math500_q3i2507_opt_b8k" and a != b


# ── 10. pad_length(F1) — 길이-매치 대조 ────────────────────────────────────────
def test_delta_vs_pad_length_reported_alongside_delta_vs_filler():
    recs = _pop(substitute=0.60, pad_length=0.30)
    s = C.summarize(recs, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200)
    assert "delta_vs_pad_length" in s and "sign_p_vs_pad_length" in s
    assert abs(s["delta_vs_pad_length"]["substitute"]["mean"] - 0.30) < 1e-9
    assert math.isfinite(s["sign_p_vs_pad_length"]["substitute"])
    md = C.to_markdown(s, mode="prompt")
    assert "pad_length" in md


def test_pad_length_columns_absent_when_pad_length_not_among_conds():
    """continue 모드 같은(또는 confirm 에서 --acts 가 안 고른) 경우 — 열 자체가 없어야 한다."""
    conds = tuple(c for c in C.prompt_conds() if c != "pad_length")
    recs = _pop()
    for r in recs:
        del r["cond"]["pad_length"]
    s = C.summarize(recs, conds, ref="plain", control="filler", k=8, seed=0, n_boot=200)
    assert "delta_vs_pad_length" not in s and "sign_p_vs_pad_length" not in s
    assert "token_confound" not in s
    assert "pad_length" not in C.to_markdown(s, mode="prompt")


def test_token_confound_flags_gains_that_dont_survive_against_pad_length():
    # substitute 는 filler 보다 낫지만(+0.25) pad_length 와는 동률(Δ=0 ≤ 0) → 토큰 혐의.
    # magnitude 는 filler 도 pad_length 도 확실히 이긴다(+0.40 / +0.15) → 혐의 없음.
    recs = _pop(substitute=0.60, magnitude=0.75, pad_length=0.60)
    s = C.summarize(recs, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200)
    assert s["token_confound"] == ["substitute"]
    assert "magnitude" not in s["token_confound"]
    assert "pad_length" not in s["token_confound"]     # 자기 자신과는 비교하지 않는다
    md = C.to_markdown(s, mode="prompt")
    assert "[TOKEN-CONFOUND?] substitute" in md
    assert "[TOKEN-CONFOUND?] magnitude" not in md


def test_pad_length_not_added_to_the_pass_rule_itself():
    """★F1: pad_length 는 진단용 둘째 잣대일 뿐 — 통과 규칙은 그대로 filler(`CONTROL`) 기준이다.
    pad_length 값을 무엇으로 바꾸든 통과 판정(승자 목록)은 바뀌지 않는다."""
    base = _pop(substitute=0.60)
    lo = C.summarize(base, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                     n_boot=200)
    hi = [dict(r, cond={**r["cond"], "pad_length": {**r["cond"]["pad_length"], "acc": 0.90}})
          for r in base]
    hi_s = C.summarize(hi, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                       n_boot=200)
    assert lo["winners"] == hi_s["winners"] == ["substitute"]


# ── 11. compliance 는 절대값이 아니라 REF[mode] 대비 차분이다(F2) ───────────────
def test_complied_rate_ref_and_delta_computed_from_recs_and_reported():
    recs = _pop(substitute=0.60)
    for r in recs:                                    # substitute 만 기준 대비 준수율을 높게
        r["cond"]["substitute"]["complied_ref"] = 0.10
        r["cond"]["substitute"]["complied"] = 0.80
    s = C.summarize(recs, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200)
    assert abs(s["complied_rate_ref"]["substitute"] - 0.10) < 1e-9
    assert abs(s["complied_rate_delta"]["substitute"] - 0.70) < 1e-9
    md = C.to_markdown(s, mode="prompt")
    assert "complied_ref" in md and "Δcomplied" in md


def test_complied_rate_delta_is_nan_when_complied_ref_is_missing():
    """구 버전 recs(complied_ref 키가 없는)는 nan 으로 떨어져 차분 조건이 항상 FAIL 이다."""
    recs = _pop(substitute=0.60)
    for r in recs:
        del r["cond"]["substitute"]["complied_ref"]
    s = C.summarize(recs, C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200)
    assert math.isnan(s["complied_rate_delta"]["substitute"])
    assert "substitute" not in s["winners"]            # 차분을 못 재면 통과할 수 없다


# ── 13. F1(0915) — 후보 없어 스킵된 단위가 다른 조건의 통계에서 안 빠진다 ──────────
def _rec_partial(present: dict, *, tag="t", unit="u"):
    """`_rec` 과 달리 딕셔너리에 없는 조건은 그 단위에 **아예 없다**는 뜻(예전엔 이런 단위가
    `all(got.values())` 에 걸려 전체가 통째로 빠졌다 — 이제는 있는 조건만으로 잰다)."""
    return {"unit_id": unit, "tag": tag, "group_id": unit, "gold": "4",
            "cond": {c: {"acc": v, "trunc": 0.0, "tokens": 100.0,
                        "complied": (0.8 if C.META_ACTS[c].rx else float("nan")),
                        "complied_ref": 0.1, "no_new_boxed": 0.0}
                    for c, v in present.items()}}


def test_units_missing_a_non_ref_control_condition_still_count_for_other_conditions():
    # 38 단위는 plain/filler/substitute/magnitude 를 다 갖는다. 2 단위는 verification_first
    # 후보가 없어 그 조건만 빠진다(ref·control 은 있다) — 예전 버그면 이 2 개가 통째로 버려져
    # substitute/magnitude 의 n 도 같이 깎였다.
    full = {"plain": 0.30, "filler": 0.30, "substitute": 0.60, "magnitude": 0.60,
           "verification_first": 0.60}
    partial = {"plain": 0.30, "filler": 0.30, "substitute": 0.60, "magnitude": 0.60}
    recs = [_rec_partial(full, unit=f"f{i}") for i in range(38)]
    recs += [_rec_partial(partial, unit=f"p{i}") for i in range(2)]
    conds = ("plain", "filler", "substitute", "magnitude", "verification_first")
    s = C.summarize(recs, conds, ref="plain", control="filler", k=8, seed=0, n_boot=200)
    # ★핵심 회귀 고정: substitute/magnitude 는 verification_first 스킵과 무관하게 40 단위 전부.
    assert s["n_paired_control"]["substitute"] == 40
    assert s["n_paired_control"]["magnitude"] == 40
    assert s["n_units_cond"]["substitute"] == 40
    # verification_first 는 그 2 개가 실제로 빠져 38 개뿐이다.
    assert s["n_paired_control"]["verification_first"] == 38
    assert s["n_units_cond"]["verification_first"] == 38
    # 최상위 n_units 는 ref∧control 이 있는 단위 수(= 전부, 이 합성 예시에선 40).
    assert s["n_units"] == 40
    md = C.to_markdown(s, mode="prompt")
    assert "| substitute | 40 |" in md
    assert "| verification_first | 38 |" in md


def test_n_paired_ref_and_control_can_differ_from_each_other():
    # ref(plain) 가 없는 단위와 control(filler) 이 없는 단위를 각각 하나씩 둔다.
    recs = [_rec_partial({"plain": 0.3, "filler": 0.3, "substitute": 0.6}, unit=f"u{i}")
           for i in range(10)]
    recs.append(_rec_partial({"filler": 0.3, "substitute": 0.6}, unit="no_ref"))
    conds = ("plain", "filler", "substitute")
    s = C.summarize(recs, conds, ref="plain", control="filler", k=8, seed=0, n_boot=200)
    # no_ref 단위는 ref 가 없어 top n_units(=ref∧control) 에서 빠지지만, control 은 있으므로
    # n_paired_control 에는 들어간다.
    assert s["n_units"] == 10
    assert s["n_paired_control"]["substitute"] == 11
    assert s["n_paired_ref"]["substitute"] == 10


# ── 14. F3(0915b) — continue 모드 준수-기준 조건은 blind 가 아니라 filler_continue ──
def test_compliance_reference_condition_differs_by_mode():
    """blind(= 완결된 4k 새 풀이)를 잣대로 쓰면 짧은 이어쓰기의 complied 와 구조적으로
    안 맞아 MIN_COMPLIED_DELTA 가 항상 음수로 시작한다 — continue 모드는 같은 길이의
    filler_continue 를 잣대로 써야 한다. prompt 모드는 그대로 REF(plain)."""
    assert C.COMPLIANCE_REF["prompt"] == C.REF["prompt"] == "plain"
    assert C.COMPLIANCE_REF["continue"] == C.CONTROL["continue"] == "filler_continue"
    assert C.COMPLIANCE_REF["continue"] != C.REF["continue"]


def test_note_line_present_only_for_continue_mode_markdown():
    s = C.summarize(_pop(), C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                    n_boot=200)
    prompt_md = C.to_markdown(s, mode="prompt")
    assert "[NOTE]" not in prompt_md
    cc = C.continue_conds()
    accs = {c: 0.3 for c in cc}
    accs["filler_continue"] = 0.35
    recs = [_rec(accs, unit=f"u{i}") for i in range(10)]
    sc = C.summarize(recs, cc, ref="blind", control="filler_continue", k=8, seed=0, n_boot=200)
    continue_md = C.to_markdown(sc, mode="continue")
    assert "[NOTE]" in continue_md
    assert "반복" in continue_md


# ── 12. PAL 경고(C3) — 문서화 요구가 실제로 텍스트에 있는지 고정 ────────────────
def test_pal_warning_present_in_docstring_and_report():
    assert "2211.10435" in C.__doc__
    assert "23.2" in C.__doc__ and "72.0" in C.__doc__
    pooled = C.summarize(_pop(), C.prompt_conds(), ref="plain", control="filler", k=8, seed=0,
                         n_boot=50)
    rep = C.render_report(pooled, {}, mode="prompt",
                          meta={"model_path": "m", "variant": "v", "seed": 1, "stage": "screen",
                                "max_tokens": 8192, "sources": {}, "n_generations": 0})
    assert "2211.10435" in rep
