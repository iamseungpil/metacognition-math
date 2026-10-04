"""math_activation_gate 회귀 시험 (CPU, 모델 없음 — MockTok).

1. 네 조건의 프롬프트 조립 — blind 바이트 동일, wait 는 열린 assistant 턴을 cue 로 이어
   쓴 것과 바이트 동일(실 토크나이저로 확인한 continue_final_message 와 동치인 트릭),
   external 은 원 풀이 뒤 새 user 턴, blind_external 은 문제 + note 뿐(원 풀이 없음).
2. 채점 — 생성물만 채점한다(원 롤아웃 접두는 안 본다).
3. wait 의 "새 박스 없음" 규칙.
4. 정답 행 선별(문제당 하나, r_corr==1).
5. 짝지은 Δ 산술.
6. 두 통과 규칙의 진리표(NaN → FAIL 포함).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_activation_gate as A  # noqa: E402
import math_ruler_pivot as P  # noqa: E402
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402

TOK = P.MockTok()
PROBLEM = "What is 2 plus 2?"
SOL = "Step 1: add them.\nStep 2: I get five.\nThus \\boxed{5}   \n"


# ── 1. 프롬프트 조립 ────────────────────────────────────────────────────────────
def test_blind_prompt_is_byte_identical_to_generation_prompt():
    assert (A.blind_prompt(TOK, "math_opt", PROBLEM)
            == render_generation_prompt(TOK, "math_opt", PROBLEM))


def test_wait_prompt_ends_with_cue_and_holds_solution_no_end_of_turn_after():
    q = A.wait_prompt(TOK, "math_opt", PROBLEM, SOL)
    head = render_generation_prompt(TOK, "math_opt", PROBLEM)
    assert q == head + SOL + A.WAIT_CUE
    assert q.endswith(A.WAIT_CUE)
    # MockTok 의 "end-of-turn" 표식은 "<assistant> " 접미 — head 는 정확히 한 번만 그 접미로
    # 끝나야 하는 자리(assistant 턴을 여는 자리)이고, cue 뒤에는 그 표식이 다시 나오지 않는다.
    assert q.count("<assistant> ") == 1
    assert not q[len(head):].rstrip().endswith("<assistant>")


def test_external_prompt_has_rollout_then_cue_as_new_user_turn():
    q = A.external_prompt(TOK, "math_opt", PROBLEM, SOL)
    assert SOL in q and A.EXTERNAL_CUE in q and A._SENTINEL not in q
    # 비평 사용자 턴이 원 풀이 뒤에 온다.
    assert q.index(SOL) < q.index(A.EXTERNAL_CUE)


def test_blind_external_prompt_has_note_but_no_rollout():
    q = A.blind_external_prompt(TOK, "math_opt", PROBLEM)
    assert A.BLIND_EXTERNAL_NOTE.strip() in q
    assert SOL not in q
    assert PROBLEM in q


def test_wait_prompt_matches_hf_continue_final_message_on_real_tokenizer():
    """실 토크나이저가 설치돼 있으면 wait_prompt == apply_chat_template(...,
    continue_final_message=True) — 없으면 스킵(CPU 시험 환경에 HF 캐시가 없을 수 있다)."""
    import pytest  # noqa: PLC0415
    model_dir = "/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507"
    if not Path(model_dir).exists():
        pytest.skip("실 토크나이저 없음 — CPU 시험 환경에는 모델 캐시가 없을 수 있다")
    from transformers import AutoTokenizer  # noqa: PLC0415
    from src.metacot.math_meta_prompt import build_math_prompt  # noqa: E402, PLC0415
    tok = AutoTokenizer.from_pretrained(model_dir)
    a = A.wait_prompt(tok, "math_opt", PROBLEM, SOL)
    msgs = build_math_prompt(PROBLEM, "math_opt") + [
        {"role": "assistant", "content": SOL + A.WAIT_CUE}]
    b = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                continue_final_message=True, enable_thinking=False)
    assert a == b


# ── 2. 채점은 생성물만 ──────────────────────────────────────────────────────────
def test_grading_uses_generated_text_only():
    # 원 오답 접두의 \boxed{5} 는 채점에 안 들어간다 — 생성물의 \boxed{4} 가 맞으면 정답.
    assert A.grade_math("some prefix \\boxed{5} more text \\boxed{4}", "4") == 1


# ── 3. wait 의 "새 박스 없음" 규칙(요약 단계) ───────────────────────────────────
def _wrong_rollout(text="wrong solution \\boxed{7}"):
    return {"roll_id": "g0#0", "group_id": "g0", "problem": PROBLEM, "gold": "4", "text": text}


# ── 4. 정답 행 선별 ──────────────────────────────────────────────────────────────
def test_select_correct_rollouts_one_per_problem_from_mixed_only():
    rolls = [
        {"group_id": "g0", "r_corr": 0, "problem": "p0", "gold": "4", "text": "w1"},
        {"group_id": "g0", "r_corr": 1, "problem": "p0", "gold": "4", "text": "c1"},
        {"group_id": "g0", "r_corr": 1, "problem": "p0", "gold": "4", "text": "c2"},
        {"group_id": "g1", "r_corr": 1, "problem": "p1", "gold": "9", "text": "all-correct"},
        {"group_id": "g2", "r_corr": 1, "problem": "p2", "gold": "1", "text": "trunc",
         "truncated": True},
        {"group_id": "g2", "r_corr": 0, "problem": "p2", "gold": "1", "text": "w"},
    ]
    sel = A.select_correct_rollouts(rolls, per_problem=1)
    # g1 은 그룹 전체가 정답(0 < sum < len 아님)이라 MIXED 가 아니다 → 빠진다.
    # g2 는 유일한 정답 후보가 잘렸다 → 빠진다.
    assert [s["group_id"] for s in sel] == ["g0"]
    assert sel[0]["text"] == "c1"


def test_group_answers_collects_last_boxed_per_group():
    rolls = [{"group_id": "g0", "text": "\\boxed{4}"}, {"group_id": "g0", "text": "\\boxed{5}"},
            {"group_id": "g1", "text": "\\boxed{9}"}]
    ga = A.group_answers(rolls)
    assert ga["g0"] == ["4", "5"] and ga["g1"] == ["9"]


# ── 5. 짝지은 Δ 산술 ────────────────────────────────────────────────────────────
def _wrec(p_blind, p_wait, p_external, p_blind_external, n=40):
    return [{"p_blind": p_blind, "p_wait": p_wait, "p_external": p_external,
             "p_blind_external": p_blind_external,
             **{f"changed_{c}": 0.1 for c in A.CONDS_WRONG},
             **{f"to_plurality_{c}": 0.05 for c in A.CONDS_WRONG},
             **{f"trunc_{c}": 0.0 for c in A.CONDS_WRONG},
             "no_new_boxed_wait": 0.2}
            for _ in range(n)]


def _crec(flip_blind, flip_wait, flip_external, n=40):
    return [{"flip_blind": flip_blind, "flip_wait": flip_wait, "flip_external": flip_external,
             **{f"changed_{c}": 0.05 for c in A.CONDS_CORRECT},
             **{f"trunc_{c}": 0.0 for c in A.CONDS_CORRECT}}
            for _ in range(n)]


def test_paired_deltas_arithmetic():
    w = A.summarize_wrong(_wrec(0.10, 0.15, 0.40, 0.12), k=8, seed=0, n_boot=200)
    assert abs(w["paired_external_minus_blind"]["mean"] - 0.30) < 1e-9
    assert abs(w["paired_wait_minus_blind"]["mean"] - 0.05) < 1e-9
    assert abs(w["paired_external_minus_blind_external"]["mean"] - 0.28) < 1e-9
    assert abs(w["paired_external_minus_wait"]["mean"] - 0.25) < 1e-9
    assert abs(w["blind_spot"] - 0.25) < 1e-9


# ── 6. 통과 규칙 진리표 ─────────────────────────────────────────────────────────
def _wsumm(eb=None, ebe=None, wb=None):
    good = {"lo": 0.1, "hi": 0.3, "mean": 0.2}
    return {"paired_external_minus_blind": eb or good,
            "paired_external_minus_blind_external": ebe or good,
            "paired_wait_minus_blind": wb or good}


def _csumm(flip_ext=0.0, flip_wait=0.0):
    return {"flip_wrong_rate": {"external": {"mean": flip_ext}, "wait": {"mean": flip_wait},
                                "blind": {"mean": 0.0}}}


def test_activation_pass_truth_table():
    good = {"lo": 0.1, "hi": 0.3, "mean": 0.2}
    weak = {"lo": 0.001, "hi": 0.03, "mean": 0.02}          # 유의하지만 +0.05 미달
    crosses_zero = {"lo": -0.1, "hi": 0.1, "mean": 0.02}
    assert A.activation_pass(_wsumm(), _csumm()) is True
    assert A.activation_pass(_wsumm(eb=weak), _csumm()) is False
    assert A.activation_pass(_wsumm(eb=crosses_zero), _csumm()) is False
    assert A.activation_pass(_wsumm(ebe=crosses_zero), _csumm()) is False
    assert A.activation_pass(_wsumm(), _csumm(flip_ext=0.11)) is False
    assert A.activation_pass(_wsumm(), _csumm(flip_ext=0.10)) is True


def test_self_trigger_pass_truth_table():
    assert A.self_trigger_pass(_wsumm(), _csumm()) is True
    crosses_zero = {"lo": -0.1, "hi": 0.1, "mean": 0.0}
    assert A.self_trigger_pass(_wsumm(wb=crosses_zero), _csumm()) is False
    assert A.self_trigger_pass(_wsumm(), _csumm(flip_wait=0.11)) is False


def test_pass_rules_nan_is_fail():
    nan_ci = {"lo": float("nan"), "hi": float("nan"), "mean": float("nan")}
    assert A.activation_pass(_wsumm(eb=nan_ci), _csumm()) is False
    assert A.self_trigger_pass(_wsumm(wb=nan_ci), _csumm()) is False
    assert A.activation_pass({}, {}) is False
    assert A.self_trigger_pass({}, {}) is False


# ── 7. F1 «리셋 자리의 메타 내용» — 새 조건의 프롬프트 바이트 ──────────────────
def _user_tail(q: str) -> str:
    """조립된 프롬프트에서 blind 대비 **덧붙은 꼬리**만 뽑는다(공통 접두를 잘라낸다)."""
    base = A.blind_prompt(TOK, "math_opt", PROBLEM)
    i = 0
    while i < min(len(base), len(q)) and base[i] == q[i]:
        i += 1
    return q[i:]


def test_fact_is_byte_identical_alias_of_blind_external():
    assert A.CONDS_ALIASES["fact"] == "blind_external"
    assert A.resolve_conds("fact") == ["blind_external"]
    assert (A.note_prompt(TOK, "math_opt", PROBLEM, "")
            == A.blind_external_prompt(TOK, "math_opt", PROBLEM))


def test_content_prompts_are_fact_line_plus_one_space_plus_sentence():
    note = A.BLIND_EXTERNAL_NOTE
    for q, extra in [
        (A.effort_prompt(TOK, "math_opt", PROBLEM), A.EFFORT_SENT),
        (A.pad_prompt(TOK, "math_opt", PROBLEM), A.PAD_SENTS),
        (A.switch_prompt(TOK, "math_opt", PROBLEM, "Heron's formula"),
         "The previous attempt used Heron's formula. Use a different approach this time."),
        (A.notx_prompt(TOK, "math_opt", PROBLEM, "5"), "In particular, the answer is not 5."),
    ]:
        # fact 줄 뒤에 공백 하나 + 그 문장. 꼬리는 fact 꼬리에서 그 문장만 늘어난 것이다.
        fact_tail = _user_tail(A.blind_external_prompt(TOK, "math_opt", PROBLEM))
        assert _user_tail(q) == fact_tail.replace(note, note + " " + extra)
        assert (note + " " + extra) in q
        assert SOL not in q            # ★오답 본문은 어느 조건에도 들어가지 않는다
        assert PROBLEM in q


def test_effort_and_pad_sentences_are_exact():
    assert A.EFFORT_SENT == ("Before giving a final answer, re-derive the full solution from "
                             "scratch and only then box it.")
    assert A.PAD_SENTS.count(".") == 3
    assert "boxed" not in A.PAD_SENTS and not any(ch.isdigit() for ch in A.PAD_SENTS)


def test_label_prompt_shows_attempt_and_asks_for_method_only():
    q = A.label_prompt(TOK, "math_opt", PROBLEM, SOL)
    assert SOL in q and A.LABEL_ASK in q and q.index(SOL) < q.index(A.LABEL_ASK)


# ── 8. 라벨 누출 방지 ───────────────────────────────────────────────────────────
def test_leak_guard_rejects_digits_boxed_and_answer():
    assert A.clean_label("Arithmetic with the number 5", "5") == (A.GENERIC_LABEL, "digit")
    assert A.clean_label("The approach is \\boxed{5}", "5") == (A.GENERIC_LABEL, "boxed")
    assert A.clean_label("guessing that it equals five", "five") == (A.GENERIC_LABEL, "answer")
    assert A.clean_label("   ", "5") == (A.GENERIC_LABEL, "empty")


def test_leak_guard_keeps_clean_label_and_trims_to_twelve_words():
    lab, why = A.clean_label('"Direct coordinate geometry with the shoelace formula."', "5")
    assert (lab, why) == ("Direct coordinate geometry with the shoelace formula", "ok")
    long = " ".join(f"w{'x' * i}" for i in range(20))          # 숫자 없는 20 단어
    lab, why = A.clean_label(long, "")
    assert why == "ok" and len(lab.split()) == A.LABEL_MAX_WORDS


def test_leak_guard_uses_first_nonempty_line():
    lab, why = A.clean_label("\n\nSubstitution\nThen I computed 42", "")
    assert (lab, why) == ("Substitution", "ok")


# ── 9. 기증자 회전 ──────────────────────────────────────────────────────────────
def test_donor_rotation_never_returns_own_label():
    labs = ["a", "b", "c", "d"]
    for i in range(len(labs)):
        d = A.rotate_donor(labs, i)
        assert d != labs[i]
    assert A.rotate_donor(labs, 0) == "b" and A.rotate_donor(labs, 3) == "a"   # 결정적 회전


def test_donor_rotation_skips_identical_text_and_falls_back():
    assert A.rotate_donor(["a", "a", "b"], 0) == "b"       # 같은 문자열은 건너뛴다
    assert A.rotate_donor(["a", "a"], 1) == A.GENERIC_LABEL
    assert A.rotate_donor(["a"], 0) == A.GENERIC_LABEL


def test_donor_scope_global_ignores_sources():
    labs = ["a", "b", "c", "d"]
    srcs = ["s0", "s1", "s1", "s1"]
    # global(기본) — sources 를 줘도 옛 동작(전체에서 회전) 그대로다
    assert A.rotate_donor(labs, 0, scope="global", sources=srcs) == "b"


def test_donor_scope_within_source_restricts_candidates():
    labs = ["a", "b", "c", "d"]
    srcs = ["s0", "s1", "s1", "s1"]
    # i=0 은 자기 소스(s0)에 자기 말고 아무도 없다 → 일반 라벨로 되돌린다
    assert A.rotate_donor(labs, 0, scope="within_source", sources=srcs) == A.GENERIC_LABEL
    # i=1 은 s1 안에서 회전 — 자기(b) 를 건너뛰고 s1 의 다음 후보(c)를 쓴다
    assert A.rotate_donor(labs, 1, scope="within_source", sources=srcs) == "c"


def test_donor_scope_within_source_without_sources_falls_back_to_global():
    labs = ["a", "b", "c", "d"]
    assert A.rotate_donor(labs, 0, scope="within_source", sources=None) == "b"


def test_donor_scope_unknown_raises():
    try:
        A.rotate_donor(["a", "b"], 0, scope="bogus")
    except SystemExit:
        return
    raise AssertionError("모르는 donor_scope 는 fail-loud 여야 한다")


# ── 10. Holm 보정 ───────────────────────────────────────────────────────────────
def test_holm_adjust_matches_step_down_definition():
    adj = A.holm_adjust({"a": 0.01, "b": 0.02, "c": 0.04})
    assert abs(adj["a"] - 0.03) < 1e-12        # 3 x .01
    assert abs(adj["b"] - 0.04) < 1e-12        # 2 x .02
    assert abs(adj["c"] - 0.04) < 1e-12        # 1 x .04, 단조화
    assert A.holm_adjust({"a": 0.5, "b": 0.9})["b"] == 1.0    # 1.0 에서 잘린다
    # 단조성: 보정된 p 는 원 순서에서 감소하지 않는다
    adj = A.holm_adjust({"a": 0.001, "b": 0.3, "c": 0.02})
    assert adj["a"] <= adj["c"] <= adj["b"]


# ── 11. 새 조건이 요약·판정에 붙는다 ────────────────────────────────────────────
F1_CONDS = ["blind", "blind_external", "fact_effort", "fact_switch", "fact_switch_donor",
            "fact_notx", "fact_pad"]


def _f1_recs(p_arm=0.70, p_fact=0.60, n=40, trunc=0.0):
    out = []
    for i in range(n):
        r = {}
        for c in F1_CONDS:
            r[f"p_{c}"] = p_fact if c in ("blind_external", "fact_pad", "blind") else p_arm
            r[f"changed_{c}"] = 0.5
            r[f"repro_{c}"] = 0.2
            r[f"tokens_{c}"] = 3000.0
            r[f"to_plurality_{c}"] = 0.1
            r[f"trunc_{c}"] = trunc
        r["p_fact_switch_donor"] = p_fact          # 기증자는 fact 수준(라벨 정박 없음)
        out.append(r)
    return out


def _f1_correct(flip=0.05, n=40):
    conds = ["blind", "fact_effort", "fact_switch", "fact_notx"]
    return [{**{f"flip_{c}": flip for c in conds},
             **{f"changed_{c}": 0.05 for c in conds},
             **{f"trunc_{c}": 0.0 for c in conds}} for _ in range(n)]


def test_summarize_wrong_adds_fact_contrasts_holm_and_diagnostics():
    w = A.summarize_wrong(_f1_recs(), k=8, seed=0, n_boot=200, conds=F1_CONDS)
    assert abs(w["paired_fact_effort_minus_fact"]["mean"] - 0.10) < 1e-9
    assert abs(w["paired_fact_pad_minus_fact"]["mean"] - 0.0) < 1e-9
    assert w["tested_arms"] == ["fact_effort", "fact_switch", "fact_notx"]
    assert set(w["holm_adjusted_p"]) == set(A.CONTENT_ARMS)   # 대조는 보정에서 빠진다
    assert abs(w["diag_switch_minus_donor"]["mean"] - 0.10) < 1e-9
    assert abs(w["diag_minus_pad"]["fact_effort"]["mean"] - 0.10) < 1e-9
    # 옛 대조는 조건이 없으면 안 나온다(그리고 예전 기본 실행에서는 그대로 나온다)
    assert "paired_external_minus_blind" not in w
    assert "paired_external_minus_blind" in A.summarize_wrong(
        _wrec(0.1, 0.15, 0.4, 0.12), k=8, seed=0, n_boot=200)


def test_summarize_wrong_pairs_per_pair_not_all_or_nothing():
    """★0915 수리의 계약: `fact_notx` 가 없는 행(그 롤아웃에 boxed 답이 없어 X 를 못 만든 경우)
    도 **다른 쌍**의 비교에서는 살아 있어야 한다 — 예전 `all(got.values())` 버그는 그 행을
    모든 비교에서 통째로 지웠다."""
    recs = _f1_recs()
    # 마지막 5개 행에서는 fact_notx 만 빠졌다고 가정(boxed 답 없음) — 나머지 조건은 있다.
    for r in recs[-5:]:
        del r["p_fact_notx"]
    w = A.summarize_wrong(recs, k=8, seed=0, n_boot=200, conds=F1_CONDS)
    # fact_effort 쌍은 fact_notx 유무와 무관하다 — 전체 n 이 그대로 쓰인다.
    assert w["n_pairs"]["paired_fact_effort_minus_fact"] == len(recs)
    # fact_notx 쌍은 그 5개가 빠진 n 만 쓴다.
    assert w["n_pairs"]["paired_fact_notx_minus_fact"] == len(recs) - 5
    assert abs(w["paired_fact_effort_minus_fact"]["mean"] - 0.10) < 1e-9


def test_summarize_correct_handles_rows_missing_a_condition():
    recs = _f1_correct()
    del recs[0]["flip_fact_notx"]        # X 후보가 없어 건너뛴 행
    c = A.summarize_correct(recs, k=8, seed=0, n_boot=200,
                            conds=["blind", "fact_effort", "fact_switch", "fact_notx"])
    assert c["n_rows"]["fact_notx"] == len(recs) - 1
    assert c["n_rows"]["fact_effort"] == len(recs)
    assert abs(c["flip_wrong_rate"]["fact_notx"]["mean"] - 0.05) < 1e-9


# ── 12. RESET-CONTENT 통과 규칙 ─────────────────────────────────────────────────
def _rsumm(mean=0.06, lo=0.02, hi=0.10, holm=0.01, trunc=0.05, arm="fact_effort",
           pad=None, donor=None):
    w = {"tested_arms": [arm],
         f"paired_{arm}_minus_fact": {"lo": lo, "hi": hi, "mean": mean},
         "holm_adjusted_p": {arm: holm},
         "trunc_rate": {arm: trunc}}
    if pad is not None:
        w["diag_minus_pad"] = {arm: pad}
    if donor is not None:
        w["diag_switch_minus_donor"] = donor
    return w


def _rcorr(flip=0.05, arm="fact_effort", blind=0.05, key="blind"):
    """★0916: flip 절은 **상대** 규칙이므로 정답-행 요약에 기준선(blind)이 있어야 한다.
    `blind=None` 이면 기준선이 없는 요약(판정 불가)을 만든다."""
    out = {"flip_wrong_rate": {arm: {"mean": flip}}}
    if blind is not None:
        out["flip_wrong_rate"][key] = {"mean": blind}
    return out


def test_reset_content_pass_rule_on_synthetic_summary():
    ok, win, _ = A.reset_content_pass(_rsumm(), _rcorr())
    assert ok and win == ["fact_effort"]
    # 효과 크기 미달(+.03 하한)
    assert A.reset_content_pass(_rsumm(mean=0.02, lo=0.005, hi=0.04), _rcorr())[0] is False
    # CI 가 0 을 품는다
    assert A.reset_content_pass(_rsumm(lo=-0.01, hi=0.12), _rcorr())[0] is False
    # Holm 보정 p 미달
    assert A.reset_content_pass(_rsumm(holm=0.06), _rcorr())[0] is False
    # 거짓 경보 — **상대** 상한(기준선 blind + FLIP_MARGIN)
    assert A.reset_content_pass(_rsumm(), _rcorr(flip=0.33, blind=0.30))[0] is False
    assert A.reset_content_pass(_rsumm(), _rcorr(flip=0.32, blind=0.30))[0] is True
    # ★절대 .10 이면 떨어졌을 조합이 통과한다 — F1(0915)의 오판정이 이것이었다.
    assert 0.32 > A.MAX_FLIP_WRONG
    # 기준선이 blind 가 아니라 fact 로만 있을 때는 fact 로 되돌린다
    assert A.reset_content_pass(_rsumm(), _rcorr(flip=0.32, blind=0.30,
                                                key="fact"))[0] is True
    # 기준선이 아예 없으면 flip 절을 판정할 수 없다 → FAIL
    assert A.reset_content_pass(_rsumm(), _rcorr(flip=0.05, blind=None))[0] is False
    # 절단 상한 초과
    assert A.reset_content_pass(_rsumm(trunc=0.21), _rcorr())[0] is False
    assert A.reset_content_pass(_rsumm(trunc=0.20), _rcorr())[0] is True
    # 값이 없으면 FAIL
    assert A.reset_content_pass({}, {}) == (False, [], {})
    nan_ci = {"lo": float("nan"), "hi": float("nan"), "mean": float("nan")}
    assert A.reset_content_pass(_rsumm(**{"mean": float("nan")}), _rcorr())[0] is False
    assert A.reset_content_pass({"tested_arms": ["fact_effort"],
                                 "paired_fact_effort_minus_fact": nan_ci,
                                 "holm_adjusted_p": {"fact_effort": 0.01},
                                 "trunc_rate": {"fact_effort": 0.0}}, _rcorr())[0] is False


def test_flip_baseline_prefers_blind_then_fact():
    assert A.flip_baseline(_rcorr(blind=0.34)) == (0.34, "blind")
    assert A.flip_baseline(_rcorr(blind=0.34, key="fact")) == (0.34, "fact")
    base, key = A.flip_baseline(_rcorr(blind=None))
    assert key == "" and not (base == base)          # nan


def test_flip_rule_line_is_in_markdown():
    w = _rsumm()
    md = A.to_markdown(w, _rcorr(flip=0.32, blind=0.30))
    assert "[FLIP-RULE]" in md and "0.3000" in md and "0.3200" in md


def test_reset_content_diagnostic_flags():
    good = {"lo": 0.02, "hi": 0.08, "mean": 0.05}
    zero = {"lo": -0.02, "hi": 0.06, "mean": 0.02}
    _, _, flags = A.reset_content_pass(_rsumm(pad=good), _rcorr())
    assert flags["fact_effort"] == []
    _, _, flags = A.reset_content_pass(_rsumm(pad=zero), _rcorr())
    assert flags["fact_effort"] == ["[TOKEN-CONFOUND?]"]
    # blind 보다 더 흔드는 팔은 플래그가 붙는다
    _, _, flags = A.reset_content_pass(_rsumm(pad=good), _rcorr(flip=0.40, blind=0.30))
    assert flags["fact_effort"] == ["[FLIP>BLIND]"]
    _, _, flags = A.reset_content_pass(
        _rsumm(arm="fact_switch", pad=zero, donor=zero),
        _rcorr(arm="fact_switch"))
    assert flags["fact_switch"] == ["[TOKEN-CONFOUND?]", "[LABEL-ANCHOR?]"]
    _, _, flags = A.reset_content_pass(
        _rsumm(arm="fact_switch", pad=good, donor=good), _rcorr(arm="fact_switch"))
    assert flags["fact_switch"] == []


# ── 13. --conds 선택 ────────────────────────────────────────────────────────────
def test_resolve_conds_default_reproduces_old_run():
    assert A.resolve_conds(None) == list(A.CONDS_WRONG)
    assert A.resolve_conds("blind,wait,external,blind_external") == list(A.CONDS_WRONG)


def test_resolve_conds_f1_selection_and_correct_population():
    spec = "blind,fact,fact_effort,fact_switch,fact_switch_donor,fact_notx,fact_pad"
    cs = A.resolve_conds(spec)
    assert cs == ["blind", "blind_external", "fact_effort", "fact_switch", "fact_switch_donor",
                  "fact_notx", "fact_pad"]
    cc = [c for c in cs if c in A.CONDS_CORRECT_ELIGIBLE]
    assert cc == ["blind", "fact_effort", "fact_switch", "fact_notx"]
    # 기본 선택의 정답 행 모집단은 예전 그대로
    assert [c for c in A.resolve_conds(None)
            if c in A.CONDS_CORRECT_ELIGIBLE] == list(A.CONDS_CORRECT)


def test_resolve_conds_rejects_unknown():
    import pytest  # noqa: PLC0415
    with pytest.raises(SystemExit):
        A.resolve_conds("fact_nonsense")


# ── 14. 정답 행 X 고르기 ────────────────────────────────────────────────────────
def test_nonplurality_wrong_answer_picks_most_common_non_plurality():
    assert A.nonplurality_wrong_answer(["7", "7", "9", ""], "7") == "9"
    assert A.nonplurality_wrong_answer(["7", "7"], "7") == ""
    assert A.nonplurality_wrong_answer([], "7") == ""


def test_group_wrong_answers_only_collects_wrong_rows():
    rolls = [{"group_id": "g0", "r_corr": 0, "text": "\\boxed{7}"},
             {"group_id": "g0", "r_corr": 1, "text": "\\boxed{4}"}]
    assert A.group_wrong_answers(rolls) == {"g0": ["7"]}


# ── 15. --resummarize: gens.jsonl → per_row/gate_summary, GPU 없이 ═════════════
def test_resummarize_activation_recomputes_from_gens(tmp_path):
    """★생성 없이 저장된 gen_r_corr 로 wrong/correct 요약과 짝지은 Δ 를 다시 만든다.
    한 오답 롤아웃에서 fact_notx 만 없어도(boxed 답 없음 가정) 다른 쌍은 살아 있어야 한다
    (0915 수리를 재생성 없이도 확인)."""
    gens = []
    conds = ["blind", "wait", "external", "blind_external"]
    for rid in ("w0", "w1", "w2"):
        for c in conds:
            for k_i in range(4):
                # external 이 blind 보다 잘 구제되도록 결정적으로 채점을 만든다
                corr = 1 if (c == "external" and k_i < 3) or (c != "external" and k_i < 1) else 0
                gens.append({"roll_id": rid, "group_id": rid, "population": "wrong",
                            "cond": c, "r_corr": 0, "gen_r_corr": corr, "text": "\\boxed{1}",
                            "truncated": 0})
    for rid in ("c0", "c1"):
        for c in ("blind", "external"):
            for k_i in range(4):
                gens.append({"roll_id": rid, "group_id": rid, "population": "correct",
                            "cond": c, "r_corr": 1, "gen_r_corr": 1, "text": "\\boxed{1}",
                            "truncated": 0})
    gens_path = tmp_path / "gens.jsonl"
    with gens_path.open("w") as fh:
        for g in gens:
            fh.write(__import__("json").dumps(g) + "\n")
    out_dir = tmp_path / "resum"
    rc = A.resummarize_activation(str(gens_path), out_dir=str(out_dir), seed=0, n_boot=100)
    assert rc == 0
    assert (out_dir / "per_row.jsonl").exists()
    assert (out_dir / "gate_summary.json").exists()

    import json as _json
    summ = _json.loads((out_dir / "gate_summary.json").read_text())
    w = summ["wrong"]
    assert abs(w["p_external"]["mean"] - 0.75) < 1e-9
    assert abs(w["p_blind"]["mean"] - 0.25) < 1e-9
    assert w["n_pairs"]["paired_external_minus_blind"] == 3
    assert "changed_rate" not in w or all(
        not (isinstance(v, float) and v == v) for v in (w.get("changed_rate") or {}).values())


# ── 14. S4″/S4′-lite — 고정 재독 팔 + 정책이 쓴 습관 노트 ──────────────────────
def test_s4_prompts_are_fact_line_plus_one_sentence():
    note = A.BLIND_EXTERNAL_NOTE
    fact_tail = _user_tail(A.blind_external_prompt(TOK, "math_opt", PROBLEM))
    for q, extra in [
        (A.reread_prompt(TOK, "math_opt", PROBLEM), A.REREAD_SENT),
        (A.habit_prompt(TOK, "math_opt", PROBLEM, "re-read the constraints first"),
         "Note to self: re-read the constraints first"),
    ]:
        assert _user_tail(q) == fact_tail.replace(note, note + " " + extra)
        assert SOL not in q and PROBLEM in q
    # 노트가 비면 fact 전용과 바이트 동일
    assert (A.habit_prompt(TOK, "math_opt", PROBLEM, "")
            == A.blind_external_prompt(TOK, "math_opt", PROBLEM))


def test_reread_sentence_is_exact():
    assert A.REREAD_SENT == ("Before solving, re-read the problem statement and list every "
                             "given quantity and constraint, then solve.")
    assert not any(ch.isdigit() for ch in A.REREAD_SENT)


def test_habit_ask_prompt_shows_attempt_and_optional_sibling_line():
    q = A.habit_ask_prompt(TOK, "math_opt", PROBLEM, SOL)
    assert SOL.strip() in q and A.HABIT_ASK in q and q.index(SOL.strip()) < q.index(A.HABIT_ASK)
    sib = A.sibling_answer_line("5", ["5", "4", "4", ""])
    q2 = A.habit_ask_prompt(TOK, "math_opt", PROBLEM, SOL, sib)
    assert sib in q2 and q2.index(sib) < q2.index(A.HABIT_ASK)


def test_sibling_line_is_gold_free_and_carries_no_correctness():
    line = A.sibling_answer_line("5", ["5", "4", "4", "4", ""])
    assert line == "Your answer: 5. Final answers across 5 attempts: 4 (3), 5 (1), no answer (1)."
    for banned in ("correct", "incorrect", "wrong", "right", "gold", "answer is"):
        assert banned not in line.lower()
    assert A.sibling_answer_line("5", []) == ""
    # 동치 표기는 한 군집으로 묶인다(예: 0.5 와 1/2 가 동치면 한 줄)
    assert A.sibling_answer_line("5", ["5", "5"]).count("(") == 1


def test_own_is_minority_is_distribution_only():
    assert A.own_is_minority("5", ["5", "4", "4", "4"]) is True
    assert A.own_is_minority("4", ["5", "4", "4", "4"]) is False
    assert A.own_is_minority("", ["4"]) is False


def test_habit_leak_guard_cases():
    # 자기 오답 X 를 부정과 함께 **한 번** → 허용
    n, why = A.clean_habit("rule out 5 and re-read the constraints", "5")
    assert why == "ok_exclusion" and n == "rule out 5 and re-read the constraints"
    # 형제 후보 Y 는 자기 오답이 아니므로 거절
    assert A.clean_habit("the answer is probably 4, try that", "5")[1] == "digit"
    # 숫자 둘 → 거절
    assert A.clean_habit("not 5 but maybe 4", "5")[1] == "two_numbers"
    # boxed / LaTeX 수식 → 거절
    assert A.clean_habit("\\boxed{5} is wrong", "5")[1] == "boxed"
    assert A.clean_habit("use \\frac{a}{b} instead", "5")[1] == "math"
    # 부정 없이 자기 답만 적으면 거절(«답 힌트»가 된다)
    assert A.clean_habit("aim for 5 again", "5")[1] == "digit"
    # 숫자 없는 순수 습관 → ok
    assert A.clean_habit("re-derive from scratch and check each constraint", "5") == (
        "re-derive from scratch and check each constraint", "ok")
    # 거절된 노트는 일반 문구로 되돌아간다(빈 자리 금지)
    assert A.clean_habit("try 4", "5")[0] == A.GENERIC_NOTE


def test_habit_class_on_six_examples():
    cases = [
        ("rule out 5 this time", "exclude"),
        ("re-read the problem's constraints", "reread"),
        ("re-derive from scratch", "rederive"),
        ("rule out 5 and re-derive from scratch", "mixed"),
        ("be careful and steady", "generic"),
        (A.GENERIC_NOTE, "generic"),
    ]
    assert [A.habit_class(n) for n, _ in cases] == [c for _, c in cases]


def test_note_variance_stat_on_synthetic_data():
    # 팔과 기준이 같은 분산이면 ratio ≈ 1 이고 신호 없음
    same = [([1, 1, 0, 0], [1, 1, 0, 0]) for _ in range(30)]
    st = A.note_variance_stat(same, seed=0, n_perm=200)
    assert abs(st["ratio"] - 1.0) < 1e-9 and st["n_rows"] == 30
    assert A.variance_signal_pass(st) is False
    # 팔이 행 안에서 섞이고 기준은 한쪽으로 쏠리면 ratio > 1.3 이고 p 가 작다
    # 팔이 행 안에서 더 흩어지면(1,1,0,0) 기준(1,1,1,0)보다 분산이 크다 → ratio = 4/3
    mixed = [([1, 1, 0, 0], [1, 1, 1, 0]) for _ in range(40)]
    st2 = A.note_variance_stat(mixed, seed=0, n_perm=400)
    assert st2["ratio"] > A.VAR_RATIO_MIN and st2["p"] < A.VAR_P_MAX
    assert A.variance_signal_pass(st2) is True
    # 자료가 없으면 nan·판정 불가
    empty = A.note_variance_stat([], seed=0, n_perm=10)
    assert empty["n_rows"] == 0 and A.variance_signal_pass(empty) is False


def _s4summ(mean=0.06, lo=0.02, hi=0.10, holm=0.01, trunc=0.05, arm="habit_self"):
    return {"tested_arms_s4": [arm],
            f"paired_{arm}_minus_fact": {"lo": lo, "hi": hi, "mean": mean},
            "holm_adjusted_p_s4": {arm: holm},
            "trunc_rate": {arm: trunc},
            f"rescue_{arm}": 0.66}


def test_s4_content_pass_rule_and_flags():
    ok, win, flags = A.s4_content_pass(_s4summ(), {})
    assert ok and win == ["habit_self"] and flags["habit_self"] == []
    assert A.s4_content_pass(_s4summ(mean=0.02, lo=0.005, hi=0.04), {})[0] is False
    assert A.s4_content_pass(_s4summ(holm=0.06), {})[0] is False
    assert A.s4_content_pass(_s4summ(trunc=0.25), {})[0] is False
    # 분산 신호·일반 문구 플래그
    w = {**_s4summ(), "note_variance": {"habit_self": {"ratio": 1.5, "p": 0.01}},
         "habit_hist": {"habit_self": {"generic": 9, "reread": 1}}}
    _, _, flags = A.s4_content_pass(w, {})
    assert flags["habit_self"] == ["[VAR-SIGNAL]", "[MOSTLY-GENERIC]"]


def test_s4_conds_resolve_with_aliases():
    assert A.resolve_conds("blind,fact,notx,fact_reread,habit_self,habit_sib") == [
        "blind", "blind_external", "fact_notx", "fact_reread", "habit_self", "habit_sib"]


def test_s4_summary_holm_is_its_own_family():
    """S4 가족의 Holm 은 F1 3팔 보정을 건드리지 않는다(같은 판에 둘 다 있어도 분리)."""
    recs = []
    for i in range(40):
        r = {"roll_id": str(i), "p_blind_external": 0.5, "p_fact_effort": 0.6,
             "p_fact_reread": 0.62, "p_habit_self": 0.61, "p_habit_sib": 0.6}
        recs.append(r)
    conds = ["blind_external", "fact_effort", "fact_reread", "habit_self", "habit_sib"]
    w = A.summarize_wrong(recs, k=8, seed=0, n_boot=200, conds=conds)
    assert set(w["holm_adjusted_p"]) == {"fact_effort"}
    assert set(w["holm_adjusted_p_s4"]) == {"fact_reread", "habit_self", "habit_sib"}
    assert w["tested_arms_s4"] == ["fact_reread", "habit_self", "habit_sib"]


# ── H2 «합의 상태 × 습관»(0916) ────────────────────────────────────────────────
def _roll(gid, corr, ans, *, trunc=0, prob="What is 2 plus 2?"):
    return {"group_id": gid, "r_corr": corr, "problem": prob, "gold": "4",
            "truncated": trunc, "text": f"work\n\\boxed{{{ans}}}" if ans else "no box"}


def test_agreement_state_classifier_on_fixed_sibling_sets():
    """네 상태 + NOANS — 분모 K 는 롤아웃 수(무응답도 센다)."""
    assert A.agreement_state(["5"] * 8)["state"] == "ALL_SAME"
    # 8개 중 하나가 무응답이고 일곱이 같다 → top=7 < K=8 → DOMINANT
    st = A.agreement_state(["5"] * 7 + [""])
    assert st["state"] == "DOMINANT" and st["top"] == 7 and abs(st["dom_frac"] - 7 / 8) < 1e-9
    assert A.agreement_state(["5"] * 5 + ["6", "7", "8"])["state"] == "DOMINANT"
    assert A.agreement_state(["5"] * 4 + ["6", "6", "7", "8"])["state"] == "SPLIT"
    assert A.agreement_state(["5", "5", "6", "6", "7", "8", "9", "10"])["state"] == "SCATTER"
    assert A.agreement_state([""] * 8)["state"] == "NOANS"
    assert A.agreement_state([])["state"] == "NOANS"
    # 수학적 동치는 한 군집이다(0.5 == 1/2)
    st2 = A.agreement_state(["0.5", "\\frac{1}{2}"] * 4)
    assert st2["state"] == "ALL_SAME" and st2["n_clusters"] == 1
    # 우세 답·자기 답 소속
    st3 = A.agreement_state(["5"] * 6 + ["7", "7"])
    assert st3["dominant_answer"] == "5"
    assert A.own_in_dominant("5", st3) and not A.own_in_dominant("7", st3)
    assert not A.own_in_dominant("", st3)


def test_all_wrong_selection_only_zero_correct_and_one_row_each():
    rolls = ([_roll("gA", 0, "5") for _ in range(8)]            # 전부 오답
             + [_roll("gB", 0, "5") for _ in range(7)] + [_roll("gB", 1, "4")]   # MIXED
             + [_roll("gC", 1, "4") for _ in range(8)])         # 전부 정답
    rows = A.select_all_wrong_rollouts(rolls, per_problem=1)
    assert [r["group_id"] for r in rows] == ["gA"]
    assert A.group_n_correct(rolls) == {"gA": 0, "gB": 1, "gC": 8}
    # 대표는 «절단 안 됐고 boxed 있는» 첫 행 — 앞의 두 행이 절단/무박스면 세 번째가 뽑힌다
    r2 = [_roll("gD", 0, "5", trunc=1), _roll("gD", 0, ""), _roll("gD", 0, "9")] \
        + [_roll("gD", 0, "5") for _ in range(5)]
    assert A.select_all_wrong_rollouts(r2)[0]["roll_id"] == "gD#2"
    # 전부 절단·무박스면 첫 행으로 되돌린다(모집단에서 빠지지 않는다)
    r3 = [_roll("gE", 0, "", trunc=1) for _ in range(8)]
    got = A.select_all_wrong_rollouts(r3)
    assert len(got) == 1 and got[0]["roll_id"] == "gE#0"


def test_group_agreement_uses_all_siblings():
    rolls = [_roll("gA", 0, "5") for _ in range(8)] + [_roll("gB", 0, str(i)) for i in range(8)]
    ag = A.group_agreement(rolls)
    assert ag["gA"]["state"] == "ALL_SAME" and ag["gB"]["state"] == "SCATTER"


def test_n_correct_sib_never_enters_any_prompt():
    """⛔gold 기반 열은 프롬프트에 **절대** 들어가지 않는다 — 프롬프트 조립기는 행의
    n_correct_sib 를 인자로도 받지 않고, 문자열 어디에도 나오지 않는다."""
    import inspect  # noqa: PLC0415
    prompts = [A.blind_prompt(TOK, "math_opt", PROBLEM),
               A.blind_external_prompt(TOK, "math_opt", PROBLEM),
               A.effort_prompt(TOK, "math_opt", PROBLEM),
               A.reread_prompt(TOK, "math_opt", PROBLEM),
               A.pad_prompt(TOK, "math_opt", PROBLEM),
               A.switch_prompt(TOK, "math_opt", PROBLEM, "algebraic expansion"),
               A.notx_prompt(TOK, "math_opt", PROBLEM, "5"),
               A.habit_prompt(TOK, "math_opt", PROBLEM, "Note"),
               A.habit_ask_prompt(TOK, "math_opt", PROBLEM, SOL,
                                  A.sibling_answer_line("5", ["5", "5", "4"])),
               A.wait_prompt(TOK, "math_opt", PROBLEM, SOL),
               A.external_prompt(TOK, "math_opt", PROBLEM, SOL)]
    for q in prompts:
        assert "n_correct_sib" not in q
        assert "correct attempt" not in q and "were correct" not in q
    for fn in (A.note_prompt, A.effort_prompt, A.switch_prompt, A.notx_prompt,
               A.habit_prompt, A.reread_prompt, A.habit_ask_prompt, A.pad_prompt):
        assert not any("correct" in p for p in inspect.signature(fn).parameters)


def _h2recs():
    """오답 행 60개 — allwrong(고합의 30) + mixed(저합의 30)."""
    recs = []
    for i in range(30):
        recs.append({"roll_id": f"aw{i}", "pop_kind": A.POP_ALLWRONG,
                     "agree_state": "ALL_SAME" if i < 20 else "SPLIT",
                     "dom_frac": 1.0 if i < 20 else 0.5, "own_in_dominant": 1,
                     "n_correct_sib": 0,
                     "p_blind_external": 0.0, "p_fact_effort": 0.125,
                     "p_fact_switch": 0.0, "p_fact_notx": 0.25,
                     "trunc_blind_external": 0.1, "trunc_fact_effort": 0.1,
                     "trunc_fact_switch": 0.1, "trunc_fact_notx": 0.1})
    for i in range(30):
        recs.append({"roll_id": f"mx{i}", "pop_kind": A.POP_MIXED,
                     "agree_state": "SCATTER", "dom_frac": 0.25, "own_in_dominant": 0,
                     "n_correct_sib": 3,
                     "p_blind_external": 0.5, "p_fact_effort": 0.6,
                     "p_fact_switch": 0.55, "p_fact_notx": 0.62,
                     "trunc_blind_external": 0.05, "trunc_fact_effort": 0.05,
                     "trunc_fact_switch": 0.05, "trunc_fact_notx": 0.05})
    return recs


H2_CONDS = ["blind_external", "fact_effort", "fact_switch", "fact_notx"]


def test_stratified_summary_keys_and_absolute_rescue():
    strat = A.stratified_summary(_h2recs(), conds=H2_CONDS, seed=1, n_boot=200)
    assert set(strat) == {A.POP_ALLWRONG, A.POP_MIXED, "base", "tested_arms"}
    assert strat["tested_arms"] == list(A.CONTENT_ARMS)
    aw = strat[A.POP_ALLWRONG]
    assert set(aw) == {"ALL", "ALL_SAME", "SPLIT"}
    assert aw["ALL"]["n_rows"] == 30 and aw["ALL_SAME"]["n_rows"] == 20
    # pass@8 = 0 모집단은 절대 구제율이 핵심 지표다
    assert "absolute_rescue" in aw["ALL"]
    assert abs(aw["ALL_SAME"]["absolute_rescue"]["fact_notx"]["mean"] - 0.25) < 1e-9
    assert aw["ALL_SAME"]["flags"] == ["[CONSENSUS-WRONG]"]
    assert aw["SPLIT"]["flags"] == []          # SPLIT 은 고합의가 아니다
    # 짝지은 Δ·부호검정·Holm(내용 3팔)
    assert abs(aw["ALL"]["paired_minus_base"]["fact_notx"]["mean"] - 0.25) < 1e-9
    assert set(aw["ALL"]["holm_adjusted_p"]) == set(A.CONTENT_ARMS)
    assert aw["ALL"]["holm_adjusted_p"]["fact_notx"] < 0.05
    mx = strat[A.POP_MIXED]
    assert set(mx) == {"ALL", "SCATTER"} and "absolute_rescue" not in mx["ALL"]
    assert mx["ALL"]["n_correct_sib_mean"] == 3
    # 마크다운에 두 모집단과 [CONSENSUS-WRONG] 이 나온다
    md = A.state_markdown(strat)
    assert A.POP_ALLWRONG in md and A.POP_MIXED in md and "[CONSENSUS-WRONG]" in md


def test_state_correct_summary_reports_flip_by_state():
    crecs = [{"roll_id": f"c{i}", "pop_kind": A.POP_CORRECT_LOWAGREE,
              "agree_state": "SPLIT" if i < 10 else "SCATTER",
              "flip_fact_switch": 0.25, "flip_fact_notx": 0.5,
              "trunc_fact_switch": 0.0, "trunc_fact_notx": 0.0} for i in range(20)]
    crecs.append({"roll_id": "main", "pop_kind": A.POP_CORRECT, "agree_state": "ALL_SAME",
                  "flip_fact_switch": 1.0})       # 주 모집단 행은 섞이지 않는다
    cs = A.state_correct_summary(crecs, seed=0, n_boot=200)
    assert cs["n_rows"] == 20 and set(cs["by_state"]) == {"ALL", "SPLIT", "SCATTER"}
    assert abs(cs["by_state"]["SPLIT"]["flip_wrong_rate"]["fact_switch"]["mean"] - 0.25) < 1e-9
    assert cs["by_state"]["ALL"]["n_rows"] == 20
    md = A.state_markdown({"base": "blind_external", "tested_arms": []}, cs)
    assert "correct_lowagree" in md


def test_population_choices_cover_the_three_modes():
    assert A.POPULATION_CHOICES["mixed"] == (A.POP_MIXED,)
    assert A.POPULATION_CHOICES["all_wrong"] == (A.POP_ALLWRONG,)
    assert A.POPULATION_CHOICES["both"] == (A.POP_MIXED, A.POP_ALLWRONG)


def test_resummarize_preserves_population_and_states(tmp_path):
    """gens.jsonl 의 pop_kind/agree_state 로 상태 격자가 재생성 없이 되살아난다."""
    gens = []
    for i in range(20):
        for kind, state, pop, p in ((A.POP_ALLWRONG, "ALL_SAME", "wrong", 0),
                                    (A.POP_MIXED, "SCATTER", "wrong", 1)):
            for cond, corr in (("blind_external", p), ("fact_notx", 1)):
                gens.append({"roll_id": f"{kind}{i}", "group_id": f"g{i}",
                             "population": pop, "pop_kind": kind, "agree_state": state,
                             "dom_frac": 1.0, "own_in_dominant": 1, "n_correct_sib": 0,
                             "cond": cond, "r_corr": 0, "gen_r_corr": corr,
                             "text": "x", "truncated": 0})
        gens.append({"roll_id": f"c{i}", "group_id": f"h{i}", "population": "correct",
                     "pop_kind": A.POP_CORRECT_LOWAGREE, "agree_state": "SPLIT",
                     "dom_frac": 0.5, "own_in_dominant": 1, "n_correct_sib": 4,
                     "cond": "fact_notx", "r_corr": 1, "gen_r_corr": 0,
                     "text": "x", "truncated": 0})
    gp = tmp_path / "gens.jsonl"
    gp.write_text("\n".join(__import__("json").dumps(g) for g in gens))
    out = tmp_path / "re"
    assert A.resummarize_activation(str(gp), out_dir=str(out), seed=3, n_boot=200) == 0
    summ = __import__("json").loads((out / "gate_summary.json").read_text())
    strat = summ["wrong"]["agree_strata"]
    assert set(strat[A.POP_ALLWRONG]) == {"ALL", "ALL_SAME"}
    assert set(strat[A.POP_MIXED]) == {"ALL", "SCATTER"}
    assert abs(strat[A.POP_ALLWRONG]["ALL_SAME"]["absolute_rescue"]["fact_notx"]["mean"]
               - 1.0) < 1e-9
    assert abs(strat[A.POP_ALLWRONG]["ALL_SAME"]["rescue"]["blind_external"]["mean"]) < 1e-9
    lw = summ["correct"]["lowagree_states"]
    assert lw["n_rows"] == 20 and "SPLIT" in lw["by_state"]
    # per_row 에 상태가 남는다
    rows = [__import__("json").loads(l) for l in (out / "per_row.jsonl").read_text().splitlines()]
    assert {r["pop_kind"] for r in rows} == {A.POP_ALLWRONG, A.POP_MIXED,
                                             A.POP_CORRECT_LOWAGREE}
    assert all(r["agree_state"] for r in rows)


def test_resolve_correct_states_default_is_all_and_restriction_normalizes():
    assert A.resolve_correct_states("") == list(A.AGREE_STATES)      # 기본 = 필터 없음
    assert A.resolve_correct_states(None) == list(A.AGREE_STATES)
    assert A.resolve_correct_states("scatter, SPLIT") == ["SPLIT", "SCATTER"]
    assert A.resolve_correct_states("SPLIT,SPLIT") == ["SPLIT"]
    import pytest  # noqa: PLC0415
    with pytest.raises(SystemExit):
        A.resolve_correct_states("SPLIT,MOSTLY")


# ── 16. 재채점 뒤 --resummarize: 1차 시도가 정답이 된 오답 단위를 뺀다 ──────────
def _resum_gens(with_fixed_unit: bool):
    """오답 단위 3개(w_fix 는 재채점으로 1차 시도가 정답이 된 단위) + 정답 단위 1개."""
    import json as _json
    rows = []
    units = [("w_fix", 1)] if with_fixed_unit else []
    units += [("w0", 0), ("w1", 0)]
    for rid, a1 in units:
        for c in ("blind", "external"):
            for k_i in range(4):
                corr = 1 if (c == "external" and k_i < 3) or k_i < 1 else 0
                rows.append({"roll_id": rid, "group_id": rid, "population": "wrong", "cond": c,
                             "r_corr": a1, "gen_r_corr": corr, "text": "\\boxed{1}",
                             "truncated": 0})
    for c in ("blind", "external"):
        for k_i in range(4):
            rows.append({"roll_id": "c0", "group_id": "c0", "population": "correct", "cond": c,
                         "r_corr": 1, "gen_r_corr": 1, "text": "\\boxed{1}", "truncated": 0})
    return "".join(_json.dumps(r) + "\n" for r in rows)


def test_resummarize_drops_units_whose_attempt1_is_now_correct(tmp_path):
    import json as _json
    a = tmp_path / "with.jsonl"
    a.write_text(_resum_gens(True))
    b = tmp_path / "without.jsonl"
    b.write_text(_resum_gens(False))
    oa, ob = tmp_path / "oa", tmp_path / "ob"
    assert A.resummarize_activation(str(a), out_dir=str(oa), seed=0, n_boot=100) == 0
    assert A.resummarize_activation(str(b), out_dir=str(ob), seed=0, n_boot=100) == 0
    sa = _json.loads((oa / "gate_summary.json").read_text())
    sb = _json.loads((ob / "gate_summary.json").read_text())
    assert sa["meta"]["n_units_dropped_a1_correct"] == 1
    assert sb["meta"]["n_units_dropped_a1_correct"] == 0
    assert sa["meta"]["n_wrong_candidates"] == sb["meta"]["n_wrong_candidates"] == 2
    # 영향 없는 단위의 숫자는 한 자리도 달라지지 않는다
    assert sa["wrong"] == sb["wrong"] and sa["correct"] == sb["correct"]
    assert len((oa / "per_row.jsonl").read_text().splitlines()) == 3


def test_resummarize_unchanged_when_no_regrade_change(tmp_path):
    """★재채점으로 바뀐 것이 없으면(오답 단위 r_corr 전부 0) 예전 숫자를 그대로 재현한다."""
    import json as _json
    p = tmp_path / "gens.jsonl"
    p.write_text(_resum_gens(False))
    o1, o2 = tmp_path / "o1", tmp_path / "o2"
    assert A.resummarize_activation(str(p), out_dir=str(o1), seed=5, n_boot=100) == 0
    assert A.resummarize_activation(str(p), out_dir=str(o2), seed=5, n_boot=100) == 0
    s1 = _json.loads((o1 / "gate_summary.json").read_text())
    s2 = _json.loads((o2 / "gate_summary.json").read_text())
    assert s1 == s2 and s1["meta"]["n_units_dropped_a1_correct"] == 0
