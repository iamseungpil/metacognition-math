"""S3 «2-시도 GRPO + 반성문 크레딧» 순수 함수 층 테스트 (docs/DESIGN_S3_trial2_0915.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import trial2 as t2  # noqa: E402
from src.training.math_meta import MATH_ARM_SPECS  # noqa: E402


# ── 팔 명세: 기존 팔이 한 글자도 안 바뀌었는가 ────────────────────────────────
def test_existing_arm_specs_unchanged():
    """기존 17개 팔의 variant/meta_term/require_meta 는 S3 추가로 바뀌지 않는다."""
    expect = {
        "M_G0": ("math_plain", None, False),
        "M_G1": ("math_opt", None, False),
        "M_JUDGE": ("math_opt", "judge", True),
        "M_PROBE": ("math_opt", "probe", True),
        "M_RAND": ("math_opt", "judge_shuffled", True),
        "M_RETRY": ("math_retry", "retry", True),
        "M_RETRY_RAND": ("math_retry", "retry_shuffled", True),
        "M_RETRY_SL": ("math_retry", "retry_sl", True),
        "M_AGREE": ("math_agree", "agree", True),
        "M_AGREE_RAND": ("math_agree", "agree_shuffled", True),
        "M_CRIT": ("math_crit", "crit", True),
        "M_DIS": ("math_dis", "dis", True),
        "M_DIS_RAND": ("math_dis", "dis_shuffled", True),
        "M_DIS0": ("math_dis", "dis_zero", False),
        "M_DIFF": ("math_diff", "diff", True),
        "M_DIFF_RAND": ("math_diff", "diff_shuffled", True),
        "M_DIFF0": ("math_diff", "diff_zero", False),
    }
    for arm, (v, term, req) in expect.items():
        s = MATH_ARM_SPECS[arm]
        assert (s["variant"], s["meta_term"], s["require_meta"]) == (v, term, req), arm
    # ★새 팔은 여기 목록에만 더한다 — 위 17개의 (variant, meta_term, require_meta) 는 불변이다.
    #   0918 수정 6: 자발적 답 수정(revision) 세 팔. 결과-only 대조는 새 팔이 아니라 M_G1 이다.
    assert set(MATH_ARM_SPECS) - set(expect) == {
        "M_TRIAL2_CREDIT", "M_TRIAL2_OUTCOME", "M_TRIAL2_SCORE",
        "M_REV_CF", "M_REV_PMI_GOLD", "M_REV_PMI_COMBO", "M_REV_PMI_CF", "M_REV_PMI_CONF"}


def test_trial2_arm_specs():
    assert MATH_ARM_SPECS["M_TRIAL2_CREDIT"]["meta_term"] == t2.CREDIT_TERM
    assert MATH_ARM_SPECS["M_TRIAL2_OUTCOME"]["meta_term"] == t2.OUTCOME_TERM
    # 두 팔은 프롬프트가 같아야 한다(대조군 프롬프트 불일치 금지)
    assert (MATH_ARM_SPECS["M_TRIAL2_CREDIT"]["variant"]
            == MATH_ARM_SPECS["M_TRIAL2_OUTCOME"]["variant"] == "math_opt")


# ── 누출 가드 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,reason", [
    ("dropped a sign when expanding", "ok"),
    ("the answer is \\boxed{42}", "boxed"),
    ("", "empty"),
    ("   \n  ", "empty"),
    ("I computed 42 instead", "digit"),
])
def test_clean_note_reasons(raw, reason):
    note, why = t2.clean_note(raw)
    assert why == reason
    assert (note == t2.GENERIC_NOTE) == (reason != "ok")


def test_clean_note_strips_attempt1_answer():
    """★핵심 가드: 반성문이 시도-1 의 최종 답을 담으면 **버린다**(그러지 않으면 이 팔은
    몰래 답을 알려 주는 팔이 된다 — 다만 정답이 아니라 오답이므로 방향까지 나쁘다)."""
    note, why = t2.clean_note("the value pi over three was wrong", "pi over three")
    assert (note, why) == (t2.GENERIC_NOTE, "answer")


def test_clean_note_first_line_and_word_cap():
    note, why = t2.clean_note("misread the constraint.\nAlso here is a long essay ...")
    assert why == "ok" and note == "misread the constraint"
    long = " ".join(["word"] * 50)
    assert len(t2.clean_note(long)[0].split()) == t2.NOTE_MAX_WORDS


def test_clean_short_text_is_the_shared_body():
    """`math_activation_gate.clean_label` 이 이 함수를 쓴다 — 두 벌이 되면 한쪽만 고쳐진다."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))
    import math_activation_gate as ag
    assert ag.BLIND_EXTERNAL_NOTE is t2.BLIND_EXTERNAL_NOTE
    assert ag.clean_label("used 3 substitutions") == (ag.GENERIC_LABEL, "digit")
    assert t2.clean_short_text("used 3 substitutions", max_words=12,
                               fallback=ag.GENERIC_LABEL) == (ag.GENERIC_LABEL, "digit")


# ── 프롬프트 ─────────────────────────────────────────────────────────────────
class _MockTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content'].strip()}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def test_attempt2_prompt_byte_compatible_with_fact_only():
    """note="" 인 시도-2 프롬프트는 G8 의 `blind_external`(= `.617` 참조)와 **바이트 동일**."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))
    import math_activation_gate as ag
    tok = _MockTok()
    assert (t2.attempt2_prompt(tok, "math_opt", "P?", "")
            == ag.blind_external_prompt(tok, "math_opt", "P?"))


def test_attempt2_prompt_never_contains_attempt1_body():
    """★G8: 오답 본문을 문맥에 두면 −10.4pp. 시도-2 프롬프트에 본문이 들어가면 안 된다."""
    body = "Step 1: I wrongly assumed the series converges."
    p = t2.attempt2_prompt(_MockTok(), "math_opt", "P?", "assumed convergence wrongly")
    assert body not in p
    assert t2.BLIND_EXTERNAL_NOTE.strip() in p
    assert "assumed convergence wrongly" in p


def test_note_ask_prompt_contains_body_and_ask():
    """반성문을 **얻는** 프롬프트만이 본문을 보는 유일한 자리다."""
    p = t2.note_ask_prompt(_MockTok(), "math_opt", "P?", "my wrong work")
    assert "my wrong work" in p and t2.NOTE_ASK in p


# ── 손잡이 ───────────────────────────────────────────────────────────────────
def test_gamma_default_and_bounds(monkeypatch):
    monkeypatch.delenv("GAMMA_TRAJ", raising=False)
    assert t2.gamma_traj() == 0.6
    monkeypatch.setenv("GAMMA_TRAJ", "1.0")
    with pytest.raises(ValueError):
        t2.gamma_traj()


def test_knob_defaults(monkeypatch):
    for k in ("NOTE_MAX_TOKENS", "NOTE_MODE", "RETRY_ONLY_WRONG", "ATTEMPT1_KL_COEF"):
        monkeypatch.delenv(k, raising=False)
    assert t2.note_max_tokens() == 64
    assert t2.note_mode() == "self"
    assert t2.retry_only_wrong() is True
    assert t2.attempt1_kl_coef(0.001) == 0.001
    monkeypatch.setenv("ATTEMPT1_KL_COEF", "0.02")
    assert t2.attempt1_kl_coef(0.001) == 0.02
    monkeypatch.setenv("NOTE_MODE", "bogus")
    with pytest.raises(ValueError):
        t2.note_mode()


def test_assert_no_sandbag_rejects_gamma_ge_1():
    t2.assert_no_sandbag(0.99)
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            t2.assert_no_sandbag(bad)


def test_attempt1_reward_never_favours_failing():
    """사행 불가 부등식: R1=1 최솟값(1.0) > R1=0 최댓값(γ)."""
    g = 0.6
    best_fail = t2.row_reward(t2.CREDIT_TERM, "a1", 1, 0.0, 1.0, g)
    worst_pass = t2.row_reward(t2.CREDIT_TERM, "a1", 1, 1.0, 0.0, g)
    assert best_fail == pytest.approx(g)
    assert worst_pass == pytest.approx(1.0)
    assert best_fail < worst_pass


# ── 슬롯 기하 ────────────────────────────────────────────────────────────────
def test_slot_layout_and_stage_of():
    lay = t2.slot_layout(2, 4)
    assert lay["a1"] == [0, 1, 2, 3, 12, 13, 14, 15]
    assert lay["note"] == [4, 5, 6, 7, 16, 17, 18, 19]
    assert lay["a2"] == [8, 9, 10, 11, 20, 21, 22, 23]
    assert t2.rollout_n(4) == 12
    for st, rows in lay.items():
        for r in rows:
            assert t2.stage_of(r, 4) == st


def test_sibling_rows_links_the_trial():
    sib = t2.sibling_rows(17, 4)          # 문제 1, 반성문 슬롯, 샘플 1
    assert sib == {"a1": 13, "note": 17, "a2": 21}
    assert t2.trial_index(17, 4) == 1
    assert t2.trial_id(21, 4, "u1") == "u1#t1"


# ── 보상·중심화 ──────────────────────────────────────────────────────────────
def _fab(k=4, p=2, r1_by_problem=((1, 0, 0, 0), (0, 0, 1, 1))):
    """작은 배치를 위조한다: P 문제 × K 샘플, 섞인 R1. R2 는 오답 trial 에만."""
    uids, stages, active, tkeys = [], [], [], []
    for pi in range(p):
        for st in t2.STAGES:
            for j in range(k):
                uids.append(f"u{pi}")
                stages.append(st)
                wrong = r1_by_problem[pi][j] == 0
                active.append(1 if st == "a1" else int(wrong))
                tkeys.append(f"u{pi}#t{j}")
    return uids, stages, active, tkeys


def test_recredit_credit_vs_outcome_row_by_row():
    """★CPU 드라이런: 2문제 × K=4, 섞인 R1 — 두 팔의 어드밴티지를 행별로 확인한다."""
    k, p = 4, 2
    r1 = {0: (1, 0, 0, 0), 1: (0, 0, 1, 1)}
    r2 = {0: (0, 1, 0, 0), 1: (1, 0, 0, 0)}      # 오답 trial 의 시도-2 결과
    uids, stages, active, tkeys = _fab(k, p, (r1[0], r1[1]))
    own = []
    for pi in range(p):
        for st in t2.STAGES:
            for j in range(k):
                own.append(float(r1[pi][j]) if st == "a1" else
                           (0.0 if st == "note" else float(r2[pi][j])))
    g = 0.6
    for term in (t2.CREDIT_TERM, t2.OUTCOME_TERM):
        rew, keys, tel = t2.recredit(term, own, stages, active, uids, tkeys, g)
        adv = []
        sums, cnts = {}, {}
        for kk, v in zip(keys, rew):
            sums[kk] = sums.get(kk, 0.0) + v
            cnts[kk] = cnts.get(kk, 0) + 1
        adv = [rew[i] - sums[keys[i]] / cnts[keys[i]] for i in range(len(rew))]

        for i, st in enumerate(stages):
            pi = 0 if uids[i] == "u0" else 1
            j = int(tkeys[i][-1])
            if st == "a1":
                want_r = r1[pi][j] + (g * r2[pi][j] if term == t2.CREDIT_TERM else 0.0)
                mean = sum(r1[pi][jj] + (g * r2[pi][jj] if term == t2.CREDIT_TERM else 0.0)
                           for jj in range(k)) / k
                assert adv[i] == pytest.approx(want_r - mean), (term, i)
            elif st == "note":
                if not active[i]:
                    assert adv[i] == 0.0
                elif term == t2.OUTCOME_TERM:
                    # ★OUTCOME 팔: 반성문 스팬 어드밴티지는 **정확히 0**
                    assert adv[i] == 0.0
                else:
                    live = [jj for jj in range(k) if r1[pi][jj] == 0]
                    mean = sum(r2[pi][jj] for jj in live) / len(live)
                    # ★CREDIT 팔: γ·(R2 − 문제 안 평균 R2)
                    assert adv[i] == pytest.approx(g * (r2[pi][j] - mean)), (i,)
            else:  # a2 — 두 팔 모두 R2 중심화
                if not active[i]:
                    assert adv[i] == 0.0
                else:
                    live = [jj for jj in range(k) if r1[pi][jj] == 0]
                    mean = sum(r2[pi][jj] for jj in live) / len(live)
                    assert adv[i] == pytest.approx(r2[pi][j] - mean), (term, i)
        assert tel["n_trials"] == k * p
        assert tel["n_retried"] == sum(1 for pi in (0, 1) for jj in range(k) if r1[pi][jj] == 0)
        assert tel["r1_mean"] == pytest.approx(3 / 8)
        # 2-시도 정확도 = R1 ∨ R2 = u0: {0:1,1:1} · u1: {0:1,2:1,3:1} = 5/8
        assert tel["two_trial_acc"] == pytest.approx(5 / 8)


def test_correct_attempt1_rows_have_no_note_or_attempt2():
    """시도-1 이 맞은 trial 은 반성문/시도-2 행이 비활성이고 보상이 정확히 0 이다."""
    uids, stages, active, tkeys = _fab(2, 1, ((1, 0),))
    own = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]     # a1: 1/0 · note: 0/0 · a2: 0/1
    rew, keys, _ = t2.recredit(t2.CREDIT_TERM, own, stages, active, uids, tkeys, 0.6)
    for i, st in enumerate(stages):
        if st != "a1" and tkeys[i].endswith("t0"):
            assert active[i] == 0 and rew[i] == 0.0
            assert keys[i].startswith("u0#dead")     # 싱글턴 → 중심화 뒤 0


def test_dead_rows_do_not_pollute_group_means():
    """비활성 행은 활성 행의 그룹 평균에 들어가지 않는다(싱글턴 그룹 키)."""
    uids = ["u0"] * 4
    stages = ["a2"] * 4
    active = [1, 1, 0, 0]
    keys = t2.row_group_keys(uids, stages, active)
    assert keys[:2] == ["u0#a2", "u0#a2"]
    assert len(set(keys[2:])) == 2 and all(k.startswith("u0#dead") for k in keys[2:])


def test_trial2_advantages_matches_manual_centering():
    adv = t2.trial2_advantages(t2.CREDIT_TERM, ["u"] * 4, ["a1"] * 4, [1] * 4,
                               [1.0, 0.0, 0.0, 1.0], [0.0] * 4, 0.6)
    assert adv == pytest.approx([0.5, -0.5, -0.5, 0.5])


def test_row_reward_rejects_unknown_term_and_stage():
    with pytest.raises(ValueError):
        t2.row_reward("nope", "a1", 1, 1.0, 0.0, 0.6)
    with pytest.raises(ValueError):
        t2.row_reward(t2.CREDIT_TERM, "nope", 1, 1.0, 0.0, 0.6)


def test_recredit_length_mismatch_dies():
    with pytest.raises(ValueError):
        t2.recredit(t2.CREDIT_TERM, [1.0], ["a1", "a1"], [1, 1], ["u", "u"],
                    ["u#t0", "u#t1"], 0.6)


def test_leak_stats():
    st = t2.leak_stats(["ok", "ok", "digit", "boxed"])
    assert st["n"] == 4 and st["ok"] == 2 and st["digit"] == 1
    assert st["generic_rate"] == pytest.approx(0.5)


def test_rollout_n_must_be_three_k():
    """슬롯 규약: run_math_arm.sh 가 M_TRIAL2_* 에 넣는 n 은 3K 여야 한다."""
    assert t2.rollout_n(8) == 24
    with pytest.raises(ValueError):
        t2.rollout_n(0)


# ── NOTE_MODE 확장(0916): switch · notx · switch_notx ────────────────────────
def _ag():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))
    import math_activation_gate as ag
    return ag


def test_note_mode_accepts_five_modes(monkeypatch):
    for m in ("self", "none", "switch", "notx", "switch_notx"):
        monkeypatch.setenv("NOTE_MODE", m.upper())     # 대소문자 정규화
        assert t2.note_mode() == m
    monkeypatch.setenv("NOTE_MODE", "switchnotx")
    with pytest.raises(ValueError):
        t2.note_mode()


def test_note_stage_only_for_self_and_switch_family():
    assert [t2.note_stage_on(m) for m in t2.NOTE_MODES] == [True, False, True, False, True]


def test_switch_prompt_is_byte_identical_to_f1_fact_switch():
    """S3 의 NOTE_MODE=switch 시도-2 프롬프트 = F1 `fact_switch` 프롬프트(같은 라벨)."""
    ag, tok, label = _ag(), _MockTok(), "a substitution argument"
    assert (t2.attempt2_prompt(tok, "math_opt", "P?", mode="switch", label=label)
            == ag.switch_prompt(tok, "math_opt", "P?", label))


def test_notx_prompt_is_byte_identical_to_f1_fact_notx():
    ag, tok = _ag(), _MockTok()
    assert (t2.attempt2_prompt(tok, "math_opt", "P?", mode="notx", notx="41")
            == ag.notx_prompt(tok, "math_opt", "P?", "41"))


def test_switch_notx_extra_is_the_two_sentences_in_order():
    """합성 팔의 extra 문자열을 **바이트로** 못 박는다."""
    assert t2.attempt2_extra("switch_notx", label="Vieta's formulas", notx="41") == (
        "The previous attempt used Vieta's formulas. Use a different approach this time. "
        "In particular, the answer is not 41.")
    tok = _MockTok()
    assert (t2.attempt2_prompt(tok, "math_opt", "P?", mode="switch_notx",
                               label="Vieta's formulas", notx="41")
            == t2.fact_prompt(tok, "math_opt", "P?",
                              t2.attempt2_extra("switch_notx", label="Vieta's formulas",
                                                notx="41")))


def test_notx_falls_back_to_fact_only_when_no_boxed():
    """a1 에 `\\boxed` 가 없으면 X 가 없다 → notx 절을 빼고 fact 전용으로 되돌린다."""
    tok = _MockTok()
    assert t2.attempt2_extra("notx", notx="") == ""
    assert (t2.attempt2_prompt(tok, "math_opt", "P?", mode="notx", notx="")
            == t2.fact_prompt(tok, "math_opt", "P?"))
    # switch_notx 는 switch 만 남는다
    assert (t2.attempt2_prompt(tok, "math_opt", "P?", mode="switch_notx", label="L", notx="")
            == t2.attempt2_prompt(tok, "math_opt", "P?", mode="switch", label="L"))


def test_label_leak_guard_and_generic_fallback():
    """라벨에 숫자·boxed·오답 문자열이 들어가면 버리고 일반 라벨로 되돌린다."""
    assert t2.clean_label("used 3 substitutions") == (t2.GENERIC_LABEL, "digit")
    assert t2.clean_label("\\boxed{41}") == (t2.GENERIC_LABEL, "boxed")
    assert t2.clean_label("the value pi over three", "pi over three") == (
        t2.GENERIC_LABEL, "answer")
    assert t2.clean_label("Vieta's formulas on the cubic")[1] == "ok"
    assert len(t2.clean_label(" ".join(["word"] * 40))[0].split()) == t2.LABEL_MAX_WORDS
    # 라벨이 비어도 프롬프트에는 일반 라벨이 들어간다(빈 자리 금지)
    assert t2.GENERIC_LABEL in t2.attempt2_extra("switch", label="")


def test_label_guard_is_the_same_function_as_the_gate_uses():
    ag = _ag()
    assert ag.clean_label("used 3 substitutions") == t2.clean_label("used 3 substitutions")
    assert (ag.LABEL_ASK, ag.GENERIC_LABEL, ag.SWITCH_TMPL, ag.NOTX_TMPL) == (
        t2.LABEL_ASK, t2.GENERIC_LABEL, t2.SWITCH_TMPL, t2.NOTX_TMPL)
    # ★F1 의 바이트를 여기에 못 박는다(trial2 로 옮겼어도 한 글자도 달라지면 안 된다)
    assert t2.LABEL_ASK == ("In at most 12 words, name the mathematical approach used in "
                            "this attempt. Do not state any number or final answer.")
    assert t2.GENERIC_LABEL == "the previous approach" and t2.LABEL_MAX_WORDS == 12
    assert t2.SWITCH_TMPL == ("The previous attempt used {label}. Use a different approach "
                              "this time.")
    assert t2.NOTX_TMPL == "In particular, the answer is not {answer}."


def test_note_stage_prompt_uses_label_ask_for_switch_family():
    tok = _MockTok()
    p = t2.note_stage_prompt(tok, "math_opt", "P?", "my wrong work", "switch")
    assert t2.LABEL_ASK in p and t2.NOTE_ASK not in p and "my wrong work" in p
    assert t2.note_stage_prompt(tok, "math_opt", "P?", "my wrong work", "switch_notx") == p
    assert t2.NOTE_ASK in t2.note_stage_prompt(tok, "math_opt", "P?", "w", "self")
    # 사전 패스가 없는 모드 → None (죽은 노트 자리)
    assert t2.note_stage_prompt(tok, "math_opt", "P?", "w", "notx") is None
    assert t2.note_stage_prompt(tok, "math_opt", "P?", "w", "none") is None


def test_clean_stage_text_picks_the_right_guard():
    assert t2.clean_stage_text("self", "misread the constraint")[1] == "ok"
    assert t2.clean_stage_text("switch", "used 3 substitutions") == (t2.GENERIC_LABEL, "digit")
    assert t2.clean_stage_text("notx", "anything") == ("", "no_note_stage")
    assert t2.clean_stage_text("none", "anything") == ("", "no_note_stage")


# ── 습관 노트 가드(allow_exclusion) — S4-lite 가 쓰는 규칙 ────────────────────
def test_clean_note_allow_exclusion_permits_one_self_exclusion():
    assert t2.clean_note("rule out 5 and re-read the constraints", "5",
                         allow_exclusion=True) == (
        "rule out 5 and re-read the constraints", "ok_exclusion")
    # 기본 경로(allow_exclusion=False)는 숫자를 그대로 거절한다 — 옛 계약 불변
    assert t2.clean_note("rule out 5 and re-read", "5")[1] == "digit"


def test_clean_habit_note_rejects_other_numbers_math_and_boxed():
    assert t2.clean_habit_note("try 4 instead", "5")[1] == "digit"
    assert t2.clean_habit_note("not 5 but maybe 4", "5")[1] == "two_numbers"
    assert t2.clean_habit_note("\\boxed{5}", "5")[1] == "boxed"
    assert t2.clean_habit_note("use \\sqrt{2}", "5")[1] == "math"
    assert t2.clean_habit_note("", "5")[1] == "empty"
    assert t2.clean_habit_note("re-read every constraint", "5") == (
        "re-read every constraint", "ok")
    assert len(t2.clean_habit_note(" ".join(["habit"] * 50), "")[0].split()) \
        == t2.NOTE_MAX_WORDS


# ── A2_RESP_LEN(0916): 시도별 생성 상한 ──────────────────────────────────────
def test_a1_a2_resp_len_default_to_board_width(monkeypatch):
    """손잡이 미설정이면 두 시도 모두 판 폭 그대로 — 기존 런과 바이트 동일."""
    for k in ("A1_RESP_LEN", "A2_RESP_LEN", "RESP_LEN"):
        monkeypatch.delenv(k, raising=False)
    assert t2.a1_max_tokens(4096) == 4096
    assert t2.a2_resp_len(4096) == 4096


def test_a2_resp_len_widens_only_attempt2(monkeypatch):
    """판 폭 6144 · A2=6144 · RESP_LEN(=A1)=4096 → 시도 1 만 4096 에서 멈춘다."""
    monkeypatch.delenv("A1_RESP_LEN", raising=False)
    monkeypatch.setenv("RESP_LEN", "4096")
    monkeypatch.setenv("A2_RESP_LEN", "6144")
    assert t2.a1_max_tokens(6144) == 4096
    assert t2.a2_resp_len(6144) == 6144


def test_a2_resp_len_over_board_width_dies(monkeypatch):
    """판 폭보다 큰 A2 는 agent-loop 이 조용히 잘라 버린다 — 즉사시킨다."""
    monkeypatch.setenv("A2_RESP_LEN", "8192")
    with pytest.raises(ValueError):
        t2.a2_resp_len(4096)


def test_launcher_widens_board_for_a2_resp_len():
    """run_math_arm.sh: A2_RESP_LEN=6144 이면 data.max_response_length 와
    max_model_len/max_num_batched_tokens 가 따라 넓어진다(E-131 프롬프트 폭은 불변)."""
    import os
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, A2_RESP_LEN="6144")
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_TRIAL2_CREDIT", "1", "100",
                        "--dry-run"], cwd=root, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "data.max_response_length=6144" in out
    assert "data.max_prompt_length=3072" in out
    assert "max_model_len=9472" in out and "max_num_batched_tokens=9472" in out
    assert "A2_RESP_LEN=6144" in out


# ── 재시도 게이트(0916, RETRY_GATE) ──────────────────────────────────────────
def _gate_env(monkeypatch, **kw):
    for k in ("RETRY_GATE", "RETRY_GATE_STATES", "RETRY_ONLY_WRONG"):
        monkeypatch.delenv(k, raising=False)
    for k, v in kw.items():
        monkeypatch.setenv(k, v)


#: 2문제 × K=4. 문제 0 = ALL_SAME(네 답이 전부 "5"), 문제 1 = SPLIT(3 + 1).
#: r1 은 **gold 채점** — 문제 0 은 전부 정답, 문제 1 은 다수결 답이 정답이고 하나만 오답.
_K = 4
_ANS = ["5", "5", "5", "5", "7", "7", "7", "9"]
_R1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_group_states_uses_activation_gate_definition():
    """상태 정의는 `math_activation_gate.agreement_state` 와 **같은 함수**다(그 스크립트가
    `trial2` 에서 import 한다) — 임계값을 복사하지 않았음을 두 경로로 고정한다."""
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts" / "local"))
    import math_activation_gate as A
    assert A.agreement_state is t2.agreement_state
    assert t2.group_states(_ANS, _K) == ["ALL_SAME"] * 4 + ["SPLIT"] * 4
    assert A.agreement_state(_ANS[:4], k=4)["state"] == "ALL_SAME"
    assert A.agreement_state(_ANS[4:], k=4)["state"] == "SPLIT"


def test_retry_gate_agree_selects_whole_low_agreement_group():
    """agree 게이트: ALL_SAME 그룹은 한 행도 안 고르고, SPLIT 그룹은 **정답 행까지** 전부."""
    rows, tel = t2.select_retry_rows(_R1, _K, gate="agree", answers=_ANS,
                                     states=t2.DEFAULT_RETRY_GATE_STATES)
    assert rows == [4, 5, 6, 7]
    assert tel["retried_by_state"] == {"SPLIT": 4.0}
    # 고른 4행 중 3행은 a1 이 **맞았다** — gold 는 로그에만 쓰인다
    assert abs(tel["retried_a1_correct_frac"] - 0.75) < 1e-9
    assert tel["n_retried"] == 4.0 and tel["n_rows"] == 8.0


def test_retry_gate_wrong_and_all_modes():
    """wrong 은 현행 그대로(오답 1행), all 은 전 행. 텔레메트리 수치도 그에 맞는다."""
    rows, tel = t2.select_retry_rows(_R1, _K, gate="wrong", answers=_ANS)
    assert rows == [7] and tel["retried_a1_correct_frac"] == 0.0
    assert tel["retried_by_state"] == {"SPLIT": 1.0}
    rows2, tel2 = t2.select_retry_rows(_R1, _K, gate="all", answers=_ANS)
    assert rows2 == list(range(8)) and tel2["n_retried"] == 8.0
    assert tel2["retried_by_state"] == {"ALL_SAME": 4.0, "SPLIT": 4.0}
    assert abs(tel2["retried_a1_correct_frac"] - 7 / 8) < 1e-9
    # answers 없이도 wrong/all 은 돈다(상태는 NA 로 센다)
    rows3, tel3 = t2.select_retry_rows(_R1, _K, gate="wrong")
    assert rows3 == [7] and tel3["retried_by_state"] == {"NA": 1.0}


def test_retry_gate_agree_needs_answers_and_rejects_bad_values(monkeypatch):
    with pytest.raises(ValueError):
        t2.select_retry_rows(_R1, _K, gate="agree")
    with pytest.raises(ValueError):
        t2.select_retry_rows(_R1, _K, gate="bogus")
    _gate_env(monkeypatch, RETRY_GATE="bogus")
    with pytest.raises(ValueError):
        t2.retry_gate()
    _gate_env(monkeypatch, RETRY_GATE_STATES="DOMINANT,NOPE")
    with pytest.raises(ValueError):
        t2.retry_gate_states()


def test_retry_gate_default_keeps_retry_only_wrong_meaning(monkeypatch):
    """RETRY_GATE 미설정이면 **현행 유지** — RETRY_ONLY_WRONG 의 옛 의미가 그대로다."""
    _gate_env(monkeypatch)
    assert t2.retry_gate() == "wrong"
    assert t2.retry_gate_states() == t2.DEFAULT_RETRY_GATE_STATES
    assert "ALL_SAME" not in t2.retry_gate_states()
    _gate_env(monkeypatch, RETRY_ONLY_WRONG="0")
    assert t2.retry_gate() == "all"
    _gate_env(monkeypatch, RETRY_ONLY_WRONG="0", RETRY_GATE="agree")
    assert t2.retry_gate() == "agree"
    _gate_env(monkeypatch, RETRY_GATE_STATES=" split , scatter ")
    assert t2.retry_gate_states() == ("SPLIT", "SCATTER")


def test_retry_gate_agree_feeds_reassembly_plan():
    """게이트가 고른 행이 `reassembly_plan` 의 `retried` 다 — ALL_SAME 그룹(문제 0)의
    note/a2 슬롯은 전부 비활성으로 남고 SPLIT 그룹(문제 1)만 켜진다(죽은 자리 규약 유지)."""
    rows, _ = t2.select_retry_rows(_R1, _K, gate="agree", answers=_ANS)
    perm, stages, active = t2.reassembly_plan(2, _K, rows)
    assert len(perm) == 2 * 3 * _K
    on = {(stages[i], active[i]) for i in range(len(stages))}
    assert ("a1", 1) in on
    # 문제 0 블록 [0,12) 의 note/a2 는 전부 0, 문제 1 블록 [12,24) 는 전부 1
    assert all(active[i] == 0 for i in range(_K, 3 * _K))
    assert all(active[i] == 1 for i in range(12 + _K, 24))
