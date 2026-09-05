r"""OPD (힌트 교사, `src/training/countdown_opd.py`) 순수 함수 단위 검증.

GPU/torch/verl 없이 CPU 만으로 돈다 — 이 모듈 자체가 순수 함수만 담기 때문이다.
teacher forward(`verl_sdc._compute_countdown_opd`)는 여기서 검증하지 않는다(그건
GPU 가 필요하고, 이 파일이 검증하는 순수 조각들 위에 얇게 얹힌 배선일 뿐이다).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.training import countdown_opd as opd                     # noqa: E402
from src.training import countdown_sites as cds                   # noqa: E402
from src.training import countdown_task as ct                     # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# build_hint
# ══════════════════════════════════════════════════════════════════════════════

def test_build_hint_no_attempts_returns_empty():
    # 시도가 아예 없는 prefix → family_dead is None → 힌트 생략.
    assert opd.build_hint([24, 15, 22, 7], 331, "Let me think about this.") == ""


def test_build_hint_dead_family_lists_live_moves():
    # nums=[1,2,3,4] target=10: 실측(스크립트로 확인) (2,'-',1) 은 해로 이어지지 않는다
    # (family_dead=1) 이고, 나머지 살아있는 첫수(1+3, 3+4 등)가 live_new_moves 에 남는다.
    nums = [1, 2, 3, 4]
    target = 10
    prefix = "2-1=1\n"
    oracle = cds.oracle_for_site(prefix, nums, target)
    assert oracle["family_dead"] == 1
    assert oracle["live_new_moves"]        # 여럿 살아 있다
    hint = opd.build_hint(nums, target, prefix)
    assert hint != ""
    assert hint.splitlines()[0] == "Hint: your current line of attack is dead."
    assert hint.splitlines()[1] == (
        "Hint: first moves that still reach the target: "
        + ", ".join(oracle["live_new_moves"]) + ".")


def test_build_hint_never_contains_witness():
    # 오라클 witness 는 실제 정답식이다 — 힌트 어디에도 나타나면 안 된다(설계 §1.1/§5).
    nums = [2, 3, 4, 5]
    target = 14   # 2*3+4+... 여러 해가 존재하는 작은 인스턴스
    oracle = cds.oracle_for_site("2+3=5\n", nums, target)
    witness = oracle.get("witness") or ""
    hint = opd.build_hint(nums, target, "2+3=5\n")
    assert witness == "" or witness not in hint
    # decoy/최종식/박스형도 없어야 한다.
    assert "\\boxed" not in hint
    assert "witness" not in hint.lower()


def test_build_hint_none_found_when_no_live_moves():
    # nums=[1,2,1,2] target=7: 실측(스크립트로 확인) — 유일하게 살아있는 첫수 "1+2" 는
    # (합계가 4개 해로 이어짐에도) 이미 pairs_pre 에 있어 live_new_moves 에서 빠지고,
    # family_dead 도 1 이다 — "none found" 분기를 실제로 때리는 인스턴스.
    nums = [1, 2, 1, 2]
    target = 7
    prefix = "1*2=99\n2-1=98\n"
    oracle = cds.oracle_for_site(prefix, nums, target)
    assert oracle["family_dead"] == 1
    assert oracle["live_new_moves"] == []
    hint = opd.build_hint(nums, target, prefix)
    assert hint == ("Hint: your current line of attack is dead.\n"
                     "Hint: first moves that still reach the target: none found.")


# ══════════════════════════════════════════════════════════════════════════════
# hinted_messages
# ══════════════════════════════════════════════════════════════════════════════

def _base_messages():
    return ct.build_prompt({"nums": [24, 15, 22, 7], "target": 331}, "new")


def test_hinted_messages_appends_hint_to_last_user_message():
    msgs = _base_messages()
    hint = "Hint: your current line of attack is dead."
    out = opd.hinted_messages(msgs, hint, "Let me try 24*15.\n")
    # user 메시지 끝에 힌트가 붙는다.
    user_msgs = [m for m in out if m["role"] == "user"]
    assert user_msgs[-1]["content"].endswith(hint)
    # assistant 프리픽스가 마지막에 추가된다.
    assert out[-1] == {"role": "assistant", "content": "Let me try 24*15.\n"}


def test_hinted_messages_empty_hint_does_not_touch_user_message():
    msgs = _base_messages()
    out = opd.hinted_messages(msgs, "", "prefix text")
    orig_user = [m["content"] for m in msgs if m["role"] == "user"]
    new_user = [m["content"] for m in out if m["role"] == "user"]
    assert orig_user == new_user
    assert out[-1] == {"role": "assistant", "content": "prefix text"}


def test_hinted_messages_does_not_mutate_input():
    msgs = _base_messages()
    before = [dict(m) for m in msgs]
    opd.hinted_messages(msgs, "Hint: x.", "prefix")
    assert msgs == before


def test_hinted_messages_no_user_message_raises():
    with pytest.raises(ValueError):
        opd.hinted_messages([{"role": "system", "content": "sys"}], "hint", "prefix")


# ══════════════════════════════════════════════════════════════════════════════
# opd_spans
# ══════════════════════════════════════════════════════════════════════════════

META = "<meta>\nconfidence: 0.6\ndecision: redirect\n</meta>"


def test_opd_spans_no_meta_returns_all_none():
    assert opd.opd_spans("no meta block here, just \\boxed{1}") == (None, None, None)


def test_opd_spans_meta_with_following_attempt_line():
    text = "before " + META + "\n5+19=24\nmore text"
    meta_start, meta_end, first_attempt_end = opd.opd_spans(text)
    assert text[meta_start:meta_end] == META
    # first_attempt_end 는 "5+19=24\n" 줄 끝(줄바꿈 포함) 다음까지.
    assert text[meta_end:first_attempt_end] == "\n5+19=24\n"
    assert text[first_attempt_end:] == "more text"


def test_opd_spans_meta_no_following_attempt_collapses_to_meta_end():
    text = "before " + META + "\njust prose, no equation here"
    meta_start, meta_end, first_attempt_end = opd.opd_spans(text)
    assert first_attempt_end == meta_end


def test_opd_spans_attempt_line_with_no_trailing_newline_ends_at_text_end():
    text = META + "\n5+19=24"    # no trailing \n
    meta_start, meta_end, first_attempt_end = opd.opd_spans(text)
    assert first_attempt_end == len(text)


def test_opd_spans_incomplete_meta_returns_all_none():
    # confidence 만 있고 decision 없음 → emitted=0.
    text = "<meta>\nconfidence: 0.6\n</meta>\n5+19=24\n"
    assert opd.opd_spans(text) == (None, None, None)


# ══════════════════════════════════════════════════════════════════════════════
# opd_reward
# ══════════════════════════════════════════════════════════════════════════════

def test_opd_reward_none_is_zero():
    assert opd.opd_reward(None, 1.0) == 0.0


def test_opd_reward_negative_kl_clips_to_zero():
    assert opd.opd_reward(-5.0, 1.0) == 0.0


def test_opd_reward_zero_kl_is_zero():
    assert opd.opd_reward(0.0, 1.0) == 0.0


def test_opd_reward_within_range_scales_linearly():
    assert opd.opd_reward(0.5, 1.0) == pytest.approx(-0.5)


def test_opd_reward_clips_at_C():
    assert opd.opd_reward(10.0, 2.0) == pytest.approx(-1.0)


def test_opd_reward_bounded_in_minus_one_zero():
    for kl in (-3.0, -0.01, 0.0, 0.1, 0.5, 1.0, 5.0, 100.0):
        r = opd.opd_reward(kl, 0.75)
        assert -1.0 <= r <= 0.0


def test_opd_reward_rejects_nonpositive_C():
    with pytest.raises(ValueError):
        opd.opd_reward(0.5, 0.0)
    with pytest.raises(ValueError):
        opd.opd_reward(0.5, -1.0)
