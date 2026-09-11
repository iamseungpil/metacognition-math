"""내용-진리 자(meta_content_rulers) 회귀 시험.

특히 `_arith_claims` 는 첫 판에서 부분식을 떼어내 채점하는 결함이 있었다
(`(22+11)*20*3 = 1980` 에서 `20*3 = 1980` 만 보고 거짓으로 셈) — 그 거짓 음성이
«검산 주장의 58% 가 거짓» 이라는 가짜 발견을 만들 뻔했다. 그 계열을 여기서 막는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))
import meta_content_rulers as M  # noqa: E402

NUMS = [24, 20, 11, 2]


def test_arith_claim_full_expression_not_subexpression():
    """괄호를 포함한 식 전체로 평가한다 — 꼬리 부분식만 떼어 쓰면 안 된다."""
    assert M._arith_claims("(22 + 11) * 20 * 3 = 1980") == (1, 1)


def test_arith_claim_detects_a_real_false_claim():
    assert M._arith_claims("20 * 3 = 61") == (1, 0)


def test_arith_claim_ignores_text_without_equation():
    assert M._arith_claims("I should try a different strategy") == (0, 0)


def test_named_moves_requires_both_operands_in_instance():
    """주어진 수로 만든 첫수만 «지목» 으로 센다 — 아무 숫자쌍이나 세면 안 된다."""
    assert M._named_moves("let me try 24 * 20", NUMS) == [(20, "*", 24)]
    assert M._named_moves("maybe 99 * 98 works", NUMS) == []


def test_named_moves_repeated_number_needs_two_copies():
    assert M._named_moves("try 2 * 2", NUMS) == []
    assert M._named_moves("try 2 * 2", [2, 2, 5, 5]) == [(2, "*", 2)]


def test_analyse_row_counts_blocks_and_position():
    row = {"group_id": "g", "r_corr": 0, "nums": NUMS, "target": 502,
           "text": "start <check> (24 * 20) = 480 </check> tail"}
    recs = M.analyse_row(row, "check")
    assert len(recs) == 1
    assert recs[0]["names_move"] == 1
    assert 0.0 < recs[0]["pos_frac"] < 1.0


def test_summarise_empty_is_safe():
    assert M.summarise([], []) == {"n_blocks": 0}
