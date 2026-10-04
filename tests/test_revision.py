r"""src/training/revision.py — 수정 구간·크레딧의 순수 함수 단위 시험(CPU)."""
import pytest

from src.training.revision import combo_save, revision_cf_credit, revision_zone


def test_no_zone_when_fewer_than_two_boxes():
    assert revision_zone("") is None
    assert revision_zone("no box at all") is None
    assert revision_zone(r"only one \boxed{7} here") is None


def test_zone_boundaries_are_first_box_end_and_last_boxed_start():
    t = r"so \boxed{7}. Wait, let me double-check. Actually \boxed{9}"
    z = revision_zone(t)
    assert z is not None
    assert z["first_answer"] == "7"
    assert z["last_answer"] == "9"
    # zone_start = 첫 박스의 닫는 `}` 바로 뒤
    assert t[z["zone_start"] - 1] == "}"
    # zone_end = 마지막 `\boxed` 가 **시작**하는 지점 — 최종 답은 구간 밖이다.
    assert t[z["zone_end"]:].startswith(r"\boxed")
    assert r"\boxed{9}" not in t[z["zone_start"]:z["zone_end"]]
    assert "double-check" in t[z["zone_start"]:z["zone_end"]]
    assert z["n_boxes"] == 2
    assert z["n_change_points"] == 1
    assert z["revised"] is True


def test_unrevised_when_answers_equal():
    z = revision_zone(r"\boxed{7} recheck \boxed{7}")
    assert z is not None and z["revised"] is False
    assert z["n_change_points"] == 0


def test_loose_equivalence_skips_notation_only_change():
    z = revision_zone(r"\boxed{0 \text{ and } -3} hmm \boxed{\{-3, 0\}}")
    assert z is not None
    assert z["revised"] is False          # 표기만 다름 — 수정이 아니다
    assert z["n_change_points"] == 0


def test_one_vs_multi_change_point():
    one = revision_zone(r"\boxed{1} a \boxed{1} b \boxed{2}")
    assert one["n_change_points"] == 1 and one["revised"] is True
    multi = revision_zone(r"\boxed{1} a \boxed{2} b \boxed{3}")
    assert multi["n_change_points"] == 2 and multi["revised"] is True
    back = revision_zone(r"\boxed{1} a \boxed{2} b \boxed{1}")
    assert back["n_change_points"] == 2 and back["revised"] is False


@pytest.mark.parametrize(
    "fc,lc,expected",
    [(False, True, 1.0), (True, False, -2.0), (True, True, 0.0), (False, False, 0.0)],
)
def test_cf_credit_truth_table(fc, lc, expected):
    assert revision_cf_credit(fc, lc, 1.0, 2.0) == expected


def test_cf_credit_uses_magnitudes():
    # 부호 오타가 보상을 뒤집지 못한다.
    assert revision_cf_credit(False, True, -3.0, 2.0) == 3.0
    assert revision_cf_credit(True, False, 1.0, -4.0) == -4.0


def test_combo_save_doubles_when_majority_wrong():
    assert combo_save(1.0, True) == 1.0
    assert combo_save(1.0, False) == 2.0
    assert combo_save(-1.5, False) == 3.0
