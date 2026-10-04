"""관문 G3 — LOO 다수결 · 정오 라벨."""
from __future__ import annotations

from mc import credit as C


def test_majority_label_loo_excludes_self():
    # 자기(0번)는 "9" 이고 형제 넷이 "3" — LOO 다수는 "3" 이다(자기 표가 안 들어간다).
    assert C.majority_label(["9", "3", "3", "3", "7"], self_idx=0) == "3"
    # 자기 표가 캐스팅보트가 되는 자리: 포함하면 2대2 동률(라벨 없음)인데 LOO 면 형제 다수가
    # 정해진다 — 자기 답 "9" 가 스스로를 다수로 만들지 못한다.
    assert C.majority_label(["9", "9", "3", "3"]) is None               # 자기 포함 → 동률
    assert C.majority_label(["9", "9", "3", "3"], self_idx=0) == "3"    # LOO → 형제 다수
    assert C.majority_label(["9", "9", "9", "3"], self_idx=0) == "9"
    # LOO 뒤 동률이면 라벨이 없다.
    assert C.majority_label(["1", "9", "3"], self_idx=0) is None


def test_majority_label_needs_two_siblings():
    assert C.majority_label(["3", "3"], self_idx=0) is None     # LOO 뒤 형제 1개
    assert C.majority_label(["3", "", ""], self_idx=0) is None
    assert C.majority_label([]) is None


def test_majority_label_uses_math_equivalence():
    assert C.majority_label(["x", "0.5", r"\frac{1}{2}", "7"], self_idx=0) in ("0.5", r"\frac{1}{2}")


def test_label_correct_gold_and_majority():
    assert C.label_correct("42", gold="42") == 1.0
    assert C.label_correct("41", gold="42") == 0.0
    assert C.label_correct("3", label="majority", sib_answers=["3", "3", "3", "9"],
                           self_idx=0) == 1.0
    assert C.label_correct("9", label="majority", sib_answers=["9", "3", "3", "3"],
                           self_idx=0) == 0.0
    # 라벨이 없으면 None(= 항이 없다, 0 이 아니다)
    assert C.label_correct("3", label="majority", sib_answers=["3", "9"], self_idx=0) is None
