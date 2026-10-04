

def test_choice_marker_stripped_from_pred():
    r"""mc/grade.py 와 같은 수리 — 표식이 붙은 박스가 gold 와 같게 채점된다(동률 유지)."""
    from src.training.math_meta import grade_math as gm
    assert gm(r"\boxed{\text{(E)}\ \dfrac{7}{2}}", r"\dfrac{7}{2}") == 1
    assert gm(r"\boxed{\text{(C) } \frac{1}{6}}", r"\frac{1}{6}") == 1
    assert gm(r"\boxed{\text{(E)}}", "E") == 1
    assert gm(r"\boxed{\text{(C) } \frac{1}{2}}", r"\frac{1}{6}") == 0
    assert gm(r"\boxed{C}", "0.20") == 0
