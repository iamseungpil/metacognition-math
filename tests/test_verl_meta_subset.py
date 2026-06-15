from src.training.verl_gdpo_data import _gold_is_rule_gradable


def test_numeric_and_boxed_gold_pass():
    assert _gold_is_rule_gradable("42")
    assert _gold_is_rule_gradable("\\frac{3}{4}")
    assert _gold_is_rule_gradable("7\\sqrt{5}")


def test_prose_gold_rejected():
    assert not _gold_is_rule_gradable("\\text{Yes, it must be a cube.}")
    assert not _gold_is_rule_gradable("Player 0 wins")
    assert not _gold_is_rule_gradable("")
