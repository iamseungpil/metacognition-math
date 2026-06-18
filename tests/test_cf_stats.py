from src.eval.cf_stats import mcnemar_exact_p, is_parsed, degeneracy_flags


def test_mcnemar_saved_gt_broke_not_significant_rejects():
    # b=7 saved, c=4 broke -> NOT significant at p<0.05 (must not be called a win)
    assert mcnemar_exact_p(b=7, c=4) > 0.05


def test_mcnemar_clear_effect_significant():
    assert mcnemar_exact_p(b=20, c=3) < 0.05


def test_mcnemar_symmetric():
    assert abs(mcnemar_exact_p(b=7, c=4) - mcnemar_exact_p(b=4, c=7)) < 1e-9


def test_parse_fail_dropped():
    assert is_parsed("") is False
    assert is_parsed("the answer is 18") is True


def test_degeneracy_flags_repetition():
    f = degeneracy_flags("the the the the the the the the", min_len=3)
    assert f["repetition"] is True


def test_degeneracy_flags_short():
    f = degeneracy_flags("hi", min_len=10)
    assert f["too_short"] is True


def test_clean_output_no_flags():
    f = degeneracy_flags("Janet sells 9 eggs at $2 each, so she makes $18.", min_len=3)
    assert f["repetition"] is False and f["too_short"] is False
