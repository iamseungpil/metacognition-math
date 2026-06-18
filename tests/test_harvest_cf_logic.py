from scripts.harvest_redirect_cf import (
    well_formed_redirect,
    splice_index,
    arm_rate,
    accept_redirect,
    expected_yield,
)


def test_well_formed_requires_switch_and_low_conf():
    good = "<|meta|>\nconfidence: 0.2\n<|switch|> try a different method\n<|/meta|>"
    assert well_formed_redirect(good) is True
    no_switch = "<|meta|>\nconfidence: 0.2\nlooks fine\n<|/meta|>"
    assert well_formed_redirect(no_switch) is False
    high_conf = "<|meta|>\nconfidence: 0.9\n<|switch|> x\n<|/meta|>"
    assert well_formed_redirect(high_conf) is False


def test_splice_index_in_band():
    assert splice_index(100, frac=0.3) == 30
    assert splice_index(100, frac=0.7) == 70
    assert splice_index(10, frac=0.3) >= 1  # never zero-length on tiny traces


def test_arm_rate():
    assert arm_rate([1, 1, 0, 0]) == 0.5
    assert arm_rate([]) == 0.0


def test_accept_only_when_redirect_beats_all_controls_by_lower_ci_margin():
    # R causally flips strongly (8/8); all controls 0/8 -> lower-CI gap clears margin
    assert accept_redirect([1] * 8, [0] * 8, [0] * 8, bprime_grades=[0] * 8, margin=0.5) is True
    # B' (plain-prose 2nd attempt) ALSO recovers -> redirect content adds nothing -> reject
    assert accept_redirect([1] * 8, [0] * 8, [0] * 8, bprime_grades=[1] * 8, margin=0.5) is False
    # noisy small gap (2/8 vs 0/8): point diff 0.25 but lower-CI < margin -> reject
    assert accept_redirect([1, 1, 0, 0, 0, 0, 0, 0], [0] * 8, [0] * 8, bprime_grades=[0] * 8, margin=0.5) is False
    # too few samples -> reject (INSUFFICIENT)
    assert accept_redirect([1, 1], [0, 0], [0, 0], bprime_grades=[0, 0], margin=0.5) is False


def test_expected_yield_product():
    y = expected_yield(emission_rate=0.5, in_band_frac=0.4, accept_prob=0.2, pool_size=50000)
    assert y == int(0.5 * 0.4 * 0.2 * 50000)  # 2000
