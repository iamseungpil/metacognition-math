"""Contracts for the R_cal repair (cal_mode, 2026-08-12).

Pinned here:
  - default cal_mode="brier_neg" is byte-identical to the legacy head
  - info_gain math: log2(conf_c/0.5) if correct else log2((1-conf_c)/0.5),
    conf clamped to [0.05, 0.95]; silence and conf=0.5 both score exactly 0;
    honest doubt on a wrong rollout is POSITIVE (the rewarding-doubt property)
  - info_gain parses confidence ONLY inside the first closed meta block
    (source unified with the CONF token mask); legacy keeps full-text parse
  - the observability arrays (conf_parsed / cal_positive / conf_gap /
    cal_group_gap / meta_first / has_think) exist in every mode
"""
import math

from src.training.dcpo_region import dcpo_region_rewards

META = "<|meta|>\nconfidence: {c}\n<|/meta|>"


def _mk(conf=None, correct=True, meta_first=False, think=True, conf_outside=None):
    gold = "42"
    body = "<think>\nsome reasoning 2+2\n</think>\n" if think else ""
    meta = META.format(c=conf) if conf is not None else ""
    tail = f"\\boxed{{{gold if correct else '7'}}}"
    outside = f"\nconfidence: {conf_outside}\n" if conf_outside is not None else ""
    if meta_first:
        return meta + "\n" + body + outside + tail
    return body + meta + outside + tail


def _run(texts, **kw):
    return dcpo_region_rewards(
        texts, ground_truth=["42"] * len(texts), group_index=[0] * len(texts), **kw
    )


def test_default_mode_is_legacy_brier():
    t = [_mk(conf=0.9, correct=True)]
    legacy = _run(t)["R_cal"][0]
    explicit = _run(t, cal_mode="brier_neg")["R_cal"][0]
    assert legacy == explicit
    assert abs(legacy - (-((0.9 - 1.0) ** 2))) < 1e-9  # -(conf-c)^2


def test_info_gain_correct_confident_positive():
    r = _run([_mk(conf=0.9, correct=True)], cal_mode="info_gain")["R_cal"][0]
    assert abs(r - math.log2(0.9 / 0.5)) < 1e-9
    assert r > 0


def test_info_gain_honest_doubt_on_wrong_is_positive():
    # The rewarding-doubt property: low confidence before a wrong answer earns
    # the SAME reward as high confidence before a correct one.
    r = _run([_mk(conf=0.1, correct=False)], cal_mode="info_gain")["R_cal"][0]
    assert abs(r - math.log2(0.9 / 0.5)) < 1e-9
    assert r > 0


def test_info_gain_misleading_overconfidence_most_negative():
    r = _run([_mk(conf=0.9, correct=False)], cal_mode="info_gain")["R_cal"][0]
    assert abs(r - math.log2(0.1 / 0.5)) < 1e-9
    assert r < -2


def test_info_gain_uninformative_and_silence_are_zero():
    out = _run(
        [_mk(conf=0.5, correct=True), _mk(conf=None, correct=True)],
        cal_mode="info_gain",
    )
    assert out["R_cal"][0] == 0.0
    assert out["R_cal"][1] == 0.0


def test_info_gain_clamp():
    # _parse_confidence already clips to [0.01, 0.99]; the head clamps further
    # to [0.05, 0.95] so the log stays bounded.
    r = _run([_mk(conf=0.99, correct=True)], cal_mode="info_gain")["R_cal"][0]
    assert abs(r - math.log2(0.95 / 0.5)) < 1e-9


def test_info_gain_ignores_conf_outside_meta_but_legacy_uses_it():
    t = [_mk(conf=None, correct=True, conf_outside=0.8)]
    ig = _run(t, cal_mode="info_gain")
    lg = _run(t, cal_mode="brier_neg")
    assert ig["R_cal"][0] == 0.0 and ig["conf_parsed"][0] == 0.0
    assert lg["conf_parsed"][0] == 1.0  # full-text regex finds it (legacy verbatim)


def test_observability_arrays_present_all_modes():
    t = [_mk(conf=0.75, correct=True), _mk(conf=0.3, correct=False, meta_first=True, think=False)]
    for mode in ("brier_neg", "info_gain"):
        out = _run(t, cal_mode=mode)
        for k in ("conf_parsed", "cal_positive", "conf_gap", "cal_group_gap",
                  "meta_first", "has_think"):
            assert k in out and len(out[k]) == 2, (mode, k)
        assert out["meta_first"] == [0.0, 1.0]
        assert out["has_think"] == [1.0, 0.0]


def test_info_gain_cal_positive_flag():
    out = _run(
        [_mk(conf=0.9, correct=True), _mk(conf=0.9, correct=False)],
        cal_mode="info_gain",
    )
    assert out["cal_positive"] == [1.0, 0.0]


def test_unknown_cal_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        _run([_mk(conf=0.9)], cal_mode="INFO_GAIN")


def test_info_gain_finds_conf_in_second_closed_block():
    t = ("<think>r</think>\n<|meta|>\nassessment: fine\n<|/meta|>\n"
         "<|meta|>\nconfidence: 0.8\n<|/meta|>\n\\boxed{42}")
    out = _run([t], cal_mode="info_gain")
    assert out["conf_parsed"][0] == 1.0
    assert abs(out["R_cal"][0] - math.log2(0.8 / 0.5)) < 1e-9


def test_info_gain_unclosed_first_block_does_not_leak_across():
    # first opener never closes; a later block closes with a conf — the parse
    # must skip the unclosed block and read the CLOSED one, never slicing
    # across two openers.
    t = ("<think>r</think>\n<|meta|>\ndangling junk\n"
         "<|meta|>\nconfidence: 0.9\n<|/meta|>\n\\boxed{42}")
    out = _run([t], cal_mode="info_gain")
    assert out["conf_parsed"][0] == 1.0
    assert abs(out["R_cal"][0] - math.log2(0.9 / 0.5)) < 1e-9


def test_info_gain_group_targets_group_rate():
    import math as _m
    # 8형제 중 4개 정답(p̂=0.5): conf 0.5 가 최적(=0, 침묵과 동률), 0.9 는 음수
    texts = [_mk(conf=0.5, correct=(i < 4)) for i in range(8)]
    out = dcpo_region_rewards(texts, ground_truth=["42"] * 8, group_index=[0] * 8,
                              cal_mode="info_gain_group")
    for v in out["R_cal"]:
        assert abs(v - 0.0) < 1e-9
    texts2 = [_mk(conf=0.9, correct=(i < 4)) for i in range(8)]
    out2 = dcpo_region_rewards(texts2, ground_truth=["42"] * 8, group_index=[0] * 8,
                               cal_mode="info_gain_group")
    expected = 0.5 * _m.log2(0.9 / 0.5) + 0.5 * _m.log2(0.1 / 0.5)
    assert all(abs(v - expected) < 1e-9 for v in out2["R_cal"])
    assert expected < 0  # p̂=0.5 인데 0.9 라 말하면 벌


def test_info_gain_group_reduces_to_binary_at_extremes():
    import math as _m
    # 전원 정답(p̂=1): 그룹판 == 이진판
    texts = [_mk(conf=0.8, correct=True) for _ in range(4)]
    g = dcpo_region_rewards(texts, ground_truth=["42"] * 4, group_index=[0] * 4,
                            cal_mode="info_gain_group")["R_cal"][0]
    b = dcpo_region_rewards(texts, ground_truth=["42"] * 4, group_index=[0] * 4,
                            cal_mode="info_gain")["R_cal"][0]
    assert abs(g - b) < 1e-9 and abs(g - _m.log2(0.8 / 0.5)) < 1e-9


def test_info_gain_group_honest_low_conf_on_hard_group_positive():
    # p̂=0.25 인 그룹에서 conf 0.25 는 양수(유익), conf 0.9 는 크게 음수
    import math as _m
    texts = [_mk(conf=0.25, correct=(i < 1)) for i in range(4)]
    out = dcpo_region_rewards(texts, ground_truth=["42"] * 4, group_index=[0] * 4,
                              cal_mode="info_gain_group")
    exp = 0.25 * _m.log2(0.25 / 0.5) + 0.75 * _m.log2(0.75 / 0.5)
    assert all(abs(v - exp) < 1e-9 for v in out["R_cal"])
    assert exp > 0
