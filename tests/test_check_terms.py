"""§13 검산 항(0909): fclaim / chk_fixed / chk_evc 원재료와 프롬프트 변형 chk."""
from src.training import countdown_rewards as C
from src.training import countdown_task as T

NUMS, TGT = [20, 10, 7, 10], 27


def test_prompt_variant_chk_exists_and_is_compact():
    p = T.PROMPT_VARIANTS["chk"]
    assert "<check>" in p and "<meta>" not in p
    assert len(p) - len(T.PROMPT_VARIANTS["plain"]) < 400


def test_false_claim_without_flag():
    r = C.check_row("... \\boxed{(20+10)+7-10}", NUMS, TGT, r_corr=0)
    assert r["fclaim"] == 1 and r["chk_fixed"] == 0 and r["chk_evc"] == 0


def test_flagged_wrong_box_is_not_false_claim():
    t = "<check> (20+10)+7-10 = 27 ✗ </check>\n\\boxed{(20+10)+7-10}"
    r = C.check_row(t, NUMS, TGT, r_corr=0)
    assert r["fclaim"] == 0 and r["chk_fixed"] == 1 and r["chk_evc"] == 0


def test_caught_and_fixed_gets_evc():
    t = "try 20*10-7-10 <check> 20*10-7-10 = 183 ✗ </check> then <check> (20+10)-(10-7) = 27 ✓ </check>\n\\boxed{(20+10)-(10-7)}"
    r = C.check_row(t, NUMS, TGT, r_corr=1)
    assert r["fclaim"] == 0 and r["chk_fixed"] == 1 and r["chk_evc"] == 1


def test_bogus_flag_on_correct_expr_gets_no_evc():
    """정답인 식을 ✗ 라고 한 뒤 같은 식을 박스 — «거짓을 잡은」 게 아니다."""
    t = "<check> (20+10)-(10-7) = 27 ✗ </check>\n\\boxed{(20+10)-(10-7)}"
    r = C.check_row(t, NUMS, TGT, r_corr=1)
    assert r["chk_evc"] == 0


def test_arm_specs_registered():
    for a in ("TAG0", "FIXED_CHK", "EVC_CHK"):
        assert "fclaim" in C.ARM_SPECS[a]["terms"]
    assert "chk_evc" in C.ARM_SPECS["EVC_CHK"]["terms"] and "chk_evc" not in C.ARM_SPECS["FIXED_CHK"]["terms"]


def test_arm_reward_signs():
    row = {"r_corr": 0, "format_ok": 1, "fclaim": 1, "chk_fixed": 0, "chk_evc": 0, "emitted": 0}
    tot0 = C.arm_reward("TAG0", row, step=50)[0]
    row2 = dict(row, fclaim=0)
    assert C.arm_reward("TAG0", row2, step=50)[0] > tot0
    row3 = {"r_corr": 1, "format_ok": 1, "fclaim": 0, "chk_fixed": 1, "chk_evc": 1, "emitted": 0}
    assert C.arm_reward("EVC_CHK", row3, step=50)[0] > C.arm_reward("TAG0", row3, step=50)[0]
