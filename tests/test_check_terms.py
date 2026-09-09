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


def test_check_region_advantage_lands_only_on_check_tokens():
    """§13-b: 문자=토큰인 가짜 토크나이저로, 그룹 중심화된 메모 값이 <check> 구간에만 더해진다."""
    import torch
    from src.training import verl_sdc as V

    class _Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)
    text_a = "ab<check>x</check>cd"      # row 0: check 있음
    text_b = "abcdefghijklmnopqrst"        # row 1: 없음(같은 그룹)
    ids = torch.tensor([[ord(ch) for ch in text_a], [ord(ch) for ch in text_b]])

    class _D:
        pass
    d = _D()
    d.batch = {"advantages": torch.zeros(2, ids.shape[1]), "responses": ids,
               "response_mask": torch.ones(2, ids.shape[1], dtype=torch.long)}
    import re
    spans = [[(m.start(), m.end()) for m in re.finditer(r"<check>.*?</check>", text_a)], []]
    V._CHK_REGION_STASH.update({"step": 1, "bs": 2, "uid": ["g", "g"], "meta": [0.5, 0.0], "spans": spans})
    out = V._countdown_add_check_region_advantage(d, tokenizer=_Tok())
    a = out.batch["advantages"]
    assert a[1].abs().sum().item() == 0
    s0, e0 = spans[0][0]
    assert torch.allclose(a[0, s0:e0], torch.full((e0 - s0,), 0.25))   # 0.5 − mean(0.25)
    assert a[0, :s0].abs().sum().item() == 0 and a[0, e0:].abs().sum().item() == 0


def test_false_alarm_is_penalised():
    """0909: 맞는 식을 ✗ 로 깎고 박스 = 허위 경보도 fclaim."""
    t = "<check> (20+10)-(10-7) = 27 ✗ </check>\n\\boxed{(20+10)-(10-7)}"
    r = C.check_row(t, NUMS, TGT, r_corr=1)
    assert r["fclaim"] == 1 and r["false_alarm"] == 1 and r["over_claim"] == 0


def test_reject_and_revise_gets_evc_even_if_final_wrong():
    """0909 완화: 진짜 틀린 식을 ✗ 로 잡고 다른 식을 박스하면, 최종 오답이어도 메타 크레딧."""
    t = "<check> 20*10-7-10 = 183 ✗ </check> keep searching\n\\boxed{20+10+7-10}"
    r = C.check_row(t, NUMS, TGT, r_corr=0)
    assert r["chk_evc"] == 1 and r["over_claim"] == 1


def test_flag_wrong_then_box_it_anyway_gets_no_evc():
    t = "<check> 20*10-7-10 = 183 ✗ </check>\n\\boxed{20*10-7-10}"
    r = C.check_row(t, NUMS, TGT, r_corr=0)
    assert r["chk_evc"] == 0 and r["fclaim"] == 0
