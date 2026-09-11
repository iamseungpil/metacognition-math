"""§13 검산 항(0909): fclaim / chk_fixed / chk_evc 원재료와 프롬프트 변형 chk."""
from src.training import countdown_rewards as C
from src.training import countdown_task as T

NUMS, TGT = [20, 10, 7, 10], 27


def test_prompt_variant_chk_exists_and_is_compact():
    p = T.PROMPT_VARIANTS["chk"]
    assert "<check>" in p and "<meta>" not in p
    assert len(p) - len(T.PROMPT_VARIANTS["plain"]) < 500   # v4 재탐색 허가 한 문장 추가(428자)


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


def test_persist_requires_real_extra_search():
    """P2: ✗ 후 새 시도 없이 다른 식만 박스하면 persist 0 (EVC 의 허점)."""
    lazy = "<check> 20*10-7-10 = 183 ✗ </check>\n\\boxed{20+10+7-10}"  # ✗ 뒤 등식 0개
    r = C.check_row(lazy, NUMS, TGT, r_corr=0)
    assert r["chk_evc"] == 1 and r["chk_persist"] == 0 and r["chk_solved"] == 0
    real = ("<check> 20*10-7-10 = 183 ✗ </check>\n20+10 = 30\n30-7 = 23\n10-7 = 3\n"
            "30-3 = 27\n\\boxed{(20+10)-(10-7)}")
    r2 = C.check_row(real, NUMS, TGT, r_corr=1)
    assert r2["chk_persist"] == 1 and r2["chk_solved"] == 1


def test_persist_arm_registered_and_ordered():
    sp = C.ARM_SPECS["PERSIST_CHK"]["terms"]
    assert "chk_persist" in sp and "chk_solved" in sp and "chk_evc" not in sp
    row = {"r_corr": 1, "format_ok": 1, "fclaim": 0, "chk_persist": 1, "chk_solved": 1, "emitted": 0}
    row2 = dict(row, chk_solved=0)
    assert C.arm_reward("PERSIST_CHK", row, step=50)[0] > C.arm_reward("PERSIST_CHK", row2, step=50)[0]


def test_tax0_is_n0_terms_with_chk_prompt():
    """R2: TAX0 는 항이 N0/OPT 와 같고(corr+format) 프롬프트만 chk — 순수 문법 세금 대조군."""
    assert C.ARM_SPECS["TAX0"]["terms"] == ("corr", "format")
    assert C.ARM_SPECS["TAX0"]["prompt_variant"] == "chk"
    assert "fclaim" in C.ARM_SPECS["TAG0"]["terms"] and "fclaim" not in C.ARM_SPECS["TAX0"]["terms"]


def test_evcm_mask_zeroes_decorative_check_but_keeps_solved_and_overclaim():
    """§13-c: keep 규칙 — solved/over_claim 은 통과, 장식적 ✓·정직한 ✗ 는 check 토큰 어드밴티지 0."""
    import torch, re
    from src.training import verl_sdc as V
    class _Tok:
        def decode(self, ids, skip_special_tokens=False): return "".join(chr(i) for i in ids)
    texts = ["ab<check>x</check>cd", "ab<check>y</check>cd"]
    ids = torch.tensor([[ord(ch) for ch in t] for t in texts])
    class _D: pass
    d = _D(); d.batch = {"advantages": torch.ones(2, ids.shape[1]), "responses": ids,
                         "response_mask": torch.ones(2, ids.shape[1], dtype=torch.long)}
    spans = [[(m.start(), m.end()) for m in re.finditer(r"<check>.*?</check>", t)] for t in texts]
    V._CHK_REGION_STASH.update({"step": 1, "bs": 2, "uid": ["g", "g"], "meta": [0.0, 0.0],
                                "spans": spans, "keep": [1, 0], "mask_on": True})
    out = V._countdown_add_check_region_advantage(d, tokenizer=_Tok())
    a = out.batch["advantages"]; s0, e0 = spans[1][0]
    assert a[0].sum().item() == ids.shape[1]            # keep=1: 그대로
    assert a[1, s0:e0].abs().sum().item() == 0 and a[1, :s0].sum().item() == s0   # keep=0: check 토큰만 0
    assert "chk_mask" in C.ARM_SPECS["EVCM_CHK"] and C.ARM_SPECS["EVCM_CHK"]["terms"] == ("corr", "format", "fclaim")


def test_evca_arm_registered_as_evcm_superset():
    """§14 R4 EVCA: EVCM 과 항·데이터·프롬프트 바이트 동일, chk_mask 값만 "amplify"."""
    evcm, evca = C.ARM_SPECS["EVCM_CHK"], C.ARM_SPECS["EVCA_CHK"]
    for k in ("terms", "meta_form", "require_meta", "data_hint", "prompt_variant"):
        assert evcm[k] == evca[k]
    assert evcm["chk_mask"] is True and evca["chk_mask"] == "amplify"


def test_evca_amplifies_solved_keeps_overclaim_zeroes_neutral():
    """§14 R4: keep 배열이 배율(float)일 때 — 0 은 지움(EVCM 과 동일), 1 은 그대로,
    >1(CHK_AMP) 은 그 구간 어드밴티지를 배율만큼 키운다(새 보너스가 아니라 기존 크레딧 증폭)."""
    import torch, re
    from src.training import verl_sdc as V
    class _Tok:
        def decode(self, ids, skip_special_tokens=False): return "".join(chr(i) for i in ids)
    texts = ["ab<check>x</check>cd", "ab<check>y</check>cd", "ab<check>z</check>cd"]
    ids = torch.tensor([[ord(ch) for ch in t] for t in texts])
    class _D: pass
    d = _D(); d.batch = {"advantages": torch.ones(3, ids.shape[1]), "responses": ids,
                         "response_mask": torch.ones(3, ids.shape[1], dtype=torch.long)}
    spans = [[(m.start(), m.end()) for m in re.finditer(r"<check>.*?</check>", t)] for t in texts]
    # row0: over_claim(1.0, 그대로) · row1: 중립(0.0, 지움) · row2: solved(2.0, 증폭)
    V._CHK_REGION_STASH.update({"step": 1, "bs": 3, "uid": ["g", "g", "g"], "meta": [0.0, 0.0, 0.0],
                                "spans": spans, "keep": [1.0, 0.0, 2.0], "mask_on": True})
    out = V._countdown_add_check_region_advantage(d, tokenizer=_Tok())
    a = out.batch["advantages"]
    s1, e1 = spans[1][0]; s2, e2 = spans[2][0]
    assert a[0].sum().item() == ids.shape[1]                              # keep=1.0: 그대로
    assert a[1, s1:e1].abs().sum().item() == 0                            # keep=0.0: check 토큰 0
    assert torch.allclose(a[2, s2:e2], torch.full_like(a[2, s2:e2], 2.0))  # keep=2.0: 원래 1.0 이 2.0 으로
    assert a[2, :s2].sum().item() == s2                                  # check 밖 토큰은 안 건드림


def test_evcas_arm_registered_and_symmetric_amplifies_negative_side():
    """§14 R4b EVCAS: EVCM/EVCA 와 항·데이터·프롬프트 바이트 동일, chk_mask="amplify_sym".
    solved 는 CHK_AMP 배, over_claim 은 CHK_AMP_NEG 배로 기존 부호를 증폭(새 벌점 아님)."""
    evcm, evcas = C.ARM_SPECS["EVCM_CHK"], C.ARM_SPECS["EVCAS_CHK"]
    for k in ("terms", "meta_form", "require_meta", "data_hint", "prompt_variant"):
        assert evcm[k] == evcas[k]
    assert evcas["chk_mask"] == "amplify_sym"

    import torch, re
    from src.training import verl_sdc as V
    class _Tok:
        def decode(self, ids, skip_special_tokens=False): return "".join(chr(i) for i in ids)
    texts = ["ab<check>x</check>cd"]
    ids = torch.tensor([[ord(ch) for ch in t] for t in texts])
    class _D: pass
    # 음수 어드밴티지(over_claim 은 대개 틀린 롤아웃) 위에서 증폭이 부호를 지키며 키우는지 확인.
    d = _D(); d.batch = {"advantages": torch.full((1, ids.shape[1]), -1.0), "responses": ids,
                         "response_mask": torch.ones(1, ids.shape[1], dtype=torch.long)}
    spans = [[(m.start(), m.end()) for m in re.finditer(r"<check>.*?</check>", t)] for t in texts]
    V._CHK_REGION_STASH.update({"step": 1, "bs": 1, "uid": ["g"], "meta": [0.0],
                                "spans": spans, "keep": [2.0], "mask_on": True})
    out = V._countdown_add_check_region_advantage(d, tokenizer=_Tok())
    a = out.batch["advantages"]; s0, e0 = spans[0][0]
    assert torch.allclose(a[0, s0:e0], torch.full_like(a[0, s0:e0], -2.0))  # -1.0 이 -2.0 으로(부호 유지, 크기만 증폭)


def test_scale_and_additive_bonus_compose_on_same_row():
    """§14 R4c 결합판을 위한 봉인 테스트(0911 수리): chk_mask 배율과 그룹 중심화 보너스(c)가
    같은 행에서 **둘 다** 적용돼야 한다 — 예전엔 배율이 걸리면 `continue` 로 보너스 덧셈이
    조용히 스킵됐다(지금까지 어떤 발사된 팔도 이 조합을 안 써서 관측된 적은 없다)."""
    import torch, re
    from src.training import verl_sdc as V
    class _Tok:
        def decode(self, ids, skip_special_tokens=False): return "".join(chr(i) for i in ids)
    # 그룹 중심화(group_mean_subtract)는 같은 그룹 안에서 평균을 빼므로, 행이 하나뿐이면
    # c 가 항상 0 이 된다 — 2행짜리 그룹(메타 1.0/0.0)으로 c=0.5 를 만든다.
    texts = ["ab<check>x</check>cd", "abcdefghijklmnopqrst"]
    ids = torch.tensor([[ord(ch) for ch in t] for t in texts])
    class _D: pass
    d = _D(); d.batch = {"advantages": torch.full((2, ids.shape[1]), 1.0), "responses": ids,
                         "response_mask": torch.ones(2, ids.shape[1], dtype=torch.long)}
    spans = [[(m.start(), m.end()) for m in re.finditer(r"<check>.*?</check>", texts[0])], []]
    # row0: scale=2.0(EVCA 증폭) 와 c=+0.5(예: chk_fixed 그룹 중심화 보너스) 가 동시에 걸린다.
    V._CHK_REGION_STASH.update({"step": 1, "bs": 2, "uid": ["g", "g"], "meta": [1.0, 0.0],
                                "spans": spans, "keep": [2.0, 1.0], "mask_on": True})
    out = V._countdown_add_check_region_advantage(d, tokenizer=_Tok())
    a = out.batch["advantages"]; s0, e0 = spans[0][0]
    # 1.0 * 2.0(배율) + 0.5(보너스) = 2.5 — 배율만(2.0) 도 보너스만(1.5) 도 아니어야 한다.
    assert torch.allclose(a[0, s0:e0], torch.full_like(a[0, s0:e0], 2.5))


def test_fixeda_arm_registered_as_fixed_plus_evca():
    """§14 R4c FIXEDA: FIXED_CHK(chk_fixed 보너스) + EVCA(chk_mask=amplify) 결합."""
    fixed, evca, hybrid = C.ARM_SPECS["FIXED_CHK"], C.ARM_SPECS["EVCA_CHK"], C.ARM_SPECS["FIXEDA_CHK"]
    assert hybrid["terms"] == fixed["terms"] == ("corr", "format", "fclaim", "chk_fixed")
    assert hybrid["chk_mask"] == evca["chk_mask"] == "amplify"
    for k in ("meta_form", "require_meta", "data_hint", "prompt_variant"):
        assert hybrid[k] == fixed[k] == evca[k]


def test_check_row_honest_flag_true_only_when_final_expr_is_self_flagged_bad():
    """§14 R4d NOSURR 재료: 최종 박스 식을 스스로 ✗ 로 표시하고 그대로 제출했을 때만
    honest_flag=1. over_claim(플래그 없이 틀림)과는 배타적이어야 한다."""
    nums, target = [5, 18, 9, 7], 94
    # 틀린 최종식을 스스로 ✗ 로 표시하고 그대로 박스 — honest_flag=1, over_claim=0.
    text_honest = ("try (18*5)+9-7 <check> (18*5)+9-7 = 92 ✗ </check> "
                   "\\boxed{(18*5)+9-7}")
    row = C.check_row(text_honest, nums, target, r_corr=0)
    assert row["honest_flag"] == 1 and row["over_claim"] == 0
    # 검산 없이(또는 ✓로) 틀린 답 제출 — over_claim=1, honest_flag=0.
    text_overclaim = "\\boxed{(18*5)+9-7}"
    row2 = C.check_row(text_overclaim, nums, target, r_corr=0)
    assert row2["over_claim"] == 1 and row2["honest_flag"] == 0
    # 정답이면 honest_flag=0(정의상 r_corr=0 조건 있음).
    row3 = C.check_row(text_honest, nums, target, r_corr=1)
    assert row3["honest_flag"] == 0


def test_nosurr_arm_registered():
    """§14 R4d NOSURR: EVCAS와 형제(EVCM과 항 동일, chk_mask="amplify_nosurr")."""
    evcm, nosurr = C.ARM_SPECS["EVCM_CHK"], C.ARM_SPECS["NOSURR_CHK"]
    assert nosurr["terms"] == evcm["terms"] == ("corr", "format", "fclaim")
    assert nosurr["chk_mask"] == "amplify_nosurr"


def test_mark_honest_surrender_early_group_relative():
    """§14 R4d: honest_flag 행이 같은 그룹 동료의 최대 길이 대비 60% 미만이면
    honest_surrender_early=1. 길게 계속 시도하다 정직하게 포기한 행(80%)은 0."""
    from src.training.verl_sdc import _mark_honest_surrender_early
    rows = [
        {"honest_flag": 1, "_resp_char_len": 100},   # 그룹 최댓값(1000)의 10% — 조기 포기
        {"honest_flag": 0, "_resp_char_len": 1000},  # 최댓값 자신(다른 이유로 틀림, honest_flag 없음)
        {"honest_flag": 1, "_resp_char_len": 850},   # 최댓값의 85% — 충분히 시도함, 조기 포기 아님
    ]
    groups = {"g0": [0, 1, 2]}
    _mark_honest_surrender_early(rows, groups, frac=0.6)
    assert rows[0]["honest_surrender_early"] == 1
    assert rows[1]["honest_surrender_early"] == 0   # honest_flag 자체가 0
    assert rows[2]["honest_surrender_early"] == 0   # 충분히 길게 시도함


def test_amplify_nosurr_scale_covers_overclaim_and_early_surrender():
    """§14 R4d: chk_mask="amplify_nosurr" 배율 선택 로직 — chk_solved 는 CHK_AMP,
    over_claim 과 honest_surrender_early 는 둘 다 CHK_AMP_NEG, 나머지는 0.0."""
    import importlib
    V = importlib.import_module("src.training.verl_sdc")
    import src.training.countdown_rewards as _cdr
    rows = [
        {"chk_solved": 1, "over_claim": 0, "honest_surrender_early": 0},
        {"chk_solved": 0, "over_claim": 1, "honest_surrender_early": 0},
        {"chk_solved": 0, "over_claim": 0, "honest_surrender_early": 1},
        {"chk_solved": 0, "over_claim": 0, "honest_surrender_early": 0},
    ]
    _chk_mask_spec = "amplify_nosurr"
    _amp = float(_cdr.CHK_AMP)
    _amp_neg = float(_cdr.CHK_AMP_NEG)
    _keep = []
    for r in rows:
        if V._bool01_local(r.get("chk_solved", 0)):
            _keep.append(_amp)
        elif V._bool01_local(r.get("over_claim", 0)):
            _keep.append(_amp_neg)
        elif _chk_mask_spec == "amplify_nosurr" and V._bool01_local(r.get("honest_surrender_early", 0)):
            _keep.append(_amp_neg)
        else:
            _keep.append(0.0)
    assert _keep == [_amp, _amp_neg, _amp_neg, 0.0]


def test_lenbonus_arm_registered_without_chk_content_terms():
    """§18 R4e LENBONUS_CHK: chk_fixed/chk_evc 등 check-내용 항은 전혀 안 들어가고
    len_bonus만 대신 들어간다 — FIXED_CHK 대비 confound 대조군."""
    fixed, lenbonus = C.ARM_SPECS["FIXED_CHK"], C.ARM_SPECS["LENBONUS_CHK"]
    assert "chk_fixed" in fixed["terms"] and "chk_fixed" not in lenbonus["terms"]
    assert "len_bonus" in lenbonus["terms"]
    assert lenbonus["prompt_variant"] == "chk"
    assert lenbonus.get("chk_mask") is None or "chk_mask" not in lenbonus


def test_lenbonus_raw_term_is_length_threshold_not_check_content():
    """len_bonus는 <check> 내용과 무관하게 응답 문자 길이만 본다."""
    short_row = {"r_corr": 1, "format_ok": 1, "fclaim": 0, "_resp_char_len": C.LEN_BONUS_CHARS - 1}
    long_row = {"r_corr": 1, "format_ok": 1, "fclaim": 0, "_resp_char_len": C.LEN_BONUS_CHARS}
    _, comp_short = C.arm_reward("LENBONUS_CHK", short_row, step=100, warmup_steps=20)
    _, comp_long = C.arm_reward("LENBONUS_CHK", long_row, step=100, warmup_steps=20)
    assert comp_short["len_bonus"] == 0.0
    assert comp_long["len_bonus"] > 0.0


def test_lenbonus_row_with_check_tags_but_short_gets_no_bonus():
    """check 태그를 정확히 썼어도 짧으면(< LEN_BONUS_CHARS) len_bonus는 0 —
    check '내용'이 아니라 순수 길이만 본다는 것을 확인."""
    row = {"r_corr": 1, "format_ok": 1, "fclaim": 0,
           "_resp_char_len": min(50, C.LEN_BONUS_CHARS - 1)}
    _, comp = C.arm_reward("LENBONUS_CHK", row, step=100, warmup_steps=20)
    assert comp["len_bonus"] == 0.0
