"""countdown_rewards 단위 테스트 — GPU·torch·verl 없이 CPU 로 돈다.

무엇을 지키는 테스트인가(원장에 있는 실제 사고들):
  · **경계값**: 0/1/±clip/sign=0/warmup 양끝. 원장 0730 "판정 밴드 문턱도 대장 검사 대상".
  · **팔 정체 diff(G8)**: ARM_SPECS 가 사양의 여덟 팔과 글자 그대로 일치하는가.
  · **재사용이 진짜인가**: r_meta_pos 가 dcpo_pmi_shift 와, r_meta_ctx 가
    dcpo_rmeta_forms 와 **수치적으로 동일**한가. 복제가 슬며시 생기면 여기서 깨진다.
  · **무효 레버 방지**: 켜진 항의 원재료가 없으면 조용히 0 이 아니라 예외인가.
  · **gold 불필요**: 행에 gold 를 심어도 p̂ 가 안 바뀌는가.

실행:  python -m pytest src/training/tests/test_countdown_rewards.py -q
   또는 python src/training/tests/test_countdown_rewards.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

from src.training import countdown_rewards as cr
from src.training.dcpo_pmi_shift import pmi_shift_reward
from src.training.dcpo_rmeta_forms import FORMULAS


# ══════════════════════════════════════════════════════════ 0. 의존성 규율

def test_no_torch_or_verl_import():
    """순수 함수 규율 — 소스에 torch/verl import 가 없어야 CPU 테스트가 성립한다."""
    src = Path(cr.__file__).read_text()
    for bad in ("import torch", "from torch", "import verl", "from verl"):
        assert bad not in src, f"{bad} 가 들어왔다 — CPU 테스트가 깨진다"
    assert "torch" not in sys.modules or True  # 다른 테스트가 로드했을 수 있으므로 정적검사만


# ══════════════════════════════════════════════════════════ 1. p̂

def _rows(flags):
    return [{"r_corr": f} for f in flags]


def test_phat_basic_fractions():
    assert cr.compute_phat(_rows([0, 0, 0, 0])) == 0.0
    assert cr.compute_phat(_rows([1, 1, 1, 1])) == 1.0
    assert cr.compute_phat(_rows([1, 0, 0, 0])) == 0.25
    assert cr.compute_phat(_rows([1, 1, 0, 0])) == 0.5


def test_phat_ignores_gold_keys():
    """★gold 불필요 — 행에 gold 계열 키를 포이즌해도 p̂ 는 한 톨도 안 바뀐다."""
    clean = _rows([1, 0, 0, 0])
    dirty = [dict(r) for r in clean]
    for r in dirty:
        for k in cr.GOLD_KEYS:
            r[k] = "(3+7)*8"          # 포이즌: 정답을 통째로 심는다
    assert cr.compute_phat(dirty) == cr.compute_phat(clean) == 0.25


def test_phat_with_injected_grader_uses_target_not_gold():
    """채점기 주입 경로 — target 은 프롬프트에 있으므로 gold 없이 채점된다."""
    def grader(row):                      # countdown.grade 자리
        return int(eval(row["expr"]) == row["target"])   # noqa: S307
    rows = [{"expr": "(3+7)*8", "target": 80, "gold": "무시돼야 한다"},
            {"expr": "(3+7)*7", "target": 80}]
    assert cr.compute_phat(rows, grader=grader) == 0.5


def test_phat_empty_group_raises_not_zero():
    """빈 그룹에 0.0 을 주면 r_gate 가 +1(발화 장려)로 읽혀 배선 버그가 학습 신호가 된다."""
    try:
        cr.compute_phat([])
        assert False, "예외가 나야 한다"
    except ValueError:
        pass


def test_phat_missing_key_raises():
    try:
        cr.compute_phat([{"text": "x"}])
        assert False, "예외가 나야 한다"
    except KeyError:
        pass


def test_phat_loo_removes_self():
    loo = cr.compute_phat_loo(_rows([1, 0, 0, 0]))
    assert loo[0] == 0.0                       # 자기(정답)를 빼면 나머지 3개 전부 오답
    assert all(abs(v - 1 / 3) < 1e-12 for v in loo[1:])
    assert abs(cr.compute_phat(_rows([1, 0, 0, 0])) - 0.25) < 1e-12   # 사양 경로는 그대로


# ══════════════════════════════════════════════════════════ 2. r_gate

def test_r_gate_boundaries():
    assert cr.r_gate(1, 0.0) == 1.0      # 전원 실패 → 발화에 +1
    assert cr.r_gate(1, 1.0) == -1.0     # 전원 성공 → 발화에 −1
    assert cr.r_gate(1, 0.5) == 0.0      # 중립
    assert cr.r_gate(1, 0.25) == 0.5
    assert cr.r_gate(1, 0.75) == -0.5


def test_r_gate_zero_when_silent():
    for p in (0.0, 0.5, 1.0):
        assert cr.r_gate(0, p) == 0.0


def test_r_gate_nan_phat_fails_closed():
    assert cr.r_gate(1, float("nan")) == 0.0
    assert cr.r_gate(1, None) == 0.0


def test_r_gate_out_of_range_raises():
    for bad in (-0.1, 1.1, 8.0):
        try:
            cr.r_gate(1, bad)
            assert False, f"p̂={bad} 는 예외여야 한다"
        except ValueError:
            pass


# ══════════════════════════════════════════════════════════ 3. r_meta_pos (재사용 확인)

def test_r_meta_pos_is_exactly_dcpo_pmi_shift():
    """★복제 금지 검사 — 값이 갈리면 여기서 즉시 깨진다."""
    for o in (-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0):
        for c in (-3.0, -1.0, -0.2, 0.0, 0.2, 1.0, 3.0):
            assert cr.r_meta_pos(o, c) == pmi_shift_reward(o, c, **cr.SHIFT_PARAMS)


def test_r_meta_pos_save_and_derail_asymmetry():
    save = cr.r_meta_pos(-0.5, 0.5)      # 오답쪽 → 정답쪽
    derail = cr.r_meta_pos(0.5, -0.5)    # 정답쪽 → 오답쪽
    assert abs(save - (1.0 + 1.0)) < 1e-9        # clip(+1.0) + save 1.0
    assert abs(derail - (-1.0 - 2.0)) < 1e-9     # clip(−1.0) − derail 2.0
    assert abs(derail) > abs(save)               # derail ≥ save (사양의 비대칭)


def test_r_meta_pos_clip_at_two():
    assert abs(cr.r_meta_pos(-10.0, 10.0) - (2.0 + 1.0)) < 1e-9    # clip ±2 + save
    assert abs(cr.r_meta_pos(10.0, -10.0) - (-2.0 - 2.0)) < 1e-9


def test_r_meta_pos_no_reversal_at_exact_zero():
    """0 을 정확히 스치는 것은 뒤집기가 아니다(연속항만)."""
    assert cr.r_meta_pos(-1.0, 0.0) == -0.0 + (0.0 - (-1.0))   # 연속항 +1.0, 보너스 0
    assert cr.r_meta_pos(0.0, 1.0) == 1.0


def test_r_meta_pos_nan_fails_closed():
    assert cr.r_meta_pos(float("nan"), 1.0) == 0.0
    assert cr.r_meta_pos(1.0, float("inf")) == 0.0


# ══════════════════════════════════════════════════════════ 4. r_meta_mul (sign 처리)

def test_r_meta_mul_multiplies_sign():
    base = cr.r_meta_pos(-0.5, 0.5)
    assert cr.r_meta_mul(-0.5, 0.5, 3.7) == base * 1.0
    assert cr.r_meta_mul(-0.5, 0.5, -0.001) == base * -1.0


def test_r_meta_mul_sign_zero_is_zero():
    """★명시적 결정: 방향이 없으면 신용도 없다(0.0). +1 로 두면 C 가 그 그룹에서 B 가 된다."""
    assert cr.r_meta_mul(-0.5, 0.5, 0.0) == 0.0
    assert cr.r_meta_mul(-0.5, 0.5, -0.0) == 0.0
    assert cr.SIGN_ZERO_VALUE == 0.0


def test_r_meta_mul_sign_nan_fails_closed():
    assert cr.r_meta_mul(-0.5, 0.5, float("nan")) == 0.0
    assert cr.r_meta_mul(-0.5, 0.5, None) == 0.0


def test_r_meta_mul_uses_mul_params_so_spec_conflict_is_one_edit():
    """사양 충돌(표는 reversal 없음)을 함수가 아니라 상수로 뒤집을 수 있어야 한다."""
    save, derail = cr.SHIFT_PARAMS_MUL["reversal_save"], cr.SHIFT_PARAMS_MUL["reversal_derail"]
    assert (save, derail) == (1.0, 2.0)          # 현재 채택: 지시(항목 4) = reversal 포함
    no_rev = dict(cr.SHIFT_PARAMS_MUL, reversal_save=0.0, reversal_derail=0.0)
    assert cr.r_meta_mul(-0.5, 0.5, 1.0, params=no_rev) == 1.0   # 연속항만


# ══════════════════════════════════════════════════════════ 5. r_meta_ctx (재사용 확인)

def test_r_meta_ctx_is_exactly_rmeta_forms_ans_clip():
    """★clip 정의를 dcpo_rmeta_forms 와 묶어 둔다 — 오프라인 스윕과 온라인 학습이 같은 식."""
    for a in (-4.0, -2.0, -0.3, 0.0, 0.3, 2.0, 4.0):
        for b in (-4.0, -0.3, 0.0, 0.3, 4.0):
            want = FORMULAS["ans_clip"]({"pmi_ans_real": a, "pmi_ans_donor": b})
            assert cr.r_meta_ctx(a, b, 1.0) == want
            assert cr.r_meta_ctx(a, b, -1.0) == -want


def test_r_meta_ctx_clip_and_sign_zero():
    assert cr.r_meta_ctx(10.0, -10.0, 1.0) == cr.CTX_CLIP        # +2 로 잘린다
    assert cr.r_meta_ctx(-10.0, 10.0, 1.0) == -cr.CTX_CLIP
    assert cr.r_meta_ctx(10.0, -10.0, 0.0) == 0.0                # C 와 같은 sign=0 규칙
    assert cr.r_meta_ctx(float("nan"), 1.0, 1.0) == 0.0


# ══════════════════════════════════════════════════════════ 6. r_len / floor / warmup

def test_r_len():
    assert cr.r_len(0) == 0.0
    assert cr.r_len(100) == 1.0
    assert cr.r_len(250) == 2.5
    assert cr.r_len(float("nan")) == 0.0


def test_r_meta_floor():
    assert cr.r_meta_floor(1) == 1.0 and cr.r_meta_floor(0) == 0.0


def test_warmup_scale_boundaries():
    assert cr.warmup_scale(0, 20) == 0.0
    assert cr.warmup_scale(10, 20) == 0.5
    assert cr.warmup_scale(20, 20) == 1.0
    assert cr.warmup_scale(150, 20) == 1.0       # 이후 상시
    assert cr.warmup_scale(-5, 20) == 0.0
    assert cr.warmup_scale(1, 20) == 0.05
    assert cr.warmup_scale(0, 0) == 1.0          # 워밍업 없음
    assert cr.warmup_scale(0, -1) == 1.0


def test_warmup_scale_is_monotone():
    vals = [cr.warmup_scale(s, 20) for s in range(0, 25)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


# ══════════════════════════════════════════════════════════ 7. ARM_SPECS = 사양 (G8 근거)

SPEC_TABLE = {   # 사양 §보상 을 손으로 옮긴 것. 코드가 아니라 **사양**이 원본이다.
    "A": (("corr", "format", "meta_floor"), "new", "corr"),
    "B": (("corr", "format", "meta_floor", "meta_pos"), "new", "cur"),
    "C": (("corr", "format", "meta_floor", "meta_mul"), "new", "mul"),
    "D": (("corr", "format", "meta_floor", "meta_ctx"), "new", "ctx"),
    "E": (("corr", "format", "meta_floor", "gate"), "new", "gate"),
    "F": (("corr", "format", "meta_floor", "meta_mul", "gate"), "new", "full"),
    "G": (("corr", "format", "meta_floor", "len"), "new", "neg"),
    "H": (("corr", "format", "meta_floor", "meta_mul", "gate"), "old", "oldfmt"),
}


def test_arm_specs_match_spec_table():
    assert sorted(cr.ARM_SPECS) == list("ABCDEFGH")
    for arm, (terms, form, label) in SPEC_TABLE.items():
        got = cr.ARM_SPECS[arm]
        assert tuple(got["terms"]) == terms, f"{arm} 항 집합 불일치"
        assert got["meta_form"] == form, f"{arm} 메타 형식 불일치"
        assert got["label"] == label


def test_common_terms_are_identical_across_all_arms():
    """공통(처치 아님)이 팔마다 다르면 그 실험은 팔 비교가 아니다."""
    for arm in cr.ARM_SPECS:
        for t in ("corr", "format", "meta_floor"):
            assert t in cr.ARM_SPECS[arm]["terms"]
    assert cr.TERMS["corr"]["weight"] == 1.0
    assert cr.TERMS["format"]["weight"] == 0.35
    assert cr.TERMS["meta_floor"]["weight"] == 0.02


def test_H_is_F_reward_with_old_form_only():
    """H 의 존재 이유 = 보상은 같고 형식만 다르다. 보상이 갈리면 형식 효과가 아니다."""
    assert tuple(cr.ARM_SPECS["H"]["terms"]) == tuple(cr.ARM_SPECS["F"]["terms"])
    assert cr.ARM_SPECS["H"]["meta_form"] != cr.ARM_SPECS["F"]["meta_form"]


def test_F_is_C_plus_E():
    c = set(cr.ARM_SPECS["C"]["terms"])
    e = set(cr.ARM_SPECS["E"]["terms"])
    assert set(cr.ARM_SPECS["F"]["terms"]) == c | e


def test_warmup_applies_to_meta_and_gate_only():
    warmed = {t for t, cfg in cr.TERMS.items() if cfg["warmup"]}
    assert warmed == {"meta_pos", "meta_mul", "meta_ctx", "gate", "len"}
    for t in ("corr", "format", "meta_floor"):
        assert not cr.TERMS[t]["warmup"]


def test_arm_signatures_are_all_distinct():
    """★G8: 여덟 서명이 전부 달라야 팔 diff 가 성립한다."""
    sigs = cr.all_arm_signatures()
    assert len(set(sigs.values())) == 8, sigs
    assert all(cr.SPEC_VERSION in s for s in sigs.values())
    assert "form=old" in sigs["H"] and "form=new" in sigs["F"]
    assert sigs["F"] != sigs["H"]          # 보상은 같아도 형식이 다르면 다른 팔


# ══════════════════════════════════════════════════════════ 8. arm_reward 조립

def _row(**kw):
    base = dict(r_corr=1, format_ok=1, emitted=1, pmi_open=-0.5, pmi_close=0.5,
                pmi_self=1.0, pmi_donor=0.0, adv_corr=0.8, meta_n_tok=50)
    base.update(kw)
    return base


def test_arm_A_has_no_meta_term():
    total, comps = cr.arm_reward("A", _row(), step=999)
    assert set(comps) == {"corr", "format", "meta_floor"}
    assert abs(total - (1.0 + 0.35 + 0.02)) < 1e-9


def test_arm_B_adds_shift_after_warmup():
    total, comps = cr.arm_reward("B", _row(), step=999)
    assert abs(comps["meta_pos"] - cr.W_META * cr.r_meta_pos(-0.5, 0.5)) < 1e-9
    assert abs(total - (1.0 + 0.35 + 0.02 + 2.0)) < 1e-9


def test_warmup_zeroes_meta_terms_at_step_zero_but_not_corr():
    total, comps = cr.arm_reward("B", _row(), step=0, warmup_steps=20)
    assert comps["meta_pos"] == 0.0
    assert comps["corr"] == 1.0 and comps["format"] == 0.35 and comps["meta_floor"] == 0.02
    half, comps_h = cr.arm_reward("B", _row(), step=10, warmup_steps=20)
    assert abs(comps_h["meta_pos"] - 0.5 * comps_meta_full()) < 1e-9


def comps_meta_full():
    _, c = cr.arm_reward("B", _row(), step=20, warmup_steps=20)
    return c["meta_pos"]


def test_arm_C_silent_when_group_tied():
    """★sign=0 그룹에서 C 의 메타 항이 침묵한다 — 결정의 대가가 눈에 보여야 한다."""
    _, comps = cr.arm_reward("C", _row(adv_corr=0.0), step=999)
    assert comps["meta_mul"] == 0.0
    _, comps2 = cr.arm_reward("C", _row(adv_corr=-1.0), step=999)
    assert comps2["meta_mul"] < 0.0


def test_arm_E_gate_requires_phat():
    try:
        cr.arm_reward("E", _row(), step=999)
        assert False, "phat 없이 gate 팔이 돌면 안 된다"
    except ValueError:
        pass
    _, comps = cr.arm_reward("E", _row(), step=999, phat=0.0)
    assert comps["gate"] == cr.W_GATE * 1.0


def test_arm_F_is_sum_of_C_and_E_components():
    row, phat = _row(), 0.0
    _, cf = cr.arm_reward("F", row, step=999, phat=phat)
    _, cc = cr.arm_reward("C", row, step=999)
    _, ce = cr.arm_reward("E", row, step=999, phat=phat)
    assert abs(cf["meta_mul"] - cc["meta_mul"]) < 1e-12
    assert abs(cf["gate"] - ce["gate"]) < 1e-12


def test_arm_G_len_term():
    _, comps = cr.arm_reward("G", _row(meta_n_tok=250), step=999)
    assert comps["len"] == cr.W_LEN * 2.5


def test_meta_terms_are_zero_when_not_emitted():
    for arm in ("B", "C", "D", "G"):
        _, comps = cr.arm_reward(arm, _row(emitted=0), step=999, phat=0.0)
        metas = [v for k, v in comps.items() if k not in ("corr", "format")]
        assert all(v == 0.0 for v in metas), (arm, comps)


def test_missing_ingredient_raises_not_silently_zero():
    """★무효 레버 방지 — 켜진 항의 원재료가 없으면 즉사."""
    bad = _row(); bad.pop("pmi_close")
    try:
        cr.arm_reward("B", bad, step=999)
        assert False, "KeyError 가 나야 한다"
    except KeyError as e:
        assert "meta_pos:pmi_close" in str(e)
    # 반면 그 항을 안 켜는 팔에서는 같은 행이 정상 통과한다
    cr.arm_reward("A", bad, step=999)


def test_unknown_arm_raises():
    try:
        cr.arm_reward("Z", _row(), step=1)
        assert False
    except ValueError:
        pass


def test_nonfinite_pmi_does_not_poison_total():
    total, comps = cr.arm_reward("B", _row(pmi_close=float("nan")), step=999)
    assert math.isfinite(total) and comps["meta_pos"] == 0.0


# ══════════════════════════════════════════════════════════ 9. 메타 파싱

NEW_META = ("Let me think.\n<meta>\nconfidence: 0.7\n"
            "The pairing I keep trying reuses 25 too early; the approach should be "
            "reconsidered rather than refined.\ndecision: redirect\n</meta>\n"
            "So I will regroup. \\boxed{(3+7)*8}")
OLD_META = ("Working.\nconfidence: 0.6 | The current grouping is unlikely to reach the "
            "target. | decision: verify\nTherefore \\boxed{(3+7)*8}")


def test_parse_new_form():
    m = cr.parse_meta(NEW_META, form="new")
    assert m["emitted"] == 1 and m["form"] == "new" and m["n_blocks"] == 1
    assert m["confidence"] == 0.7 and m["decision"] == "redirect"
    assert "confidence" not in m["body"] and "decision" not in m["body"]
    assert m["start"] == NEW_META.index("<meta>")


def test_parse_old_form():
    m = cr.parse_meta(OLD_META, form="old")
    assert m["emitted"] == 1 and m["form"] == "old"
    assert m["confidence"] == 0.6 and m["decision"] == "verify"
    assert m["body"].startswith("The current grouping")


def test_form_mismatch_is_not_emitted():
    """H 의 형식으로 새 형식을 내면 발화로 안 센다 — 형식 효과를 재려면 이게 엄격해야 한다."""
    assert cr.parse_meta(NEW_META, form="old")["emitted"] == 0
    assert cr.parse_meta(OLD_META, form="new")["emitted"] == 0
    assert cr.meta_form_ok(NEW_META, "old") == 0
    assert cr.meta_form_ok(NEW_META, "new") == 1


def test_incomplete_new_block_not_emitted():
    txt = "<meta>\nconfidence: 0.5\nsome thought\n</meta>"     # decision 없음
    m = cr.parse_meta(txt, form="new")
    assert m["emitted"] == 0 and m["n_blocks"] == 1


def test_two_blocks_fail_form_check():
    txt = NEW_META + "\n" + NEW_META
    assert cr.parse_meta(txt, form="new")["n_blocks"] == 2
    assert cr.meta_form_ok(txt, "new") == 0


def test_confidence_is_not_clamped():
    """0 과 1 을 구별해야 confidence 고유값 수가 의미를 갖는다(기존 파서는 클램프한다)."""
    z = cr.parse_meta("<meta>\nconfidence: 0\nx\ndecision: verify\n</meta>", form="new")
    o = cr.parse_meta("<meta>\nconfidence: 1\nx\ndecision: verify\n</meta>", form="new")
    assert z["confidence"] == 0.0 and o["confidence"] == 1.0


def test_format_ok_row_combines_expr_and_meta_form():
    ok = lambda t: 1            # noqa: E731  — 식은 항상 통과시키는 스텁
    no = lambda t: 0            # noqa: E731
    assert cr.format_ok_row(NEW_META, "F", parse_expr_ok=ok) == 1
    assert cr.format_ok_row(NEW_META, "H", parse_expr_ok=ok) == 0   # H 는 옛 형식
    assert cr.format_ok_row(OLD_META, "H", parse_expr_ok=ok) == 1
    assert cr.format_ok_row(NEW_META, "F", parse_expr_ok=no) == 0   # 식 형식 불통과


# ══════════════════════════════════════════════════════════ 10. 텔레메트리

def _trow(text, phat, **kw):
    r = {"text": text, "phat": phat, "final_expr": "(3+7)*8", "n_tok": 100,
         "truncated": 0, "r_corr": 0, "pmi_open": -0.5, "pmi_close": 0.5}
    r.update(kw)
    return r


def test_emit_rate():
    rows = [_trow(NEW_META, 0.0), _trow("no meta here", 0.0)]
    assert cr.emit_rate(rows, form="new") == 0.5


def test_meta_position_char_and_token():
    r = _trow(NEW_META, 0.0)
    pos = cr.meta_position_frac(r, form="new")
    assert 0.0 < pos < 0.5
    r2 = _trow(NEW_META, 0.0, meta_start_tok=5, n_tok=100)
    assert cr.meta_position_frac(r2, form="new") == 0.05


def test_meta_position_stats_flags_meta_first():
    first = "<meta>\nconfidence: 0.9\nx\ndecision: verify\n</meta>" + " tail" * 200
    rows = [_trow(first, 0.0) for _ in range(4)]
    st = cr.meta_position_stats(rows, form="new")
    assert st["frac_meta_first"] == 1.0 and st["n"] == 4


def test_selectivity_index():
    hard = [_trow(NEW_META, 0.0) for _ in range(4)]            # 어려움 → 전부 발화
    easy = [_trow("plain", 1.0) for _ in range(4)]             # 쉬움 → 전부 침묵
    s = cr.selectivity_index(hard + easy, form="new")
    assert s["selectivity"] == 1.0 and s["emit_hard"] == 1.0 and s["emit_easy"] == 0.0
    flat = cr.selectivity_index([_trow(NEW_META, 0.0), _trow(NEW_META, 1.0)], form="new")
    assert flat["selectivity"] == 0.0                          # 무차별 발화


def test_selectivity_nan_when_one_side_empty():
    s = cr.selectivity_index([_trow(NEW_META, 0.0)], form="new")
    assert math.isnan(s["selectivity"])                        # 0 으로 채우면 '무차별'로 오독


def test_boilerplate_rate_ignores_confidence_digits():
    a = "<meta>\nconfidence: 0.88\nSame stock sentence.\ndecision: verify\n</meta>"
    b = "<meta>\nconfidence: 0.42\nSame stock sentence.\ndecision: verify\n</meta>"
    c = "<meta>\nconfidence: 0.5\nA genuinely different remark about the pairing.\n" \
        "decision: redirect\n</meta>"
    rep = cr.boilerplate_rate([_trow(a, 0.0), _trow(b, 0.0), _trow(c, 0.0)], form="new")
    assert abs(rep["boilerplate_rate"] - 2 / 3) < 1e-9 and rep["n_unique"] == 2


def test_answer_leak_detects_verbatim_expression():
    leaky = ("<meta>\nconfidence: 0.9\nThe answer is (3+7)*8 so just write it.\n"
             "decision: verify\n</meta>")
    clean = ("<meta>\nconfidence: 0.4\nThe pairing strategy so far is too greedy.\n"
             "decision: redirect\n</meta>")
    assert cr.answer_leak(leaky, "(3+7)*8") == 1
    assert cr.answer_leak(leaky, "\\boxed{(3+7)*8}") == 1        # 정규화가 장식을 벗긴다
    assert cr.answer_leak(clean, "(3+7)*8") == 0
    # 바깥 괄호만 다른 경우: 최종 식이 ((3+7)*8) 인데 메타는 괄호 없이 적었다
    assert cr.answer_leak(leaky, "((3+7)*8)") == 1
    rate = cr.answer_leak_rate([_trow(leaky, 0.0), _trow(clean, 0.0)], form="new")
    assert rate == 0.5


def test_answer_leak_requires_final_expr():
    try:
        cr.answer_leak("anything", None)
        assert False, "None 이면 예외 — 조용한 0 은 중단조건을 무력화한다"
    except ValueError:
        pass


def test_meta_arithmetic_detection():
    assert cr.meta_has_arithmetic("try 24 + 2 = 26 next") == 1
    assert cr.meta_has_arithmetic("the approach is too greedy") == 0


def test_group_emit_dispersion():
    g_all = [_trow(NEW_META, 0.0) for _ in range(4)]
    g_none = [_trow("plain", 0.0) for _ in range(4)]
    g_mixed = [_trow(NEW_META, 0.0), _trow("plain", 0.0)]
    d = cr.group_emit_dispersion([g_all, g_none, g_mixed], form="new")
    assert abs(d["frac_all_emit"] - 1 / 3) < 1e-9
    assert abs(d["frac_all_silent"] - 1 / 3) < 1e-9
    assert abs(d["frac_mixed"] - 1 / 3) < 1e-9


def test_phat_distribution_reports_sign_zero_fraction():
    groups = [_rows([0, 0, 0, 0]), _rows([1, 1, 1, 1]), _rows([1, 0, 0, 0])]
    d = cr.phat_distribution(groups)
    assert d["frac_zero"] == 1 / 3 and d["frac_one"] == 1 / 3
    assert abs(d["frac_sign_zero"] - 2 / 3) < 1e-9      # ★곱셈 팔이 침묵하는 비율


def test_decision_and_confidence_distributions():
    rows = [_trow(NEW_META, 0.0), _trow(NEW_META.replace("redirect", "verify"), 0.0)]
    d = cr.decision_distribution(rows, form="new")
    assert d["verify"] == 0.5 and d["redirect"] == 0.5
    c = cr.confidence_unique(rows, form="new")
    assert c["n_unique"] == 1 and c["values"] == [0.7]


def test_component_means_flags_dead_and_constant_levers():
    comps = [{"corr": 1.0, "meta_pos": 0.0, "format": 0.35},
             {"corr": 0.0, "meta_pos": 0.0, "format": 0.35}]
    m = cr.component_means(comps)
    assert "meta_pos" in m["dead"]         # 평균 0 = 무효 레버
    assert "format" in m["dead"]           # 분산 0 = 센터링으로 사라지는 상수 레버
    assert "corr" not in m["dead"]


def test_shift_diag_counts_match_reward_criterion():
    rows = [{"pmi_open": -1.0, "pmi_close": 1.0},     # save
            {"pmi_open": 1.0, "pmi_close": -1.0},     # derail
            {"pmi_open": 0.1, "pmi_close": 0.2},      # 없음
            {"pmi_open": float("nan"), "pmi_close": 0.0}]
    d = cr.shift_diag(rows)
    assert d["n_save"] == 1 and d["n_derail"] == 1 and d["n_fail"] == 1


def test_length_stats():
    rows = [_trow("x", 0.0, n_tok=v, truncated=(v > 900)) for v in (100, 200, 1000)]
    st = cr.length_stats(rows)
    assert st["len_p50"] == 200 and abs(st["trunc_rate"] - 1 / 3) < 1e-9


def test_telemetry_report_has_every_spec_metric():
    groups = [[_trow(NEW_META, 0.0), _trow("plain", 0.0)]]
    rep = cr.telemetry_report(groups, form="new")
    for k in ("emit_rate", "meta_position", "selectivity", "boilerplate",
              "answer_leak_rate", "group_emit", "phat", "decision", "confidence",
              "length", "shift", "arith_in_meta_rate"):
        assert k in rep, f"사양의 지표 {k} 가 리포트에 없다"


# ══════════════════════════════════════════════════════════ 11. 중단 조건

def test_check_abort_thresholds():
    ok = {"emit_rate": 0.5, "boilerplate": {"boilerplate_rate": 0.3},
          "answer_leak_rate": 0.05}
    assert cr.check_abort(ok) == []
    bad = {"emit_rate": 0.19, "boilerplate": {"boilerplate_rate": 0.51},
           "answer_leak_rate": 0.11}
    got = {v["metric"] for v in cr.check_abort(bad)}
    assert got == {"emit_rate", "boilerplate_rate", "answer_leak_rate"}


def test_check_abort_boundaries_are_strict():
    """문턱 위/아래가 아니라 **정확히 문턱**일 때 통과인지 — 판정 밴드도 검사 대상이다."""
    edge = {"emit_rate": 0.2, "boilerplate": {"boilerplate_rate": 0.5},
            "answer_leak_rate": 0.1}
    assert cr.check_abort(edge) == []      # < 0.2 / > 0.5 / > 0.1 이므로 등호는 통과


def test_check_abort_reports_missing_metric_as_missing_not_pass():
    """★안 잰 칸은 통과가 아니다 — 원장 0731 '상시 WARN 은 소음이 아니다'."""
    got = cr.check_abort({"emit_rate": 0.5})
    missing = {v["metric"] for v in got if v["status"] == "missing"}
    assert missing == {"boilerplate_rate", "answer_leak_rate"}


def test_check_abort_nan_is_missing():
    got = cr.check_abort({"emit_rate": float("nan"),
                          "boilerplate": {"boilerplate_rate": 0.1},
                          "answer_leak_rate": 0.0})
    assert got and got[0]["status"] == "missing"


def test_negative_control_discards_all_when_G_wins():
    scores = {"A": 0.30, "B": 0.31, "C": 0.33, "D": 0.32, "E": 0.34, "F": 0.35, "G": 0.36}
    v = cr.check_negative_control(scores)
    assert v["status"] == "DISCARD_ALL" and set(v["beaten"]) == set(cr.LAUNCHED_TREATMENT_ARMS)


def test_launched_arms_are_six_and_exclude_D_H():
    """★이번 판이 발사하는 팔은 여섯이다. D(문맥대조)·H(옛형식)는 ARM_SPECS 에
    정의가 남아 있지만 돌지 않는다 — 정의가 있다는 이유로 "돌았다"고 읽히면 안 된다."""
    assert cr.LAUNCHED_ARMS == ("A", "B", "C", "E", "F", "G")
    assert "D" not in cr.LAUNCHED_ARMS and "H" not in cr.LAUNCHED_ARMS
    assert set(cr.LAUNCHED_ARMS) <= set(cr.ARM_SPECS)
    assert cr.SPEC_VERSION.startswith("countdown-6arm")   # 서명이 판을 정직하게 가리킨다
    # 폐기 규칙이 실제로 발동 가능해야 한다 (기본값에 안 도는 팔이 끼면 영원히 incomplete)
    assert set(cr.LAUNCHED_TREATMENT_ARMS) <= set(cr.LAUNCHED_ARMS)


def test_negative_control_ok_when_G_loses_all():
    scores = {"B": 0.31, "C": 0.33, "D": 0.32, "E": 0.34, "F": 0.35, "G": 0.20}
    assert cr.check_negative_control(scores)["status"] == "ok"


def test_negative_control_incomplete_is_not_pass():
    v = cr.check_negative_control({"G": 0.2, "B": 0.3})
    assert v["status"] == "incomplete" and set(v["missing"]) == set(cr.LAUNCHED_TREATMENT_ARMS) - {"B"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
