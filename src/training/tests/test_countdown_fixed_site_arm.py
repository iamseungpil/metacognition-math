r"""고정 자리 학습 배선(FT/M0/MT) 회귀 테스트 — CPU 전용.

무엇을 검산하는가:
  1. `r_timing` 진리표(6 가지 + family_dead=None).
  2. `r_live_new` — 새 수 안/밖 × followed 0/1.
  3. `arm_reward("MT", row)` 손계산 합계.
  4. `arm_signature` 에 `timing@0.5*w+live_new@1*w` 가 그대로 박히는가(G8 근거).
  5. "정상 행에서 첫 메타 앞 텍스트로 오라클을 다시 돌리면 그 프리픽스를 직접
     넣은 것과 같은 라벨이 나온다" — `verl_sdc._compute_countdown_arm_stash` 가
     쓰는 접근(파싱으로 프리픽스를 잘라낸 뒤 `countdown_sites.oracle_for_site` 를
     부른다)을 이 테스트가 **같은 두 줄로 재현**한다(verl 미설치 환경이라 그 함수
     자체는 못 부른다 — 아래 `_runtime_family_dead_and_live_new` 가 그 두 줄과
     바이트 동일해야 한다는 게 이 테스트의 계약이다).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

from src.training import countdown_rewards as cr           # noqa: E402
from src.training import countdown_sites as cs             # noqa: E402

NUMS = [3, 5, 7, 9]
TARGET = 24  # 오라클(§enumerate_solutions, 이 파일 헤더 실측): 3+5/3+7/3+9/5+7/5+9/7+9/3*7/5*9 는
             # 해로 이어지고(count>0), 3*9/9-3/9/3/5-3/7-3/9-5/7*9/9-7 등은 죽었다(count 0).


# ══════════════════════════════════════════════════════════════════════════════
# 1. r_timing — 6 가지 + None
# ══════════════════════════════════════════════════════════════════════════════

def _row(family_dead, dec_redirect, dec_verify):
    return {"family_dead": family_dead, "dec_redirect": dec_redirect, "dec_verify": dec_verify}


def test_r_timing_family_dead_none_is_zero_regardless_of_decision():
    for dr, dv in [(1, 0), (0, 1), (0, 0)]:
        assert cr.r_timing(_row(None, dr, dv)) == 0.0


def test_r_timing_truth_table_six_cases():
    # fd=1(계열 죽음)
    assert cr.r_timing(_row(1, 1, 0)) == 1.0    # redirect — 옳은 타이밍
    assert cr.r_timing(_row(1, 0, 1)) == -1.0   # verify — 안 갈아탐
    assert cr.r_timing(_row(1, 0, 0)) == -1.0   # 무결정 — 안 갈아탐
    # fd=0(계열 생존)
    assert cr.r_timing(_row(0, 1, 0)) == -1.0   # redirect — 성급한 포기
    assert cr.r_timing(_row(0, 0, 1)) == 0.0    # verify — 정상
    assert cr.r_timing(_row(0, 0, 0)) == 0.0    # 무결정 — 정상


# ══════════════════════════════════════════════════════════════════════════════
# 2. r_live_new — in/not-in × followed 0/1
# ══════════════════════════════════════════════════════════════════════════════

def _lrow(fm, live, followed):
    return {"first_move_after_meta": fm, "live_new_moves": live, "followed": followed}


def test_r_live_new_in_live_and_followed_is_one():
    assert cr.r_live_new(_lrow("3+7", ["3+7", "5+9"], 1)) == 1.0


def test_r_live_new_in_live_but_not_followed_is_zero():
    assert cr.r_live_new(_lrow("3+7", ["3+7", "5+9"], 0)) == 0.0


def test_r_live_new_not_in_live_and_followed_is_zero():
    assert cr.r_live_new(_lrow("9-3", ["3+7", "5+9"], 1)) == 0.0


def test_r_live_new_not_in_live_and_not_followed_is_zero():
    assert cr.r_live_new(_lrow("9-3", ["3+7", "5+9"], 0)) == 0.0


def test_r_live_new_no_move_found_is_zero():
    assert cr.r_live_new(_lrow(None, ["3+7"], 1)) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. arm_reward("MT", row) — 손계산
# ══════════════════════════════════════════════════════════════════════════════

def test_arm_reward_mt_hand_computed():
    row = {
        "r_corr": 1, "format_ok": 1, "emitted": 1,
        "family_dead": 1, "dec_redirect": 1, "dec_verify": 0,
        "first_move_after_meta": "3+7", "live_new_moves": ["3+7"], "followed": 1,
    }
    total, comps = cr.arm_reward("MT", row, step=100)  # step>=20 → warmup scale 1.0
    # corr(1.0)*1 + format(1.0)*0.35 + meta_floor(1.0)*0.02 + timing(+1)*0.5 + live_new(1)*1.0
    assert comps["corr"] == 1.0
    assert comps["format"] == 0.35
    assert comps["meta_floor"] == 0.02
    assert comps["timing"] == 0.5
    assert comps["live_new"] == 1.0
    assert abs(total - (1.0 + 0.35 + 0.02 + 0.5 + 1.0)) < 1e-9


def test_arm_reward_mt_hand_computed_negative_timing_zero_live_new():
    row = {
        "r_corr": 0, "format_ok": 0, "emitted": 1,
        "family_dead": 0, "dec_redirect": 1, "dec_verify": 0,   # 성급한 포기 → -1
        "first_move_after_meta": "9-3", "live_new_moves": ["3+7"], "followed": 1,  # 죽은 수 → 0
    }
    total, comps = cr.arm_reward("MT", row, step=100)
    assert comps["corr"] == 0.0
    assert comps["format"] == 0.0
    assert comps["meta_floor"] == 0.02
    assert comps["timing"] == -0.5     # r_timing=-1 * weight 0.5
    assert comps["live_new"] == 0.0
    assert abs(total - (0.0 + 0.0 + 0.02 - 0.5 + 0.0)) < 1e-9


def test_arm_reward_mt_not_emitted_zeroes_timing_and_live_new():
    row = {
        "r_corr": 1, "format_ok": 1, "emitted": 0,
        "family_dead": 1, "dec_redirect": 1, "dec_verify": 0,
        "first_move_after_meta": "3+7", "live_new_moves": ["3+7"], "followed": 1,
    }
    total, comps = cr.arm_reward("MT", row, step=100)
    assert comps["timing"] == 0.0
    assert comps["live_new"] == 0.0
    assert comps["meta_floor"] == 0.0   # 발화 안 했으니 바닥값도 0


# ══════════════════════════════════════════════════════════════════════════════
# 4. arm_signature — G8 근거
# ══════════════════════════════════════════════════════════════════════════════

def test_ft_m0_mt_signatures():
    ft, m0, mt = cr.arm_signature("FT"), cr.arm_signature("M0"), cr.arm_signature("MT")
    assert "timing@0.5*w+live_new@1*w" in ft
    assert "timing@0.5*w+live_new@1*w" in mt
    assert "timing" not in m0 and "live_new" not in m0
    # M0 의 항은 A 와 동일(_COMMON) — 데이터만 다르다는 것이 노트에 남아 있어야 한다.
    assert tuple(cr.ARM_SPECS["M0"]["terms"]) == tuple(cr.ARM_SPECS["A"]["terms"])
    assert cr.ARM_SPECS["FT"]["data_hint"] == "normal"
    assert cr.ARM_SPECS["M0"]["data_hint"] == "mixed"
    assert cr.ARM_SPECS["MT"]["data_hint"] == "mixed"


# ══════════════════════════════════════════════════════════════════════════════
# 5. 런타임 family_dead/live_new_moves — verl_sdc 가 쓰는 두 줄과 바이트 동일해야 한다
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_normal_response(pre_meta_body: str, meta: str, after: str) -> str:
    return f"{pre_meta_body}\n<meta>\n{meta}\n</meta>\n{after}"


def _runtime_family_dead_and_live_new(full_response_text: str, meta_form: str, nums, target):
    """`verl_sdc._compute_countdown_arm_stash` 의 FT/M0/MT 분기와 **같은 두 줄**
    (parse_meta 로 첫 메타 앞을 자르고 `oracle_for_site` 호출)을 재현한다 —
    verl 미설치 환경이라 그 함수 자체를 못 부르므로, 이 헬퍼가 그 로직의
    회귀 앵커다. verl_sdc.py 를 고칠 때 이 두 줄도 같이 고쳐야 한다.
    """
    m = cr.parse_meta(full_response_text, meta_form)
    resp_pre_meta = full_response_text[: int(m["start"])] if m.get("start") is not None else full_response_text
    oracle = cs.oracle_for_site(resp_pre_meta, nums, target)
    return oracle["family_dead"], oracle["live_new_moves"]


def test_runtime_family_dead_matches_direct_oracle_call_normal_row():
    # 마지막 두 시도 둘 다 죽은 수(9-3, 3*9) → family_dead=1.
    pre_meta = "9-3=6 3*9=27"
    text = _synthetic_normal_response(pre_meta, "confidence: 0.9\ndecision: redirect", "3+7=10\n\\boxed{24}")
    fd_runtime, live_runtime = _runtime_family_dead_and_live_new(text, "new", NUMS, TARGET)
    fd_direct = cs.family_dead_label(pre_meta, NUMS, cs.enumerate_solutions(NUMS, TARGET)[0])
    oracle_direct = cs.oracle_for_site(pre_meta, NUMS, TARGET)
    assert fd_runtime == fd_direct == 1
    assert live_runtime == oracle_direct["live_new_moves"]


def test_runtime_family_dead_matches_direct_oracle_call_alive_case():
    # 마지막 두 시도: 하나는 죽고(9-3) 하나는 산다(3+7) → family_dead=0.
    pre_meta = "9-3=6 3+7=10"
    text = _synthetic_normal_response(pre_meta, "confidence: 0.5\ndecision: verify", "\\boxed{24}")
    fd_runtime, live_runtime = _runtime_family_dead_and_live_new(text, "new", NUMS, TARGET)
    oracle_direct = cs.oracle_for_site(pre_meta, NUMS, TARGET)
    assert fd_runtime == oracle_direct["family_dead"] == 0
    assert live_runtime == oracle_direct["live_new_moves"]


def test_runtime_no_meta_uses_whole_text_as_prefix():
    # <meta> 가 없으면 parse_meta().start 는 None — 프리픽스 = 응답 전체.
    text = "9-3=6 3*9=27 no meta here"
    fd_runtime, _ = _runtime_family_dead_and_live_new(text, "new", NUMS, TARGET)
    oracle_direct = cs.oracle_for_site(text, NUMS, TARGET)
    assert fd_runtime == oracle_direct["family_dead"]
