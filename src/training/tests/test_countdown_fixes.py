"""0904 감사 — Countdown 경로 결함 10건 수리에 대한 회귀 테스트.

원 감사가 지목한 파일 셋(countdown_task.py / countdown_rewards.py / verl_sdc.py)
+ countdown_pmi.py / scripts/steer_prompts.py / configs/countdown_6arm.yaml 을 건드렸다.
GPU·ray·실 트레이너 없이 CPU 에서 도는 것만 직접 함수호출로 검사하고, 배선이
`verl_sdc.py` 소스 문자열 자체에 남아 있어야 하는 자리(무거운 트레이너 상태를 필요로
해 여기서 직접 실행할 수 없는 자리)는 텍스트 검사로 못박는다 — 이 파일의 다른 테스트
(`test_osd_wiring.py::test_osd_term_name_is_single_sourced` 등)도 같은 방식을 쓴다.
"""
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

from src.training import countdown_pmi as cp                     # noqa: E402
from src.training import countdown_rewards as cr                 # noqa: E402
from src.training import countdown_task as ct                    # noqa: E402

_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_VERL_SDC_SRC = (_REPO_ROOT / "src" / "training" / "verl_sdc.py").read_text()


# ══════════════════════════════════════════════════════════════════════════════
# 결함 1 — grade 가 grade_from_gt/_solvable/프롬프트와 같은 규칙(중간값 양의 정수)을
# 지키는가.
# ══════════════════════════════════════════════════════════════════════════════

def test_grade_rejects_negative_intermediate():
    """(2-5)*(1-4) — 두 뺄셈이 전부 중간에 음수를 거친다. 최종값은 9(=(-3)*(-3))로
    맞지만 Countdown 규칙 위반이므로 0점이어야 한다."""
    assert ct.grade(r"\boxed{(2-5)*(1-4)}", [2, 5, 1, 4], 9) == 0


def test_grade_rejects_non_integer_intermediate():
    """((1/3)*5)*15 — 최종값은 정확히 25 지만 (1/3) 이 중간에 비정수다."""
    assert ct.grade(r"\boxed{((1/3)*5)*15}", [1, 3, 5, 15], 25) == 0


def test_grade_rejects_unary_minus():
    """-3+28 — 단항 부호는 `_fold_countdown` 이 지원하지 않는 노드다(파싱 자체가
    Countdown 규칙 밖)."""
    assert ct.grade(r"\boxed{-3+28}", [3, 28], 25) == 0


def test_grade_accepts_valid_expression():
    """중간값이 전부 양의 정수인 정상 식은 여전히 1점이어야 한다(회귀 방지)."""
    assert ct.grade(r"\boxed{3+4}", [3, 4], 7) == 1
    assert ct.grade(r"\boxed{(3+4)*2}", [3, 4, 2], 14) == 1


def test_grade_still_rejects_wrong_value_and_wrong_multiset():
    """새 규칙이 기존 ⓐⓑⓒ 검사를 깨지 않았는지 — 값이 틀리거나 수가 안 맞으면 0."""
    assert ct.grade(r"\boxed{3+4}", [3, 4], 8) == 0            # 값 불일치
    assert ct.grade(r"\boxed{3+4}", [3, 4, 5], 7) == 0          # 다중집합 불일치


def test_grade_from_gt_uses_the_same_rule():
    gt = ct.make_ground_truth([2, 5, 1, 4], 9)
    assert ct.grade_from_gt(r"\boxed{(2-5)*(1-4)}", gt) == 0


# ══════════════════════════════════════════════════════════════════════════════
# 결함 2 — parse_ok 가 암묵적 곱셈(Call)·FloorDiv 를 거부하는가.
# ══════════════════════════════════════════════════════════════════════════════

def test_parse_ok_rejects_implicit_multiplication():
    """`2(3+4)` 는 파이썬 문법으로는 함수호출(Call)로 파싱된다 — 산술식이 아니다."""
    assert ct.parse_ok(r"\boxed{2(3+4)}") == 0


def test_parse_ok_rejects_floordiv():
    """`8//3+1` 은 Countdown 이 쓰지 않는 FloorDiv 연산자다."""
    assert ct.parse_ok(r"\boxed{8//3+1}") == 0


def test_parse_ok_rejects_unary_and_pow():
    assert ct.parse_ok(r"\boxed{-3+4}") == 0
    assert ct.parse_ok(r"\boxed{3**2}") == 0


def test_parse_ok_accepts_valid_expression():
    """회귀 방지 — 정상 식은 여전히 1이어야 한다."""
    assert ct.parse_ok(r"\boxed{(2+3)*4}") == 1
    assert ct.parse_ok(r"\boxed{(2-5)*(1-4)}") == 1  # parse_ok 는 값·중간값을 안 본다


# ══════════════════════════════════════════════════════════════════════════════
# 결함 3 — emitted 의 정의를 countdown_rewards.parse_meta 하나로 통일.
# ══════════════════════════════════════════════════════════════════════════════

def test_parse_meta_empty_block_is_not_emitted():
    """빈 <meta></meta> — PMI 스코어러의 emitted(스팬 유무)와 달리, 정본 정의는
    신뢰도·decision 이 둘 다 있어야 emitted=1 이다."""
    assert cr.parse_meta("<meta></meta>", "new")["emitted"] == 0


def test_parse_meta_full_block_is_emitted():
    text = "<meta>\nconfidence: 0.5\nsome judgement.\ndecision: verify\n</meta>"
    assert cr.parse_meta(text, "new")["emitted"] == 1


def test_parse_meta_none_form_is_never_emitted():
    """N0(=ARM_SPECS 의 meta_form 'none') 팔은 메타 형식 자체가 없다 —
    `_compute_countdown_arm_stash` 는 이 팔에서 emitted 를 무조건 0 으로 덮는다."""
    assert cr.ARM_SPECS["N0"]["meta_form"] == "none"


def test_emit_definition_overwrite_is_wired_in_verl_sdc():
    """★배선 검사(무거운 트레이너 없이는 `_compute_countdown_arm_stash` 를 통째로
    돌릴 수 없다 — 이 파일의 다른 테스트와 같은 이유로 소스에 직접 못박는다).
    PMI 쪽 emitted 를 그대로 두면 빈 <meta></meta> 도 발화로 세어진다(감사결함3).
    """
    assert '_cdr.parse_meta(r.get("text") or "", _meta_form)["emitted"]' in _VERL_SDC_SRC, (
        "emitted 재정의가 countdown_rewards.parse_meta 를 쓰지 않는다 — "
        "PMI 쪽 정의(빈 메타도 emitted=1)로 되돌아갔을 위험.")
    assert "EMIT-DEFN" in _VERL_SDC_SRC, "두 정의가 갈릴 때의 진단 로그가 없어졌다."


# ══════════════════════════════════════════════════════════════════════════════
# 결함 4 — plan_next 견고성.
# ══════════════════════════════════════════════════════════════════════════════

def _meta(next_line: str, after: str = "") -> str:
    return (f"<meta>\nconfidence: 0.5\nnext: {next_line}\nsome judgement.\n"
            f"decision: verify\n</meta>{after}")


def test_plan_next_good_and_followed():
    text = _meta("5+8", " Attempt: 5+8=13")
    assert cr.plan_next(text, [5, 8], 13) == (1, 1)


def test_plan_next_good_but_not_followed():
    """next 는 해를 살리지만(1), 메타 뒤 첫 시도가 다른 두 수를 쓴다(0)."""
    text = _meta("5+8", " Attempt: 10-2=8")
    assert cr.plan_next(text, [5, 8, 3], 16) == (1, 0)


def test_plan_next_bad_but_followed():
    """5-8 은 중간값이 음수라 목표에 못 닿는다(0). 메타 뒤에서 같은 두 수를 실제로
    다시 결합하면 followed 는 그래도 1 이다 — "이행했나"와 "그 계획이 좋았나"는
    별개의 질문이다."""
    text = _meta("5-8", " 5-8=-3, that's wrong, let's try something else.")
    assert cr.plan_next(text, [5, 8], 13) == (0, 1)


def test_plan_next_no_meta():
    assert cr.plan_next("just prose, no meta at all", [5, 8], 13) == (0, 0)


def test_plan_next_parenthesised_pair():
    text = _meta("(25+3)", " (25+3) first")
    assert cr.plan_next(text, [25, 3], 28) == (1, 1)


def test_plan_next_two_candidates_only_first_evaluated():
    """`next: 5+8 or 25-7` — 25 나 7 은 nums 에 없으므로, 코드가 두 번째 후보를
    잘못 골랐다면 `_apply_move` 가 즉시 실패해 ok=0 이 된다. ok=1 이 나오면 첫
    후보(5+8)만 읽었다는 뜻이다."""
    text = _meta("5+8 or 25-7", " 5+8=13")
    assert cr.plan_next(text, [5, 8], 13) == (1, 1)


def test_plan_next_prose_after_meta_does_not_count_as_followed():
    """메타 뒤에 등장하는 "3-4 ideas" 같은 산문이 우연히 정규식과 같은 모양이어도
    (여기서는 애초에 next 의 두 수와 다르므로) followed 로 잡히면 안 된다."""
    text = _meta("5+8", " Let's think of 3-4 ideas here, then verify 5+8=13")
    ok, followed = cr.plan_next(text, [5, 8], 13)
    assert followed == 0, "산문 속 우연한 숫자쌍이 이행으로 잘못 세어졌다"


def test_multiset_has_pair_respects_duplicate_counts():
    """★결함4 수리 — followed 판정에 다중집합 검사를 추가했다. 같은 값 두 개를
    묶으려면 그 값이 실제로 두 번 있어야 한다."""
    assert cr._multiset_has_pair([5, 5, 8], 5, 5) is True
    assert cr._multiset_has_pair([5, 8], 5, 5) is False
    assert cr._multiset_has_pair([5, 8], 5, 8) is True
    assert cr._multiset_has_pair([5, 8], 5, 9) is False


# ══════════════════════════════════════════════════════════════════════════════
# 결함 5 — 중단 임계값 오버라이드(COUNTDOWN_ABORT_ARITH / COUNTDOWN_ABORT_PATIENCE).
# ══════════════════════════════════════════════════════════════════════════════

def _abort_report(**over):
    rep = {"emit_rate": 0.5, "boilerplate": {"boilerplate_rate": 0.3},
           "answer_leak_rate": 0.05, "arith_in_meta_rate": 0.03,
           "false_claim_rate": 0.0, "confidence": {"mean": 0.5}}
    rep.update(over)
    return rep


@pytest.fixture(autouse=True)
def _clean_abort_env():
    for k in ("COUNTDOWN_ABORT_ARITH", "COUNTDOWN_ABORT_PATIENCE"):
        os.environ.pop(k, None)
    yield
    for k in ("COUNTDOWN_ABORT_ARITH", "COUNTDOWN_ABORT_PATIENCE"):
        os.environ.pop(k, None)


def test_abort_arith_threshold_default_unchanged():
    """오버라이드가 없으면 기존 0.02 그대로 — arith_in_meta_rate=0.03 은 위반."""
    hits = {v["metric"] for v in cr.check_abort(_abort_report(arith_in_meta_rate=0.03))
            if v["status"] == "abort"}
    assert "arith_in_meta_rate" in hits
    assert "abort_arith" not in cr.arm_signature("A")


def test_abort_arith_threshold_override_changes_verdict_and_signature():
    os.environ["COUNTDOWN_ABORT_ARITH"] = "0.05"
    # 0.03 은 이제 0.05 밑이므로 통과해야 한다.
    hits = {v["metric"] for v in cr.check_abort(_abort_report(arith_in_meta_rate=0.03))
            if v["status"] == "abort"}
    assert "arith_in_meta_rate" not in hits
    assert "abort_arith=0.05" in cr.arm_signature("A")


def test_abort_patience_default_and_override():
    assert cr.get_abort_patience() == 3
    os.environ["COUNTDOWN_ABORT_PATIENCE"] = "7"
    assert cr.get_abort_patience() == 7


def test_abort_arith_signature_is_conditional_on_arm_terms_not_override_alone():
    """오버라이드가 다른 팔의 서명도 똑같이 갱신하는지 — 조건부이되 팔에 따라
    빠지면 안 된다(무조건 로직이라 arm 과 무관해야 정상)."""
    os.environ["COUNTDOWN_ABORT_ARITH"] = "0.1"
    assert "abort_arith=0.1" in cr.arm_signature("B")
    assert "abort_arith=0.1" in cr.arm_signature("N0")


# ══════════════════════════════════════════════════════════════════════════════
# 결함 6 — 계산량 정합: PMI 항 없는 팔은 score_pmi_shift 를 건너뛰고, OSD 기본값이
# "측정 모드"로 항상 켜지지 않는다.
# ══════════════════════════════════════════════════════════════════════════════

def test_countdown_pmi_empty_rows_matches_score_pmi_shift_schema():
    rows = cp.empty_rows(3)
    assert len(rows) == 3
    for r in rows:
        assert set(r.keys()) == {"pmi_open", "pmi_close", "meta_n_tok",
                                 "emitted", "path", "meta_first", "scored"}
        assert math.isnan(r["pmi_open"]) and math.isnan(r["pmi_close"])
        assert r["emitted"] == 0 and r["scored"] is False


def test_countdown_pmi_empty_diag_has_the_keys_the_caller_reads():
    diag = cp.empty_diag(5)
    assert diag["B"] == 5
    assert diag["scored"] == 0
    assert diag["ref_error"] is None
    assert diag["attempted"] == 0


def test_pmi_skip_is_wired_for_non_pmi_arms():
    """★배선 검사. `_compute_countdown_arm_stash` 를 트레이너 없이 통째로 돌릴 수
    없으므로(다른 이 파일 테스트와 같은 사유) 소스에 직접 못박는다: PMI 항이 없는
    팔(A/N0/E/G/H/OSD/PL/R 등)은 `_cdp.empty_rows`/`empty_diag` 경로로 가야 한다."""
    assert "rows, diag = _cdp.empty_rows(bs), _cdp.empty_diag(bs)" in _VERL_SDC_SRC


def test_osd_default_is_off_not_measurement_mode():
    """★수리 전에는 `COUNTDOWN_OSD` 기본값이 "1"이라 PMI 항 없는 팔도 매 스텝
    OSD ref forward 를 물었다. 기본은 "0"이어야 한다(osd 항을 쓰는 팔은
    `_osd_terms` 로 여전히 무조건 켜진다 — 이 문자열 검사는 "기본값"만 잠근다)."""
    assert '_osd_on = bool(_osd_terms) or os.environ.get("COUNTDOWN_OSD", "0") == "1"' \
        in _VERL_SDC_SRC


def test_arm_specs_terms_partition_matches_pmi_skip_condition():
    """meta_pos/meta_mul/meta_ctx 어느 것도 없는 팔이 실제로 존재하는지 — 없으면
    위 배선 검사가 아무 팔도 커버하지 못하는 죽은 코드가 된다."""
    pmi_terms = {"meta_pos", "meta_mul", "meta_ctx"}
    no_pmi_arms = [a for a, spec in cr.ARM_SPECS.items()
                   if not (pmi_terms & set(spec["terms"]))
                   and "meta_pos_full" not in spec["terms"]]
    assert no_pmi_arms, "PMI 스킵 경로를 실제로 타는 팔이 하나도 없다"
    assert "A" in no_pmi_arms and "N0" in no_pmi_arms


# ══════════════════════════════════════════════════════════════════════════════
# 결함 7 — COUNTDOWN_6ARM 은 SDC 리전 마스크(레거시 수학 헤드가 읽는 자리)를
# 계산하지 않는다.
# ══════════════════════════════════════════════════════════════════════════════

def test_sdc_region_mask_bypass_is_wired_for_countdown_mode():
    """★배선 검사(무거운 이유는 위와 동일). COUNTDOWN_6ARM 은
    `self.reward_funcs == [countdown_arm_reward]` 하나뿐이라 `build_sdc_region_masks`
    의 결과를 아무도 안 읽는데, 예전엔 롤아웃마다 그 계산을 무조건 돌렸다."""
    assert "if _mode_sync == _COUNTDOWN_MODE:" in _VERL_SDC_SRC
    # ★두 번째 `build_sdc_region_masks(` 호출부(`MetaCotSDCRewardManager.__call__`
    #   의 동기 경로 — 첫 호출부는 별개의 async 폴백 함수다)가 우회 블록보다
    #   **뒤**에 있어야 한다(건너뛰기가 먼저 걸려야 한다).
    idx_bypass = _VERL_SDC_SRC.index("if _mode_sync == _COUNTDOWN_MODE:")
    idx_call = _VERL_SDC_SRC.index("masks = build_sdc_region_masks(", idx_bypass)
    assert idx_bypass < idx_call


def test_countdown_6arm_reward_config_has_exactly_one_head():
    """COUNTDOWN_6ARM 이 정말로 레거시 헤드를 REWARD_CONFIGS 에 안 갖고 있는지 —
    이게 참이어야 결함7의 "이미 구조적으로 우회된다"는 전제가 성립한다."""
    from src.training import verl_sdc as vs
    cfg = vs.REWARD_CONFIGS["COUNTDOWN_6ARM"]
    assert cfg["keys"] == ["countdown_arm"]
    assert len(cfg["funcs"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 결함 8 — 체크포인트 보관 개수.
# ══════════════════════════════════════════════════════════════════════════════

def test_countdown_6arm_config_limits_ckpt_retention():
    import yaml
    cfg_path = _REPO_ROOT / "configs" / "countdown_6arm.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["trainer"]["max_actor_ckpt_to_keep"] == 2
    assert cfg["trainer"]["max_critic_ckpt_to_keep"] == 2


# ══════════════════════════════════════════════════════════════════════════════
# 결함 9 — /scratch 하드코딩 제거(SDC_LOG_DIR).
# ══════════════════════════════════════════════════════════════════════════════

def test_faulthandler_uses_sdc_log_dir_override(tmp_path, monkeypatch):
    import faulthandler

    from src.training import verl_sdc as vs
    monkeypatch.setenv("SDC_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("DCPO_FAULTHANDLER_SEC", "9999")
    try:
        vs._dcpo_install_faulthandler("pytest_tag")
        assert (tmp_path / "faulthandler_pytest_tag.log").exists()
    finally:
        faulthandler.cancel_dump_traceback_later()


def test_faulthandler_default_dir_is_scratch_logs_when_unset(monkeypatch):
    """오버라이드하지 않으면 기존 동작(`/scratch/logs`)과 바이트 동일해야 한다 —
    이 값 자체가 바뀌면 기존 amlt 로그 수집 경로가 깨진다."""
    monkeypatch.delenv("SDC_LOG_DIR", raising=False)
    monkeypatch.delenv("DCPO_FAULTHANDLER_SEC", raising=False)
    assert 'os.environ.get("SDC_LOG_DIR", "/scratch/logs")' in _VERL_SDC_SRC


# ══════════════════════════════════════════════════════════════════════════════
# 결함 10 — P3 프롬프트가 countdown_task.py 의 1급 변형으로 등록됐는가.
# ══════════════════════════════════════════════════════════════════════════════

def _old_construction_p3() -> str:
    """`scripts/steer_prompts.py` 가 이번 수리 **전에** P3 를 조립하던 그대로의
    공식을, 그 파일이 바뀌어도 흔들리지 않도록 이 테스트 안에 스냅샷으로 굳힌다."""
    import re

    ban = ("★Do NOT do arithmetic in here — no expressions, no equalities, "
           "no combining of numbers, no candidate answer. "
           "Assess the approach; do not solve the puzzle.")
    ban_narrow = ("★Do NOT write a complete expression that uses ALL the given numbers — "
                  "that is the answer and it does not belong here. "
                  "Partial groupings you have ruled out or intend to try next are fine.")
    old_mandate = ("Write `decision: verify` when the confidence you just wrote is high and the "
                   "current line of search deserves to be pushed through and checked. Write "
                   "`decision: redirect` when that confidence is low and the current line of "
                   "search should be abandoned for a different family of groupings. The decision "
                   "must follow from the confidence.")
    new_mandate = ("Write `decision: redirect` when `next` names a grouping from a different "
                   "family than the one you have been exploring; write `decision: verify` when "
                   "`next` says you will check the current line. The decision must follow from "
                   "`ruled_out` and `next`. Confidence reports how likely the current family is "
                   "to succeed; it does not dictate the decision.")
    struct = """<meta>
confidence: <a single number between 0 and 1>
ruled_out: <the specific partial groupings you have already tried and eliminated, \
comma-separated, e.g. `25*3, (25+3)*7`. Write `none` only if you have tried nothing yet. \
List only groupings that actually appear in your work above and that actually failed.>
next: <the specific partial grouping you will try next, e.g. `8*7 first`. It must be one you \
have NOT already combined above, and it must use FEWER than all of the given numbers. \
If you are going to verify instead of changing course, write what you will check.>
<One sentence judging YOUR OWN APPROACH: which family of groupings you are exploring, and \
whether that family is worth continuing. ★Do NOT write a complete expression that uses ALL \
the given numbers — that is the answer and it does not belong here.>
decision: verify
</meta>"""
    block_re = re.compile(r"<meta>\nconfidence:.*?\n</meta>", re.DOTALL)
    ex_head = "\n\nExample of the block, for numbers [25, 3, 7, 8] and target 68:\n\n"
    ex_judge = ("The multiply-25-first family overshoots badly and I keep having to subtract "
                "back, so it is not worth continuing.")

    p0 = ct.SOLVE_SYS_NEW
    assert ban in p0 and old_mandate in p0
    p1 = p0.replace(ban, ban_narrow)
    p2_body, n = block_re.subn(struct, p1, count=1)
    assert n == 1
    p2 = p2_body.replace(old_mandate, new_mandate)
    ex_p2 = ex_head + ("<meta>\nconfidence: 0.3\nruled_out: 25*3, (25+3)*7\nnext: 8*7 first\n"
                       + ex_judge + "\ndecision: redirect\n</meta>")
    return p2 + ex_p2


def test_p3_prompt_variant_is_byte_identical_to_old_construction():
    assert ct.PROMPT_VARIANTS["p3"] == _old_construction_p3()


def test_build_p3_prompt_function_matches_registered_variant():
    assert ct.build_p3_prompt() == ct.PROMPT_VARIANTS["p3"]


def test_build_parquet_accepts_p3_variant(tmp_path):
    out = tmp_path / "cd_p3.parquet"
    info = ct.build_parquet(3, seed=0, out_path=out, variant="p3", n_nums=4)
    assert info["variant"] == "p3"
    assert info["rows"] == 3


def test_steer_prompts_p3_is_imported_back_from_countdown_task():
    """steer_prompts.py 가 자기 안에서 P3 를 다시 조립하지 않고 countdown_task 의
    결과를 되돌려 받는지 — 소스 문자열로 못박는다(무거운 REPO 환경변수 sys.path
    조작 때문에 이 스크립트를 직접 import 하는 것은 다른 테스트에 부작용을 준다)."""
    src = (_REPO_ROOT / "scripts" / "steer_prompts.py").read_text()
    assert 'P3 = PROMPT_VARIANTS["p3"]' in src
    assert "P3 = P2 + EX_P2" not in src
