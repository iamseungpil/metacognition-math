r"""`scripts/local/build_coupling_sft.py` 순수 함수 회귀 테스트 — CPU 전용, 모델·
parquet I/O 없이 딕셔너리로만 검증한다.

무엇을 지키는가:
  · `row_passes_filters` — 필터 6(+1)단계가 스펙 순서대로 걸리는가, 특히
    `require_redirect_for_dead` 는 family_dead==1 인 행에만 추가 조건을 건다
    (family_dead==0/None 행은 손대지 않는다).
  · `cap_per_key` — 그룹당 상한, 입력 순서 보존, `max_n<=0` 은 무제한.
  · `build_sft_row` — messages 는 힌트 없는 [system,user,assistant(=full_text)],
    wrong_prefix=prefix, scenario="redirect" — 그리고 `full_text` 가 site 의
    prefix 로 시작하지 않으면(다른 모드가 섞였다) 즉사한다.
  · `summarize_stages` — 누적 kept 카운트가 손으로 계산한 값과 맞는가.

실행:  python -m pytest tests/test_build_coupling_sft.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.local.build_coupling_sft import (  # noqa: E402
    build_sft_row, cap_per_key, compute_site_gain_ok, compute_site_success_rate,
    continuation_mentions_hint, meta_before_solve_ok, meta_text_is_template,
    row_passes_filters, summarize_stages,
)

# 0907 감사(세 오염 수리) 이후 기본(clean) full_text — 힌트 언급 없음, 메타는 실제
# 판단 문장(템플릿 아님), </meta> 뒤에 시도 등식이 있고 그 앞에는 "이미 풀렸다" 신호가
# 없다. 세 새 필터가 기본 True 로 켜져도 기존(pre-0907) 스테이지 테스트들이 여전히
# 통과하도록 이 문자열을 모든 기존 테스트의 기본값으로 쓴다.
_CLEAN_FULL_TEXT = ("PREFIX. 5+3=8, too low.\n<meta>\nconfidence: 0.4\nThe additive family "
                    "keeps landing short, so it may not be worth continuing.\n"
                    "decision: verify\n</meta>\n9-1=8\n\\boxed{9-1}")


def _cont_row(site_id="s1", *, r_corr=1, emitted=1, novel=1, followed=1, truncated=0,
             decision="redirect", mode="hint", k_index=0, full_text=_CLEAN_FULL_TEXT,
             n_tokens=10, prefix="PREFIX. ", target=None):
    return {
        "site_id": site_id, "mode": mode, "k_index": k_index, "r_corr": r_corr,
        "emitted": emitted, "novel": novel, "followed": followed, "truncated": truncated,
        "decision": decision, "full_text": full_text, "hint_text": "Hint: ...",
        "n_tokens": n_tokens, "prefix": prefix, "target": target,
    }


def _site_row(prefix="PREFIX. ", family_dead=1, nums=(1, 2, 3, 4), target=10):
    prompt = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Numbers: [1, 2, 3, 4]\nTarget: 10"},
        {"role": "assistant", "content": prefix},
    ]
    return {"prompt": prompt, "prefix": prefix, "family_dead": family_dead,
            "nums": list(nums), "target": target, "cut_type": "own-meta"}


# ══════════════════════════════════════════════════════════════════════════════
# row_passes_filters
# ══════════════════════════════════════════════════════════════════════════════

def test_all_filters_pass_on_clean_row():
    row = {**_cont_row(), "family_dead": 0}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=False)
    assert ok and stage == ""


@pytest.mark.parametrize("field,bad_value,expected_stage", [
    ("mode", "meta", "mode_hint"),
    ("r_corr", 0, "r_corr"),
    ("emitted", 0, "emitted"),
    ("novel", 0, "novel"),
    ("followed", 0, "followed"),
    ("truncated", 1, "not_truncated"),
])
def test_each_stage_rejects_its_own_field(field, bad_value, expected_stage):
    row = {**_cont_row(), field: bad_value, "family_dead": 0}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=False)
    assert ok is False
    assert stage == expected_stage


def test_redirect_for_dead_flag_off_ignores_decision():
    row = {**_cont_row(decision="verify"), "family_dead": 1}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=False)
    assert ok is True


def test_redirect_for_dead_flag_on_requires_redirect_when_family_dead():
    row = {**_cont_row(decision="verify"), "family_dead": 1}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=True)
    assert ok is False and stage == "redirect_for_dead"

    row_ok = {**_cont_row(decision="redirect"), "family_dead": 1}
    ok2, _ = row_passes_filters(row_ok, require_redirect_for_dead=True)
    assert ok2 is True


def test_redirect_for_dead_flag_on_does_not_touch_alive_family():
    row = {**_cont_row(decision="verify"), "family_dead": 0}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=True)
    assert ok is True


def test_redirect_for_dead_flag_on_does_not_touch_none_family():
    row = {**_cont_row(decision="verify"), "family_dead": None}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=True)
    assert ok is True


# ══════════════════════════════════════════════════════════════════════════════
# cap_per_key
# ══════════════════════════════════════════════════════════════════════════════

def test_cap_per_key_limits_each_group_and_keeps_order():
    rows = [{"site_id": "a", "i": 0}, {"site_id": "a", "i": 1}, {"site_id": "a", "i": 2},
            {"site_id": "b", "i": 0}]
    out = cap_per_key(rows, lambda r: r["site_id"], 2)
    assert [(r["site_id"], r["i"]) for r in out] == [("a", 0), ("a", 1), ("b", 0)]


def test_cap_per_key_zero_or_negative_means_unlimited():
    rows = [{"site_id": "a", "i": i} for i in range(5)]
    assert cap_per_key(rows, lambda r: r["site_id"], 0) == rows
    assert cap_per_key(rows, lambda r: r["site_id"], -1) == rows


# ══════════════════════════════════════════════════════════════════════════════
# build_sft_row
# ══════════════════════════════════════════════════════════════════════════════

def test_build_sft_row_messages_have_no_hint_and_full_text_as_assistant():
    cont = _cont_row(full_text="PREFIX. cont with a solution.")
    site = _site_row(prefix="PREFIX. ")
    row = build_sft_row(cont, site)
    assert row["messages"][0] == {"role": "system", "content": "SYS"}
    assert row["messages"][1]["content"] == "Numbers: [1, 2, 3, 4]\nTarget: 10"
    assert "Hint:" not in row["messages"][1]["content"]     # 힌트가 안 들어갔다
    assert row["messages"][-1] == {"role": "assistant",
                                   "content": "PREFIX. cont with a solution."}
    assert row["wrong_prefix"] == "PREFIX. "
    assert row["scenario"] == "redirect"


def test_build_sft_row_does_not_mutate_site_prompt():
    site = _site_row()
    original = [dict(m) for m in site["prompt"]]
    build_sft_row(_cont_row(full_text="PREFIX. x"), site)
    assert site["prompt"] == original


def test_build_sft_row_rejects_full_text_not_starting_with_prefix():
    cont = _cont_row(full_text="SOMETHING ELSE ENTIRELY")
    site = _site_row(prefix="PREFIX. ")
    with pytest.raises(ValueError):
        build_sft_row(cont, site)


# ══════════════════════════════════════════════════════════════════════════════
# summarize_stages
# ══════════════════════════════════════════════════════════════════════════════

def test_summarize_stages_cumulative_counts():
    rows = [
        {**_cont_row(site_id="s1"), "family_dead": 1},                     # passes all
        {**_cont_row(site_id="s2", r_corr=0), "family_dead": 1},            # dies at r_corr
        {**_cont_row(site_id="s3", novel=0), "family_dead": 0},             # dies at novel
        {**_cont_row(site_id="s4", decision="verify"), "family_dead": 1},   # dies only w/ flag
    ]
    summ = summarize_stages(rows, require_redirect_for_dead=False)
    assert summ["total_rows"] == 4
    assert summ["kept_by_stage"]["mode_hint"] == 4
    assert summ["kept_by_stage"]["r_corr"] == 3
    assert summ["kept_by_stage"]["novel"] == 2
    assert summ["final_kept"] == 2
    assert summ["kept_by_family_dead"] == {"1": 2}

    summ2 = summarize_stages(rows, require_redirect_for_dead=True)
    assert summ2["final_kept"] == 1                 # s4 also dies now
    assert "redirect_for_dead" in summ2["kept_by_stage"]


def test_summarize_stages_mean_n_tokens():
    rows = [
        {**_cont_row(site_id="s1", n_tokens=10), "family_dead": 0},
        {**_cont_row(site_id="s2", n_tokens=20), "family_dead": 0},
    ]
    summ = summarize_stages(rows, require_redirect_for_dead=False)
    assert summ["mean_n_tokens"] == 15.0


def test_summarize_stages_empty_survivors_mean_is_nan():
    import math
    rows = [{**_cont_row(r_corr=0), "family_dead": 0}]
    summ = summarize_stages(rows, require_redirect_for_dead=False)
    assert summ["final_kept"] == 0
    assert math.isnan(summ["mean_n_tokens"])


# ══════════════════════════════════════════════════════════════════════════════
# --no_require_novel / --no_require_followed
# ══════════════════════════════════════════════════════════════════════════════

def test_no_require_novel_lets_non_novel_rows_pass():
    row = {**_cont_row(novel=0), "family_dead": 0}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=False, require_novel=False)
    assert ok is True and stage == ""
    # default (require_novel=True) still rejects it
    ok2, stage2 = row_passes_filters(row, require_redirect_for_dead=False)
    assert ok2 is False and stage2 == "novel"


def test_no_require_followed_lets_non_followed_rows_pass():
    row = {**_cont_row(followed=0), "family_dead": 0}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=False, require_followed=False)
    assert ok is True and stage == ""
    ok2, stage2 = row_passes_filters(row, require_redirect_for_dead=False)
    assert ok2 is False and stage2 == "followed"


def test_summarize_stages_skips_novel_followed_checks_when_disabled():
    rows = [{**_cont_row(site_id="s1", novel=0, followed=0), "family_dead": 0}]
    summ = summarize_stages(rows, require_redirect_for_dead=False,
                             require_novel=False, require_followed=False)
    assert "novel" not in summ["kept_by_stage"]
    assert "followed" not in summ["kept_by_stage"]
    assert summ["final_kept"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# site-gain gate: compute_site_success_rate / compute_site_gain_ok / row+summary wiring
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_site_success_rate_averages_per_site():
    rows = [
        {"site_id": "a", "r_corr": 1}, {"site_id": "a", "r_corr": 0},
        {"site_id": "b", "r_corr": 1}, {"site_id": "b", "r_corr": 1},
    ]
    rates = compute_site_success_rate(rows)
    assert rates == {"a": 0.5, "b": 1.0}


def test_compute_site_gain_ok_thresholds_on_gain_and_max_base():
    hint_rates = {"a": 0.6, "b": 0.9, "c": 0.5}
    baseline_rates = {"a": 0.4, "b": 0.85}   # "c" missing from baseline
    ok = compute_site_gain_ok(hint_rates, baseline_rates, min_site_gain=0.1)
    assert ok == {"a": True, "b": False, "c": False}   # c: no baseline -> conservative False

    ok_capped = compute_site_gain_ok(hint_rates, baseline_rates, min_site_gain=0.1, max_base=0.5)
    assert ok_capped == {"a": True, "b": False, "c": False}   # b already fails gain

    hint_rates2 = {"a": 0.6}
    baseline_rates2 = {"a": 0.55}   # gain 0.05, below both thresholds
    ok2 = compute_site_gain_ok(hint_rates2, baseline_rates2, min_site_gain=0.1, max_base=0.5)
    assert ok2 == {"a": False}


def test_row_passes_filters_site_gain_stage():
    row_ok = {**_cont_row(site_id="a"), "family_dead": 0}
    row_bad = {**_cont_row(site_id="z"), "family_dead": 0}
    site_gain_ok = {"a": True}
    ok, _ = row_passes_filters(row_ok, require_redirect_for_dead=False, site_gain_ok=site_gain_ok)
    assert ok is True
    ok2, stage2 = row_passes_filters(row_bad, require_redirect_for_dead=False,
                                      site_gain_ok=site_gain_ok)
    assert ok2 is False and stage2 == "site_gain"   # missing from mapping -> rejected


def test_summarize_stages_site_gain_stage_present_only_when_requested():
    rows = [{**_cont_row(site_id="a"), "family_dead": 0}]
    summ_off = summarize_stages(rows, require_redirect_for_dead=False)
    assert "site_gain" not in summ_off["kept_by_stage"]

    summ_on = summarize_stages(rows, require_redirect_for_dead=False,
                                site_gain_ok={"a": True})
    assert summ_on["kept_by_stage"]["site_gain"] == 1

    summ_on_reject = summarize_stages(rows, require_redirect_for_dead=False,
                                       site_gain_ok={"a": False})
    assert summ_on_reject["kept_by_stage"]["site_gain"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# summarize_stages: redirect_share
# ══════════════════════════════════════════════════════════════════════════════

def test_summarize_stages_redirect_share():
    rows = [
        {**_cont_row(site_id="s1", decision="redirect"), "family_dead": 0},
        {**_cont_row(site_id="s2", decision="verify"), "family_dead": 0},
        {**_cont_row(site_id="s3", decision="redirect"), "family_dead": 0},
    ]
    summ = summarize_stages(rows, require_redirect_for_dead=False)
    assert summ["redirect_share"] == pytest.approx(2 / 3)


# ══════════════════════════════════════════════════════════════════════════════
# 0907 감사 — 오염① no_hint_mention: continuation_mentions_hint / 필터 배선
# ══════════════════════════════════════════════════════════════════════════════

def test_continuation_mentions_hint_detects_the_word():
    assert continuation_mentions_hint("as the hint suggests, try 9-1", "Hint: try 9 minus 1.")
    assert continuation_mentions_hint("Hinted at nothing new here", None)
    assert not continuation_mentions_hint("try 9-1=8 next", "Hint: try 9 minus 1.")


def test_continuation_mentions_hint_detects_verbatim_hint_sentence():
    hint_text = "Try subtracting one from nine first. Then add the rest."
    assert continuation_mentions_hint(
        "So I will do this: Try subtracting one from nine first.", hint_text)
    assert not continuation_mentions_hint("So I will try something else entirely.", hint_text)
    # too-short fragments don't count as a verbatim match (accidental overlap risk)
    assert not continuation_mentions_hint("Then.", "Then.")


def test_row_passes_filters_no_hint_mention_stage():
    bad_text = ("PREFIX. as the hint suggests, 9-1=8\n<meta>\nconfidence: 0.4\n"
                "This looks promising.\ndecision: verify\n</meta>\n5+3=8\n\\boxed{5+3}")
    row = {**_cont_row(full_text=bad_text), "family_dead": 0}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=False)
    assert ok is False and stage == "no_hint_mention"

    ok2, stage2 = row_passes_filters(
        row, require_redirect_for_dead=False, require_no_hint_mention=False)
    assert ok2 is True and stage2 == ""


# ══════════════════════════════════════════════════════════════════════════════
# 0907 감사 — 오염② no_template_meta: meta_text_is_template / 필터 배선
# ══════════════════════════════════════════════════════════════════════════════

def test_meta_text_is_template_detects_literal_instruction_phrases():
    assert meta_text_is_template(
        "<meta>\nconfidence: 0.4\nOne or two sentences judging your own approach so far.\n"
        "decision: verify\n</meta>")
    assert meta_text_is_template("<meta>\nconfidence: X\ngood\ndecision: verify\n</meta>")
    assert not meta_text_is_template(
        "<meta>\nconfidence: 0.4\nThe multiply-first family keeps overshooting badly.\n"
        "decision: redirect\n</meta>")


def test_meta_text_is_template_detects_six_word_verbatim_copy():
    copied = ("<meta>\nconfidence: 0.5\nWhich family of groupings you are exploring, and "
              "whether that family is worth continuing here.\ndecision: verify\n</meta>")
    assert meta_text_is_template(copied)


def test_row_passes_filters_no_template_meta_stage():
    bad_text = ("PREFIX. 5+3=8.\n<meta>\nconfidence: X\njudging your own approach so far\n"
                "decision: verify\n</meta>\n9-1=8\n\\boxed{9-1}")
    row = {**_cont_row(full_text=bad_text), "family_dead": 0}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=False)
    assert ok is False and stage == "no_template_meta"

    ok2, stage2 = row_passes_filters(
        row, require_redirect_for_dead=False, require_no_template_meta=False)
    assert ok2 is True and stage2 == ""


# ══════════════════════════════════════════════════════════════════════════════
# 0907 감사 — 오염③ meta_before_solve: meta_before_solve_ok / 필터 배선
# ══════════════════════════════════════════════════════════════════════════════

def test_meta_before_solve_ok_requires_attempt_between_meta_and_boxed():
    good = "PREFIX. <meta>\nconfidence: 0.4\ngood\ndecision: verify\n</meta>\n9-1=8\n\\boxed{9-1}"
    assert meta_before_solve_ok(good, target=10)

    no_attempt = "PREFIX. <meta>\nconfidence: 0.4\ngood\ndecision: verify\n</meta>\n\\boxed{9-1}"
    assert not meta_before_solve_ok(no_attempt, target=10)

    no_meta = "PREFIX. 9-1=8\n\\boxed{9-1}"
    assert not meta_before_solve_ok(no_meta, target=10)


def test_meta_before_solve_ok_rejects_post_solve_meta():
    # target(10) already reached in an equation BEFORE </meta> -> trailing verify, not a redirect
    already_hit_target = ("PREFIX. 9+1=10. <meta>\nconfidence: 0.9\ngood\n"
                          "decision: verify\n</meta>\n9+1=10\n\\boxed{9+1}")
    assert not meta_before_solve_ok(already_hit_target, target=10)

    # "works" signals the line before the meta already solved it, regardless of target
    already_worked = ("PREFIX. 4*3=12, close. This works! <meta>\nconfidence: 0.9\ngood\n"
                      "decision: verify\n</meta>\n4*3=12\n\\boxed{4*3}")
    assert not meta_before_solve_ok(already_worked, target=12)


def test_row_passes_filters_meta_before_solve_stage():
    bad_text = "PREFIX. <meta>\nconfidence: 0.4\ngood\ndecision: verify\n</meta>\n\\boxed{5+3}"
    row = {**_cont_row(full_text=bad_text), "family_dead": 0}
    ok, stage = row_passes_filters(row, require_redirect_for_dead=False)
    assert ok is False and stage == "meta_before_solve"

    ok2, stage2 = row_passes_filters(
        row, require_redirect_for_dead=False, require_meta_before_solve=False)
    assert ok2 is True and stage2 == ""


# ══════════════════════════════════════════════════════════════════════════════
# 0907 감사 — summarize_stages 배선 + 전부 끄면 v1 이전 동작과 동일
# ══════════════════════════════════════════════════════════════════════════════

def test_summarize_stages_counts_new_pollution_stages():
    bad_hint_text = ("PREFIX. hint says try 9-1.\n<meta>\nconfidence: 0.4\ngood\n"
                     "decision: verify\n</meta>\n5+3=8\n\\boxed{5+3}")
    rows = [
        {**_cont_row(site_id="s1"), "family_dead": 0},                          # clean, passes
        {**_cont_row(site_id="s2", full_text=bad_hint_text), "family_dead": 0},  # dies at no_hint_mention
    ]
    summ = summarize_stages(rows, require_redirect_for_dead=False)
    assert summ["kept_by_stage"]["no_hint_mention"] == 1
    assert summ["final_kept"] == 1


def test_all_three_new_filters_disabled_matches_pre_0907_behavior():
    contaminated = ("PREFIX. hint says try 9-1.\n<meta>\nconfidence: X\n"
                    "judging your own approach so far\ndecision: verify\n</meta>\n\\boxed{5+3}")
    row = {**_cont_row(full_text=contaminated), "family_dead": 0}
    ok, stage = row_passes_filters(
        row, require_redirect_for_dead=False,
        require_no_hint_mention=False, require_no_template_meta=False,
        require_meta_before_solve=False)
    assert ok is True and stage == ""
