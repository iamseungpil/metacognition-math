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
    row_passes_filters, summarize_stages,
)


def _cont_row(site_id="s1", *, r_corr=1, emitted=1, novel=1, followed=1, truncated=0,
             decision="redirect", mode="hint", k_index=0, full_text="PREFIX. cont",
             n_tokens=10):
    return {
        "site_id": site_id, "mode": mode, "k_index": k_index, "r_corr": r_corr,
        "emitted": emitted, "novel": novel, "followed": followed, "truncated": truncated,
        "decision": decision, "full_text": full_text, "hint_text": "Hint: ...",
        "n_tokens": n_tokens,
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
