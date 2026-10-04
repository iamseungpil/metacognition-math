"""screen_by_rollouts.py 회귀 시험(0914). 합성 texts.jsonl + train parquet 로
mixed-only + 예산 필터(mean_tok/trunc_rate) 를 검증한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import screen_by_rollouts as S  # noqa: E402


def _write_rollouts(path, spec: dict[str, list[dict]]):
    """spec: {problem: [ {r_corr, n_tok, truncated}, ... ]} — 한 problem 당 그룹 행들."""
    with open(path, "w", encoding="utf-8") as fh:
        for problem, samples in spec.items():
            for j, s in enumerate(samples):
                row = {"group_id": f"g_{problem}", "problem": problem}
                row.update(s)
                fh.write(json.dumps(row) + "\n")


def test_load_group_stats_computes_pass_rate_mean_tok_trunc_rate(tmp_path):
    path = tmp_path / "texts.jsonl"
    _write_rollouts(path, {
        "p0": [{"r_corr": 1, "n_tok": 100, "truncated": 0},
               {"r_corr": 0, "n_tok": 200, "truncated": 1}],
    })
    stats = S.load_group_stats(str(path))
    st = stats[S.norm_problem("p0")]
    assert st["pass_rate"] == 0.5
    assert st["mean_tok"] == 150.0
    assert st["trunc_rate"] == 0.5
    assert st["n"] == 2


def test_load_group_stats_conflicting_group_ids_raise(tmp_path):
    path = tmp_path / "texts.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"group_id": "gA", "problem": "same", "r_corr": 1, "n_tok": 1,
                             "truncated": 0}) + "\n")
        fh.write(json.dumps({"group_id": "gB", "problem": "same", "r_corr": 0, "n_tok": 1,
                             "truncated": 0}) + "\n")
    try:
        S.load_group_stats(str(path))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def _row(problem, gold="1"):
    return {"problem": problem, "gold": gold, "prompt": [{"role": "user", "content": problem}],
            "extra_info": {"problem": problem, "gold": gold}}


def test_screen_rows_keeps_only_mixed_within_budget():
    rows = [_row("mixed_ok"), _row("all_pass"), _row("all_fail"), _row("mixed_too_long"),
            _row("mixed_too_trunc"), _row("no_rollout")]
    stats = {
        S.norm_problem("mixed_ok"): {"pass_rate": 0.5, "mean_tok": 3000, "trunc_rate": 0.0, "n": 8},
        S.norm_problem("all_pass"): {"pass_rate": 1.0, "mean_tok": 1000, "trunc_rate": 0.0, "n": 8},
        S.norm_problem("all_fail"): {"pass_rate": 0.0, "mean_tok": 1000, "trunc_rate": 0.0, "n": 8},
        S.norm_problem("mixed_too_long"): {"pass_rate": 0.25, "mean_tok": 6000, "trunc_rate": 0.0, "n": 8},
        S.norm_problem("mixed_too_trunc"): {"pass_rate": 0.25, "mean_tok": 3000, "trunc_rate": 0.5, "n": 8},
        # "no_rollout" absent on purpose
    }
    kept, counters = S.screen_rows(rows, stats, max_mean_tok=5000, max_trunc_rate=0.2,
                                   variant="math_opt")
    kept_problems = {r["problem"] for r in kept}
    assert kept_problems == {"mixed_ok"}
    assert counters["n_in"] == 6
    assert counters["n_kept"] == 1
    assert counters["n_no_stats"] == 1
    assert counters["n_not_mixed"] == 2
    assert counters["n_too_long"] == 1
    assert counters["n_too_trunc"] == 1
    r = kept[0]
    assert r["extra_info"]["group_pass_rate"] == 0.5


def test_screen_rows_rebuilds_prompt_for_variant():
    """★버그 회귀(0914): kept 행은 입력 parquet 의 prompt(math_opt 등)를 그대로
    두면 안 되고 --variant 로 재생성해야 한다 — math_retry 로 스크리닝했는데
    forced 아닌 kept 행이 math_opt 프롬프트로 학습되던 버그."""
    rows = [_row("p0")]
    stats = {S.norm_problem("p0"): {"pass_rate": 0.5, "mean_tok": 100, "trunc_rate": 0.0, "n": 8}}

    kept, _ = S.screen_rows(rows, stats, max_mean_tok=5000, max_trunc_rate=0.2,
                            variant="math_retry")
    expected = S.build_math_prompt("p0", "math_retry")
    assert kept[0]["prompt"] == expected
    assert kept[0]["extra_info"]["prompt_variant"] == "math_retry"
    # system prompt must be the math_retry variant, not the input's math_opt
    assert kept[0]["prompt"][0]["content"] == S.MATH_PROMPT_VARIANTS["math_retry"]


def test_screen_rows_default_variant_is_idempotent_math_opt():
    """default --variant=math_opt: kept 행 prompt 재생성이 입력(이미 math_opt 로
    빌드된 build_math_parquet.py 산출물)과 동일해야 한다(멱등)."""
    rows = [_row("p0")]
    rows[0]["prompt"] = S.build_math_prompt("p0", "math_opt")  # input as build_math_parquet.py would build it
    stats = {S.norm_problem("p0"): {"pass_rate": 0.5, "mean_tok": 100, "trunc_rate": 0.0, "n": 8}}

    kept, _ = S.screen_rows(rows, stats, max_mean_tok=5000, max_trunc_rate=0.2,
                            variant="math_opt")
    assert kept[0]["prompt"] == rows[0]["prompt"]
    assert kept[0]["extra_info"]["prompt_variant"] == "math_opt"


def test_apply_forced_redirect_marks_fraction_of_kept_pool():
    kept = [_row(f"p{i}") for i in range(20)]
    for r in kept:
        r["extra_info"]["group_pass_rate"] = 0.5  # all mixed by construction
    out, n_forced = S.apply_forced_redirect(kept, forced_frac=0.25, forced_variant="math_retry_forced",
                                            seed=11)
    assert n_forced == 5
    forced = [r for r in out if r["extra_info"].get("forced_redirect") == 1]
    assert len(forced) == 5
    for r in forced:
        assert r["extra_info"]["prompt_variant"] == "math_retry_forced"


def test_apply_forced_redirect_zero_frac_is_noop():
    kept = [_row("p0")]
    out, n_forced = S.apply_forced_redirect(kept, forced_frac=0.0, forced_variant="math_retry_forced",
                                            seed=11)
    assert n_forced == 0
    assert out == kept


def _allwrong_stat(agree_state):
    return {"pass_rate": 0.0, "mean_tok": 100, "trunc_rate": 0.0, "n": 8,
            "agree_state": agree_state}


def test_screen_rows_default_excludes_allwrong():
    """★기본(--include_allwrong_states 미지정)은 옛 동작 그대로 — pass_rate==0 은 상태와
    무관하게 전부 걸러진다."""
    rows = [_row("mixed_ok"), _row("dominant_allwrong"), _row("all_same_allwrong")]
    stats = {
        S.norm_problem("mixed_ok"): {"pass_rate": 0.5, "mean_tok": 100, "trunc_rate": 0.0, "n": 8,
                                     "agree_state": "SPLIT"},
        S.norm_problem("dominant_allwrong"): _allwrong_stat("DOMINANT"),
        S.norm_problem("all_same_allwrong"): _allwrong_stat("ALL_SAME"),
    }
    kept, counters = S.screen_rows(rows, stats, max_mean_tok=5000, max_trunc_rate=0.2,
                                   variant="math_opt")
    assert {r["problem"] for r in kept} == {"mixed_ok"}
    assert counters["n_kept_mixed"] == 1
    assert counters["n_kept_allwrong"] == 0
    assert kept[0]["extra_info"]["agree_state"] == "SPLIT"


def test_screen_rows_include_allwrong_states_keeps_dominant_not_all_same():
    rows = [_row("mixed_ok"), _row("dominant_allwrong"), _row("all_same_allwrong")]
    stats = {
        S.norm_problem("mixed_ok"): {"pass_rate": 0.5, "mean_tok": 100, "trunc_rate": 0.0, "n": 8,
                                     "agree_state": "SPLIT"},
        S.norm_problem("dominant_allwrong"): _allwrong_stat("DOMINANT"),
        S.norm_problem("all_same_allwrong"): _allwrong_stat("ALL_SAME"),
    }
    kept, counters = S.screen_rows(rows, stats, max_mean_tok=5000, max_trunc_rate=0.2,
                                   variant="math_opt",
                                   include_allwrong_states=["DOMINANT", "SPLIT", "SCATTER", "NOANS"])
    kept_problems = {r["problem"] for r in kept}
    assert kept_problems == {"mixed_ok", "dominant_allwrong"}
    assert counters["n_kept_mixed"] == 1
    assert counters["n_kept_allwrong"] == 1
    dominant_row = next(r for r in kept if r["problem"] == "dominant_allwrong")
    assert dominant_row["extra_info"]["group_pass_rate"] == 0.0
    assert dominant_row["extra_info"]["agree_state"] == "DOMINANT"


def test_screen_rows_allwrong_still_respects_length_and_trunc_filters():
    rows = [_row("dominant_too_long"), _row("dominant_too_trunc")]
    stats = {
        S.norm_problem("dominant_too_long"): {"pass_rate": 0.0, "mean_tok": 6000, "trunc_rate": 0.0,
                                              "n": 8, "agree_state": "DOMINANT"},
        S.norm_problem("dominant_too_trunc"): {"pass_rate": 0.0, "mean_tok": 100, "trunc_rate": 0.5,
                                               "n": 8, "agree_state": "DOMINANT"},
    }
    kept, counters = S.screen_rows(rows, stats, max_mean_tok=5000, max_trunc_rate=0.2,
                                   variant="math_opt", include_allwrong_states=["DOMINANT"])
    assert kept == []
    assert counters["n_too_long"] == 1
    assert counters["n_too_trunc"] == 1


def test_load_group_stats_computes_agree_state(tmp_path):
    """★엔드투엔드: 진짜 texts.jsonl 로 all-wrong DOMINANT 문제를 --include_allwrong_states 로
    남기고 ALL_SAME 문제는 남기지 않는다(final_answer 로부터 gold 없는 합의 상태 계산)."""
    path = tmp_path / "texts.jsonl"
    _write_rollouts(path, {
        "mixed_p": [{"r_corr": 1, "n_tok": 100, "truncated": 0, "final_answer": "1"},
                    {"r_corr": 0, "n_tok": 100, "truncated": 0, "final_answer": "2"}],
        "dominant_p": [{"r_corr": 0, "n_tok": 100, "truncated": 0, "final_answer": "3"}] * 6
                      + [{"r_corr": 0, "n_tok": 100, "truncated": 0, "final_answer": "4"}] * 2,
        "all_same_p": [{"r_corr": 0, "n_tok": 100, "truncated": 0, "final_answer": "5"}] * 8,
    })
    stats = S.load_group_stats(str(path))
    assert stats[S.norm_problem("mixed_p")]["agree_state"] == "DOMINANT" \
        or stats[S.norm_problem("mixed_p")]["pass_rate"] == 0.5
    assert stats[S.norm_problem("dominant_p")]["agree_state"] == "DOMINANT"
    assert stats[S.norm_problem("dominant_p")]["pass_rate"] == 0.0
    assert stats[S.norm_problem("all_same_p")]["agree_state"] == "ALL_SAME"
    assert stats[S.norm_problem("all_same_p")]["pass_rate"] == 0.0

    rows = [_row("mixed_p"), _row("dominant_p"), _row("all_same_p")]
    kept, _ = S.screen_rows(rows, stats, max_mean_tok=5000, max_trunc_rate=0.2, variant="math_opt",
                            include_allwrong_states=["DOMINANT", "SPLIT", "SCATTER", "NOANS"])
    kept_problems = {r["problem"] for r in kept}
    assert kept_problems == {"mixed_p", "dominant_p"}
    assert "all_same_p" not in kept_problems
