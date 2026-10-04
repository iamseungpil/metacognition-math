#!/usr/bin/env python
r"""math_verify_perform_gate — 설계 계약 고정.

왜: 이 게이트가 거짓 양성을 내는 길은 셋뿐이다 — (a) 조건 dict 를 고쳤는데 표·판정이 안 따라옴,
  (b) `plain` 이 기준선이 아니게 되어 Δ 가 «프롬프트가 달라진 효과»를 섞음, (c) 짝짓기가
  단위를 통째로 버려 표본이 조용히 줄어듦(구 content gate 의 버그). 여기서 셋을 못 박는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))

from math_verify_perform_gate import (  # noqa: E402
    ACT_CONDS, CONTROL, CONTROL2, MAX_TRUNC, MIN_PERFORMED, PASS_DELTA, PASS_DELTA_REREAD, REF,
    VERIFY_CONDS, cond_prompt, conds_all, final_answer, gen_record, load_excluded_units,
    paired_delta, parse_conds, perform_causal_pass, resummarize_gens, reread_confound_flags,
    summarize,
)
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402


class _Tok:
    """chat template 을 흉내내는 최소 토크나이저 — 바이트 동일성만 보면 되므로 이걸로 충분."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
        s = "".join(f"<|{m['role']}|>{m['content']}" for m in msgs)
        return s + ("<|assistant|>" if add_generation_prompt else "")


def test_conds_dict_is_single_source_of_truth():
    names = conds_all()
    assert names == tuple(VERIFY_CONDS)
    assert REF in names and CONTROL in names
    assert set(ACT_CONDS) == {"recompute_tmpl", "backward_tmpl", "magnitude_tmpl"}
    assert all(VERIFY_CONDS[c].act is not None for c in ACT_CONDS)
    assert VERIFY_CONDS[REF].suffix == "" and VERIFY_CONDS[CONTROL].act is None


def test_plain_prompt_is_byte_identical():
    tok, prob = _Tok(), "What is 2+2?"
    assert cond_prompt(tok, "math_opt", prob, "plain") == render_generation_prompt(
        tok, "math_opt", prob)


def test_act_prompts_only_append_suffix_to_user_turn():
    tok, prob = _Tok(), "What is 2+2?"
    base = cond_prompt(tok, "math_opt", prob, "plain")
    for c in (*ACT_CONDS, CONTROL):
        got = cond_prompt(tok, "math_opt", prob, c)
        assert got != base
        # system 은 그대로고 접미만 사용자 턴 끝에 붙는다 → 접미를 빼면 기준선과 같아진다
        assert got.replace(VERIFY_CONDS[c].suffix, "", 1) == base


def test_all_conditions_demand_final_section():
    for c in (*ACT_CONDS, CONTROL):
        assert "## Final" in VERIFY_CONDS[c].suffix       # 구조 동일(대조군 포함)
        assert "\\boxed{}" in VERIFY_CONDS[c].suffix


def test_final_answer_prefers_final_section():
    t = r"## Method 2\nresult \boxed{99}" + "\n## Final\n" + r"\boxed{12}"
    assert final_answer(t) == "12"
    assert final_answer(r"no section, just \boxed{7}") == "7"
    # 섹션은 있으나 박스가 없으면 전체 마지막으로 폴백
    assert final_answer(r"\boxed{5}" + "\n## Final\nthe answer stands") == "5"


def _unit(uid, accs):
    return {"unit_id": uid, "tag": "t", "group_id": uid, "gold": "1",
            "cond": {c: {"acc": v, "trunc": 0.0, "tokens": 100.0, "performed": 1.0,
                         "changed": 0.0, "acc_changed": float("nan"),
                         "cat": {"confirm_right": 1.0, "confirm_wrong": 0.0,
                                 "revise_right": 0.0, "revise_wrong": 0.0, "none": 0.0}}
                     for c, v in accs.items()}}


def test_pairing_drops_only_the_pairs_it_lacks():
    """★구 content gate 의 버그를 안 옮긴다: 한 조건이 빠진 단위도 나머지 쌍에는 남는다."""
    recs = [_unit("a", {"plain": 0.0, "tmpl_null": 0.0, "recompute_tmpl": 1.0}),
            _unit("b", {"plain": 0.0, "recompute_tmpl": 1.0}),          # tmpl_null 없음
            _unit("c", {"plain": 0.0, "tmpl_null": 0.0, "recompute_tmpl": 1.0})]
    ci_ctl, _, n_ctl = paired_delta(recs, "recompute_tmpl", "tmpl_null", seed=0, n_boot=50)
    ci_ref, _, n_ref = paired_delta(recs, "recompute_tmpl", "plain", seed=0, n_boot=50)
    assert n_ctl == 2 and n_ref == 3          # b 는 tmpl_null 쌍에서만 빠진다
    assert abs(ci_ctl["mean"] - 1.0) < 1e-9 and abs(ci_ref["mean"] - 1.0) < 1e-9
    summ = summarize(recs, conds_all(), k=1, seed=0, n_boot=50)
    assert summ["n_units"] == 3
    assert summ["n_units_per_cond"]["tmpl_null"] == 2
    assert summ["n_units_per_cond"]["recompute_tmpl"] == 3
    assert summ["n_pairs_vs_control"]["recompute_tmpl"] == 2


def _summ(delta, performed, trunc, name="recompute_tmpl"):
    return {"acts_ranked": [name],
            "delta_vs_control": {name: {"mean": delta, "lo": delta - 0.01, "hi": delta + 0.01,
                                        "n": 50}},
            "performed_rate": {name: performed}, "trunc_rate": {name: trunc}}


def test_pass_rule_on_synthetic_summaries():
    ok, win = perform_causal_pass(_summ(0.10, 0.80, 0.05))
    assert ok and win == ["recompute_tmpl"]
    assert not perform_causal_pass(_summ(PASS_DELTA - 0.005, 0.80, 0.05))[0]   # 효과크기 미달
    assert not perform_causal_pass(_summ(0.10, MIN_PERFORMED - 0.01, 0.05))[0]  # 수행률 미달
    assert not perform_causal_pass(_summ(0.10, 0.80, MAX_TRUNC + 0.01))[0]      # 절단 초과
    assert not perform_causal_pass(_summ(float("nan"), 0.80, 0.05))[0]          # nan 은 FAIL
    ci = _summ(0.10, 0.80, 0.05)
    ci["delta_vs_control"]["recompute_tmpl"]["lo"] = -0.02                      # CI 가 0 포함
    assert not perform_causal_pass(ci)[0]


def test_summary_flags_and_pass_propagate():
    recs = [_unit(f"u{i}", {c: (1.0 if c == "recompute_tmpl" else 0.0) for c in conds_all()})
            for i in range(20)]
    for r in recs:                                  # 절단·수행률을 나쁜 쪽으로 흔든다
        r["cond"]["backward_tmpl"]["trunc"] = 0.5
        r["cond"]["magnitude_tmpl"]["performed"] = 0.1
    summ = summarize(recs, conds_all(), k=1, seed=0, n_boot=200)
    assert summ["acts_ranked"][0] == "recompute_tmpl"
    assert summ["pass_perform_causal"] == 1 and summ["winners"] == ["recompute_tmpl"]
    assert "backward_tmpl" in summ["budget_flag"]
    assert "magnitude_tmpl" in summ["perform_low"]
    assert set(summ["category_hist"]["plain"]) >= {"confirm_right", "none"}


# ══ 0915 확인 실험: 되읽기 대조 ═══════════════════════════════════════════════════
def test_reread_null_is_a_second_control():
    assert CONTROL2 == "reread_null" and CONTROL2 in conds_all()
    c = VERIFY_CONDS[CONTROL2]
    assert c.act is None                       # 대조군이다(수행을 시험하지 않는다)
    assert "## Recheck" in c.suffix and "## Final" in c.suffix
    assert "no new calculations" in c.suffix   # tmpl_null 처럼 새 계산을 금지하되
    assert "every numeric quantity" in c.suffix  # **문제를 다시 읽게** 한다 — 그것이 차이다


def test_reread_null_prompt_only_appends_suffix():
    tok, prob = _Tok(), "What is 2+2?"
    base = cond_prompt(tok, "math_opt", prob, "plain")
    got = cond_prompt(tok, "math_opt", prob, CONTROL2)
    assert got != base and got.replace(VERIFY_CONDS[CONTROL2].suffix, "", 1) == base


def test_parse_conds_subsets_and_fails_loud():
    assert parse_conds("") == conds_all()
    # 정의 순서로 되돌린다(호출 순서를 흔들어도 시드 오프셋이 안 바뀐다)
    got = parse_conds("backward_tmpl,reread_null,tmpl_null")
    assert got == tuple(c for c in conds_all()
                        if c in {"backward_tmpl", "reread_null", "tmpl_null"})
    assert got.index("tmpl_null") < got.index("backward_tmpl")
    try:
        parse_conds("tmpl_null,nope")
    except SystemExit:
        return
    raise AssertionError("모르는 조건은 fail-loud 여야 한다")


def test_exclude_units_from_filters_by_unit_id(tmp_path):
    p = tmp_path / "per_problem.jsonl"
    p.write_text('{"unit_id": "a::g1"}\n\n{"unit_id": "a::g2"}\n')
    used = load_excluded_units(str(p))
    assert used == {"a::g1", "a::g2"}
    assert load_excluded_units("") == set()
    pool = [{"unit_id": u} for u in ("a::g1", "a::g3", "a::g2", "a::g4")]
    assert [u["unit_id"] for u in pool if u["unit_id"] not in used] == ["a::g3", "a::g4"]


def _summ2(d_ctl, d_rr, name="backward_tmpl"):
    return {"conds": [name], "acts_ranked": [name],
            "delta_vs_control": {name: {"mean": d_ctl, "lo": d_ctl - 0.01,
                                        "hi": d_ctl + 0.01, "n": 50}},
            "delta_vs_reread": {name: {"mean": d_rr, "lo": d_rr - 0.01, "hi": d_rr + 0.01,
                                       "n": 50}},
            "performed_rate": {name: 0.80}, "trunc_rate": {name: 0.05}}


def test_reread_confound_flag_logic():
    assert reread_confound_flags(_summ2(0.05, -0.01)) == ["backward_tmpl"]   # 이득이 사라짐
    assert reread_confound_flags(_summ2(0.05, 0.00)) == ["backward_tmpl"]    # ≤ 0 이면 플래그
    assert reread_confound_flags(_summ2(0.05, 0.04)) == []                   # 대조를 넘김
    assert reread_confound_flags(_summ2(-0.02, -0.03)) == []                 # 애초에 이득 없음
    nan = _summ2(0.05, float("nan"))
    assert reread_confound_flags(nan) == []       # ★대조가 없으면 헛경보를 안 낸다
    # 대조군 자체는 플래그 대상이 아니다
    assert reread_confound_flags(_summ2(0.05, -0.01, name=CONTROL)) == []


def test_confirm_stage_requires_the_reread_gate():
    s = _summ2(0.10, 0.06)
    assert perform_causal_pass(s, stage="screen")[0]
    assert perform_causal_pass(s, stage="confirm") == (True, ["backward_tmpl"])
    # screen 은 통과하지만 되읽기 대조를 못 넘는 팔 — confirm 에서 떨어져야 한다
    assert perform_causal_pass(_summ2(0.10, -0.01), stage="screen")[0]
    assert not perform_causal_pass(_summ2(0.10, -0.01), stage="confirm")[0]
    assert not perform_causal_pass(_summ2(0.10, PASS_DELTA_REREAD - 0.005),
                                   stage="confirm")[0]          # 효과크기 미달
    wide = _summ2(0.10, 0.06)
    wide["delta_vs_reread"]["backward_tmpl"]["lo"] = -0.02       # CI 가 0 포함
    assert not perform_causal_pass(wide, stage="confirm")[0]
    assert not perform_causal_pass(_summ(0.10, 0.80, 0.05), stage="confirm")[0]  # 대조 없음


def test_summarize_carries_stage_and_reread_columns():
    recs = [_unit(f"u{i}", {c: (1.0 if c == "recompute_tmpl" else 0.0) for c in conds_all()})
            for i in range(20)]
    summ = summarize(recs, conds_all(), k=1, seed=0, n_boot=200, stage="confirm")
    assert summ["stage"] == "confirm" and summ["control2"] == CONTROL2
    assert summ["n_pairs_vs_reread"]["recompute_tmpl"] == 20
    assert abs(summ["delta_vs_reread"]["recompute_tmpl"]["mean"] - 1.0) < 1e-9
    for key in ("slot_rate", "compare_rate", "recovered_match_rate",
                "recovered_in_problem_rate", "revise_rate", "acc_given_revise",
                "verdict_hist", "performed_tmpl_sanity"):
        assert key in summ and set(summ[key]) == set(conds_all())
    assert summ["pass_perform_causal"] == 1        # 두 대조를 다 넘는다


# ══ gen_record 의 탐지기 배선 ═════════════════════════════════════════════════════
_TMPL_BACKWARD = r"""The speed is \boxed{60}.

## Backward check
distance = 60 * 2 = 120
Recovered: 120 vs stated: 120

## Compare
Same.

## Final
\boxed{60}"""


def test_gen_record_uses_template_detector_for_act_conds():
    r = gen_record(_TMPL_BACKWARD, "backward_tmpl", "60", truncated=0, tokens=100,
                   problem="A train covers 120 km in 2 hours. What is its speed?")
    assert r["detector"] == "template" and r["performed"] == 1.0
    assert r["sections"] == {"main": 1, "slot": 1, "compare": 1}
    assert r["slot_values"]["recovered"] == "120"
    assert r["recovered_matches_stated"] == 1.0
    assert r["recovered_in_problem"] == 1.0     # ★«무효 검산» 신호 — 문제에 이미 있던 수다
    assert r["category"] == "confirm_right" and r["revised"] == 0.0


def test_gen_record_controls_keep_free_text_detector_and_sanity():
    for c in ("plain", CONTROL, CONTROL2):
        r = gen_record(_TMPL_BACKWARD, c, "60", truncated=0, tokens=100, problem="")
        assert r["detector"] == "freetext"
        assert r["performed_tmpl_sanity"] == 0.0   # recompute 잣대로는 템플릿 수행이 아니다
        assert "recovered_in_problem" not in r


# ══ --resummarize: gens.jsonl → per_problem/gate_summary, GPU 없이 ═════════════════
def test_resummarize_gens_recomputes_from_stored_text(tmp_path):
    """★생성 없이 저장된 text 를 CURRENT 탐지기로 다시 잰다 — compliance(형식)와
    performed(수행)가 갈리는 사례(계산줄 없는 backward_tmpl)를 한 unit 에 심어 확인한다."""
    tmpl_no_calc = _TMPL_BACKWARD.replace(
        "distance = 60 * 2 = 120\n", "")   # 슬롯·섹션은 다 있지만 계산줄이 없다
    gens = [
        # unit u1 — backward_tmpl: 계산 있음 → performed=1, compliant=1
        {"unit_id": "src::g0", "tag": "src", "cond": "backward_tmpl", "text": _TMPL_BACKWARD,
         "r_corr": 1, "trunc": 0, "tokens": 50},
        # unit u1 — 대조군 plain
        {"unit_id": "src::g0", "tag": "src", "cond": "plain", "text": "\\boxed{60}",
         "r_corr": 1, "trunc": 0, "tokens": 5},
        # unit u2 — backward_tmpl: 슬롯·섹션은 있지만 계산줄이 없다 → performed=0, compliant=1
        {"unit_id": "src::g1", "tag": "src", "cond": "backward_tmpl", "text": tmpl_no_calc,
         "r_corr": 1, "trunc": 0, "tokens": 50},
        {"unit_id": "src::g1", "tag": "src", "cond": "plain", "text": "\\boxed{60}",
         "r_corr": 1, "trunc": 0, "tokens": 5},
    ]
    gens_path = tmp_path / "gens.jsonl"
    with gens_path.open("w") as fh:
        for g in gens:
            fh.write(__import__("json").dumps(g) + "\n")
    out_dir = tmp_path / "resum"
    rc = resummarize_gens(str(gens_path), out_dir=str(out_dir), seed=0, n_boot=50,
                          conds_spec="plain,backward_tmpl")
    assert rc == 0
    assert (out_dir / "per_problem.jsonl").exists()
    assert (out_dir / "gate_summary.json").exists()
    assert (out_dir / "gate_summary.md").exists()

    import json as _json
    recs = [_json.loads(ln) for ln in open(out_dir / "per_problem.jsonl")]
    by_unit = {r["unit_id"]: r for r in recs}
    assert by_unit["src::g0"]["cond"]["backward_tmpl"]["performed"] == 1.0
    assert by_unit["src::g0"]["cond"]["backward_tmpl"]["compliant"] == 1.0
    assert by_unit["src::g1"]["cond"]["backward_tmpl"]["performed"] == 0.0
    assert by_unit["src::g1"]["cond"]["backward_tmpl"]["compliant"] == 1.0

    summ = _json.loads((out_dir / "gate_summary.json").read_text())
    pooled = summ["pooled"]
    assert pooled["compliance_rate"]["backward_tmpl"] == 1.0
    assert pooled["performed_rate"]["backward_tmpl"] == 0.5
    md = (out_dir / "gate_summary.md").read_text()
    assert "compliance" in md
