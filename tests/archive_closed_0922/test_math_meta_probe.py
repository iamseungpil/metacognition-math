"""math_meta_probe 의 CPU 단위 테스트 — «행위 자리의 자»가 거짓말하지 않는지 본다.

GPU 없이 도는 것만 건드린다: 행위 스팬 찾기(META_ACTS 전 조건 + 없음 → None), 자리 넷의
문자→토큰 인덱스 해석(오프셋 있는 mock 토크나이저), «없는 자리»를 세는지(대체하지 않는지),
§C1 세 숫자가 «문제 안에서만 사는 층» 과 «문제 간에서만 사는 층» 을 실제로 가르는지,
층 이어붙이기가 무정보 단일 층을 이기는지, 그리고 [CONFOUND?] 가 터지는지.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))

import math_meta_probe as M  # noqa: E402
import math_ruler_pivot as P  # noqa: E402
from math_meta_content_gate import META_ACTS  # noqa: E402

BOX = r"Therefore \boxed{42}."

# 조건마다 «그 행위를 실제로 한» 응답 한 개. 단계(pre/post/mid)에 맞는 자리에 문장을 둔다.
PRESENT = {
    "substitute": f"Work.\n\n{BOX}\n\nLet me substitute it back into the original problem.\n",
    "special_case": f"Work.\n\n{BOX}\n\nLet me test a simple special case, n=1.\n",
    "recompute": f"Work.\n\n{BOX}\n\nLet me solve it by a different method now.\n",
    "constraint": f"The problem asks for the number of x.\n\nWork.\n\n{BOX}",
    "plan": f"My approach: use symmetry.\n\nWork.\n\n{BOX}",
    "magnitude": f"The sign should be positive here.\n\nWork.\n\n{BOX}",
    "stepcheck": f"Step 1 gives 6.\n\nLet me double-check this arithmetic.\n\nWork.\n\n{BOX}",
    "verification_first": f"The candidate answer is wrong.\n\nWork.\n\n{BOX}",
    "verification_first_random": f"The given answer turns out to be incorrect.\n\nW.\n\n{BOX}",
    "backward_mask": f"Work.\n\n{BOX}\n\nWorking backwards from 42 recovers the value 7.\n",
    # pad_length 의 rx 는 «수식 전 산문 문장 3개» 근사라 문장부호 셋이 앞에 있어야 한다.
    "pad_length": f"Care matters. Haste hurts. Slow is fine.\n\nWork.\n\n{BOX}",
}
ABSENT = f"Some neutral work with numbers.\n\nMore work.\n\n{BOX}"


class CharTok:
    """문자 하나 = 토큰 하나인 mock 토크나이저(오프셋 제공). 프롬프트 토큰은 항상 접두다."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) % 1000 for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 for c in text]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"


# ── 행위 스팬 ────────────────────────────────────────────────────────────────

def test_every_act_has_a_phase():
    """★META_ACTS 에 행위를 더하면서 ACT_PHASE 를 안 채우면 그 행위의 자리가 조용히 «pre» 로
    떨어진다(답 뒤 검사인데 앞에서 찾는다). 빠진 이름을 그대로 찍어 준다."""
    acts = {n for n, a in META_ACTS.items() if a.rx is not None}
    assert not (acts - set(M.ACT_PHASE)), f"ACT_PHASE 에 없는 행위: {sorted(acts - set(M.ACT_PHASE))}"


@pytest.mark.parametrize("cond", sorted(PRESENT))
def test_locate_act_span_finds_each_act(cond):
    sp = M.locate_act_span(PRESENT[cond], cond)
    assert sp is not None, cond
    s, e = sp
    assert 0 <= s < e <= len(PRESENT[cond])


@pytest.mark.parametrize("cond", sorted(PRESENT))
def test_locate_act_span_absent_is_none(cond):
    """★없으면 None — 전체 응답으로 조용히 물러서지 않는다."""
    assert M.locate_act_span(ABSENT, cond) is None


@pytest.mark.parametrize("cond", sorted(n for n, a in META_ACTS.items() if a.rx is None))
def test_controls_have_no_act_span(cond):
    assert M.locate_act_span(PRESENT["substitute"], cond) is None


def test_post_act_before_the_answer_does_not_count():
    """대입은 «답 뒤» 행위다 — \\boxed 앞에만 그 어휘가 있으면 자리가 없다."""
    t = "Let me substitute it back first.\n\nWork.\n\n" + BOX
    assert M.locate_act_span(t, "substitute") is None


def test_post_act_without_boxed_is_none():
    assert M.locate_act_span("Let me substitute it back into the problem.", "substitute") is None


def test_pre_act_span_stops_before_the_solution():
    t = PRESENT["constraint"]
    s, e = M.locate_act_span(t, "constraint")
    assert t[s:e] == "The problem asks for the number of x."


def test_plan_compliance_window_is_the_head_only():
    """`plan` 은 준수 판정과 같게 서두 600자 안에서만 본다(끝의 'another approach' 는 계획이 아니다)."""
    t = "x" * (M.PLAN_RX_HEAD_CHARS + 10) + "\n\nMy approach: symmetry.\n\n" + BOX
    assert M.locate_act_span(t, "plan") is None


def test_span_is_capped_when_there_are_no_paragraph_breaks():
    t = BOX + " Let me substitute it back. " + "y" * 5000
    s, e = M.locate_act_span(t, "substitute")
    assert e - s <= M.MAX_ACT_CHARS + 40


def test_continue_mode_uses_the_first_paragraph():
    t = "Substituting gives 5=5. OK.\n\nSo the answer stands."
    assert M.locate_act_span(t, "substitute", mode="continue") == (0, 27)
    assert M.locate_act_span(t, "filler_continue", mode="continue") is None


# ── 자리 넷 ─────────────────────────────────────────────────────────────────

def test_char_positions_answer_and_post_answer():
    t = PRESENT["substitute"]
    pos = M.char_positions(t, "substitute")
    assert t[pos["answer_start"]] == " " and t[pos["answer_start"] + 1] == "\\"
    assert t[pos["post_answer"]] == "."          # \boxed{42} 가 닫힌 직후
    assert pos["last"] == len(t) - 1
    assert pos["meta_end"] == M.locate_act_span(t, "substitute")[1] - 1


def test_post_answer_is_missing_when_nothing_follows():
    t = "Work.\n\n" + r"\boxed{42}"
    pos = M.char_positions(t, "plain")
    assert pos["post_answer"] is None and pos["meta_end"] is None
    assert pos["answer_start"] is not None


def test_char_positions_all_missing_without_boxed():
    pos = M.char_positions("no answer here", "plain")
    assert pos["answer_start"] is None and pos["post_answer"] is None
    assert pos["last"] == len("no answer here") - 1


def test_token_index_at_on_offsets():
    offs = [(0, 0), (0, 3), (3, 7), (7, 8)]   # (0,0) = 특수토큰(어떤 문자도 안 덮는다)
    assert M.token_index_at(offs, 0) == 1
    assert M.token_index_at(offs, 4) == 2
    assert M.token_index_at(offs, 7) == 3
    assert M.token_index_at(offs, 99) is None
    assert M.token_index_at(offs, None) is None


def test_prepare_row_resolves_positions_to_token_indices():
    tok = CharTok()
    prompt, text = "PROMPT>", PRESENT["substitute"]
    got = M.prepare_row(tok, prompt, text, "substitute", "prompt")
    assert got["n_prompt"] == len(prompt)
    ch = M.char_positions(text, "substitute")
    for name, c in ch.items():          # 문자=토큰이므로 인덱스는 len(prompt)+문자오프셋
        assert got["pos"][name] == (None if c is None else len(prompt) + c)


def test_prepare_row_fails_loud_when_prompt_is_not_a_token_prefix():
    class Weird(CharTok):
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            out = super().__call__(text, return_offsets_mapping=return_offsets_mapping)
            out["input_ids"] = out["input_ids"][::-1]      # 접두 성질을 깨뜨린다
            return out
    assert M.prepare_row(Weird(), "PROMPT>", "abc" + BOX, "plain", "prompt") is None


def test_extract_reads_four_positions_in_one_forward():
    """★자리 넷을 job 하나로 읽는다 — forward 호출이 행 수만큼만 나와야 한다."""
    tok = CharTok()
    units = {"t::g0": {"problem": "1+1?", "gold": "2", "text": "", "cand_vf": None}}
    rows = [{"unit_id": "t::g0", "cond": "substitute", "group_id": "t::g0",
             "text": PRESENT["substitute"], "r_corr": 1.0, "trunc": 0.0, "n_tok": 10.0}]
    seen = {}

    def fwd(jobs, layers):
        seen["n_jobs"] = len(jobs)
        seen["n_ats"] = len(jobs[0].hidden_ats)
        return P.mock_forward_factory(dim=4)(jobs, layers)
    got, diag = M.extract(rows, units, tok, fwd, [1], variant="math_opt", mode="prompt")
    assert seen["n_jobs"] == 1 and seen["n_ats"] == 4
    assert diag["n_rows_forwarded"] == 1 and diag["n_roundtrip_fail"] == 0
    assert set(got[0]["hidden"]) == set(M.POSITIONS)


# ── F3: gens.jsonl 의 cand 를 읽어 verification_first(_random) 을 되살린다 ──────
def test_extract_passes_recorded_cand_through_for_verification_first_random():
    """★F3 회귀: cand= 없이 act_prompt(tok, variant, problem, cond) 를 부르면 KeyError 로
    죽었다(옛 버그). gens.jsonl 이 적어 둔 cand 를 읽어 그대로 넘기면 죽지 않고 forward 까지
    간다 — verification_first_random 은 재계산이 불가능하므로(무작위) 기록된 값이 **유일한
    경로**다."""
    tok = CharTok()
    units = {"t::g0": {"problem": "1+1?", "gold": "2", "text": "", "cand_vf": None}}
    rows = [{"unit_id": "t::g0", "cond": "verification_first_random", "group_id": "t::g0",
             "text": PRESENT["verification_first_random"], "r_corr": 1.0, "trunc": 0.0,
             "n_tok": 10.0, "cand": "37"}]
    got, diag = M.extract(rows, units, tok, P.mock_forward_factory(dim=4), [1],
                          variant="math_opt", mode="prompt")
    assert diag["n_prompt_unrecoverable"] == 0
    assert diag["n_rows_forwarded"] == 1
    assert len(got) == 1 and set(got[0]["hidden"]) == set(M.POSITIONS)


def test_extract_skips_and_counts_rows_missing_cand_for_verification_first_random():
    """★F3: 후보값이 없고(구 gens.jsonl, 또는 기록 실패) 재계산도 불가능한(무작위 대조)
    행은 KeyError 로 죽지 않고 **버리고 센다** — 지어낸 후보값으로 넘어가지 않는다."""
    tok = CharTok()
    units = {"t::g0": {"problem": "1+1?", "gold": "2", "text": "", "cand_vf": None}}
    rows = [{"unit_id": "t::g0", "cond": "verification_first_random", "group_id": "t::g0",
             "text": PRESENT["verification_first_random"], "r_corr": 1.0, "trunc": 0.0,
             "n_tok": 10.0, "cand": None}]
    got, diag = M.extract(rows, units, tok, P.mock_forward_factory(dim=4), [1],
                          variant="math_opt", mode="prompt")
    assert diag["n_prompt_unrecoverable"] == 1
    assert diag["n_rows_forwarded"] == 0
    assert got == []


def test_extract_skips_and_counts_when_cand_key_is_entirely_absent():
    """load_gens 는 항상 "cand" 키를 넣지만(없으면 None), 방어적으로 키 자체가 없는 행도
    fail-loud KeyError 를 삼키고 스킵+카운트해야 한다(크래시 금지)."""
    tok = CharTok()
    units = {"t::g0": {"problem": "1+1?", "gold": "2", "text": "", "cand_vf": None}}
    rows = [{"unit_id": "t::g0", "cond": "verification_first_random", "group_id": "t::g0",
             "text": PRESENT["verification_first_random"], "r_corr": 1.0, "trunc": 0.0,
             "n_tok": 10.0}]                    # "cand" 키 자체가 없다
    got, diag = M.extract(rows, units, tok, P.mock_forward_factory(dim=4), [1],
                          variant="math_opt", mode="prompt")
    assert diag["n_prompt_unrecoverable"] == 1 and got == []


# ── 합성 데이터: §C1 세 숫자가 두 층을 가르는가 ────────────────────────────────

def _synthetic(n_problems: int = 24, k: int = 8, seed: int = 0) -> list[dict]:
    """층 1 = 문제 **안**에서 정오를 가르는 신호, 층 2 = 문제 **간**(난이도)만 가르는 신호.
    라벨은 문제마다 섞이게(0<pass<1) 만든다 — 그래야 문제 안 AUC 가 정의된다."""
    rs = np.random.RandomState(seed)
    rows = []
    for g in range(n_problems):
        d = (g + 1) / (n_problems + 1)                 # 문제 난이도(= 정답률)
        n_corr = max(1, min(k - 1, int(round(d * k))))
        ys = [1.0] * n_corr + [0.0] * (k - n_corr)
        for j, y in enumerate(ys):
            within = np.array([y * 2.0 + rs.randn() * 0.4, rs.randn(), rs.randn()])
            between = np.array([d * 4.0, 0.0, 0.0]) + rs.randn(3) * 0.01
            rows.append({"unit_id": f"t::g{g}#{j}", "group_id": f"t::g{g}", "cond": "substitute",
                         "r_corr": y, "trunc": 0.0, "n_tok": 100.0,
                         "hidden": {"last": {1: within, 2: between}}})
    return rows


def _cell(table, layers):
    return next(r for r in table if r["layers"] == layers and r["position"] == "last")


def test_three_way_decomposition_separates_within_from_between():
    table, trunc = M.build_table(_synthetic(), [1, 2], ["last"], seed=11, proj_dim=0)
    within, between = _cell(table, "L1"), _cell(table, "L2")
    # 문제 안에서 사는 층은 문제 안 AUC 가 높고, 문제 간에서만 사는 층은 .5 근처로 무너진다.
    assert within["auc_within"] > 0.75, within
    assert between["auc_within"] < 0.65, between
    # 그런데 **풀링** AUC 는 둘 다 높다 — 풀링만 보면 둘을 못 가른다(§C1 의 요점).
    assert between["auc_pooled"] > 0.70 and within["auc_pooled"] > 0.70
    # 난이도 ρ 는 문제 간 신호에서 크다(★문제 안에서 완벽한 자도 문제 평균은 난이도를 따라가므로
    # ρ 하나로는 둘을 못 가른다 — 가르는 것은 «문제 안 AUC» 다. 이것이 §C1 표의 존재 이유다).
    assert between["difficulty_rho"] > 0.8
    assert within["n_mixed_used"] >= 20 and np.isfinite(within["ci_lo"])
    assert trunc["substitute"] == 0.0


def test_layer_concat_beats_the_uninformative_single_layer():
    """★«여러 층» 주장은 이 표로 시험 가능해야 한다 — 이어붙인 판이 무정보 층을 이긴다."""
    table, _ = M.build_table(_synthetic(), [1, 2], ["last"], seed=11, proj_dim=0)
    cat = _cell(table, "top4")          # 층이 둘뿐이면 «상위 4개» = 있는 만큼(=둘)
    assert cat["auc_within"] > _cell(table, "L2")["auc_within"] + 0.10


def test_missing_positions_are_counted_not_imputed():
    rows = _synthetic(n_problems=12)
    for i, r in enumerate(rows):
        if i % 3 == 0:                  # 1/3 행에만 meta_end 가 있다
            r["hidden"]["meta_end"] = r["hidden"]["last"]
    table, _ = M.build_table(rows, [1], ["meta_end", "post_answer"], seed=11, proj_dim=0)
    me = next(r for r in table if r["position"] == "meta_end")
    assert me["n_rows"] == len(rows) // 3 and me["n_missing"] == len(rows) - len(rows) // 3
    pa = next(r for r in table if r["position"] == "post_answer")
    assert pa["n_rows"] == 0 and pa["n_missing"] == len(rows) and pa["layers"] == "-"


def test_nontrunc_variant_drops_and_counts_truncated_rows():
    rows = _synthetic()
    for r in rows[:40]:
        r["trunc"] = 1.0
    table, trunc = M.build_table(rows, [1], ["last"], seed=11, proj_dim=0)
    cell = _cell(table, "L1")
    assert cell["n_dropped_trunc"] == 40
    assert trunc["substitute"] == pytest.approx(40 / len(rows))


# ── [CONFOUND?] ─────────────────────────────────────────────────────────────

def _fake_table(aucs: dict) -> list[dict]:
    return [{"cond": c, "position": "last", "layers": "L1", "auc_within": a} for c, a in aucs.items()]


def test_confound_flag_fires_when_auc_tracks_truncation():
    aucs = {"a": 0.55, "b": 0.60, "c": 0.65, "d": 0.70, "e": 0.75}
    trunc = {"a": 0.05, "b": 0.10, "c": 0.15, "d": 0.20, "e": 0.25}
    cf = M.confound_check(_fake_table(aucs), trunc)
    assert cf["flag"] and cf["rho"] == pytest.approx(1.0) and cf["n_conds"] == 5


def test_confound_flag_fires_on_the_exact_inverse_too():
    """cd9 G4 는 정확도 순서가 절단률 순서의 **정확한 역순**이었다 — 그것도 잡아야 한다."""
    aucs = {"a": 0.75, "b": 0.70, "c": 0.65, "d": 0.60}
    trunc = {"a": 0.05, "b": 0.10, "c": 0.15, "d": 0.20}
    cf = M.confound_check(_fake_table(aucs), trunc)
    assert cf["flag"] and cf["rho"] == pytest.approx(-1.0)


def test_confound_flag_quiet_when_uncorrelated():
    aucs = {"a": 0.55, "b": 0.60, "c": 0.65, "d": 0.70}      # 순위 1,2,3,4
    trunc = {"a": 0.15, "b": 0.05, "c": 0.20, "d": 0.10}     # 순위 3,1,4,2 → ρ = 0
    cf = M.confound_check(_fake_table(aucs), trunc)
    assert not cf["flag"] and abs(cf["rho"]) < M.CONFOUND_RHO


def test_confound_is_undefined_with_too_few_conditions():
    cf = M.confound_check(_fake_table({"a": 0.7, "b": 0.6}), {"a": 0.1, "b": 0.2})
    assert not cf["flag"] and not np.isfinite(cf["rho"])


def test_markdown_carries_the_confound_line_and_best_within():
    aucs = {"a": 0.55, "b": 0.60, "c": 0.65, "d": 0.70}
    trunc = {"a": 0.05, "b": 0.10, "c": 0.15, "d": 0.20}
    table = [{**r, "n_rows": 50, "n_missing": 0, "trunc_rate": trunc[r["cond"]],
              "ci_lo": 0.5, "ci_hi": 0.8, "n_mixed_used": 10, "difficulty_rho": 0.1,
              "auc_pooled": 0.6, "auc_within_nontrunc": 0.6, "n_dropped_trunc": 0}
             for r in _fake_table(aucs)]
    md = M.to_markdown(table, trunc, {"n_rows_forwarded": 200}, "unit-test")
    assert "[CONFOUND?]" in md and "BEST-WITHIN: d last L1 0.700" in md


# ── 잡다한 순수 함수 ─────────────────────────────────────────────────────────

def test_layer_sets_has_singles_and_one_concat_each():
    got = M.layer_sets([18, 33, 34, 35, 36])
    assert [n for n, _s in got] == ["L18", "L33", "L34", "L35", "L36", "top4", "all"]
    assert dict(got)["top4"] == (33, 34, 35, 36)


def test_layer_sets_dedupes_when_all_equals_top4():
    assert [n for n, _s in M.layer_sets([35, 36])] == ["L35", "L36", "top4"]


def test_projector_is_seed_stable_and_off_when_small():
    assert M.projector(8, 0) is None and M.projector(8, 16) is None
    a, b = M.projector(64, 8, seed=3), M.projector(64, 8, seed=3)
    assert a.shape == (64, 8) and np.allclose(a, b)


def test_subsample_keeps_problems_balanced():
    rows = [{"group_id": f"g{i % 4}", "cond": "x", "i": i} for i in range(40)]
    got = M.subsample_rows(rows, 8)
    assert len(got) == 8 and len({r["group_id"] for r in got}) == 4


def test_load_gens_and_units_roundtrip(tmp_path):
    roll = tmp_path / "src_tag" / "texts.jsonl"
    roll.parent.mkdir()
    roll.write_text(json.dumps({"group_id": "g0", "problem": "1+1?", "gold": "2",
                                "text": "two", "r_corr": 1}) + "\n")
    gens = tmp_path / "gate" / "gens.jsonl"
    gens.parent.mkdir()
    gens.write_text(json.dumps({"unit_id": "src_tag::g0", "tag": "src_tag", "cond": "plain",
                                "gen_r_corr": 1, "truncated": 0, "n_tok": 7, "text": "hi"}) + "\n")
    (gens.parent / "gate_summary.json").write_text(json.dumps(
        {"meta": {"mode": "prompt", "variant": "math_opt", "rollouts": [str(roll)]}}))
    units, meta = M.load_units(str(gens))
    assert meta["mode"] == "prompt"
    assert units["src_tag::g0"]["problem"] == "1+1?" and units["src_tag::g0"]["text"] == ""
    assert units["src_tag::g0#0"]["text"] == "two"       # continue 단위는 롤아웃 본문을 갖는다
    row = M.load_gens(str(gens))[0]
    assert row["r_corr"] == 1.0 and row["group_id"] == "src_tag::g0"


def test_job_hidden_ats_is_additive():
    """★math_ruler_pivot.Job 확장이 기존 호출자를 안 건드린다(기본값은 빈 목록)."""
    assert P.Job([1, 2, 3], hidden_at=2).hidden_ats == []
    assert P.Job([1, 2, 3], hidden_ats=[0, 2, 9]).hidden_ats == [0, 2]   # 범위 밖은 버린다


# ── MLP 머리(grouped_oof_probe_mlp) — HSRM 대조: 천장이 «선형 머리가 약하다» 때문인지 ────
#    «신호가 없다» 때문인지 가른다. math_ruler_pivot.grouped_oof_probe_mlp 를 직접 시험한다
#    (CPU 전용, numpy 만 — torch 없음).

def _linear_groups(n_groups=40, per_group=5, seed=0, d=5):
    """y 가 특징의 선형 함수(선형 분리 가능)인 합성 데이터."""
    rng = np.random.RandomState(seed)
    w_true = rng.randn(d)
    groups, X = [], []
    for g in range(n_groups):
        for _ in range(per_group):
            X.append(rng.randn(d))
            groups.append(f"g{g}")
    X = np.array(X)
    y = (X @ w_true > 0).astype(float)
    return X, y, groups


def _xor_groups(n_groups=40, per_group=6, seed=0, d=4):
    """y = (x0>0) XOR (x1>0) — 선형으로 못 가른다(나머지 차원은 잡음)."""
    rng = np.random.RandomState(seed)
    groups, X, y = [], [], []
    for g in range(n_groups):
        for _ in range(per_group):
            x = rng.randn(d)
            groups.append(f"g{g}")
            X.append(x)
            y.append(1.0 if (x[0] > 0) != (x[1] > 0) else 0.0)
    return np.array(X), np.array(y), groups


def test_mlp_head_solves_the_easy_linearly_separable_case():
    """쉬운 경우에서 MLP 가 선형을 깨지 않는다 — 둘 다 문제 안 AUC 가 높아야 한다."""
    X, y, groups = _linear_groups()
    pr_lin = P.grouped_oof_probe(X, y, groups, seed=3)
    pr_mlp = P.grouped_oof_probe_mlp(X, y, groups, seed=3, epochs=300, lr=5e-2)
    assert P.auc(y, pr_lin["oof"]) > 0.9
    assert P.auc(y, pr_mlp["oof"]) > 0.9


def test_mlp_head_beats_linear_on_xor_like_labels():
    """★머리가 실제로 더 표현력이 큰지를 증명하는 시험 — 선형 분리 불가능한 라벨에서 MLP 가
    선형을 크게 이겨야 한다."""
    X, y, groups = _xor_groups()
    pr_lin = P.grouped_oof_probe(X, y, groups, seed=5)
    pr_mlp = P.grouped_oof_probe_mlp(X, y, groups, seed=5, hidden=32, epochs=400, lr=0.1)
    auc_lin, auc_mlp = P.auc(y, pr_lin["oof"]), P.auc(y, pr_mlp["oof"])
    assert auc_lin < 0.6, auc_lin              # 선형 머리는 XOR 을 못 푼다
    assert auc_mlp > auc_lin + 0.15, (auc_lin, auc_mlp)


def test_grouped_oof_probe_and_mlp_share_the_same_fold_split():
    """fold 배정 동일성: 같은 seed 면 grouped_oof_probe 와 grouped_oof_probe_mlp 가 같은 행을
    같은 fold 에 넣는다 — 스플리터(_group_fold_ids)를 공유해야 한다(따로 안 만든다)."""
    groups = [f"g{i // 4}" for i in range(80)]
    fold_ids_a, _fo_a, k_a = P._group_fold_ids(groups, 5, seed=7)
    fold_ids_b, _fo_b, k_b = P._group_fold_ids(groups, 5, seed=7)
    assert k_a == k_b == 5
    assert np.array_equal(fold_ids_a, fold_ids_b)

    rng = np.random.RandomState(1)
    X = rng.randn(80, 5)
    y = (X[:, 0] > 0).astype(float)
    pr_lin = P.grouped_oof_probe(X, y, groups, seed=7)
    pr_mlp = P.grouped_oof_probe_mlp(X, y, groups, seed=7, epochs=20)
    # 같은 fold 배정이면 어느 행이 out-of-fold 로 채점됐는지(=NaN 이 아닌지)도 같다.
    assert np.array_equal(np.isfinite(pr_lin["oof"]), np.isfinite(pr_mlp["oof"]))


def test_mlp_head_is_deterministic_given_fixed_seed():
    X, y, groups = _linear_groups(n_groups=20, per_group=4, seed=1)
    pr1 = P.grouped_oof_probe_mlp(X, y, groups, seed=9, epochs=50)
    pr2 = P.grouped_oof_probe_mlp(X, y, groups, seed=9, epochs=50)
    assert np.allclose(pr1["oof"], pr2["oof"], equal_nan=True)


# ── 독립 점검 수용 기준(0915) — 평문 GD(lr=1e-2, epochs=200) 는 XOR 에서 grouped 5-fold OOF
#    AUC .562(선형 .486 과 거의 같음)로 «머리가 못 푼다» 가 아니라 «최적화가 200 스텝 안에
#    안 끝난다» 였다. Adam 으로 바꾼 지금 이 다섯 기준을 **전부** 만족해야, 실제 자료에서
#    이 머리가 null 을 내는 것이 «신호가 선형으로도 비선형으로도 안 읽힌다» 를 뒷받침하지,
#    «옵티마이저가 미수렴」을 가리키지 않는다. 조건은 요청서 그대로: grouped 5-fold, seed=0,
#    n=1200, groups of 4 연속 행, 2-D 특징 — grouped_oof_probe_mlp 의 기본 하이퍼파라미터만
#    쓴다(실제 자료가 아니라 이 시험에 맞춰 튜닝했다).

def _acceptance_groups(n: int, per_group: int = 4) -> list[str]:
    return [f"g{i // per_group}" for i in range(n)]


def _acceptance_xor(seed: int = 0, n: int = 1200, per_group: int = 4):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(float)
    return X, y, _acceptance_groups(n, per_group)


def _acceptance_linear(seed: int = 0, n: int = 1200, per_group: int = 4):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    w = rng.randn(2)
    y = (X @ w > 0).astype(float)
    return X, y, _acceptance_groups(n, per_group)


def _acceptance_noise(seed: int = 0, n: int = 1200, per_group: int = 4):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2)
    y = rng.randint(0, 2, n).astype(float)          # X 와 무관한 셔플 라벨
    return X, y, _acceptance_groups(n, per_group)


def test_acceptance_1_xor_mlp_solves_it_linear_stays_at_chance():
    """기준 1: XOR 표적에서 MLP OOF AUC ≥ .90, **같은 데이터**에서 선형은 우연 근처에
    머문다 — 둘 다 확인해야 «데이터가 쉬웠다» 가 아니라 «머리가 비선형을 얻었다» 는 증명이
    된다."""
    X, y, groups = _acceptance_xor()
    auc_mlp = P.auc(y, P.grouped_oof_probe_mlp(X, y, groups, seed=0)["oof"])
    auc_lin = P.auc(y, P.grouped_oof_probe(X, y, groups, seed=0)["oof"])
    assert auc_mlp >= 0.90, auc_mlp
    assert auc_lin < 0.60, auc_lin


def test_acceptance_2_linearly_separable_no_regression():
    """기준 2: 선형 분리 가능한 표적에서 MLP OOF AUC ≥ .99 — 최적화를 더 강하게 돌려도
    쉬운 경우를 깨지 않는다."""
    X, y, groups = _acceptance_linear()
    auc_mlp = P.auc(y, P.grouped_oof_probe_mlp(X, y, groups, seed=0)["oof"])
    assert auc_mlp >= 0.99, auc_mlp


def test_acceptance_3_pure_noise_manufactures_no_signal():
    """기준 3(가장 중요한 새 시험): 라벨을 셔플한 순수 잡음에서 MLP OOF AUC 가 [.40, .60]
    안에 있어야 한다 — 머리가 지나치게 유연해서 없는 신호를 만들어내면 안 된다는 반대쪽
    실패를 지킨다."""
    X, y, groups = _acceptance_noise()
    auc_mlp = P.auc(y, P.grouped_oof_probe_mlp(X, y, groups, seed=0)["oof"])
    assert 0.40 <= auc_mlp <= 0.60, auc_mlp


def test_acceptance_4_determinism_same_seed_identical_oof_vector():
    """기준 4: 같은 seed 로 두 번 돌리면 OOF 벡터가 완전히 같다(Adam 의 모든 항이 seed 로
    결정되는 np.random.RandomState 뿐, 시각·스레드에 의존하는 난수를 안 쓴다)."""
    X, y, groups = _acceptance_xor()
    oof1 = P.grouped_oof_probe_mlp(X, y, groups, seed=0)["oof"]
    oof2 = P.grouped_oof_probe_mlp(X, y, groups, seed=0)["oof"]
    assert np.array_equal(oof1, oof2, equal_nan=True)          # 허용 오차 없이 완전히 같다


def test_acceptance_5_runtime_under_30s_for_2000_by_256():
    """기준 5: 2,000×256 합성 데이터의 grouped 5-fold OOF 가 CPU 에서 30 초 안에 끝난다."""
    rng = np.random.RandomState(0)
    n, d = 2000, 256
    X = rng.randn(n, d)
    w = rng.randn(d)
    y = (X @ w > 0).astype(float)
    groups = _acceptance_groups(n, per_group=4)
    t0 = time.time()
    P.grouped_oof_probe_mlp(X, y, groups, seed=0)
    elapsed = time.time() - t0
    assert elapsed < 30.0, elapsed


def test_mlp_head_standardizes_with_train_fold_stats_only(monkeypatch):
    """★누출 방지: 특징 0 이 fold 마다 크게 다른 상수 오프셋을 갖는다(1·10·100·1000·10000) —
    표준화가 train fold 통계만 썼다면 `_standardize` 에 넘어간 배열의 평균이 그 fold 를 뺀
    나머지로만 계산돼 전체 평균과 큰 차이가 난다(테스트 fold 가 섞였으면 그 차이가 줄어든다)."""
    groups = [f"g{i}" for i in range(30)]
    fold_ids, _fo, k = P._group_fold_ids(groups, 5, seed=8)
    offsets = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
    rng = np.random.RandomState(1)
    X = rng.randn(30, 3)
    X[:, 0] += offsets[fold_ids]
    y = (rng.randn(30) > 0).astype(float)

    calls: list[np.ndarray] = []
    orig = P._standardize

    def spy(Z):
        calls.append(np.array(Z, copy=True))
        return orig(Z)

    monkeypatch.setattr(P, "_standardize", spy)
    P.grouped_oof_probe_mlp(X, y, groups, seed=8, n_folds=5, epochs=3)
    assert len(calls) == k
    full_mean = X[:, 0].mean()
    for c in calls:
        assert c.shape[0] < len(y)                                   # test fold 는 안 들어간다
        assert abs(c[:, 0].mean() - full_mean) > 50, "표준화 입력이 전체 평균에 가깝다 — 누출"


# ── math_meta_probe --head mlp 배선 ───────────────────────────────────────────

def test_probe_cell_head_linear_is_unchanged_default():
    """head 를 안 주면(기본 "linear") 옛 동작과 같다 — 새 옵션이 기존 호출자를 안 건드린다."""
    X, y, groups = _linear_groups(n_groups=15, per_group=4, seed=2)
    got_default = M.probe_cell(X, y, groups, seed=11)
    got_explicit = M.probe_cell(X, y, groups, seed=11, head="linear")
    assert got_default["auc_within"] == pytest.approx(got_explicit["auc_within"], nan_ok=True)


def test_probe_cell_head_mlp_runs_and_returns_the_same_shape():
    X, y, groups = _linear_groups(n_groups=15, per_group=4, seed=2)
    got = M.probe_cell(X, y, groups, seed=11, head="mlp")
    assert set(got) == {"auc_pooled", "auc_within", "ci_lo", "ci_hi", "n_mixed_used",
                        "difficulty_rho", "n_problems", "n_rows", "n_folds_used"}


def test_build_table_default_head_has_no_head_column():
    """head="linear"(기본)면 표에 "head" 키가 아예 없다 — 옛 표와 바이트 단위로 같다."""
    rows = _synthetic(n_problems=12)
    table, _trunc = M.build_table(rows, [1], ["last"], seed=11, proj_dim=0)
    assert not any("head" in r for r in table)


def test_build_table_head_mlp_reports_both_heads():
    """head="mlp" 면 같은 셀에서 선형·MLP 둘 다 나와야 한다(선형 숫자가 비교로 남는다)."""
    rows = _synthetic(n_problems=12)
    table, _trunc = M.build_table(rows, [1], ["last"], seed=11, proj_dim=0, head="mlp")
    cell_l1 = [r for r in table if r["layers"] == "L1" and r["position"] == "last"]
    assert {r["head"] for r in cell_l1} == {"linear", "mlp"}
    md = M.to_markdown(table, {"substitute": 0.0}, {"n_rows_forwarded": len(rows)}, "unit-test")
    assert "| head |" in md
    assert "BEST-WITHIN:" in md and ("/linear" in md or "/mlp" in md)
