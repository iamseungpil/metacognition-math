"""cd9 사전등록 수정 6 — M_CRIT(비평 정보이득) 회귀 시험 (CPU, 모의 채점기).

프롬프트 변형(두 번째 시도 금지·블록 문법 공유) / 비평 파싱 / 누출 가드 / S+(가장 짧은 정답 형제)
선택 / IG 항의 부호·클리핑 / donor 가 **다른 문제** / 미정의 행이 그룹 중심화에서 빠짐 /
텔레메트리 키 / 중단 규칙 / 런처 dry-run(RESP_LEN 4096·math_critique_eval.py·env 전달) /
held-out 평가 요약(모의 생성기).
"""
from __future__ import annotations

import math
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_critique_eval as CE  # noqa: E402
import math_ruler_pivot as P  # noqa: E402
from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS  # noqa: E402
from src.training import math_meta as M  # noqa: E402
from src.training.critique_scorer import CritiqueScorer  # noqa: E402

TOK = P.MockTok()
GOOD = "The substitution in the middle step is unjustified; a correct approach must verify the domain first."
BAD = "Looks fine to me."


def _block(critique: str = GOOD, conf: str = "0.4", decision: str | None = "verify") -> str:
    d = f"decision: {decision}\n" if decision else ""
    return f"<meta>\nconfidence: {conf}\n{critique}\n{d}</meta>"


def _text(answer: str, critique: str = GOOD, decision: str | None = "verify") -> str:
    return f"work here \\boxed{{{answer}}}\n" + _block(critique, decision=decision) + "\n"


def _mock_scorer(fn=None):
    """모델 없는 채점기. 기본 규칙: plain −1.5 / 좋은 비평 −1.0 / donor −1.2."""
    def default(prompt, target):
        if GOOD in prompt:
            return -1.0
        if "DONOR" in prompt:
            return -1.2
        return -1.5
    return CritiqueScorer(mock_fn=fn or default, tokenizer=TOK)


# ── 1. 프롬프트 변형 ────────────────────────────────────────────────────────────
def test_math_crit_variant_has_block_grammar_and_no_second_attempt():
    p = MATH_PROMPT_VARIANTS["math_crit"]
    assert "<meta>" in p and "confidence:" in p and "decision: verify" in p
    assert "Second attempt" not in p and "LAST \\boxed" not in p
    assert "do not solve the problem again" in p
    # 블록 머리·꼬리 두 줄은 math_retry / math_opt 와 글자 그대로 같아야 같은 파서가 붙는다.
    retry = MATH_PROMPT_VARIANTS["math_retry"]
    for line in ("<meta>", "confidence: <a single number between 0 and 1>", "decision: verify",
                 "</meta>"):
        assert line in p and line in retry


def test_arm_spec_and_launcher_variant():
    spec = M.require_arm("M_CRIT")
    assert spec["variant"] == "math_crit" and spec["meta_term"] == "crit"
    assert spec["require_meta"] is True
    assert "M_CRIT" in M._CRIT_ARMS and "M_CRIT" not in M._RETRY_LIKE_ARMS


# ── 2. 블록 파싱·비평 추출 ──────────────────────────────────────────────────────
def test_parse_crit_row_extracts_critique_and_answer():
    r = M.parse_crit_row(_text("42"), "42", "prob")
    assert r["emitted"] == 1 and r["n_blocks"] == 1
    assert r["critique"] == GOOD and r["crit_words"] == M.critique_len(GOOD)
    assert r["first_answer"] == "42" and r["first_correct"] == 1 and r["r_corr"] == 1
    assert r["leaked"] == 0 and r["decision"] == "verify"
    assert r["has_second_attempt"] == 0 and r["boxed_in_meta"] == 0


def test_parse_crit_row_wrong_answer_and_no_meta():
    r = M.parse_crit_row("just \\boxed{7}", "42", "prob")
    assert r["first_correct"] == 0 and r["emitted"] == 0 and r["critique"] == ""


def test_parse_crit_row_critique_excludes_confidence_and_decision_lines():
    r = M.parse_crit_row(_text("42"), "42", "prob")
    assert "confidence" not in r["critique"] and "decision" not in r["critique"]


# ── 3. 누출 가드 ────────────────────────────────────────────────────────────────
def test_leak_guard_single_source_and_row_flag():
    assert M.critique_leaks("you should have gotten 42", "42")
    assert M.critique_leaks("the value \\boxed{3} is wrong", "42")
    assert not M.critique_leaks(GOOD, "42")
    r = M.parse_crit_row(_text("42", critique="the answer 42 is wrong"), "42", "prob")  # 자기 첫 답 누출
    assert r["leaked"] == 1


def test_gate_script_reexports_the_same_leak_guard():
    import math_critique_resolve_gate as C
    assert C.critique_leaks is M.critique_leaks and C.critique_len is M.critique_len
    assert C.assign_donors is M.assign_donors and C.NOTE_PREFIX == M.NOTE_PREFIX


# ── 4. S+ 선택 ──────────────────────────────────────────────────────────────────
def _rows_for_sibling():
    rows = [
        {"text": "x" * 50, "first_correct": 1, "truncated": 0},     # 0: 정답, 긴 것
        {"text": "x" * 10, "first_correct": 1, "truncated": 0},     # 1: 정답, **가장 짧음**
        {"text": "x" * 5, "first_correct": 1, "truncated": 1},      # 2: 정답이지만 잘림 → 제외
        {"text": "x" * 3, "first_correct": 0, "truncated": 0},      # 3: 오답 → 제외
    ]
    return rows


def test_select_s_plus_picks_shortest_correct_untruncated():
    rows = _rows_for_sibling()
    got = M.select_s_plus(rows, ["g0"] * 4)
    # ★최단 정답(1)이 표적이되, 1 자신은 차순위 정답(0)을 받는다(자기 표적 금지).
    assert got == [1, 0, 1, 1]


def test_select_s_plus_none_when_group_has_no_correct_row():
    rows = [{"text": "a", "first_correct": 0, "truncated": 0} for _ in range(3)]
    assert M.select_s_plus(rows, ["g0"] * 3) == [None, None, None]


def test_select_s_plus_is_per_group():
    rows = _rows_for_sibling() + [{"text": "y" * 9, "first_correct": 1, "truncated": 0}]
    got = M.select_s_plus(rows, ["g0"] * 4 + ["g1"])
    # ★자기 자신은 표적이 될 수 없다: 정답 행은 다른 정답 형제를, 없으면 None.
    for i, g in enumerate(got[:4]):
        assert g != i and (g is None or rows[g]["first_correct"] == 1)
    assert got[4] is None


def test_truncate_to_first_boxed():
    t = "work \\boxed{42}\n<meta>\nconfidence: 0.4\nblah\ndecision: verify\n</meta>"
    assert M.truncate_to_first_boxed(t) == "work \\boxed{42}"
    assert M.truncate_to_first_boxed("no box here") == "no box here"


# ── 5. IG 항: 부호와 클리핑 ─────────────────────────────────────────────────────
def test_crit_term_sign_and_clip():
    assert M.crit_term(0.5, 0.3, 0.05) == 1.0          # (0.2)/0.05 = 4 → clip
    assert M.crit_term(0.3, 0.5, 0.05) == -1.0
    assert math.isclose(M.crit_term(0.03, 0.01, 0.05), 0.4)
    assert M.crit_term(0.1, 0.1, 0.05) == 0.0
    assert M.crit_term(float("nan"), 0.0, 0.05) == 0.0


def _crit_rows(texts, golds, problems, uids, truncated=None):
    spec = M.require_arm("M_CRIT")
    return M.compute_rows(texts, golds, problems, "M_CRIT", uids=uids,
                          truncated=truncated or [0] * len(texts)) if spec else None


def _two_group_batch():
    """그룹 g0: 오답 1행(비평 있음) + 정답 1행(S+). 그룹 g1: 같은 모양(donor 공급)."""
    texts = [_text("7", GOOD), _text("42", BAD),
             _text("9", "DONOR critique about a different geometry setup"),
             _text("5", "DONOR critique about another algebraic identity")]
    golds = ["42", "42", "5", "5"]
    problems = ["p0", "p0", "p1", "p1"]
    uids = ["g0", "g0", "g1", "g1"]
    return texts, golds, problems, uids


def test_annotate_crit_ig_defines_rows_and_signs_positive_for_useful_critique():
    texts, golds, problems, uids = _two_group_batch()
    rows = _crit_rows(texts, golds, problems, uids)
    assert all(r["meta_defined"] == 0 for r in rows), "채점 전에는 전부 미정의여야 한다"
    M.annotate_crit_ig(rows, uids, scorer=_mock_scorer(), variant="math_crit",
                       rng=random.Random(0), weight=0.5, scale=0.05)
    r0 = rows[0]                                  # GOOD 비평: note −1.0, plain −1.5, donor −1.2
    assert r0["meta_defined"] == 1
    assert math.isclose(r0["ig"], 0.5) and math.isclose(r0["ig_donor"], 0.3)
    assert math.isclose(r0["ig_delta"], 0.2)
    assert r0["crit_term"] == 1.0 and math.isclose(r0["meta_val"], 0.5)


def test_annotate_crit_ig_negative_when_critique_hurts():
    texts, golds, problems, uids = _two_group_batch()
    rows = _crit_rows(texts, golds, problems, uids)

    def fn(prompt, target):
        if GOOD in prompt:
            return -2.0                            # 비평이 정답 형제를 **덜** 그럴듯하게 만든다
        if "DONOR" in prompt:
            return -1.4
        return -1.5
    M.annotate_crit_ig(rows, uids, scorer=_mock_scorer(fn), variant="math_crit",
                       rng=random.Random(0), scale=0.05)
    assert rows[0]["ig"] < 0 and rows[0]["crit_term"] == -1.0 and rows[0]["meta_val"] < 0


def test_annotate_crit_ig_donor_comes_from_a_different_problem():
    texts, golds, problems, uids = _two_group_batch()
    rows = _crit_rows(texts, golds, problems, uids)
    seen: list[str] = []
    M.annotate_crit_ig(rows, uids, scorer=_mock_scorer(lambda p, t: (seen.append(p), -1.0)[1]),
                       variant="math_crit", rng=random.Random(0))
    for i, r in enumerate(rows):
        if r.get("donor_idx") is not None:
            assert uids[r["donor_idx"]] != uids[i], "donor 는 **다른 문제**의 비평이어야 한다"


def test_undefined_rows_get_zero_term_and_are_excluded_from_centering():
    # 누출 비평 / 비평 없음 / 다중 블록 — 셋 다 미정의여야 한다.
    texts = [_text("7", "the answer 7 is wrong"),       # 누출(자기 첫 답)
             "no meta at all \\boxed{7}",               # 미발화
             _block() + _text("7", GOOD),               # 블록 2개(첫 답 앞에도 하나)
             _text("42", BAD)]                          # 정답 형제(S+)
    uids = ["g0"] * 4
    rows = _crit_rows(texts, ["42"] * 4, ["p0"] * 4, uids)
    M.annotate_crit_ig(rows, uids, scorer=_mock_scorer(), variant="math_crit",
                       rng=random.Random(0))
    assert rows[0]["meta_defined"] == 0 and rows[0]["meta_val"] == 0.0     # 누출
    assert rows[1]["meta_defined"] == 0 and rows[1]["meta_val"] == 0.0     # 미발화
    assert rows[2]["meta_defined"] == 0 and rows[2]["multi_block"] == 1    # 다중 블록


def test_no_correct_sibling_means_undefined():
    texts = [_text("7", GOOD), _text("8", GOOD)]
    uids = ["g0", "g0"]
    rows = _crit_rows(texts, ["42", "42"], ["p0", "p0"], uids)
    M.annotate_crit_ig(rows, uids, scorer=_mock_scorer(), variant="math_crit", rng=random.Random(0))
    assert all(r["meta_defined"] == 0 for r in rows)


def test_answer_span_reward_is_the_single_boxed():
    texts, golds, problems, uids = _two_group_batch()
    rows = _crit_rows(texts, golds, problems, uids)
    assert [r["answer_total"] for r in rows] == [0.0, 1.0, 0.0, 1.0]


def test_s_plus_target_is_truncated_to_first_boxed():
    """채점 표적(S+)에 메타 블록이 섞이면 IG 가 «형식 맞추기»를 잰다 — 박스까지만."""
    texts, golds, problems, uids = _two_group_batch()
    rows = _crit_rows(texts, golds, problems, uids)
    seen: list[str] = []
    M.annotate_crit_ig(rows, uids, scorer=_mock_scorer(lambda p, t: (seen.append(t), -1.0)[1]),
                       variant="math_crit", rng=random.Random(0))
    assert seen and all("<meta>" not in t and t.endswith("}") for t in seen)


# ── 6. 텔레메트리 ───────────────────────────────────────────────────────────────
def test_crit_telemetry_keys_and_values():
    texts, golds, problems, uids = _two_group_batch()
    rows = _crit_rows(texts, golds, problems, uids)
    M.annotate_crit_ig(rows, uids, scorer=_mock_scorer(), variant="math_crit", rng=random.Random(0))
    rep = M.telemetry(rows, arm="M_CRIT", step=7)
    for k in ("crit_rows", "crit_defined_rate", "leak_rate", "ig_mean", "ig_donor_mean",
              "ig_delta_mean", "crit_words_mean"):
        assert k in rep, k
    assert rep["crit_rows"] == 4 and rep["leak_rate"] == 0.0
    assert 0.0 < rep["crit_defined_rate"] <= 1.0
    line = M.format_tel(rep)
    assert "crit_rows=" in line and "ig_delta=" in line and "leak=" in line
    # 재시도 팔 줄은 찍히지 않아야 한다(이 팔엔 재시도가 없다).
    assert "redirect|wrong=" not in line


def test_leak_rate_denominator_is_parsed_critiques():
    rows = [M.parse_crit_row(_text("7", "the answer 7 is wrong"), "42", "p"),
            M.parse_crit_row(_text("7", GOOD), "42", "p"),
            M.parse_crit_row("no meta \\boxed{7}", "42", "p")]
    for r in rows:
        r["multi_block"] = 0
    tel = M.crit_telemetry(rows)
    assert tel["crit_rows"] == 2 and math.isclose(tel["leak_rate"], 0.5)


def test_crit_telemetry_nan_when_nothing_parsed():
    rows = [M.parse_crit_row("no meta \\boxed{7}", "42", "p")]
    rows[0]["multi_block"] = 0
    tel = M.crit_telemetry(rows)
    assert tel["crit_rows"] == 0 and math.isnan(tel["leak_rate"])


# ── 7. 중단 규칙 ────────────────────────────────────────────────────────────────
def _rep(**kw):
    base = {"step": 10, "arm": "M_CRIT", "emit_rate": 0.9, "boxed_in_meta": 0.0,
            "multi_block_rate": 0.0, "boilerplate_rate": 0.1, "n_emitted": 100,
            "leak_rate": 0.05, "crit_defined_rate": 0.8, "acc": 0.7}
    base.update(kw)
    return base


def _names(hits):
    return {h["metric"] for h in hits if h["status"] == "abort"}


def test_abort_rules_for_crit_arm():
    assert _names(M.check_abort(_rep(), arm="M_CRIT")) == set()
    assert "emit_rate" in _names(M.check_abort(_rep(emit_rate=0.1), arm="M_CRIT"))
    assert "leak_rate" in _names(M.check_abort(_rep(leak_rate=0.4), arm="M_CRIT"))
    assert "crit_defined_rate" in _names(M.check_abort(_rep(crit_defined_rate=0.2), arm="M_CRIT"))
    assert "boxed_in_meta" in _names(M.check_abort(_rep(boxed_in_meta=0.03), arm="M_CRIT"))
    assert "multi_block_rate" in _names(M.check_abort(_rep(multi_block_rate=0.2), arm="M_CRIT"))


def test_abort_warmups_for_crit_arm():
    # emit_rate·crit_defined_rate 는 step≤5, leak_rate 는 step≤3 까지 봐준다.
    assert _names(M.check_abort(_rep(step=3, emit_rate=0.0, crit_defined_rate=0.0,
                                     leak_rate=0.9), arm="M_CRIT")) == set()
    assert _names(M.check_abort(_rep(step=5, emit_rate=0.0, crit_defined_rate=0.0),
                                arm="M_CRIT")) == set()
    assert "leak_rate" in _names(M.check_abort(_rep(step=4, leak_rate=0.9), arm="M_CRIT"))
    assert "emit_rate" in _names(M.check_abort(_rep(step=6, emit_rate=0.0), arm="M_CRIT"))


def test_retry_only_rules_do_not_touch_crit_arm():
    hits = M.check_abort(_rep(), arm="M_CRIT")
    assert not any(h["metric"] in ("redirect_rate", "trunc_rate", "agree_line_rate") for h in hits)


def test_crit_rules_do_not_touch_other_arms():
    rep = {"step": 10, "arm": "M_G1", "emit_rate": 0.9, "boxed_in_meta": 0.0,
           "multi_block_rate": 0.0, "boilerplate_rate": 0.1, "n_emitted": 100, "acc": 0.7}
    hits = M.check_abort(rep, arm="M_G1")
    assert not any(h["metric"] in ("leak_rate", "crit_defined_rate") for h in hits)


# ── 8. 런처 dry-run ─────────────────────────────────────────────────────────────
def test_run_math_arm_crit_defaults_resp_len_4096_and_uses_critique_eval():
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_CRIT", "2", "30", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "ARM=M_CRIT" in out and "VARIANT=math_crit" in out
    assert "RESP_LEN=4096" in out
    assert "EVAL_SCRIPT=math_critique_eval.py" in out
    assert "math_critique_eval.py" in out and "math500_crit_8k" in out
    # ★env 전달: 네 MATH_CRIT_* 값이 런처에서 export 돼 Ray env 목록으로 간다.
    assert "MATH_CRIT_W=0.5" in out and "MATH_CRIT_SCALE=0.05" in out
    assert "MATH_CRIT_MAX_TOK=2048" in out


def test_ray_env_forwards_math_crit_vars():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    for k in ("MATH_CRIT_W", "MATH_CRIT_SCALE", "MATH_CRIT_MAX_TOK", "MATH_CRIT_SCORER_PATH"):
        assert f'"{k}"' in src, f"{k} 가 Ray runtime_env 목록에 없다(워커가 못 읽는다)"


def test_stash_wires_crit_scorer_and_truncated():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    assert "_mm._NEEDS_TRUNC_TERMS" in src
    assert "annotate_crit_ig" in src and "get_scorer" in src


# ── 9. held-out 평가(모의 생성기) ───────────────────────────────────────────────
def _mock_generate(per_problem):
    def gen(prompts):
        assert len(prompts) == len(per_problem)
        return [[(t, 0, len(t) // 4) for t in texts] for texts in per_problem]
    return gen


def test_eval_summary_on_mocked_generations():
    problems = [{"problem": "p0", "gold": "42"}, {"problem": "p1", "gold": "5"}]
    per_problem = [[_text("42", GOOD), _text("7", GOOD)],
                   [_text("5", GOOD), _text("9", "the answer 9 is not trustworthy")]]
    rows, tel = CE.evaluate(problems, _mock_generate(per_problem), num_samples=2)
    assert len(rows) == 4 and tel["n_groups"] == 2
    assert math.isclose(tel["first_acc"], 0.5)
    assert tel["emit_rate"] == 1.0 and tel["crit_rows"] == 4
    assert math.isclose(tel["leak_rate"], 0.25)          # 네 비평 중 하나가 답을 흘렸다
    assert "selectivity_mixed" in tel and tel["n_mixed_problems"] == 2
    # 재풀이 대상 = 첫 답 오답 ∧ 비평 성립(누출 제외) → 한 행(\boxed{7})뿐
    idxs = CE.wrong_rows(rows)
    assert idxs == [1]


def test_eval_resolve_metrics_with_mocked_resolver():
    problems = [{"problem": "p0", "gold": "42"}, {"problem": "p1", "gold": "5"}]
    per_problem = [[_text("42", GOOD), _text("7", GOOD)],
                   [_text("5", GOOD), _text("9", GOOD)]]
    rows, tel = CE.evaluate(problems, _mock_generate(per_problem), num_samples=2)
    idxs = CE.wrong_rows(rows)
    assert len(idxs) == 2

    def resolver(prompts):
        # 비평이 붙은 프롬프트면 다 맞히고, blind 면 절반만 맞힌다 → rescue_delta = +0.5
        out = []
        for q in prompts:
            gold = "42" if "p0" in q else "5"
            if GOOD in q:
                out.append([f"\\boxed{{{gold}}}", f"\\boxed{{{gold}}}"])
            else:
                out.append([f"\\boxed{{{gold}}}", "\\boxed{0}"])
        return out
    summ = CE.resolve_eval(rows, TOK, "math_crit", resolver, idxs=idxs)
    assert summ["n_resolved"] == 2
    assert math.isclose(summ["rescue_crit"], 1.0) and math.isclose(summ["rescue_blind"], 0.5)
    assert math.isclose(summ["rescue_delta"], 0.5) and summ["frac_crit_gt_blind"] == 1.0


def test_eval_resolve_prompts_differ_only_by_the_note():
    rows = [M.parse_crit_row(_text("7", GOOD), "42", "p0")]
    rows[0].update({"crit_ok": 1, "group_id": "g0"})
    prompts, index = CE.build_resolve_requests(rows, TOK, "math_crit", [0])
    assert index == [(0, "crit"), (0, "blind")]
    assert prompts[0].replace(M.NOTE_PREFIX + GOOD, "", 1) == prompts[1]


def test_eval_ig_uses_the_same_training_definition():
    problems = [{"problem": "p0", "gold": "42"}, {"problem": "p1", "gold": "5"}]
    per_problem = [[_text("42", BAD), _text("7", GOOD)],
                   [_text("5", BAD), _text("9", "DONOR critique of another problem")]]
    rows, _ = CE.evaluate(problems, _mock_generate(per_problem), num_samples=2)
    got = CE.ig_eval(rows, _mock_scorer(), seed=0)
    assert set(got) == {"crit_defined_rate", "ig_mean", "ig_donor_mean", "ig_delta_mean"}
    assert got["crit_defined_rate"] > 0


# ── 10. 채점기 ──────────────────────────────────────────────────────────────────
def test_scorer_mock_mode_and_length_check():
    sc = _mock_scorer()
    assert sc.score_meanlogp(["a " + GOOD, "b"], ["t", "t"]) == [-1.0, -1.5]
    try:
        sc.score_meanlogp(["a"], ["t", "t"])
    except RuntimeError as e:
        assert "prompts" in str(e)
    else:
        raise AssertionError("길이 불일치는 즉사해야 한다")


def test_scorer_path_requires_explicit_source(monkeypatch):
    import src.training.critique_scorer as CS
    monkeypatch.delenv("MATH_CRIT_SCORER_PATH", raising=False)
    try:
        CS.scorer_path(None)
    except RuntimeError as e:
        assert "채점기 경로가 없다" in str(e)
    else:
        raise AssertionError("경로 없이 조용히 기본값을 쓰면 안 된다")
    monkeypatch.setenv("MATH_CRIT_SCORER_PATH", "/x/y")
    assert CS.scorer_path("/ignored") == "/x/y"


def test_crit_weight_and_scale_env(monkeypatch):
    monkeypatch.delenv("MATH_CRIT_W", raising=False)
    monkeypatch.delenv("MATH_CRIT_SCALE", raising=False)
    assert M.crit_weight() == 0.5 and M.crit_scale() == 0.05
    monkeypatch.setenv("MATH_CRIT_W", "0.25")
    monkeypatch.setenv("MATH_CRIT_SCALE", "0.1")
    assert M.crit_weight() == 0.25 and M.crit_scale() == 0.1


def test_select_s_plus_never_returns_self():
    """정답 행의 표적은 자기 자신이 아닌 형제여야 한다(자기 서술 IG 구멍 봉쇄)."""
    from src.training import math_meta as MM
    rows = [{"first_correct": 1, "truncated": 0, "text": "a" * 10},
            {"first_correct": 1, "truncated": 0, "text": "b" * 50},
            {"first_correct": 0, "truncated": 0, "text": "c" * 5}]
    out = MM.select_s_plus(rows, ["g", "g", "g"])
    assert out[0] == 1 and out[1] == 0 and out[2] == 0
    assert MM.select_s_plus([rows[0]], ["g"]) == [None]
