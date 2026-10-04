"""cd9 0914d — M_DIS(불일치 진단) 회귀 시험 (CPU, 모의 생성기).

스케치(메타 제거·꼬리 길이·답 줄) / 커밋 파싱 / 자기증류 크레딧 진리표(gold 미참조 포함) /
진단 플래그 / 프롬프트 변형(시스템은 math_opt 와 동일, 사용자 턴 어서션) / parquet 빌더
(합성 jsonl) / 팔 명세와 두 대조군의 차이 / verl_sdc 배선(스태시·Ray env) / 텔레메트리·중단
규칙 / 런처 dry-run(RESP_LEN 2048·MAX_PROMPT 2048·math_dis_eval.py) / held-out 평가(모의
생성기) / gate_judgment 가 이 평가의 요약 키를 읽는가.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import build_math_dis_parquet as BP  # noqa: E402
import math_dis_eval as DE  # noqa: E402
from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS, build_math_prompt  # noqa: E402
from src.training import math_dis as MD  # noqa: E402
from src.training import math_meta as M  # noqa: E402

DIAG = ("Candidate 1 and Candidate 3 disagree at the substitution step; Candidate 2 drops a "
        "factor of two before the final division.")


def _block(diag: str = DIAG, commit: str | None = "3") -> str:
    c = f"Commit: {commit}\n" if commit is not None else ""
    return f"<meta>\n{diag}\n{c}</meta>"


def _text(answer: str, diag: str = DIAG, commit: str | None = "3") -> str:
    return _block(diag, commit) + f"\nSo the answer is \\boxed{{{answer}}}\n"


def _row(answer: str, cands, commit: str | None = "3", diag: str = DIAG, **kw) -> dict:
    return MD.parse_dis_row(_text(answer, diag, commit), kw.pop("gold", "42"),
                            kw.pop("problem", "p0"), cands, kw.pop("cand_correct", None))


# ── 1. 스케치 ───────────────────────────────────────────────────────────────────
def test_make_sketch_strips_meta_and_appends_answer_line():
    t = ("first step\n" + _block("some earlier meta talk", "2")
         + "\nthen more work \\boxed{7}\n trailing")
    s = MD.make_sketch(t)
    assert "<meta>" not in s and "some earlier meta talk" not in s
    assert s.endswith("Final answer: \\boxed{7}")
    assert "then more work" in s and "first step" in s


def test_make_sketch_takes_the_tail_before_the_last_boxed():
    body = "x" * 1200
    s = MD.make_sketch(f"{body} \\boxed{{5}} tail after")
    head = s.split("\nFinal answer:")[0]
    # 마지막 \boxed **앞** SKETCH_CHARS 자만(그 뒤 꼬리는 들어오지 않는다)
    assert len(head) <= MD.SKETCH_CHARS and "tail after" not in head
    assert head.endswith("x")


def test_make_sketch_raises_when_candidate_has_no_boxed_answer():
    r"""0914 리뷰 D3(fail-loud): 예전엔 \boxed 없는 후보에 빈 `\boxed{}` 를 붙여 넘어갔다 —
    그러면 잘린 롤아웃이 «빈 답» 후보가 되어 자기 군집을 이뤄 다수결 동점을 만들거나 이기고,
    커밋되면 미정의가 아니라 −1(오답)로 채점된다. 이제는 답 없는 후보를 건네면 즉사한다 —
    호출자(builder/eval)가 후보로 뽑기 전에 걸러야 한다."""
    try:
        MD.make_sketch("y" * 900)
    except ValueError as e:
        assert "boxed" in str(e).lower() or "\\boxed" in str(e)
    else:
        raise AssertionError("\\boxed 없는 후보는 make_sketch 가 즉사해야 한다")


# ── 2. 커밋 파싱 ────────────────────────────────────────────────────────────────
def test_parse_commit_ok_out_of_range_and_missing():
    assert MD.parse_commit("<meta>\nblah\nCommit: 2\n</meta>") == 2
    assert MD.parse_commit("commit:4") == 4
    assert MD.parse_commit("Commit: 5") is None          # 범위 밖(N_CAND=4)
    assert MD.parse_commit("Commit: 0") is None
    assert MD.parse_commit("<meta>\nno commit here\n</meta>") is None
    assert MD.parse_commit("") is None
    # 첫 매치가 구속력을 갖는다 — 나중에 바꿔 쓰면 말 바꾸기가 공짜가 된다
    assert MD.parse_commit("Commit: 1 ... Commit: 4") == 1


# ── 3. 자기증류 크레딧 진리표 ────────────────────────────────────────────────────
def _group(answers):
    return [{"final_answer": a} for a in answers]


def test_credit_plus_one_when_commit_matches_group_plurality():
    r = _row("42", ["9", "9", "42", "7"], commit="3")
    assert MD.dis_row_credit(r, _group(["42", "42", "7"])) == (1.0, True)


def test_credit_minus_one_when_commit_differs_from_plurality():
    r = _row("9", ["9", "9", "42", "7"], commit="1")
    assert MD.dis_row_credit(r, _group(["42", "42", "7"])) == (-1.0, True)


def test_credit_uses_math_equivalence_not_string_equality():
    r = _row("0.5", ["\\frac{1}{2}", "3", "4", "5"], commit="1")
    assert MD.dis_row_credit(r, _group(["0.5", "0.5", "3"])) == (1.0, True)


def test_credit_undefined_cases_are_all_member_false():
    g = _group(["42", "42", "7"])
    cands = ["9", "9", "42", "7"]
    # (a) 커밋 미파싱
    assert MD.dis_row_credit(_row("42", cands, commit=None), g) == (0.0, False)
    # (b) 범위 밖
    assert MD.dis_row_credit(_row("42", cands, commit="7"), g) == (0.0, False)
    # (c) 메타 블록 없음
    no_meta = MD.parse_dis_row("just work \\boxed{42}", "42", "p0", cands)
    assert MD.dis_row_credit(no_meta, g) == (0.0, False)
    # (d) 블록 둘
    multi = MD.parse_dis_row(_block() + "\n" + _block() + "\n\\boxed{42}", "42", "p0", cands)
    assert multi["multi_block"] == 1 and MD.dis_row_credit(multi, g) == (0.0, False)
    # (e) 메타 안 \boxed(답 누출)
    leak = MD.parse_dis_row(_block("the answer must be \\boxed{42}", "3") + "\n\\boxed{42}",
                            "42", "p0", cands)
    assert leak["boxed_in_meta"] == 1 and MD.dis_row_credit(leak, g) == (0.0, False)


def test_credit_undefined_when_group_plurality_is_a_tie_or_empty():
    r = _row("42", ["9", "9", "42", "7"], commit="3")
    assert MD.dis_row_credit(r, _group(["42", "7"])) == (0.0, False)       # 1:1 동률
    assert MD.dis_row_credit(r, _group(["", ""])) == (0.0, False)          # 답이 없다
    assert MD.dis_row_credit(r, _group(["42", "42", "7", "7"])) == (0.0, False)


def test_credit_never_consults_gold():
    """gold 도 cand_correct 도 **없는** 행으로 불러도 같은 값이 나와야 한다 — 메타 크레딧이
    gold 를 읽으면 «판단»이 아니라 «정답 복사»를 보상하게 된다."""
    r = _row("42", ["9", "9", "42", "7"], commit="3")
    for k in ("gold", "cand_correct", "r_corr"):
        r.pop(k, None)
    assert MD.dis_row_credit(r, _group(["42", "42", "7"])) == (1.0, True)
    # 다수답이 **오답**이어도 크레딧은 +1 이다(라벨은 gold 가 아니라 형제 다수결이다)
    r2 = _row("9", ["9", "9", "42", "7"], commit="1")
    r2.pop("gold", None)
    assert MD.dis_row_credit(r2, _group(["9", "9", "42"])) == (1.0, True)


# ── 3b. LEAVE-ONE-OUT (0914 리뷰 D1) ────────────────────────────────────────────
def test_credit_is_leave_one_out_self_vote_must_not_resolve_a_tie():
    r"""자기 자신을 다수결에 포함시키면(버그) 형제끼리는 진짜로 동률인데 자기 표가 캐스팅보트가
    되어 자기 커밋과 같은 답이 «다수» 로 뽑히고, 그 다수와 같다며 크레딧 +1 을 받는다(순응
    유인·conformity attractor). LOO(자기 제외)면 형제 둘은 정말로 동률이라 미정의다."""
    row = _row("9", ["9", "42", "7", "3"], commit="1")     # 자기 답도 "9", 커밋도 "9"
    siblings = _group(["9", "42"])                          # 형제만 보면 1:1 동률
    whole_group_with_self = [row] + siblings                # 버그: 자기 자신이 literally 들어 있다
    # (대조) 예전 버그라면 plurality_answer(["9","9","42"]) = "9"(2 vs 1) → 자기 커밋과 일치 → (1.0, True)
    assert MD.plurality_answer([r.get("final_answer") for r in whole_group_with_self]) == "9"
    # LOO(정답): 자기 자신을 빼면 형제는 진짜 동률 → 미정의
    assert MD.dis_row_credit(row, whole_group_with_self) == (0.0, False)
    # self_idx 로 명시해도 같다(자기 자신은 인덱스 0)
    assert MD.dis_row_credit(row, whole_group_with_self, self_idx=0) == (0.0, False)
    # (대조) 애초에 형제만(이미 LOO 된 입력) 주면 당연히 같은 결과
    assert MD.dis_row_credit(row, siblings) == (0.0, False)


def test_credit_identity_not_equality_finds_self_among_equal_looking_dicts():
    r"""verl 행은 딕셔너리라 값이 우연히 같을 수 있다 — `==` 로 자기 자신을 찾으면(`list.index`
    류) «가짜 쌍둥이»(값은 같지만 다른 객체)를 엉뚱하게 지운다. object identity(`is`)로 찾아야
    **진짜** 자기 자신만 빠진다."""
    row = _row("9", ["9", "42", "7", "3"], commit="1")      # 자기 답 "9", 커밋 "9"
    twin = dict(row)                                         # 값은 완전히 같은 다른 객체
    siblings = _group(["42", "42"])                          # 형제 답은 "42" 로 통일
    group = siblings + [twin, row]                           # 진짜 자기 자신은 맨 끝(인덱스 3)
    # identity 로 `row`(마지막 원소)만 뺀다 — twin 은 (값은 같지만) 살아남아 형제로 집계된다
    credit, defined = MD.dis_row_credit(row, group)
    # 남은 것: ["42","42", twin("9")] → 다수 "42"(2 vs 1) — 커밋 "9" 와 다르므로 −1
    assert (credit, defined) == (-1.0, True)
    # self_idx 로 **진짜** 위치(3)를 명시해도 같은 결과
    assert MD.dis_row_credit(row, group, self_idx=3) == (-1.0, True)


def test_credit_undefined_when_loo_leaves_fewer_than_two_answered_siblings():
    r"""자기 자신을 뺀 뒤 답 있는 형제가 2 미만이면(다수를 논할 수 없다) 미정의다 — 형제가
    하나뿐이거나(동률 여부와 무관하게) 아예 없을 때."""
    row = _row("9", ["9", "42", "7", "3"], commit="1")
    one_sib = [row] + _group(["42"])                # 자기 제외 뒤 형제 1명뿐
    assert MD.dis_row_credit(row, one_sib) == (0.0, False)
    assert MD.dis_row_credit(row, one_sib, self_idx=0) == (0.0, False)
    no_sib = [row]                                   # 자기 혼자인 그룹
    assert MD.dis_row_credit(row, no_sib) == (0.0, False)


# ── 4. 진단 플래그 ──────────────────────────────────────────────────────────────
def test_dis_row_flags():
    r = _row("42", ["9", "9", "42", "7"], commit="3")
    f = MD.dis_row_flags(r)
    assert f == {"has_meta": 1, "commit_parsed": 1, "boxed_in_meta": 0, "multi_block": 0,
                 "cites_two_plus": 1, "commit_is_minority": 1}   # 후보 다수는 "9", 커밋은 "42"
    # 후보 다수답을 골랐으면 minority 가 아니다
    f2 = MD.dis_row_flags(_row("9", ["9", "9", "42", "7"], commit="1"))
    assert f2["commit_is_minority"] == 0
    # 한 후보만 인용하면 cites_two_plus 가 아니다
    f3 = MD.dis_row_flags(_row("42", ["9", "9", "42", "7"], diag="Candidate 3 looks right."))
    assert f3["cites_two_plus"] == 0 and f3["commit_parsed"] == 1
    # 후보 다수가 동률이면 «소수»를 말할 수 없다
    f4 = MD.dis_row_flags(_row("42", ["9", "7", "42", "1"], commit="3"))
    assert f4["commit_is_minority"] == 0


def test_all_agree_and_plurality():
    assert MD.all_agree(["7", "7.0", "7", "7"]) == 1
    assert MD.all_agree(["7", "8", "7", "7"]) == 0
    assert MD.all_agree(["7", "", "7", "7"]) == 0
    assert MD.plurality_answer(["7", "7", "8"]) == "7"
    assert MD.plurality_answer(["7", "8"]) is None
    assert MD.plurality_answer(["", ""]) is None


# ── 5. 프롬프트 변형 ────────────────────────────────────────────────────────────
def test_math_dis_variant_shares_the_math_opt_system_prompt():
    assert MATH_PROMPT_VARIANTS["math_dis"] == MATH_PROMPT_VARIANTS["math_opt"]


def test_build_math_prompt_requires_the_candidate_marker():
    turn = MD.build_dis_user_turn("2+2?", ["a", "b", "c", "d"])
    msgs = build_math_prompt(turn, "math_dis")
    assert msgs[0]["content"] == MATH_PROMPT_VARIANTS["math_opt"]
    assert MD.CANDIDATE_MARKER in msgs[1]["content"] and "[Candidate 4]" in msgs[1]["content"]
    assert "Commit: <candidate number>" in msgs[1]["content"]
    try:
        build_math_prompt("2+2?", "math_dis")
    except ValueError as e:
        assert "Candidate 1" in str(e)
    else:
        raise AssertionError("후보 없는 사용자 턴은 즉사해야 한다")


def test_build_dis_user_turn_requires_exactly_n_cand():
    try:
        MD.build_dis_user_turn("p", ["a", "b", "c"])
    except ValueError as e:
        assert "N_CAND" in str(e)
    else:
        raise AssertionError("후보 수가 다르면 즉사해야 한다")


# ── 6. parquet 빌더(합성 jsonl) ─────────────────────────────────────────────────
def _write_rollouts(path: Path, spec):
    with path.open("w") as fh:
        for gi, (problem, answers) in enumerate(spec):
            for ai, a in enumerate(answers):
                fh.write(json.dumps({
                    "group_id": f"g{gi}", "problem": problem, "gold": "42",
                    "text": f"work {ai} \\boxed{{{a}}}", "final_answer": a,
                    "r_corr": int(a == "42"), "truncated": 0, "n_tok": 10}) + "\n")


def _src_rows():
    return [{"data_source": "math_meta", "prompt": [], "problem": "p0", "gold": "42",
             "reward_model": {"style": "rule", "ground_truth": "42"},
             "extra_info": {"problem": "p0", "gold": "42", "level": "Level 5"}},
            {"data_source": "math_meta", "prompt": [], "problem": "p1", "gold": "42",
             "reward_model": {"style": "rule", "ground_truth": "42"},
             "extra_info": {"problem": "p1", "gold": "42", "level": "Level 5"}},
            {"data_source": "math_meta", "prompt": [], "problem": "p2-no-rollouts", "gold": "1",
             "reward_model": {"style": "rule", "ground_truth": "1"},
             "extra_info": {"problem": "p2-no-rollouts", "gold": "1", "level": "Level 5"}}]


def test_builder_on_synthetic_rollouts(tmp_path):
    jl = tmp_path / "texts.jsonl"
    _write_rollouts(jl, [("p0", ["42", "42", "42", "42", "9"]),      # 전부 일치
                         ("p1", ["42", "9", "7", "42", "9"])])       # 불일치
    cands = BP.load_candidates(str(jl))
    assert set(cands) == {"p0", "p1"} and len(cands["p0"]) == 5
    train, val, stats = BP.build_records(_src_rows(), cands, val_n=1, seed=0)
    recs = train + val
    assert stats["n_kept"] == 2 and stats["n_no_cand"] == 1
    assert math.isclose(stats["all_agree_frac"], 0.5)   # 두 문제 중 하나만 전부 일치
    for r in recs:
        assert len(r["cand_answers"]) == MD.N_CAND and len(r["cand_correct"]) == MD.N_CAND
        assert r["cand_agree_all"] in (0, 1)
        assert r["extra_info"]["cand_answers"] == r["cand_answers"]
        assert r["extra_info"]["prompt_variant"] == "math_dis"
        assert r["prompt"][0]["content"] == MATH_PROMPT_VARIANTS["math_opt"]
        user = r["prompt"][1]["content"]
        assert user.count("[Candidate ") == MD.N_CAND and "Final answer:" in user
        # ★후보의 gold 정오는 **실려는 있되** 프롬프트엔 절대 새지 않는다(지표 전용).
        assert "cand_correct" not in user
    p0 = [r for r in recs if r["problem"] == "p0"][0]
    assert p0["cand_answers"] == ["42"] * 4 and p0["cand_agree_all"] == 1
    p1 = [r for r in recs if r["problem"] == "p1"][0]
    assert p1["cand_answers"] == ["42", "9", "7", "42"] and p1["cand_agree_all"] == 0
    assert p1["cand_correct"] == [1, 0, 0, 1]


def test_builder_drops_problem_with_fewer_than_n_cand_usable_candidates(tmp_path):
    r"""0914 리뷰 D3: 잘렸거나(\boxed 없음) 답이 빈 롤아웃은 후보로 세지 않는다. 3개는 쓸
    만하고 1개는 잘렸으면(N_CAND=4 에 못 미친다) 그 문제 전체를 드롭하고 센다
    (n_dropped_truncated) — 빈 답 후보가 몰래 들어가 자기 군집을 이루고 다수결 동점을
    만들거나 이기는 것을 막는다."""
    jl = tmp_path / "texts.jsonl"
    with jl.open("w") as fh:
        for ai, a in enumerate(["42", "9", "7"]):           # 쓸 만한 후보 3개
            fh.write(json.dumps({"group_id": "g0", "problem": "p0", "gold": "42",
                                 "text": f"work {ai} \\boxed{{{a}}}", "final_answer": a,
                                 "r_corr": int(a == "42"), "truncated": 0, "n_tok": 10}) + "\n")
        fh.write(json.dumps({"group_id": "g0", "problem": "p0", "gold": "42",              # 잘림
                             "text": "work truncated, no box here",
                             "final_answer": "", "r_corr": 0, "truncated": 1, "n_tok": 10}) + "\n")
    cands = BP.load_candidates(str(jl))
    assert len(cands["p0"]) == 4 and cands["p0"][3]["truncated"] == 1   # 원 목록엔 4개가 다 있다
    train, val, stats = BP.build_records([_src_rows()[0]], cands, val_n=0, seed=0)
    assert train == [] and val == []
    assert stats["n_kept"] == 0 and stats["n_dropped_truncated"] == 1 and stats["n_no_cand"] == 0


def test_builder_cli_writes_both_parquets(tmp_path):
    import pandas as pd
    jl = tmp_path / "texts.jsonl"
    _write_rollouts(jl, [("p0", ["42"] * 5), ("p1", ["42", "9", "7", "42", "9"])])
    src = tmp_path / "src.parquet"
    pd.DataFrame(_src_rows()).to_parquet(src, index=False)
    rc = BP.main(["--rollouts", str(jl), "--train_parquet", str(src),
                  "--out_dir", str(tmp_path), "--val_n", "1"])
    assert rc == 0
    tp = pd.read_parquet(tmp_path / "math_train_math_dis.parquet")
    vp = pd.read_parquet(tmp_path / "math_val_math_dis.parquet")
    assert len(tp) == 1 and len(vp) == 1
    for df in (tp, vp):
        for col in ("cand_answers", "cand_correct", "cand_agree_all", "prompt", "problem", "gold"):
            assert col in df.columns


# ── 7. 팔 명세와 두 대조군 ──────────────────────────────────────────────────────
def test_arm_specs_present_and_controls_differ_only_as_specified():
    base = M.MATH_ARM_SPECS["M_DIS"]
    assert base["variant"] == "math_dis" and base["meta_term"] == "dis" and base["require_meta"]
    for arm, term in (("M_DIS_RAND", "dis_shuffled"), ("M_DIS0", "dis_zero")):
        s = M.MATH_ARM_SPECS[arm]
        assert s["variant"] == base["variant"], "대조군은 프롬프트가 같아야 한다"
        assert s["meta_term"] == term
    # ★M_DIS0 만 require_meta=False (크레딧 0 인 팔에서 발화 감소는 관측 대상이지 사고가 아니다)
    assert M.MATH_ARM_SPECS["M_DIS_RAND"]["require_meta"] is True
    assert M.MATH_ARM_SPECS["M_DIS0"]["require_meta"] is False
    assert M._DIS_ARMS == {"M_DIS", "M_DIS_RAND", "M_DIS0"}


def _rows_for(arm, answers=("42", "42", "42", "9"), commit="3"):
    # ★기본 answers 는 LOO(0914 리뷰 D1) 로 자기 자신을 빼도 항상 남은 셋이 "42" 로 다수를
    #   이룬다(2:1 이상) — 3개 짜리("42","42","9")면 "42" 행이 자기를 빼는 순간 나머지가
    #   "42"/"9" 1:1 동률이 되어 미정의가 된다. credit/defined 값을 직접 검증하는 테스트는
    #   이 기본값을 쓴다.
    cands = ["9", "9", "42", "7"]
    texts = [_text(a, DIAG, commit) for a in answers]
    return M.compute_rows(texts, ["42"] * len(texts), ["p0"] * len(texts), arm,
                          uids=["g0"] * len(texts),
                          cand_answers=[cands] * len(texts),
                          cand_correct=[[0, 0, 1, 1]] * len(texts))


def test_m_dis_rows_credit_and_weight(monkeypatch):
    monkeypatch.delenv("MATH_DIS_W", raising=False)
    rows = _rows_for("M_DIS")
    assert all(r["dis_defined"] == 1 and r["meta_defined"] == 1 for r in rows)
    assert all(math.isclose(r["dis_credit"], 1.0) for r in rows)   # 커밋 "42" = 그룹 다수답
    assert all(math.isclose(r["meta_val"], 0.5) for r in rows)     # MATH_DIS_W 기본 0.5
    assert all(math.isclose(r["answer_total"], float(r["r_corr"])) for r in rows)
    monkeypatch.setenv("MATH_DIS_W", "0.25")
    assert math.isclose(_rows_for("M_DIS")[0]["meta_val"], 0.25)


def test_m_dis0_has_zero_meta_value_but_identical_rows(monkeypatch):
    monkeypatch.setenv("MATH_DIS_W", "0.5")
    base, zero = _rows_for("M_DIS"), _rows_for("M_DIS0")
    assert all(r["meta_val"] == 0.0 for r in zero)
    # 크레딧·정의 여부·답 보상은 바이트 동일 — 갈리는 것은 메타 스팬에 얹히는 값뿐이다.
    for a, b in zip(base, zero):
        assert a["dis_credit"] == b["dis_credit"] and a["dis_defined"] == b["dis_defined"]
        assert a["answer_total"] == b["answer_total"]


def test_m_dis_rand_flips_signs_per_uid_group_not_per_row():
    r"""0914 리뷰 D2: M_DIS_RAND 는 M_RAND/M_RETRY_RAND 와 같은 **그룹 단위** 부호 반전이어야
    한다(행 단위로 뒤집으면 같은 그룹 안 크레딧이 갈려 그룹 중심화 뒤 분산이 M_DIS 와 달라진다).
    그룹마다 4행(자기 제외 뒤에도 "42" 가 3개 남아 항상 다수) — LOO 로 자기를 빼도 크레딧이
    항상 +1 로 정의되도록 고정한 뒤, RAND 가 **그룹 단위**로만 부호를 흔드는지를 본다."""
    n_groups, group_size = 10, 4
    n = n_groups * group_size
    answers = ["42"] * n
    cands = ["9", "9", "42", "7"]
    uids = [f"g{i}" for i in range(n_groups) for _ in range(group_size)]
    kw = dict(uids=uids, cand_answers=[cands] * n, cand_correct=[[0, 0, 1, 1]] * n)
    base = M.compute_rows([_text(a) for a in answers], ["42"] * n, ["p0"] * n, "M_DIS", **kw)
    assert all(math.isclose(r["dis_credit"], 1.0) for r in base)   # 픽스처 전제 확인
    rand = M.compute_rows([_text(a) for a in answers], ["42"] * n, ["p0"] * n, "M_DIS_RAND", **kw)
    for gi in range(n_groups):
        block = rand[gi * group_size:(gi + 1) * group_size]
        assert len({r["dis_credit"] for r in block}) == 1, "같은 uid 그룹은 같은 부호를 받아야 한다"
    signs = {r["dis_credit"] for r in rand}
    assert signs == {1.0, -1.0}, "그룹 사이에는 부호가 갈려야 한다(전부 같은 부호면 대조군이 아니다)"
    # 고정 시드 — 다시 계산해도 같은 부호열(스텝이 달라져도 재현돼야 한다는 스펙 지시)
    rand2 = M.compute_rows([_text(a) for a in answers], ["42"] * n, ["p0"] * n, "M_DIS_RAND", **kw)
    assert [r["dis_credit"] for r in rand2] == [r["dis_credit"] for r in rand]
    # 정의된 행 집합은 M_DIS 와 같다(반전은 부호만이다)
    assert [r["meta_defined"] for r in rand] == [r["meta_defined"] for r in base]


def test_undefined_rows_are_excluded_from_group_centering():
    """미정의 행은 meta_val 0 이고 member 에서 빠진다 — verl_sdc._math_add_meta_region_advantage
    가 읽는 계약(감사 3)."""
    cands = ["9", "9", "42", "7"]
    texts = [_text("42", DIAG, "3"), "no meta at all \\boxed{42}", _text("42", DIAG, "3")]
    rows = M.compute_rows(texts, ["42"] * 3, ["p0"] * 3, "M_DIS", uids=["g0"] * 3,
                          cand_answers=[cands] * 3, cand_correct=[[0, 0, 1, 1]] * 3)
    assert [r["meta_defined"] for r in rows] == [1, 0, 1]
    assert rows[1]["meta_val"] == 0.0
    from src.training.dcpo_region import group_mean_subtract
    centered = group_mean_subtract([r["meta_val"] for r in rows], ["g0"] * 3,
                                   member=[r["meta_defined"] for r in rows]).reshape(-1)
    assert float(centered[1]) == 0.0


# ── 8. 텔레메트리 · 중단 규칙 ───────────────────────────────────────────────────
def test_dis_telemetry_keys_and_values():
    rows = _rows_for("M_DIS", answers=("42", "42", "9"))
    tel = M.telemetry(rows, arm="M_DIS", step=10)
    for k in ("dis_rows", "dis_emit_rate", "commit_parsed", "dis_defined_rate", "dis_credit_mean",
              "commit_is_minority_rate", "cites_two_plus_rate", "diag_words_mean", "final_acc",
              "cand_acc_mean", "commit_cand_correct_rate"):
        assert k in tel, k
    assert tel["dis_emit_rate"] == 1.0 and tel["commit_parsed"] == 1.0
    assert math.isclose(tel["commit_is_minority_rate"], 1.0)   # 커밋 "42" vs 후보 다수 "9"
    assert math.isclose(tel["commit_cand_correct_rate"], 1.0)  # 후보 3 은 실제로 정답
    assert math.isclose(tel["cand_acc_mean"], 0.5)
    assert "dis_credit=" in M.format_tel(tel)


def test_dis_telemetry_nan_when_nothing_parsed():
    rows = M.compute_rows(["plain \\boxed{42}"], ["42"], ["p0"], "M_DIS", uids=["g0"],
                          cand_answers=[["1", "2", "3", "4"]], cand_correct=[[0, 0, 0, 0]])
    tel = M.dis_telemetry(rows)
    assert tel["dis_rows"] == 0 and tel["commit_parsed"] == 0.0
    assert math.isnan(tel["cites_two_plus_rate"]) and math.isnan(tel["dis_credit_mean"])


def _rep(**kw):
    rep = {"step": 10, "arm": "M_DIS", "emit_rate": 0.95, "boxed_in_meta": 0.0,
           "multi_block_rate": 0.0, "boilerplate_rate": 0.1, "n_emitted": 100, "acc": 0.6,
           "dis_emit_rate": 0.95, "commit_parsed": 0.93, "dis_defined_rate": 0.85}
    rep.update(kw)
    return rep


def _names(hits):
    return {h["metric"] for h in hits if h["status"] == "abort"}


def test_abort_rules_for_dis_arm():
    assert _names(M.check_abort(_rep(), arm="M_DIS")) == set()
    assert "dis_emit_rate" in _names(M.check_abort(_rep(dis_emit_rate=0.7), arm="M_DIS"))
    assert "commit_parsed" in _names(M.check_abort(_rep(commit_parsed=0.7), arm="M_DIS"))
    assert "dis_defined_rate" in _names(M.check_abort(_rep(dis_defined_rate=0.2), arm="M_DIS"))
    assert "boxed_in_meta" in _names(M.check_abort(_rep(boxed_in_meta=0.03), arm="M_DIS"))
    assert "multi_block_rate" in _names(M.check_abort(_rep(multi_block_rate=0.2), arm="M_DIS"))


def test_abort_warmups_for_dis_arm():
    # 발화·커밋은 step≤3, 정의율은 step≤5 까지 봐준다.
    assert _names(M.check_abort(_rep(step=3, dis_emit_rate=0.0, commit_parsed=0.0, emit_rate=0.0,
                                     dis_defined_rate=0.0), arm="M_DIS")) == set()
    assert "dis_emit_rate" in _names(M.check_abort(_rep(step=4, dis_emit_rate=0.0), arm="M_DIS"))
    assert _names(M.check_abort(_rep(step=5, dis_defined_rate=0.0), arm="M_DIS")) == set()
    assert "dis_defined_rate" in _names(M.check_abort(_rep(step=6, dis_defined_rate=0.0), arm="M_DIS"))


def test_dis_rules_do_not_touch_other_arms_and_vice_versa():
    hits = M.check_abort(_rep(), arm="M_DIS")
    assert not any(h["metric"] in ("redirect_rate", "trunc_rate", "agree_line_rate", "leak_rate",
                                   "crit_defined_rate") for h in hits)
    rep = {"step": 10, "arm": "M_G1", "emit_rate": 0.9, "boxed_in_meta": 0.0,
           "multi_block_rate": 0.0, "boilerplate_rate": 0.1, "n_emitted": 100, "acc": 0.7}
    assert not any(h["metric"] in ("dis_emit_rate", "commit_parsed", "dis_defined_rate")
                   for h in M.check_abort(rep, arm="M_G1"))
    # ★M_DIS0 은 require_meta=False — 발화가 0 이어도 메타 전용 규칙이 죽이지 않는다.
    assert _names(M.check_abort(_rep(arm="M_DIS0", step=20, emit_rate=0.0, dis_emit_rate=0.0,
                                     commit_parsed=0.0, dis_defined_rate=0.0),
                                arm="M_DIS0")) == set()


# ── 9. verl_sdc 배선 ───────────────────────────────────────────────────────────
def test_stash_reads_cand_columns_and_fails_closed():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    assert "_mm._DIS_TERMS" in src
    assert 'nt_col(nt, "cand_answers"' in src and 'nt_col(nt, "cand_correct"' in src
    assert "cand_answers=_cand_ans" in src and "cand_correct=_cand_corr" in src


def test_ray_env_forwards_math_dis_w():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    assert '"MATH_DIS_W"' in src, "MATH_DIS_W 가 Ray runtime_env 목록에 없다(워커가 못 읽는다)"


# ── 10. 런처 dry-run ────────────────────────────────────────────────────────────
def test_run_math_arm_dis_defaults():
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_DIS", "2", "30", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "ARM=M_DIS" in out and "VARIANT=math_dis" in out
    assert "RESP_LEN=2048" in out
    assert "EVAL_SCRIPT=math_dis_eval.py" in out
    assert "math_dis_eval.py" in out and "math500_dis_8k" in out
    assert "MATH_DIS_W=0.5" in out
    assert "data.max_prompt_length=2048" in out   # 후보 4개가 프롬프트에 들어간다
    assert "math_train_math_dis.parquet" in out and "math_val_math_dis.parquet" in out


def test_run_math_arm_dis_controls_share_the_launch_contract():
    for arm in ("M_DIS_RAND", "M_DIS0"):
        r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", arm, "2", "30", "--dry-run"],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "RESP_LEN=2048" in r.stdout and "EVAL_SCRIPT=math_dis_eval.py" in r.stdout
        assert "VARIANT=math_dis" in r.stdout


# ── 11. held-out 평가(모의 생성기) ──────────────────────────────────────────────
def _mock_gen(per_problem):
    def gen(prompts):
        assert len(prompts) == len(per_problem)
        return [[(t, 0, max(1, len(t) // 4)) for t in texts] for texts in per_problem]
    return gen


def test_eval_candidates_and_summary_on_mocked_generations():
    problems = [{"problem": "p0", "gold": "42"}, {"problem": "p1", "gold": "5"}]
    # 후보 5개: p0 은 다수가 정답(42), p1 은 다수가 오답(9)
    cand_texts = [[f"work \\boxed{{{a}}}" for a in ("42", "42", "7", "42", "42")],
                  [f"work \\boxed{{{a}}}" for a in ("9", "9", "5", "9", "5")]]
    cands = DE.sample_candidates(problems, _mock_gen(cand_texts))
    assert [len(c) for c in cands] == [5, 5]
    assert cands[0][0]["final_answer"] == "42" and cands[0][0]["r_corr"] == 1
    # 진단 응답 N=2: p0 은 맞히고, p1 은 소수(정답 5)를 구제한다
    dis_texts = [[_text("42", DIAG, "1"), _text("42", DIAG, "1")],
                 [_text("5", DIAG, "3"), _text("5", DIAG, "3")]]
    rows, tel = DE.evaluate(problems, cands, _mock_gen(dis_texts))
    assert len(rows) == 4 and tel["n_groups"] == 2
    assert math.isclose(tel["acc_dis"], 1.0)
    assert math.isclose(tel["acc_majority4"], 0.5)      # p0 다수 42(정답), p1 다수 9(오답)
    assert math.isclose(tel["acc_majority5"], 0.5)
    assert math.isclose(tel["pass_at_4"], 1.0)
    assert math.isclose(tel["minority_rescue"], 1.0) and tel["n_minority_problems"] == 1
    assert math.isclose(tel["majority_break"], 0.0)
    assert tel["commit_parsed"] == 1.0 and tel["dis_emit_rate"] == 1.0
    assert tel["cites_two_plus_rate"] == 1.0 and tel["diag_words_mean"] > 0
    assert tel["tokens_per_problem_dis"] > tel["tokens_per_problem_majority4"] > 0
    assert tel["tokens_per_problem_majority5"] > tel["tokens_per_problem_majority4"]
    assert set(DE._SUMMARY_KEYS) <= set(tel), "요약 키가 텔레메트리에 전부 있어야 한다"
    assert "nan" not in DE.format_summary(tel).split("acc_dis")[1][:20]


def test_eval_prompt_is_built_with_the_training_turn_builder():
    problems = [{"problem": "p0", "gold": "42"}]
    cands = DE.sample_candidates(problems, _mock_gen([[f"w \\boxed{{{a}}}" for a in "1234"] + ["w \\boxed{5}"]]))
    seen = {}

    def gen(prompts):
        seen["user"] = prompts[0][1]["content"]
        seen["system"] = prompts[0][0]["content"]
        return [[(_text("42"), 0, 10)]]
    DE.evaluate(problems, cands, gen)
    assert seen["system"] == MATH_PROMPT_VARIANTS["math_opt"]
    expect = MD.build_dis_user_turn("p0", [MD.make_sketch(f"w \\boxed{{{a}}}") for a in "1234"])
    assert seen["user"] == expect


def test_eval_majority_is_nan_when_undecided():
    assert math.isnan(DE.majority_correct([{"final_answer": "1"}, {"final_answer": "2"}],
                                          "1", k=2))
    assert DE.majority_correct([{"final_answer": "1"}, {"final_answer": "1"}], "1", k=2) == 1.0


# ── 12. 게이트가 이 평가의 키를 읽는가 ──────────────────────────────────────────
def test_gate_judgment_reads_acc_dis(tmp_path, monkeypatch):
    import gate_judgment as G
    monkeypatch.setattr(G, "WORK", tmp_path)
    d = tmp_path / "eval" / "cd9_M_DIS_s2_r2048" / "step_30" / "math500_dis_8k"
    d.mkdir(parents=True)
    (d / "telemetry.json").write_text(json.dumps({"acc_dis": 0.42, "acc_majority4": 0.40}))
    got = G.read_acc("cd9_M_DIS_s2_r2048", 30, eval_subdir="math500_dis_8k", acc_key="acc_dis")
    assert math.isclose(got, 0.42)
