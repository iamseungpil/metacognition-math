"""math_cited_site_gate 회귀 시험 (CPU, 모델 없음 — 생성은 전부 모의).

1. segment_steps — 번호형/단락형 분절, 최소 길이 병합, \\boxed 단계는 인용 불가.
2. 지목 파싱(정상/형식 위반/범위 밖/인용 불가).
3. 앞부분 문맥이 «생성 문맥 + text[:start]» 와 바이트 동일 (+ 지목 프롬프트가 풀이를 원문
   그대로 담는다 — 템플릿이 끝 공백을 지워도).
4. 무작위 대조는 절대 지목과 같지 않다.
5. 준수(adherence) 계산.
6. 요약 통과 규칙.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_cited_site_gate as G  # noqa: E402

SOL = (
    "Step 1: Let x be the unknown side length of the rectangle we are looking for.\n"
    "We are told the perimeter equals 30 units in total.\n"
    "Step 2: Write the perimeter equation 2x + 2y = 30 and simplify it to x + y = 15.\n"
    "ok\n"                                     # ← 40자 미만: 앞 단계에 합쳐진다
    "Therefore y = 15 - x and the area becomes A = x(15 - x) for all valid x values.\n"
    "\n"
    "Maximizing the area gives x = 7.5, so the rectangle is a square of side 7.5.\n"
    "Thus the maximal area is \\boxed{56.25} square units for this configuration.\n"
)


def _segs_text(text=SOL):
    segs = G.segment_steps(text)
    return segs, [text[a:b] for a, b in segs]


# ── 1. 분절 ─────────────────────────────────────────────────────────────────────
def test_segment_steps_covers_text_and_splits_on_headers_and_blank_lines():
    segs, parts = _segs_text()
    assert segs[0][0] == 0 and segs[-1][1] == len(SOL), "구간이 텍스트 전체를 덮어야 한다"
    assert all(b == c for (_, b), (c, _) in zip(segs, segs[1:])), "구간이 인접해야 한다"
    assert parts[0].startswith("Step 1:") and parts[1].startswith("Step 2:")
    assert any(p.startswith("Therefore") for p in parts)
    assert any(p.lstrip().startswith("Maximizing") for p in parts), "빈 줄에서도 잘린다"
    assert any(p.startswith("Thus") for p in parts)


def test_short_segment_is_merged_into_previous():
    segs, parts = _segs_text()
    assert all((b - a) >= G.MIN_STEP_CHARS or i == len(segs) - 1
               for i, (a, b) in enumerate(segs)), "짧은 구간이 남으면 안 된다"
    step2 = [p for p in parts if p.startswith("Step 2:")]
    assert len(step2) == 1 and "ok\n" in step2[0], "40자 미만 'ok' 줄은 앞 단계에 합쳐진다"


def test_leading_short_segment_pulls_next():
    txt = "Hi.\n\n" + "A" * 80 + "\n\n" + "B" * 80 + "\n"
    segs = G.segment_steps(txt)
    assert segs[0][0] == 0 and txt[segs[0][0]:segs[0][1]].startswith("Hi.")
    assert (segs[0][1] - segs[0][0]) >= G.MIN_STEP_CHARS, "첫 구간이 짧으면 다음을 당겨 온다"


def test_boxed_step_excluded_from_citable():
    segs = G.segment_steps(SOL)
    cit = G.citable_indices(SOL, segs)
    boxed_seg = next(i for i, (a, b) in enumerate(segs) if "\\boxed" in SOL[a:b])
    assert boxed_seg not in cit, "최종 \\boxed 가 든 단계는 인용 불가"
    assert all(i < boxed_seg for i in cit), "그 뒤 단계도 인용 불가(앞부분에 답이 들어간다)"
    assert len(cit) >= G.MIN_CITABLE
    # 박스가 없으면 전부 인용 가능.
    plain = "a" * 50 + "\n\n" + "b" * 50
    ps = G.segment_steps(plain)
    assert G.citable_indices(plain, ps) == list(range(len(ps)))


# ── 2. 지목 파싱 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,want", [
    ("suspect: 2", 1),
    ("  Suspect:  3 \n", 2),
    ("suspect: **2**", 1),
    ("I think suspect: 1 is wrong", 0),
    ("step 2 looks wrong", None),      # 형식 위반
    ("suspect: 0", None),              # 범위 밖(1-기반)
    ("suspect: 9", None),              # 범위 밖
    ("suspect: 5", None),              # 범위 안이지만 인용 불가 단계
])
def test_parse_citation(raw, want):
    assert G.parse_citation(raw, 6, [0, 1, 2, 3]) == want


def test_declared_step_map_and_citation_uses_it():
    """모델이 스스로 단 'Step k' 번호가 구간 순번과 다를 때, 지목은 **머리말 번호**로 읽힌다."""
    txt = ("Intro paragraph that is long enough to be its own segment on its own here.\n\n"
           "Filler paragraph that is also long enough to survive the merge rule as one.\n\n"
           "### Step 1: set up the equation with enough characters to stand alone here.\n\n"
           "### Step 2: solve it, again with plenty of characters to stand alone here.\n\n"
           "### Step 3: conclude with \\boxed{7} and enough characters to stand alone.\n")
    segs = G.segment_steps(txt)
    sm = G.declared_step_map(txt, segs)
    assert set(sm) == {1, 2, 3}
    assert sm[1] >= 2, "Step 1 은 구간 1번이 아니다(앞에 서두 단락 둘)"
    cit = G.citable_indices(txt, segs)
    assert sm[3] not in cit, "\\boxed 가 든 Step 3 은 인용 불가"
    assert G.parse_citation("suspect: 2", len(segs), cit, sm) == sm[2]
    assert G.parse_citation("suspect: 3", len(segs), cit, sm) is None, "인용 불가 단계"
    assert G.parse_citation("suspect: 9", len(segs), cit, sm) is None, "없는 번호"
    # 머리말이 없으면 구간 순번(1-기반)으로 되돌아간다.
    assert G.parse_citation("suspect: 2", len(segs), cit, {}) == 1


# ── 3. 문맥 바이트 동일 ─────────────────────────────────────────────────────────
class FakeTok:
    """assistant 본문의 끝 공백을 지우는 템플릿(Qwen3.5 함정 재현)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False,
                            enable_thinking=None):
        out = ""
        for m in msgs:
            c = m["content"].rstrip() if m["role"] == "assistant" else m["content"]
            out += f"<|im_start|>{m['role']}\n{c}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out


def test_continuation_prompt_is_context_plus_prefix():
    from src.metacot.math_meta_prompt import render_generation_prompt
    tok = FakeTok()
    segs = G.segment_steps(SOL)
    start = segs[2][0]
    got = G.continuation_prompt(tok, "math_opt", "  2+2?  ", SOL, start)
    want = render_generation_prompt(tok, "math_opt", "  2+2?  ") + SOL[:start]
    assert got == want and got.endswith(SOL[:start])


def test_citation_prompt_keeps_solution_verbatim():
    from src.metacot.math_meta_prompt import render_generation_prompt
    tok = FakeTok()
    p = G.build_citation_prompt(tok, "math_opt", "2+2?", SOL)
    head = render_generation_prompt(tok, "math_opt", "2+2?")
    assert p.startswith(head + SOL), "풀이는 원 생성 문맥 뒤에 원문 그대로(끝 개행 포함)"
    assert G.CITATION_ASK in p and p.endswith("<|im_start|>assistant\n")


# ── 4. 무작위 대조 ──────────────────────────────────────────────────────────────
def test_random_other_never_equals_cited():
    rng = random.Random(0)
    cit = [0, 1, 2, 3, 4]
    for cited in cit:
        for _ in range(50):
            o = G.random_other(cited, cit, rng)
            assert o is not None and o != cited and o in cit
    assert G.random_other(2, [2], rng) is None


# ── 5. 준수 ────────────────────────────────────────────────────────────────────
def test_adherence_computation():
    orig = "Step 2: Write the perimeter equation 2x + 2y = 30 and simplify it to x + y = 15."
    assert G.differs_from_original(orig, orig) == 0, "그대로 다시 쓰면 준수 0"
    assert G.differs_from_original(orig, "Instead, use the area formula A = x*y directly.") == 1
    assert G.jaccard3(orig, orig) == pytest.approx(1.0)
    assert G.jaccard3(orig, "") == 0.0
    # 앞 300자만 본다 — 그 뒤가 아무리 달라도 앞이 같으면 준수 0.
    long_same = orig + " " * 5 + "X" * 400 + " totally different tail words here"
    assert G.differs_from_original(orig, long_same) == 0


# ── 6. 요약 / 통과 규칙 ─────────────────────────────────────────────────────────
def _rec(pc, pr, pe=0.1, last=0, adh=1.0):
    return {"p_cited": pc, "p_random": pr, "p_early": pe, "cite_last": last,
            "rel_pos": 0.5, "adherence": adh, "adherent": int(adh >= 0.5), "trunc_rate": 0.0}


def test_summary_pass_rule_pass():
    recs = [_rec(0.5, 0.2) for _ in range(30)]
    s = G.summarize(recs, n_no_citation=10, k=8, n_boot=500)
    assert s["n_rollouts"] == 30 and s["no_citation_rate"] == pytest.approx(0.25)
    assert s["paired_cited_minus_random"]["mean"] == pytest.approx(0.3)
    assert s["paired_cited_minus_random"]["lo"] > 0
    assert s["frac_cited_gt_random"] == 1.0 and s["sign_test_p"] < 0.01
    assert s["adherence_rate"] == 1.0 and s["n_adherent"] == 30
    assert s["pass"] == 1 and "PASS" in G.to_markdown(s)


def test_summary_pass_rule_fails_when_ci_includes_zero_or_delta_small():
    # (a) 효과 없음 — CI 가 0 을 포함.
    null = [_rec(0.3 + 0.1 * (i % 2), 0.3 + 0.1 * ((i + 1) % 2)) for i in range(30)]
    s = G.summarize(null, k=8, n_boot=500)
    assert s["pass"] == 0 and "FAIL" in G.to_markdown(s)
    # (b) 일관되지만 +.05 미만 — 부호검정은 유의해도 탈락.
    tiny = [_rec(0.22, 0.20) for _ in range(30)]
    s2 = G.summarize(tiny, k=8, n_boot=500)
    assert s2["sign_test_p"] < 0.01 and s2["paired_cited_minus_random"]["lo"] > 0
    assert s2["pass"] == 0, "평균 Δ 가 +.05 이하면 통과가 아니다"
    # (c) 빈 입력 — 조용히 통과하면 안 된다.
    assert G.summarize([], k=8, n_boot=100)["pass"] == 0


def test_summary_stratifies_by_cite_last_and_adherence():
    recs = ([_rec(0.6, 0.2, last=1, adh=1.0) for _ in range(10)]
            + [_rec(0.4, 0.3, last=0, adh=0.0) for _ in range(10)])
    s = G.summarize(recs, k=8, n_boot=300)
    assert s["cite_last_rate"] == pytest.approx(0.5)
    assert s["p_cited_when_last"]["mean"] == pytest.approx(0.6)
    assert s["p_cited_when_not_last"]["mean"] == pytest.approx(0.4)
    assert s["n_adherent"] == 10
    assert s["paired_cited_minus_random_adherent"]["mean"] == pytest.approx(0.4)


# ── 롤아웃 선별(MIXED · 오답 · 문제당 상한) ────────────────────────────────────
def test_select_wrong_rollouts_mixed_only():
    def r(g, c, t=0):
        return {"group_id": g, "problem": g, "gold": "1", "text": SOL, "r_corr": c, "truncated": t}
    rolls = ([r("mix", 0), r("mix", 0), r("mix", 0), r("mix", 1)]
             + [r("allwrong", 0), r("allwrong", 0)]
             + [r("allright", 1), r("allright", 1)]
             + [r("trunc", 0, 1), r("trunc", 1)])
    got = G.select_wrong_rollouts(rolls, per_problem=2)
    assert [x["group_id"] for x in got] == ["mix", "mix"], "MIXED 그룹의 오답만, 문제당 2개"
    kept, dropped = G.prepare_rollouts(rolls, per_problem=2, require_declared=False)
    assert len(kept) == 2 and dropped == 0
    # ★기본(require_declared=True)은 'Step k' 머리말 번호가 부족한 SOL 판을 버린다.
    kept2, dropped2 = G.prepare_rollouts(rolls, per_problem=2)
    assert kept2 == [] and dropped2 == 2
    assert all(len(x["citable"]) >= G.MIN_CITABLE for x in kept)
    short, d2 = G.prepare_rollouts([r("mix", 0), r("mix", 1)], per_problem=2,
                                   require_declared=False)
    assert short and d2 == 0


def test_cite_options_prefers_declared_steps_and_falls_back():
    txt = "".join(f"### Step {i}: work line with plenty of characters here to stand.\n\n"
                  for i in range(1, 6)) + "Thus the answer is \\boxed{7} and we are done here.\n"
    segs = G.segment_steps(txt)
    cit = G.citable_indices(txt, segs)
    sm = G.declared_step_map(txt, segs)
    opts = G.cite_options(segs, cit, sm)
    assert opts == sorted(i for i in sm.values() if i in set(cit))
    assert len(opts) >= G.MIN_CITABLE and all(i in cit for i in opts)
    # 머리말이 모자라면(SOL 은 Step 1·2 둘뿐) 단락 후보로 되돌아간다.
    ssegs = G.segment_steps(SOL)
    scit = G.citable_indices(SOL, ssegs)
    assert G.cite_options(ssegs, scit, G.declared_step_map(SOL, ssegs)) == list(scit)
