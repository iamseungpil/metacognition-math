"""math_sites `--site_source own_meta` 회귀 시험 (CPU, 모델 없음).

1. 합성 롤아웃에서 자기 멈춤 자리 추출 — 앞부분은 `<meta>` 직전에서 끝나고, 블록은
   `</meta>` 까지 통째로, 결정은 form="math" 로 파싱된다.
2. 모드별 fed 텍스트 — own 은 원 롤아웃과 바이트 동일, verify/redirect 는 심은 블록.
3. own_judgment_correct 진리표.
4. movable_only 가 두 원천 모두에 걸린다 / make_sites 상한.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_sites as M  # noqa: E402

BLOCK = ("<meta>\nconfidence: 0.3\nThe substitution is getting messy; the method may be wrong.\n"
         "decision: redirect\n</meta>")
TEXT = "Let x = 2y.\nThen 4y^2 + y = 10.\n" + BLOCK + "\nTry factoring instead.\n\\boxed{2}"


def _roll(gid="g1", text=TEXT, r_corr=1, truncated=0):
    return {"group_id": gid, "problem": f"P{gid}", "gold": "2", "text": text,
            "r_corr": r_corr, "truncated": truncated}


def test_own_meta_site_extraction():
    sites = M.own_meta_sites(_roll(), n_max=2)
    assert len(sites) == 1, "메타 블록 하나 → 자리 하나"
    s = sites[0]
    assert s["prefix"] == "Let x = 2y.\nThen 4y^2 + y = 10.\n"
    assert not s["prefix"].endswith("<meta>") and "<meta>" not in s["prefix"]
    assert s["own_meta"] == BLOCK and s["own_meta"].endswith("</meta>")
    assert s["own_decision"] == "redirect" and s["own_confidence"] == pytest.approx(0.3)
    assert s["site_id"] == f"g1@{TEXT.index('<meta>')}"


def test_own_meta_no_decision_is_none_and_cap():
    nodec = "a\n<meta>\nconfidence: 0.9\nfine so far\n</meta>\nb\n<meta>\nconfidence: 0.1\nx\n</meta>\n"
    sites = M.own_meta_sites(_roll(text=nodec), n_max=1)
    assert len(sites) == 1 and sites[0]["own_decision"] is None
    assert sites[0]["own_confidence"] == pytest.approx(0.9)
    assert len(M.own_meta_sites(_roll(text=nodec), n_max=5)) == 2
    assert M.own_meta_sites(_roll(text="no meta here\n"), n_max=2) == []
    # ★메타-먼저(prefix 0자) 자리는 기본으로 버린다; min_prefix_chars=0 이면 받는다.
    first = "<meta>\nconfidence: 0.5\nx\ndecision: verify\n</meta>\nthen\n"
    assert M.own_meta_sites(_roll(text=first), 2) == []
    got = M.own_meta_sites(_roll(text=first), 2, min_prefix_chars=0)
    assert len(got) == 1 and got[0]["prefix"] == "" and got[0]["rel_pos"] == 0.0
    s0 = M.own_meta_sites(_roll(), 1)[0]
    assert s0["rel_pos"] == pytest.approx(TEXT.index("<meta>") / len(TEXT))


def test_modes_text_own_meta():
    s = M.own_meta_sites(_roll(), 1)[0]
    own = M.build_fed("own", s["prefix"], None, s["own_meta"])
    # ★바이트 동일: own fed 는 원 롤아웃의 </meta> 까지와 같다(원 생성 문맥 재현).
    assert own == TEXT[:TEXT.index("</meta>") + len("</meta>")]
    assert M.build_fed("nometa", s["prefix"], None, s["own_meta"]) == s["prefix"]
    v = M.build_fed("verify", s["prefix"], None, s["own_meta"])
    r = M.build_fed("redirect", s["prefix"], None, s["own_meta"])
    assert v == s["prefix"].rstrip() + "\n" + M.SEED_VERIFY
    assert r == s["prefix"].rstrip() + "\n" + M.SEED_REDIRECT
    assert BLOCK not in v and BLOCK not in r
    with pytest.raises(ValueError):
        M.build_fed("own", s["prefix"], None, None)
    assert "own" in M.ALL_MODES
    assert M.DEFAULT_MODES["own_meta"].split(",") == ["nometa", "own", "verify", "redirect"]
    assert M.SOURCE_VARIANT == {"cut": "math_plain", "own_meta": "math_opt"}


@pytest.mark.parametrize("own,best,want", [
    ("verify", "verify", 1), ("redirect", "redirect", 1),
    ("verify", "redirect", 0), ("redirect", "verify", 0),
    ("verify", "tie", None), ("verify", None, None), (None, "verify", None), (None, None, None),
])
def test_own_judgment_correct_truth_table(own, best, want):
    assert M.own_judgment_correct(own, best) == want


def test_own_meta_summary():
    recs = [{"own_decision": "verify", "own_judgment_correct": 1, "delta_own": 0.5},
            {"own_decision": "redirect", "own_judgment_correct": 0, "delta_own": -0.25},
            {"own_decision": None, "own_judgment_correct": None, "delta_own": 0.0},
            {"own_decision": "verify", "own_judgment_correct": None, "delta_own": None}]
    s = M.own_meta_summary(recs)
    assert s["own_decision_rate"] == pytest.approx(0.75)
    assert s["n_own_judged"] == 2 and s["own_judgment_acc"] == pytest.approx(0.5)
    assert s["delta_own"] == pytest.approx((0.5 - 0.25 + 0.0) / 3)
    assert M.own_meta_summary([])["own_judgment_acc"] is None


def test_movable_only_and_make_sites_both_sources():
    rolls = [_roll("g1", r_corr=1), _roll("g1", r_corr=0),      # movable
             _roll("g2", r_corr=1), _roll("g2", r_corr=1),      # 결판(8/8 류)
             _roll("g3", r_corr=0, truncated=1)]                # 잘림 → 제외
    assert [r["group_id"] for r in M.select_sources(rolls, movable_only=False)] == ["g1", "g2"]
    assert [r["group_id"] for r in M.select_sources(rolls, movable_only=True)] == ["g1"]
    # keep 은 원천 후보만 거르고 movable 판정은 그룹 전체로: g1 의 첫 롤아웃(메타 없음)을
    # 건너뛰고 둘째(메타 있음)를 원천으로, g2 는 여전히 결판이라 제외.
    rolls2 = [_roll("g1", text="no meta\n\\boxed{2}", r_corr=1), _roll("g1", r_corr=0),
              _roll("g2", r_corr=1), _roll("g2", text="no meta\n", r_corr=1)]
    has = lambda r: "<meta>" in r["text"]  # noqa: E731
    got = M.select_sources(rolls2, movable_only=True, keep=has)
    assert [(r["group_id"], r["r_corr"]) for r in got] == [("g1", 0)]
    assert [r["group_id"] for r in M.select_sources(rolls2, movable_only=False, keep=has)] == ["g1", "g2"]
    rng = random.Random(0)
    own = M.make_sites(M.select_sources(rolls, True), "own_meta", 2, 10, rng)
    assert [s["site_id"].split("@")[0] for s in own] == ["g1"] and "own_meta" in own[0]
    long_text = "".join(f"line {i} of the derivation\n" for i in range(40)) + "\\boxed{2}"
    cut = M.make_sites([_roll("g1", text=long_text), _roll("g2", text=long_text)], "cut", 2, 3, rng)
    assert len(cut) == 3 and all("own_meta" not in s for s in cut)
    with pytest.raises(ValueError):
        M.make_sites([_roll()], "bogus", 1, 1, rng)


def test_audit_fixes_prompt_variant_and_donor_boxed(tmp_path):
    """0914 감사 버그 5·6·10 회귀: meta 모드도 base 프롬프트, \\boxed 기증 블록 제외, 채점은 prefix+cont."""
    import re
    import math_sites as M
    src = open(M.__file__).read()
    assert 'variant = "math_new" if mode == "meta"' not in src          # 버그5
    assert "build_math_prompt(problem, variant)" in src                 # 버그10
    assert '"\\\\boxed" not in m.group(0)' in src                        # 버그6a
    assert 'c = _grade(s["prefix"] + x.text, s["gold"])' in src        # 버그6b


def test_cut_points_respects_range():
    import random
    import math_sites as M
    text = "\n".join(f"line{i}" for i in range(100)) + "\n"
    cs = M.cut_points(text, 50, random.Random(1), 0.10, 0.50)
    assert cs and all(0.10 * len(text) <= c <= 0.50 * len(text) for c in cs)
    assert max(M.cut_points(text, 50, random.Random(1))) > 0.6 * len(text)   # 기본 .10~.80
