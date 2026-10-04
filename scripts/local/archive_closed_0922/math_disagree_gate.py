#!/usr/bin/env python
r"""math_disagree_gate — «자기 표본들의 **불일치**를 읽고 하나를 고른다» 게이트.

cd9 가 네 번 측정한 것: 이 4B non-thinking 정책에서 **풀이 하나를 평가하는 메타인지 블록은
인과 정보가 없다** — 자기 지목 자리·자기 비평·문맥 안 재시도·풀기 전 계획, 넷 다 own ≤ donor
였다(docs/HYPOTHESIS_LEDGER_cd9.md). 문제 안에서 살아 있는 신호는 하나뿐이다: 같은 문제에
대한 **자기 표본들 사이의 불일치**(형제 답 점유율, 문제별 AUC .754).

후보 방법 D 는 말로 하는 메타인지를 그 신호 위에 올린다 — 자기 후보 풀이 K 개를 읽고
**어디서 갈리는지·어느 가정이 틀렸는지** 진단한 뒤 하나에 **커밋**한다. 학습 **전에**,
**같은 표본 예산**에서 묻는다: (1) 그게 같은 K 후보의 다수결을 넘는가(SELECT) (2) 넘는다면
**진단 텍스트** 덕인가, 그냥 «고르기» 덕인가(CONTENT).

다섯 조건(같은 variant system 프롬프트, 생성 조건은 N=8·temperature 1.0):
    (1) majority    생성 없음 — K 후보 최종답의 다수결(동점이면 먼저 나온 것). 기준선.
    (2) diagnose    후보 K 개의 **끝부분**을 보여 주고 진단 2~4문장 + `Commit: n` + \boxed.
    (3) pick        같은 후보, "설명하지 말고 고르기만" — **진단 텍스트의 내용 대조**.
    (4) pick_blind  후보를 **최종답만** 보여 주고 (3) 과 같은 지시 — 끝부분 추론이 값이 있나.
    (5) solve_fresh 문제만(render_generation_prompt 와 바이트 동일) 한 표본 더 → K+1 다수결.

★(3) 이 없으면 «후보를 보여 주고 고르게 한 효과»를 «진단이 유용했다»로 오독한다(cd9 EVC·
  critique 판정과 같은 함정). ★(5) 가 없으면 «표본 하나 더»면 되는 것을 메타인지의 값으로
  읽는다. ★형식 미달(Commit 없음·범위 밖)은 **다수결의 답으로 채점한다** — 버리면 «파싱된
  것만»이라는 선택 편향이, 0 으로 치면 형식 실패가 Δ 를 과장한다(이 규약 아래 형식 미달은
  다수결을 넘을 수 없고, 넘는 부분만 진단의 값이다).
★읽는 자리 넷: (a) all(MIXED 전체) (b) **disagree**(all_agree 의 여집합 — K 후보가 전부
  동치는 아닌 문제. 선택이 애초에 **정의되는** 유일한 자리다. all 은 51%가 all_agree 라
  거기서는 diagnose 든 pick 이든 다수결과 같을 수밖에 없어 모든 Δ 를 희석한다) (c)
  **minority-correct**(다수결이 틀렸고 어떤 후보는 맞은 문제 — 어떤 선택기든 다수결을
  넘을 수 있는 **유일한** 자리) (d) all-agree(구조상 Δ≈0 인지 확인 — disagree 의 여집합).

통과 규칙(0914 개정 — disagree 부분집합으로 판정, all/minority_correct/all_agree 는 계속 보고):
  SELECT PASS  : (disagree) diagnose − majority 의 95% CI 가 0 제외 ∧ 평균 ≥ +0.03
  CONTENT PASS : (disagree) diagnose − pick     의 95% CI 가 0 제외 ∧ 평균 > 0

사용(예):
  python scripts/local/math_disagree_gate.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path <hf> --variant math_opt --n 8 --k_cand 4 --max_problems 200 --seed 11 \
      --out_dir /hdd_data/seungpil/scratch/eval/disagree_gate_s1
  # 생성량: 문제 × 4조건(diagnose/pick/pick_blind/solve_fresh) × N
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_cited_site_gate import bootstrap_ci, sign_test_p  # noqa: E402
from src.metacot.math_meta_prompt import (  # noqa: E402
    build_math_prompt, render_chat_messages, render_generation_prompt,
)
from src.training.math_dis import SKETCH_CHARS, sketch_tail, strip_meta_blocks  # noqa: E402
from src.training.math_meta import (  # noqa: E402
    answers_equivalent, grade_math, last_boxed, selftest_math_verify,
)

_NAN = float("nan")
PASS_DELTA_SELECT = 0.03        # diagnose − majority 의 하한(평균)
CONDS = ("majority", "diagnose", "pick", "pick_blind", "solve_fresh")
GEN_CONDS = ("diagnose", "pick", "pick_blind")      # 짧은 생성(max_tokens 2048 — s1 이후 개정)
# ★s1(0914) 판정: SELECT/CONTENT 는 "all" 대신 "disagree"(all_agree 의 여집합) 에서 잰다.
#   all/minority_correct/all_agree 는 계속 계산·보고한다(위 머리말 참조).
SUBSETS = ("all", "disagree", "minority_correct", "all_agree")
GATE_SUBSET = "disagree"    # ★SELECT/CONTENT 통과 판정이 읽는 부분집합
# 짝지은 비교 (좌 − 우)
PAIRS = (("diagnose", "majority"), ("pick", "majority"), ("diagnose", "pick"),
         ("pick", "pick_blind"), ("diagnose", "solve_fresh"))

_CAND_HEAD = "\n\nHere are {k} candidate solutions' endings:\n{cands}\n\n"
_BLIND_HEAD = "\n\nHere are {k} candidate final answers:\n{cands}\n\n"
# ★s1(0914) 판정: max_tokens=600 에서 diagnose malformed_rate .738 — 원인은 형식이 아니라
#   모델이 후보를 비교하는 대신 문제를 처음부터 다시 풀기 시작해(예: "Step 1: Compute
#   f(2001)..." 를 후보마다 반복) 예산을 태우고 `Commit:` 에 못 닿은 것이었다(gens.jsonl
#   diagnose 표본 10개 직접 확인 — 8/10 이 재풀이 도중 잘림). 그래서 **재풀이 금지**를 명시
#   지시로 못박고, `Commit:` 을 진단 직후 **자기 줄**에 쓰라고 위치까지 못박는다.
DIAGNOSE_ASK = (
    "Do NOT solve the problem from scratch and do not redo the candidates' computations at "
    "length. Compare the candidates. They may disagree. Write a DIAGNOSIS: which candidates "
    "disagree, at which step, and which assumption or computation is wrong (2-4 sentences, "
    "cite candidate numbers). Write `Commit: <n>` on its own line immediately after the "
    "diagnosis, then give that candidate's final answer in \\boxed{}."
)
PICK_ASK = ("Choose the correct candidate. Write `Commit: <n>` and its final answer in "
            "\\boxed{}. Do not explain. Answer in one line.")

_COMMIT_RE = re.compile(r"Commit\s*(?:\*\*)?\s*[:.\-]?\s*(?:\*\*)?\s*(?:Candidate\s*)?#?\s*(\d+)",
                        re.I)
_CITE_RE = re.compile(r"Candidate\s*#?\s*(\d+)", re.I)
_WORDS_RE = re.compile(r"[A-Za-z]{2,}")


# ── 후보 조립 ──────────────────────────────────────────────────────────────────
# ★0914 리뷰 D4: `SKETCH_CHARS`·메타 제거·«마지막 \boxed 앞 chars 자» 추출은
#   `src/training/math_dis.py`(학습이 쓰는 정본)가 단일 진실 원천이다 — 이 파일은 더 이상
#   따로 정의하지 않고 그 함수(strip_meta_blocks/sketch_tail)를 그대로 쓴다. 두 곳이 갈리면
#   이 게이트가 재는 것과 학습이 실제로 보는 것이 달라진다(이 파일 머리말의 문제의식과 같다).
#   함수 **이름**(strip_meta/candidate_sketch)과 **시그니처**는 이 파일의 기존 호출부·CLI
#   (`--sketch_chars`)와 아래 테스트가 그대로 참조하므로 얇은 위임 래퍼로 유지한다.
def strip_meta(text: str) -> str:
    """<meta>…</meta> 블록을 전부 지운 본문 — math_dis.strip_meta_blocks 그대로.
    ★메타가 남으면 «자기 메타를 읽는 효과» 가 «끝부분 추론을 읽는 효과» 와 섞인다."""
    return strip_meta_blocks(text)


def candidate_sketch(text: str, *, chars: int = SKETCH_CHARS) -> str:
    r"""후보의 «끝부분 추론» = 메타를 지운 본문에서 **마지막 \boxed 앞** 마지막 chars 자
    (math_dis.sketch_tail 그대로 — 이 파일은 답 줄을 붙이지 않고 render_candidates 에서
    따로 렌더링하므로 sketch_tail 의 «답 줄 없는» 출력을 그대로 쓴다).
    ★끝부분인 이유: K=4~8 개의 풀이 전체는 프롬프트가 3만 토큰을 넘고, 형제 불일치는 마지막
      몇 단계에서 갈린다. \boxed 를 빼는 것은 답을 answer 줄로 따로 보여 주기 때문이다."""
    return sketch_tail(text, chars=chars)


def build_candidates(rows: Sequence[dict], *, chars: int = SKETCH_CHARS) -> list[dict]:
    r"""롤아웃 행 K 개 → 후보 K 개 {answer, sketch}. answer 는 행의 final_answer,
    없으면 본문의 마지막 \boxed."""
    return [{"answer": (str(r.get("final_answer") or "").strip()
                        or last_boxed(strip_meta(r.get("text") or ""))),
             "sketch": candidate_sketch(r.get("text") or "", chars=chars)} for r in rows]


def select_candidate_sets(rolls: Sequence[dict], *, k_cand: int = 4, max_problems: int = 200,
                          sketch_chars: int = SKETCH_CHARS) -> list[dict]:
    """MIXED 그룹(0 < 그룹 정답률 < 1)의 문제당 한 행 — 후보는 **파일 순서 앞 k_cand 개**.
    ★파일 순서인 이유: 후보 집합이 난수에 안 매여야 재실행이 같은 자리를 본다. 롤아웃이
      k_cand 개에 못 미치는 그룹은 버린다(다수결의 분모가 달라지면 조건 간 짝이 깨진다)."""
    rows: dict = {}
    for r in rolls:
        rows.setdefault(r["group_id"], []).append(r)
    out = []
    for g, rs in rows.items():           # dict 는 삽입 순서를 지킨다 = 파일 순서
        v = [int(r["r_corr"]) for r in rs]
        if not (0 < sum(v) < len(v)) or len(rs) < k_cand:
            continue
        # ★잘린 롤아웃(\boxed 없음)은 후보에서 뺀다 — 빈 답이 자기 군집을 이뤄 다수결 동점을
        #   이기면 majority 기준선이 «빈 답» 이 되어 diagnose 가 공짜로 이긴다. 빼고도 K 를 못
        #   채우면 그 문제는 드롭한다(위 len 검사와 같은 규칙).
        usable = [r for r in rs if not int(r.get("truncated", 0) or 0)]
        if len(usable) < k_cand:
            continue
        head = usable[:k_cand]
        out.append({"group_id": g, "problem": head[0]["problem"], "gold": head[0]["gold"],
                    "p_group": sum(v) / len(v),
                    "cands": build_candidates(head, chars=sketch_chars)})
    return out[:max_problems] if max_problems and max_problems > 0 else out


# ── 프롬프트 ───────────────────────────────────────────────────────────────────
def render_candidates(cands: Sequence[dict], *, blind: bool = False) -> str:
    """후보 목록 문자열. blind 면 **최종답만**(끝부분 추론 없음)."""
    return "\n\n".join(
        f"[Candidate {i}] " + ("" if blind else f"...{c['sketch']}\n")
        + f"final answer: \\boxed{{{c['answer']}}}" for i, c in enumerate(cands, 1))


def cond_prompt(tok, variant: str, problem: str, cands: Sequence[dict], cond: str) -> str:
    """조건별 생성 프롬프트. solve_fresh 는 render_generation_prompt 와 **바이트 동일**이고,
    나머지 셋은 같은 system 아래 user 턴 끝에 후보 목록 + 지시만 붙는다.
    ★새 system 문구를 만들지 않는다 — 유일한 차이가 user 접미여야 Δ 가 «프롬프트가 달라진
      효과» 를 안 섞는다."""
    if cond == "solve_fresh":
        return render_generation_prompt(tok, variant, problem)
    if cond not in GEN_CONDS:
        raise ValueError(f"unknown generated condition: {cond!r}")
    blind = cond == "pick_blind"
    head = (_BLIND_HEAD if blind else _CAND_HEAD).format(
        k=len(cands), cands=render_candidates(cands, blind=blind))
    suffix = head + (DIAGNOSE_ASK if cond == "diagnose" else PICK_ASK)
    msgs = build_math_prompt(problem, variant)
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + suffix}
    return render_chat_messages(tok, msgs)


# ── 다수결·부분집합 ────────────────────────────────────────────────────────────
def answer_clusters(answers: Sequence[str]) -> list[list[int]]:
    """동치 군집(math_verify 동치; 등장 순서 유지)."""
    cls: list[list[int]] = []
    for i, a in enumerate(answers):
        for cl in cls:
            if answers_equivalent(answers[cl[0]], a):
                cl.append(i)
                break
        else:
            cls.append([i])
    return cls


def majority_vote(answers: Sequence[str]) -> str:
    """최대 동치 군집의 답 — 동점이면 **먼저 나온** 것(max 는 첫 최댓값을 돌려준다)."""
    return answers[max(answer_clusters(answers), key=len)[0]] if answers else ""


def answer_correct(ans: str, gold: str) -> int:
    r"""답 문자열이 gold 와 맞는가 — 채점은 grade_math(math_verify) 하나만 쓴다."""
    a = str(ans or "").strip()
    return int(bool(a) and bool(grade_math(f"\\boxed{{{a}}}", gold)))


def subset_flags(cands: Sequence[dict], gold: str) -> dict:
    """{maj_answer, maj_corr, all_agree, minority_correct}. minority_correct = 다수결이 틀렸고
    **어떤 후보는 맞은** 문제 — 선택기가 다수결을 넘을 수 있는 유일한 자리."""
    ans = [c["answer"] for c in cands]
    maj = majority_vote(ans)
    maj_corr = answer_correct(maj, gold)
    any_corr = any(answer_correct(a, gold) for a in ans)
    return {"maj_answer": maj, "maj_corr": maj_corr,
            "all_agree": int(len(answer_clusters(ans)) == 1),
            "minority_correct": int((not maj_corr) and any_corr)}


# ── 커밋 파싱·채점 ─────────────────────────────────────────────────────────────
# ★0914 리뷰 D4: **여기는 math_dis.parse_commit 을 쓰지 않는다** — SKETCH_CHARS/스케치와 달리
#   두 파서는 실제로 다른 것을 파싱하도록 설계됐다. math_dis.parse_commit 은 학습 보상용으로
#   ①첫 매치만 구속력을 갖고(말 바꾸기 방지) ②엄격한 "commit:" 패턴만 받고 ③N_CAND=4 고정이다.
#   이 게이트의 parse_commit 은 오프라인 판정용으로 ①**마지막** 매치를 쓰고(스펙상 최종 결정은
#   진단 뒤 맨 끝에 온다) ②마크다운 굵게·구두점 변형(`**Commit:**`, `Commit Candidate 4` 등)까지
#   받고 ③k(4 또는 8)를 인자로 받는다 — 둘을 하나로 합치면 한쪽의 의도된 동작이 깨진다
#   (검증: math_dis.parse_commit("**Commit:** 2") 는 None 을 낸다 — 이 파일의
#   test_parse_commit_variants 가 기대하는 2 가 아니다). 그래서 `SKETCH_CHARS`/스케치 추출은
#   math_dis 에서 가져오되(위), parse_commit 은 이 파일 고유로 남긴다.
def parse_commit(text: str, k: int) -> int | None:
    """`Commit: n` → 1-기반 번호. 없거나 범위 밖이면 None(=형식 미달).
    ★**마지막** 매치를 쓴다 — 스펙상 최종 결정은 진단 뒤 맨 끝에 온다."""
    ms = list(_COMMIT_RE.finditer(text or ""))
    for m in reversed(ms):
        n = int(m.group(1))
        if 1 <= n <= k:
            return n
    return None


def score_generation(text: str, cands: Sequence[dict], maj_answer: str) -> dict:
    r"""생성물 하나 → {answer, commit, fallback, malformed}. Commit 이 없거나 범위 밖이면
    **다수결의 답**으로 채점하고(형식 미달이 다수결을 넘을 수 없게), Commit 은 있는데
    \boxed 가 없으면 그 후보의 최종답으로 떨어진다(fallback)."""
    n = parse_commit(text, len(cands))
    if n is None:
        return {"answer": maj_answer, "commit": None, "fallback": 0, "malformed": 1}
    box = last_boxed(text or "")
    if not box:
        return {"answer": cands[n - 1]["answer"], "commit": n, "fallback": 1, "malformed": 0}
    return {"answer": box, "commit": n, "fallback": 0, "malformed": 0}


def diag_stats(text: str) -> dict:
    """진단 텍스트 진단량 — 단어 수와 «후보를 둘 이상 인용했는가»."""
    t = text or ""
    return {"diag_words": len(_WORDS_RE.findall(t)),
            "cites2": int(len({m.group(1) for m in _CITE_RE.finditer(t)}) >= 2)}


# ── 요약 ───────────────────────────────────────────────────────────────────────
def _p(vals: Sequence[float]) -> float:
    return (sum(vals) / len(vals)) if vals else _NAN


def _subset(recs: Sequence[dict], name: str) -> list[dict]:
    return list(recs) if name == "all" else [r for r in recs if r.get(name)]


def _block(recs: Sequence[dict], *, seed: int, n_boot: int) -> dict:
    """한 부분집합의 조건별 정확도 + 짝지은 Δ."""
    out: dict = {"n": len(recs)}
    for i, c in enumerate(CONDS):
        out[f"p_{c}"] = bootstrap_ci([r[f"p_{c}"] for r in recs], seed=seed + i, n_boot=n_boot)
    for j, (a, b) in enumerate(PAIRS):
        d = [r[f"p_{a}"] - r[f"p_{b}"] for r in recs]
        out[f"paired_{a}_minus_{b}"] = bootstrap_ci(d, seed=seed + 20 + j, n_boot=n_boot)
        out[f"sign_p_{a}_minus_{b}"] = sign_test_p(d)
    return out


def summarize(recs: Sequence[dict], *, k_cand: int = 0, n: int = 0, seed: int = 0,
              n_boot: int = 2000) -> dict:
    """per-problem 기록 → 게이트 요약(세 부분집합 × 다섯 조건 × 짝지은 Δ + 행동 통계)."""
    by = {s: _block(_subset(recs, s), seed=seed + 100 * i, n_boot=n_boot)
          for i, s in enumerate(SUBSETS)}
    commit_hist = dict(Counter(str(i) for r in recs for i in (r.get("commit_hist") or [])))
    # ★자리 편향 진단(0914) — commit_hist 가 {1:73,2:22,3:38,4:82} 처럼 자리마다 크게
    #   다르면 모델이 "내용을 비교해 고른다"가 아니라 "특정 자리(끝/처음)를 선호한다"일 수
    #   있다. commit_position_bias = 가장 많이 뽑힌 자리의 점유율(무정보면 1/k_cand 근처).
    #   commit_matches_majority_rate = 커밋이 다수결과 같은 답으로 떨어진 비율(전부 pooled —
    #   문제당 평균이 아니라 커밋 하나하나를 센다).
    n_commits = sum(commit_hist.values())
    commit_position_bias = (max(commit_hist.values()) / n_commits) if n_commits else _NAN
    match_flags = [f for r in recs for f in (r.get("commit_minority_flags") or [])]
    commit_matches_majority_rate = ((len(match_flags) - sum(match_flags)) / len(match_flags)
                                    if match_flags else _NAN)
    out = {
        "n_problems": len(recs), "k_cand": k_cand, "n_samples": n, "by_subset": by,
        "commit_hist": commit_hist,
        "commit_position_bias": commit_position_bias,
        "commit_matches_majority_rate": commit_matches_majority_rate,
        "frac_all_agree": _p([r["all_agree"] for r in recs]),
        "frac_minority_correct": _p([r["minority_correct"] for r in recs]),
        "frac_majority_correct": _p([r["maj_corr"] for r in recs]),
        "commit_is_minority_rate": bootstrap_ci([r["commit_is_minority_rate"] for r in recs],
                                                seed=seed + 41, n_boot=n_boot),
        "diag_words": bootstrap_ci([r["diag_words"] for r in recs], seed=seed + 42, n_boot=n_boot),
        "cites_two_or_more_rate": bootstrap_ci([r["cites2_rate"] for r in recs], seed=seed + 43,
                                               n_boot=n_boot),
        "malformed_rate": _p([r["malformed_rate"] for r in recs]),
        "fallback_rate": _p([r["fallback_rate"] for r in recs]),
        "trunc_rate": _p([r["trunc_rate"] for r in recs]),
        # MINORITY-RESCUE: 다수결이 0 인 자리에서 선택기가 건진 비율
        "minority_rescue_diagnose": by["minority_correct"].get("p_diagnose"),
        "minority_rescue_pick": by["minority_correct"].get("p_pick"),
    }
    out["pass_select"] = int(select_pass(out))
    out["pass_content"] = int(content_pass(out))
    return out


def _finite(ci: dict | None, *keys) -> bool:
    ci = ci or {}
    return all(isinstance(ci.get(x), (int, float)) and math.isfinite(float(ci[x])) for x in keys)


def _pair_pass(summ: dict, key: str, thresh: float) -> bool:
    """**disagree**(all_agree 의 여집합) 부분집합의 짝지은 Δ 가 «CI 가 0 제외 ∧ 평균 ≥
    thresh» 인가. nan 이면 FAIL. ★"all" 이 아니라 "disagree" 를 읽는다 — all 은 51%가
    all_agree 라 그 문제들에서 모든 조건이 다수결과 같아 Δ 를 구조적으로 희석한다."""
    ci = ((summ.get("by_subset") or {}).get(GATE_SUBSET) or {}).get(key)
    if not _finite(ci, "lo", "hi", "mean"):
        return False
    return bool((ci["lo"] > 0 or ci["hi"] < 0) and ci["mean"] >= thresh)


def select_pass(summ: dict) -> bool:
    """SELECT — (disagree) diagnose − majority 의 CI 가 0 제외 ∧ 평균 ≥ +0.03."""
    return _pair_pass(summ, "paired_diagnose_minus_majority", PASS_DELTA_SELECT)


def content_pass(summ: dict) -> bool:
    """CONTENT — (disagree) diagnose − pick 의 CI 가 0 제외 ∧ 평균 > 0."""
    return _pair_pass(summ, "paired_diagnose_minus_pick", 1e-12)


def _f(v) -> str:
    if isinstance(v, dict) and "mean" in v:
        return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}] (n={v.get('n')})"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


def to_markdown(summ: dict) -> str:
    top = ["n_problems", "k_cand", "n_samples", "frac_all_agree", "frac_minority_correct",
           "frac_majority_correct", "commit_is_minority_rate", "commit_position_bias",
           "commit_matches_majority_rate", "diag_words", "cites_two_or_more_rate",
           "malformed_rate", "fallback_rate", "trunc_rate"]
    lines = ["## math_disagree_gate — 자기 표본 불일치를 진단하고 하나에 커밋한다", "",
             "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_f(summ.get(k))} |" for k in top]
    lines += [f"| commit_hist | {json.dumps(summ.get('commit_hist', {}), sort_keys=True)} |"]
    # ★조건별(pooled 아님) malformed/trunc/reached_commit — s1 은 이 셋을 GEN_CONDS 전체로
    #   뭉뚱그려 보고해 diagnose 하나만 나쁜 것을 가렸다(malformed .738 은 diagnose 값이었다).
    cs = summ.get("cond_stats") or {}
    if cs:
        lines += ["", "### 조건별 malformed / trunc / reached_commit", "",
                  "| cond | malformed_rate | trunc_rate | reached_commit_rate |", "|---|---|---|---|"]
        lines += [f"| {c} | {_f(cs[c]['malformed_rate'])} | {_f(cs[c]['trunc_rate'])} | "
                 f"{_f(cs[c]['reached_commit_rate'])} |" for c in GEN_CONDS if c in cs]
        void = [c for c in GEN_CONDS if isinstance(cs.get(c, {}).get("malformed_rate"), float)
                and math.isfinite(cs[c]["malformed_rate"]) and cs[c]["malformed_rate"] > 0.3]
        if void:
            lines += ["", f"**[VOID?]** malformed_rate > 0.3 for: {', '.join(void)} — "
                     "이 조건의 판정은 형식 실패에 오염됐을 수 있다(다수결 폴백이 Δ 를 0으로 끈다)."]
    for s in SUBSETS:
        b = (summ.get("by_subset") or {}).get(s) or {}
        lines += ["", f"### subset: {s} (n={b.get('n')})", "", "| metric | value |", "|---|---|"]
        lines += [f"| p_{c} | {_f(b.get(f'p_{c}'))} |" for c in CONDS]
        for a, c in PAIRS:
            lines.append(f"| paired_{a}_minus_{c} | {_f(b.get(f'paired_{a}_minus_{c}'))} "
                         f"(sign p={_f(b.get(f'sign_p_{a}_minus_{c}'))}) |")
    lines += ["",
              f"**MINORITY-RESCUE** — diagnose {_f(summ.get('minority_rescue_diagnose'))} · "
              f"pick {_f(summ.get('minority_rescue_pick'))} (다수결은 이 자리에서 정의상 0)",
              f"**SELECT {'PASS' if summ.get('pass_select') else 'FAIL'}** — ({GATE_SUBSET}) "
              f"diagnose−majority CI 가 0 제외 ∧ 평균 ≥ +{PASS_DELTA_SELECT}",
              f"**CONTENT {'PASS' if summ.get('pass_content') else 'FAIL'}** — ({GATE_SUBSET}) "
              "diagnose−pick CI 가 0 제외 ∧ 평균 > 0", ""]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout 산출물 texts.jsonl")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default="math_opt", help="그 롤아웃을 만든 프롬프트 변형")
    ap.add_argument("--n", type=int, default=8, help="조건당 표본 수")
    ap.add_argument("--k_cand", type=int, default=4, help="후보 개수(4 또는 8)")
    ap.add_argument("--max_problems", type=int, default=200)
    ap.add_argument("--sketch_chars", type=int, default=SKETCH_CHARS)
    ap.add_argument("--seed", type=int, default=11)
    # ★s1(0914) 판정: 600 에서 diagnose malformed_rate .738 — 모델이 재풀이하다 잘렸다
    #   (gens.jsonl 표본 확인). 2048 로 올려 «Commit:」에 닿을 여유를 준다.
    ap.add_argument("--max_tokens", type=int, default=2048, help="diagnose/pick/pick_blind")
    ap.add_argument("--solve_tokens", type=int, default=8192,
                    help="solve_fresh — 후보 롤아웃과 같은 8192 예산(매치드 컴퓨트 대조가 불리하지 않게)")
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    selftest_math_verify()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    probs = select_candidate_sets([json.loads(l) for l in open(a.rollouts)], k_cand=a.k_cand,
                                  max_problems=a.max_problems, sketch_chars=a.sketch_chars)
    if not probs:
        raise SystemExit("[dis] 후보가 없다 — 입력 롤아웃에 MIXED 그룹이 있는지 확인하라.")
    for p in probs:
        p.update(subset_flags(p["cands"], p["gold"]))
    print(f"[dis] MIXED 문제 {len(probs)}개 (후보 K={a.k_cand}) · "
          f"all-agree {sum(p['all_agree'] for p in probs)} · "
          f"minority-correct {sum(p['minority_correct'] for p in probs)}", flush=True)

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.solve_tokens + 8192, enforce_eager=True)
    tok = llm.get_tokenizer()

    ix = [(pi, c) for pi in range(len(probs)) for c in GEN_CONDS]
    reqs = [cond_prompt(tok, a.variant, probs[pi]["problem"], probs[pi]["cands"], c)
            for pi, c in ix]
    print(f"[dis] 선택 요청 {len(reqs)}개 x N={a.n}", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.n, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))
    freqs = [cond_prompt(tok, a.variant, p["problem"], p["cands"], "solve_fresh") for p in probs]
    fouts = llm.generate(freqs, SamplingParams(n=a.n, temperature=1.0, top_p=1.0,
                                               max_tokens=a.solve_tokens, seed=a.seed))

    agg: dict = {}
    rows = []

    def push(pi: int, c: str, x, **kw) -> None:
        """조건별 집계와 gens.jsonl 행을 한 자리에서 만든다(두 루프가 같은 필드를 쓴다)."""
        t = int(x.finish_reason == "length")
        agg.setdefault((pi, c), []).append({"trunc": t, **kw})
        rows.append({"group_id": probs[pi]["group_id"], "cond": c, "r_corr": kw["r_corr"],
                     "commit": kw.get("commit"), "answer": kw.get("answer"),
                     "truncated": t, "text": x.text})

    for (pi, c), o in zip(ix, outs):
        p = probs[pi]
        for x in o.outputs:
            sc = score_generation(x.text, p["cands"], p["maj_answer"])
            d = diag_stats(x.text) if c == "diagnose" else {"diag_words": _NAN, "cites2": _NAN}
            push(pi, c, x, r_corr=answer_correct(sc["answer"], p["gold"]), **sc, **d)
    for pi, (p, o) in enumerate(zip(probs, fouts)):
        base = [c["answer"] for c in p["cands"]]
        for x in o.outputs:
            ans = last_boxed(x.text)          # ★K+1 다수결 — «표본 하나 더» 대안의 값
            push(pi, "solve_fresh", x, answer=ans,
                 r_corr=answer_correct(majority_vote(base + ([ans] if ans else [])), p["gold"]))

    recs = []
    for pi, p in enumerate(probs):
        got = {c: agg.get((pi, c)) for c in GEN_CONDS + ("solve_fresh",)}
        if not all(got.values()):
            continue                    # 네 생성 조건이 다 있어야 짝 비교가 성립한다
        dg = got["diagnose"]
        commits = [x["commit"] for x in dg if x["commit"]]
        minority = [int(not answers_equivalent(p["cands"][n - 1]["answer"], p["maj_answer"]))
                    for n in commits]
        recs.append({
            **{k: p[k] for k in ("group_id", "problem", "gold", "p_group", "maj_answer",
                                 "maj_corr", "all_agree", "minority_correct")},
            "disagree": int(not p["all_agree"]),   # ★all_agree 의 여집합 — SELECT/CONTENT 판정 자리
            "cand_answers": [c["answer"] for c in p["cands"]],
            "p_majority": float(p["maj_corr"]),      # ★생성 없는 기준선 = 0/1
            **{f"p_{c}": _p([x["r_corr"] for x in got[c]]) for c in got},
            "commit_hist": commits, "commit_is_minority_rate": _p(minority),
            "commit_minority_flags": minority,  # ★commit_matches_majority_rate 가 pooled 로 쓴다
            "diag_words": _p([x["diag_words"] for x in dg]),
            "cites2_rate": _p([x["cites2"] for x in dg]),
            "malformed_rate": _p([x["malformed"] for c in GEN_CONDS for x in got[c]]),
            "fallback_rate": _p([x["fallback"] for c in GEN_CONDS for x in got[c]]),
            "trunc_rate": _p([x["trunc"] for c in got for x in got[c]])})

    # ★조건별(pooled 아님) malformed/trunc/reached_commit — s1 이 놓친 자리(diagnose 하나만
    #   나빴는데 GEN_CONDS 전체로 뭉뚱그려 보고했다)를 메운다.
    cond_stats = {}
    for c in GEN_CONDS:
        items = [x for pi in range(len(probs)) for x in agg.get((pi, c), [])]
        cond_stats[c] = {
            "malformed_rate": _p([x["malformed"] for x in items]),
            "trunc_rate": _p([x["trunc"] for x in items]),
            "reached_commit_rate": _p([1.0 - x["malformed"] for x in items]),
        }
    void_conds = [c for c in GEN_CONDS if cond_stats[c]["malformed_rate"] > 0.3]
    if void_conds:
        print(f"[VOID?] malformed_rate > 0.3 for conditions: {void_conds} — 판정이 형식 실패에 "
              "오염됐을 수 있다(다수결 폴백이 Δ 를 0으로 끈다).", flush=True)

    summ = summarize(recs, k_cand=a.k_cand, n=a.n, seed=a.seed, n_boot=a.n_boot)
    summ.update({"model_path": a.model_path, "variant": a.variant, "rollouts": a.rollouts,
                 "seed": a.seed, "n_candidate_problems": len(probs),
                 "sketch_chars": a.sketch_chars, "n_generations": len(rows),
                 "cond_stats": cond_stats, "max_tokens": a.max_tokens})

    for name, data in (("per_problem.jsonl", recs), ("gens.jsonl", rows)):
        with (out / name).open("w") as fh:
            fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
    (out / "gate_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    (out / "gate_summary.md").write_text(to_markdown(summ))
    print(to_markdown(summ))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
