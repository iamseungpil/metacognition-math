#!/usr/bin/env python
r"""math_plan_gate — «푸는 **앞에서** 세운 계획»이 인과 정보를 갖는가.

cd9~0914 까지 측정된 것: 풀이 **중간**(자기 지목 자리)과 **끝**(비평·재시도)의 메타인지는
이 4B 정책에게 전부 인과적으로 무력하다 — 심어 준 메타는 donor 와 구별되지 않고, 자기 비평은
눈감고 다시 푸는 것을 못 넘고, 문맥 안 재시도는 0/1,248 을 살린다. 안 재 본 자리는 하나
남았다: **풀기 전**. Thought Anchors 는 planning 문장이 반사실 중요도가 가장 크다고 보고한다.

    A. plan     문제만 주고 "풀기 전에 서로 다른 접근 3개와 그중 고를 하나" 를 쓰게 한다
                (temperature 0.8, 1회). 답 누출은 거른다.
    B. solve    같은 variant system 프롬프트 아래 K 개씩 (temperature 1.0):
        (i)   blind        문제만 — 기준선(render_generation_prompt 와 **바이트 동일**)
        (ii)  plan_1..3    문제 + "Follow this approach: <계획 j>"  (셋 다, 각 K)
        (iii) donor_1..3   (ii) 와 같되 **서로 다른 세 문제**의 계획 — 내용 대조

읽는 법(★이 게이트가 가르는 두 가지):
  · **계획에 정보가 있는가**(ceiling) = p_best − p_donor_best. 세 계획 중 최선이 «같은 개수
    (3개)·같은 K» 로 뽑은 남의 계획 중 최선을 못 넘으면 계획은 조향이 아니라 장식이다.
    ★max 는 그 자체로 편향된 통계량이다 — 계획이 완전히 무정보(순수 잡음)여도 3개 중
    최댓값의 기댓값은 눈감고 푼 것보다 이미 +0.05~+0.08 높다(Binomial(K=8)/8 추정 3개의
    최댓값 편향). p_best 를 p_blind(1개, max 없음) 와 비교하면 이 편향을 «계획이 유용했다»
    로 오독한다(cd9 EVC 판정과 같은 함정) — donor 쪽도 **같은 3개·같은 K** 의 max 를 잡아야
    그 편향이 짝지은 차에서 상쇄된다. p_best−p_blind·spread=max−min 은 참고용 진단으로만
    남긴다(빈 접미 대비 «아무 문장이나 붙인 효과» 를 보는 용도).
  · **고를 줄 아는가**(selection) = p_choice − mean(p_plan). 계획에 정보가 있어도 모델이
    나쁜 계획을 고르면 RL 이 보상할 것은 «계획 쓰기» 가 아니라 «계획 고르기» 다. 이 통계량은
    max 가 아니라 mean 대 단일값이라 편향이 없다 — 그대로 둔다.
  두 물음은 갈린다. (c) 만 통과하면 계획은 외부에서 골라 줘야 하고, (b) 까지 통과해야
  «스스로 계획하고 고르는 것» 을 보상하는 RL 의 전제가 선다.

통과 규칙(따로 찍는다):
  PLAN-INFO  : (c) p_best − p_donor_best 의 95% CI 가 0 제외 ∧ 평균 > +0.05   [ceiling, max 대 max]
             ∧ (d) p_plan_mean − p_donor_mean 의 95% CI 가 0 제외 ∧ 평균 > 0  [level, mean 대 mean]
  SELECTION  : (b) p_choice − p_plan_mean 의 95% CI 가 0 제외 ∧ 평균 > 0

사용(예):
  python scripts/local/math_plan_gate.py \
      --rollouts /hdd_data/seungpil/scratch/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path <hf> --variant math_opt --k 8 --max_problems 150 --seed 11 \
      --out_dir /hdd_data/seungpil/scratch/eval/plan_gate_s1
  # 생성량: 문제 × 7조건(blind + plan1-3 + donor1-3) × K
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_cited_site_gate import bootstrap_ci, sign_test_p  # noqa: E402
from src.metacot.math_meta_prompt import (  # noqa: E402
    build_math_prompt, render_chat_messages, render_generation_prompt,
)
from src.training.math_meta import (  # noqa: E402
    critique_leaks, grade_math, selftest_math_verify,
)

_NAN = float("nan")
N_PLANS = 3
N_DONOR = 3                     # ★own 과 같은 개수라야 max 편향이 짝지은 차에서 상쇄된다
PASS_DELTA_CEIL = 0.05          # (c) p_best − p_donor_best 의 하한(평균)
SPREAD_THRESH = 0.25            # (e) «계획이 갈린다» 판정선(own·donor 양쪽에 적용)
ADHERE_CHARS = 300
CONDS = ("blind", "plan1", "plan2", "plan3", "donor1", "donor2", "donor3")

PLAN_ASK = (
    "\n\nBefore solving, list exactly 3 distinct candidate approaches, one per line as "
    "`Plan 1: …`, `Plan 2: …`, `Plan 3: …` (one sentence each, no calculations, no final "
    "answer). Then write `Choice: <n>` for the approach you judge most likely to succeed."
)
FOLLOW_PREFIX = "\n\nFollow this approach: "

_PLAN_RE = re.compile(r"^[ \t]*(?:[-*]\s*)?(?:\*\*\s*)?Plan\s*#?\s*(\d+)\s*(?:\*\*)?\s*[:.\-]\s*"
                      r"(.+?)[ \t]*$", re.M | re.I)
_CHOICE_RE = re.compile(r"Choice\s*(?:\*\*)?\s*[:.\-]\s*(?:\*\*)?\s*#?\s*(\d+)", re.I)
_WORD_RE = re.compile(r"[A-Za-z]{3,}")
# 계획·풀이 서두 양쪽에 흔한 기능어 — Jaccard 가 «수학 내용» 을 보게 한다.
_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "use", "using", "then", "are", "was",
    "can", "will", "have", "has", "its", "into", "each", "any", "all", "not", "but", "which",
    "them", "they", "you", "our", "one", "two", "let", "set", "get", "find", "solve", "given",
    "problem", "approach", "plan", "first", "second", "third", "step", "steps", "way", "need",
    "must", "would", "should", "there", "their", "these", "those", "also", "just", "such",
    "how", "what", "when", "where", "who", "since", "because", "may", "might", "does", "did",
}


# ── donor 배정(셋) ─────────────────────────────────────────────────────────────
def assign_donor_triples(kept: Sequence[dict], rng: random.Random,
                         n_donor: int = N_DONOR) -> list[list[int] | None]:
    r"""각 행에 **서로 다른 문제** donor `n_donor`개(기본 3)를 배정한다 → 인덱스 리스트의 리스트.
    `src.training.math_meta.assign_donors`(단일 donor)의 회전 배정을 그대로 쓰되, 한 자리마다
    하나가 아니라 `n_donor`개를 모은다.

    ★왜 셋인가: own 쪽 p_best 는 3개 계획 중 **최댓값**이라 그 자체로 편향된 통계량이다
    (Binomial(K,p)/8 추정 3개의 max 는 무정보 상태에서도 기댓값이 눈감고 푼 것보다 +0.05~
    +0.08 높다). 이 편향은 donor 쪽도 **같은 개수(3)·같은 K** 로 뽑은 최댓값과 비교해야만
    짝지은 차(paired_best_minus_donorbest)에서 상쇄된다 — donor 가 하나뿐이면 max 대 무엇도
    아닌 것을 비교하는 셈이라 편향이 그대로 새어 나간다.

    셔플한 순열 하나를 회전시켜, 각 자리에서 앞으로 걸으며 **자기 group_id 와도, 이미 그
    행에 모은 donor 들의 group_id 와도** 다른 첫 `n_donor`개를 모은다(즉 donor 셋의 원래
    문제가 서로도 달라야 한다 — «같은 문제의 계획을 donor 로 세 번 세는 것»을 막는다).
    그만큼(서로 다른 문제가 `n_donor`개)을 못 채우면 그 행은 None(호출부에서 드롭하고
    `n_dropped_no_donor` 로 센다)."""
    n = len(kept)
    if n < n_donor + 1:
        return [None] * n
    order = list(range(n))
    rng.shuffle(order)
    donors: list[list[int] | None] = [None] * n
    for pos, i in enumerate(order):
        own_gid = kept[i]["group_id"]
        used_gids = {own_gid}
        got: list[int] = []
        for step in range(1, n):
            j = order[(pos + step) % n]
            gj = kept[j]["group_id"]
            if gj in used_gids:
                continue
            got.append(j)
            used_gids.add(gj)
            if len(got) == n_donor:
                break
        donors[i] = got if len(got) == n_donor else None
    return donors


# ── 후보 선별 ──────────────────────────────────────────────────────────────────
def select_mixed_problems(rolls: Sequence[dict], *, max_problems: int = 150) -> list[dict]:
    """MIXED 그룹(0 < 그룹 정답률 < 1)의 **문제 하나당 한 행**.

    ★여기서는 롤아웃이 아니라 «문제» 가 단위다 — 계획은 풀이 앞에 서므로 특정 오답 롤아웃에
    매이지 않는다. 입력 롤아웃은 (a) MIXED 여부를 정하고 (b) 문제·gold 를 준다.
    쉬운·불가능한 문제(그룹 0% / 100%)를 빼는 이유는 천장·바닥에서 Δ 가 못 움직이기 때문.
    """
    acc: dict = {}
    first: dict = {}
    order: list = []
    for r in rolls:
        g = r["group_id"]
        acc.setdefault(g, []).append(int(r["r_corr"]))
        if g not in first:
            first[g] = r
            order.append(g)
    out = []
    for g in order:
        v = acc[g]
        if not (0 < sum(v) < len(v)):
            continue
        r = first[g]
        out.append({"group_id": g, "problem": r["problem"], "gold": r["gold"],
                    "p_group": sum(v) / len(v)})
    return out[:max_problems] if max_problems and max_problems > 0 else out


# ── A) 계획 프롬프트·파싱 ───────────────────────────────────────────────────────
def plan_prompt(tok, variant: str, problem: str) -> str:
    """계획 요청 프롬프트 = 같은 system + (문제 + PLAN_ASK) user 턴.
    ★새 system 문구를 만들지 않는다 — 계획 조건과 풀이 조건의 system 이 갈리면 Δ 가
      «계획의 효과» 가 아니라 «프롬프트가 달라진 효과» 를 섞는다."""
    msgs = build_math_prompt(problem, variant)
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + PLAN_ASK}
    return render_chat_messages(tok, msgs)


def parse_plans(text: str, n_plans: int = N_PLANS) -> tuple[list[str], int] | None:
    r"""생성물 → (계획 n개, 고른 번호 1-기반). 셋이 안 나오거나 Choice 가 없으면 None.

    ★같은 번호가 여러 번 나오면 **첫 번째**만 쓴다(모델이 계획을 다시 늘어놓는 경우).
    ★Choice 범위 밖(0 이나 4)이면 파싱 실패로 친다 — 조용히 1로 떨어뜨리면 «선택 능력»
      측정이 «기본값 1» 의 성능이 된다.
    """
    got: dict[int, str] = {}
    for m in _PLAN_RE.finditer(text or ""):
        k = int(m.group(1))
        body = m.group(2).strip().strip("*").strip()
        if k in got or not body:
            continue
        got[k] = body
    plans = [got[i] for i in range(1, n_plans + 1) if i in got]
    if len(plans) != n_plans:
        return None
    if len({p.lower() for p in plans}) != n_plans:
        return None                      # 서로 다른 접근이어야 한다(글자 동일은 버린다)
    cm = _CHOICE_RE.search(text or "")
    if not cm:
        return None
    c = int(cm.group(1))
    if not (1 <= c <= n_plans):
        return None
    return plans, c


def plans_leak(plans: Sequence[str], gold: str) -> bool:
    r"""계획이 답을 흘렸는가 — \boxed / 답 문자열 / 답과 값이 같은 숫자 토큰.
    ★정의는 `src/training/math_meta.critique_leaks` 하나를 쓴다(비평 게이트와 같은 계약).
      계획 하나라도 흘리면 그 문제를 통째로 버린다 — 한 계획이 답을 담고 있으면 blind 대조가
      «계획의 조향» 이 아니라 «답을 보여 준 효과» 를 잰다."""
    return any(critique_leaks(p, gold) for p in plans)


# ── B) 풀이 프롬프트 ───────────────────────────────────────────────────────────
def solve_prompt(tok, variant: str, problem: str, plan: str | None = None) -> str:
    """plan 이 None 이면 render_generation_prompt 와 **바이트 동일**(blind), 있으면 user 턴
    끝에 FOLLOW_PREFIX + plan 만 붙는다 — 일곱 조건(blind + plan1-3 + donor1-3)의 유일한
    차이는 이 접미다."""
    if plan is None:
        return render_generation_prompt(tok, variant, problem)
    msgs = build_math_prompt(problem, variant)
    msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + FOLLOW_PREFIX + plan.strip()}
    return render_chat_messages(tok, msgs)


# ── 준수 프록시 ────────────────────────────────────────────────────────────────
def _content_words(s: str) -> set[str]:
    return {w for w in (x.lower() for x in _WORD_RE.findall(s or "")) if w not in _STOP}


def adherence(plan: str, solution: str, *, chars: int = ADHERE_CHARS) -> float:
    """계획과 풀이 **서두**(앞 chars 자)의 내용어 Jaccard. 0~1, 한쪽이 비면 nan.
    ★인과가 아니라 프록시다 — 높다고 계획을 따랐다는 증거는 아니고, **낮으면** 계획이
      생성에 안 닿았다는 증거다(그러면 Δ≈0 은 «계획이 무용» 이 아니라 «전달 실패» 다)."""
    a, b = _content_words(plan), _content_words((solution or "")[:chars])
    if not a or not b:
        return _NAN
    return len(a & b) / len(a | b)


# ── 요약 ───────────────────────────────────────────────────────────────────────
def _p(vals: Sequence[float]) -> float:
    return (sum(vals) / len(vals)) if vals else _NAN


def per_problem_record(rec: dict) -> dict:
    """p_plan[1..3]·choice·p_donor[1..3] 가 든 행 → 파생량
    (p_choice/p_best/p_plan_mean/plan_spread/p_donor_mean/p_donor_best/donor_spread/argmax).
    ★own 쪽(p_best/plan_spread) 과 donor 쪽(p_donor_best/donor_spread) 을 **같은 모양**
    (3개 중 max/min)으로 뽑는다 — 그래야 짝지은 차에서 max-of-3 편향이 상쇄된다."""
    ps = [rec["p_plan"][i] for i in range(N_PLANS)]
    ds = [rec["p_donor"][i] for i in range(N_DONOR)]
    out = dict(rec)
    out["p_choice"] = ps[rec["choice"] - 1]
    out["p_best"] = max(ps)
    out["p_worst"] = min(ps)
    out["p_plan_mean"] = sum(ps) / N_PLANS
    out["plan_spread"] = max(ps) - min(ps)
    out["p_donor_mean"] = sum(ds) / N_DONOR
    out["p_donor_best"] = max(ds)
    out["donor_spread"] = max(ds) - min(ds)
    # ★argmax 는 동점이면 여러 개다 — 고른 것이 동점 최선이면 «맞혔다»로 친다(무작위 1/3 과
    #   비교하는 양이므로 동점을 틀렸다고 치면 아래쪽으로 치우친다).
    out["choice_is_argmax"] = int(out["p_choice"] >= out["p_best"] - 1e-12)
    return out


def summarize(recs: Sequence[dict], *, k: int = 0, seed: int = 0, n_boot: int = 2000,
              n_unparsed: int = 0, n_leaked: int = 0) -> dict:
    """per-problem 기록 → 게이트 요약. recs 는 일곱 조건(blind+plan1-3+donor1-3)이 모두
    성립한 문제만."""
    n = len(recs)
    d_choice_blind = [r["p_choice"] - r["p_blind"] for r in recs]              # (a) 참고 진단
    d_choice_mean = [r["p_choice"] - r["p_plan_mean"] for r in recs]           # (b) selection
    d_best_blind = [r["p_best"] - r["p_blind"] for r in recs]                  # 참고 진단(구 c)
    d_best_donorbest = [r["p_best"] - r["p_donor_best"] for r in recs]         # (c) ceiling — max 대 max
    d_planmean_donormean = [r["p_plan_mean"] - r["p_donor_mean"] for r in recs]  # (d) level — mean 대 mean
    d_spread_donorspread = [r["plan_spread"] - r["donor_spread"] for r in recs]
    n_att = n + n_unparsed + n_leaked
    trunc_by_cond = {}
    for c in CONDS:
        vals = [r["trunc_rate_by_cond"][c] for r in recs
                if isinstance(r.get("trunc_rate_by_cond"), dict) and c in r["trunc_rate_by_cond"]]
        trunc_by_cond[c] = _p(vals)
    out = {
        "n_problems": n, "k": k,
        "n_unparsed": int(n_unparsed), "n_leaked": int(n_leaked),
        "unparsed_rate": (n_unparsed / n_att) if n_att else _NAN,
        "leak_rate": (n_leaked / n_att) if n_att else _NAN,
        "p_blind": bootstrap_ci([r["p_blind"] for r in recs], seed=seed, n_boot=n_boot),
        "p_choice": bootstrap_ci([r["p_choice"] for r in recs], seed=seed + 1, n_boot=n_boot),
        "p_plan_mean": bootstrap_ci([r["p_plan_mean"] for r in recs], seed=seed + 2,
                                    n_boot=n_boot),
        "p_best": bootstrap_ci([r["p_best"] for r in recs], seed=seed + 3, n_boot=n_boot),
        "p_worst": bootstrap_ci([r["p_worst"] for r in recs], seed=seed + 4, n_boot=n_boot),
        "p_donor_mean": bootstrap_ci([r["p_donor_mean"] for r in recs], seed=seed + 5,
                                     n_boot=n_boot),
        "p_donor_best": bootstrap_ci([r["p_donor_best"] for r in recs], seed=seed + 17,
                                     n_boot=n_boot),
        "donor_spread": bootstrap_ci([r["donor_spread"] for r in recs], seed=seed + 18,
                                     n_boot=n_boot),
        # (a) 참고 진단 · (b) selection · (c) ceiling · (d) level
        "paired_choice_minus_blind": bootstrap_ci(d_choice_blind, seed=seed + 6, n_boot=n_boot),
        "paired_choice_minus_planmean": bootstrap_ci(d_choice_mean, seed=seed + 7, n_boot=n_boot),
        "paired_best_minus_blind": bootstrap_ci(d_best_blind, seed=seed + 8, n_boot=n_boot),
        "paired_best_minus_donorbest": bootstrap_ci(d_best_donorbest, seed=seed + 9,
                                                     n_boot=n_boot),
        "paired_planmean_minus_donormean": bootstrap_ci(d_planmean_donormean, seed=seed + 19,
                                                         n_boot=n_boot),
        "paired_spread_minus_donorspread": bootstrap_ci(d_spread_donorspread, seed=seed + 20,
                                                         n_boot=n_boot),
        "sign_p_choice_minus_blind": sign_test_p(d_choice_blind),
        "sign_p_choice_minus_planmean": sign_test_p(d_choice_mean),
        "sign_p_best_minus_blind": sign_test_p(d_best_blind),
        "sign_p_best_minus_donorbest": sign_test_p(d_best_donorbest),
        "sign_p_planmean_minus_donormean": sign_test_p(d_planmean_donormean),
        "sign_p_spread_minus_donorspread": sign_test_p(d_spread_donorspread),
        # (e) 계획이 갈리는가 · 퍼짐(own 과 donor 나란히 — donor 쪽이 이 양의 무정보 수준이다)
        "plan_spread": bootstrap_ci([r["plan_spread"] for r in recs], seed=seed + 10,
                                    n_boot=n_boot),
        "frac_spread_ge_thresh": (sum(1 for r in recs if r["plan_spread"] >= SPREAD_THRESH) / n)
                                 if n else _NAN,
        "frac_donor_spread_ge_thresh": (sum(1 for r in recs
                                            if r["donor_spread"] >= SPREAD_THRESH) / n)
                                       if n else _NAN,
        # (f) 고른 것이 최선인 비율 — 무작위면 1/3
        "frac_choice_is_argmax": bootstrap_ci([r["choice_is_argmax"] for r in recs],
                                              seed=seed + 11, n_boot=n_boot),
        "choice_hist": {str(i): sum(1 for r in recs if r["choice"] == i)
                        for i in range(1, N_PLANS + 1)},
        # (g) 준수 프록시 — 조건별(donor 는 셋의 평균)
        "adherence_choice": bootstrap_ci([r["adherence_choice"] for r in recs], seed=seed + 12,
                                         n_boot=n_boot),
        "adherence_planmean": bootstrap_ci([r["adherence_planmean"] for r in recs],
                                           seed=seed + 13, n_boot=n_boot),
        "adherence_donormean": bootstrap_ci([r["adherence_donormean"] for r in recs],
                                            seed=seed + 14, n_boot=n_boot),
        "adherence_blind": bootstrap_ci([r["adherence_blind"] for r in recs], seed=seed + 15,
                                        n_boot=n_boot),
        "plan_words": bootstrap_ci([r["plan_words"] for r in recs], seed=seed + 16, n_boot=n_boot),
        # ★조건별로 찍는다 — 계획 있는 풀이가 blind 보다 길어질 수 있어(모델이 3개 접근을
        #   다 훑거나 계획을 되풀이하며 시작) 통째로 뭉치면 «조건별로 다른 절단율» 이라는
        #   교란이 숨는다. max_tokens=4096 에서 이 교란은 ceiling 검정((c))을 아래로
        #   편향시킬 수 있다 — 절단된 롤아웃은 boxed 답을 못 내 오답 처리되기 쉽다.
        "trunc_rate_by_cond": trunc_by_cond,
    }
    out["pass_plan_info"] = int(plan_info_pass(out))
    out["pass_selection"] = int(selection_pass(out))
    return out


def _finite(ci: dict | None, *keys) -> bool:
    ci = ci or {}
    return all(isinstance(ci.get(x), (int, float)) and math.isfinite(float(ci[x])) for x in keys)


def _excludes_zero(ci: dict) -> bool:
    return ci["lo"] > 0 or ci["hi"] < 0


def plan_info_pass(summ: dict) -> bool:
    """(c) p_best−p_donor_best CI 가 0 제외 ∧ 평균 > +.05 [ceiling, max 대 max]
    ∧ (d) p_plan_mean−p_donor_mean CI 가 0 제외 ∧ 평균 > 0 [level, mean 대 mean]."""
    c = summ.get("paired_best_minus_donorbest") or {}
    d = summ.get("paired_planmean_minus_donormean") or {}
    if not (_finite(c, "lo", "hi", "mean") and _finite(d, "lo", "hi", "mean")):
        return False
    return bool(_excludes_zero(c) and c["mean"] > PASS_DELTA_CEIL
                and _excludes_zero(d) and d["mean"] > 0)


def selection_pass(summ: dict) -> bool:
    """(b) CI 가 0 제외 ∧ 평균 > 0."""
    b = summ.get("paired_choice_minus_planmean") or {}
    if not _finite(b, "lo", "hi", "mean"):
        return False
    return bool(_excludes_zero(b) and b["mean"] > 0)


def _f(v) -> str:
    if isinstance(v, dict) and "mean" in v:
        return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}] (n={v.get('n')})"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def to_markdown(summ: dict) -> str:
    keys = ["n_problems", "k", "n_unparsed", "n_leaked", "unparsed_rate", "leak_rate",
            "plan_words", "p_blind", "p_choice", "p_plan_mean", "p_best", "p_worst",
            "p_donor_mean", "p_donor_best",
            "paired_choice_minus_blind", "paired_choice_minus_planmean",
            "paired_best_minus_blind", "paired_best_minus_donorbest",
            "paired_planmean_minus_donormean", "paired_spread_minus_donorspread",
            "sign_p_choice_minus_blind", "sign_p_choice_minus_planmean",
            "sign_p_best_minus_blind", "sign_p_best_minus_donorbest",
            "sign_p_planmean_minus_donormean", "sign_p_spread_minus_donorspread",
            "plan_spread", "donor_spread", "frac_spread_ge_thresh",
            "frac_donor_spread_ge_thresh", "frac_choice_is_argmax", "choice_hist",
            "adherence_blind", "adherence_choice", "adherence_planmean", "adherence_donormean",
            "trunc_rate_by_cond"]
    lines = ["## math_plan_gate — 풀기 **전** 계획이 인과 정보를 갖는가", "",
             "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {_f(summ.get(k))} |" for k in keys]
    lines += [
        "",
        f"**PLAN-INFO {'PASS' if summ.get('pass_plan_info') else 'FAIL'}** — "
        f"(c) p_best − p_donor_best CI 가 0 제외 ∧ 평균 > +{PASS_DELTA_CEIL} [max 대 max, ceiling] "
        "∧ (d) p_plan_mean − p_donor_mean CI 가 0 제외 ∧ 평균 > 0 [mean 대 mean, level]",
        f"**SELECTION {'PASS' if summ.get('pass_selection') else 'FAIL'}** — "
        "(b) p_choice−p_plan_mean CI 가 0 제외 ∧ 평균 > 0 "
        f"(참고선: frac_choice_is_argmax vs 1/3 = {1/3:.4f})",
        "",
    ]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout 산출물 texts.jsonl")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default="math_opt", help="그 롤아웃을 만든 프롬프트 변형")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max_problems", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--plan_tokens", type=int, default=300)
    ap.add_argument("--plan_temp", type=float, default=0.8)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    selftest_math_verify()
    rng = random.Random(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rolls = [json.loads(l) for l in open(a.rollouts)]
    probs = select_mixed_problems(rolls, max_problems=a.max_problems)
    print(f"[plan] MIXED 문제 {len(probs)}개", flush=True)
    if not probs:
        raise SystemExit("[plan] 후보가 없다 — 입력 롤아웃에 MIXED 그룹이 있는지 확인하라.")

    from vllm import LLM, SamplingParams  # noqa: PLC0415
    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 2048, enforce_eager=True)
    tok = llm.get_tokenizer()

    # ── A) 계획 한 번(temperature 0.8) ───────────────────────────────────────
    preqs = [plan_prompt(tok, a.variant, p["problem"]) for p in probs]
    pouts = llm.generate(preqs, SamplingParams(n=1, temperature=a.plan_temp, top_p=1.0,
                                               max_tokens=a.plan_tokens, seed=a.seed))
    kept: list[dict] = []
    n_unparsed = n_leaked = 0
    plan_rows = []
    for p, o in zip(probs, pouts):
        raw = o.outputs[0].text
        parsed = parse_plans(raw)
        row = {"group_id": p["group_id"], "gold": p["gold"], "raw": raw,
               "parsed": int(parsed is not None)}
        if parsed is None:
            n_unparsed += 1
            plan_rows.append({**row, "leaked": 0})
            continue
        plans, choice = parsed
        leak = plans_leak(plans, p["gold"])
        plan_rows.append({**row, "plans": plans, "choice": choice, "leaked": int(leak)})
        if leak:
            n_leaked += 1
            continue
        kept.append({**p, "plans": plans, "choice": choice,
                     "plan_words": sum(len(_WORD_RE.findall(x)) for x in plans) / N_PLANS})
    with (out / "plans.jsonl").open("w") as fh:
        for r in plan_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[plan] 계획 성립 {len(kept)} / 파싱 실패 {n_unparsed} / 누출 {n_leaked}", flush=True)
    if not kept:
        raise SystemExit("[plan] 유효한 계획이 하나도 없다 — 파싱·누출 가드를 확인하라.")

    # donor = **셋의 서로 다른 문제**의 계획(그 문제가 고른 계획을 쓴다 — 조건 (ii) 와 같은
    # 종류의 문장). 셋을 못 채운 문제는 여기서 통째로 드롭한다(n_dropped_no_donor) — 부분
    # 조건만 있는 채로 풀이까지 생성하면 뒤에서 «일곱 조건 다 있어야» 필터에 걸려 버려진다.
    donor_triples = assign_donor_triples(kept, rng, n_donor=N_DONOR)
    n_dropped_no_donor = sum(1 for d in donor_triples if d is None)
    kept = [{**s, "donor_idxs": d,
             "donor_plans": [kept[j]["plans"][kept[j]["choice"] - 1] for j in d],
             "donor_group_ids": [kept[j]["group_id"] for j in d]}
            for s, d in zip(kept, donor_triples) if d is not None]
    print(f"[plan] donor 셋 성립 {len(kept)} / 드롭(donor 부족) {n_dropped_no_donor}", flush=True)
    if not kept:
        raise SystemExit("[plan] donor 셋을 채울 수 있는 문제가 하나도 없다 — 문제 수를 늘려라.")

    # ── B) 일곱 조건 풀이(blind + plan1-3 + donor1-3) ───────────────────────────
    reqs, ix, n_drop = [], [], 0
    for si, s in enumerate(kept):
        prompts = {
            "blind": solve_prompt(tok, a.variant, s["problem"], None),
            "plan1": solve_prompt(tok, a.variant, s["problem"], s["plans"][0]),
            "plan2": solve_prompt(tok, a.variant, s["problem"], s["plans"][1]),
            "plan3": solve_prompt(tok, a.variant, s["problem"], s["plans"][2]),
            "donor1": solve_prompt(tok, a.variant, s["problem"], s["donor_plans"][0]),
            "donor2": solve_prompt(tok, a.variant, s["problem"], s["donor_plans"][1]),
            "donor3": solve_prompt(tok, a.variant, s["problem"], s["donor_plans"][2]),
        }
        for c in CONDS:
            if prompts[c] is None:
                n_drop += 1
                continue
            reqs.append(prompts[c])
            ix.append((si, c))
    print(f"[plan] 풀이 요청 {len(reqs)}개 x K={a.k} "
          f"(문제 {len(kept)} x 조건 {len(CONDS)} x K={a.k} = {len(kept) * len(CONDS) * a.k}개 "
          f"생성 예정, 버린 것 {n_drop}개)", flush=True)
    outs = llm.generate(reqs, SamplingParams(n=a.k, temperature=1.0, top_p=1.0,
                                             max_tokens=a.max_tokens, seed=a.seed))

    agg: dict = {}
    rows = []
    for (si, c), o in zip(ix, outs):
        s = kept[si]
        ref = {"blind": "", "plan1": s["plans"][0], "plan2": s["plans"][1],
               "plan3": s["plans"][2], "donor1": s["donor_plans"][0],
               "donor2": s["donor_plans"][1], "donor3": s["donor_plans"][2]}[c]
        # ★blind 의 준수 프록시는 «고른 계획» 을 기준으로 잰다 — 계획 없이 푼 풀이가 이미
        #   그 계획의 어휘를 쓰고 있으면 (ii) 의 높은 Jaccard 는 조향의 증거가 아니다.
        ref_blind = s["plans"][s["choice"] - 1]
        for x in o.outputs:
            adh = adherence(ref_blind if c == "blind" else ref, x.text)
            corr = grade_math(x.text, s["gold"])
            trunc = int(x.finish_reason == "length")
            agg.setdefault((si, c), []).append({"r_corr": corr, "adh": adh, "trunc": trunc})
            rows.append({"group_id": s["group_id"], "cond": c, "r_corr": corr,
                         "adherence": adh, "truncated": trunc, "text": x.text})

    def _adh(vals):
        v = [x["adh"] for x in vals if isinstance(x["adh"], float) and math.isfinite(x["adh"])]
        return _p(v)

    recs = []
    for si, s in enumerate(kept):
        got = {c: agg.get((si, c)) for c in CONDS}
        if not all(got.values()):
            continue                     # 일곱 조건이 다 있어야 짝 비교가 성립한다
        pl = [_p([x["r_corr"] for x in got[f"plan{j}"]]) for j in (1, 2, 3)]
        dn = [_p([x["r_corr"] for x in got[f"donor{j}"]]) for j in (1, 2, 3)]
        ad = {f"plan{j}": _adh(got[f"plan{j}"]) for j in (1, 2, 3)}
        ad_donor = [_adh(got[f"donor{j}"]) for j in (1, 2, 3)]
        rec = per_problem_record({
            "group_id": s["group_id"], "problem": s["problem"], "gold": s["gold"],
            "p_group": s["p_group"], "plans": s["plans"], "choice": s["choice"],
            "plan_words": s["plan_words"], "donor_group_ids": s["donor_group_ids"],
            "donor_plans": s["donor_plans"],
            "p_blind": _p([x["r_corr"] for x in got["blind"]]),
            "p_donor": dn,
            "p_plan": pl,
            "adherence_blind": _adh(got["blind"]),
            "adherence_donormean": _p([v for v in ad_donor if math.isfinite(v)]),
            "adherence_planmean": _p([v for v in ad.values() if math.isfinite(v)]),
            "adherence_plan": [ad["plan1"], ad["plan2"], ad["plan3"]],
            "trunc_rate_by_cond": {c: _p([x["trunc"] for x in got[c]]) for c in CONDS},
        })
        rec["adherence_choice"] = rec["adherence_plan"][rec["choice"] - 1]
        recs.append(rec)

    summ = summarize(recs, k=a.k, seed=a.seed, n_boot=a.n_boot, n_unparsed=n_unparsed,
                     n_leaked=n_leaked)
    summ.update({"model_path": a.model_path, "variant": a.variant, "rollouts": a.rollouts,
                 "seed": a.seed, "n_candidates": len(probs),
                 "n_dropped_no_donor": n_dropped_no_donor, "n_dropped_missing_cond": n_drop,
                 "n_generations": len(rows)})

    with (out / "per_problem.jsonl").open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "solves.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "gate_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    (out / "gate_summary.md").write_text(to_markdown(summ))
    print(to_markdown(summ))
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
