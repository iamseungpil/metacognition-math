#!/usr/bin/env python
r"""math_anti_teacher_ruler — S1 «반면교사 자»: 틀린 시도 w 가 끌개(attractor)로 남아 있는지를
**생성 없이**(채점 전용 forward) 토큰 단위로 잰다.

무엇을 재는가. cd9 활성화 관문(`docs/analysis/CONTAM_activation_0915.md` A1)이 실측한 것:
같은 정책이 자기 오답 w 를 문맥에 들고 있으면 그 **답**을 다시 낸다(`wait` .796 / `external`
.447), 그런데 «이전 시도가 틀렸다»는 **사실 f** 만 주고 w 를 안 보여주면 .185 로 떨어진다.
즉 w 는 텍스트를 베끼게 하는 게 아니라 **답 하나를 끌어당긴다**.

가설 H2. 그 끌림은 **토큰 단위 대비**로 읽힌다:

    Δ_t = log π(y_t | x, w, f, y_<t)  −  log π(y_t | x, f, y_<t)

여기서 y 는 정책이 **사실만 준 문맥**(`blind_external`)에서 스스로 낸 재시도다. Δ 가 크다는
것은 «이 재시도는 w 를 봤더라면 더 쉽게 썼을 글»이라는 뜻이고, 그런 재시도는 (a) 옛 오답을
다시 내고 (b) 틀릴 것이다 — **문제 안에서**. 문제 안 신호라야 GRPO 의 그룹 중심화가 지우지
못한다(그래서 pooled AUC 가 아니라 **within-problem AUC** 가 관문이다).

문맥 조립은 `math_activation_gate` 의 빌더를 **그대로 import** 한다 — 생성물(gens.jsonl)을
만든 프롬프트와 한 바이트라도 달라지면 Δ 는 «끌림»이 아니라 «프롬프트 차이»를 잰다.
  C_clean = `blind_external_prompt(x)`               (x + f, w 없음)  ← y 를 만든 바로 그 문맥
  C_anti  = `external_prompt(x, w)`                  (x + w + f)
  C_donor = `external_prompt(x, w_donor)`            (같은 문제, **다른 문제의** 오답)

★통제(둘 다 필수):
  1. donor-anti — 문제 안 AUC 가 donor 에서도 같으면 그 신호는 «그 특정 끌개»가 아니라
     «assistant 턴에 뭔가 긴 글이 하나 더 있다»일 뿐이다.
  2. 길이 — ΣΔ 는 토큰 수에 비례해 자란다. ΣΔ 와 n_tok 의 상관, 그리고 **길이만으로의 AUC**
     를 같이 찍는다(길이가 이미 .75 를 내면 Δ 는 아무것도 더 말하지 않는다).

★관문(ANTI-TEACHER PASS), 셋 다:
  (1) within-problem AUC(ΣΔ 또는 boxΔ → reemit) ≥ .75
  (2) 그 같은 자에서 own − donor ≥ .10
  (3) 재순위(문제마다 ΣΔ 최소 재시도를 고름) 정확도 > maj@8 ∧ 짝 부트스트랩 CI 가 0 제외

maj@8 는 `docs/RESULTS_cd9.md`(2026-09-15 정본화)의 규약이다 — 수학 동치 군집 · 무응답 제외 ·
동률은 최초 등장 · 채점은 `grade_math("\boxed{투표답}", gold)`. 재순위·무작위·오라클도 같은
채점을 쓴다(선택자끼리 비교하려면 채점이 같아야 한다).

★생성은 **한 토큰도** 하지 않는다. 존재하는 생성물 위의 순수 채점 패스다.

사용(예):
  python scripts/local/math_anti_teacher_ruler.py \
      --gens $WORK/eval/activation_gate_s1/gens.jsonl \
      --rollouts $WORK/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path $WORK/models/Qwen3-4B-Instruct-2507 --variant math_opt \
      --out_dir $WORK/eval/anti_teacher_s1

★B1 «기억상실 교사 신호 국소화»(`--cond wait`) — 같은 기계, 다른 물음. B3 는 KL(교사‖학생)
항을 붙이려 한다: 학생 = π(·|x, w, WAIT_CUE)(자기 오답을 이어 쓰는 정책), 교사 = π(·|x, f)
(같은 정책, **오답 본문이 없는** 리셋 문맥 = `blind_external_prompt`). 학습 전에 그 로그비가
실제 학생 표본 위에서 **어디에 사는지** 본다:

    L_t = log π(y_t | x, f, y_<t) − log π(y_t | x, w, WAIT_CUE, y_<t)

y 는 활성화 게이트의 `wait` 이어쓰기(150 오답 × K=8)다. L<0 = 교사가 학생이 쓴 것을 싫어한다.
문맥은 생성기와 **같은 빌더**로 짓는다(`A.wait_prompt` / `A.blind_external_prompt`) — 한
바이트라도 다르면 L 은 «프롬프트 차이»다. 교사 문맥은 user 턴이 다르게 끝나므로 y 의 **첫
토큰들은 교사에게 분포 밖**이다 → 모든 자를 첫 16 토큰을 뺀 판(`skip16_*`)과 **같이** 낸다.
donor 대조는 **학생 쪽**을 바꾼다(남의 오답 w′ 를 앞부분으로): donor 에서도 똑같이 국소화되면
그 신호는 그 특정 끌개가 아니라 «앞부분에 무슨 오답이든 있다» 일 뿐이다.

★관문(AMNESIC-TEACHER PASS) — 같은 자(boxL 또는 tailL)에서 셋 다:
  (1) 문제 안 AUC(→reemit, 방향 고정) ≥ .75  (2) own − donor ≥ .10
  (3) 집중 비율 conc_ratio = (꼬리 ∪ 박스의 |L| 분수) / (그 구간의 **토큰 몫**) 이 평균 ≥ 2.0
      ∧ 부트스트랩 CI 하한 > 1.0 — 즉 KL 이 **재발화 결정**에 걸리지 문체에 걸리지 않는다.
      (절대 분수 ≥ .50 은 보통 길이의 이어쓰기에서 구조적으로 불가능해 쓰지 않는다. 절대
      분수는 참고용으로 요약에 남는다.)
★boxL 은 새 `\boxed` 가 없는 이어쓰기(=답을 상속한 재발화, wait 에서 다수)에서 **NaN** 이다.
  NaN 은 0.5 로 눙치지 않고 그 자에서 **뺀다** — 자마다 `n_valid` 를 찍고, boxL 의 n_valid 가
  이어쓰기의 절반에 못 미치면 마크다운이 경고한다(그 AUC 는 부분집합 위의 수다).

  python scripts/local/math_anti_teacher_ruler.py --cond wait \
      --gens $WORK/eval/activation_gate_s1/gens.jsonl \
      --rollouts $WORK/eval/mathL5_q3i2507_opt_b8k/texts.jsonl \
      --model_path $WORK/models/Qwen3-4B-Instruct-2507 --variant math_opt \
      --out_dir $WORK/eval/amnesic_teacher_s1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math_activation_gate as A  # noqa: E402  (★프롬프트 빌더 — 복사하지 않고 그대로 쓴다)
import math_geometry_probe as G  # noqa: E402  (spearman)
import math_ruler_pivot as P  # noqa: E402  (Job/_enc/hf_forward_factory/auc)
from math_cited_site_gate import bootstrap_ci  # noqa: E402
from math_critique_ig_ruler import roll_ids  # noqa: E402  (roll_id 규약을 한 곳에)
from src.training.math_meta import (  # noqa: E402
    answers_equivalent, boxed_spans, grade_math, last_boxed,
)

_NAN = float("nan")
COND_FACT = "blind_external"          # y 를 만든 조건(사실만, 원 풀이 없음)
COND_WAIT = "wait"                    # ★B1 — y 를 만든 조건(자기 오답 + WAIT_CUE 이어쓰기)
CONTEXTS = ("clean", "anti", "donor")
TAIL_TOKENS = 64                      # tailΔ 창
LOCAL_TAIL_FRAC = 0.10                # 국소화: 마지막 10% 토큰
FIRST_K_SKIP = 16                     # ★교사 문맥에서 y 의 첫 토큰들은 분포 밖이다 — 뺀 판도 같이
STOP_EARLY_CHARS = 1000               # CONTAM A2b 의 «63% 버킷»(짧게 멈춘 이어쓰기)
PASS_WITHIN_AUC = 0.75
PASS_OWN_MINUS_DONOR = 0.10
PASS_MASS_FRAC = 0.50                 # (참고용 절대 분수 — 관문에는 쓰지 않는다)
PASS_CONC_RATIO = 2.0                 # ★B1 (3) — |L| 질량 집중 비율(몫 대비 배수)
SCORES = ("sum_delta", "mean_delta", "box_delta", "tail_delta")
SKIP = f"skip{FIRST_K_SKIP}_"
SCORES_SKIP = tuple(SKIP + s for s in SCORES)
LABELS = ("reemit", "wrong")

# ── 모드(=--cond) ───────────────────────────────────────────────────────────────
# 한 벌의 기계(문맥 조립 · forward · 특징 · AUC · 부트스트랩 · 국소화)를 두 질문에 쓴다.
#   S1(기본, cond=blind_external): y 는 «사실만» 문맥에서 난 재시도.
#       Δ_t = log π(y_t | x,w,f) − log π(y_t | x,f)   (anti − clean) — «w 를 봤다면 더 쉬웠나»
#   B1(cond=wait):              y 는 **자기 오답 w + WAIT_CUE** 를 이어 쓴 이어쓰기.
#       L_t = log π(y_t | x,f) − log π(y_t | x,w,WAIT_CUE)  (teach − stud) — B3 의 KL(교사‖학생)
#       이 실제로 어디에 힘을 주는지. L<0 = «기억상실 교사가 학생이 쓴 것을 싫어한다».
#   ★orient: 가설이 정한 **한 방향**. B1 은 «재발화(reemit)일수록 L 이 더 음수» 이므로 관문은
#     −L 의 AUC(=1−AUC)를 쓴다(양방향 최대값을 쓰면 관문이 두 배로 헐거워진다).
MODE_SPECS = {
    COND_FACT: {"contexts": ("clean", "anti", "donor"),
                "own": ("anti", "clean"), "donor": ("donor", "clean"),
                "orient": +1, "inherit_answer": False,
                "out_name": "per_retry.jsonl", "gate_key": "pass_anti_teacher"},
    COND_WAIT: {"contexts": ("teach", "stud", "stud_donor"),
                "own": ("teach", "stud"), "donor": ("teach", "stud_donor"),
                "orient": -1, "inherit_answer": True,
                "out_name": "per_continuation.jsonl", "gate_key": "pass_amnesic_teacher"},
}


def mode_spec(cond: str) -> dict:
    if cond not in MODE_SPECS:
        raise SystemExit(f"[anti] 모르는 --cond {cond!r} — 가능한 것: {', '.join(MODE_SPECS)}")
    return MODE_SPECS[cond]


# ── 토큰↔문자 정렬 ──────────────────────────────────────────────────────────────
def enc_with_offsets(tok, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """(ids, [(char_start, char_end) per token]). 세 단계로 내려간다:
      1) fast 토크나이저의 return_offsets_mapping (정확)
      2) 토큰 조각 decode 를 원문에서 앞으로 찾아 맞추기 (근사 — 못 찾으면 폭 0)
      3) 공백 분할 (MockTok 용 — `MockTok.encode` 와 같은 규약이라 시험이 돈다)
    ★어느 단계든 len(ids) == len(offsets) 를 지킨다."""
    try:
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
        if len(ids) == len(offs):
            return ids, offs
    except Exception:
        pass
    ids = P._enc(tok, text)
    if hasattr(tok, "decode"):
        offs, cur = [], 0
        for i in ids:
            piece = tok.decode([i])
            j = text.find(piece, cur) if piece else -1
            if j < 0:
                offs.append((cur, cur))
            else:
                offs.append((j, j + len(piece)))
                cur = j + len(piece)
        return ids, offs
    # MockTok: 공백 분할과 1:1
    offs, cur = [], 0
    for w in text.split():
        j = text.find(w, cur)
        offs.append((j, j + len(w)))
        cur = j + len(w)
    return ids[:len(offs)], offs[:len(ids)]


def spans_to_token_mask(offsets: Sequence[tuple[int, int]],
                        char_spans: Sequence[tuple[int, int]]) -> list[bool]:
    """문자 구간들과 **겹치는** 토큰을 True 로. 폭 0 토큰은 겹치지 않는다."""
    out = []
    for (a, b) in offsets:
        out.append(any(a < e and s < b for (s, e) in char_spans))
    return out


def final_boxed_span(text: str) -> Optional[tuple[int, int]]:
    """마지막 \\boxed{...} 의 문자 구간(균형 스캐너). 없으면 None."""
    sp = boxed_spans(text or "")
    if not sp:
        return None
    _, s, e = sp[-1]
    return (s, e)


def literal_spans(text: str, needle: str) -> list[tuple[int, int]]:
    """needle 이 **문자 그대로** 나타나는 모든 구간(겹침 없이 앞에서부터). 빈 needle → []."""
    text, needle = text or "", (needle or "").strip()
    if not needle:
        return []
    out, i = [], 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            return out
        out.append((j, j + len(needle)))
        i = j + len(needle)


# ── Δ 특징 ──────────────────────────────────────────────────────────────────────
def _fsum(xs: Sequence[float]) -> float:
    v = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(v) if v else _NAN


def _fmean(xs: Sequence[float]) -> float:
    v = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return (sum(v) / len(v)) if v else _NAN


def delta_features(delta: Sequence[float], offsets: Sequence[tuple[int, int]],
                   box_span: Optional[tuple[int, int]],
                   old_ans_spans: Sequence[tuple[int, int]],
                   *, tail: int = TAIL_TOKENS,
                   local_frac: float = LOCAL_TAIL_FRAC) -> dict:
    """토큰별 Δ → 이 재시도의 스칼라 자들 + 국소화 진단.
    box_delta 는 **마지막 \\boxed 구간** 토큰 합, tail_delta 는 마지막 `tail` 토큰 합,
    oldans_delta 는 옛 오답 문자열과 겹치는 토큰 합."""
    n = len(delta)
    box_mask = spans_to_token_mask(offsets, [box_span] if box_span else [])
    old_mask = spans_to_token_mask(offsets, old_ans_spans)
    absv = [abs(float(d)) for d in delta if isinstance(d, (int, float)) and math.isfinite(float(d))]
    tot_abs = sum(absv) if absv else _NAN
    n_tail_local = max(1, int(round(n * local_frac))) if n else 0
    tail_mask = [i >= n - n_tail_local for i in range(n)] if n_tail_local else [False] * n
    abs_tail = _fsum([abs(float(d)) for d in list(delta)[n - n_tail_local:]]) if n_tail_local else _NAN
    abs_box = _fsum([abs(float(d)) for d, m in zip(delta, box_mask) if m])
    # ★꼬리 ∪ 박스 — 겹치는 토큰을 두 번 세지 않는다(B1 관문 (3) 이 쓰는 분수).
    abs_union = _fsum([abs(float(d)) for d, mb, mt in zip(delta, box_mask, tail_mask) if (mb or mt)])
    n_union = sum(1 for mb, mt in zip(box_mask, tail_mask) if (mb or mt))

    def _frac(x):
        return (x / tot_abs) if (isinstance(tot_abs, float) and math.isfinite(tot_abs)
                                 and tot_abs > 0 and isinstance(x, float)
                                 and math.isfinite(x)) else _NAN
    return {
        "n_tok": n,
        "sum_delta": _fsum(delta),
        "mean_delta": _fmean(delta),
        "box_delta": _fsum([d for d, m in zip(delta, box_mask) if m]),
        "tail_delta": _fsum(list(delta)[max(0, n - tail):]),
        "oldans_delta": _fsum([d for d, m in zip(delta, old_mask) if m]),
        "n_tok_box": int(sum(box_mask)),
        "n_tok_oldans": int(sum(old_mask)),
        "abs_mass_total": tot_abs,
        "frac_abs_mass_last10pct": _frac(abs_tail),
        "frac_abs_mass_boxed": _frac(abs_box),
        "frac_abs_mass_tail_or_box": _frac(abs_union),
        # ★집중 비율 — «그 구간이 차지하는 토큰 몫» 대비 몇 배의 |L| 질량인가. 절대 분수는
        #   긴 이어쓰기에서 구조적으로 작아진다(꼬리 10% + 박스 몇 토큰뿐이므로).
        "expected_mass_share": (n_union / n) if n else _NAN,
        "conc_ratio": ((_frac(abs_union) / (n_union / n))
                       if (n and n_union and isinstance(_frac(abs_union), float)
                           and math.isfinite(_frac(abs_union))) else _NAN),
    }


def delta_features_all(delta: Sequence[float], offsets: Sequence[tuple[int, int]],
                       box_span: Optional[tuple[int, int]],
                       old_ans_spans: Sequence[tuple[int, int]],
                       *, skip: int = FIRST_K_SKIP) -> dict:
    """전체 판 + **첫 `skip` 토큰을 뺀** 판(`skip16_` 접두). 교사 문맥은 user 턴이 다르게
    끝나므로 y 의 첫 토큰들이 교사에겐 분포 밖이다 — 두 판을 늘 같이 보고한다."""
    out = delta_features(delta, offsets, box_span, old_ans_spans)
    tail_d, tail_o = list(delta)[skip:], list(offsets)[skip:]
    sub = (delta_features(tail_d, tail_o, box_span, old_ans_spans) if tail_d
           else {k: (_NAN if isinstance(v, float) else 0) for k, v in out.items()})
    out.update({SKIP + k: v for k, v in sub.items()})
    return out


def mean_by_decile(delta: Sequence[float], n_bins: int = 10) -> list[float]:
    """위치 십분위별 평균 Δ(국소화 히스토그램의 한 줄). 토큰이 없으면 NaN 벡터."""
    n = len(delta)
    if n == 0:
        return [_NAN] * n_bins
    out = []
    for b in range(n_bins):
        lo, hi = (n * b) // n_bins, (n * (b + 1)) // n_bins
        out.append(_fmean(list(delta)[lo:max(hi, lo + 1)]))
    return out


# ── 라벨 ────────────────────────────────────────────────────────────────────────
def label_retry(gen_text: str, orig_wrong_ans: str, gold: str, *,
                inherit_answer: bool = False) -> dict:
    """reemit(옛 오답과 동치) / wrong(오답) / changed_and_right(답을 바꿨고 맞음) /
    stopped_early(1,000자 미만 — CONTAM A2b 의 «짧게 멈춘» 버킷).
    ★채점은 «박스 답을 한 번 채점» — maj@8 정본과 같은 규약이라 선택자끼리 비교 가능하다.
    ★`inherit_answer`(B1/wait 전용): 이어쓰기에 **새 박스가 없으면** 최종 답은 앞부분의 옛
    오답 그대로다(`math_activation_gate` 의 no_new_boxed 규약과 같다) — 그래서 그 경우를
    reemit 로 센다. 그 사실은 `inherited_answer` 로 남긴다."""
    ans = last_boxed(gen_text or "")
    inherited = int(inherit_answer and not ans and bool(orig_wrong_ans))
    eff = orig_wrong_ans if inherited else ans
    corr = boxed_correct(eff, gold)
    reemit = int(bool(eff) and bool(orig_wrong_ans) and answers_equivalent(orig_wrong_ans, eff))
    return {"ans": ans, "eff_ans": eff, "inherited_answer": inherited, "r_corr": corr,
            "reemit": reemit, "wrong": int(not corr),
            "changed_and_right": int(corr and not reemit), "no_answer": int(not ans),
            "n_chars": len(gen_text or ""),
            "stopped_early": int(len(gen_text or "") < STOP_EARLY_CHARS)}


def boxed_correct(ans: str, gold: str) -> int:
    """`grade_math("\\boxed{ans}", gold)` — 맨 문자열은 `\\dfrac{1}{3}` 에서 파싱이 깨진다
    (RESULTS_cd9 C4 함정). 빈 답은 0."""
    a = (ans or "").strip()
    if not a:
        return 0
    return int(grade_math("\\boxed{" + a + "}", gold))


# ── maj@8 (RESULTS_cd9 2026-09-15 정본) ─────────────────────────────────────────
def majority_answer(answers: Sequence[str]) -> Optional[str]:
    """수학 동치 군집 · 무응답 **제외** · 동률은 **최초 등장**. 표가 하나도 없으면 None.
    ★`math_dis.plurality_answer` 와 다른 점은 동률 규칙뿐이다(그쪽은 None = 미정)."""
    clusters: list[list[str]] = []
    for a in answers:
        s = str(a or "").strip()
        if not s:
            continue
        for c in clusters:
            if answers_equivalent(c[0], s):
                c.append(s)
                break
        else:
            clusters.append([s])
    if not clusters:
        return None
    top = max(len(c) for c in clusters)
    for c in clusters:                     # 최초 등장 순서 = clusters 의 순서
        if len(c) == top:
            return c[0]
    return None


def majority_correct(answers: Sequence[str], gold: str) -> int:
    v = majority_answer(answers)
    return boxed_correct(v, gold) if v else 0


# ── AUC (pooled / within-problem) ───────────────────────────────────────────────
def rank_normalize(values: Sequence[float]) -> list[float]:
    """행 안 평균 순위를 [0,1] 로. 유한하지 않은 값은 NaN 그대로. n<2 면 전부 0.5."""
    idx = [i for i, v in enumerate(values)
           if isinstance(v, (int, float)) and math.isfinite(float(v))]
    out = [_NAN] * len(values)
    if not idx:
        return out
    if len(idx) < 2:
        out[idx[0]] = 0.5
        return out
    order = sorted(idx, key=lambda i: float(values[i]))
    ranks: dict[int, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and float(values[order[j + 1]]) == float(values[order[i]]):
            j += 1
        r = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    denom = len(order) - 1
    for i in idx:
        out[i] = ranks[i] / denom
    return out


def _finite_pairs(retries: Sequence[dict], score_key: str,
                  label_key: str) -> tuple[list[int], list[float]]:
    """그 자에서 **점수가 유한한** 이어쓰기만. ★boxΔ 는 새 박스가 없으면 NaN 이다(wait 모드의
    ~70%) — NaN 을 0.5 나 0 으로 눙치지 않고 그 행에서 **뺀다**."""
    ys, ss = [], []
    for x in retries:
        v = x.get(score_key, _NAN)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            ys.append(int(x[label_key]))
            ss.append(float(v))
    return ys, ss


def n_valid(rows: Sequence[dict], score_key: str) -> int:
    return sum(1 for r in rows for x in r["retries"]
               if isinstance(x.get(score_key, _NAN), (int, float))
               and math.isfinite(float(x.get(score_key, _NAN))))


def within_auc(rows: Sequence[dict], score_key: str, label_key: str) -> dict:
    """문제(행) **안**에서만 잰 AUC 의 평균 + 두 클래스가 다 있는 행 수, 그리고 행 안
    순위정규화 점수를 모아 한 번에 잰 Mann-Whitney AUC.
    ★NaN 점수는 먼저 버리고, **남은 것들**에 두 클래스가 다 있는 행만 센다.
    rows: [{"retries": [{score_key:…, label_key:…}, …]}, …]"""
    per, pooled_y, pooled_s = [], [], []
    for r in rows:
        ys, ss = _finite_pairs(r["retries"], score_key, label_key)
        if len(set(ys)) < 2:
            continue
        a = P.auc(ys, ss)
        if math.isfinite(a):
            per.append(a)
        nz = rank_normalize(ss)
        pooled_y += ys
        pooled_s += nz
    return {"auc": (sum(per) / len(per)) if per else _NAN,
            "n_rows_both_classes": len(per),
            "n_valid": n_valid(rows, score_key),
            "auc_ranknorm_pooled": P.auc(pooled_y, pooled_s) if pooled_y else _NAN}


def pooled_auc(rows: Sequence[dict], score_key: str, label_key: str) -> float:
    """★NaN 점수는 버린다(P.auc 와 같은 규약을 여기서 명시한다). 남은 것이 없거나 한
    클래스뿐이면 NaN."""
    y, s = _finite_pairs([x for r in rows for x in r["retries"]], score_key, label_key)
    if len(set(y)) < 2:
        return _NAN
    return P.auc(y, s)


# ── 재순위 ──────────────────────────────────────────────────────────────────────
def _argmin(values: Sequence[float]) -> Optional[int]:
    idx = [i for i, v in enumerate(values)
           if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not idx:
        return None
    return min(idx, key=lambda i: (float(values[i]), i))     # 동률은 **최초 등장**


def rerank_row(retries: Sequence[dict], gold: str) -> dict:
    """한 문제에서 다섯 선택자의 정확도. random 은 표본이 아니라 **기댓값**(8개의 평균)."""
    corr = [int(x["r_corr"]) for x in retries]
    i_sum = _argmin([x["sum_delta"] for x in retries])
    i_box = _argmin([x["box_delta"] for x in retries])
    return {
        "pick_sum": corr[i_sum] if i_sum is not None else 0,
        "pick_box": corr[i_box] if i_box is not None else 0,
        "pick_sum_idx": i_sum, "pick_box_idx": i_box,
        "maj8": majority_correct([x.get("eff_ans", x["ans"]) for x in retries], gold),
        "random": (sum(corr) / len(corr)) if corr else _NAN,
        "oracle": max(corr) if corr else 0,
    }


def paired_boot(a: Sequence[float], b: Sequence[float], *, seed: int = 0,
                n_boot: int = 2000) -> dict:
    """짝(행) 부트스트랩 — a−b 의 평균과 CI."""
    return bootstrap_ci([float(x) - float(y) for x, y in zip(a, b)], seed=seed, n_boot=n_boot)


# ── donor 배정 ──────────────────────────────────────────────────────────────────
def rotate_donors(n: int) -> list[int]:
    """i → (i+1) % n 회전. **어느 행도 자기 자신을 받지 않는다**(n≥2). n<2 면 빈 배정."""
    if n < 2:
        return [-1] * n
    return [(i + 1) % n for i in range(n)]


# ── 행 조립 ─────────────────────────────────────────────────────────────────────
def build_rows(gens: Sequence[dict], rolls: Sequence[dict], *, cond: str = COND_FACT,
               limit: int = 0) -> list[dict]:
    """gens.jsonl 의 `cond` 재시도를 roll_id 로 원 오답 롤아웃(문제·gold·w)에 붙인다."""
    by_id = {rid: r for rid, r in zip(roll_ids(rolls), rolls)}
    order: list[str] = []
    buckets: dict[str, list[str]] = {}
    for g in gens:
        if g.get("population") != "wrong" or g.get("cond") != cond:
            continue
        rid = g["roll_id"]
        if rid not in by_id:
            continue
        if rid not in buckets:
            buckets[rid] = []
            order.append(rid)
        buckets[rid].append(g.get("text") or "")
    rows = []
    for rid in order:
        src = by_id[rid]
        rows.append({"roll_id": rid, "group_id": src["group_id"], "problem": src["problem"],
                     "gold": src["gold"], "wrong_text": src["text"],
                     "wrong_ans": last_boxed(src["text"]), "retry_texts": buckets[rid]})
    if limit:
        rows = rows[:limit]
    d = rotate_donors(len(rows))
    for i, r in enumerate(rows):
        r["donor_idx"] = d[i]
        r["donor_roll_id"] = rows[d[i]]["roll_id"] if d[i] >= 0 else None
        r["donor_wrong_text"] = rows[d[i]]["wrong_text"] if d[i] >= 0 else None
    return [r for r in rows if r["donor_wrong_text"]]


def build_contexts(tok, variant: str, row: dict, cond: str = COND_FACT) -> dict:
    """모드별 세 문맥. ★`math_activation_gate` 의 빌더를 그대로 부른다 — 바이트 동일이 정의상
    보장되고 `tests/test_math_anti_teacher_ruler.py` 가 그 사실을 고정한다."""
    p, w, dw = row["problem"], row["wrong_text"], row["donor_wrong_text"]
    if cond == COND_WAIT:
        ctx = {
            "teach": A.blind_external_prompt(tok, variant, p),        # 기억상실 교사 = 사실만
            "stud": A.wait_prompt(tok, variant, p, w),                # 학생 = 자기 오답 + cue
            "stud_donor": A.wait_prompt(tok, variant, p, dw),         # 대조 = 남의 오답 + cue
        }
        # ★y 를 만든 바로 그 프롬프트여야 한다(생성기와 같은 빌더 · 같은 꼬리).
        assert ctx["stud"].endswith(A.WAIT_CUE) and (w or "") in ctx["stud"]
        return ctx
    mode_spec(cond)
    return {
        "clean": A.blind_external_prompt(tok, variant, p),
        "anti": A.external_prompt(tok, variant, p, w),
        "donor": A.external_prompt(tok, variant, p, dw),
    }


def build_jobs(tok, rows: Sequence[dict], variant: str, *, max_len: int = 20480,
               cond: str = COND_FACT) -> dict:
    """재시도 하나당 세 forward(clean/anti/donor). 셋 중 하나라도 max_len 을 넘으면 그 재시도를
    통째로 버린다 — 한 문맥만 왼쪽에서 잘리면 Δ_t 가 «끌림»이 아니라 «잘림»을 잰다."""
    spec = mode_spec(cond)
    ctx_names = spec["contexts"]
    jobs: list[P.Job] = []
    index: list[dict] = []
    n_drop = 0
    for r in rows:
        ctx = build_contexts(tok, variant, r, cond)
        heads = {c: P._enc(tok, ctx[c]) for c in ctx_names}
        rec = {k: v for k, v in r.items()
               if k not in ("wrong_text", "donor_wrong_text", "retry_texts")}
        rec["retries"] = []
        for ki, y in enumerate(r["retry_texts"]):
            tids, offs = enc_with_offsets(tok, y)
            if not tids or any(len(heads[c]) + len(tids) > max_len for c in ctx_names):
                n_drop += 1
                continue
            slot = {"k": ki, "text": y, "tok_ids": tids, "offsets": offs, "jobs": {}}
            for c in ctx_names:
                ids = heads[c] + tids
                slot["jobs"][c] = len(jobs)
                jobs.append(P.Job(ids, tok_lp_spans=[(len(heads[c]), len(ids))]))
            rec["retries"].append(slot)
        if rec["retries"]:
            index.append(rec)
    return {"jobs": jobs, "index": index, "n_dropped_long": n_drop, "cond": cond}


def retry_records(res: Sequence[dict], built: dict) -> list[dict]:
    """forward 결과 → 행별(문제별) 레코드. 이어쓰기마다 Δ_t(own) 와 Δ_t(donor) 를 만든다.
    ★S1 은 Δ = anti − clean, B1 은 L = teach − stud — 어느 쪽이든 키 이름은 같다
    (`delta`/`sum_delta`/…) 라서 per_retry.jsonl 과 per_continuation.jsonl 이 같은 스키마다."""
    cond = built.get("cond", COND_FACT)
    spec = mode_spec(cond)
    ctx_names = spec["contexts"]
    (oa, ob), (da, db) = spec["own"], spec["donor"]
    rows = []
    for it in built["index"]:
        gold, wans = it["gold"], it["wrong_ans"]
        out = {k: v for k, v in it.items() if k != "retries"}
        out["retries"] = []
        for s in it["retries"]:
            lp = {c: res[s["jobs"][c]]["tok_lp"][0] for c in ctx_names}
            d_own = [a - b for a, b in zip(lp[oa], lp[ob])]
            d_don = [a - b for a, b in zip(lp[da], lp[db])]
            box = final_boxed_span(s["text"])
            old = literal_spans(s["text"], wans)
            feats = delta_features_all(d_own, s["offsets"], box, old)
            dfeats = delta_features_all(d_don, s["offsets"], box, old)
            rec = {"k": s["k"], "cond": cond, "tok_ids": s["tok_ids"],
                   "delta": d_own, "delta_donor": d_don,
                   "decile_delta": mean_by_decile(d_own)}
            rec.update({f"mean_logp_{c}": _fmean(lp[c]) for c in ctx_names})
            rec.update(feats)
            rec.update({f"donor_{k}": v for k, v in dfeats.items() if k != "n_tok"})
            rec.update(label_retry(s["text"], wans, gold,
                                   inherit_answer=spec["inherit_answer"]))
            out["retries"].append(rec)
        rows.append(out)
    return rows


def top_negative_tokens(rows: Sequence[dict], tok=None, *, top: int = 20,
                        min_count: int = 5) -> list[dict]:
    """토큰(id)별 평균 Δ 가 가장 **음수**인 것 top 개 — B1 에서는 «교사가 가장 싫어한 토큰».
    답을 되뇌는 토큰(숫자·`\\boxed`·`=`)이 나오면 KL 이 재발화 결정에 걸린다는 뜻이다."""
    agg: dict = {}
    for r in rows:
        for x in r["retries"]:
            for tid, d in zip(x["tok_ids"], x["delta"]):
                if not (isinstance(d, (int, float)) and math.isfinite(float(d))):
                    continue
                a = agg.setdefault(int(tid), [0.0, 0])
                a[0] += float(d)
                a[1] += 1
    items = [{"token_id": t, "mean_delta": s / n, "n": n}
             for t, (s, n) in agg.items() if n >= min_count]
    items.sort(key=lambda z: (z["mean_delta"], z["token_id"]))
    items = items[:top]
    for z in items:
        try:
            z["token"] = tok.decode([z["token_id"]]) if tok is not None else ""
        except Exception:
            z["token"] = ""
    return items


# ── 요약 ────────────────────────────────────────────────────────────────────────
def _orient(a: float, orient: int) -> float:
    """가설이 정한 한 방향으로 AUC 를 세운다. orient=-1 이면 −점수의 AUC(=1−AUC)."""
    if not (isinstance(a, (int, float)) and math.isfinite(float(a))):
        return _NAN
    return float(a) if orient > 0 else 1.0 - float(a)


def summarize(rows: Sequence[dict], *, seed: int = 11, n_boot: int = 2000,
              cond: str = COND_FACT) -> dict:
    spec = mode_spec(cond)
    orient = spec["orient"]
    n_rows = len(rows)
    n_ret = sum(len(r["retries"]) for r in rows)
    summ: dict = {"n_rows": n_rows, "n_retries": n_ret, "cond": cond, "orient": orient,
                  "rate": {k: _fmean([x.get(k, _NAN) for r in rows for x in r["retries"]])
                           for k in ("reemit", "wrong", "changed_and_right", "no_answer",
                                     "stopped_early", "inherited_answer")}}

    # AUC 표 — 여덟 자(전체 + 첫 16 토큰 제외) × 두 라벨, pooled 와 within-problem.
    auc_tab: dict = {}
    for sk in SCORES + SCORES_SKIP:
        for lk in LABELS:
            w = within_auc(rows, sk, lk)
            auc_tab[f"{sk}->{lk}"] = {
                "pooled": pooled_auc(rows, sk, lk),
                "within": w["auc"], "within_n_rows": w["n_rows_both_classes"],
                "n_valid": w["n_valid"],
                "within_ranknorm_mw": w["auc_ranknorm_pooled"],
                "within_donor": within_auc(rows, f"donor_{sk}", lk)["auc"],
            }
            a = auc_tab[f"{sk}->{lk}"]
            a["within_oriented"] = _orient(a["within"], orient)
            a["within_donor_oriented"] = _orient(a["within_donor"], orient)
            a["own_minus_donor"] = (a["within_oriented"] - a["within_donor_oriented"]
                                    if math.isfinite(a["within_oriented"])
                                    and math.isfinite(a["within_donor_oriented"]) else _NAN)
    summ["auc"] = auc_tab

    # 국소화.
    summ["localization"] = {
        "frac_abs_mass_tail_or_box": bootstrap_ci(
            [x.get("frac_abs_mass_tail_or_box", _NAN) for r in rows for x in r["retries"]],
            seed=seed + 23, n_boot=n_boot),
        # ★관문 (3) 이 쓰는 자 — 절대 분수가 아니라 «몫 대비 배수»(1.0 = 균일 분포).
        "conc_ratio": bootstrap_ci([x.get("conc_ratio", _NAN) for r in rows
                                    for x in r["retries"]], seed=seed + 24, n_boot=n_boot),
        "expected_mass_share": _fmean([x.get("expected_mass_share", _NAN) for r in rows
                                       for x in r["retries"]]),
        "mean_delta_by_decile": [
            _fmean([x["decile_delta"][b] for r in rows for x in r["retries"]
                    if x.get("decile_delta")]) for b in range(10)],
        "frac_abs_mass_last10pct": bootstrap_ci(
            [x.get("frac_abs_mass_last10pct", _NAN) for r in rows for x in r["retries"]],
            seed=seed + 20, n_boot=n_boot),
        "frac_abs_mass_boxed": bootstrap_ci(
            [x.get("frac_abs_mass_boxed", _NAN) for r in rows for x in r["retries"]],
            seed=seed + 21, n_boot=n_boot),
        "mean_n_tok_box": _fmean([x.get("n_tok_box", _NAN) for r in rows for x in r["retries"]]),
        "mean_n_tok_oldans": _fmean([x.get("n_tok_oldans", _NAN) for r in rows for x in r["retries"]]),
        "oldans_delta": bootstrap_ci([x.get("oldans_delta", _NAN) for r in rows for x in r["retries"]],
                                     seed=seed + 22, n_boot=n_boot),
    }

    # 길이 통제.
    all_ret = [x for r in rows for x in r["retries"]]
    summ["length_control"] = {
        "spearman_sum_delta_vs_n_tok": G.spearman([x["sum_delta"] for x in all_ret],
                                                  [x["n_tok"] for x in all_ret]),
        "spearman_mean_delta_vs_n_tok": G.spearman([x["mean_delta"] for x in all_ret],
                                                   [x["n_tok"] for x in all_ret]),
    }
    for lk in LABELS:
        rr = [{"retries": [{"n_tok": x["n_tok"], lk: x[lk]} for x in r["retries"]]}
              for r in rows]
        # ★AUC(길이만) — 0.5 에서 먼 쪽이 «길이 자체가 이미 라벨을 안다»는 뜻이다(어느 방향이든).
        summ["length_control"][f"auc_len_only_pooled->{lk}"] = pooled_auc(rr, "n_tok", lk)
        summ["length_control"][f"auc_len_only_within->{lk}"] = within_auc(rr, "n_tok", lk)["auc"]

    # 재순위.
    picks = [rerank_row(r["retries"], r["gold"]) for r in rows]
    sel = {k: [p[k] for p in picks] for k in ("pick_sum", "pick_box", "maj8", "random", "oracle")}
    summ["rerank"] = {
        "acc": {k: _fmean(v) for k, v in sel.items()},
        "pick_sum_minus_maj8": paired_boot(sel["pick_sum"], sel["maj8"], seed=seed + 30,
                                           n_boot=n_boot),
        "pick_box_minus_maj8": paired_boot(sel["pick_box"], sel["maj8"], seed=seed + 31,
                                           n_boot=n_boot),
        "pick_sum_minus_random": paired_boot(sel["pick_sum"], sel["random"], seed=seed + 32,
                                             n_boot=n_boot),
        "pick_sum_minus_oracle": paired_boot(sel["pick_sum"], sel["oracle"], seed=seed + 33,
                                             n_boot=n_boot),
    }
    summ[spec["gate_key"]] = int(gate_pass(summ) if cond == COND_FACT
                                 else gate_pass_amnesic(summ))
    return summ


def _ci_excl_zero_pos(ci: Optional[dict]) -> bool:
    ci = ci or {}
    lo, hi, mean = ci.get("lo"), ci.get("hi"), ci.get("mean")
    if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in (lo, hi, mean)):
        return False
    return (lo > 0 or hi < 0) and mean > 0


def gate_pass(summ: dict) -> bool:
    """ANTI-TEACHER PASS ⇔ (ΣΔ 또는 boxΔ)→reemit 의 within AUC ≥ .75 **이면서 같은 자에서**
    own−donor ≥ .10 ∧ 그 자의 재순위가 maj@8 보다 높고 짝 CI 가 0 을 제외한다."""
    for sk, key in (("sum_delta", "pick_sum_minus_maj8"), ("box_delta", "pick_box_minus_maj8")):
        a = (summ.get("auc") or {}).get(f"{sk}->reemit") or {}
        w, d = a.get("within", _NAN), a.get("own_minus_donor", _NAN)
        if not (isinstance(w, (int, float)) and math.isfinite(w) and w >= PASS_WITHIN_AUC):
            continue
        if not (isinstance(d, (int, float)) and math.isfinite(d) and d >= PASS_OWN_MINUS_DONOR):
            continue
        if _ci_excl_zero_pos((summ.get("rerank") or {}).get(key)):
            return True
    return False


GATE_SCORES_AMNESIC = ("box_delta", "tail_delta")


def gate_pass_amnesic(summ: dict) -> bool:
    """AMNESIC-TEACHER PASS ⇔ **같은 자**(boxL 또는 tailL)에서
      (1) 문제 안 AUC(→reemit, 방향 고정) ≥ .75
      (2) own − donor ≥ .10        (donor = 남의 오답을 앞부분으로 둔 학생 문맥)
      (3) 집중 비율 conc_ratio = frac_abs_mass_tail_or_box / expected_share 의 부트스트랩
          평균 ≥ 2.0 ∧ CI 하한 > 1.0   (expected_share = 그 구간이 차지하는 **토큰 몫**)
    셋이 다 서야 B3 의 KL 이 **재발화 결정**에 걸린다고 말할 수 있다(문체가 아니라).
    ★절대 분수(≥ .50)를 쓰지 않는 이유: 보통 길이의 이어쓰기에서는 꼬리 10% + 박스가 토큰의
    ~12% 뿐이라, 결정 지점 신호가 아무리 강해도 절대 분수가 .50 에 닿지 않는다. 비율은 그
    길이 의존을 나눠 없앤다(1.0 = 균일, 2.0 = 그 구간에 두 배로 몰렸다)."""
    cc = (summ.get("localization") or {}).get("conc_ratio") or {}
    m, lo = cc.get("mean"), cc.get("lo")
    if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in (m, lo)):
        return False
    if not (float(m) >= PASS_CONC_RATIO and float(lo) > 1.0):
        return False
    for sk in GATE_SCORES_AMNESIC:
        a = (summ.get("auc") or {}).get(f"{sk}->reemit") or {}
        w, d = a.get("within_oriented", _NAN), a.get("own_minus_donor", _NAN)
        if not (isinstance(w, (int, float)) and math.isfinite(w) and w >= PASS_WITHIN_AUC):
            continue
        if not (isinstance(d, (int, float)) and math.isfinite(d) and d >= PASS_OWN_MINUS_DONOR):
            continue
        return True
    return False


def _f(v) -> str:
    if isinstance(v, dict):
        if "mean" in v and "lo" in v:
            return f"{_f(v.get('mean'))} [{_f(v.get('lo'))}, {_f(v.get('hi'))}] (n={v.get('n')})"
        return "; ".join(f"{k}={_f(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_f(x) for x in v) + "]"
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4f}"
    return str(v)


def to_markdown(summ: dict) -> str:
    cond = summ.get("cond", COND_FACT)
    amnesic = cond == COND_WAIT
    title = ("## math_anti_teacher_ruler — 기억상실 교사 자(B1, cond=wait, 채점 전용)"
             if amnesic else "## math_anti_teacher_ruler — 반면교사 자(S1, 채점 전용)")
    L = [title, "",
         ("L_t = log π(y_t | x, f) − log π(y_t | x, w, WAIT_CUE) — 음수 = 교사가 싫어한 토큰. "
          f"방향 고정 orient={summ.get('orient')}(관문은 −L 의 AUC)." if amnesic
          else "Δ_t = log π(y_t | x, w, f) − log π(y_t | x, f)."), "",
         f"n_rows={summ.get('n_rows')} · n_retries={summ.get('n_retries')} · "
         f"rate: {_f(summ.get('rate'))}", "",
         "### AUC (자 → 라벨)", "",
         "| 자 → 라벨 | pooled | within | within(방향) | donor(방향) | own−donor | within n | "
         "rank-norm MW |",
         "|---|---|---|---|---|---|---|---|"]
    for k, a in (summ.get("auc") or {}).items():
        L.append(f"| {k} | {_f(a['pooled'])} | {_f(a['within'])} | "
                 f"{_f(a.get('within_oriented'))} | {_f(a.get('within_donor_oriented'))} | "
                 f"{_f(a['own_minus_donor'])} | {a['within_n_rows']} | "
                 f"{_f(a['within_ranknorm_mw'])} |")
    L += ["", "### 국소화 · 길이 통제", "| metric | value |", "|---|---|"]
    for k, v in (summ.get("localization") or {}).items():
        L.append(f"| localization.{k} | {_f(v)} |")
    for k, v in (summ.get("length_control") or {}).items():
        L.append(f"| length.{k} | {_f(v)} |")
    rr = summ.get("rerank") or {}
    L += ["", "### 재순위(문제당 8개 중 하나 고르기)", "| selector | acc |", "|---|---|"]
    for k, v in (rr.get("acc") or {}).items():
        L.append(f"| {k} | {_f(v)} |")
    L += ["", "| 짝 차이 | mean [CI] |", "|---|---|"]
    for k in ("pick_sum_minus_maj8", "pick_box_minus_maj8", "pick_sum_minus_random",
              "pick_sum_minus_oracle"):
        L.append(f"| {k} | {_f(rr.get(k))} |")
    if summ.get("top_negative_tokens"):
        L += ["", "### 가장 음수인 토큰 20개(평균 L 오름차순)", "| token | mean L | n |",
              "|---|---|---|"]
        for z in summ["top_negative_tokens"]:
            L.append(f"| `{(z.get('token') or z['token_id'])!r}` | {_f(z['mean_delta'])} | "
                     f"{z['n']} |")
    if amnesic:
        ok = bool(summ.get("pass_amnesic_teacher"))
        L += ["", f"**AMNESIC-TEACHER {'PASS' if ok else 'FAIL'}** — 같은 자(boxL 또는 tailL)에서 "
                  f"문제 안 AUC(→reemit, 방향 고정) ≥ {PASS_WITHIN_AUC} ∧ own−donor ≥ "
                  f"{PASS_OWN_MINUS_DONOR} ∧ 집중 비율(conc_ratio) 평균 ≥ {PASS_CONC_RATIO} ∧ "
                  "CI 하한 > 1.0"]
        bx = (summ.get("auc") or {}).get("box_delta->reemit") or {}
        nv, nr = bx.get("n_valid"), summ.get("n_retries") or 0
        if isinstance(nv, int) and nr and nv < 0.5 * nr:
            L.append(f"> ⚠ boxL 은 이어쓰기 {nr}개 중 **{nv}개**(={nv / nr:.1%})에서만 정의된다 "
                     "— 새 `\\boxed` 가 없는 이어쓰기(답을 상속한 재발화)에서는 NaN 이라 그 자의 "
                     "AUC 는 **재발화 행을 대부분 뺀 부분집합** 위의 수다. tailL 을 먼저 보라.")
        L.append("")
        return "\n".join(L)
    ok = bool(summ.get("pass_anti_teacher"))
    L += ["", f"**ANTI-TEACHER {'PASS' if ok else 'FAIL'}** — within AUC(ΣΔ 또는 boxΔ → reemit) "
              f"≥ {PASS_WITHIN_AUC} ∧ own−donor ≥ {PASS_OWN_MINUS_DONOR} ∧ 재순위 − maj@8 의 "
              "CI 가 0 제외·평균 > 0", ""]
    return "\n".join(L)


# ── main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True, help="math_activation_gate 산출물 gens.jsonl")
    ap.add_argument("--rollouts", required=True,
                    help="그 게이트의 입력 math_rollout texts.jsonl (w·problem·gold 의 출처)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--variant", default="math_opt")
    ap.add_argument("--cond", default=COND_FACT, choices=sorted(MODE_SPECS),
                    help=f"어느 조건의 생성물을 채점할지 — {COND_FACT}=S1 반면교사 자(기본), "
                         f"{COND_WAIT}=B1 기억상실 교사 자(KL(teacher‖student) 국소화)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max_len", type=int, default=20480)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="★기본 1 — anti/donor 문맥은 «문제 + 오답 전체 + 재시도 전체» 라 "
                         "7k 토큰을 넘고, [B,T,vocab] bf16 로짓이 배치당 ~2 GB 다. 프리필은 "
                         "1 개만으로도 카드를 채우므로 배치를 키워 얻는 것이 거의 없다.")
    ap.add_argument("--limit", type=int, default=0, help="행(문제) 수 상한(0=전부)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.Random(a.seed)               # 결정적 — donor 는 회전이라 난수를 쓰지 않는다

    gens = [json.loads(l) for l in open(a.gens)]
    rolls = [json.loads(l) for l in open(a.rollouts)]
    rows = build_rows(gens, rolls, cond=a.cond, limit=a.limit)
    print(f"[anti] {a.cond} 재시도가 붙은 행 {len(rows)}개 "
          f"(재시도 {sum(len(r['retry_texts']) for r in rows)}개)", flush=True)
    if not rows:
        raise SystemExit("[anti] 행이 없다 — gens 의 cond/roll_id 가 rollouts 와 맞는지 보라.")

    forward, tok, _ = P.hf_forward_factory(a.model_path, batch_size=a.batch_size,
                                           max_len=a.max_len)
    built = build_jobs(tok, rows, a.variant, max_len=a.max_len, cond=a.cond)
    print(f"[anti] forward {len(built['jobs'])}개 (행 {len(built['index'])}, "
          f"너무 긴 재시도 {built['n_dropped_long']}개 제외)", flush=True)
    res = forward(built["jobs"], [])
    recs = retry_records(res, built)

    summ = summarize(recs, seed=a.seed, n_boot=a.n_boot, cond=a.cond)
    summ["top_negative_tokens"] = top_negative_tokens(recs, tok)
    summ.update({"model_path": a.model_path, "variant": a.variant, "gens": a.gens,
                 "rollouts": a.rollouts, "cond": a.cond, "seed": a.seed,
                 "max_len": a.max_len, "n_dropped_long": built["n_dropped_long"],
                 "n_forwards": len(built["jobs"])})

    with (out / mode_spec(a.cond)["out_name"]).open("w") as fh:
        for r in recs:
            for x in r["retries"]:
                fh.write(json.dumps({"roll_id": r["roll_id"], "group_id": r["group_id"],
                                     "gold": r["gold"], "wrong_ans": r["wrong_ans"],
                                     "donor_roll_id": r["donor_roll_id"], **x},
                                    ensure_ascii=False) + "\n")
    (out / "ruler_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    md = to_markdown(summ)
    (out / "ruler_summary.md").write_text(md)
    print(md)
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
