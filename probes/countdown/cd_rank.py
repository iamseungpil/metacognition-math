"""Countdown 오라클 판 2단계 — ***PMI-shift 가 품질 사다리를 아는가.***

여덟 판이 전부 "모델이 만든 메타"만 넣었고, PMI 검정은 전부 (a) 잡음 섞인 결과를
과녁으로 삼거나 (b) 문제 간 비교였다. `cd_oracle.py` 가 그 둘을 동시에 없앤다:

    참 품질 순서가 **구성으로** 알려져 있다        O3 > O1 > R ≈ OF
    다섯 메타가 **같은 prefix** 위에 있다          문제 난이도가 상쇄된다
    n = 문제수 × 4                                 ★문제 안 비교

★그래서 이 파일이 PMI-shift 의 **정답지 채점**이다. 지금까지의 널(참경보 AUC 0.457,
난이도 0.530, 문제별 메타가치 rho −0.06)은 전부 "과녁이 흐렸다"는 변명이 가능했다.
여기서는 그 변명이 없다.

═══════════════════════════════════════════════════════════════════════════
검정 셋 — 사전등록
═══════════════════════════════════════════════════════════════════════════

T1  사다리 재현 (co-primary)
    문제마다 네 메타(R/OF/O1/O3)의 칸 값을 참 순위와 맞춘다.
      · rho̅  = 문제 안 Spearman(동점보정)의 평균 · 치환 귀무분포 대비
      · C    = **참 순위가 정한 다섯 쌍**의 문제단위 승률 평균
               (O3>O1, O3>R, O3>OF, O1>R, O1>OF).  R–OF 는 참 동점이라 제외하고
               따로 찍는다(0.5 근처면 정상, 아니면 그 사실을 보고한다).
    ⚠rho̅ 는 참 순위에 동점(R≈OF)이 있어 **상한이 1.0 이 아니다** — 완벽예측도
      0.9487 이다. 코드가 직접 계산해서 찍고, rho̅/상한 도 같이 찍는다.
      C 는 상한이 정확히 1.0 이라 co-primary 로 둔다.

T2  ★핵심 대조 O1 vs OF (primary — 이 프로젝트에서 가장 깨끗한 PMI 검정)
    같은 문장 틀 · 같은 confidence(0.9) · 같은 자릿수 구조 · **틀린 수**.
    ⇒ 여기서 갈리면 PMI 는 **내용의 정확성**을 본 것이다. 못 가르면 **구체성**만 본 것이다.
    T1 의 사다리는 O3 가 "가장 길고 confidence 가 가장 높은" 팔이라 길이·확신도와
    **구성상 얽혀 있다**. T2 만 그 얽힘에서 자유롭다. 그래서 T2 가 primary 다.
    ★짝맞춤은 가정하지 않고 **실측해서 찍는다**(길이차·숫자수차·conf차 중앙값).

T3  음성 대조군 게이트 — meta_len(토큰수) · n_digits(숫자 문자수) · conf(선언 confidence)
    ★T1 에서는 이 셋이 **구성상 사다리를 맞힌다**(O3 가 가장 길고 0.95 다). 그러니
      T1 에서 음성 대조군을 이기는 것은 어렵고, 그것이 정직한 문턱이다.
    ★T2 에서는 이 셋이 0.5 근처여야 한다. 벗어나면 O1/OF 통제가 깨진 것이고,
      그 폭을 판정문에 같이 싣는다.
    (직전 판: meta_len 0.598 > 현행 0.457 로 **길이가 이겼다**. 그 사건의 재발 방지 장치다.)

═══════════════════════════════════════════════════════════════════════════
격자 — 축1 × 축2 × 축3
═══════════════════════════════════════════════════════════════════════════
  축1 채점대상  w  증인 단독            logp(\\boxed{witness} | ctx)
                wd 증인 − 연산자교체    ★같은 수·같은 구조·연산자 하나만 다름
                sf 자기후보             모델 자신이 낸 **맞는 식** vs 자신이 낸 **틀린 식**
  축2 대조      K1 위치   meta ctx − no-meta ctx           ← 라이브 정의(close − open)
                K2 문맥   meta ctx − **같은 팔의 다른 문제** 메타 ctx
  축3 위치      P1 메타 경계
                P2 답 직전 — **중립 post**(N 팔 sample0, 메타 없이 생성된 본문)

★축2 에 대한 경고 — 이 설계에서 직접 확인되는 성질:
  문제 안 랭킹은 **문제 안에서 상수인 항을 빼도 순위가 안 바뀐다.** open(=no-meta ctx)
  은 네 팔이 공유하므로 **K1 의 순위 = close 원값의 순위**다. 코드가 이것을 배선검사로
  대조한다(`[검사]` 칸). 즉 이 설계는 "shift 의 뺄셈"에 점수를 줄 수 없다.
  뺄셈이 순위에 영향을 주는 길은 셋뿐이다:
      ① K2 처럼 **팔마다 다른** 대조를 쓸 때  (그래서 K2 donor 는 팔-매칭이다)
      ② 라이브식처럼 open 에 **비선형**으로 의존할 때 (그래서 pos_full 칸을 따로 둔다)
      ③ 길이잔차처럼 **팔마다 다른** 공변량을 뺄 때
  이걸 안 적으면 K1 과 close 를 두 개의 독립 증거로 착각한다. 하나다.

═══════════════════════════════════════════════════════════════════════════
직전 판에서 실제로 터진 버그 셋 — 여기서 어떻게 막는가
═══════════════════════════════════════════════════════════════════════════
 (a) divergent_positions 가 부분열 쌍('2' vs '12')에서 빈 슬라이스를 낸다
     → `cd_grid.pair` 를 **그대로 재사용**한다(전체문자열 폴백 내장). 경로 카운터
       {div, full} 를 콘솔과 JSON 에 남긴다. full 이 늘면 그 칸의 해석이 달라진다.
 (b) 답 직전 측정은 본문이 이미 답을 말해 74% 가 20 nats 넘게 포화됐다
     → 칸마다 **포화율**(|raw| > SAT_NATS)을 찍고, **비포화 부분집합에서 C 를 다시** 낸다.
 (c) drive 는 post 가 처치 조건에서 생성돼 구성 편향(logp(post|R) > logp(post|무메타) 98.7%)
     → P2 의 post 는 **N 팔(메타 없음) sample0** 이라 네 팔에 바이트 동일하다.
       팔 자신의 post 를 쓰는 칸은 `⛔…자기post` 로 낙인찍어 따로 둔다(--own_post).

감쇠 보정: **하지 않는다.** 참 순위가 구성으로 알려져 기준 신뢰도 = 1.0 이다
(직전 판의 0.309 와 다르다 — 그때는 상한이 0.556 이었다). 유일한 천장은 위의 동점
구조이며 코드가 `rho_ceiling()` 으로 직접 계산한다.

⚠오라클은 진단이지 학습 방법이 아니다. 여기서 나오는 어떤 수도 성능 주장이 아니다.

사용:
    python cd_rank.py --selftest              # GPU 없이 통계 배선만 검사
    python cd_rank.py --json cd_oracle.json   # 본 채점 (GPU 1장)
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import statistics as st
from collections import Counter

# ────────────────────────────────────────────────────────────── 사전등록 상수

METAS = ["R", "OF", "O1", "O3"]
TRUTH = {"R": 1.5, "OF": 1.5, "O1": 3.0, "O3": 4.0}       # 구성으로 알려진 참 품질
ORD_PAIRS = [("O3", "O1"), ("O3", "R"), ("O3", "OF"), ("O1", "R"), ("O1", "OF")]
TIE_PAIR = ("R", "OF")                                     # 참 동점 — 0.5 근처가 정상
KEY_PAIR = ("O1", "OF")                                    # ★T2 primary
PRIMARY_CELL = "증인−교체·위치·경계"                        # ★사전등록 주 칸 = 라이브 정의

NEG_TAG = "★음성 "     # 음성 대조군 칸 접두사 (게이트 판별에 쓰이므로 한 곳에서만 정의)
SAT_NATS = 20.0        # 포화 기준 — 직전 판 답직전에서 74% 가 이 선을 넘었다
POST_TOK = 600         # post 토큰 상한 (cd_grid 규약 유지)

PREREG = {
    "C_strong": 0.60, "C_lo_strong": 0.55, "C_weak": 0.55,      # T1
    "rho_strong": 0.20, "rho_lo_strong": 0.10,                  # T1
    "o1of_strong": 0.60, "o1of_lo_strong": 0.55, "o1of_weak": 0.55,   # T2
    "neg_margin": 0.02,        # T3 음성 대조군을 이 폭 이상 이겨야 생존
    "pairing_tol": 0.05,       # O1/OF 통제가 성립하는지 (음성 대조군이 0.5±tol)
    "n_min": 100, "sat_warn": 0.50,
}


# ────────────────────────────────────────────────────────── 순수 통계 (GPU 무관)

def ranks(v):
    """동점 평균 순위."""
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    sa = math.sqrt(sum(x * x for x in da))
    sb = math.sqrt(sum(x * x for x in db))
    if sa < 1e-12 or sb < 1e-12:
        return None          # 무변동 — 0 으로 세지 않는다(널과 따로 센다)
    return sum(x * y for x, y in zip(da, db)) / (sa * sb)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def rho_ceiling(truth_vec):
    """**동점 없는** 예측이 낼 수 있는 Spearman 의 최대값 — 직접 계산한다.

    참 순위에 동점(R≈OF)이 있으므로, 예측이 그 둘을 어느 쪽으로든 가르면 벌점이 붙는다.
    PMI 는 연속값이라 정확한 동점이 사실상 안 나온다 ⇒ **이 값이 실질 천장**이다.
    (예측이 R/OF 를 정확히 동점 처리하면 1.0 도 가능하지만, 측도 0 의 사건이다.)
    """
    best = -2.0
    for perm in itertools.permutations(range(len(truth_vec))):
        r = spearman([float(p) for p in perm], truth_vec)
        if r is not None:
            best = max(best, r)
    return best


def perm_null_rho(score_vec, truth_vec):
    """한 문제의 치환 귀무 = 점수벡터를 팔에 붙이는 모든 배치(4! = 24). 근사 없음."""
    out = []
    for perm in itertools.permutations(score_vec):
        r = spearman(list(perm), truth_vec)
        if r is not None:
            out.append(r)
    return out


def winrate1(a, b):
    """한 문제의 쌍대 승. 귀무 0.5."""
    return 1.0 if a > b else 0.5 if a == b else 0.0


def boot_cols(cols, B=2000, seed=0, idx=None):
    """★문제 단위 부트스트랩 — 여러 통계가 **같은 재표집**을 공유해야 짝이 유지된다.

    cols: {key: [문제별 값 or None]} (모두 같은 길이·같은 문제 순서)
    idx : 부분집합(비포화 등). None 이면 전체.
    반환: {key: (obs, lo, hi, n)}
    """
    keys = list(cols)
    if not keys:
        return {}
    order = list(range(len(cols[keys[0]]))) if idx is None else list(idx)
    n = len(order)
    if n == 0:
        return {k: (None, None, None, 0) for k in keys}
    rng = random.Random(seed)
    sub = {k: [cols[k][i] for i in order] for k in keys}
    out = {}
    obs = {}
    for k in keys:
        v = [x for x in sub[k] if x is not None]
        obs[k] = (sum(v) / len(v) if v else None, len(v))
    draws = {k: [] for k in keys}
    for _ in range(B):
        sel = [rng.randrange(n) for _ in range(n)]
        for k in keys:
            col = sub[k]
            v = [col[i] for i in sel if col[i] is not None]
            draws[k].append(sum(v) / len(v) if v else None)
    for k in keys:
        d = sorted(x for x in draws[k] if x is not None)
        o, cnt = obs[k]
        out[k] = ((o, d[int(.025 * len(d))], d[int(.975 * len(d))], cnt)
                  if o is not None and d else (o, None, None, cnt))
    return out


def residualize(vals, cov):
    """문제 안에서 공변량(길이)에 회귀하고 잔차를 돌려준다.

    ⚠4점 2모수 — 자유도 2. 잡음에 약하니 보조 칸으로만 읽는다.
    """
    n = len(vals)
    mv, mc = sum(vals) / n, sum(cov) / n
    sxx = sum((c - mc) ** 2 for c in cov)
    if sxx < 1e-12:
        return list(vals)
    b = sum((c - mc) * (v - mv) for c, v in zip(cov, vals)) / sxx
    return [v - b * (c - mc) for v, c in zip(vals, cov)]


# ────────────────────────────────────────────────────── 자기검사 (GPU 없이 돈다)

def selftest() -> None:
    tv = [TRUTH[m] for m in METAS]
    ceil4 = rho_ceiling(tv)
    print(f"[selftest] 참 순위 {dict(zip(METAS, tv))}")
    print(f"[selftest] ★rho 실질 상한(동점 없는 예측) = {ceil4:.4f}  ≠ 1.0 — 참 순위의 "
          f"동점(R≈OF) 때문이다. 기준 신뢰도는 1.0 이므로 **감쇠보정은 하지 않는다**")
    print(f"[selftest] O3 제외 3팔(O1>R≈OF) rho 상한  = "
          f"{rho_ceiling([TRUTH[m] for m in ('R', 'OF', 'O1')]):.4f}")
    nl = perm_null_rho([0.1, 0.2, 0.3, 0.4], tv)
    m = sum(nl) / len(nl)
    sd = math.sqrt(st.pvariance(nl))
    print(f"[selftest] 치환 귀무 rho: 평균 {m:+.12f} (0 이어야) · 표준편차 {sd:.4f} · {len(nl)}배치")
    assert abs(m) < 1e-9, "치환 귀무 평균이 0 이 아니다 — 대칭성이 깨졌다"
    for n in (150, 200, 250, 300):
        print(f"[selftest] n={n:3d} → rho̅ 95% 귀무밴드 ±{1.96*sd/math.sqrt(n):.4f} · "
              f"쌍대승률 95% 귀무밴드 ±{1.96*0.5/math.sqrt(n):.4f} · "
              f"Bonferroni(15칸) ±{2.93*0.5/math.sqrt(n):.4f}")
    perfect = {"R": 0.0, "OF": 0.0, "O1": 1.0, "O3": 2.0}
    c = st.mean(winrate1(perfect[a], perfect[b]) for a, b in ORD_PAIRS)
    assert abs(c - 1.0) < 1e-12
    assert abs(winrate1(perfect["R"], perfect["OF"]) - 0.5) < 1e-12
    rev = {k: -v for k, v in perfect.items()}
    assert abs(st.mean(winrate1(rev[a], rev[b]) for a, b in ORD_PAIRS)) < 1e-12
    print(f"[selftest] 완벽예측 C = 1.0 · 역예측 C = 0.0 · 동점쌍 R–OF = 0.5  ✅통계 배선 통과")


# ────────────────────────────────────────────────────────────────────── 본체

_BX = re.compile(r"\\boxed\{([^{}]*)\}")
_DIG = re.compile(r"\d")


def _nrm(s):
    return (s or "").replace(" ", "").strip()


def build_cells():
    """칸 정의. 각 칸 = (row → {팔: 값 or None}, 포화기준키, 편향낙인).

    raw 키 규칙:  f"{tgt}_{ctx}"  또는  f"{tgt}_{ctx}:{팔}"    tgt ∈ {w, wd, sf}
      O   메타 없음(경계)              OP  메타 없음 + 중립 post
      M   메타 m(경계)                 MP  메타 m + 중립 post
      D   같은 팔의 **다른 문제** 메타   DP  그것 + 중립 post
      MPo 메타 m + **자기 post**       ⛔구성 편향 — 낙인용
    """
    TN = {"w": "증인단독", "wd": "증인−교체", "sf": "자기후보"}

    def delta(tgt, close, open_, open_per_arm):
        def f(r):
            out = {}
            for m in METAS:
                a = r["raw"].get(f"{tgt}_{close}:{m}")
                b = r["raw"].get(f"{tgt}_{open_}:{m}" if open_per_arm
                                 else f"{tgt}_{open_}")
                out[m] = None if (a is None or b is None) else a - b
            return out
        return f

    C = {}
    for tgt in ("w", "wd", "sf"):
        C[f"{TN[tgt]}·위치·경계"] = (delta(tgt, "M", "O", False), f"{tgt}_M", False)
        C[f"{TN[tgt]}·문맥·경계"] = (delta(tgt, "M", "D", True), f"{tgt}_M", False)
    # 답 직전은 증인−교체 축에서만 (cd_grid 규약 유지 — 비용/다중검정 억제)
    C["증인−교체·위치·답직전"] = (delta("wd", "MP", "OP", False), "wd_MP", False)
    C["증인−교체·문맥·답직전"] = (delta("wd", "MP", "DP", True), "wd_MP", False)
    C["⛔증인−교체·위치·답직전·자기post"] = (delta("wd", "MPo", "OP", False), "wd_MPo", True)

    # 배선검사 — K1 과 순위가 **같아야** 한다(open 이 문제 안 상수라서)
    C["[검사]증인−교체·close원값"] = (
        lambda r: {m: r["raw"].get(f"wd_M:{m}") for m in METAS}, "wd_M", False)

    # ★라이브식 pos_full — open 에 **비선형**으로 의존하므로 K1 과 순위가 다를 수 있다
    def live(r):
        out, o = {}, r["raw"].get("wd_O")
        for m in METAS:
            c = r["raw"].get(f"wd_M:{m}")
            if o is None or c is None:
                out[m] = None
                continue
            v = max(-2.0, min(2.0, c - o))
            if o < 0 < c:
                v += 1.0            # save
            elif o > 0 > c:
                v -= 2.0            # derail
            out[m] = v
        return out
    C["라이브 pos_full(clip±2+역전)"] = (live, "wd_M", False)

    base = C["증인−교체·위치·경계"][0]

    def resid(r):
        v = base(r)
        ks = [m for m in METAS if v[m] is not None]
        if len(ks) < 3:
            return {m: None for m in METAS}
        rv = residualize([v[m] for m in ks], [float(r["meta_len"][m]) for m in ks])
        out = {m: None for m in METAS}
        for m, x in zip(ks, rv):
            out[m] = x
        return out
    C["증인−교체·위치·경계(길이잔차)"] = (resid, "wd_M", False)

    # ★음성 대조군 — 모델을 전혀 안 쓴다
    C[NEG_TAG + "meta_len"] = (lambda r: {m: float(r["meta_len"][m]) for m in METAS}, None, False)
    C[NEG_TAG + "n_digits"] = (lambda r: {m: float(r["n_digits"][m]) for m in METAS}, None, False)
    C[NEG_TAG + "conf"] = (lambda r: {m: float(r["conf"][m]) for m in METAS}, None, False)
    return C


def rank_mismatch(f1, f2, rows):
    """두 칸의 **문제 안 순위**가 다른 행 수. K1 vs close 배선검사용.

    수학적으로 둘은 같아야 한다(open 이 문제 안 상수라 뺄셈이 순서를 보존한다). 유일한
    예외는 **부동소수 정밀도**다: close 값들의 차이가 open 의 크기에 대한 ULP 보다 작으면
    뺄셈이 그 차이를 없애 동점을 만든다. 합성검사(0818)에서 실제로 4/60 행이 이 경로로
    갈렸다 — close 가 6.6899999999999995 vs 6.689999999999998 (차이 1.8e-15) 였고
    open 이 −13.31 이라 둘 다 정확히 20.0 이 됐다.
    실측 PMI 는 팔 간 차이가 O(0.01~1) nats 이라 이 경로는 거의 안 나온다. 그래서
    **0 이 아니면 그 자체가 보고 대상**이다(신호가 ULP 크기라는 뜻이니까).
    """
    bad = tie1 = tie2 = 0
    for r in rows:
        a = [f1(r)[m] for m in METAS]
        b = [f2(r)[m] for m in METAS]
        if any(x is None for x in a) or any(x is None for x in b):
            continue
        ra, rb = ranks(a), ranks(b)
        bad += 1 if ra != rb else 0
        tie1 += 1 if len(set(a)) < len(a) else 0
        tie2 += 1 if len(set(b)) < len(b) else 0
    return bad, tie1, tie2


def score_cell(cellf, rows, satkey, B, seed, strict_idx=None):
    """한 칸에 T1(rho̅ · C) + T2(O1 vs OF) + 동점쌍 + 포화율 + 비포화 재판정을 전부 건다."""
    tv = [TRUTH[m] for m in METAS]
    pair_keys = [f"{a}>{b}" for a, b in ORD_PAIRS] + [f"{TIE_PAIR[0]}>{TIE_PAIR[1]}"]
    cols = {"rho": [], **{k: [] for k in pair_keys}, "C": []}
    nulls_v, n_flat = [], 0
    sat_hit = sat_tot = 0
    unsat = []
    for pos, r in enumerate(rows):
        v = cellf(r)
        vals = [v[m] for m in METAS]
        rr = None
        if all(x is not None for x in vals):
            rr = spearman(vals, tv)
            if rr is None:
                n_flat += 1
            else:
                nulls_v.append(st.pvariance(perm_null_rho(vals, tv)))
        cols["rho"].append(rr)
        for a, b in ORD_PAIRS + [TIE_PAIR]:
            cols[f"{a}>{b}"].append(None if (v[a] is None or v[b] is None)
                                    else winrate1(v[a], v[b]))
        cols["C"].append(None if any(cols[f"{a}>{b}"][-1] is None for a, b in ORD_PAIRS)
                         else st.mean(cols[f"{a}>{b}"][-1] for a, b in ORD_PAIRS))
        if satkey:
            hit = False
            for m in METAS:
                x = r["raw"].get(f"{satkey}:{m}")
                if x is not None:
                    sat_tot += 1
                    if abs(x) > SAT_NATS:
                        sat_hit, hit = sat_hit + 1, True
            if not hit:
                unsat.append(pos)
        else:
            unsat.append(pos)

    kk = f"{KEY_PAIR[0]}>{KEY_PAIR[1]}"
    bs = boot_cols(cols, B=B, seed=seed)
    bs_u = boot_cols({"C": cols["C"], kk: cols[kk]},
                     B=max(500, B // 4), seed=seed + 1, idx=unsat)
    bs_s = boot_cols({kk: cols[kk]}, B=max(500, B // 4), seed=seed + 2,
                     idx=strict_idx) if strict_idx is not None else {kk: (None,) * 4}
    band = (1.96 * math.sqrt(sum(nulls_v)) / len(nulls_v)) if nulls_v else None
    return {"rho": bs["rho"], "C": bs["C"],
            "pairs": {k: bs[k] for k in pair_keys},
            "rho_null_band": band, "n_flat": n_flat,
            "sat": (sat_hit / sat_tot) if sat_tot else None,
            "unsat": {"C": bs_u["C"], "o1of": bs_u[kk], "n": len(unsat)},
            "strict": bs_s[kk],          # ★엄격 짝맞춤 부분집합에서의 O1 vs OF
            "cols": cols}


# ───────────────────────────────────────────────────────────────── 판정 문구

def verdict_pair(v, lo, hi):
    """T2 (O1 vs OF) 판정 밴드."""
    if v is None or lo is None:
        return "표본부족"
    if v >= PREREG["o1of_strong"] and lo > PREREG["o1of_lo_strong"]:
        return "★가른다"
    if v >= PREREG["o1of_weak"] and lo > 0.5:
        return "약함(방향만)"
    if hi is not None and hi < 0.5:
        return "⛔역방향 유의"
    return "널"


def verdict_ladder(C, Clo, Chi, rho, band):
    """T1 (사다리) 판정 밴드."""
    if C is None or Clo is None:
        return "표본부족"
    if C >= PREREG["C_strong"] and Clo > PREREG["C_lo_strong"]:
        return "★재현"
    if C >= PREREG["C_weak"] and Clo > 0.5:
        return "약함"
    if Chi is not None and Chi < 0.5:
        return "⛔역순 유의"
    if rho is not None and band is not None and abs(rho) <= band:
        return "확정널(치환밴드 안)"
    return "널"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="cd_oracle.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="cd_rank.json")
    ap.add_argument("--limit", type=int, default=0, help="0=전부 (디버그용 부분집합)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--own_post", action="store_true",
                    help="⛔구성 편향 칸(자기 post)도 계산한다 — 비용 +25%, 낙인 표시됨")
    ap.add_argument("--selftest", action="store_true", help="GPU 없이 통계 배선만 검사")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return

    b = json.load(open(args.json))
    items, ACC, PS, ARMS = b["items"], b["per_problem"], b["per_sample"], b["arms"]
    POOL = b["pool"][:args.limit] if args.limit else b["pool"]

    print("\n[0] 1단계(cd_oracle) 판정 A 이어받기")
    o3w = b.get("o3_wiring")
    print(f"    형식 {b.get('fmt', float('nan')):.1%} · 모집단 acc(N)=0 : "
          f"{len(b['pool'])} (사용 {len(POOL)})")
    print("    SAVE율  " + " · ".join(f"{c} {b['save_rate'][c]*100:.1f}%" for c in ARMS))
    wiring_ok = o3w is not None and o3w >= 0.80
    if wiring_ok:
        print(f"    ✅배선 P(SAVE|O3) = {o3w*100:.1f}%")
    else:
        print(f"    ⛔배선 P(SAVE|O3) = {o3w} < 0.80 — **생성쪽** 채널이 고장이다.")
        print(f"      ★이 파일의 참 순위는 구성이지 실측이 아니므로 검정 자체는 유효하다.")
        print(f"      다만 '메타가 읽히기는 하나'가 미확인이므로 모든 판정은 **조건부**다.")

    # ── 연산자교체 오답. cd_oracle.py 는 이것을 저장하지 않는다(cd_main.py 만 저장했다).
    #    ⇒ 여기서 **결정론적으로** 다시 만든다. 시드를 문제 인덱스에 고정해 재현 가능하게.
    from countdown import swap_op_decoy, grade
    n_dec = 0
    for i, it in enumerate(items):
        if not it.get("decoy"):
            it["decoy"] = swap_op_decoy(it["witness"], it["nums"], it["target"],
                                        random.Random(10_000 + i))
        n_dec += 1 if it.get("decoy") else 0
    print(f"[1] 연산자교체 오답 {n_dec}/{len(items)}"
          + ("" if n_dec == len(items) else "  ⚠없는 문제는 wd 칸에서 제외된다"))

    # ── 자기후보: 모델 자신이 낸 **맞는 식** vs 자신이 낸 **틀린 식**
    #   ⚠per_sample 은 sample0 의 text 만 저장한다 → 팔당 1개뿐이라 n 이 작다(찍는다).
    #   ⚠오라클 팔(O1/O3)의 text 는 증인을 그대로 베낀다 → **비오라클 팔을 먼저** 쓴다.
    NON_OR = [a for a in ("N", "Np", "R", "OF") if a in ARMS]
    OR_ARM = [a for a in ("O1", "O3") if a in ARMS]
    selfc = {}
    for i in POOL:
        it = items[i]
        good, gsrc, bad = None, None, Counter()
        for group in (NON_OR, OR_ARM):
            for a in group:
                t = (PS[a][i][0] or {}).get("text")
                if not t:
                    continue
                for e in (_nrm(x) for x in _BX.findall(t)):
                    if not e:
                        continue
                    if grade(f"\\boxed{{{e}}}", it["nums"], it["target"]):
                        if good is None:
                            good, gsrc = e, a
                    elif a in NON_OR:
                        bad[e] += 1
            if good is not None:
                break
        d = bad.most_common(1)[0][0] if bad else None
        if good is not None and d is not None and good == d:
            good = d = None
        selfc[i] = (good, gsrc, d)
    n_sf = sum(1 for i in POOL if selfc[i][0] and selfc[i][2])
    print(f"[2] 자기후보 가능 {n_sf}/{len(POOL)} — 비오라클 출처 "
          f"{sum(1 for i in POOL if selfc[i][1] in NON_OR)} · 오라클 출처 "
          f"{sum(1 for i in POOL if selfc[i][1] in OR_ARM)} (후자는 증인 베끼기일 수 있다)")

    # ── K2 donor: **같은 팔·다른 문제**. 문제 안에서 팔마다 다르므로 순위를 실제로 바꾼다.
    #    (문제 안 상수인 대조는 순위를 못 바꾼다 — 파일 상단 '축2 경고' 참조)
    donor = {i: {m: items[POOL[(k + 1) % len(POOL)]][m] for m in METAS}
             for k, i in enumerate(POOL)}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cd_grid import pair, solo          # ★(a) 폴백 내장된 검증된 경로를 재사용

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    PATH, rows = {}, []
    for k, i in enumerate(POOL):
        it = items[i]
        w, dec = it["witness"], it.get("decoy")
        g_self, _, d_self = selfc[i]

        # ★cd_oracle.py 가 실제 생성에 쓴 문자열과 **바이트 동일**하게 만든다:
        #     req = prompt + prefix + ("" | f"\n<meta>{m}</meta>") + "\n"
        base = it["prompt"] + it["prefix"]
        ctxO = base + "\n"
        ctxM = {m: base + f"\n<meta>{it[m]}</meta>" + "\n" for m in METAS}
        ctxD = {m: base + f"\n<meta>{donor[i][m]}</meta>" + "\n" for m in METAS}

        # 중립 post = **N 팔(메타 없음)** sample0 본문의 boxed 앞까지 → 네 팔에 동일
        t0 = (PS["N"][i][0] or {}).get("text") or ""
        j0 = t0.find("\\boxed{")
        pid = tok(t0[:j0] if j0 > 0 else "", add_special_tokens=False)["input_ids"][:POST_TOK]
        post = tok.decode(pid) if len(pid) >= 5 else ""

        raw = {}

        def fill(key, ctx):
            raw[f"w_{key}"] = solo(model, tok, ctx, w)
            raw[f"wd_{key}"] = pair(model, tok, ctx, w, dec, stats=PATH)
            raw[f"sf_{key}"] = pair(model, tok, ctx, g_self, d_self, stats=PATH)

        def fill_wd(key, ctx):
            raw[f"wd_{key}"] = pair(model, tok, ctx, w, dec, stats=PATH)

        fill("O", ctxO)
        for m in METAS:
            fill(f"M:{m}", ctxM[m])
            fill(f"D:{m}", ctxD[m])
        if post:
            fill_wd("OP", ctxO + post)
            for m in METAS:
                fill_wd(f"MP:{m}", ctxM[m] + post)
                fill_wd(f"DP:{m}", ctxD[m] + post)
                if args.own_post:
                    to = (PS[m][i][0] or {}).get("text") or ""
                    jo = to.find("\\boxed{")
                    ip = tok(to[:jo] if jo > 0 else "",
                             add_special_tokens=False)["input_ids"][:POST_TOK]
                    if len(ip) >= 5:
                        fill_wd(f"MPo:{m}", ctxM[m] + tok.decode(ip))

        rows.append({
            "i": i, "has_post": bool(post),
            "acc": {m: ACC[m][i] for m in METAS},
            "meta_len": {m: len(tok(it[m], add_special_tokens=False)["input_ids"])
                         for m in METAS},
            "n_digits": {m: len(_DIG.findall(it[m])) for m in METAS},
            "conf": {m: (lambda g: float(g.group(1)) if g else 0.5)(
                re.search(r"confidence:\s*([0-9.]+)", it[m])) for m in METAS},
            "raw": raw,
        })
        if (k + 1) % 25 == 0:
            print(f"    …{k+1}/{len(POOL)}  PMI 슬라이스 경로 {PATH}")
    print(f"[3] 원재료 완료 {len(rows)} · post 있는 문제 {sum(r['has_post'] for r in rows)} · "
          f"PMI 발산슬라이스 경로 {PATH}")
    print(f"    (full = 전체문자열 폴백 = 버그(a) 카운터. 이 비율이 크면 그 칸은 "
          f"'발산토큰 대비'가 아니라 '문자열 길이 대비'다 — 해석을 바꿔라)")

    # ── ★T2 짝맞춤 실측 (가정하지 않는다)
    dl = [r["meta_len"]["O1"] - r["meta_len"]["OF"] for r in rows]
    dd = [r["n_digits"]["O1"] - r["n_digits"]["OF"] for r in rows]
    dc = [r["conf"]["O1"] - r["conf"]["OF"] for r in rows]
    strict = [p for p, r in enumerate(rows)
              if abs(r["meta_len"]["O1"] - r["meta_len"]["OF"]) <= 1
              and r["n_digits"]["O1"] == r["n_digits"]["OF"]]
    print(f"[4] ★O1/OF 짝맞춤 실측  길이차 중앙값 {st.median(dl):+.1f}토큰"
          f"(평균 {st.mean(dl):+.2f}) · 숫자수차 중앙값 {st.median(dd):+.1f} · "
          f"conf차 평균 {st.mean(dc):+.3f}")
    print(f"    ★엄격 짝맞춤 부분집합(길이차≤1 토큰 **그리고** 숫자수 동일) "
          f"{len(strict)}/{len(rows)} — T2 를 여기서 다시 낸다.")
    print(f"    ⚠OF 의 거짓 결합은 연산자가 무작위라 값의 자릿수가 O1 과 어긋날 수 있다"
          f"(예: 곱셈이면 자릿수가 는다). 짝맞춤은 **가정하지 않고 위 수로 판정한다**.")

    CELLS = build_cells()
    res = {nm: score_cell(f, rows, sk, args.boot, args.seed, strict_idx=strict)
           for nm, (f, sk, _) in CELLS.items()}

    # ── 배선검사: K1(위치 뺄셈)과 close 원값의 **문제 안 순위**가 같아야 한다
    nbad, t1, t2 = rank_mismatch(CELLS[PRIMARY_CELL][0],
                                 CELLS["[검사]증인−교체·close원값"][0], rows)
    ok = nbad == 0
    print(f"\n[5] 배선검사  K1 과 close원값의 순위가 다른 행 {nbad}/{len(rows)} "
          f"(동점 행: K1 {t1} · close {t2})")
    if ok:
        print(f"    ✅동일 — open 은 문제 안 상수라 순위에 기여하지 않는다(설계상 예상).")
    else:
        print(f"    ⚠불일치 {nbad}행. 수학적으로 유일한 경로는 **부동소수 ULP 붕괴**다"
              f"(close 값 차이가 open 크기의 ULP 보다 작으면 뺄셈이 동점을 만든다).")
        print(f"      ⇒ 이 비율이 크면 팔 간 PMI 차이가 잡음 수준이라는 뜻이다. "
              f"그 자체를 판정문에 실어라. 그 외 원인이면 결과를 읽기 전에 조사하라.")
    print(f"    ⇒ 즉 이 판에서 'shift 의 뺄셈'은 K2·라이브식·길이잔차를 통해서만 검정된다. "
          f"K1 과 close 는 **독립 증거 둘이 아니라 하나**다.")

    ceil4 = rho_ceiling([TRUTH[m] for m in METAS])
    print(f"\n[6] ★격자 — 참 순위 O3>O1>R≈OF (구성으로 알려짐 · 기준 신뢰도 1.0 "
          f"⇒ **감쇠보정 없음**)")
    print(f"    rho 실질 상한 {ceil4:.4f}(참 순위의 동점 때문 · 연속값 예측은 여기가 천장) · "
          f"C 상한 1.000 · 귀무 rho̅ 0 · 귀무 승률 0.500")

    # ★라이브식 진단 — clip(±2) 이 사다리 꼭대기를 뭉개는가 (합성검사에서 실제로 관측됨:
    #   완벽신호에서도 C 가 1.000 → 0.900 으로 떨어졌다. O1 과 O3 가 둘 다 상한에 붙어서다.)
    nclip = ntot = 0
    for r in rows:
        o = r["raw"].get("wd_O")
        for m in METAS:
            c = r["raw"].get(f"wd_M:{m}")
            if o is not None and c is not None:
                ntot += 1
                nclip += 1 if abs(c - o) > 2.0 else 0
    lv = res["라이브 pos_full(clip±2+역전)"]
    tie_top = st.mean(1 if x == 0.5 else 0 for x in lv["cols"]["O3>O1"] if x is not None)
    print(f"    ★라이브식 진단  clip(±2) 에 걸린 (문제×팔) {nclip}/{ntot} = "
          f"{(nclip/ntot*100 if ntot else float('nan')):.1f}%  · "
          f"그 결과 O3 vs O1 이 동점이 된 비율 {tie_top*100:.1f}%")
    print(f"      ⇒ clip 이 높으면 **라이브 보상은 O1 과 O3 를 원리적으로 못 가른다** "
          f"(사다리 꼭대기가 뭉개진다). 그 경우 낮은 C 는 '신호 없음'이 아니라 '해상도 없음'이다.")
    hdr = (f"\n {'칸':32s} {'C[5쌍]':>22s} {'rho̅':>22s} {'rho/상한':>8s} "
           f"{'★O1>OF':>22s} {'R>OF':>6s} {'포화':>6s} {'n':>5s}")
    print(hdr)
    print("─" * 130)
    for nm in CELLS:
        R = res[nm]
        C, Clo, Chi, nC = R["C"]
        rh, rlo, rhi, _ = R["rho"]
        o1, o1lo, o1hi, _ = R["pairs"][f"{KEY_PAIR[0]}>{KEY_PAIR[1]}"]
        tie = R["pairs"][f"{TIE_PAIR[0]}>{TIE_PAIR[1]}"][0]
        if C is None or rh is None:
            print(f" {nm:32s} 표본부족 n={nC}")
            continue
        s_c = f"{C:.3f}[{Clo:.3f},{Chi:.3f}]"
        s_r = f"{rh:+.3f}[{rlo:+.3f},{rhi:+.3f}]"
        s_o = f"{o1:.3f}[{o1lo:.3f},{o1hi:.3f}]"
        s_t = "  n/a" if tie is None else f"{tie:.3f}"
        s_s = "  n/a" if R["sat"] is None else f"{R['sat']*100:5.1f}%"
        star = "★" if (Clo > 0.5 or Chi < 0.5) else " "
        print(f"{star}{nm:32s} {s_c:>22s} {s_r:>22s} {rh/ceil4:8.3f} "
              f"{s_o:>22s} {s_t:>6s} {s_s:>6s} {nC:5d}")

    # ── 판정
    print(f"\n[7] ★판정 (사전등록 밴드: T1 C≥{PREREG['C_strong']}&하한>{PREREG['C_lo_strong']} · "
          f"T2 ≥{PREREG['o1of_strong']}&하한>{PREREG['o1of_lo_strong']} · "
          f"음성대조군 +{PREREG['neg_margin']} 초과)")
    negs = [k for k in CELLS if k.startswith(NEG_TAG)]
    negC = {k: res[k]["C"][0] for k in negs}
    negK = {k: res[k]["pairs"][f"{KEY_PAIR[0]}>{KEY_PAIR[1]}"][0] for k in negs}
    short = lambda k: k[len(NEG_TAG):]
    print("  음성 C     : " + " · ".join(f"{short(k)} {v:.3f}" for k, v in negC.items()))
    print(f"    ★T1 게이트는 **meta_len 만** 쓴다(직전 판에서 실제로 우리를 이긴 그 대조군, "
          f"0.598 vs 0.457). n_digits·conf 는 참 순위와 **구성상 얽혀** 있다 — O3 가 "
          f"가장 길고 숫자가 많고 confidence 0.95 다. 그 둘은 게이트가 아니라 "
          f"'구성 얽힘의 크기'를 보여주는 참조선이다. 그래서 T2 가 primary 다.")
    print("  음성 O1>OF : " + " · ".join(f"{short(k)} {v:.3f}" for k, v in negK.items())
          + "   ← 0.5 근처여야 O1/OF 통제가 성립")
    broke = [short(k) for k, v in negK.items() if abs(v - .5) > PREREG["pairing_tol"]]
    if broke:
        print(f"  ⚠짝맞춤 경고 — {broke} 가 0.5±{PREREG['pairing_tol']} 밖이다. "
              f"O1/OF 대조가 구체성 통제를 잃었다. 그 폭을 판정문에 같이 실어라.")

    print(f"\n  {'칸':32s} {'T1 사다리':>18s} {'★T2 O1vsOF':>14s}  음성게이트 / 비고")
    surv = []
    for nm in CELLS:
        if nm.startswith(NEG_TAG) or nm.startswith("[검사]"):
            continue
        R = res[nm]
        C, Clo, Chi, nC = R["C"]
        o1, o1lo, o1hi, _ = R["pairs"][f"{KEY_PAIR[0]}>{KEY_PAIR[1]}"]
        v1 = verdict_ladder(C, Clo, Chi, R["rho"][0], R["rho_null_band"])
        v2 = verdict_pair(o1, o1lo, o1hi)
        g1 = (C is not None
              and abs(C - .5) > abs(negC[NEG_TAG + "meta_len"] - .5) + PREREG["neg_margin"])
        g2 = o1 is not None and all(abs(o1 - .5) > abs(v - .5) + PREREG["neg_margin"]
                                    for v in negK.values())
        so, solo_, sohi, son = R["strict"]
        note = f"  엄격짝맞춤 O1>OF={'n/a' if so is None else f'{so:.3f}'}(n={son})"
        if nm.startswith("⛔"):
            note += "  ⛔구성편향(자기post)"
        if nC < PREREG["n_min"]:
            note += f"  ⚠n={nC}<{PREREG['n_min']}"
        if R["sat"] is not None and R["sat"] > PREREG["sat_warn"]:
            u = R["unsat"]
            uc = "n/a" if u["C"][0] is None else f"{u['C'][0]:.3f}"
            uo = "n/a" if u["o1of"][0] is None else f"{u['o1of'][0]:.3f}"
            note += f"  ⚠포화{R['sat']*100:.0f}% → 비포화 C={uc} O1>OF={uo} (n={u['n']})"
        print(f"  {nm:32s} {v1:>18s} {v2:>14s}  T1{'✅' if g1 else '⛔'} "
              f"T2{'✅' if g2 else '⛔'}{note}")
        if g2 and v2.startswith("★") and not nm.startswith("⛔"):
            surv.append(nm)

    P = res[PRIMARY_CELL]
    o1, o1lo, o1hi, n1 = P["pairs"][f"{KEY_PAIR[0]}>{KEY_PAIR[1]}"]
    Cp, Cplo, Cphi, _ = P["C"]
    print(f"\n  ═══ ★사전등록 주 칸 = {PRIMARY_CELL} (라이브 정의) ═══")
    print(f"    ★T2 O1 vs OF = {o1:.4f} [{o1lo:.4f},{o1hi:.4f}]  n={n1}  → "
          f"{verdict_pair(o1, o1lo, o1hi)}")
    so, solo_, sohi, son = P["strict"]
    if so is not None:
        print(f"    ★T2(엄격 짝맞춤: 길이차≤1 & 숫자수 동일) = {so:.4f} "
              f"[{solo_:.4f},{sohi:.4f}]  n={son}  → {verdict_pair(so, solo_, sohi)}")
    else:
        print(f"    ★T2(엄격 짝맞춤) 표본부족 n={son}")
    print(f"      ⇒ 가르면 PMI 는 **내용의 정확성**을 본다. 못 가르면 **구체성**만 본다.")
    print(f"    T1 C = {Cp:.4f} [{Cplo:.4f},{Cphi:.4f}] → {verdict_ladder(Cp, Cplo, Cphi, P['rho'][0], P['rho_null_band'])}")
    print(f"    T1 rho̅ = {P['rho'][0]:+.4f} (치환 귀무밴드 ±{P['rho_null_band']:.4f} · "
          f"상한 {ceil4:.4f} · 상한대비 {P['rho'][0]/ceil4:+.3f})")
    p_sat = "n/a" if P["sat"] is None else f"{P['sat']*100:.1f}%"
    print(f"    포화율(|PMI|>{SAT_NATS:.0f} nats) = {p_sat}"
          f"   ← 직전 판 '답 직전'에서 74% 가 여기 걸렸다")
    print(f"\n  ⇒ 걸린 칸 {len(CELLS)}개(음성·검사 포함). 다중검정 문턱: 승률 |v−0.5| ≥ "
          f"{1.96*0.5/math.sqrt(max(1, n1)):.3f}(단일) / "
          f"{2.93*0.5/math.sqrt(max(1, n1)):.3f}(Bonferroni 15칸). "
          f"주 칸만 사전등록이고 나머지는 **탐색**이다.")
    print(f"  ⇒ 음성 대조군을 이기고 O1/OF 를 가른 칸: {surv if surv else '없음 — 격자 폐기'}")
    if not wiring_ok:
        print(f"  ⚠위 판정은 전부 **조건부** — 1단계 배선(P(SAVE|O3)≥0.80)이 미달이다.")

    json.dump({"model": args.model, "src": args.json, "n": len(rows),
               "wiring_ok": wiring_ok, "o3_wiring": o3w,
               "truth": TRUTH, "rho_ceiling": ceil4, "prereg": PREREG,
               "pmi_path": PATH, "primary_cell": PRIMARY_CELL,
               "k1_equals_close": ok, "survivors": surv,
               "o1of_pairing": {"len_med": st.median(dl), "dig_med": st.median(dd),
                                "conf_mean": st.mean(dc)},
               "cells": {nm: {k: v for k, v in res[nm].items() if k != "cols"}
                         for nm in CELLS},
               "cols": {nm: res[nm]["cols"] for nm in CELLS},
               "rows": rows}, open(args.out, "w"))
    print(f"\n[8] wrote {args.out}")


if __name__ == "__main__":
    main()
