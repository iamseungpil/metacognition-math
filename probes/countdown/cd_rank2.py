"""Countdown 오라클 판 2단계 (확정본) — ***PMI-shift 가 메타 품질을 아는가.***

`cd_rank.py`(0818 04:14) 를 대체한다. 적대검토 넷에서 살아남은 것만 넣었다.
정본(`/home/v-seungplee/metacognition-math`)은 한 글자도 안 건드린다.

═══════════════════════════════════════════════════════════════════════════
★이 판의 정답지는 **부분집합에서만** 정답지다  (0818 CPU 실측, seed 0 재생성 400문제)
═══════════════════════════════════════════════════════════════════════════
  O3 가 witness 를 **문자 그대로** 담는다            400/400 = 100.0%
  O1 이 **발산 연산자 자리**를 문자로 지목한다        81/400 =  20.2%
  OF 가 산술·규칙상 **유효한 등식**이다               250/400 =  62.5%
      (음수/0이하 130 · 비정수 나눗셈 104 · nums 밖 11 — `countdown.py:248-249` 가
       나누어떨어짐도 양수도 검사하지 않는다)

  ⇒ 참 품질 순서 O3>O1>{R,OF} 는 **축자 겹침 순위와 공선**이고, O1 vs OF 는
    "맞는 조언 vs 틀린 조언"이 아니라 **"참인 등식 vs 거짓 등식"**과 공선이다.

  모델을 **전혀 안 쓰는** 대조군의 실측값 (4팔 · R 은 cd_main 실측 텍스트 대입):
      칸                전체 400        ★CLEAN 199
      lex(축자겹침)     C5 0.800 T2 0.601   C5 0.800 T2 0.500
      conf(선언 확신도) C5 0.900 T2 0.500   C5 0.900 T2 0.500
      n_digits          C5 0.708 T2 0.599   C5 0.675 T2 0.503
      meta_len(토큰)    C5 0.476 T2 0.599   C5 0.425 T2 0.503
  ⇒ **전체집합의 T2 는 자릿수 세기(0.599)와 구별되지 않는다.** 사전등록 문턱 0.60 을
    자릿수 계산기가 통과한다. T1(4팔 사다리)은 conf 0.900 · lex 0.800 아래라
    **이 정답지로는 물을 수 없다** — 배선/참고 칸으로 강등한다.

★CLEAN = (O1 이 발산 자리를 안 덮음) ∧ (OF 가 유효 등식) — n=199/400.
  이 부분집합에서 **네 음성 대조군이 전부 0.500~0.503 으로 붕괴한다.**
  그래서 사전등록 주 검정은 여기 하나다:

      ★주 검정 = 라이브 pos_full(clip±2+역전) · 증인−연산자교체 · 메타경계 ·
                 O1 vs OF · **CLEAN 부분집합**

  라이브 정의는 `src/training/dcpo_pmi_shift.py:104-117` 그대로다:
      shift = close − open;  cont = clip(shift, ±2);  +1 save(o<0<c) / −2 derail(o>0>c)

═══════════════════════════════════════════════════════════════════════════
축2 경고 (그대로 유지 — 검산으로 확인됨)
═══════════════════════════════════════════════════════════════════════════
  문제 안 랭킹은 문제 안에서 **상수인 항**을 빼도 안 바뀐다. open(no-meta ctx)은 네 팔이
  공유하므로 **K1(위치차)의 순위 = close 원값의 순위**다. 즉 K1 은 shift 를 한 비트도
  안 담는다. 뺄셈이 순위를 바꾸는 길은 셋뿐이다:
      ① K2 팔-매칭 도너(팔마다 다른 대조)   ② 라이브식(open 에 비선형)   ③ 길이잔차
  ⇒ **shift 검정으로 인정하는 칸은 이 셋뿐**이고, 주 칸은 그중 라이브식이다.
    K1/close 는 독립 증거 둘이 아니라 하나이며 `[참고]` 로 낙인한다.

═══════════════════════════════════════════════════════════════════════════
판정 등급 — 1단계(판정 A)와의 관계
═══════════════════════════════════════════════════════════════════════════
  `verdict_A` 를 **읽어서 나란히 찍는다.** 다만 A 로 B 를 봉인하지 않는다:
    · TRUTH 는 조언에 담긴 **과제 정보량**이지 행동효과가 아니다(O3 는 정답 전체를 담는다).
    · cd_main 실측 MDE(95%) ≈ 2.8pp / 기준선 3.3% ⇒ **A 의 널은 대부분 검정력 부족**이다.
      검정력 없는 널로 B 를 봉인하면 동전던지기로 판정문을 정하게 된다.
  대신 등급을 붙인다:  A양성 → "행동으로도 확인된 사다리"  ·  A널 → "정보량 사다리(조건부)"

⚠오라클은 진단이지 학습 방법이 아니다. 여기 어떤 수도 성능 주장이 아니다.

사용:
    python cd_rank2.py --selftest              # GPU 없이 통계·부분집합·합성 end-to-end
    python cd_rank2.py --json cd_oracle.json   # 본 채점 (GPU 1장)
    python cd_rank2.py --json cd_oracle.json --own_post   # ⛔편향 칸 추가
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

METAS = ["R", "OF", "O1", "O3"]                 # ★넷이다. Np 는 메타 텍스트가 없다
TRUTH = {"R": 1.5, "OF": 1.5, "O1": 3.0, "O3": 4.0}
ORD_PAIRS = [("O3", "O1"), ("O3", "R"), ("O3", "OF"), ("O1", "R"), ("O1", "OF")]
ORD3 = [("O1", "R"), ("O1", "OF")]              # ★O3 제외 사다리 (O3 는 100% 복사)
TIE_PAIR = ("R", "OF")
KEY_PAIR = ("O1", "OF")

PRIMARY_CELL = "★라이브 pos_full(clip±2+역전)"   # 라이브 보상식 그대로 (dcpo_pmi_shift.py:104-117)
PRIMARY_SUB = "CLEAN"                            # ★주 검정은 이 부분집합에서만
REF_TAG = "[참고]"
NEG_TAG = "★음성 "
SAT_NATS = 20.0
POST_TOK = 600

PREREG = {
    # ★문턱은 **하드코딩하지 않는다** — 실행 시 max(음성 대조군)+margin 과
    #   Bonferroni 밴드 중 **큰 쪽**으로 유도한다. 아래는 하한(이보다 낮아질 수 없다).
    "floor_strong": 0.60,      # 어떤 경우에도 이 아래를 ★강 이라 부르지 않는다
    "floor_weak": 0.55,
    "neg_margin": 0.02,        # 음성 대조군을 이 폭 이상 이겨야 생존
    "pairing_tol": 0.03,       # CLEAN 에서 음성 T2 가 0.5±tol 안이어야 통제 성립
    "n_min": 100,
    "sat_warn": 0.50,
    "boot_B": 2000,
}


# ────────────────────────────────────────────────────────── 순수 통계 (GPU 무관)

def ranks(v):
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
        return None                     # 무변동 — 0 으로 세지 않는다
    return sum(x * y for x, y in zip(da, db)) / (sa * sb)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def rho_ceiling(truth_vec):
    """참 순위에 동점이 있으면 **연속값 예측의 실질 천장이 1.0 이 아니다** — 직접 계산."""
    best = -2.0
    for perm in itertools.permutations(range(len(truth_vec))):
        r = spearman([float(p) for p in perm], truth_vec)
        if r is not None:
            best = max(best, r)
    return best


def perm_null_sd(truth_vec):
    """치환 귀무(팔 라벨 순열)의 정확 표준편차. 근사 없음."""
    out = []
    for perm in itertools.permutations(range(len(truth_vec))):
        r = spearman([float(p) for p in perm], truth_vec)
        if r is not None:
            out.append(r)
    return math.sqrt(st.pvariance(out)), st.mean(out), len(out)


def z_bonf(m):
    """양측 0.05 의 Bonferroni z. scipy 없이 이분법으로 역누적정규."""
    from math import erf, sqrt
    target = 1.0 - 0.025 / max(1, m)
    lo, hi = 0.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def winrate1(a, b):
    return 1.0 if a > b else 0.5 if a == b else 0.0


def boot_cols(cols, B, seed, idx=None):
    """★문제 단위 부트스트랩 — 여러 통계가 **같은 재표집**을 공유한다(짝 유지).

    반환 {key: (obs, lo, hi, n_valid)}.  n_valid 는 **그 통계의** 표본수다
    (rho 는 무변동 행에서 None 이라 C 와 n 이 다르다 — 그래서 칸마다 따로 찍는다).
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
    obs, draws = {}, {k: [] for k in keys}
    for k in keys:
        v = [x for x in sub[k] if x is not None]
        obs[k] = (sum(v) / len(v) if v else None, len(v))
    for _ in range(B):
        sel = [rng.randrange(n) for _ in range(n)]
        for k in keys:
            col = sub[k]
            v = [col[i] for i in sel if col[i] is not None]
            draws[k].append(sum(v) / len(v) if v else None)
    out = {}
    for k in keys:
        d = sorted(x for x in draws[k] if x is not None)
        o, cnt = obs[k]
        out[k] = ((o, d[int(.025 * len(d))], d[int(.975 * len(d))], cnt)
                  if (o is not None and d) else (o, None, None, cnt))
    return out


def residualize(vals, cov):
    """문제 안에서 공변량에 회귀한 잔차. ⚠4점 2모수 — 보조 칸으로만 읽는다."""
    n = len(vals)
    mv, mc = sum(vals) / n, sum(cov) / n
    sxx = sum((c - mc) ** 2 for c in cov)
    if sxx < 1e-12:
        return list(vals)
    b = sum((c - mc) * (v - mv) for c, v in zip(cov, vals)) / sxx
    return [v - b * (c - mc) for v, c in zip(vals, cov)]


# ────────────────────────────────────────── 정답지 감사 (모델 없이 items 만으로)

_INNER = re.compile(r"\((\d+)\s*([-+*/])\s*(\d+)\)")
_DIG = re.compile(r"\d")
_OFP = re.compile(r"pursue is (\d+) ([-+*/]) (\d+) = (-?\d+)")


def _all_boxed(text):
    """모든 \\boxed{...} — 균형괄호 스캔. ⚠정규식판은 중첩에서 실패한다
    (`countdown.py:66-80` 이 그래서 스캐너를 쓴다. cd_rank.py:263 은 그 회귀였다)."""
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            break
        k, dep = j + 7, 1
        while k < len(text) and dep:
            dep += (text[k] == "{") - (text[k] == "}")
            k += 1
        if dep == 0:
            out.append(text[j + 7:k - 1].strip())
        i = j + 7
    return out


def _strip(s):
    return (s or "").replace(" ", "").replace("(", "").replace(")", "")


def div_pos(witness, decoy):
    """witness 와 decoy 가 갈리는 **단일 문자 위치**. 아니면 None."""
    if not decoy or len(decoy) != len(witness):
        return None
    p = [i for i in range(len(witness)) if witness[i] != decoy[i]]
    return p[0] if len(p) == 1 else None


def lex_covers(meta, witness, p):
    """★축자 겹침 공변량 — 메타가 **발산 연산자 자리**를 문자로 지목하는가.

    공백·괄호를 지운 정규형에서, p 를 포함하고 양쪽에 숫자가 있는 witness 부분열이
    메타 안에 그대로 나오면 1. 구성으로 O3=1(항상) · O1=발산이 인용 결합 안일 때 ·
    R/OF=0 이지만, **모델이 쓴 R 이 우연히 지목하는 경우도 잡는다**(그게 요점이다).
    """
    if p is None or not meta:
        return 0.0
    keep = [i for i, c in enumerate(witness) if c not in "()"]
    if p not in keep:
        return 0.0
    wn = "".join(witness[i] for i in keep)
    pn = keep.index(p)
    mn = _strip(meta)
    for i in range(pn, -1, -1):
        for j in range(pn + 1, len(wn) + 1):
            if j - i < 3:
                continue
            s = wn[i:j]
            if not any(c.isdigit() for c in s[:pn - i]):
                continue
            if not any(c.isdigit() for c in s[pn - i + 1:]):
                continue
            if s in mn:
                return 1.0
    return 0.0


def of_is_valid(meta_of, nums):
    """OF 가 **산술·게임규칙상 유효한 등식**인가.

    `countdown.py:248-249` 는 나누어떨어짐도 양수도 검사하지 않는다 ⇒ 실측 37.5% 가
    거짓 등식이거나 규칙 위반이다("13 / 25 = 0", "3 - 19 = -16"). 그런 팔은
    '그럴듯하지만 틀린 조언'이 아니라 **'명백히 거짓인 진술'**이라 통제군이 못 된다.
    """
    m = _OFP.search(meta_of or "")
    if not m:
        return False
    c, op, d, v = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    if c not in nums or d not in nums:
        return False
    if op == "/" and (d == 0 or c % d != 0):
        return False
    if op == "-" and c - d <= 0:
        return False
    return v > 0


def audit_items(items, idx):
    """정답지 감사 — 부분집합 인덱스(문제 위치 기준)와 진단 수치를 만든다."""
    o3v = o1cov = ofv = ndp = 0
    nonlex, valid = [], []
    for pos, i in enumerate(idx):
        it = items[i]
        w, d = it["witness"], it.get("decoy")
        p = div_pos(w, d)
        ndp += p is not None
        o3v += _strip(w) in _strip(it["O3"])
        c1 = lex_covers(it["O1"], w, p)
        o1cov += c1
        ok = of_is_valid(it["OF"], it["nums"])
        ofv += ok
        if c1 == 0.0:
            nonlex.append(pos)
        if ok:
            valid.append(pos)
    clean = sorted(set(nonlex) & set(valid))
    return {"n": len(idx), "o3_verbatim": o3v, "o1_covers_div": o1cov,
            "of_valid": ofv, "single_div": ndp,
            "NONLEX": nonlex, "OFVALID": valid, "CLEAN": clean}


# ────────────────────────────────────────────────────────────────── 칸 정의

def build_cells(own_post=False):
    """칸 = (row → {팔: 값 or None}, 포화기준키, 낙인).

    raw 키:  f"{tgt}_{ctx}" 또는 f"{tgt}_{ctx}:{팔}"   tgt ∈ {w, wd, sf}
      O  메타없음(경계) · M 메타(경계) · D 같은 팔의 **다른 문제** 메타
      OP/MP/DP  + 중립 post(N 팔 sample0)      MPo  + **자기 post** ⛔구성편향
    """
    TN = {"w": "증인단독", "wd": "증인−교체", "sf": "자기후보"}

    def delta(tgt, close, open_, per_arm):
        def f(r):
            out = {}
            for m in METAS:
                a = r["raw"].get(f"{tgt}_{close}:{m}")
                b = r["raw"].get(f"{tgt}_{open_}:{m}" if per_arm else f"{tgt}_{open_}")
                out[m] = None if (a is None or b is None) else a - b
            return out
        return f

    C = {}
    # ★라이브식이 주 칸 — 맨 앞에 둔다
    def live(r):
        out, o = {}, r["raw"].get("wd_O")
        for m in METAS:
            c = r["raw"].get(f"wd_M:{m}")
            if o is None or c is None:
                out[m] = None
                continue
            v = max(-2.0, min(2.0, c - o))
            if o < 0 < c:
                v += 1.0
            elif o > 0 > c:
                v -= 2.0
            out[m] = v
        return out
    C[PRIMARY_CELL] = (live, "wd_O", False)

    # shift 를 실제로 담는 나머지 둘
    C["증인−교체·문맥(팔매칭도너)·경계"] = (delta("wd", "M", "D", True), "wd_O", False)
    base = delta("wd", "M", "O", False)

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
    C["증인−교체·위치·경계(길이잔차)"] = (resid, "wd_O", False)

    # 나머지 축 — 전부 **탐색**
    for tgt in ("w", "sf"):            # wd 문맥칸은 위에서 이미 정의했다 (중복 금지)
        C[f"{TN[tgt]}·문맥(도너)·경계"] = (delta(tgt, "M", "D", True), f"{tgt}_O", False)
    C["증인−교체·위치·답직전"] = (delta("wd", "MP", "OP", False), "wd_OP", False)
    C["증인−교체·문맥(도너)·답직전"] = (delta("wd", "MP", "DP", True), "wd_OP", False)
    if own_post:
        C["⛔증인−교체·위치·답직전·자기post"] = (delta("wd", "MPo", "OP", False), "wd_OP", True)

    # ★참고 — shift 정보 0. K1 의 문제 안 순위 ≡ close 원값의 순위(수학적 항등)
    C[REF_TAG + "증인−교체·위치·경계(=close)"] = (base, "wd_O", False)
    C[REF_TAG + "증인−교체·close원값"] = (
        lambda r: {m: r["raw"].get(f"wd_M:{m}") for m in METAS}, "wd_O", False)
    C[REF_TAG + "증인단독·위치·경계"] = (delta("w", "M", "O", False), "w_O", False)

    # ★음성 대조군 — 모델을 전혀 안 쓴다. lex 가 이번 판의 진짜 문턱이다.
    C[NEG_TAG + "lex(축자겹침)"] = (lambda r: dict(r["lex"]), None, False)
    C[NEG_TAG + "conf"] = (lambda r: {m: float(r["conf"][m]) for m in METAS}, None, False)
    C[NEG_TAG + "n_digits"] = (lambda r: {m: float(r["n_digits"][m]) for m in METAS}, None, False)
    C[NEG_TAG + "meta_len"] = (lambda r: {m: float(r["meta_len"][m]) for m in METAS}, None, False)
    return C


def rank_mismatch(f1, f2, rows):
    """두 칸의 문제 안 순위가 다른 행 수. 유일한 수학적 경로는 **부동소수 ULP 붕괴**다."""
    bad = t1 = t2 = 0
    for r in rows:
        a = [f1(r)[m] for m in METAS]
        b = [f2(r)[m] for m in METAS]
        if any(x is None for x in a) or any(x is None for x in b):
            continue
        bad += 1 if ranks(a) != ranks(b) else 0
        t1 += 1 if len(set(a)) < len(a) else 0
        t2 += 1 if len(set(b)) < len(b) else 0
    return bad, t1, t2


def score_cell(cellf, rows, satkey, subs, B, seed):
    """한 칸에 T1(rho̅·C5·C3) + T2(전체/CLEAN/엄격) + 포화 + 비포화 재판정을 건다.

    ★포화 판정은 **open(`wd_O` 등 팔-불변 값)** 으로 한다. 이전 판은 네 팔 중 하나라도
      포화하면 버렸는데, O3 가 구조적으로 가장 크므로 그것은 **처치에 대한 조건화**였다.
    """
    tv = [TRUTH[m] for m in METAS]
    pk = [f"{a}>{b}" for a, b in ORD_PAIRS] + [f"{TIE_PAIR[0]}>{TIE_PAIR[1]}"]
    cols = {"rho": [], "C5": [], "C3": [], **{k: [] for k in pk}}
    n_flat = 0
    sat_arm = {m: [0, 0] for m in METAS}
    unsat = []
    for pos, r in enumerate(rows):
        v = cellf(r)
        vals = [v[m] for m in METAS]
        rr = None
        if all(x is not None for x in vals):
            rr = spearman(vals, tv)
            if rr is None:
                n_flat += 1
        cols["rho"].append(rr)
        for a, b in ORD_PAIRS + [TIE_PAIR]:
            cols[f"{a}>{b}"].append(None if (v[a] is None or v[b] is None)
                                    else winrate1(v[a], v[b]))
        cols["C5"].append(None if any(cols[f"{a}>{b}"][-1] is None for a, b in ORD_PAIRS)
                          else st.mean(cols[f"{a}>{b}"][-1] for a, b in ORD_PAIRS))
        cols["C3"].append(None if any(cols[f"{a}>{b}"][-1] is None for a, b in ORD3)
                          else st.mean(cols[f"{a}>{b}"][-1] for a, b in ORD3))
        # 포화 — 팔별로 세어 표에 남기고, **제외 판정은 팔-불변 open 으로만** 한다
        if satkey:
            tgt, ctx = satkey.split("_", 1)
            ck = "M" if ctx == "O" else "MP"
            for m in METAS:
                x = r["raw"].get(f"{tgt}_{ck}:{m}")
                if x is not None:
                    sat_arm[m][1] += 1
                    sat_arm[m][0] += abs(x) > SAT_NATS
        if satkey:
            o = r["raw"].get(satkey)
            if o is None or abs(o) <= SAT_NATS:
                unsat.append(pos)
        else:
            unsat.append(pos)

    kk = f"{KEY_PAIR[0]}>{KEY_PAIR[1]}"
    out = {"n_flat": n_flat, "cols": cols,
           "sat_arm": {m: (sat_arm[m][0] / sat_arm[m][1] if sat_arm[m][1] else None)
                       for m in METAS}}
    o_all = r_open = 0
    for r in rows:
        if satkey:
            o = r["raw"].get(satkey)
            o_all += 1
            r_open += 1 if (o is not None and abs(o) > SAT_NATS) else 0
    out["sat_open"] = (r_open / o_all) if (satkey and o_all) else None
    bs = boot_cols(cols, B, seed)
    out["rho"], out["C5"], out["C3"] = bs["rho"], bs["C5"], bs["C3"]
    out["pairs"] = {k: bs[k] for k in pk}
    for nm, idx in subs.items():
        b2 = boot_cols({"C5": cols["C5"], "C3": cols["C3"], kk: cols[kk]},
                       max(500, B // 2), seed + 7, idx=idx)
        out[f"sub_{nm}"] = {"C5": b2["C5"], "C3": b2["C3"], "o1of": b2[kk]}
    b3 = boot_cols({"C5": cols["C5"], kk: cols[kk]}, max(500, B // 2), seed + 9, idx=unsat)
    out["unsat"] = {"C5": b3["C5"], "o1of": b3[kk], "n": len(unsat)}
    return out


# ───────────────────────────────────────────────────────────────── 판정 문구

def make_bands(neg_vals, n_primary, n_cells_ci):
    """★문턱 유도 — 하드코딩하지 않는다.

    strong = max( 0.5 + 단일 95% 밴드,  max|음성−0.5| + 0.5 + margin,  하한 0.60 )
    exp    = 0.5 + Bonferroni(실제 CI 개수) 밴드      ← 탐색 칸에 쓴다
    """
    single = 1.96 * 0.5 / math.sqrt(max(1, n_primary))
    bonf = z_bonf(max(1, n_cells_ci)) * 0.5 / math.sqrt(max(1, n_primary))
    negmax = max([abs(v - 0.5) for v in neg_vals if v is not None] or [0.0])
    strong = max(0.5 + single, 0.5 + negmax + PREREG["neg_margin"], PREREG["floor_strong"])
    return {"single": single, "bonf": bonf, "negmax": negmax,
            "strong": strong, "lo_strong": max(0.5 + 1e-9, strong - single),
            "weak": max(PREREG["floor_weak"], 0.5 + negmax + PREREG["neg_margin"] / 2),
            "exp": 0.5 + bonf}


def verdict_pair(v, lo, hi, bands):
    if v is None or lo is None:
        return "표본부족"
    if v >= bands["strong"] and lo > bands["lo_strong"]:
        return "★가른다"
    if v >= bands["weak"] and lo > 0.5:
        return "약함(방향만)"
    if hi is not None and hi < 0.5:
        return "⛔역방향 유의"
    return "널"


def verdict_ladder(C, Clo, Chi, rho, band, bands):
    if C is None or Clo is None:
        return "표본부족"
    if C >= bands["strong"] and Clo > bands["lo_strong"]:
        return "★재현"
    if C >= bands["weak"] and Clo > 0.5:
        return "약함"
    if Chi is not None and Chi < 0.5:
        return "⛔역순 유의"
    if rho is not None and band is not None and abs(rho) <= band:
        return "확정널(치환밴드 안)"
    return "널"


# ────────────────────────────────────────────────────────── 자기검사 (GPU 없이)

def _synth_rows(n, mode, seed=0):
    """합성 행 — 배선·문턱·부분집합 경로를 GPU 없이 전부 밟는다."""
    rng = random.Random(seed)
    sig = {"R": 0.0, "OF": 0.0, "O1": 1.0, "O3": 2.0}
    rows = []
    for i in range(n):
        o = rng.gauss(0, 3)
        raw = {"wd_O": o, "w_O": o, "sf_O": o, "wd_OP": o, "w_OP": o}
        for m in METAS:
            if mode == "perfect":
                c = o + sig[m] * 1.0
            elif mode == "big":
                c = o + sig[m] * 5.0          # clip(±2) 이 뭉개는 크기
            else:
                c = o + rng.gauss(0, 1)
            raw[f"wd_M:{m}"] = c
            raw[f"wd_D:{m}"] = o + rng.gauss(0, 1)
            raw[f"w_M:{m}"] = c
            raw[f"w_D:{m}"] = o + rng.gauss(0, 1)
            raw[f"sf_M:{m}"] = c
            raw[f"sf_D:{m}"] = o + rng.gauss(0, 1)
            raw[f"wd_MP:{m}"] = c
            raw[f"wd_DP:{m}"] = o + rng.gauss(0, 1)
        rows.append({"i": i, "has_post": True, "raw": raw,
                     "acc": {m: 0.0 for m in METAS},
                     "meta_len": {"R": 45, "OF": 39, "O1": 39, "O3": 41},
                     "n_digits": {"R": 4, "OF": 6, "O1": 6, "O3": 9},
                     "conf": {"R": 0.7, "OF": 0.9, "O1": 0.9, "O3": 0.95},
                     "lex": {"R": 0.0, "OF": 0.0, "O1": 0.0, "O3": 1.0}})
    return rows


def selftest() -> None:
    tv = [TRUTH[m] for m in METAS]
    c4 = rho_ceiling(tv)
    c3 = rho_ceiling([TRUTH[m] for m in ("R", "OF", "O1")])
    sd, mu, nb = perm_null_sd(tv)
    print(f"[st] 참 순위 {dict(zip(METAS, tv))}")
    print(f"[st] rho 실질 상한 4팔 {c4:.4f} · 3팔 {c3:.4f}  (참 순위 동점 R≈OF 때문 · "
          f"기준 신뢰도 1.0 ⇒ 감쇠보정 없음)")
    print(f"[st] 치환 귀무 평균 {mu:+.1e} (0 이어야) · sd {sd:.4f} · 배치 {nb}")
    assert abs(mu) < 1e-9
    for n in (100, 150, 199, 250, 300):
        print(f"[st] n={n:3d} → rho̅ ±{1.96*sd/math.sqrt(n):.4f} · 승률 단일 ±{1.96*.5/math.sqrt(n):.4f}"
              f" · Bonf(m=40) ±{z_bonf(40)*.5/math.sqrt(n):.4f} · Bonf(m=140) ±{z_bonf(140)*.5/math.sqrt(n):.4f}")
    perfect = {"R": 0.0, "OF": 0.0, "O1": 1.0, "O3": 2.0}
    assert abs(st.mean(winrate1(perfect[a], perfect[b]) for a, b in ORD_PAIRS) - 1.0) < 1e-12
    assert abs(winrate1(perfect["R"], perfect["OF"]) - 0.5) < 1e-12
    assert abs(st.mean(winrate1(-perfect[a], -perfect[b]) for a, b in ORD_PAIRS)) < 1e-12

    # ── 축자 겹침 공변량 배선 검사
    w, d = "((9*2)+13)", "((9+2)+13)"
    p = div_pos(w, d)
    o1 = "confidence: 0.9 | The pairing to pursue is 9 * 2 = 18; ... | decision: redirect"
    o3 = f"confidence: 0.95 | The full expression that reaches the target is {w}; ..."
    assert p == 3 and lex_covers(o3, w, p) == 1.0 and lex_covers(o1, w, p) == 1.0
    assert lex_covers("confidence: 0.9 | The pairing to pursue is 5 + 4 = 9; ...", w, p) == 0.0
    w2, d2 = "((9*2)+13)", "((9*2)-13)"
    p2 = div_pos(w2, d2)
    assert p2 == 6 and lex_covers(o1, w2, p2) == 0.0 and lex_covers(o3, w2, p2) == 1.0
    assert of_is_valid("The pairing to pursue is 12 / 4 = 3; ", [12, 4]) is True
    assert of_is_valid("The pairing to pursue is 13 / 25 = 0; ", [13, 25]) is False
    assert of_is_valid("The pairing to pursue is 3 - 19 = -16; ", [3, 19]) is False
    print("[st] ✅lex_covers · of_is_valid · 순위·치환 배선 통과")

    # ── 합성 end-to-end (부분집합·문턱·판정문 경로 전부)
    for mode in ("perfect", "big", "noise"):
        rows = _synth_rows(120, mode, seed=1)
        subs = {"CLEAN": list(range(0, 120, 2)), "STRICT": list(range(0, 120, 3))}
        cells = build_cells()
        res = {nm: score_cell(f, rows, sk, subs, 300, 0) for nm, (f, sk, _) in cells.items()}
        neg = [res[k]["pairs"][f"{KEY_PAIR[0]}>{KEY_PAIR[1]}"][0]
               for k in cells if k.startswith(NEG_TAG)]
        bands = make_bands(neg, len(subs["CLEAN"]), 40)
        P = res[PRIMARY_CELL]
        v, lo, hi, n = P[f"sub_CLEAN"]["o1of"]
        pos_all = res[REF_TAG + "증인−교체·위치·경계(=close)"]
        vp = pos_all["sub_CLEAN"]["o1of"][0]
        nbad, _, _ = rank_mismatch(cells[REF_TAG + "증인−교체·위치·경계(=close)"][0],
                                   cells[REF_TAG + "증인−교체·close원값"][0], rows)
        print(f"[st] 합성 {mode:8s} → 주칸 T2(CLEAN) {v:.3f} [{lo:.3f},{hi:.3f}] n={n} "
              f"{verdict_pair(v, lo, hi, bands):>10s} · 무클립 pos {vp:.3f} · "
              f"C5 {P['C5'][0]:.3f} · K1≠close 행 {nbad}")
        if mode == "perfect":
            assert v > 0.9, "완벽신호에서 주칸이 안 갈린다 — 배선 고장"
        if mode == "big":
            assert v < vp + 1e-9, "clip 이 손해를 안 준다 — 라이브식 구현 확인"
    print("[st] ✅합성 end-to-end 통과 (perfect 는 갈리고, big 은 clip 으로 뭉개지고, "
          "noise 는 널)")


# ────────────────────────────────────────────────────────────────────── 본체

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="cd_oracle.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", default="cd_rank2.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--boot", type=int, default=PREREG["boot_B"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--own_post", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    selftest()
    if args.selftest:
        return

    b = json.load(open(args.json))
    items, ACC, PS, ARMS = b["items"], b["per_problem"], b["per_sample"], b["arms"]
    POOL = b["pool"][:args.limit] if args.limit else b["pool"]

    # ── [0] 1단계 판정 A 이어받기  (★verdict_A 를 실제로 읽는다)
    print("\n[0] 1단계(cd_oracle) — 배선·중단조건·판정 A")
    o3w = b.get("o3_wiring")
    fmt = b.get("fmt", float("nan"))
    stops = {"fmt<0.80": (fmt < 0.80), "pool<100": (len(b["pool"]) < 100),
             "o3wiring<0.80": (o3w is None or o3w < 0.80)}
    print(f"    형식 {fmt:.1%} · 모집단 acc(N)=0 {len(b['pool'])} (사용 {len(POOL)}) · "
          f"P(SAVE|O3) {('%.1f%%' % (o3w*100)) if o3w is not None else 'n/a'}")
    print("    ★중단조건 " + " · ".join(f"{k}{'⛔발동' if v else '✅'}" for k, v in stops.items())
          + "   ← cd_oracle.py 는 ①②를 인쇄만 하고 막지 않는다. 여기서 다시 검사한다.")
    print("    SAVE율  " + " · ".join(f"{c} {b['save_rate'][c]*100:.1f}%" for c in ARMS)
          + "   (N 은 모집단 정의상 항상 0.0%)")
    VA = b.get("verdict_A") or {}
    for k, v in VA.items():
        print(f"    판정A {k:8s} {v['delta']*100:+6.2f}pp "
              f"[{v['ci'][0]*100:+6.2f},{v['ci'][1]*100:+6.2f}]  "
              f"{'유의' if v['excludes_zero'] else '널'}")
    base_np = b["save_rate"].get("Np", 0.0)
    mde = 1.96 * math.sqrt(2 * base_np * (1 - base_np) / max(1, len(b["pool"]))) if base_np else None
    if mde:
        print(f"    ★검정력  기준선 Np {base_np*100:.2f}% · n {len(b['pool'])} ⇒ "
              f"MDE(95%) ≈ {mde*100:.2f}pp = 상대 {(base_np+mde)/base_np:.2f}배. "
              f"이보다 작은 차이의 '널'은 **검정력 부족과 구별되지 않는다**.")
    a_pos = bool(VA.get("O1-OF", {}).get("excludes_zero")) and VA.get("O1-OF", {}).get("delta", 0) > 0
    grade_txt = ("행동으로도 확인된 사다리" if a_pos else
                 "정보량 사다리(조건부 — 행동 확인 없음/검정력 부족)")
    print(f"    ⇒ 이 판의 정답지 등급: **{grade_txt}**")
    wiring_ok = (o3w is not None and o3w >= 0.80)

    # ── [1] decoy 재생성 + ★정답지 감사
    from countdown import swap_op_decoy, grade
    for i, it in enumerate(items):
        if not it.get("decoy"):
            it["decoy"] = swap_op_decoy(it["witness"], it["nums"], it["target"],
                                        random.Random(10_000 + i))
    aud = audit_items(items, POOL)
    print(f"\n[1] ★정답지 감사 (모델 없이 items 만으로) — n={aud['n']}")
    print(f"    O3 가 witness 를 문자 그대로 담음        {aud['o3_verbatim']}/{aud['n']}"
          f" = {aud['o3_verbatim']/aud['n']:.1%}   ⇒ O3 는 **복사 채널**이지 품질 증거가 아니다")
    print(f"    O1 이 발산 연산자 자리를 문자로 지목     {aud['o1_covers_div']:.0f}/{aud['n']}"
          f" = {aud['o1_covers_div']/aud['n']:.1%}   ⇒ 이 층은 T2 에서 제외한다")
    print(f"    OF 가 산술·규칙상 유효한 등식            {aud['of_valid']}/{aud['n']}"
          f" = {aud['of_valid']/aud['n']:.1%}   ⇒ 나머지는 '틀린 조언'이 아니라 '거짓 진술'")
    print(f"    단일문자 발산 decoy                      {aud['single_div']}/{aud['n']}")
    subs = {"CLEAN": aud["CLEAN"], "NONLEX": aud["NONLEX"], "OFVALID": aud["OFVALID"]}
    print(f"    ★부분집합  CLEAN {len(aud['CLEAN'])} · 비겹침 {len(aud['NONLEX'])} · "
          f"OF유효 {len(aud['OFVALID'])}")
    if len(aud["CLEAN"]) < PREREG["n_min"]:
        print(f"    ⚠CLEAN n={len(aud['CLEAN'])} < {PREREG['n_min']} — 주 검정이 검정력 부족이다. "
              f"'널'을 '신호 없음'으로 읽지 마라.")

    # ── [2] 자기후보 (n 이 작다 — 탐색 칸)
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
                for e in (x.replace(" ", "").strip() for x in _all_boxed(t)):
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
        selfc[i] = (None, None, None) if (good and d and good == d) else (good, gsrc, d)
    n_sf = sum(1 for i in POOL if selfc[i][0] and selfc[i][2])
    print(f"[2] 자기후보 가능 {n_sf}/{len(POOL)} (비오라클 출처 "
          f"{sum(1 for i in POOL if selfc[i][1] in NON_OR)}) — ⚠sample0 만 text 가 저장돼 "
          f"n 이 구조적으로 작다. 이 축은 **탐색**이다.")

    donor = {i: {m: items[POOL[(k + 1) % len(POOL)]][m] for m in METAS}
             for k, i in enumerate(POOL)}

    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cd_grid import pair, solo                     # 폴백 내장된 검증된 경로 재사용

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda").eval()

    PATH, rows = {}, []
    for k, i in enumerate(POOL):
        it = items[i]
        w, dec = it["witness"], it.get("decoy")
        p = div_pos(w, dec)
        g_self, _, d_self = selfc[i]
        # cd_oracle.py:144-145 와 **바이트 동일**한 문맥
        base = it["prompt"] + it["prefix"]
        ctxO = base + "\n"
        ctxM = {m: base + f"\n<meta>{it[m]}</meta>" + "\n" for m in METAS}
        ctxD = {m: base + f"\n<meta>{donor[i][m]}</meta>" + "\n" for m in METAS}
        t0 = (PS["N"][i][0] or {}).get("text") or ""
        j0 = t0.find("\\boxed{")
        pid = tok(t0[:j0] if j0 > 0 else "", add_special_tokens=False)["input_ids"][:POST_TOK]
        post = tok.decode(pid) if len(pid) >= 5 else ""

        raw = {}

        def fill(key, ctx):
            raw[f"w_{key}"] = solo(model, tok, ctx, w)
            raw[f"wd_{key}"] = pair(model, tok, ctx, w, dec, stats=PATH)
            raw[f"sf_{key}"] = pair(model, tok, ctx, g_self, d_self, stats=PATH)

        fill("O", ctxO)
        for m in METAS:
            fill(f"M:{m}", ctxM[m])
            fill(f"D:{m}", ctxD[m])
        if post:
            raw["wd_OP"] = pair(model, tok, ctxO + post, w, dec, stats=PATH)
            for m in METAS:
                raw[f"wd_MP:{m}"] = pair(model, tok, ctxM[m] + post, w, dec, stats=PATH)
                raw[f"wd_DP:{m}"] = pair(model, tok, ctxD[m] + post, w, dec, stats=PATH)
                if args.own_post:
                    to = (PS[m][i][0] or {}).get("text") or ""
                    jo = to.find("\\boxed{")
                    ip = tok(to[:jo] if jo > 0 else "",
                             add_special_tokens=False)["input_ids"][:POST_TOK]
                    if len(ip) >= 5:
                        raw[f"wd_MPo:{m}"] = pair(model, tok, ctxM[m] + tok.decode(ip),
                                                  w, dec, stats=PATH)
        rows.append({
            "i": i, "has_post": bool(post), "div_pos": p,
            "acc": {m: ACC[m][i] for m in METAS},
            "save": {m: (1 if ACC[m][i] > 1e-9 else 0) for m in METAS},
            "meta_len": {m: len(tok(it[m], add_special_tokens=False)["input_ids"])
                         for m in METAS},
            "n_digits": {m: len(_DIG.findall(it[m])) for m in METAS},
            "conf": {m: (lambda g: float(g.group(1)) if g else 0.5)(
                re.search(r"confidence:\s*([0-9.]+)", it[m])) for m in METAS},
            "lex": {m: lex_covers(it[m], w, p) for m in METAS},
            "raw": raw,
        })
        if (k + 1) % 25 == 0:
            print(f"    …{k+1}/{len(POOL)}  PMI 슬라이스 경로 {PATH}")
    print(f"[3] 원재료 완료 {len(rows)} · post 있는 문제 {sum(r['has_post'] for r in rows)} · "
          f"발산슬라이스 경로 {PATH}")
    print(f"    (full = 전체문자열 폴백. 이 비율이 크면 그 칸은 '발산토큰 대비'가 아니라 "
          f"'문자열 길이 대비'다 — 해석을 바꿔라)")

    # ── [4] 짝맞춤 실측 + 엄격 부분집합
    dl = [r["meta_len"]["O1"] - r["meta_len"]["OF"] for r in rows]
    dd = [r["n_digits"]["O1"] - r["n_digits"]["OF"] for r in rows]
    strict = [p for p, r in enumerate(rows)
              if abs(r["meta_len"]["O1"] - r["meta_len"]["OF"]) <= 1
              and r["n_digits"]["O1"] == r["n_digits"]["OF"]]
    subs["STRICT"] = strict
    subs["CLEAN∩STRICT"] = sorted(set(strict) & set(aud["CLEAN"]))
    print(f"[4] O1/OF 짝맞춤 실측 — 토큰길이차 중앙 {st.median(dl):+.1f}(평균 {st.mean(dl):+.2f}, "
          f"|차|>1 {sum(abs(x)>1 for x in dl)/len(dl):.1%}) · 숫자수차 중앙 {st.median(dd):+.1f}")
    print(f"    엄격(길이차≤1 & 숫자수 동일) {len(strict)} · CLEAN∩엄격 {len(subs['CLEAN∩STRICT'])}")
    for nm in ("STRICT", "CLEAN∩STRICT"):
        n = len(subs[nm])
        if n:
            hw = 1.96 * 0.5 / math.sqrt(n)
            print(f"    ⚠{nm} n={n} → 95% 반폭 ±{hw:.4f}. 'lo>0.5+단일밴드' 이려면 관측 "
                  f"≥ {0.5+2*hw:.3f} 가 필요하다 — **이 부분집합은 보고용이지 게이트가 아니다**.")

    # ── [5] 격자
    CELLS = build_cells(own_post=args.own_post)
    res = {nm: score_cell(f, rows, sk, subs, args.boot, args.seed)
           for nm, (f, sk, _) in CELLS.items()}

    nbad, t1, t2 = rank_mismatch(CELLS[REF_TAG + "증인−교체·위치·경계(=close)"][0],
                                 CELLS[REF_TAG + "증인−교체·close원값"][0], rows)
    print(f"\n[5] 배선검사 K1(위치차) vs close원값 순위 불일치 {nbad}/{len(rows)} "
          f"(동점 행 K1 {t1} · close {t2})")
    print(f"    {'✅동일 — open 은 문제 안 상수(설계상 예상). K1 은 shift 를 안 담는다.' if nbad == 0 else '⚠불일치 — 유일한 경로는 부동소수 ULP 붕괴다. 팔 간 PMI 차이가 잡음 수준이라는 뜻이니 그 자체를 판정문에 실어라.'}")

    # 라이브식 해상도 진단 — O1/OF **쌍에 대해서만**
    kk = f"{KEY_PAIR[0]}>{KEY_PAIR[1]}"
    nclip = ntot = 0
    for r in rows:
        o = r["raw"].get("wd_O")
        for m in METAS:
            c = r["raw"].get(f"wd_M:{m}")
            if o is not None and c is not None:
                ntot += 1
                nclip += abs(c - o) > 2.0
    lv_tie = st.mean(1 if x == 0.5 else 0
                     for x in res[PRIMARY_CELL]["cols"][kk] if x is not None)
    pos_tie = st.mean(1 if x == 0.5 else 0
                      for x in res[REF_TAG + "증인−교체·위치·경계(=close)"]["cols"][kk]
                      if x is not None)
    print(f"\n[6] ★라이브식 해상도 진단  clip(±2) 에 걸린 (문제×팔) "
          f"{nclip}/{ntot} = {(nclip/ntot*100 if ntot else float('nan')):.1f}%")
    print(f"    O1 vs OF 가 **동점이 된** 비율:  라이브 {lv_tie*100:.1f}%  vs  무클립 {pos_tie*100:.1f}%")
    print(f"    ⇒ 라이브 쪽 동점이 크게 높으면 낮은 T2 는 '신호 없음'이 아니라 "
          f"**'라이브식이 해상도를 버렸다'**이고, 수리는 clip 상향/제거다.")

    # ── [7] 문턱 유도
    negs = [k for k in CELLS if k.startswith(NEG_TAG)]
    n_ci = sum(1 for nm in CELLS if not nm.startswith(NEG_TAG)) * (3 + len(ORD_PAIRS) + 1 + 3 * len(subs))
    negK_clean = {k: res[k][f"sub_{PRIMARY_SUB}"]["o1of"][0] for k in negs}
    negK_all = {k: res[k]["pairs"][kk][0] for k in negs}
    negC_clean = {k: res[k][f"sub_{PRIMARY_SUB}"]["C5"][0] for k in negs}
    negC3_clean = {k: res[k][f"sub_{PRIMARY_SUB}"]["C3"][0] for k in negs}
    n_prim = len(subs[PRIMARY_SUB])
    bands = make_bands(list(negK_clean.values()), n_prim, n_ci)
    short = lambda k: k[len(NEG_TAG):]
    print(f"\n[7] ★음성 대조군 (모델을 전혀 안 쓴다) — 이것이 문턱이다")
    print("    T2 전체집합 : " + " · ".join(f"{short(k)} {v:.3f}" for k, v in negK_all.items()))
    print(f"    T2 {PRIMARY_SUB:6s}: " + " · ".join(f"{short(k)} {v:.3f}" for k, v in negK_clean.items())
          + "   ← 전부 0.5±{:.2f} 안이어야 통제 성립".format(PREREG["pairing_tol"]))
    broke = [short(k) for k, v in negK_clean.items()
             if v is not None and abs(v - .5) > PREREG["pairing_tol"]]
    if broke:
        print(f"    ⚠통제 붕괴 — {broke} 가 밴드 밖이다. 주 검정의 해석을 그만큼 깎아라.")
    print("    T1 C5 " + PRIMARY_SUB + " : " + " · ".join(f"{short(k)} {v:.3f}" for k, v in negC_clean.items()))
    print("    T1 C3 " + PRIMARY_SUB + " : " + " · ".join(f"{short(k)} {v:.3f}" for k, v in negC3_clean.items()))
    print(f"    ⇒ ★T1(사다리)은 **이 정답지로 물을 수 없다**: conf·lex 가 구성상 사다리를 "
          f"맞힌다. T1 칸은 전부 **참고**로만 읽고, 판정은 T2 로 한다.")
    print(f"[7b] 문턱 유도 (하드코딩 아님)  n(주)={n_prim} · CI 개수 m={n_ci}")
    print(f"     단일 95% 밴드 ±{bands['single']:.4f} · Bonferroni 밴드 ±{bands['bonf']:.4f} · "
          f"max|음성−0.5| {bands['negmax']:.4f}")
    print(f"     ⇒ ★강 문턱 {bands['strong']:.4f} (하한 > {bands['lo_strong']:.4f}) · "
          f"약함 {bands['weak']:.4f} · 탐색칸 문턱 {bands['exp']:.4f}")

    # ── [8] 격자 표
    ceil4 = rho_ceiling([TRUTH[m] for m in METAS])
    sd_null, _, _ = perm_null_sd([TRUTH[m] for m in METAS])
    print(f"\n[8] 격자 — 참 순위 O3>O1>R≈OF · rho 상한 {ceil4:.4f} · C 상한 1.000")
    hdr = (f" {'칸':34s} {'T2 CLEAN':>21s} {'T2 전체':>9s} {'C5':>9s} {'C3':>9s} "
           f"{'rho̅':>9s} {'n_rho':>6s} {'n_C':>5s} {'포화(open)':>10s}")
    print(hdr)
    print("─" * 150)
    for nm in CELLS:
        R = res[nm]
        vc, lc, hc, nc = R[f"sub_{PRIMARY_SUB}"]["o1of"]
        va = R["pairs"][kk][0]
        C5, _, _, n5 = R["C5"]
        C3v = R["C3"][0]
        rh, _, _, nr = R["rho"]
        s = lambda x: "  n/a" if x is None else f"{x:.3f}"
        s_c = "표본부족" if vc is None else f"{vc:.3f}[{lc:.3f},{hc:.3f}]"
        star = "★" if (vc is not None and lc is not None
                       and (lc > bands["exp"] or (hc is not None and hc < 1 - bands["exp"]))) else " "
        so = "  n/a" if R["sat_open"] is None else f"{R['sat_open']*100:8.1f}%"
        print(f"{star}{nm:34s} {s_c:>21s} {s(va):>9s} {s(C5):>9s} {s(C3v):>9s} "
              f"{s(rh):>9s} {nr:6d} {n5:5d} {so:>10s}")
    print("    (★ = Bonferroni 밴드 밖. 주 칸만 사전등록이고 나머지는 **탐색**이다.)")

    # ── [9] ★행동과 나란히  (0731 「네 번째 칸 = 기준선」)
    print(f"\n[9] ★행동 vs PMI — 같은 표에 놓는다 (둘이 갈리면 그 자체가 결론이다)")
    P = res[PRIMARY_CELL]
    mrank = {}
    for m in METAS:
        vv = []
        for pos in subs[PRIMARY_SUB]:
            v = CELLS[PRIMARY_CELL][0](rows[pos])
            if all(v[x] is not None for x in METAS):
                vv.append(ranks([v[x] for x in METAS])[METAS.index(m)])
        mrank[m] = st.mean(vv) if vv else float("nan")
    print(f"    {'팔':4s} {'참 품질':>7s} {'SAVE율(전체pool)':>16s} {'주칸 평균순위(CLEAN)':>20s}")
    for m in METAS:
        print(f"    {m:4s} {TRUTH[m]:7.1f} {b['save_rate'].get(m, float('nan'))*100:15.1f}% "
              f"{mrank[m]:20.2f}")
    print(f"    ⇒ PMI 가 사다리를 맞혀도 SAVE 가 안 움직이면 결론은 "
          f"**'PMI 는 정보량은 읽고 유용성은 못 읽는다'**이지 '성공'이 아니다.")

    # ── [10] 판정
    print(f"\n[10] ★판정")
    print(f"  {'칸':34s} {'T2(CLEAN)':>12s} {'음성게이트':>10s}  비고")
    surv = []
    for nm in CELLS:
        if nm.startswith(NEG_TAG) or nm.startswith(REF_TAG):
            continue
        R = res[nm]
        vc, lc, hc, nc = R[f"sub_{PRIMARY_SUB}"]["o1of"]
        v2 = verdict_pair(vc, lc, hc, bands)
        g = (vc is not None and all(abs(vc - .5) > abs(x - .5) + PREREG["neg_margin"]
                                    for x in negK_clean.values() if x is not None))
        note = ""
        if nc < PREREG["n_min"]:
            note += f" ⚠n={nc}<{PREREG['n_min']}"
        if R["sat_open"] is not None and R["sat_open"] > PREREG["sat_warn"]:
            u = R["unsat"]
            uo = u["o1of"][0]
            uos = "n/a" if uo is None else f"{uo:.3f}"
            note += f" ⚠open 포화 {R['sat_open']*100:.0f}% → 비포화 O1>OF={uos} (n={u['n']})"
        sa = R["sat_arm"]
        if any(x is not None and x > PREREG["sat_warn"] for x in sa.values()):
            note += "  팔별포화 " + "/".join(f"{m}{(sa[m] or 0)*100:.0f}%" for m in METAS)
        if nm.startswith("⛔"):
            note += "  ⛔구성편향(자기post)"
        print(f"  {nm:34s} {v2:>12s} {'✅' if g else '⛔':>10s} {note}")
        if g and v2.startswith("★") and not nm.startswith("⛔"):
            surv.append(nm)

    vc, lc, hc, nc = P[f"sub_{PRIMARY_SUB}"]["o1of"]
    va, la, ha, na = P["pairs"][kk]
    pos = res[REF_TAG + "증인−교체·위치·경계(=close)"]
    pc = pos[f"sub_{PRIMARY_SUB}"]["o1of"]
    print(f"\n  ═══ ★사전등록 주 검정 ═══")
    print(f"    칸 = {PRIMARY_CELL}   (= dcpo_pmi_shift.py:104-117 의 라이브 보상식)")
    print(f"    부분집합 = {PRIMARY_SUB} (O1 이 발산 자리를 안 덮음 ∧ OF 가 유효 등식)")
    print(f"    ★T2 O1 vs OF = {vc if vc is None else round(vc,4)} "
          f"[{'n/a' if lc is None else round(lc,4)},{'n/a' if hc is None else round(hc,4)}] "
          f"n={nc}  → {verdict_pair(vc, lc, hc, bands)}")
    print(f"      참고: 전체집합 {va if va is None else round(va,4)} "
          f"(⚠자릿수 세기 대조군과 구별 불가) · 무클립 pos(CLEAN) "
          f"{pc[0] if pc[0] is None else round(pc[0],4)}")
    for nm in ("STRICT", "CLEAN∩STRICT"):
        x = P[f"sub_{nm}"]["o1of"]
        print(f"      {nm}: {x[0] if x[0] is None else round(x[0],4)} (n={x[3]}) "
              f"— 보고용(검정력 부족)")
    print(f"    ⇒ 가르면 PMI 는 **내용의 정확성**을 본다. 못 가르면 **구체성/복사**만 본다.")
    print(f"    ⇒ 라이브 < 무클립 이면 결함은 PMI 가 아니라 **clip(±2)** 이다.")
    print(f"  ⇒ 생존 칸: {surv if surv else '없음'}")
    print(f"  ⇒ 정답지 등급: {grade_txt}" + ("" if wiring_ok else "  ⚠배선 미달이라 전부 조건부"))
    print(f"  ⛔이 판이 **답하지 않는** 것: O3 는 100% 복사라 사다리 증거가 아니다 · "
          f"T1 은 conf/lex 와 공선이라 물을 수 없다 · 전체집합 T2 는 자릿수와 공선이다.")

    json.dump({"model": args.model, "src": args.json, "n": len(rows),
               "wiring_ok": wiring_ok, "o3_wiring": o3w, "stops": stops,
               "verdict_A": VA, "mde_pp": (mde * 100 if mde else None),
               "truth_grade": grade_txt,
               "audit": {k: v for k, v in aud.items()
                         if k not in ("NONLEX", "OFVALID", "CLEAN")},
               "subsets": {k: len(v) for k, v in subs.items()},
               "bands": bands, "n_ci": n_ci, "prereg": PREREG,
               "truth": TRUTH, "rho_ceiling": ceil4, "rho_null_sd": sd_null,
               "pmi_path": PATH, "primary_cell": PRIMARY_CELL, "primary_sub": PRIMARY_SUB,
               "k1_equals_close": nbad == 0, "clip_rate": (nclip / ntot if ntot else None),
               "tie_live_vs_pos": [lv_tie, pos_tie], "survivors": surv,
               "neg_T2_clean": negK_clean, "neg_T2_all": negK_all,
               "mean_rank_primary": mrank,
               "cells": {nm: {k: v for k, v in res[nm].items() if k != "cols"} for nm in CELLS},
               "cols": {nm: res[nm]["cols"] for nm in CELLS},
               "rows": rows}, open(args.out, "w"))
    print(f"\n[11] wrote {args.out}")


if __name__ == "__main__":
    main()
