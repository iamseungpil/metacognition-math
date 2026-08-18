"""Countdown 메타인지 RL — 보상 여덟 팔(A~H)과 텔레메트리의 **단일 정의처**.

왜 별도 모듈인가. 이 저장소에는 "선언된 판정식과 채택된 수치가 불일치"한 전례가 있고
(원장 0726: Stage G 선언대로면 0.000 FAIL, 채택된 per_cell 로는 0.785 통과), "선언은 있고
배선은 없는 무효 레버"로 실험을 여러 번 날린 전례가 있다. 그래서:

  * **팔의 정체(ARM_SPECS)는 이 파일 한 곳에만 있다.** 런처·트레이너·분석이 전부 여기서
    읽는다. 팔별 diff(G8)의 근거는 `arm_signature()` 한 함수다.
  * **이미 있는 식은 재사용한다.** 위치대조(shift)는 `dcpo_pmi_shift.pmi_shift_reward`,
    문맥대조 clip 은 `dcpo_rmeta_forms.FORMULAS["ans_clip"]` 를 그대로 호출한다.
    복제하면 두 곳이 갈린다.
  * **켜진 항의 원재료가 없으면 조용히 0 을 흘리지 않고 즉사한다**(무효 레버 방지).
    반대로 원재료가 있는데 NaN/inf 면 0.0 으로 fail-closed(포이즌 행이 그룹 형제의
    센터링을 망치지 못하게).

torch·verl 을 import 하지 않는다 — 전부 순수 함수라 CPU 로 테스트된다.
(transitively 들어오는 것은 numpy 뿐이며, 이는 `dcpo_pmi_shift` 가 이미 쓰는 것이다.)

──────────────────────────────────────────────────────────────────────────────
사양(0818 확정판)과 이 파일의 대응
──────────────────────────────────────────────────────────────────────────────
공통(처치 아님):  w_corr 1.0 · w_format 0.35 · meta_floor 0.02 · 나머지 0
  A corr    R_corr 만
  B cur     + clip(shift,±2) + reversal(save 1.0/derail 2.0)        현행 재현
  C mul     + [위 항] × sign(A_corr)                                 ★곱하기
  D ctx     + clip(문맥대조,±2) × sign(A_corr)                        ★계산방식
  E gate    + −(2·p̂−1) × 1{메타 냈다}                                ★게이팅
  F full    C + E
  G neg     + meta_len/100                                          가짜 대조군
  H oldfmt  F 와 같은 보상, **옛 메타 형식**                           형식의 값어치

⚠**사양 내부 충돌 하나(사람 판단 필요, 임의로 정하지 않았다)**
  사양의 팔 표는 `C = clip(shift,±2) × sign` 이라 적어 **reversal 항이 없다**. 그러나
  구현 지시(항목 4)는 `r_meta_mul = r_meta_pos × sign` 이고 `r_meta_pos`(항목 3)는
  reversal 을 포함한 현행식이다. 두 읽기가 다르다.
  → 여기서는 **지시(항목 4)를 따랐다**: C·F·H 의 meta 항은 reversal 을 포함한다.
  → 표 쪽 읽기를 채택하려면 `SHIFT_PARAMS_MUL` 의 save/derail 을 0.0 으로 두면 된다.
    **함수를 고치지 말고 이 상수만 고쳐라** — 그래야 팔 정체가 한 곳에 남는다.

⚠**사양이 값을 안 정한 것(미확인, 기본값을 선언으로 박아둔다)**
  w_meta(=W_META) · w_gate(=W_GATE) · w_len(=W_LEN) 의 크기. 사양은 warmup 대상이라고만
  적고 크기를 안 준다. 여기서는 전부 1.0 으로 선언했다. 바꾸려면 상수 한 줄.

──────────────────────────────────────────────────────────────────────────────
행(row) 규약 — 호출자가 채운다. 전부 롤아웃 하나에 대한 스칼라
──────────────────────────────────────────────────────────────────────────────
    text          모델 **응답** 텍스트 (프롬프트 제외 — meta_position 이 응답 기준이다)
    r_corr        1{식이 목표수를 만들고 주어진 수를 각각 정확히 한 번씩 쓴다}
                  ★countdown.grade(text, nums, target) 의 출력. gold 불필요.
    format_ok     1{\\boxed 식이 파싱되고 + 그 팔의 메타 형식을 지켰다}
                  ★`format_ok_row()` 헬퍼 참조
    emitted       1{그 팔의 형식에 맞는 메타를 냈다}  ★`parse_meta()` 의 emitted
    pmi_open      메타 **앞** 위치의  logp(증인식|문맥) − logp(연산자교체오답|문맥)
    pmi_close     메타 **뒤** 위치
    pmi_self      답 직전 위치, 문맥 = **내** 메타
    pmi_donor     답 직전 위치, 문맥 = **같은 문제 다른 롤아웃**의 메타 (D 의 문맥대조)
    adv_corr      correctness 어드밴티지(시퀀스 수준). 부호만 쓴다.
    meta_n_tok    메타 토큰 수 (G 전용)
    group_id      같은 프롬프트에서 나온 롤아웃끼리 같은 값 (p̂ 의 단위)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

# ── 재사용: 복제 금지. 이 두 줄이 "두 곳이 갈리는" 사고를 막는다. ──────────────
from src.training.dcpo_pmi_shift import (
    compute_pmi_shift_reward as _compute_pmi_shift_reward,
    pmi_shift_reward as _pmi_shift_reward,
)
from src.training.dcpo_rmeta_forms import FORMULAS as _RMETA_FORMULAS

__all__ = [
    # 상수 / 팔 정체
    "SPEC_VERSION", "W_CORR", "W_FORMAT", "META_FLOOR", "W_META", "W_GATE", "W_LEN",
    "SHIFT_PARAMS", "SHIFT_PARAMS_MUL", "CTX_CLIP", "TERMS", "ARM_SPECS",
    "arm_signature", "all_arm_signatures",
    # 보상
    "compute_phat", "compute_phat_loo", "sign_of",
    "r_gate", "r_meta_pos", "r_meta_mul", "r_meta_ctx", "r_len", "r_meta_floor",
    "warmup_scale", "arm_reward",
    # 파싱 / 형식
    "parse_meta", "meta_form_ok", "format_ok_row",
    # 텔레메트리
    "emit_rate", "meta_position_frac", "meta_position_stats", "selectivity_index",
    "boilerplate_rate", "answer_leak", "answer_leak_rate", "meta_has_arithmetic",
    "arithmetic_in_meta_rate", "group_emit_dispersion", "phat_distribution",
    "decision_distribution", "confidence_unique", "component_means", "shift_diag",
    "length_stats", "telemetry_report",
    # 중단 조건
    "ABORT_RULES", "check_abort", "check_negative_control",
    "LAUNCHED_ARMS", "LAUNCHED_TREATMENT_ARMS",
]

SPEC_VERSION = "countdown-6arm-0818"

# ★이 실험이 **실제로 발사하는** 팔. ARM_SPECS 에는 D(문맥대조)·H(옛형식)도 정의돼
#   있지만 이번 판은 여섯만 돌린다(D 는 오프라인에서 위치와 구별되지 않았고 — 0.527 vs
#   0.464 — H 는 형식 두 벌 데이터가 필요해 후속으로 미뤘다). 이 목록이 판정 함수의
#   기본값을 정한다. 정의가 남아 있다는 이유로 "돌았다"고 읽히면 안 된다.
LAUNCHED_ARMS: tuple = ("A", "B", "C", "E", "F", "G")
LAUNCHED_TREATMENT_ARMS: tuple = ("B", "C", "E", "F")   # A=기준, G=가짜 대조군

# ══════════════════════════════════════════════════════════════════════════════
# 1. 상수 — 사양의 숫자는 전부 여기 한 번씩만 등장한다
# ══════════════════════════════════════════════════════════════════════════════

W_CORR = 1.0        # 공통. 처치 아님.
W_FORMAT = 0.35     # 공통. 처치 아님.
META_FLOOR = 0.02   # 공통. 처치 아님. 메타를 냈다는 사실 자체에 붙는 바닥값.
                    # ⚠미확인: 사양의 "meta_floor 0.02" 가 (a) 발화 보너스인지
                    # (b) R_meta 의 하한 clamp(max(R_meta, 0.02))인지 명시가 없다.
                    # 여기서는 (a) 발화 보너스로 배선했다 — 0.89→0.54 발화 침식
                    # 전례에 대한 방어라는 읽기다. (b) 로 바꾸려면 `arm_reward` 의
                    # meta_floor 분기 한 곳만 고치면 된다.

# 사양이 크기를 안 준 처치 항의 무게(미확인 → 1.0 으로 선언).
W_META = 1.0
W_GATE = 1.0
W_LEN = 1.0

# 위치대조(shift) 파라미터. B 는 사양 그대로.
SHIFT_PARAMS = dict(scale=1.0, clip=2.0, reversal_save=1.0, reversal_derail=2.0,
                    reversal_min_magnitude=0.0)
# C·F·H 가 sign 을 곱하기 전에 쓰는 파라미터. 위 ⚠사양 충돌 참조 —
# 표 쪽 읽기(reversal 없음)를 채택하려면 save/derail 을 0.0 으로.
SHIFT_PARAMS_MUL = dict(SHIFT_PARAMS)

CTX_CLIP = 2.0      # 문맥대조 clip. `dcpo_rmeta_forms._clip` 의 기본값과 같아야 한다
                    # (그 함수를 그대로 호출하므로 구조적으로 같다 — 테스트가 지킨다).

# sign(adv_corr) == 0 일 때의 정책. 아래 `r_meta_mul` 주석 참조.
SIGN_ZERO_VALUE = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. 작은 순수 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _f(x) -> float:
    """float 로 만들되 NaN/inf 는 0.0 (fail-closed)."""
    return float(x) if _finite(x) else 0.0


def sign_of(x) -> float:
    """부호. NaN/inf/None 은 0.0 (fail-closed) — 무한대에 부호를 주면 그 행이 그룹을 먹는다."""
    if not _finite(x):
        return 0.0
    v = float(x)
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


def _bool01(x) -> int:
    return 1 if bool(x) else 0


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(float(v) for v in xs) / len(xs) if xs else 0.0


def _quantile(xs: Sequence[float], q: float) -> float:
    """선형보간 분위수. numpy 없이 — 이 파일의 텔레메트리는 의존성 0 으로 돈다."""
    ys = sorted(float(v) for v in xs)
    if not ys:
        return float("nan")
    if len(ys) == 1:
        return ys[0]
    p = min(1.0, max(0.0, float(q))) * (len(ys) - 1)
    lo = int(math.floor(p))
    hi = int(math.ceil(p))
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (p - lo)


# ══════════════════════════════════════════════════════════════════════════════
# 3. p̂ — 그룹의 자가검증 성공률 (gold 를 안 쓴다)
# ══════════════════════════════════════════════════════════════════════════════

# compute_phat 이 **절대 읽지 않는** 키들. 행에 이 키가 있어도 p̂ 는 바뀌지 않아야 한다
# (테스트가 포이즌 값으로 확인한다). "gold 불필요"는 주석이 아니라 검사 대상이다.
GOLD_KEYS = ("gold", "witness", "solution", "ground_truth", "answer", "target_expr")


def compute_phat(
    group_rows: Sequence[Mapping],
    *,
    solved_key: str = "r_corr",
    grader: Callable[[Mapping], int] | None = None,
) -> float:
    r"""p̂_g = 그룹 g 에서 **목표수를 실제로 만든** 롤아웃의 비율.

    ★gold 가 필요 없다. target 이 프롬프트에 있고 식을 평가하면 되기 때문이다.
    두 가지 경로 중 하나로 쓴다:

      1. 호출자가 이미 채점해 뒀다 → 각 행의 `solved_key`(기본 "r_corr") 를 읽는다.
         그 값은 `countdown.grade(text, nums, target)` 의 출력이어야 한다.
      2. 채점기를 넣는다 → `grader(row) -> 0/1` 을 호출한다.
         예:  grader=lambda r: grade(r["text"], r["nums"], r["target"])

    빈 그룹은 **예외**다. 0.0 을 돌려주면 r_gate 가 +1(발화 장려) 로 읽혀 배선 사고가
    조용히 학습 신호가 된다. 그룹은 rollout.n 개가 항상 있어야 하므로 빈 그룹은 버그다.
    """
    rows = list(group_rows)
    if not rows:
        raise ValueError("compute_phat: 빈 그룹. rollout.n 개가 있어야 한다 — 배선 버그다.")
    vals = []
    for r in rows:
        if grader is not None:
            v = grader(r)
        else:
            if solved_key not in r:
                raise KeyError(
                    f"compute_phat: 행에 {solved_key!r} 가 없다. 채점 결과를 채우거나 "
                    f"grader= 를 넘겨라 (조용히 0 을 흘리지 않는다).")
            v = r[solved_key]
        vals.append(1.0 if _bool01(v) else 0.0)
    return sum(vals) / len(vals)


def compute_phat_loo(
    group_rows: Sequence[Mapping],
    *,
    solved_key: str = "r_corr",
    grader: Callable[[Mapping], int] | None = None,
) -> list[float]:
    r"""행별 **leave-one-out** p̂ — 자기 자신을 뺀 그룹 성공률.

    사양의 p̂ 는 그룹 전체(자기 포함)다. 이 함수는 그 대안이며 **기본 경로가 아니다**.
    왜 두는가: r_gate 는 p̂ 로 발화를 상벌하는데, 자기 포함 p̂ 는 자기 정답 여부가
    자기 게이트 보상에 1/n 만큼 직접 섞인다(n=8 이면 0.125 — 게이트가 재려는 "이 문제가
    어려운가"에 "내가 맞혔는가"가 새어든다). LOO 는 그 누출을 0 으로 만든다.
    ★채택 여부는 사람이 정한다. 채택하면 `arm_reward(phat=...)` 에 이 값을 넣으면 되고
    함수는 한 줄도 안 바뀐다.
    """
    rows = list(group_rows)
    if not rows:
        raise ValueError("compute_phat_loo: 빈 그룹.")
    if len(rows) == 1:
        raise ValueError("compute_phat_loo: n=1 그룹에는 leave-one-out 이 정의되지 않는다.")
    solved = []
    for r in rows:
        if grader is not None:
            v = grader(r)
        else:
            if solved_key not in r:
                raise KeyError(f"compute_phat_loo: 행에 {solved_key!r} 가 없다.")
            v = r[solved_key]
        solved.append(1.0 if _bool01(v) else 0.0)
    tot = sum(solved)
    n = len(solved)
    return [(tot - s) / (n - 1) for s in solved]


# ══════════════════════════════════════════════════════════════════════════════
# 4. 보상 항 — 전부 순수 함수, 전부 여기 한 번씩만
# ══════════════════════════════════════════════════════════════════════════════

def r_gate(emitted, phat) -> float:
    r"""E 팔. R_gate = −(2·p̂ − 1) × 1{메타 냈다}.

    p̂=0 (그룹 전원 실패, 오라클 판에서 63.7%) → +1 : 어려울 때 내면 상.
    p̂=1 (그룹 전원 성공)                     → −1 : 이미 되는데 내면 벌.
    p̂=0.5                                    →  0 : 중립.
    안 냈으면 0 — 발화 자체에만 걸리는 항이라 침묵에는 상벌이 없다.

    p̂ 가 NaN/None 이면 0.0 (fail-closed). p̂ 는 [0,1] 밖이면 안 되므로 클램프하지 않고
    **예외**를 던진다 — 범위를 벗어난 p̂ 는 집계 버그이고, 클램프하면 그 버그가 숨는다.
    """
    if not _bool01(emitted):
        return 0.0
    if not _finite(phat):
        return 0.0
    p = float(phat)
    if not (0.0 - 1e-9 <= p <= 1.0 + 1e-9):
        raise ValueError(f"r_gate: p̂={p} 가 [0,1] 밖이다 — 집계 버그다.")
    return -(2.0 * p - 1.0)


def r_meta_pos(pmi_open, pmi_close, *, params: Mapping | None = None) -> float:
    r"""B 팔(현행식). clip(shift,±2) + reversal(save 1.0 / derail 2.0).

    ★`dcpo_pmi_shift.pmi_shift_reward` 를 **그대로 호출한다**. 복제하지 않는다 —
    두 곳에 같은 식이 있으면 언젠가 갈리고, 그때 "선언된 판정식 ≠ 채택된 수치"가 된다.
    이 파일이 더하는 것은 사양의 파라미터(SHIFT_PARAMS)를 한 곳에 고정하는 것뿐이다.
    """
    p = dict(SHIFT_PARAMS if params is None else params)
    return float(_pmi_shift_reward(pmi_open, pmi_close, **p))


def r_meta_mul(pmi_open, pmi_close, adv_corr_sign, *, params: Mapping | None = None) -> float:
    r"""C 팔. r_meta_pos × sign(A_corr).

    ★sign == 0 일 때(그룹 전원 동점 → correctness 어드밴티지가 0) → **0.0** 을 돌려준다.
      왜 0 인가(명시적 결정):
        · 시퀀스 GRPO 에서 그룹이 전원 정답이거나 전원 오답이면 correctness 어드밴티지가
          정확히 0 이다. 즉 "이 롤아웃이 좋았나"에 대한 방향이 **없다**. 방향이 없는데
          부호를 임의로 +1 로 두면 그 항은 부호 없는 위치보너스가 되고, 그러면 C 는
          그 그룹에서 **B 와 같은 팔이 된다** — 팔 정체가 데이터에 따라 흔들린다.
          곱셈 팔의 요점은 "정답 방향일 때만 메타에 신용을 준다"이므로 방향 없음 = 신용 없음.
        · −1 로 두는 것은 더 나쁘다(전원 정답 그룹에서 메타를 벌한다).
      ⚠**이 결정의 대가는 반드시 재야 한다**: Countdown 오라클 판에서 p̂=0 이 63.7%,
        p̂=1 이 1.0% 로 **그룹의 약 64.8% 에서 correctness 기울기가 정확히 0** 이다.
        따라서 C·F·H 의 meta 항은 그 그룹들에서 통째로 침묵한다. 텔레메트리
        `frac_sign_zero` 와 성분별 평균(R_meta)이 이것을 드러내며, R_meta 평균이 0 이면
        그 팔은 무효 레버다. 이것은 버그가 아니라 **곱셈 설계의 실제 성질**이고,
        E(게이트)가 바로 그 64.8% 를 겨냥한다는 점에서 F 의 존재 이유이기도 하다.
    """
    s = sign_of(adv_corr_sign)
    if s == 0.0:
        return SIGN_ZERO_VALUE
    base = r_meta_pos(pmi_open, pmi_close,
                      params=SHIFT_PARAMS_MUL if params is None else params)
    out = base * s
    return out if _finite(out) else 0.0


def r_meta_ctx(pmi_self, pmi_donor, adv_corr_sign) -> float:
    r"""D 팔. clip(문맥대조, ±2) × sign(A_corr).

    문맥대조 = PMI(내 메타) − PMI(같은 문제 다른 롤아웃의 메타).
    ★clip 은 `dcpo_rmeta_forms.FORMULAS["ans_clip"]` 를 **그대로 호출**해 얻는다.
      그 파일이 오프라인 스윕(공식 고르기)의 정의처이므로, 여기서 다시 clip 을 쓰면
      오프라인에서 고른 식과 온라인에서 도는 식이 갈린다. 호출로 묶어 둔다.
    sign == 0 처리는 `r_meta_mul` 과 **같은 규칙**(0.0). 두 곱셈 팔이 다른 규칙을 쓰면
    D 와 C 의 차이가 "계산방식"이 아니라 "동점 처리"가 되어 버린다.
    """
    s = sign_of(adv_corr_sign)
    if s == 0.0:
        return SIGN_ZERO_VALUE
    if not (_finite(pmi_self) and _finite(pmi_donor)):
        return 0.0
    base = float(_RMETA_FORMULAS["ans_clip"]({
        "pmi_ans_real": float(pmi_self),
        "pmi_ans_donor": float(pmi_donor),
    }))
    out = base * s
    return out if _finite(out) else 0.0


def r_len(meta_token_count) -> float:
    r"""G 팔 — **가짜 대조군**. meta_len/100.

    학습에 쓰라고 있는 게 아니다. G 가 B~F 중 하나를 이기면 그 위의 전부가 길이의 대리일
    뿐이라는 뜻이고, **여덟 팔 전체를 폐기**한다(사양의 중단 조건).
    """
    if not _finite(meta_token_count):
        return 0.0
    return float(meta_token_count) / 100.0


def r_meta_floor(emitted) -> float:
    """공통 바닥값. 메타를 냈다는 사실 자체에 붙는 상수(처치 아님)."""
    return 1.0 if _bool01(emitted) else 0.0


def warmup_scale(step, warmup_steps: int = 20) -> float:
    """0→1 선형 워밍업. step 0 에서 0.0, step ≥ warmup_steps 에서 1.0.

    warmup_steps ≤ 0 이면 워밍업 없음(1.0). step < 0 은 0.0 으로 클램프.
    """
    try:
        w = int(warmup_steps)
    except (TypeError, ValueError):
        raise ValueError(f"warmup_scale: warmup_steps={warmup_steps!r} 가 정수가 아니다.")
    if w <= 0:
        return 1.0
    if not _finite(step):
        raise ValueError(f"warmup_scale: step={step!r} 가 유한수가 아니다.")
    s = float(step)
    if s <= 0:
        return 0.0
    if s >= w:
        return 1.0
    return s / float(w)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 팔 정체 — ★G8(팔 diff)의 유일한 근거. 다른 어느 파일에도 복사하지 마라.
# ══════════════════════════════════════════════════════════════════════════════

# 항 등록부: 각 항이 무엇을 필요로 하고, 워밍업을 받는가.
#   needs  : 그 항이 켜졌을 때 **반드시** 행에 있어야 하는 키(없으면 즉사 = 무효 레버 방지)
#   warmup : w_meta·w_gate 계열인가 (사양: step 0→20 선형, 이후 상시)
#   weight : 기본 무게
TERMS: dict[str, dict] = {
    "corr":       {"needs": ("r_corr",),                                   "warmup": False, "weight": W_CORR},
    "format":     {"needs": ("format_ok",),                                "warmup": False, "weight": W_FORMAT},
    # meta_floor 는 워밍업을 **안 받는다**(명시적 결정): 발화 침식을 막는 바닥값인데
    # 워밍업을 받으면 바닥이 도착하기 전에 발화가 무너질 수 있다. 사양은 warmup 대상으로
    # w_meta·w_gate 만 적었고 meta_floor 는 "공통(처치 아님)" 쪽에 있다.
    "meta_floor": {"needs": ("emitted",),                                  "warmup": False, "weight": META_FLOOR},
    "meta_pos":   {"needs": ("emitted", "pmi_open", "pmi_close"),          "warmup": True,  "weight": W_META},
    "meta_mul":   {"needs": ("emitted", "pmi_open", "pmi_close", "adv_corr"), "warmup": True, "weight": W_META},
    "meta_ctx":   {"needs": ("emitted", "pmi_self", "pmi_donor", "adv_corr"), "warmup": True, "weight": W_META},
    "gate":       {"needs": ("emitted",),                                  "warmup": True,  "weight": W_GATE},
    "len":        {"needs": ("emitted", "meta_n_tok"),                     "warmup": True,  "weight": W_LEN},
}

_COMMON = ("corr", "format", "meta_floor")   # 공통 = 처치 아님. 여덟 팔 전부 동일.

ARM_SPECS: dict[str, dict] = {
    "A": {"label": "corr",   "terms": _COMMON,                          "meta_form": "new",
          "note": "R_corr 만. 메타 항 없음 — 대조군."},
    "B": {"label": "cur",    "terms": _COMMON + ("meta_pos",),          "meta_form": "new",
          "note": "현행 재현. clip(shift,±2) + reversal(save 1.0/derail 2.0)."},
    "C": {"label": "mul",    "terms": _COMMON + ("meta_mul",),          "meta_form": "new",
          "note": "★곱하기. B 의 항 × sign(A_corr). sign=0 이면 0(전원 동점 그룹은 침묵)."},
    "D": {"label": "ctx",    "terms": _COMMON + ("meta_ctx",),          "meta_form": "new",
          "note": "★계산방식. 문맥대조(내 메타 − 도너 메타) × sign(A_corr)."},
    "E": {"label": "gate",   "terms": _COMMON + ("gate",),              "meta_form": "new",
          "note": "★게이팅. −(2p̂−1)×1{발화}. p̂ 는 gold 없이 그룹 자가검증률."},
    "F": {"label": "full",   "terms": _COMMON + ("meta_mul", "gate"),   "meta_form": "new",
          "note": "C + E."},
    "G": {"label": "neg",    "terms": _COMMON + ("len",),               "meta_form": "new",
          "note": "가짜 대조군. meta_len/100. 이것이 B~F 를 이기면 전부 폐기."},
    "H": {"label": "oldfmt", "terms": _COMMON + ("meta_mul", "gate"),   "meta_form": "old",
          "note": "F 와 **같은 보상**, 옛 한 줄 메타 형식. 형식의 값어치를 잰다."},
}


def arm_signature(arm: str) -> str:
    """팔 하나의 **정체 서명** — 이 문자열이 다르면 다른 팔이다.

    ★G8(팔별 diff)의 근거. 런처·로그·분석이 전부 이 문자열을 찍으면, 어떤 팔이
    실제로 무엇을 켜고 돌았는지가 사후에 한 줄로 확인된다("선언은 있고 배선은 없음" 방지).
    """
    spec = _require_arm(arm)
    parts = []
    for t in spec["terms"]:
        cfg = TERMS[t]
        parts.append(f"{t}@{cfg['weight']:g}{'*w' if cfg['warmup'] else ''}")
    extra = ""
    if "meta_pos" in spec["terms"]:
        extra += "|shift=" + _params_sig(SHIFT_PARAMS)
    if "meta_mul" in spec["terms"]:
        extra += "|shiftmul=" + _params_sig(SHIFT_PARAMS_MUL)
    if "meta_ctx" in spec["terms"]:
        extra += f"|ctxclip={CTX_CLIP:g}"
    if {"meta_mul", "meta_ctx"} & set(spec["terms"]):
        extra += f"|sign0={SIGN_ZERO_VALUE:g}"
    return (f"{SPEC_VERSION}|{arm}={spec['label']}|form={spec['meta_form']}"
            f"|{'+'.join(parts)}{extra}")


def _params_sig(p: Mapping) -> str:
    return ",".join(f"{k}={float(v):g}" for k, v in sorted(p.items()))


def all_arm_signatures() -> dict[str, str]:
    return {a: arm_signature(a) for a in sorted(ARM_SPECS)}


def _require_arm(arm: str) -> dict:
    if arm not in ARM_SPECS:
        raise ValueError(f"팔 {arm!r} 는 없다. 가능: {sorted(ARM_SPECS)}")
    return ARM_SPECS[arm]


# ══════════════════════════════════════════════════════════════════════════════
# 6. 조립 — 팔 하나의 총보상. 항이 합쳐지는 자리는 여기 하나뿐이다.
# ══════════════════════════════════════════════════════════════════════════════

def arm_reward(
    arm: str,
    row: Mapping,
    *,
    step: int,
    warmup_steps: int = 20,
    phat: float | None = None,
) -> tuple[float, dict[str, float]]:
    r"""팔 `arm` 의 총보상과 **성분별 기여**를 돌려준다.

    Returns (total, components) — components 는 무게·워밍업이 **이미 곱해진** 실지급액이라
    그대로 평균 내면 사양의 "성분별 보상 평균"이 된다(하나가 0 이면 무효 레버).

    켜진 항의 원재료가 행에 없으면 **KeyError 로 즉사**한다. 조용히 0 을 흘리면 그 팔은
    선언만 있고 배선이 없는 팔이 되고, 우리는 그것으로 실험을 여러 번 날렸다.
    `gate` 항이 켜졌는데 `phat` 이 None 이면 마찬가지로 즉사한다.
    """
    spec = _require_arm(arm)
    terms = spec["terms"]
    scale = warmup_scale(step, warmup_steps)

    # 원재료 사전검사 — 계산 전에 통째로 확인한다(부분 계산 후 죽으면 로그가 헷갈린다).
    missing = []
    for t in terms:
        for k in TERMS[t]["needs"]:
            if k not in row:
                missing.append(f"{t}:{k}")
    if missing:
        raise KeyError(
            f"arm_reward({arm}): 켜진 항의 원재료 누락 {missing}. "
            f"조용히 0 을 흘리지 않는다 — 무효 레버 방지.")
    if "gate" in terms and phat is None:
        raise ValueError(f"arm_reward({arm}): gate 항이 켜졌는데 phat=None 이다.")

    emitted = _bool01(row.get("emitted", 0))
    raw: dict[str, float] = {}

    if "corr" in terms:
        raw["corr"] = 1.0 if _bool01(row["r_corr"]) else 0.0
    if "format" in terms:
        raw["format"] = 1.0 if _bool01(row["format_ok"]) else 0.0
    if "meta_floor" in terms:
        raw["meta_floor"] = r_meta_floor(emitted)
    if "meta_pos" in terms:
        raw["meta_pos"] = r_meta_pos(row["pmi_open"], row["pmi_close"]) if emitted else 0.0
    if "meta_mul" in terms:
        raw["meta_mul"] = (r_meta_mul(row["pmi_open"], row["pmi_close"], row["adv_corr"])
                           if emitted else 0.0)
    if "meta_ctx" in terms:
        raw["meta_ctx"] = (r_meta_ctx(row["pmi_self"], row["pmi_donor"], row["adv_corr"])
                           if emitted else 0.0)
    if "gate" in terms:
        raw["gate"] = r_gate(emitted, phat)
    if "len" in terms:
        raw["len"] = r_len(row["meta_n_tok"]) if emitted else 0.0

    comps: dict[str, float] = {}
    for t, v in raw.items():
        w = float(TERMS[t]["weight"])
        s = scale if TERMS[t]["warmup"] else 1.0
        comps[t] = _f(w * s * float(v))
    total = _f(sum(comps.values()))
    return total, comps


# ══════════════════════════════════════════════════════════════════════════════
# 7. 메타 파싱 — 두 형식. 왜 기존 파서를 안 쓰나는 아래 주석 참조.
# ══════════════════════════════════════════════════════════════════════════════
#
# `rewards.py::_parse_meta_blocks_with_spans` / `_parse_confidence` 를 검토했고 **쓰지
# 않았다**. 이유 셋(추측 아님, 그 코드를 읽고 확인한 것):
#   1. 태그 어휘가 다르다 — 그 파서는 `<|meta|>`(특수토큰) / `[META]` 만 안다.
#      이 사양은 평문 `<meta>…</meta>` 와 옛 한 줄 형식이다.
#   2. `_parse_confidence` 가 신뢰도를 **[0.01, 0.99] 로 클램프**한다. 사양의 텔레메트리
#      "confidence 고유값 수"는 0 과 1 을 구별해야 하므로 그 클램프가 지표를 파괴한다.
#   3. free-text fallback 이 본문의 "probability/확률" 같은 단어까지 메타로 센다 →
#      emit_rate 가 부풀고, emit_rate < 0.2 중단 조건이 무력해진다.
# 즉 재사용이 아니라 **오염**이 될 자리라 새 파서를 썼다. (shift·clip 처럼 식이 같은
# 자리는 전부 재사용했다.)

_META_BLOCK = re.compile(r"<meta>(.*?)</meta>", re.IGNORECASE | re.DOTALL)
_CONF = re.compile(r"confidence\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_DECISION = re.compile(r"decision\s*:\s*(verify|redirect)", re.IGNORECASE)
_OLD_LINE = re.compile(
    r"confidence\s*:\s*([0-9]*\.?[0-9]+)\s*\|(.*?)\|\s*decision\s*:\s*(verify|redirect)",
    re.IGNORECASE)
# "산수 금지" 위반 탐지 — 메타 안의 `a op b = c`.
_ARITH = re.compile(r"\d+\s*[+\-*/]\s*\d+\s*=\s*\d+")


def parse_meta(text: str, form: str = "new") -> dict:
    r"""응답 텍스트에서 메타 블록 **하나**(첫 번째)를 뽑는다.

    form:
      "new" — 사양의 A~G 형식.  <meta>\nconfidence: x\n…\ndecision: verify\n</meta>
      "old" — H 의 옛 형식.      confidence: 0.6 | 한 문장 | decision: verify
      "any" — new 를 먼저 시도하고 없으면 old.

    Returns dict:
      emitted    1 = 그 형식의 **완결된** 메타가 있다(신뢰도와 decision 을 둘 다 가진다)
      form       실제로 맞은 형식 ("new"/"old"/None)
      start,end  응답 텍스트 안의 문자 오프셋 (meta_position 계산용)
      body       신뢰도·decision 줄을 뺀 **평가 문장**
      raw        메타 블록 전체 텍스트
      confidence float 또는 None (★클램프하지 않는다 — 0 과 1 을 구별해야 한다)
      decision   "verify"/"redirect"/None
      n_blocks   그 형식의 블록이 몇 개 나왔나 (2 이상이면 형식 위반 신호)
    """
    if form not in ("new", "old", "any"):
        raise ValueError(f"parse_meta: form={form!r} 는 없다 (new/old/any).")
    t = text or ""
    if form in ("new", "any"):
        got = _parse_new(t)
        # "any" 에서도 <meta> 블록이 **있기만 하면** 그 결과를 돌려준다(불완전해도).
        # 그래야 "새 형식을 시도했는데 decision 을 빠뜨렸다"가 옛 형식 미발화로 뭉개지지
        # 않고 emitted=0 · n_blocks=1 로 보인다 — 형식 붕괴의 진단이 남는다.
        if form == "new" or got["n_blocks"] > 0:
            return got
    return _parse_old(t)


def _empty_meta() -> dict:
    return {"emitted": 0, "form": None, "start": None, "end": None, "body": "",
            "raw": "", "confidence": None, "decision": None, "n_blocks": 0}


def _parse_new(text: str) -> dict:
    ms = list(_META_BLOCK.finditer(text))
    out = _empty_meta()
    out["n_blocks"] = len(ms)
    if not ms:
        return out
    m = ms[0]
    inner = m.group(1)
    conf = _CONF.search(inner)
    dec = _DECISION.search(inner)
    body_lines = [ln.strip() for ln in inner.splitlines()
                  if ln.strip()
                  and not _CONF.match(ln.strip())
                  and not _DECISION.match(ln.strip())]
    out.update({
        "emitted": 1 if (conf is not None and dec is not None) else 0,
        "form": "new",
        "start": m.start(), "end": m.end(),
        "body": " ".join(body_lines),
        "raw": m.group(0),
        "confidence": float(conf.group(1)) if conf else None,
        "decision": dec.group(1).lower() if dec else None,
    })
    return out


def _parse_old(text: str) -> dict:
    ms = list(_OLD_LINE.finditer(text))
    out = _empty_meta()
    out["n_blocks"] = len(ms)
    if not ms:
        return out
    m = ms[0]
    out.update({
        "emitted": 1, "form": "old", "start": m.start(), "end": m.end(),
        "body": m.group(2).strip(), "raw": m.group(0),
        "confidence": float(m.group(1)), "decision": m.group(3).lower(),
    })
    return out


def meta_form_ok(text: str, form: str) -> int:
    """그 팔의 형식을 **정확히 하나** 지켰나. 0/1.

    블록이 둘 이상이면 0 이다 — 사양은 메타 하나를 내고 그 결정에 맞게 추론을 잇는
    구조이고, 여러 개면 meta_position·정형문 같은 지표가 정의되지 않는다.
    """
    p = parse_meta(text, form=form)
    return 1 if (p["emitted"] and p["n_blocks"] == 1) else 0


def format_ok_row(text: str, arm: str, *, parse_expr_ok: Callable[[str], int]) -> int:
    r"""사양의 `w_format` 이 붙는 형식 준수 = 식 형식 ∧ 메타 형식.

    `parse_expr_ok` 는 **주입**한다 — Countdown 의 식 파서(`countdown.parse_ok`)는
    검증된 코드이므로 여기서 복제하지 않는다. 예:
        from countdown import parse_ok
        format_ok_row(text, "F", parse_expr_ok=parse_ok)

    ⚠H 는 이 항에서도 F 와 달라진다(옛 형식을 지켜야 1 을 받는다). 그것이 "형식의
    값어치"를 재는 팔의 설계 의도다 — 보상식은 F 와 같고 형식만 다르다.
    """
    spec = _require_arm(arm)
    return 1 if (_bool01(parse_expr_ok(text)) and meta_form_ok(text, spec["meta_form"])) else 0


# ══════════════════════════════════════════════════════════════════════════════
# 8. 텔레메트리 — 사양의 지표 전부. 전부 순수 함수.
# ══════════════════════════════════════════════════════════════════════════════
#
# 행 규약(텔레메트리용): 각 행은 최소한 다음을 가진다.
#   emitted, meta(=parse_meta 결과 dict, 없으면 여기서 계산), text, phat, group_id
# 그룹 규약: rows 를 group_id 로 묶은 list[list[row]] 또는 dict[gid, rows].

def _meta_of(row: Mapping, form: str = "any") -> dict:
    m = row.get("meta")
    if isinstance(m, Mapping):
        return dict(m)
    return parse_meta(row.get("text", ""), form=form)


def emit_rate(rows: Sequence[Mapping], *, form: str = "any") -> float:
    """발화율. **< 0.2 면 중단**."""
    rows = list(rows)
    if not rows:
        return 0.0
    return _mean([_bool01(_meta_of(r, form)["emitted"]) for r in rows])


def meta_position_frac(row: Mapping, *, form: str = "any") -> float | None:
    r"""★메타가 응답의 몇 % 지점에 나오나 (0.0 = 맨 앞, 1.0 = 맨 뒤).

    문자 오프셋 기준: meta_start / len(응답). 토큰 오프셋이 있으면
    row["meta_start_tok"]/row["n_tok"] 를 우선 쓴다(트레이너 쪽이 더 정확하다).
    전례: rq3 에서 온폴리시 97.2% 가 meta-first 였다 — 즉 "메타 뒤에 추론이 온다"가
    아니라 "메타가 첫 줄이고 그 뒤가 전부"였다. 이 지표가 없으면 그 붕괴가 안 보인다.
    발화가 없으면 None.
    """
    if row.get("meta_start_tok") is not None and row.get("n_tok"):
        n = float(row["n_tok"])
        return float(row["meta_start_tok"]) / n if n > 0 else None
    m = _meta_of(row, form)
    if not m["emitted"] or m["start"] is None:
        return None
    text = row.get("text", "")
    return (m["start"] / len(text)) if len(text) > 0 else None


def meta_position_stats(rows: Sequence[Mapping], *, form: str = "any",
                        first_frac: float = 0.1) -> dict:
    """meta_position 의 분포 + **meta-first 비율**(위치 ≤ first_frac)."""
    vals = [v for v in (meta_position_frac(r, form=form) for r in rows) if v is not None]
    if not vals:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"),
                "p90": float("nan"), "frac_meta_first": float("nan")}
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "p50": _quantile(vals, 0.5),
        "p90": _quantile(vals, 0.9),
        "frac_meta_first": _mean([1.0 if v <= first_frac else 0.0 for v in vals]),
    }


def selectivity_index(rows: Sequence[Mapping], *, form: str = "any",
                      lo: float = 0.25, hi: float = 0.75) -> dict:
    r"""★선택성 지수 = emit(p̂ < lo) − emit(p̂ > hi).

    양수 = 어려운 문제에서 더 낸다(우리가 원하는 것). 0 = 무차별 발화.
    각 행에 그 행이 속한 그룹의 p̂ 가 `phat` 키로 있어야 한다.
    한쪽 표본이 비면 그 항은 NaN — **0 으로 채우지 않는다**(0 은 "무차별"로 오독된다).
    """
    a = [_bool01(_meta_of(r, form)["emitted"]) for r in rows if _finite(r.get("phat")) and float(r["phat"]) < lo]
    b = [_bool01(_meta_of(r, form)["emitted"]) for r in rows if _finite(r.get("phat")) and float(r["phat"]) > hi]
    ea = _mean(a) if a else float("nan")
    eb = _mean(b) if b else float("nan")
    idx = (ea - eb) if (a and b) else float("nan")
    return {"selectivity": idx, "emit_hard": ea, "emit_easy": eb,
            "n_hard": len(a), "n_easy": len(b)}


def _norm_body(s: str) -> str:
    """정형문 비교용 정규화: 소문자 + 공백 축약 + 숫자 제거.

    숫자를 지우는 이유 — 정형문은 신뢰도 숫자만 흔들리고 문장은 같다(b4p2 전례).
    숫자를 남기면 같은 정형문이 서로 다른 문장으로 세어져 정형문 비율이 낮게 나온다.
    ⚠부작용: 이 문제의 수를 언급하는 **좋은** 메타도 숫자가 지워진다. 그래서 정형문
    판정은 "숫자를 지워도 같은 문장"이라는 보수적 정의이고, 이 정의로도 0.5 를 넘으면
    붕괴다. (문제별 수를 쓰는 메타는 문장 자체가 달라 보통 안 뭉친다.)
    """
    s = re.sub(r"[0-9]+", "", (s or "").lower())
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;|")


def boilerplate_rate(rows: Sequence[Mapping], *, form: str = "any", top_k: int = 5) -> dict:
    """정형문 비율 = **최빈 메타 문장**의 점유율(발화한 행 중). > 0.5 면 중단."""
    bodies = [_norm_body(_meta_of(r, form)["body"]) for r in rows
              if _bool01(_meta_of(r, form)["emitted"])]
    bodies = [b for b in bodies if b]
    if not bodies:
        return {"boilerplate_rate": float("nan"), "n_emitted": 0,
                "n_unique": 0, "top": [], "topk_share": float("nan")}
    c = Counter(bodies)
    top = c.most_common(top_k)
    return {
        "boilerplate_rate": top[0][1] / len(bodies),
        "n_emitted": len(bodies),
        "n_unique": len(c),
        "top": [(b[:80], n / len(bodies)) for b, n in top],
        "topk_share": sum(n for _, n in top) / len(bodies),
    }


def _norm_expr(e: str) -> str:
    """식 비교용 정규화: 공백·LaTeX 장식 제거."""
    s = (e or "")
    for a, b in (("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"),
                 ("\\left", ""), ("\\right", ""), ("\\boxed", ""), ("$", "")):
        s = s.replace(a, b)
    return re.sub(r"[\s{}]", "", s)


def answer_leak(meta_text: str, final_expr: str | None, *,
                extra_grader: Callable[[str], int] | None = None) -> int:
    r"""★답 누출 — 메타에 **최종 식이 그대로** 들어 있나. 0/1.

    사양의 정의를 문자 그대로 잰다("메타에 최종 식이 그대로 든 비율"): 정규화한 최종 식이
    정규화한 메타 안에 부분문자열로 있으면 1. 바깥 괄호만 다른 경우도 잡는다.
    `final_expr` 이 None 이면 **예외** — 조용히 0 을 돌려주면 누출률 0.1 중단 조건이
    무력해진다(그 지표는 "안 쟀다"가 아니라 "없다"로 읽힌다).
    `extra_grader` 로 채점기를 넣으면(예: lambda s: grade(s, nums, target)) 메타 자체가
    정답 식을 담은 경우도 함께 잡는다 — 이 저장소의 `countdown.grade` 를 복제하지 않기
    위한 주입 지점이다.
    """
    if final_expr is None:
        raise ValueError("answer_leak: final_expr 이 None 이다 — 누출률을 못 잰다. "
                         "식 추출기를 돌려 값을 넣어라(조용히 0 을 흘리지 않는다).")
    m = _norm_expr(meta_text)
    e = _norm_expr(final_expr)
    if e and e in m:
        return 1
    if len(e) > 2 and e[0] == "(" and e[-1] == ")" and e[1:-1] in m:
        return 1
    if extra_grader is not None and _bool01(extra_grader(meta_text or "")):
        return 1
    return 0


def answer_leak_rate(rows: Sequence[Mapping], *, form: str = "any",
                     expr_key: str = "final_expr",
                     extra_grader: Callable[[Mapping, str], int] | None = None) -> float:
    """답 누출률(발화한 행 중). **> 0.1 이면 중단**."""
    vals = []
    for r in rows:
        m = _meta_of(r, form)
        if not _bool01(m["emitted"]):
            continue
        if expr_key not in r:
            raise KeyError(f"answer_leak_rate: 행에 {expr_key!r} 가 없다 — 최종 식을 넣어라.")
        g = (lambda s, _r=r: extra_grader(_r, s)) if extra_grader else None
        vals.append(answer_leak(m["raw"], r[expr_key], extra_grader=g))
    return _mean(vals) if vals else float("nan")


def meta_has_arithmetic(meta_text: str) -> int:
    """메타 안에 `a op b = c` 산수가 있나. 사양의 '★산수 금지' 위반 탐지."""
    return 1 if _ARITH.search(meta_text or "") else 0


def arithmetic_in_meta_rate(rows: Sequence[Mapping], *, form: str = "any") -> float:
    vals = [meta_has_arithmetic(_meta_of(r, form)["raw"]) for r in rows
            if _bool01(_meta_of(r, form)["emitted"])]
    return _mean(vals) if vals else float("nan")


def _as_groups(groups) -> list[list[Mapping]]:
    if isinstance(groups, Mapping):
        return [list(v) for v in groups.values()]
    return [list(g) for g in groups]


def group_emit_dispersion(groups, *, form: str = "any") -> dict:
    """그룹 내 발화 분산 — 전원발화/전원침묵 그룹의 비율.

    둘의 합이 1 에 가까우면 발화가 **문제 단위로만** 갈린다는 뜻이고, 그러면 그룹 안에서
    '냈다 vs 안 냈다'를 비교하는 어떤 어드밴티지도 만들어지지 않는다(게이트 팔이 조용히
    무효가 되는 경로).
    """
    gs = _as_groups(groups)
    if not gs:
        return {"frac_all_emit": float("nan"), "frac_all_silent": float("nan"),
                "frac_mixed": float("nan"), "mean_within_var": float("nan"), "n_groups": 0}
    all_e = all_s = 0
    varis = []
    for g in gs:
        es = [_bool01(_meta_of(r, form)["emitted"]) for r in g]
        if not es:
            continue
        mu = _mean(es)
        varis.append(_mean([(e - mu) ** 2 for e in es]))
        if mu == 1.0:
            all_e += 1
        elif mu == 0.0:
            all_s += 1
    n = len(gs)
    return {"frac_all_emit": all_e / n, "frac_all_silent": all_s / n,
            "frac_mixed": 1.0 - (all_e + all_s) / n,
            "mean_within_var": _mean(varis) if varis else float("nan"), "n_groups": n}


def phat_distribution(groups, *, solved_key: str = "r_corr",
                      grader: Callable[[Mapping], int] | None = None) -> dict:
    r"""p̂ 분포 — 게이트 표류 감시. frac_zero/frac_one 이 핵심이다.

    오라클 판 기준선: p̂=0 이 63.7% · p̂=1 이 1.0%.  ★p̂=0 또는 1 인 그룹에서는
    correctness 어드밴티지가 정확히 0 이므로 곱셈 팔(C·F·H)의 메타 항이 침묵한다.
    `frac_sign_zero` 가 바로 그 비율이며, r_meta_mul 의 sign=0 결정의 대가를 잰다.
    """
    gs = _as_groups(groups)
    ps = [compute_phat(g, solved_key=solved_key, grader=grader) for g in gs if g]
    if not ps:
        return {"n_groups": 0}
    return {
        "n_groups": len(ps),
        "mean": _mean(ps),
        "frac_zero": _mean([1.0 if p == 0.0 else 0.0 for p in ps]),
        "frac_one": _mean([1.0 if p == 1.0 else 0.0 for p in ps]),
        "frac_sign_zero": _mean([1.0 if p in (0.0, 1.0) else 0.0 for p in ps]),
        "p10": _quantile(ps, 0.1), "p50": _quantile(ps, 0.5), "p90": _quantile(ps, 0.9),
    }


def decision_distribution(rows: Sequence[Mapping], *, form: str = "any") -> dict:
    """decision 분포 — verify/redirect 비율(발화한 행 중)."""
    ds = [_meta_of(r, form)["decision"] for r in rows if _bool01(_meta_of(r, form)["emitted"])]
    ds = [d for d in ds if d]
    if not ds:
        return {"verify": float("nan"), "redirect": float("nan"), "n": 0}
    c = Counter(ds)
    return {"verify": c.get("verify", 0) / len(ds),
            "redirect": c.get("redirect", 0) / len(ds), "n": len(ds)}


def confidence_unique(rows: Sequence[Mapping], *, form: str = "any", ndigits: int = 6) -> dict:
    """confidence 고유값 수 — 지금 2 개. 형식 변경으로 느나.

    ★클램프하지 않은 원값을 센다. (기존 `rewards._parse_confidence` 는 [0.01,0.99] 로
    클램프해서 0 과 1 을 구별 못 한다 — 그래서 그 파서를 안 썼다.)
    """
    vs = [_meta_of(r, form)["confidence"] for r in rows]
    vs = [round(float(v), ndigits) for v in vs if v is not None]
    c = Counter(vs)
    return {"n_unique": len(c), "n": len(vs),
            "values": sorted(c), "top": c.most_common(5)}


def component_means(components: Sequence[Mapping[str, float]], *, dead_eps: float = 1e-12) -> dict:
    """성분별 보상 평균 + **죽은 성분** 목록.

    사양: "하나가 0 이면 무효 레버". 평균이 0 인 것뿐 아니라 **분산이 0**인 것도 잡는다 —
    모든 행에 같은 상수를 주는 항은 GRPO 에서 센터링으로 사라져 학습에 아무 영향이 없다
    (평균은 0 이 아닌데 레버는 무효인 경우).
    """
    keys = sorted({k for c in components for k in c})
    out = {"means": {}, "std": {}, "dead": [], "n": len(components)}
    for k in keys:
        vs = [float(c.get(k, 0.0)) for c in components]
        mu = _mean(vs)
        var = _mean([(v - mu) ** 2 for v in vs])
        out["means"][k] = mu
        out["std"][k] = math.sqrt(max(0.0, var))
        if abs(mu) <= dead_eps or var <= dead_eps:
            out["dead"].append(k)
    return out


def shift_diag(rows: Sequence[Mapping], *, params: Mapping | None = None) -> dict:
    """n_save / n_derail — `dcpo_pmi_shift.compute_pmi_shift_reward` 를 그대로 쓴다.

    직접 세지 않는 이유: 세는 기준(eps 게이트)이 보상이 실제로 지급하는 기준과 갈리면
    "선언된 판정식 ≠ 채택된 수치"가 그대로 재발한다. 그 파일이 둘을 같이 계산한다.
    """
    p = dict(SHIFT_PARAMS if params is None else params)
    _, diag = _compute_pmi_shift_reward(list(rows), **p)
    n = max(1, len(list(rows)))
    return {"n_save": diag["n_save"], "n_derail": diag["n_derail"],
            "save_rate": diag["n_save"] / n, "derail_rate": diag["n_derail"] / n,
            "n_fail": sum(1 for f in diag["failures"] if f)}


def length_stats(rows: Sequence[Mapping], *, len_key: str = "n_tok",
                 trunc_key: str = "truncated") -> dict:
    """응답 길이 p95 · 절단률."""
    ls = [float(r[len_key]) for r in rows if _finite(r.get(len_key))]
    tr = [_bool01(r.get(trunc_key, 0)) for r in rows]
    return {"len_mean": _mean(ls) if ls else float("nan"),
            "len_p50": _quantile(ls, 0.5) if ls else float("nan"),
            "len_p95": _quantile(ls, 0.95) if ls else float("nan"),
            "trunc_rate": _mean(tr) if tr else float("nan")}


def telemetry_report(groups, *, form: str = "any", components=None,
                     solved_key: str = "r_corr",
                     expr_key: str = "final_expr",
                     compute_leak: bool = True) -> dict:
    """사양의 텔레메트리 **전부**를 한 번에. 로거는 이 dict 하나만 찍으면 된다.

    한 곳에 모으는 이유: 지표가 호출처마다 흩어지면 "쟀다고 선언했는데 안 잰" 칸이 생긴다.
    entropy·KL 은 트레이너만 아는 값이라 여기서 계산하지 않는다 — 호출자가 결과 dict 에
    합치고, 아래 `check_abort` 는 그 칸이 없으면 없다고 말한다(조용히 통과시키지 않는다).
    """
    gs = _as_groups(groups)
    rows = [r for g in gs for r in g]
    rep = {
        "n_rows": len(rows), "n_groups": len(gs),
        "emit_rate": emit_rate(rows, form=form),
        "meta_position": meta_position_stats(rows, form=form),
        "selectivity": selectivity_index(rows, form=form),
        "boilerplate": boilerplate_rate(rows, form=form),
        "arith_in_meta_rate": arithmetic_in_meta_rate(rows, form=form),
        "group_emit": group_emit_dispersion(gs, form=form),
        "phat": phat_distribution(gs, solved_key=solved_key),
        "decision": decision_distribution(rows, form=form),
        "confidence": confidence_unique(rows, form=form),
        "length": length_stats(rows),
        "shift": shift_diag(rows),
        "acc": _mean([_bool01(r.get(solved_key, 0)) for r in rows]) if rows else float("nan"),
    }
    rep["answer_leak_rate"] = (answer_leak_rate(rows, form=form, expr_key=expr_key)
                              if compute_leak else float("nan"))
    if components is not None:
        rep["components"] = component_means(list(components))
    return rep


# ══════════════════════════════════════════════════════════════════════════════
# 9. 중단 조건 — 코드에 박는다(사양)
# ══════════════════════════════════════════════════════════════════════════════

ABORT_RULES = {
    "emit_rate":        {"op": "<", "thr": 0.2,
                         "why": "발화가 무너지면 어떤 메타 팔도 잴 것이 없다"},
    "boilerplate_rate": {"op": ">", "thr": 0.5,
                         "why": "최빈 문장이 과반이면 메타가 아니라 상투구다"},
    "answer_leak_rate": {"op": ">", "thr": 0.1,
                         "why": "메타가 답을 담으면 메타 보상이 정답 보상의 사본이 된다"},
}


def check_abort(report: Mapping) -> list[dict]:
    r"""사양의 중단 조건을 재고, **위반 목록**을 돌려준다(빈 리스트 = 통과).

    지표가 없거나 NaN 이면 `missing` 로 보고한다 — 조용히 통과시키지 않는다.
    "상시 WARN 은 소음이 아니다"(원장 0731): 4건 중 2건이 실질이었다.
    """
    out = []
    vals = {
        "emit_rate": report.get("emit_rate"),
        "boilerplate_rate": (report.get("boilerplate") or {}).get("boilerplate_rate"),
        "answer_leak_rate": report.get("answer_leak_rate"),
    }
    for name, rule in ABORT_RULES.items():
        v = vals.get(name)
        if v is None or not _finite(v):
            out.append({"metric": name, "status": "missing", "value": v,
                        "why": "지표가 없다 — 안 쟀다는 뜻이지 통과가 아니다"})
            continue
        v = float(v)
        bad = (v < rule["thr"]) if rule["op"] == "<" else (v > rule["thr"])
        if bad:
            out.append({"metric": name, "status": "abort", "value": v,
                        "threshold": rule["thr"], "op": rule["op"], "why": rule["why"]})
    return out


def check_negative_control(arm_scores: Mapping[str, float],
                           *, neg_arm: str = "G",
                           treatment_arms: Sequence[str] = LAUNCHED_TREATMENT_ARMS) -> dict:
    r"""★가짜 대조군 규칙: **G(길이)가 B~F 중 하나라도 이기면 전부 폐기.**

    `arm_scores` 는 팔 → 최종 지표(높을수록 좋음, 예: held-out 정답률). 비교 대상 팔이
    빠져 있으면 `incomplete` 로 보고한다 — 없는 팔을 이기지 못했다고 통과시키지 않는다.
    """
    missing = [a for a in (neg_arm, *treatment_arms) if a not in arm_scores]
    if missing:
        return {"status": "incomplete", "missing": missing,
                "why": "빠진 팔이 있다 — G 가 이겼는지 판정할 수 없다"}
    g = float(arm_scores[neg_arm])
    beaten = [a for a in treatment_arms if g >= float(arm_scores[a])]
    return {"status": "DISCARD_ALL" if beaten else "ok",
            "neg_score": g, "beaten": beaten,
            "why": ("길이 대조군이 처치 팔 이상이다 — 위의 전부가 길이의 대리다"
                    if beaten else "길이 대조군이 모든 처치 팔에 진다")}
