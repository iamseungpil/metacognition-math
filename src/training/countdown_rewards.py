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
import os
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
    "NORMALIZE_TERMS", "TERM_MAX_ABS",
    "arm_signature", "all_arm_signatures",
    # 보상
    "compute_phat", "compute_phat_loo", "sign_of",
    "r_gate", "r_meta_pos", "r_meta_mul", "r_meta_ctx", "r_len", "r_meta_floor",
    "r_osd", "plan_next", "plan_followed",
    "SC_K_STUCK", "SC_CONF_HI", "W_EXPLORE", "W_VERIFY", "W_EARLY",
    "r_explore", "r_explore_g", "r_verify", "r_early",
    "W_TIMING", "W_LIVE_NEW", "r_timing", "r_timing2", "r_live_new",
    "OPD_TERM", "OPD_C", "OPD_C_PROVISIONAL", "r_opd_meta",
    "OPD_TERM_C", "opd_center_rows", "r_opd_meta_c",
    "CF_TERM", "W_CF", "cf_center_rows", "r_cf_meta",
    "CF_GROUP_TERM", "cf_group_rows", "r_cf_group",
    "WHEN_TERM", "VTR_TAU", "VTR_WHEN_W", "resolved_vtr_tau", "resolved_vtr_when_w",
    "vtr_batch_gate", "r_when",
    "OSD_TERM", "OSD_C", "OSD_C_PROVISIONAL", "OSD_W_MAX", "OSD_LEAK_NGRAM",
    "INV_TERM", "INV_SCOPE", "INV_FORM", "INV_AGG", "INV_TAU", "INV_C",
    "INV_TAU_PROVISIONAL", "INV_MIN_PROSE_TOK", "INV_FALSE_CLAIM_PEN", "r_meta_inv",
    "warmup_scale", "arm_reward",
    # 파싱 / 형식
    "parse_meta", "meta_form_ok", "format_ok_row",
    # 텔레메트리
    "emit_rate", "meta_position_frac", "meta_position_stats", "selectivity_index",
    "boilerplate_rate", "answer_leak", "answer_leak_rate", "meta_has_arithmetic",
    "arithmetic_in_meta_rate", "false_claim_rate", "group_emit_dispersion", "phat_distribution",
    "decision_distribution", "confidence_unique", "component_means", "shift_diag",
    "META_TERMS", "rmeta_magnitude", "group_variance_decomposition",
    "meta_outcome_discrimination",
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

# ── SC(자기제어) / SCg 팔 — SC_DESIGN.md(2026-09-04) ────────────────────────────
# 왜 새 팔인가. B~H·OSD·R 은 전부 "메타가 정답과 정렬됐는가(=근거-진리 또는 PMI)"를
# 재는데, 그 잣대는 모델 자신의 **상태**(막혔는가·과신했는가)를 안 본다. SC 는 응답
# **프리픽스**(메타 이전 텍스트)에서 관측 가능한 상태만으로 두 조건부 습관을 겨냥한다:
#   ① 막히면(prefix 안 시도 수 ≥ K_S) 새 계열로 갈아타는가(decision=redirect ∧
#      다음 쌍이 프리픽스에 없던 새 결합 ∧ 실제로 그 쌍을 이었는가) → `explore`
#   ② 과신하면(confidence ≥ CONF_HI) 메타 뒤에 실제 재계산(참인 등식)을 보였는가,
#      그리고 verify 라 선언해놓고 안 쟀으면서 틀렸으면 벌 → `verify`
#   ③ 시작하자마자(prefix 시도 0) 메타부터 내는 것은 비용 → `early_cost`
# 근거-진리(완전 열거로 next 가 해를 살리는지)는 SC 에서 **안 쓴다** — SC 는 모델
# 자신의 신호만으로 충분한지를 보는 대조다. SCg 는 `explore` 에 `plan_ok`(근거-진리)를
# 곱해 "내부 신호만으로 충분한가"의 대조점을 만든다(SC_DESIGN.md §행 특징).
SC_K_STUCK = 4       # gs0 분포에서 사전 고정할 후보값 중 하나(설계 §행 특징). 발사 전
                     # 실측으로 교체될 수 있다 — 그때는 이 상수 한 줄만 바뀐다.
SC_CONF_HI = 0.8     # 과신 임계. INV_TAU 류와 달리 confidence 는 [0,1] 스칼라라
                     # 분위수 실측 없이도 "0.8 이상"이 사전등록된 임계로 고정 가능하다.
W_EXPLORE = 1.0      # SC_DESIGN.md: w_x = 1.0
W_VERIFY = 0.5       # SC_DESIGN.md: w_v = 0.5
W_EARLY = 0.25       # SC_DESIGN.md: w_e = 0.25

# ── 고정 자리(site-anchored) 팔 FT/M0/MT — 2026-09-04 「고정 자리 학습 배선」 ──────
# 왜 새 항인가. SC/SCg 는 프리픽스 **상태**(시도수·확신)만 보고 완전열거(오라클)를
# 안 쓴다. FT/MT 는 반대로 `countdown_sites` 의 오라클 라벨(family_dead·live_new_moves —
# "이 계열이 죽었는가", "이 첫수가 새로 산 살아있는 해로 이어지는가")을 직접 보상에
# 태운다 — SC 의 "모델 자기 신호만"과 대조되는 "근거-진리를 아는 고정 자리에서
# 재개했을 때" 조건이다. `family_dead`/`live_new_moves`/`first_move_after_meta` 는
# 이 파일이 계산하지 않는다 — site 행은 `countdown_sites.oracle_for_site`(빌드 시점,
# parquet 컬럼), normal 행은 `verl_sdc._compute_countdown_arm_stash`(런타임, 같은
# `countdown_sites` 함수를 응답 프리픽스에 호출)가 채운다. 여기는 그 결과만 읽는다.
W_TIMING = 0.5       # 사용자 지시. warmup 대상.
W_LIVE_NEW = 1.0     # 사용자 지시. warmup 대상.

# ── OSD (Outcome-Signed Surprisal Drop) — PMI-shift 의 대체 항 (2026-08-25) ──
# 왜 새 항인가. PMI-shift 는 오프라인 판별력 **AUC 0.52(=우연)** 로 측정돼 폐기됐다.
# OSD 는 "남의 오답 후보"를 끌어오지 않고 **자기 궤적만** 본다:
#   메타 구간(<meta>…</meta>)을 **그것만** 지웠을 때, 뒤따르는 응답 창 W 의 평균
#   logP 가 얼마나 떨어지는가 = Δcert. 두 문맥은 메타 유무만 다르고 W 는 동일하다.
#   샘플링 없음 — teacher-forced forward 두 번뿐이다.
# 이 파일의 역할은 그 Δcert 를 **받아서 정답 부호를 붙이고 정규화**하는 것뿐이다.
# Δcert 를 실제로 계산하는 자리(토큰 인덱스·forward)는 `countdown_pmi` 쪽이다.
OSD_TERM = "osd"     # ★항 이름의 **단일 정의처**. verl_sdc 의 fail-loud 가드가 이것을
                      #   읽는다. 문자열을 두 곳에 두면 갈리고, 갈린 순간 "선언된 레버,
                      #   배선 0" 이 조용히 성립한다(0825 적대검증에서 실제로 잡혔다:
                      #   가드가 "meta_osd" 를 보는데 항은 "osd" 라 OSD 팔이 A 팔과
                      #   비트 동일한 보상을 냈다).
OSD_C = 0.366        # ★학습 조건 실측 p90 의 중앙값(0826 A반 8걸음, 4수·new).
                      #   관문값 1.053 은 «원본 모델 · 검증셋» 조건이라 2.8배 컸다.
                      #   정규화 계약(무게 = 그 항의 최대 기여)이 성립하려면 학습이
                      #   실제로 보는 분포로 나눠야 한다.
                      # ⚠**미확인**: 아직 실측이 없다. 여기 박힌 0.10 은 **잠정값**이고
                      #   발사 전에 실측 p90 으로 교체해야 한다. 잠정 여부가
                      #   `arm_signature` 에 '?' 로 새겨지므로(아래 OSD_C_PROVISIONAL)
                      #   로그가 "어느 c 로 돌았나"를 숨길 수 없다 — 이 저장소의
                      #   "선언된 판정식 ≠ 채택된 수치" 전례에 대한 방어다.
OSD_C_PROVISIONAL = False  # 실측값을 박았다(0826 관문). 서명의 '?' 가 사라진다.
OSD_W_MAX = 200       # W 창 최대 길이. W = 응답[t1+1 .. t1+L], L=min(200, boxed 끝까지).
                      # 여기 선언해 두는 이유: 창 길이는 **판정식의 일부**이고, 팔의
                      # 정체는 이 파일 한 곳에만 있어야 한다(스코어러가 이 값을 읽는다).
OSD_LEAK_NGRAM = 8    # 누출 가드: 메타 본문과 W 가 이만큼 연속 n-그램을 공유하면 Δcert:=0.
                      # 왜: 메타에 미래 토큰을 미리 써 두면 Δcert 가 "예언 보너스"로
                      # 부풀고, 그것은 메타인지가 아니라 **복사**다.

# ── 도치 자 (inverted ruler) — R 팔의 «메타 처벌» 항 (2026-08-31) ───────────────
# 정의:  sh 는 «정답 힌트를 준 문맥 − 안 준 문맥» 의 메타 프로즈 logp 이고,
#        inv = agg(hint) − agg(plain)  (아래 form/scope 조합).
#        해로운 메타일수록 inv 가 **높다**(정답을 알아도 어색해지지 않는다).
#
# ★어느 «도치» 인가 — 이것이 이 항의 전부다.
#   «도치 자» 라는 이름 아래 서로 다른 통계가 둘 있었고 라벨 판별력은 그중 **하나에만**
#   있다. 2026-08-31 통일 재측정(84 라벨 사이트, `scripts/inv_ruler_unified.py`,
#   scope 2 × form 2 × agg 3 = 12 칸)의 실측:
#     scope=reencode, form=a2d, agg=min : d(harm−help) **+1.036**, AUC 0.210
#                                         [0.100, 0.351]  ← CI 가 0.5 를 배제
#     나머지 11 칸                       : |d| <= 0.14, AUC CI 전부 0.5 포함 (= 우연)
#   이 칸은 `scripts/reverse_ruler.py` 의 `V1_prose_min` 과 ρ=**0.974** 로 같은 자다.
#   ⇒ scope/form/agg 를 하나라도 바꾸면 **신호가 사라진다**. 서명에 셋 다 박는 이유다.
INV_TERM = "meta_inv"   # ★항 이름의 **단일 정의처**. verl_sdc 의 fail-loud 가드가 읽는다
                        #   (0825 "meta_osd vs osd" 키 불일치 사고의 재발 방지).
INV_SCOPE = "reencode"  # 프로즈 문자 구간을 **따로 인코딩**해 채점(= reverse_ruler 판)
INV_FORM = "a2d"        # agg(hint) − agg(plain)  «집계의 차이»
INV_AGG = "min"         # 토큰 최솟값
INV_MIN_PROSE_TOK = 3   # 프로즈 토큰이 이보다 적으면 못 잰다 → 항 0

# ★단측(one-sided) — 게이밍 방어의 핵심.
#     R_inv = − clip( max(0, inv − τ)/c + P·1{거짓선언} , 0, 1 )   ∈ [−1, 0]
#   상을 주지 않는다. 상으로 쓰면 «그 항의 argmax» 를 사냥하게 되고, 이 자의 argmax 는
#   실측상 «정답을 베낀 메타»(answer_leak 변형 +8.04, real −0.49)다.
#   단측이면 그 꼬리는 벌만 커질 뿐 이득이 없고, 벌을 피하는 길은 «답을 안 베끼는 것» 이다.
#   ⚠탈출구가 «메타를 안 내기» 이므로 `META_FLOOR` 와 `emit_rate < 0.2` abort 를 함께 켠다.
INV_TAU = 0.25          # ⚠**잠정**. 84 라벨 사이트 분포의 p50(+0.250). 라벨셋은 «극단»
                        #   선별이라 학습 분포가 아니다. 발사 전 측정모드(COUNTDOWN_INV=1)
                        #   8 걸음으로 학습 분포를 재고 교체한다 — OSD_C 관문과 같은 규약.
                        # ⛔검수(0831) 정정: **«p50 으로 교체» 는 틀린 지침이다.**
                        #   τ 를 분포의 p50 에 두면 «정직한 메타의 정확히 절반» 이 벌을
                        #   받는다 — 정의상 그렇다. 라벨 84 사이트 실측:
                        #     τ=p50 → 벌비율 0.500, 평균 R_inv −0.273
                        #     발화의 기대 순이득 = META_FLOOR(+0.02) + (−0.273) = **−0.25**
                        #   즉 «메타를 안 내기» 가 기대값에서 12 배 유리해진다. 이 항은 벌만
                        #   있고 탈출구가 공짜라, τ 를 분포 한가운데 두는 순간 발화 침식이
                        #   설계상 보장된다(0.89→0.54 전례가 이미 있다).
                        #   τ 를 올리면 침식압과 라벨 분리력이 **함께** 줄지만 비율은 좋아진다:
                        #     τ    발화순이득   분리(도움−해로움)   분리/침식압
                        #     p50    −0.2525          0.2613           1.03
                        #     p70    −0.1281          0.1512           1.18
                        #     p80    −0.0747          0.1119           1.50
                        #     p90    −0.0399          0.0708           1.77
                        #   ⇒ 관문에서 읽을 값은 p50 이 아니라 **정직한 메타의 벌비율이
                        #     10% 이하로 내려가는 분위수(≈p90)** 이고, 그 τ 에서도
                        #     `inv_gaming` 의 6 변형이 여전히 잡히는지 함께 확인해야 한다
                        #     (conf_wrong·answer_leak 은 G2/누출로 τ 와 무관하게 잡힌다).
INV_C = 1.12            # ⚠**잠정**. 같은 라벨셋에서 max(0, inv−τ) 의 p90(1.124).
                        #   «무게 = 그 항의 최대 절대 기여» 계약이 성립하려면 학습이 실제로
                        #   보는 분포로 나눠야 한다(OSD_C 0.366 이 관문값 1.053 과 2.8배
                        #   달랐던 전례).
INV_TAU_PROVISIONAL = True   # 실측 전이면 서명에 '?' 가 박힌다(OSD_C_PROVISIONAL 규약).

# G2 — «메타가 target 에 닿지 않는 식을 긍정형으로 맞다고 선언» 하면 이 크기의 벌.
#   왜 필요한가(실측): `_ARITH`(아래 743행)는 `a op b = c` 만 보고 **텔레메트리 전용**이라
#   보상식에도 ABORT 에도 없다. 게이밍 프로브의 conf_wrong 문장은 `=` 가 없어 그 정규식에
#   안 걸린다. `countdown_inv.false_claim_in_meta` 는 그 37/37 을 전부 잡고 실제 혼잣말
#   84 개에서 거짓양성 0 이었다(2026-08-31 실측).
#   1.0 이면 거짓선언 하나로 이 항이 포화(−1)한다.
INV_FALSE_CLAIM_PEN = 1.0

# ── OPD (힌트 교사, on-policy distillation) — OPT 사다리의 「무엇을」 항 (2026-09-06) ──
# 왜 새 항인가. FT/MT(timing/live_new)는 오라클 판정(옳은 타이밍으로 갈아탔는가)만
# 주고, cd8 판정표가 보여준 실패(OPT_M v3: 판정 자리 성공률 +6pp 지만 새 쌍 시도율은
# 4.4%→2.6%로 되레 줄었다)는 "결과를 흉내내는 지름길"에 취약하다는 뜻이다. OPD 는
# 결과 판정이 아니라 **분포**를 준다 — 오라클(family_dead·live_new_moves, 정답 식은
# 절대 안 준다)로 조건화한 문맥에서 정책 자신이 얼마나 다르게 말했을지를 재는
# on-policy KL 이다. 힌트·삽입·스팬 정의는 `countdown_opd.py`(순수 함수). 실제
# teacher forward 는 `verl_sdc._compute_countdown_opd`(GPU) — 상세는
# docs/DESIGN_opd_hint_teacher.md.
OPD_TERM = "opd_meta"   # ★항 이름의 **단일 정의처**. verl_sdc 의 fail-loud 가드가 이것을
                        #   읽는다(OSD_TERM/INV_TERM 과 같은 "문자열 두 곳 갈림" 방지 규약).
OPD_C = 0.075           # ⚠**잠정**. `scripts/local/opd_probe.py`(2026-09-06, judge_cd7_OPT
                        #   30사이트) 의 `kl_meta_mean` p95(0.0753, 설계 §7 표)를 반올림한
                        #   값 — 학습 온폴리시 KL 분포가 아니라 오프라인 판정 자리 표본이라
                        #   OSD_C/INV_TAU 와 같은 이유로 **잠정**이다. 발사 전 학습 조건
                        #   실측(COUNTDOWN_OPD=1 측정 모드)으로 교체한다.
OPD_C_PROVISIONAL = True

# ── OPD 그룹 중심화(opd_meta_c) — OPT_OPDC 팔 (2026-09-06, docs/DESIGN_opd_hint_teacher.md §10) ──
# 왜 새 항인가(실측 실패 원인). `opd_meta`(단측 벌 `−clip(kl,0,C)/C`)는 **발화한 행에만**
# 걸리고 미발화 행은 0 이다 — 즉 "메타를 냈다 → (거의 항상) 벌을 받는다"가 되어, 정책이
# 벌을 피하는 가장 싼 길인 "메타를 그만 낸다"로 수렴했다(cd7 실측: 발화율 7%→0.8%,
# 15 스텝. `docs/RESULTS_cd7.md` "그룹 중심화 OPD" 항목). 문제는 벌의 **부호**가 아니라
# **기준선**이다 — 모든 발화 행이 같은 절대 기준(0)과 비교되므로 "평균적인 메타"조차
# 벌을 받는다.
# 해법: 같은 GRPO 그룹(같은 프롬프트의 8 롤아웃) 안에서, 메타를 내고 힌트가 있어 opd_kl
# 을 잰 행끼리만 **서로** 비교한다. `r_i = clip((mean_kl_group − kl_i)/C, −1, +1)` —
# 그룹 평균보다 KL 이 낮은(힌트 교사와 더 가까운) 메타는 +, 높은 메타는 −. 그룹 평균
# 자체는 **기댓값 0**이다(정의상 `sum(mean−kl_i) = 0`) → "발화 자체"에는 중립이고,
# "발화했다면 어느 메타가 더 나은가"만 순위를 매긴다. 채점 가능 행이 1개 이하인
# 그룹(비교 상대가 없다)과 NaN(포이즌, `r_opd_meta` 와 같은 fail-closed 규약)은 0.
OPD_TERM_C = "opd_meta_c"   # ★OPD_TERM 과 같은 "단일 정의처" 규약 — verl_sdc 의 게이트가
                            #   {OPD_TERM, OPD_TERM_C} 둘 다 읽는다.

# ── 반사실 쌍둥이(counterfactual twin) — OPT_CF 팔 (2026-09-07, §8) ──────────────
# 왜 새 항인가. cd8 "같은 자리 인과 검사"(docs/RESULTS_cd7.md 09-07 01:10)의 실측: 학습
# 전 정책에서 스스로 낸 메모는 nometa 대비 같은 자리 성공률을 **바꾸지 못한다**
# (−0.001, 95% CI 가 0 을 포함). FT/MT/OPT_T/OPT_MT 는 전부 오라클(완전열거) 판정이
# "올바른 타이밍"이었는지만 보고, OPD/OPDC 는 힌트 교사의 분포에 얼마나 가까운지만
# 보는데, 둘 다 "그 메모가 **이 문제에서 실제로 정답률을 바꿨는가**"는 안 잰다.
# 이 항은 그것을 직접 잰다 — 같은 자리(site_id)에서 메타를 **허가**한 프롬프트(main)
# 와 메타 문장 자체가 없는 **plain** 프롬프트(twin, 처치 없음)를 한 배치에 같이 태워,
# main 의 정답률을 twin 그룹의 평균 정답률과 비교한다. twin 은 반사실(counterfactual)
# 기준선 — "이 자리에서 메타 없이 이어쓰면 얼마나 맞히는가"의 그룹 평균이다.
CF_TERM = "cf_meta"     # ★단일 정의처 규약(OSD_TERM/INV_TERM/OPD_TERM 과 같은 이유).
W_CF = 1.0              # 사용자 지시(§8). warmup 대상. OPD/INV 류와 달리 잠정 실측치가
                        #   아니라 사용자가 고정한 값이라 '?' 표시가 없다.


def cf_center_rows(rows: Sequence[Mapping]) -> list[float]:
    r"""OPT_CF 팔. 배치 **전체**(여러 GRPO 그룹을 포괄)에서 `cf_key`(=site_id)로
    짝지은 main/twin 쌍을 찾아, main 행의 `corr` 을 그 twin **그룹의 평균 corr** 과
    비교한다.

    `opd_center_rows`(같은 그룹 안에서 자기들끼리 상대 순위)와 다른 점: 이 함수가
    비교하는 두 편은 **서로 다른 프롬프트**(같은 자리 접합, 메타 허가 문장 유무만
    다름)이고 따라서 서로 다른 GRPO 그룹(uid)에 속한다 — "메모를 허가한 정책이,
    메모 자체가 없는 반사실 쌍둥이보다 이 자리에서 실제로 더 잘 푸는가"라는 인과
    대조다.

    입력 `rows` 는 **배치 전체**의 행 딕셔너리 목록, 각 원소가 최소
    `cf_role`("main"/"twin"/"none"), `cf_key`(str, "" 는 미짝), `emitted`,
    `corr`(={0,1}, r_corr 그대로) 를 갖는다.

    규칙 (그 외는 전부 0.0):
      * `cf_role == "twin"` 인 행은 **항상** 0.0 — 비교 기준일 뿐 처치 대상이 아니다.
      * `cf_role == "none"`(일반 롤아웃) 도 0.0 — 반사실 쌍이 없다.
      * `cf_role == "main"` 이고 `emitted` 인 행만 채점 대상:
        `r = clip(corr − mean(같은 cf_key 의 twin 그룹 corr), −1, +1)`.
      * `cf_role == "main"` 인데 미발화, 또는 같은 배치에 그 `cf_key` 의 twin 행이
        **하나도 없으면**(데이터 셔플이 쌍을 갈랐거나 이 자리가 twin이 없는 경우)
        0.0 — 비교 불가를 조용히 숨기지 않고 그냥 무처치로 둔다.

    반환값은 입력과 같은 길이·순서의 리스트.
    """
    rows = list(rows)
    twin_corr: dict = {}
    for r in rows:
        if str(r.get("cf_role", "none")) == "twin":
            key = r.get("cf_key")
            twin_corr.setdefault(key, []).append(_f(r.get("corr", 0.0)))
    twin_mean = {k: (sum(v) / len(v)) for k, v in twin_corr.items() if v}
    out: list[float] = []
    for r in rows:
        if str(r.get("cf_role", "none")) == "main" and _bool01(r.get("emitted", 0)):
            m = twin_mean.get(r.get("cf_key"))
            if m is None:
                out.append(0.0)
            else:
                x = _f(r.get("corr", 0.0)) - m
                out.append(max(-1.0, min(1.0, x)))
        else:
            out.append(0.0)
    return out


def r_cf_meta(cf_meta_raw: float | None) -> float:
    r"""OPT_CF 팔의 항 값. `cf_meta_raw` 는 `cf_center_rows` 가 배치 단위로 미리
    계산해 행에 얹어 둔 결과(이미 [-1,+1] 클립됨) — `r_opd_meta_c` 와 같은 얇은
    통과 함수(그룹/배치 문맥이 필요한 계산은 `verl_sdc._compute_countdown_arm_stash`
    에서 한 번만 한다는 패턴을 그대로 따른다).

    `cf_meta_raw is None` 이면(배선 사고가 아니라, `cf_center_rows` 가 이미 모든
    "0 이어야 하는" 칸을 float 0.0 으로 채우므로 정상 경로에서는 None 이 들어오지
    않는다 — 방어적으로만 0.0) 0.0.
    """
    if cf_meta_raw is None:
        return 0.0
    return float(cf_meta_raw)


# ── 반사실 쌍둥이(그룹 내부판) — OPT_CFG 팔 (2026-09-07) ────────────────────────
# 왜 새 항인가. `cf_center_rows`(OPT_CF)는 main(메타 허가 프롬프트) 과 twin(plain
# 프롬프트) 을 **서로 다른 프롬프트**로 비교한다. 실측 교란: 학습 전 정책이 같은 500
# 문제에서 `plain` 0.426 vs `opt` 0.352(7.4pp) — 즉 twin 기준선이 "메타가 없어서"가
# 아니라 "plain 프롬프트라서" 부풀려 있고, 그 결과 cf_meta 는 발화에 불리하게 편향된다.
# 해법: 프롬프트를 아예 바꾸지 않는다. 같은 GRPO 그룹(=같은 프롬프트, uid 하나) 안에서
# 실제로 메타를 낸 롤아웃(E)과 안 낸 롤아웃(NE)을 갈라, E 의 corr 을 그 그룹 NE 의 평균
# corr 과 비교한다 — 프롬프트가 동일하므로 위 7.4pp 교란이 구조적으로 없다.
CF_GROUP_TERM = "cf_group"   # ★단일 정의처 규약(CF_TERM 과 같은 이유).


def cf_group_rows(rows: Sequence[Mapping]) -> list[float]:
    r"""OPT_CFG 팔. 배치 전체(여러 GRPO 그룹을 포괄)를 받아, `cf_center_rows` 가
    쓰는 것과 같은 그룹 단위(`group_id`, `_compute_countdown_arm_stash` 의 uid)로
    묶은 뒤 **그룹 안에서** emitted==1(E) 대 emitted==0(NE) 로 가른다.

    입력 `rows` 는 배치 전체의 행 딕셔너리 목록, 각 원소가 최소
    `group_id`(str/hashable, 같은 프롬프트 롤아웃끼리 같은 값), `emitted`, `corr`
    (={0,1}, r_corr 그대로) 를 갖는다. site 행·일반 행 구분은 필요 없다 — site 접두가
    붙은 행도 프롬프트가 같은 그룹 안이면 그대로 섞인다.

    규칙 (그 외는 전부 0.0):
      * 그룹 안에 NE(`emitted==0`)가 1개 이상 **그리고** E(`emitted==1`)가 1개 이상
        있어야 채점한다. 둘 중 하나라도 없으면(전원 발화 또는 전원 무발화) 그 그룹은
        비교 불가 — 그룹 전체 0.0.
      * 채점 대상 그룹에서 E 행마다 `r = clip(corr − mean(NE 의 corr), −1, +1)`.
      * NE 행은 (채점 대상 그룹이라도) 항상 0.0 — 비교 기준일 뿐 처치 대상이 아니다
        (`cf_center_rows` 의 twin==0 과 같은 관례).

    NaN-safe: `corr`/`emitted` 가 비유한이면 `_f`/`_bool01` 로 fail-closed 처리한다
    (포이즌 행이 그룹 평균을 망치지 못하게).

    반환값은 입력과 같은 길이·순서의 리스트.
    """
    rows = list(rows)
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get("group_id"), []).append(i)
    out = [0.0] * len(rows)
    for _gid, ix in groups.items():
        ne_ix = [i for i in ix if not _bool01(rows[i].get("emitted", 0))]
        e_ix = [i for i in ix if _bool01(rows[i].get("emitted", 0))]
        if not ne_ix or not e_ix:
            continue
        ne_mean = sum(_f(rows[i].get("corr", 0.0)) for i in ne_ix) / len(ne_ix)
        for i in e_ix:
            x = _f(rows[i].get("corr", 0.0)) - ne_mean
            out[i] = max(-1.0, min(1.0, x))
    return out


def r_cf_group(cf_group_raw: float | None) -> float:
    r"""OPT_CFG 팔의 항 값. `cf_group_raw` 는 `cf_group_rows` 가 배치 단위로 미리
    계산해 행에 얹어 둔 결과(이미 [-1,+1] 클립됨) — `r_cf_meta` 와 같은 얇은 통과
    함수(그룹/배치 문맥이 필요한 계산은 `verl_sdc._compute_countdown_arm_stash` 에서
    한 번만 한다는 패턴을 그대로 따른다).

    `cf_group_raw is None` 이면(배선 사고가 아니라, `cf_group_rows` 가 이미 모든
    "0 이어야 하는" 칸을 float 0.0 으로 채우므로 정상 경로에서는 None 이 들어오지
    않는다 — 방어적으로만) 0.0.
    """
    if cf_group_raw is None:
        return 0.0
    return float(cf_group_raw)


# ── OPT_VTR/OPT_VTRW(§12) — 온라인 검증 게이트 + "언제" 보상 (2026-09-07/08) ─────
# 왜 새 항인가. OPT_OPDG(§11)의 게이트는 **오프라인**이다 — 학습 전(고정) 힌트 교사
# 롤아웃으로 자리를 한 번 채점해 두고, 학습 내내 그 값을 그대로 쓴다. 정책이 학습
# 중에 바뀌면(발화 습관·능력 모두) 오프라인 게이트가 "지금" 그 자리에서 힌트가
# 실제로 도움이 되는지와 어긋날 수 있다. OPT_VTR 은 배치마다 **온라인**으로 다시
# 잰다 — 그 스텝에 실제로 뽑은 힌트 twin 롤아웃(K=4, `build_cf_twins.py --mode
# hint`)의 평균 정답률에서, 같은 배치 main 그룹의 **무발화** 롤아웃 평균 정답률을
# 뺀 값이 tau 이상이면 그 자리를 그 스텝에서 게이트 통과로 본다. 온라인 비교가
# 불가능한 자리(무발화 main 표본이 그 배치에 없다 — 예: 발화율이 높아져 전원
# 발화한 그룹)는 오프라인 `opd_gate`(build_gate_sites.py 산출) 로 **폴백**한다.
# 게이트를 통과한 행은 OPT_OPDG 와 같은 `opd_meta_c` 증류를 받고, 그 외는 순수
# 결과 GRPO 만 받는다(`verl_sdc._compute_countdown_arm_stash` 가 이 함수의 출력을
# `opd_center_rows(gate=…)` 에 그대로 넘긴다).
WHEN_TERM = "when"      # ★단일 정의처 규약(CF_TERM 과 같은 이유).
VTR_TAU = 0.10          # 사용자 지시(§12). `VTR_TAU` 환경변수로 오버라이드.
VTR_WHEN_W = 0.2        # 사용자 지시(§12). `VTR_WHEN_W` 환경변수로 오버라이드.


def resolved_vtr_tau() -> float:
    """온라인 검증 게이트 임계값 — `VTR_TAU` 로 오버라이드(기본 0.10).

    `_resolved_arith_threshold`(이 파일 하단)와 같은 패턴: 오버라이드 여부를
    `arm_signature` 가 찍어 "선언된 값 = 실행된 값"을 로그 한 줄로 확인한다.
    """
    v = os.environ.get("VTR_TAU")
    return float(VTR_TAU) if v is None else float(v)


def _vtr_tau_overridden() -> bool:
    return os.environ.get("VTR_TAU") is not None


def resolved_vtr_when_w() -> float:
    """"when" 보상 무게 — `VTR_WHEN_W` 로 오버라이드(기본 0.2)."""
    v = os.environ.get("VTR_WHEN_W")
    return float(VTR_WHEN_W) if v is None else float(v)


def _vtr_when_w_overridden() -> bool:
    return os.environ.get("VTR_WHEN_W") is not None


def vtr_batch_gate(rows: Sequence[Mapping], *, tau: float | None = None) -> dict[str, tuple[int, str]]:
    r"""OPT_VTR/OPT_VTRW 팔. 배치 **전체**에서 `vtr_key`(=site_id) 별로 온라인 검증
    게이트를 계산한다 — 계산 불가하면 오프라인 `opd_gate` 로 폴백한다(§12).

    입력 `rows` 는 배치 전체의 행 딕셔너리 목록, 각 원소가 최소
    `vtr_role`("main"/"twin"/"none"), `vtr_key`(str, "" 는 사이트 무관),
    `emitted`(main 행에서만 의미), `corr`(={0,1}, r_corr 그대로), `opd_gate`
    (오프라인 폴백값, 0/1, `build_gate_sites.py` 산출 — 없으면 0) 를 갖는다.

    자리(`vtr_key`)마다:
      * `hint_corrs`  = twin 행(`vtr_role=="twin"`)의 corr 전부(그 자리 K개).
      * `nometa_corrs`= main 행(`vtr_role=="main"`) 중 **무발화**(`emitted==0`)
        인 것의 corr(그 자리 그룹 안에서 이번 배치가 실제로 뽑은 무발화 롤아웃).
      * 둘 다 1개 이상이면 **온라인**: `gate = 1 if mean(hint_corrs) −
        mean(nometa_corrs) >= tau else 0`, source="online".
      * 아니면(둘 중 하나라도 비었다) **폴백**: 그 자리 main 행의 `opd_gate`
        값(첫 번째로 발견된 것, 전부 같은 값이어야 정상이지만 방어적으로 첫 값만
        본다) 을 그대로 쓴다, source="fallback". main 행도 그 자리 twin 행도
        배치에 전혀 없으면(자리 자체가 이 배치에 없음) 표에 아예 안 실린다.

    반환값: `{site_id: (gate:int, source:"online"|"fallback")}`. `vtr_key==""`
    (사이트 없음, normal 행) 은 표에 안 들어간다.
    """
    tau = resolved_vtr_tau() if tau is None else float(tau)
    hint_by_site: dict[str, list[float]] = {}
    nometa_by_site: dict[str, list[float]] = {}
    fallback_by_site: dict[str, int] = {}
    seen_sites: set[str] = set()
    for r in rows:
        role = str(r.get("vtr_role", "none"))
        key = r.get("vtr_key") or ""
        if not key:
            continue
        seen_sites.add(key)
        if role == "twin":
            hint_by_site.setdefault(key, []).append(_f(r.get("corr", 0.0)))
        elif role == "main":
            if key not in fallback_by_site:
                fallback_by_site[key] = _bool01(r.get("opd_gate", 0))
            if not _bool01(r.get("emitted", 0)):
                nometa_by_site.setdefault(key, []).append(_f(r.get("corr", 0.0)))
    out: dict[str, tuple[int, str]] = {}
    for key in seen_sites:
        hints = hint_by_site.get(key, [])
        nometas = nometa_by_site.get(key, [])
        if hints and nometas:
            delta = (sum(hints) / len(hints)) - (sum(nometas) / len(nometas))
            out[key] = (int(delta >= tau), "online")
        else:
            out[key] = (fallback_by_site.get(key, 0), "fallback")
    return out


def r_when(row: Mapping) -> float:
    r"""OPT_VTRW 팔의 "when" 보상. site 행(`is_site_row`)에서 실제로 메타를 낸
    행에서만 채점한다 — 발화를 요구하지 않는다(§5 사양: "no meta emitted -> 0,
    never force emission").

    `family_dead`(마지막 두 시도의 첫수 계열이 이미 죽었는가, `r_timing` 과 같은
    오라클) 와 `dec_redirect`(decision=="redirect" 였는가) 의 정합만 본다 —
    verify/무결정은 전부 "continue"(계열을 안 버렸다) 로 묶는다.

        emitted==0                                    → 0.0 (발화 강제 아님)
        family_dead is None(판정불가, 시도 없음)          → 0.0 (r_timing 과 같은 관례)
        family_dead==1(죽음) ∧ redirect                → +w  (옳게 갈아탐)
        family_dead==0(생존) ∧ ¬redirect(continue)     → +w  (옳게 계속함)
        family_dead==1 ∧ ¬redirect                     → −w  (죽었는데 안 갈아탐)
        family_dead==0 ∧ redirect                       → −w  (안 죽었는데 갈아탐)

    범위 {−w, 0, +w}, w=`resolved_vtr_when_w()`.
    """
    if not _bool01(row.get("emitted", 0)):
        return 0.0
    fd = row.get("family_dead")
    if fd is None:
        return 0.0
    fd = _bool01(fd)
    redirect = _bool01(row.get("dec_redirect", 0))
    w = resolved_vtr_when_w()
    match = (redirect and fd == 1) or ((not redirect) and fd == 0)
    return w if match else -w

SHIFT_PARAMS = dict(scale=1.0, clip=2.0, reversal_save=1.0, reversal_derail=2.0,
                    reversal_min_magnitude=0.0)
# C·F·H 가 sign 을 곱하기 전에 쓰는 파라미터. 위 ⚠사양 충돌 참조 —
# 표 쪽 읽기(reversal 없음)를 채택하려면 save/derail 을 0.0 으로.
SHIFT_PARAMS_MUL = dict(SHIFT_PARAMS)

CTX_CLIP = 2.0      # 문맥대조 clip. `dcpo_rmeta_forms._clip` 의 기본값과 같아야 한다
                    # (그 함수를 그대로 호출하므로 구조적으로 같다 — 테스트가 지킨다).

# ── 항별 정규화 (2026-08-20, 사용자 결정) ─────────────────────────────────────
# 각 항의 **원값 최대 절대치**로 나눠 [-1,1] 로 맞춘 뒤 무게를 곱한다.
# ⇒ 무게가 곧 «그 항이 줄 수 있는 최대 절대 기여» 가 된다.
#
# 왜 필요한가 (정규화 없을 때 실측):
#   corr        [0, 1]        폭 1.0   ← 목표
#   meta_pos    [-4, +3]      폭 7.0   ← corr 의 4배
#   gate        [-1, +1]      폭 2.0
#   len(G)      [0, inf)      무제한
#   ⇒ 「문제를 풀었다+메타 평범」 1.370  <  「못 풀었다+메타 최고」 3.370.
#     보상이 *"퍼즐은 포기하고 반성문을 잘 써라"* 라고 말한다. 사전등록 §1 의
#     «메타인지는 목적이 아니라 정확도를 올리는 수단» 과 정반대다.
#
# ⚠이것은 **처치 크기 변경**이다. 사전등록은 w_meta/w_gate 의 크기를 준 적이 없고
#   (아래 W_META 주석의 "미확인 → 1.0 으로 선언" 참조) 이 값은 발사 전에 고정된다.
#   `arm_signature` 에 상태가 박히므로 로그만 봐도 어느 판인지 구분된다.
NORMALIZE_TERMS: bool = True

# ══════════════════════════════════════════════════════════════════════════════
# ★0909 «검산(check)» 메타 행동 — 자기 주장 감시. 프롬프트 변형 `chk`(countdown_task) 전용.
#   메타 칸 점수를 세 방식으로 나란히 잰다(docs/PREREGISTRATION §13):
#     fclaim   (답 칸, 공통): 박스 식이 정답이 아닌데 ✗ 표시가 없다 → 거짓 주장, −W_FCLAIM.
#     chk_fixed(FIXED_CHK)  : 형식에 맞는 <check> 가 하나라도 있으면 +W_CHK — 내용 안 봄(선행 연구식).
#     chk_evc  (EVC_CHK)    : ✗ 로 표시한 식이 **정말** 틀렸고(값≠목표 또는 규칙 위반) 최종 박스가
#                             그 식과 다르며 정답이면 +W_CHK — «거짓을 잡고 고쳤다」. 그 외 0.
#   검산 안의 산수에는 점수를 주지 않는다 — 점수는 «자기 주장이 틀렸음을 알아채고 행동을
#   바꿨나」에만 붙는다(사용자 지적 0909: 계산을 앞당긴 것은 메타인지가 아니다).
# ══════════════════════════════════════════════════════════════════════════════
W_FCLAIM = 0.5
W_CHK = 0.5
_CHECK_RE = re.compile(r"<check>\s*(.*?)\s*</check>", re.S)


def _norm_expr(e: str) -> str:
    e = e.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/").replace("$", "")
    e = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", e)
    return re.sub(r"\s+", "", e)


def parse_checks(text: str) -> list[dict]:
    """<check> 블록들을 (expr, mark) 로. mark ∈ {"ok","bad",None}. `expr = value` 의 좌변만 쓴다."""
    out = []
    for body in _CHECK_RE.findall(text or ""):
        mark = "bad" if ("\u2717" in body or "\u274c" in body) else ("ok" if ("\u2713" in body or "\u2714" in body) else None)
        left = body.split("=")[0]
        left = re.sub(r"[\u2713\u2714\u2717\u274c]", "", left).strip()
        if left:
            out.append({"expr": _norm_expr(left), "mark": mark})
    return out


def check_row(text: str, nums, target, r_corr: int) -> dict:
    """행 하나의 검산 원재료. countdown_task.grade/extract_expr/eval_countdown 재사용(복제 금지)."""
    from src.training import countdown_task as _ct   # noqa: PLC0415
    checks = parse_checks(text)
    final = _ct.extract_expr(text)
    final_n = _norm_expr(final) if final else None
    def _wrong(expr: str) -> bool:
        try:
            v = _ct.eval_countdown(expr)
        except Exception:
            return True
        if v is None:
            return True
        used = sorted(int(x) for x in re.findall(r"\d+", expr))
        return not (v == int(target) and used == sorted(int(x) for x in nums))
    flagged_bad = {c["expr"] for c in checks if c["mark"] == "bad"}
    has_box = final is not None
    fclaim = int(has_box and not _bool01(r_corr) and (final_n not in flagged_bad))
    chk_fixed = int(any(c["mark"] is not None for c in checks))
    chk_evc = int(_bool01(r_corr) and any(e != final_n and _wrong(e) for e in flagged_bad))
    return {"fclaim": fclaim, "chk_fixed": chk_fixed, "chk_evc": chk_evc, "n_checks": len(checks)}


def r_fclaim(fclaim) -> float:
    return -1.0 if _bool01(fclaim) else 0.0


TERM_MAX_ABS: dict = {
    "fclaim":     1.0, "chk_fixed": 1.0, "chk_evc": 1.0,   # ★0909 check
    "corr":       1.0,   # {0,1}
    "format":     1.0,   # {0,1}
    "meta_floor": 1.0,   # {0,1}
    "meta_pos":   4.0,   # clip 2 + derail 2 (이론 최대. 실측은 §1.5 — derail 은 115 중 2건)
    "meta_mul":   4.0,   # meta_pos x sign
    "meta_ctx":   2.0,   # CTX_CLIP
    "gate":       1.0,   # [-1,+1]
    "len":        4.0,   # meta 400 토큰에서 포화 — 가짜 대조군의 무제한을 닫는다
    # r_osd 가 이미 y·clip(·,±1) 로 [-1,1] 이라 정규화는 사실상 항등이다. 그래도
    # **명시한다** — `.get(t, 1.0)` 기본값에 기대면 나중에 항이 조용히 스케일을 잃는다.
    "osd":        1.0,   # [-1,+1] (정의상)
    "meta_pos_full": 4.0,   # r_meta_pos 와 같은 식(clip 2 + derail 2) → 같은 상한
    # r_meta_inv 가 이미 [−1, 0] 이라 정규화는 항등이다. `.get(t, 1.0)` 기본값에 기대지
    # 않고 **명시한다**(osd 와 같은 이유).
    "meta_inv":   1.0,   # [−1, 0] (정의상)
    # ── SC/SCg — 전부 함수 정의상 이미 [-1,1] 안이다. `.get(t, 1.0)` 기본값에 기대지
    #   않고 명시한다(osd·meta_inv 와 같은 이유 — 나중에 항이 조용히 스케일을 잃지 않게).
    "explore":    1.0,   # stuck×dec_redirect×novel×followed ∈ {0,1}
    "explore_g":  1.0,   # explore × plan_ok ∈ {0,1}
    "verify":     1.0,   # hi×(dec_verify×checked − (1−checked)×1[y<0]) ∈ [-1,1]
    "early_cost": 1.0,   # −early ∈ {-1,0}
    # ── FT/M0/MT (고정 자리) — 정의상 이미 각 범위 안이다. osd/meta_inv 와 같은 이유로
    #   `.get(t, 1.0)` 기본값에 기대지 않고 명시한다.
    "timing":     1.0,   # r_timing ∈ {-1,0,+1}
    # timing2 (OPT_MT2, 0906): r_timing2 도 같은 범위 {-1,0,+1} — family_dead=None
    # 칸의 redirect 만 0→-1 로 바뀐 것 뿐(위 timing2 항목/`r_timing2` docstring 참조).
    "timing2":    1.0,   # r_timing2 ∈ {-1,0,+1} (weight 는 TERMS 에서 W_TIMING)
    "live_new":   1.0,   # r_live_new ∈ {0,1}
    # r_opd_meta 가 이미 [−1, 0] 이라 정규화는 항등이다. osd/meta_inv 와 같은 이유로
    # `.get(t, 1.0)` 기본값에 기대지 않고 명시한다.
    "opd_meta":   1.0,
    # opd_center_rows 가 이미 clip(·,−1,+1) 로 [-1,1] 이라 정규화는 항등이다.
    # opd_meta 와 같은 이유로 `.get(t, 1.0)` 기본값에 기대지 않고 명시한다.
    "opd_meta_c": 1.0,
    # cf_center_rows 가 이미 clip(·,−1,+1) 로 [-1,1] 이라 정규화는 항등이다.
    # opd_meta_c 와 같은 이유로 `.get(t, 1.0)` 기본값에 기대지 않고 명시한다.
    "cf_meta": 1.0,
    # cf_group_rows 가 이미 clip(·,−1,+1) 로 [-1,1] 이라 정규화는 항등이다.
    # cf_meta 와 같은 이유로 `.get(t, 1.0)` 기본값에 기대지 않고 명시한다.
    "cf_group": 1.0,
    # r_when 이 이미 [−w,+w](w=resolved_vtr_when_w(), 기본 0.2) 로 무게를 내장한다
    # (`arm_reward` 의 일반 무게×스케일 곱은 여기서는 항등 1.0×scale 만 얹는다) —
    # cf_meta/cf_group 과 같은 이유로 `.get(t, 1.0)` 기본값에 기대지 않고 명시한다.
    "when": 1.0,
}

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


def r_osd(delta_cert, r_corr, *, c: float | None = None) -> float:
    r"""★OSD 팔. R_osd = y · clip(Δcert / c, ±1),  y = +1(정답) / −1(오답).

    Δcert = (1/|W|)·[ logP(W | 프롬프트 ⊕ 응답[..t1])        # 메타를 **포함**한 문맥
                    − logP(W | 프롬프트 ⊕ 응답[..t0)) ]      # 메타 구간만 **제거**한 문맥
    즉 "내 메타가 내 뒤 문장을 얼마나 덜 놀랍게 만들었나"(토큰당 나트).

    ★왜 결과 부호(y)를 곱하나 — 이 항의 전부가 여기 있다.
      Δcert 그 자체는 "확신이 늘었다"만 말한다. 확신은 그 자체로 미덕이 아니다.
      정답 부호를 곱하면 네 칸이 전부 우리가 원하는 방향이 된다:
        Δcert>0 & 정답 → +  : 도움이 된 메타에 상.
        Δcert>0 & 오답 → −  : **확신에 차서 틀린** 롤아웃에 벌(과신 억제).
        Δcert<0 & 정답 → −  : 자기 답을 흔들어 놓고 맞힌 경우엔 신용을 주지 않는다.
        Δcert<0 & 오답 → +  : **의심을 표하고 틀린** 롤아웃에 상(정직한 불확실성).
      ⇒ 보상하는 것은 "확신"이 아니라 **확신과 결과의 정렬**, 곧 calibration 이다.

    ★왜 이것이 전원 오답 그룹을 살리나.
      현재 그룹의 57% 는 전원 오답이라 correctness 어드밴티지가 통째로 0 이고, C·F 의
      곱셈 항은 그 그룹에서 침묵한다(`r_meta_mul` 주석의 sign==0 대가). OSD 는 y 가
      그룹 전원 −1 이어도 **Δcert 가 롤아웃마다 다르므로** 그룹 내 분산이 남는다.
      GRPO 는 그룹 평균을 빼고 남는 분산만 쓰므로, 거기서 기울기가 살아난다.

    ★왜 `adv_corr` 이 아니라 `r_corr` 인가 (C·F 와 다른 점).
      `adv_corr` 은 그룹 동점에서 정확히 0 이 되고, 그래서 C·D·F 는 `SIGN_ZERO_VALUE`
      라는 별도 정책이 필요했다. y 는 **롤아웃 자신의 채점 결과**라 0 이 될 수 없다
      (0/1 뿐). ⇒ 이 항은 `sign_of()==0` 문제도, `SIGN_ZERO_VALUE` 도 쓰지 않는다.
      그 대가는 정직하게 적어 둔다: y 가 라벨(`r_corr`) 자체이므로 **R_meta 의 AUC 는
      정의상 1.000** 인 항진명제다. 그래서 `meta_outcome_discrimination` 의
      `sign_injected` 집합에 `"osd"` 를 등록해 "AUC 해석 대상 아님"으로 표시한다.
      OSD 팔의 판정은 auc 가 아니라 `auc_total`/`inversion_rate` 와 정확도로 한다.

    ★c 는 왜 상수로 나누나. Δcert 는 토큰당 나트라 절대 크기가 모델·프롬프트에 따라
      다르다. 사전 측정한 |Δcert| 의 p90 으로 나눠야 "무게 = 그 항의 최대 절대 기여"
      라는 §정규화 계약이 성립한다. c 는 클램프하지 않고 **예외**로 지킨다
      (`r_gate` 의 p̂ 범위 검사와 같은 규약) — 0 이나 음수 c 는 집계 버그다.

    ★NaN 정책. 행 하나가 NaN 이면 **0.0 fail-closed**(포이즌 행이 그룹 형제의 센터링을
      망치지 못하게 — 이 파일의 `_f` 규약과 동일). "행 드롭"은 구조적으로 불가능하다:
      `verl_sdc.countdown_arm_reward` 가 `len(total) != len(completions)` 이면 즉사시킨다.
      배치 단위 실패(스코어러가 죽었다/NaN 비율 폭발)의 fail-loud 는 **호출자의 몫**이며,
      PMI 의 ref 실패 가드와 같은 자리에 둔다.

    ★누출 가드는 여기 없다. `verl_sdc._osd_leak_guard()` 가 True 면 **호출자가 delta_cert 를 0 으로
      만들어** 넘긴다 — 그래야 텔레메트리가 누출 비율을 따로 잴 수 있다(가드가 이 함수
      안에 숨으면 "얼마나 걸렸나"가 로그에서 사라진다. `answer_leak_rate` 선례).
    """
    # ★c 는 **호출 시점**에 읽는다(기본 인자로 캡처하지 않는다). 관문이 실측 p90 을
    #   재고 `OSD_C` 를 교체하는데, 기본 인자로 굳히면 서명은 새 값을 찍고 보상은 옛
    #   값을 쓴다 — 이 파일이 존재하는 이유인 "선언된 판정식 ≠ 채택된 수치" 그 자체다.
    #   `r_meta_pos(params=None)` 관례와 같은 모양.
    cc = OSD_C if c is None else c
    if not _finite(cc):
        raise ValueError(f"r_osd: c={cc!r} 가 유한수가 아니다 — c 는 |Δcert| 의 사전 측정 p90 이다.")
    cc = float(cc)
    if cc <= 0.0:
        raise ValueError(f"r_osd: c={cc} 는 양수여야 한다(|Δcert| 의 p90). 집계 버그다.")
    # ★None 과 NaN 을 **다르게** 다룬다. 스코어러가 둘을 일부러 구분해 넘긴다.
    #   None = "못 쟀다"(스코어러가 안 돌았거나 죽었다) → 조용한 0 은 이 팔을 A 팔로
    #          위장시킨다. 이 저장소가 명시적으로 금지하는 "선언된 레버, 배선 0" 이다.
    #   NaN  = "쟀는데 비유한"(포이즌 행) → 0.0 fail-closed. 형제의 센터링을 지킨다.
    if delta_cert is None:
        raise ValueError(
            "r_osd: delta_cert=None — 스코어러가 이 행을 재지 못했다. 조용한 0 을 돌려주면 "
            "OSD 팔이 A 팔과 동일해지고 그것이 '선언된 레버, 배선 0' 이다. "
            "verl_sdc 의 OSD 스코어러 배선/COUNTDOWN_OSD 환경변수를 확인하라.")
    if not _finite(delta_cert):
        return 0.0
    y = 1.0 if _bool01(r_corr) else -1.0
    return float(y * max(-1.0, min(1.0, float(delta_cert) / cc)))


def r_meta_inv(inv_raw, false_claim=0, *, tau: float | None = None,
               c: float | None = None, fc_pen: float | None = None) -> float:
    r"""★R 팔. **단측 벌만** 준다.

        R_inv = − clip( max(0, inv − τ)/c  +  P·1{거짓선언} , 0, 1 )   ∈ [−1, 0]

    inv = `countdown_inv` 가 잰 도치 점수(scope=reencode, form=a2d, agg=min).
    해로운 메타일수록 **높다** — 라벨 84 사이트에서 harm +0.24 / help −1.53,
    d=+1.036, AUC 0.210 [0.100, 0.351] (2026-08-31 통일 재측정).

    ★왜 «벌만» 인가 — 이 함수가 존재하는 이유다.
      −inv 를 **상**으로 주면 이 항의 argmax 를 사냥하게 된다. 그 argmax 가 무엇인지
      학습 전에 이미 실측했다(37 사이트 × 6 적대 변형):
        answer_leak(정답 베끼기) +8.04 · conf_wrong(확신에 찬 오답) +0.76 · real −0.49
      즉 **정답을 베낀 메타가 이 자에서 가장 높다**. 상으로 쓰면 −inv 가 가장 낮으므로
      그쪽으로 안 갈 것 같지만, 반대편 꼬리(gibberish −0.46 등 «내용 없는 문장»)가
      최댓값이 되어 «아무 말도 안 하는 메타» 로 붕괴한다. 어느 방향이든 상은 사냥된다.
      단측 벌은 «τ 아래면 전부 0» 이라 사냥할 최댓값 자체가 없다. 탈출구는 정직한 혼잣말
      이고, 그 손실은 0 이다.
    ★왜 그래도 안전하지 않은가(정직하게 적어 둔다). 벌만 있으면 «메타를 안 내기» 라는
      더 싼 탈출구가 열린다. 손실은 `META_FLOOR`(0.02) 뿐이다. 그래서 이 항을 켠 팔은
      `emit_rate < 0.2` ABORT 를 **반드시** 함께 켜고 매 스텝 발화율을 본다
      (0.89→0.54 침식 전례).

    ★G2(거짓선언)는 왜 같은 항에 들어가나. 따로 항을 만들면 팔의 정체가 둘로 갈리고
      `check_negative_control` 의 팔 비교가 «무엇을 처치했나» 를 말하지 못한다. 하나의
      항 안에서 **더한 뒤 함께 클립**하므로 이 팔의 총 벌은 여전히 [−1, 0] 이다.

    ★NaN 정책. inv 가 NaN(=쟀는데 비유한)이면 그 부분만 0 이고 **거짓선언 벌은 그대로**
      간다 — 거짓선언은 텍스트만 보는 판정이라 스코어러와 무관하게 항상 정의된다.
    ★None 정책. inv 가 None 이면 **즉사**한다. "못 쟀다" 를 조용한 0 으로 바꾸면 이 팔이
      A 팔로 위장한다(`r_osd` 와 같은 규약, 0825 사고).
    """
    tt = INV_TAU if tau is None else tau
    cc = INV_C if c is None else c
    pp = INV_FALSE_CLAIM_PEN if fc_pen is None else fc_pen
    if not _finite(tt):
        raise ValueError(f"r_meta_inv: tau={tt!r} 가 유한수가 아니다.")
    if not _finite(cc) or float(cc) <= 0.0:
        raise ValueError(
            f"r_meta_inv: c={cc!r} 는 양수여야 한다(max(0, inv−τ) 의 사전 측정 p90). 집계 버그다.")
    if not _finite(pp) or float(pp) < 0.0:
        raise ValueError(f"r_meta_inv: fc_pen={pp!r} 는 0 이상이어야 한다.")
    if inv_raw is None:
        raise ValueError(
            "r_meta_inv: inv_raw=None — 스코어러가 이 행을 재지 못했다. 조용한 0 을 돌려주면 "
            "R 팔이 A 팔과 동일해지고 그것이 '선언된 레버, 배선 0' 이다. "
            "verl_sdc 의 INV 스코어러 배선/COUNTDOWN_INV 환경변수를 확인하라.")
    pen = 0.0
    if _finite(inv_raw):
        pen += max(0.0, float(inv_raw) - float(tt)) / float(cc)
    pen += float(pp) * (1.0 if _bool01(false_claim) else 0.0)
    return float(-min(1.0, max(0.0, pen)))


def r_meta_floor(emitted) -> float:
    """공통 바닥값. 메타를 냈다는 사실 자체에 붙는 상수(처치 아님)."""
    return 1.0 if _bool01(emitted) else 0.0


# ── SC(자기제어) / SCg 항 — `countdown_selfcontrol.sc_row()` 가 채운 행 필드를 읽는다.
#   ★다른 meta 항(r_meta_pos 등)과 달리 **row 하나를 통째로** 받는다. 이유: SC 의
#   재료는 pmi 같은 스칼라 하나가 아니라 stuck·decision·novel·followed·checked·y 처럼
#   여러 불리언/스칼라의 조합이고, 그 조합 자체가 「상태 조건부 습관」의 정의다.
#   함수 시그니처를 (row) 하나로 두면 TERMS["needs"] 의 키 목록과 함수 본문이 항상
#   같은 이름을 가리켜, 이름이 갈리는 사고(0825 meta_osd/osd)를 원천적으로 막는다.

def r_explore(row: Mapping) -> float:
    r"""SC 팔. explore = stuck × 1[decision=redirect] × novel × followed.  범위 {0,1}.

    막혔는데(stuck) redirect 를 선언하고, 그 next 가 프리픽스에 없던 새 쌍(novel)이며,
    실제로 그 쌍을 이었다(followed) — 넷이 전부 참일 때만 1이다. 어느 하나가 빠지면
    "말만 redirect"이고 그건 이 항이 겨냥하는 습관이 아니다.
    """
    return float(_bool01(row["stuck"]) * _bool01(row["dec_redirect"])
                * _bool01(row["novel"]) * _bool01(row["followed"]))


def r_explore_g(row: Mapping) -> float:
    r"""SCg 팔. explore_g = explore × plan_ok(근거-진리, 완전열거).  범위 {0,1}.

    SC 의 `explore`(모델 자기 신호만)에 "그 next 가 실제로 해를 살리는가"(완전열거
    `_solvable`)를 곱한다 — SC_DESIGN.md 의 대조 취지: "근거-진리 없이 내부 신호만으로
    충분한가"를 SC 대 SCg 로 검정한다. `plan_ok` 는 `plan_next()`(=PL 팔이 이미 쓰는
    같은 완전열거)가 채운다 — 여기서 다시 계산하지 않는다(복제 금지 규약).
    """
    return float(r_explore(row) * _bool01(row["plan_ok"]))


def r_verify(row: Mapping) -> float:
    r"""SC/SCg 팔. verify = hi × ( dec_verify×checked − (1−checked)×1[y<0] ).  범위 [-1,1].

    hi(과신, confidence≥CONF_HI) 가 아니면 0 — 확신이 낮을 때는 이 항이 겨냥하는
    "과신했으면 검산해라"가 애초에 적용 대상이 아니다. hi 인데:
      dec=verify ∧ checked  → +1  (확신하고 실제로 재계산까지 했다 — 상)
      checked 아님 ∧ 오답(y=-1) → −1  (확신에 차 놓고 검산도 안 하고 틀렸다 — 벌)
      그 외(예: dec=verify 인데 checked=0, 또는 checked=1 인데 오답) → 0
    ★`y` 는 `r_corr`(행 자신의 채점 결과)라 0 이 될 수 없다 — `r_osd` 와 같은 이유로
      sign_of()==0 문제가 없다.
    """
    if not _bool01(row["hi"]):
        return 0.0
    checked = _bool01(row["checked"])
    wrong = 1 if float(row["y"]) < 0 else 0
    dec_v = _bool01(row["dec_verify"])
    val = float(dec_v * checked - (1 - checked) * wrong)
    return max(-1.0, min(1.0, val))


def r_early(row: Mapping) -> float:
    r"""SC/SCg 팔. early_cost = −early.  범위 {-1,0}.

    early = 1[프리픽스 시도 수 == 0] — 시작하자마자(아무것도 안 해보고) 메타부터
    내는 것에 대한 비용. §18 실측(SC_DESIGN.md)에서 해로움이 확인된 패턴이다.
    """
    return float(-_bool01(row["early"]))


def r_timing(row: Mapping) -> float:
    r"""FT/MT 팔. 고정 자리 개입의 **타이밍** 보상 — `family_dead`(마지막 두 시도의
    첫수 계열이 이미 죽었는가, `countdown_sites.family_dead_label`) 와 decision
    (redirect/verify) 의 정합. 범위 {-1,0,+1}.

        family_dead is None (시도가 아예 없었다 — 판정 불가)        → 0
        family_dead==1 (계열 죽음) ∧ redirect                       → +1  (옳은 타이밍)
        family_dead==0 (계열 생존) ∧ redirect                       → −1  (성급한 포기)
        family_dead==1 ∧ (verify ∨ redirect 아님)                   → −1  (죽은 계열에서 안 갈아탐)
        그 외 (family_dead==0 ∧ verify 또는 무결정)                  →  0

    `dec_redirect`/`dec_verify` 는 `countdown_selfcontrol.sc_row` 와 같은 이름 —
    `decision` 문자열을 여기서 다시 파싱하지 않는다(복제 금지 규약).
    """
    fd = row["family_dead"]
    if fd is None:
        return 0.0
    fd = _bool01(fd)
    redirect = _bool01(row["dec_redirect"])
    verify = _bool01(row["dec_verify"])
    if fd == 1:
        return 1.0 if redirect else -1.0
    # fd == 0
    return -1.0 if redirect else 0.0


def r_timing2(row: Mapping) -> float:
    r"""OPT_MT2 팔. `r_timing` 과 완전히 같되 **판정불가(family_dead is None) 칸을 없앤다**.

    동기(cd7 실측, 0906, `docs/RESULTS_cd7.md` "OPT_MT v3 s1(pre-fix) 스텝 100"): `r_timing`
    은 `family_dead is None`(아직 아무 시도도 안 해봤다 — 계열 생사를 판정할 근거가 없다)
    일 때 **0**을 준다. 정책은 이것을 착취했다 — 스텝 100에 이르면 문제의 ~100%에서 응답의
    **첫 토큰부터** "redirect" 메타를 낸다. 시도를 하나도 안 했으니 비용이 0이고, 우연히
    계열이 죽어 있던(운 좋은) 문제에서는 +1까지 받는다. "언제 포기할지 판단"이라는 처치의
    취지 자체가 무의미해진다 — 판단할 시도가 없는데 포기부터 하는 것.

    수리: 시도 없이 낸 redirect 는 "성급한 포기"의 극단이므로, 계열이 살아있는데 낸
    redirect(fd==0 ∧ redirect → −1)와 **같은 벌**을 준다. `family_dead is None` 이고
    redirect 가 **아니면**(verify 를 냈거나 무결정) 여전히 판정 근거가 없으므로 0 —
    "아직 아무것도 안 해봤다"는 사실 자체를 벌하지 않는다, 벌하는 것은 어디까지나
    "아무것도 안 해보고 포기(redirect)한 것"이다.

    truth table (범위 {-1,0,+1}, `r_timing` 과 다른 칸만 ★표):
        family_dead is None ∧ redirect                              → −1  ★(신규, 구식은 0)
        family_dead is None ∧ not redirect(verify 또는 무결정)         →  0  (변화 없음)
        family_dead==1 (계열 죽음) ∧ redirect                        → +1  (변화 없음, 옳은 타이밍)
        family_dead==0 (계열 생존) ∧ redirect                        → −1  (변화 없음, 성급한 포기)
        family_dead==1 ∧ (verify ∨ redirect 아님)                    → −1  (변화 없음, 죽은 계열에서 안 갈아탐)
        family_dead==0 ∧ (verify ∨ redirect 아님)                    →  0  (변화 없음)

    `r_timing` 은 이 실험과 무관하게 그대로 둔다(OPT_MT 는 손대지 않는다) — 이 함수는
    OPT_MT2 전용 새 항이다.
    """
    fd = row["family_dead"]
    redirect = _bool01(row["dec_redirect"])
    if fd is None:
        return -1.0 if redirect else 0.0
    fd = _bool01(fd)
    verify = _bool01(row["dec_verify"])
    if fd == 1:
        return 1.0 if redirect else -1.0
    # fd == 0
    return -1.0 if redirect else 0.0


def r_live_new(row: Mapping) -> float:
    r"""FT/MT 팔. 메타 뒤 **첫 결합**이 새로 산(=`live_new_moves`, 프리픽스에 없던 쌍이며
    실제로 해로 이어지는) 수이고 실제로 그것을 이었는가(followed). 범위 {0,1}.

    `first_move_after_meta` 는 `countdown_sites.canon_move`/`move_key_str` 과 같은
    정규형 문자열("a+b" 류)이다 — site 행은 빌드 시점(`countdown_sites`), normal 행은
    런타임(`verl_sdc._compute_countdown_arm_stash`)이 같은 함수로 채운다.
    """
    fm = row["first_move_after_meta"]
    live = row["live_new_moves"] or ()
    followed = _bool01(row["followed"])
    return 1.0 if (fm is not None and fm in live and followed) else 0.0


def r_opd_meta(opd_kl, *, c: float | None = None) -> float:
    r"""OPT_OPD 팔. R_opd_meta = −clip(opd_kl, 0, C) / C ∈ [−1, 0]. 단측 벌만(설계 §2.2/§4).

    `opd_kl` 은 `verl_sdc._compute_countdown_opd` 가 잰 `mean_{t∈span}(lp_student−
    lp_teacher)`(학생 표본 위 KL(student‖teacher) 추정 ≥0 기대; E-132 로 부호 확정, `countdown_opd.opd_reward` 와 **같은 공식** —
    거기는 순수 헬퍼로 테스트 전용이고, 팔의 정체는 이 파일이 유일한 정의처라는 규약을
    지키기 위해 여기서 다시 한 번 짧게 정의한다. `r_osd`/`r_meta_inv` 도 같은 이유로
    자기 스코어러 모듈의 공식을 복제한다).

    `opd_kl is None`(잴 스팬이 없었다 — 메타 없음 또는 힌트를 만들 시도 이력이 없음)
    이면 **0.0**(침묵, `r_timing` 의 `family_dead is None → 0` 과 같은 관례).
    """
    if opd_kl is None:
        return 0.0
    cc = float(OPD_C if c is None else c)
    if not (cc > 0):
        raise ValueError(f"r_opd_meta: c={cc!r} 는 0보다 커야 한다.")
    x = max(0.0, min(float(opd_kl), cc))
    return -x / cc


def opd_center_rows(kls: Sequence[float | None], c: float | None = None,
                    gate: Sequence | None = None) -> list[float]:
    r"""OPT_OPDC 팔. 한 GRPO 그룹(같은 프롬프트의 롤아웃들) 안에서 `opd_kl` **끼리**
    상대 순위를 매기는 그룹 중심화 보상. `r_opd_meta`(절대 기준 0 과 비교하는 단측 벌)와
    달리 이 함수는 그룹 자신의 평균을 기준으로 삼는다 — 위 `OPD_TERM_C` 주석 참조.

    입력 `kls` 는 **한 그룹**의 행들이 낸 `opd_kl` 값 리스트(순서 = 그 그룹 안 행 순서),
    각 원소는 `float | None | nan` 이다:
      * `None` = 메타를 안 냈거나 힌트를 만들 시도 이력이 없다(잴 스팬 없음) — 채점 제외.
      * `nan`  = 쟀는데 비유한(포이즌 행) — `r_opd_meta` 와 같은 fail-closed 규약으로
        채점에서 **제외**한다(평균에도 안 들어가고, 그 행 자신도 0 을 받는다).
      * 유한 실수 = 채점 대상.

    채점 대상이 **2개 미만**인 그룹은 비교 상대가 없으므로 전원 0.0(발화·미채점 불문).
    그 외에는 채점 대상의 평균 `m` 을 구해 각 채점 대상 행에
    `clip((m − kl_i)/c, −1, +1)` 을 준다 — `sum(m − kl_i) = 0` 이므로 그룹 안에서
    기댓값 0(발화 자체는 중립), 그룹 평균보다 KL 이 낮은(힌트 교사와 더 가까운) 메타만
    상대적으로 + 를 받는다. 채점 제외 행(None/nan)은 항상 0.0.

    반환값은 입력과 같은 길이·순서의 리스트다.

    `gate`(OPT_OPDG, §11) — 없으면(None) 위 계약 그대로다(OPT_OPDC 와 바이트 동일).
    주면 같은 길이의 0/1 마스크로 읽어 **게이트 통과 행만** 채점 대상으로 삼는다:
    센터링 평균 `m` 도 게이트 통과 행들만으로 구하고, 게이트 밖 행은 kl 이 유한해도
    항상 0.0 이다. 게이트 통과 채점 대상이 2개 미만인 그룹은 (게이트 안팎 불문)
    전원 0.0 — 위의 "채점 대상 2 미만 → 전원 0" 규약을 그대로 물려받는다.
    """
    cc = float(OPD_C if c is None else c)
    if not (cc > 0):
        raise ValueError(f"opd_center_rows: c={cc!r} 는 0보다 커야 한다.")
    if gate is not None and len(gate) != len(kls):
        raise ValueError(
            f"opd_center_rows: gate 길이({len(gate)}) 가 kls({len(kls)}) 와 다르다.")
    scored_idx = [i for i, k in enumerate(kls) if k is not None and _finite(k)]
    if gate is not None:
        scored_idx = [i for i in scored_idx if _bool01(gate[i])]
    out = [0.0] * len(kls)
    if len(scored_idx) < 2:
        return out
    m = sum(float(kls[i]) for i in scored_idx) / len(scored_idx)
    for i in scored_idx:
        x = (m - float(kls[i])) / cc
        out[i] = max(-1.0, min(1.0, x))
    return out


def r_opd_meta_c(opd_kl_c: float | None) -> float:
    r"""OPT_OPDC 팔의 항 값. `opd_kl_c` 는 `opd_center_rows` 가 그룹 단위로 미리 계산해
    행에 얹어 둔 결과(이미 [-1,+1] 클립됨) — 이 함수는 `arm_reward` 의 다른 항과 같은
    "행 하나 → 값 하나" 호출 패턴을 지키기 위한 얇은 통과 함수일 뿐이다. 실제 계산은
    그룹 문맥이 필요해 `verl_sdc._compute_countdown_arm_stash` 에서 그룹별로 한 번
    수행된다(§10, `phat`/`adv_corr` 와 같은 "그룹 단위 선계산 → 행 필드로 전달" 패턴).

    `opd_kl_c is None`(센터링 자체가 안 돌았다 — 배선 사고) 이면 0.0 이 아니라 이 값을
    그대로 반환하지 않고 **KeyError 대신 조용히 0**을 주는 것도 사고다 — 그래서 이 함수는
    호출되지 않고, `arm_reward` 는 `row["opd_kl_c"]` 를 `needs` 로 강제해 그 필드가 아예
    없으면 즉사한다(무효 레버 방지). `None` 이 **값으로** 들어오는 경우(그룹 채점
    2 미만이라 이 행이 애초에 대상이 아니었다)만 여기서 0.0 을 준다 — `opd_center_rows`
    는 그 경우도 0.0(float)으로 채우므로 실제로는 float 만 들어온다.
    """
    if opd_kl_c is None:
        return 0.0
    return float(opd_kl_c)


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
    "fclaim":     {"needs": ("fclaim",),                                   "warmup": False, "weight": W_FCLAIM},
    "chk_fixed":  {"needs": ("chk_fixed",),                                "warmup": False, "weight": W_CHK},
    "chk_evc":    {"needs": ("chk_evc",),                                  "warmup": False, "weight": W_CHK},
    "format":     {"needs": ("format_ok",),                                "warmup": False, "weight": W_FORMAT},
    # meta_floor 는 워밍업을 **안 받는다**(명시적 결정): 발화 침식을 막는 바닥값인데
    # 워밍업을 받으면 바닥이 도착하기 전에 발화가 무너질 수 있다. 사양은 warmup 대상으로
    # w_meta·w_gate 만 적었고 meta_floor 는 "공통(처치 아님)" 쪽에 있다.
    "meta_floor": {"needs": ("emitted",),                                  "warmup": False, "weight": META_FLOOR},
    "plan":       {"needs": ("emitted", "plan_ok", "plan_followed"),       "warmup": True,  "weight": W_META},
    "meta_pos":   {"needs": ("emitted", "pmi_open", "pmi_close"),          "warmup": True,  "weight": W_META},
    "meta_mul":   {"needs": ("emitted", "pmi_open", "pmi_close", "adv_corr"), "warmup": True, "weight": W_META},
    "meta_ctx":   {"needs": ("emitted", "pmi_self", "pmi_donor", "adv_corr"), "warmup": True, "weight": W_META},
    "gate":       {"needs": ("emitted",),                                  "warmup": True,  "weight": W_GATE},
    "len":        {"needs": ("emitted", "meta_n_tok"),                     "warmup": True,  "weight": W_LEN},
    # osd: 처치 항이므로 meta 계열과 같이 워밍업을 받는다(step 0→20 선형).
    # needs 에 "format_ok" 는 없다 — `format` 항이 _COMMON 이라 모든 팔의 행에 이미
    # 보장돼 있고, 사양의 "형식 위반 → R_osd=0" 은 `arm_reward` 에서 그 키를 직접 읽어
    # 처리한다(없으면 KeyError 로 즉사 = 조용한 0 아님).
    "osd":        {"needs": ("emitted", "delta_cert", "r_corr", "format_ok"),           "warmup": True,  "weight": W_META},
    # ★P 팔(R2_full): 식 «전체» 토큰당 평균 logp 의 (gold−decoy)x(close−open) shift.
    "meta_pos_full": {"needs": ("emitted", "pmi_open_full", "pmi_close_full"),
                      "warmup": True, "weight": W_META},
    # ★R 팔(도치): 단측 벌. osd 와 같은 이유로 `format_ok` 를 함께 본다 — 메타 구간의
    #   경계가 형식으로 정의되므로 형식이 깨진 행의 inv 는 다른 것을 잰 값이다.
    "meta_inv": {"needs": ("emitted", "inv_raw", "inv_false_claim", "format_ok"),
                 "warmup": True, "weight": W_META},
    # ── SC/SCg. needs 는 `countdown_selfcontrol.sc_row()` 가 채우는 필드명 그대로다
    #   (pmi_open 류 원값이 아니라 이미 조합된 불리언/스칼라 — r_explore 등 참조).
    "explore":    {"needs": ("stuck", "dec_redirect", "novel", "followed"),
                   "warmup": True, "weight": W_EXPLORE},
    "explore_g":  {"needs": ("stuck", "dec_redirect", "novel", "followed", "plan_ok"),
                   "warmup": True, "weight": W_EXPLORE},
    "verify":     {"needs": ("hi", "dec_verify", "checked", "y"),
                   "warmup": True, "weight": W_VERIFY},
    # ★early_cost 는 워밍업을 **안 받는다**(meta_floor 와 같은 이유, 명시적 결정):
    #   이것은 "메타를 켜는 보상"이 아니라 "너무 이른 메타에 매기는 비용(바닥형 가드)"
    #   이다. 워밍업을 받으면 초반 스텝에 그 가드가 꺼진 채로 조기발화 습관이 굳을 수
    #   있고, 처음부터 걸려 있어야 "시작하자마자 메타부터 내지 마라"가 스텝 0부터 성립한다.
    "early_cost": {"needs": ("early",), "warmup": False, "weight": W_EARLY},
    # ── FT/M0/MT (고정 자리, 2026-09-04) — needs 는 site 행(빌드 시점,
    #   `countdown_sites.oracle_for_site`) 과 normal 행(런타임,
    #   `verl_sdc._compute_countdown_arm_stash`) 이 **같은 이름**으로 채운다.
    "timing":     {"needs": ("emitted", "family_dead", "dec_redirect", "dec_verify"),
                   "warmup": True, "weight": W_TIMING},
    # ── timing2 (OPT_MT2, 2026-09-06) — `timing` 과 needs/warmup/**weight(W_TIMING)** 전부
    #   동일. 다른 것은 `r_timing2` 의 한 칸(`family_dead is None ∧ redirect → -1`) 뿐이라
    #   OPT_MT 와의 차이가 그 칸 하나로 격리된다.
    "timing2":    {"needs": ("emitted", "family_dead", "dec_redirect", "dec_verify"),
                   "warmup": True, "weight": W_TIMING},
    "live_new":   {"needs": ("emitted", "first_move_after_meta", "live_new_moves", "followed"),
                   "warmup": True, "weight": W_LIVE_NEW},
    # ── OPD (힌트 교사, 2026-09-06) — needs 는 `verl_sdc._compute_countdown_opd` 가
    #   채우는 행 필드명 그대로다. `opd_n_tok` 은 진단용(구간 길이)일 뿐 보상 계산에는
    #   안 쓰므로 needs 에 안 넣는다(design §4: "보상 계산엔 안 쓴다 — 로그·중단 규칙 전용").
    OPD_TERM:     {"needs": ("emitted", "opd_kl"), "warmup": True, "weight": W_META},
    # ── OPD 그룹 중심화(opd_meta_c, 2026-09-06) — needs 에 "opd_kl" 이 아니라
    #   "opd_kl_c" 를 요구한다: 그룹 중심화는 그룹 문맥이 필요해 `verl_sdc.
    #   _compute_countdown_arm_stash` 가 그룹별로 미리 계산해 행에 얹어 둔 값을 읽는다
    #   (`phat`/`adv_corr` 와 같은 "그룹 단위 선계산 → 행 필드" 패턴). 이 필드가 행에
    #   없으면 즉사한다 — OPT_OPDC 가 돌았는데 센터링이 배선 안 된 채 조용히 0 을 흘리는
    #   사고를 막는다(OPD_TERM 과 같은 이유).
    OPD_TERM_C:   {"needs": ("emitted", "opd_kl_c"), "warmup": True, "weight": W_META},
    # ── 반사실 쌍둥이(cf_meta, 2026-09-07, §8) — needs 에 "corr" 원값이 아니라
    #   "cf_meta_raw" 를 요구한다: main/twin 비교는 배치 문맥(다른 GRPO 그룹)이
    #   필요해 `verl_sdc._compute_countdown_arm_stash` 가 배치별로 한 번 계산해
    #   행에 얹어 둔 값을 읽는다(opd_kl_c 와 같은 "배치 단위 선계산 → 행 필드" 패턴).
    #   이 필드가 행에 없으면 즉사한다 — OPT_CF 가 돌았는데 배선이 안 된 채 조용히
    #   0 을 흘리는 사고를 막는다(OPD_TERM/OPD_TERM_C 와 같은 이유).
    CF_TERM:      {"needs": ("emitted", "cf_meta_raw"), "warmup": True, "weight": W_CF},
    # ── 반사실 쌍둥이 그룹판(cf_group, 2026-09-07) — needs 는 CF_TERM 과 같은 이유로
    #   "corr" 원값이 아니라 "cf_group_raw" 를 요구한다: 그룹 내부 emit/no-emit 비교는
    #   그룹 문맥이 필요해 `verl_sdc._compute_countdown_arm_stash` 가 배치별로 한 번
    #   계산해 행에 얹어 둔 값을 읽는다(cf_meta_raw 와 같은 "배치 단위 선계산 → 행
    #   필드" 패턴). 이 필드가 행에 없으면 즉사한다 — OPT_CFG 가 돌았는데 배선이 안
    #   된 채 조용히 0 을 흘리는 사고를 막는다(CF_TERM 과 같은 이유).
    CF_GROUP_TERM: {"needs": ("emitted", "cf_group_raw"), "warmup": True, "weight": W_CF},
    # ── "when" 보상(OPT_VTRW, §12) — `r_timing` 과 같은 원재료(emitted/family_dead/
    #   dec_redirect), 단 무게(±VTR_WHEN_W)는 `r_when` 안에 이미 내장돼 있으므로 여기
    #   무게는 항등 1.0(위 TERM_MAX_ABS["when"] 주석 참조). warmup 은 다른 처치 항
    #   (timing/live_new)과 같이 받는다.
    WHEN_TERM:    {"needs": ("emitted", "family_dead", "dec_redirect"),
                   "warmup": True, "weight": 1.0},
}

_COMMON = ("corr", "format", "meta_floor")   # 공통 = 처치 아님. 여덟 팔 전부 동일.

ARM_SPECS: dict[str, dict] = {
    # ★2026-09-02 N0: 맨 GRPO. 메타 지시문 없는 프롬프트(variant plain, DATA_SUFFIX=_4num_plain)
    #   + 정답·형식만. meta_floor 없음. «메타가 필요한가»의 진짜 대조군.
    "N0": {"label": "plain", "terms": ("corr", "format"),               "meta_form": "none",
           "note": "맨 GRPO. 메타 지시문 없음, 메타 항 없음, 발화 보너스 없음, 형식 = boxed 만."},
    "A": {"label": "corr",   "terms": _COMMON,                          "meta_form": "new",
          "note": "R_corr 만. 메타 항 없음 — 대조군."},
    # ★0902 P 팔(계획): P3 프롬프트(ruled_out/next) + «next 가 해를 살리고(완전 열거) 실제로 이행했는가».
    #   치환 A/B 로 확인된 유일한 내용 신호(막힘 +8.1, 2시드). 메타 텍스트를 사람이 채점하지 않는다.
    "PL": {"label": "plan",  "terms": _COMMON + ("plan",),               "meta_form": "new",
          "note": "★계획 보상. 1[next 해 생존] × 1[이행]. 데이터 _4num_p3."},
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
    # ★A~H 와 달리 키가 한 글자가 아니다. 사양의 여덟 팔은 그대로 두고 **뒤에 붙인**
    #   대체 처치임을 키가 그대로 말하게 했다(PMI-shift AUC 0.52 폐기의 후속).
    "OSD": {"label": "osd", "terms": _COMMON + ("osd",),                "meta_form": "new",
            "note": "★자기 궤적 놀람감소 x 정답부호. PMI-shift 대체."},
    # ★P 팔: 갈아끼우기 검정에서 «해로운 메타» 탐지 AUC 0.587 [0.505,0.664] 로
    #   유일하게 0.5 를 배제한 내용 신호(현행 1토큰 PMI 는 0.522). 입력만 다르고
    #   보상식은 r_meta_pos 를 그대로 쓴다.
    "P": {"label": "full", "terms": _COMMON + ("meta_pos_full",), "meta_form": "new",
          "note": "★개선 후보 R2. 식 전체 토큰당 평균 logp 의 (gold−decoy)x(close−open) shift."},
    # ★R 팔: 도치 자(정답 힌트 조건화)의 **단측 벌**. 상은 없다.
    #   12 칸 통일 재측정에서 라벨 판별력이 있는 유일한 칸(reencode/a2d/min, d=+1.04)이고,
    #   그 칸은 `reverse_ruler.V1_prose_min` 과 ρ=0.974 로 같은 자다.
    "R": {"label": "inv", "terms": _COMMON + ("meta_inv",), "meta_form": "new",
          "note": "★도치 단측 벌. −clip(max(0,inv−τ)/c + P·거짓선언, 0, 1). 상 없음."},
    # ★SC/SCg (2026-09-04, SC_DESIGN.md): 「막히면 탐색(새 계열 redirect), 과신하면
    #   실제 재계산(verify)」 을 모델 **자신의 신호**(프리픽스 시도수·confidence·decision)
    #   로 보상한다. 근거-진리(완전열거)는 SC 에서 안 쓴다. data = `_4num_p3`(P3 프롬프트,
    #   ruled_out/next 구조 — PL 팔과 같은 프롬프트 변형).
    "SC": {"label": "selfctrl", "terms": _COMMON + ("explore", "verify", "early_cost"),
           "meta_form": "new",
           "note": "★자기제어. 막힘→redirect+novel+followed(explore) · "
                   "과신→verify+checked(verify) · 조기메타 비용(early_cost). "
                   "근거-진리 없음 — 내부 신호만. 데이터 _4num_new (0904: p3 는 메타 96% 가 시도 전이라 부적합)."},
    # ★SCg = SC 의 대조점. explore 에만 plan_ok(근거-진리, PL 팔과 같은 완전열거)를
    #   곱한다 — 「내부 신호만으로 충분한가」를 SC 대 SCg 로 검정한다(SC_DESIGN.md).
    "SCg": {"label": "selfctrl_g", "terms": _COMMON + ("explore_g", "verify", "early_cost"),
            "meta_form": "new",
            "note": "★SC + 근거-진리 대조. explore_g = explore × plan_ok. 데이터 _4num_p3 (next: 필드 필요)."},
    # ★SC_GH = 굿하트 압력시험(관문 G-F). 정답 항을 **뺀** SC — 메타 항만 최적화하면 정책이
    #   무엇으로 수렴하는지 20스텝만 보고 표본을 눈으로 검사한다. 학습 주장에 쓰지 않는다.
    "SC_GH": {"label": "selfctrl_goodhart", "terms": ("format", "meta_floor", "explore", "verify", "early_cost"),
              "meta_form": "new",
              "note": "★굿하트 압력시험. corr 없음. 20스텝 뒤 메타 텍스트 퇴화 여부만 본다. 데이터 _4num_new."},
    # ★FT/M0/MT (2026-09-04, 「고정 자리 학습 배선」): 오라클(완전 열거)로 라벨링한
    #   "고정 자리"(site — 부분 응답 접합점) 롤아웃에서 재개해 타이밍·새 수 처치를 준다.
    #   `data_hint` 는 이 파일이 읽지 않는다 — `scripts/local/run_arm.sh` 가 이 값으로
    #   train/val parquet 을 고른다("normal"=일반 롤아웃 전용, "mixed"=site+normal 섞음,
    #   `$WORK/data/sites_v1/mixed_train.parquet`). SC/SCg 와 달리 근거-진리
    #   (`countdown_sites` 완전열거)를 **쓴다** — SC 의 "모델 자기 신호만"과의 대조가
    #   여기서는 "일반 롤아웃에서 재개(FT)" 대 "고정 자리에서 재개(MT)"로 바뀐다.
    "FT": {"label": "fixedtiming", "terms": _COMMON + ("timing", "live_new"), "meta_form": "new",
           "data_hint": "normal",
           "note": "★고정 자리 처치, 일반 데이터. family_dead×decision 정합(timing) + "
                   "메타 뒤 첫수가 새로 산 살아있는 수인가(live_new). 응답은 처음부터 "
                   "생성한다(재개 없음) — data_hint=normal."},
    "M0": {"label": "mixed_ctrl", "terms": _COMMON, "meta_form": "new",
           "data_hint": "mixed",
           "note": "★대조군. 항은 A(=_COMMON) 와 **동일** — 이름이 다른 이유는 데이터가 다르기 "
                   "때문이다(FT/MT 와 같은 mixed_train.parquet, site 3000+normal 3000). "
                   "M0 를 A 의 로그와 섞어 읽지 마라 — 반드시 mixed 데이터로 발사해야 "
                   "FT/MT 와 같은 데이터 위에서 처치 유무만 갈린다."},
    "MT": {"label": "mixed_timing", "terms": _COMMON + ("timing", "live_new"), "meta_form": "new",
           "data_hint": "mixed",
           "note": "★FT 의 처치를 고정 자리(mixed_train.parquet) 데이터에서 반복. site 행은 "
                   "프리픽스가 이미 프롬프트에 접합돼 있으므로 family_dead/live_new_moves 는 "
                   "빌드 시점 오라클(parquet 컬럼)에서, normal 행은 런타임에 같은 함수로 계산한다."},
    # ★OPT/OPT_M (2026-09-05): 메타를 **허용하되 요구하지 않는다**. N0(사양의 진짜
    #   대조군)와 terms 는 같다(정답·형식만, meta_floor 없음) — «강제된 메타의 −7pp
    #   비용이 사라지는가, 그리고 모델이 스스로 «언제» 메타를 쓸지 배우는가»를 재는
    #   것이 요점이지 새 보상 항을 더하는 것이 아니다. 프롬프트만 `opt`(허가 문장,
    #   `countdown_task.PROMPT_VARIANTS["opt"]`)로 바꾼다. `meta_form` 은 "new" 로
    #   두어 발화가 나오면 다른 팔과 같은 계기(`parse_meta`/텔레메트리)로 잡히게 하되,
    #   `require_meta: False` 로 `format_ok_row` 의 형식 점수에서 메타 요구를 뗀다
    #   (N0 처럼 `meta_form: "none"` 을 쓰면 파싱 계기까지 꺼져 발화율을 못 잰다).
    # ★0909 §13 검산 사다리(RL 만, SFT 없음, 프롬프트 chk). 답 칸(corr+fclaim)은 셋이 같고 메모 칸만 다르다.
    "TAG0":      {"label": "chk_tag_only", "terms": ("corr", "format", "fclaim"), "meta_form": "new",
                  "require_meta": False, "data_hint": "normal", "prompt_variant": "chk",
                  "note": "문법(check 허가)만, 메모 칸 점수 0 — 세금 대조군."},
    "FIXED_CHK": {"label": "chk_fixed_bonus", "terms": ("corr", "format", "fclaim", "chk_fixed"), "meta_form": "new",
                  "require_meta": False, "data_hint": "normal", "prompt_variant": "chk",
                  "note": "check 썼으면 +W_CHK(내용 무관) — 선행 연구(고정 검증 보상) 절제군."},
    "EVC_CHK":   {"label": "chk_effect_verified", "terms": ("corr", "format", "fclaim", "chk_evc"), "meta_form": "new",
                  "require_meta": False, "data_hint": "normal", "prompt_variant": "chk",
                  "note": "✗ 로 잡은 식이 정말 틀렸고 최종 답이 다른 정답일 때만 +W_CHK — 효과-검증 메타 크레딧."},
    "OPT": {"label": "optional", "terms": ("corr", "format"), "meta_form": "new",
            "require_meta": False, "data_hint": "normal",
            "note": "★메타 허용·비요구. N0 와 항은 같고 프롬프트만 opt(강제→허가). "
                    "emit_rate<0.2 중단 규칙은 이 팔에서 끈다(발화가 선택이므로) — "
                    "check_abort 참고."},
    "OPT_M": {"label": "optional_mixed", "terms": ("corr", "format"), "meta_form": "new",
              "require_meta": False, "data_hint": "mixed",
              "note": "★OPT 의 데이터만 mixed(고정 자리 배치, opt 프롬프트, mixed_train_v2_opt)."},
    # ★OPT 사다리(0905): 메모는 허용만 하고, 지도(정답표) 두 줄로 «상태에 맞는 행동»만 상벌한다.
    #   좋은 메모 = 죽은 계열에서 redirect + 안 가본 출구 있는 길로 이행(상). 나쁜 메모 = 산 계열에서
    #   redirect, 죽은 계열에서 빈말(벌). 안 쓰면 0 — «불필요할 땐 안 쓴다»는 결과 보상만으로 수렴.
    "OPT_T": {"label": "optional_timing", "terms": ("corr", "format", "timing", "live_new"),
              "meta_form": "new", "require_meta": False, "data_hint": "normal",
              "note": "★OPT + 지도 항(timing 0.5·w, live_new 1.0·w). 메모 강제 없음, 발화 바닥 없음."},
    "OPT_MT": {"label": "optional_timing_mixed", "terms": ("corr", "format", "timing", "live_new"),
               "meta_form": "new", "require_meta": False, "data_hint": "mixed",
               "note": "★OPT_T + 같은 자리 배치 절반(mixed_train_v2_opt)."},
    # ★OPT_MT2 (2026-09-06, `docs/RESULTS_cd7.md` "OPT_MT v3 s1(pre-fix) 스텝 100"):
    #   OPT_MT 와 완전히 같되 timing → timing2 하나만 바뀐다. 동기는 OPT_MT 가 스텝
    #   100 에 이르러 착취한 무효 레버다 — `r_timing` 은 `family_dead is None`(아직
    #   아무 시도도 안 해봤다)일 때 항상 0 을 주므로, 정책은 응답 첫 토큰부터 "redirect"
    #   메타를 내고(비용 0) 계열이 우연히 죽어 있던 문제에서는 +1 까지 챙겼다 —
    #   판정 대상인 「~100% 첫 토큰 redirect」 자체가 이 레버의 산물이다. `r_timing2`
    #   는 그 칸(판정불가 ∧ redirect)을 계열 생존 중 redirect(−1)와 같은 벌로 막는다
    #   (`r_timing2` docstring 의 truth table 참조). OPT_MT 는 이 실험과 무관하게
    #   손대지 않는다 — `timing` 항·`r_timing` 함수 둘 다 그대로다.
    "OPT_MT2": {"label": "optional_timing2_mixed", "terms": ("corr", "format", "timing2", "live_new"),
                "meta_form": "new", "require_meta": False, "data_hint": "mixed",
                "note": "★OPT_MT 와 동일(같은 자리 배치 절반, mixed_train_v2_opt) + timing→timing2"
                        "(무시도 redirect 를 더는 공짜로 두지 않는다 — 위 note 참조)."},
    # ★OPT_OPD (2026-09-06, docs/DESIGN_opd_hint_teacher.md §4/§10): 힌트 교사
    #   on-policy distillation. OPT 와 항이 같고(corr·format, meta_floor 없음,
    #   require_meta=False) opd_meta 하나만 더한다 — FT/MT/OPT_T 의 "결과 판정"과
    #   달리 "이 상태를 알았다면 정책 자신이 다음에 뭐라고 썼을지"의 분포(온폴리시 KL)를
    #   준다. data_hint 는 (기본값) "normal" — OPT/OPT_T 와 같은 일반 롤아웃, opt
    #   프롬프트(허가문, `run_arm.sh` 의 OPT 계열 분기가 그대로 적용된다). §10 조건부
    #   GO: 20스텝 스모크런으로 §6 1차 지표(판정 자리 새 쌍 이행률)·3차(누출률)를 본다.
    "OPT_OPD": {"label": "optional_opd", "terms": ("corr", "format", OPD_TERM),
                "meta_form": "new", "require_meta": False, "data_hint": "normal",
                "note": "★힌트 교사(오라클 상태-요약 조건화) 메타 구간 on-policy 증류. "
                        "OPT 와 항 동일 + opd_meta. 오라클은 family_dead/live_new_moves만 "
                        "쓴다 — 정답 식(witness)은 절대 안 준다."},
    # ★OPT_OPDC (2026-09-06, docs/DESIGN_opd_hint_teacher.md §10): OPT_OPD 의
    #   단측 벌(절대 기준 0)이 cd7 스모크에서 "메타를 그만 낸다"로만 수렴한 실패
    #   (발화율 7%→0.8%, 15 스텝, `docs/RESULTS_cd7.md` "그룹 중심화 OPD")를 고치는
    #   판. 항 하나만 opd_meta → opd_meta_c 로 바뀐다 — 나머지는 OPT_OPD 와 완전히
    #   같다(코드가 아니라 데이터로 이 사실을 확인할 것: `arm_signature` 가 term 이름
    #   차이 하나로만 갈린다).
    "OPT_MTC": {"label": "optional_timing_mixed_opd_centered",
                "terms": ("corr", "format", "timing", "live_new", OPD_TERM_C),
                "meta_form": "new", "require_meta": False, "data_hint": "mixed",
                "note": "★OPT_MT(고정 자리 절반 + 정답표 타이밍) + opd_meta_c. 자율 출발(OPT_OPDC) 에서는 "
                        "발화 6% 라 그룹당 채점 메타가 2개 미만이어서 항이 굶었다(0906 스모크). 자리 행은 "
                        "발화 40~60% 라 중심화 항이 실제로 분산을 갖는다."},
    "OPT_OPDC": {"label": "optional_opd_centered", "terms": ("corr", "format", OPD_TERM_C),
                 "meta_form": "new", "require_meta": False, "data_hint": "normal",
                 "note": "★OPT_OPD 와 항 동일 + opd_meta_c(그룹 중심화) 대신. 같은 그룹의 "
                         "발화·채점된 메타끼리만 kl 을 상대 비교한다 — 발화 자체엔 기대값 "
                         "중립, 힌트 교사에 더 가까운 메타만 상대적으로 밀어올린다."},
    # ★OPT_OPDG (2026-09-07, §11): OPT_OPDC 에 «교사 신뢰도 게이트» 하나만 더한 판.
    #   항 구성은 OPT_OPDC 와 완전히 같고(opd_meta_c), 달라지는 것은 그 항을 **어느
    #   행에서 켜는가**뿐이다 — `extra_info.opd_gate == 1`(그 자리에서 힌트 교사가
    #   nometa 대비 tau 이상 이겼다, `scripts/local/build_gate_sites.py`)인 행만
    #   센터링에 참여하고 나머지는 순수 결과 GRPO 로 남는다. `opd_gate` 는 그래서
    #   서명에도 박힌다(`arm_signature` 참조) — 항 이름만으로는 OPT_OPDC 와 안 갈린다.
    "OPT_OPDG": {"label": "optional_opd_gated", "terms": ("corr", "format", OPD_TERM_C),
                 "meta_form": "new", "require_meta": False, "data_hint": "mixed",
                 "opd_gate": True,
                 "note": "★교사 게이트 증류: opd_meta_c 를 extra_info.opd_gate==1 인 자리에서만 "
                         "적용한다(자리 단위 교사 신뢰도 게이트). 게이트 밖 행은 순수 결과 GRPO."},
    # ★OPT_CF (2026-09-07, §8 「반사실 쌍둥이」): cd8 "같은 자리 인과 검사"(docs/
    #   RESULTS_cd7.md 09-07 01:10)의 실측 — 학습 전 정책에서 스스로 낸 메모는 nometa
    #   대비 같은 자리 성공률을 바꾸지 못한다(−0.001, CI 가 0 포함). OPD/OPDC 는
    #   "힌트 교사와 얼마나 가까운가", timing/live_new 는 "오라클 타이밍이 맞았는가"
    #   만 재고 **결과가 실제로 바뀌었는가**는 안 잰다. cf_meta 는 그것을 직접 잰다:
    #   같은 자리(site_id)에서 메타를 허가한 프롬프트(main)와 메타 문장 자체가 없는
    #   plain 프롬프트(twin, 처치 없음 반사실 기준선)를 한 배치에 같이 태워 main 의
    #   corr 을 twin 그룹 평균과 비교한다. `data_hint="mixed_cf"` 는 이 파일이 안
    #   읽는다 — `scripts/local/run_arm.sh` 가 `mixed_train_v3c_cf_opt.parquet`(twin
    #   포함판)로 라우팅한다.
    "OPT_CF": {"label": "optional_cf_twin", "terms": ("corr", "format", CF_TERM),
               "meta_form": "new", "require_meta": False, "data_hint": "mixed_cf",
               "note": "★반사실 쌍둥이. main(메타 허가) 의 corr 을 같은 자리 twin(메타 "
                       "없음) 그룹의 평균 corr 과 비교(clip ±1) — 「메모가 이 문제에서 "
                       "실제로 정답률을 바꿨는가」의 직접 인과 대조. twin 행 자체는 "
                       "cf_meta=0(비교 기준일 뿐 처치 대상 아님), 일반 롤아웃도 0."},
    # ★OPT_CFG (2026-09-07, §10): OPT_CF 의 **그룹 내부판**. 실측 교란 — 학습 전
    #   정책이 같은 500 문제에서 `plain` 0.426 vs `opt` 0.352(7.4pp): OPT_CF 의 twin
    #   기준선(plain 프롬프트)이 그 자체로 부풀려 있어 cf_meta 가 발화에 불리하게
    #   편향된다. 이 팔은 twin 프롬프트를 쓰지 않는다 — 같은 GRPO 그룹(=같은 opt
    #   프롬프트) 안에서 실제로 메타를 낸 롤아웃(E)과 안 낸 롤아웃(NE)을 갈라 E 의
    #   corr 을 그 그룹 NE 평균 corr 과 비교한다(clip ±1). 프롬프트가 동일하므로 위
    #   7.4pp 교란이 구조적으로 없다. `data_hint="mixed"` — OPT_M/OPT_MT 와 같은
    #   `${MIXED_DATA}_opt.parquet` 로 라우팅된다(twin 데이터 불필요).
    #   알려진 한계(§10): 그룹 안에서 「누가 메타를 냈는가」는 정책 자신의 선택 —
    #   내생적(endogenous) 선택 편향이 있다(모델이 이미 풀 수 있다고 느낀 문제에서만
    #   메타를 낼 수도 있다). OPT_CF(서로 다른 프롬프트 쌍둥이)는 이 편향이 없는 대신
    #   위 7.4pp 교란이 있다 — 둘을 상호 보조적 이차 확인으로만 쓴다.
    "OPT_CFG": {"label": "optional_cf_group", "terms": ("corr", "format", CF_GROUP_TERM),
                "meta_form": "new", "require_meta": False, "data_hint": "mixed",
                "note": "★반사실 쌍둥이 그룹 내부판. 같은 프롬프트 그룹 안에서 emit(E) 의 "
                        "corr 을 no-emit(NE) 그룹 평균 corr 과 비교(clip ±1) — OPT_CF 의 "
                        "twin 프롬프트 교란(plain 0.426 vs opt 0.352) 을 없앤다. 한계: "
                        "emit/no-emit 선택이 내생적(§10) — OPT_CF 를 이차 확인으로 유지."},
    # ★OPT_VTR (2026-09-08, §12 「온라인 검증 게이트」): OPT_OPDG(§11, 오프라인 게이트)
    #   의 항 구성(corr·format·opd_meta_c)을 그대로 쓰되, **어느 행이 게이트 통과인가**
    #   를 매 배치 온라인으로 다시 잰다 — `vtr_batch_gate`(위 정의) 가 그 스텝에 실제로
    #   뽑은 힌트 twin(K=4, `build_cf_twins.py --mode hint`) 평균 정답률과 main 그룹
    #   무발화 평균 정답률의 차이를 tau(=`VTR_TAU`) 와 비교하고, 비교 불가하면
    #   `extra_info.opd_gate`(오프라인)로 폴백한다. `data_hint="mixed_vtr"` 는 이
    #   파일이 안 읽는다 — `scripts/local/run_arm.sh` 가
    #   `mixed_train_v4_vtr_opt.parquet`(힌트 twin 포함판)로 라우팅한다.
    "OPT_VTR": {"label": "optional_vtr_gated", "terms": ("corr", "format", OPD_TERM_C),
                "meta_form": "new", "require_meta": False, "data_hint": "mixed_vtr",
                "opd_gate": True, "vtr_online_gate": True,
                "note": "★온라인 검증 게이트 증류. opd_meta_c 를 그 스텝 실측 "
                        "«힌트 twin 평균 − 무발화 main 평균 >= tau» 인 자리에서만 "
                        "켠다(온라인 불가 시 오프라인 opd_gate 로 폴백). 게이트 밖 "
                        "행은 순수 결과 GRPO."},
    # ★OPT_VTRW (2026-09-08, §12): OPT_VTR + "when" 보상(WHEN_TERM). 게이트 증류는
    #   "어떻게(교사와 얼마나 가까운가)"만 재므로, "언제(죽은 계열에서 갈아탔는가)"를
    #   직접 상벌하는 항을 더한다 — 발화한 site 행에서만 채점, 미발화는 절대 0(발화
    #   강제 금지, §5).
    "OPT_VTRW": {"label": "optional_vtr_gated_when",
                 "terms": ("corr", "format", OPD_TERM_C, WHEN_TERM),
                 "meta_form": "new", "require_meta": False, "data_hint": "mixed_vtr",
                 "opd_gate": True, "vtr_online_gate": True,
                 "note": "★OPT_VTR + when(±VTR_WHEN_W, family_dead·dec_redirect 정합). "
                         "발화를 요구하지 않는다 — 미발화는 항상 0."},
    # ★OPT_OPDGW (2026-09-08 20:3x, §12-d): OPT_OPDG(오프라인 게이트 + 밀집 증류, 트윈 없음) +
    #   when 보상. 실측 근거 — VTR(온라인 게이트, 트윈 34%) held-out s30 .555 < N0 .640 < 이고
    #   g3dn(오프라인 게이트 과표집 + 밀집) 스텝 22 학습 corr .611 ≈ N0 궤적(.605), VTRW 의
    #   when 일치율 .63→.87. 즉 «무엇을(밀집 증류)»은 오프라인 게이트로, «언제(when)»는 보상으로.
    "OPT_OPDGW": {"label": "optional_opd_gated_when",
                  "terms": ("corr", "format", OPD_TERM_C, WHEN_TERM),
                  "meta_form": "new", "require_meta": False, "data_hint": "mixed",
                  "opd_gate": True,
                  "note": "★OPT_OPDG + when(±VTR_WHEN_W). 트윈 없음 — mixed_train_v4_gate3 권장."},
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
    # ★반드시 **조건부**다. 무조건 붙이면 A~H 여덟 서명이 전부 바뀌어 팔 diff 의
    #   로그 연속성이 끊긴다(OSD 는 추가이지 개정이 아니다).
    #   c 를 서명에 박는 이유: "선언된 판정식 = 채택된 수치"를 사후에 한 줄로 확인해야
    #   한다. 실측 전 잠정값이면 '?' 가 붙어 로그가 그 사실을 숨기지 못한다.
    if "osd" in spec["terms"]:
        extra += (f"|osdc={OSD_C:g}{'?' if OSD_C_PROVISIONAL else ''}"
                  f",wmax={OSD_W_MAX:d},ngram={OSD_LEAK_NGRAM:d}")
    # ★정규화 상태는 정체의 일부다 — 안 박으면 서명이 거짓말을 한다.
    if "meta_pos_full" in spec["terms"]:
        extra += "|shiftfull=" + _params_sig(SHIFT_PARAMS)
    # ★scope/form/agg 를 전부 박는다 — 12 칸 중 신호가 있는 칸은 하나뿐이라 하나만 달라도
    #   «검증한 자» 가 아니게 된다. τ·c 가 잠정이면 '?' 가 붙어 로그가 그 사실을 숨기지 못한다.
    if INV_TERM in spec["terms"]:
        _q = "?" if INV_TAU_PROVISIONAL else ""
        extra += (f"|inv=scope={INV_SCOPE},form={INV_FORM},agg={INV_AGG},"
                  f"tau={INV_TAU:g}{_q},c={INV_C:g}{_q},"
                  f"fcpen={INV_FALSE_CLAIM_PEN:g},minprose={INV_MIN_PROSE_TOK:d}")
    # ★c 를 서명에 박는 이유는 OSD_C/INV_TAU 와 같다 — "선언된 판정식 = 채택된 수치"를
    #   사후에 한 줄로 확인해야 한다. 실측 전 잠정값이면 '?' 가 붙는다.
    if OPD_TERM in spec["terms"]:
        extra += f"|opd_c={OPD_C:g}{'?' if OPD_C_PROVISIONAL else ''}"
    # ★OPD_TERM 과 별개 조각(`opdc_c=`)이다 — 두 항이 같은 OPD_C 상수를 공유해도
    #   서명 문자열은 term 이름(`parts` 루프의 `t@weight`)만으로 이미 갈리지만,
    #   OSD_C/INV_TAU 규약(잠정값이면 '?')을 이 항에도 그대로 지키기 위해 명시한다.
    if OPD_TERM_C in spec["terms"]:
        extra += f"|opdc_c={OPD_C:g}{'?' if OPD_C_PROVISIONAL else ''}"
    # ★게이트는 «정체»다 — OPT_OPDG 는 항 구성이 OPT_OPDC 와 바이트 동일이라 이 조각이
    #   없으면 두 팔의 서명이 같아져 로그가 거짓말을 한다(조건부라 기존 서명은 안 깨진다).
    if spec.get("opd_gate"):
        extra += "|opdgate=on"
    # ★OPT_VTR/OPT_VTRW(§12) — 게이트가 온라인(매 배치 재계산)인지를 박는다. 없으면
    #   OPT_OPDG 와 opdgate=on 이 같아 "오프라인 vs 온라인"이 로그에서 안 갈린다.
    if spec.get("vtr_online_gate"):
        _tq = "?" if _vtr_tau_overridden() else ""
        extra += f"|vtr_tau={resolved_vtr_tau():g}{_tq}"
    if WHEN_TERM in spec["terms"]:
        _wq = "?" if _vtr_when_w_overridden() else ""
        extra += f"|when_w={resolved_vtr_when_w():g}{_wq}"
    # ★SC/SCg 정체 — K_S(막힘 임계)·conf_hi(과신 임계)·세 항의 무게가 이 팔의 전부다.
    #   하나라도 안 박으면 "어느 K_S 로 돌았나"가 로그에서 사라진다(OSD_C/INV_TAU 와
    #   같은 규약). 무게는 위 `parts` 루프가 이미 `t@weight` 로 찍으므로 여기서는
    #   `parts` 가 안 담는 두 임계값만 더한다.
    if {"explore", "explore_g", "verify", "early_cost"} & set(spec["terms"]):
        extra += f"|sc_k={SC_K_STUCK:d},sc_conf_hi={SC_CONF_HI:g}"
    extra += f"|norm={'on' if NORMALIZE_TERMS else 'off'}"
    # ★수리(0904, 감사결함5): P0-보정 임계값을 오버라이드했으면 서명에 박는다 —
    #   안 박으면 "이 로그가 어느 중단 임계값 아래서 났는가"를 사후에 확인할 수 없다.
    #   오버라이드가 **없으면** 붙이지 않는다(조건부) — 무조건 붙이면 지금까지의
    #   전체 팔 서명이 다 바뀌어 로그 연속성이 끊긴다(OSD/INV 항과 같은 규약).
    if _abort_arith_overridden():
        extra += f"|abort_arith={_resolved_arith_threshold():g}"
    # ★OPT/OPT_M — `require_meta: False` 는 `format_ok_row` 의 채점을 바꾸므로 서명에
    #   박는다. 조건부다(다른 팔은 전부 True 기본값이라 안 붙어야 기존 서명이 안 깨진다).
    if not spec.get("require_meta", True):
        extra += "|require_meta=0"
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
    if "fclaim" in terms:
        raw["fclaim"] = r_fclaim(row["fclaim"])
    if "chk_fixed" in terms:
        raw["chk_fixed"] = 1.0 if _bool01(row["chk_fixed"]) else 0.0
    if "chk_evc" in terms:
        raw["chk_evc"] = 1.0 if _bool01(row["chk_evc"]) else 0.0
    if "format" in terms:
        raw["format"] = 1.0 if _bool01(row["format_ok"]) else 0.0
    if "meta_floor" in terms:
        raw["meta_floor"] = r_meta_floor(emitted)
    if "meta_pos" in terms:
        raw["meta_pos"] = r_meta_pos(row["pmi_open"], row["pmi_close"]) if emitted else 0.0
    if "plan" in terms:
        raw["plan"] = (1.0 if (emitted and _bool01(row["plan_ok"]) and _bool01(row["plan_followed"])) else 0.0)
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
    if "osd" in terms:
        # 사양: "메타 미발화 **또는 형식 위반** → R_osd = 0". 다른 메타 항은 emitted 만
        # 보지만 OSD 는 형식도 본다 — 메타 구간의 경계(t0/t1)가 형식으로 정의되므로
        # 형식이 깨진 행의 Δcert 는 애초에 다른 것을 잰 값이다.
        # `format_ok` 는 _COMMON 의 `format` 항이 이미 요구하는 키라 반드시 행에 있다.
        raw["osd"] = (r_osd(row["delta_cert"], row["r_corr"])
                      if (emitted and _bool01(row["format_ok"])) else 0.0)

    if "meta_pos_full" in terms:
        raw["meta_pos_full"] = (r_meta_pos(row["pmi_open_full"], row["pmi_close_full"])
                                if emitted else 0.0)

    if INV_TERM in terms:
        # osd 와 같은 규약: 미발화 **또는 형식 위반** → 0. 메타 구간의 경계가 형식으로
        # 정의되므로 형식이 깨진 행의 inv 는 애초에 다른 것을 잰 값이다.
        raw[INV_TERM] = (r_meta_inv(row["inv_raw"], row["inv_false_claim"])
                         if (emitted and _bool01(row["format_ok"])) else 0.0)

    # ── SC/SCg. explore·explore_g·verify 는 decision/confidence 가 원재료라 미발화면
    #   정의상 이미 0(decision=None→dec_redirect·dec_verify=0, confidence=None→hi=0)
    #   이지만, 다른 meta 항과 같은 관례(if emitted else 0.0)로 명시해 둔다 — 그래야
    #   "emitted 게이팅을 빠뜨렸다"는 감사 질문에 코드로 바로 답할 수 있다.
    if "explore" in terms:
        raw["explore"] = r_explore(row) if emitted else 0.0
    if "explore_g" in terms:
        raw["explore_g"] = r_explore_g(row) if emitted else 0.0
    if "verify" in terms:
        raw["verify"] = r_verify(row) if emitted else 0.0
    if "early_cost" in terms:
        # ★emitted 로 게이팅하지 않는다(명시적 결정). `early`는 "메타가 없었다"가
        #   아니라 "프리픽스 시도 수가 0"이라는 뜻이고, 이 비용은 meta_floor 처럼
        #   워밍업도 없이 스텝 0부터 늘 걸려 있어야 하는 바닥형 가드다(TERMS 주석 참조).
        raw["early_cost"] = r_early(row)
    if "timing" in terms:
        raw["timing"] = r_timing(row) if emitted else 0.0
    if "timing2" in terms:
        raw["timing2"] = r_timing2(row) if emitted else 0.0
    if "live_new" in terms:
        raw["live_new"] = r_live_new(row) if emitted else 0.0
    if OPD_TERM in terms:
        raw[OPD_TERM] = r_opd_meta(row["opd_kl"]) if emitted else 0.0
    if OPD_TERM_C in terms:
        # 센터링은 emitted 게이팅과 무관하게 이미 0 이다(opd_center_rows 가 None/nan/
        # 미채점 행에 0.0 을 채운다) — 그래도 다른 meta 항과 같은 관례(if emitted else
        # 0.0)를 명시해 "emitted 게이팅을 빠뜨렸다"는 감사 질문에 코드로 바로 답한다.
        raw[OPD_TERM_C] = r_opd_meta_c(row["opd_kl_c"]) if emitted else 0.0
    if CF_TERM in terms:
        # 센터링은 emitted 게이팅과 무관하게 이미 0 이다(cf_center_rows 가 twin/none/
        # 미발화 main 행에 0.0 을 채운다) — OPD_TERM_C 와 같은 이유로 관례를 명시한다.
        raw[CF_TERM] = r_cf_meta(row["cf_meta_raw"]) if emitted else 0.0

    if CF_GROUP_TERM in terms:
        # 센터링은 emitted 게이팅과 무관하게 이미 0 이다(cf_group_rows 가 NE 행·비교
        # 불가 그룹에 0.0 을 채운다) — CF_TERM 과 같은 이유로 관례를 명시한다.
        raw[CF_GROUP_TERM] = r_cf_group(row["cf_group_raw"]) if emitted else 0.0

    if WHEN_TERM in terms:
        # `r_when` 자체가 emitted==0/family_dead is None 을 이미 0.0 으로 처리한다
        # (§5: 발화 강제 금지) — 그래도 다른 meta 항과 같은 관례를 명시해 "emitted
        # 게이팅을 빠뜨렸다"는 감사 질문에 코드로 바로 답한다.
        raw[WHEN_TERM] = r_when(row) if emitted else 0.0

    comps: dict[str, float] = {}
    for t, v in raw.items():
        w = float(TERMS[t]["weight"])
        s = scale if TERMS[t]["warmup"] else 1.0
        x = float(v)
        if NORMALIZE_TERMS:
            # 항별 정규화: 원값을 그 항의 최대 절대치로 나누고 [-1,1] 로 포화시킨다.
            # 포화가 실제로 무는 항은 `len`(무제한) 하나뿐 — 나머지는 정의상 |x|<=max.
            m = float(TERM_MAX_ABS.get(t, 1.0)) or 1.0
            x = max(-1.0, min(1.0, x / m))
        comps[t] = _f(w * s * x)
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


# ── ★0902 계획 항: 메타의 `next:` 첫수가 해를 살리는가(완전 열거) · 그 수를 실제로 이행했는가 ──────────
_NEXT_RE = re.compile(r"\bnext\s*:\s*(.*)$", re.I)   # parse_meta 는 본문 줄을 공백으로 합친다
_PAIR_RE = re.compile(r"\(?\s*(\d{1,4})\s*([+\-*/×÷])\s*(\d{1,4})\s*\)?")


def _solvable(vals, target: int) -> bool:
    """남은 값들로 목표 도달 가능한가 — 완전 열거(move_space_probe.solvable 와 동형)."""
    from fractions import Fraction
    import itertools as _it
    vals = [Fraction(int(v)) for v in vals]; target = Fraction(int(target))
    def go(v):
        if len(v) == 1:
            return v[0] == target
        for i, j in _it.combinations(range(len(v)), 2):
            a, b = v[i], v[j]; rest = [v[k] for k in range(len(v)) if k not in (i, j)]
            cands = [a + b, a * b, a - b, b - a]
            if b: cands.append(a / b)
            if a: cands.append(b / a)
            for x in cands:
                if x > 0 and x.denominator == 1 and go(rest + [x]):
                    return True
        return False
    return go(vals)


def _apply_move(nums, a: int, o: str, b: int):
    """nums 에서 a,b 를 빼고 a∘b 를 넣은 다중집합. 사용 불가(수가 없음·비정수)면 None."""
    from collections import Counter
    c = Counter(int(v) for v in nums)
    if c[a] == 0 or c[b] == 0 or (a == b and c[a] < 2):
        return None
    o = {"×": "*", "÷": "/"}.get(o, o)
    v = {"+": a + b, "*": a * b, "-": a - b, "/": (a // b if b and a % b == 0 else None)}[o]
    if v is None or v <= 0:
        return None
    c[a] -= 1; c[b] -= 1; rest = list(c.elements())
    return rest + [v]


def _multiset_has_pair(nums, a: int, b: int) -> bool:
    """`nums` 다중집합에서 a·b 를 **동시에** 뽑을 수 있는가 (같은 값이면 두 자리 필요).

    ★수리(0904, 감사결함4) 근거: `plan_followed` 는 "메타 뒤 첫 시도"를 정규식
    `_PAIR_RE` 로만 찾았다. 그 정규식은 산술 시도와 우연히 같은 모양인 산문
    ("3-4 ideas about it")도 구분 없이 잡는다. 두 수가 **원래 문제의 다중집합에
    실제로 있는 값**이라는 조건을 추가로 걸면, 적어도 "지목한 두 수 자체가
    이 문제에 등장하지도 않는" 우연한 문자열은 걸러진다(완전한 산문 구분은 정규식
    으로는 원리적으로 불가능하다 — 이 검사는 최소 방어선이다).
    """
    c = Counter(int(v) for v in nums)
    if a == b:
        return c[a] >= 2
    return c.get(a, 0) >= 1 and c.get(b, 0) >= 1


def plan_followed(after_text: str, nums, a: int, b: int) -> int:
    r"""메타 **뒤** 첫 시도가 (a,b) 를 실제로 결합하는가. 0/1.

    ★`plan_next` 에서 분리한 헬퍼(SC_DESIGN.md 배선 요구): `plan_next` 는 "next 가
    해를 살리는가"(완전열거 `_solvable`)와 "실제로 따라갔는가"를 **함께** 계산해서
    반환하는데, `countdown_selfcontrol.sc_row` 의 `followed` 는 완전열거 없이 이
    "따라갔는가"만 필요하다(SC 는 근거-진리를 안 쓴다 — SCg 만 `plan_ok` 를 따로
    붙인다). 완전열거를 매번 다시 도는 대신 이 부분만 떼어 두 자리에서 같은 함수를
    부른다 — 복제하면 "따라갔다"의 정의가 두 곳에서 갈릴 수 있다.
    """
    first = _PAIR_RE.search(after_text or "")
    return int(bool(first)
               and {int(first.group(1)), int(first.group(3))} == {int(a), int(b)}
               and _multiset_has_pair(nums, a, b))


def plan_next(text: str, nums, target: int) -> tuple[int, int]:
    """(plan_ok, plan_followed).

    첫 메타의 `next:` 를 읽어
      (a) plan_ok       그 첫수 뒤로 **원래 다중집합**(nums) 기준 목표 도달이 가능한가
                        (`_apply_move` 로 한 수를 접은 뒤 `_solvable` 완전열거).
      (b) plan_followed 메타 **뒤** 첫 시도가 그 두 수를 실제로 결합하는가.
    next 가 없거나 파싱 불가면 (0, 0).

    ★"next 는 원래 수 기준" 규약: `next:` 가 이름 붙인 두 수는 응답이 지금까지
    접어온 **중간값**이 아니라 항상 문제의 원래 다중집합 `nums` 안에서 찾는다.
    예를 들어 접두에서 이미 `25+3=28` 을 만들었어도 `next: 28*7` 은 (0, 0) 이다
    (28 이 nums 에 없으므로). 이는 의도된 설계다 — P3 프롬프트가 요구하는 것은
    "지금까지의 결합을 이어가는 계획"이 아니라 "**새로운** 부분 그룹화"이고, 중간값을
    허용하면 채점기가 임의의 트리 깊이를 추적해야 해서(응답 파싱만으로는 어떤
    중간값이 어떤 부분식에서 나왔는지 항상 복원되지 않는다) 검증 자체가 게임당하기
    쉬워진다.
    """
    m = parse_meta(text, "new")
    if not m.get("emitted"):
        return 0, 0
    body = m.get("body", "") or ""
    mm = _NEXT_RE.search(body)
    if not mm:
        return 0, 0
    # "5+8 or 25-7" 같은 복수 후보는 **첫 번째만** 본다(re.search 는 leftmost match).
    pr = _PAIR_RE.search(mm.group(1))
    if not pr:
        return 0, 0
    a, o, b = int(pr.group(1)), pr.group(2), int(pr.group(3))
    new = _apply_move(nums, a, o, b)
    ok = int(new is not None and _solvable(new, int(target)))
    after = text[int(m.get("end", 0)):] if m.get("end") else text.split("</meta>", 1)[-1]
    followed = plan_followed(after, nums, a, b)
    return ok, followed


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

    ★OPT/OPT_M (2026-09-05): `meta_form` 은 "new" 지만(파싱·텔레메트리는 다른 팔과
    같은 계기로 잡아야 한다) 메타 블록 자체는 선택이라 **형식 점수에 요구하면 안
    된다**. `spec["require_meta"]` 가 그 결합을 끊는다 — N0 처럼 `meta_form == "none"`
    으로 우회하면 파싱 계기까지 같이 꺼지므로(사양이 요구하는 텔레메트리를 잃는다)
    별도 플래그가 필요했다. 기존 팔은 전부 `require_meta` 를 안 적으므로 기본값
    True 로 지금까지의 동작이 한 글자도 안 바뀐다.
    """
    spec = _require_arm(arm)
    if spec["meta_form"] == "none":        # ★N0 맨 GRPO: 메타 형식을 요구하지 않는다 (메타 발화에 형식 점수를 주면 대조군이 아니다)
        return 1 if _bool01(parse_expr_ok(text)) else 0
    if not spec.get("require_meta", True):  # ★OPT/OPT_M: 메타는 허용이지 요구가 아니다
        return 1 if _bool01(parse_expr_ok(text)) else 0
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


# ★`osd_leak_guard` 는 제거했다(0825 적대검증). 살아 있는 가드는
#   `verl_sdc._osd_leak_guard` 이며 **토큰 id 기준** 8-그램이다(사양).
#   여기 있던 판본은 공백 분리 **단어** 기준이라 2~3배 헐거웠고 호출자가
#   0명인 죽은 코드였다. 두 판본이 공존하면 언젠가 느슨한 쪽이 채택된다.
#   누출 가드의 정의처는 스코어러 한 곳뿐이다.

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


def false_claim_rate(rows: Sequence[Mapping], *, form: str = "any",
                     key: str = "inv_false_claim") -> float:
    r"""메타가 «target 에 닿지 않는 식» 을 긍정형으로 선언한 비율(발화 행 기준).

    ★판정처는 `countdown_inv.false_claim_in_meta` **하나**다. 여기서는 스코어러가 이미
      행에 찍어 둔 결과를 셀 뿐이다 — 판정을 복제하면 두 곳이 갈리고, 갈리는 순간
      ABORT 가 보는 수와 보상이 보는 수가 달라진다(0825 키 불일치 사고의 모양).
    ★INV 스코어러가 안 돌면 이 키가 없다 ⇒ **NaN** 을 돌려주고 `check_abort` 가
      "missing" 으로 크게 남긴다. 조용한 0 은 «안 쟀다» 를 «깨끗하다» 로 위장시킨다.
    """
    em = [r for r in rows if _bool01(_meta_of(r, form)["emitted"])]
    vals = [r[key] for r in em if r.get(key) is not None]
    if not vals:
        return float("nan")
    return sum(1.0 for v in vals if _bool01(v)) / len(vals)


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
            # ★mean 은 G3(캘리브레이션 감시)의 원재료다. 신뢰도가 전부 0.9 쪽으로 붙으면
            #   "확신을 표하는 것" 자체가 값싼 전략이 된 것이고, 도치 자처럼 «정답을 아는
            #   눈에 그럴듯한가» 를 재는 항 아래에서 그것은 정확히 게이밍의 모양이다.
            "mean": (sum(vs) / len(vs)) if vs else float("nan"),
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


# ── R_meta 크기 계기 (C-012 가 만든 것) ───────────────────────────────────────
# ★새 메타 항은 **반드시** 여기 등록한다. 미등록 항은 크기계기·그룹분산분해·AUC 에서
#   통째로 사라져 "쟀는데 0" 과 "안 쟀다" 가 구별되지 않는다.
META_TERMS: tuple = ("meta_pos", "meta_mul", "meta_ctx", "gate", "len", "osd",
                     "meta_pos_full", "plan", INV_TERM,
                     "explore", "explore_g", "verify", "early_cost",
                     "timing", "timing2", "live_new", OPD_TERM, OPD_TERM_C, CF_TERM,
                     CF_GROUP_TERM)


def rmeta_magnitude(components: Sequence[Mapping[str, float]],
                    *, totals: Sequence[float] | None = None,
                    warmup: float | None = None,
                    dead_eps: float = 1e-12) -> dict:
    r"""메타 항의 **크기**를 분해한다. `component_means` 의 메타 전용 확대경이다.

    ★왜 평균만으로는 부족한가 (C-012).
      base b3p 붕괴의 근인은 "PMI-shift 가 작아서"가 아니었다. `pmi_shift` 는 무게 0.8 로
      메타 헤드 중 **가장 컸다**. 문제는 그것이 **그룹 중심화되어 평균이 정확히 0** 이라
      발화 자체를 끌어당기는 **순 견인력이 없었다**는 것이다. 반대편에는 `len_cost`·
      절단페널티·`R_corr` 이 전부 **무조건** 걸려 있었고, 유일한 무조건 견인력이던
      `meta_floor` 가 0.0 으로 꺼져 있었다. ⇒ 발화율 1.00 → 0.0137.

      따라서 세 숫자를 **따로** 봐야 한다:
        · `mean`     — 순 견인력. 0 이면 "발화하라"는 압력이 없다(C-012 의 그 칸).
        · `abs_mean` — 지급 규모. 이것이 0 이면 항이 아예 안 지급된다(무효 레버).
        · `std`      — GRPO 가 실제로 쓰는 양. 그룹 센터링 후 남는 것은 분산이다.
                       평균이 0 이 아니어도 분산이 0 이면 학습에 아무 영향이 없다.

      `mean≈0` 이면서 `std>0` 은 **정상**(재분배형 신호)이지만, 그때 발화를 지탱하는 것은
      `meta_floor` 뿐이므로 `meta_floor` 지급액을 같이 읽어야 한다 — `floor_vs_meta` 가 그것.

    Returns 항별 dict + 최상위 요약. 순수 함수이며 보상값을 바꾸지 않는다.
    """
    comps = [dict(c) for c in components]
    n = len(comps)
    if n == 0:
        return {"n": 0, "present": [], "verdict": ["no rows"]}

    def _stats(vs: Sequence[float]) -> dict:
        mu = _mean(vs)
        var = _mean([(v - mu) ** 2 for v in vs])
        av = [abs(v) for v in vs]
        return {
            "mean": mu,
            "abs_mean": _mean(av),
            "std": math.sqrt(max(0.0, var)),
            "frac_zero": _mean([1.0 if abs(v) <= dead_eps else 0.0 for v in vs]),
            "p95_abs": _quantile(av, 0.95),
            "max_abs": max(av) if av else 0.0,
        }

    present = [k for k in META_TERMS if any(k in c for c in comps)]
    out: dict = {"n": n, "present": present, "terms": {}}
    for k in present:
        out["terms"][k] = _stats([float(c.get(k, 0.0)) for c in comps])

    floor_abs = _mean([abs(float(c.get("meta_floor", 0.0))) for c in comps])
    meta_abs = sum(out["terms"][k]["abs_mean"] for k in present)
    out["meta_floor_abs_mean"] = floor_abs
    out["meta_abs_mean_total"] = meta_abs
    out["floor_vs_meta"] = (floor_abs / meta_abs) if meta_abs > dead_eps else float("inf")

    if totals is not None and len(totals) == n:
        tot_abs = _mean([abs(float(t)) for t in totals])
        out["total_abs_mean"] = tot_abs
        out["meta_share_of_total"] = (meta_abs / tot_abs) if tot_abs > dead_eps else float("nan")

    # ★warmup 구간(step<20)에는 처치 항이 축소돼 있다. warmup=0 이면 정의상 0 이므로
    #   그것을 «DEAD LEVER» 라고 부르면 첫 20스텝이 전부 오경보가 된다.
    out["warmup"] = float("nan") if warmup is None else float(warmup)
    in_warmup = warmup is not None and float(warmup) < 1.0
    zeroed_by_warmup = warmup is not None and float(warmup) <= dead_eps

    verdict: list = []
    if zeroed_by_warmup:
        out["verdict"] = ["warmup=0 — 처치 항이 아직 안 켜졌다(판정 대상 아님)"]
        return out
    for k in present:
        st = out["terms"][k]
        tag = " [warmup 중]" if in_warmup else ""
        if st["abs_mean"] <= dead_eps:
            verdict.append(f"{k}: DEAD LEVER — 지급액 0 (항이 켜졌는데 아무것도 안 준다){tag}")
        elif st["std"] <= dead_eps:
            verdict.append(f"{k}: DEAD LEVER — 분산 0 (상수는 GRPO 센터링에서 사라진다){tag}")
        elif abs(st["mean"]) <= 1e-6:
            verdict.append(f"{k}: 순 견인력 ~0 (재분배형). 발화는 meta_floor 가 지탱한다 — C-012")
    if floor_abs <= dead_eps and present:
        verdict.append("meta_floor: 0 — C-012 의 붕괴 조건(유일한 무조건 견인력 꺼짐)")
    out["verdict"] = verdict or ["ok"]
    return out


def group_variance_decomposition(group_components, *, eps: float = 1e-6,
                                 small_std: float = 0.05) -> dict:
    r"""그룹 **내** 분산 분해 — `norm_adv_by_std_in_grpo` 가 True 일 때의 실제 영향력.

    ★왜 `rmeta_magnitude` 만으로는 부족한가.
      verl 의 GRPO 는 `A_i = (R_i − μ_g) / (σ_g + 1e-6)` 이다(core_algos.py).
      **σ 로 나누는 순간 보상의 절대 크기는 버려진다** — 그룹마다 어드밴티지가 단위분산으로
      재정규화되므로, `W_META` 가 정하는 것은 «크기» 가 아니라 «그룹 내 분산에서 메타가
      차지하는 몫» 이 된다. 따라서 영향력을 재려면 **그룹 내 분산을 쪼개야** 한다.

      ⚠Countdown 에서 이것이 특히 위험한 이유: 오라클 측정에서 **p̂ 가 0 또는 1 인 그룹이
      64.8%** 다. 그 그룹에서는 `corr` 이 상수라 분산에 **한 톨도 기여하지 않는다** ⇒
      σ_g 가 사실상 메타(+format)만으로 만들어지고, 나누고 나면 그 그룹은
      **메타만으로 만든 단위분산 그라디언트**를 낸다. 정답을 하나도 못 맞힌 그룹이
      정답이 갈린 그룹과 **같은 세기로** 학습에 들어간다.

      그리고 ε 이 1e-6 로 매우 작아, σ_g 가 작은 그룹은 사소한 차이가 ±2~3 으로 증폭된다
      (예: 0.02 짜리 format 차이 하나가 σ=0.007 에서 −2.5 가 된다). `frac_tiny_std` 와
      `amp_p95` 가 그 위험을 잰다.

    Args:
        group_components: 그룹의 리스트. 각 그룹은 `arm_reward` 가 돌려준 성분 dict 의 리스트.

    Returns: 그룹 평균 통계 + 위험 지표. 순수 함수.
    """
    gs = [list(g) for g in group_components if g]
    if not gs:
        return {"n_groups": 0, "verdict": ["no groups"]}

    def _std(vs):
        if len(vs) < 2:
            return 0.0
        mu = _mean(vs)
        return math.sqrt(max(0.0, _mean([(v - mu) ** 2 for v in vs])))

    st, sm, sn, sc, shares, tiny, amps, corr_const = [], [], [], [], [], 0, [], 0
    for g in gs:
        tot  = [sum(float(v) for v in c.values()) for c in g]
        meta = [sum(float(c.get(k, 0.0)) for k in META_TERMS) for c in g]
        non  = [t - m for t, m in zip(tot, meta)]
        corr = [float(c.get("corr", 0.0)) for c in g]
        s_t, s_m, s_n, s_c = _std(tot), _std(meta), _std(non), _std(corr)
        st.append(s_t); sm.append(s_m); sn.append(s_n); sc.append(s_c)
        if s_t > eps:
            shares.append((s_m ** 2) / (s_t ** 2))
        if s_t < small_std:
            tiny += 1
        amps.append(1.0 / (s_t + eps))
        if s_c <= eps:
            corr_const += 1

    n = len(gs)
    out = {
        "n_groups": n,
        "std_total": _mean(st), "std_meta": _mean(sm),
        "std_nonmeta": _mean(sn), "std_corr": _mean(sc),
        "meta_var_share": _mean(shares) if shares else float("nan"),
        "frac_corr_constant": corr_const / n,
        "frac_tiny_std": tiny / n,
        "amp_p95": _quantile(amps, 0.95),
    }

    v = []
    if out["frac_corr_constant"] >= 0.8:
        v.append(f"corr 상수 그룹 {out['frac_corr_constant']:.0%} — 그 그룹의 σ 는 메타가 만든다")
    if out["meta_var_share"] == out["meta_var_share"] and out["meta_var_share"] >= 0.5:
        v.append(f"메타가 그룹 내 분산의 {out['meta_var_share']:.0%} — /σ 후 어드밴티지를 지배한다")
    if out["frac_tiny_std"] >= 0.1:
        v.append(f"σ<{small_std} 인 그룹 {out['frac_tiny_std']:.0%} — ε=1e-6 증폭 위험")
    out["verdict"] = v or ["ok"]
    return out


def meta_outcome_discrimination(rows, components, *, solved_key: str = "r_corr",
                                group_ids=None) -> dict:
    r"""★R_meta 가 «푼 롤아웃»과 «못 푼 롤아웃»을 가르는가 — AUC.

    **«좋은 메타인지» 를 런타임에 검정하는 유일한 지표다.** 나머지 텔레메트리(발화율·
    상투구·답누출·선택성)는 전부 «메타가 망가지지 않았나» 를 보는 **가드**이고, 「이 메타가
    좋은가」는 결국 «그 메타를 낸 롤아웃이 문제를 풀었나» 로만 검정된다.

    ⚠왜 이것이 이번 판의 핵심 계기인가 (docs/reports/2026-08-17 §1.5 실측, 수학 세대):

        PMI-shift (현행 공식)   AUC 0.457 [0.363, 0.559]   ← 못 가림
        meta_len (음성 대조군)  AUC 0.598 [0.489, 0.699]   ← 🚨현행보다 높다
        -surprise               AUC 0.656
        -confidence             AUC 0.635
        둘 합침                 AUC 0.714

      즉 **가짜 대조군이 우리 공식을 이겼다.** 그리고 이긴 조합에 PMI 를 도로 넣으면
      0.714 -> 0.677 로 떨어진다 — 보태는 게 아니라 잡음이었다.
      원인 진단은 §1.7 티처포싱 포화이고, **Countdown 의 짝지어진 오답이 그 결함을
      고친다는 것이 이 판의 가설이다.** 이 함수가 그 가설을 매 스텝 검정한다.

      읽는 법:  AUC ~ 0.5  -> R_meta 는 «좋은 메타» 를 못 가리고 있다. 그 상태에서
                             B/C/F 의 널은 «공식이 나쁘다» 가 아니라 «신호가 없다» 이다.
                AUC > 0.6  -> 짝지어진 오답이 실제로 결함을 고쳤다. 수학 대비 전진.

    Args:
        rows: 행 dict 리스트(각 행에 `solved_key`). `components` 와 **같은 순서·길이**.
        components: `arm_reward` 가 돌려준 성분 dict 리스트.

    Returns: auc / n_pos / n_neg / mean_pos / mean_neg / gap. 순수 함수.
    """
    rs, cs = list(rows), list(components)
    if len(rs) != len(cs):
        raise ValueError(f"rows({len(rs)}) 와 components({len(cs)}) 길이가 다르다 — 배선 버그다.")
    if group_ids is not None and len(group_ids) != len(rs):
        raise ValueError(f"group_ids({len(group_ids)}) 길이가 rows({len(rs)}) 와 다르다 — 배선 버그다.")
    has_meta_term = any(k in c for c in cs for k in META_TERMS)
    # ★그룹 내로 제한하는 이유 (파일럿 0821 에서 잡은 오독):
    #   전체를 한 통에 넣고 재면 «그룹 간 난이도» 가 «메타 품질» 로 둔갑한다.
    #   E 팔의 `gate = -(2p̂-1)` 은 **정의상** 어려운 그룹(=오답이 많은 그룹)에서 높으므로,
    #   통짜 AUC 는 항상 0.5 아래로 나오고 그것을 «거꾸로다» 라고 부르면 상시 오경보다.
    #   GRPO 는 애초에 **같은 그룹 안에서만** 비교하므로(그룹 평균을 뺀다) 지표도 그래야 한다.
    #   group_ids 를 주면 같은 그룹 쌍만 세고, 안 주면 통짜(=구 동작)로 돈다.
    pairs, tot_pairs = [], []
    for r, c in zip(rs, cs):
        m = sum(float(c.get(k, 0.0)) for k in META_TERMS)
        y = _bool01(r.get(solved_key, 0))
        pairs.append((m, y))
        tot_pairs.append((sum(float(v) for v in c.values()), y))
    pos = [m for m, y in pairs if y == 1]
    neg = [m for m, y in pairs if y == 0]
    out = {"n_pos": len(pos), "n_neg": len(neg),
           "mean_pos": _mean(pos) if pos else float("nan"),
           "mean_neg": _mean(neg) if neg else float("nan")}
    out["gap"] = (out["mean_pos"] - out["mean_neg"]) if (pos and neg) else float("nan")
    if not pos or not neg:
        # 한쪽이 비면 AUC 는 정의되지 않는다. 0.5 로 채우면 «못 가림» 으로 오독된다.
        out["scope"] = "within_group" if group_ids is not None else "pooled"
        out["auc"] = float("nan")
        out["auc_total"] = float("nan")
        out["inversion_rate"] = float("nan")
        out["verdict"] = ["정답/오답 한쪽이 비어 AUC 정의 불가 — 잴 수 없었다는 뜻이다"]
        return out
    gids = list(group_ids) if group_ids is not None else None
    out["scope"] = "within_group" if gids is not None else "pooled"
    out["auc"] = _auc(pairs, gids)
    # ★목표 부합의 직접 측정 — 총보상이 «푼 쪽»을 위로 매기는가.
    #   사전등록 §1 «메타인지는 목적이 아니라 정확도를 올리는 수단» 을 수치로 옮기면
    #   «총보상 AUC = 1.0» 이다. 1 에서 모자라는 만큼이 곧 **목표 역전율**이다.
    out["auc_total"] = _auc(tot_pairs, gids)
    out["inversion_rate"] = 1.0 - out["auc_total"]

    v = []
    # ★B6(감사 0821): `meta_mul`/`meta_ctx` 는 정의상 `... x sign(adv_corr)` 이고
    #   `adv_corr` 의 부호가 곧 AUC 의 라벨(`r_corr`) 이다 ⇒ AUC 가 **항상 1.000** 이 되는
    #   항진명제다. C·F 에서 «ok» 로 읽히면 판정이 순환논리가 된다. 해석 대상에서 뺀다.
    #   `osd` 도 같은 항진명제다: R_osd = y·clip(...) 이고 y 는 `r_corr` 에서 왔다.
    #   등록하지 않으면 OSD 팔 로그가 매 스텝 "AUC 1.000 ok" 로 읽히고 그 판정은
    #   순환논리다(그래서 폐기된 PMI-shift 보다 나빠 보이는 일이 절대 없다).
    sign_injected = sorted({"meta_mul", "meta_ctx", "osd"} & {k for c in cs for k in c})
    if sign_injected:
        out["auc_interpretable"] = False
        out["verdict"] = [f"{sign_injected} 은 결과 부호(meta_mul·meta_ctx=sign(adv_corr), "
                          "osd=y=±1(r_corr))가 곱해져 있어 AUC 라벨과 "
                          "순환한다 — auc 는 해석 대상이 아니다(C·F·OSD팔). "
                          f"auc_total/inversion 은 유효: {out['inversion_rate']:.1%}"]
        return out
    out["auc_interpretable"] = True
    # ★그룹 내로 제한하면 «전역엔 정답도 오답도 있는데 같은 그룹 안에는 없는» 경우가 생긴다
    #   (예: 어려운 그룹은 전원 오답, 쉬운 그룹은 전원 정답). 그때 AUC 는 NaN 이고,
    #   그것을 «ok» 로 흘리면 «쟀는데 괜찮았다» 로 오독된다 — 안 쟀다는 뜻이어야 한다.
    if out["auc"] != out["auc"]:
        scope_txt = "같은 그룹 안에" if gids is not None else ""
        out["verdict"] = [f"AUC 정의 불가 — {scope_txt} (정답, 오답) 쌍이 하나도 없다. "
                          "잴 수 없었다는 뜻이지 통과가 아니다"]
        return out
    if not has_meta_term:
        v.append("이 팔에는 메타 항이 없다 — auc 는 해석 대상이 아니다(A팔)")
    elif 0.45 <= out["auc"] <= 0.55:
        v.append(f"AUC {out['auc']:.3f} — R_meta 가 정답/오답을 못 가린다. "
                 "이 상태의 널은 «공식이 나쁘다» 가 아니라 «신호가 없다» 다")
    elif out["auc"] < 0.45:
        v.append(f"AUC {out['auc']:.3f} — ★거꾸로다. R_meta 가 **못 푼 쪽**을 높게 매긴다")
    if out["inversion_rate"] > 0.05:
        v.append(f"★목표 역전율 {out['inversion_rate']:.1%} — 총보상이 «못 푼 롤아웃» 을 "
                 "«푼 롤아웃» 위로 매기는 쌍이 그만큼이다. 처치 무게가 과하다는 뜻이다")
    out["verdict"] = v or [f"AUC {out['auc']:.3f} · 역전 {out['inversion_rate']:.1%} ok"]
    return out


def _auc(pairs, group_ids=None) -> float:
    """Mann-Whitney U -> AUC. 동점은 0.5 로 센다.

    group_ids 가 주어지면 **같은 그룹 안의 (정답, 오답) 쌍만** 센다 — 그룹 간 난이도
    교락을 제거한다. 비교 가능한 쌍이 하나도 없으면 NaN(0.5 로 채우면 «못 가림» 으로 오독).
    """
    if group_ids is not None:
        by_g: dict = {}
        for (m, y), g in zip(pairs, group_ids):
            by_g.setdefault(g, []).append((m, y))
        wins = ties = tot = 0.0
        for grp in by_g.values():
            ps = [m for m, y in grp if y == 1]
            ns = [m for m, y in grp if y == 0]
            for a in ps:
                for b in ns:
                    tot += 1
                    if a > b:
                        wins += 1
                    elif a == b:
                        ties += 1
        return (wins + 0.5 * ties) / tot if tot else float("nan")
    pos_n = sum(1 for _, y in pairs if y == 1)
    neg_n = len(pairs) - pos_n
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    vals = sorted((m, i) for i, (m, _) in enumerate(pairs))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[vals[k][1]] = avg
        i = j + 1
    rsum = sum(ranks[i] for i, (_, y) in enumerate(pairs) if y == 1)
    return (rsum - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n)


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
        "false_claim_rate": false_claim_rate(rows, form=form),
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

# ★2026-08-31 확장(G4). 셋에서 여섯으로. 앞의 셋은 한 글자도 안 바뀐다.
#   왜 늘렸나 — 설계검토 실측: 기존 셋 중 어느 것도 «확신에 찬 오답» 을 못 막는다.
#     · `arith_in_meta_rate` 는 **텔레메트리 전용**이라 보상식에도 여기에도 없었다.
#     · `answer_leak` 은 최종 boxed 식과의 문자열 일치라 **틀린 식은 원리적으로** 안 걸린다.
#     · `meta_form_ok` 은 confidence 가 «있기만» 하면 1 이라 0.99 를 매번 써도 통과한다.
#   새 셋은 그 구멍을 정확히 하나씩 막는다.
ABORT_RULES = {
    "emit_rate":        {"op": "<", "thr": 0.2,
                         "why": "발화가 무너지면 어떤 메타 팔도 잴 것이 없다"},
    "boilerplate_rate": {"op": ">", "thr": 0.5,
                         "why": "최빈 문장이 과반이면 메타가 아니라 상투구다"},
    "answer_leak_rate": {"op": ">", "thr": 0.1,
                         "why": "메타가 답을 담으면 메타 보상이 정답 보상의 사본이 된다"},
    "arith_in_meta_rate": {"op": ">", "thr": 0.02,
                           "why": "메타가 계산을 대신하기 시작하면 그것은 메타인지가 아니라 "
                                  "본문이다(실측 기저 0.001)"},
    "false_claim_rate": {"op": ">", "thr": 0.02,
                         "why": "닿지 않는 식을 «맞다»고 선언하는 것이 도치 자의 argmax 다 "
                                "— 이 수가 오르면 게이밍이 시작된 것이다"},
    "confidence_mean":  {"op": ">", "thr": 0.8,
                         "why": "신뢰도가 한쪽으로 붙으면 «확신 표하기» 가 값싼 전략이 된 것이다"},
}


# ★수리(0904, 감사결함5): `arith_in_meta_rate` 의 임계값 0.02 는 **P0 프롬프트로
#   실측한 기저**(주석 0.001)에서 정한 값이다. 다른 프롬프트 변형(P3 등)은 메타
#   본문에 산술이 섞이는 자연 기저율이 달라질 수 있는데, 그때마다 이 파일을 고쳐
#   재배포하면 "무엇을 기준으로 판단했는지"가 커밋 로그에만 남고 실행 로그에는
#   안 남는다. 환경변수로 오버라이드하고 `arm_signature()` 가 그 사실을 찍게 한다
#   ("선언은 있고 배선은 없음" 방지 원칙 — 이 파일 `arm_signature` docstring과 같다).
def _resolved_arith_threshold() -> float:
    """`arith_in_meta_rate` 중단 임계값 — `COUNTDOWN_ABORT_ARITH` 로 오버라이드."""
    v = os.environ.get("COUNTDOWN_ABORT_ARITH")
    return float(ABORT_RULES["arith_in_meta_rate"]["thr"]) if v is None else float(v)


def _abort_arith_overridden() -> bool:
    return os.environ.get("COUNTDOWN_ABORT_ARITH") is not None


def get_abort_patience() -> int:
    """중단 조건 연속 위반 허용 스텝 수 — `COUNTDOWN_ABORT_PATIENCE` 로 오버라이드(기본 3).

    ★verl_sdc.py 의 `_ABORT_STREAK` 판정이 이 값을 읽는다. 프롬프트 변형별로
    노이즈 수준이 달라 patience 를 조정해야 할 수 있어 P0 고정값을 벗어난다.
    """
    v = os.environ.get("COUNTDOWN_ABORT_PATIENCE")
    return 3 if v is None else int(v)


MIN_EMITTED_FOR_RATE_RULES = 64   # 선택 발화(require_meta=False) 팔에서 비율 중단 규칙을 판정할 최소 발화 행 수


def check_abort(report: Mapping, *, arm: str | None = None) -> list[dict]:
    r"""사양의 중단 조건을 재고, **위반 목록**을 돌려준다(빈 리스트 = 통과).

    지표가 없거나 NaN 이면 `missing` 로 보고한다 — 조용히 통과시키지 않는다.
    "상시 WARN 은 소음이 아니다"(원장 0731): 4건 중 2건이 실질이었다.

    ★OPT/OPT_M (2026-09-05): `arm` 을 주면 그 팔의 사양을 본다. `require_meta: False`
    인 팔은 `emit_rate < 0.2` 중단 규칙만 끈다 — 발화가 **선택**이므로 낮은 emit_rate
    가 붕괴가 아니라 "안 쓰기로 골랐다"일 수 있다. boilerplate/answer_leak/arith 규칙은
    그대로 둔다(발화된 행에 한해 조건부로 계산되는 지표라 emit_rate 와 성격이 다르다).
    `arm` 을 생략하면(기본값) 지금까지와 완전히 같게 여섯 규칙을 전부 본다 — 기존
    호출부(`verl_sdc.py`)를 건드리지 않아도 회귀가 없다.
    """
    out = []
    vals = {
        "emit_rate": report.get("emit_rate"),
        "boilerplate_rate": (report.get("boilerplate") or {}).get("boilerplate_rate"),
        "answer_leak_rate": report.get("answer_leak_rate"),
        "arith_in_meta_rate": report.get("arith_in_meta_rate"),
        "false_claim_rate": report.get("false_claim_rate"),
        "confidence_mean": (report.get("confidence") or {}).get("mean"),
    }
    skip = set()
    optional_meta = arm is not None and not _require_arm(arm).get("require_meta", True)
    if optional_meta:
        skip.add("emit_rate")
    # ★0905 OPT 실측: 발화가 6% 면 스텝당 발화 행이 ~30개라 «발화 행 기준 비율»(메타 내 산수·
    #   정형문·답 누출·거짓 주장·확신 평균)이 메타 1~2개로 임계를 넘나든다(0.056 > 0.02).
    #   비율 규칙은 표본이 충분할 때만 판정한다 — 그렇지 않으면 «못 쟀다»로 기록하고 죽이지 않는다.
    #   메타를 요구하는 팔은 발화 행이 수백이라 이 가드가 사실상 작동하지 않는다(행동 불변).
    _RATE_RULES = {"boilerplate_rate", "answer_leak_rate", "arith_in_meta_rate", "false_claim_rate", "confidence_mean"}
    try:
        n_emit = float(report.get("emit_rate") or 0.0) * float(report.get("n_rows") or 0)
    except (TypeError, ValueError):
        n_emit = 0.0
    for name, rule in ABORT_RULES.items():
        if name in skip:
            continue
        if optional_meta and name in _RATE_RULES and n_emit < MIN_EMITTED_FOR_RATE_RULES:
            out.append({"metric": name, "status": "missing", "value": vals.get(name),
                        "why": f"발화 행 {n_emit:.0f} < {MIN_EMITTED_FOR_RATE_RULES} — 비율 판정 표본 부족(선택 발화 팔)"})
            continue
        v = vals.get(name)
        if v is None or not _finite(v):
            out.append({"metric": name, "status": "missing", "value": v,
                        "why": "지표가 없다 — 안 쟀다는 뜻이지 통과가 아니다"})
            continue
        v = float(v)
        thr = _resolved_arith_threshold() if name == "arith_in_meta_rate" else rule["thr"]
        bad = (v < thr) if rule["op"] == "<" else (v > thr)
        if bad:
            out.append({"metric": name, "status": "abort", "value": v,
                        "threshold": thr, "op": rule["op"], "why": rule["why"]})
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
