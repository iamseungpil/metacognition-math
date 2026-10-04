#!/usr/bin/env bash
# scripts/local/run_math_arm.sh ARM SEED [STEPS=100] [--dry-run] [--check-labels-only]
#
# cd9 수학 무대: MATH_META 팔 하나를 GPU 하나에서 끝까지 돌린다(학습 → 판정 스텝 머지 →
# math500 롤아웃 평가). run_arm.sh(Countdown) 의 관례를 그대로 따르되 팔·데이터·모델만
# 수학으로 바꿨다 — gpu_queue.py 잡의 cmd 로 쓰이며 CUDA_VISIBLE_DEVICES 는 건드리지 않는다.
#
# ARM    src/training/math_meta.MATH_ARM_SPECS 키: M_G0 M_G1 M_JUDGE M_PROBE M_RAND M_RETRY
#        M_RETRY_RAND M_RETRY_SL M_AGREE M_AGREE_RAND M_CRIT M_DIS M_DIS_RAND M_DIS0
#        M_DIFF M_DIFF_RAND M_DIFF0 M_TRIAL2_CREDIT M_TRIAL2_OUTCOME M_TRIAL2_SCORE
# SEED   data.seed (run_arm.sh 와 같은 이유로 rollout seed 는 못 건다)
# STEPS  trainer.total_training_steps (기본 100)
#
# LINEAGE = cd9_<ARM>_s<SEED>. 프롬프트 변형은 팔 명세가 정한다(M_G0=math_plain, 나머지
# math_opt) — 호출자가 고를 수 없다(Countdown E-134/E-135 의 «대조군 프롬프트 불일치» 방지).
#
# 환경변수(선택): MATH_JUDGE_LABELS(판단 라벨 json; M_JUDGE/M_RAND 는 없으면 트레이너가 즉사)
#   ACC_FLOOR_FROM=<lineage>(설정 시 MATH_ACC_FLOOR 미설정이면 $WORK/eval/<lineage>/step_30/
#   math500/telemetry.json 의 acc-0.01 로 MATH_ACC_FLOOR 를 자동 채운다; 파일 없으면 즉사)
#   MATH_JUDGE_W(기본 0.5) MATH_ACC_FLOOR(사전등록 «acc < M_G0 − 1pp» 문턱; 설정 시 3-스텝
#   연속 중단 규칙에 참여) MODEL_PATH RESP_LEN(기본 4096) SLIM PAGED REF_OFFLOAD ACTOR_OFFLOAD
#   VLLM_UTIL. MATH_JUDGE_LABELS/MATH_JUDGE_W/MATH_ACC_FLOOR 는 verl_sdc.main 이 Ray runtime_env
#   로 실어 워커까지 전달한다(드라이버 export 만으로는 워커가 못 본다).
# ★M_RETRY_SL(0914): M_RETRY 와 발사 규약 동일(RESP_LEN 6144·math_retry_eval.py). 추가로
#   MATH_SL_W(기본 0.5)·MATH_SL_WARMUP_STEPS(기본 10)가 Ray env 로 워커에 실린다 — 결정 토큰
#   자기지도 항의 가중치와 «2배로 미는» 워밍업 구간이다.
# ★재시도류 팔(M_RETRY/M_RETRY_RAND/M_AGREE/M_AGREE_RAND, 사전등록 수정 3/0914b): RESP_LEN
#   미설정이면 기본 **6144**(두 번 풀어야 한다 — 4096 이면 두 번째 시도가 잘려 trunc_rate 규칙이
#   죽인다; 명시적 RESP_LEN 은 존중). MATH_RETRY_W(기본 0.5)·MATH_RETRY_LEN_COST(기본 0.2)도 Ray
#   env 로 워커에 실린다. M_AGREE/M_AGREE_RAND 는 추가로 MATH_AGREE_MON_W(기본 0.5)·
#   MATH_AGREE_CTL_W(기본 0.5)도 실린다. 사후 평가는 math_rollout.py 가 아니라
#   scripts/local/math_retry_eval.py(--variant math_retry|math_agree) 다 — math_rollout 은 첫
#   답/판단/재시도 분해가 없어 이 팔들의 판정 지표를 내지 못한다.
# ★M_CRIT(수정 6, 0914c): 비평 정보이득 팔. RESP_LEN 기본 **4096**(문맥 안 두 번째 시도가 없다),
#   사후 평가는 scripts/local/math_critique_eval.py(첫 답·비평 분해 + 답 없는 재풀이 + IG).
#   MATH_CRIT_W(기본 0.5)·MATH_CRIT_SCALE(기본 0.05 nats/tok)·MATH_CRIT_MAX_TOK(기본 2048)·
#   MATH_CRIT_SCORER_PATH(얼어붙은 채점기; 미설정이면 actor init)가 Ray env 로 워커에 실린다.
#   ★채점기(4B bf16 ≈ 8 GB)가 vLLM 과 같은 GPU 를 쓰므로 VLLM_UTIL 기본을 0.3 으로 낮춘다.
# ★M_DIS/M_DIS_RAND/M_DIS0(0914d): 불일치 진단 팔. RESP_LEN 기본 **2048**(진단 한 블록 + 답만),
#   MAX_PROMPT 기본 **2048**(후보 4개가 프롬프트에 들어간다), 사후 평가는
#   scripts/local/math_dis_eval.py(후보 재표집 → 진단 → 다수결 4/5 대조 + 토큰 비용).
#   데이터는 scripts/local/build_math_dis_parquet.py 가 만든 math_train_math_dis.parquet /
#   math_val_math_dis.parquet (DATA_TRAIN/DATA_VAL 기본값이 VARIANT=math_dis 로 자동으로 가리킨다).
#   MATH_DIS_W(기본 0.5)가 Ray env 로 워커에 실린다 — M_DIS0 은 팔 명세가 가중치 0 을 강제하므로
#   이 값을 무시한다. 판정 게이트는 `gate_judgment.py --eval-subdir math500_dis_8k --acc-key acc_dis`.
# ★M_DIFF/M_DIFF_RAND/M_DIFF0(0914e): 난이도 판단 팔. RESP_LEN 기본 **4096**(한 블록 + 한 번 풀기 —
#   문맥 안 재시도가 없다), MAX_PROMPT 기본 1024(프롬프트는 문제 + 지시 한 문단뿐), 사후 평가는
#   scripts/local/math_diff_eval.py(N=8 표집 → 캘리브레이션 + **같은 예산에서의 배분** 표).
#   데이터는 scripts/local/build_math_parquet.py --variant math_diff(레벨 섞인 기본 소스 —
#   배분은 쉬운 문제와 어려운 문제가 **같이 있어야** 값을 낸다; L5 전용 parquet 를 쓰면 안 된다).
#   MATH_DIFF_W(기본 0.5)가 Ray env 로 워커에 실린다 — M_DIFF0 은 팔 명세가 가중치 0 을 강제하므로
#   이 값을 무시한다. 판정 게이트는 `gate_judgment.py --eval-subdir math500_diff_8k --acc-key acc_first`.
# ★M_TRIAL2_CREDIT/M_TRIAL2_OUTCOME(S3, 0915, docs/DESIGN_S3_trial2_0915.md): 2-시도 팔.
#   시도 1 이 틀리면 **문맥을 통째로 버리고**(G8: 본문 제시 −10.4pp · 사실만 고지 +3.4pp) 사실 줄 +
#   자기 반성문 한 문장만 들고 다시 푼다. 슬롯 규약상 **ROLLOUT_N=24**(=3K, K=8)이고 RESP_LEN 은
#   4096(두 시도가 서로 다른 행이라 M_RETRY 의 6144 가 필요 없다). 손잡이 GAMMA_TRAJ(기본 .6)·
#   NOTE_MAX_TOKENS(64)·NOTE_MODE(self|none|switch|notx|switch_notx)·RETRY_ONLY_WRONG(1)·
#   RETRY_GATE(wrong|agree|all, 미설정=현행)·RETRY_GATE_STATES·ATTEMPT1_KL_COEF 가 Ray env 로
#   워커에 실린다. 사후 평가는 math_rollout.py = **시도-1 pass@1**(판정 게이트 ① ≥ .744) —
#   EVAL_SCRIPT 를 바꾸지 않는 이유는 math500_8k 산출물이 M_G0/M_G1 과 **같은 자**여야 하기
#   때문이다. 게이트 ②(2-시도 정확도 vs G8 `.617`)는 학습이 끝난 뒤 따로 돌린다:
#     scripts/local/math_trial2_eval.py --stage two_trial --rollouts <난이도5 texts.jsonl> \
#         --model_path <merged ckpt> --k 8 --max_tokens 8192 --out_dir <eval>/trial2_two_8k
#   (같은 스크립트의 --stage attempt1 은 math_rollout.py 를 그대로 부르는 얇은 래퍼다.)
#   학습 중에는 [TRIAL2] two_trial_acc 텔레메트리로만 본다.
# ★M_TRIAL2_SCORE(H2, 0921, docs/PLAN_h2_twoturn_0921.md §1·§5): 밀도 1 리셋 2턴 팔.
#   M_TRIAL2_CREDIT 과 파이프라인·슬롯 기하(ROLLOUT_N=24=3K)·MAX_PROMPT(3072) 동일하고
#   시도-2 보상만 R2 + α⁺·max(R2−R1,0) − α⁻·max(R1−R2,0) 이다. 정책은 문제 안에서 자기 답의
#   정오를 못 가리므로(AUC .51~.55) 고르기를 배우게 하지 않고, 파괴율(.078 → 표적 .04)만
#   깎는다. RESP_LEN·A2_RESP_LEN 기본 **6144**(H1 과 같은 6k+6k 예산 — 재시도 중앙값 4,675
#   토큰이라 4096 이면 R2 가 «판단»이 아니라 «예산»으로 0 이 된다). 손잡이
#   SCORE_ALPHA_POS(1.0)·SCORE_ALPHA_NEG(2.0)가 Ray env 로 워커에 실린다.
# ★감사 2: 판단 팔(M_JUDGE/M_RAND)은 발사 전에 math_meta.check_labels_cover 로 라벨 키가 학습
#   parquet 의 문제와 교집합이 있는지 확인한다(0 이면 카운트를 찍고 즉사). --check-labels-only 는
#   그 검사만 하고 끝낸다(테스트·수동 확인용).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

usage() {
  echo "Usage: $0 ARM SEED [STEPS=100] [--dry-run]" >&2
  exit 1
}

DRY_RUN=0
CHECK_LABELS_ONLY=0
POSITIONAL=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --check-labels-only) CHECK_LABELS_ONLY=1 ;;
    *) POSITIONAL+=("$a") ;;
  esac
done
set -- "${POSITIONAL[@]:-}"
[ -z "${1:-}" ] && usage

ARM="${1:?ARM required}"
SEED="${2:?SEED required}"
STEPS="${3:-100}"
# ★재시도류 팔(M_RETRY*·M_AGREE*)은 두 번 풀 예산이 필요하다 — 미설정일 때만 6144, 명시적
#   RESP_LEN 은 그대로.
case "${ARM}" in
  M_RETRY|M_RETRY_RAND|M_RETRY_SL|M_AGREE|M_AGREE_RAND) _RESP_LEN_DEFAULT=6144 ;;
  # ★M_CRIT(수정 6): 문맥 안 두 번째 시도가 **없다**(프롬프트가 금지) — 4096 이면 충분하고,
  #   6144 로 두면 «비평 자리에서 다시 푸는» 여유만 준다.
  M_CRIT) _RESP_LEN_DEFAULT=4096 ;;
  # ★M_DIS*(0914d): 응답은 «진단 한 블록 + 답» 뿐이다(문제를 처음부터 다시 풀지 않는다 — 후보
  #   4개가 이미 프롬프트에 있다). 2048 이면 넉넉하고, 더 주면 진단 자리에서 다시 푸는 여유만 준다.
  #   ★대신 **프롬프트**가 길다(문제 + 후보 스케치 4×400자 + 지시) — MAX_PROMPT 를 2048 로 올린다.
  M_DIS|M_DIS_RAND|M_DIS0) _RESP_LEN_DEFAULT=2048 ;;
  # ★S3 2-시도 팔: 두 시도가 **서로 다른 행**이다(문맥을 잇지 않는다) — 한 행의 예산은
  #   M_G0 과 같은 4096 으로 충분하다(M_RETRY 의 6144 는 한 응답에 두 번 푸느라 필요했다).
  M_TRIAL2_CREDIT|M_TRIAL2_OUTCOME) _RESP_LEN_DEFAULT=4096 ;;
  # ★M_TRIAL2_SCORE(H2, 0921): 재시도류 팔의 6144 규약을 따른다 — 두 시도가 서로 다른 행이라도
  #   각각 6k 를 써야 총 12k 예산 비교(base 단일 패스 12,288)가 성립하고, 4096 은 재시도의
  #   57% 를 잘라 R2 를 예산 인공물로 만든다(0916 관측: 재시도 중앙값 4,675 토큰).
  M_TRIAL2_SCORE) _RESP_LEN_DEFAULT=6144 ;;
  # ★M_REV_*(0918 수정 6): 크레딧이 걸린 행동(첫 답 → 재검토 → 다른 답)이 일어나는 행은
  #   길다 — 참조 롤아웃에서 수정 행 평균 4,446 토큰(전체 평균 2,885 대비 +1,607).
  #   4096 이면 바로 그 행들만 골라 잘라내 처치가 사라진다. 참조 측정과 같은 8192 로 둔다.
  M_REV_CF|M_REV_PMI_GOLD|M_REV_PMI_COMBO|M_REV_PMI_CF|M_REV_PMI_CONF) _RESP_LEN_DEFAULT=8192 ;;
  *) _RESP_LEN_DEFAULT=4096 ;;
esac
case "${ARM}" in
  M_DIS|M_DIS_RAND|M_DIS0) _MAX_PROMPT_DEFAULT=2048 ;;
  # ★S3 2-시도 팔(0916): 세 단계가 **한 배치**에 실리고 그중 반성문 호출의 프롬프트가 가장
  #   길다 — verl 은 세 단계를 하나의 판(폭 = 프롬프트 폭 + 응답 폭)으로 concat 해야 하고
  #   E-131 가드는 그 프롬프트 폭이 data.max_prompt_length 와 **정확히 같기를** 요구한다
  #   (폭이 밀리면 모든 행의 응답 꼬리가 채점에서 사라진다). 예산 계산:
  #     시스템·chat 템플릿 ~40 + 문제 ≤ ~900 + 시도-1 꼬리 NOTE_CTX_MAX_TOKENS=1536
  #     + NOTE_ASK ~40 ≈ 2,520 → 여유 550 을 두고 **3072**.
  #   MAX_MODEL_LEN 은 아래에서 MAX_PROMPT + RESP_LEN + 256 = 7,424 로 따라 온다(vLLM 은
  #   반성문 호출에서 프롬프트 ~2.5k + 64 토큰만 쓰므로 넉넉하다). 학습 판이 5,120 → 7,168 로
  #   1.4배 넓어지니 큐 제출 시 need_mb 는 40000 → **46000** 으로 올린다.
  M_TRIAL2_CREDIT|M_TRIAL2_OUTCOME|M_TRIAL2_SCORE) _MAX_PROMPT_DEFAULT=3072 ;;
  *) _MAX_PROMPT_DEFAULT=1024 ;;
esac
RESP_LEN="${RESP_LEN:-${_RESP_LEN_DEFAULT}}"
# ★S3 2-시도 팔의 슬롯 규약: 한 문제당 3K 행(시도1 K + 반성문 K + 시도2 K)이 필요하다
#   (설계 §2 — verl 의 fit() 은 행 수를 늘릴 수 없으므로 rollout.n 을 3K 로 잡아 슬롯을 빌린다).
#   K=8 을 유지하려면 n=24. 다른 팔은 그대로 8.
case "${ARM}" in
  M_TRIAL2_CREDIT|M_TRIAL2_OUTCOME|M_TRIAL2_SCORE) ROLLOUT_N="${ROLLOUT_N:-24}" ;;
  *) ROLLOUT_N="${ROLLOUT_N:-8}" ;;
esac
# ★EVAL_MAX_TOKENS(0914 수리): 판정 스텝(30/50/100) math500 eval은 RESP_LEN(학습 rollout
#   예산 — M_G0/M_G1 4096)과 별개로 docs/RESULTS_cd9.md 의 사전학습 베이스라인(8192)과 같은
#   예산으로 재야 truncation 이 왜곡한 acc 를 baseline 과 비교 가능해진다(M_G1 s30 eval 에서
#   trunc_rate 11.5% 관측 — RESP_LEN=4096 eval 은 비교 불가였다). 그래서 모든 사후(post-train)
#   eval 은 RESP_LEN 과 무관하게 EVAL_MAX_TOKENS(기본 8192, 오버라이드 가능)를 쓴다.
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-8192}"
# eval 출력 폴더 이름에 예산을 새긴다(math500 -> math500_8k, math500_retry -> math500_retry_8k)
# — 그래야 다른 예산으로 다시 잰 eval 이 같은 폴더를 덮어쓰지 않고, gate_judgment.py 가 보는
# --eval-subdir 이 실제로 무슨 예산으로 채점됐는지 이름만 보고 알 수 있다.
EVAL_BUDGET_TAG="$((EVAL_MAX_TOKENS / 1024))k"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

# ★cd9-verl09: VERL_ENV=<venv root> 로 격리 env(verl 0.9 / vllm 0.20 / transformers 5.x —
#   Qwen3.5 학습용, docs/ENV_verl09.md)를 쓴다. env.sh 가 simplerl 을 PATH 앞에 붙인 뒤에
#   이 venv 의 bin 을 그 앞에 다시 붙여 `python`·`verl.model_merger`·math_rollout.py 가 전부
#   같은 env 에서 돌게 한다. 미설정이면 바이트 동일(simplerl).
if [ -n "${VERL_ENV:-}" ]; then
  if [ ! -x "${VERL_ENV}/bin/python" ]; then
    echo "[run_math_arm] FATAL: VERL_ENV=${VERL_ENV} has no bin/python" >&2
    exit 1
  fi
  export PATH="${VERL_ENV}/bin:${PATH}"
  export VIRTUAL_ENV="${VERL_ENV}"
  echo "[run_math_arm] VERL_ENV=${VERL_ENV} -> python=$(command -v python) ($(python -c 'import verl,vllm,transformers;print("verl",verl.__version__,"vllm",vllm.__version__,"tf",transformers.__version__)' 2>&1 | tail -1))"
fi

# ── ARM validity + prompt variant: 단일 진실 원천은 MATH_ARM_SPECS 다(fail closed). ──
if ! VARIANT=$(python -c "
from src.training.math_meta import MATH_ARM_SPECS
import sys
sys.exit(1) if '${ARM}' not in MATH_ARM_SPECS else print(MATH_ARM_SPECS['${ARM}']['variant'])
" 2>/dev/null); then
  echo "[run_math_arm] FATAL: ARM='${ARM}' not in src.training.math_meta.MATH_ARM_SPECS" >&2
  exit 1
fi

# ★0914 amendment 1: 학습 env(transformers 4.57/vllm 0.10)가 qwen3_5 를 모른다 → 정책은 Instruct-2507.
MODEL_PATH="${MODEL_PATH:-/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507}"
LINEAGE="cd9_${ARM}_s${SEED}"
if [ "${RESP_LEN}" != "4096" ]; then
  LINEAGE="${LINEAGE}_r${RESP_LEN}"
fi
CONFIG_NAME="${CONFIG_NAME:-countdown_6arm}"   # 예산/배관 절반만 쓴다; 처치 키는 아래 override
DATA_TRAIN="${DATA_TRAIN:-${WORK}/data/math_train_${VARIANT}.parquet}"   # env 재지정 허용(테스트·수동 검사)
# ★B8(0914): build_math_parquet.py 의 val 은 forced/mixed 접미사를 절대 달지 않는다(val 은
#   항상 unforced) — DATA_VAL 기본값도 접미사 없는 이름으로 고정한다.
DATA_VAL="${DATA_VAL:-${WORK}/data/math_val_${VARIANT}.parquet}"
# ★B1(0914): DATA_TRAIN 파일명이 build_math_parquet.py 의 `_f{F}`/`_fmix{F}` 접미사를 달고
#   있으면(강제탐색 parquet) LINEAGE 에도 그 접미사를 반영한다 — 그래야 forced 런과 unforced
#   런이 같은 ckpt_dir/resume 을 공유하지 않는다(둘은 학습 분포가 다른 별개의 실험이다).
# ★B2: echo 정규식은 `_f`(mix)?`(숫자)`.parquet` 만 매치(math_train_math_retry_f0.25.parquet 도
#   math_train_math_retry_fmix0.25.parquet 도 매치, math_train_math_retry.parquet 은 매치 안 함).
_data_suffix=""
if [[ "$(basename "${DATA_TRAIN}")" =~ _f(mix)?([0-9.]+)\.parquet$ ]]; then
  _forced_frac="${BASH_REMATCH[2]}"
  _data_suffix="_f${BASH_REMATCH[1]}${_forced_frac}"
  echo "[run_math_arm] forced_frac=${_forced_frac}"
fi
echo "[run_math_arm] data.val_files=${DATA_VAL}"
if [ -n "${_data_suffix}" ]; then
  LINEAGE="${LINEAGE}${_data_suffix}"
fi
CKPT_DIR="${WORK}/checkpoints/${LINEAGE}"
LOG_FILE="${WORK}/logs/${LINEAGE}.log"
MAX_PROMPT="${MAX_PROMPT:-${_MAX_PROMPT_DEFAULT}}"
# ★A2_RESP_LEN(S3, 0916): 시도 2 만 넓히는 손잡이(기본 = RESP_LEN → 기존과 바이트 동일).
#   verl 의 agent-loop 은 응답을 **판 폭**(`data.max_response_length`)으로 패딩·절단하므로
#   (agent_loop.py:775-789) 판 폭을 max(RESP_LEN, A2_RESP_LEN) 로 올려야 실제로 길어진다.
#   시도 1 은 판이 넓어져도 호출 단위 max_tokens(=A1_RESP_LEN, 기본 RESP_LEN)로 4096 에서
#   멈춘다(src/training/trial2.a1_max_tokens + capped_single_turn_agent). 남는 자리는
#   오른쪽 패딩이라 채점·마스크에 영향이 없다. MAX_MODEL_LEN/max_num_batched_tokens 도 같이 따라온다.
# LR/LR_WARMUP_STYLE: LR(기본 미설정 = 기존과 바이트 동일한 1e-6) 은 actor optim lr 오버라이드
#   값을 대체하고, LR_WARMUP_STYLE(기본 미설정 = 옵션 생략)은 설정 시(예: constant)
#   actor_rollout_ref.actor.optim.warmup_style=${LR_WARMUP_STYLE} 오버라이드를 추가한다.
# ★M_TRIAL2_SCORE(H2, 0921): 시도 2 도 6144 가 기본이다(위 RESP_LEN 기본과 짝) — 다른 팔은
#   종전대로 A2_RESP_LEN 기본 = RESP_LEN 이라 바이트 동일하다.
case "${ARM}" in
  M_TRIAL2_SCORE) A2_RESP_LEN="${A2_RESP_LEN:-6144}" ;;
  *) A2_RESP_LEN="${A2_RESP_LEN:-${RESP_LEN}}" ;;
esac
A1_RESP_LEN="${A1_RESP_LEN:-${RESP_LEN}}"
_BOARD_RESP="${RESP_LEN}"
if [ "${A2_RESP_LEN}" -gt "${_BOARD_RESP}" ]; then _BOARD_RESP="${A2_RESP_LEN}"; fi
export A2_RESP_LEN A1_RESP_LEN
MAX_RESP="${MAX_RESP:-${_BOARD_RESP}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT + MAX_RESP + 256))}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((MAX_PROMPT + MAX_RESP + 256))}"

# ★판단 팔은 라벨표 없이는 M_G1 과 바이트 동일해진다 — 트레이너도 즉사하지만 여기서 먼저 막는다.
case "${ARM}" in
  M_JUDGE|M_RAND)
    if [ -z "${MATH_JUDGE_LABELS:-}" ] && [ "${DRY_RUN}" = "0" ]; then
      echo "[run_math_arm] FATAL: ${ARM} 는 MATH_JUDGE_LABELS(judgment_labels.json) 가 필요하다" >&2
      exit 2
    fi
    ;;
esac

export RAY_TMPDIR="/hdd_data/seungpil/r/$(printf '%s' "${LINEAGE}" | md5sum | cut -c1-8)"
mkdir -p "${RAY_TMPDIR}" "${CKPT_DIR}" "${WORK}/logs"
unset RAY_ADDRESS || true

export WANDB_PROJECT="metacot-math-local"
export WANDB_NAME="${LINEAGE}"
export WANDB_RUN_ID="${LINEAGE}-1"
export WANDB_RESUME=allow
# ★워커 전달 대상(verl_sdc.main 의 _local_ray_env_vars 가 «설정된 것만» 싣는다).
export MATH_JUDGE_W="${MATH_JUDGE_W:-0.5}"
export MATH_RETRY_W="${MATH_RETRY_W:-0.5}"
export MATH_RETRY_LEN_COST="${MATH_RETRY_LEN_COST:-0.2}"
# ★M_AGREE/M_AGREE_RAND 전용 가중치(모니터링/통제) — 다른 팔은 읽지 않으니 항상 export 해도 무해하다.
export MATH_AGREE_MON_W="${MATH_AGREE_MON_W:-0.5}"
export MATH_AGREE_CTL_W="${MATH_AGREE_CTL_W:-0.5}"
# ★M_RETRY_SL 전용(결정 토큰 자기지도). 다른 팔은 읽지 않으니 항상 export 해도 무해하다.
export MATH_SL_W="${MATH_SL_W:-0.5}"
export MATH_SL_WARMUP_STEPS="${MATH_SL_WARMUP_STEPS:-10}"
# ★M_CRIT 전용(비평 정보이득). MATH_CRIT_SCORER_PATH 는 **얼어붙은 채점기**(초기 정책) 경로 —
#   미설정이면 트레이너가 actor init(MODEL_PATH)을 쓴다. 다른 팔은 읽지 않으니 항상 export 해도 무해.
export MATH_CRIT_W="${MATH_CRIT_W:-0.5}"
export MATH_CRIT_SCALE="${MATH_CRIT_SCALE:-0.05}"
export MATH_CRIT_MAX_TOK="${MATH_CRIT_MAX_TOK:-2048}"
if [ -n "${MATH_CRIT_SCORER_PATH:-}" ]; then export MATH_CRIT_SCORER_PATH; fi
# ★메모리: 얼어붙은 채점기(4B bf16 ≈ 8 GB)가 vLLM 과 **같은 GPU** 에 올라간다 — M_CRIT 은
#   VLLM_UTIL 기본을 0.3 으로 낮춘다(다른 팔은 0.35 그대로). 명시적 VLLM_UTIL 은 존중.
if [ "${ARM}" = "M_CRIT" ]; then VLLM_UTIL="${VLLM_UTIL:-0.3}"; fi
# ★M_DIS 전용(불일치 진단). 다른 팔은 읽지 않으니 항상 export 해도 무해하다. M_DIS0 은 이 값을
#   무시하고 가중치 0 을 쓴다(팔 명세가 정한다 — 런처가 고를 수 없다).
export MATH_DIS_W="${MATH_DIS_W:-0.5}"
# ★M_DIFF 전용(난이도 판단). 다른 팔은 읽지 않으니 항상 export 해도 무해하다. M_DIFF0 은 이 값을
#   무시하고 가중치 0 을 쓴다(팔 명세가 정한다 — 런처가 고를 수 없다).
export MATH_DIFF_W="${MATH_DIFF_W:-0.5}"
# ★M_REV_* 전용(0918 수정 6, 자발적 답 수정 증폭). 다른 팔은 읽지 않으니 항상 export 해도 무해하다.
#   MATH_REV_W       수정 구간 크레딧의 가중치(기본 1.0). 0 으로 두면 M_G1 과 바이트 동일한
#                    무효 레버가 된다 — 아래 사전 점검이 그 경우 즉사시킨다.
#   MATH_REV_ANCHOR  PMI 앵커(gold_x|self_mx|combo). **비워 두는 것이 기본**이다 — 팔 이름이
#                    앵커를 정한다(M_REV_PMI_GOLD=gold_x, M_REV_PMI_COMBO=combo). 여기서 덮으면
#                    두 PMI 팔이 같은 처치가 될 수 있으므로 의도적으로만 설정하라.
export MATH_REV_W="${MATH_REV_W:-1.0}"
#   MATH_REV_CAP_MULT  수정 구간 크레딧 **상한**(배치 평균 |답 어드밴티지|)의 배수(기본 1.0).
#                      0919 중간 분석: 크레딧 질량이 |advantage| 의 0.16% 뿐이라 레버가
#                      사실상 무력했다 — 라우팅은 그대로 두고 세기만 이 배수로 키운다.
export MATH_REV_CAP_MULT="${MATH_REV_CAP_MULT:-1.0}"
#   MATH_REV_MASS_SHARE 0 = 끔(기본, 종전 동작). >0 이면 수정 크레딧의 총 |어드밴티지| 질량을
#                      «배치 답 어드밴티지 질량 × 이 몫»으로 맞춘다 — 크레딧 행이 몇 개든
#                      정책에 닿는 비중이 일정하다(희소 행동에 평균 클립은 방향이 반대다).
#                      켜면 MATH_REV_CAP_MULT 는 **안전 클립**으로만 남는다.
export MATH_REV_MASS_SHARE="${MATH_REV_MASS_SHARE:-0}"
#   MATH_REV_W_PMI / MATH_REV_W_CF  혼합 팔(M_REV_PMI_CF)의 두 항 가중치(각 기본 1.0).
#                      한쪽을 0 으로 두면 그 팔은 순수 팔과 바이트 동일해진다(무효 레버) —
#                      의도적으로만 그렇게 하라. 다른 팔은 이 값을 읽지 않는다.
export MATH_REV_W_PMI="${MATH_REV_W_PMI:-1.0}"
export MATH_REV_W_CF="${MATH_REV_W_CF:-1.0}"
#   MATH_REV_CONFIRM   확인(confirm) 크레딧 **모드**. 0 = 끔(기본) / 1 = 첫 답이 **틀린**
#                      확인 행만(가짜 확인 — WHY_NO_GAIN_0920 §4) / 2 = 확인 행 전부(대조).
#                      비워 두면 팔 명세가 정한다(M_REV_PMI_CONF 는 1 을 강제) — 다른 팔은 0.
if [ -n "${MATH_REV_CONFIRM:-}" ]; then export MATH_REV_CONFIRM; fi
if [ -n "${MATH_REV_ANCHOR:-}" ]; then export MATH_REV_ANCHOR; fi
case "${ARM}" in
  M_REV_CF|M_REV_PMI_GOLD|M_REV_PMI_COMBO|M_REV_PMI_CF|M_REV_PMI_CONF)
    # ★무효 레버 방지(M_PROBE scorer·M_JUDGE 라벨 점검과 같은 규약): 가중치 0 이면 이 팔은
    #   결과만 보는 M_G1 과 바이트 동일해지는데 이름만 다르게 남는다. 발사 전에 즉사시킨다.
    if [ "$(awk -v w="${MATH_REV_W}" 'BEGIN{print (w+0==0)?1:0}')" = "1" ]; then
      echo "[run_math_arm] FATAL: ${ARM} 은 MATH_REV_W=0 이면 M_G1 과 바이트 동일한 무효 레버다" >&2
      exit 2
    fi
    ;;
esac
# ★S3 2-시도 팔 전용(docs/DESIGN_S3_trial2_0915.md). 다른 팔은 읽지 않으니 항상 export 해도 무해.
#   GAMMA_TRAJ      크로스-에피소드 할인(LaMer γ_traj, 기본 .6). **1 미만이어야** 한다 —
#                   그게 «시도 1 을 틀리는 편이 낫다»(사행)를 구조적으로 막는 부등식이다.
#   NOTE_MAX_TOKENS 노트 단계(반성문/방법 라벨) 생성 상한(기본 64)
#   NOTE_MODE       리셋 자리에 적는 내용. 기본값은 **self**(변경 없음).
#                     self        자기 반성문 한 문장
#                     none        사실만 재시도(= G8 `.617` 참조 팔)
#                     switch      자기 방법 라벨 + «다른 길로»(F1 `fact_switch` +.040)
#                     notx        «답은 X 가 아니다»(X=a1 의 boxed; F1 `fact_notx` +.069)
#                     switch_notx 위 둘을 그 순서로 이은 합성 팔
#   RETRY_ONLY_WRONG 1=오답 시도-1 행에만 시도 2(기본) · 0=전 행(거짓 경보 진단용)
#   RETRY_GATE      어느 a1 행이 시도 2 를 받는가. 기본 **wrong**(= 현행, RETRY_ONLY_WRONG 그대로).
#                     wrong  gold 채점이 틀린 행만
#                     agree  **gold 없이** 같은 문제 K 개 a1 답의 합의 상태로 고른다 —
#                            추론 프로토콜(`math_activation_gate.agreement_state`)과 같은 규칙.
#                            상태 ∈ RETRY_GATE_STATES 면 그 문제의 a1 행을 **정오와 무관하게** 전부.
#                     all    전 행
#   RETRY_GATE_STATES agree 게이트가 여는 상태(쉼표, 기본 DOMINANT,SPLIT,SCATTER,NOANS = ALL_SAME 제외)
#   ATTEMPT1_KL_COEF 시도-1 토큰 참조 KL 계수(미설정이면 트레이너의 기존 KL 설정 그대로)
#   SCORE_ALPHA_POS / SCORE_ALPHA_NEG  M_TRIAL2_SCORE 의 비대칭 Δ 계수(기본 1.0 / 2.0 = 2α⁺).
#                   둘 다 ≥ 0 이어야 하고(음수면 trial2.py 가 즉사) α⁻ 가 «맞던 것을 깨뜨리지
#                   않기»를 가르치는 레버다. 다른 팔은 읽지 않으니 항상 export 해도 무해하다.
export GAMMA_TRAJ="${GAMMA_TRAJ:-0.6}"
export SCORE_ALPHA_POS="${SCORE_ALPHA_POS:-1.0}"
export SCORE_ALPHA_NEG="${SCORE_ALPHA_NEG:-2.0}"
export NOTE_MAX_TOKENS="${NOTE_MAX_TOKENS:-64}"
export NOTE_MODE="${NOTE_MODE:-self}"
export RETRY_ONLY_WRONG="${RETRY_ONLY_WRONG:-1}"
# ★미설정≠빈 문자열 규약: RETRY_GATE 를 강제로 찍으면 RETRY_ONLY_WRONG=0 의 옛 의미(전 행)가
#   조용히 wrong 으로 덮인다. 설정된 것만 싣는다(미설정이면 trial2.retry_gate() 가
#   RETRY_ONLY_WRONG 에서 wrong|all 을 고른다).
if [ -n "${RETRY_GATE:-}" ]; then export RETRY_GATE; fi
if [ -n "${RETRY_GATE_STATES:-}" ]; then export RETRY_GATE_STATES; fi
if [ -n "${ATTEMPT1_KL_COEF:-}" ]; then export ATTEMPT1_KL_COEF; fi
# ★사후 평가 스크립트는 팔이 정한다(재시도류 팔 = math_retry_eval.py). dry-run 도 이 이름을 찍는다.
case "${ARM}" in
  M_RETRY|M_RETRY_RAND|M_RETRY_SL|M_AGREE|M_AGREE_RAND) EVAL_SCRIPT="math_retry_eval.py" ;;
  # ★M_CRIT 은 첫 답/비평 분해 + 답 없는 재풀이(비평 vs blind) + IG 를 내는 전용 평가가 필요하다.
  M_CRIT) EVAL_SCRIPT="math_critique_eval.py" ;;
  # ★M_DIS 는 «후보 4개 재표집 → 진단 → 다수결 대조(4/5)» 를 내는 전용 평가가 필요하다.
  M_DIS|M_DIS_RAND|M_DIS0) EVAL_SCRIPT="math_dis_eval.py" ;;
  # ★M_DIFF 는 «캘리브레이션 + 같은 예산에서의 표본 배분» 을 내는 전용 평가가 필요하다.
  M_DIFF|M_DIFF_RAND|M_DIFF0) EVAL_SCRIPT="math_diff_eval.py" ;;
  *) EVAL_SCRIPT="math_rollout.py" ;;
esac
if [ -n "${MATH_JUDGE_LABELS:-}" ]; then export MATH_JUDGE_LABELS; fi
# ★inert 레버 (ii, 0914): ACC_FLOOR_FROM=<lineage> 가 있으면 그 팔의 step_30/math500_${EVAL_BUDGET_TAG}
#   eval acc 를 읽어 MATH_ACC_FLOOR=acc-0.01 로 채운다(사전등록 «acc < M_G0 − 1pp»). 파일이 없으면
#   즉사(기준 팔이 아직 안 끝났는데 조용히 문턱 없이 도는 것을 막는다). MATH_ACC_FLOOR 를 직접
#   설정하면 이 자동 계산을 건너뛰고 그 값을 그대로 쓴다(명시값 우선).
#   ★EVAL_MAX_TOKENS 수리(0914b): 폴더 이름이 math500 -> math500_${EVAL_BUDGET_TAG} 로 바뀌었으므로
#   기준 팔도 같은 예산으로 재야 apples-to-apples 다(둘 다 기본 8192 아니면 명시적으로 맞춰 부를 것).
if [ -z "${MATH_ACC_FLOOR:-}" ] && [ -n "${ACC_FLOOR_FROM:-}" ]; then
  _floor_tel="${WORK}/eval/${ACC_FLOOR_FROM}/step_30/math500_${EVAL_BUDGET_TAG}/telemetry.json"
  if [ ! -s "${_floor_tel}" ]; then
    echo "[run_math_arm] FATAL: ACC_FLOOR_FROM=${ACC_FLOOR_FROM} 인데 ${_floor_tel} 가 없다 — 기준 팔 eval 이 아직 안 끝났다" >&2
    exit 1
  fi
  MATH_ACC_FLOOR="$(python -c "import json; print(json.load(open('${_floor_tel}'))['acc'] - 0.01)")"
  echo "[run_math_arm] ACC_FLOOR_FROM=${ACC_FLOOR_FROM} -> MATH_ACC_FLOOR=${MATH_ACC_FLOOR}"
fi
if [ -n "${MATH_ACC_FLOOR:-}" ]; then export MATH_ACC_FLOOR; fi

# ── ★감사 2: 판단 팔은 라벨↔학습 문제 교집합을 발사 전에 확인한다(교집합 0 → 즉사, 카운트 출력). ──
check_labels_cover() {
  python - "${DATA_TRAIN}" "${MATH_JUDGE_LABELS}" <<'PY'
import sys
from src.training.math_meta import check_labels_cover
try:
    st = check_labels_cover(sys.argv[1], sys.argv[2])
except RuntimeError as e:
    print(f"[run_math_arm] FATAL: {e}", file=sys.stderr); sys.exit(3)
print(f"[run_math_arm] labels cover: n_train={st['n_train']} n_labels={st['n_labels']} n_cover={st['n_cover']}")
PY
}
case "${ARM}" in
  M_JUDGE|M_RAND)
    if [ -n "${MATH_JUDGE_LABELS:-}" ] && { [ "${DRY_RUN}" = "0" ] || [ "${CHECK_LABELS_ONLY}" = "1" ]; }; then
      [ -s "${DATA_TRAIN}" ] || { echo "[run_math_arm] FATAL: missing/empty data file ${DATA_TRAIN} (labels check needs it)" >&2; exit 1; }
      check_labels_cover
    fi
    ;;
esac
if [ "${CHECK_LABELS_ONLY}" = "1" ]; then
  echo "[run_math_arm] --check-labels-only: done."
  exit 0
fi

TRAIN_CMD=(python -u -m src.training.verl_sdc
  "--config-name=${CONFIG_NAME}"
  "++mode=MATH_META"
  "++algorithm.math_arm=${ARM}"
  "trainer.experiment_name=${LINEAGE}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=1"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.actor.optim.lr=${LR:-1e-6}"
  "data.train_files=${DATA_TRAIN}"
  "data.val_files=${DATA_VAL}"
  "++data.seed=${SEED}"
  # ★Qwen3.5 하이브리드도 기본 thinking ON — math_rollout.py 와 같은 조건(enable_thinking=False).
  "++data.apply_chat_template_kwargs.enable_thinking=false"
  "data.train_batch_size=${TRAIN_BATCH:-64}"
  "data.max_prompt_length=${MAX_PROMPT}"
  "data.max_response_length=${MAX_RESP}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.35}"
  "actor_rollout_ref.actor.optim.optimizer_impl=$([ "${SLIM:-0}" = "1" ] && echo bitsandbytes.optim || echo torch.optim)"
  "actor_rollout_ref.actor.optim.optimizer=$([ "${SLIM:-0}" = "1" ] && { [ "${PAGED:-0}" = "1" ] && echo PagedAdamW8bit || echo AdamW8bit; } || echo AdamW)"
  "actor_rollout_ref.ref.fsdp_config.param_offload=$([ "${REF_OFFLOAD:-0}" = "1" ] && echo true || echo false)"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$([ "${ACTOR_OFFLOAD:-0}" = "1" ] && echo true || echo false)"
  "++actor_rollout_ref.model.enable_activation_offload=true"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "++trainer.total_training_steps=${STEPS}"
)
if [ -n "${LR_WARMUP_STYLE:-}" ]; then
  TRAIN_CMD+=( "actor_rollout_ref.actor.optim.warmup_style=${LR_WARMUP_STYLE}" )
fi
TRAIN_CMD+=(
  "trainer.resume_mode=auto"
  "trainer.save_freq=5"
  "trainer.test_freq=0"
  "++trainer.val_before_train=False"
  "++hydra.searchpath=[pkg://verl/trainer/config]"
)

echo "[run_math_arm] LINEAGE=${LINEAGE} ARM=${ARM} VARIANT=${VARIANT} SEED=${SEED} STEPS=${STEPS} RESP_LEN=${RESP_LEN} EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS} MODEL_PATH=${MODEL_PATH} MATH_JUDGE_LABELS=${MATH_JUDGE_LABELS:-unset} MATH_JUDGE_W=${MATH_JUDGE_W} MATH_ACC_FLOOR=${MATH_ACC_FLOOR:-unset} MATH_RETRY_W=${MATH_RETRY_W} MATH_RETRY_LEN_COST=${MATH_RETRY_LEN_COST} MATH_AGREE_MON_W=${MATH_AGREE_MON_W} MATH_AGREE_CTL_W=${MATH_AGREE_CTL_W} MATH_SL_W=${MATH_SL_W} MATH_SL_WARMUP_STEPS=${MATH_SL_WARMUP_STEPS} MATH_CRIT_W=${MATH_CRIT_W} MATH_CRIT_SCALE=${MATH_CRIT_SCALE} MATH_CRIT_MAX_TOK=${MATH_CRIT_MAX_TOK} MATH_CRIT_SCORER_PATH=${MATH_CRIT_SCORER_PATH:-unset} MATH_DIS_W=${MATH_DIS_W} MATH_DIFF_W=${MATH_DIFF_W} MATH_REV_W=${MATH_REV_W} MATH_REV_ANCHOR=${MATH_REV_ANCHOR:-by-arm} MATH_REV_CAP_MULT=${MATH_REV_CAP_MULT} MATH_REV_MASS_SHARE=${MATH_REV_MASS_SHARE} MATH_REV_W_PMI=${MATH_REV_W_PMI} MATH_REV_W_CF=${MATH_REV_W_CF} MATH_REV_CONFIRM=${MATH_REV_CONFIRM:-by-arm} ROLLOUT_N=${ROLLOUT_N} GAMMA_TRAJ=${GAMMA_TRAJ} SCORE_ALPHA_POS=${SCORE_ALPHA_POS} SCORE_ALPHA_NEG=${SCORE_ALPHA_NEG} A2_RESP_LEN=${A2_RESP_LEN} A1_RESP_LEN=${A1_RESP_LEN} NOTE_MODE=${NOTE_MODE} NOTE_MAX_TOKENS=${NOTE_MAX_TOKENS} RETRY_ONLY_WRONG=${RETRY_ONLY_WRONG} RETRY_GATE=${RETRY_GATE:-unset} RETRY_GATE_STATES=${RETRY_GATE_STATES:-unset} EVAL_SCRIPT=${EVAL_SCRIPT} LR=${LR:-1e-6} LR_WARMUP_STYLE=${LR_WARMUP_STYLE:-unset}"
echo "[run_math_arm] data.train_files=${DATA_TRAIN}"
echo "[run_math_arm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset! queue should have set this>}"
echo "[run_math_arm] exact train command:"
printf '  %q' "${TRAIN_CMD[@]}"; echo
echo "[run_math_arm] log file: ${LOG_FILE}"

# ★EVAL_MAX_TOKENS: 판정 스텝 eval 은 RESP_LEN 이 아니라 EVAL_MAX_TOKENS 로 돈다(위 주석).
#   dry-run 은 학습·머지를 안 돌리므로 실제 EVAL_CMD(STEP 별로 아래 루프에서 조립)를 낼 수
#   없지만, 예산 검증(테스트)을 위해 같은 인자로 조립되는 템플릿을 여기서 한 번 찍는다 —
#   --max_tokens 값만 실질 검증 대상이고 나머지는 STEP/MERGED_DIR 자리표시자다.
EVAL_SUBDIR_TEMPLATE="math500"
[ "${EVAL_SCRIPT}" = "math_retry_eval.py" ] && EVAL_SUBDIR_TEMPLATE="math500_retry"
[ "${EVAL_SCRIPT}" = "math_critique_eval.py" ] && EVAL_SUBDIR_TEMPLATE="math500_crit"
[ "${EVAL_SCRIPT}" = "math_dis_eval.py" ] && EVAL_SUBDIR_TEMPLATE="math500_dis"
[ "${EVAL_SCRIPT}" = "math_diff_eval.py" ] && EVAL_SUBDIR_TEMPLATE="math500_diff"
EVAL_SUBDIR_TEMPLATE="${EVAL_SUBDIR_TEMPLATE}_${EVAL_BUDGET_TAG}"
echo "[run_math_arm] eval command template (per judgment step, STEP substituted at runtime):"
printf '  %q' python "${_SCRIPT_DIR}/${EVAL_SCRIPT}" --dataset math500 \
  --model_path "${WORK}/merged/${LINEAGE}/step_<STEP>" --variant "${VARIANT}" \
  --num_samples 8 --seed 11 --max_tokens "${EVAL_MAX_TOKENS}" \
  --out_dir "${WORK}/eval/${LINEAGE}/step_<STEP>/${EVAL_SUBDIR_TEMPLATE}"
echo

if [ "${DRY_RUN}" = "1" ]; then
  echo "[run_math_arm] --dry-run: not executing."
  exit 0
fi

for f in "${DATA_TRAIN}" "${DATA_VAL}"; do
  if [ ! -s "${f}" ]; then
    echo "[run_math_arm] FATAL: missing/empty data file ${f} (run scripts/local/build_math_parquet.py --variant ${VARIANT} first)" >&2
    exit 1
  fi
done

echo "[run_math_arm] starting training, appending to ${LOG_FILE}" | tee -a "${LOG_FILE}"
STARTUP_LOCK="${QUEUE_ROOT:-/hdd_data/seungpil/queue}/.startup.lock"
STARTUP_HOLD_SEC="${STARTUP_HOLD_SEC:-300}"
exec 9>"${STARTUP_LOCK}"
flock 9
set +e
"${TRAIN_CMD[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
( sleep "${STARTUP_HOLD_SEC}"; flock -u 9 ) &
wait "${TRAIN_PID}"
TRAIN_RC=$?
flock -u 9 2>/dev/null || true
set -e

if [ "${TRAIN_RC}" != "0" ]; then
  echo "[run_math_arm] FATAL: training exited ${TRAIN_RC} (see ${LOG_FILE})" >&2
  exit "${TRAIN_RC}"
fi

# ── 판정 스텝: 머지 → math500 롤아웃 평가(학습과 같은 프롬프트 변형·채점기). ─────────
JUDGMENT_STEPS=(30 50 100)
for STEP in "${JUDGMENT_STEPS[@]}"; do
  ACTOR_DIR="${CKPT_DIR}/global_step_${STEP}/actor"
  [ -d "${ACTOR_DIR}" ] || continue
  MERGED_DIR="${WORK}/merged/${LINEAGE}/step_${STEP}"
  mkdir -p "${MERGED_DIR}"
  if python -m verl.model_merger merge --backend fsdp \
      --local_dir "${ACTOR_DIR}" --target_dir "${MERGED_DIR}" >> "${LOG_FILE}" 2>&1 \
     && [ -f "${MERGED_DIR}/config.json" ] && ls "${MERGED_DIR}"/*safetensors* >/dev/null 2>&1; then
    rm -f "${ACTOR_DIR}"/optim* "${ACTOR_DIR}"/model*
  else
    echo "[run_math_arm] step ${STEP}: merge FAILED — skipping eval" >&2
    continue
  fi
  EVAL_SUBDIR="math500"
  # ★재시도 팔은 math_retry_eval.py(첫 답/판단/재시도 분해·세 곡선). --max_tokens 규약은 같다.
  if [ "${EVAL_SCRIPT}" = "math_retry_eval.py" ]; then
    EVAL_SUBDIR="${EVAL_SUBDIR}_retry"
  elif [ "${EVAL_SCRIPT}" = "math_critique_eval.py" ]; then
    EVAL_SUBDIR="${EVAL_SUBDIR}_crit"
  elif [ "${EVAL_SCRIPT}" = "math_dis_eval.py" ]; then
    EVAL_SUBDIR="${EVAL_SUBDIR}_dis"
  elif [ "${EVAL_SCRIPT}" = "math_diff_eval.py" ]; then
    EVAL_SUBDIR="${EVAL_SUBDIR}_diff"
  fi
  # ★EVAL_MAX_TOKENS 예산을 폴더 이름에 새긴다(math500_8k / math500_retry_8k) — RESP_LEN 기반
  #   구eval(math500/math500_retry, 4096·6144 예산)과 안 섞이고 gate_judgment.py 가 어느 예산인지
  #   폴더명만으로 안다.
  EVAL_SUBDIR="${EVAL_SUBDIR}_${EVAL_BUDGET_TAG}"
  EVAL_OUT="${WORK}/eval/${LINEAGE}/step_${STEP}/${EVAL_SUBDIR}"
  python "${_SCRIPT_DIR}/${EVAL_SCRIPT}" --dataset math500 --model_path "${MERGED_DIR}" \
    --variant "${VARIANT}" --num_samples 8 --seed 11 --max_tokens "${EVAL_MAX_TOKENS}" \
    --out_dir "${EVAL_OUT}" >> "${LOG_FILE}" 2>&1 \
    || echo "[run_math_arm] step ${STEP}: eval FAILED (see ${LOG_FILE})" >&2
done

echo "[run_math_arm] LINEAGE=${LINEAGE} done."
