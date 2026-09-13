#!/usr/bin/env bash
# scripts/local/run_math_arm.sh ARM SEED [STEPS=100] [--dry-run] [--check-labels-only]
#
# cd9 수학 무대: MATH_META 팔 하나를 GPU 하나에서 끝까지 돌린다(학습 → 판정 스텝 머지 →
# math500 롤아웃 평가). run_arm.sh(Countdown) 의 관례를 그대로 따르되 팔·데이터·모델만
# 수학으로 바꿨다 — gpu_queue.py 잡의 cmd 로 쓰이며 CUDA_VISIBLE_DEVICES 는 건드리지 않는다.
#
# ARM    src/training/math_meta.MATH_ARM_SPECS 키: M_G0 M_G1 M_JUDGE M_PROBE M_RAND
# SEED   data.seed (run_arm.sh 와 같은 이유로 rollout seed 는 못 건다)
# STEPS  trainer.total_training_steps (기본 100)
#
# LINEAGE = cd9_<ARM>_s<SEED>. 프롬프트 변형은 팔 명세가 정한다(M_G0=math_plain, 나머지
# math_opt) — 호출자가 고를 수 없다(Countdown E-134/E-135 의 «대조군 프롬프트 불일치» 방지).
#
# 환경변수(선택): MATH_JUDGE_LABELS(판단 라벨 json; M_JUDGE/M_RAND 는 없으면 트레이너가 즉사)
#   MATH_JUDGE_W(기본 0.5) MATH_ACC_FLOOR(사전등록 «acc < M_G0 − 1pp» 문턱; 설정 시 3-스텝
#   연속 중단 규칙에 참여) MODEL_PATH RESP_LEN(기본 4096) SLIM PAGED REF_OFFLOAD ACTOR_OFFLOAD
#   VLLM_UTIL. MATH_JUDGE_LABELS/MATH_JUDGE_W/MATH_ACC_FLOOR 는 verl_sdc.main 이 Ray runtime_env
#   로 실어 워커까지 전달한다(드라이버 export 만으로는 워커가 못 본다).
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
RESP_LEN="${RESP_LEN:-4096}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

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
DATA_VAL="${DATA_VAL:-${WORK}/data/math_val_${VARIANT}.parquet}"
CKPT_DIR="${WORK}/checkpoints/${LINEAGE}"
LOG_FILE="${WORK}/logs/${LINEAGE}.log"
MAX_PROMPT="${MAX_PROMPT:-1024}"
MAX_RESP="${MAX_RESP:-${RESP_LEN}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT + RESP_LEN + 256))}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((MAX_PROMPT + RESP_LEN + 256))}"

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
if [ -n "${MATH_JUDGE_LABELS:-}" ]; then export MATH_JUDGE_LABELS; fi
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
  "actor_rollout_ref.rollout.n=8"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "data.train_files=${DATA_TRAIN}"
  "data.val_files=${DATA_VAL}"
  "++data.seed=${SEED}"
  # ★Qwen3.5 하이브리드도 기본 thinking ON — math_rollout.py 와 같은 조건(enable_thinking=False).
  "+data.apply_chat_template_kwargs.enable_thinking=false"
  "data.train_batch_size=64"
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
  "trainer.resume_mode=auto"
  "trainer.save_freq=5"
  "trainer.test_freq=0"
  "++trainer.val_before_train=False"
  "++hydra.searchpath=[pkg://verl/trainer/config]"
)

echo "[run_math_arm] LINEAGE=${LINEAGE} ARM=${ARM} VARIANT=${VARIANT} SEED=${SEED} STEPS=${STEPS} RESP_LEN=${RESP_LEN} MODEL_PATH=${MODEL_PATH} MATH_JUDGE_LABELS=${MATH_JUDGE_LABELS:-unset} MATH_JUDGE_W=${MATH_JUDGE_W} MATH_ACC_FLOOR=${MATH_ACC_FLOOR:-unset}"
echo "[run_math_arm] data.train_files=${DATA_TRAIN}"
echo "[run_math_arm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset! queue should have set this>}"
echo "[run_math_arm] exact train command:"
printf '  %q' "${TRAIN_CMD[@]}"; echo
echo "[run_math_arm] log file: ${LOG_FILE}"

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
  EVAL_OUT="${WORK}/eval/${LINEAGE}/step_${STEP}/math500"
  python "${_SCRIPT_DIR}/math_rollout.py" --dataset math500 --model_path "${MERGED_DIR}" \
    --variant "${VARIANT}" --num_samples 8 --seed 11 --max_tokens "${RESP_LEN}" \
    --out_dir "${EVAL_OUT}" >> "${LOG_FILE}" 2>&1 \
    || echo "[run_math_arm] step ${STEP}: eval FAILED (see ${LOG_FILE})" >&2
done

echo "[run_math_arm] LINEAGE=${LINEAGE} done."
