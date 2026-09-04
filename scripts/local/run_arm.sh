#!/usr/bin/env bash
# scripts/local/run_arm.sh ARM SEED [STEPS=100] [VARIANT] [--dry-run]
#
# Run ONE Countdown RL arm end-to-end on ONE GPU: train -> merge judgment steps
# -> held-out eval -> HF upload. Designed to be the "cmd" of one gpu_queue.py job
# (the queue sets CUDA_VISIBLE_DEVICES; this script does not touch it, so it runs
# on whatever GPU it is handed).
#
# ARM      one of src/training/countdown_rewards.ARM_SPECS keys, e.g. N0 A PL B C D E F
# SEED     int, used for data.seed (below is why NOT actor_rollout_ref.rollout.seed:
#          verl 0.7.1's rollout.yaml — checked under
#          /hdd_data/seungpil/envs/simplerl/lib/python3.10/site-packages/verl/trainer/config/rollout/rollout.yaml —
#          has NO `seed` key at all. `data.seed` DOES exist
#          (.../trainer/config/data/legacy_data.yaml:65, default null), so that is
#          the only seed override this script can verify. Rollout sampling
#          randomness therefore is NOT separately seeded per arm/run here —
#          UNVERIFIED whether that matters at this budget; noted, not fixed.)
# STEPS    trainer.total_training_steps (default 100)
# VARIANT  prompt/data variant for non-N0 arms (default p3); ignored for N0, which
#          always uses the `plain` variant (countdown_rewards.ARM_SPECS["N0"] note:
#          "메타 지시문 없음... variant plain, DATA_SUFFIX=_4num_plain").
#
# LINEAGE = cd7_<ARM>_<VARIANT>_s<SEED>   (VARIANT here is the EFFECTIVE data variant,
#           i.e. "plain" for N0, so lineages stay unambiguous.)
#
# Isolation: this job gets its own RAY_TMPDIR (so its Ray head process's sockets
# don't collide with sibling jobs on other GPUs of this box) and its own Ray
# instance (no RAY_ADDRESS is set — verl_sdc.py's main() only takes the
# "attach to existing cluster" branch when RAY_ADDRESS is present; see
# src/training/verl_sdc.py:4918-4934 vs 4942-4964). `ray stop --force` is NEVER
# called from this script because other jobs' Ray instances share the box.
#
# UNVERIFIED / risk noted, not solved here: verl_sdc.py's own `ray.init()` call
# (no RAY_ADDRESS branch) does not pin a GCS port, so two concurrent invocations
# of this script MAY race for Ray's default port unless Ray's own auto-port-pick
# on conflict is reliable at this Ray version. Not fixed because the task's env
# var/isolation list did not ask for a port scheme and verl_sdc.py is out of
# scope (another agent owns it concurrently).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

usage() {
  echo "Usage: $0 ARM SEED [STEPS=100] [VARIANT] [--dry-run]" >&2
  exit 1
}

DRY_RUN=0
POSITIONAL=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    *) POSITIONAL+=("$a") ;;
  esac
done
set -- "${POSITIONAL[@]:-}"
[ -z "${1:-}" ] && usage

ARM="${1:?ARM required}"
SEED="${2:?SEED required}"
STEPS="${3:-100}"
VARIANT_ARG="${4:-p3}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

# ── ARM validity (fail closed like the cluster launcher does — see
#    countdown_rl_6arm.yaml's `case "$ARM" in [ABCEFG]) ;; *) FATAL` gate; the
#    valid set here is the current ARM_SPECS keys, which is a superset of the
#    6-arm cluster round). ─────────────────────────────────────────────────
if ! ARM_OK=$(python -c "
from src.training.countdown_rewards import ARM_SPECS
import sys
sys.exit(0 if '${ARM}' in ARM_SPECS else 1)
" 2>&1); then
  echo "[run_arm] FATAL: ARM='${ARM}' not in src.training.countdown_rewards.ARM_SPECS" >&2
  exit 1
fi

if [ "${ARM}" = "N0" ]; then
  DATA_VARIANT="plain"
else
  DATA_VARIANT="${VARIANT_ARG}"
fi

# ── data_hint (고정 자리, 0904). ARM_SPECS[ARM] 이 "mixed" 를 선언하면(현재 M0/MT)
#    site(3000)+normal(3000) 을 섞은 mixed_train.parquet 로 발사해야 한다 — 항 이름이
#    같아도(M0 는 A 와 동일) 데이터가 다르면 다른 실험이다(countdown_rewards.py
#    ARM_SPECS 의 M0 note 참조). 다른 팔은 이 키가 없으므로 기본값 "normal" —
#    "지금까지"(일반 롤아웃 전용 parquet)와 바이트 동일하게 행동한다.
DATA_HINT=$(python -c "
from src.training.countdown_rewards import ARM_SPECS
print(ARM_SPECS['${ARM}'].get('data_hint', 'normal'))
")

if [ "${DATA_HINT}" = "mixed" ]; then
  LINEAGE="cd7_${ARM}_${DATA_VARIANT}_s${SEED}_mixed"
else
  LINEAGE="cd7_${ARM}_${DATA_VARIANT}_s${SEED}"
fi
CONFIG_NAME="${CONFIG_NAME:-countdown_6arm}"
if [ "${DATA_HINT}" = "mixed" ]; then
  # site(3000, 프리픽스가 이미 프롬프트에 접합) + normal(3000, 빈 assistant 메시지
  # 부착) 를 섞은 고정 자리 학습 parquet. countdown_sites.py 헤더 참조.
  DATA_TRAIN="${WORK}/data/sites_v1/mixed_train.parquet"
else
  DATA_TRAIN="${WORK}/data/countdown_train_4num_${DATA_VARIANT}.parquet"
fi
DATA_VAL="${WORK}/data/countdown_val_4num_${DATA_VARIANT}.parquet"
CKPT_DIR="${WORK}/checkpoints/${LINEAGE}"
LOG_FILE="${WORK}/logs/${LINEAGE}.log"
# ── 예산 (고정 자리 처치, 0904 사용자 지시값). site 프롬프트는 프리픽스가 이미
#    접합돼 있어 일반 프롬프트보다 길다 — prompt/response/model_len 전부 키운다.
#    "${VAR:-default}" 라 호출자가 이미 env 로 값을 줬으면 그쪽을 존중한다
#    (그래서 data_hint 분기를 그 **기본값**에만 건다 — "지금까지"와 바이트 동일한
#    경로는 이 분기가 없어도 원래 기본값 그대로다).
if [ "${DATA_HINT}" = "mixed" ]; then
  MAX_PROMPT="${MAX_PROMPT:-2048}"
  MAX_RESP="${MAX_RESP:-2048}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-4352}"
  MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4352}"
else
  MAX_PROMPT="${MAX_PROMPT:-1024}"
  MAX_RESP="${MAX_RESP:-2560}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
  MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
fi

export RAY_TMPDIR="/hdd_data/seungpil/ray/${LINEAGE}"
mkdir -p "${RAY_TMPDIR}" "${CKPT_DIR}" "${WORK}/logs"
unset RAY_ADDRESS || true

export WANDB_PROJECT="metacot-countdown-local"
export WANDB_NAME="${LINEAGE}"
export WANDB_RUN_ID="${LINEAGE}-1"
export WANDB_RESUME=allow

TRAIN_CMD=(python -u -m src.training.verl_sdc
  "--config-name=${CONFIG_NAME}"
  "++mode=COUNTDOWN_6ARM"
  "++algorithm.countdown_arm=${ARM}"
  "trainer.experiment_name=${LINEAGE}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=1"
  "actor_rollout_ref.model.path=/hdd_data/seungpil/scratch/models/Qwen3-4B"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.n=8"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "data.train_files=${DATA_TRAIN}"
  "data.val_files=${DATA_VAL}"
  "++data.seed=${SEED}"
  # ★0904: Qwen3 하이브리드는 기본이 thinking ON 이라 <meta> 발화가 0 이 된다(cd6 prereg A.2 와 동일 결정).
  #   학습 전 평가(countdown_gs0_eval.py)도 enable_thinking=False 라 이 옵션이 있어야 조건이 같다.
  "+data.apply_chat_template_kwargs.enable_thinking=false"
  "data.train_batch_size=64"
  "data.max_prompt_length=${MAX_PROMPT}"
  "data.max_response_length=${MAX_RESP}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.35"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
  "++actor_rollout_ref.model.enable_activation_offload=true"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "++trainer.total_training_steps=${STEPS}"
  "trainer.resume_mode=auto"
  "trainer.save_freq=5"
  "trainer.test_freq=0"
  "++trainer.val_before_train=False"
  "++hydra.searchpath=[pkg://verl/trainer/config]"
)

if [ "${DATA_HINT}" = "mixed" ]; then
  # ★고정 자리 재개(site 행) — 마지막 메시지가 이미 assistant(프리픽스)다. 이 두
  #   키가 verl 0.7.1 agent-loop 로 실제 도달하려면 sitecustomize.py 의
  #   `_patch_verl_agent_loop_chat_template` 이 걸려 있어야 한다 — 패치 없이 이
  #   override 만 주면 `AgentLoopBase.apply_chat_template` 이
  #   `add_generation_prompt` 중복 키워드로 즉사한다(그 함수 조사 기록 참조).
  TRAIN_CMD+=(
    "+data.apply_chat_template_kwargs.continue_final_message=true"
    "+data.apply_chat_template_kwargs.add_generation_prompt=false"
    # ★verl 0.7.1 rl_dataset.py:maybe_filter_out_long_prompts 의 `doc2len` 도
    #   같은 하드코딩(`add_generation_prompt=True` + `**apply_chat_template_kwargs`)
    #   버그를 갖고 있는데, 그쪽은 예외를 **삼키고**(broad except) 그 행을
    #   "너무 길다"로 오분류해 조용히 필터링한다 — 이 버그는 크래시가 아니라
    #   site 행 3000개가 전부 소리 없이 사라지는 데이터 유실이라 sitecustomize
    #   패치로도 못 막는다(그 메서드는 agent-loop 가 아니다). 필터를 꺼서 그
    #   경로 자체를 피한다 — max_prompt_length 초과를 걸러야 할 이유가 애초에
    #   없다(이미 2048 로 넉넉히 키웠다).
    "data.filter_overlong_prompts=false"
  )
fi

echo "[run_arm] LINEAGE=${LINEAGE} ARM=${ARM} SEED=${SEED} STEPS=${STEPS} DATA_VARIANT=${DATA_VARIANT} DATA_HINT=${DATA_HINT}"
echo "[run_arm] data.train_files=${DATA_TRAIN}"
echo "[run_arm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset! queue should have set this>}"
echo "[run_arm] exact train command:"
printf '  %q' "${TRAIN_CMD[@]}"; echo
echo "[run_arm] log file: ${LOG_FILE}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[run_arm] --dry-run: not executing."
  exit 0
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[run_arm] WARNING: CUDA_VISIBLE_DEVICES is unset. This script expects the queue" >&2
  echo "[run_arm]          worker to have set it to a single GPU index (0-3)." >&2
fi

for f in "${DATA_TRAIN}" "${DATA_VAL}"; do
  if [ ! -s "${f}" ]; then
    echo "[run_arm] FATAL: missing/empty data file ${f} (run scripts/local/make_data.sh first)" >&2
    exit 1
  fi
done

echo "[run_arm] starting training, appending to ${LOG_FILE}" | tee -a "${LOG_FILE}"
# ── Ray 기동 직렬화. 0904 실측: 4 잡이 동시에 ray.init 하면 «node timed out during startup»
#    (raylet/agent 등록 경합)로 죽는다. 기동 창(기본 300초) 동안만 전역 flock 을 잡고,
#    그 뒤에는 잠금을 풀어 다른 잡이 기동하게 한다. 학습 자체는 병렬이다.
STARTUP_LOCK="${QUEUE_ROOT:-/hdd_data/seungpil/queue}/.startup.lock"
STARTUP_HOLD_SEC="${STARTUP_HOLD_SEC:-300}"
exec 9>"${STARTUP_LOCK}"
echo "[run_arm] waiting for startup lock ${STARTUP_LOCK}" | tee -a "${LOG_FILE}"
flock 9
echo "[run_arm] startup lock acquired $(date -Is); holding ${STARTUP_HOLD_SEC}s" | tee -a "${LOG_FILE}"
set +e
"${TRAIN_CMD[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
( sleep "${STARTUP_HOLD_SEC}"; flock -u 9; echo "[run_arm] startup lock released $(date -Is)" >> "${LOG_FILE}" ) &
wait "${TRAIN_PID}"
TRAIN_RC=$?
flock -u 9 2>/dev/null || true
set -e

if [ "${TRAIN_RC}" != "0" ]; then
  echo "[run_arm] FATAL: training exited ${TRAIN_RC} (see ${LOG_FILE})" >&2
  exit "${TRAIN_RC}"
fi
echo "[run_arm] training exited 0"

# ── merge + prune + eval + upload, per judgment step present on disk. ───────
JUDGMENT_STEPS=(30 50 100)
for STEP in "${JUDGMENT_STEPS[@]}"; do
  GS_DIR="${CKPT_DIR}/global_step_${STEP}"
  ACTOR_DIR="${GS_DIR}/actor"
  if [ ! -d "${ACTOR_DIR}" ]; then
    echo "[run_arm] step ${STEP}: no ${ACTOR_DIR}, skipping (arm may not have trained this far)"
    continue
  fi

  MERGED_DIR="${WORK}/merged/${LINEAGE}/step_${STEP}"
  mkdir -p "${MERGED_DIR}"
  echo "[run_arm] step ${STEP}: merging ${ACTOR_DIR} -> ${MERGED_DIR}"
  # verl 0.7.1 CLI confirmed at
  # /hdd_data/seungpil/envs/simplerl/lib/python3.10/site-packages/verl/model_merger/__main__.py
  if python -m verl.model_merger merge --backend fsdp \
      --local_dir "${ACTOR_DIR}" --target_dir "${MERGED_DIR}" \
      >> "${LOG_FILE}" 2>&1; then
    if [ -f "${MERGED_DIR}/config.json" ] && ls "${MERGED_DIR}"/*safetensors* >/dev/null 2>&1; then
      echo "[run_arm] step ${STEP}: merge verified (config.json + safetensors present) — pruning shards"
      rm -f "${ACTOR_DIR}"/optim* "${ACTOR_DIR}"/model*
    else
      echo "[run_arm] step ${STEP}: merge command exited 0 but config.json/safetensors missing in ${MERGED_DIR} — NOT pruning ${ACTOR_DIR}" >&2
      continue
    fi
  else
    echo "[run_arm] step ${STEP}: merge FAILED (see ${LOG_FILE}) — NOT pruning ${ACTOR_DIR}, skipping eval/upload for this step" >&2
    continue
  fi

  EVAL_OUT="${WORK}/eval/${LINEAGE}/step_${STEP}"
  mkdir -p "${EVAL_OUT}"
  echo "[run_arm] step ${STEP}: eval -> ${EVAL_OUT}"
  # countdown_gs0_eval.py CLI confirmed at scripts/countdown_gs0_eval.py:108-123.
  # 500 problems x 8 rollouts per task spec: val parquet already has 500 rows
  # (scripts/local/make_data.sh), so --limit 0 (= all) x --num_samples 8.
  python scripts/countdown_gs0_eval.py \
    --model_path "${MERGED_DIR}" \
    --data "${DATA_VAL}" \
    --meta_format "${EVAL_META_FORMAT:-${DATA_VARIANT}}" \
    --num_samples 8 \
    --seed 11 \
    --limit 0 \
    --out_dir "${EVAL_OUT}" \
    >> "${LOG_FILE}" 2>&1 || echo "[run_arm] step ${STEP}: eval FAILED (see ${LOG_FILE}), continuing to upload anyway" >&2

  echo "[run_arm] step ${STEP}: uploading ${MERGED_DIR} to HF"
  python "${_SCRIPT_DIR}/hf_upload.py" \
    --lineage "${LINEAGE}" --step "${STEP}" --local-dir "${MERGED_DIR}" \
    >> "${LOG_FILE}" 2>&1 || echo "[run_arm] step ${STEP}: upload FAILED/SKIPPED (see ${LOG_FILE}), not failing the job for this" >&2
done

echo "[run_arm] LINEAGE=${LINEAGE} done."
