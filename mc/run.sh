#!/usr/bin/env bash
# mc/run.sh ARM SEED [STEPS=50] [--dry-run]
#
# 발사기 — 진입점은 **`mc.trainer`** 다. 유일한 팔(0925 정리): SPONT_PFX(mc/train_hook.MC_ARM_SPECS).
#
# SPONT_PFX(0923·0925) = 고정 첫 박스 앞부분(DATA_TRAIN) 뒤 K 이어 쓰기 · verl GRPO · plain · constant ·
# LABEL=gold(기본) · RESP_LEN 4096 · ROLLOUT_N 8 · 에이전트 루프 mc_prefix(mc/agent_loop.yaml) ·
# MAX_PROMPT 8704. PFX_FORK=ch(수정 28) = 섞인 묶음을 교차 적합 이웃으로 재배분.
# PFX_TRUNC=mask(수정 32) = 상한에서 잘린 이어쓰기는 학습에서 제외. PFX_BREAK_W=w(수정 43) = 맞은 첫 답을 뒤집은 실패 벌 × w.
# PFX_REP=c(수정 43) = 같은 답 3번째 확인 뒤 말에만 작은 벌(끝낸·잘린 행 모두). PFX_REP_HARD=1(수정 46) = 그 구간 결과 칭찬 차단·벌 상한 없음.
# 앞부분 표집 무게 = extra_info[PFX_WEIGHT_KEY=weight](수정 25). 계보 mc_SPONT_PFX_<label>_s<seed>_r<len>
# [_fork<mode>][_tm][_bw<w>][_rp<c>][_<TAG>]. (R2·zero·RIGHT_SHARE·STOP·TAIL0·loop 는 실패로 삭제 — 수정 33/38/43)
#
# ★PROMPT_VARIANT=plain: 프롬프트는 parquet 의 `prompt` 컬럼에 구워져 있으므로(verl RLHFDataset)
#   plain 으로 구운 DATA_TRAIN/DATA_VAL 을 함께 줘야 한다(mc/pool.py 가 PROMPT_VARIANT 로 굽는다;
#   어긋나면 mc/trainer.check_prompt_lengths 가 즉사). 평가·롤아웃은 이 env 만으로 갈린다.
# ★LR_SCHED=cosine(기본)|constant — constant 는 워밍업 0. 끊어 학습(STEPS 를 늘려 다시 부르면
#   resume)에서 일정이 안 갈리게 SPONT_PFX 기본은 constant 다.
# ★TRAIN_DONE 마커 = `steps=<도달 스텝> <시각>`. 마커 스텝 ≥ STEPS 면 학습을 건너뛰고, 작으면
#   resume_mode=auto 로 최신 global_step 에서 STEPS 까지 잇는다(옛 마커 = 스텝 없음 → 완료로 본다).
# ★EVAL_STEPS: 미설정 = 모든 global_step 병합·12k 평가 · `10,25` = 그 스텝만 · `0` = 평가 없음.
#
# env   LABEL=gold|majority · RESP_LEN=4096 · OUTCOME_MODE=group · LR_SCHED
#       PFX_FORK=ch · PFX_TRUNC=keep|mask · PFX_BREAK_W · PFX_REP · PFX_WEIGHT_KEY · PFX_DISTILL(_SHUF·_ROWS) · PFX_ALLOC(_SHUF) · PFX_KEEP · KL_COEF · PFX_GUARD_ABS
#       MODEL_PATH · DATA_TRAIN · TRAIN_BATCH · VLLM_UTIL
#       REF_OFFLOAD/ACTOR_OFFLOAD · TAG · SAVE_FREQ · EVAL_STEPS · MC_DUMP_ADV
#
# ⛔`SLIM`/`PAGED` 는 **없다** — verl09 env 에 bitsandbytes 가 없어 8비트 옵티마이저는 즉사한다
#   (2026-09-21). 옵티마이저는 torch AdamW 고정, 메모리는 optimizer_offload=true(항상 켬).
# ★GPU 는 큐가 CUDA_VISIBLE_DEVICES 로 준다(여기서 건드리지 않는다). wandb·git 은 쓰지 않는다.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
DRY_RUN=0
POSITIONAL=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    *) POSITIONAL+=("$a") ;;
  esac
done
set -- "${POSITIONAL[@]:-}"

ARM="${1:?ARM required (SPONT_PFX)}"
SEED="${2:?SEED required}"
STEPS="${3:-50}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/../scripts/local/env.sh"
cd "${REPO_ROOT}"
if [ -n "${VERL_ENV:-}" ]; then
  [ -x "${VERL_ENV}/bin/python" ] || { echo "[mc] FATAL: VERL_ENV has no bin/python" >&2; exit 1; }
  export PATH="${VERL_ENV}/bin:${PATH}"; export VIRTUAL_ENV="${VERL_ENV}"
fi

case "${ARM}" in
  SPONT_PFX) ;;
  *) echo "[mc] FATAL: ARM=${ARM} 은 SPONT_PFX 여야 한다(다른 팔은 0925 에 폐기)" >&2; exit 1 ;;
esac
: "${PROMPT_VARIANT:=plain}" "${LR_SCHED:=constant}" "${LABEL:=gold}"
: "${RESP_LEN:=4096}" "${ROLLOUT_N:=8}" "${MAX_PROMPT:=8704}"
: "${DATA_TRAIN:=${WORK}/data/mc_pfx_200_plain.parquet}"
: "${DATA_VAL:=${WORK}/data/math_val_plain.parquet}"
PFX_ARGS=("actor_rollout_ref.rollout.agent.agent_loop_config_path=${_SCRIPT_DIR}/agent_loop.yaml"
          "actor_rollout_ref.rollout.agent.default_agent_loop=mc_prefix")

export OUTCOME_MODE="${OUTCOME_MODE:-group}"
[ "${OUTCOME_MODE}" = "group" ] || { echo "[mc] FATAL: SPONT_PFX 는 OUTCOME_MODE=group 전용이다" >&2; exit 1; }
LR_SCHED="${LR_SCHED:-cosine}"
case "${LR_SCHED}" in cosine|constant) ;; *) echo "[mc] FATAL: LR_SCHED=${LR_SCHED} 은 cosine|constant" >&2; exit 1 ;; esac

MODEL_PATH="${MODEL_PATH:-/hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507}"
RESP_LEN="${RESP_LEN:-8192}"
# ★계보 = ckpt 디렉터리. 프롬프트·체크비용·포크·LR 일정이 다른 런이 같은 디렉터리를 쓰면
#   `resume_mode=auto` 가 **남의 체크포인트**에서 이어 버린다(0921 발견) — 다른 것은 전부 박는다.
export PROMPT_VARIANT="${PROMPT_VARIANT:-math_opt}"
V4=""                                    # PFX-B 기본은 옛 이름 그대로(계보에 아무것도 안 박는다)
[ -n "${PFX_FORK:-}" ] && V4="${V4}_fork${PFX_FORK}"   # CH-Fork 수정 28
[ "${PFX_TRUNC:-keep}" = "mask" ] && V4="${V4}_tm"                  # 수정 32: 잘린 이어쓰기 = 학습에서 제외
[ -n "${PFX_BREAK_W:-}" ] && V4="${V4}_bw${PFX_BREAK_W}"             # 수정 43: 뒤집은 실패 벌 × w
[ -n "${PFX_REP:-}" ] && V4="${V4}_rp${PFX_REP}$([ "${PFX_REP_HARD:-}" = 1 ] && echo h || true)"   # 수정 43: 반복 확인 벌(h = 수정 46 구간 칭찬 차단·상한 없음)
[ -n "${PFX_ADV_CAP:-}" ] && V4="${V4}_ac${PFX_ADV_CAP}"           # 수정 48: 말 단위 adv 상한
[ -n "${PFX_DISTILL:-}" ] && V4="${V4}_ds${PFX_DISTILL}$([ "${PFX_DISTILL_ROWS:-wrong}" = all ] && echo a || true)$([ "${PFX_DISTILL_SHUF:-}" = 1 ] && echo x || true)"   # 수정 51/52: 자기 풀이 가린 자기 자신 증류(말당 β·d) · a = 59 모든 행 · x = 55 위약(자리 섞기)
[ -n "${PFX_ALLOC:-}" ] && V4="${V4}_al${PFX_ALLOC}$([ "${PFX_ALLOC_SHUF:-}" = 1 ] && echo x || true)"   # 수정 61: 눈 가린 나 − 어제의 나 말 단위 배분(곱) · x = 61e 위약
[ -n "${PFX_KEEP:-}" ] && V4="${V4}_kp${PFX_KEEP}"           # 수정 53: 맞은 줄 원래의 나 유지
[ -n "${KL_COEF:-}" ] && V4="${V4}_kl${KL_COEF}"                   # 수정 53: 전역 KL 손실 계수(기본 yaml .002)
[ "${PROMPT_VARIANT}" = "plain" ] || V4="${V4}_${PROMPT_VARIANT}"
[ "${LR_SCHED}" = "constant" ] || V4="${V4}_${LR_SCHED}"
LINEAGE="mc_${ARM}_${LABEL:-gold}_s${SEED}_r${RESP_LEN}${V4}"
# TAG: 같은 ARM·LABEL·SEED 의 변형(예: 다른 풀·스모크)을 계보에서 가른다.
[ -n "${TAG:-}" ] && LINEAGE="${LINEAGE}_${TAG}"
# CKPT_DIR 는 덮어쓸 수 있다(테스트가 마커 로직을 임시 디렉터리에서 검증한다).
CKPT_DIR="${CKPT_DIR:-${WORK}/checkpoints/${LINEAGE}}"
LOG_FILE="${WORK}/logs/${LINEAGE}.log"
DATA_TRAIN="${DATA_TRAIN:-${WORK}/data/math_train_math_opt.parquet}"
DATA_VAL="${DATA_VAL:-${WORK}/data/math_val_math_opt.parquet}"

ROLLOUT_N="${ROLLOUT_N:-16}"
export LABEL="${LABEL:-gold}"
export MC_CKPT_DIR="${CKPT_DIR}"
# ★MC_DUMP_ADV=N (기본 0) — step ≤ N 에서 어드밴티지를 가산 전/후로 떨군다.
if [ -n "${MC_DUMP_ADV:-}" ]; then export MC_DUMP_ADV; fi

# ★학습 풀 math_opt 1턴 프롬프트 max 1,891 · p99 1,079 토큰 — 1024/1536 이면 verl 이 긴(어려운)
#   문제를 **조용히 버린다**. 2048 로 두고 트레이너가 발사 전에 전 행을 검사한다.
MAX_PROMPT="${MAX_PROMPT:-2048}"
MAX_RESP="${RESP_LEN}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT + MAX_RESP + 256))}"

export RAY_TMPDIR="/hdd_data/seungpil/r/$(printf '%s' "${LINEAGE}" | md5sum | cut -c1-8)"
mkdir -p "${RAY_TMPDIR}" "${CKPT_DIR}" "${WORK}/logs"
# 루트 디스크 보호(0922): 임시·컴파일 캐시를 전부 HDD 로 돌린다.
export TMPDIR="/hdd_data/seungpil/tmp"
export TORCHINDUCTOR_CACHE_DIR="/hdd_data/seungpil/cache/torchinductor"
export TRITON_CACHE_DIR="/hdd_data/seungpil/cache/triton"
export VLLM_CACHE_ROOT="/hdd_data/seungpil/cache/vllm"
mkdir -p "${TMPDIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}"
unset RAY_ADDRESS || true
export WANDB_MODE=disabled      # wandb 금지(운용 규칙)
# ★2026-09-21 사고: Ray 가 num_cpus(124)만큼 워커를 **선기동**해 NFS import 타임아웃으로 죽고
#   main_task 가 영원히 대기했다. raylet env 로만 먹으므로 ray.init 전에(= 여기서) 심는다.
export RAY_enable_worker_prestart=0
export RAY_prestart_worker_first_driver=0

# constant 는 verl 0.9 FSDPOptimizerConfig 의 warmup_style(→lr_scheduler_type) + 워밍업 비율 0.
LR_ARGS=()
if [ "${LR_SCHED}" = "constant" ]; then
  LR_ARGS=("actor_rollout_ref.actor.optim.warmup_style=constant"
           "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0")
fi
TRAIN_CMD=(python -u -m mc.trainer
  "--config-name=${CONFIG_NAME:-countdown_6arm}"
  "++mode=MATH_META"
  "++algorithm.math_arm=${ARM}"
  "trainer.experiment_name=${LINEAGE}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.project_name=metacot-math-local"
  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=1"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.actor.optim.lr=${LR:-1e-6}"
  ${LR_ARGS[@]+"${LR_ARGS[@]}"}
  ${PFX_ARGS[@]+"${PFX_ARGS[@]}"}
  "data.train_files=${DATA_TRAIN}"
  "data.val_files=${DATA_VAL}"
  "++data.seed=${SEED}"
  "++data.apply_chat_template_kwargs.enable_thinking=false"
  "data.train_batch_size=${TRAIN_BATCH:-32}"
  "data.max_prompt_length=${MAX_PROMPT}"
  "data.max_response_length=${MAX_RESP}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.35}"
  # ★torch AdamW 고정 — verl09 env 에 bitsandbytes 가 없다(위 머리말).
  "actor_rollout_ref.actor.optim.optimizer_impl=torch.optim"
  "actor_rollout_ref.actor.optim.optimizer=AdamW"
  "actor_rollout_ref.ref.fsdp_config.param_offload=$([ "${REF_OFFLOAD:-0}" = "1" ] && echo true || echo false)"
  # ★8비트 옵티마이저를 못 쓰므로 메모리는 이 오프로드가 맡는다 — 끄는 손잡이를 두지 않는다.
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
  "actor_rollout_ref.actor.fsdp_config.param_offload=$([ "${ACTOR_OFFLOAD:-0}" = "1" ] && echo true || echo false)"
  # 엔트로피 항이 10k토큰 × 152k 어휘 로짓 사본을 남겨 1행 backward +17GB — 끈다(수정 10).
  "actor_rollout_ref.actor.entropy_coeff=0"
  ${KL_COEF:+"actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}"}
  "++actor_rollout_ref.model.enable_activation_offload=true"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "++trainer.total_training_steps=${STEPS}"
  "trainer.resume_mode=auto"
  "trainer.save_freq=${SAVE_FREQ:-5}"
  "trainer.test_freq=0"
  "++trainer.val_before_train=False"
  "++hydra.searchpath=[pkg://verl/trainer/config]"
)

echo "[mc] LINEAGE=${LINEAGE} ARM=${ARM} SEED=${SEED} STEPS=${STEPS} LABEL=${LABEL} RESP_LEN=${RESP_LEN} ROLLOUT_N=${ROLLOUT_N} PROMPT_VARIANT=${PROMPT_VARIANT} OUTCOME_MODE=${OUTCOME_MODE} LR_SCHED=${LR_SCHED} LR=${LR:-1e-6} TRAIN_BATCH=${TRAIN_BATCH:-32} PFX_FORK=${PFX_FORK:-} PFX_BREAK_W=${PFX_BREAK_W:-} PFX_REP=${PFX_REP:-} PFX_TRUNC=${PFX_TRUNC:-keep} MC_DUMP_ADV=${MC_DUMP_ADV:-0} EVAL_STEPS=${EVAL_STEPS:-all} MODEL_PATH=${MODEL_PATH}"
echo "[mc] ckpt=${CKPT_DIR} log=${LOG_FILE} data=${DATA_TRAIN}"
echo "[mc] exact train command:"; printf '  %q' "${TRAIN_CMD[@]}"; echo
# ★이어 학습 안전(0926): 계보 이름에 없는 손잡이(LR·데이터·표집 무게 …)가 바뀐 채 같은 체크포인트를 잇지 못하게 한다.
RUN_ENV="LR=${LR:-1e-6} DATA_TRAIN=${DATA_TRAIN} PFX_WEIGHT_KEY=${PFX_WEIGHT_KEY:-weight} TRAIN_BATCH=${TRAIN_BATCH:-32} ROLLOUT_N=${ROLLOUT_N} RESP_LEN=${RESP_LEN} MAX_PROMPT=${MAX_PROMPT} LR_SCHED=${LR_SCHED} PROMPT_VARIANT=${PROMPT_VARIANT} LABEL=${LABEL:-gold} PFX_FORK=${PFX_FORK:-} PFX_CHECK_COST= PFX_CHECK_W="   # 빈 두 칸 = 옛 RUN_ENV.txt 와 같은 문자열(R2 삭제)
[ "${PFX_TRUNC:-keep}" = "keep" ] || RUN_ENV="${RUN_ENV} PFX_TRUNC=${PFX_TRUNC}"          # 기본값이면 옛 기록과 같은 문자열
[ -z "${PFX_BREAK_W:-}" ] || RUN_ENV="${RUN_ENV} PFX_BREAK_W=${PFX_BREAK_W}"
[ -z "${PFX_REP:-}" ] || RUN_ENV="${RUN_ENV} PFX_REP=${PFX_REP}"
[ -z "${PFX_REP_HARD:-}" ] || RUN_ENV="${RUN_ENV} PFX_REP_HARD=${PFX_REP_HARD}"
[ -z "${PFX_ADV_CAP:-}" ] || RUN_ENV="${RUN_ENV} PFX_ADV_CAP=${PFX_ADV_CAP}"
[ -z "${PFX_DISTILL:-}" ] || RUN_ENV="${RUN_ENV} PFX_DISTILL=${PFX_DISTILL}"
[ -z "${PFX_DISTILL_SHUF:-}" ] || RUN_ENV="${RUN_ENV} PFX_DISTILL_SHUF=${PFX_DISTILL_SHUF}"
[ -z "${PFX_DISTILL_ROWS:-}" ] || RUN_ENV="${RUN_ENV} PFX_DISTILL_ROWS=${PFX_DISTILL_ROWS}"
[ -z "${PFX_KEEP:-}" ] || RUN_ENV="${RUN_ENV} PFX_KEEP=${PFX_KEEP}"
[ -z "${PFX_ALLOC:-}" ] || RUN_ENV="${RUN_ENV} PFX_ALLOC=${PFX_ALLOC}"
[ -z "${PFX_ALLOC_SHUF:-}" ] || RUN_ENV="${RUN_ENV} PFX_ALLOC_SHUF=${PFX_ALLOC_SHUF}"
[ -z "${KL_COEF:-}" ] || RUN_ENV="${RUN_ENV} KL_COEF=${KL_COEF}"
[ -z "${PFX_GUARD_ABS:-}" ] || RUN_ENV="${RUN_ENV} PFX_GUARD_ABS=${PFX_GUARD_ABS}"
if [ -f "${CKPT_DIR}/RUN_ENV.txt" ] && [ "$(cat "${CKPT_DIR}/RUN_ENV.txt")" != "${RUN_ENV}" ]; then
  echo "[mc] FATAL: ${CKPT_DIR}/RUN_ENV.txt 와 설정이 다르다 — 이어 학습 금지" >&2
  diff <(tr ' ' '\n' < "${CKPT_DIR}/RUN_ENV.txt") <(tr ' ' '\n' <<< "${RUN_ENV}") >&2
  exit 1
fi
[ "${DRY_RUN}" = "1" ] || echo "${RUN_ENV}" > "${CKPT_DIR}/RUN_ENV.txt"
# ── 학습 ─────────────────────────────────────────────────────────────────────
# ★운용 사고(옛 런처): 학습은 끝났는데 병합·평가가 rc≠0 이면 재시도 래퍼가 **학습을 처음부터
#   다시** 띄웠다. 그래서 도달 스텝을 마커로 못 박고, 재시도는 병합·평가만 한다.
TRAIN_DONE="${CKPT_DIR}/TRAIN_DONE"
DONE_STEPS=""
if [ -f "${TRAIN_DONE}" ]; then
  DONE_STEPS="$(sed -n 's/^steps=\([0-9][0-9]*\).*/\1/p' "${TRAIN_DONE}" | head -1)"
fi
if [ -f "${TRAIN_DONE}" ] && { [ -z "${DONE_STEPS}" ] || [ "${DONE_STEPS}" -ge "${STEPS}" ]; }; then
  echo "[mc] TRAIN_DONE 있음($(cat "${TRAIN_DONE}")) — 학습을 건너뛰고 병합·평가만 한다"
elif [ "${DRY_RUN}" = "1" ]; then
  echo "[mc] --dry-run: 학습을 돌리지 않는다(마커 steps=${DONE_STEPS:-없음} < STEPS=${STEPS} → 실제 실행 시 최신 global_step 에서 이어 학습)"
else
  # ★rc 75 = 중단 규칙(ABORTED.txt) — 큐가 재시도하지 않는다.
  set +e
  "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  rc="${PIPESTATUS[0]}"
  set -e
  echo "[mc] train rc=${rc}"
  if [ "${rc}" != "0" ]; then exit "${rc}"; fi
  REACHED="$(cat "${CKPT_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || echo "${STEPS}")"
  echo "steps=${REACHED} $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${TRAIN_DONE}"
  echo "[mc] TRAIN_DONE 기록: ${TRAIN_DONE} (steps=${REACHED})"
fi

# ── 병합 + 12k 등예산 평가 ───────────────────────────────────────────────────
# 실패는 **학습 rc 와 분리**한다: rc 3 으로 끝내고, 재시도해도 위 마커 때문에 학습은 다시 돌지 않는다.
EVAL_DATA="${EVAL_DATA:-${WORK}/data/math_eval_L5_800.parquet}"
EVAL_K="${EVAL_K:-8}"
EVAL_SEED="${EVAL_SEED:-11}"
POST_RC=0
post_train() {
  local step actor merged
  for step in $(ls -1 "${CKPT_DIR}" 2>/dev/null | sed -n 's/^global_step_\([0-9]*\)$/\1/p' | sort -n); do
    actor="${CKPT_DIR}/global_step_${step}/actor"
    [ -d "${actor}" ] || continue
    # EVAL_STEPS=10,25 처럼 주면 그 스텝만(12k 평가 1개 ≈ 2~3 GPU시간). `0` 이면 아무 스텝도 안 한다.
    if [ -n "${EVAL_STEPS:-}" ] && ! echo ",${EVAL_STEPS}," | grep -q ",${step},"; then continue; fi
    # 병합은 mc/probe.merged_model 한 곳 — ${WORK}/models/merged_<계보>_step<N>, 있으면 재사용.
    if ! merged="$(python -m mc.probe --merge_only "${actor}" 2>> "${LOG_FILE}")"; then
      echo "[mc] step ${step}: merge FAILED" >&2; POST_RC=3; continue
    fi
    if ! python -m mc.eval --protocol single --model_path "${merged}" \
        --dataset "parquet:${EVAL_DATA}" --k "${EVAL_K}" --seed "${EVAL_SEED}" \
        --out_dir "${WORK}/eval/${LINEAGE}_step${step}_single" \
        >> "${LOG_FILE}" 2>&1; then
      echo "[mc] step ${step} single: eval FAILED" >&2; POST_RC=3
    fi
  done
}

if [ "${DRY_RUN}" = "1" ]; then
  echo "[mc] --dry-run: post-train = merge -> ${WORK}/models/merged_${LINEAGE}_step<N> (EVAL_STEPS=${EVAL_STEPS:-all})"
  echo "[mc] --dry-run: eval = mc.eval {single} parquet:${EVAL_DATA} k=${EVAL_K} seed=${EVAL_SEED}"
  echo "[mc] --dry-run: done."
  exit 0
fi
post_train
if [ "${POST_RC}" != "0" ]; then
  echo "[mc] post-train FAILED (rc ${POST_RC}) — 학습은 끝났다(${TRAIN_DONE}); 재시도는 병합·평가만 돈다" >&2
  exit 3
fi
echo "[mc] done — train + merge + 12k eval 완료"
