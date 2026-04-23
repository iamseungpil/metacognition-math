#!/bin/bash
# Launch veRL-native SDC_SHARED training on the current node.
# Auto-detects GPU count + VRAM, picks matching config.
#
# Usage:
#   scripts/launch_sdc_verl.sh                            # foreground, auto detect
#   DETACH=1 scripts/launch_sdc_verl.sh                   # background + pid/path files
#   CONFIG=verl_sdc_e21r_shared_40g8 scripts/launch_sdc_verl.sh
#
# Prereq: scripts/bootstrap_sdc_node.sh already ran (env + code present).

set -euo pipefail

SCRATCH="${SCRATCH:-/scratch}"
CODE_DIR="${CODE_DIR:-${SCRATCH}/metacognition}"
CKPT_DIR="${CKPT_DIR:-${SCRATCH}/checkpoints}"
LOG_DIR="${LOG_DIR:-${SCRATCH}/logs}"
HF_DATA_REPO="${HF_DATA_REPO:-iamseungpil/metacot}"
MODEL_NAME="${MODEL_NAME:-v8_meta_inside_strict_sft}"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# ── Load env ──
if [ -f "${SCRATCH}/env.sh" ]; then
    source "${SCRATCH}/env.sh"
fi
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

cd "${CODE_DIR}"

# ── GPU auto-detect → config pick ──
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
VRAM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')

if [ -z "${CONFIG:-}" ]; then
    # Order matters: most-specific (largest VRAM) first
    if [ "${NGPU}" -eq 8 ] && [ "${VRAM_GB}" -le 45 ]; then
        CONFIG="verl_sdc_e21r_shared_40g8"
    elif [ "${NGPU}" -eq 8 ] && [ "${VRAM_GB}" -ge 140 ]; then
        CONFIG="verl_sdc_e21r_shared"  # H200 x8 can safely run the base config
    elif [ "${NGPU}" -eq 4 ] && [ "${VRAM_GB}" -ge 140 ]; then
        CONFIG="verl_sdc_e21r_shared_h200_4x16k"
    elif [ "${NGPU}" -eq 4 ] && [ "${VRAM_GB}" -ge 80 ]; then
        CONFIG="verl_sdc_e21r_shared"  # A100 80GB x4 baseline
    else
        echo "[launch] no matching config for ${NGPU}x ${VRAM_GB}GB — set CONFIG= manually"; exit 1
    fi
fi
echo "[launch] NGPU=${NGPU} VRAM=${VRAM_GB}GB → config=${CONFIG}"

MODEL_OVERRIDE_ARG=""
if [ -n "${MODEL_PATH:-}" ]; then
    MODEL_OVERRIDE_ARG="actor_rollout_ref.model.path=${MODEL_PATH}"
    echo "[launch] model override → ${MODEL_PATH}"
fi

# ── Asset staging ──
if [ ! -f "data/verl_train_redirect.parquet" ]; then
    echo "[launch] data parquet missing in ${CODE_DIR}/data, pulling from HF dataset repo"
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_DATA_REPO}",
    repo_type="dataset",
    token="${HF_TOKEN:-}",
    local_dir="${CODE_DIR}",
    allow_patterns=[
        "data/verl_train_redirect.parquet",
        "data/verl_val_redirect.parquet",
        "data/verl_train_redirect_base.parquet",
        "data/verl_val_redirect_base.parquet",
    ],
)
PY
fi

if [ -z "${MODEL_PATH:-}" ]; then
    DEFAULT_MODEL_PATH="${SCRATCH}/models/${MODEL_NAME}"
    if [ ! -f "${DEFAULT_MODEL_PATH}/config.json" ]; then
        echo "[launch] model missing at ${DEFAULT_MODEL_PATH}, staging from HF dataset repo"
        python scripts/ensure_hf_model.py \
            --repo-id "${HF_DATA_REPO}" \
            --repo-type dataset \
            --model-name "${MODEL_NAME}" \
            --output-dir "${DEFAULT_MODEL_PATH}"
    fi
    MODEL_OVERRIDE_ARG="actor_rollout_ref.model.path=${DEFAULT_MODEL_PATH}"
    echo "[launch] using staged model → ${DEFAULT_MODEL_PATH}"
fi

# ── Resume-from-HF hook: pull latest global_step_N if present ──
HF_CKPT_REPO="${HF_CKPT_REPO:-iamseungpil/metacot-sdc-verl-shared}"
RESUME_ARG=""
if [ -n "${HF_TOKEN:-}" ]; then
python - <<PY >"${LOG_DIR}/resume_probe.log" 2>&1 || true
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
try:
    files = api.list_repo_files(repo_id="${HF_CKPT_REPO}", repo_type="model")
except Exception:
    files = []
steps = set()
for f in files:
    if f.startswith("checkpoints/") and "/global_step_" in f:
        try:
            steps.add(int(f.split("/global_step_")[1].split("/")[0]))
        except Exception:
            pass
if steps:
    print(max(steps))
PY
    LATEST_STEP=$(tail -1 "${LOG_DIR}/resume_probe.log" 2>/dev/null | grep -E '^[0-9]+$' || true)
else
    LATEST_STEP=""
    echo "[launch] HF_TOKEN not set — skipping resume probe"
fi

if [ -n "${LATEST_STEP}" ]; then
    RESUME_DIR="${CKPT_DIR}/${CONFIG}/global_step_${LATEST_STEP}"
    if [ ! -d "${RESUME_DIR}" ]; then
        echo "[launch] pulling HF ckpt step ${LATEST_STEP} → ${RESUME_DIR}"
        python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${HF_CKPT_REPO}",
    token="${HF_TOKEN}",
    allow_patterns=["checkpoints/${CONFIG}/global_step_${LATEST_STEP}/**"],
    local_dir="${SCRATCH}",
)
PY
    fi
    RESUME_ARG="trainer.resume_from_path=${RESUME_DIR} trainer.resume_mode=auto"
    echo "[launch] will resume from step ${LATEST_STEP}"
fi

# ── Background checkpoint push daemon ──
PUSH_LOG="${LOG_DIR}/push_daemon.log"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[launch] HF_TOKEN not set — skipping checkpoint push daemon"
elif ! pgrep -f "push_ckpts_to_hf.py" >/dev/null 2>&1; then
    echo "[launch] starting checkpoint push daemon (every 600s)"
    nohup python scripts/push_ckpts_to_hf.py \
        --ckpt_dir "${CKPT_DIR}/${CONFIG}" \
        --repo_id "${HF_CKPT_REPO}" \
        --token "${HF_TOKEN}" \
        --interval 600 \
        --config_name "${CONFIG}" \
        >"${PUSH_LOG}" 2>&1 &
    echo "[launch] push daemon PID=$! → ${PUSH_LOG}"
fi

# ── Launch training ──
TRAIN_TS="$(date +%Y%m%d_%H%M%S)"
TRAIN_LOG="${LOG_DIR}/train_${CONFIG}_${TRAIN_TS}.log"
PID_FILE="${SCRATCH}/sdc_direct.pid"
PATH_FILE="${SCRATCH}/sdc_direct.path"

CMD=(
    python -u -m src.training.verl_sdc
    "--config-name=${CONFIG}"
    "trainer.default_local_dir=${CKPT_DIR}/${CONFIG}"
)

if [ -n "${RESUME_ARG}" ]; then
    # shellcheck disable=SC2206
    CMD+=( ${RESUME_ARG} )
fi
if [ -n "${MODEL_OVERRIDE_ARG}" ]; then
    # shellcheck disable=SC2206
    CMD+=( ${MODEL_OVERRIDE_ARG} )
fi

echo "[launch] stopping stale Ray runtime if present"
ray stop --force >/dev/null 2>&1 || true
echo "${TRAIN_LOG}" > "${PATH_FILE}"

if [ "${DETACH:-0}" = "1" ]; then
    echo "[launch] starting detached training — log=${TRAIN_LOG}"
    nohup "${CMD[@]}" >"${TRAIN_LOG}" 2>&1 </dev/null &
    echo $! > "${PID_FILE}"
    echo "[launch] detached PID=$(cat "${PID_FILE}") path=$(cat "${PATH_FILE}")"
else
    echo "[launch] starting foreground training — log=${TRAIN_LOG}"
    "${CMD[@]}" 2>&1 | tee "${TRAIN_LOG}"
fi
