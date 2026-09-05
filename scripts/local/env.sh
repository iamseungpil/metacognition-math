#!/usr/bin/env bash
# scripts/local/env.sh — SOURCE this, don't execute it: `source scripts/local/env.sh`
#
# Common environment for every script under scripts/local/. Idempotent (safe to
# source twice). Tolerant of a missing .env (repo ships only .env.example — see
# CLAUDE.md "Key Tokens"): a missing HF_TOKEN/WANDB_API_KEY degrades features
# (no HF upload, offline wandb) instead of erroring, because this box may be
# used before tokens are provisioned.

# Resolve repo root from this file's location so `source`-ing works from anywhere.
_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_ROOT="$(cd "${_ENV_SH_DIR}/../.." && pwd)"

# ── 1. Repo tokens (.env) — optional. ────────────────────────────────────────
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
else
  echo "[env.sh] no ${REPO_ROOT}/.env found (only .env.example ships in git) — continuing without tokens" >&2
fi

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[env.sh] HF_TOKEN not set — hf_upload.py will skip uploads" >&2
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  export WANDB_MODE=offline
  echo "[env.sh] WANDB_API_KEY not set — WANDB_MODE=offline" >&2
fi

# ── 2. Conda env activation snippet (being installed per task; tolerant). ───
_SIMPLERL_ACTIVATE=/hdd_data/seungpil/envs/activate_simplerl.sh
if [ -f "${_SIMPLERL_ACTIVATE}" ]; then
  # shellcheck disable=SC1090
  source "${_SIMPLERL_ACTIVATE}"
else
  echo "[env.sh] WARNING: ${_SIMPLERL_ACTIVATE} not found — simplerl env not activated" >&2
fi

# activate_simplerl.sh sets PYTHONPATH to the repo root already; keep it explicit
# here too in case env.sh is sourced without that file being present.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ── 3. Work root. ─────────────────────────────────────────────────────────
export WORK=/hdd_data/seungpil/scratch
mkdir -p "${WORK}/logs" "${WORK}/checkpoints" "${WORK}/models" "${WORK}/data" "${WORK}/merged" "${WORK}/eval"

# ── 4. Ray / vLLM / misc knobs (per task spec). ──────────────────────────────
# NOTE: verl_sdc.py's own ray.init() call hardcodes
# `_system_config={"agent_register_timeout_ms": 600000}` directly (see comment
# at verl_sdc.py:4945 — the code claims RAY_* env vars did NOT change this
# value for them, hence the direct _system_config passthrough). Exporting the
# env var below is therefore probably a no-op against the currently-checked-out
# verl_sdc.py, but it is harmless and is what the task spec asked for, and it
# does matter for the bare `ray start`/`ray.init()` defaults used elsewhere
# (e.g. gpu_queue.py, ad hoc debugging).
export RAY_agent_register_timeout_ms=600000
export VLLM_USE_V1=1
export LOCAL_RANK=0
export PYTHONNOUSERSITE=1
export SDC_LOG_DIR="${WORK}/logs"
export TRITON_CACHE_DIR="${WORK}/triton_cache"; mkdir -p "${TRITON_CACHE_DIR}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "[env.sh] REPO_ROOT=${REPO_ROOT} WORK=${WORK} HF_TOKEN=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset) WANDB_MODE=${WANDB_MODE:-online}"
# 0905: 루트 디스크(200GB)를 타 프로젝트가 채워 12MB 까지 떨어짐 → ~/.cache 계열(vLLM 컴파일·torch inductor)도 /hdd_data 로.
export XDG_CACHE_HOME="/hdd_data/seungpil/xdg_cache"; mkdir -p "${XDG_CACHE_HOME}"
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME}/vllm"; export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"; export FLASHINFER_WORKSPACE_BASE="${XDG_CACHE_HOME}/flashinfer"
