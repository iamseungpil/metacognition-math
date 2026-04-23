#!/bin/bash
# Idempotent bootstrap for SDC veRL training on an AMLT Singularity node.
# Works for 40G8-A100-NvLink and 80G4-A100-NvLink targets.
#
# Stages:
#   1. Create conda env 'simplerl' (skip if /scratch/simplerl.done exists)
#   2. Install torch + veRL 0.7.1 + compatible vLLM/Ray stack
#      Wheels pulled from HF CDN when present (faster than PyPI on BSC).
#   3. Sync code to /scratch/metacognition from HF code_snapshot tarball
#      (see memory: feedback_hf_bootstrap — no SSH base64 for big uploads)
#   4. Write /scratch/simplerl.done sentinel + export HF/WANDB tokens
#   5. nvidia-smi verification
#
# Env vars (override if needed):
#   HF_TOKEN  : optional HuggingFace token for iamseungpil/metacot
#   WANDB_KEY / WANDB_API_KEY : optional WandB API key
#   CODE_REPO : HF dataset ID hosting code_snapshot tarball
#               (default iamseungpil/metacot, path code_snapshots/metacognition.tar.gz)

set -euo pipefail

HF_TOKEN="${HF_TOKEN:-}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
CODE_REPO="${CODE_REPO:-iamseungpil/metacot}"
CODE_TAR_PATH="${CODE_TAR_PATH:-code_snapshots/metacognition.tar.gz}"
CODE_TAR_REVISION="${CODE_TAR_REVISION:-}"  # pin to a specific HF revision if set
FORCE_SYNC="${FORCE_SYNC:-0}"                 # set to 1 to force code re-download
SCRATCH="${SCRATCH:-/scratch}"
DONE_MARKER="${SCRATCH}/simplerl.done"
CODE_DIR="${SCRATCH}/metacognition"
CODE_VERSION_FILE="${CODE_DIR}/.bootstrap_version"

echo "[bootstrap] scratch=${SCRATCH} done=${DONE_MARKER}"
mkdir -p "${SCRATCH}"

# ── Stage 0: quick exit if already bootstrapped ──
if [ -f "${DONE_MARKER}" ]; then
    echo "[bootstrap] ${DONE_MARKER} exists — env already installed, skipping install"
    ENV_OK=1
else
    ENV_OK=0
fi

# ── Stage 1+2: conda env + packages ──
if [ "${ENV_OK}" -eq 0 ]; then
    echo "[bootstrap] creating simplerl conda env + packages"
    source /opt/conda/etc/profile.d/conda.sh

    if ! conda env list | grep -qE '^simplerl\b'; then
        conda create -n simplerl python=3.10 -y
    fi
    conda activate simplerl

    pip install --upgrade pip setuptools wheel

    # Torch from PyTorch CDN (fast)
    # H200 SDC runs were previously stable on the 4k regime with the torch 2.6 / vLLM 0.8.3 line.
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

    # Core ML deps
    pip install accelerate datasets dill hydra-core omegaconf numpy pybind11 \
        tensordict "transformers==4.57.6" peft liger-kernel word2number \
        "math-verify[antlr4_11_0]==0.6.0" deepspeed wandb pandas pyarrow pyyaml

    # Match the vLLM 0.8.3 requirement line directly to avoid resolver backtracking.
    pip install "ray[cgraph]==2.43.0"

    # vLLM + veRL — keep versions aligned with veRL 0.7.1 metadata.
    WHEEL_DIR="${SCRATCH}/wheels"
    mkdir -p "${WHEEL_DIR}"
    for pkg in vllm-0.8.3 verl-0.7.1; do
        url="https://huggingface.co/datasets/${CODE_REPO}/resolve/main/wheels/${pkg}-py3-none-any.whl"
        wheel_path="${WHEEL_DIR}/${pkg}-py3-none-any.whl"
        if [ ! -f "${wheel_path}" ] && [ -n "${HF_TOKEN}" ]; then
            echo "[bootstrap] trying HF wheel ${url}"
            if curl -sfL -H "Authorization: Bearer ${HF_TOKEN}" -o "${wheel_path}" "${url}"; then
                echo "[bootstrap] got ${pkg} from HF"
            else
                echo "[bootstrap] HF wheel miss, using pip for ${pkg}"
                rm -f "${wheel_path}"
            fi
        fi
    done

    if [ -f "${WHEEL_DIR}/vllm-0.8.3-py3-none-any.whl" ]; then
        pip install --no-deps "${WHEEL_DIR}/vllm-0.8.3-py3-none-any.whl"
    else
        pip install --no-deps "vllm==0.8.3"
    fi

    if [ -f "${WHEEL_DIR}/verl-0.7.1-py3-none-any.whl" ]; then
        pip install --no-deps "${WHEEL_DIR}/verl-0.7.1-py3-none-any.whl"
    else
        pip install --no-deps "verl==0.7.1"
    fi

    # Verify imports
    python - <<'PY'
import torch, ray, hydra, omegaconf
import vllm, verl
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}")
print(f"ray={ray.__version__} hydra={hydra.__version__}")
print(f"vllm={vllm.__version__} verl={verl.__version__}")
PY

    # veRL 0.7.1 + vLLM 0.8.x async server compatibility:
    # some vLLM builds route through a V0 fallback while veRL still calls the
    # V1 AsyncLLM entrypoint, which raises "VLLM_USE_V1=False". Patch the
    # installed veRL launcher to fall back to AsyncLLMEngine explicitly.
    python - <<'PY'
from pathlib import Path
import inspect
import verl.workers.rollout.vllm_rollout.vllm_async_server as vas

path = Path(inspect.getfile(vas))
text = path.read_text()

if "from vllm.engine.async_llm_engine import AsyncLLMEngine" not in text:
    text = text.replace(
        "from vllm.v1.engine.async_llm import AsyncLLM\n",
        "from vllm.v1.engine.async_llm import AsyncLLM\nfrom vllm.engine.async_llm_engine import AsyncLLMEngine\n",
    )

old = "        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)\n"
new = """        try:\n            engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)\n        except ValueError as exc:\n            if \"VLLM_USE_V1=False\" not in str(exc):\n                raise\n            engine_client = AsyncLLMEngine.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)\n"""

if old in text and new not in text:
    text = text.replace(old, new)

old = """        if self.config.prometheus.enable:\n            if self.config.prometheus.served_model_name:\n                # Extract model name from path if it's a full path\n                served_model_name = self.config.prometheus.served_model_name\n                if \"/\" in served_model_name:\n                    # If it's a full path, extract the last part as model name\n                    served_model_name = served_model_name.split(\"/\")[-1]\n                args[\"served_model_name\"] = served_model_name\n\n        # mtp\n        if self.config.mtp.enable and self.config.mtp.enable_rollout:\n            speculative_config = {\n                \"method\": self.config.mtp.method,\n                \"num_speculative_tokens\": self.config.mtp.num_speculative_tokens,\n"""
new = """        prom_cfg = getattr(self.config, \"prometheus\", None)\n        prom_enable = getattr(prom_cfg, \"enable\", False)\n        prom_served_model_name = getattr(prom_cfg, \"served_model_name\", None)\n        if isinstance(prom_cfg, dict):\n            prom_enable = prom_cfg.get(\"enable\", False)\n            prom_served_model_name = prom_cfg.get(\"served_model_name\")\n        if prom_enable:\n            if prom_served_model_name:\n                # Extract model name from path if it's a full path\n                served_model_name = prom_served_model_name\n                if \"/\" in served_model_name:\n                    # If it's a full path, extract the last part as model name\n                    served_model_name = served_model_name.split(\"/\")[-1]\n                args[\"served_model_name\"] = served_model_name\n\n        # mtp\n        mtp_cfg = getattr(self.config, \"mtp\", None)\n        mtp_enable = getattr(mtp_cfg, \"enable\", False)\n        mtp_enable_rollout = getattr(mtp_cfg, \"enable_rollout\", False)\n        mtp_method = getattr(mtp_cfg, \"method\", None)\n        mtp_num_speculative_tokens = getattr(mtp_cfg, \"num_speculative_tokens\", None)\n        if isinstance(mtp_cfg, dict):\n            mtp_enable = mtp_cfg.get(\"enable\", False)\n            mtp_enable_rollout = mtp_cfg.get(\"enable_rollout\", False)\n            mtp_method = mtp_cfg.get(\"method\")\n            mtp_num_speculative_tokens = mtp_cfg.get(\"num_speculative_tokens\")\n        if mtp_enable and mtp_enable_rollout:\n            speculative_config = {\n                \"method\": mtp_method,\n                \"num_speculative_tokens\": mtp_num_speculative_tokens,\n"""

if old in text and new not in text:
    text = text.replace(old, new)

path.write_text(text)
print(f"[bootstrap] patched {path}")
PY

    touch "${DONE_MARKER}"
    echo "[bootstrap] env install complete → ${DONE_MARKER}"
fi

# ── Stage 3: code sync from HF ──
# Check if we need to sync: missing, force-sync, or revision mismatch
NEED_SYNC=0
if [ ! -d "${CODE_DIR}/src/training" ]; then
    echo "[bootstrap] code dir missing — will sync"
    NEED_SYNC=1
elif [ "${FORCE_SYNC}" = "1" ]; then
    echo "[bootstrap] FORCE_SYNC=1 — will re-sync"
    NEED_SYNC=1
elif [ -n "${CODE_TAR_REVISION}" ]; then
    CURRENT_REV="$(cat "${CODE_VERSION_FILE}" 2>/dev/null || echo 'none')"
    if [ "${CURRENT_REV}" != "${CODE_TAR_REVISION}" ]; then
        echo "[bootstrap] revision mismatch (current=${CURRENT_REV} target=${CODE_TAR_REVISION}) — will sync"
        NEED_SYNC=1
    else
        echo "[bootstrap] code at pinned revision ${CODE_TAR_REVISION}"
    fi
else
    # Check tarball etag vs stored etag to detect remote changes
    STORED_ETAG="$(cat "${CODE_VERSION_FILE}" 2>/dev/null || echo 'none')"
    if [ -n "${HF_TOKEN}" ]; then
        REMOTE_ETAG="$(curl -sI -H "Authorization: Bearer ${HF_TOKEN}" \
            "https://huggingface.co/datasets/${CODE_REPO}/resolve/main/${CODE_TAR_PATH}" \
            | grep -i '^etag:' | awk '{print $2}' | tr -d '"\r\n')"
    else
        REMOTE_ETAG=""
    fi
    if [ -n "${REMOTE_ETAG}" ] && [ "${REMOTE_ETAG}" != "${STORED_ETAG}" ]; then
        echo "[bootstrap] remote etag changed (${STORED_ETAG} → ${REMOTE_ETAG}) — will sync"
        NEED_SYNC=1
    else
        echo "[bootstrap] code up-to-date or HF sync unavailable (etag=${REMOTE_ETAG:-none})"
    fi
fi

if [ "${NEED_SYNC}" = "1" ]; then
    if [ -z "${HF_TOKEN}" ]; then
        echo "[bootstrap] cannot sync code from HF without HF_TOKEN" >&2
        exit 1
    fi
    echo "[bootstrap] syncing code from HF ${CODE_REPO}/${CODE_TAR_PATH}"
    RESOLVE_URL="https://huggingface.co/datasets/${CODE_REPO}/resolve/${CODE_TAR_REVISION:-main}/${CODE_TAR_PATH}"
    curl -sfL -H "Authorization: Bearer ${HF_TOKEN}" \
        -o "${SCRATCH}/metacognition.tar.gz" \
        "${RESOLVE_URL}"
    # Back up existing and extract fresh
    if [ -d "${CODE_DIR}" ]; then
        rm -rf "${CODE_DIR}.prev"
        mv "${CODE_DIR}" "${CODE_DIR}.prev" 2>/dev/null || true
    fi
    mkdir -p "${CODE_DIR}"
    tar xzf "${SCRATCH}/metacognition.tar.gz" -C "${SCRATCH}/"
    # Record revision / etag for next bootstrap
    if [ -n "${CODE_TAR_REVISION}" ]; then
        echo "${CODE_TAR_REVISION}" > "${CODE_VERSION_FILE}"
    else
        echo "${REMOTE_ETAG:-unknown}" > "${CODE_VERSION_FILE}"
    fi
    echo "[bootstrap] code extracted to ${CODE_DIR}"
fi

# ── Stage 4: export tokens ──
cat > "${SCRATCH}/env.sh" <<EOF
export HF_TOKEN='${HF_TOKEN}'
export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}'
export WANDB_API_KEY='${WANDB_KEY}'
export PYTHONPATH='${CODE_DIR}:\${PYTHONPATH:-}'
export VLLM_USE_V1='1'
source /opt/conda/etc/profile.d/conda.sh
conda activate simplerl
EOF
echo "[bootstrap] wrote ${SCRATCH}/env.sh — 'source /scratch/env.sh' before training"

# ── Stage 5: verify GPU ──
nvidia-smi | head -20

# ── Summary ──
NGPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "[bootstrap] DONE — ${NGPU}x ${GPU_NAME}"
