#!/bin/bash
# Idempotent bootstrap for shared veRL training on an AMLT Singularity node.
# Works for SDC and baseline GRPO launchers on H100/H200/A100-class nodes.
#
# Stages:
#   1. Create conda env 'simplerl' (skip if /scratch/simplerl_v3.done exists)
#   2. Install torch + veRL 0.7.1 + compatible vLLM/Ray stack
#      Wheels pulled from HF CDN when present (faster than PyPI on BSC).
#   3. Sync code to /scratch/metacognition from HF code_snapshot tarball
#      (see memory: feedback_hf_bootstrap — no SSH base64 for big uploads)
#   4. Write /scratch/simplerl_v3.done sentinel + export HF/WANDB tokens
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
DONE_MARKER="${SCRATCH}/simplerl_v3.done"
CODE_DIR="${SCRATCH}/metacognition"
CODE_VERSION_FILE="${CODE_DIR}/.bootstrap_version"
HF_CACHE_DIR="${HF_CACHE_DIR:-${SCRATCH}/hf_cache}"
HF_HOME="${HF_HOME:-${HF_CACHE_DIR}}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_CACHE_DIR}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_CACHE_DIR}/transformers}"

echo "[bootstrap] scratch=${SCRATCH} done=${DONE_MARKER}"
mkdir -p "${SCRATCH}"
mkdir -p "${HF_CACHE_DIR}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}"
chmod 700 "${HF_CACHE_DIR}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" 2>/dev/null || true
export HF_HOME
export HUGGINGFACE_HUB_CACHE
export TRANSFORMERS_CACHE

# YAML's `pip install --user huggingface_hub` (no version pin) installs hub 1.x
# to ~/.local which then SHADOWS the env's hub 0.36.2 (env site is loaded after
# user site for some Python configs). transformers 4.57.6 requires hub<1.0 —
# so we remove the user-site hub and disable user-site for env Python.
rm -rf "${HOME}/.local/lib/python"*"/site-packages/huggingface_hub"* 2>/dev/null || true
export PYTHONNOUSERSITE=1

# ── Stage 0: quick exit if already bootstrapped ──
if [ -f "${DONE_MARKER}" ]; then
    echo "[bootstrap] ${DONE_MARKER} exists — env already installed, skipping install"
    ENV_OK=1
else
    ENV_OK=0
fi

# ── Stage 1+2: conda env + packages ──
if [ "${ENV_OK}" -eq 0 ]; then
    source /opt/conda/etc/profile.d/conda.sh

    # Fast path: conda-pack tarball (~5GB) — bypass pip install entirely.
    # Built locally with all verl source patches baked in. Pulls from HF in ~30s
    # vs the pip-install path which takes 10-15 min and breaks on package
    # availability changes.
    PACK_URL="https://huggingface.co/datasets/${CODE_REPO}/resolve/main/env_snapshots/simplerl_v3.tar.gz"
    PACK_PATH="${SCRATCH}/simplerl_v3.tar.gz"
    # E.4 fix: /opt/conda is READ-ONLY on Basic-tier nodes (mkdir Permission
    # denied), and the amlt YAML runs ${SIMPLERL_DIR}/bin/python where the YAML
    # sets SIMPLERL_DIR=/scratch/conda_envs/simplerl. Respect that env var and
    # default to the WRITABLE /scratch path so the conda-pack env lands where the
    # run command looks for it.
    SIMPLERL_DIR="${SIMPLERL_DIR:-/scratch/conda_envs/simplerl}"

    if [ ! -d "${SIMPLERL_DIR}/bin" ] && [ -n "${HF_TOKEN}" ]; then
        echo "[bootstrap] fast-path: pulling conda-pack env from HF -> ${SIMPLERL_DIR}"
        if curl -sfL -H "Authorization: Bearer ${HF_TOKEN}" -o "${PACK_PATH}" "${PACK_URL}"; then
            echo "[bootstrap] extracting env (~5GB → ${SIMPLERL_DIR})"
            mkdir -p "${SIMPLERL_DIR}"
            tar -xzf "${PACK_PATH}" -C "${SIMPLERL_DIR}"
            "${SIMPLERL_DIR}/bin/conda-unpack" 2>/dev/null || true
            rm -f "${PACK_PATH}"
            conda activate "${SIMPLERL_DIR}" 2>/dev/null || true
            echo "[bootstrap] fast-path complete"
        else
            echo "[bootstrap] HF env tarball miss — falling back to pip install"
            rm -f "${PACK_PATH}"
        fi
    fi

    if [ ! -d "${SIMPLERL_DIR}/bin" ]; then
        echo "[bootstrap] slow-path: creating simplerl conda env + pip install"
        if ! conda env list | grep -qE '^simplerl\b'; then
            conda create -n simplerl python=3.10 -y
        fi
        conda activate simplerl

        pip install --upgrade pip setuptools wheel

    # vllm 0.10.2 is the "verified working" line per requirements.txt.
    # vllm 0.17.0 was tried but its torch 2.8 + cuda 12.8 transitive install
    # exceeds 60 min on AMLT BSC nodes, causing bootstrap timeouts.
    # Our verl source patches (collective_rpc/reset_mm_cache hasattr guards,
    # logprobs-mode strip) handle vllm 0.10.x missing APIs.
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126

    # Core ML deps — omegaconf pinned to <2.4 because verl 0.7.1 uses
    # `from omegaconf import MISSING` which was removed in omegaconf 2.4+.
    # omegaconf 2.3.0 is the only non-dev 2.3.x release on PyPI; pin exactly to
    # avoid resolver failure. math-verify[antlr4_9_3] aligns with omegaconf
    # 2.3.0's antlr4-python3-runtime==4.9.* requirement (the [antlr4_11_0]
    # variant bumps antlr to 4.11.0 which conflicts with omegaconf<2.4).
    pip install accelerate datasets dill hydra-core "omegaconf==2.3.0" numpy pybind11 \
        tensordict "transformers==4.57.6" peft liger-kernel word2number \
        "math-verify[antlr4_9_3]==0.6.0" deepspeed wandb pandas pyarrow pyyaml
    pip install "trl==0.19.1"

    # vLLM 0.8.3 + verl 0.7.1 runtime deps — installed explicitly because we
    # invoke both with --no-deps below to avoid ray resolver conflicts. These
    # are the imports vllm and verl actually need at runtime (msgspec in
    # vllm/sequence.py, torchdata for verl ray_trainer, etc.). Missing any of
    # them silently breaks training at launch.
    pip install msgspec openai prometheus-fastapi-instrumentator \
        partial-json-parser lm-format-enforcer "outlines==0.1.11" cloudpickle tiktoken \
        pyzmq prometheus_client blake3 msgpack lark einops torchdata codetiming \
        cachetools compressed-tensors py-cpuinfo importlib_metadata gguf \
        mistral-common depyf pillow psutil setproctitle aiohttp fastapi \
        uvicorn uvloop watchfiles httptools websockets python-multipart \
        llguidance "xgrammar==0.1.17" numba

    pip install "ray[cgraph]==2.43.0"

    # vllm 0.10.2 — cu126 compatible, fast install. Patched at runtime via
    # the verl source sed below to handle missing collective_rpc / reset_mm_cache
    # / new init_app_state signature.
    pip install "vllm==0.10.2"

    # veRL 0.7.1 — install with --no-deps to avoid resolver pulling old pins.
    pip install --no-deps "verl==0.7.1"

        # Ensure pkg_resources stays available (some pip resolver chains strip it).
        pip install --upgrade --force-reinstall setuptools 2>&1 | tail -3 || true
    fi  # end of slow-path pip-install fallback

    # ── Activate env (works for both fast-path and slow-path) ──
    conda activate simplerl

    # vLLM 0.10.2 transitively pulls opencv-python-headless. cv2 import
    # SEGFAULTs on the AMLT acpt-torch2.7.1 image (libGL/glibc mismatch),
    # which kills the Ray worker during verl→mistral_common→cv2 import chain.
    # Removing it makes mistral_common's `is_opencv_installed()` return False
    # and skip the import. opencv isn't actually used by verl/vllm rollout.
    pip uninstall -y opencv-python-headless 2>&1 | tail -2 || true

    # flash-attn — required by verl 0.7.1's left_right_2_no_padding /
    # unpad_input path (engine_workers.compute_old_log_prob). Conda-pack
    # snapshot pre-dates this discovery, so install on-demand into the
    # simplerl site-packages via --target (plain pip install lands in the
    # wrong env on these AMLT images). Idempotent: skips if already present.
    if ! /opt/conda/envs/simplerl/bin/python -c "import flash_attn" 2>/dev/null; then
        echo "[bootstrap] installing flash-attn into simplerl"
        /opt/conda/envs/simplerl/bin/python -m pip install --no-build-isolation --no-deps \
            --target /opt/conda/envs/simplerl/lib/python3.10/site-packages \
            --upgrade flash-attn==2.8.3 2>&1 | tail -3 || \
            echo "[bootstrap] flash-attn install failed — verl unpad path will crash"
    else
        echo "[bootstrap] flash-attn already present in simplerl"
    fi

    # Write env.sh BEFORE the verify heredoc so a verify failure cannot leave
    # launch_sdc_verl.sh stranded on the ptca default env. Bootstrap.done is
    # touched by the tmux wrapper regardless of this script's exit status, so
    # the only way to guarantee simplerl activation downstream is to emit the
    # sentinel file eagerly — any later re-write is a no-op.
    cat > "${SCRATCH}/env.sh" <<EOF
export HF_TOKEN='${HF_TOKEN}'
export HUGGING_FACE_HUB_TOKEN='${HF_TOKEN}'
export WANDB_API_KEY='${WANDB_KEY}'
export PYTHONPATH='${CODE_DIR}:\${PYTHONPATH:-}'
export VLLM_USE_V1='1'
export PYTHONNOUSERSITE='1'
export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
source /opt/conda/etc/profile.d/conda.sh
conda activate simplerl
EOF
    echo "[bootstrap] early env.sh written — launch scripts will activate simplerl"

    # Verify imports (non-fatal: hydra's pkg_resources dep has flipped in
    # recent setuptools releases, which we've seen break only the verify step
    # even though the underlying packages work. `|| true` keeps bootstrap.done
    # from being a lie when only the diagnostic import crashes.)
    python - <<'PY' || true
import torch, ray, hydra, omegaconf
import vllm, verl
import trl
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}")
print(f"ray={ray.__version__} hydra={hydra.__version__}")
print(f"vllm={vllm.__version__} verl={verl.__version__} trl={trl.__version__}")
PY

    # veRL 0.7.1 + vLLM 0.8.x async server compatibility:
    # some vLLM builds route through a V0 fallback while veRL still calls the
    # V1 AsyncLLM entrypoint, which raises "VLLM_USE_V1=False". Patch the
    # installed veRL launcher to fall back to AsyncLLMEngine explicitly.
    #
    # IMPORTANT: do NOT `import verl.workers...vllm_async_server` here — that
    # module crashes on load in vllm 0.8.3 because `run_headless` is a 0.9+
    # symbol. Resolve the file path via `import verl` (package __init__ is
    # side-effect free) and patch the file textually before anything imports it.
    python - <<'PY'
from pathlib import Path
import verl
import os.path

vas_path = Path(os.path.dirname(verl.__file__)) / "workers/rollout/vllm_rollout/vllm_async_server.py"
path = vas_path
text = path.read_text()

if "from vllm.engine.async_llm_engine import AsyncLLMEngine" not in text:
    text = text.replace(
        "from vllm.v1.engine.async_llm import AsyncLLM\n",
        "from vllm.v1.engine.async_llm import AsyncLLM\nfrom vllm.engine.async_llm_engine import AsyncLLMEngine\n",
    )

# Guard run_headless import (vllm 0.9+ has it, vllm 0.8.3 doesn't)
old_rh = "from vllm.entrypoints.cli.serve import run_headless"
new_rh = "try:\n    from vllm.entrypoints.cli.serve import run_headless\nexcept ImportError:\n    run_headless = None"
if old_rh in text and new_rh not in text:
    text = text.replace(old_rh, new_rh)

old = "        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)\n"
new = """        try:\n            engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)\n        except ValueError as exc:\n            if \"VLLM_USE_V1=False\" not in str(exc):\n                raise\n            engine_client = AsyncLLMEngine.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)\n"""

if old in text and new not in text:
    text = text.replace(old, new)

old = """        if self.config.prometheus.enable:\n            if self.config.prometheus.served_model_name:\n                # Extract model name from path if it's a full path\n                served_model_name = self.config.prometheus.served_model_name\n                if \"/\" in served_model_name:\n                    # If it's a full path, extract the last part as model name\n                    served_model_name = served_model_name.split(\"/\")[-1]\n                args[\"served_model_name\"] = served_model_name\n\n        # mtp\n        if self.config.mtp.enable and self.config.mtp.enable_rollout:\n            speculative_config = {\n                \"method\": self.config.mtp.method,\n                \"num_speculative_tokens\": self.config.mtp.num_speculative_tokens,\n"""
new = """        prom_cfg = getattr(self.config, \"prometheus\", None)\n        prom_enable = getattr(prom_cfg, \"enable\", False)\n        prom_served_model_name = getattr(prom_cfg, \"served_model_name\", None)\n        if isinstance(prom_cfg, dict):\n            prom_enable = prom_cfg.get(\"enable\", False)\n            prom_served_model_name = prom_cfg.get(\"served_model_name\")\n        if prom_enable:\n            if prom_served_model_name:\n                # Extract model name from path if it's a full path\n                served_model_name = prom_served_model_name\n                if \"/\" in served_model_name:\n                    # If it's a full path, extract the last part as model name\n                    served_model_name = served_model_name.split(\"/\")[-1]\n                args[\"served_model_name\"] = served_model_name\n\n        # mtp\n        mtp_cfg = getattr(self.config, \"mtp\", None)\n        mtp_enable = getattr(mtp_cfg, \"enable\", False)\n        mtp_enable_rollout = getattr(mtp_cfg, \"enable_rollout\", False)\n        mtp_method = getattr(mtp_cfg, \"method\", None)\n        mtp_num_speculative_tokens = getattr(mtp_cfg, \"num_speculative_tokens\", None)\n        if isinstance(mtp_cfg, dict):\n            mtp_enable = mtp_cfg.get(\"enable\", False)\n            mtp_enable_rollout = mtp_cfg.get(\"enable_rollout\", False)\n            mtp_method = mtp_cfg.get(\"method\")\n            mtp_num_speculative_tokens = mtp_cfg.get(\"num_speculative_tokens\")\n        if mtp_enable and mtp_enable_rollout:\n            speculative_config = {\n                \"method\": mtp_method,\n                \"num_speculative_tokens\": mtp_num_speculative_tokens,\n"""

if old in text and new not in text:
    text = text.replace(old, new)

# vLLM 0.10.2 doesn't accept --logprobs-mode CLI arg (added in vLLM 0.11+).
# Strip that key from the args dict so verl 0.7.1 can launch on vllm 0.10.x.
import re
text = re.sub(r'\n\s*"logprobs_mode": self\.config\.logprobs_mode,', "", text)

# verl 0.7.1 union_numpy_dict asserts identical values for same-key fields,
# which trips on `data_source` when GRPO does group sampling (rollout.n > 1).
# Issue: github.com/volcengine/verl/issues/2155
# Patch the protocol.py to skip the assertion for `data_source` (it's a row-id
# index that varies across rollouts in a group, not a tensor to merge).
proto_path = Path(os.path.dirname(verl.__file__)) / "protocol.py"
proto_text = proto_path.read_text()
old_assert = 'assert _deep_equal(tensor_dict1[key], tensor_dict2[key], visited=set()), (\n                f"`{key}` in tensor_dict1 and tensor_dict2 are not the same object."\n            )'
new_assert = 'if key not in ("data_source", "raw_prompt", "raw_prompt_ids", "uid"):\n                assert _deep_equal(tensor_dict1[key], tensor_dict2[key], visited=set()), (\n                    f"`{key}` in tensor_dict1 and tensor_dict2 are not the same object."\n                )'
if old_assert in proto_text and new_assert not in proto_text:
    proto_text = proto_text.replace(old_assert, new_assert)
    proto_path.write_text(proto_text)
    print(f"[bootstrap] patched {proto_path} (skip data_source assert)")

# vLLM 0.8.3 AsyncLLM lacks reset_mm_cache() and collective_rpc(); guard both.
# Idempotent: skip if guard already present (conda-pack env may have it baked).
if "if hasattr(engine_client, 'reset_mm_cache')" not in text:
    text = text.replace(
        "        await engine_client.reset_mm_cache()\n",
        "        if hasattr(engine_client, 'reset_mm_cache'):\n            await engine_client.reset_mm_cache()\n",
    )
if "if hasattr(engine_client, 'collective_rpc')" not in text:
    text = text.replace(
        "        await engine_client.collective_rpc(\n            method=\"monkey_patch_model\", kwargs={\"vocab_size\": len(self.model_config.tokenizer)}\n        )\n",
        "        if hasattr(engine_client, 'collective_rpc'):\n            await engine_client.collective_rpc(\n                method=\"monkey_patch_model\", kwargs={\"vocab_size\": len(self.model_config.tokenizer)}\n            )\n",
    )

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
export HF_HOME='${HF_HOME}'
export HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
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
