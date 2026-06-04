"""Centralized environment + paths for CTSD experiments.

All env vars and model/data paths consolidated here. Single source of truth.
Karpathy: no magic strings scattered across scripts.
"""
from __future__ import annotations
import os
from pathlib import Path

# Models
TEACHER_MODEL = os.environ.get(
    "CTSD_TEACHER", "/home/v-seungplee/sft_e20a_local"
)
STUDENT_R10V2 = os.environ.get(
    "CTSD_STUDENT_R10V2", "/home/v-seungplee/student_ckpts/R10v2_merged_step306"
)
SFT_V8_STRICT = os.environ.get(
    "CTSD_SFT_V8_STRICT",
    "/home/v-seungplee/sft_v8_strict_local/models/v8_meta_inside_strict_sft/checkpoint-254",
)

# Eval data (HuggingFace)
HF_DATASET = "iamseungpil/metacot"
EVAL_R10V2_V8 = "eval/R10v2_step300_16k_2026_05_17/R10v2_step300_16k.json"
EVAL_R10V2_E20A = "eval/r10v2_e20a_step275_16k_node/r10v2_e20a_step275_16k.json"

# Special token IDs (Qwen3 tokenizer, verified 2026-05-26)
META_OPEN_ID = 151669      # <|meta|>
META_CLOSE_ID = 151670     # <|/meta|>

# Output paths. Default is the local-A100 host path (unchanged); override via
# CTSD_REPORTS_DIR on cluster nodes where /home/v-seungplee is not writable
# (e.g. amlt H200 → /scratch/reports). Behaviour-preserving: same default locally.
REPORTS_DIR = Path(os.environ.get("CTSD_REPORTS_DIR", "/home/v-seungplee/metacognition/reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
DEFAULT_SEED = 20260528
STRATIFIED_SAMPLE_SEED = 20260528  # fixed for paper-grade reproducibility

# AGL = agl-dev Azure OpenAI endpoint (gpt-5.5). NOT Agent Lightning.
# Codex (`codex exec`) already uses this — confirmed working.
# Auth = runtime AAD JWT (AZURE_OPENAI_API_KEY, ~1h expiry), injected by codex,
# so NOT a static .env key. For LLM-generation tasks (e.g. meta rewrite),
# prefer calling `codex exec` as the gpt-5.5 gateway (no key management).
AGL_ENDPOINT = os.environ.get("AGL_ENDPOINT", "https://agl-dev.cognitiveservices.azure.com/openai")
AGL_MODEL = os.environ.get("AGL_MODEL", "gpt-5.5")
AGL_API_VERSION = os.environ.get("AGL_API_VERSION", "2025-04-01-preview")


def agl_generate(prompt: str, timeout: int = 120) -> str:
    """Call gpt-5.5 via codex exec (the agl-dev gateway). Returns model text.

    Uses codex as the authenticated gateway so we don't manage the AAD JWT.
    For batch generation, call repeatedly or use the live AZURE_OPENAI_API_KEY
    JWT directly with an Azure OpenAI client if available in env.
    """
    import subprocess
    r = subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", "--color", "never"],
        input=prompt, capture_output=True, text=True, timeout=timeout,
    )
    return r.stdout
