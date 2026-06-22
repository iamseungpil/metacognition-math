#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/metacognition_sdc_code.tar.gz}"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGE_DIR}"' EXIT

mkdir -p "${STAGE_DIR}/metacognition"
cd "${ROOT}"

for path in \
    src \
    scripts \
    configs \
    tests \
    docs \
    data \
    ANALYSIS_MAP.md \
    REPORT_REFERENCES.md \
    README.md \
    run_c_h200_basic.yaml \
    node_recovery_0415.yaml
do
    if [ -e "${path}" ]; then
        cp -a "${path}" "${STAGE_DIR}/metacognition/"
    fi
done

rm -rf \
    "${STAGE_DIR}/metacognition/.git" \
    "${STAGE_DIR}/metacognition/gnosis_repo" \
    "${STAGE_DIR}/metacognition/legacy" \
    "${STAGE_DIR}/metacognition/checkpoints" \
    "${STAGE_DIR}/metacognition/results" \
    "${STAGE_DIR}/metacognition/data/"*.rejections.jsonl 2>/dev/null || true

find "${STAGE_DIR}/metacognition" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${STAGE_DIR}/metacognition" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

# Redaction pass: cp -a bypasses .gitattributes export-ignore (that only applies
# to `git archive`), so legacy scripts with hardcoded tokens get staged. Scrub
# secret literals from the STAGED copies (repo originals untouched) before tar.
# Patterns: GitHub PAT (ghp_+36), HuggingFace (hf_+34), and 40-hex WANDB keys —
# the WANDB rule is line-scoped (must mention WANDB/wandb_key/api_key) so bare
# 40-hex git commit SHAs are left intact.
find "${STAGE_DIR}/metacognition" -type f \( -name "*.sh" -o -name "*.py" \) -print0 \
    | xargs -0 -r perl -i -pe '
        s/ghp_[A-Za-z0-9]{36}/REDACTED_SECRET/g;
        s/hf_[A-Za-z0-9]{34}/REDACTED_SECRET/g;
        if (/WANDB|wandb_key|api_key/i) {
            s/(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])/REDACTED_SECRET/g;
        }
    '

tar czf "${OUT}" -C "${STAGE_DIR}" metacognition
echo "${OUT}"
