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

tar czf "${OUT}" -C "${STAGE_DIR}" metacognition
echo "${OUT}"
