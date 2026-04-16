#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python scripts/run_smoke.py
python scripts/extract_behavior_uncertainty.py \
  --targets configs/targets.json \
  --outdir results
python scripts/run_critic.py
python scripts/render_report.py
echo "loop_ok"
