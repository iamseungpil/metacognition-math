#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <model_path> <input_path> <output_dir> <dataset_mode:{naive|epistemic}> <claim_bearing:{0|1}> [example_bank ...]" >&2
  exit 1
fi

bash scripts/run_self_distill_roundtrip.sh fixed_k_repair "$@"
