#!/bin/bash
# R1 평가 전용 잡(GPU 1장, 40GB 가능): r1_eval.sh EVAL_STEM SOURCE
#   SOURCE = base(Qwen3-4B-Instruct-2507, get_data.sh 가 받아 둠) | HF 모델 저장소(iamseungpil/metacot-cd9-models) 안 폴더 이름(병합된 step-60 모델)
#   평가 규약 = a79_rlrt_eval.sh 와 같다(온도 .7 · top-p .8 · top-k 20 · 최대 38,912 말 · seed 11). 시험 = AMC23·Minerva Math 두 개를 k=16.
#   출력 = $ROOT/scratch/eval/rl_<EVAL_STEM>_{amc23,minerva}/ (a79 이름 규약 — 이 서버 분석 스크립트가 같은 이름으로 찾는다).
#   env: BENCHES(시험 목록 덮어쓰기) · MC_GPU_UTIL(기본 .80) · MC_MAX_NUM_SEQS(기본 128, 40GB 용) · KEEP_MODEL=1(받은 모델 안 지움) · RUN_NAME(기본 EVAL_STEM)
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/hf_env.sh"
STEM=${1:?EVAL_STEM}; SRC=${2:?SOURCE}; D=$ROOT/scratch; RUN=${RUN_NAME:-$STEM}
BENCHES=${BENCHES:-"amc23:hf:math-ai/amc23:test:16 minerva:hf:math-ai/minervamath:test:16"}; export BENCHES
export MC_MAX_NUM_SEQS=${MC_MAX_NUM_SEQS:-128} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
mkdir -p $D/remote_meta
printf "EVAL_GLOB=rl_%s_amc23 rl_%s_minerva\nMODEL=%s\n" "$STEM" "$STEM" "$SRC" > $D/remote_meta/$RUN.env
if [ "$SRC" = base ]; then M=$D/models/Qwen3-4B-Instruct-2507
else M=$("$PY" -I "$R_DIR/hf_sync.py" fetch-model "${MODELS_REPO:-iamseungpil/metacot-cd9-models}" "$SRC" | tail -1); fi
[ -s "$M/config.json" ] || { echo "[r1] 모델 없음: $M" >&2; exit 2; }
bash "$R_DIR/a79_rlrt_eval.sh" "$STEM=$M"
[ "$SRC" = base ] || [ "${KEEP_MODEL:-}" = 1 ] || rm -rf "$M"
echo "[r1] $STEM done"
