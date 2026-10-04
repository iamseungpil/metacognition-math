#!/usr/bin/env bash
# 수정 습관 팔 체크포인트 평가 한 벌: 병합 → L5 800×8 롤아웃(8k) → math_revision_eval(base 대조).
# 사용: bash scripts/local/eval_rev_ckpt.sh <LINEAGE> <STEP>   예) cd9_M_G1_s1_r8192 25
set -euo pipefail
LIN="$1"; STEP="$2"
source scripts/local/env.sh >/dev/null 2>&1 || true
export HF_HOME="${WORK}/hf_home"
C="${WORK}/checkpoints/${LIN}/global_step_${STEP}"
M="${WORK}/models/merged_${LIN}_step${STEP}"
# 런처가 이미 병합해 둔 사본이 있으면 그것을 쓴다(런처는 병합 뒤 raw 가중치를 지운다).
[ -d "${WORK}/merged/${LIN}/step_${STEP}" ] && [ -f "${WORK}/merged/${LIN}/step_${STEP}/config.json" ] && M="${WORK}/merged/${LIN}/step_${STEP}"
O="${WORK}/eval/mathL5_${LIN}_step${STEP}_b8k"
BASE="${WORK}/eval/mathL5_q3i2507_opt_b8k/texts.jsonl"
[ -d "$M" ] || /hdd_data/seungpil/envs/verl09/bin/python -m verl.model_merger merge --backend fsdp --local_dir "$C/actor" --target_dir "$M"
[ -f "$O/texts.jsonl" ] || /hdd_data/seungpil/envs/qwen35/bin/python scripts/local/math_rollout.py \
  --dataset "parquet:${WORK}/data/math_eval_L5_800.parquet" --model_path "$M" --variant math_opt \
  --num_samples 8 --seed 11 --max_tokens 8192 --gpu_util 0.80 --out_dir "$O"
/hdd_data/seungpil/envs/qwen35/bin/python scripts/local/math_revision_eval.py --rollouts "$O/texts.jsonl" --compare "$BASE" --out "${WORK}/eval/revision_eval_${LIN}_step${STEP}"
echo "[eval_rev_ckpt] done -> ${WORK}/eval/revision_eval_${LIN}_step${STEP}/summary.md"
