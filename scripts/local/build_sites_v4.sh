#!/usr/bin/env bash
# scripts/local/build_sites_v4.sh
#
# E-133 오염 수리: `sites_v1`(및 v2/v3/v3c) 은 held-out val(countdown_val_4num_*,
# 500문제)에서 뽑은 평가 롤아웃에서 자리를 채굴해, RL/SFT 처치 팔이 held-out
# 문제로 학습했다(N0/OPT 는 아니었음 — 팔 간 비교 자체가 무효). 이 스크립트는
# TRAIN 셋 롤아웃(`train1500_gs0_new`, `train1500_A_new` — 1500 train 문제 × 8)
# 에서 다시 채굴해 `$WORK/data/sites_v4/` 아래 깨끗한 산출물을 만든다.
#
# 산출물:
#   sites_train.parquet        자리(재개 학습용)
#   sites_judge.parquet        자리(판정용)
#   mixed_train_v4.parquet     normal(new) 50% + site 50%
#   mixed_train_v4_opt.parquet 위와 같되 시스템 메시지만 opt(메타 허용만)
#   mixed_train_v4_cf_opt.parquet  opt 판의 반사실 쌍둥이(main/twin 쌍 선두)
#   summary.json               build_sites.py 가 남긴 컷 종류/family_dead/버킷 집계
#
# 하드 불변: 위 다섯 parquet 전부 (nums,target) 이 countdown_val_4num_opt.parquet
# 과 절대 겹치지 않는다 — check_no_val_overlap.py 가 겹치면 이 스크립트를 실패시킨다.
#
# CPU 전용 — GPU 를 만지지 않는다.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

PY="${PY:-/hdd_data/seungpil/envs/simplerl/bin/python}"
OUT_DIR="${WORK}/data/sites_v4"
mkdir -p "${OUT_DIR}"

N_TRAIN="${N_TRAIN:-3000}"
N_JUDGE="${N_JUDGE:-1000}"
SEED="${SEED:-7}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1900}"
ROLLOUTS="gs0=eval/train1500_gs0_new/texts.jsonl,A=eval/train1500_A_new/texts.jsonl"
TRAIN_PARQUET="data/countdown_train_4num_new.parquet"
VAL_PARQUET="${WORK}/data/countdown_val_4num_opt.parquet"

for tag_path in "eval/train1500_gs0_new/texts.jsonl" "eval/train1500_A_new/texts.jsonl"; do
  f="${WORK}/${tag_path}"
  if [ ! -s "${f}" ]; then
    echo "[build_sites_v4] FATAL: missing/empty rollout file ${f}" >&2
    echo "[build_sites_v4]   (두 롤아웃 잡이 아직 안 끝났을 수 있다 — 완료 후 재실행)" >&2
    exit 1
  fi
done

echo "[build_sites_v4] [1/4] build_sites.py -> ${OUT_DIR}"
"${PY}" "${_SCRIPT_DIR}/build_sites.py" \
  --out_dir "${OUT_DIR}" \
  --rollouts "${ROLLOUTS}" \
  --train_parquet "${TRAIN_PARQUET}" \
  --n_train "${N_TRAIN}" --n_judge "${N_JUDGE}" --seed "${SEED}" \
  --max_prompt_tokens "${MAX_PROMPT_TOKENS}"

mv -f "${OUT_DIR}/mixed_train.parquet" "${OUT_DIR}/mixed_train_v4.parquet"
echo "[build_sites_v4]   -> ${OUT_DIR}/mixed_train_v4.parquet"

echo "[build_sites_v4] [2/4] opt 판(시스템 메시지만 permission 프롬프트로 교체)"
"${PY}" "${_SCRIPT_DIR}/make_opt_variant.py" \
  --in "${OUT_DIR}/mixed_train_v4.parquet" \
  --out "${OUT_DIR}/mixed_train_v4_opt.parquet" \
  --variant opt

echo "[build_sites_v4] [3/4] 반사실 쌍둥이(OPT_CF)"
"${PY}" "${_SCRIPT_DIR}/build_cf_twins.py" \
  --in "${OUT_DIR}/mixed_train_v4_opt.parquet" \
  --out "${OUT_DIR}/mixed_train_v4_cf_opt.parquet"

echo "[build_sites_v4] [4/4] val 겹침 하드 불변 검사"
"${PY}" "${_SCRIPT_DIR}/check_no_val_overlap.py" \
  --val "${VAL_PARQUET}" \
  "${OUT_DIR}/sites_train.parquet" \
  "${OUT_DIR}/sites_judge.parquet" \
  "${OUT_DIR}/mixed_train_v4.parquet" \
  "${OUT_DIR}/mixed_train_v4_opt.parquet" \
  "${OUT_DIR}/mixed_train_v4_cf_opt.parquet"

echo "[build_sites_v4] done. summary:"
cat "${OUT_DIR}/summary.json"
