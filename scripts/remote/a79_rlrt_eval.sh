#!/usr/bin/env bash
# 수정 70 RLRT 평가 규약(Kim 외 2026 부록 G): avg@16·pass@16 · 온도 .7 · top-p .8 · top-k 20 · 최대 38,912 말.
# 시험 = AIME24·AIME25·HMMT25·MATH500(로컬) + 어려운 개발셋 hdev(k8) + 있으면 data/eval_extra_*.parquet(내려받은 AIME26·HMMT26·AMC23·OOD).
# 10-14: 생성만 GPU 에서 하고 채점·요약은 a79_gen.py 가 CPU 백그라운드로(summary.json 은 몇 분 늦게 생김; 내용은 mc.eval 과 같음).
# 원격 추가: BENCHES="태그:hf:<id>:<split>:<k> …" 로 시험 목록을 통째로 바꾼다(R1: amc23·minerva) · MC_GPU_UTIL/MC_MAX_NUM_SEQS 로 40GB 카드 조정 · 평가 시험 분포는 원본과 같음.
# 사용: bash a79_rlrt_eval.sh 이름=모델경로 …  → eval/rl_<이름>_<시험>
set -euo pipefail
ROOT=${ROOT:-/hdd_data/seungpil}; R_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=${REPO:-$(cd "$R_DIR/../.." && pwd)}   # scripts/remote 복사본: 경로는 ROOT·REPO 로(기본 = 이 박스)
export ROOT REPO
cd "$REPO"
export PROMPT_VARIANT=plain HF_HOME=$ROOT/scratch/hf_home HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1} VLLM_USE_DEEP_GEMM=0 MC_EAGER=0
PY=${PY:-$ROOT/envs/verl09/bin/python}; D=$ROOT/scratch; A=$R_DIR
S="--temperature 0.7 --top_p 0.8 --top_k 20 --max_tokens 38912 --seed 11"
for nm in "$@"; do
  n=${nm%%=*}; m=${nm#*=}
  for b in ${BENCHES:-aime24:hf:HuggingFaceH4/aime_2024:train:16 aime25:hf:math-ai/aime25:test:16 hmmt25:hf:MathArena/hmmt_feb_2025:train:16 \
           math500:hf:HuggingFaceH4/MATH-500:test:16 hdev:parquet:$D/data/hdev_all.parquet:8 \
           $(for f in $D/data/eval_extra_*.parquet; do [ -e "$f" ] && t=${f##*/eval_extra_} && echo "${t%.parquet}:parquet:$f:16"; done)}; do
    tag=${b%%:*}; rest=${b#*:}; k=${rest##*:}; ds=${rest%:*}; o=$D/eval/rl_${n}_${tag}
    [ -s $o/summary.json ] || [ -s $o/texts.jsonl ] || $PY $A/a79_gen.py --protocol single --model_path $m --dataset $ds --k $k $S --out_dir $o
  done
done
echo "[a79] $* done"
