#!/bin/bash
# 수정 70 RLRT 세팅 맞춤 — a75 와 같은 단계(앞부분 만들기·거르기·관문), 출처 SRC(기본 dapo, a77_dapo.py)·모델 M2(기본 주 모델 2507)·꼬리표 TG(기본 dapo).
#   pool h · prep h · screen h · gate · base · train u  (env M2 · TG · DISTILL · KL · GUARD · SEED · NOPFX · RLEN)
set -euo pipefail
ROOT=${ROOT:-/hdd_data/seungpil}; R_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=${REPO:-$(cd "$R_DIR/../.." && pwd)}   # scripts/remote 복사본: 경로는 ROOT·REPO 로(기본 = 이 박스)
export ROOT REPO
cd "$REPO"
export PROMPT_VARIANT=plain HF_HOME=$ROOT/scratch/hf_home VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
PY=${PY:-$ROOT/envs/verl09/bin/python}
D=$ROOT/scratch; A=$R_DIR; NCH=12
M2=${M2:-$D/models/Qwen3-4B-Instruct-2507}; TG=${TG:-dapo}; SRC=${SRC:-dapo}
OUT=$D/data/mc_pfx_commit_${TG}_screen_plain_le6000.parquet
wait_for() { for _ in $(seq $((${WAIT_H:-2} * 30))); do local ok=1; for f in "$@"; do [ -s "$f" ] || ok=0; done
             [ $ok = 1 ] && return 0; sleep 120; done; echo "[a77] 기다림 초과: $*" >&2; return 1; }
case "$1" in
  pool)
    MC_EAGER=0 $PY -m mc.pool --dataset parquet:$D/data/${SRC}_src_$2.parquet --model_path $M2 --k 4 --max_tokens 8192 --seed 7 \
      --select all --out $D/data/${TG}_pool_$2.parquet ;;
  prep)
    wait_for $D/data/${TG}_pool_$2.summary.json
    $PY $A/a75_model2.py mid $TG $2
    $PY -m mc.shift_check --boxes --only_selected --pool_texts $D/data/${TG}_mid_$2.texts.jsonl --model_path $M2 --out_dir $D/eval/box_scores_${TG}_$2
    $PY -m mc.pool --pfx_from $D/data/${TG}_mid_$2.texts.jsonl --commit_scores $D/eval/box_scores_${TG}_$2/box_scores.jsonl \
      --probe_selection $D/eval/probe_commit_selection.json --val $D/data/math_val_plain.parquet --model_path $M2 \
      --out $D/data/${TG}_pfx_$2.parquet
    $PY -m mc.pool --pfx_screen $D/data/${TG}_pfx_$2.parquet --out $D/eval/${TG}_sel_$2.tmp.json
    mv $D/eval/${TG}_sel_$2.tmp.json $D/eval/${TG}_sel_$2.json ;;
  screen)
    wait_for $D/eval/${TG}_sel_$2.json
    MC_EAGER=0 $PY -m mc.probe --model_path $M2 --selection $D/eval/${TG}_sel_$2.json --k 8 --max_tokens 8192 --seed 11 \
      --out_dir $D/eval/${TG}_screen_$2 ;;
  gate)
    [ -s $OUT ] || { WAIT_H=3 wait_for $(for h in $(seq 0 $((NCH - 1))); do echo $D/eval/${TG}_screen_$h/summary.json; done) || true
                     $PY $A/a75_model2.py weights $TG $M2; }
    [ -s $D/data/mc_nopfx_${TG}.parquet ] || $PY $A/a75_model2.py nopfx $TG ;;
  base)                                                    # 두 번째 모델 base 의 hdev(32k k8) — 학습 팔과 같은 시험
    MC_EAGER=0 $PY -m mc.eval --protocol single --max_tokens 32768 --model_path $M2 --dataset parquet:$D/data/hdev_all.parquet \
      --k 8 --seed 11 --out_dir $D/eval/hdevall_${TG}base_k8 ;;
  train)
    WAIT_H=8 wait_for ${OUT%.parquet}.summary.json
    [ -s $OUT ] || { echo "[a77] 관문 미달 — 학습 안 함" >&2; exit 3; }
    NP=$([ "${NOPFX:-}" = 1 ] && echo np || true); RL=${RLEN:-8192}; SD=${SEED:-2}
    TR=$([ -n "$NP" ] && echo $D/data/mc_nopfx_${TG}.parquet || echo $OUT)
    T=u${DISTILL:+ds$DISTILL}${KL:+kl$KL}$TG$NP$([ "$RL" = 8192 ] || echo "r$RL")$([ "$SD" = 2 ] || echo "sd$SD")
    L=mc_SPONT_PFX_gold_s${SD}_r${RL}_tm_bw2_rp0.25${DISTILL:+_ds$DISTILL}${KL:+_kl$KL}_$TG$NP
    [ -e $D/checkpoints/$L/TRAIN_DONE ] || env -u PFX_FORK -u PFX_ADV_CAP -u PFX_DISTILL -u PFX_DISTILL_SHUF -u PFX_KEEP -u KL_COEF -u PFX_GUARD_ABS \
      ${DISTILL:+PFX_DISTILL=$DISTILL} ${KL:+KL_COEF=$KL} ${GUARD:+PFX_GUARD_ABS=$GUARD} PFX_TRUNC=mask PFX_BREAK_W=2 PFX_REP=0.25 \
      LR_SCHED=constant REF_OFFLOAD=1 ACTOR_OFFLOAD=1 LABEL=gold PFX_WEIGHT_KEY=sample_weight LR=5e-6 TRAIN_BATCH=32 RESP_LEN=$RL MAX_PROMPT=6000 \
      DATA_VAL=$D/data/math_val_plain.parquet VERL_ENV=$ROOT/envs/verl09 EVAL_STEPS=0 SAVE_FREQ=10 TAG=$TG$NP MODEL_PATH=$M2 \
      DATA_TRAIN=$TR bash mc/run.sh SPONT_PFX $SD 20
    M=$($PY -m mc.probe --merge_only $D/checkpoints/$L/global_step_20/actor)
    MC_EAGER=0 $PY -m mc.eval --protocol single --max_tokens 32768 --model_path $M --dataset parquet:$D/data/hdev_all.parquet \
      --k 8 --seed 11 --out_dir $D/eval/hdevall_${T}_s20_k8 ;;
  *) echo "usage: $0 pool h | prep h | screen h | gate | base | train u" >&2; exit 2 ;;
esac
echo "[a77] $* done"
