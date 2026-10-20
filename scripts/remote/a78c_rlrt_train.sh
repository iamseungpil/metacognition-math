#!/bin/bash
# a78c (10-14): a78b 와 같고 끊김 대비만 더함 — GPU 4·5 를 빌려 쓰다 양보해야 할 수 있어서 10스텝마다 저장(SAVE_FREQ 10) ·
#   최신 체크포인트 하나만 남김(백그라운드 정리: latest_checkpointed_iteration.txt 보다 작은 global_step_* 삭제) · 재시작 시 resume_mode=auto 로 이어 학습.
# a78b (10-13): a78 과 같고 길이·배치만 바꿈 — 응답 12,288(앞부분 팔은 이어쓰기 6,144 = 같은 전체 예산) · 문제 32/스텝 · 미니배치 16(스텝당 갱신 2번 = a78 과 같은 갱신 수). 20,480·64 는 한 H100 에서 update_actor OOM · 스텝당 ≈1.5 h(스모크 10-13).
# 수정 70 RLRT 세팅 맞춤 학습(주 모델 2507, DAPO 데이터) — 팔 하나 = 잡 하나. 레시피(모든 팔 같음): 문제 64/스텝 · 미니배치 32 · 롤아웃 8 ·
# 응답 20,480 · LR 1e-6 · clip .2/.28 · 60스텝(20마다 저장) · mask·BREAK_W 2·REP .25(예비 실험과 같은 바탕). 끝나면 병합 → a79 평가.
#   팔: grpo(앞부분 없음, KL 0) · grpokl(앞부분 + 끈 .05 = 리셋만 뺌) · reset(앞부분 + 리셋 .5, KL 0 = 끈만 뺌) · ours(앞부분 + 리셋 .5 + 끈 .05 =
#       풀이 중간 두 선생님) · rlsd · rlrt · sdpo(수정 72; 앞부분 없음, KL 0). grpokl·reset·ours 셋이 «두 선생님이 함께 있어야 한다» 를 가른다(수정 70c).
#   모든 팔 같은 절대 파괴 가드 .12 · vLLM 메모리 .4(두 오프로드 켬; .6 시작 실패·.55 는 old_log_prob 에서 11.7 GiB OOM, 10-13). 논문과 다른 점(문서화): adv std 정규화 켬 · 워밍업 0 · 롤아웃 IS 자르기 없음 · REP·mask 공통.
# 원격 추가 env: ROOT·REPO · MODEL(기본 4B-Instruct-2507) · PTAG(앞부분 데이터 꼬리표, 기본 dapo; 두 번째 모델은 dapo17) · EVTAG(평가 이름 앞붙임, 예 q17_) · N_GPUS(기본 1) · RUN_NAME(큐 잡 이름; push_results 키) · VLLM_UTIL(기본 .4).
# 사용: ARM=ours SEED=3 bash a78_rlrt_train.sh   ·   SMOKE=1 → 2스텝·꼬리표 _smk·평가 없음(20,480 길이 메모리 시험)
set -euo pipefail
ROOT=${ROOT:-/hdd_data/seungpil}; R_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPO=${REPO:-$(cd "$R_DIR/../.." && pwd)}   # scripts/remote 복사본: 경로는 ROOT·REPO 로(기본 = 이 박스)
export ROOT REPO
cd "$REPO"
export PROMPT_VARIANT=plain HF_HOME=$ROOT/scratch/hf_home VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
PY=${PY:-$ROOT/envs/verl09/bin/python}; D=$ROOT/scratch; A=$R_DIR
SD=${SEED:-3}; ST=${STEPS:-60}; RL=${RL_OVERRIDE:-12288}; SMK=$([ "${SMOKE:-}" = 1 ] && echo _smk || true); [ -z "$SMK" ] || ST=2; PTAG=${PTAG:-dapo}; EVTAG=${EVTAG:-}; PFX=$D/data/mc_pfx_commit_${PTAG}_screen_plain_le6000.parquet; NOP=$D/data/mc_nopfx_${PTAG}.parquet
case "${ARM:?}" in
  grpo)   X="KL_COEF=0";                                       TR=$NOP; TG=${PTAG}_np ;;
  grpokl) X="KL_COEF=0.05";                                    TR=$PFX; TG=$PTAG ;;
  pfx)    X="KL_COEF=0";                                       TR=$PFX; TG=$PTAG ;;   # 10-21 대조: 앞부분·끈 없음·선생님 없음(reset 의 짝)
  reset)  X="PFX_DISTILL=0.5 KL_COEF=0";                       TR=$PFX; TG=$PTAG ;;
  ours)   X="PFX_DISTILL=0.5 KL_COEF=0.05";                    TR=$PFX; TG=$PTAG ;;
  rlsd)   X="PFX_FORK=rlsd KL_COEF=0 PFX_TEACHER=policy";      TR=$NOP; TG=${PTAG}_np ;;   # 수정 74: 선생님·학생 = 학습 중 정책(논문 식)
  rlrt)   X="PFX_FORK=rlrt KL_COEF=0 PFX_TEACHER=policy";      TR=$NOP; TG=${PTAG}_np ;;
  sdpo)   X="PFX_FORK=sdpo KL_COEF=0";                         TR=$NOP; TG=${PTAG}_np ;;   # 수정 72 SDPO 식 · 선생님 얼린 ref(RLRT 의 SDPO 실행 teacher_update_rate 0 과 같음)
  ourson) X="PFX_DISTILL=0.5 KL_COEF=0.05 PFX_TEACHER=policy";  TR=$PFX; TG=$PTAG ;;     # 수정 74 절제: 눈 가린 나 = 학습 중 정책(끈은 그대로 얼린 ref)
  *) echo "ARM=grpo|grpokl|pfx|reset|ours|ourson|rlsd|rlrt|sdpo" >&2; exit 2 ;;
esac
[ "$TR" = "$PFX" ] && RL=${RL_PFX:-6144}   # 앞부분 팔: 앞부분(≤6,000) + 이어쓰기 ≤ 12.1k = GRPO 전체 풀이 예산(12,288)과 맞춤(10-13 ours 스모크 update OOM)
[ -s $TR ] || { echo "[a78] 데이터 없음: $TR (dapo_gate 전)" >&2; exit 3; }
TG=${TG}_r${RL}b${TB_OVERRIDE:-32}$SMK; X="$X PFX_GUARD_ABS=0.12"; U="-u PFX_FORK -u PFX_DISTILL -u PFX_DISTILL_SHUF -u PFX_KEEP -u PFX_ADV_CAP -u PFX_ONPOL -u PFX_TEACHER -u KL_COEF -u PFX_GUARD_ABS -u CKPT_DIR"
OUTL=$(env $U $X PFX_TRUNC=mask PFX_BREAK_W=2 PFX_REP=0.25 LR_SCHED=constant LABEL=gold \
  CLIP_HIGH=0.28 MINI_BATCH=16 TRAIN_BATCH=${TB_OVERRIDE:-32} RESP_LEN=$RL MAX_PROMPT=6000 TAG=$TG CKPT_DIR=$(mktemp -d) bash mc/run.sh SPONT_PFX $SD $ST --dry-run | grep -oE "LINEAGE=[^ ]+" | head -1 || true)
L=${OUTL#LINEAGE=}; [ -n "$L" ] || { echo "[a78] 계보 계산 실패" >&2; exit 2; }
EVN=${EVTAG}${ARM}_r${RL}b${TB_OVERRIDE:-32}$([ "$ST" = 60 ] || echo "_st$ST")_s${SD}   # 평가 폴더 이름 줄기(rl_<EVN>_<시험>)
CK=$D/checkpoints/$L; mkdir -p "$CK"
[ -n "$SMK" ] || { mkdir -p $D/remote_meta; printf "LINEAGE=%s\nEVAL_GLOB=rl_%s_*\nMODEL=%s\nN_GPUS=%s\n" "$L" "$EVN" "${MODEL:-default}" "${N_GPUS:-1}" > $D/remote_meta/${RUN_NAME:-$EVN}.env || echo "[a78c] 경고: remote_meta 못 씀" >&2; }   # push_results.sh 가 읽는다
( while sleep 300; do n=$(cat "$CK/latest_checkpointed_iteration.txt" 2>/dev/null) || continue; for c in "$CK"/global_step_*; do [ -d "$c" ] || continue; s=${c##*_}; [ "$s" -lt "$n" ] 2>/dev/null && rm -rf "$c" && echo "[a78c] 옛 체크포인트 삭제 $c (최신 $n)"; done; done ) & CLEAN=$!
trap 'kill $CLEAN 2>/dev/null' EXIT
[ -e $D/checkpoints/$L/TRAIN_DONE ] && grep -q "steps=$ST " $D/checkpoints/$L/TRAIN_DONE || env $U $X VLLM_UTIL=${VLLM_UTIL:-0.4} \
  PFX_TRUNC=mask PFX_BREAK_W=2 PFX_REP=0.25 LR_SCHED=constant REF_OFFLOAD=1 ACTOR_OFFLOAD=1 LABEL=gold PFX_WEIGHT_KEY=sample_weight LR=1e-6 \
  CLIP_HIGH=0.28 MINI_BATCH=16 TRAIN_BATCH=${TB_OVERRIDE:-32} RESP_LEN=$RL MAX_PROMPT=6000 DATA_VAL=$D/data/math_val_plain.parquet VERL_ENV=$ROOT/envs/verl09 \
  EVAL_STEPS=0 SAVE_FREQ=10 TAG=$TG DATA_TRAIN=$TR ${MODEL:+MODEL_PATH=$MODEL} ${N_GPUS:+N_GPUS=$N_GPUS} bash mc/run.sh SPONT_PFX $SD $ST
[ -z "$SMK" ] || { echo "[a78] smoke $ARM 2스텝 끝 — $L"; exit 0; }
M=$($PY -m mc.probe --merge_only $D/checkpoints/$L/global_step_$ST/actor)
for c in $D/checkpoints/$L/global_step_*; do [ "$c" = "$D/checkpoints/$L/global_step_$ST" ] || rm -rf "$c"; done   # 10-13: 중간 체크포인트(47 GB씩) 삭제 — 마지막 스텝만 남김(연장용)
bash $A/a79_rlrt_eval.sh $EVN=$M   # 10-13: 100스텝 연장은 다른 평가 이름
echo "[a78c] $ARM s$SD → $L done"
