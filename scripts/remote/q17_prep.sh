#!/bin/bash
# R2 두 번째 모델(Qwen3-1.7B) 앞부분 준비 — 추론만, GPU 1장. 4B 앞부분(= 모델이 첫 \boxed 까지 스스로 푼 앞부분)은 모델마다 다시 만들어야 한다.
#   파이프라인 = a77_dapo.sh(원본 a77_dapo.sh 의 복사본): pool(base k4 롤아웃) → prep(1–3/4 맞은 문제 고름 · 박스 점수 · 앞부분) → screen(k8 거르기) → gate(무게·관문·nopfx).
#   usage: q17_prep.sh group G   묶음 h(0..11) 중 h % PREP_GROUPS == G 인 것을 순서대로(기본 4그룹 = GPU 4장 병렬; 끝난 단계는 건너뜀 = 이어 하기)
#          q17_prep.sh gate      12묶음 screen 이 다 있으면 gate (group 의 마지막 끝낸 잡이 자동으로 부른다)
#          q17_prep.sh all       한 GPU 에서 처음부터 끝까지
#   env: MODEL(기본 $ROOT/scratch/models/Qwen3-1.7B) · TG(기본 dapo17 → mc_pfx_commit_dapo17_screen_plain_le6000.parquet) · FORCE=1(관문 미달이어도 탐색용으로 씀) · RUN_NAME
set -euo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/hf_env.sh" 2>/dev/null || { ROOT=${ROOT:-/hdd_data/seungpil}; R_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); }
D=$ROOT/scratch; export M2=${MODEL:-$D/models/Qwen3-1.7B} TG=${TG:-dapo17} ROOT; G=${PREP_GROUPS:-4}; A=$R_DIR
OUT=$D/data/mc_pfx_commit_${TG}_screen_plain_le6000.parquet; mkdir -p $D/remote_meta
RUN=${RUN_NAME:-q17_prep_${2:-all}}
stage() {  # stage h : 끝난 산출물이 있으면 건너뛴다
  local h=$1
  [ -s $D/data/${TG}_pool_$h.summary.json ] || bash $A/a77_dapo.sh pool $h
  [ -s $D/eval/${TG}_sel_$h.json ] || bash $A/a77_dapo.sh prep $h
  [ -s $D/eval/${TG}_screen_$h/summary.json ] || bash $A/a77_dapo.sh screen $h
}
gate() { bash $A/a77_dapo.sh gate; [ -s $OUT ] || { echo "[q17] 관문 미달 — 앞부분 데이터 없음 (FORCE=1 로 탐색용 사용 가능)" >&2; exit 3; }
         printf "EXTRA_FILES=data/mc_pfx_commit_%s_screen_plain_le6000.parquet data/mc_pfx_commit_%s_screen_plain_le6000.summary.json data/mc_nopfx_%s.parquet\n" $TG $TG $TG >> $D/remote_meta/$RUN.env; }
all_screens() { for h in $(seq 0 11); do [ -s $D/eval/${TG}_screen_$h/summary.json ] || return 1; done; }
printf "MODEL=%s\nTG=%s\n" "$M2" "$TG" > $D/remote_meta/$RUN.env
case "${1:-}" in
  group) g=${2:?group G}; for h in $(seq 0 11); do if [ $((h % G)) = $g ]; then stage $h; fi; done
         touch $D/remote_meta/${TG}_group_$g.done
         if all_screens && mkdir $D/remote_meta/${TG}_gate.lock 2>/dev/null; then gate; fi ;;   # 마지막으로 끝낸 그룹이 gate 를 맡는다
  gate)  gate ;;
  all)   for h in $(seq 0 11); do stage $h; done; gate ;;
  *) echo "usage: $0 group G | gate | all" >&2; exit 2 ;;
esac
echo "[q17] ${1} ${2:-} done"
