#!/usr/bin/env bash
# scripts/local/run_arm.sh ARM SEED [STEPS=100] [VARIANT] [--dry-run]
#
# Run ONE Countdown RL arm end-to-end on ONE GPU: train -> merge judgment steps
# -> held-out eval -> HF upload. Designed to be the "cmd" of one gpu_queue.py job
# (the queue sets CUDA_VISIBLE_DEVICES; this script does not touch it, so it runs
# on whatever GPU it is handed).
#
# ARM      one of src/training/countdown_rewards.ARM_SPECS keys, e.g. N0 A PL B C D E F
# SEED     int, used for data.seed (below is why NOT actor_rollout_ref.rollout.seed:
#          verl 0.7.1's rollout.yaml — checked under
#          /hdd_data/seungpil/envs/simplerl/lib/python3.10/site-packages/verl/trainer/config/rollout/rollout.yaml —
#          has NO `seed` key at all. `data.seed` DOES exist
#          (.../trainer/config/data/legacy_data.yaml:65, default null), so that is
#          the only seed override this script can verify. Rollout sampling
#          randomness therefore is NOT separately seeded per arm/run here —
#          UNVERIFIED whether that matters at this budget; noted, not fixed.)
# STEPS    trainer.total_training_steps (default 100)
# VARIANT  prompt/data variant for non-N0/OPT/OPT_M arms (default p3); ignored for
#          N0, which always uses the `plain` variant (countdown_rewards.ARM_SPECS["N0"]
#          note: "메타 지시문 없음... variant plain, DATA_SUFFIX=_4num_plain"), and
#          ignored for OPT/OPT_M/OPT_T/OPT_MT/OPT_MT2/OPT_OPD/OPT_OPDC/OPT_OPDG/OPT_MTC/OPT_CF/
#          OPT_CFG/OPT_VTR/OPT_VTRW, which always use the `opt` variant (permission, not
#          mandate — countdown_task.PROMPT_VARIANTS["opt"]). OPT_CF's twin rows carry the
#          `plain` system message inside the parquet itself (build_cf_twins.py) —
#          VARIANT/DATA_VARIANT here only pick the val file for OPT_CF. OPT_VTR/OPT_VTRW's
#          hint-twin rows keep the `opt` system message and instead splice a hint line into
#          the user message (build_cf_twins.py --mode hint) — same reasoning, VARIANT/
#          DATA_VARIANT here only pick the val file.
#
# LINEAGE = cd7_<ARM>_<VARIANT>_s<SEED>   (VARIANT here is the EFFECTIVE data variant,
#           i.e. "plain" for N0, so lineages stay unambiguous.)
#
# Isolation: this job gets its own RAY_TMPDIR (so its Ray head process's sockets
# don't collide with sibling jobs on other GPUs of this box) and its own Ray
# instance (no RAY_ADDRESS is set — verl_sdc.py's main() only takes the
# "attach to existing cluster" branch when RAY_ADDRESS is present; see
# src/training/verl_sdc.py:4918-4934 vs 4942-4964). `ray stop --force` is NEVER
# called from this script because other jobs' Ray instances share the box.
#
# UNVERIFIED / risk noted, not solved here: verl_sdc.py's own `ray.init()` call
# (no RAY_ADDRESS branch) does not pin a GCS port, so two concurrent invocations
# of this script MAY race for Ray's default port unless Ray's own auto-port-pick
# on conflict is reliable at this Ray version. Not fixed because the task's env
# var/isolation list did not ask for a port scheme and verl_sdc.py is out of
# scope (another agent owns it concurrently).
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

usage() {
  echo "Usage: $0 ARM SEED [STEPS=100] [VARIANT] [--dry-run]" >&2
  exit 1
}

DRY_RUN=0
POSITIONAL=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    *) POSITIONAL+=("$a") ;;
  esac
done
set -- "${POSITIONAL[@]:-}"
[ -z "${1:-}" ] && usage

ARM="${1:?ARM required}"
SEED="${2:?SEED required}"
STEPS="${3:-100}"
VARIANT_ARG="${4:-p3}"

# ── RESP_LEN (0906, OPT_MT-L): 응답 길이 예산 knob. 기본 2048 = 지금까지와 바이트
#    동일(고정 자리 mixed 분기의 기존 하드코딩 값). OPT_MT 계열은 메모 뒤 재시도가
#    길어져 응답의 31~38% 가 잘린다(docs/RESULTS_cd7.md OPT_MT-L 절) — 잘림이
#    held-out 손실의 원인인지 직접 검사하려면 예산만 키운 짝 실험이 필요하다.
#    명시적으로 설정됐는지(값이 아니라 존재 여부)를 RESP_LEN_SET 로 따로 기억해
#    둔다 — non-mixed 분기는 "건드리지 않음"이 기본이고, 사용자가 RESP_LEN 을
#    직접 준 경우에만 그쪽에도 유사한 override 를 적용한다.
if [ -n "${RESP_LEN+x}" ]; then RESP_LEN_SET=1; else RESP_LEN_SET=0; fi
RESP_LEN="${RESP_LEN:-2048}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

# ── ARM validity (fail closed like the cluster launcher does — see
#    countdown_rl_6arm.yaml's `case "$ARM" in [ABCEFG]) ;; *) FATAL` gate; the
#    valid set here is the current ARM_SPECS keys, which is a superset of the
#    6-arm cluster round). ─────────────────────────────────────────────────
if ! ARM_OK=$(python -c "
from src.training.countdown_rewards import ARM_SPECS
import sys
sys.exit(0 if '${ARM}' in ARM_SPECS else 1)
" 2>&1); then
  echo "[run_arm] FATAL: ARM='${ARM}' not in src.training.countdown_rewards.ARM_SPECS" >&2
  exit 1
fi

IS_OPT_PROMPT=0
if [ "${ARM}" = "N0" ]; then
  DATA_VARIANT="plain"
elif [ "${ARM}" = "OPT" ] || [ "${ARM}" = "OPT_M" ] || [ "${ARM}" = "OPT_T" ] || [ "${ARM}" = "OPT_MT" ] || [ "${ARM}" = "OPT_MT2" ] || [ "${ARM}" = "OPT_OPD" ] || [ "${ARM}" = "OPT_OPDC" ] || [ "${ARM}" = "OPT_MTC" ] || [ "${ARM}" = "OPT_CF" ] || [ "${ARM}" = "OPT_CFG" ] || [ "${ARM}" = "OPT_OPDG" ] || [ "${ARM}" = "OPT_VTR" ] || [ "${ARM}" = "OPT_VTRW" ] || [ "${ARM}" = "OPT_OPDGW" ]; then
  # ★E-135(2026-09-07): 「opt 프롬프트 팔」 판정을 여기 한 곳에서만 한다. 예전엔 아래 mixed 분기에
  #   팔 이름 목록이 따로 있어, 새 팔을 한쪽에만 넣으면 학습은 «강제(new)» 데이터를 읽는 사고가 났다.
  IS_OPT_PROMPT=1
  # ★OPT/OPT_M (2026-09-05): 메타 허용·비요구 팔은 항상 `opt` 프롬프트(허가 문장)로
  #   발사한다 — N0 가 항상 `plain` 인 것과 같은 이유다. VARIANT_ARG 를 그대로 두면
  #   호출자가 실수로 p3/new 데이터를 붙여 강제 프롬프트로 발사할 수 있다.
  #   ★OPT_OPD(0906, docs/DESIGN_opd_hint_teacher.md §4): 힌트 교사 항도 OPT 계열과
  #   같은 «허용·비요구» 프롬프트 위에서 돈다 — data_hint 는 (기본값) "normal" 이라
  #   아래 mixed 분기는 안 탄다(고정 자리 배치 없음, §6 스모크는 일반 롤아웃 전용).
  #   ★OPT_OPDC(0906, §10): 그룹 중심화 판. OPT_OPD 와 항 하나만 다르고(opd_meta →
  #   opd_meta_c) 프롬프트·data_hint 는 완전히 같다.
  #   ★OPT_CF(2026-09-07, §8): main 행은 opt 프롬프트(메타 허가) 그대로다 — twin
  #   행의 plain 프롬프트는 `scripts/local/build_cf_twins.py` 가 parquet 안에서
  #   이미 만들어 뒀다(행마다 다른 시스템 메시지). 여기서 VARIANT_ARG 를 "opt" 로
  #   고정하는 것은 val 파일 선택(`countdown_val_4num_opt.parquet`)에만 쓰인다.
  #   ★OPT_VTR/OPT_VTRW(2026-09-08, §12): main 행은 OPT_CF 와 같은 opt 프롬프트.
  #   힌트 twin 행은 시스템 메시지를 그대로 두고 user 메시지에 힌트를 끼운다
  #   (`build_cf_twins.py --mode hint`) — VARIANT/DATA_VARIANT 는 여기서도 val
  #   파일 선택에만 쓰인다.
  DATA_VARIANT="opt"
elif [ "${ARM}" = "TAG0" ] || [ "${ARM}" = "TAX0" ] || [ "${ARM}" = "FIXED_CHK" ] || [ "${ARM}" = "EVC_CHK" ] || [ "${ARM}" = "EVCM_CHK" ] || [ "${ARM}" = "EVCA_CHK" ] || [ "${ARM}" = "EVCAS_CHK" ] || [ "${ARM}" = "PERSIST_CHK" ]; then
  # ★0909 §13 검산 사다리: 항상 chk 프롬프트(plain + check 한 줄 허가).
  DATA_VARIANT="chk"
else
  DATA_VARIANT="${VARIANT_ARG}"
fi

# ── data_hint (고정 자리, 0904). ARM_SPECS[ARM] 이 "mixed" 를 선언하면(현재 M0/MT)
#    site(3000)+normal(3000) 을 섞은 mixed_train.parquet 로 발사해야 한다 — 항 이름이
#    같아도(M0 는 A 와 동일) 데이터가 다르면 다른 실험이다(countdown_rewards.py
#    ARM_SPECS 의 M0 note 참조). 다른 팔은 이 키가 없으므로 기본값 "normal" —
#    "지금까지"(일반 롤아웃 전용 parquet)와 바이트 동일하게 행동한다.
DATA_HINT=$(python -c "
from src.training.countdown_rewards import ARM_SPECS
print(ARM_SPECS['${ARM}'].get('data_hint', 'normal'))
")

# ★OPT_CF(§8): data_hint="mixed_cf" 는 고정 자리 + 반사실 쌍둥이 판이다. 예산·
#   continue_final_message 오버라이드는 "mixed"(FT/M0/MT/OPT_M/OPT_MT...) 와
#   완전히 같아야 한다 — 데이터가 다를 뿐 배치 기하(site 프리픽스가 프롬프트에
#   접합됨)는 동일하다. `IS_MIXED_LIKE` 로 그 두 값을 한 곳에서 함께 다룬다.
# ★OPT_VTR/OPT_VTRW(§12): data_hint="mixed_vtr" 는 고정 자리 + 힌트 twin(K=4) 판이다.
#   같은 이유로 "mixed"/"mixed_cf" 와 함께 `IS_MIXED_LIKE` 로 다룬다.
IS_MIXED_LIKE=0
if [ "${DATA_HINT}" = "mixed" ] || [ "${DATA_HINT}" = "mixed_cf" ] || [ "${DATA_HINT}" = "mixed_vtr" ]; then
  IS_MIXED_LIKE=1
fi

# SITES_DIR(0907, E-133 수리): mixed 계열이 읽는 site parquet 디렉터리 이름
# ($WORK/data/<SITES_DIR>/ 아래). 기본 sites_v1 은 "지금까지"와 바이트 동일하다.
# sites_v1(및 v2/v3/v3c) 은 held-out val 에서 채굴돼 오염됐다(E-133) —
# 새 라운드는 SITES_DIR=sites_v4 로 재개해야 한다(scripts/local/build_sites_v4.sh).
SITES_DIR="${SITES_DIR:-sites_v1}"

# INIT_TAG(선택, 대문자/숫자, 예 SFT1): 초기 모델을 바꾼 팔의 계보를 구분한다 — 계보명 arm 자리에 붙어
#   `cd7_OPT_CF_SFT1_opt_s1_mixed` 꼴이 된다(keeper 의 _LIN 정규식 arm 그룹 [A-Z0-9_]+ 가 그대로 받는다).
# MODEL_PATH(선택): 초기 모델 경로(기본 Qwen3-4B). INIT_TAG 없이 MODEL_PATH 만 바꾸는 것은 금지(계보 충돌).
MODEL_PATH="${MODEL_PATH:-/hdd_data/seungpil/scratch/models/Qwen3-4B}"
INIT_TAG="${INIT_TAG:-}"
if [ -n "${INIT_TAG}" ] && ! [[ "${INIT_TAG}" =~ ^[A-Z0-9]+$ ]]; then echo "[run_arm] FATAL: INIT_TAG 는 대문자/숫자만 (${INIT_TAG})"; exit 2; fi
if [ -z "${INIT_TAG}" ] && [ "${MODEL_PATH}" != "/hdd_data/seungpil/scratch/models/Qwen3-4B" ]; then echo "[run_arm] FATAL: MODEL_PATH 를 바꾸면 INIT_TAG 도 줘야 한다(계보 충돌 방지)"; exit 2; fi
LINEAGE="cd7_${ARM}${INIT_TAG:+_${INIT_TAG}}_${DATA_VARIANT}_s${SEED}"
if [ "${IS_MIXED_LIKE}" = "1" ]; then
  LINEAGE="${LINEAGE}_mixed"
fi
# ★RESP_LEN suffix는 _mixed 뒤에 붙인다 — 값이 기본(2048)과 다를 때만, 체크포인트/
#   merged/eval/logs 가 2048 계보와 절대 충돌하지 않게.
if [ "${RESP_LEN}" != "2048" ]; then
  LINEAGE="${LINEAGE}_r${RESP_LEN}"
fi
# ★OPD_FULL_SPAN=1 (0908, §12-b): 게이트 자리 전 구간 증류. 계보 접미사 _fs 로 분리.
if [ "${OPD_FULL_SPAN:-0}" = "1" ]; then
  LINEAGE="${LINEAGE}_fs"
fi
# ★OPD_DENSE=1 (0908, §12-c): 토큰별 밀집 OPD 어드밴티지. 계보 접미사 _dn.
if [ "${OPD_DENSE:-0}" = "1" ]; then
  LINEAGE="${LINEAGE}_dn"
fi
# ★CHK_REGION=1 (0909, §13-b): 메모 칸 항을 <check> 구간 토큰에만 어드밴티지로. 계보 접미사 _rg.
if [ "${CHK_REGION:-0}" = "1" ]; then
  LINEAGE="${LINEAGE}_rg"
fi
CONFIG_NAME="${CONFIG_NAME:-countdown_6arm}"
if [ "${DATA_HINT}" = "mixed_cf" ]; then
  # ★OPT_CF(§8): 반사실 쌍둥이 판 — main(site, opt 프롬프트) + twin(같은 자리,
  #   plain 프롬프트, `extra_info.cf_role=twin`) + normal. `scripts/local/
  #   build_cf_twins.py` 가 `mixed_train_v3c_opt.parquet` 에서 만든다.
  DATA_TRAIN="${WORK}/data/${SITES_DIR}/${MIXED_DATA:-mixed_train_v3c}_cf_opt.parquet"
elif [ "${DATA_HINT}" = "mixed_vtr" ]; then
  # ★OPT_VTR/OPT_VTRW(2026-09-08, §12): 온라인 검증 게이트 판 — main(site, opt
  #   프롬프트, `extra_info.vtr_role=main`) + 힌트 twin(같은 자리, 같은 opt 프롬프트에
  #   힌트 삽입, K=4, `extra_info.vtr_role=twin`) + normal. `scripts/local/
  #   build_cf_twins.py --mode hint` 가 `mixed_train_v4_gate_opt.parquet`(오프라인
  #   opd_gate 폴백 포함판)에서 만든다.
  DATA_TRAIN="${WORK}/data/${SITES_DIR}/${MIXED_DATA:-mixed_train_v4}_vtr_opt.parquet"
elif [ "${DATA_HINT}" = "mixed" ]; then
  # site(3000, 프리픽스가 이미 프롬프트에 접합) + normal(3000, 빈 assistant 메시지
  # 부착) 를 섞은 고정 자리 학습 parquet. countdown_sites.py 헤더 참조.
  # OPT_M 은 opt 프롬프트(메모 허용만)로 시스템 메시지를 바꾼 같은 자리 데이터를 쓴다.
  # ★E-134(2026-09-07): 예전엔 `_opt` 가 기본값 문자열 안에만 있어 MIXED_DATA 를 넘기면 opt 팔이
  #   메타 «강제」(new) 데이터를 읽었다(대조군 프롬프트 불일치). 이제 MIXED_DATA 는 항상 «기본 이름」
  #   (예: mixed_train_v3c) 이고, opt 팔이면 런처가 `_opt` 를 붙인다.
  MIXED_BASE="${MIXED_DATA:-mixed_train_v2}"
  case "${MIXED_BASE}" in *_opt) echo "[run_arm] FATAL: MIXED_DATA 에 _opt 를 붙이지 말 것(런처가 붙인다): ${MIXED_BASE}"; exit 2;; esac
  if [ "${IS_OPT_PROMPT}" = "1" ]; then
    DATA_TRAIN="${WORK}/data/${SITES_DIR}/${MIXED_BASE}_opt.parquet"
  else
    DATA_TRAIN="${WORK}/data/${SITES_DIR}/${MIXED_BASE}.parquet"
  fi
else
  DATA_TRAIN="${WORK}/data/countdown_train_4num_${DATA_VARIANT}.parquet"
fi

# ★E-133 가드: sites_v1 계열(자리·SFT 가 held-out val 문제에서 채굴됨)은 평가가 오염된다.
#   의도적으로 재현할 때만 ALLOW_CONTAMINATED=1 로 연다.
case "${DATA_TRAIN}" in
  */sites_v1/*)
    if [ "${ALLOW_CONTAMINATED:-0}" != "1" ]; then
      echo "[run_arm] FATAL: ${DATA_TRAIN} 는 오염된 sites_v1 계열이다(E-133). SITES_DIR=sites_v4 MIXED_DATA=mixed_train_v4 로 발사하거나 ALLOW_CONTAMINATED=1 을 명시하라."
      exit 2
    fi
    echo "[run_arm] WARNING: 오염된 sites_v1 데이터를 명시적으로 사용한다(ALLOW_CONTAMINATED=1)."
    ;;
esac
DATA_VAL="${WORK}/data/countdown_val_4num_${DATA_VARIANT}.parquet"
CKPT_DIR="${WORK}/checkpoints/${LINEAGE}"
LOG_FILE="${WORK}/logs/${LINEAGE}.log"
# ── 예산 (고정 자리 처치, 0904 사용자 지시값). site 프롬프트는 프리픽스가 이미
#    접합돼 있어 일반 프롬프트보다 길다 — prompt/response/model_len 전부 키운다.
#    "${VAR:-default}" 라 호출자가 이미 env 로 값을 줬으면 그쪽을 존중한다
#    (그래서 data_hint 분기를 그 **기본값**에만 건다 — "지금까지"와 바이트 동일한
#    경로는 이 분기가 없어도 원래 기본값 그대로다).
if [ "${IS_MIXED_LIKE}" = "1" ]; then
  MAX_PROMPT="${MAX_PROMPT:-2048}"
  # ★RESP_LEN(0906): 기본값 2048 이면 이 세 줄은 예전 하드코딩(2048/4352/4352)과
  #   바이트 동일하다. RESP_LEN 을 올리면(예: 3072) model_len/batched_tokens 도
  #   같이 늘려 예산 전체가 일관되게 커지게 한다(2048 프롬프트 + RESP_LEN + 256 여유).
  MAX_RESP="${MAX_RESP:-${RESP_LEN}}"
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT + RESP_LEN + 256))}"
  MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((MAX_PROMPT + RESP_LEN + 256))}"
else
  MAX_PROMPT="${MAX_PROMPT:-1024}"
  if [ "${RESP_LEN_SET}" = "1" ]; then
    # RESP_LEN 을 명시적으로 준 경우에만 non-mixed 분기도 건드린다 — "지금까지"
    # 경로(RESP_LEN 미지정)는 원래 기본값 그대로 유지한다.
    MAX_RESP="${MAX_RESP:-${RESP_LEN}}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT + RESP_LEN + 256))}"
    MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$((MAX_PROMPT + RESP_LEN + 256))}"
  else
    MAX_RESP="${MAX_RESP:-2560}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
  fi
fi

# ★0904: AF_UNIX 소켓 경로 107바이트 제한 — /hdd_data/…/<긴 계보명>/session_…/sockets/plasma_store 가 넘쳤다.
#   Ray 임시 디렉터리는 소켓·로그뿐이라(객체 저장소는 /dev/shm) 루트 디스크의 짧은 경로로 둔다.
export RAY_TMPDIR="/hdd_data/seungpil/r/$(printf '%s' "${LINEAGE}" | md5sum | cut -c1-8)"   # 0905: 루트 디스크(200GB)가 찼다 — 짧은 경로(소켓 107B 제한)로 /hdd_data 에 둔다
mkdir -p "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}" "${CKPT_DIR}" "${WORK}/logs"
unset RAY_ADDRESS || true

export WANDB_PROJECT="metacot-countdown-local"
export WANDB_NAME="${LINEAGE}"
export WANDB_RUN_ID="${LINEAGE}-1"
export WANDB_RESUME=allow

TRAIN_CMD=(python -u -m src.training.verl_sdc
  "--config-name=${CONFIG_NAME}"
  "++mode=COUNTDOWN_6ARM"
  "++algorithm.countdown_arm=${ARM}"
  "trainer.experiment_name=${LINEAGE}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.nnodes=1"
  "trainer.n_gpus_per_node=1"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.n=8"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "data.train_files=${DATA_TRAIN}"
  "data.val_files=${DATA_VAL}"
  "++data.seed=${SEED}"
  # ★0904: Qwen3 하이브리드는 기본이 thinking ON 이라 <meta> 발화가 0 이 된다(cd6 prereg A.2 와 동일 결정).
  #   학습 전 평가(countdown_gs0_eval.py)도 enable_thinking=False 라 이 옵션이 있어야 조건이 같다.
  "+data.apply_chat_template_kwargs.enable_thinking=false"
  "data.train_batch_size=64"
  "data.max_prompt_length=${MAX_PROMPT}"
  "data.max_response_length=${MAX_RESP}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_UTIL:-0.35}"
  # SLIM=1(0907): 8-bit AdamW(bitsandbytes) — 갱신 피크 73GB→~50GB, 공유 카드(이웃 5~14GB) 옆에서 학습 가능.
  #   최적화 수치가 바뀌므로 같은 설정끼리만 비교한다(계보 태그 INIT_TAG 로 구분).
  "actor_rollout_ref.actor.optim.optimizer_impl=$([ "${SLIM:-0}" = "1" ] && echo bitsandbytes.optim || echo torch.optim)"
  # ★0909 PAGED=1: bitsandbytes PagedAdamW8bit — 옵티마이저 상태를 압력 시 CPU 로 페이징한다.
  #   수식은 AdamW8bit 과 같고 «어디에 두느냐»만 다르다(방법 변경 아님). 외부 잡이 카드를 27~62GB
  #   오가며 액터 업데이트 순간 OOM 을 내는 상황(0909 새벽 9회)에서 피크 ~8GB 를 줄인다.
  "actor_rollout_ref.actor.optim.optimizer=$([ "${SLIM:-0}" = "1" ] && { [ "${PAGED:-0}" = "1" ] && echo PagedAdamW8bit || echo AdamW8bit; } || echo AdamW)"
  # REF_OFFLOAD=1: ref(KL) 워커 파라미터를 CPU 로 내려 GPU 피크 ~8GB 절감(공유 카드용). 기본 false(기존 팔과 동일).
  "actor_rollout_ref.ref.fsdp_config.param_offload=$([ "${REF_OFFLOAD:-0}" = "1" ] && echo true || echo false)"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true"
  # ★0909 ACTOR_OFFLOAD=1: 외부 잡이 카드마다 32GB 를 상주시킬 때(0909 trm pretrain ×4) 45GB 안에서 돌기 위한
  #   파라미터 CPU 오프로드. 방법이 아니라 메모리 배치라 계보 접미사 없음. 속도 저하 감수.
  "actor_rollout_ref.actor.fsdp_config.param_offload=$([ "${ACTOR_OFFLOAD:-0}" = "1" ] && echo true || echo false)"
  "++actor_rollout_ref.model.enable_activation_offload=true"
  "actor_rollout_ref.rollout.enforce_eager=true"
  "++trainer.total_training_steps=${STEPS}"
  "trainer.resume_mode=auto"
  "trainer.save_freq=5"
  "trainer.test_freq=0"
  "++trainer.val_before_train=False"
  "++hydra.searchpath=[pkg://verl/trainer/config]"
)

if [ "${IS_MIXED_LIKE}" = "1" ]; then
  # ★고정 자리 재개(site 행) — 마지막 메시지가 이미 assistant(프리픽스)다. 이 두
  #   키가 verl 0.7.1 agent-loop 로 실제 도달하려면 sitecustomize.py 의
  #   `_patch_verl_agent_loop_chat_template` 이 걸려 있어야 한다 — 패치 없이 이
  #   override 만 주면 `AgentLoopBase.apply_chat_template` 이
  #   `add_generation_prompt` 중복 키워드로 즉사한다(그 함수 조사 기록 참조).
  TRAIN_CMD+=(
    "+data.apply_chat_template_kwargs.continue_final_message=true"
    "+data.apply_chat_template_kwargs.add_generation_prompt=false"
    # ★verl 0.7.1 rl_dataset.py:maybe_filter_out_long_prompts 의 `doc2len` 도
    #   같은 하드코딩(`add_generation_prompt=True` + `**apply_chat_template_kwargs`)
    #   버그를 갖고 있는데, 그쪽은 예외를 **삼키고**(broad except) 그 행을
    #   "너무 길다"로 오분류해 조용히 필터링한다 — 이 버그는 크래시가 아니라
    #   site 행 3000개가 전부 소리 없이 사라지는 데이터 유실이라 sitecustomize
    #   패치로도 못 막는다(그 메서드는 agent-loop 가 아니다). 필터를 꺼서 그
    #   경로 자체를 피한다 — max_prompt_length 초과를 걸러야 할 이유가 애초에
    #   없다(이미 2048 로 넉넉히 키웠다).
    "data.filter_overlong_prompts=false"
  )
fi

# ★OPT_CF(§8) 전용 셔플 오버라이드. `configs/countdown_6arm.yaml` 의 `data.shuffle`
#   기본값은 True — verl 은 매 에폭 **데이터셋 자체**를 섞은 뒤 `train_batch_size`
#   (64) 로 순서대로 자른다. `build_cf_twins.py` 는 main/twin 쌍을 (main,twin,
#   main,twin,...) 로 인접 배치해 뒀는데, 셔플이 켜져 있으면 그 인접성이 에폭마다
#   깨져 `cf_center_rows`(배치 전체에서 cf_key 로 짝짓는다)가 짝을 못 찾는 행이
#   늘어난다 — 처치가 조용히 무효 레버가 된다. 그래서 OPT_CF 만 셔플을 끈다: 쌍은
#   항상 짝수 인덱스(0,2,4,...)에서 시작하고 배치 크기(64)도 짝수라, 셔플이 꺼져
#   있으면 어떤 쌍도 배치 경계에 걸리지 않는다(파일 순서가 고정되므로). 다른 팔은
#   이 분기를 안 타 "지금까지"(shuffle=True)와 바이트 동일하다.
#   ★OPT_VTR/OPT_VTRW(§12): 같은 이유로 같은 오버라이드가 필요하다 — 블록이
#   (main, twin*4) 로 인접 배치돼 있고(`build_cf_twins.py --mode hint`), 셔플이
#   켜지면 온라인 게이트(`vtr_batch_gate`)가 같은 배치에서 main/twin 을 못 찾는
#   행이 늘어 폴백(오프라인)으로만 돌게 된다.
if [ "${ARM}" = "OPT_CF" ] || [ "${ARM}" = "OPT_VTR" ] || [ "${ARM}" = "OPT_VTRW" ]; then
  TRAIN_CMD+=("data.shuffle=false")
fi

echo "[run_arm] LINEAGE=${LINEAGE} ARM=${ARM} SEED=${SEED} STEPS=${STEPS} DATA_VARIANT=${DATA_VARIANT} DATA_HINT=${DATA_HINT} RESP_LEN=${RESP_LEN} INIT_TAG=${INIT_TAG:-none} MODEL_PATH=${MODEL_PATH} SLIM=${SLIM:-0}"
echo "[run_arm] data.train_files=${DATA_TRAIN}"
echo "[run_arm] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset! queue should have set this>}"
echo "[run_arm] exact train command:"
printf '  %q' "${TRAIN_CMD[@]}"; echo
echo "[run_arm] log file: ${LOG_FILE}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[run_arm] --dry-run: not executing."
  exit 0
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[run_arm] WARNING: CUDA_VISIBLE_DEVICES is unset. This script expects the queue" >&2
  echo "[run_arm]          worker to have set it to a single GPU index (0-3)." >&2
fi

for f in "${DATA_TRAIN}" "${DATA_VAL}"; do
  if [ ! -s "${f}" ]; then
    echo "[run_arm] FATAL: missing/empty data file ${f} (run scripts/local/make_data.sh first)" >&2
    exit 1
  fi
done

echo "[run_arm] starting training, appending to ${LOG_FILE}" | tee -a "${LOG_FILE}"
# ── Ray 기동 직렬화. 0904 실측: 4 잡이 동시에 ray.init 하면 «node timed out during startup»
#    (raylet/agent 등록 경합)로 죽는다. 기동 창(기본 300초) 동안만 전역 flock 을 잡고,
#    그 뒤에는 잠금을 풀어 다른 잡이 기동하게 한다. 학습 자체는 병렬이다.
STARTUP_LOCK="${QUEUE_ROOT:-/hdd_data/seungpil/queue}/.startup.lock"
STARTUP_HOLD_SEC="${STARTUP_HOLD_SEC:-300}"
exec 9>"${STARTUP_LOCK}"
echo "[run_arm] waiting for startup lock ${STARTUP_LOCK}" | tee -a "${LOG_FILE}"
flock 9
echo "[run_arm] startup lock acquired $(date -Is); holding ${STARTUP_HOLD_SEC}s" | tee -a "${LOG_FILE}"
set +e
"${TRAIN_CMD[@]}" >> "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!
( sleep "${STARTUP_HOLD_SEC}"; flock -u 9; echo "[run_arm] startup lock released $(date -Is)" >> "${LOG_FILE}" ) &
wait "${TRAIN_PID}"
TRAIN_RC=$?
flock -u 9 2>/dev/null || true
set -e

if [ "${TRAIN_RC}" != "0" ]; then
  echo "[run_arm] FATAL: training exited ${TRAIN_RC} (see ${LOG_FILE})" >&2
  exit "${TRAIN_RC}"
fi
echo "[run_arm] training exited 0"

# ── merge + prune + eval + upload, per judgment step present on disk. ───────
JUDGMENT_STEPS=(30 50 100)
for STEP in "${JUDGMENT_STEPS[@]}"; do
  GS_DIR="${CKPT_DIR}/global_step_${STEP}"
  ACTOR_DIR="${GS_DIR}/actor"
  if [ ! -d "${ACTOR_DIR}" ]; then
    echo "[run_arm] step ${STEP}: no ${ACTOR_DIR}, skipping (arm may not have trained this far)"
    continue
  fi

  MERGED_DIR="${WORK}/merged/${LINEAGE}/step_${STEP}"
  mkdir -p "${MERGED_DIR}"
  echo "[run_arm] step ${STEP}: merging ${ACTOR_DIR} -> ${MERGED_DIR}"
  # verl 0.7.1 CLI confirmed at
  # /hdd_data/seungpil/envs/simplerl/lib/python3.10/site-packages/verl/model_merger/__main__.py
  if python -m verl.model_merger merge --backend fsdp \
      --local_dir "${ACTOR_DIR}" --target_dir "${MERGED_DIR}" \
      >> "${LOG_FILE}" 2>&1; then
    if [ -f "${MERGED_DIR}/config.json" ] && ls "${MERGED_DIR}"/*safetensors* >/dev/null 2>&1; then
      echo "[run_arm] step ${STEP}: merge verified (config.json + safetensors present) — pruning shards"
      rm -f "${ACTOR_DIR}"/optim* "${ACTOR_DIR}"/model*
    else
      echo "[run_arm] step ${STEP}: merge command exited 0 but config.json/safetensors missing in ${MERGED_DIR} — NOT pruning ${ACTOR_DIR}" >&2
      continue
    fi
  else
    echo "[run_arm] step ${STEP}: merge FAILED (see ${LOG_FILE}) — NOT pruning ${ACTOR_DIR}, skipping eval/upload for this step" >&2
    continue
  fi

  EVAL_OUT="${WORK}/eval/${LINEAGE}/step_${STEP}"
  mkdir -p "${EVAL_OUT}"
  echo "[run_arm] step ${STEP}: eval -> ${EVAL_OUT}"
  # countdown_gs0_eval.py CLI confirmed at scripts/countdown_gs0_eval.py:108-123.
  # 500 problems x 8 rollouts per task spec: val parquet already has 500 rows
  # (scripts/local/make_data.sh), so --limit 0 (= all) x --num_samples 8.
  # ★RESP_LEN(0906): held-out eval 예산도 학습 예산에 맞춰 늘린다 — 512 여유는
  #   기존 하드코딩(2048 학습 → 2560 eval)과 같은 비율. RESP_LEN 기본값(2048)에서는
  #   2560 으로 예전과 바이트 동일.
  python scripts/countdown_gs0_eval.py \
    --model_path "${MERGED_DIR}" \
    --data "${DATA_VAL}" \
    --meta_format "${EVAL_META_FORMAT:-${DATA_VARIANT}}" \
    --num_samples 8 \
    --seed 11 \
    --limit 0 \
    --max_tokens "$((RESP_LEN + 512))" \
    --out_dir "${EVAL_OUT}" \
    >> "${LOG_FILE}" 2>&1 || echo "[run_arm] step ${STEP}: eval FAILED (see ${LOG_FILE}), continuing to upload anyway" >&2

  echo "[run_arm] step ${STEP}: uploading ${MERGED_DIR} to HF"
  python "${_SCRIPT_DIR}/hf_upload.py" \
    --lineage "${LINEAGE}" --step "${STEP}" --local-dir "${MERGED_DIR}" \
    >> "${LOG_FILE}" 2>&1 || echo "[run_arm] step ${STEP}: upload FAILED/SKIPPED (see ${LOG_FILE}), not failing the job for this" >&2
done

echo "[run_arm] LINEAGE=${LINEAGE} done."
