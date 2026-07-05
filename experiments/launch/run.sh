#!/usr/bin/env bash
# experiments/launch/run.sh — compose a science yaml with an infra yaml and run.
#
# Usage:
#   ./run.sh <science.yaml> <infra.yaml> [--seed N]
#
#   science.yaml : experiments/configs/science/*.yaml (what the experiment IS)
#   infra.yaml   : experiments/configs/infra/*.yaml   (where/how it runs)
#   --seed N     : override the science-yaml seed (T5 seed sweeps)
#
# Behavior by science `mode`:
#   eval : EXECUTES scripts/eval_vllm_1030.py for every model x pass in the
#          science yaml (both arms in the same job, same seed).
#   sft  : merges `overrides` onto `derives_from`, writes the merged config to
#          experiments/launch/generated/, then EXECUTES
#          accelerate launch src/training/sft.py --config <merged>.
#   rl   : composes and PRINTS the exact verl command with all params merged.
#          It is printed, not executed, because RL needs the verl 0.7.1 env
#          bootstrapped by scripts/bootstrap_sdc_node.sh (Ray + vLLM stack);
#          on MSR, submit through the repo-root h100std_*.yaml amlt jobs, which
#          also handle the 6h-window HF checkpoint relay
#          (scripts/pull_resume_ckpt.py / scripts/push_ckpts_to_hf.py).
#
# Secrets: NEVER passed here. Export HF_TOKEN / WANDB_API_KEY from .env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Interpreter for YAML parsing/merging (needs PyYAML; every training env ships
# it). Override with RUN_PYTHON=... if your default python3 lacks PyYAML.
PYPARSE=""
for cand in "${RUN_PYTHON:-}" python3 python /usr/bin/python3; do
    [ -n "$cand" ] || continue
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import yaml" >/dev/null 2>&1; then
        PYPARSE="$cand"
        break
    fi
done
[ -n "$PYPARSE" ] || { echo "[run] no python with PyYAML found — pip install pyyaml or set RUN_PYTHON" >&2; exit 1; }

usage() {
    echo "usage: $0 <science.yaml> <infra.yaml> [--seed N]" >&2
    echo "  science: experiments/configs/science/*.yaml" >&2
    echo "  infra:   experiments/configs/infra/*.yaml" >&2
    exit 1
}

[ $# -ge 2 ] || usage
SCI="$1"
INFRA="$2"
shift 2

SEED_CLI=""
while [ $# -gt 0 ]; do
    case "$1" in
        --seed) [ $# -ge 2 ] || usage; SEED_CLI="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

# Resolve config paths (accept absolute, cwd-relative, or repo-relative).
resolve() {
    if [ -f "$1" ]; then echo "$1"
    elif [ -f "$ROOT/$1" ]; then echo "$ROOT/$1"
    else echo "config not found: $1" >&2; exit 1
    fi
}
SCI="$(resolve "$SCI")"
INFRA="$(resolve "$INFRA")"

# ── YAML accessors (python3 -c one-liners; dotted keys, list indices ok) ─────
ycfg() {  # ycfg <file> <dotted.key> [default] -> scalar (lists join with space)
    "$PYPARSE" -c '
import sys, yaml
path, key = sys.argv[1], sys.argv[2]
default = sys.argv[3] if len(sys.argv) > 3 else ""
cur = yaml.safe_load(open(path))
for part in key.split("."):
    if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
        cur = cur[int(part)]
    elif isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        print(default, end=""); raise SystemExit
if isinstance(cur, bool):
    print(str(cur).lower(), end="")
elif isinstance(cur, list):
    print(" ".join(str(x) for x in cur), end="")
else:
    print("" if cur is None else cur, end="")
' "$@"
}

ylen() {  # ylen <file> <dotted.key> -> list length (0 if absent)
    "$PYPARSE" -c '
import sys, yaml
cur = yaml.safe_load(open(sys.argv[1]))
for part in sys.argv[2].split("."):
    cur = cur.get(part) if isinstance(cur, dict) else None
print(len(cur) if isinstance(cur, list) else 0)
' "$@"
}

ylist() {  # ylist <file> <dotted.key> -> one list item per line
    "$PYPARSE" -c '
import sys, yaml
cur = yaml.safe_load(open(sys.argv[1]))
for part in sys.argv[2].split("."):
    cur = cur.get(part) if isinstance(cur, dict) else None
if isinstance(cur, list):
    print("\n".join(str(x) for x in cur))
' "$@"
}

MODE="$(ycfg "$SCI" mode)"
EXP="$(ycfg "$SCI" experiment)"
SEED="${SEED_CLI:-$(ycfg "$SCI" seed 42)}"

RUN_MODE="$(ycfg "$INFRA" run_mode local)"
TP="$(ycfg "$INFRA" eval_tp_size 1)"
GMU="$(ycfg "$INFRA" gpu_memory_utilization 0.85)"
SCRATCH="$(ycfg "$INFRA" scratch_dir ./scratch)"
ACC_CFG="$(ycfg "$INFRA" accelerate_config configs/accelerate_sft.yaml)"
TRAIN_PY="$(ycfg "$INFRA" train_python python3)"
VERL_PY="$(ycfg "$INFRA" verl_python python3)"
EVAL_PY="$(ycfg "$INFRA" eval_python python3)"
SMOKE="$(ycfg "$INFRA" smoke false)"
case "$SCRATCH" in /*) : ;; *) SCRATCH="$ROOT/${SCRATCH#./}" ;; esac

echo "[run] experiment=$EXP mode=$MODE seed=$SEED infra=$(ycfg "$INFRA" infra) smoke=$SMOKE"
[ "$RUN_MODE" = "amlt" ] && echo "[run] NOTE: this infra submits training through the repo-root amlt yamls; direct execution below is for ON-NODE use after scripts/bootstrap_sdc_node.sh."

case "$MODE" in

# ── EVAL: fully executed ─────────────────────────────────────────────────────
eval)
    EVAL_SCRIPT="$ROOT/$(ycfg "$SCI" eval_script scripts/eval_vllm_1030.py)"
    [ -f "$EVAL_SCRIPT" ] || { echo "[run] eval script missing: $EVAL_SCRIPT" >&2; exit 1; }
    TEMP="$(ycfg "$SCI" temperature 0.7)"
    TOPP="$(ycfg "$SCI" top_p 0.95)"
    MAXTOK="$(ycfg "$SCI" max_tokens 16384)"
    MML="$(ycfg "$SCI" max_model_len 20480)"
    MAXPROB="$(ycfg "$SCI" max_problems 500)"
    if [ "$SMOKE" = "true" ]; then
        MAXPROB=10; MAXTOK=2048
        echo "[run] SMOKE: max_problems=10 max_tokens=2048 num_samples=1 (plumbing only, never for tables)"
    fi

    NMODELS="$(ylen "$SCI" models)"
    NPASSES="$(ylen "$SCI" passes)"
    [ "$NMODELS" -gt 0 ] && [ "$NPASSES" -gt 0 ] || { echo "[run] eval yaml needs models[] and passes[]" >&2; exit 1; }

    # Both arms MUST be present in the same job — verify every path up front.
    m=0
    while [ "$m" -lt "$NMODELS" ]; do
        MP="$(ycfg "$SCI" "models.$m.path")"
        if [ ! -d "$MP" ]; then
            echo "[run] MISSING model dir: $MP ($(ycfg "$SCI" "models.$m.name"))" >&2
            echo "      stage it first — merge $(ycfg "$SCI" "models.$m.hf_checkpoint") from the HF relay repo (recipe in the science yaml header)" >&2
            exit 1
        fi
        m=$((m + 1))
    done

    OUTROOT="$ROOT/results/eval_1030_${EXP}_seed${SEED}"
    mkdir -p "$OUTROOT"
    m=0
    while [ "$m" -lt "$NMODELS" ]; do
        MNAME="$(ycfg "$SCI" "models.$m.name")"
        MPATH="$(ycfg "$SCI" "models.$m.path")"
        p=0
        while [ "$p" -lt "$NPASSES" ]; do
            PNAME="$(ycfg "$SCI" "passes.$p.name")"
            BENCHES="$(ycfg "$SCI" "passes.$p.benchmarks")"
            NS="$(ycfg "$SCI" "passes.$p.num_samples" 1)"
            [ "$SMOKE" = "true" ] && NS=1
            RUN_NAME="${MNAME}_${PNAME}_seed${SEED}"
            echo "================ EVAL $RUN_NAME ================"
            # shellcheck disable=SC2086  # BENCHES is a space-separated nargs+ list
            (cd "$ROOT" && PYTHONPATH="$ROOT" "$EVAL_PY" "$EVAL_SCRIPT" \
                --model_path "$MPATH" \
                --model_name "$RUN_NAME" \
                --output_dir "$OUTROOT/$RUN_NAME" \
                --benchmarks $BENCHES \
                --max_problems "$MAXPROB" \
                --max_tokens "$MAXTOK" \
                --temperature "$TEMP" \
                --top_p "$TOPP" \
                --tp_size "$TP" \
                --gpu_memory_utilization "$GMU" \
                --max_model_len "$MML" \
                --num_samples "$NS" \
                --seed "$SEED")
            p=$((p + 1))
        done
        m=$((m + 1))
    done
    echo "[run] eval complete -> $OUTROOT"
    ;;

# ── SFT: merge science overrides onto the real config, then execute ─────────
sft)
    BASE_CFG="$ROOT/$(ycfg "$SCI" derives_from)"
    [ -f "$BASE_CFG" ] || { echo "[run] derives_from missing: $BASE_CFG" >&2; exit 1; }
    GEN_DIR="$SCRIPT_DIR/generated"
    mkdir -p "$GEN_DIR"
    MERGED="$GEN_DIR/${EXP}_seed${SEED}.yaml"
    "$PYPARSE" -c '
import sys, yaml
base, sci, out, seed, smoke = sys.argv[1:6]
cfg = yaml.safe_load(open(base))
sci_doc = yaml.safe_load(open(sci))
cfg.update(sci_doc.get("overrides") or {})
# seed provenance in run_name (sft.py exposes no training-seed knob; its
# internal train_test_split seed is fixed at 42).
cfg["run_name"] = "%s-seed%s" % (sci_doc["experiment"], seed)
if smoke == "true":
    cfg["num_train_epochs"] = 1
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("[run] merged sft config -> %s" % out)
' "$BASE_CFG" "$SCI" "$MERGED" "$SEED" "$SMOKE"
    echo "[run] launching SFT (dataset must be staged at <repo>/$(ycfg "$SCI" data.train_parquet) — hf_hub_download of that path from $(ycfg "$SCI" data.hf_dataset))"
    (cd "$ROOT" && PYTHONPATH="$ROOT" "$TRAIN_PY" -m accelerate.commands.launch \
        --config_file "$ROOT/$ACC_CFG" \
        "$ROOT/src/training/sft.py" --config "$MERGED")
    ;;

# ── RL: compose and PRINT the exact verl command (not executed) ─────────────
rl)
    VCFG="$(ycfg "$SCI" verl_config_name)"
    EXPNAME="$(ycfg "$SCI" experiment_name "$EXP")"
    INIT="$(ycfg "$SCI" init_model)"
    STEPS="$(ycfg "$SCI" total_training_steps 300)"
    SAVEF="$(ycfg "$SCI" save_freq 10)"
    VALF="$(ycfg "$SCI" val_freq 25)"
    TRAINPQ="$(ycfg "$SCI" data.train_parquet)"
    VALPQ="$(ycfg "$SCI" data.val_parquet)"

    CMD=( "$VERL_PY" -u -m src.training.verl_sdc
        "--config-name=$VCFG"
        "trainer.experiment_name=$EXPNAME"
        "trainer.default_local_dir=$SCRATCH/checkpoints/$EXPNAME"
        "actor_rollout_ref.model.path=$INIT"
        "data.train_files=$ROOT/$TRAINPQ"
        "data.val_files=$ROOT/$VALPQ"
        "trainer.total_training_steps=$STEPS"
        "trainer.save_freq=$SAVEF"
        "trainer.test_freq=$VALF"
        "++actor_rollout_ref.actor.data_loader_seed=$SEED"
    )
    while IFS= read -r ov; do
        [ -n "$ov" ] && CMD+=( "$ov" )
    done < <(ylist "$SCI" hydra_overrides)
    CMD+=( '++hydra.searchpath=[pkg://verl/trainer/config]' )

    echo
    echo "[run] RL mode ($(ycfg "$SCI" reward_mode)) — command COMPOSED, not executed."
    echo "[run] Requires the verl 0.7.1 env from scripts/bootstrap_sdc_node.sh; on MSR"
    echo "[run] submit via the repo-root amlt yamls (6h-window HF ckpt relay:"
    echo "[run] scripts/pull_resume_ckpt.py before launch, scripts/push_ckpts_to_hf.py"
    echo "[run] in the background, config_name=$EXPNAME). Run on-node:"
    echo
    echo "cd $ROOT && \\"
    echo "PYTHONPATH=$ROOT LOCAL_RANK=0 WANDB_NAME=$EXPNAME \\"
    printf '%q ' "${CMD[@]}"
    echo
    ;;

*)
    echo "[run] unknown mode '$MODE' in $SCI (expected sft|rl|eval)" >&2
    exit 1
    ;;
esac
