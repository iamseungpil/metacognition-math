#!/bin/bash
# run_e190_cf_eval.sh — node-side body of the E-190 offline counterfactual eval.
#
# Lives in the repo (and therefore in the code tarball) rather than inside the
# amlt YAML on purpose: nested quoting inside an amlt command string parses fine
# at submit and breaks at node runtime (0717 incident). The YAML now only sets
# ARM/GS and calls this.
#
# Env in: ARM (b2p|b3s|b3sh|...), GS, REPO, HF_TOKEN, SIMPLERL_DIR.
set -o pipefail

: "${ARM:?ARM not set}"
: "${GS:?GS not set}"
: "${REPO:?REPO not set}"
PYBIN="${SIMPLERL_DIR}/bin/python"
CFG="rq3v2f_${ARM}"
ROOT=/scratch/metacognition
VAL="${ROOT}/data/verl_val_meta_mix.parquet"
MERGED="/scratch/eval_results/merged_${ARM}_gs${GS}"
BASE="/scratch/eval_results/cf_${ARM}_gs${GS}"
CKPT="/scratch/checkpoints/${CFG}/global_step_${GS}/actor"

echo "[e190] arm=${ARM} config=${CFG} gs=${GS}"

# --- G2 wiring check: refuse to run a stale code asset. The whole point of this
# --- launch is the two new fields; a tarball without them would silently
# --- reproduce the nine prior runs and look like a fresh measurement.
for needle in repeat_arm_a meta_signature_without; do
    grep -q "${needle}" "${ROOT}/src/eval/eval_counterfactual_difficulty.py" \
        || { echo "[e190] STALE code asset: '${needle}' missing; ABORT window"; exit 1; }
done
[ -f "${ROOT}/experiments/analysis/e190_cf_gates.py" ] \
    || { echo "[e190] STALE code asset: e190_cf_gates.py missing; ABORT window"; exit 1; }
echo "[e190] code asset OK"

"${PYBIN}" "${ROOT}/scripts/pull_parquets.py" 2>&1 | tail -3
ls -la "${VAL}" || { echo "[e190] val parquet missing"; exit 1; }

CFG_NAME="${CFG}" "${PYBIN}" - <<'PYDL'
import os
from huggingface_hub import snapshot_download
cfg, gs = os.environ["CFG_NAME"], os.environ["GS"]
snapshot_download(repo_id=os.environ["REPO"], repo_type="model", token=os.environ["HF_TOKEN"],
                  allow_patterns=["checkpoints/%s/global_step_%s/**" % (cfg, gs)], local_dir="/scratch")
print("[e190] downloaded %s gs%s" % (cfg, gs))
PYDL

ls -la "${CKPT}" || { echo "[e190] checkpoint missing ${CKPT}"; exit 1; }
mkdir -p "${MERGED}"
"${PYBIN}" -m verl.model_merger merge --backend fsdp --local_dir "${CKPT}" --target_dir "${MERGED}"
rc=$?
[ "${rc}" -eq 0 ] || { echo "[e190] merge FAILED rc=${rc}"; exit 1; }

MERGED_DIR="${MERGED}" "${PYBIN}" - <<'PYTOK'
import json, os
p = os.path.join(os.environ["MERGED_DIR"], "tokenizer_config.json")
if os.path.exists(p):
    c = json.load(open(p))
    c.pop("extra_special_tokens", None)
    json.dump(c, open(p, "w"), indent=2)
    print("[e190] tokenizer_config patched")
PYTOK

export PYTHONPATH="${ROOT}"
cd "${ROOT}" || exit 1
"${PYBIN}" -m src.eval.eval_counterfactual_difficulty \
    --model_path "${MERGED}" --val_parquet "${VAL}" --out "${BASE}.jsonl" \
    --tp_size 4 --max_new_tokens 16384 --repeat_arm_a 2>&1 | tee "${BASE}.log"
rc=$?
[ "${rc}" -eq 0 ] || { echo "[e190] cf eval FAILED rc=${rc}"; exit 1; }

echo "================ E-190 ${ARM} gs${GS} — stratified ================"
"${PYBIN}" -m src.eval.eval_counterfactual_difficulty_summarize --jsonl "${BASE}.jsonl" \
    2>&1 | tee "${BASE}_summary.txt"
echo "================ E-190 ${ARM} gs${GS} — GATES + PRIMARY ================"
"${PYBIN}" -m experiments.analysis.e190_cf_gates --jsonl "${BASE}.jsonl" \
    2>&1 | tee "${BASE}_gates.txt"
echo "======================================================================"

ARMN="${ARM}" GSN="${GS}" "${PYBIN}" - <<'PYUP'
import os
from huggingface_hub import HfApi
arm, gs = os.environ["ARMN"], os.environ["GSN"]
api = HfApi(token=os.environ["HF_TOKEN"])
base = "/scratch/eval_results/cf_%s_gs%s" % (arm, gs)
for suf in [".jsonl", "_summary.txt", "_gates.txt", ".log"]:
    fn = base + suf
    try:
        api.upload_file(path_or_fileobj=fn, repo_id="iamseungpil/metacot-rv", repo_type="dataset",
                        path_in_repo="eval/e190_%s_gs%s/%s" % (arm, gs, os.path.basename(fn)))
        print("[e190] uploaded " + os.path.basename(fn))
    except Exception as e:
        print("[e190] upload skip %s %s" % (fn, str(e)[:120]))
PYUP

echo "[e190] ${ARM} gs${GS} DONE"
