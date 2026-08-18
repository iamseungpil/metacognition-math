#!/usr/bin/env python
"""Generate countdown_rl_8arm.yaml.

The eight arm job blocks MUST be byte-identical except for `name:` and `ARM:`.
Generating them from one template is the only way to guarantee that, which is
what makes the G8 diff (arm-vs-arm) readable.
"""
import io

OUT = "/home/v-seungplee/metacognition-math/countdown_rl_8arm.yaml"

ARMS = [
    ("A", "corr",   "R_corr only (no meta reward)"),
    ("B", "cur",    "clip(shift,+-2) + reversal(save 1.0/derail 2.0) - current recipe"),
    ("C", "mul",    "clip(shift,+-2) x sign(A_corr) - MULTIPLY"),
    ("D", "ctx",    "context contrast x sign(A_corr) - COMPUTATION"),
    ("E", "gate",   "-(2*p_hat-1) x 1{meta emitted} - GATING"),
    ("F", "full",   "C + E"),
    ("G", "neg",    "meta_len/100 - FAKE control (length)"),
    ("H", "oldfmt", "same reward as F, OLD one-line meta format"),
]

DESCRIPTION = (
    "COUNTDOWN 8-ARM SEQUENCE-LEVEL GRPO (Qwen3-4B, NO SFT, prompt-only). Eight single-GPU "
    "arms A..H that differ by EXACTLY ONE key: ++algorithm.countdown_arm, which indexes "
    "src/training/countdown_rewards.ARM_SPECS. NO reward weight is written in this yaml - the "
    "weights live in ARM_SPECS so that `diff` of two arm blocks in this file shows one line "
    "(ARM) and nothing else. Sequence-level GRPO: the DCPO 3-region routing is NOT used and "
    "src/training/dcpo_region.py is NOT touched, because correctness already flows to every "
    "token, so the meta tokens do not need a region to receive credit. Shared (NOT treatment): "
    "w_corr 1.0, w_format 0.35, meta_floor 0.02, all other heads 0, w_meta/w_gate linear warmup "
    "over steps 0..20 - all of those live in ARM_SPECS/config, not here. Budget: train_batch 64 "
    "x rollout.n 8, 150 steps, lr 1e-6, response cap 3072. Arm G (length) is the falsifier: if G "
    "beats any of B..F the whole family is discarded. Job cd8_gs0 is the pre-training origin "
    "photo (C-035) and must finish BEFORE the eight training arms are judged. Every staging "
    "input is guarded fail-closed (code tar, countdown parquets, config, ARM_SPECS key, init "
    "model) because a silent staging miss has burned a 3h window with zero output before. "
    "H100x1 x8 msrresrchbasicvc Standard."
)

ENV_BLOCK = """      env:
        _AZUREML_SINGULARITY_JOB_UAI: "/subscriptions/c4c534bc-9978-4974-9c87-551f7c5754ef/resourceGroups/msra-sh-aml-rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/msra-sh-aml-uai"
        HF_TOKEN: ${HF_TOKEN}
        HUGGING_FACE_HUB_TOKEN: ${HF_TOKEN}
        HF_HUB_DOWNLOAD_TIMEOUT: "60"
        HF_HUB_DISABLE_XET: "1"
        GH_TOKEN: ${GH_TOKEN}
        CODE_TAR_REVISION: "REPLACE_WITH_COUNTDOWN_ASSET_ID"
        SIMPLERL_DIR: /scratch/conda_envs/simplerl
        WANDB_API_KEY: ${WANDB_API_KEY}
        WANDB_KEY: ${WANDB_API_KEY}
        WANDB_PROJECT: metacot-countdown-8arm
        WANDB_RUN_GROUP: countdown_8arm
        WANDB_RESUME: allow
        WANDB_INIT_TIMEOUT: "300"
        WANDB_TAGS: countdown,8arm,seq_grpo,qwen3_4b,prompt_only,no_sft
        PYTHONFAULTHANDLER: "1"
        CKPT_REPO: iamseungpil/metacot-h200-triobj-dcpo-v3
        DATA_REPO: iamseungpil/metacot-sdc-data
        INIT_MODEL: Qwen/Qwen3-4B
        CONFIG_NAME: countdown_grpo_qwen3_4b
        DATA_TRAIN: /scratch/metacognition/data/countdown_train.parquet
        DATA_VAL: /scratch/metacognition/data/countdown_val.parquet
"""

# ── the shared node-side body. Byte-identical in all eight arm jobs. ───────────
# Discipline: NOT ONE single quote inside (the `bash -c '...'` wrapper owns the
# only pair), and no comment or blank line ever follows a backslash-continued line.
TRAIN_BODY = r"""    command:
      - nvidia-smi
      - |
        bash -c '
        set +e
        set -x
        mkdir -p /scratch/logs /scratch/checkpoints /scratch/models /scratch/eval_results
        cd /scratch
        source /opt/conda/etc/profile.d/conda.sh
        conda activate ptca
        # inline GPU spinner from t~0 to close the Singularity suspend-on-idle window.
        nohup python -c "
        import torch, time
        g = torch.cuda.device_count()
        ts = [torch.randn(2048, 2048, device=f\"cuda:{i}\", dtype=torch.float16) for i in range(g)]
        while True:
            for i in range(g): ts[i] = (ts[i] @ ts[i]) * 0.5
            time.sleep(2)
        " > /scratch/logs/gpu_keeper_inline.log 2>&1 &
        pip install -q "huggingface_hub<1.0" 2>&1 | tail -1

        # ── ARM. The ONLY per-job variable in this file. Everything else derives from
        # it, so two arm blocks differ by exactly one line.
        case "$${ARM}" in [A-H]) ;; *) echo "[YAML] FATAL ARM=$${ARM} is not one of A..H; ABORT window"; sleep 60; exit 1;; esac

        set +x  # do NOT leak GH_TOKEN into xtrace logs
        curl -fsSL -H "Authorization: token $${GH_TOKEN}" -H "Accept: application/octet-stream" -o /tmp/metacognition.tar.gz https://api.github.com/repos/iamseungpil/metacognition-math/releases/assets/$${CODE_TAR_REVISION} || { echo "[YAML] FATAL code tar $${CODE_TAR_REVISION} download failed; ABORT window"; sleep 300; exit 1; }
        set -x
        tar -tzf /tmp/metacognition.tar.gz >/dev/null || { echo "[YAML] FATAL code tar corrupt; ABORT window"; sleep 300; exit 1; }
        tar -xzf /tmp/metacognition.tar.gz -C /scratch || { echo "[YAML] FATAL code tar extract failed; ABORT window"; sleep 300; exit 1; }
        echo "$${CODE_TAR_REVISION}" > /scratch/metacognition/.bootstrap_version

        nohup python /scratch/metacognition/scripts/gpu_keeper.py > /scratch/logs/gpu_keeper_early.log 2>&1 &
        bash /scratch/metacognition/scripts/bootstrap_sdc_node.sh
        export PATH="$${SIMPLERL_DIR}/bin:$${PATH}"
        export CONDA_PREFIX="$${SIMPLERL_DIR}"
        PYBIN=$${SIMPLERL_DIR}/bin/python
        # hard-remove hf_xet (env flag + pip uninstall alone are insufficient).
        pip uninstall -y hf_xet 2>&1 | tail -1 || true
        find $${SIMPLERL_DIR}/lib -maxdepth 3 -name "hf_xet*" -exec rm -rf {} + 2>/dev/null || true
        export PYTHONPATH=/scratch/metacognition
        cd /scratch/metacognition

        # ══ STAGING GUARD 1/4 - CODE. A tar built before the countdown files exist
        # would otherwise let verl start, fail on an import, and drop the window into
        # the trailing sleep, holding an H100 for hours with zero output.
        for F in configs/$${CONFIG_NAME}.yaml src/training/countdown_rewards.py src/training/countdown_task.py; do
          test -f /scratch/metacognition/$${F} || { echo "[YAML] FATAL staging: $${F} missing from code tar $${CODE_TAR_REVISION}; ABORT window"; sleep 300; exit 1; }
        done
        echo "[YAML] staging guard 1/4 CODE ok"

        # ══ STAGING GUARD 2/4 - ARM KEY. The launcher carries NO reward weight; it
        # carries a key. If the key does not resolve, the run would silently train a
        # different arm than its name claims. Resolve it here and PRINT the spec, so
        # the actual weights land in the run log as provenance.
        $${PYBIN} -c "
        import os, sys
        sys.path.insert(0, \"/scratch/metacognition\")
        from src.training.countdown_rewards import ARM_SPECS, SPEC_VERSION, arm_signature
        arm = os.environ[\"ARM\"]
        assert arm in ARM_SPECS, \"ARM %s not in ARM_SPECS %s\" % (arm, sorted(ARM_SPECS))
        print(\"[YAML] SPEC_VERSION=%s\" % SPEC_VERSION)
        print(\"[YAML] ARM_SPECS[%s] = %r\" % (arm, ARM_SPECS[arm]))
        print(\"[YAML] arm_signature(%s) = %s\" % (arm, arm_signature(arm)))
        open(\"/scratch/arm_label.txt\", \"w\").write(ARM_SPECS[arm][\"label\"])
        " || { echo "[YAML] FATAL staging: ARM_SPECS[$${ARM}] did not resolve; ABORT window"; sleep 300; exit 1; }
        echo "[YAML] staging guard 2/4 ARM_SPECS ok"

        # ── LINEAGE derives from the ARM_SPECS LABEL, never from the job name. That is
        # the one wiring that cannot drift: editing ARM without editing `name:` (the
        # copy-paste failure an eight-block file invites) now moves the checkpoints and
        # the wandb run together with the reward, so a mislabeled block is visible as a
        # name/lineage mismatch in the log instead of silently mixing two arms into one
        # lineage. `name:` is cosmetic; LINEAGE is authoritative.
        ARM_LABEL=$$(cat /scratch/arm_label.txt)
        test -n "$${ARM_LABEL}" || { echo "[YAML] FATAL empty arm label; ABORT window"; sleep 300; exit 1; }
        export LINEAGE=cd8_$${ARM_LABEL}
        export WANDB_NAME=$${LINEAGE}
        export WANDB_RUN_ID=$${LINEAGE}-1
        echo "[YAML] ARM=$${ARM} LABEL=$${ARM_LABEL} LINEAGE=$${LINEAGE} (job name should read cd8_$${ARM_LABEL})"

        # ══ STAGING GUARD 3/4 - DATA. Countdown parquets are pulled explicitly (they
        # are not in the fixed list scripts/pull_parquets.py walks).
        $${PYBIN} -c "
        import os, time
        from huggingface_hub import hf_hub_download
        tok = os.environ[\"HF_TOKEN\"]
        for fn in [\"countdown_train.parquet\", \"countdown_val.parquet\"]:
            for a in range(5):
                try:
                    p = hf_hub_download(repo_id=os.environ[\"DATA_REPO\"], repo_type=\"dataset\", filename=fn,
                                        token=tok, local_dir=\"/scratch/metacognition/data\")
                    print(\"[data]\", p, os.path.getsize(p), flush=True); break
                except Exception as e:
                    print(\"[data] attempt\", a, repr(e)[:160], flush=True); time.sleep(20*(a+1))
        "
        test -s $${DATA_TRAIN} || { echo "[YAML] FATAL staging: $${DATA_TRAIN} missing or empty; ABORT window"; sleep 300; exit 1; }
        test -s $${DATA_VAL} || { echo "[YAML] FATAL staging: $${DATA_VAL} missing or empty; ABORT window"; sleep 300; exit 1; }
        echo "[YAML] staging guard 3/4 DATA ok"

        # ══ STAGING GUARD 4/4 - INIT. Stock Qwen3-4B, NO SFT (prompt-only regime).
        $${PYBIN} -c "
        import os, time
        from huggingface_hub import snapshot_download
        ok = False
        for a in range(6):
            try:
                snapshot_download(repo_id=os.environ[\"INIT_MODEL\"], token=os.environ.get(\"HF_TOKEN\"),
                                  local_dir=\"/scratch/models/init\", max_workers=2,
                                  allow_patterns=[\"*.json\", \"*.safetensors\", \"*.txt\", \"*.model\"])
                ok = True; break
            except Exception as e:
                print(\"[init] attempt\", a, repr(e)[:160], flush=True); time.sleep(30*(a+1))
        assert ok, \"[init] snapshot_download exhausted retries\"
        "
        test -f /scratch/models/init/config.json || { echo "[YAML] FATAL staging: init model incomplete; ABORT window"; sleep 300; exit 1; }
        echo "[YAML] staging guard 4/4 INIT ok"

        # ── EOS PROBE (observability only, NO patching). The spec flags an open item:
        # generation_config lists TWO eos ids [151645, 151643] and it is unconfirmed
        # whether the verl response mask accepts a list. Print both so the answer is in
        # the log of every run instead of in nobody head.
        $${PYBIN} -c "
        import json, os
        c = json.load(open(\"/scratch/models/init/config.json\"))
        g = {}
        p = \"/scratch/models/init/generation_config.json\"
        if os.path.exists(p): g = json.load(open(p))
        t = {}
        q = \"/scratch/models/init/tokenizer_config.json\"
        if os.path.exists(q): t = json.load(open(q))
        e = g.get(\"eos_token_id\")
        print(\"[EOS] config.eos_token_id=%r generation_config.eos_token_id=%r tokenizer.eos_token=%r pad=%r\" % (c.get(\"eos_token_id\"), e, t.get(\"eos_token\"), g.get(\"pad_token_id\")))
        if isinstance(e, list): print(\"[EOS] WARN generation_config.eos_token_id is a LIST - confirm the verl response mask consumes a list, else the mask may only close on the first id\")
        "

        # ── pusher --token contract patch (argparse required=True in older tarball
        # copies kills the pusher at boot and NO checkpoint ever reaches HF). Fail closed.
        sed -i "s/ap.add_argument(\"--token\", required=True)/ap.add_argument(\"--token\", default=os.environ.get(\"HF_TOKEN\"))/" /scratch/metacognition/scripts/push_ckpts_to_hf.py
        grep -q "add_argument(\"--token\", default=os.environ.get(\"HF_TOKEN\"))" /scratch/metacognition/scripts/push_ckpts_to_hf.py || { echo "[YAML] FATAL pusher --token contract patch did not land; ABORT window"; sleep 300; exit 1; }

        # ══ LINEAGE FRESHNESS. Prints, for THIS lineage, the highest resumable step and
        # whether ANY file exists under it. ANY=0 is the proof the lineage name is NEW
        # and a gs0 cold start cannot overwrite somebody evidence. World size is 1 here
        # (one GPU), so a complete step is model+extra_state+optim >= 1 shard each.
        RGS=$$($${PYBIN} -c "
        import os, re, time
        from collections import Counter
        from huggingface_hub import HfApi
        lin = os.environ[\"LINEAGE\"]
        fs = None
        for i in range(3):
            try:
                fs = HfApi(token=os.environ[\"HF_TOKEN\"]).list_repo_files(os.environ[\"CKPT_REPO\"], repo_type=\"model\")
                break
            except Exception:
                time.sleep(30)
        if fs is None:
            print(-1, 0); raise SystemExit
        cm = Counter(); ce = Counter(); co = Counter()
        pat = re.compile(re.escape(lin) + r\"/global_step_(\d+)/actor/(model|extra_state|optim)_world_size_\d+_rank_\d+\.pt$$\")
        for f in fs:
            m = pat.search(f)
            if m: {\"model\": cm, \"extra_state\": ce, \"optim\": co}[m.group(2)][int(m.group(1))] += 1
        steps = sorted(s for s, n in cm.items() if n >= 1 and ce[s] >= 1 and co[s] >= 1)
        print((steps[-1] if steps else 0), (1 if (cm or ce or co) else 0))
        ")
        echo "[YAML] lineage freshness $${LINEAGE}: RGS=$${RGS}  (format: STEP ANY; ANY=0 means the lineage is NEW on $${CKPT_REPO})"
        RGS_STEP=$${RGS%% *}; RGS_ANY=$${RGS##* }
        case "$${RGS_STEP}" in ""|*[!0-9-]*) echo "[YAML] FATAL lineage probe empty/garbled ($${RGS}); ABORT window"; sleep 300; exit 1;; esac
        if [ "$${RGS_STEP}" = "-1" ]; then echo "[YAML] FATAL HF unreachable for the lineage probe; ABORT window"; sleep 300; exit 1; fi
        if [ "$${RGS_ANY}" = "0" ]; then echo "[YAML] LINEAGE $${LINEAGE} IS NEW (0 files) - fresh gs0 start confirmed"; fi
        for rp in 1 2 3; do
          $${PYBIN} /scratch/metacognition/scripts/pull_resume_ckpt.py --repo $${CKPT_REPO} --config_name $${LINEAGE} --local_dir /scratch/checkpoints/$${LINEAGE} 2>&1 | tail -5
          ls -d /scratch/checkpoints/$${LINEAGE}/global_step_* >/dev/null 2>&1 && break
          echo "[YAML] pull_resume attempt $${rp} landed no local checkpoint; retry"; sleep 30
        done
        LOCAL_GS=$$(ls -d /scratch/checkpoints/$${LINEAGE}/global_step_* 2>/dev/null | grep -oE "global_step_[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
        echo "[YAML] RGS(HF)=$${RGS_STEP} ANY=$${RGS_ANY} LOCAL_GS(pulled)=$${LOCAL_GS:-NONE}"
        if [ "$${RGS_ANY}" = "1" ] && [ -z "$${LOCAL_GS}" ]; then
          echo "[YAML] ABORT: HF already carries lineage $${LINEAGE} (complete gs=$${RGS_STEP}) but the resume pull produced nothing; refusing a gs0 cold start that would overwrite it."
          sleep 300
          exit 1
        fi

        # ── checkpoint pusher. Started AFTER the resume pull on purpose: a pusher that
        # is already scanning the directory while pull_resume writes into it races the
        # download and can commit half-written shards.
        $${PYBIN} -m pip install --quiet --upgrade hf_xet 2>&1 | tail -2 || echo "[YAML] hf_xet install failed (non-fatal)"
        nohup env -u HF_HUB_DISABLE_XET $${PYBIN} /scratch/metacognition/scripts/push_ckpts_to_hf.py --ckpt_dir /scratch/checkpoints/$${LINEAGE} --repo_id $${CKPT_REPO} --interval 90 --keep 1 --squash_every 3 --config_name $${LINEAGE} > /scratch/logs/push_$${LINEAGE}.log 2>&1 &
        PUSH_PID=$$!
        echo "[YAML] pusher pid=$${PUSH_PID}"

        nohup python /scratch/metacognition/scripts/gpu_keeper.py > /scratch/logs/gpu_keeper.log 2>&1 &
        # `amlt run -i` sets LOCAL_RANK to empty; export a valid 0 so Ray worker actors
        # inherit a usable value when spawned.
        export LOCAL_RANK=0
        # ══ THE RUN. ++algorithm.countdown_arm is the FIRST override on purpose: it is
        # the single treatment key. Every other override below is a budget/shape knob that
        # is IDENTICAL in all eight arms; no reward weight appears anywhere in this file.
        # Sequence-level GRPO - no region routing, dcpo_region.py untouched.
        $${PYBIN} -u -m src.training.verl_sdc \
            --config-name=$${CONFIG_NAME} \
            ++algorithm.countdown_arm=$${ARM} \
            trainer.experiment_name=$${LINEAGE} \
            trainer.default_local_dir=/scratch/checkpoints/$${LINEAGE} \
            trainer.project_name=$${WANDB_PROJECT} \
            trainer.nnodes=1 \
            trainer.n_gpus_per_node=1 \
            actor_rollout_ref.model.path=/scratch/models/init \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.n=8 \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.rollout.top_k=-1 \
            actor_rollout_ref.rollout.top_p=1.0 \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            data.train_files=$${DATA_TRAIN} \
            data.val_files=$${DATA_VAL} \
            data.train_batch_size=64 \
            data.max_response_length=3072 \
            actor_rollout_ref.rollout.max_model_len=4096 \
            actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
            ++actor_rollout_ref.model.enable_activation_offload=true \
            actor_rollout_ref.rollout.enforce_eager=true \
            ++trainer.total_training_steps=150 \
            trainer.resume_mode=auto \
            trainer.save_freq=5 \
            trainer.test_freq=25 \
            ++trainer.val_before_train=False \
            ++hydra.searchpath=[pkg://verl/trainer/config] \
            > /scratch/logs/verl_main.log 2>&1 &
        VERL_PID=$$!
        # NO PIPE on the training process: a `| tee` block-buffers and loses its buffer
        # on a process-group SIGKILL, which is how nine silent deaths left an empty log.
        # verl writes unbuffered straight to a file, `tail -F` mirrors it into std_log, and
        # the 10s heartbeat separates preemption (abrupt stop, memory fine) from host OOM
        # (MemAvailable to the floor) from a silent hang (gpu0used frozen).
        tail -n +1 -F /scratch/logs/verl_main.log &
        TAIL_PID=$$!
        ( while kill -0 $${VERL_PID} 2>/dev/null; do echo "[HB $$(date)] $$(grep MemAvailable /proc/meminfo) gpu0used=$$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)MB"; sleep 10; done ) &
        HB_PID=$$!
        wait $${VERL_PID}
        VERL_RC=$$?
        kill $${TAIL_PID} $${HB_PID} 2>/dev/null || true
        echo "[YAML] verl_sdc $${LINEAGE} rc=$${VERL_RC}; tail of verl_main.log:"
        tail -60 /scratch/logs/verl_main.log 2>/dev/null

        # ── FINAL SYNC PUSH. The background pusher can freeze mid-run, so push the
        # highest local step synchronously and VERIFY the shard count on HF before
        # calling it durable. Kill the pusher BY PID: a pkill -f pattern also matches
        # this very bash -c command line and has SIGTERMed the window itself, which is
        # how one run lost its final weights.
        kill $${PUSH_PID} 2>/dev/null || true
        sleep 5
        kill -9 $${PUSH_PID} 2>/dev/null || true
        sleep 2
        kill -0 $${PUSH_PID} 2>/dev/null && echo "[YAML] WARN pusher $${PUSH_PID} still alive" || echo "[YAML] pusher $${PUSH_PID} stopped"
        FINAL=$$(ls -d /scratch/checkpoints/$${LINEAGE}/global_step_* 2>/dev/null | sort -V | tail -1)
        FN=$$(basename "$${FINAL}")
        echo "[YAML] final sync push target: $${FINAL}"
        if [ -n "$${FINAL}" ]; then
          for i in 1 2 3 4 5 6 7 8 9 10; do
            OUT=$$($${PYBIN} -c "
        import os
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ[\"HF_TOKEN\"])
        lin = os.environ[\"LINEAGE\"]
        api.upload_folder(folder_path=\"$${FINAL}\", path_in_repo=\"checkpoints/%s/$${FN}\" % lin,
                          repo_id=os.environ[\"CKPT_REPO\"], repo_type=\"model\", ignore_patterns=[\"*.tmp\", \"*.lock\"])
        fs = api.list_repo_files(os.environ[\"CKPT_REPO\"], repo_type=\"model\")
        nm = sum(1 for f in fs if (\"$${FN}/actor/model_world_size\" in f) and lin in f and f.endswith(\".pt\"))
        ne = sum(1 for f in fs if (\"$${FN}/actor/extra_state_world_size\" in f) and lin in f and f.endswith(\".pt\"))
        no = sum(1 for f in fs if (\"$${FN}/actor/optim_world_size\" in f) and lin in f and f.endswith(\".pt\"))
        print(\"SHARDS=%d\" % min(nm, ne, no))
        " 2>&1)
            echo "$${OUT}" | tail -2
            echo "$${OUT}" | grep -qE "SHARDS=[1-9]" && { echo "[YAML] FINAL PUSH DURABLE $${FN}"; break; } || { echo "[YAML] final push retry $$i"; sleep 60; }
          done
        fi
        echo "[YAML] ARM $${ARM} ($${LINEAGE}) WINDOW END rc=$${VERL_RC}"
        # DELIBERATE DEVIATION from the RQ3 launchers, which end in `sleep 86400`.
        # Eight arms holding a GPU each for a day after the run ends is eight GPU-days
        # of idle. 1800s is enough to read the tail and pull anything the final sync
        # push missed; the checkpoints are already durable on HF by this line.
        sleep 1800
        '
"""

GS0_BODY = r"""    command:
      - nvidia-smi
      - |
        bash -c '
        set +e
        set -x
        mkdir -p /scratch/logs /scratch/models /scratch/eval_results/cd8_gs0
        cd /scratch
        source /opt/conda/etc/profile.d/conda.sh
        conda activate ptca
        nohup python -c "
        import torch, time
        g = torch.cuda.device_count()
        ts = [torch.randn(2048, 2048, device=f\"cuda:{i}\", dtype=torch.float16) for i in range(g)]
        while True:
            for i in range(g): ts[i] = (ts[i] @ ts[i]) * 0.5
            time.sleep(2)
        " > /scratch/logs/gpu_keeper_inline.log 2>&1 &
        pip install -q "huggingface_hub<1.0" 2>&1 | tail -1

        set +x  # do NOT leak GH_TOKEN into xtrace logs
        curl -fsSL -H "Authorization: token $${GH_TOKEN}" -H "Accept: application/octet-stream" -o /tmp/metacognition.tar.gz https://api.github.com/repos/iamseungpil/metacognition-math/releases/assets/$${CODE_TAR_REVISION} || { echo "[YAML] FATAL code tar $${CODE_TAR_REVISION} download failed; ABORT window"; sleep 300; exit 1; }
        set -x
        tar -tzf /tmp/metacognition.tar.gz >/dev/null || { echo "[YAML] FATAL code tar corrupt; ABORT window"; sleep 300; exit 1; }
        tar -xzf /tmp/metacognition.tar.gz -C /scratch || { echo "[YAML] FATAL code tar extract failed; ABORT window"; sleep 300; exit 1; }

        nohup python /scratch/metacognition/scripts/gpu_keeper.py > /scratch/logs/gpu_keeper_early.log 2>&1 &
        bash /scratch/metacognition/scripts/bootstrap_sdc_node.sh
        export PATH="$${SIMPLERL_DIR}/bin:$${PATH}"
        export CONDA_PREFIX="$${SIMPLERL_DIR}"
        PYBIN=$${SIMPLERL_DIR}/bin/python
        pip uninstall -y hf_xet 2>&1 | tail -1 || true
        export PYTHONPATH=/scratch/metacognition
        cd /scratch/metacognition

        # ══ STAGING GUARDS (same fail-closed rule as the training arms).
        for F in scripts/countdown_gs0_eval.py src/training/countdown_rewards.py src/training/countdown_task.py; do
          test -f /scratch/metacognition/$${F} || { echo "[YAML] FATAL staging: $${F} missing from code tar $${CODE_TAR_REVISION}; ABORT window"; sleep 300; exit 1; }
        done
        $${PYBIN} -c "
        import os, time
        from huggingface_hub import hf_hub_download
        for fn in [\"countdown_val.parquet\"]:
            for a in range(5):
                try:
                    p = hf_hub_download(repo_id=os.environ[\"DATA_REPO\"], repo_type=\"dataset\", filename=fn,
                                        token=os.environ[\"HF_TOKEN\"], local_dir=\"/scratch/metacognition/data\")
                    print(\"[data]\", p, os.path.getsize(p), flush=True); break
                except Exception as e:
                    print(\"[data] attempt\", a, repr(e)[:160], flush=True); time.sleep(20*(a+1))
        "
        test -s $${DATA_VAL} || { echo "[YAML] FATAL staging: $${DATA_VAL} missing or empty; ABORT window"; sleep 300; exit 1; }
        $${PYBIN} -c "
        import os, time
        from huggingface_hub import snapshot_download
        ok = False
        for a in range(6):
            try:
                snapshot_download(repo_id=os.environ[\"INIT_MODEL\"], token=os.environ.get(\"HF_TOKEN\"),
                                  local_dir=\"/scratch/models/init\", max_workers=2,
                                  allow_patterns=[\"*.json\", \"*.safetensors\", \"*.txt\", \"*.model\"])
                ok = True; break
            except Exception as e:
                print(\"[init] attempt\", a, repr(e)[:160], flush=True); time.sleep(30*(a+1))
        assert ok, \"[init] snapshot_download exhausted retries\"
        "
        test -f /scratch/models/init/config.json || { echo "[YAML] FATAL staging: init model incomplete; ABORT window"; sleep 300; exit 1; }
        echo "[YAML] staging guards ok"

        # ══ UPLOAD-PREFIX FRESHNESS. Reusing a live eval prefix has silently overwritten
        # evidence before, so refuse to write into a prefix that already has files.
        $${PYBIN} -c "
        import os
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ[\"HF_TOKEN\"])
        fs = api.list_repo_files(os.environ[\"CKPT_REPO\"], repo_type=\"model\")
        n = sum(1 for f in fs if f.startswith(os.environ[\"UPLOAD_PREFIX\"] + \"/\"))
        print(\"[YAML] upload prefix %s currently holds %d files\" % (os.environ[\"UPLOAD_PREFIX\"], n))
        assert n == 0, \"upload prefix is NOT new - refusing to overwrite existing evidence\"
        " || { echo "[YAML] FATAL upload prefix not fresh; ABORT window"; sleep 300; exit 1; }

        upload_pass() {
          $${PYBIN} -c "
        import os, glob
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ[\"HF_TOKEN\"])
        for fn in glob.glob(\"/scratch/eval_results/cd8_gs0/**\", recursive=True):
            if not os.path.isfile(fn): continue
            if not (fn.endswith(\".parquet\") or fn.endswith(\".json\") or fn.endswith(\".log\")): continue
            rel = os.path.relpath(fn, \"/scratch/eval_results/cd8_gs0\")
            try:
                api.upload_file(path_or_fileobj=fn, path_in_repo=os.environ[\"UPLOAD_PREFIX\"] + \"/\" + rel,
                                repo_id=os.environ[\"CKPT_REPO\"], repo_type=\"model\")
            except Exception as e:
                print(\"upload skip\", fn, str(e)[:80])
        print(\"[YAML] pass upload done\")
        " || echo "[YAML] upload skipped"
        }

        export LOCAL_RANK=0
        # ══ ORIGIN PHOTO (C-035). Two passes, because two prompt formats are on trial:
        # the A..G block format and the H one-line format. Both are measured on the
        # UNTRAINED model so every telemetry number at step 0 has its own baseline
        # beside it - emit_rate, meta_position, selectivity, boilerplate share, answer
        # leak, p_hat distribution, decision split, confidence cardinality.
        for MFMT in new old; do
        NAME=cd8_gs0_$${MFMT}
        echo "================ GS0 $${NAME} ================"
        $${PYBIN} scripts/countdown_gs0_eval.py \
            --model_path /scratch/models/init \
            --data $${DATA_VAL} \
            --meta_format $${MFMT} \
            --num_samples 16 \
            --max_tokens 3072 \
            --temperature 1.0 \
            --top_p 1.0 \
            --top_k=-1 \
            --seed 42 \
            --tp_size 1 \
            --out_dir /scratch/eval_results/cd8_gs0/$${NAME} \
            2>&1 | tee /scratch/eval_results/cd8_gs0/$${NAME}.log || echo "[YAML] $${NAME} FAILED, continue"
        upload_pass
        done
        echo "[YAML] COUNTDOWN gs0 ORIGIN PHOTO DONE -> $${UPLOAD_PREFIX}"
        sleep 3600
        '
"""

GS0_ENV = """      env:
        _AZUREML_SINGULARITY_JOB_UAI: "/subscriptions/c4c534bc-9978-4974-9c87-551f7c5754ef/resourceGroups/msra-sh-aml-rg/providers/Microsoft.ManagedIdentity/userAssignedIdentities/msra-sh-aml-uai"
        HF_TOKEN: ${HF_TOKEN}
        HUGGING_FACE_HUB_TOKEN: ${HF_TOKEN}
        HF_HUB_DOWNLOAD_TIMEOUT: "60"
        HF_HUB_DISABLE_XET: "1"
        GH_TOKEN: ${GH_TOKEN}
        CODE_TAR_REVISION: "REPLACE_WITH_COUNTDOWN_ASSET_ID"
        SIMPLERL_DIR: /scratch/conda_envs/simplerl
        VLLM_WORKER_MULTIPROC_METHOD: spawn
        PYTHONFAULTHANDLER: "1"
        CKPT_REPO: iamseungpil/metacot-h200-triobj-dcpo-v3
        DATA_REPO: iamseungpil/metacot-sdc-data
        INIT_MODEL: Qwen/Qwen3-4B
        DATA_VAL: /scratch/metacognition/data/countdown_val.parquet
        UPLOAD_PREFIX: eval/cd8_gs0
"""

HEADER = '''description: "{desc}"

# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION NOTE. ${{HF_TOKEN}}, ${{GH_TOKEN}} and ${{WANDB_API_KEY}} are expanded by
# the SUBMITTING shell, not on the node. They are only valid if .env was sourced
# first:   set -a; source /home/v-seungplee/metacognition-math/.env; set +a
# A submission without that lands empty tokens in the job env and the code-tar
# download fails on the node, which the staging guard turns into an immediate
# exit 1 rather than a three-hour idle window.
#
# ARM WIRING. Each training job differs from its neighbours by ONE line: `ARM: "X"`.
# That letter is passed through to ++algorithm.countdown_arm and looked up in
# src/training/countdown_rewards.ARM_SPECS. No reward weight is written anywhere in
# this file, so `diff <(sed -n JOB_A_RANGE) <(sed -n JOB_G_RANGE)` is one line.
#
# NODE SHAPE. Eight single-GPU training jobs plus one single-GPU origin-photo job.
# On the ND48_H100_v5 instance type (4 x H100 per node) the eight arms occupy two
# nodes worth of GPUs.
# ─────────────────────────────────────────────────────────────────────────────

target:
  service: sing
  name: msrresrchbasicvc
  workspace_name: msra-sh-aml-ws

environment:
  registry: mcr.microsoft.com
  image: aifx/acpt/stable-ubuntu2204-cu126-py310-torch28x:latest

code:
  local_dir: $CONFIG_DIR/

jobs:
'''


def job_header(name, dur):
    return (
        "  - name: {name}\n"
        "    # SKU UNVERIFIED. One GPU per arm. The single-GPU string is confirmed only in\n"
        "    # the A100 form (80G1-A100, archive/launchers_retired_0727/a100g1_sft_*.yaml);\n"
        "    # 80G1-H100 is the analogous H100 name and has NO precedent in this repo, so\n"
        "    # confirm it against `amlt target info sing msrresrchbasicvc` before submitting.\n"
        "    # If it is rejected, the fallback is 2 jobs at 80G4-H100 running 4 arms each,\n"
        "    # pinned with CUDA_VISIBLE_DEVICES - which costs the one-key G8 diff.\n"
        "    sku: 80G1-H100\n"
        "    sla_tier: Standard\n"
        "    priority: high\n"
        "    identity: managed\n"
        "    submit_args:\n"
        "      max_run_duration_seconds: {dur}\n"
    ).format(name=name, dur=dur)


def main():
    buf = io.StringIO()
    buf.write(HEADER.format(desc=DESCRIPTION))

    # gs0 first: the origin photo must exist before the arms are judged.
    buf.write("  # ══ ORIGIN PHOTO (C-035): untrained Qwen3-4B on Countdown, both meta formats.\n")
    buf.write(job_header("cd8_gs0", 21600))
    buf.write(GS0_ENV)
    buf.write(GS0_BODY)
    buf.write("\n")

    for letter, short, note in ARMS:
        buf.write("  # ══ ARM {L} ({s}): {n}\n".format(L=letter, s=short, n=note))
        buf.write(job_header("cd8_{}".format(short), 604800))
        buf.write(ENV_BLOCK)
        buf.write('        ARM: "{}"\n'.format(letter))
        buf.write(TRAIN_BODY)
        if letter != ARMS[-1][0]:
            buf.write("\n")

    text = buf.getvalue()
    with open(OUT, "w") as fh:
        fh.write(text)
    print("wrote", OUT, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
