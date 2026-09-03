# scripts/local — single-machine Countdown RL pipeline (GPUs 0-3)

## 0. One-time setup
```bash
source scripts/local/env.sh          # loads .env (tolerant if missing), activates
                                      # simplerl env, sets $WORK, Ray/vLLM knobs
bash scripts/local/make_data.sh      # builds countdown_{train,val}_4num_<variant>.parquet
                                      # for plain/new/p3 (skips p3 until it exists
                                      # in countdown_task.PROMPT_VARIANTS)
```

## 1. Start workers (one process per GPU 0-3)
```bash
python scripts/local/gpu_queue.py start-workers 0 1 2 3
```
Each worker polls `/hdd_data/seungpil/queue/pending/`, claims the highest-priority
job atomically, runs it with `CUDA_VISIBLE_DEVICES=<its GPU>`, and refuses to
start a job while `/hdd_data usage > 85%` or `/ free < 15GB` (logs `[DISK] waiting`).

## 2. Submit the standard wave (one arm per GPU)
```bash
for arm in N0 A B C; do
  python scripts/local/gpu_queue.py submit \
    --name "cd7_${arm}_p3_s0" --priority 10 \
    --cmd "bash scripts/local/run_arm.sh ${arm} 0 100 p3"
done
```
`run_arm.sh ARM SEED [STEPS=100] [VARIANT=p3]` trains one arm, then for each of
steps {30,50,100} that exists it merges the FSDP shards to bf16
(`verl.model_merger`), prunes the raw shards once the merge is verified, runs
the held-out eval (`scripts/countdown_gs0_eval.py`), and uploads the merged
checkpoint via `hf_upload.py`. Preview the exact command without running
anything: `bash scripts/local/run_arm.sh B 0 100 p3 --dry-run`.

## 3. Check status
```bash
python scripts/local/gpu_queue.py status   # pending/running/done/failed + GPU memory
tail -f $WORK/logs/cd7_B_p3_s0.log         # per-lineage training/merge/eval log
tail -f $WORK/logs/queue_cd7_B_p3_s0.log   # queue wrapper's own stdout/stderr
```

## 4. What gets uploaded where
Only **merged bf16 checkpoints** (`$WORK/merged/<LINEAGE>/step_N/`) go to HF —
raw FSDP shards under `$WORK/checkpoints/` never leave the box. Destination:
`iamseungpil/metacot-countdown-local` on the Hub, at
`runs/<LINEAGE>/step_<N>/` (kept forever) and `runs/<LINEAGE>/latest/`
(overwritten each time; `hf_upload.py` never deletes anything outside that one
`latest/` prefix). No `HF_TOKEN` in `.env` -> upload step logs
`[hf] no HF_TOKEN, skipping upload` and exits 0; the job is not failed.

## 5. Disk hygiene
```bash
python scripts/local/disk_guard.py prune          # dry-run: shows what it would delete
python scripts/local/disk_guard.py prune --apply  # actually delete
```
Only touches `global_step_N/` dirs that are (a) not among the 2 newest for
their lineage AND (b) already merged (`merged/<lineage>/step_N/config.json`
exists). The queue's own disk guard (in `gpu_queue.py worker`) is separate and
just pauses new job starts, logging `[DISK] waiting`.

## 6. Stopping safely
```bash
touch /hdd_data/seungpil/queue/STOP        # pause: workers finish current job, then idle
python scripts/local/gpu_queue.py stop-workers   # SIGTERM every worker process (after STOP,
                                                  # or any time — training jobs are NOT
                                                  # auto-killed, only the worker loop is)
rm /hdd_data/seungpil/queue/STOP           # resume
```
Never run `ray stop --force` from any of these scripts — GPUs 0-3 may host
other users' Ray instances too; each `run_arm.sh` job gets its own
`RAY_TMPDIR`/Ray instance instead.
