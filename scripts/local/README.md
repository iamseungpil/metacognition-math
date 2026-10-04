# scripts/local — single-machine RL pipeline (GPUs 0-3)

## cd9 math (현행)

현재 도는 무대는 수학(M_RETRY). 아래 §0~6 은 Countdown(prior round, cd7)용 스크립트로 남겨두되
지금 발사는 이 절만 쓴다.

**1. parquet 빌드** (level 5, forced mixed):
```bash
python scripts/local/build_math_parquet.py --level "Level 5" \
  --forced_frac 0.25 --forced_from_rollouts /hdd_data/seungpil/.../pass_rates.json
```

**2. M_RETRY 제출** (GPU 2·3만, `gpu_queue.py` 큐 사용):
```bash
python scripts/local/gpu_queue.py submit --name cd9_M_RETRY_s0 --gpus 1 \
  --cmd "cd /home/ubuntu/seungpil/metacognition-math && \
    DATA_TRAIN=\$WORK/data/math_train_math_opt_fmix0.25.parquet \
    DATA_VAL=\$WORK/data/math_val_math_opt.parquet \
    RESP_LEN=6144 \
    bash scripts/local/run_math_arm.sh M_RETRY 0 100"
```
**2b. M_RETRY_SL 제출** (M_RETRY + 결정 토큰 자기지도 CE — step-1 에 redirect=0.000 이라
RL 만으로는 희소 행동을 못 찾는다. 라벨은 그 행 자신의 first_correct, 교사 없음):
```bash
python scripts/local/gpu_queue.py submit --name cd9_M_RETRY_SL_s2 --gpus 1 \
  --cmd "cd /home/ubuntu/seungpil/metacognition-math && \
    DATA_TRAIN=/hdd_data/seungpil/scratch/data/math_train_math_retry_fmix0.25.parquet \
    DATA_VAL=/hdd_data/seungpil/scratch/data/math_val_math_retry.parquet \
    bash scripts/local/run_math_arm.sh M_RETRY_SL 2 30"
```
`MATH_SL_W`(기본 0.5)·`MATH_SL_WARMUP_STEPS`(기본 10, 그 구간엔 2배)로 세기를 조절한다.
중단 규칙 `redirect_rate<0.02` 는 이 팔만 min_step 15(SL 이 일할 시간을 준다).

**2c. M_CRIT 제출** (비평 정보이득 — 문맥 안 재시도를 **빼고** 비평만 남긴 팔. cd9 정박 진단:
같은 문맥 이어쓰기의 구제율은 0.000 이었다. 비평의 값은 «얼어붙은 채점기로 잰 정보 이득 −
남의 비평으로 잰 정보 이득»으로 채점한다):
```bash
python scripts/local/gpu_queue.py submit --name cd9_M_CRIT_s2 --gpus 1 \
  --cmd "\
    DATA_TRAIN=/hdd_data/seungpil/scratch/data/math_train_math_crit.parquet \
    DATA_VAL=/hdd_data/seungpil/scratch/data/math_val_math_crit.parquet \
    bash scripts/local/run_math_arm.sh M_CRIT 2 30"
```
`MATH_CRIT_W`(기본 0.5)·`MATH_CRIT_SCALE`(기본 0.05 nats/tok)·`MATH_CRIT_MAX_TOK`(기본 2048)·
`MATH_CRIT_SCORER_PATH`(얼어붙은 채점기; 미설정이면 actor init = `MODEL_PATH`)로 조절한다.
★채점기(4B bf16 ≈ 8 GB)가 vLLM 과 같은 GPU 에 올라가므로 런처가 `VLLM_UTIL` 기본을 0.3 으로
낮춘다. 사후 평가는 `math_critique_eval.py`(→ `$WORK/eval/<lineage>/step_<N>/math500_crit_8k/`):
첫 답 정확도·누출율·IG 와 **답 없는 재풀이**(비평 K=4 vs blind K=4)의 `rescue_delta` 를 낸다.

**2d. M_DIS 제출** (불일치 진단 — 한 풀이 안의 메타는 네 관문 전부에서 무력했다(own ≤ donor).
문제 안에서 살아 있는 유일한 신호는 **자기 샘플들의 불일치**(문제별 AUC .754)이므로, 프롬프트가
이 정책 자신의 후보 4개를 보여 주고 `<meta>` 한 블록에 진단 + `Commit: n` 을 쓰게 한다. 메타
크레딧의 라벨은 gold 가 아니라 **그 그룹 형제들의 다수답**이다 — TTRL 식 자기 증류):

먼저 학습 문제에 대한 후보 롤아웃 → parquet:
```bash
# (i) TRAIN 문제 롤아웃(8 샘플·8192 토큰). 후보는 plain 정책으로 뽑는다(--variant math_opt).
python scripts/local/gpu_queue.py submit --name cd9_DIS_cand_train --gpus 1 \
  --cmd "cd /home/ubuntu/seungpil/metacognition-math && \
    python scripts/local/math_rollout.py \
      --dataset parquet:/hdd_data/seungpil/scratch/data/math_train_math_crit.parquet \
      --model_path /hdd_data/seungpil/scratch/models/Qwen3-4B-Instruct-2507 \
      --variant math_opt --num_samples 8 --max_tokens 8192 --seed 11 \
      --out_dir /hdd_data/seungpil/scratch/eval/dis_cand_train"
# (ii) parquet 빌드(앞 4개를 후보로, 200행 val)
python scripts/local/build_math_dis_parquet.py \
  --rollouts /hdd_data/seungpil/scratch/eval/dis_cand_train/texts.jsonl \
  --train_parquet /hdd_data/seungpil/scratch/data/math_train_math_crit.parquet
```
그 다음 팔 제출(대조군 `M_DIS_RAND`/`M_DIS0` 도 같은 데이터·같은 발사 규약):
```bash
python scripts/local/gpu_queue.py submit --name cd9_M_DIS_s2 --gpus 1 \
  --cmd "cd /home/ubuntu/seungpil/metacognition-math && \
    bash scripts/local/run_math_arm.sh M_DIS 2 30"
```
`MATH_DIS_W`(기본 0.5)로 크레딧 세기를 조절한다(M_DIS0 은 팔 명세가 0 을 강제 — env 무시).
RESP_LEN 기본 **2048**(진단 + 답만), MAX_PROMPT 기본 **2048**(후보 4개가 프롬프트에 들어간다),
LINEAGE 는 `cd9_M_DIS_s2_r2048`. 사후 평가는 `math_dis_eval.py`
(→ `$WORK/eval/<lineage>/step_<N>/math500_dis_8k/`): 후보 4+1 개를 **다시 뽑아** 진단 N=4 를
돌리고 `acc_dis` / `acc_majority4` / `acc_majority5`(대등 비용) / `pass_at_4` / `minority_rescue`
와 문제당 토큰을 같이 낸다. 게이트는
`gate_judgment.py --eval-subdir math500_dis_8k --acc-key acc_dis`.

`DATA_TRAIN`/`DATA_VAL` 는 build_math_parquet.py 출력 경로로 지정(env 재지정, 기본은
`$WORK/data/math_{train,val}_<variant>.parquet`). `RESP_LEN` 재시도 팔은 미설정이면 자동 6144
(두 번 풀 예산). 워커는 **GPU 2·3만** — `gpu_queue.py start-workers 2 3`.

**3. 평가 위치**: `math_retry_eval.py` 가 판정 스텝마다 `$WORK/eval/<lineage>/step_<N>/math500_retry/`
에 첫 답/판단/재시도 분해 지표를 쓴다(`math_rollout.py` 는 이 분해가 없어 M_RETRY 판정 불가).

> **0922 아카이브 주의** — 위 2a~2d 절이 부르는 사후 평가 스크립트
> (`math_rollout.py` · `math_revision_eval.py` · `math_retry_eval.py` · `math_critique_eval.py` ·
> `math_dis_eval.py` · `math_diff_eval.py` · `math_sites.py` · `build_math_*.py`)는 전부
> `scripts/local/archive_closed_0922/` 로 옮겨졌다. 현행 단일 통과 12k 평가는 **`python -m mc.eval
> --protocol single`**(구현 `mc/eval.py` + `mc/rollout.py`)이 대신하며, 라이브 큐 잡도 이쪽을 쓴다.
> `run_math_arm.sh` 의 판정-스텝 eval 호출은 이 이름들을 그대로 참조하지만 실패가 치명적이지
> 않다(`|| echo ... eval FAILED`) — 학습은 그대로 끝난다. 되살리려면 아카이브에서 꺼내 오라.
>
> `math_cited_site_gate.py` 도 아카이브됐다. 유일하게 살아 있던 쓰임인 `bootstrap_ci` 는
> `math_pmi_shift_probe.py` 안으로 산술 그대로 옮겼다(시험 `tests/test_pmi_shift_probe.py` 통과).

자세한 팔 정의·중단 규칙은 `src/training/math_meta.py`, 설계 근거는
`docs/PREREGISTRATION_cd9_math_judgment.md` + `docs/RESULTS_cd9.md`.

## Countdown (prior round, cd7) — 아래는 이전 라운드 참고용

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

## Qwen3.5 학습 — 격리 env `verl09` (cd9-verl09)

simplerl(verl 0.7.1)은 `qwen3_5` 를 모른다. Qwen3.5-4B 학습은 `/hdd_data/seungpil/envs/verl09`
(verl 0.9.0 / vllm 0.20.2 / transformers 5.10.4 / torch 2.11+cu130 — 구축·검증·위험은
`docs/ENV_verl09.md`) 에서 돌린다. `run_math_arm.sh` 는 `VERL_ENV` 를 받으면 그 venv 의 bin 을
PATH 맨 앞에 붙인다(학습·`verl.model_merger`·`math_rollout.py` 전부 같은 env). yaml 은
`configs/math_meta_verl09.yaml`(0.9 스키마 감사 결과는 그 헤더).

```bash
python scripts/local/gpu_queue.py submit --name cd9v09_smoke_M_G0_s0 --gpus 1 \
  --cmd "cd /home/ubuntu/seungpil/metacognition-math && VERL_ENV=/hdd_data/seungpil/envs/verl09 CONFIG_NAME=math_meta_verl09 MODEL_PATH=/hdd_data/seungpil/scratch/models/Qwen3.5-4B bash scripts/local/run_math_arm.sh M_G0 0 5"
```
