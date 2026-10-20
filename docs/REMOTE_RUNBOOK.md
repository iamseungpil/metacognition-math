# 원격 서버 실행 안내 (4×A100-40GB)

이 문서는 프로젝트를 처음 보는 사람이 원격 서버에서 실험을 돌리고, 결과를 Hugging Face(HF)로 돌려보내기 위한 것이다.
모든 명령은 복사해서 쓸 수 있게 적었다. `ROOT` 는 원격 서버의 작업 폴더(디스크가 넉넉한 곳)이다.

## 1. 목적 (5줄)

1. 정책 모델이 자기 풀이를 «첫 `\boxed` 뒤에 되짚어 틀린 답을 고치는» 버릇을 강화 학습(GRPO)으로 키우는 방법을 시험한다. 정확도가 목표이고 버릇은 수단이다.
2. 비교 팔: `grpokl`(앞부분 + KL 끈) · `pfx`(앞부분만, 끈·교사 없음) · `reset`(앞부분 + 눈 가린 자기 자신 증류). 앞부분 = 모델이 첫 `\boxed` 까지 스스로 푼 풀이.
3. **R1 (평가 전용)**: 이 서버(H100)에서 이미 학습한 4B 모델(Qwen3-4B-Instruct-2507 기반) 7개를 새 시험 AMC23 · Minerva Math 에서 잰다. GPU 1장 × 4개 병렬.
4. **R2 (두 번째 모델)**: Qwen3-1.7B 로 같은 팔 3개 × 씨앗 3·5·7 을 학습한다(GPU 4장 FSDP). 앞부분은 모델마다 다시 만들어야 하므로 먼저 준비 단계(추론만)를 돈다.
5. 4B 학습(약 70 GB/장 필요)은 원격에서 하지 않는다. 결과는 HF 비공개 저장소로만 돌려보낸다. wandb 는 쓰지 않는다.

## 2. 준비 (Setup)

NVIDIA 드라이버가 580 이상이어야 한다(`nvidia-smi` 로 확인; torch 2.11+cu130). 아니면 §9 의 «미확인» 을 보라.

```bash
export ROOT=/data/seungpil            # 원격 서버의 큰 디스크 위 폴더 (아래 전부 이 값을 쓴다)
mkdir -p $ROOT && cd $ROOT
git clone -b cd9-reset-distill-1004 https://github.com/iamseungpil/metacognition-math.git   # 또는 알려준 태그: git clone --branch <태그> ...
cd metacognition-math && export REPO=$PWD

# 1) 환경 (verl09: python 3.10 · torch 2.11+cu130 · vllm 0.20.2 · ray 2.58 · verl 0.9.0 + 로컬 수정 3곳)
bash scripts/remote/env/setup_env.sh              # 끝에 "verl 로컬 수정 OK" 가 나와야 한다

# 2) HF 토큰 — 파일에만 둔다(커밋 금지). 둘 중 하나:
echo 'HF_TOKEN=hf_xxx' > $REPO/.env               # .env 는 gitignore 대상. 출력·로그에 찍지 말 것
# export HF_TOKEN=hf_xxx

# 3) 데이터 + 기본 모델 + 시험 데이터셋 캐시
bash scripts/remote/get_data.sh                    # → $ROOT/scratch/{data,eval,models,hf_home}
ls $ROOT/scratch/data | wc -l                      # 16 (dapo_src_0..11, math_val_plain, hdev_all, eval_extra_*)
```

받는 것(전부 합쳐 데이터 약 9 MB + 모델 약 10 GB):

| 무엇 | 어디서 | 어디로 |
|---|---|---|
| DAPO 문제 묶음 12개, val/hdev/eval_extra parquet, 탐침 선택 json | HF 비공개 `iamseungpil/metacot-cd9-data` | `$ROOT/scratch/data`, `$ROOT/scratch/eval` |
| Qwen3-4B-Instruct-2507, Qwen3-1.7B | HF `Qwen/…` | `$ROOT/scratch/models/` |
| AIME24/25, HMMT25, MATH-500, AMC23, Minerva 시험 | HF 데이터셋 | `$ROOT/scratch/hf_home` (평가는 오프라인으로 읽는다) |
| 4B 학습 모델(R1 용, 폴더당 8 GB) | HF 비공개 `iamseungpil/metacot-cd9-models` | R1 잡이 알아서 받고 평가 뒤 지운다 |

## 3. 스모크 시험 (SMOKE)

**3-1. 평가 스모크 (지금 바로, GPU 1장)** — 엔진·데이터셋·채점이 도는지와 40 GB 메모리를 본다.

```bash
export CUDA_VISIBLE_DEVICES=0
BENCHES="amc23:hf:math-ai/amc23:test:2" bash scripts/remote/a79_rlrt_eval.sh smoke=$ROOT/scratch/models/Qwen3-4B-Instruct-2507
cat $ROOT/scratch/eval/rl_smoke_amc23/summary.json | head -20     # acc 가 있으면 성공 (채점은 몇 분 늦게 생긴다)
rm -rf $ROOT/scratch/eval/rl_smoke_amc23
# 메모리가 모자라면:  MC_GPU_UTIL=0.70 MC_MAX_NUM_SEQS=64  를 앞에 붙인다 (분포는 안 바뀐다)
```

**3-2. R2 학습 스모크 (§4-1 의 앞부분 준비가 끝난 뒤, GPU 4장, 2스텝)** — **R2 는 이 시험을 통과하기 전에 제출하지 않는다.** 40 GB 에서 메모리가 맞는지 아직 검증되지 않았다.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 MODEL=$ROOT/scratch/models/Qwen3-1.7B PTAG=dapo17 EVTAG=q17_ \
  SMOKE=1 ARM=reset SEED=3 bash scripts/remote/a78c_rlrt_train.sh      # 2스텝 · 평가 없음 · 계보 꼬리표 _smk
# 학습 중 다른 창에서: watch -n 5 nvidia-smi --query-gpu=index,memory.used --format=csv
```

통과 = 2스텝이 끝나고 `[a78] smoke reset 2스텝 끝` 이 나옴. OOM 이면 순서대로 시도(레시피를 바꾸므로 **결과표에 표시**): `VLLM_UTIL=0.3` → `TB_OVERRIDE=16` → `RL_PFX=4096`. 어느 값으로 통과했는지 기록해 두라.
스모크 산출물은 지운다: `rm -rf $ROOT/scratch/checkpoints/*_smk $ROOT/scratch/logs/*_smk.log`.

## 4. 큐 (Queue)

잡 목록은 `scripts/remote/remote_jobs.tsv`(20개). 그룹 세 개를 순서대로 낸다. 큐는 GPU 하나에 워커 하나다.

```bash
cd $REPO
export QUEUE_DONE_HOOK=$REPO/scripts/remote/queue_done_hook.sh   # 잡이 끝나면 자동으로 HF 업로드 (§5)
export QUEUE_STATUS_GPUS=0,1,2,3
# ★ 허락된 GPU 번호만 쓴다. 남의 프로세스·GPU 는 건드리지 않는다(§6).
```

### 4-1. R2 준비 (Qwen3-1.7B 앞부분, 추론만) + R1 평가 — GPU 4장, 각 1장씩

```bash
python3 scripts/local/gpu_queue.py start-workers 0 1 2 3
bash scripts/remote/submit_jobs.sh r2prep      # q17_prep_g0..g3: 12개 묶음을 4그룹으로 병렬(그룹당 약 5시간). 마지막으로 끝난 그룹이 관문(gate)을 자동 실행
bash scripts/remote/submit_jobs.sh r1          # r1_*: 7개 평가 (기준 모델 + 학습 모델 6개), 우선순위가 낮아 준비 잡 뒤에 선다
python3 scripts/local/gpu_queue.py status
```

준비가 끝나면 `$ROOT/scratch/data/mc_pfx_commit_dapo17_screen_plain_le6000.parquet` 와 `.summary.json` 이 생긴다.
`summary.json` 에서 `gate_pass` 를 확인하라(관문 = 정보 있는 틀린 첫 답 ≥ 100). 미달이면 gate 가 종료 코드 3 으로 끝나며, 탐색용으로 쓰려면
`FORCE=1 bash scripts/remote/q17_prep.sh gate` 로 다시 돌린다(결과표에 «탐색» 표시). 관문을 다시 돌리려면 `rm -r $ROOT/scratch/remote_meta/dapo17_gate.lock`.

### 4-2. R2 학습 (GPU 4장을 한 잡이 다 쓴다) — 스모크 통과 후에만

```bash
python3 scripts/local/gpu_queue.py stop-workers        # R1 이 다 끝난 뒤. 워커를 하나만 남겨 잡을 직렬로 돌린다
python3 scripts/local/gpu_queue.py start-workers 0     # 잡 명령이 CUDA_VISIBLE_DEVICES=0,1,2,3 을 스스로 건다
bash scripts/remote/submit_jobs.sh r2rl --smoke-ok     # q17_{grpokl,pfx,reset}_s{3,5,7} 9개, 직렬
```

## 5. 결과 돌려보내기

**권장: 큐 완료 훅** (§4 에서 `QUEUE_DONE_HOOK` 를 켰다면 자동). 잡이 `done` 이면 평가 결과를, `failed` 이면 로그만(`FAILED` 표시) 올린다.
수동으로 올리려면(잡 이름 = RUN_NAME):

```bash
bash scripts/remote/push_results.sh r1_base                 # 올리는 것: 평가 폴더마다 summary.json · meta.json · texts.jsonl.gz,
                                                            #   체크포인트의 RUN_ENV.txt · TRAIN_DONE · pfx_rollouts.jsonl.gz, 학습·큐 로그(gz), DONE
bash scripts/remote/push_results.sh q17_reset_s3 --weights  # 병합된 최종 모델(수 GB)까지 — 필요할 때만
```

훅 대신 반복문을 쓰려면: `nohup bash scripts/remote/push_loop.sh > $ROOT/scratch/logs/push_loop.log 2>&1 &` (10분마다 큐의 done/failed 잡을 한 번씩 올린다).
저장소 = HF 비공개 데이터셋 `iamseungpil/metacot-cd9-remote` 의 `runs/<RUN_NAME>/` (없으면 비공개로 자동 생성).
평가 채점은 백그라운드로 몇 분 늦게 끝나므로 push 가 최대 30분(`WAIT_SUMMARY_MIN`) 기다린다.

**이 서버(H100)에서 받기:**

```bash
cd /home/ubuntu/seungpil/metacognition-math
export HF_ENV_FILE=/home/ubuntu/seungpil/.env      # 토큰은 파일에서만 읽는다
bash scripts/remote/pull_results.sh --list         # 무엇이 올라왔나 (DONE 이 있어야 받는다)
bash scripts/remote/pull_results.sh                # 안 받은 DONE 런 전부
```

받은 평가 폴더는 `/hdd_data/seungpil/scratch/eval/<같은 이름>/` (`rl_<줄기>_<시험>`, texts.jsonl 은 압축을 푼다)로, 나머지는 `scratch/remote_runs/<RUN_NAME>/` 로 간다.
같은 이름의 평가 폴더가 이미 있으면 덮지 않고 경고만 낸다(`--force` 로 덮기). 이미 받은 런은 `.pulled` 표시로 건너뛴다.

## 6. 규칙

- **남의 프로세스·GPU 를 건드리지 않는다.** 쓰도록 허락된 GPU 번호만 `start-workers` 에 준다. `kill`·`pkill` 로 이름 매칭을 하지 말 것(잡은 큐의 `stop-workers` 로).
- **wandb 금지**(`WANDB_MODE=disabled` 가 이미 걸려 있다). git 은 사용자가 직접 한다 — 코드를 원격에서 고치지 말고, 고칠 일이 있으면 알려 달라.
- **디스크 85% 미만.** 큐가 `ROOT` 사용률 85% 를 넘으면 새 잡을 시작하지 않는다(`[DISK] waiting`). 시작 전 `df -h $ROOT`. 체크포인트는 최신 하나만 남기고(스크립트가 지운다), 끝난 R2 런의 `global_step_*` 는 push 뒤 지워도 된다. `/tmp`·홈에 큰 파일 금지.
- **돌고 있는 셸 스크립트를 제자리에서 편집하지 않는다.** bash 는 오프셋으로 읽기 때문에 도는 잡이 깨진다. 고칠 때는 새 파일을 만들어 `mv` 로 덮고, 잡을 내기 전에 고친다.
- 토큰을 출력·커밋하지 않는다. 같은 잡 이름을 두 번 내지 않는다(같은 계보 폴더를 두 프로세스가 쓰면 안 된다).
- 끊긴 학습은 같은 명령으로 다시 내면 `resume_mode=auto` 로 최신 체크포인트에서 이어진다.

## 7. 잡별 예상 시간

| 잡 | 개수 | 예상(시간/개) | 근거 |
|---|---|---|---|
| `q17_prep_g0..g3` (1.7B 앞부분 준비) | 4 | 약 5 | 이 서버 4B 실측: pool 묶음당 45–100분, prep 8–16분, screen 6–25분, 묶음 3개/그룹. 1.7B·A100 환산 — 추정 |
| `r1_base` | 1 | 약 2 | AMC23 40×16 + Minerva 272×16 샘플, 최대 38,912 말, A100 환산 — 추정 |
| `r1_<학습 모델 6개>` | 6 | 약 2.5 | 위 + 모델 8 GB 내려받기. 12k 응답 팔은 더 길 수 있음 |
| `q17_{grpokl,pfx,reset}_s{3,5,7}` (60스텝 학습 + 평가) | 9 | 약 10 (**미검증**) | 이 서버 4B 는 1 H100 에서 26–33시간(평가 포함). 1.7B·4×A100 은 스모크 전에는 모른다. 직렬 9개 ≈ 90시간 |

전체: 4 GPU 로 준비+R1 약 8–10시간, R2 는 스모크 결과에 따라 3–4일.

## 8. 결과 확인 (이 서버)

- 평가 요약: `/hdd_data/seungpil/scratch/eval/rl_<줄기>_<amc23|minerva|aime24|…>/summary.json` (`acc`, k 별 평균 정확도). 기존 분석 스크립트가 같은 이름으로 찾는다.
- R1 줄기 이름: `base`, `sdpo_r12288b32_s3`, `grpo_r12288b32_s3`, `ours_r6144b32_s3`, `ours_r6144b32_s5`, `reset_r6144b32_s5`, `grpokl_r6144b32_s5`.
- R2 줄기 이름: `q17_<팔>_r6144b32_s<씨앗>` (시험: aime24/aime25/hmmt25/math500/hdev/mathl5/mmlupro).
- 학습 로그·실행 설정: `scratch/remote_runs/<RUN_NAME>/{train_log.txt.gz, ckpt/RUN_ENV.txt}`.

## 9. 아직 확인 안 된 것

- **40 GB 에서 R2 학습 메모리**: 스모크 전에는 모른다. 특히 `N_GPUS=4` 다중 GPU 경로(verl FSDP + `mc` 의 얼린 참조 워커·에이전트 루프)는 이 서버에서 한 번도 돌려 본 적이 없다(여기선 항상 1장).
- **드라이버/CUDA**: 환경 핀은 CUDA 13(cu130, 드라이버 580+)이다. 원격이 더 낮은 드라이버면 torch/vllm 짝을 새로 정해야 한다. `pip install` 해상은 이 서버에서 시험하지 못했다(핀 244줄 + `verl09_local.patch` 는 순정 휠에 깨끗이 적용되는 것까지만 확인).
- **Qwen3-1.7B 앞부분 관문**: 정보 있는 틀린 첫 답 ≥ 100 이 1.7B 에서 되는지는 준비 후에야 안다.
- Minerva Math 정답 일부는 `np.arcsin(10/13)` 같은 수식 문자열이라 채점기가 맞는 답도 틀림으로 볼 수 있다(모델 간 비교용으로만 쓸 것).
- 이 서버에 있고 HF 모델 저장소에 **없는** step-60 모델(reset s3, grpokl s3, pfx, rlrt, rlsd)은 R1 에서 빠졌다.
