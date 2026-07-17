# LOCAL_RUN — 로컬 4×80GB 서버에서 RQ3 학습 재현 가이드 (2026-07-17)

동료가 클러스터(amlt) 없이 자기 서버에서 B2/B3pkg 학습을 재현하기 위한
가이드다. 결론부터: **repo만으로 ~90%는 간다** — 풀 pip 레시피와 verl
패치는 `scripts/bootstrap_sdc_node.sh`에, 정확한 학습 커맨드는 런처 yaml에
전부 있다. 다만 아래 **갭 3개는 현재 턴키가 아니다**: ① 전 config/스크립트가
`/scratch/...` 경로 하드코딩, ② 로컬 런처 스크립트가 없음(실행 시퀀스가
amlt yaml `command:` 블록 안에만 존재), ③ conda-pack
tarball(`env_snapshots/simplerl_v4.tar.gz`)의 빌드 레시피가 repo에 미문서.
갭은 숨기지 않고 각 절에 "현재 불가/수동필요"로 표기한다.

**팀원 연결 주소 (0717 확인: 전부 PUBLIC — 다운로드에 토큰·권한 불필요)**:

| 용도 | 주소 |
|---|---|
| 데이터 parquet·SFT init 모델·conda-pack env | https://huggingface.co/datasets/iamseungpil/metacot |
| RL 체크포인트(B0 gs300·B2 gs140 등, resume/평가용) | https://huggingface.co/iamseungpil/metacot-h200-triobj-dcpo-v3 |
| RV 데이터 계보(학생 롤아웃 덤프 — 재생성 시에만) | https://huggingface.co/datasets/iamseungpil/metacot-rv |
| 코드 | https://github.com/iamseungpil/metacognition-math |

## 0. 전제조건

- **GPU**: 80GB급 × 4 (config 전체가 tensor_parallel=4, n_gpus_per_node=4,
  gpu_memory_utilization=0.45, logprob micro_bs=2 등 4×80GB 전제로 튜닝됨.
  다른 GPU 수는 다수 지점 동시 수정 필요 — 권장하지 않음).
- **토큰**: 다운로드에는 불필요(위 repo 전부 PUBLIC — 0717 확인).
  `HF_TOKEN`은 **업로드(체크포인트 push)할 때만** 자기 계정 것으로 준비.
  `WANDB_API_KEY`(선택 — 없으면 아래처럼 `trainer.logger=['console']`
  오버라이드하되, 커스텀 dcpo/* 메타 지표는 wandb 직접호출이라 콘솔엔 안
  남는다는 점 유의). 토큰 값은 `.env`에만 두고 문서·코드에 절대 기록 금지.
- **코드**: git checkout이면 충분. (클러스터의 GitHub release tarball /
  HF 코드싱크 흐름은 로컬에선 불필요.)

## 1. 환경 구축

두 경로가 있다. 어느 쪽이든 마지막에 **`python scripts/patch_math_verify.py`
수동 실행 필수** — math_verify의 SIGALRM이 Ray 워커 스레드에서 no-op이 돼
정답을 False로 채점하거나 hang하는 버그의 패치인데, 어떤 런처/bootstrap도
자동 호출하지 않는다(conda-pack에는 구운 것으로 추정 — **갭**).

**경로 A (빠름, 토큰 필요)**: HF에서 conda-pack 다운로드.
`hf_hub_download(repo_id="iamseungpil/metacot", repo_type="dataset",
filename="env_snapshots/simplerl_v4.tar.gz")` → 원하는 디렉토리에 untar →
`conda-unpack`. Xet-backed라 plain curl은 403 — hf_hub_download를 써야 한다.
**갭**: 이 tarball의 빌드 스크립트는 repo에 없다. 재현 불가 시 경로 B.

**경로 B (느림, repo만으로 재현)**: `scripts/bootstrap_sdc_node.sh`
113–169행이 사실상의 레시피다. 요지: conda python=3.10 →
torch==2.7.1(cu126) → transformers==4.57.6, omegaconf==2.3.0, math-verify
[antlr4_9_3]==0.6.0, trl==0.19.1, ray[cgraph]==2.43.0 등 → **vllm==0.10.2**
(이때 torch가 2.8.0+cu128로 올라감 — 이게 실제 런타임 torch, 정상) →
verl==0.7.1 `--no-deps` → flash-attn==2.8.3(`--no-build-isolation
--no-deps` + site-packages target) → huggingface_hub<1.0. 이후
`pip uninstall opencv-python-headless hf_xet`(cv2 segfault·Xet 404 함정),
`PYTHONNOUSERSITE=1`.
**갭**: 경로 B는 requirements.txt와 드리프트(ray 2.43.0 vs 2.54.1 등) —
실제 rq3 런이 쓴 조합은 conda-pack 내용물이라 repo에서 검증 불가.

**verl 소스 패치 2건**(bootstrap 246–313행이 텍스트 패치로 idempotent 적용;
conda-pack에는 구워져 있다고 주장됨): ① `vllm_async_server.py` 호환 가드
일체, ② `verl/protocol.py`의 union_numpy_dict `_deep_equal` assert 우회
(rollout.n=8 GRPO 그룹에 필수). 경로 B로 직접 깔았다면
`SCRATCH=<작업디렉토리> SIMPLERL_DIR=<env경로> bash
scripts/bootstrap_sdc_node.sh`를 한 번 돌려 패치를 적용하는 것이 안전하다
(repo를 `$SCRATCH/metacognition`에 미리 두면 code-sync 단계가 no-op).

**런타임 자동 패치**: repo 루트의 `sitecustomize.py`가 agent-loop
postprocess를 패치한다 — `PYTHONPATH`가 repo 루트를 가리키기만 하면 작동.

## 2. 데이터 / init 모델 스테이징

작업 루트를 `$WORK`라 하자(클러스터의 `/scratch` 대체).
`mkdir -p $WORK/{logs,checkpoints,models,data}`.

**parquet**: 두 경로가 있다 (rq3 arm은 meta_mix 2개만 씀).
- **재생성(권장 — 다운로드 불필요, 0717 codex 실행검증)**: 입력 2개
  (`data/v8_meta_inside_think.parquet`, `data/v8_base_matched_clean.parquet`)는
  git에 추적돼 clone에 포함된다.
  `python -m src.training.verl_gdpo_data --mode meta_mix` 실행 →
  train 5,344행 / val 594행이 나오는지 확인(현행 완제품과 행 단위 일치 검증됨).
- **다운로드**: PUBLIC dataset `iamseungpil/metacot`의
  `data/verl_train_meta_mix.parquet`·`data/verl_val_meta_mix.parquet`를
  `hf_hub_download`로 `$WORK/data`에 받는다.
  (`scripts/pull_parquets.py`는 `/scratch` 하드코딩이라 로컬에선 비권장 — **갭**.)

**init 모델** (B2/B3pkg 공통 = meta-SFT):

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="iamseungpil/metacot", repo_type="dataset",
    allow_patterns=["models/b23_rv_unmasked_sft/**"], local_dir="<tmp>")
# → $WORK/models/b23_rv_unmasked_sft 로 복사 후,
# tokenizer_config.json에서 "extra_special_tokens" 키 제거 (yaml 인라인 로직과 동일)
```

B0은 `models/b0_gold_sft`로 동일 절차.

## 3. 학습 커맨드 (런처 CLI 오버라이드 전사)

repo 루트에서, `export PYTHONPATH=<repo루트> LOCAL_RANK=0 VLLM_USE_V1=1`
(+`HF_TOKEN`; 디버깅엔 `PYTHONFAULTHANDLER=1 DCPO_DEBUG=1`).
`/scratch` → `$WORK` 치환은 아래처럼 `data.train_files`/`val_files`/
`default_local_dir`/`model.path` 하이드라 오버라이드로 해결한다(configs
수정 불필요·금지). **갭 1건**: `src/training/verl_sdc.py`의 faulthandler
로그 경로 `/scratch/logs`(49–53행)는 오버라이드 불가 — `/scratch`가 없는
서버면 쓰기 가능한 `/scratch/logs`를 만들어 두거나(sudo mkdir + chown)
해당 기능 실패를 감수한다(학습 자체는 진행).

**B2** (`h100std_rq3_b2.yaml` 171–190행 전사 + 로컬 경로 치환):

```bash
cd <repo루트> && python -u -m src.training.verl_sdc \
  --config-name=base_matched_grpo_h100_4x4k \
  trainer.experiment_name=rq3_b2 \
  trainer.default_local_dir=$WORK/checkpoints/rq3_b2 \
  trainer.project_name=metacot-dcpo-v4 \
  data.train_files=$WORK/data/verl_train_meta_mix.parquet \
  data.val_files=$WORK/data/verl_val_meta_mix.parquet \
  actor_rollout_ref.model.path=$WORK/models/b23_rv_unmasked_sft \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.top_p=1.0 \
  data.max_response_length=8192 \
  actor_rollout_ref.rollout.max_model_len=10240 \
  actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
  ++algorithm.norm_adv_by_std_in_grpo=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  trainer.resume_mode=auto trainer.save_freq=10 trainer.test_freq=50 \
  ++trainer.val_before_train=False \
  "trainer.logger=['console']" \
  ++hydra.searchpath=[pkg://verl/trainer/config]
```

(`data.train_files`/`val_files` 오버라이드 2줄과 `trainer.logger`만 원본
런처 대비 추가 — 나머지는 yaml의 오버라이드 그대로다. WANDB를 쓰려면
logger 줄을 빼고 `WANDB_API_KEY`를 export.)

**B3pkg**: 위와 동일하되 다음만 교체 —
`--config-name=triobj_dcpo_v4_stage3b_h100_4x4k`,
`trainer.experiment_name=rq3_b3pkg`,
`trainer.default_local_dir=$WORK/checkpoints/rq3_b3pkg`,
`trainer.save_freq=5`, 그리고 추가 2줄
`++algorithm.dcpo_rmeta_source=pmi_shift ++algorithm.dcpo_w_over=0.0`.
env에 `DCPO_META_CLOSE_FORCE=1 DCPO_META_CLOSE_N=96` 추가. 나머지 triobj
head 가중치(w_meta .8 + 80-step warmup / w_format .35 / w_emit .1 /
w_cal .3 / len .08 / trunc_open .3)는 config 기본값에서 온다.
⚠️ config yaml의 `dcpo_rmeta_source: cf_group` 기본값을 CLI가 뒤집는
구조임에 주의(CODE_MAP.md §2 함정).

**체크포인트/재개**: verl이
`$WORK/checkpoints/<arm>/global_step_N/actor/{model,optim,extra_state}_world_size_4_rank_R.pt`
(각 4샤드) + `latest_checkpointed_iteration.txt`를 쓴다.
`trainer.resume_mode=auto`가 로컬 디렉토리에서 재개하므로 선점 없는 로컬
서버에선 **클러스터의 HF 릴레이 일체(push_ckpts_to_hf.py /
pull_resume_ckpt.py / RGS 가드)를 통째로 생략**해도 된다.

## 4. Eval (held-out 1030)

1. **머지**: `global_step_300/actor`를 `python -m verl.model_merger merge
   --backend fsdp --local_dir .../actor --target_dir <merged>`로 합치고
   tokenizer_config.json의 `extra_special_tokens` 키 제거.
   (**갭**: verl 0.7.1과 model_merger CLI 호환은 repo 내 미검증 주장.)
2. **실행**: 벤치마크는 전부 공개 HF dataset(openai/gsm8k,
   HuggingFaceH4/MATH-500, HuggingFaceH4/aime_2024) — 토큰 갭 없음.

```bash
python scripts/eval_vllm_1030.py \
  --model_path <merged> --model_name rq3_<arm>_gs300_1030 \
  --output_dir results/eval_1030_rq3_<arm>_gs300_16k \
  --benchmarks gsm8k math500 aime2024 --max_problems 500 \
  --max_tokens 16384 --max_model_len 20480 \
  --temperature 0.7 --top_p 0.95 --num_samples 8 --tp_size 4 --seed 42
```

3. **재채점 필수**: 런타임 `is_correct`는 잠정치 — 최종 수치는
   `experiments/analysis/aggregate_tables.py`로 재채점(구 채점기는 math500
   gold의 ~21%를 잘못 깎는다).
4. **갭**: `experiments/configs/science/eval_1030.yaml`의 `models:` 블록은
   pre-rq3 arm을 가리킴 — rq3 ckpt로 돌리려면 그 블록을 수동 수정하거나
   위처럼 스크립트를 직접 호출한다.

## 5. 갭 요약표

| # | 갭 | 심각도 | 우회로 |
|---|---|---|---|
| 1 | ~~private HF 토큰~~ **해소(0717)**: repo 전부 PUBLIC 확인 | 해소 | 상단 "팀원 연결 주소" 표의 URL로 직접 다운로드(토큰 불필요) |
| 2 | 로컬 런처 부재(시퀀스가 amlt yaml 안에만) | BLOCKER | 본 문서 §2–3의 전사 커맨드 사용 |
| 3 | `/scratch` 하드코딩(configs·pull_parquets·faulthandler) | MAJOR | 하이드라 오버라이드 + 수동 다운로드; faulthandler 경로만 우회 불가 |
| 4 | conda-pack 빌드 레시피 미문서·requirements 드리프트 | MAJOR | 경로 A(다운로드) 또는 경로 B(bootstrap 전사) + 패치 수동 적용 |
| 5 | patch_math_verify.py 자동 호출 없음 | MAJOR | 설치 후 반드시 수동 실행 |
| 6 | 4×80GB 전제 하드와이어 | MINOR | 동일 스펙 서버 사용 권장 |
| 7 | eval_1030.yaml models 블록 stale / model_merger 미검증 | MINOR | 스크립트 직접 호출·수동 확인 |
