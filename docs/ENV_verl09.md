# ENV verl09 — Qwen3.5 학습용 격리 환경 (cd9-verl09, 2026-09-14)

> **상태(2026-09-14)**: Qwen3.5 학습 트랙 **보류**. Ray 시작 직후 정지가 두 번 재현됐다(스모크 1차·
> 2차 모두 GPU 0%, 필터/로더 워커 문제로 추정 — `docs/RESULTS_cd9.md` 2026-09-14 08:30/09:40 절).
> 정책은 Qwen3-4B-Instruct-2507 로 대체(사전등록 amendment 1). 이 env 는 이후 재개용으로 보존한다.

`/hdd_data/seungpil/envs/verl09` — verl 0.9.0 + vllm 0.20.2 + transformers 5.x. 목적은 하나:
**Qwen3.5-4B(`model_type=qwen3_5`, 하이브리드 gated-delta-net + full attention)를 우리 트레이너
`src/training/verl_sdc.py`로 학습**하는 것. 공유 학습 env `simplerl`(verl 0.7.1 / vllm 0.10.2 /
transformers 4.57.6)은 qwen3_5 를 모르므로 건드리지 않고 따로 만들었다. 추론 전용 env
`/hdd_data/seungpil/envs/qwen35`(vllm 0.29)와도 별개다.

## 1. 구축 방법 (재현)

```bash
# simplerl 의 python 3.10.20 으로 venv 를 판다 (qwen35 env 와 같은 방식; simplerl 은 읽기만 한다).
/hdd_data/seungpil/envs/simplerl/bin/python -m venv /hdd_data/seungpil/envs/verl09
export TMPDIR=/hdd_data/seungpil/tmp PIP_CACHE_DIR=/hdd_data/seungpil/tmp/pipcache   # 루트 디스크 금지
/hdd_data/seungpil/envs/verl09/bin/python -m pip install --upgrade pip
/hdd_data/seungpil/envs/verl09/bin/python -m pip install --no-cache-dir \
    "vllm==0.20.2" "verl==0.9.0" "transformers>=5.5.3,<5.11" \
    math-verify datasets pytest flash-linear-attention
```

- `uv` 없음, 시스템 python 에 pip 없음 → 위 방식이 이 박스에서 유일하게 단순한 길이다.
- **`causal-conv1d` 는 설치하지 않는다.** PyPI 에 휠이 없어 소스 빌드로 들어가며, 빌드 격리
  env 에 별도 torch(cu13) 를 통째로 받고 CUDA 컴파일을 시작한다(한 번 시도 후 중단). verl 0.9 의
  `verl/models/transformers/qwen3_5.py` 가 `causal_conv1d_fn is None` 이면
  `_packed_causal_conv1d_fallback`(순수 torch) 로 처리하므로 학습에 필수가 아니다.
- `flash-attn` 은 없다(설치하지 않는다 — vllm 0.20 은 `flash_attn` 패키지가 보이면 `flash_attn.ops.triton.rotary` 까지
  요구해 심(shim) 패키지로는 죽는다, 0921 실측). verl 0.9 는 sdpa·`use_remove_padding: False` 여도
  `workers/utils/padding.py` 가 `flash_attn.bert_padding` 을 import 하므로, **verl 쪽만** 고쳤다:
  `verl/utils/attention_utils.py` 의 import 를 try/except 로 감싸고 순수 torch 사본
  `verl/utils/_bert_padding_torch.py`(BSD-3, flash-attn 2.8.3 의 bert_padding.py)로 폴백한다.

## 2. 설치된 핵심 버전 (`pip freeze` 전체 244 행은 `/hdd_data/seungpil/tmp/probe/verl09_freeze.txt`)

| 패키지 | 버전 | 비고 |
|---|---|---|
| python | 3.10.20 | simplerl 의 conda python (venv base) |
| torch / torchvision / torchaudio | 2.11.0+cu130 / 0.26.0 / 2.11.0 | vllm 0.20.2 가 `torch==2.11.0` 을 핀; PyPI 기본 휠은 CUDA 13.0 (nvidia-*-cu13). 드라이버 580.105.08 = CUDA 13.0 지원 확인 |
| vllm | 0.20.2 | 레지스트리에 `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`, `Qwen3NextForCausalLM` 있음. **`Qwen3_5ForCausalLM` 은 없다** — 체크포인트 아키텍처가 ConditionalGeneration(VL) 이라 그것을 쓴다 |
| verl | 0.9.0 | `transformers!=5.6.0,<5.11,>=5.5.3`, `tensordict<=0.10.0` 핀 |
| transformers | 5.10.4 | `AutoConfig` → `Qwen3_5Config`(text_config 32층, layer_types = full_attention + linear_attention) |
| tensordict | 0.10.0 | |
| ray | 2.58.0 | |
| flash-linear-attention / fla-core | 0.5.2 | Qwen3.5 gated-delta-net Triton 커널 (필수) |
| flashinfer-python / -cubin | 0.6.8.post1 | vllm 의존 |
| triton | 3.6.0 | |
| numpy | 2.2.6 | |
| datasets / pyarrow | 5.0.1 / 25.0.1 | |
| math-verify | 0.9.0 | |
| hydra-core / omegaconf | 1.3.6 / 2.3.1 | |
| peft / accelerate | 0.20.0 / 1.15.0 | |
| xgrammar | 0.2.3 | |

## 3. 검증한 것 (CPU 전용, GPU 6 에서 `torch.cuda.is_available()` 만)

- `torch.version.cuda == 13.0`, `torch.cuda.is_available() == True`.
- `AutoConfig.from_pretrained(/hdd_data/seungpil/scratch/models/Qwen3.5-4B)` OK.
- `python -m src.training.verl_sdc --help` 가 import·hydra 까지 통과(`_VERL_VERSION=(0,9,0)`).
- `configs/math_meta_verl09.yaml` 을 hydra 로 compose → `migrate_legacy_reward_impl` →
  `HFModelConfig` 변환 → `hf_tokenizer/hf_processor` → `create_rl_dataset`(math_plain parquet 7,295행)
  → `collate_fn` 까지 CPU 로 통과. 자세한 스키마 감사는 그 yaml 헤더.
- 회귀: simplerl(0.7.1) 에서 `pytest tests src/training/tests` = **13 failed / 1464 passed**
  (기존 13 건 그대로, 신규 실패 0). verl09 env 에서의 결과는 아래 «미검증» 참조.

## 4. 포팅에서 깨진 것과 수리 (`src/training/verl_sdc.py`, 전부 verl 버전 게이트 — 0.7.1 경로 바이트 동일)

| 깨진 곳 | 0.9 에서의 증상 | 수리 |
|---|---|---|
| `from verl.workers.fsdp_workers import CriticWorker` | 모듈 자체가 삭제됨 (`ImportError`) | `except ImportError` → `verl.workers.engine_workers.TrainingWorker`(verl 자신의 `main_ppo_v0.add_critic_worker` 와 동일). critic.enable=False 라 실제 생성은 안 됨 |
| `from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler` | `verl.trainer.ppo.utils` 로 이동 | `except ImportError` 폴백 |
| `RayWorkerGroup.update_weights(self, global_steps)` 몽키패치 | 0.9 의 `CheckpointEngineManager` 가 `update_weights(global_steps=…, mode=…)` 로 호출 → `TypeError` | 래퍼가 `**kwargs` 를 전달. (0.9 는 워커 메서드가 `@register` 되어 인스턴스 바인딩이 이 클래스 속성을 가리므로 사실상 비활성) |
| `_CompatAsyncLLM` 심(vllm<0.8 폴백) | 0.9 의 `vllm_async_server` 가 `inspect.signature(AsyncLLM.from_vllm_config)` 로 kwargs 를 필터 → `(*args, **kwargs)` 심이 **모든 kwargs 를 지움** | `_VERL_GE_0_8` 이면 심 설치를 건너뜀 |
| `hf_processor(Qwen3.5-4B)` → `Qwen3VLProcessor` 의 `chat_template=None` | verl `RLHFDataset` 이 processor 가 있으면 `processor.apply_chat_template` 를 쓰므로 **모든 행이 overlong 필터에서 예외 → 데이터셋 길이 0** | `main_task` 에서 tokenizer 의 chat_template 을 processor 로 복사(verl 0.9 `HFModelConfig` 가 agent-loop 워커 쪽에 하는 것과 동일). 텍스트 모델(processor None)엔 no-op |
| `scripts/local/run_math_arm.sh` `+data.apply_chat_template_kwargs.enable_thinking=false` | 새 yaml 이 그 키를 이미 가지므로 hydra `+` 가 «already at» 으로 실패 | `++` 로 변경(키 유무 모두 허용) |

그 외 확인만 하고 손대지 않은 것: `compute_advantage` 시그니처 동일, `_compute_reward_colocate`
호출 경로 동일(`use_rm` 뒤집기 트릭 유효), `no_padding_2_padding` 은 `losses` 모듈 네임스페이스에
여전히 있음, `ppo_loss` 시그니처 동일, `migrate_legacy_reward_impl`·`collate_fn`·`copy_to_local`·
`hf_tokenizer` 동일.

## 5. 미검증 / 위험 (GPU 잡을 돌리지 않았다)

1. **FSDP + 하이브리드 gated-delta-net 층.** verl 0.9 는 `Qwen3_5DecoderLayer`/`GatedDeltaNet`
   forward 를 몽키패치하고(`verl/models/transformers/qwen3_5.py`), FSDP wrap 정책은
   `_no_split_modules` 의 `Qwen3_5TextDecoderLayer`(추상 이름) 를 건너뛰도록 되어 있다
   (`fsdp_utils.py`). 우리 yaml 의 `wrap_policy.min_num_params: 0` 경로가 이 모델에서
   어떻게 감싸는지는 실행해 봐야 안다. fla 커널의 backward 가 gradient checkpointing +
   activation offload 와 겹칠 때의 안정성도 미검증.
2. **가중치 동기화(FSDP → vLLM).** 아키텍처가 `Qwen3_5ForConditionalGeneration` 이라
   FSDP 도 vLLM 도 **비전 타워를 함께 적재**한다(텍스트 데이터라 픽셀은 절대 안 들어감).
   naive checkpoint-engine 의 이름 매핑(`model.language_model.*`, `model.visual.*`)이
   맞는지는 첫 `update_weights` 에서 판가름난다. 메모리: 4B 텍스트 + ~0.4B 비전을 두 번
   (FSDP bf16 + vLLM) 얹고 ref 워커까지 한 H100 에 올린다 — `VLLM_UTIL=0.35` 로 시작하되
   OOM 이면 `ACTOR_OFFLOAD=1 REF_OFFLOAD=1`.
3. **thinking 비활성.** `enable_thinking=false` 가 tokenizer 템플릿에서 빈 `<think>\n\n</think>`
   블록을 렌더하는 것은 확인했지만, vLLM 롤아웃이 agent-loop 의 processor 템플릿(동일 문자열)
   으로 같은 프롬프트를 만드는지는 로그의 첫 롤아웃 텍스트로 확인해야 한다.
4. **미설치 커널.** `causal-conv1d` 없음(torch 폴백 — 느릴 수 있음), `flash-attn` 없음
   (sdpa 로 충분). `use_remove_padding=True` 로 바꾸려면 둘 다 필요해질 수 있다.
5. **legacy `reward_model` 블록.** `migrate_legacy_reward_impl` 가 0.7.1·0.9 모두에서 예외를 내고
   verl_sdc 가 삼킨다(기존 동작). `reward:` 블록이 이미 새 레이아웃이라 무해하지만, 0.9 가
   이 함수를 더 엄격히 요구하게 되면 top-level `reward_model` 블록을 지우는 게 맞다.
6. **회귀 테스트는 두 env 모두 13 failed / 1464 passed** (같은 13 건 — 기존 ModuleNotFound·
   spec-table 불일치, 신규 0). CPU 테스트라 GPU 경로(FSDP·vLLM·weight sync)는 덮지 않는다.
7. **NFS 위의 datasets 캐시.** overlong 필터(`num_proc=31`) 정리 단계에서
   `OSError: [Errno 16] Device or resource busy: '.nfs…'` 경고가 찍힌다(HF_HOME 이 /hdd_data NFS).
   결과는 정상(64/64)이지만 실제 잡에서 같은 경고가 나면 무시해도 된다; 반복되면
   `data.filter_overlong_prompts_workers` 를 낮춘다.
8. `torch 2.11+cu130` 은 이 박스에서 처음 쓰는 CUDA 13 스택이다. 드라이버는 지원하지만
   NCCL(2.28, cu13)·Triton 3.6 커널 컴파일은 첫 잡에서 확인된다. Triton 캐시는 env.sh 가
   `/hdd_data/seungpil/scratch/triton_cache` 로 돌린다.

## 6. 쓰는 법

`scripts/local/run_math_arm.sh` 가 `VERL_ENV` 를 받는다(미설정이면 simplerl 그대로):

```bash
cd /home/ubuntu/seungpil/metacognition-math && git checkout cd9-verl09
python scripts/local/gpu_queue.py submit --name cd9v09_smoke_M_G0_s0 --gpus 1 \
  --cmd "cd /home/ubuntu/seungpil/metacognition-math && \
         VERL_ENV=/hdd_data/seungpil/envs/verl09 CONFIG_NAME=math_meta_verl09 \
         MODEL_PATH=/hdd_data/seungpil/scratch/models/Qwen3.5-4B \
         bash scripts/local/run_math_arm.sh M_G0 0 5"
```

5 스텝 스모크(`STEPS=5`, `save_freq=5` 라 `global_step_5` 가 남고, 판정 스텝 30/50/100 은
없으므로 머지·평가 단계는 건너뛴다). 성공 판정: 로그에 `[SDC] processor.chat_template was
empty — synced`, `[MATH]` 보상 라인, `update_weights` 타이밍이 5 번 찍히고 rc 0.

## 0928 verl 0.9 패치 — vLLM 가중치 전송 소켓에 잡 꼬리표(동시 학습 충돌 방지)
- 증상(0927 22:35): 두 학습 잡이 각자 Ray 를 띄워 job id 가 둘 다 `01000000` → 같은 `/tmp/rl-colocate-zmq-01000000-replica-0-rank-0.sock` → 한 잡이 다른 잡의 IPC 핸들을 받아 `TypeError: 'str' object is not callable` 로 정지.
- 패치(원본 백업 `/hdd_data/seungpil/tmp/verl09_vllm_rollout.py.orig`, `verl09_utils.py.orig`):
  `verl/workers/rollout/vllm_rollout/vllm_rollout.py`(송신) 와 `utils.py`(수신) 의 job_id 뒤에 `os.environ.get("MC_ZMQ_TAG", "")`.
- `mc/trainer.py:_ray_env` 가 `MC_ZMQ_TAG = -mc<드라이버 pid>` 를 모든 Ray 워커에 넘김(vLLM 워커 프로세스는 상속) → 잡마다 다른 소켓. 꼬리표가 없으면 옛 동작 그대로.
