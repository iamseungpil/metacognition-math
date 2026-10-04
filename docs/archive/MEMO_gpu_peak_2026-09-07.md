# GPU 피크 조사 (0907) — Countdown 6-arm 단일-GPU GRPO, 73~76GB 원인

읽기 전용 조사. verl 0.7.1 소스(`/hdd_data/seungpil/envs/simplerl/.../verl`)와
`configs/countdown_6arm.yaml` + `scripts/local/run_arm.sh` 오버라이드만 근거.
GPU 사용·설정 변경 없음.

## 핵심 발견 — 가정이 틀렸다

과제 메모의 "bf16 4B 파라미터 8GB" 가정이 틀렸다. `fsdp_workers.py:386-389`:
```python
torch_dtype = fsdp_config.get("model_dtype", None)
if torch_dtype is None:
    torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
```
`model_dtype`를 우리 yaml/오버라이드 어디서도 안 준다 → **actor는 HF 로드 시점부터
fp32**. FSDP `MixedPrecision(param_dtype=bf16, reduce_dtype=fp32, ...)`
(`fsdp_workers.py:568-573`, `fsdp_config.dtype` 기본 "bfloat16")는 **forward 연산만**
bf16로 캐스팅하고, 마스터 파라미터·그래디언트·옵티마이저 상태는 fp32로 상주한다.
즉 4B 모델의 상주 메모리는:
- 파라미터(fp32) 16GB
- 그래디언트(fp32, reduce_dtype 강제) 16GB
- AdamW exp_avg(fp32) 16GB + exp_avg_sq(fp32) 16GB

**= 64GB**, 이게 `update_actor` 동안 GPU에 동시에 올라간다. 여기에 forward용 bf16
캐스트 사본(~8GB, 순간적)과 vocab logits/entropy 버퍼(1시퀀스 3072×151936×4B ≈
1.9GB, log_probs/entropy 중복 보관 포함 수GB)를 더하면 73~76GB와 정확히 맞는다.
"메타인지 게이트"류 판단이 아니라 순수 산수이므로 이 결론은 소스만으로 확정적이다.

## 질문별 답

**1) update_actor 중 params+grad+Adam 전부 GPU?**
그렇다. `fsdp_workers.py:1004` `load_fsdp_optimizer(...)`가 미니배치 루프 **진입 전**
옵티마이저 상태 전체를 GPU로 올리고, `:1038` `offload_fsdp_optimizer(...)`가 **모든
micro-batch·모든 mini-batch(ppo_epochs 포함) 끝난 뒤** 딱 한 번 내린다. "필요한
순간에만 조금씩"이 아니라 "통째로 올렸다 통째로 내림" 방식 — `optimizer_offload`는
스텝 사이(호출 간)에만 절약하지, 스텝 도중 피크는 전혀 못 줄인다.
`grad_offload`는 **FSDP 경로에 존재하지 않는다** — `grep -rn grad_offload`로 전체
verl 트리를 훑으면 `megatron_workers.py`/`engine/megatron/*`에만 있다
(`workers/config/engine.py:86`은 Megatron `EngineConfig`의 필드). FSDP
`FSDPEngineConfig`엔 grad_offload 필드 자체가 없으므로 우리 yaml에 넣어도 조용히
무시되거나(dataclass가 strict면 에러) 아무 효과가 없다.

**2) ref 워커 = GPU 상주 두 번째 전체 모델?**
그렇다. `init_model`에서 rollout(vLLM) 다음 순서로 `_build_model_optimizer(...,
role="ref")`가 별도 FSDP 인스턴스를 만든다(`fsdp_workers.py:948`). ref는
`bfloat16`(actor와 달리 `self._is_actor`가 False라 기본 bf16, ~8GB)이고,
`ref.fsdp_config.param_offload` 기본값 False(`FSDPEngineConfig.param_offload:
bool = False`, 우리 yaml 미설정) → **항상 GPU 상주**, offload/load 왕복이 없다.
`need_reference_policy` (`trainer/ppo/utils.py:72-76`)는
`config.algorithm.use_kl_in_reward or actor.use_kl_loss` — PMI/OSD 코드 경로와
무관하게 `use_kl_loss=True`만으로 매 스텝 ref forward가 강제된다는 launcher 주석은
소스와 일치한다.
**⚠️ REF_OFFLOAD=1 오버라이드(이미 배선됐지만 기본 꺼짐) 리스크**: `param_offload=true`로
바꿔도 절약되는 건 **init 시점의 8GB뿐일 수 있다** — `compute_ref_log_prob`
(`fsdp_workers.py:1152` 이하)에도, `ray_trainer.py:_compute_ref_log_prob`
(`:1105-1128`)에도 `load_fsdp_model_to_gpu`/`offload_fsdp_model_to_cpu` 래핑이
**없다**(actor 쪽 `compute_log_prob`엔 `:1099-1100`에 있음 — 대조됨). 즉 매 스텝
ref forward 직전에 GPU로 되불러오는 코드가 안 보인다. 이 프로젝트 코드
(`src/training/verl_sdc.py`)에도 `compute_ref_log_prob`를 몽키패치하는 곳은
없었다(grep 확인). **소스만으로는 REF_OFFLOAD=1이 8GB를 안전하게 절약하는지,
아니면 CPU 상주 shard로 forward를 걸어 죽거나 조용히 느려지는지 판정 불가** —
반드시 짧은 스모크로 실측 검증 필요.

**3) 로짓/엔트로피 메모리, micro-batch=1의 의미**
`dp_actor.py:update_policy` (`:548, :565`)에서 `mini_batch.split(ppo_mini_batch_size)`
후 `mini_batch.split(ppo_micro_batch_size_per_gpu)` — **split 대상은 "시퀀스" 단위
row**(rollout.n=8로 이미 펼쳐진 프롬프트×롤아웃 결과), **prompt 그룹이 아니다**.
`ppo_micro_batch_size_per_gpu: 1` = 진짜 시퀀스 1개씩 forward. full-vocab fp32
logits(`dp_actor.py:369` `logits.div_(temperature)` 이전에 `(bsz, resp_len, vocab)`
캐스트 없음 — fp32 그대로) 한 장이 1×3072×151936×4B ≈ 1.87GB, 여기서 log_probs
gather + entropy(`entropy_from_logits`) 계산에 로짓 텐서가 최소 한 번 더 파생되므로
순간적으로 수GB 추가.
`use_dynamic_bsz`(:75), `ppo_max_token_len_per_gpu`(actor config에 존재,
`use_dynamic_bsz=False`라 현재 안 쓰임), `entropy_from_logits_with_chunking`(:81-84,
config에 `false`로 명시 존재), `entropy_checkpointing`(:126, 274-277, 372-375,
`torch.utils.checkpoint.checkpoint`로 감싸는 실제 코드 존재), `use_fused_kernels`
(:68, config에 `false`로 존재) — **다섯 개 다 verl 0.7.1에 실재하고 우리 yaml에
이미 키가 있다**. 지금은 전부 최소-메모리 방향(micro=1, chunking/checkpointing
off인데 이건 켜는 쪽이 절약, use_fused_kernels off인데 이건 커널에 따라 다름)으로
설정돼 있어 여기서 짤 여지는 `entropy_checkpointing=true` 정도(수백MB~1-2GB, 시퀀스
1개뿐이라 재계산 비용도 작음)뿐이다.

**4) 재개(load_checkpoint) 피크**
`fsdp_checkpoint_manager.py:load_checkpoint` (:102-178)는 `torch.load(...,
weights_only=False)`로 optimizer per-rank 파일을 읽고 `self.optimizer.load_state_dict(...)`
호출 — **`map_location` 지정이 없다**. 저장 시(`save_checkpoint`,
`ShardedOptimStateDictConfig(offload_to_cpu=True)`)는 CPU 텐서로 직렬화되므로
`torch.load` 자체는 CPU에 복원되지만, PyTorch `Optimizer.load_state_dict`는 로드된
state 텐서를 **해당 파라미터가 있는 디바이스로 캐스트**한다. 그런데 워커 레벨
`load_checkpoint`(`fsdp_workers.py:1225-1236`)는 `checkpoint_manager.load_checkpoint`를
부르기 **전에 이미 `load_fsdp_model_to_gpu(self.actor_module_fsdp)`를 호출**해
모델을 GPU에 올려둔다 — 그래서 옵티마이저 상태도 GPU로 캐스트되며 로드된다. 결과:
파라미터(fp32, 16GB, 이미 GPU) + 방금 복원된 옵티마이저 전체(fp32, 32GB, GPU)가
동시에 존재 — 그 뒤에야 `offload_fsdp_model_to_cpu`/`offload_fsdp_optimizer`
(`:1234, :1236`)가 내린다. **verl 0.7.1엔 "옵티마이저를 CPU에 로드"로 강제하는
Hydra 플래그가 없다** — 이건 코드에 하드코딩된 순서 문제라 오버라이드로 못 고친다.
고치려면 (a) 두 콜 순서를 바꾸거나(모델 GPU 로드 전에 optimizer.load_state_dict를
CPU에서 먼저 실행) (b) `torch.load(..., map_location="cpu")`를 명시하는 소스 패치가
필요 — `run_arm.sh`가 이미 하고 있는 sitecustomize 몽키패치 패턴과 같은 종류의
작업이다. 이 조사에선 코드 변경을 하지 않았다.

**5) 초기화 중 `_sync_params_and_buffers` 피크**
`init_model` 순서: actor 빌드+즉시 offload(:913-919, 이 시점 optimizer.state는
비어 있어 `offload_fsdp_optimizer`는 사실상 no-op — Adam state는 lazy init이라
첫 `.step()`까지 텐서가 없다) → `_build_rollout`(vLLM 엔진, hybrid engine이라 actor
weight를 per-tensor로 동기화해야 함) → ref 모델 빌드(GPU 상주, 위 참고).
`_sync_params_and_buffers`라는 심볼 자체는 `fsdp_workers.py`에 없다 — PyTorch FSDP
내부(`torch.distributed.fsdp._runtime_utils`) 함수로, hybrid engine이 vLLM에 가중치를
동기화하려고 actor FSDP를 다시 GPU로 올려 **unshard(전체 비샤딩 사본)**할 때 트리거
되는 것으로 보인다. 이 시점 ref 모델은 아직 빌드 전(코드 순서상 ref는 rollout 다음)
이므로 이 피크엔 ref가 안 걸린다. 대신 vLLM 엔진 자체가 이미 gpu_memory_utilization
비율만큼 예약해 둔 상태에서, actor unshard 사본(fp32 16GB 파라미터 전체를 한 랭크에
재구성)이 겹친다 — `VLLM_UTIL` 0.30→0.15로 피크가 안 바뀐 것과 정합적이다(vLLM
쪽 KV캐시 예약 크기가 원인이 아니라, actor unshard가 원인이므로 vLLM 비율을
바꿔도 무관). **단, 이 지점의 정확한 텐서 구성은 소스만으로 확정 불가** — 그
순간의 `torch.cuda.memory_snapshot()`이 있어야 vLLM weight buffer / actor unshard
사본 / ref(빌드 전이면 0) 각각의 기여를 나눌 수 있다.

## 확정 못 한 것 (소스만으로는 판정 불가)
- REF_OFFLOAD=1이 실제로 동작하는지(질문 2의 ⚠️) — 코드상 재로드 경로가 안 보임.
- 초기화 피크의 정확한 텐서 구성비(질문 5) — 실측 메모리 스냅샷 필요.
- `optimizer_impl=bitsandbytes.optim` 스타일 8-bit AdamW는 `build_optimizer`
  (`workers/config/optimizer.py:173-198`)가 **제네릭하게 지원**하지만
  `bitsandbytes`/`torchao` 둘 다 이 env(`/hdd_data/seungpil/envs/simplerl`)에
  **미설치**(`ModuleNotFoundError` 확인) — 설치 없이는 그림의 떡.

## 오버라이드 제안 (순수 Hydra 키만, 우선순위 순)

| # | 오버라이드 | 근거(코드 위치) | 예상 절감 | 리스크 |
|---|---|---|---|---|
| 1 | `actor_rollout_ref.actor.entropy_checkpointing=true` | `dp_actor.py:274-277,372-375` 실제 checkpoint 래핑 존재 | 수백MB~1-2GB | 낮음. 시퀀스 1개라 재계산 비용 작음 |
| 2 | `actor_rollout_ref.ref.fsdp_config.param_offload=true` (=REF_OFFLOAD=1, 이미 배선됨) | `fsdp_workers.py:930-968` ref는 두 번째 상주 모델 | ~8GB *(init만, 스텝 중 절감은 미검증 — 위 ⚠️)* | **중간~높음 — 반드시 짧은 스모크로 먼저 검증.** 재로드 경로가 안 보여 CPU에서 forward가 걸리거나 죽을 수 있음 |
| 3 | 근본 원인(actor fp32 params+grad+Adam = 64GB)을 줄이는 오버라이드는 **없다** — `model_dtype`/`optimizer`/`optimizer_impl`은 전부 실재하는 키지만: `model_dtype=bfloat16`은 학습 수치(정밀도)를 바꿔 arm 간 byte-parity가 깨짐(CLAUDE.md의 재현성 원칙과 상충), `optimizer_impl=bitsandbytes.optim`은 패키지 미설치. 둘 다 "Hydra 오버라이드 한 줄"이 아니라 **별도 검증이 필요한 실험**이다. | `fsdp_workers.py:386-389`, `workers/config/optimizer.py:173-198` | 16~30GB급(수치 바꾸는 옵션이라 "절감"이 아니라 "다른 실험") | 높음 — 학습 동역학 변경 |
| 4 | 체크포인트 재개 피크(질문4) | `fsdp_checkpoint_manager.py:load_checkpoint` 하드코딩 순서 | — | 오버라이드로 불가능. 소스 패치 필요 |

**결론**: #1+#2(검증 통과 시)를 합쳐도 최대 ~9-10GB 절감 — 73GB → 63-64GB. **<60GB
목표에 필요한 13-16GB를 순수 Hydra 오버라이드만으로 안전하게 채울 방법은 verl
0.7.1 FSDP1 경로에 없다.** 진짜 병목은 "옵티마이저를 fp32 AdamW로, 모델을 fp32로
올려두는 구조" 자체이며, 이건 정밀도(모델 dtype) 또는 옵티마이저 종류(8-bit/torchao)
를 바꾸는 실험이지 메모리 튜닝이 아니다. 다음 스텝 후보: (a) REF_OFFLOAD=1 스모크
1회로 안전성 확인, (b) bitsandbytes 설치 후 `AdamW8bit` 별도 arm으로 스모크(수치
영향 있으므로 기존 arm과 나란히 비교), (c) 체크포인트 재개 피크만 별도로 잡고
싶다면 `fsdp_checkpoint_manager.py` 소스 패치(`map_location="cpu"` 강제) 검토.
