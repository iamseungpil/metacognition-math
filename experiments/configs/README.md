# experiments/configs — 과학 설정과 인프라 설정의 분리

이 디렉토리는 논문 실험(base vs meta, RQ1–RQ4)의 재현성을 위해 "실험이 무엇인가"(science)와 "어디서 어떻게 도는가"(infra)를 분리해 둔 얇은 설정 계층입니다. 실제 하이퍼파라미터의 원본은 여전히 `configs/`의 진짜 학습 설정 파일들이며, science yaml은 그중 과학적으로 불변이어야 하는 필드(데이터 parquet 경로, 보상 모드, 총 스텝 300, save/val 주기, seed, init 모델)만 못박아 두고 `derives_from`으로 원본을 가리킵니다. 기존 학습 코드와 설정은 단 한 줄도 수정하지 않는 additive 계층입니다.

## science/ — 실험 정의 (arm별 하나씩)

- `sft1_meta.yaml` / `sft1_base.yaml`: SFT-1 쌍둥이 (v8_meta_inside_strict vs v8_base_matched_strict, lr 2e-6 3ep 동일).
- `sft2_rv.yaml` / `sft2_base_rv.yaml`: SFT-2 쌍둥이 (rv_functional vs 메타 제거 v8_base_rv, lr 1e-5 3ep 동일).
- `rl_pmishift.yaml`: 메타 arm RL — TRIOBJ_DCPO_V4 + `++algorithm.dcpo_rmeta_source=pmi_shift` (T1 처치).
- `rl_vanilla_base.yaml`: base arm RL — VANILLA_GRPO correctness-only (T1 대조).
- `rl_gandhi_arm.yaml`: 메타 SFT 초기화 + VANILLA_GRPO (RQ2/T2, SFT 프라이밍 vs RL 보상 분해).
- `rl_ref0_nosft.yaml`: 순정 Qwen3-8B + VANILLA_GRPO (T1 참조행 REF-0 — SFT 자체 비용이 T1 표 안에서 읽히게 하는 no-SFT 기준선).
- `eval_1030.yaml`: 최종 판정 프로토콜 — held-out 1030문제, 16k 토큰, avg@8(AIME avg@16), temp 0.7, math_verify 채점, 두 arm을 같은 잡·같은 seed로.

주의할 지표 규칙: 학습 중 정확도는 wandb의 `val-aux/<ds>/correctness/mean@1`만 봅니다(`val-core/reward`는 메타 성형 합성값이라 가짜 격차가 생깁니다). 최종 판정은 반드시 1030 held-out을 math_verify로 채점하고, 난이도 층화 분석(Simpson 함정)을 함께 보고합니다.

## infra/ — 실행 환경 정의

- `msr_h100x4.yaml`: 현재 MSR 클러스터 현실(H100x4, 6시간 선점 윈도우, HF 모델 리포를 통한 체크포인트 릴레이). 학습 잡 제출은 리포 루트의 amlt yaml(`h100std_*.yaml`)로 하고, 이 파일은 노드 위에서 run.sh를 쓸 때의 경로/인터프리터를 기술합니다.
- `generic_8gpu.yaml`: 협업자 서버용 예시(선점 없음, 로컬 실행). 경로와 python 인터프리터만 자기 환경에 맞게 복사·수정하면 됩니다.
- `local_debug.yaml`: 1-GPU 스모크. `smoke: true`가 처리량 관련 값만 축소하며, 스모크 결과는 절대 표에 넣지 않습니다.

## launch/run.sh — 사용법

```bash
cd experiments/launch
./run.sh ../configs/science/eval_1030.yaml   ../configs/infra/msr_h100x4.yaml
./run.sh ../configs/science/sft1_meta.yaml   ../configs/infra/generic_8gpu.yaml
./run.sh ../configs/science/rl_pmishift.yaml ../configs/infra/msr_h100x4.yaml --seed 43
```

eval 모드는 `scripts/eval_vllm_1030.py`를 모델×패스별로 실제 실행하고, sft 모드는 science overrides를 원본 설정에 병합해 `generated/`에 쓴 뒤 `accelerate launch src/training/sft.py --config ...`로 실제 실행합니다. rl 모드는 병합된 verl 커맨드 전체를 출력만 합니다 — verl 환경은 `scripts/bootstrap_sdc_node.sh`로 준비해야 하고, MSR에서는 선점 릴레이가 포함된 amlt yaml로 제출하는 것이 정본이기 때문입니다. `--seed`는 T1 프로토콜에 내장된 시드 ×3 스윕용이며 RL에서는 `++actor_rollout_ref.actor.data_loader_seed`로 전달됩니다.

토큰(HF_TOKEN, WANDB_API_KEY, GH_TOKEN)은 오직 `.env`/잡 환경변수로만 주입하고, 어떤 설정 파일에도 적지 않습니다.
