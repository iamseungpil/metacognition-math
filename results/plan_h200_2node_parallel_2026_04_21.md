# Plan — H200 × 4 × 2-node Parallel Experiment (2026-04-21, v2)

**Target**: base self-distill 검증 (Node A) + SDC-shared v7 개선안 (Node B)
**Compute**: msrresrchbasicvc H200 × 4 × 2 노드 (신청 상태 전부 queued)
**Scope**: `/home/v-seungplee/metacognition` mainline

---

## 0. 배경 — v6 실제 실패 모드

SDC-split v6 은 300 step 완주했으나 overall -20.30pp 회귀 (MATH500 42.6 / AIME 10). 표면적으로는 `\boxed{}` 문제처럼 보이지만 실제 기전은 다름:

- **Observation 1** (report §5): 모델은 첫 번째로 정답 `\boxed{5}` 를 맞게 산출 → 그 뒤 `\boxed{5\text{hour}}}` 같은 malformed 변형을 5-150회 반복 → `max_tokens=4096` 에서 cutoff → extraction 이 last malformed 을 잡아서 오답으로 채점.
- **Observation 2**: meta emission 99.1% 유지 (controller 살아있음).
- **Mechanism** (report §6): SDC post-meta repel 이 decoy-conditioned teacher logprobs 로부터 student 를 밀어내는데, **EOS 가 decoy context 에서 high-prob token** 이라 student 가 EOS 를 피하게 됨 → `\boxed{}` 산출 후에도 EOS 대신 새 `</think> + meta + boxed` 시퀀스 재시작.

따라서 개선은 (a) consensus 토큰 보존 (이미 sdc-shared v3 에서 도입), 플러스 **(b) EOS / repetition 방어**, **(c) λ_diff 축소** 가 본질.

---

## 1. 실험 A — 베이스 Self-Distill Ladder (Node A)

### Intent
RL reward loop 없이도 `strict paired SFT → question-only best-of-N → correctness-clean teacher selection → short SFT readout` 만으로 OOD AIME acc ≥ v8_meta_inside_strict_sft 수준을 복구할 수 있는지 검증.

### Baselines (고정 참조)
`v8_meta_inside_strict_sft` (4k 디코드):
- GSM8K 92.0% / MATH500 71.6% / AIME 13.3% / overall 79.81%
- meta_emission 99.94%
(16k 디코드): AIME 36.7% (token-exhaustion 해소 시).

### Hypothesis
- **H1a** (base naive): `base_qonly_naive` 는 v8_base_matched_strict_sft (75.51%) 대비 MATH500/GSM8K 에서 ±1.5pp 이내 유지, AIME 는 변화 없음 (meta 없는 경로).
- **H1b** (meta epistemic): `meta_qonly_epistemic` 는 meta_emission ≥ 97% 유지하면서 naive 대비 **AIME 동점 이상**, MATH500 loss 는 2pp 이내.
- **H1c** (meta kl, stretch): `meta_qonly_epistemic_meta_kl` (dense top-K + meta-only KL, coef 0.15) 은 meta span 에서 teacher 분포 근접 (KL loss 감소), overall acc 는 H1b 와 동등.
- **비교 목적**: H1b 가 성공하면 "RL 없이도 self-distill 로 meta controller + OOD 를 회복 가능" 이 입증됨.

### Verification
1. **Pre-flight**:
   - `data/v8_meta_inside_strict.parquet` (4264 rows) / `v8_base_matched_strict.parquet` validator pass
   - `scripts/run_self_distill_roundtrip.sh question_only_best_of_n <model> <input> <out> <mode> <claim_bearing>` 로 `results/self_distill/<mode>/online_sdpo_regen.parquet` 생성 (없으면)
   - Config checksum: `sft_self_distill_{base_qonly_naive,meta_qonly_epistemic,meta_qonly_epistemic_meta_kl}_h200_4gpu.yaml`
2. **Smoke** (3-step, <5분): `accelerate launch src/training/sft.py --config <cfg> --max_steps 3` 성공, checkpoint shard > 10MB
3. **HF push callback**: `src/training/sft.py` 에 `HfCheckpointCallback` 추가 (step % 100 == 0 에 `scripts/hf_checkpoint_sync.sh` 호출). 없으면 구현 필요.
4. **Full run** — per_device_batch 1, accum 8 → eff 8, 2 epoch, LR 1e-6, DeepSpeed ZeRO-3 no offload.
5. **Post-training eval** (`src/eval/eval_hf.py --bench 1030`):
   - GSM8K 500 / MATH500 500 / AIME 30 / max_tokens 4096 (primary) + 16k (secondary)
   - metrics: accuracy, `meta_emission_rate`, `no_boxed_rate`, avg response tokens
6. **Fail criteria**:
   - H1a: MATH500 acc drop > 3pp → base path issue
   - H1b: AIME `no_boxed_rate` > 15% OR acc drop > 3pp vs SFT Meta
   - H1c: KL loss trajectory 증가 → meta distillation 역효과

### Failure handling
- Smoke 실패 → config diff → 재smoke. 2회 실패 시 `iterative-code-review` skill 로 sft pipeline debug.
- Full run 중 val loss divergence → 직전 checkpoint 복구 + LR ½ 감소 재시작.

---

## 2. 실험 B — SDC-shared v7 (Node B)

### Intent
v6 실패 근인 (post-meta repel 이 EOS 를 피하게 만들어 repetition loop 발생) 을 직접 수정. 구조 변화 없이 **4개 knob 조정 + 1개 신규 reward head** 로 최소 침습 개선.

### 개선 사항 (v6 대비)
1. **`sdc_lambda_diff` 감소**: 0.30 → **0.15** (post-meta diff 토큰 repel 강도 절반). EOS/closure 토큰이 diff 로 잘못 분류되는 risk 축소.
2. **`sdc_lambda_diff` curriculum**: 0 → 0.15 linear over 100 step warmup. 초기에 broad repel 없이 base GDPO 로 안정화 후 점진 도입.
3. **`sdc_shared_tau` 완화**: 0.5 → **0.3** (더 엄격한 consensus gating). T+/T- logprob diff 가 0.3 nats 이내면 shared 로 취급 → 더 많은 토큰이 shared (repel 제외) 로 분류.
4. **`meta_commit_shape_reward` 가중치 상향**: 0.35 → **0.50** + **repetition penalty** 추가 (`\boxed{}` 중복 감지). 이 reward 는 "첫 번째 `\boxed{}` 후 추가 `\boxed{}` 를 감점".
5. (이미 sdc-shared 도입됨) consensus 토큰은 repel 제외 — v6 의 broad repel 문제 근본 수정.

### Hypothesis
- **H2a** (overall recovery): SDC-v7 은 v8_meta_inside_strict_sft (4k) 대비 **overall acc 차이 ±2pp 이내** 유지하면서 **AIME acc ≥ 15%** (v6: 10%).
- **H2b** (repetition 제거): repetition penalty + `λ_diff 0.15` 조합으로 **평균 response length 가 v6 3,130 tokens (MATH500) → ≤ 2,800 tokens**. repetition 으로 인한 `max_tokens` 도달률 50% → **≤ 15%**.
- **H2c** (mechanism check): `postmeta_shared_frac` 가 τ=0.3 로 v6 의 default(0.5) 대비 증가 → shared 토큰 비율 증가가 실제 관찰됨. shared 비율 increase 와 `\boxed{}` 중복률 감소 사이 상관 > 0.5.

### Verification
1. **Pre-flight**:
   - `tests/test_verl_sdc.py` 기존 테스트 + 신규: (a) `test_lambda_diff_curriculum` (step 0/50/100 에서 λ 값), (b) `test_repetition_penalty_reward` (2 `\boxed{}` 응답에 감점 확인)
   - `python -m py_compile src/training/verl_sdc.py src/training/verl_sdc_utils.py src/training/rewards.py`
   - `verl_sdc_e21r_shared_h200_4x4k.yaml` 수정 확인:
     ```yaml
     algorithm:
       gdpo_reward_keys: [correctness, outcome_calibration, meta_structure, meta_commit_shape, postmeta_closure, repetition_penalty]
       gdpo_reward_weights: [1.0, 0.7, 0.25, 0.50, 0.45, 0.25]
       sdc_lambda_diff: 0.15
       sdc_shared_tau: 0.3
       sdc_lambda_diff_curriculum: true
       sdc_lambda_diff_warmup_steps: 100
       sdc_lambda_diff_initial: 0.0
       sdc_lambda_diff_final: 0.15
     ```
2. **Smoke** (20 step, ≤ 15분):
   - batch 에 `sdc_teacher_pos_log_probs`, `sdc_postmeta_shared_mask`, `sdc_postmeta_diff_mask` 존재
   - `postmeta_shared_frac > 0.15` (tau 축소로 증가해야 함)
   - reward mean finite, repetition_penalty mean < 0 (감점이 발동)
   - step 20 에서 λ_diff_curriculum 이 0.03 근처 (20/100 * 0.15)
3. **Autoresearch 300-step loop** — halt triggers:
   - `repetition_rate` rolling mean (window 20) > 30% → halt (v6 재발)
   - reward mean < 0.3 연속 30 step → halt (발산)
   - `postmeta_shared_frac` rolling mean < 0.1 → halt (기전 미작동)
   - `wrap_rate` rolling mean < 0.35 → halt (controller 붕괴)
4. **HF push**: `push_ckpts_to_hf.py` daemon (600s 간격) + **every 20 step** step-trigger (신규 wrapper 추가)
5. **Post-training eval**: 1030 problems, max_tokens 4096 + 16k
   - **Success gate**: AIME acc ≥ 15% AND MATH500 truncation rate < 25% AND meta_emission ≥ 95% AND overall acc ≥ SFT meta - 2pp

### Failure handling
- Halt → autoresearch 가 config 조정 제안 (λ 축소, commit_shape 추가 boost) → smoke 재수행 → 통과 시 resume
- 3회 halt 시 전략 변경: `variant: n3` (attract only, no post-meta repel) 로 500 step control run (sdc-split v6 단계별 ablation)

---

## 3. 노드 할당 및 Wall-time

### 중요 제약: `results/self_distill/<mode>/online_sdpo_regen.parquet` **아직 미생성**
→ 실험 A 는 Phase A1 (generation, vLLM rollout best-of-N) + Phase A2 (SFT training) 순차 필요.

### Wall-time 추정 (H200 × 4)
| 단계 | 예상 시간 |
|---|---|
| Bootstrap (env install) | 5-15 분 |
| **SDC-v7 train** (300 step, v6 기반) | 2h 40m + retry buffer 30m ≈ 3h |
| **A Phase A1** — best-of-N rollout (N=8, 4264 problems) | 1.5-2h (vLLM async, 4 GPU) |
| **A Phase A2** — SFT training (2 epoch, 4264 × 2/8 ≈ 1066 step) | 1-1.5h |
| A Stage 1 (base_naive) total | ≈ 3-4h |
| A Stage 2 (meta_epistemic) total | ≈ 3-4h |
| A Stage 4 (meta_kl) — needs teacher top-K after Stage 2 | +1h top-K 생성 + 1.5h SFT |

### 노드 할당 시나리오
| 시나리오 | 우선순위 | 노드 배분 |
|---|---|---|
| 1개 노드 | **SDC-v7** 우선 (v6 직접 수정) | Node1: SDC-v7 (3h) |
| 2개 노드 | **B + A1 병행** | Node1: SDC-v7 (3h); Node2: A Stage 1 (3-4h) then Stage 2 (3-4h) |
| 2개 노드 + 시간 여유 | **A Stage 3 추가** | Node2 Stage 2 후 Stage 3 (scored, 3-4h) |
| 3개 이상 | **full ladder** | Node3: Stage 4 (meta_kl) 순차 |

### 우선순위 명확화
SDC-v7 는 v6 실패 직접 수정이므로 **최우선**. 노드 1개라면 B only. 2개 확보되면 Node2 에 A Stage 1 → 2 순차 실행해서 4개 SFT variant 중 3개 완료 목표.

---

## 4. 공통 인프라

### Bootstrap + keep-alive (이미 준비)
Running 전이 즉시 `scripts/run_sdc_on_h200_node.sh` (3-tmux):
- `gpukeeper` (torch matmul 4×H200, `gpu_keeper.py`) — idle-suspend 방지
- `boot` — bootstrap_sdc_node.sh, 5-15분
- `sdc_train` / `sft_train` — 실험별 런처, 10-retry loop

### HF push
- Node B (RL): `push_ckpts_to_hf.py` daemon 600s + step-trigger (신규 wrapper) — target `iamseungpil/metacot-sdc-v7`
- Node A (SFT): `HfCheckpointCallback` (step % 100) — targets:
  - `iamseungpil/metacot-sd-base-naive`
  - `iamseungpil/metacot-sd-meta-epistemic`
  - `iamseungpil/metacot-sd-meta-kl`
- Preempt 복구: `launch_sdc_verl.sh` 의 resume probe 가 HF 최신 `global_step_N` 자동 pull

### Monitoring
- Poll loop (PID 3131285 활성) + Monitor `bs6ip3lho` (Running 전이 감지)
- 25분 `/loop` fallback heartbeat — tmux 세션 + eval metrics 중간 리포트

---

## 5. Success Criteria

실험 A: H1a AND H1b 둘 다 pass → RQ3-D self-distill mainline 검증됨.
실험 B: H2a AND H2b pass → SDC-v7 이 v6 문제 해결됨.
(H1c, H2c 는 stretch — 실패해도 A/B 결론 유지)

두 실험 모두 성공 시 **"RL(SDC-v7) vs no-RL(self-distill) 둘 다 작동하며 controller 보존"** 이 입증 — v8 active plan §7 의 전체 mainline 이 정돈됨.

---

## 6. 리스크

1. **BSC idle-suspension 재발** — gpu_keeper.py 로 bootstrap 동안 점유 (완료).
2. **Bootstrap 설치 실패** — 2회 재시도, alternative wheel (HF CDN) 지원.
3. **vLLM 16k OOM** — 4x4k fallback auto-detect.
4. **Repetition penalty reward 가 false-positive** (정당한 2차 `\boxed{}` 에 감점) → smoke 에서 실제 SFT 응답 샘플로 검증 후 튜닝.
5. **Curriculum λ 구현 버그** → unit test `test_lambda_diff_curriculum` + smoke 에서 step 별 λ 로깅.
6. **HF repo 충돌** (여러 노드 같은 repo push) → 실험별 다른 repo 사용.

---

## 7. 실행 순서 (승인 후)

로컬 사전작업 (노드 대기 중):
1. `src/training/rewards.py` 에 `repetition_penalty_reward` 추가 — `\boxed{}` 중복 감지
2. `src/training/verl_sdc_utils.py` 에 `_get_lambda_diff_schedule(step, config)` 추가
3. `configs/verl_sdc_e21r_shared_h200_4x4k.yaml` algorithm 블록 수정 (위 § 2 참조)
4. `tests/test_verl_sdc.py` 에 curriculum/repetition_penalty 테스트 추가
5. `src/training/sft.py` 에 `HfCheckpointCallback` 추가
6. `scripts/hf_checkpoint_sync.sh` 호출 wrapper 확인
7. `iterative-code-review` skill 로 위 1-6 반복 (smoke → critic → fix), 오류 없을 때까지
8. HF code snapshot 재푸시 (39MB tarball)

노드 Running 후:
9. 자동 launch (poll loop + run_sdc_on_h200_node.sh)
10. autoresearch skill: halt trigger 반응 → config 자동 조정 → resume
11. 각 실험 종료 후 `src/eval/eval_hf.py --bench 1030 --max_tokens 4096` + `--max_tokens 16k`
12. 결과 정리 → `results/report_h200_2node_v7_2026_04_21.md` (intent / hypothesis / 실측 / pass/fail)

---

## 8. Open questions (approve 전 확인)

- **Q1**: SDC-v7 에서 `sdc_lambda_shared` (현재 0.25) 유지 vs 0.0 (완전 preserve)? → 기본 유지 제안.
- **Q2**: repetition_penalty weight 0.25 적절? 0.1 / 0.25 / 0.4 중 smoke 에서 false-positive rate 보고 선택.
- **Q3**: 노드 2개 시 SDC-v7 (Node B) 와 A Stage 1-2 (Node A) **별도 HF repo** 사용 확정 (위 § 4 HF push).
- **Q4**: A Stage 4 (meta_kl) 는 Stage 2 완료 후 teacher top-K 생성 필요 → 순차.
- **Q5**: A Phase A1 (best-of-N rollout) 는 vLLM 기반이라 SDC 학습과 동일 H200 리소스 사용 → **Node2 에서 A 만 실행** (Node B 와 분리). 따라서 Node1=B, Node2=A 구도 강제.
