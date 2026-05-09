# Plan v5.9 — Framework Unification (Critic-Refined)

**작성일**: 2026-05-09 01:30Z
**선행**: v5.8 (`plan_pt_rlsd_v58_2026_05_09.md`) self-critic 후 강화
**핵심 강화점**: H1-H4 falsification 기준을 noise band 위로 끌어올림, V0_prefix·penalty scale 명시, smoke를 local-unit + GPU-1step으로 분리, decision tree에 H3 failure branch 추가, total_steps=100 정당화.

**증거 기반 (NEW):**
- RLSD 논문 (arxiv 2604.03128) **veRL 사용 확정** (PDF Section 5/Implementation, WebFetch 2026-05-09)
- 우리 R5 = veRL (논문 그대로) → ROD-PT를 TRL로 두면 **논문 + R5와 framework 모두 분리**
- OPD-Decoy의 top-K full-logit KL은 per-token PPO loss 항 → veRL/TRL 어느 쪽이든 actor loss 수정 필요. TRL 유지가 plan v5.7 §10.5 line 235와 정합.

---

## 0. Executive summary (refined)

ROD-PT를 R5와 동일한 **veRL framework**로 옮긴다. RLSD pattern은 "advantage modifier" — base PPO loss 무수정, advantage에 SDC factor + position penalty 주입. 검증은 step time → cold start fidelity → AIME signal → preempt resume 4단계 chain, 각 단계 falsification 기준이 noise band 위에 위치. OPD는 per-token KL 본질로 인해 TRL 유지.

---

## 1. Intent (한 문장, v5.8 동일)

R5와 같은 veRL framework 위에서 decoy teacher T-를 position teacher로 교체해, "메타 내용"(SDC factor) + "메타 위치"(top-K position penalty) 두 신호로 H-A(working memory disruption)를 검증한다.

---

## 2. Hypotheses (Falsifiable, v5.9에서 noise band 강화)

### H1 (framework consistency)
**가설**: ROD-PT veRL 구현이 R5와 동일 학습 루프 (advantage modifier path) 위에서 도는 한, step time은 R5의 1.0–1.5× 범위.
**검증**:
- Step 1-10 시간 측정 (R5 baseline 50-55s/step on H200×4)
- ROD-PT 평균 step time **≤ 80s** PASS (1.5× upper)
- ROD-PT 평균 step time > 100s (= 2× R5) FAIL → position teacher forward가 예상보다 무거움 → 메모리/IO 진단
**Noise resilience**: 10 step 평균이라 single-step jitter 무관.

### H2 (cold start fidelity)
**가설**: R5 step 300 ckpt를 직접 load하면 ROD-PT step 0의 정책은 R5 step 300과 정확히 동일.
**검증** (1차, deterministic):
- Step 0 ROD-PT가 sample 동일 prompt 5개 → 토큰 시퀀스가 R5 step 300의 동일 prompt sample과 byte-identical (greedy decode, temp=0)
- Mismatch 발견 → ckpt load 손상 또는 framework mismatch FAIL
**검증** (2차, statistical):
- 16k AIME eval N=30 → ROD-PT step 0 = R5 step 300 ± **5pp** (N=30 binomial 95% CI 폭)
- ±5pp 초과 → 학습 자체 시작 거부, 진단 모드
**Noise resilience**: 1차는 deterministic이라 noise 0; 2차는 noise band 명시.

### H3 (position teacher signal)
**전제**: H1 + H2 PASS 후에만 평가. 둘 중 하나 fail이면 H3 검증 보류.
**가설**: position teacher 활성 → 메타 발화 위치가 teacher top-K(K=16)와 정렬 → AIME 회복.
**검증**:
- step 100 16k AIME N=30: **AIME ≥ 22%** PASS (Plan v5.7 forecast 22-32% 의 lower bound, R5=10%, E21R=13.3% 초과)
- AIME < 17% (= R5 + N=30 noise band): position 신호 효과 없음 FAIL → H3 falsified
- 17% ≤ AIME < 22%: ambiguous → step 200 추가 학습 + 재평가
**보조 메트릭** (sanity, fail/pass 결정 안 함):
- `n_rollouts_with_meta` ≥ 0.5 (메타 발화 자체는 일어나야)
- `penalty_rate` 시간에 따라 ↓ (학습 진행으로 위치 정렬 강화)

### H4 (preempt resilience)
**전제**: push 인프라 fix (commit `4067c15` `checkpoint-*` 매칭, polling 5s) 적용된 상태.
**가설**: BSC tier preempt 시 손실 ≤ 5 step.
**검증** (1차, push 자체):
- 첫 ckpt save 시점 (TRL: step 5, veRL: global_step_5) 후 **15분 이내** HF에 등장 (16GB / 20MB/s = 13min upper bound)
- 15분 후에도 HF에 ckpt 없음 → push pipeline FAIL → resume 무의미
**검증** (2차, resume):
- 임의 preempt → 재할당 사이클에서 `[resume] OK at <ckpt>` 로그 + step N부터 재개 (N = 마지막 push된 step)
- fresh start 발생 → resume yaml 결함 → 진단

### H-fallback (H3 fail 시 next action)
- AIME < 17% → H-A (working memory disruption) 단독으로는 부족 → H-B (over-confidence) 우세 → difficulty-conditional meta gating plan v6.0으로 분기
- 17% ≤ AIME < 22% → step 200 추가 학습 → 재평가. 그래도 < 22%면 H3 partial fail로 기록, OPD-Decoy 결과 본 후 종합 판단

---

## 3. Method — veRL ROD-PT 구현 사양 (NEW: 정밀)

### 3.1 새 mode 추가
`verl_sdc.py` line 51-140 `REWARD_CONFIGS`에 `ROD_PT` 항목:
```python
"ROD_PT": {
    "funcs": [correctness_reward, meta_penalty_reward],
    "weights": [1.0, 1.0],
    "keys": ["correctness", "meta_penalty"],
},
```
Mode group:
```python
_SINGLE_TEACHER_MODES = {"RLSD_META_ATTR", "OPSD_META", "ROD_PT"}  # ROD_PT 추가
# _FORCED_META_MODES 에는 ROD_PT 추가 안 함 (자연 emit)
```

### 3.2 V0_prefix 정책 (NEW: 명확)
- ROD-PT는 **V0_prefix 사용 안 함** (자연 emit이라 student의 풀이 prefix 자체가 V0 역할)
- T_position input = `prompt + " Answer: " + gold + " " + completion[:p]`
- p = 첫 META_START 위치 (`response_ids == META_START_ID` first match, response_mask=True)
- 이는 R5의 `_build_teacher_logprob_batch(forced_meta=False)` 와 거의 같으나 completion을 p까지 자르는 부분만 다름

### 3.3 새 함수 `_build_position_teacher_batch`
`verl_sdc.py`에 추가 (~50 LOC):
```python
def _build_position_teacher_batch(tokenizer, prompt_texts, gold_answers,
                                   responses, response_mask, meta_start_id):
    # for each rollout: find first p where response_ids[p] == meta_start_id
    # if no meta emitted: skip (penalty=0 for that rollout)
    # else: build text = prompt + " Answer: " + gold + " " + decode(response[:p])
    # return DataProto for trainer._compute_ref_log_prob (returns logprobs of last token)
    # we then check if meta_start_id is in top-K of last token's logits
```
**중요**: `_compute_ref_log_prob`는 logprobs를 반환하지만 우리는 top-K 체크를 위해 last position의 full logits 필요. **veRL `_compute_ref_log_prob` 대신 새 함수 `_compute_position_top_k`**를 작성해 last position logits → top-K check만 수행 (메모리 효율).

### 3.4 Position penalty 스케일 (NEW: 정당화)
- penalty value = **-0.5** (Plan v5.7은 -1.0; 정정)
- 정당화: GRPO group-normalized advantage는 [-1, +1] 범위. -1.0 penalty면 advantage = 0 ~ -2.0 (한 sigma 이상 펄어짐) → 학습 신호 너무 강함. -0.5면 ≈ 1 sigma 처벌, 적절.
- yaml 파라미터로 노출 (`pt_position_penalty: -0.5`), ablation 가능

### 3.5 Position penalty 통합 위치
- `_attach_teacher_signals` 끝부분에서:
  ```python
  if mode == "ROD_PT":
      penalty_per_rollout = compute_position_penalty(...)  # [B] tensor, 0 or -0.5
      # Inject into rm_scores BEFORE compute_advantage:
      data.batch["token_level_rewards"][:, -1] += penalty_per_rollout  # last token
  ```
- last token에 penalty 추가하면 `compute_gdpo_outcome_advantage`가 base_advantage 계산 시 자연스럽게 반영
- `compute_sdc_gdpo_advantage`의 SDC factor (R5와 동일)는 그 위에 곱해짐

### 3.6 Cold start
- R5 ckpt: `iamseungpil/metacot-h100-rlsd-forced-meta-R5-0504/checkpoints/.../global_step_300/actor`
- yaml 직접 download (h200_r5_rl_0506b.yaml의 step 7 패턴 그대로)
- merge 불필요 (veRL 직접 load)

### 3.7 Total steps 정당화 (NEW)
- R5는 step 300까지 학습. 우리 ROD-PT는 R5 step 300 cold start 위에 추가 학습.
- total_steps = **100** (R5의 1/3)
- 정당화: position teacher signal은 학습 초기 (~50 step)에 위치 정렬 가속, 이후는 SDC factor (R5와 동일) 효과. R5가 300 step 동안 SDC factor 학습 마쳤으니 ROD-PT는 position 신호만 추가 학습.
- 검증: step 50/75/100 ckpt 모두 16k eval → AIME 추이로 saturation 시점 확인

### 3.8 Config & yaml
- `configs/verl_rod_pt_R10_h200_4x4k.yaml` (R5 yaml `verl_rlsd_forced_meta_R5_h100_4x4k.yaml` 기반):
  - `algorithm.sdc_mode: ROD_PT`
  - `pt_position_penalty: -0.5`
  - `pt_position_top_k: 16`
  - `total_training_steps: 100`
  - `actor_rollout_ref.model.path: <merged R5 step 300 path>`
- `h200_rod_pt_R10_v2.yaml` (`h200_r5_rl_0506b.yaml` 기반):
  - cold start path → R5 step 300
  - HF push repo → `iamseungpil/metacot-h200-rod-pt-R10-veRL`
  - resume yaml block (이미 commit `c8fee6f`에서 작성, veRL `global_step_*` 포맷 자동 매칭)

---

## 4. Verification Pipeline (NEW: smoke 분리)

### Phase 1a: Local unit smoke (CPU, no GPU)
- `tests/test_verl_rod_pt_units.py` 작성
- 5개 unit test:
  1. **Mode dispatch**: `_SINGLE_TEACHER_MODES`에 ROD_PT 포함되고, `_FORCED_META_MODES`에 ROD_PT 포함되지 않는다
  2. **Position search**: dummy `response_ids` (`[a, b, META_START, c]`) → p=2 정확히 찾음
  3. **No-meta rollout**: META_START 없는 rollout → penalty=0 (skip)
  4. **Penalty injection**: dummy `token_level_rewards`에 -0.5 추가 → 정확히 last token만 변경
  5. **Top-K check**: dummy logits (`[10, 5, 8, ...]`, K=2) + META_START_ID=0 → in top-K (no penalty); ID=99 → out of top-K (penalty)

PASS 조건: 5/5 unit test 통과 (pytest)

### Phase 1b: GPU 1-step smoke (실 노드, ~5분)
- 새 ckpt 받은 후 (R15 노드 또는 신규 노드) SSH 또는 yaml 1-step 모드로:
  - 작은 batch (4) + 1 training step
  - 학습 신호 sanity: `loss != NaN`, `grad_norm < 100`, `sdc_factor_mean ∈ [0.5, 2.0]`, `position_penalty` 출력 발생
- 위 5개 모두 통과 → Phase 2

### Phase 2: code-reviewer critic (Agent 호출)
- `code-reviewer` agent에 ultrathink mode로 review
- 검증 항목 (이전 reviews에서 발견된 패턴):
  - DataProto 키 충돌 (sdc_teacher_pos / sdc_teacher_neg 변경 안 함)
  - Mode dispatch 누락 (ROD_PT가 모든 분기에 처리되는지)
  - Position penalty NaN/Inf protection
  - top-K logits memory (full vocab 152K × batch × 1 = 152K × 4 = 608KB acceptable)
  - cold start path 정확성
- 출력: Critical / Warning / Suggestion 분류

### Phase 3: Iterative improve
- Critical 발견 → task-planner-analyzer → modular-code-architect 수정 → 다시 code-reviewer
- 최대 5 iteration. Critical 0 + Warning ≤ 3 → 진행
- 각 iteration에서 v5.x → v5.x.1 patch commit

### Phase 4: autoresearch launch
- 새 yaml 제출 (`metacot-rod-pt-R16-0509-verl`)
- Phase 4a: H1 (step time, 첫 10 step에서 측정) PASS 확인
- Phase 4b: H2 (cold start fidelity, step 0에서 sample 검증) PASS 확인
- Phase 4c: H4 (push within 15min) PASS 확인
- Phase 4d: step 100에서 H3 verification

각 Phase 통과 못 하면 즉시 중단 + Phase 3로 회귀 (코드 결함) 또는 H-fallback (가설 결함)

---

## 5. OPD-Decoy 결정 (TRL 유지, justification 강화)

OPD-Decoy의 top-K full-logit KL을 PPO loss 항으로 추가하려면:
- **TRL**: GRPOTrainer subclass의 `compute_loss` override 한 줄 (이미 작동, R8 step 1-2 검증됨)
- **veRL**: actor의 PPO loss를 직접 수정 또는 auxiliary loss hook 추가. veRL 0.7.1에는 `actor.aux_loss_weight` 추가 가능하나 worker.compute_loss 흐름 깊이 재배선 필요
- **결론**: TRL이 OPD에 자연. plan v5.7 §10.5 line 235와 정합. 변경 안 함

OPD R8는 H200 141GB로 옮긴 후 정상 학습 신호 확인됨 (step 1: reward 0.32, kl_pos 2.24, kl_neg 2.26). preempt + push 인프라 fix는 ROD/OPD 공통 적용됨.

---

## 6. Risk & Fallback (refined)

| Risk | Probability | Impact | Mitigation | Fallback |
|---|---|---|---|---|
| veRL `_compute_ref_log_prob`이 last-token logits 미반환 | 50% | High | 새 함수 `_compute_position_top_k` 작성 | TRL 버전 R15 유지 |
| Position penalty -0.5 너무 약함 | 20% | Medium | yaml param, ablation 가능 | -1.0 ablation, but H-A scale 비교 |
| H1 fail (step time 2x slow) | 15% | Medium | smoke로 사전 검증 | 메모리 / IO bottleneck 진단 |
| H2 fail (cold start mismatch) | 5% | Critical | byte-identical check 사전 | ckpt format 진단, R5 ckpt 검증 |
| H3 fail (AIME ≤ 17%) | 30% | Method-level | H-fallback (H-B uppercase) | difficulty-conditional gating plan v6.0 |
| BSC preempt 빈발 | 80% | Operational | push 5s + resume (commit `4067c15`+`c8fee6f`) | tier 변경 (Premium quota 확인) |

---

## 7. Decision Tree (NEW: H3 outcomes 포함)

```
veRL ROD-PT 구현 (Phase 1a-3)
├── Phase 1a-1b smoke PASS
│   ├── Phase 2 critic Critical=0 → Phase 4 launch
│   └── Phase 2 critic Critical>0 → Phase 3 fix → re-smoke → re-critic
├── Phase 1 smoke FAIL
│   ├── 코드 결함 → Phase 3
│   └── framework limit → Fallback: TRL R15 유지
└── Phase 4 launch
    ├── H1 FAIL → 메모리/IO 진단, 1.5x 못 맞추면 abort
    ├── H2 FAIL → ckpt format 진단, R5 reload 검증
    ├── H3 PASS (AIME ≥ 22%) → H8 (vs OPD-PT 비교) 다음 ablation 진행
    ├── H3 ambiguous (17-22%) → step 200 추가 학습 → 재평가
    └── H3 FAIL (< 17%) → H-fallback: H-B 가설로 plan v6.0 분기
```

---

## 8. Timeline (refined)

- **D0 (today, 01:30Z–)**: Plan v5.9 작성 ✓, codex 논의 round 1 시작, Phase 1a unit test 작성
- **D0+0.5 (~12h)**: codex round 2-3 (review-counter-review), Phase 1a smoke PASS, Phase 2 critic round 1
- **D0+1**: Phase 3 iteration 안정화 (Critical=0)
- **D0+1.5**: Phase 1b GPU smoke (현재 R15/R8 노드에서 SSH 또는 신규 노드 신청)
- **D0+2**: Phase 4 launch
- **D0+5**: H3 verification (step 100 16k AIME)

---

## 9. v5.8 → v5.9 변경 요약

| § | v5.8 → v5.9 |
|---|---|
| H1 | "±20%" → "≤ 1.5×" + 10-step 평균 (single-step jitter 무력화) |
| H2 | "±2pp" → byte-identical (1차) + ±5pp (2차, N=30 binomial CI) 분리 |
| H3 | 전제 (H1+H2 PASS) 명시; ambiguous range (17-22%) 추가 |
| H4 | push 자체 검증 (1차) + resume 검증 (2차) 분리 |
| H-fallback | NEW: H3 fail 시 next action 명시 |
| §3.2 | V0_prefix 정책 명시 (ROD-PT는 사용 안 함) |
| §3.4 | penalty -1.0 → -0.5 (GRPO advantage scale 정당화) |
| §3.5 | 통합 위치를 token_level_rewards last token으로 명시 |
| §3.7 | total_steps=100 정당화 추가 |
| §4 | smoke를 1a (local unit pytest) + 1b (GPU 1-step) 분리 |
| §5 | OPD-veRL 비교 깊이 강화 (worker.compute_loss 재배선 비용) |
| §6 | risk 표에 probability 추가 |
| §7 | decision tree에 H3 outcome branches 추가 |

---

## 10. v5.9 self-critic (next iteration trigger)

이번 iteration에서 발견된 잠재 결함:
- §3.3 `_compute_position_top_k` 새 함수 — veRL 0.7.1 API와 호환되는지 미검증. Phase 1a unit test에서 검증 필요.
- §3.5 `token_level_rewards[:, -1] += penalty` — last token이 EOS인지 또는 마지막 valid token인지 불명. response_mask로 valid last position 찾아야.
- §4 Phase 1b GPU smoke — 어느 노드에서? R15 SSH 불가 (managed singularity). 신규 노드 신청은 BSC wait.

이 3개를 v5.10에서 해결. 다음 iteration:
- §3.3을 코드 작성 시점에 검증 (codex 또는 직접)
- §3.5 last valid token로 정정
- §4 Phase 1b를 R15 cycle 끝나는 시점 (preempt 후 신규 노드 받을 때) Hijack해서 1-step 검증

---

## 11. 한 줄 요약

ROD-PT veRL 통일 (RLSD 논문 + R5와 framework 동일)을 4단계 falsifiable 가설 (H1 step time ≤1.5× / H2 cold start byte-identical / H3 AIME ≥22% / H4 push ≤15min)로 검증. 코드는 Phase 1a unit test → 1b GPU smoke → 2 critic → 3 fix iterate, 모두 PASS 후만 Phase 4 launch. H3 fail 시 H-fallback (over-confidence) plan v6.0으로 분기.
