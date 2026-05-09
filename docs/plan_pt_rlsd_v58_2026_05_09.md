# Plan v5.8 — Framework Unification (ROD-PT → veRL, OPD-Decoy → TRL)

**작성일**: 2026-05-09
**선행 문서**: `plan_pt_rlsd_v57_2026_05_07.md`, `report_meta_cot_v57_2026_05_08.md`
**변경 요지**: User 통찰 반영 — "ROD-PT는 R5 + position teacher → R5와 동일 framework(veRL)이어야 한다." Plan v5.7은 TRL 통일을 가정했으나 이는 의도와 어긋남.

---

## 0. 의도 정합성 재검토

| 항목 | Plan v5.7 (현행) | Plan v5.8 (제안) |
|---|---|---|
| ROD-PT framework | TRL GRPOTrainer subclass | **veRL** (R5 동일) |
| OPD-Decoy framework | TRL | **TRL 유지** (top-K KL 본질) |
| ROD-PT cold start | Meta SFT v8 fresh | **R5 step 300** (veRL 직접 load) |
| OPD cold start | R5 step 300 (merge to HF) | 동일 |

**근거**:
- veRL R5 (`src/training/verl_sdc.py:423-549`)는 `_attach_teacher_signals`로 DataProto에 teacher logprobs 첨부 → `compute_sdc_gdpo_advantage`가 advantage에 SDC factor 곱함. **standard PPO loss는 unmodified.**
- ROD-PT의 position teacher penalty는 rollout-level 스칼라 → reward로 환원 가능 → veRL RewardManager 확장만으로 구현 가능.
- OPD-Decoy의 top-K full-logit KL은 per-token PPO loss 항 → veRL에 추가하려면 worker.compute_loss 수정 필요 (invasive). TRL이 자연스러움.

---

## 1. Intent (한 문장)

R5와 같은 veRL framework 위에서 decoy teacher T-를 position teacher로 교체해, "메타 내용"(SDC factor) + "메타 위치"(top-K position penalty) 두 신호로 H-A(working memory disruption)를 검증한다.

---

## 2. Hypotheses (Falsifiable)

### H1 (framework consistency)
**가설**: ROD-PT를 veRL로 구현하면 R5와 동일 학습 속도(50s/step)와 동일 ckpt 형식 (`global_step_N/`)을 가진다.
**검증**: 첫 10 step에서 step time 측정. R5 (50-55s/step)의 ±20% 이내면 PASS.
**Falsifies**: step time이 R5의 2x 이상이면 framework 차이가 큼 → veRL 구현 결함 의심.

### H2 (cold start continuity)
**가설**: R5 step 300 ckpt를 cold start로 직접 사용하면 ROD-PT는 R5 baseline (AIME 10%) 위에서 시작.
**검증**: ROD-PT step 1 시점 16k AIME eval (사실상 R5 step 300 직접 eval과 같아야).
**Falsifies**: AIME ≠ 10% ± 2pp → ckpt load 손상 또는 framework mismatch.

### H3 (position teacher signal)
**가설**: position teacher가 활성화되면 메타 발화 위치가 teacher top-K(K=16)와 정렬돼 H-A 단절 문제를 완화한다.
**검증**: step 100에서 16k AIME eval. **AIME ≥ 22%면 PASS** (Plan v5.7 forecast 22-32%, baseline R5 = 10%).
**Falsifies**: AIME ≤ 13.3% (=E21R) → H-A는 position 정렬과 직교, H-B (over-confidence) 우세.

### H4 (preempt resilience)
**가설**: HF auto-push (`global_step_N/`) + auto-resume yaml로 BSC tier preempt 시 손실 ≤ 5 step.
**검증**: 임의 preempt 후 재할당 사이클에서 `[resume] OK at global_step_N` 로그 출현 + step N부터 재개.
**Falsifies**: fresh start 발생 → push/resume 인프라 결함.

---

## 3. Method — veRL ROD-PT 구현 사양

### 3.1 새 mode 추가
`verl_sdc.py:_VANILLA_MODES`, `_FORCED_META_MODES`, `_SINGLE_TEACHER_MODES`에 `ROD_PT` 추가:
- `ROD_PT_MODES = {"ROD_PT"}`
- single teacher (T- 없음, decoy off)
- forced_meta=False (학생이 자연 emit, R5와 차이)

### 3.2 Position teacher forward
새 함수 `_build_position_teacher_batch`:
- 학생의 첫 META_START 위치 p 탐색 (`response_ids[i] == META_START_ID`)
- input = prompt + V0_prefix + gold + completion[:p]
- forward → top-K 검증
- top-K에 META_START 없으면 rollout-level penalty -1.0

### 3.3 Reward 통합
`_attach_teacher_signals`에서:
- T+ (content) forward → SDC factor (R5 동일)
- T_position forward → position_penalty
- `data.batch["rm_scores"] += position_penalty` (rollout-level scalar)

### 3.4 Config & yaml
- `configs/verl_rod_pt_R10_h200_4x4k.yaml` (R5 yaml 기반 + `_ACTIVE_SDC_CONTEXT["mode"] = "ROD_PT"`)
- `h200_rod_pt_R10_v2.yaml` (R5 yaml 기반, ckpt path → `iamseungpil/metacot-h200-rod-pt-R10-veRL`)

### 3.5 Cold start
- R5 ckpt: `iamseungpil/metacot-h100-rlsd-forced-meta-R5-0504/checkpoints/.../global_step_300/actor`
- yaml에서 직접 download, no merge step (TRL 버전과 다름)

---

## 4. Verification Pipeline (iterative-code-review)

### Phase 1: Smoke test (local CPU)
- `scripts/smoke_verl_rod_pt.py` 작성
- 5 step:
  1. SDC factor compute (R5 path 동일 검증)
  2. Position teacher forward (top-K logic)
  3. Reward 통합 (rm_scores += penalty)
  4. Backward gradient (PPO standard, unmodified)
  5. Trainer import + dummy 1-step

### Phase 2: Code-reviewer critic
- `code-reviewer` agent로 ultrathink 수준 리뷰
- 검증 항목:
  - veRL DataProto 사용 정확성
  - mode dispatch 누락 없는지
  - position penalty scale (-1.0 vs group-normalized advantage)
  - cold start path 정확성
  - Hydra config 유효성

### Phase 3: Iterative improve
- Critical issues 발견 시 task-planner-analyzer → modular-code-architect 수정 → 다시 code-reviewer
- "no critical issues" 까지 반복 (최대 5 iteration)

### Phase 4: 노드 launch (autoresearch)
- 새 yaml 제출
- 첫 step 학습 신호 검증 (H1, H2)
- step 50 eval (H3 partial)
- step 100 16k eval (H3 full)
- preempt 시 resume 검증 (H4)

---

## 5. OPD-Decoy 결정 (TRL 유지)

OPD-Decoy는 top-K full-logit KL이 PPO loss 항으로 직접 들어가므로 veRL forking 비용이 크다. Plan v5.7 §10.5 그대로 TRL 유지하되:
- compute_loss Phase reorder (이미 적용됨, commit `9fa0e3f`)
- HF resume + auto-push 인프라 (이미 적용됨, commit `4067c15`)
- save_interval 5, push interval 5s (이미 적용됨)

OPD는 ROD-PT 결과 본 후 별도 평가.

---

## 6. Risk & Fallback

| Risk | Mitigation | Fallback |
|---|---|---|
| veRL ROD-PT 구현 결함 | iterative-code-review smoke→critic→improve loop | TRL 버전 유지 (현재 `meta_rod_pt_trainer.py`) |
| veRL position teacher forward 메모리 폭발 | num_rollouts=2, max_response 4096 (R5 동일) | TP=4 sharded teacher (R5 패턴) |
| BSC preempt 빈발 | resume 인프라 (commit `c8fee6f`) | save_interval 5 (≤5 step 손실) |
| H1 fail (step time 2x slow) | smoke로 사전 검증 | 메모리/IO bottleneck 진단 |
| H3 fail (AIME ≤ 13.3%) | H-B 가설 (over-confidence) 우세 | difficulty-conditional meta gating 별도 plan |

---

## 7. Decision Tree

```
veRL ROD-PT 구현 (this plan)
├── Phase 1 smoke PASS
│   ├── Phase 2 critic clean → Phase 3 → Phase 4 launch
│   └── Phase 2 critic Critical → Phase 3 fix → re-smoke → re-critic
├── Phase 1 smoke FAIL critical
│   ├── 수정 가능 → Phase 3
│   └── 수정 불가 (framework limit) → Fallback: TRL 유지 (현재 R15/R8)
└── 시간 초과 (>1 day)
    └── Fallback: TRL 유지, plan v5.8 보류
```

---

## 8. Timeline

- **D0 (today)**: Plan v5.8 작성, codex 논의, smoke test 작성
- **D0+1 (tomorrow)**: smoke PASS + critic round → 수정 → re-critic 안정화
- **D0+2**: H200 노드 신청, autoresearch launch
- **D0+5**: ROD-PT step 100/200/300 16k AIME eval → H3 verification

---

## 9. 한 줄 요약

ROD-PT를 R5와 같은 veRL framework로 통일해 (1) R5 step 300 ckpt 직접 cold start, (2) 학습 속도 50s/step 회복, (3) preempt-resume robust, (4) framework 의도-구현 정합성을 확보한다. 검증은 H1(step time)→H2(cold start)→H3(AIME 회복)→H4(resume) 4단계, 코드는 smoke→critic→improve loop, 모든 단계 통과 후만 autoresearch.
