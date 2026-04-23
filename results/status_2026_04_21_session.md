# Status Report — 2026-04-21 H200 Session

**Session**: 2026-04-21 (Opus 4.7 1M context)
**Scope**: SDC-shared H200 launch preparation + HF audit + cleanup
**Status**: Awaiting msrresrchbasicvc H200 node availability (6 jobs queued 2-6 days)

---

## 1. Session 진행 요약

### 1.1 인프라
- `node_recovery_0415.yaml` 변경: `metacognition_a100.yaml` 의 720h → 336h (VC 정책 600h 초과로 실패 건 수정)
- **신규 제출**: `h200_2nodes_0421.yaml` → `metacot-h200-2node-0421` (2× H200×4 basic VC, queued)
- **총 6 H200 BSC 잡 대기 중**:
  1. `node-recovery-h200-0415:rsp_grpo_node_1` (queued 6d)
  2. `node-recovery-h200-0415:metacognition_run_c` (queued 6d)
  3. `node-recovery-h200-0415:metacognition_e8` (queued 4d, 19:37-19:54 Running bounce)
  4. `metacot-h200-0417v2:metacognition_eval` (queued 3d)
  5. `metacot-h200-2node-0421:metacognition_h200_a` (queued ~2h)
  6. `metacot-h200-2node-0421:metacognition_h200_b` (queued ~2h)

### 1.2 BSC idle-suspend 대응 (신규 인프라)
- `scripts/gpu_keeper.py` — pre-installed torch로 4×GPU matmul 점유 (idle-suspend 방지)
- `scripts/run_sdc_on_h200_node.sh` — 3-tmux orchestrator (gpukeeper || boot || sdc_train)
- `scripts/poll_h200_and_launch.sh` — 5분 주기 6-node 감시 + Running 전이 시 base64 인코딩된 bootstrap-and-launch 명령 자동 SSH
- 중요 수정: poll 스크립트에 `cd "$HOME"` 추가 (amlt CLI 가 CWD 기반 `.amltconfig` 탐색 → /tmp 에서 실행 시 "unknown" 반환 버그 해결)

---

## 2. HF 에 이미 완료된 실험 — 16k budget eval (전부 `results/eval_1030_*` / `results/eval_d*_16k/`)

### 2.1 Baseline
| 실험 | Overall | GSM8K | MATH500 | AIME | meta_emit | HF 경로 |
|---|---|---|---|---|---|---|
| Base SFT (strict paired) | 75.92 | 92.6 | — | 33.33 | 0% | `models/v8_base_matched_strict_sft` |
| Meta SFT (strict paired) | 79.81 | 92.0 | 71.6 | 13.33 | 99.94% | `models/v8_meta_inside_strict_sft` |

### 2.2 Vanilla RL (현 mainline baseline)
| 실험 | Overall | GSM8K | MATH500 | AIME | meta_emit | HF 경로 |
|---|---|---|---|---|---|---|
| Base GRPO step 300 | **77.00** | 93.4 | 63.0 | **36.67** | ~0% | `checkpoints/verl_base_matched_0410/global_step_300` |
| Meta GDPO E21R-v2 step 300 | **81.65** | 92.6 | **74.8** | 13.33 | 82-99% | `checkpoints/verl_e21r_v2_0413/global_step_300` |

### 2.3 Self-distill (3 variants × 2 rebuilds, 전부 SFT 기반)
| 실험 | Overall | GSM8K | MATH500 | AIME | meta_emit | Δ vs baseline |
|---|---|---|---|---|---|---|
| D1 naive (base, no meta) | 68.35 | 88.8 | 51.4 | 10.0 | 0.3% | -7.57pp |
| D1 rebuilt (base) | 68.16 | 88.4 | 51.0 | 16.67 | 0.1% | -7.76pp |
| D2 epistemic (meta) | 66.89 | 87.4 | 49.8 | 10.0 | 100% | -13pp ❌ |
| D2 rebuilt (meta) | 59.81 | 78.8 | 44.0 | 6.67 | 98.9% | -16pp ❌ |
| D3b meta-KL (control-span, coef 0.15) | 58.06 | 79.4 | 39.4 | 13.33 | 98.9% | -17pp ❌ |

### 2.4 SDC (contrastive variants) — v6 completed, v7 not yet
| 실험 | Overall | GSM8K | MATH500 | AIME | meta_emit | Status |
|---|---|---|---|---|---|---|
| SDC-split v6 (stepfinal 300) | 59.51 | 79.4 | 42.6 | 10.0 | 99.1% | ❌ -20.3pp. HF `iamseungpil/metacot-sdc-split-v6-step300` (16 files, full weights) |
| **SDC-shared veRL** | — | — | — | — | — | ⏳ HF repo `metacot-sdc-verl-shared` 예약됨, weight 없음 (wandb logs 4 파일만) |

### 2.5 N3 / M1 시도 — 전부 FAILED
- 11× M1 (meta-only mask RLSD): 04-17, all `FAILED` 마커
- 13× N3 (contrastive, pre-SDC): 04-17, all `FAILED` 마커
- 7× sdc-split preceding v6: 04-18, all `FAILED` (4개는 4-36 step metrics 있음)
- **원인 미상 — trainer init 또는 환경 이슈 추정. 상세 log 분석 필요**

---

## 3. 핵심 결론 (답이 나온 질문)

1. **Vanilla meta RL (GDPO E21R-v2) 이 현 mainline 최강** — 81.65% / MATH500 74.8% / AIME 13.3% / meta 82-99%
2. **모든 self-distill 변형 실패** — naive / epistemic / meta-KL 모두 -7~17pp 회귀 (controller 복원되어도 accuracy 하락)
3. **SDC-split v6 실패** — -20.3pp, repetition loop + EOS avoidance (broad post-meta repel)
4. **Base RL 이 AIME 36.7%** — token budget 여유로 long-horizon 풀이 성공. Meta RL 은 token 고갈 (AIME 13.3%).

## 4. 답 없는 질문 (다음 실험 대상)

1. **SDC-shared (consensus preserve) 는 split 의 -20pp 를 고치는가?** — 유일한 새 data point
2. **M1/N3 FAILED 원인** — 설정/환경 이슈 디버깅 후 재시도 가치 있음 (controller + contrastive ablation)
3. **Base 에 SDC-shared 적용 시?** — 대조군 (meta controller 없이 contrastive 효과만)

---

## 5. 다음 액션

### 5.1 노드 Running 전이 시 (자동)
Poll loop + Monitor 가 감지 → `scripts/run_sdc_on_h200_node.sh`
1. gpu_keeper (BSC idle-suspend 방어)
2. bootstrap (verl 0.7.1 + vllm 0.6.3 + ray 2.10 + torch 2.4)
3. SDC-shared veRL 학습 (`verl_sdc_e21r_shared_h200_4x4k.yaml`, 기본 knob 그대로, 300 step)
4. HF push (600s 주기 daemon)
5. eval `src/eval/eval_hf.py --bench 1030` (4k + 16k budget)

### 5.2 2번째 노드 확보 시
- 옵션 A: M1 실패 원인 디버깅 후 재시도 (meta-only mask RLSD)
- 옵션 B: SDC-shared on base (`v8_base_matched_strict_sft` 학생 + `data/verl_train_redirect_base.parquet`)
- 옵션 C: sdc-uniform 또는 sdc-noise ablation (SDC v3 plan §H2 검증)

### 5.3 수정 불필요
**코드 수정 없음** 결정. 현재 구현으로 그대로 진행. 결과 보고 필요 시에만 개선.

---

## 6. 문서 / 데이터 / 체크포인트 현황

### 로컬 `/home/v-seungplee/metacognition/`
- `results/plan_h200_2node_parallel_2026_04_21.md` — 본 session 의 상세 계획 (이 파일과 연동)
- `results/plan_metacot_v8_active_2026_04_09.md` — mainline 활성 계획
- `results/plan_and_findings_consolidated_2026_04_16.md` — 종합 보고서
- `results/plan_EAD_unified_v3_2026_04_17.md` — EAD family 논문 계획
- `results/plan_SDC_v3_2026_04_19.md` / `plan_SDC_v4_veRL_2026_04_19.md` — SDC 설계 plan
- `results/plan_meta_rlsd_v2_2026_04_17.md` — M1/N3 hypothesis
- `results/report_SDC_v6_2026_04_19.md` — v6 실패 보고서
- `results/status_2026_04_21_session.md` — **본 문서**
- `docs/mainline_registry_2026_04_13.md` — mainline file index

### HF `iamseungpil/metacot` (dataset)
- `checkpoints/verl_base_matched_0410/` — base RL 체크포인트 steps 50/70/150/200/240/300
- `checkpoints/verl_e21r_v2_0413/` — meta RL 체크포인트 steps 190/220/250/300
- `sdc_runs/` — 12개 SDC 실험 디렉토리 (v5, v6 포함)
- `n3_runs/` — 31개 M1/N3 실패 run (디버깅 자료)
- `models/` — SFT 체크포인트 + self-distill 5 variant
- `data/` — v8 strict paired parquets + verl redirect subsets

### HF separate repos
- `iamseungpil/metacot-sdc-split-v6-step300` — v6 final weights (16 files)
- `iamseungpil/metacot-sdc-verl-shared` — SDC-shared 예약 (아직 weight 없음, 본 session 채울 예정)
- `iamseungpil/metacot-v8-base-matched-clean-sft` — base clean SFT
- `iamseungpil/metacot-verl-e21-historical` — E21 archive
