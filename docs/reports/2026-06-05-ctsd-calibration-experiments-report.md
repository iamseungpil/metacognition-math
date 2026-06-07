# CTSD Calibration Experiments — Status Report (2026-06-05)

## North-star
Meta-CoT(`<|meta|>` 토큰으로 메타인지를 외재화)가 **정확도에서 Base SFT를 이기고
(✅ 달성) AND calibration을 개선(❌ 미달)** 해야 한다.

현재 상태 한 줄 요약: **정확도는 풀렸고(RLVR 0.786 ≫ Base SFT 0.548), calibration이 미해결.**
최고 메타-보존 RLVR(e4_baseline)은 정확하지만 confidence를 0.29로 말하는 **과소확신**이라
ECE 0.557.

---

## 실험 아크 (왜 지금 여기에 왔나)

1. **정확도 해결.** RLVR(correctness 중심)로 v8_strict cold start에서 0.786 도달
   (e4_baseline = VANILLA_GRPO), Base SFT 0.548 대비 +24pp. 메타 토큰 ~92% 유지.
2. **calibration 갭 발견.** 같은 모델이 confidence를 0.29로 과소 표현 → ECE 0.557.
   "정답인데 자신 없다"가 핵심 병 (overconf_rate≈0).
3. **추론 steering 탐색(E.5–E.7).** confidence 강제 조건화 → 추론이 인과적으로 바뀌어
   정확도↑(E.6b +5.5pp). 하지만 confidence 숫자는 안 따라와 calibration 악화(E.7).
   → **steering은 한 방향 편향 레버일 뿐 calibration이 아니다.**
4. **RL 전이 검증(E.4).** 추론에서 통한 contrastive teacher를 RL에 넣으니 **손해**
   (conf_down 0.714 < baseline 0.786, decoy 0.588). → 추론 효과 ≠ 학습 효과.
5. **현재 두 갈래(live):**
   - **E.8** = gold-free RLSD(E.4의 gold 조건화만 제거한 단일 변수 ablation).
   - **E.9** = BCI_RLVR(calibration을 *직접* 보상). 정확도가 아니라 calibration 정면 타깃.
     주: 진짜 binned-confidence 주입은 verl 0.7.1 agent-loop 롤아웃이 prompt를 messages에서
     재토큰화해 input_ids 텐서가 없어 step-1 smoke에서 크래시 → **proper-score on
     self-emit**(correctness + outcome_calibration, 주입 없이 자기 confidence에 보상)로 전환.
     주입은 agent-loop-aware 구현으로 후속 보류.

---

## 0. 북극성 기준선 (SFT, RL 없음)

| | acc | meta emit | 의미 |
|---|---|---|---|
| **base_sft_v8** | **0.548** | 0.91 | Base SFT 정확도 기준선. RLVR 0.786이 +24pp로 이김 → 정확도 북극성 ✅ |

## 1. 추론 Steering (v8_strict SFT, 1030@16k k=8, self=unsteered, probe 채점)

| Exp | 방식 | acc (self→steer) | Δacc (p) | calib_gap | verb_conf | pass@k | 판정 |
|---|---|---|---|---|---|---|---|
| **E.6b** | conf_down logit-steer (meta 토큰에 conf 0.15 vs 0.95 방향) | 0.543→**0.583** | **+5.5pp** (9e-4) | 0.253→0.333↑ | 0.600→0.570 | 0.836→0.718 | 정확도 레버 OK, calib 미개선 |
| **E.7** | adaptive 1−c steer (학생 자기 conf의 반대로) | 0.543→0.575 | +4.3pp (7e-3) | 0.254→0.322↑ | 0.600→0.602 | 0.834→0.714 | fixed<adaptive 아님, 과소확신 악화 |
| E.5 | gold-free steer (stance+conf, gold 없이) | node-level만 | — | — | — | — | inference-only 보조 검증 |

읽는 법: conf를 낮게 밀면 추론이 신중해져 정확도↑. 하지만 verb_conf는 거의 안 움직여
정확도와 confidence가 더 벌어져 calib_gap 악화. pass@k 하락 = 토큰별 결정적 bias가
샘플 다양성을 줄임. → steering = 편향 레버, calibration 아님.

## 2. RL 학습 — full eval (RL 체크포인트, eval_vllm 1030@16k k=8, ECE 포함)

| Exp | 방식 (보상/teacher) | acc | ECE | conf | emit | 판정 |
|---|---|---|---|---|---|---|
| **e4_baseline** | VANILLA_GRPO, correctness only (no teacher) | **0.786** | **0.557** | 0.29 | 0.92 | 최고 메타-보존 RLVR, 과소확신 |
| e4_gold_conf_down | RLSD magnitude + gold conf teacher | 0.714 | 0.435 | 0.28 | 1.00 | teacher 정확도 손해 |
| e4_gold_decoy | RLSD + gold/decoy contrast teacher | 0.588 | 0.295 | 0.30 | 0.98 | 더 큰 손해 |
| e4_gold_stance | RLSD + cautious/confident stance teacher | gs300 미도착 | — | — | — | eval 대기 |

## 3. RL 학습 — 과거 런 (HF eval/ 전체, overall_accuracy + meta emission; ECE 미산출 schema)

| Run | 방식 | acc | meta emit | 비고 |
|---|---|---|---|---|
| r10v2_e20a (step275) | ROD_PT on e20a 모델 | 0.850 | **0.012** | meta 붕괴 → 실격 |
| arm1_matched_e21rv2 (step300) | E21Rv2 control reward heads | 0.730 | 0.99 | baseline 미달 |
| R10v2 (step300) | ROD_PT (content×position 2-teacher) | 0.723 | 0.98 | |
| R18b (step302) | ROD_MQ_CONTRAST (+decoy contrast) | 0.709 | 0.99 | |
| R18a (step300) | ROD_MQ (single meta-quality teacher) | 0.676 | 0.99 | |
| r5 (step300) | RLSD_FORCED_META | 0.672 | 0.96 | |
| R16v3 (step300/step310) | ROD_PT_DEGEN | 0.672 / 0.669 | 1.00 | 두 ckpt |
| R5 (step200) | RLSD_FORCED_META (이른 ckpt) | 0.670 | 0.95 | |
| R18c (step300) | GFN_OPSD_CONTRAST (listwise KL distill) | 0.668 | 0.98 | |
| rod_pt_R10 (step100) | ROD_PT (이른 ckpt) | 0.654 | 0.98 | |
| meta_opd_R7 | OPSD (KL 분포 distill) | 0.628 | 0.91 | |
| arm2_rod_pt2 (step300) | ROD_PT2_E21CTRL (recipe X) | 0.509 | 0.76 | |
| Arm3 (step200) | STABLE_GFN_C2FIX | 0.186 | 0.97 | 붕괴 |

## 4. Live (eval 전, 학습 중)

| Exp | 방식 | 상태 |
|---|---|---|
| **E.8** | gold-free RLSD (RLSD_META_CONTRAST + conf_free teacher) | 🔄 학습중 ~80/300 |
| **E.9-v2** | BCI_RLVR = correctness + outcome_calibration proper-score (self-emit) | 🔄 relaunch, 학습 진입중 |

---

## 방식 요약 (steering vs 학습)

- **Steering (E.5–E.7, 추론):** 학습 안 함. 추론 시 meta 토큰 logit에
  `α·(logit|conf_low − logit|conf_high)`를 더해 confidence 강제 조건화 → 추론 신중화로
  정확도↑. confidence 숫자 자체는 안 배움.
- **RLSD/ROD/GFN/OPSD (R5–R18, e4):** correctness advantage의 *부호*는 RLVR이 주고,
  teacher(T+/T−)가 meta 영역 advantage의 *크기*(RLSD magnitude) 또는 *분포*(GFN/OPSD KL)를
  조형. e4에서 검증된 결론 = teacher가 정확도 **손해**.
- **E.8 (gold-free RLSD):** e4의 gold 조건화만 제거한 단일 변수 ablation.
- **E.9 (BCI_RLVR, self-emit):** 유일하게 **calibration을 직접 보상**. correctness가
  GDPO dominant head라 정확도 보존, outcome_calibration(proper-scoring Brier + revision)이
  자기 confidence를 정답률에 맞추도록 학습. (진짜 binned-injection은 agent-loop 구현 보류.)

---

## 앞으로 계획

**즉시(자동 모니터링):**
- E.9-v2 학습 진입 확인 → 300스텝(~6h) → 1030@16k eval로 e4_baseline(0.786/ECE 0.557)
  대비 ECE<0.35 & acc≥0.786−1.5pp 판정.
- E.8 step 300 도달 → eval vs E.4 baseline 0.786 / conf_down 0.714.
- E.4 stance gs300 도착 시 eval 추가.

**판정 분기:**
- E.9가 ECE 자르고 정확도 유지 → calibration 절반 달성. 다음: bin-injection(agent-loop)로
  sweep 강화 or self-consistency 기반 calibration.
- E.9가 정확도 손해 → calibration weight(0.5) 낮추거나 outcome_calibration→group-Brier 전환.
- E.8이 gold-free로도 baseline 못 넘음 → teacher 방향 접고 calibration 보상에 집중.

**중기:** E.8+E.9 종합 → "정확도(teacher)와 calibration(proper-score) 중 무엇이 north-star를
동시에 만족시키나" 결론 → 최종 단일 레시피(유력: correctness + outcome_calibration) 확정.

---

## 격리/안전 메모 (E.9)
- 신규 mode/플래그(`sdc_force_inject_conf`)는 전부 additive·gated → 기존 모든 mode는
  플래그 off일 때 byte-identical. E.8 등 라이브 잡은 frozen release로 무영향.
- 코드리뷰 Critical 2건 수정(validation 오염 C1, seed EOS-충돌 I1). 로컬 8/8.
- 주입 wrap은 agent-loop 비호환으로 step-1 smoke에서 크래시 → 게이트 OFF, self-emit로 전환.
  wrap 코드는 게이트-off 상태로 남겨 후속 agent-loop-aware 구현 대기.
- commits 89685d9 → ac634b3, release asset 439609207 (self-emit).

---

## 업데이트 (2026-06-07)

### Intent 재정렬
North-star를 명문화: **"메타인지 행동을 강화해 정확도를 올린다 — 언제/무엇이 좋은 메타인지인지
규정한 metacot를 정의하고 이를 강화하는 RL을 개발한다."** calibration은 부분 목표(=유용한
메타인지의 한 신호)임을 명확히 함. `CLAUDE.md` Goal + `docs/intent_and_plan_2026_06_07.md`에 반영.

### E.9 inject — 진짜 binned-injection 학습 완료 ✅
- agent-loop-native 재구현(BCIConfAgentLoop)으로 **진짜 주입** 버전이 **gs300 완주**,
  HF `metacot-h200-e9-bci-inject` head=300 (온전한 4-shard actor + data.pt).
- step-1 smoke 통과(outcome_calibration이 bin별로 변동) → 주입 경로 정상 동작 확인됨.
- **eval(`e9-inject-eval-gs300`)이 노드 경합으로 8h+ queued** — B축 첫 직접 결과가 여기서 막힘.
  결과 나오면 e4_baseline(0.786 / ECE 0.557 / conf 0.29) 대비 acc≥~0.77 & ECE≪0.557 판정.

### 인프라 — resume-from-HF 추가 (durability gap 수정)
- 문제: cross-node preempt가 `/scratch`를 지우는데 `resume_mode=auto`는 로컬만 봐서 매번
  step 0부터 재시작 → e8/rlvr-v2가 gs15/gs10에서 정체.
- 수정: `scripts/pull_resume_ckpt.py` — HF에서 최신 global_step 다운로드 +
  `latest_checkpointed_iteration.txt` 작성 → resume_mode=auto가 그 step부터 재개.
  e8/e9 학습 YAML 3종에 verl 실행 직전 삽입. 검출 로직 스모크 4/4 통과(300/10/15/no-op).
  **적용하려면 해당 학습 잡 relaunch 필요**(현재 러닝 잡은 frozen command라 무영향).

---

## E.9 inject EVAL 결과 + 근본원인 진단 (2026-06-07)

### 결과 (1030@16k k=8, vs e4_baseline 0.786/ECE0.557/conf0.29)
| 지표 | baseline | **inject** | 판정 |
|---|---|---|---|
| accuracy | 0.786 | **0.609** | ❌ −18pp |
| ECE | 0.557 | **0.083** | ✅(착시 포함) |
| mean_conf | 0.29 | 0.602 | 과소확신 해소 |
| meta_emission | 0.92 | **0.41** | ❌ 발화 붕괴 |
| 벤치 | — | gsm8k .736 / math500 .514 / aime .096 | |

### AIME 240개 전수조사 — decoherence(붕괴)가 진짜 원인
- **75%(181/240)가 16k에서 truncate**, 그것도 **반복 LaTeX 조각 스팸**(`\frac\n\]\n\boxed\n\end...`)으로 **박스 답 미커밋**. meta_blocks=0.
- 정답 응답 median **1030 tok**(쉬운 문제 빠르게 해결) vs 오답 median **16384 tok(꽉 참)**.
- baseline 대비 악화: acc 0.163→0.096, truncate 0.67→0.75, meta 0.39→0.30.

### 학습 추세 — underfitting 아님, 점진적 붕괴
- output.log 288 step: response_length **mean ~650→~1450**, **clip_ratio 0.10→0.25** (둘 다 학습 중 ~2배 증가).
- val에서 **outcome_calibration 보상 전 카테고리 0** (meta 발화 죽어 신호 소멸) → 실효 신호는 correctness뿐.
  hard 카테고리 val reward 음수(geometry −0.13, number_theory −0.13, omni-math −0.73).
- → **더 학습하면 상한이 오르는 게 아니라 붕괴가 심해짐.** 목적함수가 유도한 실패.

### ECE 개선의 착시
ECE가 준 건 일부만 진짜(easy conf 0.71↑). 나머지는 **"안정적으로 틀리고(붕괴)+낮은 conf"가 Brier상 잘 보정된 것으로 채점**된 artifact — 모델이 *더 못해진 덕에* 낮은 confidence가 정당해짐.

### 선행연구로 본 근본원인 (3-agent 조사, 2024–2026)
1. **DCPO [Ma et al., ICML2026, arXiv:2603.09117] — 직격탄.** correctness에 Brier를 **결합**하면 정확도-gradient와 calibration-gradient가 **충돌**(Fisher 내적 음수). RLCR의 결합 Brier가 **AIME 40%→32.8%로 정확도 하락**을 그대로 관측. 처방: **gradient 분리** — correctness는 reasoning 토큰, calibration은 conf 토큰에만, advantage 마스킹.
2. **RLCR [Damani et al., 2025, 2507.16806]** = 우리 설정(correctness+Brier). "정확도 손실 없음" 주장했으나 DCPO가 hard reasoning에서 반박.
3. **Dr.GRPO [2503.20783]** — GRPO의 per-response length 정규화가 **특히 오답에서 길이를 부풀림** = 우리 "hard 문제 스팸" 시그니처. 처방: length/std 정규화 제거.
4. **DAPO [2503.14476]** — clip-higher / dynamic sampling / token-level loss / **soft overlong punishment + truncated 샘플 마스킹**(0 보상 금지) / format(box) 보상.
5. **Entropy collapse [2505.22617]** — clip_ratio 2배 = 엔트로피 붕괴 지문. 처방 KL-Cov/Clip-Cov.
6. **Yue et al. [2504.13837]** — RLVR은 쉬운 mode를 sharpen하고 **hard 문제 reasoning 경계는 축소**(easy↑/hard↓).
7. **Taming Overconfidence (PPO-M/C) [2410.09724]** — calibration 보상은 reward-hacking 당하기 쉬움.
8. 메타인지 RL: **SCoRe [2409.12917]**(Δ-correctness/진전 보상, warmup 없으면 붕괴), **Reflect-Retry-Reward [2505.24726]**(retry가 fail→success 뒤집을 때만 reflection 토큰 보상 = 우리 utility-gating과 동일, 검증됨), **Huang et al. [2310.01798]**(oracle 없는 intrinsic self-correction은 실패), **Self-Verification Dilemma [2602.03485]**(검증 과보상은 정확도 ↓), **Metacognitive Harness [2605.14186]**(모델은 calibrated FoK를 갖지만 *행동*을 못함 → 행동화하면 low-conf 16%→42%).

### 진단 방법(다음에 적용)
- ECE 단독 금지 → **AUROC(conf vs correct, discrimination)**, **accuracy-stratified ECE**, **selective-accuracy/coverage** 동시 보고(착시 차단).
- **Pass@k + 난이도층화 정확도**(easy↑/hard↓ = diversity collapse 확인).
- 학습 중 **response_length/clip_ratio·엔트로피** 추적, **truncated-no-box 비율**을 1급 지표로.

### 처방 방향 (의도 = 유용한 메타인지로 정확도↑)
A. **anti-decoherence/commit**: soft-overlong + box format 보상, truncated 마스킹, Dr.GRPO loss.
B. **gradient 분리(DCPO)**: correctness는 답 토큰, calibration은 conf 토큰에만.
C. **utility-gated meta(SCoRe/Reflect-Retry)**: meta-action이 **wrong→right 뒤집을 때만 +, right→wrong엔 −**, emission 자체엔 보상 금지.
D. **self-consistency를 conf 타깃**으로(그룹 정답률) — 신호가 죽지 않고 정확도와 상관.
E. correctness dominant 유지 + KL anchor로 붕괴 방지(warmup 후 shaping).
