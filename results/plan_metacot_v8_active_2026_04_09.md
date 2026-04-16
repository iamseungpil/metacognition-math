# Meta-CoT V8 Active Plan

Updated: 2026-04-15

This is the only claim-bearing mainline plan for the `metacognition` repository.
Older V5/V6/V7 plans remain useful as historical notes, but they are not the execution contract.

## 0. Current Stage

Current mainline stage is:

1. strict paired data fixed
2. strict paired SFT completed from raw `Qwen/Qwen3-8B`
3. paired eval + behavior + entropy analysis completed (2026-04-12)
4. Phase 3 E21 anchor completed as `side_evidence` (step 43, stopped)
5. **Phase 4 E21R-v2 (2-head GDPO) completed — 300 steps, evaluated**
6. **Base GRPO completed — 300 steps, evaluated**
7. **Phase 5 next: self-distill D1/D2 + reward redesign**

As of 2026-04-15:

1. `data/v8_meta_inside_strict.parquet` and `data/v8_base_matched_strict.parquet` are the authoritative paired SFT datasets (4264 rows each, validator pass)
2. SFT checkpoints on HF: `iamseungpil/metacot` → `models/v8_meta_inside_strict_sft` (merged safetensors, 16.4 GB) and `models/v8_base_matched_strict_sft` (sharded safetensors, 16.4 GB)
3. RL checkpoints on HF: `checkpoints/verl_e21r_v2_0413/` (steps 190/220/250/latest=300, FSDP shards) and `checkpoints/verl_base_matched_0410/` (steps 50/70/150/200/240/latest=300, FSDP shards)
4. **Step 300 eval**: E21R-v2 79.81% vs Base 75.92% (+3.88pp). GSM8K tied, MATH500 +9.8pp, AIME -20.0pp
5. **Paired analysis**: Meta-only wins 117, Base-only wins 77, net +40 (+3.88pp)
6. **Fisher exact tests**: redirect/diagnosis/epistemic all significantly correlated with lower accuracy (p<0.001) in BOTH models
7. **Calibration**: AUROC 0.522 (random), confidence collapsed to 0.96 constant (98.9%)
8. **Root cause findings (2026-04-15)**:
   a. `<|meta|>` wrapping lost: reward fallback in `_parse_meta_blocks_with_spans` makes free-text confidence reward-equivalent to wrapped. RL dropped structural tokens as zero-value overhead
   b. Confidence collapse: calibration reward E[cal]=0.3c×(2×acc-1) → c→1 optimal when acc>50%. Structural flaw, not absence
   c. Template collapse: 908/1030 completions contain identical boilerplate assessment text
   d. AIME -20pp: redirect on hard problems → token budget exhaustion (85% wrong AIME hit 4096 limit)
   e. MATH500 +9.8pp: longer reasoning (avg +115 tokens) reaches correct answers
10. **Strict paired SFT verified rerun (2026-04-15)**: base strict SFT 75.51%, meta strict SFT 75.38%, OOD(AIME) 26.67% vs 16.67%. Meta controller structure is preserved (`meta_emission=99.94%`) but OOD gain is not present at the strict-SFT stage
11. **Retrieval contract clarification (2026-04-16)**: `fixed_k_repair` supports retrieval only when an example bank is supplied. Without `--example_bank`, mainline roundtrip must be recorded as `repair-only`, not RAG-enabled
9. **Self-distill code**: fair `fixed_k_repair` + selector provenance + claim-bearing synthetic-meta gate implemented; base/meta self-distill YAML and roundtrip launcher added
10. **6 H200 nodes available**: `node-recovery-h200-0415` with 6 jobs running

### Executive Summary

**이번 단계의 최우선 질문**

RQ3의 의도는 `OOD에서 self-distill을 성공시킬 수 있는가`이다.
이를 위해 immediate mainline은 RL을 더 키우는 것이 아니라,
`strict paired SFT -> fair fixed-K repair -> reward-ranked teacher selection -> short SFT readout`
으로 고정한다.

**이번 단계의 핵심 가설**

1. `H1`: naive self-distill은 in-domain gain이 있어도 `meta emission`, `wrong-high-confidence`, OOD accuracy를 망가뜨릴 수 있다
2. `H2`: meta SFT에서 `selected_completion` 기반 claim-bearing epistemic self-distill을 하면,
   naive baseline보다 controller retention과 OOD retention이 낫다
3. `H3`: reward-guided teacher selection은 무작위/trigger-earned repair보다 더 나은 distill target을 만든다
4. `H4`: RL은 immediate mainline이 아니라 side branch다.
   reward redesign은 self-distill 비교와 섞지 않고 별도 smoke track에서만 검증한다
5. `H5`: retrieval은 현재 mainline의 필요조건이 아니다. retrieval claim은 `example_bank loaded + retrieval_nonempty_rate > 0`가 동시에 만족될 때만 허용한다

**이번 단계의 실험 순서**

1. `E1`: `strict_base_sft -> fixed_k_repair -> naive self-distill`
2. `E2`: `strict_meta_sft -> fixed_k_repair -> claim-bearing epistemic self-distill`
3. `E3`: `E1 vs E2` collapse / OOD / controller retention 비교
4. `E4`: `selected_completion`에 대해 teacher top-k를 질의하고, `control-span-weighted KL` readout을 붙인 meta self-distill 확장 점검
5. `E5`: 별도 node에서 RL reward redesign smoke (`E21R-v3-smoke`) 진행

**이번 단계의 성공 기준**

1. `E1/E2`는 같은 strict split, 같은 root/repair budget, 같은 decode setting을 사용해야 한다
2. claim-bearing lane에서 `synthetic_meta_injected_rate == 0`
3. `E2`는 `E1` 대비 `meta_emission`과 `wrong_high_confidence`에서 더 낫거나 같아야 한다
4. `E2`는 `E1` 대비 OOD combined accuracy에서 `+2pp` 이상을 목표로 한다
5. 위 조건을 만족하지 못하면 `OOD self-distill 성공`이 아니라 `collapse analysis only`로 기록한다
6. retrieval-based gain은 별도 조건이다. `retrieval_nonempty_rate == 0`이면 그 run은 retrieval evidence로 쓰지 않는다

### Phase 5 Plan: Two-Track Parallel Execution

**Track A — Self-Distill Mainline (allowed nodes only: `metacognition_eval`, `metacognition_train_b`)**
1. Build fair `fixed_k_repair` artifacts from `strict_base_sft` and `strict_meta_sft`
2. Project base lane to `naive` baseline and meta lane to claim-bearing `epistemic`
3. Build teacher top-k targets only for the meta lane that already passed claim-bearing checks
4. Train:
   - base lane: CE/SFT only
   - meta lane: CE/SFT + control-span-weighted KL on wrapped meta / diagnosis / study_need / post-meta recovery spans
5. Evaluate with `analyze_self_distill_eval.py` for collapse / OOD / controller-retention metrics
6. Record retrieval contract in every artifact:
   - `retriever_active`
   - `retrieval_enabled`
   - `retrieval_nonempty_rate`
   Retrieval is considered active evidence only if these fields show actual retrieval use

**Track B — Reward Redesign Smoke (side-evidence only; runs only when mainline node window is free)**
1. Fix calibration reward to strict wrapped-only proper scoring
2. Add explicit meta-structure preservation reward so RL cannot collapse wrapped controller state into free text
3. Keep reward smoke isolated from claim-bearing self-distill tables
4. Record the exact reward entrypoint (`compute_score_e21r_v3_smoke`) and treat it as side-evidence until rerun cleanly

Reward redesign status (2026-04-15, local code patch in progress):
1. `outcome_calibration_reward` should use strict wrapped blocks only, not free-text fallback
2. `confidence_omission_floor` and `meta_count_bonus` should count only wrapped meta blocks
3. proper scoring rule should reward moving confidence closer to truth, not merely increasing confidence on correct samples
4. training entrypoint must be checked explicitly:
   - `src/training/verl_reward.py::compute_score_e21r_v2` is the 2-head path used by the historical E21R-v2 launcher
   - `src/training/verl_gdpo.py::REWARD_CONFIGS["E21R"]` is a different confidence-centered controller recipe
   - future claim-bearing runs must record which path was used

Uniform full-trace KL is **not** the target for this phase.
The intended dense objective is:
1. select a repair teacher with control-aware scoring
2. apply CE/SFT on the selected completion
3. apply KL only on control-critical spans

**Unavailable capacity for mainline**
1. `rsp_grpo_node_1`
2. `rsp_grpo_node_2`
3. `metacognition_e8`
4. `metacognition_run_c`

### Concrete Node Allocation

**Node plan for this phase**

1. `metacognition_train_b`
   - first priority: `E1` base self-distill mainline
   - inputs: `checkpoints/v8_base_matched_strict_sft`, strict paired train slice
   - artifact path: `results/self_distill/base_fixedk_naive/`
   - train config: `configs/sft_self_distill_base_fixedk_naive.yaml`
2. `metacognition_eval`
   - first priority after resume: `E2` meta self-distill mainline + missing eval-only analyses
   - inputs: `checkpoints/v8_meta_inside_strict_sft`, strict paired train slice
   - artifact path: `results/self_distill/meta_fixedk_epistemic/`
   - train config: `configs/sft_self_distill_meta_fixedk_epistemic.yaml`
   - KL extension config: `configs/sft_self_distill_meta_fixedk_epistemic_kl.yaml`
3. `metacognition_eval`
   - if still paused, local/remote analysis work cannot be claimed as running
   - resume or reconnect must be verified before launching missing analysis
4. RL reward redesign smoke
   - only after `E1/E2/E3` or during explicitly free node windows
   - do not mix smoke checkpoints into claim-bearing tables

### Immediate Runbook

1. generate `fixed_k_repair` artifacts
   - base: `scripts/run_fixed_k_self_distill_roundtrip.sh checkpoints/v8_base_matched_strict_sft <train_split> results/self_distill/base_fixedk_naive naive 0`
   - meta: `scripts/run_fixed_k_self_distill_roundtrip.sh checkpoints/v8_meta_inside_strict_sft <train_split> results/self_distill/meta_fixedk_epistemic epistemic 1`
   - note: if no example-bank path is appended, the launcher disables retrieval (`rag_top_k=0`, `retrieval_query_mode=none`) and the artifact must be labeled `repair-only`
2. verify artifact contract
   - base/meta row counts, candidate_count, selector provenance, `synthetic_meta_injected_rate`, retrieval summary
3. train SFT readout
   - base: `configs/sft_self_distill_base_fixedk_naive.yaml`
   - meta: `configs/sft_self_distill_meta_fixedk_epistemic.yaml`
4. optional meta-lane dense targets
   - `scripts/build_teacher_topk_targets.py`
   - output: `teacher_topk_targets.parquet`
   - if the parquet is missing or has 0 valid target rows, KL lane must fail closed rather than silently reverting to CE-only
5. evaluate only new outputs
   - `scripts/analyze_self_distill_eval.py`
   - OOD and controller-retention deltas vs strict SFT baselines
6. only after `E1/E2/E3` are saved, run RL smoke and side-evidence extensions

## 1. Research Contract

### RQ1. Meta-CoT

**의도**

Meta-CoT를 스타일 문구가 아니라 `confidence-conditioned controller`로 학습한다.
모델은 ordinary derivation과 분리된 meta state를 통해 현재 경로를 계속 밀지,
verify할지, redirect할지를 드러내야 한다.

**가설**

올바른 SFT만으로도 모델은 다음을 구조적으로 배울 수 있다.

1. high-confidence but weak-support 상황에서 `verify`
2. low-confidence or anomaly 상황에서 `redirect`
3. ordinary CoT와 meta control의 분리

**검증 방법**

1. meta emission rate
2. verify / redirect rate by scenario
3. wrong-high-confidence rate
4. trigger-conditioned correction rate
5. AIME / hard subset에서의 qualitative trace audit

**해석**

1. accuracy만 오르고 controller behavior가 안 바뀌면 `style gain`, not Meta-CoT
2. verify만 늘고 redirect가 안 늘면 `verify-heavy drift`
3. low-confidence redirect와 post-trigger correction이 늘면 controller learning evidence

### RQ2. Meta-RL

**의도**

meta behavior를 surface phrase가 아니라 `verifiable reward`로 강화할 수 있는지 본다.
핵심은 "메타처럼 말하느냐"가 아니라 "confidence에 따라 행동을 바꾸느냐"이다.
RQ2는 training-time utilization의 질문이다.

**가설**

correctness에 더해 적은 수의 해석 가능한 reward만으로도 controller behavior를 강화할 수 있다.
특히 confidence revision, redirect execution, gated verify, meta floor는
controller behavior를 분해 가능하게 유지한다.

**검증 방법**

1. reward head별 mean/variance
2. smoke cases에서 reward confusion 여부
3. RL 전후 redirect/verify/post-correction 변화
4. wrong-high-confidence 감소 여부
5. difficulty-conditioned behavior shift

**해석**

1. accuracy가 유지되어도 verify-only drift면 RQ2는 약하게만 지지된다
2. redirect trigger와 correction chain이 늘면 RQ2를 지지한다
3. reward 분해가 불가능하면 RL gain은 claim-bearing evidence로 쓰지 않는다
4. stepwise reward는 local credit assignment evidence이지 inference-time search evidence가 아니다

**가설 수정 (2026-04-13)**: Controller detection behavior (WHEN)는 SFT만으로 99.8% 달성됨.
RQ2의 초점을 "controller behavior 강화"에서 "redirect execution quality 개선"으로 이동.
새 가설: reward design 또는 training paradigm 변경을 통해 redirect가 실제 accuracy 개선으로 이어지게 할 수 있다.

### RQ3. Curriculum / OOD Test-Time Adaptation

**의도**

학습된 meta state가 어려운 문제에서 failure diagnosis를 만들고,
이후 retrieval / one-example adaptation / retry의 trigger로 재사용 가능한지 본다.
RQ3는 inference-time utilization의 질문이다.
RQ3의 핵심 프레이밍은 post-hoc calibration이 아니라
`controller-mediated information acquisition`이다:
모델이 현재 정보만으로 계속 밀지 말고, 추가 예시(retrieval), 추가 시도(retry),
추가 분기(branching) 중 무엇을 살 가치가 있는지 드러낼 수 있는가를 본다.

RQ3는 두 개의 하위 질문으로 분리한다.

1. `RQ3-A`: diagnosis-triggered curriculum / retrieval retry
2. `RQ3-B`: confidence-bucket selective branching (`MCTS-lite`, side-evidence)

**가설**

controller가 단순 verify가 아니라 explicit diagnosis와 study need를 남기면,
그 출력은 downstream curriculum / RAG의 reliable trigger가 될 수 있다.
또한 low-confidence / diagnosis signal이 충분히 신뢰 가능하면,
그 signal은 selective branching budget의 prior로도 재사용될 수 있다.
즉 meta state가 uncertainty report를 넘어서
`which extra information or computation is worth buying`를 선택하는 control interface가 될 수 있다.

현재 local audit에서 확인해야 하는 리스크도 명시한다.
RAG가 nominal하게는 `study_need`를 사용하더라도,
실제 top-k가 question lexical overlap만 따라가면 controller-mediated retrieval이라는 해석은 약해진다.
따라서 RQ3-A의 기본 설계는 `study_need family / diagnosis / strategy alignment`를
question surface similarity보다 앞에 두는 typed retrieval이어야 한다.

**검증 방법**

1. root completion 기준 diagnosis quality audit
2. `root -> trigger -> retrieval/adapt -> retry` 파이프라인 smoke
3. trigger precision / trigger coverage / false-trigger rate
4. retrieval retry correction gain vs plain retry control
5. retrieval 직후 `next meta` readout: confidence recovery / trigger clear / study-need followthrough
6. selective branching gain vs plain retry / retrieval retry
7. successful path logging quality for later self-distill
8. cost-aware readout: extra retrieval / extra branches 대비 marginal gain

**해석**

1. low confidence만 있고 diagnosis가 없으면 curriculum trigger로는 불충분
2. diagnosis와 next-strategy가 명시되면 curriculum-ready evidence
3. branching gain이 없으면 confidence는 search prior로만 약하게 취급하고, search claim은 보류한다
4. curriculum / retrieval / search 결과는 RQ1/RQ2를 대체하지 못한다
5. calibration-only improvement는 RQ3 novelty가 아니다. RQ3의 핵심은 structured meta state의 downstream reusability이다.
6. novelty framing은 post-hoc calibration보다 `diagnosis -> action selection`에 둔다. confidence 수치 자체만으로는 새로운 claim을 만들지 않는다.
7. extra computation의 비용 대비 gain이 없으면, RQ3는 유용한 controller가 아니라 단순 expensive fallback으로 해석한다.
8. retrieval이 final correctness만 올리고 `next meta`를 개선하지 못하면, 정보획득이 아니라 prompt luck 가능성을 먼저 의심한다.
9. 쉬운 예시를 무조건 넣는 것은 기본 정책이 아니다. easy exemplar는 primitive study_need class에서만 gated fallback으로 허용한다.

## 2. Evidence Classes

Every artifact, run, checkpoint, or report must be labeled as exactly one of:

1. `mainline`
2. `side_evidence`
3. `historical`
4. `invalid_for_claim`

Rules:

1. `mainline` means the run obeys the frozen initializer, paired data, and paired hyperparameter contract
2. `side_evidence` means the run may still be useful diagnostically, but cannot support the main claim
3. `historical` means the artifact is preserved for context only
4. `invalid_for_claim` means the run is broken, misconfigured, or missing critical provenance

## 3. Frozen Mainline Contracts

### 3.1 Strict Paired SFT Contract

Allowed differences between the paired SFT runs:

1. dataset path
2. output dir
3. run name

Frozen shared requirements:

1. initializer must be raw `Qwen/Qwen3-8B`
2. dataset rows must be paired one-to-one
3. user prompt must match row-by-row
4. final boxed answer target must match row-by-row
5. same optimizer family, LR, epoch count, max length, and batch schedule
6. both runs must preserve `<think> ... </think>` format

### 3.2 Future Paired RL Contract

RL may begin only after strict paired SFT finishes and the alignment checklist passes.

These values are aligned with Four Habits (Gandhi et al., 2025, COLM) PPO recipe adapted for Qwen3-8B on math reasoning. Key deviations from the earlier conservative defaults:

1. `response_length=4096` is required because AIME2024 completions in our Phase 2 eval showed p95 ≥ 4096 (13.3% of meta, 23.3% of base hit the cap at 4096). The earlier 2048 value truncated too aggressively to test redirect behavior.
2. `rollout.n=4` stabilizes GDPO group normalization; n=2 left the per-reward heads too noisy in smoke tests.
3. `kl_coef=0.001` and `lr=1e-6` match Four Habits defaults which produced stable policy updates on Llama-3.2-3B; scaled to 8B with no further change.
4. `save_freq=10, test_freq=10` gives dense learning-curve visibility without slowing the run (test is cheap on the paired redirect subset).

Frozen shared keys for claim-bearing paired RL:

1. `prompt_length=2048`
2. `response_length=4096`
3. `train_batch_size=64`
4. `actor.ppo_mini_batch_size=16`
5. `actor.ppo_micro_batch_size_per_gpu=1`
6. `actor.ppo_max_token_len_per_gpu=16384`
7. `critic.ppo_mini_batch_size=16`
8. `critic.ppo_micro_batch_size_per_gpu=1`
9. `critic.ppo_max_token_len_per_gpu=32768`
10. `rollout.n=4`
11. `learning_rate=1e-6`
12. `kl_coef=0.001`
13. `temperature=0.7`
14. `top_p=0.95`
15. `rollout.tensor_model_parallel_size=2`
16. `rollout.gpu_memory_utilization=0.4`
17. `rollout.log_prob_micro_batch_size_per_gpu=16`
18. `ref.log_prob_micro_batch_size_per_gpu=16`
19. `total_training_steps=300`
20. `save_freq=10`
21. `test_freq=10`
22. `remove_previous_ckpt=False`

Allowed paired-run differences:

1. model checkpoint
2. parquet/data definition
3. reward function
4. algorithm only when the plan explicitly treats it as the experimental variable

Any RL run that violates these keys is not `mainline`.

## 4. Phase Plan

### Phase 0. Strict Paired Data

**의도**

Meta lane과 base lane이 정확히 같은 문제 슬라이스를 보도록 만든다.

**가설**

paired data parity를 강하게 보장하면 이후 SFT/RL 차이를 controller effect로 더 깔끔하게 해석할 수 있다.

**검증 방법**

1. row count equality
2. prompt equality
3. boxed answer equality
4. redirect/verify scenario counts
5. strict validator pass

**해석**

pair가 깨져 있으면 이후 paired comparison은 무효다.

Current authoritative artifacts:

1. `data/v8_meta_inside_strict.parquet`
2. `data/v8_base_matched_strict.parquet`
3. `results/strict_data/v8_strict_validation_summary.json`

### Phase 0b. Data Expansion Branches

**의도**

strict paired SFT가 너무 작은 데이터 때문에 controller representation을 충분히 못 배우는지
분리해서 검증한다. 다만 expansion branch가 active eval benchmark를 오염시키면 안 된다.

**가설**

현재 strict paired 4264쌍은 clean anchor로는 충분하지만,
redirect / diagnosis richness를 더 키우려면 hard source 확장이 도움이 될 수 있다.

**검증 방법**

1. expansion 전후 paired row count 비교
2. source / topic / difficulty coverage 비교
3. redirect low-confidence rate / verify high-confidence rate 유지 여부
4. strict validator pass 여부
5. post-SFT에서 meta emission / redirect rate / AIME qualitative shift 비교

**해석**

1. 데이터만 늘고 control semantics가 흐려지면 expansion은 기각한다
2. redirect richness가 늘고 strict paired behavior가 개선되면 accepted side branch가 된다

**허용 분기**

1. `mainline-safe expansion`
   - source: `gsm8k`, `EleutherAI/hendrycks_math`, `KbsdJames/Omni-MATH`
   - generation: `scripts/gen_control_v5_trapi.py` + `src/metacot/prompt_control_v5.py`
   - QC: `scripts/qc_control_v5_samples.py`
   - strict rebuild: `scripts/build_v8_strict_paired_data.py`
   - note: `Omni-MATH` 보강은 허용된다
2. `benchmark-contaminated side branch`
   - source includes `math500` or any active held-out eval set
   - evidence class must be `side_evidence`
   - such data may be used only for diagnostic or contamination-ablation runs
   - it must not replace the claim-bearing strict paired SFT data

**금지 규칙**

1. `math500`는 active eval benchmark이므로 claim-bearing SFT/RL training data에 넣지 않는다
2. `aime2024`도 동일하게 mainline training source로 쓰지 않는다
3. expansion branch는 항상 새 parquet 이름으로 저장하고 기존 strict parquet를 덮어쓰지 않는다

### Phase 1. Strict Paired SFT

**의도**

raw base에서 시작하는 clean paired SFT로 controller representation이 생기는지 본다.

**가설**

strict meta SFT는 strict base SFT보다 더 높은 meta emission과 더 좋은 trigger-conditioned verify/redirect behavior를 보인다.

**검증 방법**

1. benchmark accuracy
2. meta block emission
3. scenario-conditioned behavior rates
4. AIME qualitative traces
5. completion length and confidence statistics

**해석**

1. meta emission only without action shift: style imitation
2. action shift with accuracy preservation: good evidence for RQ1

Current active runs:

1. `metacognition_eval` -> `v8_meta_inside_strict_sft`
2. `metacognition_train_b` -> `v8_base_matched_strict_sft`

Fallback / follow-up branches after strict SFT readout:

1. `Branch S1`
   - keep current strict 4264 paired data as the sole mainline anchor
2. `Branch S2`
   - if controller behavior is weak but emission is healthy, expand only hard safe sources (`Omni-MATH` + held-in Hendrycks) via TRAPI and rerun paired SFT as `side_evidence`
3. `Branch S3`
   - if meta emission itself collapses relative to `E20a`, inspect formatting / truncation / data filtering first before any reward or RL change

### Phase 2. Paired Eval and Behavior Analysis

**의도**

strict SFT 결과를 math accuracy뿐 아니라 behavior, confidence, entropy 관점에서 본다.

**가설**

true Meta-CoT는 단순한 token count 변화가 아니라 confidence-conditioned behavior 변화를 남긴다.

**검증 방법**

1. `gsm8k`, `math500`, `aime2024`
2. accuracy / ECE / confidence-accuracy correlation
3. verify / redirect / diagnosis / subgoal / backward chaining rates
4. AIME hard subset qualitative audits
5. entropy split before/after meta
6. paired eval must be deterministic (`do_sample=False`) and saved as JSON + metadata + parquet

**해석**

1. easy 문제에서 verify가 높고 hard 문제에서 redirect가 높아야 controller 해석이 자연스럽다
2. hard subset에서도 redirect가 안 뜨면 later RL target remains redirect

Stage-2 artifact bundle:

1. paired eval bundles for strict meta/base checkpoints
2. confidence report
3. behavior summary + critic pass
4. AIME qualitative casebooks
5. entropy analysis for the strict meta lane
6. explicit comparison against `E20a` on:
   - meta emission rate
   - multi-meta rate
   - confidence rate
   - redirect / verify lexical + behavioral rates

### Phase 3. RL Anchor (completed as side_evidence, 2026-04-13)

**의도**

historical E21 (6-head GDPO) anchor와 paired base GRPO를 직접 비교하여
multi-reward meta RL의 baseline 성능을 확인한다.

**가설**

E21의 6 reward heads (correctness + switch_v2 + verify_v2 + conf_traj + meta_floor + meta_count_bonus)는
meta behavior를 유도하지만, reward 수가 GDPO 논문 권장 (2-3)을 초과하여
correctness gradient가 dilute될 수 있다.

**검증 방법**

1. identical paired RL budget (shared hyperparameters, Section 3.2)
2. step-matched validation comparison (meta vs base at same step)
3. per-topic accuracy trajectory
4. reward head별 gradient 기여 비율

**결과 (2026-04-13, `side_evidence`)**

E21 ran 43 steps, stopped due to declining validation after step 30.

Step 30 공정 비교 (같은 step, 같은 validation set):

| 토픽 | Meta E21 @30 | Base GRPO @30 | 차이 |
|---|---|---|---|
| algebra | 76.5% | 78.4% | -1.9pp |
| prealgebra | 67.5% | 82.5% | -15.0pp |
| number_theory | 67.9% | 64.3% | +3.6pp |
| 전체 평균 | 41.8% | 48.0% | **-6.2pp** |

**해석**

1. E21 6-head GDPO는 base correctness-only GRPO보다 -6.2pp 열등 (step 30 기준)
2. 원인 분석: (a) correctness weight 비율 32% (6 head 중 1), (b) meta block이 constraint forgetting 유발, (c) response budget 36%+ 낭비
3. 실제 응답 검수: meta interrupt 후 문제 constraint 누락 확인 (예: digit ≤ 9 조건 빠뜨림)
4. GDPO 논문 자체가 2-3 heads 권장 — 6 heads는 advantage collapse 위험
5. E21은 `side_evidence`로 분류. mainline claim에는 사용하지 않음
6. 교훈: meta reward를 많이 넣을수록 좋은 게 아님. correctness 지배력 유지가 핵심

### Phase 4. Reward Comparison Ladder

The reward ladder is intentionally staged so that each reward family answers a different
training-time hypothesis under RQ2. These are reward-family comparisons, not inference-time
adaptation methods.

#### E21. Historical Anchor

**의도**

historical reward family를 clean anchor로 남긴다.

**가설**

`correctness + switch_v2 + verify_v2 + conf_traj + meta_floor`는 accuracy를 유지하지만 verify-heavy drift를 만들 수 있다.

**추가 reward head (2026-04-12)**

`meta_count_bonus`는 strict meta SFT eval에서 발견된 single-step collapse (1559/1560 samples used exactly 1 meta block)를 직접 targeting한다. Block 개수에 per-block monotonic reward (1 block=0.1, 2=0.2, 3=0.3 cap)를 부여하지만, 내용 품질은 gating하지 않으므로 `meta_floor`와 함께만 의미가 있다.

가설: `meta_count_bonus`가 없으면 RL이 SFT single-block distribution 근처에 머무르며, 있으면 모델이 2-3 block trajectories를 탐색하기 시작한다. 이는 RL이 SFT distribution shift를 만들 수 있는지에 대한 직접 테스트이다.

해석: 만약 meta_count_bonus가 추가되어도 >1 block rate가 20% 미만이면, data bottleneck이 확정되며 Phase 0b expansion branch S2로 fallback한다.

**검증 방법**

verify concentration, redirect rarity, hard subset correction chain.

#### E21R. Confidence-Centered Redirect Controller (superseded by E21R-v2)

**의도**

confidence revision과 redirect execution을 분리해서 redirect controller를 직접 겨냥한다.

**가설**

historical E21보다 low-confidence redirect behavior를 더 직접적으로 만든다.

**검증 방법**

wrong-high-confidence 감소, low-confidence redirect 증가, post-trigger correction.

**status**: superseded. E21R의 5-head 구조는 E21과 같은 gradient dilution 위험이 있다. E21R-v2로 대체.

#### E21R-v2. Outcome-Calibration Controller (2026-04-13, active)

**의도**

GDPO 논문의 2-3 head 권장에 따라, correctness를 지배적으로 유지하면서
confidence calibration만 보조 신호로 사용한다.
Meta behavior를 직접 강제하지 않고, 모델이 "정답을 위해 meta가 필요하면 자연스럽게 사용"하게 한다.

**가설**

1. 2-head GDPO (correctness 77% + outcome_calibration 23%)는 E21의 6-head보다
   correctness 학습 속도가 빠르고, base GRPO 대비 competitive하다.
2. outcome_calibration의 trajectory component (conf drop→recovery→correct = +0.1 bonus)는
   meta를 강제하지 않으면서도, meta가 도움되는 상황에서 multi-meta 사용을 자연스럽게 유도한다.
3. wrong-high-confidence penalty (-0.3 × last_conf)가 과신 오답을 줄인다.
4. meta_floor는 GDPO head에서 제외하고 combined score에만 약하게 (0.3) 추가하여
   meta 존재를 최소한으로만 권장한다.

**Reward 구조**

```
GDPO heads (2):
  correctness              × 1.0   range [-1, +1]
  outcome_calibration      × 1.0   range [-0.4, +0.4]

Combined score에만 추가 (GDPO normalization 밖):
  meta_floor               × 0.3   range [-0.5, 0]
```

outcome_calibration 계산:
- Endpoint: `correct → +0.3 × last_conf`, `wrong → -0.3 × last_conf`
- Trajectory (multi-meta only): `correct + conf rise → +0.1`, `wrong + conf drop → +0.1`, `wrong + conf rise → -0.1`

**검증 방법**

1. step-matched validation comparison: E21R-v2 vs Base GRPO at step 30, 50, 100
2. correctness trajectory: base와 비슷한 속도로 개선되는가?
3. wrong-high-confidence rate: SFT baseline 대비 감소하는가?
4. confidence trajectory quality: correct 답에서 conf 상승, wrong 답에서 conf 하락이 나타나는가?
5. multi-meta 자연 발생: meta_count_bonus 없이도 >1 block rate가 증가하는가?
6. constraint forgetting: meta interrupt 후 problem constraint 누락 빈도가 E21보다 줄어드는가?

**결과 (2026-04-13, interim at step 50)**

| Metric | E21R-v2 | Base GRPO | vs E21 (6h) |
|---|---|---|---|
| Val @step30 | 43.8% | 48.0% | +2.0pp |
| Val @step50 | 44.9% | ~50% (est) | +3.1pp |
| Base gap @step30 | -4.2pp | — | improved from -6.2pp |

Validation trajectory: 39.9% → 39.4% → 40.5% → 43.8% → 42.2% → 44.9% (step 0→50, oscillating 40-45%)

Additional behavioral analysis (SFT eval baseline):
1. Redirect success rate: 47.7% = hard-problem baseline → redirect does not improve accuracy
2. Diagnosis diversity: 99% single template ("What is missing is...")
3. Post-redirect structure: backtracking -17.4pp, multiple attempts -23.5pp, short redirect 55% vs long 9%
4. Key insight: model knows WHEN to redirect (99.8%) but not HOW (redirect ≈ random on hard problems)

해석: E21R-v2 해석 조건 2 ("base보다 열등하면 meta format 재설계") 충족됨. Meta overhead > calibration benefit. Redirect execution quality가 핵심 병목.

**해석**

1. E21R-v2가 base와 동등하거나 우위면: calibration 신호가 meta에 실제로 도움을 줌
2. E21R-v2가 base보다 열등하면: meta overhead가 calibration 이득보다 큼 → meta format 자체를 재설계해야 함
3. multi-meta가 자연 발생하면: 강제 보상 없이도 meta 활용 가능 → RQ2 지지
4. multi-meta가 여전히 안 생기면: SFT data의 single-block distribution이 hard constraint → Phase 0b 필요

**E21 → E21R-v2 전환 근거 (2026-04-13)**

| E21 (6 heads) | E21R-v2 (2 heads) |
|---|---|
| correctness 비중 32% | **correctness 비중 77%** |
| regex 기반 switch/verify/conf | **순수 수치 calibration** |
| meta_count_bonus → reward hacking | **meta 직접 보상 없음** |
| step 30 peak → step 40 하락 | 기대: 안정적 상승 |
| base 대비 -6.2pp @step30 | 기대: base와 competitive |

#### E21S. Stepwise Confidence Delta

**현재 상태 (2026-04-13, deferred)**: E21R-v2 결과에서 redirect content가 단일 템플릿(99%)이고 redirect success rate가 baseline과 동일(47.7%)함이 확인됨. Dense blockwise reward를 적용해도 content가 빈약한 meta block에 credit을 할당하는 것은 의미가 없음. Redirect execution quality 해결 후 재개.

**의도**

meta block별 local intervention에 dense reward를 주어, endpoint-only trajectory reward보다
촘촘한 credit assignment를 제공한다.

**가설**

diagnosis에서 confidence drop, successful verify/redirect 이후 calibrated recovery가 보이면
endpoint-only reward보다 controller attribution이 좋아진다.
단, 이것은 heuristic blockwise dense reward이지 rollout-based value learning이나 MCTS는 아니다.

**검증 방법**

1. blockwise confidence slope
2. trigger-conditioned drop
3. recovery-conditioned correctness
4. stepwise reward confusion smoke

#### Phase 4b. Redirect Execution (new, 2026-04-13)

**의도**

Model이 WHEN to redirect를 이미 학습했으므로 (99.8%),
HOW to redirect (redirect 이후 실제로 답을 개선하는 방법)를 학습시킨다.
Redirect success rate가 hard-problem baseline (47.7%)을 넘는 것이 목표.

**가설**

1. 현재 redirect 실패 원인은 diagnosis content의 빈약함 (99% 단일 템플릿)과
   post-redirect continuation의 과도한 길이 (backtracking -17pp, multiple attempts -23pp)
2. 짧고 구체적인 redirect가 더 효과적임 (55% vs 9%)
3. SCoRe-style 2-turn training에서 turn-2 correctness reward만으로도
   redirect content quality가 개선될 수 있음

**실험 후보 (우선순위순)**

4b-A. Short-Redirect Constraint
- meta block을 50 token으로 cap, 초과 시 penalty
- 가설: 길이 제한이 모델을 구체적 subgoal 생성으로 유도
- 비용: E21R-v2와 동일 compute

4b-B. SCoRe-Style Two-Turn Correction
- Turn 1: 문제 풀기 (meta SFT 출력)
- Turn 2: Turn 1 결과를 입력으로 받아 수정
- Reward: Turn 2 correctness만 reward
- 비용: response length 2x → rollout.n=2로 축소 필요

4b-C. Subgoal-Based Redirect Data
- meta block이 concrete next subgoal을 생성하도록 SFT data 재구축
- 비용: data generation 1-2일, SFT retrain 0.5일

4b-D. Behavior-Uncertainty Curriculum
- behavior-uncertainty repo의 diverse redirect template을 SFT data에 혼합
- 비용: data extraction + SFT 1일

**검증 방법**

1. Redirect success rate: 47.7% baseline을 유의미하게 초과하는가? (target >55%)
2. Diagnosis diversity: unique template 비율이 50% 이상인가?
3. Post-redirect accuracy: redirect 사용 시 비사용 대비 accuracy 개선이 있는가?
4. Overall accuracy: base GRPO 대비 competitive한가?

**해석**

1. Redirect success rate > 55% + base-competitive accuracy: redirect execution 해결, E21S/E21M 재개 가능
2. 모든 4b 실험에서 redirect success rate <= baseline: meta block의 corrective value에 대한 negative evidence → meta format을 verify-only로 축소 권고

### Phase 5. Curriculum / RAG

**의도**

controller output을 downstream retrieval/adaptation trigger로 사용한다.
이 phase는 RQ3의 시작점이며, RQ2 reward comparison에서 효과가 확인된 뒤에만
claim-bearing downstream evidence로 다룬다.
Phase 5 안에서도 두 lane을 분리한다:
`E21M`은 diagnosis-triggered retry의 mainline downstream readout,
`MCTS-lite`는 confidence-bucket branching의 side-evidence readout이다.

**가설**

diagnosis quality가 충분하면 retrieval이 low-confidence-only trigger보다 더 정확하다.
그리고 diagnosis-conditioned retrieval이 plain retry보다 consistently 낫다면,
meta state는 useful information request로 해석할 수 있다.

**검증 방법**

1. trigger precision
2. trigger coverage
3. false-trigger rate
4. diagnosis completeness
5. retrieved example relevance
6. plain retry 대비 correction gain
7. study_need field completeness
8. retrieval cost 대비 gain (`gain per triggered retrieval`)

**해석**

curriculum은 mainline RL evidence가 나온 뒤의 downstream phase다.
지금 시점의 curriculum code는 준비 상태를 점검하되 claim-bearing evidence로 쓰지 않는다.
따라서 Stage 6의 첫 readout은 반드시 `plain retry` control을 포함해야 하며,
retrieval retry gain이 control보다 낫지 않으면 단순 prompt-length 효과로 해석한다.
또한 diagnosis를 붙인 retrieval이 low-confidence-only retrieval보다 낫지 않으면,
`study_need`의 정보적 가치는 아직 입증되지 않은 것으로 본다.

#### E21M. Multi-turn Retry

**의도**

`diagnose -> study_need -> retry`를 single-turn controller에서 downstream adaptation loop로 확장한다.
이 lane은 "예시를 넣어주는 방식"의 직접적인 실험축이다.

**가설**

RQ2에서 학습된 controller가 diagnosis quality를 충분히 확보하면, multi-turn retry는
retrieval / adaptation gain을 만드는 첫 번째 downstream mechanism이 된다.
특히 `study_need`가 query를 구조화하면, 단순 유사문제 retrieval보다 더 적절한 예시를 가져올 수 있다.
반대로 retrieval이 lexical problem match에만 머물면, hard/OOD에서 전략 mismatch를 불러올 수 있다.

**검증 방법**

1. turn-1 diagnosis quality
2. turn-2 correction gain
3. retrieval justification quality
4. retry 이후 confidence / answer revision consistency
5. `next meta`의 confidence recovery / trigger clear / study_need followthrough
6. plain retry control 대비 추가 이득
7. low-confidence-only retrieval 대비 추가 이득

**운영 계약**

1. E21M은 항상 `root`, `plain_retry`, `retrieval_retry` 세 결과를 같은 문제에 대해 저장한다
2. root에서 trigger가 없으면 retrieval retry를 강제로 하지 않는다
3. retrieval retry가 이겨도 retrieved example provenance를 반드시 저장한다
4. one-example adaptation은 별도 side branch로 기록하며, retrieval retry와 혼동하지 않는다
5. retrieval 평가는 question 유사도만이 아니라 solution / method / study_need 정렬을 함께 기록한다
6. example bank는 `stable_seed_library`와 `dynamic_success_library`를 분리하며, 후자는 corrected trace만 필터링해서 append한다
7. retrieval 기본 정책은 `typed study_need retrieval`이다:
   a. 먼저 `study_need family`와 strategy type으로 candidate pool을 줄인다
   b. 그 다음 diagnosis / solution / question overlap으로 re-rank한다
8. easy exemplar는 default가 아니라 gated fallback이다:
   a. `arithmetic_translation`, `exponential_growth`, 일부 `probability_counting`처럼 primitive study_need에서만 허용
   b. AIME/olympiad geometry/invariant 계열에는 benchmark-easy exemplar를 우선 삽입하지 않는다
9. teacher-only RAG를 self-distill 기본 경계로 둔다:
   a. teacher는 diagnosis + study_need + retrieved exemplar를 볼 수 있다
   b. student는 plain problem만 본다
   c. distill 대상은 exemplar 자체가 아니라 exemplar를 사용해 회복한 reasoning trace다

#### MCTS-lite. Confidence-Bucket Search

**의도**

meta block이 등장한 중간 prefix를 decision point로 보고, confidence bucket
(`low`, `mid`, `high`)에 따라 branching budget을 달리하는 inference-time search를 실험한다.
이 lane은 "confidence가 추가 계산의 prior가 될 수 있는가"를 보는 축이다.

**가설**

RQ2에서 이미 학습된 controller가 low-confidence와 diagnosis를 안정적으로 드러낸다면,
그 signal은 selective branching policy의 trigger가 될 수 있다.
단, raw confidence 자체가 value를 제공하는 것은 아니며,
branch scoring은 downstream completion quality와 repair evidence를 함께 봐야 한다.
즉 branching의 목표는 confidence를 믿는 것이 아니라,
confidence를 이용해 expensive search를 sparsely 배치하는 것이다.

**검증 방법**

1. low-confidence node trigger precision
2. confidence bucket별 branch budget utilization
3. branch usefulness vs wasted expansion
4. best-branch gain vs plain retry
5. best-branch gain vs retrieval retry
6. successful path logging quality for later self-distill
7. wasted expansion rate / gain-per-extra-branch
8. heuristic branch score breakdown audit

**해석**

1. MCTS-lite는 RQ3의 side branch이며, RQ2 reward family와 같은 축으로 비교하지 않는다
2. raw self-reported confidence는 branch prior일 수는 있어도, 단독 value로 간주하지 않는다
3. 구현은 `src/curriculum/mcts_lite.py`에 두며, mainline launch path와 분리한다
4. branching gain이 retrieval retry보다 일관되게 크지 않으면, search overhead를 정당화하지 않는다

#### RQ3-C. Search-to-Learn Distillation (deferred side branch)

**의도**

retrieval retry와 selective branching에서 얻은 successful path를 나중의 학습 신호로 재사용할 수 있는지 본다.
이 lane은 search와 example injection을 학습으로 되돌리는 연결고리다.

**가설**

RQ3-A/B에서 얻은 successful repaired trace를 모으면,
단순 root trace보다 richer한 `when-to-intervene / how-to-recover` 학습 신호를 만들 수 있다.

**검증 방법**

1. successful repaired trace rate
2. repaired trace diversity
3. distill dataset quality audit
4. subsequent SFT side-branch gain

**해석**

1. 이 lane은 RQ3의 확장 연구축이며, RQ1/RQ2의 mainline claim과 분리한다
2. retrieval/search가 성공해도 distill 이후 재현되지 않으면 training signal로서의 가치는 약하다

#### RQ3-D. Epistemic Self-Distillation (priority side branch after RQ2)

**의도**

RQ2 paired readout 직후, meta-cognition 모델에 self-distillation을 적용했을 때
epistemic verbalization이 붕괴하는지 먼저 확인한다.
그 다음, uncertainty를 보존하는 self-distillation 설계를 추가해
OOD 문제에서도 `diagnosis -> information acquisition -> justified recovery` 패턴을
학습시킬 수 있는지 본다.

이 branch의 목적은 단순히 self-distill을 하나 더 넣는 것이 아니다.
먼저 `naive distill이 왜 붕괴하는지`를 우리 controller 관점에서 계측하고,
그 다음 `meta-cognitive intervention trace를 함께 distill`할 때
붕괴를 줄일 수 있는지 검증하는 것이다.

이 lane은 두 단계로 분리한다.

1. `RQ3-D1`: naive self-distill collapse check
2. `RQ3-D2a`: offline epistemic-preserving control distill
3. `RQ3-D2b`: feedback-conditioned OOD recovery distill using teacher-only RAG / side-evidence provenance
4. `RQ3-D3`: optional dense token distill only after D1/D2 data contract is validated

**우선순위 순서 (2026-04-15 revised)**

immediate claim-bearing 비교는 `strict_base_sft`와 `strict_meta_sft`에서 시작해야 하며,
현재 가장 먼저 해야 할 일은 `공정한 fixed-K repair + reward-ranked teacher selection + SFT`를
안정화하는 것이다. SDPO / OPD 계열은 그 다음 단계의 확장으로 둔다.

1. `P1`: strict paired SFT에서 fair `fixed_k_repair` artifact collection
   - 두 모델 모두 같은 problem ids, 같은 root decode budget, 같은 `K` repair budget을 쓴다
   - retrieval은 trigger-earned path가 아니라 `question_only` 또는 완전 비사용으로 고정한다
   - selection reward는 correctness + controller-execution terms만 사용하고 `meta_count_bonus`는 selector에서 제외한다
   - 결과 row에는 `repair_candidates`, `selected_candidate_id`, `selection_score_total`, `score_margin`을 남긴다
2. `P2`: reward-ranked selected trace를 messages parquet로 투영하여 short SFT readout
   - immediate matrix는 `strict_base_sft -> self-distill` vs `strict_meta_sft -> self-distill`이다
   - 이 단계는 claim-bearing lane이며 synthetic meta 주입을 금지한다
3. `P3`: collapse / OOD / controller retention readout
   - `meta_emission`, `wrong_high_confidence`, diagnosis specificity, OOD accuracy를 함께 본다
   - fair budget에서 meta lane이 실제로 OOD self-distill에 유리한지 먼저 판단한다
4. `P4`: side-evidence `sdpo_regen` artifact collection
   - privileged teacher feedback, retrieval provenance, `study_need`-conditioned retry를 저장한다
   - 이 lane은 claim-bearing 비교와 분리된 side-evidence lane이다
5. `P5`: teacher top-k query and dense token extension
   - `sdpo_regen` 또는 selected repair traces에 대해 teacher top-k target을 수집한다
   - short readout이 통과한 뒤에만 OPD-style dense objective로 넘어간다
6. `P6`: full OPD / SDPO integration
   - `verl` loop 안의 token-wise distillation은 마지막 단계다

즉 immediate mainline은
`strict paired SFT -> fixed_k_repair -> reward-ranked selection -> short SFT readout`
이고, `sdpo_regen -> top-k -> verl integration`은 side-evidence 확장이다.

**가설**

정답 trace만을 모방하는 naive self-distill은 in-domain에서는 답을 더 짧고 confident하게 만들 수 있지만,
hard / OOD 문제에서는 uncertainty expression과 recovery behavior를 억누를 수 있다.
반대로 self-distill teacher를 `root failure -> meta diagnosis -> intervention evidence -> next-meta recovery`
형태로 구성하면, answer trace만이 아니라 epistemic control pattern도 보존할 수 있다.
다만 이것만으로는 SDPO-style의 feedback-conditioned distillation이 되는 것은 아니다.
실제로 OOD self-distill이 성공했다고 주장하려면, teacher가 본 privileged feedback
(예: teacher-only RAG exemplar, branch-side evidence)가 데이터 contract에 남아 있고,
그 feedback이 없는 naive/distill baseline보다 OOD에서 더 잘 유지되어야 한다.

보조 가설은 다음과 같다.

1. naive self-distill의 핵심 위험은 accuracy 하락 자체보다 `meta emission`, `diagnosis specificity`, `wrong-high-confidence` 악화이다
2. epistemic-preserving self-distill은 정답 reasoning만이 아니라 `언제 개입했고 왜 개입했는지`를 같이 남겨야 한다
3. OOD 성공 여부는 `epistemic wording 보존`만으로 충분하지 않고, `feedback-conditioned recovery advantage`가 실제로 보여야 한다
4. dense token distill은 데이터 contract가 맞을 때만 의미가 있다. teacher target이 epistemically collapsed되어 있으면 token-wise KL은 붕괴를 더 빠르게 복제할 수 있다
5. immediate claim-bearing lane에서는 `repair opportunity` 자체가 meta model에 유리하게 주어지면 안 된다.
   따라서 trigger-gated retrieval path는 mainline selector 생성기로 쓰지 않는다.

**구현 가이드**

1. `D1`은 successful repaired trace 또는 hint-conditioned correct trace를 teacher demonstration으로 삼는 baseline이다
   immediate implementation은 `fixed_k_repair` selected trace를 사용한다
2. `D2a/D2b` teacher context에는 반드시 다음을 포함한다:
   a. root meta state
   b. failure diagnosis
   c. study_need 또는 missing perspective
   d. retrieval / branching evidence
   e. next-meta recovery summary (`confidence_gain`, `trigger_cleared`, `low_confidence_cleared`)
3. `D2a`는 offline repaired trace imitation baseline이다.
   이 lane은 epistemic controller retention을 보는 것이 목적이며, SDPO-style feedback distill을 주장하지 않는다.
4. `D2b`는 OOD recovery lane이다.
   teacher completion 외에도 teacher가 실제로 사용한 privileged feedback provenance를 dataset row에 남겨야 하며,
   현재 구현 기준으로는 `teacher_feedback_kind`, `teacher_feedback_context_json`으로 기록한다.
   다만 provenance를 저장만 하고 student input에 넣지 않으면 D2b가 아니라 D2a 변형으로 취급한다.
   따라서 offline D2b의 최소 구현 단위는 `sdpo_regen` message projection이며,
   root failure, diagnosis, study_need, teacher-side evidence를 user/system context에 실제로 넣는다.
   direct plain eval과 계약을 맞추기 위해 first implementation은 user-conditioned prompt를 우선하며,
   system-only instruction에만 의존하는 schema는 피한다.
5. 학습은 두 층으로 분리한다:
   a. offline SFT self-distill: teacher-generated repaired dataset을 직접 모방
   b. on-policy dense token distill: SDPO-style teacher-conditioned token distillation
   c. implementation order는 `live on-policy regeneration artifact -> token-wise KL` 순서로 둔다.
      즉 먼저 student가 실제로 root attempt를 생성하고, evidence를 보고 같은 모델이 다시 푼 `sdpo_regen` artifact를 저장한 뒤,
      그 다음에만 dense teacher-logit distillation을 붙인다.
6. dense token distill은 scalar outcome reward를 token별로 나누는 방식이 아니라,
   teacher-conditioned next-token distribution에 대한 token-wise KL / distillation loss로 구현한다
7. `D2a/D2b` dataset에는 root가 원래 맞았던 쉬운 문제를 대량으로 넣지 않는다. collapse check가 흐려지기 때문이다
8. 첫 구현은 offline SFT projection을 우선한다. 즉, 공통 IR을 만든 뒤
   a. `messages` parquet로 투영해 현행 `src/training/sft.py`로 학습 가능하게 만들고
   b. claim-bearing lane은 `selected_completion` + selector provenance를 그대로 남기고 synthetic meta를 금지한다
   c. 같은 IR에서 later `prompt + privileged teacher context + ground_truth` 형태로 SDPO-style distill 경로로 확장한다
   d. 현재 구현 기준으로는 `scripts/run_online_sdpo_regen.py`가
      `fixed_k_repair`와 `sdpo_regen` 두 artifact path를 모두 만든다
9. `D2a/D2b`의 epistemic 보존은 새로운 rigid schema를 강제하지 않는다.
   기존 control-v5 분포를 유지해야 하므로 meta block 안에는 자연어 diagnosis와 `confidence:`, `study_need:` 정도만 유지한다
10. OOD readout은 최소 `aime2024`를 포함하고, 가능하면 `omni_math` 또는 `openmath_cot`을 추가한다.
    OOD 성공 판정은 반드시 같은 train budget의 `D1`과 직접 비교한다.

**선행연구와 차별점**

1. `arXiv:2603.24472`는 self-distillation이 math reasoning에서 epistemic verbalization을 억누를 수 있다고 분석한다.
   우리 D1은 이 붕괴가 현재 Meta-CoT controller에서도 재현되는지 직접 확인하는 lane이다.
2. `arXiv:2603.15500`은 uncertainty verbalization이 정보 획득과 연결된다는 분석축을 제공한다.
   우리 D2는 이를 단순 분석이 아니라 `diagnosis -> intervention -> recovery` trace distillation으로 operationalize한다.
3. `arXiv:2601.20802` SDPO는 privileged teacher context를 활용한 token-wise distillation 구현 근거를 제공한다.
   우리는 이를 바로 main branch로 넣지 않고, 먼저 offline D1/D2 data contract를 고정한 뒤 optional D3로 붙인다.
4. 최근 self-distill / entropy-preserving 계열은 overconfidence와 diversity 감소를 완화하려 하지만,
   본 계획의 직접적 novel point는 `meta-cognitive controller behavior retention`을 분리 계측한다는 점이다.
5. `thunlp/OPD` 공개 구현은 `verl` 위에서 student on-policy rollout에 대해
   teacher token signal, top-k 전략, probability weighting을 붙이는 recipe를 제공한다.
   현재 우리 코드는 이 full objective까지는 아니고, 그 직전 단계인 `live regeneration artifact`까지 구현된 상태다.

**검증 방법**

1. naive self-distill 전후 meta emission rate
2. wrong-high-confidence rate
3. response length compression
4. hard / OOD accuracy (`aime2024` required, `omni_math` or `openmath_cot` preferred)
5. retrieval trigger precision
6. retrieval 후 next-meta recovery rate
7. plain self-distill 대비 epistemic-preserving self-distill의 추가 이득
8. in-domain gain과 OOD retention의 Pareto 비교
9. diagnosis specificity: generic template 비중 vs concrete failure diagnosis 비중
10. study_need preservation rate
11. confidence-bin calibration (`ECE`, reliability bins)
12. retrieval / branching side pipeline에 넣었을 때 trigger semantics가 유지되는지
13. `D2b`에서는 training data의 `teacher_feedback_available_rate`
14. `D2b`에서는 OOD retrieval-conditioned gain: feedback-conditioned lane가 `D1`보다 OOD accuracy에서 얼마나 이득을 보는지
15. `D2b` 성공 판정은 두 readout을 함께 본다:
    a. `analyze_self_distill_eval.py`로 collapse / calibration / benchmark slice
    b. `analyze_rq3_side_eval.py`로 trigger precision / next-meta recovery / OOD combined accuracy gate

**공정성 / 데이터 계약 게이트**

autoresearch나 대규모 원격 실행 전에 다음 조건을 모두 통과해야 한다.

1. base/meta가 같은 strict SFT split, 같은 decode seed schedule, 같은 root/repair budget을 사용한다
2. claim-bearing lane에서 `synthetic_meta_injected_rate == 0` 이어야 한다
3. claim-bearing lane에서 retrieval은 trigger-earned path가 아니라 강제 대칭 path 또는 미사용이어야 한다
4. selector provenance (`candidate_count`, `selected_candidate_id`, `selection_score_total`, `score_margin`)가 artifact row에 저장돼야 한다
5. selector는 correctness + controller-execution reward를 쓰되, `meta_count_bonus` 같은 style reward는 teacher ranking에서 제외한다
6. `sdpo_regen` lane는 side-evidence label로만 해석하고 immediate base/meta claim과 합치지 않는다

**성공 기준**

RQ3의 의도는 `OOD에서 self-distill을 성공할 수 있는 방법`을 찾는 것이다.
따라서 다음 기준을 모두 만족해야만 성공으로 본다.

1. `D1`은 collapse probe로서 명확한 붕괴 신호를 보여주거나, 반대로 붕괴가 없음을 명확히 보여야 한다.
   즉, 결과가 애매하면 안 된다.
2. `D2a`는 같은 budget의 `D1`보다 `wrong_high_confidence`와 `meta_emission`에서 더 낫거나 같아야 한다.
3. `D2b`는 같은 budget의 `D1` 대비 OOD accuracy에서 유의미한 개선을 보여야 한다.
   운영 기준은 `OOD combined accuracy >= D1 + 2pp`로 둔다.
4. `D2b`는 OOD에서 retrieval/branch trigger semantics를 망가뜨리면 안 된다.
   운영 기준은 `trigger precision drop <= 10pp` 및
   `curriculum_retry.next_meta.recovery_rate >= D1`.
5. `D2b`는 feedback-conditioned distill이라는 이름에 맞게 실제 privileged feedback를 포함해야 한다.
   운영 기준은 `teacher_feedback_available_rate > 0`이며, 실험 채택 기준은 `>= 0.3`을 목표로 한다.
6. 위 기준을 만족하지 못하면 `RQ3 성공`이 아니라 `collapse analysis only` 또는 `offline controller retention only`로 기록한다.

**해석**

1. naive self-distill이 in-domain만 올리고 OOD에서 meta emission과 recovery를 잃으면 epistemic collapse로 해석한다
2. epistemic-preserving self-distill이 동일 또는 더 나은 accuracy를 유지하면서 collapse 지표를 덜 악화시키면 채택한다
3. OOD에서 retrieval / branching trigger가 보존되지 않으면, self-distill은 controller를 망가뜨린 것으로 본다
4. 이 lane의 첫 readout은 반드시 `naive self-distill vs epistemic-preserving self-distill` 쌍으로 낸다
5. D3 dense distill은 D1/D2가 모두 동작한 뒤에만 해석한다. D3가 좋아 보여도 D1 collapse readout 없이 claim하지 않는다

#### RQ3 Artifact Contract

RQ3 결과물은 문제별로 다음을 반드시 남긴다.

1. root completion + root analysis
2. trigger fired 여부
3. plain retry completion + correctness
4. retrieval retry completion + correctness + retrieved example provenance
5. retrieval score breakdown (`problem_similarity`, `diagnosis_to_solution`, `study_need_to_strategy`, `strategy_hint`)
6. retrieval 후 `next meta` readout (`confidence_gain`, `trigger_cleared`, `study_need_followthrough`)
7. selective branching result + chosen branch + branch values
8. branch score breakdown (`diagnosis`, `decomposition`, `next_strategy`, confidence prior)
9. per-lane evidence class (`mainline_downstream` vs `side_evidence`)
10. per-problem winner label (`root`, `plain_retry`, `retrieval_retry`, `mcts_lite`)
11. aggregate summary: trigger precision, retry gain, branch gain, wasted expansion
12. self-distill branch 결과가 있으면 `meta_emission`, `wrong_high_confidence`, `ood_gap`, `response_length_change`

## 5. Verification Gates

The repository is not allowed to advance to the next phase unless all gates below pass.

### Gate A. Data

1. strict validator passes
2. paired prompt/answer parity passes
3. scenario counts are reported

### Gate B. SFT Runtime

1. correct initializer logged
2. 4 GPUs active on both nodes
3. no duplicate trainer processes
4. wandb run created
5. checkpoints written locally
6. post-SFT eval bundle script prepared before SFT completion

### Gate C. Pre-RL

1. strict paired SFT completed
2. paired deterministic eval completed and decoding metadata saved
3. reward smoke tests pass
4. RL launcher passes frozen-key preflight
5. RQ3 branches are not allowed to start before RQ2 anchor and reward-family readout

### Gate D. Claim Readiness

1. evidence class recorded
2. checkpoint provenance recorded
3. eval bundle saved as JSON + metadata + parquet
4. qualitative samples saved for later audit

## 6. Analysis Contract

The minimum analysis bundle for any mainline checkpoint is:

1. benchmark accuracy on `gsm8k`, `math500`, `aime2024`
2. confidence and ECE summary
3. behavior counts: verify, redirect, diagnosis, decomposition
4. difficulty-conditioned analysis
5. AIME qualitative examples
6. if available, entropy / information-allocation diagnostics

## 7. Node Contract

Mainline `metacognition` may use only:

1. `metacognition_eval`
2. `metacognition_train_b`

Other long-lived AMLT holders are outside the mainline scheduler.

## 8. Operational Rules

1. never silently switch from raw base to a previously trained base-SFT initializer
2. never call a run `mainline` if paired shared keys differ
3. never reuse an exploratory launcher as if it were the active launcher
4. always run alignment preflight before mainline launch
5. always save eval outputs in machine-readable form for later behavior audit
6. do not use curriculum/RAG results to support the main claim before RQ1 and RQ2 are established
7. do not describe E21S as value learning, MCTS, or test-time search
8. do not place E21M or MCTS-lite inside the RQ2 reward ladder

## 9. Immediate Next Actions (updated 2026-04-13)

1. Repair runtime environment and resume E21R-v2 from the latest valid checkpoint
2. Repair runtime environment and resume Base GRPO from the latest valid checkpoint
3. At E21R-v2 step 100:
   - If overall accuracy > base - 2pp: continue to step 300
   - If overall accuracy < base - 5pp: stop early, declare meta overhead > benefit
4. After E21R-v2 completes:
   a. Run 1560-problem eval on final checkpoint
   b. Redirect success rate re-measurement
   c. Paired behavioral comparison vs base
5. Based on E21R-v2 final readout, start Phase 4b:
   a. Priority 1: Short-Redirect Constraint (4b-A) — simplest, cheapest
   b. Priority 2: SCoRe 2-turn (4b-B) — if 4b-A insufficient
   c. Priority 3: Subgoal data rebuild (4b-C) — if execution quality is data-limited
6. Defer E21S (stepwise dense reward) until redirect content quality is solved
7. Keep E21M and MCTS-lite in Phase 5 side-evidence until RQ2 paired readout is complete
8. Start `RQ3-D1 naive self-distill collapse check` immediately after RQ2 paired readout
9. Only start `RQ3-D2 epistemic-preserving self-distill` after D1 collapse analysis is saved
10. Push all checkpoints to HF every 3 hours (automated)
11. Experiment report: results/metacot_v8_experiment_report.md

## 10. RQ3 Retrieval Critic Update (2026-04-15)

Intent:

1. make teacher-only RAG retrieval depend on typed `study_need` evidence rather than generic easy-example overlap
2. separate reusable `stable_seed_library` from future `dynamic_success_library`
3. ensure saved example banks round-trip without silently dropping nested metadata

Hypothesis:

1. generic easy exemplars were still winning because `easy_bonus` could dominate even when the bank entry had no typed strategy signal
2. several families, especially `exponential_growth`, were under-covered in the temporary correct-only eval bank, so retrieval quality could not improve by scoring changes alone
3. `load_example_bank()` dropping nested metadata would break both future dynamic-memory replay and any curated seed-library workflow

Verification:

1. added `typed_strategy_bonus` and `generic_penalty` so typed exemplars are preferred over empty generic-easy entries when a query carries `study_need`
2. added `scripts/build_control_rag_seed_library.py` to build a reusable `stable_seed_library` from prior correct typed runs
3. fixed `load_example_bank()` so `ExampleRecord.to_dict()` JSON round-trips preserve nested metadata such as `study_need_family`, `source_role`, and future dynamic-lane provenance
4. refined family classification to avoid false geometry matches from bare `power` and to capture `geometric sequence` under `exponential_growth`
5. verified with:
   - `pytest -q tests/test_control_rag.py tests/test_rq3_pipeline.py tests/test_self_distill_data.py`
   - `python scripts/build_control_rag_seed_library.py --inputs results/eval_v8_E20a/eval_v8_meta_inside_E20a.json results/eval_v8_E20b/eval_v8_meta_inside_E20b_5ep.json --output tmp/control_rag_seed_library.json --require_correct --require_study_need`
   - `python scripts/audit_control_rag_real.py --bank_paths tmp/control_rag_seed_library.json`

Current readout:

1. with the appended stable seed library, top-1 typed-strategy coverage is `0.935`
2. `exponential_growth` improved from `0/6` top-1 family matches in the temporary bank-only audit to `3/7` when the stable seed library is appended
3. `probability_counting` remains the weakest family because the bank still lacks sufficiently specific typed exemplars; this is now a bank-coverage problem, not just a scorer problem

Operational follow-up:

1. `dynamic_success_library` is now materializable from saved RQ3 traces via `scripts/build_control_rag_dynamic_library.py`
2. `run_rq3_side_eval.py` accepts both flat retry fields and nested trace-bundle retry fields, so smoke and offline audit artifacts can be replayed through the same path
3. side-eval manifests now report active bank roles, making stable-seed vs dynamic-success provenance inspectable
