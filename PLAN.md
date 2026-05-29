# PLAN — Contrastive Triggered Self-Distill (CTSD) for Meta-CoT (v5, 2026-05-29)

> **v5 STATUS — what we actually know now, and the one gate that decides training**
>
> Phase A probe results (run 2026-05-28):
> - **A.1 contrastive-on-natural-meta — FAIL.** On natural R10v2 rollouts the
>   outcome-discriminating signal lives in the *body* (Cohen d=0.84), NOT in the
>   meta region (d=0.08). The model's spontaneously-emitted meta is decoration.
> - **A.2 entropy trigger — PASS.** Token entropy predicts wrongness (AUC 0.749);
>   τ candidates p85=8.28 / p90=8.59 / p95=8.91. We know *where* to intervene.
> - **A.6 controlled-meta discrimination — PASS.** When good vs bad meta is
>   *explicitly injected*, all 3 teachers separate them (EXCL_ANSWER AUC: v8=0.81,
>   E20a=0.95, base=0.94). E20a is the best teacher. The reward signal is *valid* —
>   IF the meta carries good/bad variance.
> - **R18b (contrastive RL on natural meta) — FAIL.** 70.9% < baseline 72.3%.
>
> **The honest gap (user challenge 2026-05-28):** we *inferred* "emission lacks
> variance" from the A.6-vs-R18b gap, but never (a) directly proved natural meta
> lacks variance, nor (b) verified that *force-injecting* meta improves accuracy.
>
> **DECISION (locked, user-approved 2026-05-29):** one verification gates all
> training — **A.3 inject causal test** (see §Phase A.3, rewritten below).
> - A.3 PASS → run Phase C smoke → Phase E full, *no stopping* (auto long-run).
> - A.3 FAIL (meta inject has no causal effect) → STOP, report, brainstorm new
>   experiment directions with user (do NOT auto-pivot to Phase D).
> Teacher = E20a (A.6 winner). clip_eps_w = 0.2. The thing R18b lacked and CTSD
> adds = **force-inject** the meta region so the contrastive reward has a target.
>
> **A.3 RESULT (2026-05-29) + the b-marker pivot.** A.3 ran (n=30 math500-hard,
> v8_strict, k=4) and the *content-inject* gate FAILED, well-powered (boxed 0.80-
> 0.93, real null): c good-inject 0.150 ≈ a no-inject 0.133 (c−a=+1.7pp p=0.86);
> direction c−d=−1.7pp p=0.74. Natural substantive meta 0/15 correct = distress
> marker. **BUT b marker-only was the best condition** (0.183, +5pp over a, 7-3
> per-problem wins) — though underpowered (paired p=0.32, 95% CI [−2.5,+13.3]pp).
> Reading: supplying *fixed* good content (c) does NOT beat the model's own meta,
> but *forcing the model to emit its own meta at the uncertain point* (b) is the
> promising-but-unproven signal. **User direction (2026-05-29):** go with b
> (marker-only inject) and let the **contrastive reward shape the content** during
> RL. So the training inject switched from fixed-content → **marker-only**
> (`sdc_inject_mode=marker`), and a **high-power b-vs-a re-test** (`--select mixed`,
> gsm8k+math500 headroom, n=48 k=6) runs before committing node compute.

# PLAN — Contrastive Triggered Self-Distill (CTSD) for Meta-CoT (v4, codex-reviewed 2026-05-28)

> **v4 changes from v3** (per codex direct CLI review):
> - Add control contrast probes (A.1b/c/d) to rule out artifact: shuffled T+/T-, non-meta tokens, same-answer-different-rationale.
> - Add wandb metrics for w_attr × w_contrast independence: effective weight mass, KL on meta tokens.
> - Explicit "thresholds pre-registered before looking" note in Phase A gating.
>
> **v3 changes from v2**: Remove Arm 3 (Stable-GFN P-direct) — not our method, only Kwon's framework. Remove H3 (STaR-flavored best-of-k) — not our method. Our method is *single*: SDC form + position inject + meta signal strengthening. Stable-GFN / STaR mentioned in §8 related work only, not in our implementation.

> Purpose: a clean, first-reader-accessible plan for the next 3-week sprint
> on metacognition self-distillation. Replaces all root-level `PLAN_*` docs
> after codex review converges; old plans archived under `plans/archive/`.
>
> **Revision history**:
> - v0 (initial draft, 14 self-identified issues)
> - v1 (this version, addresses all 14 issues + decision gate tightening)
>
> **CTSD = Contrastive Triggered Self-Distill**: our novel framework combining
> (i) T+ vs T- contrastive teacher direction, (ii) entropy-triggered force-inject,
> (iii) Stable-GFN's fluency stabilizer 만 차용 (force-inject 후 coherent continuation 보장 위한 base-ref penalty). Stable-GFN paradigm 의 distribution-matching 자체는 *사용 안 함*.

---

## 0. Intent (one paragraph, no jargon)

We want Qwen3-8B 의 math reasoning 을 *self-monitoring* — 풀이 중간에 *"내가
잘 가고 있나, 다시 봐야 하나"* 의 metacognition 을 emit 하면서 풀게 — 향상시킵니다.
지난 두 달의 실험 (R10v2-on-v8, Arm1-5, R18a/b/c, R10v2-on-E20a clip-sweep) 이
RLSD 의 *single-teacher* paradigm (Yang 의 T+ vs student) 이 우리 distribution 에서
*9-axis null* (Phase 0) 임을 보였습니다. 다만 2026-05-27 에 발견 — *T+ vs T-
(contrastive teacher)* 의 direction 이 *correct vs wrong rollout 에서 sign-flip
separation* (gsm8k 위에서 Cohen d=0.70). 이 발견을 *math500 + AIME* 로 generalize
시키고, force-inject 와 결합해서 우리만의 *contrastive metacognition self-distill*
framework 를 *paper §8 의 positive result* 로 정리하는 게 이 plan 의 목표.

---

## 1. Confirmed facts (2026-05-26 ~ 27)

| Fact | Evidence | Implication |
|---|---|---|
| Yang RLSD single-teacher fails | Phase 0 9-axis null (M1-M5, AUC 0.21-0.55) | Single-teacher direction is dead on our distribution |
| Kim RLRT direction (student−T+) also fails | m_tier_a (3 versions, n=100/100) | Sign-reversed also null on R10v2 rollouts |
| Teacher GENERATES more meta but LESS accurate | teacher vs student gen test (n=8): T+ meta count 3×, correct 1/8 vs student 2/8 | Self-teacher is not a good distillation target |
| Contrastive T+ vs T- shows outcome separation | m_iota (n=66, gsm8k only): d=0.70, p=0.004, sign-flip | NEW positive signal — needs robustness check on hard problems |
| Meta SFT hurts AIME | strict_meta_sft 16.67% vs strict_base_sft 26.67% | Our v8_meta_strict data is declarative not experiential |
| RL preserves SFT meta-emission, doesn't create | v8 SFT 78% emit → R10v2-on-v8 98.4%; E20a SFT 0% → R10v2-on-E20a 1.2% | RL only amplifies; SFT-stage failure cascades |
| Tokenizer identical between SFTs | special=False, same vocab, same chat template | SFT learning failure is hyperparam/step issue, not tokenizer |

---

## 2. Why our previous approach failed — 6 mechanism layers

처음 보는 사람을 위해 *왜 우리 기존 SDC GDPO RLSD 가 안 됐는지* 의 mechanism 을 단계별로:

**Layer 1 — Declarative SFT data**. v8_meta_inside_strict 의 redirect samples 가
*"A tempting first thought is X, but that's weak..."* 형식의 didactic 시범.
*실제로 wrong path 시도 후 깨달음* 의 experiential pattern 부재.

**Layer 2 — Random meta position**. Meta 의 emission 위치가 *response 의 30%
지점* 으로 fixed. *Confidence collapse 나 overconfidence 같은 reasoning state
변화에 triggered* 가 아닌 *learned position*.

**Layer 3 — Teacher 의 logp-based discrimination 불능**. T+ teacher 가 codex-
labeled good vs bad meta 를 logp 로 separable 하게 못 가름 (Phase 0 9-axis null).

**Layer 4 — Teacher 의 generation 도 quality 못 보장**. Gold-revealed teacher 가
*더 많은 meta* + *더 많은 backtracking* emit, 그러나 *accuracy 더 낮음*. 즉
distillation target 자체가 student 보다 worse.

**Layer 5 — RL reward 의 quality term 부재**. meta_penalty (-0.20) 가 *emission
여부* 만 penalize, *content quality / position appropriateness* 는 measure 안 함.
Boilerplate template 도 penalty 0. GRPO advantage normalization 에서 모든 rollout
이 같은 penalty 받으면 cancel.

**Layer 6 — Format collapse via clip sweep**. clip_eps_w=1.0 의 aggressive
amplification 으로 R10v2-on-E20a 가 *답 emit 후 newline padding mode* (98.7%
truncated, 1.3% proper EOS).

이 6 layer 모두 *분리 verifiable, 모두 동시 작용*. 한 layer 만 고친다고 해결
안 됨. 그래서 이 plan 의 *multi-axis intervention* 필요.

---

## 3. Our single method = CTSD — 세 component 의 *결합*

우리 method 는 *Yang RLSD 의 SDC form* 위에 *두 layer 추가* 한 single framework:

```
SDC base (existing)        Meta signal strengthening (NEW)        Position inject (NEW)
    │                                  │                                    │
    │  w_attr = clip(exp(sign(A) ×     │  w_contrast = clip(exp(sign(A) ×   │  Force-insert <|meta|> at
    │    (T+_logp − student)),         │    (T+_logp − T-_logp) / τ_c),     │    high-entropy positions
    │    1-ε, 1+ε)                     │    1-ε, 1+ε)                       │    during inference
    │                                  │                                    │
    └──── existing in verl_sdc.py ─────┴── NEW: T- forward pass added ──────┴── NEW: entropy hook ──┘
                                                                                       │
                                       Combined token weight:  w_final = w_attr × w_contrast
                                       Applied: meta region only (selective)
                                       Reward: A_t' = A_t × w_final
```

세 component 가 *각각 우리 finding* 또는 *우리 design*:

### H1 — Contrastive teacher direction (signal strengthening)
$w^{\text{contrast}}_t = \text{clip}(\exp(\text{sign}(A) \cdot (T^+_\text{logp} - T^-_\text{logp}) / \tau_c), 1-\epsilon, 1+\epsilon)$

m_iota (gsm8k 위에서 d=0.70) 이 보여준 *우리만의 finding*. T+ vs T- 의 *outcome-stratified separation*. Yang RLSD 의 *T+ alone* 또는 Kim RLRT 의 *student − T+ alone* 와 정성적으로 다른 axis — *두 teacher 의 contrast*. Phase A.1 가 generalization 검증.

### H2 — Entropy-triggered force-inject (position control)
inference 시 token entropy H_t > τ_high → next position 에 `<|meta|>` force-insert. *Internal trigger, single-turn, external framework 없음*. Phase A.2 가 τ_high empirical calibration. Quiet-STaR (모든 position) 도 SCoRe (external 2-turn) 도 아닌 *우리 design*.

### 세 component 의 결합 = CTSD
- *SDC form* 은 existing infrastructure 재사용 (verl_sdc.py)
- *Contrastive direction* 은 우리 finding 의 RL gradient 화
- *Position inject* 는 *조건부* (Step 2 marginal 시 추가) — Step 1 에서 contrastive 만으로 충분하면 inject 불필요

### Related work — 우리 method 가 *아닌 것* 들 (명확 구분)

다음은 *우리 CTSD 안에 들어가지 않는* paper 들. 명시적으로 demarcate.

**vs Yang RLSD (2604.03128)**: Yang 의 T+ alone w_attr 는 *우리 base*. 우리는 그 위에 *w_contrast (T+ vs T-)* 추가. Yang 의 single teacher 가 fail 한 우리 distribution 에서 dual teacher contrast 가 separation.

**vs Kim RLRT (2605.10781)**: Kim 도 single teacher (sign-reversed P_S − T+), full-response, outcome-gated. 우리는 *dual teacher + meta-region selective*. 다른 axis. *RLRT 의 코드/training 도 우리 안에 들어가지 않음*.

**vs Stable-GFN (Kwon 2605.00553)**: Kwon 의 pairwise contrastive trajectory balance (Z-free) 는 *우리 method 안에 들어가지 않음*. Distribution-matching paradigm. 우리는 *advantage-multiplicative SDC paradigm*. 다른 framework. (우리 codebase 의 STABLE_GFN mode 는 *별도 비교 baseline* 으로 *향후 ablation* 에서 활용 가능, 그러나 *CTSD 의 component 아님*.)

**vs Quiet-STaR (Zelikman 2403.09629)**: 모든 position 에서 thought generate (expensive). 우리 force-inject 는 *high-entropy position 만* — triggered, sparse. 다른 mechanism.

**vs SCoRe (Kumar 2409.12917)**: External 2-turn (turn-1 → turn-2 critique). 우리는 *internal single-turn force-insert*. External orchestration 없음.

**vs STaR (Zelikman 2022)**: Full trajectory rejection sampling for data augmentation. *우리 method 안에 들어가지 않음* — STaR 는 SFT-stage data 만들기, CTSD 는 RL-stage. 다른 paradigm.

**vs PRM-guided (rStar-Math 2501.04519)**: Step-level reward model 별도 학습. 우리는 *PRM 불필요* — outcome reward 만.

→ 우리 CTSD framework = **SDC base + H1 contrastive direction + (조건부) H2 force-inject**. 단일 RL update rule. 위 paper 들의 framework 와 *components 차원에서 demarcate*.

---

## 4. 4-Phase plan — parallel maximized

### Phase A — Local inference probes (Day 0-1, 모두 parallel, no training)

#### A.1 m_iota benchmark-stratified replicate
- **Intent**: H1 의 contrastive signal 이 gsm8k 만 의 artifact 인지 vs general.
- **Hypothesis**: Math500 + AIME 위에서도 같은 sign + Cohen d ≥ 0.4.
- **Verification**:
  - 30 gsm8k + 30 math500 + 30 aime stratified sample (R10v2-on-v8 eval JSON)
  - Same protocol as `scripts/m_iota_contrastive_outcome_split.py`
  - Compute T+/T- contrast outcome split per benchmark + pooled
  - **Equal n per benchmark** (30 each) to prevent gsm8k bias re-introduction
  - **Gate**: pooled Cohen d ≥ 0.4 AND p < 0.01 AND sign consistent across all 3 benchmarks (not just pooled)
  - **Thresholds pre-registered**: gate values locked in plan before measurement (this commit)
  - **NEW Control contrasts** (codex v4 feedback — rule out artifact):
    - **A.1b shuffled control**: T+ vs T- where T- is from a *random* problem (mismatched gold). Real contrast should be larger than shuffled (control Cohen d expectation < 0.2).
    - **A.1c non-meta control**: same contrast on body tokens (outside meta region). If body contrast d ≈ meta contrast d, signal is *not meta-specific* — artifact.
    - **A.1d same-answer-different-rationale**: within correct rollouts of *same problem*, two different meta blocks. Contrast should NOT show outcome direction (this is null-test for spurious causality).
  - **Robustness gate** (A.1 + A.1b/c/d combined): real d ≥ 0.4 AND real d ≥ 2× max(control d) — signal is causal, not artifact
  - Wall: 1 GPU × 2h (4× forward passes per rollout — real / shuffled / body / same-answer)

#### A.2 Entropy distribution mapping
- **Intent**: H2 의 trigger threshold τ_high empirical calibration.
- **Hypothesis**: Wrong rollouts have early-position entropy spike > correct rollouts; threshold τ at p85-p95 quantile gives clean separation.
- **Verification**:
  - 100 problems (50 correct + 50 wrong) from R10v2-on-v8
  - Run inference, capture per-token entropy H_t
  - Plot histogram + quantile analysis
  - Measure: P(correct | H_t > τ) vs P(correct | H_t < τ) for τ ∈ [p70, p99]
  - **Gate**: Some threshold τ where AUC(entropy vs final_correctness) ≥ 0.65
  - Wall: 1 GPU × 1h

#### A.3 Force-inject CAUSAL test (THE gate for all training) — v5, codex-reviewed
- **Intent**: answer the user's two challenges in one inference-only probe —
  (i) does the model's *natural* meta carry good/bad variance that tracks
  correctness, and (ii) does *force-injecting* meta at high-entropy positions
  causally improve accuracy, with the *direction* mattering.
- **Method** (no training; v8_strict student solves alone — no reference answer):
  - Sample = baseline-WRONG rollouts on hard benchmarks (headroom); rule fixed.
  - Per problem: baseline rollout → RAW per-token entropy (A.2 definition) →
    inject at the **argmax body-entropy position** (≥50 tok). codex P1 fix: an
    absolute τ=p90 fires on only ~10% of rollouts → empty gate; argmax targets
    the model's single most-uncertain body token and always fires. Freeze
    prefix = [prompt+tokens[:p*]].
  - From the SAME prefix, 4 conditions × k continuations:
    - **(a) no-inject** — continue normally (natural baseline)
    - **(b) marker-only** — append `<|meta|>` only; model fills + closes →
      *emission alone* (descriptive; close-rate reported)
    - **(c) good-inject** — PRODUCTIVE meta (low-conf, re-derive+verify),
      **answer-free**
    - **(d) bad-inject** — UNPRODUCTIVE meta (over-conf, skip-checking),
      **answer-free**, ≈length-matched to (c) (45 vs 47 tok)
  - **codex P0 fix**: (c)/(d) are *both answer-free* and length-matched — neither
    names gold nor decoy — so (c) vs (d) tests meta *direction*, not answer priming.
  - Grade vs gold via math_verify-backed `_check_correctness` (math_verify now
    installed → robust on fractions/MATH).
- **Pre-registered gates** (locked; over problems with all 4 conditions present
  — paired intersection):
  - **helps**: acc(c) − acc(a) ≥ +3pp AND paired p<0.05 → inject *helps* (not
    merely "differs from bad")
  - **direction**: acc(c) − acc(d) ≥ +5pp AND paired p<0.05 → direction causal
  - **PASS for training** = **helps AND direction** (good inject beats *both*
    no-inject and bad inject — rules out "d made worse" artifact, codex P0).
  - **power guard**: if baseline `boxed_rate` < 0.5, continuations are truncating
    before an answer → verdict = INCONCLUSIVE (raise `--max_new`), NOT a null.
  - **FAIL** = gates miss AND power OK → meta inject non-causal → STOP + brainstorm.
  - emission (b−a), content (c−b) reported descriptively, not gates.
- **Natural-meta variance** (answers Q-i): on (a) continuations classify natural
  meta none/boilerplate/substantive + accuracy per class + emit-rate.
- Wall: 1 A100 × ~3-5h (HF generate, batched k).

#### A.4 SFT data quality audit
- **Intent**: v8_meta_inside_strict 의 *declarative vs experiential* ratio 측정.
- **Hypothesis**: Declarative (가짜 wrong path 시범) 가 ≥ 60% — paper §8 의 mechanism 의 핵심 datum.
- **Verification**:
  - 100 random samples (not 50) from 4264 SFT rows — SE ~5% gives clear decision branch at 50%
  - GPT-4 binary classification: declarative (fake wrong path setup) vs experiential (genuine recovery from real attempt)
  - **Decision branch**:
    - Declarative < 50% → SFT data 그대로 keep, Phase C/D 진행
    - Declarative ≥ 50% → Phase D 의 v9 redesign 가 critical path
  - Wall: 1h + $30 GPT-4

### Phase B — Decision branch (Day 1)

위 4 probe 결과의 *명확 mapping table*:

| A.1 result | A.2 result | A.3 result | A.4 result | → Decision |
|---|---|---|---|---|
| d≥0.4 + 3-way sign consistent | AUC≥0.65 | coherence≥4.0 | declarative<50% | **S1**: all Phase C arms full speed, no Phase D |
| d≥0.4 | AUC≥0.65 | coherence≥4.0 | declarative≥50% | **S2a**: Phase C 진행 + Phase D parallel |
| d≥0.4 | AUC<0.65 | any | any | **S2b**: Phase C Arm 1 만 (inject 보류), Phase D 조건부 |
| d<0.4 OR sign inconsistent | any | any | any | **S3**: Phase C cancel, **PIVOT to Phase D core**, paper §8 narrative locks |
| any | any | coherence<3.0 | any | **S2c**: Arm 1+3 만 (Arm 2 force-inject 보류), need fluency redesign |

### Phase C — RL smoke training (gated by A.3 PASS)

> **v5 simplification (grounded in what already ran).** "Arm 1 = contrastive
> only" is NOT a hypothetical — it already ran as **R18b** (ROD_MQ_CONTRAST,
> α=β=0.5) and **FAILED (70.9% < 72.3%)**, because the contrastive reward had no
> good/bad variance to shape in the model's decorative natural meta (A.1). So
> Phase C is a **single canonical arm**: R18b + the one missing piece, force-inject.

#### The Phase C arm — ROD_MQ_CONTRAST_INJECT (R18b + force-inject)
- **Intent**: give the contrastive reward a meta region to shape by force-injecting
  `<|meta|>` at the max-entropy pre-answer position during rollout — the exact
  mechanism A.3 validates offline. One-axis ablation vs R18b (inject on/off).
- **Implementation (built, gated by A.3)**:
  - Inject core (pure, unit-tested ×9): `src/training/meta_inject.py`
    (`find_inject_position` = argmax body-entropy before first `\boxed`, outside
    meta spans; `plan_inject_prefixes` = batch orchestration).
  - Mode `ROD_MQ_CONTRAST_INJECT` in `verl_sdc.py` REWARD_CONFIGS — identical
    reward heads + advantage math to R18b (aliased in `verl_sdc_utils.py`
    dispatch + `_CONTRASTIVE_MODES` so T− is attached for the q_contrast term).
  - Rollout wiring `SDCRayPPOTrainer._force_inject_rollout` + fail-fast guard:
    `sdc_force_inject=true` REFUSES to launch until the two-phase DataProto repack
    is node-smoke-tested (NODE-SMOKE-REQUIRED — verl internals can't be tested in
    the local CPU env). This is the one piece that must be wired on the node first.
  - Config: `configs/verl_ctsd_inject_C_h200_4x4k.yaml` (mode + `sdc_force_inject`
    + `sdc_inject_min_tok=50`, `clip_eps_w=0.2`, `total_training_steps=50` smoke).
- **Base SFT**: v8_strict. **Teacher**: T+/T− self-distill (E20a teacher-swap = a
  separate future ablation, not this arm — keep the ablation one-axis).
- **Smoke scale**: 100 problems, 50 PPO steps.
- **Honest baseline**: R10v2-on-v8 13.3% AIME strict (NOT Kim's Qwen3-8B-Base 27.9%).
- **Validation gate**: AIME strict-boxed ≥ baseline + 3pp on 30 AIME AND p<0.05
  paired bootstrap → proceed to Phase E. Else stop / rethink.
- **First step on node**: wire + 1-step smoke the `_force_inject_rollout` repack
  (remove the fail-fast guard only after the smoke passes).

### Phase D — Conditional SFT redesign (Day 4-14, triggered by A.4)

만약 Phase A.4 의 declarative ratio ≥ 50% 면 *parallel track* 으로 시작:

- **D.1 Experiential trajectory generation**:
  - Base Qwen3-8B + simple math CoT prompt → 1000 problems × k=4 rollouts
  - **Step verifier**: sympy-based math equation verifier (not PRM — we don't have trained PRM). For each step, parse equation, verify symbolically. First step that fails verification = "first wrong step" anchor.
  - Splice: insert "Wait, that's wrong" redirect → retry until correct
  - Quality filter: GPT-4 judge of resulting experiential trajectory
  - Yield: ~500-1000 experiential SFT samples
- **D.2 v9 SFT training**: Qwen3-8B + experiential data, 1-2 days
- **D.3 v9 baseline eval**: 1030-panel strict-boxed, compare to v8_strict
- **D.4 Re-run Phase C on v9** (if v9 baseline better)

### Phase E — Full RL training (Day 7-17)

Phase C 의 best arm (또는 D.4 의 v9 best) 으로 full training:
- 300 PPO steps
- Full DAPO-Math-17k or verl_train_redirect.parquet
- AMLT H200 4-GPU or A100 8-GPU, ~1 week reservation
- Final 1030-panel eval

### Phase F — Paper §8 write-up (Day 17-21)

Multi-layer mechanism (declarative SFT + single-teacher fail + contrastive rescue +
inject + experiential SFT) + positive result table + ablation matrix.

---

## 5. Folder structure redesign (Karpathy: surgical, compact, accessible)

현재 94 scripts/ + 20+ root *.md → 새 structure:

```
/home/v-seungplee/metacognition/
├── PLAN.md                          # this file (canonical plan)
├── README.md                        # project entry point (existing)
├── CLAUDE.md                        # session config (existing)
├── plans/archive/                   # all old PLAN_*.md moved here
├── docs/                            # literature surveys + mechanism findings
├── src/                             # training code (existing, unchanged)
├── experiments/                     # NEW (replaces scripts/)
│   ├── common/                      # shared utilities
│   │   ├── eval_data.py             # eval JSON loaders
│   │   ├── logp_scoring.py          # forward-pass scoring (m_zeta primitives)
│   │   └── meta_parsing.py          # <|meta|> tokenization helpers
│   ├── probes/                      # Phase A: inference-only analysis
│   │   ├── a1_contrastive_outcome.py
│   │   ├── a2_entropy_distribution.py
│   │   ├── a3_inject_coherence.py
│   │   └── a4_sft_audit.py
│   ├── smoke/                       # Phase C: small-scale RL smoke
│   │   ├── arm1_contrastive_only.py
│   │   ├── arm2_contrastive_inject.py
│   │   └── arm3_stable_gfn.py
│   └── data_redesign/               # Phase D: experiential SFT data
│       ├── trajectory_splicer.py
│       └── quality_filter.py
├── configs/                         # AMLT yamls (existing)
├── reports/                         # measurement outputs (existing)
└── scripts/legacy/                  # all 94 old scripts moved here
```

Karpathy principles 적용:
- 새 코드 = minimum, 기존 m_zeta primitives 재사용
- 각 experiments/*/*.py = single purpose, single owner
- experiments/common/ 가 shared logic, 중복 제거
- Environment variables → `experiments/common/env.py` 하나로 모음

---

## 6. Implementation checklist (smoke → critic → fix loop per artifact)

1. **PLAN.md (this file)**:
   - [ ] codex review on intent/hypothesis/verification clarity
   - [ ] iterate until no problems found
   - [ ] lock

2. **Folder restructure**:
   - [ ] create experiments/, plans/archive/, scripts/legacy/
   - [ ] move 20+ old PLAN_*.md → plans/archive/
   - [ ] move 94 scripts → scripts/legacy/ (don't delete, preserve provenance)
   - [ ] codex review of new structure clarity

3. **experiments/common/** scaffolding:
   - [ ] eval_data.py — HF eval JSON loader, stratified sampler
   - [ ] logp_scoring.py — forward-pass scoring (extracted from m_zeta)
   - [ ] meta_parsing.py — `<|meta|>` regex + tokenization
   - [ ] env.py — environment variables, paths, model IDs
   - [ ] tests/ — smoke test each
   - [ ] codex review

4. **experiments/probes/** (Phase A):
   - [ ] a1_contrastive_outcome.py (port from m_iota, stratified sampling)
   - [ ] a2_entropy_distribution.py
   - [ ] a3_inject_coherence.py
   - [ ] a4_sft_audit.py
   - [ ] smoke test each (n=5 problems first), critic, fix
   - [ ] codex review per file

5. **Autoresearch launch** (only when all above clean):
   - [ ] launch 4 probes in parallel
   - [ ] wait for results
   - [ ] Phase B decision branch
   - [ ] AMLT yaml submission for Phase C arms

---

## 7. Decision gates (when to stop / pivot)

- After Phase A: pooled contrastive d < 0.3 OR entropy AUC < 0.55 → **PIVOT to Phase D core**, Phase C/E cancel
- After Phase C arm 1: AIME +3pp pass → proceed Arm 2+3, Phase E. Fail → Arm 2 only, OR pivot
- After Phase C all arms: any arm passes → Phase E full training. None pass → Phase D core
- After Phase E: AIME strict ≥ Kim RLRT baseline (27.9% on Qwen3-8B-Base) → paper §8 positive result. Below → mechanism narrative only

---

## 8. Related work positioning (final paper §8 narrative)

우리 contribution 의 explicit demarcation:

- **Yang RLSD (2604.03128)**: T+ alone, full-response. Fails on our distribution.
- **Kim RLRT (2605.10781)**: student−T+, outcome-gated, full-response. Reverse direction, but our distribution still null.
- **Kwon Stable-GFN (2605.00553)**: pairwise cTB for red-teaming. We adapt to math.
- **Gandhi 4 habits (2503.01307)**: behavior priors via SFT. We use as Phase D inspiration.
- **Quiet-STaR (2403.09629)**: per-position thought. We use *triggered* position only.
- **SCoRe (2409.12917)**: 2-turn external correction. We do internal single-turn.
- **Rewarding Doubt (2503.02623)**: log-scoring on declared confidence. Orthogonal to our contrastive direction.

**Our unique angle**: *Contrastive teacher direction* (T+ vs T-) + *entropy-triggered
forced inject* (internal single-turn) + *outcome-grounded habit selection* +
*Stable-GFN style P-direct distribution preservation*. None of the cited papers
combines these four; the combination is testable as ablation matrix in Phase E.

---

## 9. Risks and unknowns

- m_iota gsm8k-only signal may not generalize (Phase A.1 gate)
- Force-inject may produce frankenstein meta (Phase A.3 gate)
- v8_strict SFT may be too declarative for any path (Phase A.4 + D)
- Stable-GFN with sparse meta-region reward — convergence guarantees only at full trajectory (Arm 3 risk)
- AMLT node queue wait — buffer Day 2-5 expected (mitigation: submit Day 0)

---

## 10. v1 changes from v0 (issue → resolution)

1. **vLLM forced-prefix detail**: Phase A.3 implementation note added (vLLM's PrefixCachingScheduler not needed; use HF transformers `generate` with `prefix_allowed_tokens_fn` for token-level injection, simpler).
2. **A.4 sample size**: 50 → 100 (SE ~5% for clearer decision).
3. **Decision branch table**: 5-row explicit mapping in §Phase B.
4. **τ_c calibration**: spec added in Arm 1 (median-normalized for full clip range usage).
5. **Step verifier**: sympy-based (concrete tool, not abstract "PRM").
6. **Env vars**: explicit list — HF_TOKEN, WANDB_API_KEY, MODEL_PATH_TEACHER, MODEL_PATH_STUDENT, EVAL_JSON_PATH (centralized in `experiments/common/env.py`).
7. **Phase F detail**: paper §8 sections — "8.1 Multi-layer mechanism" / "8.2 Contrastive direction discovery" / "8.3 CTSD framework + ablation matrix" / "8.4 Limitations + future work".
8. **Smoke loop converge**: max 3 rounds per file, escalate user if still problems.
9. **TCCHI → CTSD**: cleaner acronym.
10. **Honest baseline**: R10v2-on-v8 13.3% AIME strict (not Kim 27.9% which is different SFT init).
11. **Parallel detail**: Phase A probes on single A100 sequential is ~6h; if 2 GPU available, parallel cut to 2h. Priority order if sequential: A.1 → A.2 → A.4 → A.3 (informativeness × cheapness).
12. **Equal n per benchmark**: A.1 spec tightened (30+30+30, not natural distribution).
13. **Effect size + p-value both**: gate refined (Cohen d ≥ 0.4 AND p < 0.01).
14. **clip_eps_w default**: 0.2 explicit (not 1.0) — avoid length pathology.

## 12. WandB metric specification — *intent-aligned* logging (NEW v2)

처음 보는 사람이 *우리 metacognition habit 이 reasoning 에 미치는 영향* 을 wandb dashboard 만으로 파악 가능하도록. RLSD/RLRT/4-habits paper 의 *각각의 main metric* + *우리 contrastive direction* 의 union.

### Per-batch (training-time, ~ every 10 steps)
- **`train/reward_mean`**, **`train/reward_std`** — overall reward sanity
- **`train/correctness_mean`** — base correctness fraction
- **`train/meta_emit_rate`** — *RLSD inheritance*: rollout 중 meta block 1+ emit 비율 (1.0 → mode collapse to always-emit, 0.0 → atrophy)
- **`train/meta_chars_mean`**, **`train/meta_chars_p90`** — meta length distribution (avoid overhead inflation)
- **`train/n_meta_blocks_mean`** — # blocks per response

### Per-batch contrastive signal (CTSD-specific)
- **`train/contrast_tplus_tneg_mean`** — *our new H1 signal*. Should track positive on correct, negative on wrong rollouts.
- **`train/contrast_correct_mean`**, **`train/contrast_wrong_mean`** — outcome-stratified split (the key Phase A.1 metric carried into training)
- **`train/contrast_cohen_d_correct_vs_wrong`** — effect size (should stay or grow during training; collapse → mode collapse)
- **`train/w_contrast_p50`, `train/w_contrast_p95`** — clip range usage (w near 1.0 → τ_c too high; pegged at clip → τ_c too low)
- **`train/q_meta_mean`, `train/q_meta_p90`** — existing SDC quality signal (kept for comparison)
- **`train/w_attr_p50`, `train/w_position_p50`** — existing multi-factor SDC components
- **`train/w_attr_contrast_correlation`** — NEW (v4): Pearson ρ between w_attr and w_contrast across meta tokens. Codex flagged double-count risk if highly correlated. Alert if |ρ| > 0.7 (signal: they're measuring same thing, drop one).
- **`train/effective_weight_mass`** — NEW (v4): sum(|w_combined - 1|) across meta tokens / sum(|w_combined - 1|) across all tokens. Sanity check that meta-region updates dominate.
- **`train/kl_meta_vs_ref`** — NEW (v4): KL divergence on meta-region tokens between current policy and reference (frozen SFT). Track distribution drift specifically on meta tokens.

### Per-batch 4-habits frequency (Gandhi-inspired)
- **`train/habit_verify_rate`** — `verify|check|confirm` lexical match rate per response
- **`train/habit_backtrack_rate`** — `wait|hmm|actually|reconsider|switch`
- **`train/habit_subgoal_rate`** — `step \d+|first[,:]|second[,:]`
- **`train/habit_backward_rate`** — `working backwards?|from the answer`
- **`train/habit_diversity_entropy`** — entropy over 4 habit types (1.5+ = balanced, < 0.5 = collapse to one habit)

### Per-batch entropy + injection (H2 / H3 specific)
- **`train/token_entropy_p90`** — high-entropy token quantile (trigger threshold tracking)
- **`train/inject_rate`** — % of responses where force-inject fired
- **`train/inject_position_p50`** — median fraction of response where inject occurred (compare to v8 SFT median 0.30)
- **`train/inject_coherence_estimated`** — base ref P(continuation | injected_prefix) proxy (Stable-GFN fluency stabilizer signal)

### Per-batch length / format (length pathology prevention)
- **`train/response_length_mean`, `train/response_length_p90`** — track length explosion
- **`train/eos_rate`** — % proper EOS (not truncated). Falls < 50% → length pathology warning
- **`train/boxed_rate`** — % with `\boxed{}` (format compliance)

### Per-eval (1030-panel, every 50 steps OR at end)
- **`eval/strict_boxed_overall`** — strict accuracy (primary headline)
- **`eval/strict_boxed_gsm8k`, `eval/strict_boxed_math500`, `eval/strict_boxed_aime`** — per-benchmark
- **`eval/lenient_overall`** — lenient (informational, NOT primary; honest split)
- **`eval/aime_pass_at_16`** — pass@k for diversity-preservation (Stable-GFN core claim) — sample k=16 at eval
- **`eval/aime_maj_at_16`** — majority vote (alternate)
- **`eval/per_habit_acc`** — accuracy stratified by whether response contains each habit (4-habits study)

### Pre-registered alerts (wandb alert system)
- **`alert_mode_collapse`**: trigger if `train/meta_emit_rate` < 0.1 OR > 0.99 for 50 consecutive steps
- **`alert_length_pathology`**: trigger if `train/response_length_p90` > 8000 tokens for 20 consecutive steps
- **`alert_format_collapse`**: trigger if `train/eos_rate` < 50%
- **`alert_habit_collapse`**: trigger if `train/habit_diversity_entropy` < 0.5

### Run organization
- **Project**: `metacot-ctsd` (NEW, separate from legacy `metacot-math`)
- **Run name**: `{phase}_{arm}_{seed}_{date}` (e.g., `C_arm1_s42_20260528`)
- **Resume**: each AMLT pod restart calls `wandb.init(id=run_id, resume="must")` from checkpoint
- **Tags**: `["CTSD", "phase_C", "arm1", "h200", "v8_strict_init"]`
- **Group**: `ctsd_phase_C_arms` (for cross-arm comparison)

## 13. HF checkpoint strategy — preempt-resilient + capacity-optimized (NEW v2)

H200 의 preempt 빈도 (보통 1-3일 마다) 고려. *Last-only checkpoint policy* + *atomic upload + delete* pattern.

### Strategy
- **Local save freq**: every 5 PPO steps (frequent, but fast SSD)
- **HF push freq**: every 25 steps OR when local disk > 80% capacity
- **HF repo**: `iamseungpil/metacot-ctsd-ckpt` (NEW repo, separate from data dataset)
- **Policy**: keep *latest 2* on HF (current + immediate previous, for rollback safety). Delete older.
- **Atomic upload**: upload-to-temp-path → rename → delete-old (no half-uploaded state)

### Resume sequence (on pod restart)
1. Download latest HF checkpoint to `/scratch/resume/`
2. Read `trainer_state.json` for global_step + optimizer state
3. wandb.init with same run_id + `resume="must"`
4. Load model + optimizer + lr_scheduler from HF ckpt
5. Resume from step+1

### Implementation
- `experiments/training/checkpoint_push.py`: async push to HF, last-only policy
- `experiments/training/resume_handler.py`: detect interrupted run, download latest ckpt, init from it
- AMLT yaml: `restart_on_failure: true` + `pre_run_script: experiments/training/resume_handler.py`

### Capacity estimate
- Qwen3-8B fp16 ckpt = ~16GB
- Optimizer state (Adam) = ~32GB
- Total per checkpoint = ~48GB
- 2 ckpts on HF = ~96GB (within HF model repo limit)

## 14. Monitoring + alerting plan (NEW v2)

### Real-time monitoring (during training)
- **wandb dashboard**: 4 panels per arm
  1. Reward + accuracy curves (sanity)
  2. CTSD signal curves (contrast, q_meta, w_*)
  3. Habit + format (4-habit rates, eos, boxed)
  4. Per-eval strict accuracy (every 50 steps)
- **wandb alerts** (Pre-registered, §12 above) fire to Slack/email

### Failure recovery monitoring
- **HF push success**: cron check every 30min, verify latest ckpt on HF matches local `latest_step`
- **AMLT pod health**: `amlt status` polling every 10min
- **Preempt detection**: if pod status flips paused → completed → starting, trigger resume sequence

### Manual review checkpoints
- Day 1 (after Phase A): user review of probe results, approve Phase C launch
- Day 7 (after Phase C smoke): user review of arm results, approve Phase E full
- Day 14 (mid Phase E): user mid-training review, decide continue/stop
- Day 17 (after Phase E): user review of 1030 eval, approve Phase F write-up

## 11. Open questions for further review

- Phase C 3-arm: Arm 3 (Stable-GFN P-direct) too research-y for first iteration?
- Phase D 의 experiential trajectory yield 가 충분할까 (1000 problems × 4 rollouts → ~300-500 successful splices 추정)?
- Decision gate 의 +3pp threshold가 노이즈 floor 와 분리되는지 (paired bootstrap n=30 의 95% CI 의 width)?

---

**Status**: v0 draft, pending codex review + iterate until no problems. After
convergence, implementation order: folder restructure → common/ → probes/ → smoke/.
