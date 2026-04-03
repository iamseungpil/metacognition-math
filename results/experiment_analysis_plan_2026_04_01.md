# Meta-CoT Experiment and Analysis Contract (2026-04-01)

## 1. Main Question

The project asks one main question.

`Can metacognitive control learned at test time improve OOD problem solving by changing behavior, not just style?`

We answer it through three research questions.

1. `RQ1: Meta-CoT`
   - Can the model emit parseable metacognitive state that is cleanly separated from ordinary CoT?
2. `RQ2: Meta-RL`
   - Can that metacognitive state be turned into verifiable reward signals and learned as a control policy?
3. `RQ3: Curriculum`
   - Can the same metacognitive state trigger diagnosis, retrieval, and adaptation for open-ended recovery?

## 2. Project Intent

### 2.1 Intent A: Meta-CoT must be a controller, not decorative self-talk

We do not want longer answers or generic reflection.

1. `confidence` must be a control variable that changes with risk.
2. `something feels off` or meaningful confidence drop must trigger redirect behavior.
3. `confidence remains high but support is weak` must trigger verify behavior.
4. `diagnosis` must explain why the current route is weak.
5. `study_need` must expose what skill, perspective, or object of study is missing.
6. meta blocks must stay strictly separate from ordinary derivation.

Meta blocks therefore carry only:

1. local confidence
2. anomaly / conflict notice
3. failure diagnosis
4. next control action
5. optional `study_need`

Meta blocks should not contain ordinary algebraic derivation or full CoT.

### 2.2 Intent B: Meta-RL must make those behaviors verifiable

The RL stage is not “optimize everything at once”.

1. first improve calibration
2. then improve confidence revision around intervention points
3. then test isolated behavior rewards one by one
4. only then run the combined controller

The decomposition exists to answer causal questions, not only to get a better final number.

### 2.3 Intent C: Curriculum must start from diagnosis

Curriculum and RAG are only meaningful if the model can say:

1. why the current attempt is failing
2. what missing knowledge or perspective is needed next
3. whether retrieval or one-example adaptation is actually justified

The long-term target is:

`diagnose -> expose study_need -> retrieve/adapt -> retry`

## 3. Intent / Hypothesis / Verification

### 3.1 RQ1: Meta-CoT

`Intent`

1. teach pure metacognitive control state
2. keep meta separate from derivation
3. test whether the learned state actually enables test-time adaptation on hard and OOD problems

`Hypotheses`

1. `H1a`
   - `V2/V3/V5 SFT` can teach parseable confidence and meta tags without collapsing math ability.
2. `H1b`
   - the learned meta traces will express controller state rather than decorative chain-of-thought, and will causally precede verify / redirect / diagnosis actions.
3. `H1c`
   - diagnosis and `study_need` can be made parseable enough for later curriculum use.
4. `H1d`
   - the same meta state can improve retry-time adaptation on hard slices without large regression on base accuracy.

`Verification`

1. meta parse rate
2. confidence extraction rate
3. meta purity
   - meta text contains diagnosis / control state / next action
   - meta text does not contain ordinary derivation
4. adaptation precursor coverage on hard problems
5. adaptation lift
   - `first_completion -> retry/intervention completion` delta on hard slices
   - contrast between samples with versus without verify / redirect signals
6. accuracy retention
7. qualitative samples on hard problems, especially AIME-like failures

### 3.2 RQ2: Meta-RL

`Intent`

1. learn better confidence calibration
2. learn intervention-local confidence revision
3. separate verify / redirect / diagnosis effects
4. test whether those effects combine into a real controller

`Hypotheses`

1. `H2a`
   - `E3` improves calibration even before explicit behavior rewards.
2. `H2b`
   - `E5` improves confidence revision around conflict or anomaly.
3. `H2c`
   - `E8` suppresses hard-slice overconfidence better than `E5`.
4. `H2d`
   - `E9 / E9b / E9c` isolate verify / redirect / diagnosis effects in interpretable ways.
5. `H2e`
   - `E10` combines those behavior rewards into a stronger controller than any single isolated policy.
6. `H2f`
   - `E6 / E7` are only meaningful if the probe estimates `p(correct | prefix)` rather than single-trajectory final correctness.

`Verification`

1. benchmark accuracy
2. ECE / Brier / wrong-answer mean confidence / wrong high-confidence rate
3. conflict-conditioned confidence drop
4. verify precision under high confidence
5. redirect-conditioned strategy-switch and recovery
6. diagnosis consistency and usefulness
7. repeated intervention quality on hard problems
8. qualitative response analysis, not only reward averages

### 3.3 RQ3: Curriculum

`Intent`

1. make diagnosis actionable
2. expose `study_need` as a parseable retrieval trigger
3. connect failure analysis to no-training RAG or one-example adaptation

`Hypotheses`

1. `H3a`
   - decorative low confidence alone should not trigger retrieval.
2. `H3b`
   - diagnosis plus `study_need` should trigger retrieval much more precisely.
3. `H3c`
   - retrieved examples or one-example adaptation should improve retry accuracy only when diagnosis is meaningful.

`Verification`

1. retrieval trigger precision
2. diagnosis / `study_need` coverage
3. retry prompt artifact logging
4. retry accuracy delta
5. reproducible saved outputs for later audit

## 4. Experiment Matrix

| Experiment | Intent | Hypothesis | Verification |
|---|---|---|---|
| `V2 / V3 / V5 SFT` | establish parseable meta representation | H1a, H1b, H1c | parseability, purity, confidence extraction, accuracy retention |
| `control_v5_verify_sft` | isolated verify controller | verify can be taught without redirect | high-confidence verify precision |
| `control_v5_redirect_sft` | isolated redirect controller | redirect can be taught without verify | redirect recovery and strategy switch |
| `control_v5_all_sft` | unified controller SFT | verify, redirect, diagnosis can coexist | unified behavior with limited accuracy loss |
| `E3` | pure calibration | H2a | ECE, Brier, wrong high-confidence rate |
| `E5` | calibration + confidence revision | H2b | conflict-conditioned confidence drop, no-drop wrong-commit |
| `E6` | probe calibration | H2f | `|confidence - p_hat_probe|`, probe-aligned ECE |
| `E7` | probe + blockwise stepwise | H2f | block-level probe gap, intervention-local calibration |
| `E8` | stronger anti-overconfidence shaping | H2c | hard-slice calibration, wrong-high-confidence suppression |
| `E9` | verify-only decomposition | H2d | verify precision, verify-conditioned error |
| `E9b` | redirect-only decomposition | H2d | redirect recovery, real switch fraction |
| `E9c` | diagnosis-only decomposition | H2d | diagnosis consistency, `study_need` usefulness |
| `E10` | full combined controller | H2e | verify + redirect + diagnosis closure |
| `Curriculum / RAG` | weakness-conditioned retrieval/adaptation | H3a, H3b, H3c | trigger precision, retry gain, study_need quality |

## 5. Reward Decomposition Contract

The reward family is fixed as follows.

1. `E3`
   - pure calibration baseline
2. `E5`
   - `E3 + confidence_revision`
3. `E6`
   - `E3 + probe_calibration`
4. `E7`
   - `E6 + stepwise_probe`
5. `E8`
   - `E5 + overconfidence shaping`
6. `E9`
   - `E8 + verify only`
7. `E9b`
   - `E8 + redirect only`
8. `E9c`
   - `E8 + diagnosis / decomposition only`
9. `E10`
   - full combined controller

This decomposition is necessary to answer separate questions.

1. what changes from calibration alone
2. what changes when revision is added
3. what behavior reward changes which behavior
4. whether the combined controller adds anything beyond the decomposed pieces

## 6. Probe Contract

The probe is not allowed to be a style classifier.

1. the intended object is `p(correct | prefix)`
2. a single rollout's final correctness is not sufficient supervision for prefix-local uncertainty
3. `E6/E7` are gated on prefix-conditioned targets collected from multiple continuations per prefix
4. until that condition is met, probe-free RL continues but probe-dependent RL does not

The minimum probe checks are:

1. held-out Brier
2. held-out ECE
3. correlation between stated confidence and `p_hat_probe`
4. AUROC only when held-out targets are binary enough to make AUROC meaningful
5. calibration of the probe itself after temperature scaling
6. group split by `problem_id` to avoid leakage

## 7. Curriculum Contract

Curriculum retrieval is valid only if all of the following hold.

1. low confidence alone does not trigger retrieval
2. diagnosis or `study_need` is present
3. retrieved examples are actually inserted into the retry prompt
4. retry artifacts are saved for full later analysis

The curriculum objective is not “retrieve whenever uncertain”.
It is “retrieve when the model knows why the current route is insufficient”.

## 8. Execution Gates

No new broad experiment launches until these gates are satisfied.

### Gate 1. Code stability

Required:

1. tokenizer compatibility is runtime-safe across local and remote `transformers`
2. reward configs match the documented decomposition
3. curriculum smoke passes
4. launch scripts and core modules compile

### Gate 2. Probe-free RL

Allowed when Gate 1 passes.

Runs:

1. `E3`
2. `E5`
3. `E8`
4. `E9`
5. `E9b`
6. `E9c`
7. `E10`

### Gate 3. Probe-dependent RL

Allowed only when all of the following pass.

1. prefix-conditioned targets exist
2. probe smoke passes
3. held-out probe metrics are acceptable
4. probe outputs can be loaded by the reward pipeline

Runs:

1. `E6`
2. `E7`

### Gate 4. Curriculum

Allowed only after diagnosis and `study_need` quality are good enough on held-out analysis.

Runs:

1. redirect-triggered in-context retrieval
2. one-example adaptation
3. later self-distill or RLVR-style follow-up if retry artifacts justify it

## 9. Three-Node Plan

The project uses exactly three training nodes and keeps one other node free for unrelated work.

1. `metacognition_eval`
   - role: probe-free calibration lane
   - order:
     `E3 -> E5 -> E9`
2. `metacognition_train_b`
   - role: probe-free behavior lane
   - order:
     `E8 -> E9b -> E9c`
3. `metacognition_e8`
   - role: gated lane
   - order:
     `probe target generation -> probe smoke -> E6 -> E7`
   - fallback if probe gate is not satisfied:
     `E10`

The core rule is:

`all RL runs start from the same unified SFT initialization`

Operationally, the intended flow is:

1. run `probe-free RL` on two nodes first
2. train and validate the modified probe on one node
3. run `E6/E7` only if the probe gate passes
4. push each completed checkpoint to Hugging Face
5. evaluate `base_sft`, control-v5 SFT variants, and completed RL checkpoints
6. run both quantitative and qualitative analysis before moving to behavior analysis or curriculum

## 10. Smoke / Critic / Improve Loop

Every code path must pass the same loop.

1. run one smoke or unit check
2. identify one concrete mismatch
3. fix one mismatch
4. rerun the relevant smoke
5. keep the change only if the smoke or guard improves

Minimum required checks:

1. `tests/test_rewards.py`
2. `tests/test_gdpo.py`
3. `tests/test_tokenizer_utils.py`
4. `tests/test_probe_pipeline.py`
5. `tests/test_control_rag.py`
6. `scripts/smoke_control_rag.py --skip-model`
7. `python -m py_compile` for core launch and probe scripts

## 11. Current Judgment

The direction remains aligned.

1. `Meta-CoT`
   - teaches parseable controller state
2. `Meta-RL`
   - turns that state into verifiable behavior rewards
3. `Curriculum`
   - extends only after diagnosis and `study_need` are reliable enough

The correct order is still:

`representation -> calibration/revision -> decomposed behavior control -> full controller -> curriculum`

What matters now is not another conceptual pivot.
What matters is keeping the data, code, rewards, and launch conditions aligned to this contract.
