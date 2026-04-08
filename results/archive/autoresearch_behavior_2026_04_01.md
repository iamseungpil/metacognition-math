# Autoresearch Round: Behavior-First Metacognition (2026-04-01)

## Goal

Primary research goal:

- learn an `OOD test-time control policy` that adaptively allocates reasoning effort without further task-specific training
- treat `AIME` as one hard OOD measurement axis, not as the only target

Teach two intended policies before curriculum/RAG:

1. `stuck or contradiction -> lower confidence -> switch method`
2. `high confidence -> verify independently before finalizing`

Success is not just benchmark accuracy. The target is a measurable increase in:

- contradiction-conditioned confidence drop
- effective verification
- genuine strategy switching
- recovery after revision
- difficulty-conditioned compute allocation

Interpretation:

- `AIME` is useful because it is hard and OOD-like for these models
- but the broader claim should be about `meta control under distribution shift`, not about a single benchmark

## Current Evidence

Completed models so far still look closer to `meta phrasing` than to real metacognitive control.

| Model | Overall Acc. | Verify Frac | Redirect Frac | Interpretation |
|---|---:|---:|---:|---|
| Base SFT | 71.7% | 0.13 | 0.072 | No explicit meta, decent raw solving |
| V2 SFT | 72.72% | 1.00 | 0.067 | Verification text learned; often decorative |
| V3 SFT | 72.0% | 0.941 | 0.043 | Better calibration, weak real redirect |
| E7 prev | 69.9% | 0.992 | 0.156 | More meta blocks, not better control |

Working interpretation:

- `verification` has mostly been learned as a textual ritual
- `redirect` remains weak and often not a real method switch
- current training is still partially teaching behavior form rather than control policy

## Round 1 Hypotheses

### H-B1: Combined behavior SFT
- Change: TRAPI-generate behavior-first SFT data with `straight`, `verify`, `redirect` categories.
- Expected effect: better structured confidence updates and more grounded verification.

### H-B2: Redirect-heavy SFT
- Change: train a redirect-focused variant to directly target `stuck -> switch_method`.
- Expected effect: higher genuine redirect rate, even if raw accuracy is not yet best.

### H-B3: Combined SFT + E9 GDPO
- Change: start from the best behavior SFT checkpoint and apply GDPO with behavior-first rewards.
- Expected effect: turn supervised behavior primitives into a more reliable policy.

## Planned Resource Allocation

Budget target: `12 GPUs total`

- `train_b`: current evals continue on 2 GPUs
- `eval-e8`: current evals continue on 3 GPUs
- `tops-caiman`: 4 GPUs for `H-B1` then `H-B3`
- `metacognition_e8`: 4 GPUs for `H-B2`

This keeps total active usage within the requested envelope while avoiding disruption to ongoing evals.

## Live Status

As of 2026-04-01 UTC:

- ongoing evals
  - `train_b`: `E3_500`, `E5`
  - `eval-e8`: `E7 current`, `V2 rich`, `E8`
- idle training nodes reserved for round 1
  - `tops-caiman`: `H-B1` combined behavior SFT, then `H-B3` E9 GDPO
  - `metacognition_e8`: `H-B2` redirect-heavy behavior SFT
- TRAPI generation
  - pilot complete: `data/metacot_behavior_trapi_round1.parquet`
  - requested target: `1,800` chains (`600` each for `straight`, `verify`, `redirect`)
  - valid output: `770` chains
  - observed class balance after validation:
    - `redirect`: `552`
    - `verify`: `216`
    - `straight`: `2`
  - implication: the current prompt/validator strongly favors redirect-style samples and is not yet suitable for a balanced main run

## Pilot Learnings

Round-1 pilot supports the current research direction, but also exposes a major data issue:

- the `behavior-first` framing is useful because it targets action primitives rather than meta phrasing
- however, the current generation/validation stack is collapsing the dataset toward `redirect`
- this means the next iteration should not scale to 6k yet
- first fix:
  - `straight` acceptance
  - `verify` acceptance
  - scenario-balanced generation quotas
  - remote launch reliability for SFT/GDPO jobs

## Plan Evaluation

Against the current plan, the research direction is mostly correct.

What is aligned:

- the plan now targets `meta control` rather than more meta text
- `verify when confident` and `redirect when stuck` are the right first behaviors
- `AIME` is being used as a hard slice, not the only target
- curriculum/RAG remains gated until these behaviors are reliable

What is currently off-plan:

- round-1 pilot data is not balanced enough to support the intended control policy
- the current valid set is dominated by `redirect`, which would bias training toward over-revision
- the automatic remote SFT launch did not actually produce running jobs on the two idle training nodes

Immediate autoresearch decision:

- keep the behavior-first direction
- do **not** scale this pilot dataset to the main run
- first repair:
  - balanced generation
  - remote launcher reliability
  - then continue with SFT -> E9 GDPO

## Novelty Position

The likely novelty is **not**:

- "we also use confidence rewards"
- "we also combine correctness and calibration"

Those are already close to existing calibration work.

The more defensible novelty would be:

- confidence is used as a `control variable`, not just a reported score
- meta learning is evaluated by `behavioral consequences`:
  - lower confidence when contradiction appears
  - spend more compute on harder inputs
  - switch strategy when stuck
  - verify when highly confident
- the contribution is an `actionable metacognitive controller` for OOD reasoning, rather than confidence calibration alone

## Artifacts

- TRAPI generator: `scripts/gen_behavior_trapi.py`
- Variant builder: `scripts/build_behavior_sft_variants.py`
- Prompt: `src/metacot/prompt_behavior.py`
- SFT configs:
  - `configs/sft_behavior_all.yaml`
  - `configs/sft_behavior_redirect.yaml`
  - `configs/sft_behavior_verify.yaml`
- GDPO mode: `E9` in `src/training/grpo_v2.py`
