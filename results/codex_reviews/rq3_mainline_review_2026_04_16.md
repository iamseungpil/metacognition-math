# RQ3 Mainline Review Loop

Date: 2026-04-16

## Scope

Reviewed:

1. [plan_metacot_v8_active_2026_04_09.md](/home/v-seungplee/metacognition/results/plan_metacot_v8_active_2026_04_09.md)
2. [online.py](/home/v-seungplee/metacognition/src/training/self_distill/online.py)
3. [sft.py](/home/v-seungplee/metacognition/src/training/sft.py)
4. [kl.py](/home/v-seungplee/metacognition/src/training/self_distill/kl.py)
5. [run_fixed_k_self_distill_roundtrip.sh](/home/v-seungplee/metacognition/scripts/run_fixed_k_self_distill_roundtrip.sh)
6. [run_online_sdpo_regen.py](/home/v-seungplee/metacognition/scripts/run_online_sdpo_regen.py)
7. Verified result artifacts from `strict_pair_analysis_2026_04_15` and `step300_deep_analysis`

## Iteration 1: Critical Mismatch

### Finding

The plan language was stronger than the launcher contract. `fixed_k_repair` supports retrieval, but the launcher was effectively describing a RAG-enabled path while retrieval was only real if an example bank was supplied. Without that bank, the code ran as repair-only.

### Verification

1. `online.py` loads a retriever only from `example_bank_paths`
2. `_retrieve_examples_for_fixed_k()` returns no retrieval when the retriever is `None`
3. the launcher previously still passed `rag_top_k=1` and `retrieval_query_mode=question_only`, which made the requested mode look active even when no bank existed

### Fix

1. launcher now disables retrieval explicitly when no example bank is supplied
2. runtime now warns when retrieval was requested but no retriever was loaded
3. artifact summary now records retrieval-active, retrieval-enabled, and non-empty retrieval rates

## Iteration 2: Claim Boundary Check

### Finding

The current verified results do not support an OOD-improvement claim from strict SFT or a clean metacognitive-control claim from step-300 RL.

### Verification

1. strict SFT overall accuracy is essentially tied, while OOD is worse for meta
2. step-300 RL gains are accompanied by zero wrapped meta emission, near-constant confidence, and AIME degradation

### Fix

1. active plan now explicitly limits the current claim to controller acquisition plus self-distill testing
2. report now states that RQ3 is open, not solved
3. RL reward redesign remains side-evidence until rerun under the corrected contract

## Converged Position

The intention-hypothesis-validation chain is now:

1. **Intent**: test whether self-distill can preserve useful control and improve OOD behavior
2. **Hypothesis**: reward-ranked repair teachers plus claim-bearing meta preservation outperform naive stripping
3. **Validation**: compare strict base-vs-meta self-distill with explicit collapse metrics, OOD accuracy, and retrieval provenance

This is aligned enough to proceed with bounded mainline experimentation. Retrieval-backed claims remain gated on actual example-bank usage.
