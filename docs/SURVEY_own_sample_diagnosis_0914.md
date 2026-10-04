# Survey: trained verbal metacognition over the model's OWN sample set (2026-09-14)

All arXiv ids below were verified by fetching the abstract page.

## Core hit

**AggLM — "The Majority is not always right: RL training for solution aggregation", arXiv:2509.06870** (Zhao, Aggarwal, Saha, Celikyilmaz, Weston, Kulikov; Sep 2025).
- Reads: K=8 full candidate solutions (post-`</think>` text), K swept 2–16. Generator Qwen3-1.7B (frozen, off-the-shelf); aggregator Qwen3-1.7B trained **separately**. A shared-parameter multitask variant (one model both solves and aggregates) is reported as "close performance".
- Writes: free-form review/reconcile/synthesis then a final answer ("correcting mistakes, filling in gaps, combining useful ideas") — not a forced pick among candidates.
- Label source: gold answer, binary reward `1[y~ = y*]`. RL = GRPO.
- Baselines: majority vote at same K, reward-model selection (AceMath 7B/72B), prompted aggregator without RL, oracle BoN.
- Matched compute: yes, token-budget argument — AggLM-1.7B spends ~1/3 the tokens of the solution models, so aggregating 8 beats maj@16+ at comparable tokens.
- Gain: AIME25 50.0 vs 45.9 maj (+4.1); HMMT24 33.3 vs 29.0 (+4.3). Largest when the majority bloc is small.
- Minority rescue: explicitly analyzed — "outperforms majority vote in the harder cases when the correct answer is infrequent, on par when frequent". Training-set balancing of easy/hard is the stated key ingredient.
- **Not tested**: diagnosis-text vs choice-only ablation; candidate-order shuffling / position-bias control.

## Neighbors

| Paper | id | reads own K? | writes | label | trained? | size | baseline maj@K | gain |
|---|---|---|---|---|---|---|---|---|
| Universal Self-Consistency | 2311.17311 | yes, full | choice | — | prompted | PaLM2/GPT | yes | ≈0 on math |
| CISC | 2502.06233 | own paths | scalar conf | — | prompted | 9 models | yes | maj@K with ~40% fewer paths |
| GenSelect | 2507.17797 | N candidates | long comparative reasoning | — | prompted | QwQ / R1 | pointwise/pairwise | beats scoring baselines |
| Recursive Self-Aggregation (RSA) | 2509.26626 | own population, recursive subsets | new solution | gold (RLVR) | **RL-trained aggregation** | Qwen3-4B-Instruct-2507, Gemini 3 Flash | parallel/sequential scaling | gains; no explicit maj@K matched table in abstract |
| MAPR (meta-awareness) | 2510.03259 | nothing (pre-solve) | predicted pass-rate/length/concepts | own rollout stats | RL (GRPO) | Qwen3-4B | GRPO | +13% avg math, AIME25 +83% rel |
| Self-Calibration | 2503.00031 | SC distilled offline | scalar conf | own samples (SC) | SFT distill | — | SC | efficiency |
| DINCO | 2509.25532 | self-generated distractors | per-claim conf | — | inference-only | — | SC@100 | DINCO@10 > SC@100 |
| TTRL | 2504.16084 | own samples | nothing verbal | majority vote | RL | — | maj@n | surpasses maj@n |
| Co-rewarding | 2508.00410 | own/paraphrase views | nothing verbal | self-consistency | RL | Llama-3.2-3B, Qwen3-8B-Base | — | +7.5% |
| GenRM | 2408.15240 | one solution at a time | CoT verification | gold | trained | — | discriminative RM | — |
| PPV / delegation | 2606.08098 | 128 samples | no text (entropy+embedding delegation) | none | untrained | — | majority | +1.5pp MMLU-Pro |
| VecCISC | 2605.08070 | own traces, clustered | choice | — | prompted | — | CISC/SC | small |

Related-but-off-axis: LLM-Blender/PairRanker (cross-model candidates, trained ranker, no self-sample framing), Self-MoA (2502.00674, prompted single-model MoA), multi-agent debate with trained aggregator (cross-agent, not own-sample).

## Verdict

1. **Closest**: AggLM (2509.06870). Differences in our design: (i) the *same* policy reads its own samples inside one rollout (self-reference), not a separately-trained aggregator over a frozen generator; (ii) the written object is an explicit *diagnosis* — where the K endings disagree and which assumption is wrong — as a rewarded span, not a free synthesis; (iii) labels derived from the model's own rollouts (self-distillation) rather than gold-answer RLVR; (iv) the commit step is a choice over its own candidates, scored against maj@K at matched token budget with content controls.
2. **"Diagnosis text > choice-only" has never been tested.** GenRM argues CoT-before-score > direct score for a *single* solution; no paper ablates diagnosis-vs-choice for a trained selector over its own K.
3. **Minority-correct rescue at ≤8B**: yes, AggLM reports it at **1.7B** (and cross-model generalization), which is the only clean precedent; RSA at 4B shows aggregation gains without a minority-rescue decomposition.
4. **Remaining gap**: no one has trained a *single* policy, on labels from its own rollouts, to write an explicit disagreement/assumption diagnosis over its own K samples and commit — nor isolated how much of any gain comes from the diagnosis content versus the mere act of choosing.
