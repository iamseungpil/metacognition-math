# When does metacognition actually help? A discriminating survey

**Question.** Prior work claims (a) LLMs carry usable internal signals of correctness and (b) metacognitive
behaviours (verification, backtracking, self-correction, planning, confidence) can be trained to raise accuracy.
**Under what conditions were those gains obtained, and do any of those conditions hold for us?**

**Our policy and results, for reference.** Qwen3-4B-Instruct-2507, **non-thinking**, already RLVR'd by its vendor.
MATH-500 pass@1 .744; pass@8 ≫ pass@1 on AIME25/HMMT25 (+23–27 pts). Failed gates:
(G-donor) inserted mid-solution metacognition ≈ a block copied from another problem (own +.025 vs donor +.024);
(G-geom) end-of-block hidden-state geometry does not separate right from wrong **within** a problem
(per-problem AUC .51–.55); verbalized confidence within-problem AUC .50; **sibling-plurality (agreement with
majority of 8 samples) AUC .78**; (G-retry) in-context retry after a completed wrong answer rescues 0/1,248,
seeded "redirect" instructions overridden 95.6%, 86% verbatim restatement;
(G-critique) own critique with answer hidden makes re-solving **worse** than no critique (.332 vs .361) and worse
than a critique from another problem (.382); teacher-forced likelihood of the correct sibling drops more under own
critique than under a generic hint; (G-rl) 30 steps correctness-only GRPO: held-out accuracy unchanged,
meta emission .31→.43.

---

## Master table

Abbreviations — **Where**: B=before solving, M=mid-solution, E=end-of-solution, X=across samples.
**Class**: base / instr = instruction-tuned non-thinking / think = long-CoT or R1-distill / RLVR' = already RL-trained.
**W/A**: within-problem (right-vs-wrong samples of the *same* prompt) vs across-problem (difficulty ranking).
**CF control**: does the paper run a content-free control (donor / shuffled / random / generic / format-only), and does the gain survive?

| # | Paper | Signal / behaviour | Where | Model class & size | Baseline the gain is against | Metric | W/A | Content-free control | Magnitude |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Kadavath et al. 2022**, *LMs (Mostly) Know What They Know* ([2207.05221](https://arxiv.org/abs/2207.05221)) | P(True) self-eval; P(IK) "do I know this" | E (P(True) judges a finished sample); **X when 5 samples are shown** | Anthropic dense LMs 800M/3B/12B/**52B**, base+RLHF | Chance (AUROC .5); and P(True) with one sample vs with 5 comparison samples | AUROC, Brier, calibration | **Across-problem for P(IK); within-problem for P(True) but weak on math** | ✅ **two of them.** (a) *Distracting hints generated for a different question* (donor) → lower P(IK) than good hints; (b) **two 12B models with different pretraining, TriviaQA questions one gets right and the other wrong** — model-specific P(IK) gap is **only ~6 pts** (.463 vs .409 / .477 vs .408) | On **GSM8K the P(True) AUROC axis runs .45–.70**; the 52B model tops out ≈.65–.70 and *smaller models are near chance*. Zero-shot "P(True) is poorly calibrated, and typically it lies close to 50%". Showing 5 T=1 samples "improves performance **significantly** on all short-form tasks" |
| 2 | **Kuhn/Farquhar et al. 2023–24**, *Semantic entropy* ([Nature 630:625](https://www.nature.com/articles/s41586-024-07421-0)) | Entropy over *meaning-clusters* of multiple generations | **X (10 samples per question)** | LLaMA-2-Chat 7/13/70B, Falcon 40B, Mistral 7B — instr, not thinking | Naive token entropy, P(True), embedding regression | AUROC / rejection-accuracy | **Across-problem**, short-form QA; **no competition math** | none (it *is* a sampling method) | Clear AUROC gains over naive entropy and P(True); requires ~10 generations |
| 3 | **Semantic Entropy Probes** ([2406.15927](https://arxiv.org/abs/2406.15927)) | Linear probe on hidden states predicting semantic entropy — **the single-rollout version of #2** | E, single generation | Llama-2-7B/70B, Llama-3-70B, Mistral-7B, Phi-3-3.8B (instr) | Accuracy probes; sampling-based SE | AUROC | Across-problem, QA only, **no math** | none | In-dist ≈ accuracy probes (±1–3 AUROC); OOD +7.7–10.5 over accuracy probes; **"SEPs cannot match the performance of these methods" (the 10× costlier sampling baselines)** |
| 4 | **Huang et al. 2023**, *LLMs Cannot Self-Correct Reasoning Yet* ([2310.01798](https://arxiv.org/abs/2310.01798)) | Intrinsic in-context self-correction | E→retry | GPT-3.5, GPT-4, GPT-4-Turbo, **Llama-2** (instr, non-thinking) | Same model, standard prompting, matched # calls | Accuracy | Within-problem (same prompt, revise) | ✅ **oracle-label condition is the positive control** | GSM8K **GPT-3.5 75.9→75.1→74.7**; **GPT-4 95.5→91.5→89.0**; GPT-4-Turbo 91.5→88.0→90.0; **Llama-2 62.0→43.5→36.5**. With oracle labels: 75.9→84.3, 95.5→97.5. "For GSM8K, **74.7% of the time GPT-3.5 retains its initial answer**… more likely to modify a correct answer to an incorrect one" |
| 5 | **Huang et al. 2023, Table 7** (same paper, matched-compute arm) | Multi-agent debate | X | GPT-3.5 | **Self-consistency at equal # responses** | Accuracy | X | this *is* the control | 1 resp 76.7 · SC@3 82.5 · **Debate r1 (6) 83.2 vs SC@6 85.3** · **Debate r2 (9) 83.0 vs SC@9 88.2**. Debate loses to plain majority vote at every matched budget |
| 6 | **Tyen et al. 2024**, *LLMs cannot find reasoning errors, but can correct them given the location* ([ACL Findings 826](https://aclanthology.org/2024.findings-acl.826/)) | Error **localisation** vs error **correction**, separated | M (find the first bad step) | GPT-4, GPT-4-Turbo, GPT-3.5-Turbo, PaLM-2 Unicorn, Gemini Pro | Random-location backtracking; no-backtracking | Mistake-finding accuracy; Δaccuracy | Within-problem | ✅ **"with random location" column** | Mistake finding: GPT-4 best **avg 39.8 / 52.9 / 43.4** across three prompting styles; PaLM-2 **22.0/21.7/23.7**. Backtracking Δacc on originally-wrong traces, **oracle vs random**: multistep-arith **+18.04 vs +10.59**; tracking-objects **+43.92 vs +20.39**; logical-deduction **+36.86 vs +21.57**; word-sorting **+23.53 vs +11.76**; Dyck **+18.06 vs +5.16**. **~40–70% of the "self-correction" gain is reproduced by a random location.** |
| 7 | **Kumar et al. 2024, SCoRe** ([2409.12917](https://arxiv.org/abs/2409.12917)) | Multi-turn **online RL** for self-correction, 2-stage + shaped reward | E→retry, trained | Gemini 1.5 Flash / 1.0 Pro (large instr; not open, not small) | Base model, Self-Refine, STaR, Pair-SFT | Acc@t1, Acc@t2, Δ(t1,t2) | Within-problem | partial (STaR/Pair-SFT are the "SFT can't do it" controls) | MATH: base 52.6/41.4/**−11.2**; Self-Refine 52.8/51.8/−1.0; STaR 53.6/54.0/+0.4; Pair-SFT 52.4/54.2/+1.8; **SCoRe 60.0/64.4/+4.4** (Δ^{i→c} 5.8, Δ^{c→i} 1.4). **Note: 7.4 of the 11.8-pt Acc@t2 gain over Pair-SFT comes from Acc@t1 rising 52.6→60.0 — i.e. ordinary RLVR, not self-correction.** Matched compute: at a 32-sample budget, parallel sampling **+7.4%**, parallel+one self-correction round **+10.5%** |
| 8 | **Bensal et al. 2025, Reflect–Retry–Reward** ([2505.24726](https://arxiv.org/abs/2505.24726)) | Self-reflection after failure, GRPO on the reflection tokens only | E→reflect→retry | Qwen2/2.5 1.5B/3B/7B, Llama-3.1/3.2 3B/8B, Phi-3.5-mini 3.8B, Palmyra 1.7B — **instr, not RLVR'd, not thinking** | Same model first attempt | Accuracy | Within-problem | ❌ **no generic/random-reflection control** | APIGen Qwen2-1.5B 32.6→48.6, Qwen2-7B 66.4→72.2, Llama-3.1-8B 64.9→68.7. Countdown Qwen2.5-1.5B 6.0→34.9, 7B 31.7→41.6, Llama-3.1-8B 2.2→8.8. **⚠ The model is TOLD it failed by an external binary verifier before it reflects.** **⚠ No MATH/AIME evaluation** — the two tasks are function-calling and Countdown equation writing |
| 9 | **Thought Anchors 2025** ([2506.19143](https://arxiv.org/abs/2506.19143)) | Sentence-level counterfactual importance; "plan generation" and "uncertainty management" sentences | M | **DeepSeek-R1-Distill-Qwen-14B** (thinking), + R1-Distill-Llama-8B | n/a — pure analysis | KL / Δ P(correct) | Within-problem (resample forward from sentence *i*, 100×) | ❌ semantic filtering (cos<0.8) only; **no unrelated/donor-sentence insertion control** | Plan-generation and uncertainty-management sentences have the highest counterfactual importance; active computation lowest. **No accuracy gain is demonstrated** — intervention only ever *hurts* (ablating 512 receiver heads: 64.1→27.7, vs 37.3 for random heads) |
| 10 | **Gandhi et al. 2025**, *Cognitive Behaviors that Enable Self-Improving Reasoners* ([2503.01307](https://arxiv.org/abs/2503.01307)) | Verification, backtracking, subgoal setting, backward chaining — as **priors before RL** | B (priming) then whole trajectory | Qwen-2.5-3B vs **Llama-3.2-3B** — 3B **base**, behaviour-poor | Same model under identical RL, un-primed | Accuracy on Countdown | Across-problem | ✅ **two**: (a) priming with *incorrect* solutions containing the behaviours gives "identical performance" to priming with correct ones; (b) **empty-CoT and length-matched placeholder-token controls ≈30–35%**, i.e. "mere allocation of additional tokens without cognitive behaviors fails" | Post-RL Countdown: Qwen-2.5-3B ≈**60%**, Llama-3.2-3B ≈**30%**, from a similar low start. All-strategies-primed Llama matches/exceeds Qwen's trajectory; behaviour-enriched OpenWebMath continued pretraining lets Llama reach comparable performance. **The gain exists only because the starting policy LACKS the behaviours** |
| 11 | **Wang et al. 2025**, *Beyond the 80/20 Rule* ([2506.01939](https://arxiv.org/abs/2506.01939)) | High-entropy "forking" tokens; gradient restricted to top-20% | M, per token | Qwen3-8B / 14B / 32B **base** | **Vanilla DAPO** (already a strong RLVR baseline) | Accuracy AIME24/25 | Not a per-sample score | ⚠ bottom-80% control ✅, but **no random-20% mask control** | 32B **+7.71 / +11.04**; 14B +5.21/+4.79; **8B ≈ +0.5–0.83 (noise)** — an explicit small-model negative. Top-20% entropy positions overlap **86.67%** between base and converged RL model |
| 12 | **Zhao et al. 2025, Intuitor / RLIF** ([2505.19590](https://arxiv.org/abs/2505.19590)) | Self-certainty (KL to uniform) as the *only* reward | E, normalised **within the GRPO prompt group** | Qwen2.5 1.5–14B **base**, Qwen3-14B, Llama3.2-3B-Instruct, OLMo-2-7B-SFT | Base model; **GRPO with gold reward** | Accuracy; Mann-Whitney U on correct vs incorrect self-certainty (**no AUROC**) | Within-problem by construction | ✅ **coin-flip 0/1 reward → severe degradation** | Qwen2.5-3B: GSM8K .673→.792, MATH500 .544→.612, CRUXEval .236→.416. **Gold-reward GRPO beats it in-domain (.826/.636)**. Llama3.2-3B-Instruct MATH500 .436→.476 (GRPO .494); OLMo CRUXEval .238→**.215 (down)**; **training collapsed on Llama3.2-3B-Base** |
| 13 | **No Free Lunch: Rethinking Internal Feedback** ([2506.17219](https://arxiv.org/abs/2506.17219)) | Audit of #12 | E / token-level | Qwen2.5-3B(-Instruct), Qwen3-1.7B/**4B** base, Qwen2.5-Math-1.5B | Pre-training model; RLVR | Accuracy over steps | — | — | **"RLIF can boost… base LLMs at the beginning phase… when training progresses, performance degrades even below the model before training"** and **"RLIF yields little improvement for instruction-tuned models."** Qwen3-1.7B MATH500 .587→.720@step20→.702@step80→below start. Mechanism attributed to **initial policy entropy**: high-entropy (base) checkpoints gain, instruction-tuned ones do not |
| 14 | **Shao et al. 2025, Spurious Rewards** ([2506.10947](https://arxiv.org/abs/2506.10947)) | random / incorrect / format-only rewards | E | Qwen2.5-Math-1.5B/7B, Qwen2.5-1.5B/7B, **Llama3.1-8B(-Instr), Llama3.2-3B(-Instr), OLMo2-7B(-SFT)** | Same model pass@1 pre-RLVR | Accuracy MATH-500 | — | ✅ **the paper is the control** | Qwen2.5-Math-7B MATH-500: gold **+29.1**, incorrect **+24.1**, **random +21.4**, format ≈+13.8. Code-reasoning frequency 65.0%→**95.6%** under random rewards. **Llama3 / OLMo2 show minimal gain or degrade under spurious rewards** |
| 15 | **TTRL** ([2504.16084](https://arxiv.org/abs/2504.16084)) | Majority-vote pseudo-label as reward | **X (same prompt, n samples)** | Qwen2.5-Math-1.5/7B, Qwen2.5-7/32B, Qwen3-8B, LLaMA-3.1-8B | Same model pass@1; maj@n; RL-with-gold ceiling | pass@1 / avg@16 | Within-problem via vote | ❌ no shuffled-vote control | Qwen2.5-Math-7B AIME24 **12.9→40.2**, MATH-500 46.7→83.4, AMC 35.6→68.1. **Negatives**: GPQA 29.1→27.7; gains shrink with difficulty (MATH-500 L1 +175% vs **L5 +75%**); LLaMA-3.1-8B only 4.6→10.0 |
| 16 | **Setlur et al. 2024, Rewarding Progress (PAVs)** ([2410.08146](https://arxiv.org/abs/2410.08146)) | Step-level *progress* = ΔP(correct) **under a prover policy μ ≠ π** | M, per step | Gemma 2B/9B/27B SFT'd on MATH | ORM best-of-N; ORM-only RL; RFT | Accuracy, compute/sample efficiency | Within-problem | ✅ **analytic null: prover = base policy ⇒ the update is identical to outcome reward alone** | Search **>8%** over ORM at 1.5–5× less compute; online RL **>6–7%** accuracy at **5–6×** sample efficiency; vs RFT +11% (2B), +15% (9B) |
| 17 | **RLPR** ([2506.18254](https://arxiv.org/abs/2506.18254)) | Mean token probability of the **reference answer** | E | Qwen2.5-0.5/3/7B(+Instr), Llama3.1-8B-Instr, Gemma2-2B-it | Rule-verifier RLVR; General-Reasoner; VeriFree | Accuracy **+ ROC-AUC of the reward** | ✅ reports AUROC of reward vs correctness | partial (ablation, no shuffled reference) | Reward AUROC **.97 math / .81 general** vs rule-verifier .95/.61. Qwen2.5-7B MATH-500 63.0→78.0, Minerva 37.6→56.5. **Negative: AIME24 16.3 vs 17.7 for rule RLVR — loses on competition math.** Note the signal is **not internal**: it needs the gold answer |
| 18 | **Quiet-STaR** ([2403.09629](https://arxiv.org/abs/2403.09629)) | Latent rationales rewarded by Δ next-token likelihood | B/M, every token | **Mistral-7B base only** | Same model continued-pretrained without thoughts | Zero-shot accuracy | not per-sample correctness | ❌ no random-thought control | GSM8K **5.9→10.9**, CQA 36.3→47.2, zero-shot, no task FT. Absolute levels are base-model-scale |
| 19 | **Cui et al. 2025**, *Entropy Mechanism of RL* ([2505.22617](https://arxiv.org/abs/2505.22617)) | R = −a·exp(H)+b; performance is *traded from* entropy | training-level | 11 base models, 4 families (Qwen2.5 0.5–32B, Mistral, Llama, DeepSeek-Math) | GRPO | Accuracy | — | — | Qwen2.5-32B AIME24 +10.5/+15.0 over GRPO; **7B only +1.8/+2.0 avg**. Ceiling set by the entropy the initial policy brought in |
| 20 | **Yue et al. 2025**, *Does RL Really Incentivize Reasoning Capacity…* ([2504.13837](https://arxiv.org/abs/2504.13837)) | pass@k boundary | X | Qwen2.5-7/14/32B, LLaMA-3.1-8B, R1-Distill 7/14B | **Base model at matched k** | pass@k, coverage, perplexity | Within-problem (solvable-set coverage) | is itself the negative control for RLVR | RLVR wins at k=1, **base wins at large k**; RL solutions already lie in the base sampling support. **Distillation** does raise the pass@k curve |
| 21 | **Reasoning Boundary Paradox** ([2510.02230](https://arxiv.org/abs/2510.02230)) | Why RLVR shrinks pass@k | X | multiple math benchmarks | base | pass@k | — | — | Two mechanisms: **negative interference** and a **winner-take-all effect** that "disproportionately reinforces problems with high likelihood, correct solutions under the base model, while suppressing initially low-likelihood ones" |
| 22 | **Zhang et al. 2025**, *Reasoning Models Know When They're Right* ([2504.05419](https://arxiv.org/abs/2504.05419)) | **Supervised linear probe** on hidden states at intermediate answers | M (at each intermediate answer chunk) | R1-Distill-Qwen-1.5/7/32B, R1-Distill-Llama-8/70B, QwQ-32B — **all thinking**; + Llama-3.1-8B-Instruct as a non-reasoning contrast | chance; prior-token baselines | ROC-AUC, ECE | ROC **pooled across problems** (the paper never states a within-problem ROC) | ❌ none | AUROC GSM8K .82–.91, MATH .80–.89, AIME .69–.91, ECE<0.1. Transfers MATH↔GSM8K, **does not** transfer across domains. **Payoff is token saving, not accuracy: −24% tokens at 88.2% (same as baseline).** **§4.4: "the probe trained on non-reasoning model representations performs much worse than its reasoning counterpart"; "the encoded information on answer correctness is more prominent in reasoning models"** |
| 23 | **Self-Correction Bench 2025** ([2507.02778](https://arxiv.org/abs/2507.02778)) | "Self-correction blind spot"; the **"Wait"** token | E→retry | **14 open-source non-reasoning models** (Llama-4-Maverick-17B, DeepSeek-V3, Qwen2.5-72B, Llama-3.3-70B, Qwen2-7B…) + reasoning models (R1, Qwen3-thinking, phi-4-reasoning-plus) | same model, same injected error presented as the *user's* error | correction rate | Within-problem | the user-error framing **is** the control | Blind spot **64.5%** on non-reasoning models ≈ 0 on reasoning models. Appending **"Wait"**: blind-spot −89.3%, accuracy +156% macro; GSM8K-SC **0.183→0.796**, PRM800K-SC 0.200→0.504, SCLI5 0.499→0.957. **⚠ errors are INJECTED, not the model's own natural errors.** Root cause: SFT corpora have correction-marker density 1–2 at the 95th percentile, RL-generated reasoning corpora **30–170** |
| 24 | **Reasoning Models Struggle to Control their CoT** ([2603.05706](https://arxiv.org/abs/2603.05706)) | Instruction-following *inside* the reasoning trace | M | Claude Sonnet 4.5 and others | same model controlling the final answer | controllability rate | Within-problem | — | **2.7% CoT controllability vs 61.9% output controllability** — a 59.2-pt gap. **"CoT controllability… decreases with more RL training, test-time compute, and increased problem difficulty"** |
| 25 | **Are Reasoning LLMs Robust to Interventions on their CoT?** ([2602.07470](https://arxiv.org/abs/2602.07470)) | 7 benign/neutral/adversarial CoT interventions | M | multiple open-weight RLLMs | un-intervened trace | accuracy, CoT length | Within-problem | intervention taxonomy, no donor control | RLLMs "**generally robust, reliably recovering from diverse perturbations**"; robustness ↑ with size; adversarial noise inflates CoT length **>200%** |
| 26 | **Distilling Self-Consistency into Verbal Confidence** ([2604.24070](https://arxiv.org/abs/2604.24070)) | Distil a sampling signal into a **single-rollout** verbal confidence | E | **Gemma 3 4B-it** — our scale and class | pre-registered protocol | AUROC | Across-problem (TriviaQA, MMLU) | ✅ **shuffled-target control = 0.501** | **Pre-registered phase failed: AUROC 0.554 → 0.509** (label-entropy collapse). Post-hoc rescue without the modal filter: TriviaQA AUROC 0.774; MMLU 54.2%→77.4% |
| 27 | **When LLMs Agree, Are They Right?** ([2607.08065](https://arxiv.org/abs/2607.08065)) | Audit of self-consistency / cross-model agreement as a confidence signal | X | frontier + mid-tier models; 265k samples | — | rank correlation, AURC | Within-problem | — | Agreement is "a positive but **weak** predictor (ρ 0.20–0.59)"; on GPQA self-consistency ρ=0.31 vs verbalized 0.21; **on AIME verbalized confidence parsed on only 3/50 cases and P(True) ρ=0.12.** "Self-consistency is a **conditional proxy for correctness, not a standalone confidence score**"; 77% agreement on GPQA yet **48% of those wrong** |
| 28 | **LLMs cannot spot math errors, even when allowed to peek into the solution** ([2509.01395](https://arxiv.org/abs/2509.01395)) | First-error localisation in stepwise math solutions | M | SOTA LLMs, VtG + PRM800K | — | localisation accuracy | Within-problem | — | Models "**struggle to locate the first error step… even when given access to the reference solution**" |
| 29 | **Small Language Models Need Strong Verifiers to Self-Correct Reasoning** ([2404.17140](https://arxiv.org/abs/2404.17140), ACL Findings 2024) | Self-correction with a self-verifier vs an external strong verifier | E→retry | small LMs (1B–13B class), instr | same model without correction; correction with a strong verifier | Accuracy (GSM8K, math + commonsense) | Within-problem | the strong-verifier arm is the control | Small models "struggle with self-correction when subjected to a **weak self-verifier** fine-tuned on self-generated solutions"; correction is effective **only with strong verifiers** |
| 30 | **Mind the Gap: Self-Improvement Capabilities of LLMs** ([2412.02674](https://arxiv.org/abs/2412.02674)) | **Generation–verification gap** as the scaling quantity | X / E | Qwen-1.5-0.5B, Qwen-2-0.5B, Llama-2-7B up to frontier | own generation accuracy | gen–ver gap | Within-problem | — | Small models "**cannot self-improve, with non-positive generator–verifier gaps for nearly all verification methods, even though the models have non-trivial generation accuracy**." Self-improvement requires a minimum instruction-following/reasoning floor |
| 31 | **s1: Simple test-time scaling** ([2501.19393](https://arxiv.org/abs/2501.19393)) | Budget forcing: suppress `</think>`, append **"Wait"** | M (extends thinking) | s1-32B = 1k-sample SFT of **Qwen2.5-32B-Instruct** | no-extrapolation; **"2× without string"**; majority voting on the base instruct model | Accuracy | Within-problem (sequential) | ✅ **the best content-free control in the whole table**: *"2× without string"* (double the thinking budget, no injected token) = **50.0 / 90.2 / 55.1** (AIME24/MATH500/GPQA) vs no-extrapolation **50.0 / 93.0 / 57.6** — **zero AIME gain, and it HURTS MATH500 and GPQA**. Only *"2× Wait"* helps: **53.3 / 93.0 / 59.6** | AIME24 50 → 57 end-to-end; "flattens out at six times"; over-suppression "can lead the model into repetitive loops". **⚠ The majority-voting curve is only plotted, never tabulated** — no matched-token number exists |
| 32 | **L1 / LCPO** ([2503.04697](https://arxiv.org/abs/2503.04697)) | RL with a length constraint stated in the prompt | whole generation | L1-1.5B from DeepScaleR-1.5B-Preview (Qwen2.5-1.5B-Instruct lineage) | **GPT-4o and Llama-3.3-70B at matched output length**; s1 at matched budget | Accuracy vs tokens | Across-problem | matched-length comparison is the control | ≈ **+2 pts average over GPT-4o at 858 vs 867 tokens**; vs s1 *"over 100–150% relative and 20–25% absolute gains at both 512 and 1024 token budgets"*. (Per-benchmark cells disagreed between fetch passes — treat as unverified) |
| 33 | **ThinkPrune** ([2504.01296](https://arxiv.org/abs/2504.01296)) | RL with an iteratively tightened token cap (4k→3k→2k) | whole CoT | R1-Distill-Qwen-1.5B, DeepScaleR-1.5B-Preview, QwQ-32B (thinking) | the un-pruned same model | Accuracy vs tokens | Across-problem | — | *"reasoning length … reduced by half with only 2% drop"* (AIME24, 1.5B). **Negative worth noting: on the two already-RL-saturated models pruning COSTS accuracy (−2.5, −3.2 avg pts); only the unsaturated distill improved** (table read, unverified) |
| 34 | **AdaptThink** ([2505.13417](https://arxiv.org/abs/2505.13417)) | RL to choose **Think vs NoThink per problem** — the closest published "metacognitive control" | **B (mode choice before reasoning)** | R1-Distill-Qwen-**1.5B and 7B** (thinking) | original Thinking model **and NoThinking prompting alone** | Accuracy + length | Within-problem decision, across-problem evaluation | the NoThinking-alone arm is the control | **1.5B +2.4% acc / −53.0% length; 7B +2.3% acc / −40.1% length.** NoThinking alone is far worse (1.5B AIME24 29.4→14.0; 7B 53.5→24.2) and competitive only on **MATH Level 1–3** — the gain is in the *selection*, not the cheap mode |
| 35 | **Arora & Zanette 2025** ([2502.04463](https://arxiv.org/abs/2502.04463)) | RL reward = correctness − α·length | whole CoT | R1-Distill-Qwen-1.5B/7B | same model at α=0 | Accuracy vs tokens | Across-problem | — | Prose claim: 7B on MATH *"decreases by 36% … while the accuracy loss is only 2.2%"*. ⚠ appendix table read contradicts the prose — cite the prose only |
| 36 | **Self-Consistency** ([2203.11171](https://arxiv.org/abs/2203.11171)) | majority vote over k paths | **X** | PaLM-540B, code-davinci-002, LaMDA-137B, UL2-20B | greedy CoT (**k=1 vs k=40 — not matched compute**) | Accuracy | X | — | GSM8K PaLM-540B **56.5 → 74.4 (+17.9) at k=40**; +11.0 SVAMP, +12.2 AQuA. Authors' own caveat: *"the gain is relatively lower for smaller models"* |
| 37 | **Universal Self-Consistency** ([2311.17311](https://arxiv.org/abs/2311.17311)) | the LLM itself picks the most consistent of k | **X** | PaLM 2-L, gpt-3.5-turbo | greedy; **and standard SC** | Accuracy | X | standard SC is the control | k=8: GSM8K 85.7→90.2, MATH 30.8→37.4. **USC matches but does not beat standard SC** (90.2 vs 90.4; 37.4 vs 37.9) — letting the model do the selection buys generality, not accuracy |
| 38 | **Snell et al. 2024**, *Scaling Test-Time Compute Optimally* ([2408.03314](https://arxiv.org/abs/2408.03314)) | **difficulty-aware** allocation between verifier search and sequential revisions | X + sequential | PaLM 2-S* + PRM | **best-of-N at matched compute** | Accuracy (MATH) | Across-problem (difficulty bins from base pass rate) | matched-compute by construction | *"outperform best-of-N using up to **4× less** test-time compute (e.g. 64 samples verses 256)"*; search *"16 verses 64"*. **Direction matters for us: "easy questions benefit more from sequential revisions, whereas on difficult questions it is optimal to strike a balance between sequential and parallel"**, and beam search over-optimizes on Level 1–2 |
| 39 | **Brown et al. 2024**, *Large Language Monkeys* ([2407.21787](https://arxiv.org/abs/2407.21787)) | repeated sampling; coverage vs selection | **X** | Llama-3-8B/70B-Instruct, Gemma, DeepSeek-Coder-V2 | majority voting / reward models at the same k | pass@k coverage vs selected accuracy | Within-problem (solvable set) | — | **MATH, Llama-3-8B-Instruct: coverage 79.8%@100 → 95.3%@10,000, while majority voting / RM selection go only 38.7% → 39.8%.** A **~55-point** generation–selection gap. *"methods for picking from a sample collection … plateau beyond several hundred samples"* |
| 40 | **SRPO (long-horizon)** ([2608.23493](https://arxiv.org/abs/2608.23493)) | **Reset-with-memory**: distil a failed trajectory into a "reflection patch", prepend it, regenerate from a clean state; reflection-conditioned teacher scores give dense token-level signal | after a failed rollout | **Qwen3-8B** | scaled SFT | Accuracy | Within-problem | ❌ none reported | AIME'24 **73.3%** at **8% of scaled-SFT FLOPs**. Per-baseline deltas and inference-compute matching not reported |
| 41 | **Credit Assignment with Resets** ([2605.25507](https://arxiv.org/abs/2605.25507)) | Model **self-localises the erroneous step**, resets *there*, samples suffix continuations, learns from their rewards | **M (self-chosen reset position)** | not stated in abstract | GRPO **and RRPO (Random-Reset PO)** | Accuracy | Within-problem | ✅ **RRPO is a random-position control — the closest published analogue of our "citation site vs random site" counterfactual** | "consistently outperforms GRPO and RRPO, using only the model itself, no external supervision". Sizes/numbers not in abstract — needs a full read |
| 42 | **MAPR**, *Verifying Meta-Awareness via Predictive Rewards* ([2510.03259](https://arxiv.org/abs/2510.03259)) | Model **predicts its own rollout statistics** (length, pass-rate, concepts); prediction accuracy is the reward | **B (before solving)** | **Qwen3-4B / 8B / 14B**, Llama-3.1-8B-Instruct, Gemma-2-9B-IT | **GRPO** | Accuracy | Across-problem (predicting *problem-level* difficulty) | ❌ no placebo-meta control | **AIME25 11.77 → 21.56 Pass@1 (Qwen3-4B vs GRPO), +13.04% avg relative over 6 math benchmarks**, 1.28× training speedup; meta-prediction ≈15.5% of training tokens. **The single most important precedent for us — and note it is an ACROSS-PROBLEM, BEFORE-SOLVING signal, not within-problem mid-solution** (single-pass read, verify before citing) |
| 43 | **SVR: Self-Verifying Refinement** ([2607.28457](https://arxiv.org/abs/2607.28457)) | Joint verdict + confidence RL; keep the answer only if verdict=correct AND confidence high, else refine | E→retry, trained | **Qwen3.5-2B** instruct | fixed ten-turn inference | Accuracy vs turns | Within-problem | ❌ | 0.563 macro-avg accuracy at **2.99 average turns** over 7 math datasets. **Compute matching not stated.** Mechanically almost exactly our M_RETRY arm, at 2B |
| 44 | **SEVRA**, *Think Again or Think Longer?* ([2606.19808](https://arxiv.org/abs/2606.19808)) | Serving-layer gate: keep the initial answer or invoke verification | E | **frozen Qwen3-4B** — our exact policy class | always-verify; **and simply generating longer** | Accuracy vs tokens | Within-problem gate | matched-budget arm present | Selective verification **76.3%** vs always-verify **75.5%**, −26.8% post-generation tokens — **but the paper's own honest negative: a plain 8,192-token initial solve reaches 76.0% with 28% fewer TOTAL tokens.** *The gate loses to "just think longer" at matched budget on a Qwen3-4B.* |
| 45 | **MaR**, *Metacognition as Reward* ([2605.23384](https://arxiv.org/abs/2605.23384)) | Trajectory reward over knowledge coverage + regulation fidelity + correctness | whole trajectory | Qwen3.5-9B | base model; **vanilla DAPO** | Accuracy, 22 benchmarks | Across-problem | ❌ | **+7.7% over base, +11.0% over vanilla DAPO**. Reward is still shaped on metacognitive *form*, not on demonstrated usefulness |
| 46 | **RLMF** ([2606.32032](https://arxiv.org/abs/2606.32032)) · **C3RL/CAS** ([2607.01612](https://arxiv.org/abs/2607.01612)) · **SRRL** ([2607.05541](https://arxiv.org/abs/2607.05541)) · **CPT** ([2606.00869](https://arxiv.org/abs/2606.00869)) · **Cog-Rethinker** ([2510.15979](https://arxiv.org/abs/2510.15979)) · **Learning to Self-Verify** ([2602.07594](https://arxiv.org/abs/2602.07594)) | 2026 metacognition-RL cluster | varies | Qwen3-4B / OLMo-3-7B / 14B / unspecified | RL baselines | mostly calibration or abstention | mixed | ❌ **none runs a content-free / placebo-meta control** | RLMF "surpasses standard RL by up to 63%" but targets **faithful uncertainty expression (calibration)**, the sub-goal we explicitly distinguish from accuracy. CPT at 14B: +2.2 math-avg, +5.6 abstention-F1. SRRL: **GSM8K only, no numbers**. *Learning to Self-Verify* reports a **directional asymmetry: better generation does not improve verification, but verification training does improve generation** |
| 47 | **The Self-Correction Illusion** ([2606.05976](https://arxiv.org/abs/2606.05976)) | **Role relabelling** of self-generated errors | E→retry | 12 model-domain combinations, closed APIs + open weights down from 70B-class | same error attributed to self | explicit-correction rate | Within-problem | the self-attributed arm is the control | Relabelling a self-generated error as coming from an external role raises explicit correction by **23–93 percentage points, significant in 10 of 12 settings** (`<memory>` framing best for math). **Independent 2026 replication of row 23's blind spot** |
| 48 | **When Small Models Are Right for Wrong Reasons** ([2601.00513](https://arxiv.org/abs/2601.00513)) | Audit of reasoning integrity + intervention comparison | whole trace | **7–9B** models, 10,734 traces | no-intervention | Reasoning Integrity Score; effect size d | Within-problem | — | **50–69% of correct answers contain fundamentally flawed reasoning.** **"Meta-cognitive interventions harmed performance (d = −0.14 to −0.33)"** while RAG helped (d = 0.23–0.93). A direct 2026 negative at our scale |


---

## Answers (≤150 words each)

### Q1. Any within-problem discrimination from a single rollout's internal signal at ~4B?

**No — and the strong claims are across-problem or on thinking models.** Row 22 (AUROC .80–.91) probes
R1-distill/QwQ models, pools ROC across problems, and its §4.4 says the probe *"performs much worse"* on a
non-reasoning 8B instruct model. Row 1's own two-12B control puts the **model-specific** part of P(IK) at
**~6 points** (.463 vs .409); its GSM8K P(True) AUROC tops out ≈.65–.70 **at 52B**, and zero-shot P(True)
*"lies close to 50%"*. Row 3: *"SEPs cannot match"* the sampling baselines. Row 26 is our scale —
Gemma 3 4B-it, pre-registered, **.554 → .509**, shuffled control .501. Row 27: on AIME, verbalized confidence
parses in **3/50** cases and P(True) ρ = **0.12**. Everything that works reads **across samples**
(rows 1-brainstorming, 2, 15, 36). Our .51–.55 single-rollout vs **.78** sibling-plurality is the published
pattern, not an anomaly.

### Q2. Has in-context self-correction ever rescued wrong answers on a non-thinking instruct model without an external signal or teacher?

**No clean case.** Row 4's only positive column is **oracle labels** (75.9→84.3); intrinsic is 75.9→**74.7**,
GPT-4 95.5→**89.0**, Llama-2 62.0→**36.5**. Row 8 (1.5–8B instruct) is **told it failed by an external
verifier**, and never evaluates MATH/AIME. Row 6 needs an **oracle error location** — and a
random location reproduces **40–70%** of the gain (+18.04 vs +10.59). Row 7 is the one self-generated success:
**multi-turn online RL** with a shaped i→c reward, on Gemini 1.5 Flash — Δ is only **+4.4**, with
**7.4 of its 11.8-pt lead coming from Acc@t1**. So the enablers, when it works, are: an external failure signal,
an oracle location, multi-turn RL with a correction-shaped reward, or **thinking-mode marker density**
(row 23: blind spot **64.5%** non-reasoning vs ≈0 reasoning; "Wait" removes **89.3%** — on *injected* errors).
Scale is not the enabler (rows 4, 24, 28).

### Q3. Whose gains are really difficulty-aware compute? (matched-compute numbers)

**Nearly all.** Debate **83.0@9** vs self-consistency **88.2@9** (row 5). SCoRe adds **+3.1** over matched
parallel sampling (**+10.5 vs +7.4** at a 32-sample budget), and **7.4 of its 11.8-pt lead is Acc@t1** (row 7).
s1 is the exception that proves it: extra tokens alone give **50.0** AIME24 (zero gain) and *hurt* MATH500/GPQA;
only the injected token gives **53.3** (row 31). SEVRA on a **frozen Qwen3-4B**: gate **76.3%** vs **76.0%** for
simply solving longer **with 28% fewer total tokens** (row 44). Probing buys **−24% tokens at identical 88.2%**
(row 22). Snell's genuine matched win is **pure difficulty-aware allocation** — *"4× less test-time compute"* —
with no in-context metacognition (row 38). Brown: coverage **95.3** vs selection **39.8** (row 39). Net: strip
RLVR gains, extra samples and allocation and **≈+2 to +3 points** remain.

### Q4. Which single condition most plausibly explains our failures, and the one experiment?

**Condition: our policy is the *non-thinking fused mode of an already-RLVR'd model* — a low-entropy,
correction-marker-poor reasoning distribution.** Not 4B scale (self-evaluation is near chance at 70B, in GPT-4
and in Sonnet 4.5 — rows 4, 24, 27, 28; row 48: metacognitive interventions *harm* 7–9B models, d = −0.14…−0.33).
Not K=8 (the same 8 samples yield **.78**). Within-problem framing is real but explains only two of five gates.
Rows **13** (*"little improvement for instruction-tuned models"*, mechanism = initial policy entropy), **23**
(SFT marker density 1–2 vs **30–170** for RL reasoning corpora), **22 §4.4**, **24** (*"CoT controllability
decreases with more RL training"*), **11** (86.67% entropy-position overlap) and **20/21** explain all five.
**Experiment:** Self-Correction Bench's attribution contrast (rows 23, 47) on our own 1,248 failures —
own-retry / user-framed / +"Wait" / donor / matched-compute re-solve.

---

## Appendix — supporting detail behind each answer

### A1 — supporting detail. Is there ANY paper showing within-problem discrimination from a single rollout's internal signal at ~4B scale?

**No. Not one.** Every paper in the table that reports a *strong* correctness signal reads it **across samples**,
and the one paper that reports strong single-rollout discrimination reads it **across problems** on **thinking** models.

- **Row 22** is the closest claim in the literature ("Reasoning Models Know When They're Right", AUROC .80–.91 on
  MATH/GSM8K). Three disqualifiers: every primary model is a **long-CoT/R1-distill or QwQ** model; the ROC is
  **pooled over problems** (the paper never reports a per-problem ROC); and §4.4 states outright that
  *"the probe trained on non-reasoning model representations performs much worse than its reasoning counterpart"*
  and that *"the encoded information on answer correctness is more prominent in reasoning models."*
  Its non-reasoning contrast model is Llama-3.1-**8B**-Instruct — twice our size, and it fails there too.
- **Row 1** is the origin of "LLMs know what they know", and it contains the decisive control: two 12B models with
  different pretraining, scored on the TriviaQA items **one gets right and the other gets wrong**. The
  model-specific component of P(IK) is **~6 points** (.463 vs .409 / .477 vs .408). The rest is item difficulty.
  That is the cleanest published statement that *"internal signals exist"* is overwhelmingly an
  **across-problem (difficulty) finding**. On math specifically, its P(True) AUROC on GSM8K tops out ≈.65–.70
  **at 52B**, and *"zero-shot, P(True) is poorly calibrated, and typically it lies close to 50%."*
- **Row 3** is the explicit single-rollout-vs-sampling comparison: *"SEPs cannot match the performance of these
  methods"* (the 10× costlier sampling baselines). QA only, no math.
- **Row 26** is our exact scale and class — **Gemma 3 4B-it**, pre-registered: single-rollout verbal confidence
  AUROC **.554 → .509**, shuffled-target control **.501**. Only after abandoning the pre-registered filter does it
  reach .774, and that is **across-problem** on TriviaQA/MMLU, not within-problem on math.
- **Row 27**, on competition math: verbalized confidence **parsed on 3/50 AIME cases**; P(True) ρ = **0.12**.

**What *does* work is across-samples, every time**: Kadavath's "brainstormed" 5-sample prompt (row 1), semantic
entropy's 10 generations (row 2), TTRL's majority vote (row 15), Intuitor's within-group normalisation (row 12).
Our own numbers reproduce this split exactly — single-rollout geometry .51–.55 and verbalized confidence .50,
**sibling plurality .78**. We are not an anomaly; we are the published pattern.

### A2 — supporting detail. Has in-context self-correction ever rescued wrong answers on a non-thinking instruct model without an external signal or teacher?

**No clean case exists.** Every positive result in the table buys its rescue with one of four things we do not have.

1. **An external correctness signal.** Row 4's only positive column is the *oracle-label* condition
   (GSM8K 75.9→84.3, 95.5→97.5); its intrinsic column is 75.9→**74.7** and 95.5→**89.0**, and Llama-2 collapses
   62.0→**36.5**. Row 8 (Reflect-Retry-Reward) looks like a counterexample — 1.5B–8B instruct models, large gains —
   but **an external binary verifier tells the model it failed before it reflects**, and the tasks are
   function-calling and Countdown equation writing; **MATH/AIME are never evaluated**, and there is **no
   generic-reflection control**.
2. **An oracle error location.** Row 6 separates the two abilities and finds localisation is the broken one
   (GPT-4 averages ≈40/53/43% across three prompting styles; PaLM-2 Unicorn ~22%). Given the location, correction
   works — but the **random-location control reproduces 40–70% of the gain** (multistep arithmetic +18.04 oracle
   vs **+10.59 random**), so even the oracle-located gain is only partly about knowing *where*.
3. **Multi-turn online RL with an explicitly shaped correction reward.** Row 7 (SCoRe) is the one method that
   makes Δ(t1,t2) positive from self-generated data alone — and it needed a two-stage RL recipe with a KL-anchored
   first attempt plus a progress-shaped reward, on **Gemini 1.5 Flash**. Even then Δ is **+4.4**, and
   **7.4 of the 11.8-point Acc@t2 gain over the best prior method comes from Acc@t1 rising 52.6→60.0** — ordinary
   single-turn RLVR, not self-correction. Note also that plain SFT variants (STaR, Pair-SFT) all land at Δ ≈ 0–1.8.
4. **Thinking mode, or an injected correction marker that substitutes for it.** Row 23 is the most informative
   result for us: the capability *exists* in non-thinking models but is **not activated** — blind-spot rate
   **64.5%** across 14 non-reasoning models, ≈0 for reasoning models. Appending **"Wait"** removes 89.3% of the
   blind spot (GSM8K-SC .183→.796). The stated cause is distributional: SFT corpora carry correction-marker
   density **1–2** at the 95th percentile, RL-generated reasoning corpora **30–170**. But the errors there are
   **injected and externally framed** — which is precisely the condition our 0/1,248 retry does not satisfy.

Scale alone is not the missing ingredient: GPT-4 degrades under intrinsic self-correction (row 4), Claude Sonnet 4.5
has **2.7%** CoT controllability (row 24), and row 28 finds SOTA models cannot localise a math error **even with the
reference solution in context**.

### A3 — supporting detail. Whose gains are actually *difficulty-aware compute*, not in-context metacognition? Numbers at matched compute.

Almost all of them, once you hold compute fixed. The matched-compute arms in this table are the most damaging
evidence against the in-context-metacognition story, and several of the papers run the control themselves.

| Claim | Matched-compute number | Verdict |
|---|---|---|
| **Multi-agent debate** (row 5) | Debate r1 (6 responses) **83.2** vs **SC@6 85.3**; debate r2 (9) **83.0** vs **SC@9 88.2** | The entire "critique each other" gain is *less* than spending the same calls on plain majority vote |
| **SCoRe self-correction** (row 7) | At a 32-sample budget: parallel sampling **+7.4%**, parallel + one self-correction round **+10.5%** | Self-correction adds **≈+3.1 pts over pure parallel sampling**. Also, **7.4 of its 11.8-pt Acc@t2 lead over Pair-SFT is Acc@t1 rising 52.6→60.0** — ordinary RLVR |
| **s1 budget forcing** (row 31) | *"2× without string"* = **50.0 / 90.2 / 55.1** vs no-extrapolation **50.0 / 93.0 / 57.6**; *"2× Wait"* = **53.3 / 93.0 / 59.6** | **More tokens alone buy nothing (and hurt).** The +3.3 AIME24 is attributable to the injected token, not the compute — the one clean win for a metacognitive *marker* over matched compute |
| **SEVRA verification gate on a frozen Qwen3-4B** (row 44) | Gate **76.3%**; always-verify 75.5%; **plain 8,192-token initial solve 76.0% with 28% fewer total tokens** | **The gate loses to "just think longer" at matched budget — on our exact policy class.** This is the single most on-point matched-compute null in the literature |
| **Probing for early exit** (row 22) | −24% tokens at **88.2%**, i.e. *identical* accuracy | The payoff of a good internal signal is **token saving, not accuracy** |
| **Snell et al. compute-optimal allocation** (row 38) | *"outperform best-of-N using up to 4× less test-time compute (64 samples verses 256)"*; search *"16 verses 64"* | This is a **real** matched-compute win — and it is purely **difficulty-aware allocation over a verifier**, with **no in-context metacognition at all** |
| **Brown et al.** (row 39) | coverage 79.8→**95.3** vs selection 38.7→**39.8** | The binding constraint is **selection**, not generation or reflection. ~55 points sit on the table |
| **AdaptThink** (row 34) | +2.4% / −53% length (1.5B), +2.3% / −40% (7B) | A genuine simultaneous win — but the decision is **before solving**, is **problem-level difficulty**, and NoThinking alone is far worse (AIME24 29.4→14.0) |
| **Self-consistency** (row 36) | +17.9 GSM8K **at k=40 vs k=1** | Not matched compute at all; the canonical "gain" is 40× the budget |

Two further direction-of-effect findings that cut against a "re-check the hard ones" policy:
**Snell** finds *"easy questions benefit more from sequential revisions, whereas on difficult questions it is optimal
to strike a balance between sequential and parallel computation"*, and beam search **over-optimizes on Levels 1–2**.
**TTRL** (row 15) finds its gains shrink monotonically with difficulty (MATH-500 L1 +175% vs **L5 +75%**).
Competition math — where our gates were run — is precisely the regime where the literature says sequential
self-revision is *least* effective and parallel breadth is most effective.

**Net reading of Q3.** Strip out (i) the Acc@t1 improvement that is ordinary RLVR, (ii) the extra samples, and
(iii) the difficulty-aware allocation, and what remains attributable to *in-context metacognition* across this
whole literature is: SCoRe's **+3.1 pts over matched parallel sampling**, s1's **+3.3 AIME24 from an injected
token**, and AdaptThink's **+2.3–2.4%** from a *pre-solve mode choice*. That is the honest size of the prize.

### A4 — supporting detail. Which single condition most plausibly explains our failures — and the one experiment that tests it

**Ruling out the candidates first.**

- ***K=8 noise*** — ruled out by our own data. The same 8 samples yield a **.78** sibling-plurality AUC. The
  samples carry the signal; only the single-rollout readout is dead. Row 26 further shows a **4B** model reaching
  AUROC .774 when the signal is really there, against a shuffled control of .501.
- ***4B scale*** — ruled out, or at least not sufficient. Self-evaluation is near chance at far larger scale
  (rows 4, 24, 27, 28), and row 22's non-reasoning failure case is an **8B** model. Row 11's small-model null
  (8B ≈ +0.5 vs 32B +7.71) says scale governs *how much RL on a mid-solution signal buys*, not whether the signal
  exists.
- ***Within-problem framing*** — real, and a genuine part of the story (row 1's ~6-point model-specific component;
  row 22's pooled ROC). But it explains only the two discrimination gates, not the retry, redirect, critique or
  GRPO gates.

**The condition that explains all five gates at once: our policy is the *non-thinking fused mode of an
already-RLVR'd model* — a low-entropy, correction-marker-poor reasoning distribution.**
Qwen3's post-training is long-CoT cold start → **reasoning RL (GRPO with verifiers)** → **thinking-mode fusion** →
general RL ([2505.09388](https://arxiv.org/abs/2505.09388)). The non-thinking mode is manufactured by *removing*
the long-CoT behaviour after the entropy has already been spent on RL. Every symptom we measured is a documented
consequence of exactly that state:

- **Donor ≈ own inserted metacognition (+.025 vs +.024)** — row 23: correction markers in the SFT/fused
  distribution have density 1–2, so an inserted meta block is read as generic text, not as a control signal; and
  row 25: RLLMs are "generally robust, reliably recovering from diverse perturbations", i.e. the trajectory
  reasserts itself regardless of what you insert.
- **Redirect overridden 95.6%, 86% verbatim restatement** — row 24 measures this directly: 2.7% CoT
  controllability vs 61.9% output controllability, and *"CoT controllability decreases with more RL training."*
- **Retry rescues 0/1,248** — row 23's 64.5% blind spot on non-reasoning models, with reasoning models at ≈0.
- **Within-problem geometry .51–.55** — row 22 §4.4: correctness encoding is "more prominent in reasoning models"
  and the probe "performs much worse" on a non-reasoning instruct model.
- **30 steps of correctness-only GRPO changes nothing** — row 13: RLIF-style gains occur on **base** checkpoints
  and *"yield little improvement for instruction-tuned models"*, with the mechanism attributed to **initial policy
  entropy**; row 11: the top-20% entropy token positions overlap **86.67%** between base and converged-RL models;
  rows 20/21: RLVR sharpens inside the base support and a winner-take-all effect suppresses low-likelihood
  solutions. Our pass@8 ≫ pass@1 (+23–27) is the signature of a policy whose *coverage* is intact and whose
  *selection* is the binding constraint — exactly Yue et al.'s and Brown et al.'s diagnosis.

**The one apparent counter-example, and why it does not rescue the mid-solution framing.**
**MAPR** (row 42) reports **AIME25 11.77 → 21.56 Pass@1 on Qwen3-4B over a GRPO baseline** — same family, same
size, same benchmark class as us. But its signal is **before solving and across problems**: the model predicts
*its own rollout statistics* (expected length, expected pass rate, concepts) for the problem, and prediction
accuracy is the reward. That is difficulty self-estimation — the thing Kadavath's P(IK) actually measures well
(row 1) and the thing our across-problem numbers would presumably also support. It is **not** the mid-solution,
within-problem, "is this particular line of work going wrong" signal that all five of our gates tested.
MAPR is therefore evidence *for* pivoting, not evidence that our gates were mis-run. Note also that **none** of
the 2026 metacognition-RL papers (rows 42–46) runs a content-free / placebo-meta control of the kind s1's
"2× without string" (row 31) or RRPO (row 41) provides — our G2 donor gate is stricter than the entire
2026 literature, which is why we are seeing nulls they would not have detected.

Finally, the most direct scale-matched negative: **row 48**, 7–9B models, 10,734 traces —
*"meta-cognitive interventions harmed performance (d = −0.14 to −0.33)"* while retrieval helped
(d = 0.23–0.93) — together with **row 30**'s *"small models cannot self-improve; the generation–verification
gap is non-positive"*. Our .332-vs-.361 critique result is that finding, reproduced.


**The one experiment to run: Self-Correction Bench's attribution contrast (row 23,
[2507.02778](https://arxiv.org/abs/2507.02778)), on our own 1,248 natural failures.**
Take each wrong solution our policy produced and re-present it in four conditions, holding the content fixed:
(a) **own completed answer, plain retry** — our current 0/1,248 baseline;
(b) **same text framed as a *user's* answer to critique** — the blind-spot control;
(c) **own answer with `Wait` appended** — the marker intervention;
(d) **donor**: another problem's wrong answer framed as the user's, to keep the content-free control in place;
(e) **matched compute**: the same total token budget spent on a fresh independent solve instead of any re-check —
the arm SEVRA (row 44) ran on a frozen Qwen3-4B and lost, and the arm SCoRe (row 7) and Huang (row 5) ran and
mostly lost. **No re-check arm should be reported without (e).**
Score rescue rate in all five. Cost: five decode passes over 1,248 items, no training.

- If **(b) and (c) rescue substantially while (a) stays at ~0**, the capability is present but *not activated*:
  the binding condition is the non-thinking fused distribution, and the fix is to **install the marker
  distribution** (SFT on correction markers / run the thinking mode / prime behaviours as in row 10) *before*
  any metacognition RL. M_DIST and the self-citation counterfactual would then be premature, not wrong.
- If **(b) and (c) also rescue ≈0**, the capability genuinely is not there for this policy on competition math,
  and the only family in this entire table that works at our scale and class is the **across-sample** one
  (rows 2, 15, 27 and our own .78 sibling plurality). The programme should then be redirected from
  "reward good in-context metacognition" to "learn *when* to spend samples and how to select among them" —
  which is also where the matched-compute evidence in Q3 points.


---

## Provenance and caveats

Verbatim-from-paper (safe to cite): Huang Tables 2/3/4/7; Kadavath Table 2 and the §4.1–4.2 quotes; Tyen Tables 4/6;
SCoRe Tables 1/2 and §6.2; s1 Table 4; Snell's two "4×" sentences; Brown's MATH coverage sentence; the abstract
claims of rows 8, 13, 14, 20, 21, 23, 24, 26, 27, 28, 30, 47, 48.

**Single-pass table/figure reads — verify before quoting in a paper:** ThinkPrune Table 1; L1 per-benchmark cells
(two fetch passes disagreed; the "+2 pts at ~860 tokens" average is consistent); Arora & Zanette appendix tables
(they contradict the paper's own prose — cite the prose); USC per-benchmark cells; MAPR's 11.77→21.56 and model
list. **s1's majority-voting AIME24 accuracy at 64 samples is never stated numerically in the text — do not cite a
number for it.** Row 41 (RRPO control) and row 43 report no model sizes in their abstracts and need a full read.

**Structural gap worth claiming.** Of 48 rows, only seven run a content-free control on their own signal:
Kadavath's donor hints and two-model cross-experiment (row 1), Tyen's random location (row 6), Gandhi's
placeholder-token prime (row 10), Intuitor's coin-flip reward (row 12), Spurious Rewards in its entirety (row 14),
Setlur's analytic prover=policy null (row 16), s1's "2× without string" (row 31) and RRPO (row 41).
**None of the 2026 metacognition-RL papers (rows 42–46) runs one.** Our G2 donor gate is stricter than the
standard of the field it is testing — which is the most likely reason we are seeing nulls that they did not.
