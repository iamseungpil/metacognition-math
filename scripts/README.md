# Meta-CoT analysis scripts

## analyze_ev_signature_meta.py

Computes the four Epistemic Verbalization (EV) signature metrics defined in
`plan_EAD_unified` Section 6 on Meta-CoT eval traces. The script consumes the
JSON bundle produced by `src.eval.eval_hf` (keys: `results[].completion`,
`results[].is_correct`, `results[].num_meta_blocks`, `results[].benchmark`),
loads the matching HuggingFace checkpoint with `AutoModelForCausalLM` in
bf16/device_map="auto", performs a single forward pass per completion to
recover per-token full-vocabulary Shannon entropy and top-1/top-2
probabilities, and then reports:

1. `delta_H +/- 5` mean Shannon-entropy difference over the 5-token windows
   before vs. after each EV marker, split by correctness.
2. `d_M` Mahalanobis distance between EV pairs `(marker-token, next-token)`
   and neutral pairs drawn from random positions, in
   `(H_t, top1_prob, top1-top2 margin)` space, with a bootstrap 95% CI.
3. `I(M_c ; Y | D)` mutual information between the meta-content indicator
   `M_c` and correctness `Y`, conditioned on a difficulty tercile `D`
   computed from per-benchmark accuracy (plug-in histogram estimator).
4. `C_t` cumulative confidence gain `sum (1 - H_s / log2 V)` over 5
   post-marker tokens, with Cohen's d between correct and incorrect subsets
   (easily extended to SFT vs. RL splits at the caller level).

Two marker modes are supported: `--marker_mode meta` locates
`<|meta|>...<|/meta|>` spans (SFT policies) and `--marker_mode confidence`
regex-matches free-text `confidence: 0.XX` (RL policies that dropped the
`<|meta|>` wrap).

Example:

```bash
python scripts/analyze_ev_signature_meta.py \
    --model_path checkpoints/v8_meta_inside_strict_sft \
    --eval_json   results/eval_v8_meta_inside_strict_sft/eval_1030_v8_meta_inside_strict_sft.json \
    --output_dir  results/ev_signature/ \
    --max_samples 200 \
    --marker_mode meta
```

Outputs `ev_signature_stats.json` (all four metrics with per-tercile breakdown
and bootstrap CI) and `ev_signature_per_sample.csv` (per-sample raw values for
downstream analyses).

Smoke test: `pytest tests/test_analyze_ev_signature.py -q` exercises all four
metrics on synthetic traces and does not require a GPU.
