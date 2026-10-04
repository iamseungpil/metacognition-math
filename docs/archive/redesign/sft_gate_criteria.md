# SFT-Quality GATE — PASS criteria before an SFT checkpoint earns RL GPU

**Purpose.** RL (GRPO / PMI-shift) is expensive. A metacognition SFT checkpoint
must first prove it is *RL-ready*: it emits well-formed metas reliably, its meta
actually shifts the gold-vs-decoy belief in a way that discriminates correct from
wrong (the substrate the PMI-shift reward amplifies), and it has not paid for that
behavior with an accuracy collapse. This gate is the go/no-go before spending GPU
on RL.

**How to measure.** Run `scripts/measure_sft_gate.py` on the merged SFT checkpoint
(see command at the bottom). It orchestrates `scripts/eval_vllm_1030.py` (emission
+ accuracy) and `src/eval/pmi_shift_signal.py` (PMI-shift signal + confound check)
and writes `sft_gate.json` with a `gate` block. Thresholds below are mirrored as
constants at the top of that script — keep the two in sync.

The slice is a **held-out math slice** (default `math500`, `--max_problems 300`,
`max_tokens 4096`). The greedy accuracy check must use the **same slice** for both
the candidate and its matched Base SFT.

---

## The four quantities and their thresholds

### (a) `emission_at_temp1` >= 0.90  — the meta habit is real, not a temp-0 artifact
Meta emission rate (fraction of completions with a closed `<|meta|>...<|/meta|>`
block) measured at **temperature 1.0**. Measuring under sampling, not greedy,
guards against a checkpoint that only emits metas on the single greedy path.
- **FAIL implies:** the SFT did not internalize *when* to emit. Revisit the SFT
  data mix / meta-density and the cold-start format (`data/*_meta_*_sft.parquet`,
  `src/metacot/prompt*.py`). Do NOT try to fix this with RL — RL amplifies an
  existing habit, it does not install one.

### (b) `wellformed_rate` >= 0.95  — emitted metas are properly closed
Of the completions that emit an opening `<|meta|>` tag, the fraction that also
produce a matching `<|/meta|>` close (`num_meta_blocks > 0`). Unclosed metas run
into the answer and break both parsing and the PMI-shift OPEN/CLOSE split.
- **FAIL implies:** a tokenizer/terminator or truncation problem. Check that the
  meta tokens are non-special and the EOS/PAD invariant holds (eval enforces
  `eos=<|im_end|>`, `pad=<|endoftext|>`); check `max_tokens` truncation
  (`finish_reason=="length"`); check the SFT target actually closed every meta.

### (c) `accuracy_greedy` not collapsed vs base  — the meta did not cost the answer
Accuracy at **temperature 0** on the same slice. Gate: `accuracy_greedy >=
base_accuracy_greedy - 0.05` (matched Base SFT on the identical slice, passed via
`--base_accuracy_greedy`). This number is **required**: if `--base_accuracy_greedy`
is omitted the accuracy leg **FAILS** the gate (fail-closed — without the matched
base we cannot detect a collapse, so we do not pass). A weak `accuracy > 0` guard is
no longer accepted.
- **FAIL implies:** the meta overhead is displacing solution reasoning (the known
  failure mode: meta = 56% of tokens, 31% truncation → MATH 56.7% vs base 76.7%).
  Revisit meta length/budget in SFT and `max_tokens`, not the RL reward.

### (d) `pmi_signal` — the meta produces a real SAVE signal (the RL substrate)
From `src/eval/pmi_shift_signal.py`, run on temp=1.0 rollouts that carry a closed
meta block. Three sub-conditions, **all** required:

1. **`auc_shift` clearly > 0.5** (gate: `> 0.55`). The per-rollout PMI *shift*
   (`PMI_close − PMI_open` on the divergent gold-vs-decoy tokens) separates correct
   from wrong rollouts. AUC ~0.5 = the meta carries no discriminative belief update.
2. **`n_save_reversal` > 0.** At least some rollouts flip decoy→gold across the
   meta (`PMI_open < 0 < PMI_close`) — the meta genuinely *rescues* a wrong-leaning
   belief. Zero SAVE reversals = nothing for the PMI-shift reward to credit.
3. **own!=gold confound not the sole explanation.** On the subset where the model's
   own final answer != gold, the shift must still point toward gold and discriminate
   correct-from-wrong. This leg is **FAIL-CLOSED** — it PASSES only when **all** of:
   - **n-floor:** `n_own_ne_gold >= 30`. Below the floor the discrimination is
     statistically inconclusive → the leg **FAILS** (an inconclusive is a FAIL, not
     a pass). Constant `CONFOUND_N_FLOOR` in the script.
   - **direction:** `mean_pmi_close_own_ne_gold > 0` (belief stays toward gold, not
     the own/decoy answer).
   - **AUC:** `auc_shift_own_ne_gold` is **computable** and **> 0.5**. An uncomputable
     AUC (single correctness class in the subset) is treated as **inconclusive =
     FAIL** — it does **not** pass leniently on a mean-only check.
   - **placebo gap:** `confound.placebo.placebo_gap_own_ne_gold` is **present** and
     **> 0** (real-meta shift clearly exceeds the content-destroyed / shuffled-meta
     placebo shift). This rules out a meta that moves belief purely by its *presence*
     (presence-as-confidence), not its *content*. If the field is **absent** from the
     `pmi_shift_signal` output the leg **FAILS loudly** (it is not silently skipped).

   If any sub-check fails, `shift` may be the model favoring its OWN answer (A.6
   answer-identity) or mere meta-presence, not a real gold-belief update. The script
   reports the verdict as `confound.verdict_genuine_not_sole_own_identity` and lists
   the failing sub-checks in `confound.verdict_reasons`.

- **FAIL (1 low AUC):** the meta is decorative — it does not move the answer belief.
  The PMI-shift reward has no gradient to stand on. Revisit the SFT meta *content*
  (does the meta actually reason toward/against the answer?), not RL hyperparams.
- **FAIL (2 no SAVE):** metas never reverse a wrong lean. Either emission is on
  already-correct paths only, or metas never engage with the answer. Revisit the
  SFT trajectory selection (include error→fix / redirect exemplars).
- **FAIL (3 confounded):** the "signal" is own-answer identity, not belief update.
  This is the A.6 confound the reward was designed to avoid — a checkpoint that
  passes 1+2 but fails 3 would train the reward to reward tautology. Revisit the
  decoy construction and, upstream, whether the meta ever disagrees with the final
  answer in the SFT data (a meta that only ever agrees cannot show a genuine
  gold-shift on own!=gold rows). Also check `pmi_signal.dropped` — high
  `n_len_mismatch` / `n_zero_divergent` or a small `n_own_ne_gold` can starve this
  check; a tiny own!=gold n is an *inconclusive*, treat as FAIL until n is adequate.

---

## Overall gate

**PASS** iff **all** hold:
```
emission_at_temp1        >= 0.90
wellformed_rate          >= 0.95
accuracy_greedy          >= base_accuracy_greedy - 0.05   (base REQUIRED; omitted => FAIL)
pmi_signal.auc_shift     >  0.55
pmi_signal.n_save_reversal > 0
pmi_signal.confound.verdict_genuine_not_sole_own_identity == true
    where genuine requires ALL of:
      n_own_ne_gold                       >= 30            (else inconclusive => FAIL)
      mean_pmi_close_own_ne_gold          >  0
      auc_shift_own_ne_gold  computable AND >  0.5          (None => FAIL)
      placebo.placebo_gap_own_ne_gold present AND > 0       (absent => FAIL)
```
Any single FAIL blocks RL; fix the upstream knob named for that check and
re-measure. The script's `gate.PASS` field encodes exactly this conjunction.

---

## Run it (after SFT finishes; GPU required)

```bash
set -a; source .env; set +a
python scripts/measure_sft_gate.py \
  --model_path checkpoints/<my_meta_sft_merged> \
  --output_dir results/sft_gate_<name>/ \
  --benchmarks math500 --max_problems 300 --tp_size 4 \
  --base_accuracy_greedy <matched_base_greedy_acc_on_same_slice>
# add --reuse to skip stages whose artifacts already exist.
```
Output: `results/sft_gate_<name>/sft_gate.json` (+ per-stage eval parquets and
`pmi_shift/pmi_signal.json`). Read `gate.PASS`.
