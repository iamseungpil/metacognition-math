## Self-Distill Design Review

Date: 2026-04-14

### Intent

Add two explicit self-distill lanes after RQ2:

1. `D1 naive self-distill`
2. `D2 epistemic-preserving self-distill`

with a later optional `D3` dense token distill path.

### Verified Findings

1. The current offline SFT path is reusable as long as the output artifact keeps the standard `messages` column expected by `src/training/sft.py`.
2. A silent mismatch risk exists if self-distill builders encode controller state as extra prompt turns or system text, because `src/eval/eval_hf.py` evaluates with a single user prompt only.
3. The current repo already has the right analysis vocabulary for collapse checks:
   - `meta_emission_rate`
   - `wrong_high_confidence`
   - confidence bins / ECE
   - hard-vs-GSM slices
4. The current repo already has the right controller vocabulary for D2:
   - `diagnosis_text`
   - `study_need`
   - `confidence_gain`
   - `trigger_cleared`
5. The most important contract decision is to keep a stable teacher-trace IR and project it into:
   - current `messages` parquet for offline SFT now
   - teacher-conditioned distillation inputs later

### Converged Design

1. Implement offline D1/D2 first.
2. Use a common normalization layer instead of writing separate one-off dataset builders.
3. Keep D2 close to the current control-v5 natural-language meta format; do not introduce rigid `trigger:` or schema-heavy control markup.
4. Only add D3 after D1/D2 collapse results are available.

### Open Risks

1. If D2 examples are synthesized in a format too different from current control-v5 traces, trigger behavior may regress even if the dataset looks richer on paper.
2. If D1 removes too much structure, the baseline becomes unrealistic; it should mimic the “successful but epistemically thinner” teacher regime, not random stripping.
3. If evaluation only checks accuracy, the claimed contribution is underidentified. Collapse metrics must be saved with every readout.
