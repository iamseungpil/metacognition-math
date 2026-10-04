# Redirect-Priming v2 (Counterfactual) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The NEXT phase after each task is karpathy-guidelines + a smoke critic, so every task ends with a runnable smoke check.

**Goal:** Prove that *functional redirect* metacognition causally raises math accuracy, by priming a discrete `<|switch|>` redirect behavior (counterfactually-harvested), rewarding it at RL only when it causally helps, and measuring the effect with the confound-controlled estimand **R − B′**.

**Architecture:** PRE-GATES first (cheap STOP-gates: no GPU training until each passes) → Stage A counterfactual harvest → Stage B priming SFT (`<|switch|>` token) → Stage C DCPO-v4 RL with a `-inf`-masked, behavior-graded counterfactual reward → R−B′ eval. Builds on `verl_sdc`/`dcpo_region`/`cf_prefix_agent`/`sft.py`.

**Tech Stack:** Python 3.10 (metaprobe env), verl (FSDP+vLLM), HF transformers, math_verify, pytest. No scipy/statsmodels (absent in-env → pure-python stats).

**Spec:** `docs/superpowers/specs/2026-06-18-redirect-priming-counterfactual-design.md` (REV-6).

---

## File Structure (created / modified)

- **NEW** `src/training/switch_ban_processor.py` — hard `-inf` `<|switch|>` LogitsProcessor (Task 1).
- **NEW** `src/eval/redirect_behavior_detector.py` — LLM-judge + regex redirect-behavior detector with measured recall (Task 2).
- **NEW** `src/eval/cf_stats.py` — pure-python McNemar exact, parse-gate, degeneracy health-gate (Task 3).
- **NEW** `scripts/pg0_yield_pilot.py`, `scripts/pg1_separability_smoke.py` — STOP-gate smokes (Tasks 4-5).
- **NEW** `scripts/harvest_redirect_cf.py` — Stage A counterfactual harvest (Task 6).
- **MODIFY** tokenizer/data + `scripts/build_meta_template_sft.py` — `<|switch|>` chain + mask (Tasks 7-8).
- **MODIFY** `src/training/cf_prefix_agent.py`, `src/training/dcpo_region.py`, `src/training/verl_sdc.py` — `-inf` CF + behavior grading + k=4-8 continuous + negative term + tripwires (Task 9).
- **NEW** `configs/triobj_dcpo_v4_redirect_cf_h100_4x4k.yaml` (Task 10).
- **MODIFY** `src/eval/eval_counterfactual_difficulty.py` + `_summarize.py` — R−B′, placebo, health/parse gate, McNemar (Task 11).
- **Tests** under `tests/` alongside existing `test_dcpo_region.py` etc.

---

## Task 1: SwitchBanLogitsProcessor (true −inf hard ban)

Resolves spec §7 / round-5 I-2,I-6: live code uses soft `-100.0`; vLLM `logit_bias` is additive/clamped, not −inf. Need a real processor (pattern from `meta_close_processor.py`).

**Files:** Create `src/training/switch_ban_processor.py`; Test `tests/test_switch_ban_processor.py`.

- [ ] **Step 1: Write failing test**
```python
# tests/test_switch_ban_processor.py
import torch
from src.training.switch_ban_processor import SwitchBanLogitsProcessor
def test_switch_logit_forced_to_neg_inf():
    p = SwitchBanLogitsProcessor(ban_ids=[151670])
    logits = torch.zeros(151700); logits[151670] = 50.0  # switch token leads by a lot
    out = p([1,2,3], logits)
    assert out[151670] == float("-inf")
    assert out[5] == 0.0  # other tokens untouched
```
- [ ] **Step 2: Run → FAIL** `pytest tests/test_switch_ban_processor.py -v` (module not found).
- [ ] **Step 3: Implement** (stateless, picklable for Ray, mirrors `meta_close_processor.py`):
```python
# src/training/switch_ban_processor.py
import torch
class SwitchBanLogitsProcessor:
    """Sets the logit of each banned id to -inf at every position. True hard ban
    (vLLM logit_bias is additive and a primed model can override -100)."""
    def __init__(self, ban_ids):
        self.ban_ids = [int(i) for i in ban_ids]
    def __call__(self, token_ids, logits):
        for i in self.ban_ids:
            logits[i] = float("-inf")
        return logits
```
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Smoke (vLLM honors it)** — `scripts/` one-off: load a tiny model, generate with the processor on a token id, assert that id never appears in 50 sampled outputs. Record result; if vLLM ignores the processor, fall back to `-1e9` and re-assert. **This is part of PG3.**
- [ ] **Step 6: Commit** `git commit -m "feat: SwitchBanLogitsProcessor true -inf hard ban (PG3 primitive)"`

## Task 2: redirect-behavior detector (measured recall)

Resolves §5.3 / round-5 M-3: token ban ≠ behavior ban; need to detect plain-prose redirect with a measured recall.

**Files:** Create `src/eval/redirect_behavior_detector.py`; Test `tests/test_redirect_behavior_detector.py`.

- [ ] **Step 1: Failing test** — given 6 hand-labeled strings (3 redirect: "wait, that's wrong, let me try a different approach"; 3 non: straight continuation), `detect_redirect(text)` returns correct booleans; `measure_recall(labeled)` returns a float.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — regex pre-filter `(?i)(instead|different (approach|method)|let me reconsider|that('?s| is) wrong|start over|backtrack|scrap that)` OR induced `<|switch|>` surface form; `detect_redirect` returns `regex_hit OR llm_judge(text)` (llm_judge via the project's judge util; injectable for test with a stub). `measure_recall(labeled_redirects)` = fraction detected.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Smoke** — run `measure_recall` on ~30 hand-labeled redirect traces (reuse harvested examples); print recall. **Gate (PG1): recall ≥ 0.8 else STOP.**
- [ ] **Step 6: Commit.**

## Task 3: cf_stats (pure-python McNemar + parse + health gates)

Resolves C-3 / round-5: no scipy/statsmodels; summarizer has raw saved/broke only.

**Files:** Create `src/eval/cf_stats.py`; Test `tests/test_cf_stats.py`.

- [ ] **Step 1: Failing tests** (3 cases the spec pre-registers):
```python
from src.eval.cf_stats import mcnemar_exact_p, is_parsed, degeneracy_flags
def test_mcnemar_saved_gt_broke_not_significant_rejects():
    # b=7 saved, c=4 broke -> not significant at p<0.05
    assert mcnemar_exact_p(b=7, c=4) > 0.05
def test_mcnemar_clear_effect_significant():
    assert mcnemar_exact_p(b=20, c=3) < 0.05
def test_parse_fail_dropped():
    assert is_parsed("") is False and is_parsed("the answer is 18") is True
def test_degeneracy_flags_repetition():
    f = degeneracy_flags("the the the the the the", min_len=3)
    assert f["repetition"] is True
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** — `mcnemar_exact_p(b,c)`: two-sided exact binomial `min(b,c) ~ Binom(b+c, 0.5)` via `math.comb` summation (no scipy); `is_parsed(text)` wraps the project's `_extract_answer_fallback(text) != ''`; `degeneracy_flags(text, min_len)`: repetition (max n-gram repeat ratio), length floor, `%`-no-final-answer.
- [ ] **Step 4: Run → PASS** (all 3 pre-registered cases: saved>broke-not-sig REJECTS, parse-fail DROPS, repetition flagged).
- [ ] **Step 5: Commit** `"feat: pure-python cf_stats McNemar+parse+degeneracy (PG3)"`

## Task 4: PG0 — yield pilot (cheapest STOP-gate, run FIRST)

Resolves round-5 I-7. Spec §0 PG0.

**Files:** Create `scripts/pg0_yield_pilot.py`.

- [ ] **Step 1:** Implement a tiny pilot: sample N=200 train problems, rollout the SFT init (k=8) via `run_online_sdpo_regen.py` infra, compute `emission_rate × in_band_frac([0.125,0.5]) × accept_prob`, project accepted-redirect count for the full pool.
- [ ] **Step 2: Run** `PY scripts/pg0_yield_pilot.py --pool_size <N> --target 1500`. Expected: prints projected accepted count + GO/STOP vs pre-registered target (1500 for a 15–30% mix). **STOP the whole project here if projected < target** (costs minutes, not GPU-hours).
- [ ] **Step 3: Commit + record the numbers in the spec §0.**

## Task 5: PG1 — two-sided separability smoke (STOP-gate)

Resolves C-1 / round-4 I-4 / round-5. Uses Task 1 + Task 2 + Task 3.

**Files:** Create `scripts/pg1_separability_smoke.py`.

- [ ] **Step 1:** On ≥20 labeled redirect traces, for each: arm-R (normal) vs arm-B (SwitchBan processor, Task 1). Compute (a) redirect-behavior rate both arms (Task 2 detector), (b) verify/conf meta emission rate, (c) degeneracy flags both arms (Task 3).
- [ ] **Step 2: Run.** Expected printout + asserts: (a) arm-B behavior-rate ≪ arm-R; (b) verify/conf emission ≈ arm-R; (c) arm-B degeneracy ≈ arm-R; detector recall ≥ 0.8. **Any fail → STOP (experiment invalid before GPU).**
- [ ] **Step 3: Commit + record PG1 numbers (the pre-registered false-negative bound) in spec §0.**

## Task 6: Stage A — counterfactual harvest

Spec §4.A. **Files:** Create `scripts/harvest_redirect_cf.py` (gen base = `run_online_sdpo_regen.py`).

- [ ] **Step 1: Unit test** the accept rule on synthetic arm grades: `accept(R=[1,1,0,1], Nprime=[0,0,0,0], Nc=[0,0,0,0], margin=0.5)` True; equal grades False; use `cf_stats` lower-CI-bound.
- [ ] **Step 2: Run → FAIL; Step 3: Implement** gates 1–7 (frozen pass-rate band, failed pool, splice 30–70%, well-formed `<|switch|>`, **prefix-forced 3-4 arm R/B′/N′/Nc with SwitchBan + behavior detector on banned arms (round-5 I-1)**, lower-CI accept + fresh holdout, ≤2/problem, record source ids).
- [ ] **Step 4: Smoke** on 50 problems: prints accepted count, false-accept estimate, behavior-rate of banned arms. **Step 5: Commit.**

## Task 7: Stage B — `<|switch|>` token chain

Spec §4.B / round-3 M-5. **Files:** tokenizer config + data regen + `build_meta_template_sft.py`.

- [ ] **Step 1:** Add `<|switch|>` to tokenizer, assign id; `resize_token_embeddings` + verify `lm_head` tie; init embedding from mean of `decision`/`switch` think-token embeddings.
- [ ] **Step 2: Smoke test** — encode/decode round-trips the token to one id; embedding row is non-zero; lm_head shares it. **Step 3:** Regenerate primed SFT parquet with `<|switch|>` at the redirect decision point (from Task 6 harvest, mixed 15–30% with normal v8 traces). **Step 4:** assert RL config + frozen-ref + PMI base reference the resized vocab id. **Step 5: Commit.**

## Task 8: Stage B — `tokenize_row` segment loss-mask

Spec §4.B / round-2 I-7, round-3 M-5. **Files:** Modify `src/training/sft.py:83-106`; Test `tests/test_sft_segment_mask.py`.

- [ ] **Step 1: Failing test** — a row with explicit `loss_mask` over `[prompt+wrong_prefix]` produces labels = −100 on those token-id indices and real labels on `[meta…<|switch|>…]+[continuation]`.
- [ ] **Step 2: Run → FAIL; Step 3: Implement** — extend `tokenize_row` to accept token-id-index segment boundaries (chat-template-robust, not char offsets); define precedence with the `teacher_kl` control-span path (state: teacher_kl OFF for redirect-priming SFT). **Step 4: Run → PASS. Step 5: Commit.**

## Task 9: Stage C — counterfactual reward (−inf, behavior-graded, k=4-8, negative term, tripwires)

Spec §3 / rounds 1-5 (C-4,I-6,C-2,I-1,I-2,I-3,round-5 C-1). **Files:** Modify `cf_prefix_agent.py` (use SwitchBan processor instead of `logit_bias=-100`), `dcpo_region.py` (R_meta continuous + negative term + behavior+degeneracy grading of `c_without`), `verl_sdc.py` (k=4-8 reshape not k=1, tripwires); Tests `tests/test_dcpo_region_cf.py`.

- [ ] **Step 1: Failing tests**: (a) `c_without` graded as wrong when degeneracy-flagged (not "meta saved it"); (b) R_meta continuous in [−1,1] from k=4 mean; (c) negative term fires when `emit switch AND c_without correct`; (d) continuous rmeta_pos/neg thresholds (pre-registered, e.g. ±0.25).
- [ ] **Step 2: Run → FAIL; Step 3: Implement**: cf_prefix_agent uses `SwitchBanLogitsProcessor` (Task 1) for the suppressed arm; k=4-8 draws → continuous `c_without = mean(correct & not-degenerate & behavior-absent)`; `R_meta = c_with − c_without − λ·1[emit_switch AND c_without_correct]` (λ small); positive only on frozen hard band; **runtime tripwires**: halt if `c_without`-arm redirect-behavior > PG1 bound, if band-emission < ⅓ post-prime rate (10-step slope), or if `acc_without` falls while rmeta_pos rises (step<30). **Step 4: Run → PASS.** **Step 5:** V4+cf integration test (current tests assert cf=False). **Step 6: Commit.**

## Task 10: Stage C — config

**Files:** Create `configs/triobj_dcpo_v4_redirect_cf_h100_4x4k.yaml`.

- [ ] **Step 1:** Derive from `triobj_dcpo_v4_stage3b_h100_4x4k.yaml`; set `sdc_counterfactual=true`, k=4-8, `len_cost` warmup RAMP (not step-0), hard-band gating, subset `f` s.t. `f·emit·B≥10`, PMI weight 0. **Step 2: Smoke** — `yaml.safe_load` + assert keys. **Step 3: Commit.**

## Task 11: Eval rewrite — R−B′, placebo, gates, McNemar

Spec §5. **Files:** Modify `src/eval/eval_counterfactual_difficulty.py` + `_summarize.py`; Test `tests/test_eval_cf_rml.py`.

- [ ] **Step 1: Failing tests**: arm-B uses SwitchBan (token id 0 survivors), B′ allows prose restart with only switch banned, placebo P bans a matched non-redirect meta token; rows dropped on parse-fail (Task 3) and arm-B-still-redirects (Task 2); effect = R−B′ over bias sweep; McNemar significance from `cf_stats`.
- [ ] **Step 2: Run → FAIL; Step 3: Implement** §5.1-5.7 (R/B′/P/non-meta arms, behavior+parse+degeneracy+truncation gates, bias-delta sweep monotonicity, pure-python McNemar, same-regime no-regression floor, S1/S2 gates). **Step 4: Run → PASS. Step 5: Commit.**

## Task 12: PG2/PG4/PG5/PG6 gating runs (after code, before full RL)

- [ ] **PG2** (CF leak <50% AND ≥10 graded nonzero on primed ckpt) — run CF producer; gate.
- [ ] **PG4** (primed-SFT meta-off 1030 ≥ v8-init, under true −inf + behavior rate near-zero) — gate.
- [ ] **PG6** (primed-SFT acc_with−acc_without < target final gap — RL has headroom) — gate.
- [ ] **PG5** (20-30 step micro-RL Stage-2, FULL len_cost in window, band-emission + acc_without tripwires live) — gate.
- [ ] Only if ALL pass → launch full Stage-C RL via **autoresearch** (H100 only).

---

## Self-Review
- **Spec coverage:** PG0-PG6 (Tasks 4,5,12) ✓; Stage A (6) ✓; Stage B switch-token+mask (7,8) ✓; Stage C reward (9,10) ✓; eval R−B′ (11) ✓; primitives LogitsProcessor/detector/stats (1,2,3) ✓; S1/S2/S3 gates (11) ✓.
- **Placeholder scan:** none — each task names files, a test, a smoke, a commit.
- **Type consistency:** `SwitchBanLogitsProcessor(ban_ids)` (T1) reused in cf_prefix_agent (T9) + eval (T11); `cf_stats.mcnemar_exact_p/is_parsed/degeneracy_flags` (T3) reused in T9 grading + T11 eval; `detect_redirect/measure_recall` (T2) reused in PG1 (T5) + T9 tripwire + T11.
- **Ordering:** primitives (1-3) → cheap STOP-gates (4 PG0, 5 PG1) → harvest (6) → prime (7,8) → RL code (9,10) → eval (11) → empirical gates (12) → autoresearch. Each PG can STOP before GPU spend.
