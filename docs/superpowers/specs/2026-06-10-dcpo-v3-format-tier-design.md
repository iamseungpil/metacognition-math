# TRIOBJ_DCPO_V3k — Three-Tier Format Recovery (REPLACE / DISCARD / REWARD)

Date: 2026-06-10 · Branch: ctsd-phase-c · Status: DESIGN (plan phase, pre-implementation)
Predecessors: 2026-06-09-dcpo-3region-design.md, 2026-06-09-dcpo-v3-counterfactual-design.md

## 0. Problem (measured, run ...ab03684-7068, 512 rollouts)

SFT sampling habit produces five malformation classes around the meta delimiters:

| class       | rate | surface shape                                   | current (pre-v3k) handling                       |
|-------------|------|--------------------------------------------------|--------------------------------------------------|
| no_meta     | 34%  | no meta tokens at all                            | R_meta=0 (correct)                                |
| wellformed  | 17%  | `<|meta|> content <|/meta|>`                     | full routing (correct)                            |
| SWAPPED     | 25%  | `</think> content <|/meta|>` (close-only)        | stray close IGNORED (dcpo_region.py:244-245) → meta content lands in ANSWER_REGION = **misrouted R_corr** |
| DRIFT       | 19%  | `<|meta|> content </think>` (open-only; incl 8% dup-open) | clamp+gate: R_meta forced 0, no conf mask, whole block in FORMAT_VIOLATION |
| REVERSED    | 5%   | `<|/meta|> content <|meta|>`                     | close ignored + open treated as drift/truncation  |

Meta content carries a strong content signature regardless of delimiters: lines
matching `confidence:` / `assessment:` / `action:`. Detection anchors on this.

Token ids: `<|meta|>`=151669, `<|/meta|>`=151670, `<think>`=151667, `</think>`=151668.

## 1. Verified ground truth (read 2026-06-10, this machine)

### V1 — verl old_log_prob ordering (verl source NOT on this machine)

Searched: `pip show verl` in all 4 conda envs (amlt/metaprobe/metavllm/nodejs),
`find` over $HOME and /scratch — **verl 0.7.1 source is NOT present locally**.
The prior session that wrote the CF wrap traced it against verl source
(verl_sdc.py:2381-2397 cites agent_loop.py:523/491-493/552/571/808-820,
vllm_async_server.py:549/557).

**ASSUMPTION A1 (stated explicitly, runtime-validated by §8):** in verl 0.7.1
`RayPPOTrainer.fit()`, `old_log_prob` is computed by a separate actor forward
(`actor_rollout_wg.compute_log_prob(batch)`) AFTER `generate_sequences` returns
and after `batch = batch.union(gen_batch_output)` — i.e. on the very tensors our
wrap can mutate, NOT from rollout-engine-returned logprobs.

Repo-internal supporting evidence (file:line):
- src/training/bci_agent_loop.py:79-83 — the E.9 seed-injection loop deliberately
  sets `response_logprobs=None` because "verl recomputes old_log_prob over the
  full response in the actor forward (standard guided-REINFORCE: the forced seed
  gets a valid advantage/gradient)". E.9 ran live; this is the exact same
  replace-tokens-then-let-actor-recompute mechanism v3k relies on.
- src/training/verl_sdc.py:2095-2097 + 2207-2210 — CF wrap docstrings: the wrap
  runs "RIGHT AFTER main generation", stash lands on gen_output "BEFORE
  sleep_replicas()"; sleep_replicas runs in ray_trainer after the wrap returns.
- src/training/verl_sdc.py:120-139 — the populator runs inside
  `patched_compute_advantage` (via `_attach_teacher_signals`, verl_sdc.py:1392-1393),
  which is the ADVANTAGE stage, i.e. AFTER old_log_prob. **Therefore the populator
  is TOO LATE for token replacement; the CF wrap is the only correct site.**

Residual risk: if `actor_rollout_ref.rollout.calculate_log_probs`-style reuse of
vLLM `rollout_log_probs` were active, replaced tokens would carry stale
log-probs. §8's first-live-step DCPO_DBG assertion catches this.

### V2 — which tensors hold response ids that the log-prob pass reads

In the CF wrap (`_dcpo_cf_generate_sequences`, verl_sdc.py:2234-2236, 2269-2272)
`gen_output.batch` carries: `responses`, `prompts`, `attention_mask`, optional
`response_mask`. verl's agent-loop postprocess (cited at verl_sdc.py:2394-2395)
additionally emits the concatenated `input_ids` (+ `position_ids`); the actor
log-prob forward consumes `input_ids`/`attention_mask`/`position_ids` and slices
the response tail, while advantage/masking code consumes `responses`.
Our own consumers all read `data.batch["responses"]`:
- populator decode: verl_sdc.py:160-167 (`_decode_response(prompts, responses, attention_mask)`)
- sync `__call__` from-scratch block: verl_sdc.py:1923-1931
- sync `__call__` prefilled block: verl_sdc.py:1770-1777
- CF prefix build: verl_sdc.py:2269, 2277-2289

**Replacement must therefore write BOTH:**
1. `gen_output.batch["responses"][row, pos] = new_id`
2. `gen_output.batch["input_ids"][row, prompt_len + pos] = new_id` — IF the key
   exists (defensive `.get`); assert old value == the id being replaced before
   writing (coherence guard). `prompt_len = gen_output.batch["prompts"].shape[-1]`.
`attention_mask` / `position_ids` / `response_mask` are untouched (1:1 same-length
substitution — no length change, no re-pad, no position shift).

### V3 — where conf / R_cal parsing happens (recovered content must feed it)

Two independent consumers, BOTH re-decode from `data.batch["responses"]` at
reward time, so tier-1 replaced ids flow in with zero extra plumbing:
- **CONF token mask** (R_cal routing target): `build_dcpo_region_masks` Pass B,
  dcpo_region.py:249-287 — `_parse_confidence_charspan` over the decoded
  META_CONTENT span; spans exist ONLY for closed (or legacy-unclosed) blocks.
  Drift rows currently produce NO span → empty CONF mask (tier-3 must add the
  recovered content span to `spans`).
- **R_cal scalar**: dcpo_region_rewards, dcpo_region.py:397 —
  `conf[i] = _parse_confidence(t)` over the FULL decoded text
  (rewards.py:701-712; regex is delimiter-agnostic, keyword `confidence:` etc).
  NOTE the existing asymmetry: a drift row today can get a NONZERO R_cal scalar
  (full-text parse) with an EMPTY CONF mask — the scalar still shifts sibling
  baselines through group centering while routing nowhere. Tier-3 closes this
  hole by making mask and scalar agree.

## 2. The pure parser — ONE function, reused everywhere (Karpathy lock)

New in `src/training/dcpo_region.py` (single source of truth; masks, rewards,
CF producer, and test harness all call it — NO duplicated regex across files):

```python
def classify_dcpo_format(resp_ids, response_mask, decode_fn,
                         meta_open=151669, meta_close=151670,
                         think_close=151668) -> dict:
    """{fmt_class, replacement_plan, meta_content_span, conf_required_signature,
        violation_positions, format_ok_positions, answer_start}"""
```

Returns:
- `fmt_class`: one of `wellformed | no_meta | swapped | dup_open | reversed |
  drift | truncation | discard` (tier-1 = swapped/dup_open/reversed).
- `replacement_plan`: list of `(pos, old_id, new_id)` (empty for non-tier-1).
- `meta_content_span`: `(lo, hi)` exclusive-hi content token span (post-plan
  coordinates for tier-1; recovered span for drift; None otherwise).
- `violation_positions`: list of token positions for FORMAT_VIOLATION
  (drift: the single double-duty `</think>` index; discard: every garbage
  meta-delimiter index; else empty).
- `format_ok_positions`: the closer `<|/meta|>` index for wellformed rows only.

### 2.1 Per-class detection rules (token-id level)

Delimiters are detected on TOKEN IDS only (never full-text regex — a literal
"<|meta|>" surface string in prose tokenizes differently and must not trigger).
Let `O` = real-token positions of 151669, `C` = of 151670, `K` = of 151668,
scanned over positions where `response_mask` is truthy.

Signature check (the only decode): `_has_meta_signature(decode_fn(ids[lo:hi]))`
= regex `(?im)^\s*(confidence|assessment|action)\s*:` finds ≥1 line marker.
(Reuses the keyword family of `_parse_confidence`, rewards.py:704.)

Priority-ordered classification (first match wins):

1. **no_meta**: `len(O)==0 and len(C)==0` → done, no plan.
2. **wellformed**: `len(O)==1, len(C)==1, O[0]<C[0]`, and no `K` strictly inside
   `(O[0], C[0])` → content span `(O[0]+1, C[0])`; `format_ok_positions=[C[0]]`.
3. **SWAPPED** (tier-1): `len(O)==0, len(C)==1`; let `t = max(k in K, k < C[0])`
   (the `</think>` opening the block); require `t` exists and
   signature(`ids[t+1:C[0]]`) → plan `[(t, 151668, 151669)]`.
4. **DUP_OPEN** (tier-1): `len(O)==2, len(C)==0`; no `K` strictly inside
   `(O[0], O[1])` (a `</think>` between them = drifted first span — fall through
   to discard, matching the dcpo_region.py:224-227 dup-open-after-drift edge);
   signature(`ids[O[0]+1:O[1]]`) → plan `[(O[1], 151669, 151670)]`.
5. **REVERSED** (tier-1): `len(O)==1, len(C)==1, C[0]<O[0]`;
   signature(`ids[C[0]+1:O[0]]`) → plan `[(C[0],151670,151669), (O[0],151669,151670)]`.
6. **DRIFT** (tier-3): `len(O)==1, len(C)==0`; exists `k in K, k>O[0]`; take the
   FIRST such `k`; signature(`ids[O[0]+1:k]`) → no plan;
   `meta_content_span=(O[0]+1, k)`; `violation_positions=[k]` (the double-duty
   `</think>` token ONLY — kills the R_corr leak where a correct drifted rollout
   reinforces `</think>` at w1.0); `answer_start=k+1`.
7. **TRUNCATION**: `len(O)>=1, len(C)==0`, NO `K` after the last open, run ends
   at pad → gate as today (R_meta 0, CF skip, NO penalty — length, not habit).
8. **DISCARD** (tier-2): everything else — multiple/crossing blocks, `{)}`-style
   interleavings, >2 meta tokens not matching 4-5, signature check fails for a
   tier-1/3 candidate, or §2.2 replacement validation fails.
   `violation_positions` = ALL meta-delimiter token positions present
   (`O ∪ C`, plus the drift-`</think>` when identifiable).

### 2.2 Replacement validation (tier-1 only, mandatory)

Apply the plan to a COPY of the ids; re-run `classify_dcpo_format` on the copy;
require `fmt_class == "wellformed"`. On failure → reclassify as `discard`
(strategy: "replacement-validation fails → discard"). This is cheap (pure
python) and runs in the CF wrap before any tensor write.

## 3. Tier semantics

### TIER 1 — REPLACE (`swapped` / `dup_open` / `reversed`, ~30% of rollouts)

1:1 token-id substitution: SAME length, masks/positions/attention unchanged.
After replacement the sequence IS wellformed → full normal routing
(R_corr/R_meta/R_cal), conf parsed, CF prefix cut at the now-correct opener.
- Site: inside `_dcpo_cf_generate_sequences` (verl_sdc.py:2206), immediately
  after the main gen returns and BEFORE `_dcpo_cf_build_prefixes` (line 2244) —
  post-generation, pre-old_log_prob (Assumption A1), pre-CF-cut.
- Writes per §1-V2: `responses` + `input_ids` tail, both rows asserted-coherent.
- old_log_prob is then recomputed on the REPLACED ids by verl's separate pass →
  PPO ratio at replaced positions is exactly 1 on the first minibatch; the
  gradient is `advantage × ∇logπ(corrected_tag)` — token-local STaR-style
  correction (π(correct tag) rises when the row's routed advantage is positive).
- `R_format = 0` for replaced rows AND **no format mask positions at all**
  (neither OK nor VIOLATION). Rationale beyond the strategy text: if replaced
  rows' tags sat in FORMAT_OK with R_format=0, Dr.GRPO centering against
  wellformed siblings (+1) would route NEGATIVE advantage onto the corrected
  tags — the exact conflicting signal the 0 is meant to avoid.
- Routing detail: the (corrected) opener joins META_CONTENT (same as wellformed,
  §5) so it receives the R_meta advantage; the corrected closer receives nothing
  (FORMAT_OK is wellformed-only).
- Gate: `algorithm.dcpo_format_replace` (yaml, default **true**), effective only
  under `mode==TRIOBJ_DCPO_V3`.

### TIER 2 — DISCARD (`discard`)

Regions untrustworthy — flowing anything = misrouting:
- `R_corr = R_meta = R_cal = 0.0` for the row (scalar level), AND
  ANSWER/META_CONTENT/CONF masks all-zero for the row (token level) — both are
  needed: empty masks stop routing, zeroed scalars stop the row from injecting
  ±1 into its own heads. The row is additionally EXCLUDED from the three
  content-head sibling group means (populator writes `dcpo_head_member`=0,
  compose threads it as `member_mask` into `group_mean_subtract`) — a forced-0
  is not a real reward, and averaging it in would shift every sibling by
  (d/n)·mean(siblings). The FORMAT head keeps every row (see §10 risk 2,
  CLOSED).
- `R_format = -1`, routed onto `violation_positions` (the identifiable garbage
  delimiter tokens) via FORMAT_VIOLATION.
- CF skipped (slot None).
- Diagnostics: `dcpo/discard_rate`.

### TIER 3 — REWARD (`drift`, ~19%)

Needs an INSERTION to fix (length change = invasive) → no replace; instead
content-anchored lenient region recovery:
- `META_CONTENT = (open+1 .. think_close-1)` — plays R_meta (CF runs, cut at the
  opener), conf parsed (CONF mask via Pass B over the recovered span; R_cal
  scalar already parses from full text, now mask agrees).
- ANSWER_REGION = everything after the `</think>` (+ pre-open prefix, as today:
  it is outside META_REGION).
- `R_format = -1` routed ONTO `violation_positions = [think_close_idx]` — the
  double-duty `</think>` token itself (CHANGED from the current whole-block
  FORMAT_VIOLATION at dcpo_region.py:194-195).
- The v3-current gates that force drift R_meta to 0 are LIFTED for drift
  (kept for truncation): dcpo_region.py:406-409 + 462-466 and the CF-wrap
  unclosed skip at verl_sdc.py:2314-2333 (see §6 insertion points).

### WELLFORMED rows

- Normal routing, PLUS opener `<|meta|>` INCLUDED in META_CONTENT (R_meta
  teaches WHEN to start meta), PLUS closer `<|/meta|>` in the new FORMAT_OK
  mask receiving the R_format head's group-centered POSITIVE advantage
  (`R_format = +1`) — two-sided format signal.
- This intentionally breaks the v2 invariant "tag tokens in NEITHER content nor
  answer" — v3-only (flag-gated), v2 masks byte-identical.

### TRUNCATION / NO_META rows

Unchanged: truncation = R_meta 0, CF skip, R_format 0 (no penalty);
no_meta = R_meta 0 (no CF), R_format 0.

## 4. Head routing table per class

R_format raw values per row: wellformed +1 · drift −1 · discard −1 ·
replaced/no_meta/truncation 0. ONE head, group-mean-subtracted ONCE
(Dr.GRPO, no /std), routed onto the row's own format positions.

| fmt_class  | R_corr | R_meta | R_cal | R_format | META_CONTENT            | ANSWER_REGION       | CONF        | FORMAT_OK     | FORMAT_VIOLATION         | CF gen |
|------------|--------|--------|-------|----------|--------------------------|----------------------|-------------|---------------|---------------------------|--------|
| wellformed | ±1     | c_w−c_wo | Brier | **+1**  | content **+ opener tag** | resp − META_REGION   | conf run    | closer token  | ∅                         | run    |
| replaced   | ±1     | c_w−c_wo | Brier | 0       | content + opener (post-replace) | resp − META_REGION | conf run | ∅            | ∅                         | run (cut at corrected opener) |
| drift      | ±1 (post-`</think>` answer) | c_w−c_wo (**ungated**) | Brier | **−1** | recovered (open+1..k−1)  | k+1..end (+ pre-open) | conf run in recovered span | ∅ | `</think>` token ONLY | run    |
| discard    | **0**  | **0**  | **0** | **−1**   | ∅ (zeroed)               | ∅ (zeroed)           | ∅           | ∅             | garbage delimiter tokens   | skip   |
| truncation | ±1     | 0 (gated) | 0/Brier-as-today | 0 | ∅                     | resp − META_REGION   | ∅           | ∅             | ∅                         | skip   |
| no_meta    | ±1     | 0      | 0     | 0        | ∅                        | all real tokens      | ∅           | ∅             | ∅                         | skip   |

Advantage composition (extends `compose_dcpo_region_advantage`,
dcpo_region.py:558-613, backward-compatible optional param):

```
A_token = ( w_corr·Â_corr·ANSWER + w_meta·Â_meta·META_CONTENT + w_cal·Â_cal·CONF
          + w_format·Â_format·(FORMAT_OK ∪ FORMAT_VIOLATION) ) · response_mask
```

FORMAT_OK and FORMAT_VIOLATION are per-row disjoint by construction (a row is
exactly one class), so the union routes each row's centered Â_format onto its
own positions: positive-relative onto wellformed closers, negative-relative onto
drift `</think>` / discard garbage. Implementation: new optional
`format_ok_mask=None` param; `None` → byte-identical to current 4-head compose
(v2 + existing-test compatibility).

## 5. FORMAT_OK / FORMAT_VIOLATION mask semantics (exact)

- `FORMAT_VIOLATION[t]=1` iff position `t` is a wrong-format token the policy is
  being pushed AWAY from: drift rows → exactly the double-duty `</think>` index;
  discard rows → every garbage meta-delimiter index. All other rows: all-zero.
  (CHANGE vs current: today it is the whole drift-clamped block o..k−1.)
- `FORMAT_OK[t]=1` iff `t` is the closer `<|/meta|>` of an ORIGINALLY-wellformed
  row (fmt_class==wellformed; replaced rows excluded by design, §3-Tier-1).
- Invariants (asserted in tests): FORMAT_OK ∩ FORMAT_VIOLATION = ∅ (per row and
  globally); FORMAT_VIOLATION ⊆ response_mask; FORMAT_OK ⊆ META_REGION;
  FORMAT_VIOLATION ∩ META_CONTENT = ∅ (drift content excludes `</think>`;
  discard content is empty); CONF ⊆ META_CONTENT; for v3k wellformed/replaced
  rows: opener ∈ META_CONTENT, closer ∈ META_REGION \ META_CONTENT.
- v2 path (`clamp_unclosed=False` legacy branch) emits both masks all-zero —
  byte-identical.

## 6. Exact insertion points (file:line, surgical)

1. **`src/training/dcpo_region.py`**
   a. ADD `classify_dcpo_format` (+ `_has_meta_signature`) after
      `first_meta_token_index` (after line 79). Pure python, decode_fn-injected.
   b. `build_dcpo_region_masks` (line 97): new kwarg
      `fmt_mode: str = "v3"` → extend to accept `fmt_class`/span output of the
      parser via a new kwarg `fmt: dict | None = None` (None → legacy v3
      behavior so v2/v3-pre-k callers unchanged). When `fmt` given:
      wellformed/replaced → opener into META_CONTENT + FORMAT_OK at closer;
      drift → META_CONTENT = recovered span, append it to `spans` (so Pass B
      lines 249-287 parses CONF), FORMAT_VIOLATION = {k} only; discard → zero
      ANSWER/META_CONTENT/CONF, FORMAT_VIOLATION = garbage positions; returns
      additionally `FORMAT_OK` + `fmt_class`.
   c. `dcpo_region_rewards` (line 310): new kwarg `fmt_class=None` (len-B list).
      When given: drift rows bypass the unclosed R_meta gate (lines 406-409 +
      462-466 keep gating truncation only); discard rows force
      R_corr/R_meta/R_cal = 0; `format_penalty` per-row per §4 table (−1 drift,
      −1 discard, +1 wellformed, 0 else — RENAME-IN-VALUE only, key unchanged).
      `fmt_class=None` → current v3 behavior verbatim (and `gate_unclosed=False`
      → v2 verbatim).
   d. `compose_dcpo_region_advantage` (line 558): optional `format_ok_mask=None`;
      route `Â_format` onto `fv + ok` when given (line 608-611 block).
2. **`src/training/verl_sdc.py` — CF wrap (the replacement site)**
   a. In `_dcpo_cf_generate_sequences` (line 2206), after
      `gen_output = self._dcpo_cf_orig_generate(gen_batch)` (line 2225) and the
      validate early-return (2227), BEFORE `_dcpo_cf_build_prefixes` (2244):
      call new `self._dcpo_format_classify_and_replace(gen_output)` —
      per row: run `classify_dcpo_format` on the valid response ids; if tier-1
      and `dcpo_format_replace`: validate (§2.2), write `responses` +
      `input_ids` tail (§1-V2); stash `gen_output.non_tensor_batch["dcpo_fmt_class"]`
      (object array of class strings, the populator's authoritative source) +
      `["dcpo_fmt_replaced"]` (0/1 float) — they flow through fit()'s union like
      `cf_texts` (verl_sdc.py:215). Gated on `mode==TRIOBJ_DCPO_V3` AND the
      yaml knob; otherwise byte-identical pass-through.
   b. Wrap INSTALL gate (line 2098 + 2145): currently only
      `sdc_counterfactual`; extend the init flag to
      `self._dcpo_cf or (mode==V3 and dcpo_format_replace)` so replacement still
      runs if CF were ever turned off (v3 yaml has both true; this is belt-and-
      suspenders, not a behavior change for the live config).
   c. CF unclosed skip-gate (lines 2314-2333): narrow from "no `<|/meta|>`
      anywhere" to fmt_class ∈ {truncation, discard} (drift now RUNS CF —
      tier-3 plays R_meta). Use the stashed `dcpo_fmt_class` instead of the
      text check (one parser, no duplicated logic).
   d. CF prefix cut (line 2279): unchanged code — `first_meta_token_index` now
      naturally finds the corrected opener for replaced rows (replacement
      happened first).
3. **`src/training/verl_sdc.py` — populator `_populate_dcpo_region_keys` (120-268)**
   - Read `data.non_tensor_batch.get("dcpo_fmt_class")`; if absent (wrap not
     installed) fall back to running `classify_dcpo_format` here WITHOUT
     replacement and treating tier-1 rows as DISCARD (conservative — too late to
     replace, old_log_prob already computed).
   - Pass `fmt=` into `build_dcpo_region_masks` (line 177) and `fmt_class=` into
     `_compute_dcpo_heads_stash` (line 219, threading through line 107).
   - Stack `data.batch["dcpo_format_ok_mask"]` (v3-only, mirror of the
     format_violation stack at lines 198-199).
4. **`src/training/verl_sdc.py` — sync `__call__` DCPO blocks**
   (prefilled path 1764-1817 AND from-scratch path 1916-1972): mirror item 3
   exactly (five-way sync rule; these two blocks are the async populator's
   sync twins and have crashed twice from drift).
5. **`src/training/verl_sdc.py` — `_compute_dcpo_heads_stash` (90-117)**:
   thread `fmt_class` kwarg through to `dcpo_region_rewards`.
6. **`src/training/verl_sdc_utils.py` — `_compute_dcpo_region_advantage` (238-294)**:
   optional `dcpo_format_ok_mask` read (line ~272, same presence-gate pattern as
   `_fv_mask`), passed as `format_ok_mask`. Absence-tolerant (older ckpts/v2).
7. **`configs/triobj_dcpo_v3_h100_4x4k.yaml`** (algorithm block, after line 63):
   `dcpo_format_replace: true` (+ comment). No new reward keys (format_penalty
   already at line 55-56) → gdpo_reward_keys/weights UNCHANGED.
8. **`src/training/verl_sdc.py` — observability**
   - `_log_dcpo_trend_scalars` (271-323): add `dcpo/replaced_rate`,
     `dcpo/discard_rate`, `dcpo/drift_rate`, `dcpo/wellformed_rate` from the
     stashed class counts; KEEP `dcpo/meta_unclosed_rate` (continuity: still =
     textual unclosed = drift + truncation); `dcpo/format_penalty_rate` keeps
     meaning "rows with R_format<0" (now drift+discard).
   - `_log_dcpo_rollout_table` (326-369): add column `fmt_class` (and
     `replaced` bool) after `unclosed`.
   - §8 assertion scalars: `dcpo/replaced_oldlp_mean`, `dcpo/sampled_oldlp_mean`.

## 7. Five-way sync checklist (extends the three-way rule; three prior crashes)

| # | surface | v3k state |
|---|---------|-----------|
| 1 | `REWARD_CONFIGS['TRIOBJ_DCPO_V3']` funcs/keys/weights (verl_sdc.py:785-804) | UNCHANGED — 5 funcs/keys, weights [1.0,0.5,0.3,0.0,0.1]; format_penalty VALUES change (−1/0 → +1/0/−1), key set does not |
| 2 | yaml `gdpo_reward_keys`/`gdpo_reward_weights` (triobj_dcpo_v3...yaml:55-56) | UNCHANGED lists; ADD `dcpo_format_replace: true` knob only |
| 3 | populator `non_tensor_batch` writes (verl_sdc.py:228-246) | UNCHANGED key set (correctness, meta_region_utility, cal_region_reward, meta_emission, format_penalty); NEW batch tensors `dcpo_format_ok_mask` (v3-only) alongside `dcpo_format_violation_mask` |
| 4 | compose params (`compose_dcpo_region_advantage` ↔ `_compute_dcpo_region_advantage`) | NEW optional `format_ok_mask` on both sides, presence-gated, absence = byte-identical |
| 5 | masks (`build_dcpo_region_masks` output keys ↔ populator stacks ↔ sync `__call__` stacks ×2) | NEW `FORMAT_OK` key; ALL THREE stack sites (populator + 2 sync blocks) must add it under the same `_is_v3` gate |

Guard tests: extend `test_v3_yaml_reward_lists_match_reward_configs` +
`test_populate_writes_every_gdpo_reward_key` (test_dcpo_v3.py:209, 229) and add
a mask-stack-parity test asserting populator and both sync blocks stack the same
v3 mask key set.

## 8. Runtime DCPO_DBG assertion (validates Assumption A1 on the first live step)

verl source is absent locally, so the old_log_prob ordering is validated AT
RUNTIME instead of by code reading:

1. **At the replacement site (CF wrap), every step:**
   - hard-assert `"old_log_probs" not in gen_output.batch` and
     `"rollout_log_probs" not in gen_output.batch` — if either exists, the
     engine returned/precomputed log-probs that replacement would invalidate →
     print `[DCPO_DBG] FORMAT-REPLACE ABORT` and skip ALL replacement for the
     run (rows degrade to discard; training never silently trains on stale
     ratios).
   - per replaced position: assert the pre-write value equals the plan's
     `old_id` in BOTH `responses` and `input_ids` tail; post-write re-read both.
2. **At the populator (advantage stage, old_log_probs now in batch), first
   replaced step only (then every N=50):**
   - assert `data.batch["responses"][row, pos] == new_id` for every recorded
     replacement (replacement survived fit()'s union);
   - assert `old_log_probs[row, pos]` finite;
   - log `dcpo/replaced_oldlp_mean` vs `dcpo/sampled_oldlp_mean`: the corrected
     tag is a token the policy did NOT sample, so its old_log_prob should sit
     well below the sampled-token mean on step 1. If
     `replaced_oldlp_mean > sampled_oldlp_mean − 0.5` print a LOUD
     `[DCPO_DBG] OLD-LOGPROB-CONSISTENCY SUSPECT` warning (heuristic, warn-only;
     the hard guarantees are the two asserts above).
   The replacement record travels as `non_tensor_batch["dcpo_fmt_replace_plan"]`
   (object array of per-row `[(pos, old, new), ...]`, empty lists for
   non-replaced rows).

## 9. Test matrix (extend tests/test_dcpo_v3.py; pure-python, metaprobe env)

| group | cases |
|-------|-------|
| parser classes | one test per class × the measured surface shapes: no_meta, wellformed, swapped (`</think> sig <|/meta|>`), dup_open (`<|meta|> sig <|meta|>`), reversed, drift (`<|meta|> sig </think> ans`), truncation, discard (crossing blocks; 3+ meta tokens; signatureless swapped candidate) |
| signature anchor | swapped WITHOUT `confidence:/assessment:/action:` lines → discard (not replaced); signature present but no preceding `</think>` → discard |
| replacement plans | exact `(pos, old, new)` tuples per tier-1 class; plan applied → re-classify == wellformed; validation-failure path → discard |
| same-length invariant | len(ids) unchanged; attention/positions untouched (plan never inserts/deletes) |
| mask building w/ fmt | wellformed: opener ∈ META_CONTENT, closer ∈ FORMAT_OK; replaced: opener ∈ META_CONTENT, FORMAT_OK empty; drift: META_CONTENT = recovered span, CONF parsed inside it, FORMAT_VIOLATION == {`</think>` idx} ONLY; discard: ANSWER/META/CONF all-zero, FORMAT_VIOLATION == garbage positions; §5 invariants |
| reward gating | drift row WITH positive CF → R_meta ≠ 0 (gate lifted — updates test_rmeta_gated_unclosed_even_with_positive_cf:563 expectation for drift; truncation case at :575 stays gated); discard row → R_corr=R_meta=R_cal=0, format=−1; wellformed → format=+1; replaced → format=0 |
| compose routing | Â_format lands on FORMAT_OK ∪ FORMAT_VIOLATION only; format_ok_mask=None → byte-identical to current output (regression for v2 + pre-k v3); update test_compose_format_head_routes_only_on_violation_mask:605 for the union |
| v2 byte-identity | clamp_unclosed=False / gate_unclosed=False / fmt=None paths byte-identical (extend :527/:547); v2 populator stacks NO format_ok/violation masks |
| five-way sync | yaml↔REWARD_CONFIGS lists (:209), populator writes every key (:229), NEW: three mask-stack sites stack identical v3 key sets |
| CF interplay | replaced row: `first_meta_token_index` on post-plan ids == corrected opener; drift row not in the CF skip set; truncation/discard in it |
| existing-drift tests | test_mask_drift_clamps_at_think_close:454 / test_format_penalty_only_for_drift_rows:587 updated to v3k semantics (violation = single token; drift R_meta ungated) — v2-flagged variants preserved |

## 10. Risks / noted tensions (resolved or accepted)

1. **CF skip-gate vs tier-3** (found, resolved §6-2c): the live unclosed-meta CF
   skip (verl_sdc.py:2314-2333) would starve drift rows of their counterfactual.
2. **Discard zeros shift sibling baselines** (was accepted; CLOSED 2026-06-10):
   a forced-0 R_corr entering the group mean shifts every sibling by
   (d/n)·mean(siblings) — e.g. one discard in an all-correct group of 4 handed
   each sibling a spurious +0.25 at w_corr. The harness measured discard as the
   LARGEST class on real data (30.3% of full responses vs the ~0% the 512-rollout
   taxonomy implied — the close-led no-opener shapes were lumped into "swapped
   25%"), so the bias was upgraded from accepted to fixed: exclusion-aware
   `group_mean_subtract(member=...)` + compose `member_mask` + populator
   `dcpo_head_member` (discard rows contribute nothing to and receive nothing
   from the content-head means; FORMAT head keeps every row). `dcpo/discard_rate`
   still makes the mass visible. OPEN follow-up (user decision): extend recovery
   to the double-close/no-opener discard shapes (CCK/CK/CKK/C ≈ 73% of discards)
   — not 1:1-fixable under §2.1 rule 3.
3. **Replaced-row format neutrality** (found, resolved §3): R_format=0 +
   membership in FORMAT_OK would have routed negative centered advantage onto
   corrected tags; replaced rows therefore carry NO format positions.
4. **Invariant break is v3-only**: opener-in-META_CONTENT violates the v2 tag
   invariant; all new mask behavior keyed off the `fmt` kwarg / `_is_v3` gates.
5. **R_cal scalar/mask asymmetry for drift** (pre-existing, §1-V3): closed by
   tier-3 recovered spans.
6. **All-wellformed groups center R_format to 0**: Dr.GRPO relative signal —
   intended (no absolute format farming).
7. **Replacement requires the wrap**: populator fallback treats unreplaced
   tier-1 as discard (never half-replaced); install gate widened (§6-2b).

## 11. Deletions / non-goals

- No refactor of v2 paths, no global re-whiten change, no new reward KEYS.
- No insertion-based repair (drift stays tier-3 by design — length change would
  desync masks/positions/old_log_prob).
- DO NOT git commit in this phase.
