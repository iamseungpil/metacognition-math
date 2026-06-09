# TRIOBJ_DCPO_V3 — Counterfactual Meta-Ablation (causal meta-utility)

Status: DESIGN v3 (user-approved; surgical, additive). Date: 2026-06-09.
Owner: metacognition-math / verl_sdc training path.
Predecessor: `TRIOBJ_DCPO_V2` (`docs/superpowers/specs/2026-06-09-dcpo-3region-design.md`;
flat-`+1` transition table R_meta, two-pass INSTR prompt, `h200_triobj_dcpo_v2.yaml`).

This spec defines a NEW additive verl mode `TRIOBJ_DCPO_V3`. It MUST NOT alter the behavior of
any existing mode. Every change is gated behind `sdc_mode == "TRIOBJ_DCPO_V3"` (added to
`_REGION_ROUTED_MODES`, `verl_sdc.py:598`) or a new `REWARD_CONFIGS` key, exactly like the
existing additive-mode pattern.

---

## §1 Goal + intent alignment (causal meta-utility)

**North star.** Reward USEFUL metacognition by its CAUSAL effect: *does deleting the meta make
the answer worse?* v2 used a transition proxy (`answer1→answer2` flip table) that only captures
metacognition which produced a *visible answer revision* — it is blind to in-line verification
that prevents an error before any answer is written, and it can be hacked by writing a wrong
draft on purpose to then "revise" it. v3 replaces the proxy with the real counterfactual.

**Definition.** For a main rollout that emits `<think>...<|meta|>verify+confidence<|/meta|>...answer`,
the meta-utility is

```
R_meta = c_with - c_without
       = correct(main answer with meta in context)
       - correct(counterfactual answer generated from the SAME prefix with <|meta|> suppressed)
```

`R_meta ∈ {-1, 0, +1}`: `+1` when the meta turned a wrong answer right, `-1` when meta turned a
right answer wrong, `0` when meta did not change the outcome (the natural / no-harm case). The
counterfactual continues the model's *own* prefix up to (but not including) the first `<|meta|>`,
so it is the policy's own "what would I have answered without stopping to verify" — a within-rollout,
policy-conditioned comparison, not a teacher or a different prompt.

**Intent alignment.** This is the literal operationalization of the MEMORY north star (reframe:
define + reward the metacognition we want, calibration is a sub-goal). The 3-region routing from v2
is KEPT so the credit lands on the right tokens (utility→meta-content, calibration→conf, correctness
→answer). Only the *definition of the meta head* changes (proxy → causal delta), plus the prompt and
the rollout plumbing that produces the counterfactual.

**Why drop the two-pass INSTR (evidence).** v1/v2 prepend a two-pass INSTR to every prompt via a
parquet rewrite (`h200_triobj_dcpo_v2.yaml:60-82`). Measured harm: with the INSTR, the model emits
`<|meta|>` at eval only **65%** of the time (v1); the SFT/e8 plain-redirect prompt (no INSTR) emits
`<|meta|>` **99.8%** at eval. The INSTR suppresses the very behavior we train. v3 trains on the plain
redirect prompt so the model emits its natural single-pass `<think>...<|meta|>...answer` (ONE answer,
meta inside think), and the counterfactual — not a two-box transition — supplies the causal signal.

---

## §2 Data flow (8 inference / 4 GRPO; counterfactuals inference-only)

Per training prompt, GRPO group size `K=4`:

```
PROMPT (plain redirect, NO two-pass INSTR)
  │
  ├─ MAIN gen (actor_rollout_wg.generate_sequences, n=4 repeat)  ── 4 rollouts, TRAINED (GRPO group=4)
  │     each rollout r_i = <think> ... <|meta|> verify+conf <|/meta|> ... answer
  │
  └─ for EACH main rollout r_i  (i=1..4):
        firstMetaIdx_i = index of FIRST <|meta|> (id 151669) in r_i's response tokens
        prefix_i       = prompt_ids + response_ids[:firstMetaIdx_i]      # strictly BEFORE the tag
        CF gen (2nd generate_sequences on prefix_i, <|meta|> SUPPRESSED) ── 4 rollouts, INFERENCE-ONLY
              cf_i = model continues prefix_i WITHOUT ever emitting <|meta|>
              cf_correct_i = correct(extract_answer(cf_i))
```

- **Total inference = 8 per prompt** (4 main + 4 counterfactual).
- **The 4 counterfactuals are INFERENCE-ONLY**: generated under `no_grad`, NOT placed in the GRPO
  group, NOT scored for advantage, discarded after `cf_correct` is graded. They contribute exactly
  one scalar each (`cf_correct_i`) consumed by `R_meta`.
- **One counterfactual per main rollout, regardless of how many meta blocks the rollout has.** We cut
  at the FIRST `<|meta|>` and suppress the token for the rest of the continuation, so `N` meta blocks
  still collapse to ONE counterfactual (the "remove ALL meta" semantics = cut at first + suppress).
- **No-meta rollout ⇒ natural delta 0.** If `r_i` contains no `<|meta|>` token, `firstMetaIdx_i = None`,
  `prefix_i = whole response`, so `cf_i ≈ r_i` and `R_meta_i = 0`. (Implementation short-circuits: when
  no meta is present we skip the CF gen entirely and set `R_meta_i = 0`.)

---

## §3 The verl 2nd-generation hook (insertion point + signature + suppression)

**Feasibility: EASY–MEDIUM. CONFIRMED.** A second batched generation on custom prefixes within the
same trainer step is structurally supported by verl, and the SDC codebase already contains a
purpose-built, currently-stubbed hook for exactly this (`_force_inject_rollout`). No retrain-loop
restructure is needed.

### 3.1 Why a 2nd in-step gen is legal (existence proofs)

- **verl itself does it:** the REMAX branch calls `self.async_rollout_manager.generate_sequences(
  gen_baseline_batch)` a second time on a modified copy in the *same* step
  (`/tmp/verl_src/verl/trainer/ppo/ray_trainer.py:1335`, batch built :1331-1332).
- **`@auto_await` makes it synchronous:** `AgentLoopManager.generate_sequences` is `async def` but
  decorated `@auto_await` (`agent_loop.py:1029`; impl `ray_utils.py:97`). Called outside a running loop
  (as in `fit()`), it does `asyncio.run(coro)` and returns a *materialized* `DataProto` synchronously
  (`ray_utils.py:122-123`). From the trainer's view it is a normal blocking call — callable again inline,
  no `await`/event-loop plumbing.
- **SDC already wires a custom 2nd-gen path:** `_bci_generate_sequences` (`verl_sdc.py:1803`) wraps the
  manager's `generate_sequences` and injects per-sample custom data (`agent_name`, token-id lists) via
  `non_tensor_batch` (:1826-1834). This is the exact template for "custom prefix per sample".

### 3.2 The two runtime constraints (why MEDIUM not trivially EASY)

- **(A) Replicas sleep after main gen.** `self.checkpoint_manager.sleep_replicas()` runs immediately
  after the main gen (`ray_trainer.py:1322`). The CF generate MUST happen *before* that sleep (or wake
  replicas again). Cleanest: do the CF pass at the `generate_sequences` boundary where replicas are
  guaranteed awake — i.e. inside the wrapped generate (what the BCI wrap relies on). **This is why the
  CF gen lives in the rollout/fit phase, NOT in `compute_advantage`** (see §5.2).
- **(B) Mask/length coherence for downstream reward.** The CF responses are graded *as text* (answer
  extraction), so we do NOT need to splice them back into the main batch's masks. The CF outputs are a
  side channel; only `cf_correct[i]` (a float array, length B) is carried forward. This sidesteps the
  v2-era mask-coherence worry entirely (we never route advantage through CF tokens).

### 3.3 Exact insertion point + call signature

**Insertion point: fill the existing `SDCRayPPOTrainer._force_inject_rollout` stub** (`verl_sdc.py:1870`,
currently `NotImplementedError` at :1895), gated behind a NEW flag `algorithm.sdc_counterfactual`
(byte-identical when off). Do NOT use `patched_compute_advantage` / `_attach_teacher_signals` to
*produce* the CF gen — those run after `sleep_replicas()` with the engine asleep (`verl_sdc.py:2299`,
`:1146`); they are the right place to *consume* `cf_correct`, not to generate it (§5.2, and the
`_generate_v0_prefixes` busy-rollout deadlock TODO at `verl_sdc.py:922-930` is the standing warning).

Wire it at the `generate_sequences` boundary exactly like `_bci_generate_sequences` (`verl_sdc.py:1791-1792`):
under `sdc_counterfactual`, override `mgr.generate_sequences` so that after the main gen returns it (a) cuts
prefixes, (b) runs the CF gen, (c) grades, (d) stashes `cf_correct` onto the returned `DataProto`'s
`non_tensor_batch`, all before control returns to `ray_trainer.py:1322`'s `sleep_replicas()`.

```python
# inside the generate_sequences wrapper, after main gen, replicas still awake:
def generate_with_counterfactual(self, gen_batch: "DataProto") -> "DataProto":
    gen_output = self._orig_generate_sequences(gen_batch)        # MAIN, K=4 already repeated
    if not bool(getattr(self.config.algorithm, "sdc_counterfactual", False)):
        return gen_output
    cf_batch = self._build_cf_batch(gen_batch, gen_output)        # §3.4 — per-row prefix ids
    cf_out   = self._orig_generate_sequences(cf_batch)            # 2nd gen, <|meta|> suppressed (§3.5)
    cf_correct = self._grade_cf(cf_out, gen_batch)               # decode→extract→math_verify, length B
    gen_output.non_tensor_batch["cf_correct"] = np.asarray(cf_correct, dtype=np.float32)
    return gen_output
```

`self._orig_generate_sequences` is the manager's bound `generate_sequences` captured before the override
(mirror `verl_sdc.py:1792`). The CF call reuses the SAME engine (no re-init); replicas awake by construction.

### 3.4 `cf_batch` construction (mirror the BCI pattern)

Per main rollout `i` (B = `len(gen_batch) * 1`, already at K-repeat granularity so B = num_prompts·K):

1. `resp_ids_i = gen_output.batch["responses"][i]`, `resp_mask_i = gen_output.batch["response_mask"][i]`.
2. `j = first_meta_index(resp_ids_i, resp_mask_i)` (NEW helper, §6.1). If `j is None`: mark
   `skip_i = True` (no CF needed; `R_meta_i` will be 0). Else `prefix_ids_i = prompt_ids_i +
   resp_ids_i[:j].tolist()` (strip left-pad from `prompt_ids_i`).
3. Route to a custom agent_loop that accepts pre-tokenized prefix ids (the stock `single_turn_agent_loop`
   only takes `raw_prompt` chat messages — `single_turn_agent_loop.py:37,45,54`; it has no token-id-prefix
   entry). Reuse/clone the BCI agent-loop pattern that splices a token-id list:

```python
B = len(gen_batch)
cf_batch = gen_batch.select(...)  # copy meta_info; we overwrite non_tensor below
cf_batch.non_tensor_batch["agent_name"]  = np.array(["cf_prefix_agent"] * B, dtype=object)
prefix = np.empty(B, dtype=object)        # dtype=object so numpy does NOT collapse equal-length lists
for i in range(B):
    prefix[i] = [] if skip[i] else list(prefix_ids[i])
cf_batch.non_tensor_batch["prefix_ids"] = prefix
cf_batch.meta_info = {**gen_batch.meta_info, "validate": False}   # carry global_steps/temperature
# per-call suppression of <|meta|> (§3.5) attached via the agent_loop's sampling_params
```

Skipped rows (`prefix == []`) can be passed as a 1-token no-op or filtered out of `cf_batch` and
re-inserted as `cf_correct = main_correct` afterward; filtering is cleaner (fewer wasted tokens). The
manager returns a `DataProto` with `prompts/responses/input_ids/attention_mask/response_mask/position_ids`
left/right-padded and concatenated (`agent_loop.py:462-469`), so `cf_out.responses` decodes directly.

### 3.5 Token suppression (`<|meta|>` = 151669) on the CF call ONLY

verl forwards the per-request `sampling_params` dict **verbatim** into vLLM — no key allowlist. Decisive
line: `vllm_async_server.py:549` `SamplingParams(max_tokens=max_tokens, **sampling_params)`. The dict is
threaded unmodified `single_turn_agent_loop.py:54 → agent_loop.py:157 → server.generate.remote`. vLLM's
`SamplingParams` supports `logit_bias: dict[int,float]` (`sampling_params.py:202/250`).

**Use `logit_bias` (cleanest: raw token id, no tokenizer round-trip, JSON-serializable across Ray):**

```python
# in cf_prefix_agent's run(), copy the shared dict before mutating:
cf_sp = dict(base_sampling_params)
cf_sp["logit_bias"] = {151669: -100.0}     # drive <|meta|> logit to ~ -inf → effective hard ban
# main rollout's sampling dict is UNTOUCHED — only this 2nd call suppresses the token.
```

`-100.0` is an effective ban while remaining a soft bias (numerically safe). If an absolute guarantee is
required, add a one-line `logits_processors` entry masking index 151669 to `-inf`; avoid `bad_words`
(expects a decodable string; a special token may not round-trip through the tokenizer cleanly).

### 3.6 Fallback if the custom-prefix agent-loop proves HARD

If wiring `cf_prefix_agent` to ingest token-id prefixes is blocked by a verl-version detail, two
fallbacks (in preference order):

1. **Chat-message prefix via stock `single_turn_agent_loop`.** Set `non_tensor_batch["raw_prompt"]` to
   the prompt messages + a partial assistant turn = decoded `response[:firstMeta]`, leave `agent_name`
   default, attach `logit_bias` via sampling. Simpler plumbing; weaker fidelity (the partial assistant
   continuation is re-tokenized from text, not the exact prefix ids — acceptable for a *grading-only*
   side channel).
2. **Separate counterfactual pass (decoupled step).** Run the CF gen as its own `generate_sequences`
   call on a freshly-built `DataProto` of prefixes right after main gen (REMAX-style, `ray_trainer.py:1335`
   is the existence proof), still before `sleep_replicas()`. Same engine, no agent-loop customization,
   at the cost of one extra batch assembly. This is the safest fallback and is the recommended path if
   `_force_inject_rollout`'s repack fights the live runtime.

Honest assessment: the token-id-prefix agent loop is the *clean* path and is already 90% built (BCI). The
chat-message fallback is trivially available and loses only exact-prefix fidelity, which does not matter
for a graded scalar. We are not blocked.

---

## §4 The 3 rewards + region routing (KEEP routing, REPLACE R_meta)

Region masks (`build_dcpo_region_masks`, `verl_sdc.py`-VERIFIED on real `<|meta|>` responses: detects
META_REGION / META_CONTENT / `CONF='0.22'`) and region routing (`compose_dcpo_region_advantage`:
`A_corr*answer_mask + A_meta*meta_content_mask + A_cal*conf_mask`, VERIFIED) are KEPT unchanged. Tag tokens
are in META_REGION but NOT META_CONTENT (`dcpo_region.py:115,119`), so the delta routes to exactly the
inner meta-content span.

| Head | Definition (v3) | Routed to (mask) | Group op |
|------|-----------------|------------------|----------|
| `R_corr` | correctness of MAIN final answer (lenient `_extract_answer_fallback`) on ANSWER tokens | ANSWER | group-mean-subtract |
| `R_meta` | **`c_with − c_without`** = `correct(main) − correct(counterfactual)` ∈ {−1,0,+1} | META_CONTENT | group-mean-subtract |
| `R_cal` | **`−(conf − c_with)²`** Brier; `conf` parsed from inside meta | CONF | (Brier; per-instance) |

`c_with[i] = 1.0 if main correct else 0.0`. `c_without[i] = cf_correct[i]` (from §3, the graded
counterfactual). `conf[i]` = parsed confidence number inside the meta block.

**DROP entirely (all v2 carry-over):** two-pass INSTR; two-box `answer1→answer2` transition; the
transition table (`dcpo_region.py:301-326`); `format_penalty`/`format_credit`; single-pass enforcement;
`w_warmup`; sandbag `canary`/`clamp_f` circuit-breaker; `p_lo/p_hi` warrant gate (as an R_meta gate —
`p_hat`/`group_acc` stay as returned diagnostics only).

**KEEP off:** KL/entropy disabled (Dr.GRPO implicit reg) stays off, as in v2.

---

## §5 Exact files / edits (surgical)

### 5.1 `src/training/dcpo_region.py` — R_meta → delta + 2 new helpers

`META_OPEN_DEFAULT = 151669` already exists (`dcpo_region.py:38`); `dcpo_region_rewards` at `:188`;
`compose_dcpo_region_advantage` at `:393` (UNCHANGED).

**Edit A — add 2 helpers (after imports, ~line 40):**

```python
def first_meta_index(resp_ids, response_mask=None, meta_open: int = META_OPEN_DEFAULT):
    """Token position of the FIRST <|meta|> among real response tokens, else None.
    The counterfactual 'without-meta' prefix is resp_ids[:i] (strictly before the tag)."""
    ids = [int(t) for t in (resp_ids.tolist() if hasattr(resp_ids, "tolist") else list(resp_ids))]
    rm = ([True] * len(ids) if response_mask is None
          else (response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)))
    for i, t in enumerate(ids):
        if i < len(rm) and not rm[i]:
            continue
        if t == meta_open:
            return i
    return None


def cf_answer_from_prefix(text: str):
    """TEXT-fallback counterfactual answer = extract from the pre-(first-)meta prefix only.
    Used by dcpo_region_rewards ONLY when no real regenerated cf rollout is supplied."""
    if "<|meta|>" not in text:
        return None
    prefix = text.split("<|meta|>", 1)[0]
    return _extract_answer_fallback(prefix) or None
```

`first_meta_index` is the TOKEN-level cut the producer (§3.4) uses to build the CF prefix.
`cf_answer_from_prefix` is the TEXT-level fallback grader used inside `dcpo_region_rewards` when the real
CF rollout was not threaded through (keeps the head functional from a single rollout).

**Edit B — rewrite `dcpo_region_rewards` body (the R_meta half only):**

1. Add optional arg `cf_completions=None` (parallel list of CF rollout texts, len B, from §3 producer).
2. Trim the per-rollout primitives loop to compute only `answer2`/`c2`/`conf` (final answer, correctness,
   parsed confidence) per rollout, and `c_without[i]`:
   ```python
   if cf_completions is not None:
       cf_txt = _get_text(cf_completions[i])
       cf_ans = _extract_answer_fallback(cf_txt) if cf_txt else None
   else:
       cf_ans = cf_answer_from_prefix(texts[i])       # text fallback
   c_without[i] = bool(_check_correctness(cf_ans, gts[i])) if cf_ans is not None else None
   ```
3. KEEP the group `p_hat`/`group_acc` block (returned as diagnostics).
4. DELETE `w_warmup`, the sandbag block, and the transition table.
5. Replace the head assignment:
   ```python
   for i in range(B):
       R_corr[i] = 1.0 if c2[i] else -1.0
       c_with    = 1.0 if c2[i] else 0.0
       R_meta[i] = 0.0 if c_without[i] is None else (c_with - (1.0 if c_without[i] else 0.0))  # {-1,0,+1}
       R_cal[i]  = -((conf[i] - c_with) ** 2) if conf[i] is not None else 0.0
   ```
6. Return dict: keep `R_corr/R_meta/R_cal/p_hat/group_acc`. Keep `canary_pass1_acc`/`sandbag_clamp` as
   constant `[1.0]*B` stubs so existing wandb keys + `_populate_dcpo_region_keys` stay alive without
   touching `verl_sdc.py`.
7. Keep DROPPED kwargs (`eps/p_lo/p_hi/warmup_steps/sandbag_*/format_*`) in the signature as
   accepted-but-ignored (absorbed by `**cfg`) for caller compatibility.

### 5.2 `src/training/verl_sdc.py` — attach `cf_correct`, trigger CF gen

- **Add `"TRIOBJ_DCPO_V3"` to `_REGION_ROUTED_MODES`** (`verl_sdc.py:598`) and register the v3
  `REWARD_CONFIGS` key.
- **CF gen trigger (PRODUCER, rollout phase):** fill `_force_inject_rollout` (`:1870`, currently
  `NotImplementedError` `:1895`) per §3.3–3.5, gated behind NEW flag `algorithm.sdc_counterfactual`.
  Wire it at the `generate_sequences` boundary like `_bci_generate_sequences` (`:1791-1792`). It writes
  `data.non_tensor_batch["cf_correct"]` (float32, length B) onto the main gen output BEFORE
  `sleep_replicas()` (`ray_trainer.py:1322`).
- **CONSUMER, `_populate_dcpo_region_keys` (`:117-207`):** this is the unique site with decoded main
  responses (`:147-159`), ground truths (`:160-163`), the full uid group + step, runs in the main process
  BEFORE the GDPO assertion/advantage. `main_correct` already exists as `_heads["R_corr"]` (`:196`). Read
  `cf_correct` from `data.non_tensor_batch` and compute `R_meta = main_correct − cf_correct` here (thread
  `cf_completions`/`cf_correct` into `_compute_dcpo_heads_stash` → `dcpo_region_rewards`). Do NOT trigger
  the CF generation from here — that is the busy-rollout deadlock `_generate_v0_prefixes` warns about
  (`:922-930`); the engine is asleep at this point.
- **Fail-safe:** if `sdc_counterfactual` is on but `cf_correct` is absent on the batch (producer
  short-circuited / all rows skipped), `_populate_dcpo_region_keys` falls back to
  `cf_answer_from_prefix` (text path) so the step never crashes.

### 5.3 Config + amlt yaml — REMOVE the two-pass INSTR parquet-rewrite

New yaml `h200_triobj_dcpo_v3.yaml` (copy of v2), with:

- **DELETE the entire "TWO-PASS PROMPT REWRITE" block** (`h200_triobj_dcpo_v2.yaml:60-82`): the `INSTR`
  string, `rewrite()`, and both `rewrite(...)` calls. Train directly on the plain redirect parquets
  (`verl_train_redirect.parquet` / `verl_val_redirect.parquet`) — the SFT/e8 prompt — so the model emits
  its natural single-pass meta (99.8% emit vs 65% with INSTR).
- Set `sdc_mode: TRIOBJ_DCPO_V3`, `algorithm.sdc_counterfactual: true`.
- KEEP: Dr.GRPO (KL/entropy off), GRPO group `n=4`, region-routing config, masking config.
- DROP from reward config: `format_penalty`, `format_credit`, `w_warmup`, `sandbag_*`, `p_lo`/`p_hi` as
  R_meta gate (leave as ignored if present).

---

## §6 Helpers summary

- `first_meta_index(resp_ids, response_mask, meta_open=151669)` — token cut point for the CF prefix
  (producer, §3.4 / §5.1-A).
- `cf_answer_from_prefix(text)` — text-level CF answer fallback grader (consumer fallback, §5.1-A).
- Reused, UNCHANGED: `_extract_answer_fallback` (`rewards.py:71`), `_check_correctness` (`rewards.py:27`),
  `_parse_confidence` (`rewards.py:701`), `correctness_reward` (`rewards.py:857`, pure decode grader, no
  GPU/group — usable to grade `cf_correct` in the producer).

---

## §7 Cost (~2× generation)

Inference doubles (8 vs 4 per prompt): +4 CF rollouts. CF rollouts are short on average — they start from
the first-meta prefix (typically deep into `<think>`), so the remaining continuation is usually just the
answer tail, not a full trace; expected CF token cost < main token cost. No extra training-graph cost (CF
is `no_grad`, never in GRPO, never backproped). No second engine/init (same replicas, still awake). Net:
roughly +50–100% wall-clock on the rollout phase, 0 on the optimizer phase. Acceptable.

---

## §8 Anti-hack (counterfactual is a policy-independent comparison)

- **Within-rollout, same policy, same prefix.** `c_with` and `c_without` share the identical prefix up to
  the first `<|meta|>`; the ONLY difference is whether meta is allowed to continue. The model cannot inflate
  `R_meta` by making the non-meta path artificially bad — the non-meta path IS its own continuation of its
  own prefix.
- **Kills the v2 "write a wrong draft then revise" exploit.** There is no separate draft answer to
  sandbag; the counterfactual is generated fresh from before the meta, so deliberately writing a wrong
  pre-meta answer would also corrupt `c_without` (and the prefix the CF continues), not hand free credit.
- **No teacher, no gold conditioning** in the CF path → no answer-leak channel (cf. the contrastive-teacher
  confound in MEMORY). `R_meta` is grounded purely in task correctness deltas.
- **Group-mean-subtract on R_meta** keeps the head zero-centered per prompt; a model that emits useless
  meta on every rollout in a group gets ~0 advantage, not a constant reward floor.
- **No-meta ⇒ 0, not penalty.** Choosing not to emit meta is never punished (delta 0), so the model is
  rewarded for *useful* meta, not for meta volume.

---

## §9 Test plan

**Pure-python unit (no GPU):**

1. `first_meta_index`: response with one/many/zero `<|meta|>` ids → returns first index / first index /
   `None`; respects `response_mask` (skips masked positions).
2. CF prefix cut: `prefix_ids == prompt_ids + resp_ids[:firstMeta]`, length matches, last prefix token ≠
   151669.
3. R_meta delta: synth `(c_with, c_without)` ∈ {(1,0),(0,1),(1,1),(0,0),(c_with, None)} → R_meta ∈
   {+1,−1,0,0,0}. Verify group-mean-subtract centers a 4-rollout group.
4. Region masks unchanged: on a real `<|meta|>` response, META_CONTENT excludes tag tokens; R_meta routes
   to META_CONTENT only; R_cal to CONF; R_corr to ANSWER (re-run v2's verified mask assertions).
5. `cf_answer_from_prefix`: response with meta → extracts pre-meta answer; no-meta → `None`.

**verl step-1 smoke (1 node, K=4, `sdc_counterfactual=true`):**

6. CF gen produces a no-meta answer: assert 0 occurrences of token 151669 in every `cf_out.responses`
   (logit_bias suppression works end-to-end across the Ray boundary).
7. `cf_correct` present on the batch, length B, ∈ {0.0,1.0}; no-meta rows short-circuited to R_meta 0.
8. `R_meta` is non-degenerate: `std(R_meta) > 0` over the step (log via `DCPO_DEBUG`); print
   `(c_with, c_without, R_meta)` for row 0 and the per-step `+1/−1/0` histogram.
9. Replicas-awake ordering: CF gen completes before `sleep_replicas()` (no wake/deadlock); step finishes;
   `sdc_counterfactual=false` is byte-identical to a v2 step (gating proof).

---

## §10 Risks

1. **Custom-prefix agent-loop repack vs live verl runtime** (the only MEDIUM item). Mitigation: BCI
   pattern is the template; §3.6 chat-message and separate-pass fallbacks are both available; node-smoke
   (test 6–9) gates enablement, exactly as the existing `_force_inject_rollout` guard demands
   (`verl_sdc.py:1893-1895`).
2. **`logit_bias` not absolute.** −100 is effectively a ban but soft. Mitigation: assert 0 occurrences in
   smoke test 6; escalate to a `logits_processors` `-inf` mask only if any leak observed.
3. **CF length blow-up / non-termination** (model wanders without meta). Mitigation: cap CF `max_tokens`
   (reuse main response budget); log CF close-rate + truncation rate; truncated CF still graded (extractor
   returns None → c_without None → R_meta 0, conservative).
4. **Sleep-ordering regression** if verl bumps and moves `sleep_replicas()`. Mitigation: pin the call at
   the `generate_sequences` wrapper boundary (replicas awake by construction), not at a `fit()` line
   number; smoke test 9 guards it.
5. **Prompt change (drop INSTR) shifts the data distribution** vs v2 runs. Intended (that is the point),
   but it makes v3 not directly comparable to v2 curves. Mitigation: note in run metadata; the e8/SFT
   plain-redirect prompt is the established baseline, and 99.8% meta-emit is the target behavior.
6. **Text-fallback divergence.** `cf_answer_from_prefix` (pre-meta extract) ≠ a real regenerated CF when
   the pre-meta prefix has no boxed answer yet (common: meta fires before any answer). Then fallback yields
   None → R_meta 0, *under-crediting* useful meta. Mitigation: the real producer path (§3) is the primary;
   fallback is only a crash-guard, and its conservative bias (0 not wrong-sign) is acceptable.

---

## Changelog

- ADD mode `TRIOBJ_DCPO_V3` to `_REGION_ROUTED_MODES` (`verl_sdc.py:598`) + new `REWARD_CONFIGS` key.
- ADD `algorithm.sdc_counterfactual` flag; fill `_force_inject_rollout` stub (`verl_sdc.py:1870-1895`)
  with: main-gen → first-meta prefix cut → 2nd `generate_sequences` with `logit_bias{151669:-100}` →
  grade → attach `cf_correct` (float32, len B) to `non_tensor_batch`, all before `sleep_replicas()`.
- ADD custom `cf_prefix_agent` agent-loop (token-id prefix ingest, BCI-pattern) OR fallback to
  chat-message prefix / separate CF pass (§3.6).
- ADD helpers `first_meta_index` + `cf_answer_from_prefix` to `dcpo_region.py` (~line 40).
- REWRITE `dcpo_region_rewards` R_meta: transition table → `R_meta = c_with − c_without`; add
  `cf_completions` arg; keep `R_corr`/`R_cal`; keep `p_hat`/`group_acc` + constant `canary`/`sandbag`
  stubs; DROP `w_warmup`/sandbag/transition/format logic; keep dropped kwargs as ignored.
- WIRE `cf_correct` consumption in `_populate_dcpo_region_keys` (`verl_sdc.py:117-207`): `R_meta =
  main_correct(_heads["R_corr"]) − cf_correct`; text fallback if absent. NO CF generation from this site
  (deadlock; `:922-930`).
- ADD yaml `h200_triobj_dcpo_v3.yaml`: DELETE two-pass INSTR parquet-rewrite (`v2.yaml:60-82`), train on
  plain redirect parquets; set `sdc_mode: TRIOBJ_DCPO_V3`, `sdc_counterfactual: true`; KEEP Dr.GRPO/region
  routing/masking; DROP format/warmup/sandbag reward kwargs.
- KEEP UNCHANGED: `build_dcpo_region_masks`, `compose_dcpo_region_advantage` (region routing verified);
  `rewards.py` helpers; all non-v3 modes (additive, gated).
- TESTS: pure-python units (cut/delta/masks/cf-extraction) + verl step-1 smoke (no-meta CF via token 0
  count, `std(R_meta)>0` via `DCPO_DEBUG`, replicas-awake ordering, off==byte-identical).
