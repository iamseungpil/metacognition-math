# s3b Meta-Format Priming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix meta forming (open/close pairing) via format SFT-priming + token-embedding transplant + a best-effort decoding guard, keep PMI RL, so the meta channel survives RL instead of plain-ifying.

**Architecture:** Three small, independent code pieces (bug fix, embedding-transplant helper, meta-template rebuilder) + a short format-SFT pass + an optional decoding logits-processor flag + a widened auto-correction, then an RL stage that inherits s3 (data/composition/PMI) but starts from the primed checkpoint. SFT/decoding handle *form*; PMI handles *usefulness*. We never touch the s3 data, composition, or PMI reward.

**Tech Stack:** Python, transformers (SFT, embeddings), pandas (data rebuild), verl/vLLM (RL rollout), pytest (`/home/v-seungplee/miniconda3/envs/metaprobe/bin/python`), amlt H100.

**Spec:** docs/superpowers/specs/2026-06-15-s3b-meta-format-priming-design.md

**Scope note:** two phases — **Priming** (Tasks 1–6: code + short SFT) and **RL** (Tasks 7–9). Each phase is independently testable; the RL phase only launches after the primed checkpoint is verified (Task 6).

---

### Task 1: Fix signal.alarm grading bug (§3.5, isolated, do first)

**Files:**
- Modify: `src/training/rewards.py:42-56` (`_check_correctness`)
- Test: `tests/test_rewards_timeout.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rewards_timeout.py
from src.training import rewards
def test_check_correctness_no_signal_alarm_typeerror(monkeypatch):
    """Even off the main thread / with a math_verify that calls signal.alarm,
    grading must not raise and must grade a correct numeric answer True."""
    # correct numeric answer should grade True without TypeError flood
    assert rewards._check_correctness("\\boxed{42}", "42") is True
    assert rewards._check_correctness("\\boxed{1/2}", "0.5") is True
    # wrong stays False
    assert rewards._check_correctness("\\boxed{7}", "42") is False
```

- [ ] **Step 2: Run test to verify current state**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_rewards_timeout.py -v`
Expected: may PASS on metaprobe (no SIGALRM issue locally) — the fix is for the NODE's math_verify. So ALSO assert the call passes a *positive* timeout, not None (the real fix). Add:
```python
def test_verify_called_with_positive_timeout(monkeypatch):
    calls={}
    import src.training.rewards as R
    if not R.HAS_MATH_VERIFY: return
    def fake_verify(g,p,timeout_seconds=None): calls["t"]=timeout_seconds; return True
    monkeypatch.setattr(R, "verify", fake_verify)
    R._check_correctness("\\boxed{42}", "42")
    assert calls["t"] is not None and calls["t"] > 0   # FAILS now (None)
```
Expected: FAIL on the second test (timeout is None).

- [ ] **Step 3: Apply the fix**

In `_check_correctness`, change the verify call from `timeout_seconds=None` to a positive timeout, guarded so a SIGALRM-in-thread failure still falls back:
```python
            # math_verify's SIGALRM only works in the main thread; in Ray worker
            # threads signal.alarm(None) raised TypeError -> log flood + fallback.
            # Pass a positive timeout (math_verify guards thread-safety internally
            # in current versions); the try/except still rescues any failure.
            gold_parsed = parse(str(gold), extraction_mode="first_match", parsing_timeout=5)
            pred_parsed = parse(str(pred_text), extraction_mode="first_match", parsing_timeout=5)
            if bool(verify(gold_parsed, pred_parsed, timeout_seconds=5)):
                return True
```

- [ ] **Step 4: Run tests to verify pass**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_rewards_timeout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/rewards.py tests/test_rewards_timeout.py
git commit -m "fix(rewards): positive math_verify timeout (kill signal.alarm(None) flood)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Meta-token embedding transplant helper (§3.1b)

**Files:**
- Create: `src/training/meta_token_init.py`
- Test: `tests/test_meta_token_init.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meta_token_init.py
import torch
from src.training.meta_token_init import transplant_meta_embeddings_from_think

class _FakeEmb:
    def __init__(self,n,d): self.weight=torch.nn.Parameter(torch.randn(n,d))
class _FakeModel:
    def __init__(self,n,d): self._e=_FakeEmb(n,d)
    def get_input_embeddings(self): return self._e
    def get_output_embeddings(self): return self._e
class _FakeTok:
    def __init__(self,m): self.m=m
    def convert_tokens_to_ids(self,t): return self.m[t]

def test_meta_rows_become_think_rows():
    tok=_FakeTok({"<think>":10,"</think>":11,"<|meta|>":12,"<|/meta|>":13})
    model=_FakeModel(20,4)
    transplant_meta_embeddings_from_think(model, tok)
    w=model.get_input_embeddings().weight
    assert torch.allclose(w[12], w[10])   # <|meta|> <- <think>
    assert torch.allclose(w[13], w[11])   # <|/meta|> <- </think>
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_meta_token_init.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

```python
# src/training/meta_token_init.py
"""Transplant the native think-token embeddings into the added meta tokens so the
added (zero-prior) <|meta|>/<|/meta|> inherit the strong open/close pairing prior of
<think>/</think> (spec 2026-06-15-s3b §3.1b). Call AFTER resize_token_embeddings."""
import torch

def transplant_meta_embeddings_from_think(model, tokenizer,
        pairs=(("<|meta|>","<think>"),("<|/meta|>","</think>"))):
    def _id(t):
        i=tokenizer.convert_tokens_to_ids(t)
        if i is None or i<0: raise ValueError(f"token {t!r} not in tokenizer")
        return i
    with torch.no_grad():
        for emb in {id(model.get_input_embeddings()): model.get_input_embeddings(),
                    id(model.get_output_embeddings()): model.get_output_embeddings()}.values():
            if emb is None: continue
            W=emb.weight
            for meta_t, think_t in pairs:
                W[_id(meta_t)] = W[_id(think_t)].clone()
    return model
```

- [ ] **Step 4: Run to verify pass**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_meta_token_init.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/meta_token_init.py tests/test_meta_token_init.py
git commit -m "feat(sft): meta-token embedding transplant from think tokens

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Short fixed-template meta rebuilder (§3.1a)

**Files:**
- Create: `src/training/meta_template.py`
- Test: `tests/test_meta_template.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meta_template.py
from src.training.meta_template import rebuild_meta_block

def test_rebuild_extracts_labels_and_caps():
    raw=("confidence: 0.22\nThe current route is weak because ...\n"
         "assessment: boundary tracing needed\naction: switch to boundary view\n"
         "study_need: composite regions\n")
    out=rebuild_meta_block(raw, max_chars=200)
    # fixed order, only known labels, capped
    assert out.startswith("confidence: 0.22")
    assert "assessment:" in out and "action:" in out
    assert "study_need:" not in out   # not in the fixed 3-line template
    assert len(out) <= 200

def test_missing_label_skipped():
    out=rebuild_meta_block("confidence: 0.5\naction: verify the boundary\n", max_chars=200)
    assert out.startswith("confidence: 0.5")
    assert "action:" in out
    assert "assessment:" not in out   # absent -> skipped, not fabricated
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_meta_template.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement (fixed 3-line template, longest label coverage)**

```python
# src/training/meta_template.py
"""Rebuild a variable-length meta block into a short FIXED-ORDER template so the
closing position becomes predictable (spec 2026-06-15-s3b §3.1a). Keeps only the
three most common labels in a fixed order; each value is single-line, trimmed; the
whole block is char-capped. Labels absent in the source are skipped (never fabricated)."""
import re

_TEMPLATE_LABELS = ("confidence", "assessment", "action")  # observed coverage 0.89/0.65/0.47

def rebuild_meta_block(raw_body: str, max_chars: int = 320) -> str:
    lines = {}
    for lab in _TEMPLATE_LABELS:
        m = re.search(rf"(?im)^\s*{lab}\s*:\s*(.+)$", raw_body or "")
        if m:
            val = re.sub(r"\s+", " ", m.group(1)).strip()
            if val:
                lines[lab] = val
    out = "\n".join(f"{lab}: {lines[lab]}" for lab in _TEMPLATE_LABELS if lab in lines)
    return out[:max_chars].rstrip()
```
(`max_chars` 320 ≈ ~80 tokens; tune in Task 4.)

- [ ] **Step 4: Run to verify pass**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_meta_template.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/training/meta_template.py tests/test_meta_template.py
git commit -m "feat(sft): fixed-template meta-block rebuilder (predictable close)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Build the templated SFT parquet

**Files:**
- Create: `scripts/build_meta_template_sft.py`
- Output: `data/v8_meta_template_sft.parquet`

- [ ] **Step 1: Write the build script**

```python
# scripts/build_meta_template_sft.py
"""Rewrite v8_meta_inside_think assistant responses: replace each <|meta|>…<|/meta|>
body with the fixed short template (meta_template.rebuild_meta_block). Everything
outside the meta block is byte-identical. Output a new SFT parquet."""
import json, re, argparse, pandas as pd
from src.training.meta_template import rebuild_meta_block

_META = re.compile(r"<\|meta\|>(.*?)<\|/meta\|>", re.S)

def _rewrite(content, max_chars):
    def repl(m):
        body = rebuild_meta_block(m.group(1), max_chars=max_chars)
        return f"<|meta|>\n{body}\n<|/meta|>"
    return _META.sub(repl, content)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_path", default="data/v8_meta_inside_think.parquet")
    ap.add_argument("--out_path", default="data/v8_meta_template_sft.parquet")
    ap.add_argument("--max_chars", type=int, default=320)
    a=ap.parse_args()
    df=pd.read_parquet(a.in_path); rows=[]
    for _,r in df.iterrows():
        msgs=json.loads(r["messages"]) if isinstance(r["messages"],str) else r["messages"]
        for x in msgs:
            if isinstance(x,dict) and x.get("role")=="assistant":
                x["content"]=_rewrite(x.get("content",""), a.max_chars)
        rr=dict(r); rr["messages"]=json.dumps(msgs); rows.append(rr)
    out=pd.DataFrame(rows); out.to_parquet(a.out_path, index=False)
    print(f"wrote {len(out)} rows -> {a.out_path}")

if __name__=="__main__": main()
```

- [ ] **Step 2: Build + verify**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m scripts.build_meta_template_sft`
Then verify meta blocks are now short + paired:
```bash
/home/v-seungplee/miniconda3/envs/metaprobe/bin/python - <<'PY'
import pandas as pd,json,re,numpy as np
df=pd.read_parquet("data/v8_meta_template_sft.parquet")
def asst(m):
    for x in (json.loads(m) if isinstance(m,str) else m):
        if x.get("role")=="assistant": return x["content"]
    return ""
ls=[]
for i in range(min(500,len(df))):
    a=asst(df.iloc[i]["messages"]); m=re.search(r"<\|meta\|>(.*?)<\|/meta\|>",a,re.S)
    if m: ls.append(len(m.group(1)))
import numpy as np
print("rows",len(df),"meta blocks",len(ls),"mean_chars",round(np.mean(ls),1),"p90",round(np.percentile(ls,90),1),"CV",round(np.std(ls)/np.mean(ls),2))
assert np.mean(ls) < 350 and np.std(ls)/np.mean(ls) < 0.45   # short + low variability
PY
```
Expected: mean chars < 350, CV < 0.45 (was 0.69).

- [ ] **Step 3: Commit the script**

```bash
git add scripts/build_meta_template_sft.py
git commit -m "feat(sft): build templated meta SFT parquet

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Format-SFT wiring (embedding transplant + templated data)

**Files:**
- Modify: `src/training/sft.py:374` (after resize_token_embeddings)
- Create: node yaml `h100std_s3b_format_sft.yaml`

- [ ] **Step 1: Wire the transplant into sft.py behind a flag**

After `model.resize_token_embeddings(len(tokenizer))` (line 374), add:
```python
        import os as _os
        if _os.environ.get("S3B_META_EMB_TRANSPLANT", "0") == "1":
            from src.training.meta_token_init import transplant_meta_embeddings_from_think
            transplant_meta_embeddings_from_think(model, tokenizer)
            print("[s3b] transplanted meta-token embeddings from think tokens")
```
(Flag-gated → byte-identical when off. Verify `S3B_META_EMB_TRANSPLANT` only flips this.)

- [ ] **Step 2: Create the format-SFT node yaml**

`h100std_s3b_format_sft.yaml`: clone the existing v8_strict SFT node yaml; set `S3B_META_EMB_TRANSPLANT=1`, train_data=`data/v8_meta_template_sft.parquet`, init from `v8_meta_inside_strict` checkpoint, **low LR (e.g. 5e-6), 1 epoch**, output → HF `iamseungpil/metacot` models/`v8_s3b_format_primed`. (Mirror the staging/push pattern of the existing SFT yaml.)

- [ ] **Step 3: Validate yaml + sft.py import**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -c "import ast; ast.parse(open('src/training/sft.py').read()); import yaml; yaml.safe_load(open('h100std_s3b_format_sft.yaml')); print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/training/sft.py h100std_s3b_format_sft.yaml
git commit -m "feat(sft): s3b format-SFT (embedding transplant flag + templated data yaml)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: (RUN, human-gated) format-SFT + verify priming worked

- [ ] **Step 1: Launch format-SFT** (operator, after code review): `amlt run h100std_s3b_format_sft.yaml s3b-format-sft -y`
- [ ] **Step 2: Verify the primed checkpoint** — sample ~64 rollouts from the primed ckpt and check `wellformed_rate` jumped vs s3 step-1 (0.22). **Gate:** wellformed > 0.5 on the primed ckpt before spending the RL node. If not, iterate the template/transplant (do NOT launch RL).

(This task is a run+gate, not code. The implementing agent stops after Task 5/7/8 code is ready and reports; the operator runs Task 6.)

---

### Task 7: Constrained-decoding logits-processor (§3.2, flag, best-effort)

**Files:**
- Create: `src/training/meta_close_processor.py`
- Modify: `src/training/verl_sdc.py` (rollout sampling_params build — inject when flag on)
- Test: `tests/test_meta_close_processor.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_meta_close_processor.py
import torch
from src.training.meta_close_processor import MetaCloseLogitsProcessor

def test_forces_close_after_budget():
    p=MetaCloseLogitsProcessor(meta_open=12, meta_close=13, max_open_tokens=3)
    V=20
    # before any open: no-op
    lg=torch.zeros(V); out=p([1,2,3], lg.clone()); assert torch.allclose(out,lg)
    # after open, within budget: forbid a 2nd open (id12 -> -inf), close not forced yet
    p2=MetaCloseLogitsProcessor(12,13,max_open_tokens=3)
    o=p2([12,5], torch.zeros(V)); assert o[12]==float("-inf")
    # at budget: force close (only id13 finite)
    p3=MetaCloseLogitsProcessor(12,13,max_open_tokens=2)
    o=p3([12,5,6], torch.zeros(V))
    assert o[13]>-1e30 and o[0]==float("-inf")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_meta_close_processor.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement (vLLM-style logits_processor: callable(token_ids, logits)->logits)**

```python
# src/training/meta_close_processor.py
"""vLLM logits_processor that bounds the meta block: after <|meta|>, forbid a 2nd
open, and once `max_open_tokens` have passed without a close, force <|/meta|>.
Stateless across the (token_ids, logits) call contract (reconstructs state from
token_ids each step) so it is picklable for Ray workers (spec §3.2 best-effort)."""
import torch

class MetaCloseLogitsProcessor:
    def __init__(self, meta_open:int, meta_close:int, max_open_tokens:int=96):
        self.o=meta_open; self.c=meta_close; self.maxn=max_open_tokens
    def __call__(self, token_ids, logits):
        # find last unmatched open
        depth=0; since=None
        for k,t in enumerate(token_ids):
            if t==self.o: depth+=1; since=0
            elif t==self.c and depth>0: depth-=1; since=None
            elif since is not None: since+=1
        if depth<=0 or since is None:
            return logits
        if since>=self.maxn:                       # force close
            mask=torch.full_like(logits, float("-inf")); mask[self.c]=logits[self.c]
            return mask
        logits[self.o]=float("-inf")               # within budget: forbid 2nd open
        return logits
```

- [ ] **Step 4: Wire behind a flag in verl_sdc rollout sampling_params**

Where the rollout `sampling_params` dict is built (same area that splats `cf_logit_bias`/`logit_bias` into `SamplingParams(**sampling_params)`), add when `dcpo_meta_close_force` env/config is set:
```python
        import os as _os
        if _os.environ.get("DCPO_META_CLOSE_FORCE", "0") == "1":
            from src.training.meta_close_processor import MetaCloseLogitsProcessor
            sampling_params.setdefault("logits_processors", []).append(
                MetaCloseLogitsProcessor(meta_open=151669, meta_close=151670,
                                         max_open_tokens=int(_os.environ.get("DCPO_META_CLOSE_N","96"))))
```
(BEST-EFFORT: if verl/vLLM rejects `logits_processors` in sampling_params or Ray can't pickle it, the flag stays OFF and s3b relies on Tasks 1–6. Document this in the config comment. Default OFF → byte-identical.)

- [ ] **Step 5: Run tests + commit**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_meta_close_processor.py -v`
Expected: PASS.
```bash
git add src/training/meta_close_processor.py tests/test_meta_close_processor.py src/training/verl_sdc.py
git commit -m "feat(dcpo): best-effort meta-close logits processor (flag, default off)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Widen tier-1 auto-correction to recover some discard (§3.4)

**Files:**
- Modify: `src/training/dcpo_region.py` (`classify_dcpo_format` discard branch, rule 8 ~line 343)
- Test: `tests/test_dcpo_region.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# in tests/test_dcpo_region.py
def test_multi_open_recovers_first_pair_when_flag_on():
    from src.training.dcpo_region import classify_dcpo_format
    # ids: <|meta|> sig <|/meta|> ... <|meta|> stray   (a stray 2nd open after a valid pair)
    O,C=151669,151670
    ids=[O,1,2,C,9,9,O,3]   # first pair valid, trailing stray open
    rm=[True]*len(ids)
    dec=lambda xs: "confidence: 0.5"   # signature present
    out=classify_dcpo_format(ids, rm, dec, recover_first_pair=True)
    assert out["fmt_class"] in ("wellformed","dup_open","swapped","reversed")  # recovered, not discard
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_region.py::test_multi_open_recovers_first_pair_when_flag_on -v`
Expected: FAIL — unexpected kwarg `recover_first_pair` (or classifies discard).

- [ ] **Step 3: Implement — add `recover_first_pair` param; in the final discard branch, if a valid first open→close pair exists, re-run the classifier on the truncated prefix**

Add `recover_first_pair: bool = False` to `classify_dcpo_format` signature. Just before the final `return _discard()` (rule 8), add:
```python
    if recover_first_pair and len(O) >= 1 and len(C) >= 1:
        first_o = O[0]; first_c = min(c for c in C if c > first_o) if any(c > first_o for c in C) else None
        if first_c is not None:
            # keep only the first valid pair span; reclassify that prefix
            sub_ids = ids[:first_c+1]
            sub_rm = rmask[:first_c+1]
            return classify_dcpo_format(sub_ids, sub_rm, decode_fn,
                meta_open=meta_open, meta_close=meta_close, think_close=think_close,
                tier1_to_discard=tier1_to_discard, _validate_plan=_validate_plan,
                recover_first_pair=False)
    return _discard()
```
Thread `recover_first_pair` from the populator via a config knob `dcpo_recover_first_pair` (default False → byte-identical).

- [ ] **Step 4: Run tests (new + full dcpo suite) + commit**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_region.py tests/test_dcpo_anchor_emit.py -q`
Expected: PASS (new test + no regressions).
```bash
git add src/training/dcpo_region.py tests/test_dcpo_region.py
git commit -m "feat(dcpo): widen auto-correction to recover first valid meta pair (flag)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: s3b RL config + node yaml + release

**Files:**
- Create: `configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml`
- Create: `h100std_triobj_dcpo_v4_s3b.yaml`

- [ ] **Step 1: RL config (clone s3, init from primed ckpt, enable flags)**

Copy `configs/triobj_dcpo_v4_stage3_h100_4x4k.yaml` → stage3b. Keep ALL s3 knobs (data=meta_mix, composition anchor/emit/len_cost/trunc, PMI). Add:
```yaml
  dcpo_recover_first_pair: true      # Task 8 widened auto-correction
trainer:
  project_name: metacot-dcpo-v4
  experiment_name: triobj_dcpo_v4_stage3b_h100_4x4k
  default_local_dir: /scratch/checkpoints/triobj_dcpo_v4_stage3b_h100_4x4k
```

- [ ] **Step 2: Node yaml (init = primed ckpt, optional decoding flag)**

Clone `h100std_triobj_dcpo_v4_s3.yaml` → s3b. Change stage3→stage3b, WANDB_NAME=dcpo_v4_s3b, `actor_rollout_ref.model.path` = the `v8_s3b_format_primed` checkpoint (staged like the SFT), env `DCPO_META_CLOSE_FORCE=1` `DCPO_META_CLOSE_N=96` (decoding assist; if it errors at runtime, operator sets it to 0 and reruns — s3b still valid). `CODE_TAR_REVISION` set after release.

- [ ] **Step 3: Validate both yamls**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -c "import yaml; yaml.safe_load(open('configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml')); yaml.safe_load(open('h100std_triobj_dcpo_v4_s3b.yaml')); print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Full regression suite**

Run: `/home/v-seungplee/miniconda3/envs/metaprobe/bin/python -m pytest tests/test_dcpo_region.py tests/test_dcpo_anchor_emit.py tests/test_verl_meta_subset.py tests/test_rewards_timeout.py tests/test_meta_token_init.py tests/test_meta_template.py tests/test_meta_close_processor.py -q`
Expected: all pass.

- [ ] **Step 5: Build release asset + set CODE_TAR_REVISION** (mirror the s3 REST sequence: git archive → token scan → POST releases → POST asset → set id in s3b node yaml). Commit configs + yamls.

```bash
git add configs/triobj_dcpo_v4_stage3b_h100_4x4k.yaml h100std_triobj_dcpo_v4_s3b.yaml
git commit -m "feat(dcpo): s3b RL config + node yaml (primed init, recover-pair, decode flag)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review notes

- **Spec coverage:** §3.1a → Tasks 3,4; §3.1b → Task 2,5; §3.1c → Task 5,6; §3.2 → Task 7; §3.3 (PMI kept) → Task 9 inherits s3; §3.4 → Task 8; §3.5 → Task 1. HF1-4 hypotheses verified at Task 6 (priming) + post-RL eval.
- **Human gates:** Task 6 (verify priming before RL node) and Task 9 Step 5 (release/launch) are operator-run; the ultracode agent produces all CODE (Tasks 1-5,7-9 code) + leaves runs for me to verify/launch.
- **Default-off:** every new knob (S3B_META_EMB_TRANSPLANT, DCPO_META_CLOSE_FORCE, dcpo_recover_first_pair) defaults off → existing runs byte-identical.
- **Deferred:** redirect Harvest/Prime merge; difficulty-stratified eval; native verl guided decoding if it lands.
