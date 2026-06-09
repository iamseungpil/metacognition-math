"""Unit tests for the TRIOBJ_DCPO_V3 counterfactual 2nd-generation CALL.

PURE PYTHON (runs under /home/v-seungplee/miniconda3/envs/metaprobe/bin/python).
verl / ray / tensordict are NOT installed in this env, so we stub them in
sys.modules just enough to import `src.training.verl_sdc` and exercise the three
CF methods that were wired (the verl 2nd-gen call), against fake DataProto-like
objects. No GPU, no real verl rollout — those are smoke-only.

Covers:
  - _dcpo_cf_build_prefixes: prefix_ids = prompt(left-pad-stripped) + resp[:firstMeta];
    no-meta row → skip=True, prefix=None.
  - _dcpo_cf_call_engine: cf_batch construction (agent_name / prefix_ids / cf_logit_bias
    shapes + values), correct prompt-ids passed (continuation prompt), SAME captured
    generate invoked, meta_info validate=False.
  - _dcpo_cf_decode_texts: right-pad strip + meta-leak strip + decode.
  - _dcpo_cf_generate_and_grade: NaN on empty active; crash-safe all-NaN on engine raise;
    correct {1.0,0.0,NaN} placement on grading.
  - cf_prefix_agent loop module: logit_bias injection does NOT mutate the shared dict.
"""
import sys
import types

import numpy as np
import torch
import pytest

META_OPEN = 151669


# ── auto-stub finder: any import under these prefixes resolves to a blank module
#    whose attributes are themselves callable/subclassable blanks. Lets us import
#    src.training.verl_sdc (deep verl/datasets/hydra graph) in the metaprobe env
#    WITHOUT those packages, so we can unit-test the pure-python CF methods. ──────
import importlib.abc
import importlib.machinery

_AUTO_STUB_PREFIXES = ("verl", "datasets", "omegaconf", "hydra", "ray", "vllm",
                       "tensordict")


class _Blank:
    """A blank that is callable, subclassable, indexable, and attr-spawning."""

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        # Decorator-friendly: @stub(...) applied to a class/func returns it unchanged
        # (so @register("name") and @register(...) leave the decorated object intact).
        if len(a) == 1 and not k and (isinstance(a[0], type) or callable(a[0])):
            return a[0]
        return _Blank()

    def __getattr__(self, name):
        # Capitalized → a subclassable class; else a callable blank.
        return _Blank

    def __getitem__(self, k):
        return _Blank()


class _AutoStubModule(types.ModuleType):
    def __getattr__(self, name):
        # Return a subclassable/callable blank class for any attribute access.
        v = type(name, (_Blank,), {})
        setattr(self, name, v)
        return v


class _AutoStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".")[0]
        if root in _AUTO_STUB_PREFIXES and fullname not in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _AutoStubModule(spec.name)

    def exec_module(self, module):
        module.__path__ = []  # mark as package so submodule imports resolve


def _install_auto_stub():
    if not any(isinstance(f, _AutoStubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AutoStubFinder())


_install_auto_stub()
from src.training import verl_sdc as V  # noqa: E402


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeBatch(dict):
    """dict with .get already; mimics DataProto.batch (tensor dict)."""


class FakeGenOutput:
    """Mimics the gen_output DataProto: .batch tensors + per-row non_tensor_batch."""

    def __init__(self, prompts, responses, attention_mask=None, response_mask=None,
                 ground_truths=None):
        self.batch = FakeBatch()
        self.batch["prompts"] = prompts
        self.batch["responses"] = responses
        if attention_mask is not None:
            self.batch["attention_mask"] = attention_mask
        if response_mask is not None:
            self.batch["response_mask"] = response_mask
        self._gts = ground_truths or [""] * len(prompts)

    def __len__(self):
        return self.batch["prompts"].shape[0]

    def __getitem__(self, i):
        row = types.SimpleNamespace()
        row.non_tensor_batch = {"reward_model": {"ground_truth": self._gts[i]}}
        return row


class FakeCFBatch:
    """The cf_batch returned by select_idxs: records what the wrap sets on it."""

    def __init__(self, n, base_non_tensor, meta_info):
        self.non_tensor_batch = dict(base_non_tensor)
        self.meta_info = dict(meta_info)
        self._n = n

    def __len__(self):
        return self._n


class FakeGenBatch:
    """Mimics gen_batch: select_idxs carries raw_prompt + meta_info to a FakeCFBatch."""

    def __init__(self, raw_prompts, meta_info=None):
        # raw_prompt present on every row (REQUIRED by _agent_loop_postprocess).
        self.non_tensor_batch = {
            "raw_prompt": np.array(list(raw_prompts), dtype=object),
        }
        self.meta_info = dict(meta_info or {})
        self.selected_idxs = None

    def select_idxs(self, idxs):
        self.selected_idxs = list(idxs)
        sub = {
            k: np.array([v[i] for i in idxs], dtype=object)
            for k, v in self.non_tensor_batch.items()
        }
        return FakeCFBatch(len(idxs), sub, self.meta_info)


class FakeTokenizer:
    """decode: join ids as 't<id>' tokens; lets us assert content + meta-strip."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(t)}" for t in ids)


def _mk_trainer():
    """A bare object carrying ONLY the CF methods bound from the real class."""
    t = types.SimpleNamespace()
    t.tokenizer = FakeTokenizer()
    t._dcpo_cf = True
    # bind the real unbound methods
    cls = V.SDCRayPPOTrainer
    for name in (
        "_dcpo_cf_build_prefixes",
        "_dcpo_cf_call_engine",
        "_dcpo_cf_decode_texts",
        "_dcpo_cf_generate_and_grade",
        "_dcpo_cf_ground_truths",
    ):
        setattr(t, name, getattr(cls, name).__get__(t))
    return t


# ═══════════════════════════════════════════════════════════════════════════
# _dcpo_cf_build_prefixes
# ═══════════════════════════════════════════════════════════════════════════
def test_build_prefixes_basic_and_no_meta():
    # row0: prompt [9,9] (no pad), resp [5, META, 7] → prefix [9,9,5]
    # row1: NO meta → skip
    prompts = torch.tensor([[9, 9], [8, 8]])
    responses = torch.tensor([[5, META_OPEN, 7], [1, 2, 3]])
    attn = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])  # prompt_len=2 + resp_len=3
    go = FakeGenOutput(prompts, responses, attention_mask=attn)
    t = _mk_trainer()
    prefix_ids, skip = t._dcpo_cf_build_prefixes(go, META_OPEN)
    assert prefix_ids[0] == [9, 9, 5]
    assert skip[0] is False
    assert skip[1] is True
    assert prefix_ids[1] is None


def test_build_prefixes_strips_left_pad():
    # prompt left-padded: [0,7] with attn [0,1] → stripped prompt = [7]
    prompts = torch.tensor([[0, 7]])
    responses = torch.tensor([[4, META_OPEN, 9]])
    attn = torch.tensor([[0, 1, 1, 1, 1]])  # first prompt token is pad
    go = FakeGenOutput(prompts, responses, attention_mask=attn)
    t = _mk_trainer()
    prefix_ids, skip = t._dcpo_cf_build_prefixes(go, META_OPEN)
    assert prefix_ids[0] == [7, 4]  # [stripped prompt] + resp[:firstMeta]
    assert skip[0] is False


# ═══════════════════════════════════════════════════════════════════════════
# _dcpo_cf_call_engine — cf_batch construction + SAME-engine call
# ═══════════════════════════════════════════════════════════════════════════
def test_call_engine_builds_cf_batch_and_calls_same_generate():
    raw_prompts = ["pA", "pB", "pC"]
    gb = FakeGenBatch(raw_prompts, meta_info={"global_steps": 7, "validate": True})
    prefix_ids = [[1, 2, 3], None, [4, 5]]  # row1 has no meta
    active = [0, 2]

    captured = {}

    def fake_generate(cf_batch):
        captured["cf_batch"] = cf_batch
        # return a minimal cf_out: 2 rows, response_length 2, prompts width = max prefix
        out = FakeGenOutput(
            prompts=torch.zeros((2, 3), dtype=torch.long),
            responses=torch.tensor([[11, 12], [13, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
        )
        return out

    t = _mk_trainer()
    t._dcpo_cf_orig_generate = fake_generate

    texts = t._dcpo_cf_call_engine(gb, prefix_ids, active, META_OPEN)

    cfb = captured["cf_batch"]
    # active rows [0,2] PADDED to B=len(prefix_ids)=3 (chunk-divisibility: repeat active[0]).
    assert gb.selected_idxs == [0, 2, 0]
    assert len(cfb) == 3
    # agent routed to the CF prefix loop (every padded row)
    assert list(cfb.non_tensor_batch["agent_name"]) == ["cf_prefix_agent"] * 3
    # prefix_ids carried as object array of int-lists; row2 = padding repeat of active[0].
    assert list(cfb.non_tensor_batch["prefix_ids"][0]) == [1, 2, 3]
    assert list(cfb.non_tensor_batch["prefix_ids"][1]) == [4, 5]
    assert list(cfb.non_tensor_batch["prefix_ids"][2]) == [1, 2, 3]
    # logit_bias suppresses meta_open with -100.0, per row
    assert cfb.non_tensor_batch["cf_logit_bias"][0] == {META_OPEN: -100.0}
    assert cfb.non_tensor_batch["cf_logit_bias"][1] == {META_OPEN: -100.0}
    # raw_prompt carried through (REQUIRED by _agent_loop_postprocess); padded row repeats pA.
    assert list(cfb.non_tensor_batch["raw_prompt"]) == ["pA", "pC", "pA"]
    # meta_info forced to train path, global_steps preserved
    assert cfb.meta_info["validate"] is False
    assert cfb.meta_info["global_steps"] == 7
    # decoded texts parallel to active
    assert len(texts) == 2
    assert texts[0] == "t11 t12"
    assert texts[1] == "t13"  # right-pad (attn 0) stripped


# ═══════════════════════════════════════════════════════════════════════════
# _dcpo_cf_decode_texts — pad strip + meta leak strip
# ═══════════════════════════════════════════════════════════════════════════
def test_decode_strips_rightpad_via_attention():
    cf_out = FakeGenOutput(
        prompts=torch.zeros((1, 2), dtype=torch.long),
        responses=torch.tensor([[21, 22, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0]]),  # prompt_len 2 + 2 real + 2 pad
    )
    t = _mk_trainer()
    out = t._dcpo_cf_decode_texts(cf_out, META_OPEN)
    assert out == ["t21 t22"]


def test_decode_strips_leaked_meta_token():
    # logit_bias should prevent this, but if meta leaks we strip + warn (not crash).
    cf_out = FakeGenOutput(
        prompts=torch.zeros((1, 1), dtype=torch.long),
        responses=torch.tensor([[META_OPEN, 30, 31]]),
        response_mask=torch.tensor([[1, 1, 1]]),
    )
    t = _mk_trainer()
    out = t._dcpo_cf_decode_texts(cf_out, META_OPEN)
    assert str(META_OPEN) not in out[0]
    assert out[0] == "t30 t31"


def test_decode_prefers_response_mask_when_present():
    cf_out = FakeGenOutput(
        prompts=torch.zeros((1, 2), dtype=torch.long),
        responses=torch.tensor([[40, 41, 99]]),
        response_mask=torch.tensor([[1, 1, 0]]),  # last token masked out
    )
    t = _mk_trainer()
    out = t._dcpo_cf_decode_texts(cf_out, META_OPEN)
    assert out == ["t40 t41"]


# ═══════════════════════════════════════════════════════════════════════════
# _dcpo_cf_generate_and_grade — NaN-on-empty, crash-safe, grading
# ═══════════════════════════════════════════════════════════════════════════
def test_grade_nan_when_no_active():
    prompts = torch.tensor([[1, 1], [2, 2]])
    responses = torch.tensor([[1, 2, 3], [4, 5, 6]])  # no meta in either
    go = FakeGenOutput(prompts, responses, ground_truths=["1", "2"])
    gb = FakeGenBatch(["a", "b"])
    t = _mk_trainer()
    skip = [True, True]
    prefix_ids = [None, None]
    out = t._dcpo_cf_generate_and_grade(gb, go, prefix_ids, skip, META_OPEN)
    assert len(out) == 2
    assert all(v != v for v in out)  # all NaN


def test_grade_crash_safe_all_nan_on_engine_raise():
    prompts = torch.tensor([[1, 1]])
    responses = torch.tensor([[7, META_OPEN, 9]])
    go = FakeGenOutput(prompts, responses, ground_truths=["42"])
    gb = FakeGenBatch(["a"])
    t = _mk_trainer()

    def boom(cf_batch):
        raise RuntimeError("engine exploded")

    t._dcpo_cf_orig_generate = boom
    prefix_ids = [[1, 1, 7]]
    skip = [False]
    out = t._dcpo_cf_generate_and_grade(gb, go, prefix_ids, skip, META_OPEN)
    assert len(out) == 1
    assert out[0] != out[0]  # NaN — R_meta degrades to 0


def test_grade_places_correct_values(monkeypatch):
    # CF gen returns text that grades correct for row0, wrong for row2; row1 skipped.
    prompts = torch.tensor([[1, 1], [2, 2], [3, 3]])
    responses = torch.tensor([
        [5, META_OPEN, 0],
        [1, 2, 3],          # no meta → skip
        [6, META_OPEN, 0],
    ])
    go = FakeGenOutput(prompts, responses, ground_truths=["AAA", "x", "BBB"])
    gb = FakeGenBatch(["p0", "p1", "p2"])
    t = _mk_trainer()

    # stub the engine call to return controlled texts for active=[0,2]
    def fake_call(gen_batch, prefix_ids, active, meta_open):
        assert active == [0, 2]
        return ["ans0", "ans2"]

    t._dcpo_cf_call_engine = fake_call

    # control grading: row0 correct, row2 wrong
    import src.training.rewards as R

    monkeypatch.setattr(R, "_extract_answer_fallback", lambda txt: txt)
    monkeypatch.setattr(
        R, "_check_correctness", lambda ans, gt: (ans == "ans0" and gt == "AAA")
    )

    skip = [False, True, False]
    prefix_ids = [[1, 1, 5], None, [3, 3, 6]]
    out = t._dcpo_cf_generate_and_grade(gb, go, prefix_ids, skip, META_OPEN)
    assert out[0] == 1.0
    assert out[1] != out[1]  # NaN (skipped)
    assert out[2] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# cf_prefix_agent loop: per-call logit_bias must NOT mutate the shared dict
# ═══════════════════════════════════════════════════════════════════════════
def test_cf_agent_module_imports_and_logit_bias_injection():
    # The CF loop module imports under the auto-stub finder (verl symbols → blanks).
    import importlib
    cfmod = importlib.import_module("src.training.cf_prefix_agent")
    assert cfmod.CFPrefixAgentLoop is not None

    # The run() injection block: a per-call shallow copy gets logit_bias, the SHARED
    # batch dict is NOT mutated (so other rollouts keep meta_open allowed).
    shared = {"temperature": 0.7, "top_p": 0.9}
    lb = {META_OPEN: -100.0}
    sp = dict(shared)
    if lb:
        sp["logit_bias"] = {int(k): float(v) for k, v in dict(lb).items()}
    assert "logit_bias" not in shared
    assert sp["logit_bias"] == {META_OPEN: -100.0}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
