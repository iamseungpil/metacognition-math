from __future__ import annotations

import os
import sys

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training import verl_sdc as verl_sdc_mod
from src.training.verl_sdc_utils import (
    build_sdc_region_masks,
    compute_sdc_gdpo_advantage,
    postmeta_closure_reward,
)


class DummyCharTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(ch) for ch in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_build_sdc_region_masks_preserves_boxed_wrapper():
    tok = DummyCharTokenizer()
    text = "<|meta|>m<|/meta|> The answer is \\boxed{42}."
    ids = tok.encode(text)
    masks = build_sdc_region_masks(tok, ids, text)
    shared = masks["postmeta_shared_mask"]
    diff = masks["postmeta_diff_mask"]

    assert shared.sum().item() > 0
    assert diff.sum().item() > 0
    boxed_idx = text.index("\\boxed{")
    # wrapper prefix should be shared
    assert shared[boxed_idx].item() == 1.0
    # numeric payload should remain differential
    payload_idx = text.index("4", boxed_idx)
    assert diff[payload_idx].item() == 1.0


def test_postmeta_closure_reward_prefers_clean_commit():
    good = [[{"content": "<|meta|>x<|/meta|> Therefore, \\boxed{42}"}]]
    bad = [[{"content": "<|meta|>x<|/meta|> therefore therefore therefore no boxed tail tail"}]]
    good_score = postmeta_closure_reward(good)[0]
    bad_score = postmeta_closure_reward(bad)[0]
    assert good_score > bad_score


def test_compute_sdc_gdpo_advantage_routes_regions():
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    batch = {
        "prompts": torch.zeros(2, 2, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=torch.long),
        "sdc_meta_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        "sdc_postmeta_shared_mask": torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "sdc_postmeta_diff_mask": torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        "sdc_body_mask": torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
        "old_log_probs": torch.tensor([[-2.0, -2.0, -2.0, -2.0], [-2.2, -2.2, -2.2, -2.2]]),
        "sdc_teacher_pos_log_probs": torch.tensor([[-1.0, -0.3, -0.1, -0.2], [-2.8, -2.4, -2.9, -2.6]]),
        "sdc_teacher_neg_log_probs": torch.tensor([[-1.5, -0.35, -2.2, -1.4], [-1.2, -1.1, -0.8, -1.0]]),
    }
    non_tensor_batch = {
        "uid": np.asarray(["g1", "g1"], dtype=object),
        "correctness": np.asarray([1.0, -1.0], dtype=np.float32),
        "outcome_calibration": np.asarray([0.5, -0.2], dtype=np.float32),
        "meta_structure": np.asarray([0.2, -0.1], dtype=np.float32),
        "meta_commit_shape": np.asarray([0.3, -0.3], dtype=np.float32),
        "postmeta_closure": np.asarray([0.4, -0.4], dtype=np.float32),
    }

    class Cfg(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    cfg = Cfg(
        gdpo_reward_keys=["correctness", "outcome_calibration", "meta_structure", "meta_commit_shape", "postmeta_closure"],
        gdpo_reward_weights=[1.0, 0.7, 0.25, 0.35, 0.45],
        sdc_clip_eps_w=0.2,
        sdc_log_ratio_clamp=10.0,
        sdc_shared_tau=0.5,
        sdc_lambda_meta=0.5,
        sdc_lambda_shared=0.25,
        sdc_lambda_diff=0.30,
    )
    token_level_rewards = torch.zeros_like(response_mask)
    adv, ret = compute_sdc_gdpo_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=non_tensor_batch["uid"],
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        config=cfg,
    )
    assert adv.shape == response_mask.shape
    assert ret.shape == response_mask.shape
    assert torch.all(torch.isfinite(adv))
    # token 1 is teacher-consensus shared structure for sample 0, so it should
    # not be penalized more harshly than the differential token 2.
    assert adv[0, 1].abs().item() <= adv[0, 2].abs().item() + 1e-6


def test_attach_teacher_signals_adds_token_logprobs():
    tok = DummyCharTokenizer()

    class FakeTrainer:
        def __init__(self):
            self.calls = 0

        def _compute_ref_log_prob(self, batch):
            self.calls += 1
            bsz = batch.batch["response_mask"].shape[0]
            resp_len = batch.batch["response_mask"].shape[1] - int(
                (batch.batch["attention_mask"][0] - batch.batch["response_mask"][0]).sum().item()
            )
            vals = torch.full((bsz, resp_len), -0.5 * self.calls, dtype=torch.float32)
            return DataProto.from_dict(tensors={"ref_log_prob": vals})

    old_ctx = dict(verl_sdc_mod._ACTIVE_SDC_CONTEXT)
    verl_sdc_mod._ACTIVE_SDC_CONTEXT["trainer"] = FakeTrainer()
    verl_sdc_mod._ACTIVE_SDC_CONTEXT["tokenizer"] = tok
    try:
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([[ord("Q"), ord("?")]], dtype=torch.long),
                "responses": torch.tensor([[ord("a"), ord("b"), ord("c"), ord("d")]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
                "response_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
            },
            non_tensors={
                "reward_model": np.asarray([{"ground_truth": "42"}], dtype=object),
            },
        )
        out = verl_sdc_mod._attach_teacher_signals(data)
        assert "sdc_teacher_pos_log_probs" in out.batch.keys()
        assert "sdc_teacher_neg_log_probs" in out.batch.keys()
        assert out.batch["sdc_teacher_pos_log_probs"].shape == out.batch["responses"].shape
        assert out.batch["sdc_teacher_neg_log_probs"].shape == out.batch["responses"].shape
    finally:
        verl_sdc_mod._ACTIVE_SDC_CONTEXT.clear()
        verl_sdc_mod._ACTIVE_SDC_CONTEXT.update(old_ctx)


def test_is_gdpo_estimator_handles_string_and_enum():
    assert verl_sdc_mod._is_gdpo_estimator("gdpo") is True
    assert verl_sdc_mod._is_gdpo_estimator(AdvantageEstimator.GDPO) is True
    assert verl_sdc_mod._is_gdpo_estimator("grpo") is False


def test_decode_response_returns_response_only_text():
    # Regression guard: _decode_response must never leak prompt content into
    # the string that reward heads see. If this breaks, correctness/outcome/meta
    # rewards can pattern-match on \boxed{} or <|meta|> examples that live inside
    # the problem prompt and silently hack themselves.
    tok = DummyCharTokenizer()
    prompt_text = "Q: worked example \\boxed{99}. <|meta|>hint<|/meta|> A:"
    response_text = "I think 7"
    prompt_ids = torch.tensor(tok.encode(prompt_text), dtype=torch.long)
    response_ids = torch.tensor(tok.encode(response_text), dtype=torch.long)
    prompt_length = prompt_ids.numel()
    attention_mask = torch.ones(prompt_length + response_ids.numel(), dtype=torch.long)

    text, valid_resp = verl_sdc_mod._decode_response(
        tok, prompt_ids, response_ids, attention_mask, prompt_length
    )
    assert text == response_text
    assert "\\boxed" not in text
    assert "<|meta|>" not in text
    assert valid_resp.numel() == response_ids.numel()


def test_correctness_reward_contrast_documents_contamination_fix():
    # Pairs with the fix in _decode_response: on response-only text the wrong
    # answer is correctly penalized, but on prompt+response concat the reward
    # flips to +1 purely because the prompt contains a gold \boxed{} example.
    # Keeping both assertions documents why response-only decode matters.
    from src.training.rewards import correctness_reward

    response_only = [[{"content": "Final answer: 7"}]]
    contaminated = [[{"content": "Question with \\boxed{42} example. Final: 7"}]]
    gt = ["42"]

    assert correctness_reward(response_only, ground_truth=gt)[0] == -1.0
    assert correctness_reward(contaminated, ground_truth=gt)[0] == 1.0


def test_build_sdc_region_masks_falls_back_on_degenerate_offsets():
    # When the tokenizer returns many (0,0) offsets — the failure mode for
    # added-vocab chat tokens like <|meta|> on some forks — build_sdc_region_masks
    # must trigger the same manual_offset_scan fallback that meta/postmeta
    # masks use. Before this fix, shared/diff collapsed while meta/postmeta
    # relocated, leaving the SDC pressure applied to the wrong tokens.
    from unittest.mock import patch

    tok = DummyCharTokenizer()
    text = "<|meta|>plan<|/meta|> Therefore, \\boxed{42}."
    ids = tok.encode(text)

    # Degenerate: only first token carries a real offset; rest report (0,0).
    degenerate = [(0, 1)] + [(0, 0)] * (len(ids) - 1)

    with patch(
        "src.training.verl_sdc_utils.assistant_offsets",
        return_value=([int(i) for i in ids], degenerate),
    ):
        masks = build_sdc_region_masks(tok, ids, text)

    shared = masks["postmeta_shared_mask"]
    diff = masks["postmeta_diff_mask"]
    assert shared.numel() == len(ids)
    # Fallback must recover real offsets — not all-zero, not collapsed at idx 0.
    assert shared[1:].sum().item() >= 1.0
    # And the boxed numeric payload must still land in the differential region.
    payload_idx = text.index("4", text.index("\\boxed{"))
    assert diff[payload_idx].item() == 1.0


def test_teacher_prompt_omits_synthetic_answer_separator():
    # The teacher batch must condition on exactly what the actor sees
    # ({prompt}{answer}) — not inject a synthetic " Answer: " that the policy
    # never generates, which would shift the conditional distribution.
    tok = DummyCharTokenizer()
    prompt_texts = ["Q: 1+1=? A: "]
    gold = ["2"]
    batch = verl_sdc_mod._build_teacher_logprob_batch(
        tokenizer=tok,
        prompt_texts=prompt_texts,
        answer_texts=gold,
        responses=torch.zeros(1, 1, dtype=torch.long),
        response_mask=torch.zeros(1, 1, dtype=torch.long),
    )
    attn = batch.batch["attention_mask"][0]
    valid_len = int(attn.sum().item())
    decoded = tok.decode(batch.batch["input_ids"][0][:valid_len].tolist())
    assert "Answer:" not in decoded
    assert decoded.startswith(prompt_texts[0])
    assert decoded.endswith(gold[0])


def test_reward_loop_score_handles_unrecognized_verl_data_source():
    out = verl_sdc_mod.reward_loop_score(
        data_source="hendrycks_math/algebra",
        solution_str="<|meta|>check<|/meta|> Therefore, \\boxed{42}",
        ground_truth="42",
    )
    assert isinstance(out, dict)
    assert "score" in out
    assert out["correctness"] == 1.0
    assert out["data_source"] == "hendrycks_math/algebra"
