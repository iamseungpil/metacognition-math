"""GROUP-BRANCH COUNTERFACTUAL without-arm via PLACEBO-META prefix (design 2026-06-22).

WHY THIS EXISTS (the bug it fixes — confirmed in live logs rv-cfgroup-2):
The cf_group counterfactual splits each GRPO group into with-meta / without-meta
sub-arms; the answer-delta (correct_with - correct_without) is R_meta. The
without-arm was produced by BANNING the meta-open (151669) + meta-close (151670)
tokens (cf_groupban_agent). But the warm-up SFT init (v8_rv_confidence_warmup) is
hardwired to emit '<|meta|>' immediately after '<think>'. Banning the open token
leaves the model with NO learned continuation -> it emits '<think>\\n\\n</think>'
(EMPTY reasoning) and ends. Observed: without-arm rows ans2='', response_length
min=5, R_corr=-1 (all wrong). acc_without ~= 0 makes Δ = acc_with - ~0 ~= acc_with,
so the reward DEGENERATES to "reward correctness / always emit meta" — the exact
generic-verify collapse cf_group was meant to cure. The counterfactual is INVALID.

THE FIX (PLACEBO-META prefix-forcing): instead of BANNING meta, FORCE a contentless
PLACEBO meta block as the START of the without-arm RESPONSE, then let the model
generate the solving continuation ON-DISTRIBUTION. The model sees a complete (but
contentless) meta block, proceeds to solve normally -> acc_without becomes a FAIR
"solve with no genuine metacognition" baseline. The genuine metacognition (real
verify/redirect content) is what the with-arm has and the without-arm lacks -> Δ
now measures genuine-meta usefulness.

DESIGN POINTS (verified against the codebase):
  - SFT response format (scripts/build_v8_strict_paired_data.py:294,318; sft.py:90)
    is `<think>\\n{think}\\n</think>\\n\\nThe answer is $...$.` with meta blocks
    INSIDE <think>. So the on-distribution contentless opener is
    `'<think>\\n' + PLACEBO_META + '\\n'` = `'<think>\\n<|meta|>\\nLet me continue.\\n<|/meta|>\\n'`.
  - MASKING: single_turn (single_turn_agent_loop.py:63) masks [1]*len over its OWN
    generated '<think>\\n<|meta|>...' opener as RESPONSE. So the placebo opener MUST
    be in the RESPONSE (response_mask=1), NOT the prompt — keeps both arms trained
    from <think> onward (clean counterfactual). This is the OPPOSITE of
    cf_prefix_agent (which puts the forced prefix in prompt_ids because the V3 cf
    path is inference-only / never trained).
  - SCORED GRPO MEMBER: unlike the V3 cf path, without-arm rows are real scored
    group members. response_logprobs=None (verl actor recomputes old_log_prob over
    the FULL placebo+continuation response in a separate forward pass — same pattern
    cf_prefix_agent.py:99 relies on; agent_loop.py:628-631 only injects
    rollout_log_probs when response_logprobs is not None).
  - NO logit_bias on the placebo arm (the placebo block is self-closed).

Registered via @register AND via configs/cf_agents_combined.yaml so the Ray rollout
workers resolve it by `_target_`. ADDITIVE: the default single_turn_agent loop is
untouched (with-arm rows keep it).

HONEST LIMIT: the DEFINITIVE check — that forcing the placebo actually makes the
without-arm SOLVE (non-empty, acc_without>0) on the live SFT model — is NOT
unit-testable (needs verl + vLLM + the real model on a GPU node). Both prior
cf_group bugs (stash loss, degenerate ban) PASSED unit tests and only surfaced
live. The helpers below (placebo_opener_str / build_placebo_output) are the
unit-testable contract; the live behaviour requires a 1-step NODE SMOKE.
"""
from __future__ import annotations

# ── PURE helpers (no verl/vLLM deps) — importable under the metaprobe test env ──
# PLACEBO_META is numpy-only (no verl/torch), so importing it here is test-safe.
from src.training.dcpo_pmi import PLACEBO_META


def placebo_opener_str() -> str:
    """The exact contentless on-distribution opener forced as the without-arm
    response prefix: '<think>\\n' + PLACEBO_META + '\\n'.

    Mirrors the SFT response format (meta block INSIDE <think>) so the model
    continues solving on-distribution after the closed placebo block.
    """
    return "<think>\n" + PLACEBO_META + "\n"


def build_placebo_output(
    prompt_ids: list[int],
    placebo_ids: list[int],
    gen_ids: list[int],
    response_length: int,
) -> dict:
    """Pure AgentLoopOutput-field contract for the placebo without-arm (the
    GRPO-member split). UNIT-TESTABLE without verl/DataProto/vLLM.

      prompt_ids        : UNCHANGED — the placebo is NOT in the prompt (prompt
                          parity with the with-arm single_turn so the only
                          with/without difference is genuine vs placebo meta).
      response_ids      : (placebo_ids + gen_ids)[:response_length] — the placebo
                          IS the trained response prefix, then the continuation.
      response_mask     : [1]*len(response_ids) — all-1 over placebo + continuation
                          (matches single_turn masking its OWN <think><|meta|> opener).
      response_logprobs : None — verl actor recomputes old_log_prob over the full
                          response (forced placebo tokens have no sampling logprobs).
    """
    response_ids = (list(placebo_ids) + list(gen_ids))[:response_length]
    return {
        "prompt_ids": list(prompt_ids),
        "response_ids": response_ids,
        "response_mask": [1] * len(response_ids),
        "response_logprobs": None,
    }


# ── verl-dependent agent loop (guarded so the pure helpers import without verl) ──
try:  # pragma: no cover — exercised only on the verl/vLLM node, not in metaprobe
    import logging
    import os
    from typing import Any
    from uuid import uuid4

    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
    from verl.utils.profiler import simple_timer
    from verl.workers.rollout.replica import TokenOutput

    _logger = logging.getLogger(__file__)
    _logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

    @register("cf_placebo_agent")
    class CFPlaceboAgentLoop(SingleTurnAgentLoop):
        """SingleTurn-derived loop that FORCES a contentless placebo meta block as
        the START of the TRAINED response, then lets the model solve on-distribution
        (cf_group without-meta sub-arm, PLACEBO mode).

        Builds prompt_ids IDENTICALLY to the with-arm single_turn (await
        apply_chat_template) so the counterfactual prompt parity holds, then forces
        the placebo opener tokens as the response prefix (response_mask=1, TRAINED)
        — the opposite of cf_prefix_agent's inference-only prompt-prefix.
        """

        async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
            messages = list(kwargs["raw_prompt"])

            # 1. images/videos + chat template — IDENTICAL to with-arm single_turn
            #    (prompt parity -> clean counterfactual; the only difference is the
            #    forced placebo meta vs the with-arm's genuine meta).
            multi_modal_data = await self.process_vision_info(messages)
            images = multi_modal_data.get("images")
            videos = multi_modal_data.get("videos")
            prompt_ids = await self.apply_chat_template(
                messages, images=images, videos=videos
            )

            # 2. cache the placebo opener token ids once (response-side continuation
            #    text, NOT apply_chat_template — it is forced into the response).
            placebo_ids = getattr(self, "_placebo_ids", None)
            if placebo_ids is None:
                placebo_ids = self.tokenizer.encode(
                    placebo_opener_str(), add_special_tokens=False
                )
                self._placebo_ids = placebo_ids

            # 3. continue raw from prompt + placebo (no logit_bias). vLLM continues
            #    the forced placebo block then the model solves on-distribution.
            #    CAP max_tokens by the placebo length: the placebo is the response
            #    PREFIX (response = placebo + gen, capped at response_length), so the
            #    real gen budget is response_length - len(placebo). Without the cap the
            #    request length is prompt + placebo + response_length — for prompts near
            #    the prompt/response boundary that exceeds max_model_len, an over-length
            #    edge case in the async agent loop that the with-arm (prompt +
            #    response_length) never hits (the only cf_group-vs-PMI rollout diff, and
            #    only on long prompts -> the intermittent hang). This makes the placebo
            #    request length EXACTLY match the with-arm, and stops wasting tokens that
            #    get truncated anyway.
            _sp = dict(sampling_params)
            _cap = max(1, int(self.response_length) - len(placebo_ids))
            if _sp.get("max_tokens") is None or int(_sp.get("max_tokens") or 0) > _cap:
                _sp["max_tokens"] = _cap
            metrics: dict[str, Any] = {}
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=list(prompt_ids) + list(placebo_ids),
                    sampling_params=_sp,
                    image_data=images,
                    video_data=videos,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = (
                    output.num_preempted if output.num_preempted is not None else -1
                )

            # 4. GRPO-member split: placebo is the TRAINED response prefix.
            fields = build_placebo_output(
                prompt_ids, placebo_ids, list(output.token_ids), self.response_length
            )

            out = AgentLoopOutput(
                prompt_ids=fields["prompt_ids"],
                response_ids=fields["response_ids"],
                # SCORED GRPO member, but forced placebo tokens have no sampling
                # logprobs -> None -> verl actor recomputes old_log_prob over the
                # FULL placebo+continuation response.
                response_logprobs=None,
                response_mask=fields["response_mask"],
                routed_experts=None,
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics=metrics,
                extra_fields=output.extra_fields,
            )
            out.extra_fields.update({"turn_scores": [], "tool_rewards": []})
            return out

except ImportError:  # pragma: no cover — metaprobe (no verl): pure helpers only.
    pass
