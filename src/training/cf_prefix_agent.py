"""TRIOBJ_DCPO_V3 counterfactual agent loop — prefix-ingesting continuation.

The CF 2nd-generation must CONTINUE from a PRE-TOKENIZED prefix
  prefix_ids_i = [original prompt + main-response[:firstMetaIdx]]
with the <|meta|> token (id 151669) SUPPRESSED for that call only. The stock
single_turn_agent loop re-applies the chat template + re-tokenizes `raw_prompt`
(single_turn_agent_loop.py:45 → agent_loop.py:343-352), so it CANNOT continue
from raw prefix ids. This loop ingests `prefix_ids` directly and drives the
continuation primitive `server_manager.generate(prompt_ids=...)` (verl
agent_loop.py:135-164 → vllm_async_server.py:557 builds
TokensPrompt(prompt_token_ids=prompt_ids) → vLLM continues raw, NO chat template).

Token suppression: verl splats the sampling_params dict verbatim into
`SamplingParams(max_tokens=..., **sampling_params)` (vllm_async_server.py:549) with
NO key filtering, so a per-call shallow-copied dict with
`logit_bias = {151669: -100.0}` reaches vLLM's SamplingParams.logit_bias directly.
We do NOT mutate the shared batch dict — we copy it per call.

Per-sample fields arrive via non_tensor_batch (splatted into kwargs at
agent_loop.py:523), set by the gated CF wrap
`SDCRayPPOTrainer._dcpo_cf_call_engine`:
  - prefix_ids     : list[int]   — the continuation prompt (pre-tokenized)
  - cf_logit_bias  : dict        — {151669: -100.0}

Registered via the @register decorator AND via configs/cf_prefix_agent.yaml
(agent_loop_config_path) so the Ray rollout workers resolve it by `_target_`.
ADDITIVE: the default single_turn_agent loop is untouched (it is selected for
every non-CF row by non_tensor_batch["agent_name"]).
"""
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("cf_prefix_agent")
class CFPrefixAgentLoop(SingleTurnAgentLoop):
    """SingleTurn-derived loop that CONTINUES from a pre-tokenized prefix with
    <|meta|> suppressed via per-call logit_bias (TRIOBJ_DCPO_V3 counterfactual).

    Extends SingleTurnAgentLoop only to inherit `self.response_length` /
    `process_vision_info` plumbing; it overrides run() to bypass the chat-template
    re-tokenization and feed `prefix_ids` straight to the engine.
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # 1. PRE-TOKENIZED continuation prompt — NO chat template, NO re-tokenize.
        raw_prefix = kwargs.get("prefix_ids")
        prompt_ids = [int(t) for t in list(raw_prefix)] if raw_prefix is not None else []

        # 2. per-call sampling params: shallow-copy the SHARED batch dict and inject
        #    logit_bias so we never mutate the dict other rollouts use.
        sp = dict(sampling_params)
        lb = kwargs.get("cf_logit_bias")
        if lb:
            # vLLM SamplingParams.logit_bias = {token_id: bias}; -100.0 masks <|meta|>.
            sp["logit_bias"] = {int(k): float(v) for k, v in dict(lb).items()}

        # 3. continue raw from prefix_ids (vLLM TokensPrompt, no chat template).
        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sp,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_ids = list(output.token_ids)[: self.response_length]
        response_mask = [1] * len(response_ids)

        out = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            # The CF rollouts are inference-only (never scored for advantage), so
            # rollout logprobs are irrelevant; drop them.
            response_logprobs=None,
            response_mask=response_mask,
            routed_experts=None,
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        out.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return out
