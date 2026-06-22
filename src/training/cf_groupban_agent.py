"""GROUP-BRANCH COUNTERFACTUAL without-meta agent loop (design 2026-06-21).

The cf_group R_meta needs a without-meta sub-arm generated IN the main rollout
(no second decode). Each without-meta row is a real GRPO group member produced
on the NORMAL chat-template path, but with the meta-open + meta-close tokens
banned via per-call logit_bias, so the model cannot open a <|meta|> block.

UNLIKE cf_prefix_agent (which ingests pre-tokenized `prefix_ids` and bypasses
the chat template for a continuation decode), this loop keeps the standard
single_turn `raw_prompt` / chat-template path — it only injects `logit_bias`
into the sampling params before calling the stock SingleTurnAgentLoop.run().
That is the minimal surgical difference: same generation, meta tags suppressed.

Token suppression: verl splats the sampling_params dict verbatim into
`SamplingParams(max_tokens=..., **sampling_params)` (vllm_async_server.py:549)
with NO key filtering, so `logit_bias = {151669: -100.0, 151670: -100.0}`
reaches vLLM's SamplingParams.logit_bias directly. We shallow-copy per call so
the shared batch dict other rollouts use is never mutated (verbatim 3 lines from
cf_prefix_agent.py:61-65).

Per-row fields arrive via non_tensor_batch (splatted into kwargs at
agent_loop.py:523), set by the gated cf_group gen wrap
`SDCRayPPOTrainer._dcpo_cf_group_generate_sequences`:
  - cf_logit_bias : dict — {151669: -100.0, 151670: -100.0}  (or None for with-arm)

Registered via @register AND via configs/cf_groupban_agent.yaml
(agent_loop_config_path) so the Ray rollout workers resolve it by `_target_`.
ADDITIVE: the default single_turn_agent loop is untouched (with-arm rows keep it).
"""
import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("cf_groupban_agent")
class CFGroupBanAgentLoop(SingleTurnAgentLoop):
    """SingleTurn loop that bans the meta-open + meta-close tokens via per-row
    logit_bias on the NORMAL chat-template path (cf_group without-meta sub-arm).

    Keeps the stock raw_prompt / chat-template generation (does NOT ingest
    prefix_ids); only injects sampling_params['logit_bias'] then delegates to
    SingleTurnAgentLoop.run().
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Shallow-copy the SHARED batch dict and inject logit_bias so we never
        # mutate the dict other rollouts use (verbatim cf_prefix_agent.py:61-65).
        sp = dict(sampling_params)
        lb = kwargs.get("cf_logit_bias")
        if lb:
            # vLLM SamplingParams.logit_bias = {token_id: bias}; -100.0 bans the
            # meta-open (151669) + meta-close (151670) tags.
            sp["logit_bias"] = {int(k): float(v) for k, v in dict(lb).items()}
        return await super().run(sp, **kwargs)
