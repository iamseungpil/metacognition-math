"""E.9 BCI-RLVR custom agent loop — agent-loop-native binned-confidence injection.

verl 0.7.1 rolls out via the agent loop, which builds `prompt_ids` from `messages`
through the chat template and generates a response — there is NO input `input_ids`
tensor to splice a seed into (the earlier tensor-repack approach crashed on exactly
this). This loop works the agent-loop way, entirely in token-id LISTS:

  1. build prompt_ids from messages (same as SingleTurnAgentLoop)
  2. generate the continuation conditioned on  prompt_ids + seed  (the seed is the
     binned confidence meta block, e.g. "<|meta|>\nconfidence: 0.60\n<|/meta|>\n")
  3. return  response_ids = seed + continuation , prompt_ids = original prompt

So the seed lands at the START of the trained response (response_mask=1) — the policy
learns to EMIT the calibrated confidence — and the prompt is unchanged. No tensor
repack, no padding surgery.

The per-sample seed token-ids arrive via the non_tensor_batch field
`bci_conf_seed_ids`, set by the gated wrap `SDCRayPPOTrainer._bci_generate_sequences`
(bin index = row % rollout.n, matching gen_batch.repeat(n, interleave=True)).

Registered both via the @register decorator and via configs/bci_agent_loop.yaml
(actor_rollout_ref.rollout.agent.agent_loop_config_path) so the Ray rollout workers
resolve it by `_target_`. Additive: the default single_turn_agent loop is untouched
(validation, which the wrap no-ops, keeps using it).
"""
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("bci_conf_agent")
class BCIConfAgentLoop(SingleTurnAgentLoop):
    """SingleTurn loop that force-seeds a binned confidence meta block at the
    start of the assistant response (trained), so a proper-scoring reward can
    select the calibrated confidence bin (E.9 BCI-RLVR)."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        raw_seed = kwargs.get("bci_conf_seed_ids")
        seed_ids = [int(t) for t in list(raw_seed)] if raw_seed is not None else []

        # 1. images/videos + prompt_ids (identical to SingleTurnAgentLoop)
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, images=images, videos=videos)

        # 2. generate the continuation conditioned on prompt + seed
        gen_prompt_ids = list(prompt_ids) + seed_ids
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=gen_prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        # 3. response = seed + continuation, both trained (response_mask=1). The
        #    prompt stays original → the seeded confidence is in the RESPONSE.
        continuation = list(output.token_ids)
        response_ids = (seed_ids + continuation)[: self.response_length]
        response_mask = ([1] * (len(seed_ids) + len(continuation)))[: self.response_length]

        out = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            # rollout logprobs cover only the continuation (the seed was forced),
            # so they no longer align with response_ids → drop them; verl recomputes
            # old_log_prob over the full response in the actor forward (standard
            # guided-REINFORCE: the forced seed gets a valid advantage/gradient).
            response_logprobs=None,
            response_mask=response_mask,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        out.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return out
