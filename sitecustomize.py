"""Local startup patches for veRL runtime.

This module is auto-imported by Python when the repository root is on
``PYTHONPATH``. We use it to patch veRL's agent-loop postprocess path so
validation batches with heterogeneous prompt lengths do not crash during
concatenation.
"""

from __future__ import annotations


def _patch_verl_agent_loop() -> None:
    # ★0921: 두 패치는 verl 0.7.1 전용이다(0.9 는 `_postprocess(..., validate=)` 시그니처가
    # 다르고 chat_template 결함도 없다). verl09 env(mc/)에서는 손대지 않는다.
    try:
        import verl as _verl
        if tuple(int(x) for x in str(getattr(_verl, "__version__", "0")).split(".")[:2]) >= (0, 8):
            return
    except Exception:
        return
    try:
        import numpy as np
        import torch
        from tensordict import TensorDict
        from verl import DataProto
        from verl.protocol import list_of_dict_to_dict_of_list
        from verl.experimental.agent_loop import agent_loop as ag
    except Exception:
        return

    if getattr(ag.AgentLoopWorker, "_metacognition_postprocess_patched", False):
        return

    def _pad_stack(tensors, pad_value=0):
        if not tensors:
            raise ValueError("Expected a non-empty tensor list")
        if len(tensors) == 1:
            return tensors[0]

        max_len = max(int(t.shape[-1]) for t in tensors)
        if all(int(t.shape[-1]) == max_len for t in tensors):
            return torch.cat(tensors, dim=0)

        padded = []
        for tensor in tensors:
            if int(tensor.shape[-1]) == max_len:
                padded.append(tensor)
                continue
            pad_shape = list(tensor.shape)
            pad_shape[-1] = max_len - int(tensor.shape[-1])
            pad_tensor = torch.full(
                pad_shape,
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded.append(torch.cat([tensor, pad_tensor], dim=-1))
        return torch.cat(padded, dim=0)

    def _concat_tensordicts_with_padding(batch_list):
        if not batch_list:
            return None
        if batch_list[0] is None:
            return None

        keys = list(batch_list[0].keys())
        merged = {}
        for key in keys:
            values = [batch[key] for batch in batch_list]
            if isinstance(values[0], torch.Tensor):
                merged[key] = _pad_stack(values, pad_value=0)
            else:
                merged[key] = torch.cat(values, dim=0)

        total_batch = sum(int(batch.batch_size[0]) for batch in batch_list)
        return TensorDict(merged, batch_size=[total_batch])

    def _patched_postprocess(self, inputs, input_non_tensor_batch=None):
        prompt_ids = _pad_stack([item.prompt_ids for item in inputs], pad_value=0)
        response_ids = _pad_stack([item.response_ids for item in inputs], pad_value=0)
        response_mask = _pad_stack([item.response_mask for item in inputs], pad_value=0)
        attention_mask = _pad_stack([item.attention_mask for item in inputs], pad_value=0)
        input_ids = _pad_stack([item.input_ids for item in inputs], pad_value=0)
        position_ids = _pad_stack([item.position_ids for item in inputs], pad_value=0)

        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = _pad_stack(
                [item.response_logprobs for item in inputs],
                pad_value=0,
            )
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = _pad_stack(
                [item.routed_experts for item in inputs],
                pad_value=0,
            )

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [item.reward_score for item in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(
                scores, dtype=torch.float32
            )
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([item.num_turns for item in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        reward_extra_infos = [item.extra_fields.get("reward_extra_info", {}) for item in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys()) if reward_extra_infos else []
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        multi_modal_inputs_list = [item.multi_modal_inputs for item in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [item.metrics.model_dump() for item in inputs]
        extra_fields = {}
        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
        }
        all_keys = set(key for item in inputs for key in item.extra_fields) | default_extra_keys
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [item.extra_fields.get(key) for item in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    ag.AgentLoopWorker._postprocess = _patched_postprocess
    ag.AgentLoopWorker._metacognition_postprocess_patched = True

    def _patched_concat(data):
        batch_lst = [item.batch for item in data]
        new_batch = _concat_tensordicts_with_padding(batch_lst)

        non_tensor_batch = list_of_dict_to_dict_of_list(list_of_dict=[d.non_tensor_batch for d in data])
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = np.concatenate(val, axis=0)

        merged_meta_info = {}
        if data:
            all_metrics = []
            for d in data:
                for k, v in d.meta_info.items():
                    if k == "metrics":
                        if v is not None:
                            if isinstance(v, list):
                                all_metrics.extend(v)
                            else:
                                all_metrics.append(v)
                    else:
                        if k in merged_meta_info:
                            assert merged_meta_info[k] == v, f"Conflicting values for meta_info key '{k}'"
                        else:
                            merged_meta_info[k] = v

            if all_metrics:
                merged_meta_info["metrics"] = list_of_dict_to_dict_of_list(all_metrics)

        cls = type(data[0]) if len(data) > 0 else DataProto
        return cls(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=merged_meta_info)

    DataProto.concat = staticmethod(_patched_concat)
    DataProto._metacognition_concat_patched = True


def _patch_verl_agent_loop_chat_template() -> None:
    r"""고정 자리(site) 롤아웃 — `AgentLoopBase.apply_chat_template` 이 어시스턴트
    프리픽스 이어붙이기(`continue_final_message=True, add_generation_prompt=False`)를
    **무시하는 게 아니라 그 조합에서 즉사한다**는 것을 확인했다(0904 조사).

    verl 0.7.1, `verl/experimental/agent_loop/agent_loop.py`
    (`/hdd_data/seungpil/envs/simplerl/lib/python3.10/site-packages/verl/...`),
    `AgentLoopBase.apply_chat_template` 의 비-processor 분기(347-349행, processor
    분기는 318-320행도 같은 결함):

        tokenized_prompt = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.tokenizer, messages, tools=tools,
                add_generation_prompt=True,      # ← 하드코딩
                tokenize=True,
                **self.apply_chat_template_kwargs,   # ← data.apply_chat_template_kwargs
            ),
        )

    `data.apply_chat_template_kwargs.add_generation_prompt=false` 를 hydra override 로
    넘기면 `self.apply_chat_template_kwargs` 에 `add_generation_prompt` 키가 들어가고,
    위 호출은 그 이름을 **명시 키워드로 두 번** 받는다 — 즉 무시가 아니라
    `TypeError: apply_chat_template() got multiple values for keyword argument
    'add_generation_prompt'` 로 배치 전체가 죽는다(agent-loop 워커 안이라 스택이
    Ray 로그에 묻히기 쉽다). `continue_final_message` 는 하드코딩 충돌이 없어 그대로
    통과하지만, 위 크래시가 먼저 나므로 site 프롬프트(마지막 메시지가 assistant 인
    프리픽스 이어붙이기)는 이 경로로 **아예 롤아웃되지 않는다**.

    최소 패치점: `AgentLoopBase.apply_chat_template` 그 자체. 마지막 메시지가
    assistant 면 `continue_final_message=True, add_generation_prompt=False` 를
    **기본값으로**(설정이 이미 명시했으면 그 값을 존중 — `setdefault`) 주고,
    `add_generation_prompt` 를 호출부에 하드코딩하지 않는다 — 그래야 `**kwargs` 와
    충돌할 자리가 애초에 없다. `initialize_system_prompt`(같은 파일 267행)는
    시스템 프롬프트 하나만 인코딩하는 별개 호출이라 손대지 않는다.
    """
    # ★0921: 두 패치는 verl 0.7.1 전용이다(0.9 는 `_postprocess(..., validate=)` 시그니처가
    # 다르고 chat_template 결함도 없다). verl09 env(mc/)에서는 손대지 않는다.
    try:
        import verl as _verl
        if tuple(int(x) for x in str(getattr(_verl, "__version__", "0")).split(".")[:2]) >= (0, 8):
            return
    except Exception:
        return
    try:
        from verl.experimental.agent_loop import agent_loop as ag
        from verl.utils.chat_template import apply_chat_template as _verl_apply_chat_template
    except Exception:
        return

    if getattr(ag.AgentLoopBase, "_metacognition_chat_template_patched", False):
        return

    _orig_apply_chat_template = ag.AgentLoopBase.apply_chat_template

    async def _patched_apply_chat_template(self, messages, tools=None, images=None,
                                            videos=None, remove_system_prompt=False):
        # processor(비전) 경로는 건드리지 않는다 — Countdown site 는 텍스트 전용이고,
        # 이 저장소가 검증한 것은 텍스트 토크나이저 경로뿐이다.
        if self.processor is not None:
            return await _orig_apply_chat_template(
                self, messages, tools=tools, images=images, videos=videos,
                remove_system_prompt=remove_system_prompt)

        kwargs = dict(self.apply_chat_template_kwargs)
        last_role = messages[-1]["role"] if messages else None
        if last_role == "assistant":
            # ★site 프리픽스: 마지막 메시지가 이미 assistant 텍스트라, 그 뒤에 새
            #   assistant 턴을 여는 generation prompt 를 또 붙이면 안 된다 — 이미
            #   assistant 턴 **안**이다. `setdefault`/`.get` 이라 launcher/config 가
            #   명시적으로 다른 값을 줬으면 그쪽을 존중한다(조용히 덮어쓰지 않는다).
            kwargs.setdefault("continue_final_message", True)
            kwargs["add_generation_prompt"] = kwargs.get("add_generation_prompt", False)
        else:
            kwargs.setdefault("add_generation_prompt", True)

        tokenized_prompt = await self.loop.run_in_executor(
            None,
            lambda: _verl_apply_chat_template(
                self.tokenizer, messages, tools=tools, tokenize=True, **kwargs),
        )
        from verl.utils.tokenizer import normalize_token_ids
        prompt_ids = normalize_token_ids(tokenized_prompt)
        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt):]
        return prompt_ids

    ag.AgentLoopBase.apply_chat_template = _patched_apply_chat_template
    ag.AgentLoopBase._metacognition_chat_template_patched = True


_patch_verl_agent_loop()
_patch_verl_agent_loop_chat_template()
