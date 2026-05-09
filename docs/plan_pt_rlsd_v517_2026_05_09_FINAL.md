# Plan v5.17 — FINAL (R5 코드 reading 기반 hard-commit, plan iteration 종료)

**작성일**: 2026-05-09 04:25Z
**상태**: **TERMINAL** — 추가 plan iteration 없음. 모든 결정은 R5 코드 직접 read로 lock-in.
**다음 단계**: code 작성 + iterative-code-review smoke→critic→improve loop.

---

## Hard-commit 사실 (R5 코드 직접 read)

| 항목 | 사실 (file:line) | v5.16에서 어긋난 점 |
|---|---|---|
| `compute_sdc_gdpo_advantage` signature | **free function**, no self (`verl_sdc_utils.py:201`) | v5.16 `self._current_lambda()` ❌ |
| `lam_meta` source | `config.get("sdc_lambda_meta", 0.5)` (line 318) | v5.16 `_current_lambda()` 미정의 ❌ |
| Factor pattern | **4-region**: meta + shared + diff + body (line 322-327) | v5.16 2-region (meta vs ~meta) ❌ |
| `ref_log_prob` shape | `[B, response_length]` (`dp_actor.py:128`) | OK |
| `ref_log_prob[i, t]` 의미 | `responses[i, t]`의 log_prob | v5.16 [i,p] 정확 ✓ |

---

## 0. Intent (1 문장)

R5의 `w_meta`를 `w_meta × w_position`로 교체. 다른 region (shared/diff/body)은 R5 그대로. **이게 ROD-PT 구현의 전부**.

---

## 1. 핵심 변경 (verl_sdc_utils.py 한 곳, ~10 lines)

`compute_sdc_gdpo_advantage` (line 201) 안 ROD_PT mode dispatch:

```python
elif sdc_mode == "ROD_PT":
    # ROD-PT: R5 attribution × position factor (PRODUCT, RLSD invariant 보존)
    log_prob_meta = batch.get("sdc_position_log_prob_meta", torch.zeros(student_logp.size(0), device=device))
    if log_prob_meta.dim() == 1:
        log_prob_meta = log_prob_meta.unsqueeze(1)  # [B, 1]
    w_position = torch.clamp(
        torch.exp(sign * log_prob_meta), 1.0 - clip_eps, 1.0 + clip_eps
    )  # [B, 1]
    w_meta = w_attr * w_position  # PRODUCT, broadcast [B, T] × [B, 1] → [B, T]
```

위치: 기존 `if sdc_mode == "RLSD_META_CONTRAST" or "RLSD_FORCED_META": ... w_meta = clamp(exp(sign * combined_log), ...)` 분기 옆에 **새 elif 추가**. 나머지 (shared, diff, body factor + masked_whiten) 모두 R5 그대로.

---

## 2. 핵심 변경 (verl_sdc.py 한 곳, ~40 lines)

`_attach_teacher_signals` (line 423) 안 ROD_PT branch (R5 builder 직접 호출):

```python
# After existing T+ forward (line 480-496)
if mode == "ROD_PT":
    META_START_ID = tokenizer.convert_tokens_to_ids("<|meta|>")
    target_device = response_tensor.device
    full_log_prob_meta = torch.zeros(B, device=target_device)

    # For each rollout, find first META_START position p in response
    rollout_indices = []
    for b in range(B):
        valid = (response_tensor[b] == META_START_ID) & response_mask[b].bool()
        nz = valid.nonzero(as_tuple=True)[0]
        if nz.numel() > 0:
            rollout_indices.append((b, int(nz[0].item())))

    if rollout_indices:
        # R5 builder 직접 호출 — answer_texts=gold, responses=rollout response 그대로
        # (META_START는 response[p]에 자연 발현 상태로 이미 있음)
        # response_mask는 [:p+1] 까지만 1.0으로 truncate (META 위치까지만 valid)
        N = len(rollout_indices)
        T_resp = response_tensor.size(1)
        truncated_mask = torch.zeros((N, T_resp), dtype=response_mask.dtype, device=response_mask.device)
        truncated_responses_list = []
        prompt_texts_subset = []
        gold_subset = []
        for i, (b, p) in enumerate(rollout_indices):
            truncated_mask[i, :p+1] = 1.0
            truncated_responses_list.append(response_tensor[b])  # full response, mask handles validity
            prompt_texts_subset.append(prompt_texts[b])
            gold_subset.append(gold_answers[b])
        truncated_responses = torch.stack(truncated_responses_list, dim=0)

        position_batch = _build_teacher_logprob_batch(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts_subset,
            answer_texts=gold_subset,
            responses=truncated_responses,
            response_mask=truncated_mask,
            v0_prefixes=None,
            forced_meta=False,
        )
        position_batch.meta_info["temperature"] = rollout_temp
        pos_out = trainer._compute_ref_log_prob(position_batch)
        # ref_log_prob[i, t] = log_prob of responses[i, t] given preceding context
        # → ref_log_prob[i, p] = log_prob(META | prompt + gold + response[:p])
        ref_log_probs = pos_out.batch["ref_log_prob"]  # [N, T_resp]

        for i, (b, p) in enumerate(rollout_indices):
            full_log_prob_meta[b] = ref_log_probs[i, p]

    data.batch["sdc_position_log_prob_meta"] = full_log_prob_meta
```

---

## 3. mode dispatch (verl_sdc.py:143-152)

```python
_SINGLE_TEACHER_MODES = {"RLSD_META_ATTR", "OPSD_META", "ROD_PT"}  # ROD_PT 추가
# _FORCED_META_MODES 에는 추가 안 함 (자연 emit)
```

`REWARD_CONFIGS["ROD_PT"]` 추가:
```python
"ROD_PT": {
    "funcs": [correctness_reward, meta_penalty_reward],
    "weights": [1.0, 1.0],
    "keys": ["correctness", "meta_penalty"],
},
```

---

## 4. yaml 사양

`configs/verl_rod_pt_R10_h200_4x4k.yaml`: R5 yaml `verl_rlsd_forced_meta_R5_h100_4x4k.yaml` 그대로 복사 + 변경 3곳:
- `algorithm.sdc_mode: ROD_PT`
- `algorithm.gdpo_reward_keys: [correctness, meta_penalty]` (R5와 동일, position은 reward dim 아님)
- `algorithm.sdc_lambda_meta: 0.5` (R5 기본)

`h200_rod_pt_R10_v2.yaml`: R5 yaml `h200_r5_rl_0506b.yaml` 그대로 복사 + 변경 2곳:
- HF push repo: `iamseungpil/metacot-h200-rod-pt-R10-veRL`
- runtime config: `configs/verl_rod_pt_R10_h200_4x4k.yaml`

---

## 5. 가설 + 검증 (간소)

| H | 가설 | PASS 기준 | FAIL → action |
|---|---|---|---|
| H1 | step time ≤ 1.5× R5 (50-55s) | ≤ 80s avg | I/O 진단 |
| H2 | cold start = R5 step 300 | byte-identical sample | ckpt 진단 |
| H3 | AIME ≥ 30% @ step 100 | (Wilson CI [16.7%, 47%]) | H-fallback (H-B) |
| H4 | 첫 ckpt 15분 이내 HF push | global_step_5 등장 | push pipeline 진단 |
| H5 | position factor std > 0.05 | wandb metric | clip_eps 조정 |
| H6 | log_prob_meta median ∈ [-3, -0.05] | first 10 step 측정 | mode 폐기 |

---

## 6. Implementation pipeline (이 plan 종료 후 즉시)

1. **Code patch**: §1-3 changes to verl_sdc.py + verl_sdc_utils.py
2. **Smoke test** (`tests/test_verl_rod_pt.py`): mode dispatch + position factor pure function
3. **code-reviewer critic** (round 1)
4. **수정 → re-smoke → re-critic** until 0 critical
5. **yaml 작성** + tarball push
6. **Cancel R15 (TRL fallback)**: veRL ROD-PT submit
7. **autoresearch monitor**: H1→H4→H6→H3 chain

---

## 7. 한 줄 요약

R5의 `w_meta`를 `w_meta × w_position` 한 줄 변경. 나머지 R5 그대로. plan iteration 종료. 코드 작성 단계로 즉시 이동.
