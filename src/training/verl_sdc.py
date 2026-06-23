"""veRL-native Shared-preserve SDC trainer (verl 0.7.1 compatible).

Preserves the original SDC intent:
  - scalar/group advantage via GDPO reward heads (correctness, outcome_calibration,
    meta_structure, meta_commit_shape, postmeta_closure)
  - token-wise credit shaped by teacher T+ / T- log-probs on meta/postmeta regions
  - free-text `confidence:` fallback detection (see feedback_reward_fallback)

Refactor notes (2026-04-20):
  verl 0.7.1 removed the `reward_fn`/`val_reward_fn` kwargs from
  `RayPPOTrainer.__init__`.  Reward is now routed through either the
  `RewardLoopManager` (async workers) or `config.reward.custom_reward_function`.
  To keep the SDC-specific reward+side-effect pipeline intact (meta masks,
  reward_extra_infos, teacher signals), we use a thin subclass
  `SDCRayPPOTrainer` that (1) accepts the legacy kwargs and (2) overrides
  `_compute_reward_colocate` to call our in-process reward manager.  This is
  the minimum change that preserves the intent while adopting the 0.7.1
  initialization contract (processor, train_dataset, val_dataset, collate_fn,
  train_sampler).
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import traceback
from typing import Callable, List

import numpy as np
import ray
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role

# ── DEADLOCK DIAGNOSTIC (gated; off by default — zero effect on normal runs) ──
# When DCPO_FAULTHANDLER_SEC is set, dump EVERY thread's Python stack on a
# repeating timer to /scratch/logs/faulthandler_trainer.log. A hung run (the
# cf_group async agent-loop + vLLM deadlock: GPU 0% util, no step) then leaves the
# deadlocked stack on disk even when amlt log capture is empty — retrieve via
# `amlt ssh :job -c "tail -300 /scratch/logs/faulthandler_*.log"` (interactive
# jobs only). This file is the TRAINER process; cf_placebo_agent dumps the ROLLOUT
# Ray-actor process (where the await likely hangs).
def _dcpo_install_faulthandler(tag: str):  # pragma: no cover — node-only diagnostic
    import faulthandler
    sec = os.environ.get("DCPO_FAULTHANDLER_SEC")
    if not sec:
        return
    try:
        os.makedirs("/scratch/logs", exist_ok=True)
        fh = open(f"/scratch/logs/faulthandler_{tag}.log", "a", buffering=1)
        faulthandler.dump_traceback_later(int(sec), repeat=True, file=fh)
        print(f"[DCPO] faulthandler self-dump every {sec}s -> "
              f"/scratch/logs/faulthandler_{tag}.log", flush=True)
    except Exception as _e:
        print(f"[DCPO] faulthandler setup skipped ({tag}): {_e}", flush=True)


_dcpo_install_faulthandler("trainer")

from src.metacot.prompt import META_END, META_START
from src.training.rewards import (
    compute_degeneration_penalty,
    correctness_reward,
    degeneration_penalty_reward,
    meta_commit_shape_reward,
    meta_penalty_reward,
    meta_structure_reward,
    outcome_calibration_reward,
    # ── C1/C2 next-wave arms (deliverables #1/#2): ADDITIVE imports. None of
    # the above existing imports change; these are only referenced by the NEW
    # REWARD_CONFIGS entries (ROD_PT2_E21CTRL, STABLE_GFN_C2FIX). The C2-fix
    # reward and the E21Rv2 control heads are reused from rewards.py verbatim.
    confidence_omission_floor,
    confidence_revision_reward,
    meta_count_bonus,
    meta_penalty_adaptive_reward,
    redirect_execution_reward,
    verify_execution_reward,
)
# TRIOBJ_META_V1 (ADDITIVE): the NEW meta-revision-utility head lives in its own
# module and is referenced ONLY by the new REWARD_CONFIGS['TRIOBJ_META_V1'] entry.
# No existing import/head changes.
from src.training.meta_revision_rewards import meta_revision_utility_reward
# TRIOBJ_DCPO_V2 (ADDITIVE): the NEW DCPO 3-region reward/mask helpers live in
# their own module and are referenced ONLY by REWARD_CONFIGS['TRIOBJ_DCPO_V2'],
# the _REGION_ROUTED_MODES gate, and the mode-gated mask-stack block. No existing
# import/head/mode changes.
from src.training.dcpo_region import (
    build_dcpo_region_masks,
    classify_dcpo_format,
    dcpo_region_rewards,
    first_meta_token_index,
    cf_answer_from_prefix,
    cf_group_arm_split,
    cf_group_route_row,
    cfgroup_scalar_summary,
    compute_cf_group_heads,
    TRUSTED_META_CLASSES,
    signature_suppression_ids,
)
from src.training._decoy_utils import _rule_based_decoy
# TRIOBJ_DCPO_V4 (ADDITIVE): the dense likelihood-delta (PMI) R_meta core is a
# pure numpy module (zero verl deps, shared with the offline probe). Referenced
# ONLY by the V4 populator block + _compute_dcpo_v4_pmi_rmeta below.
from src.training.dcpo_pmi import (
    PLACEBO_META,
    SpliceAlignmentError,
    compute_pmi_rows,
    splice_and_align,
    split_first_meta,
)
# SYNC PAIR (round 2 IMPORTANT-4): this import list is mirrored by the
# verl_sdc_utils STUB in tests/test_bci_isolation_regression.py
# (_install_verl_stubs) — adding a name here without adding the stub attr
# breaks that suite STANDALONE (hidden in the full suite by import order).
from src.training.verl_sdc_utils import (
    build_sdc_region_masks,
    compute_sdc_gdpo_advantage,
    dcpo_length_cost,
    dcpo_w_meta_warmup_scale,
    postmeta_closure_reward,
)


# ── TRIOBJ_DCPO_V2 reward-head wiring (ADDITIVE) ───────────────────────────────
# The three DCPO heads are GROUP-dependent (R_meta warrant uses the group p_hat),
# but the reward manager calls each reward_fn per-key without group structure. So
# the manager runs ONE mode-gated pre-pass (`_compute_dcpo_heads_stash`) that calls
# dcpo_region_rewards once with uid+step and stashes the per-rollout head lists; the
# three thin wrappers below just read the stash so REWARD_CONFIGS['TRIOBJ_DCPO_V2']
# keeps the exact 3-func/3-key GDPO contract. Pre-existing modes never touch this.
_DCPO_HEAD_STASH: dict = {"R_corr": None, "R_meta": None, "R_cal": None,
                         "p_hat": None, "group_acc": None,
                         "canary_pass1_acc": None, "sandbag_clamp": None}


def _compute_dcpo_heads_stash(
    completions, ground_truth, group_index, step, config,
    cf_completions=None, cf_correct=None, gate_unclosed=True, fmt_class=None,
):
    algo = getattr(config, "algorithm", None) if config is not None else None
    # Robust knob read (OmegaConf DictConfig supports .get; plain object uses getattr).
    def _read(name, default):
        try:
            if algo is not None and hasattr(algo, "get"):
                return algo.get(name, default)
            return getattr(algo, name, default) if algo is not None else default
        except Exception:
            return default
    # v3 R_meta = c_with - c_without uses only completions + cf_correct (+ uid/step for
    # grouping/diagnostics). The v2 reward knobs (eps/p_lo/warmup/sandbag/format_*) are
    # gone — no longer read or passed.
    out = dcpo_region_rewards(
        completions,
        ground_truth=ground_truth,
        group_index=group_index,
        step=step,
        cf_completions=cf_completions,   # v3: regenerated counterfactual texts (or None)
        cf_correct=cf_correct,           # v3: pre-graded CF correctness (producer) or None
        gate_unclosed=gate_unclosed,     # v3-only unclosed gate/penalty (v2 byte-identical)
        fmt_class=fmt_class,             # v3k: per-row parser classes (three-tier routing)
        # s1b collapse fix: asymmetric format head (see dcpo_region docstring);
        # default 1.0 = pre-fix verbatim for every existing config.
        format_neg=float(_read("dcpo_format_neg", 1.0)),
        # spec 2026-06-15 §3.3: medium penalty for opened-then-truncated meta
        # rows. Default 0.0 -> truncation stays format-neutral (byte-identical).
        trunc_open_penalty=float(_read("dcpo_trunc_open_penalty", 0.0) or 0.0),
    )
    _DCPO_HEAD_STASH.update(out)
    return out


# Round 2 M-A: under TRIOBJ_DCPO_V4 the R_meta source must be an EXPLICIT
# decision — the old `read("dcpo_rmeta_source", "cf")` default silently fell
# open onto the deprecated CF-regeneration path (plausible nonzero values, no
# log line) whenever the knob was missing OR the algorithm config was
# unreadable (the reader swallows exceptions into its default).
_V4_RMETA_SOURCES = ("cf", "pmi", "none", "cf_group")
_V4_RMETA_MISSING = object()


def _v4_rmeta_source_strict(read_knob) -> str:
    """Read algorithm.dcpo_rmeta_source; RAISE on absent/unreadable/invalid."""
    raw = read_knob("dcpo_rmeta_source", _V4_RMETA_MISSING)
    if raw is _V4_RMETA_MISSING or raw is None:
        raise ValueError(
            "TRIOBJ_DCPO_V4 requires algorithm.dcpo_rmeta_source to be set "
            f"explicitly (one of {_V4_RMETA_SOURCES}); the deprecated 'cf' path "
            "is opt-in only, never a silent fallback (review round 2 M-A)")
    src = str(raw)
    if src not in _V4_RMETA_SOURCES:
        raise ValueError(
            f"algorithm.dcpo_rmeta_source={src!r} not in {_V4_RMETA_SOURCES}")
    return src


def _populate_dcpo_region_keys(data) -> None:
    """TRIOBJ_DCPO_V2: write the 3 GDPO reward keys + 3 token masks into `data`.

    AUTHORITATIVE, GROUP-AWARE, MAIN-PROCESS population. Called from the
    `_REGION_ROUTED_MODES` short-circuit in `_attach_teacher_signals` — i.e.
    inside `patched_compute_advantage`, immediately BEFORE
    `compute_sdc_gdpo_advantage` runs the GDPO assertion + reads the heads.

    Why here (and not in `reward_loop_score`): the R_meta head is GROUP-dependent
    (its warrant uses the group p_hat), so it can only be computed once per batch
    with the full `uid` group structure + `step`. The Ray RewardLoopWorker actors
    that run `reward_loop_score` see one rollout at a time with no group, so they
    can only emit a 0.0 placeholder for `meta_region_utility` / `cal_region_reward`
    (R16 robustness pattern). This main-process write OVERWRITES that placeholder
    with the authoritative group-aware values before the assertion/advantage.

    Mirrors the synchronous `MetaCotSDCRewardManager.__call__` DCPO block exactly,
    but sources tokenizer/trainer/config from `_ACTIVE_SDC_CONTEXT` instead of
    `self` (the async-rollout path bypasses `__call__`, so neither the masks nor
    the keys are otherwise populated). Idempotent.
    """
    tokenizer = _ACTIVE_SDC_CONTEXT.get("tokenizer")
    trainer = _ACTIVE_SDC_CONTEXT.get("trainer")
    if tokenizer is None:
        raise RuntimeError("TRIOBJ_DCPO_V2: tokenizer context not initialized")
    # KARPATHY lock "v2 mode byte-identical": EVERYTHING v3-format-fix below
    # (unclosed clamp/gate in masks+heads, FORMAT_VIOLATION stack, the
    # format_penalty key) is gated on this flag — TRIOBJ_DCPO_V2 keeps the
    # legacy 3-mask/3-key population verbatim, so the 4th head can never arm
    # on a v2 async run (its yaml has neither dcpo_w_format nor the key).
    # TRIOBJ_DCPO_V4 joins via _DCPO_V3_FMT_MODES (same format machinery
    # verbatim; only the R_meta SOURCE differs — see the v4 block below).
    _is_v3 = _ACTIVE_SDC_CONTEXT.get("mode", "") in _DCPO_V3_FMT_MODES

    bs = len(data)
    response_length = data.batch["responses"].shape[-1]
    prompt_length = data.batch["prompts"].shape[-1]

    # v3k three-tier fmt machinery (parser-driven, spec §6-3). The CF wrap
    # stashes dcpo_fmt_replaced (0/1 per row) when token replacement ran; if
    # absent (replace knob off / wrap not installed) every row classifies HERE
    # with tier1_to_discard=True — replacement at this advantage-stage site is
    # TOO LATE (old_log_prob already computed), so unreplaced tier-1 rows are
    # conservatively demoted to discard (never half-replaced, spec risk 7).
    # Effective class per row: the ORIGINAL stashed class for replaced rows
    # (tier-1 names = "replaced" semantics downstream), else the parser class.
    _fmt_cls_stash = data.non_tensor_batch.get("dcpo_fmt_class", None) if _is_v3 else None
    _fmt_rep_stash = data.non_tensor_batch.get("dcpo_fmt_replaced", None) if _is_v3 else None
    _fmt_classes: list = []

    decoded_responses: list[str] = []
    ground_truths: list[str] = []
    dcpo_ans, dcpo_meta_c, dcpo_conf, dcpo_fmt, dcpo_fmt_ok = [], [], [], [], []
    dcpo_trunc = []  # TRUNC_OPEN: opened-then-truncated opener (spec §3.3)
    for i in range(bs):
        item = data[i]
        text, response_ids = _decode_response(
            tokenizer,
            item.batch["prompts"],
            item.batch["responses"],
            item.batch["attention_mask"],
            prompt_length,
        )
        decoded_responses.append(text)
        gt = item.non_tensor_batch.get("reward_model", {})
        if isinstance(gt, dict):
            gt = gt.get("ground_truth", "")
        ground_truths.append(str(gt))

        _rids = response_ids.tolist()
        _rmask = [True] * len(_rids)
        _decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
        if _is_v3:
            # v3k: ONE parser call drives masks + rewards + diagnostics. A
            # replaced row's ids are ALREADY the corrected (wellformed) ids —
            # the CF wrap mutated `responses` before old_log_prob — so the
            # parser naturally yields the wellformed regions for it.
            _rep = bool(
                _fmt_rep_stash is not None
                and i < len(_fmt_rep_stash)
                and float(_fmt_rep_stash[i]) > 0.5
            )
            _fmt = classify_dcpo_format(_rids, _rmask, _decode, tier1_to_discard=not _rep)
            if _rep and _fmt_cls_stash is not None and i < len(_fmt_cls_stash):
                _fmt_classes.append(str(_fmt_cls_stash[i]))  # original tier-1 name
            else:
                _fmt_classes.append(_fmt["fmt_class"])
            rmasks = build_dcpo_region_masks(
                _rids, _rmask, _decode, clamp_unclosed=True, fmt=_fmt, fmt_replaced=_rep)
        else:
            rmasks = build_dcpo_region_masks(_rids, _rmask, _decode, clamp_unclosed=False)

        def _pad_bool(arr) -> torch.Tensor:
            out = torch.zeros(response_length, dtype=torch.float32)
            n = min(response_length, len(arr))
            if n > 0:
                out[:n] = torch.as_tensor(arr[:n], dtype=torch.float32)
            return out

        dcpo_ans.append(_pad_bool(rmasks["ANSWER_REGION"]))
        dcpo_meta_c.append(_pad_bool(rmasks["META_CONTENT"]))
        dcpo_conf.append(_pad_bool(rmasks["CONF"]))
        dcpo_fmt.append(_pad_bool(rmasks["FORMAT_VIOLATION"]))
        dcpo_fmt_ok.append(_pad_bool(rmasks["FORMAT_OK"]))
        dcpo_trunc.append(_pad_bool(rmasks["TRUNC_OPEN"]))

    data.batch["dcpo_answer_mask"] = torch.stack(dcpo_ans, dim=0)
    data.batch["dcpo_meta_content_mask"] = torch.stack(dcpo_meta_c, dim=0)
    data.batch["dcpo_conf_mask"] = torch.stack(dcpo_conf, dim=0)
    # 4th routed head's token spans: FORMAT_VIOLATION (-side: drift </think> /
    # discard garbage) + FORMAT_OK (+side: wellformed closers, v3k two-sided
    # signal). Consumed by _compute_dcpo_region_advantage — v3-ONLY: it
    # activates the head on key PRESENCE, so stacking these for v2 would
    # silently arm it (review finding). FIVE-WAY SYNC #5: the two sync
    # __call__ DCPO blocks must stack the SAME v3 mask key set.
    if _is_v3:
        data.batch["dcpo_format_violation_mask"] = torch.stack(dcpo_fmt, dim=0)
        data.batch["dcpo_format_ok_mask"] = torch.stack(dcpo_fmt_ok, dim=0)
        # TRUNC_OPEN target for the un-centered open-meta-then-truncation penalty
        # (spec §3.3). Always stacked for v3; compose ignores it unless
        # dcpo_trunc_open_penalty>0 (default 0 -> byte-identical).
        data.batch["dcpo_trunc_open_mask"] = torch.stack(dcpo_trunc, dim=0)

    completions = [[{"content": t}] for t in decoded_responses]
    _uid = data.non_tensor_batch.get("uid", None)
    _step = int(getattr(trainer, "global_steps", 0) or 0)
    _config = getattr(trainer, "config", None)

    # TRIOBJ_DCPO_V3 (ADDITIVE): consume the counterfactual TEXTS the PRODUCER
    # (_dcpo_cf_generate_sequences, §3) stashed onto the batch BEFORE sleep_replicas().
    # We do NOT trigger the CF generation here — the engine is asleep at this consume
    # site. GRADING happens HERE (dcpo_region_rewards cf_completions path) because this
    # is where the real ground_truths are available — the producer's gen_output lacks
    # non_tensor 'reward_model' (grading there saw gt="" → c_without≡0, the v3b bug).
    # If absent (producer off / all rows skipped / v2 mode), cf_texts stays None and
    # dcpo_region_rewards falls back to the text path (cf_answer_from_prefix), so the
    # step never crashes (spec §5.2 fail-safe). None elements = skipped rows.
    _cf_texts = data.non_tensor_batch.get("cf_texts", None)
    if _cf_texts is not None:
        _cf_texts = [None if t is None else str(t) for t in list(_cf_texts)]

    _heads = _compute_dcpo_heads_stash(
        completions, ground_truths, _uid, _step, _config,
        cf_completions=_cf_texts,
        gate_unclosed=_is_v3,   # v2 byte-identical: no unclosed gate/penalty
        fmt_class=(_fmt_classes if _is_v3 else None),  # v3k three-tier routing
    )

    # v3k §8 runtime DCPO_DBG check (validates Assumption A1 on live steps):
    # replacement survived fit()'s union + old_log_probs exist & are finite at
    # the replaced positions. Warn-level oldlp-consistency heuristic inside.
    if _is_v3:
        _dcpo_fmt_replace_runtime_check(data, _step)

    # AUTHORITATIVE group-aware GDPO reward keys (overwrite any async placeholder).
    # R_corr -> 'correctness', R_meta -> 'meta_region_utility', R_cal -> 'cal_region_reward'.
    # float32 arrays of length B, written BEFORE compute_gdpo_outcome_advantage asserts.
    data.non_tensor_batch["correctness"] = np.asarray(_heads["R_corr"], dtype=np.float32)
    data.non_tensor_batch["meta_region_utility"] = np.asarray(_heads["R_meta"], dtype=np.float32)
    data.non_tensor_batch["cal_region_reward"] = np.asarray(_heads["R_cal"], dtype=np.float32)
    # meta_emission (OBSERVABILITY-ONLY, weight 0.0): it is listed in gdpo_reward_keys,
    # so the GDPO assertion requires it on this ASYNC path too — the RewardLoopWorker
    # placeholder set does not include it (v3g step-1 crash 2026-06-10: "GDPO reward
    # key 'meta_emission' not found in non_tensor_batch"). Same formula as
    # meta_emission_reward; weight 0.0 keeps it out of the advantage.
    data.non_tensor_batch["meta_emission"] = np.asarray(
        meta_emission_reward(completions), dtype=np.float32)
    # format_penalty (4th ROUTED head, w_format 0.1): listed in v3's
    # gdpo_reward_keys, so the GDPO assertion requires it on this ASYNC path too
    # (three-way sync rule — same crash class as meta_emission above). Sourced
    # from the heads (text-level meta_drift mirror) so it matches the
    # FORMAT_VIOLATION mask rows. v3-ONLY: writing it for v2 (whose keys list
    # has 3 entries) would arm the 4th head in _compute_dcpo_region_advantage.
    if _is_v3:
        data.non_tensor_batch["format_penalty"] = np.asarray(
            _heads.get("format_penalty", [0.0] * bs), dtype=np.float32)
        # v3k tier-2 exclusion membership (spec §10 risk 2, CLOSED): 0.0 =
        # discard row. _compute_dcpo_region_advantage threads this to compose
        # as member_mask so the forced-0 R_corr/R_meta/R_cal scalars stay OUT
        # of sibling group means (one discard in an all-correct group of n
        # would otherwise hand every sibling a spurious +1/n at w_corr where
        # exclusion gives the correct no-gradient 0). The row itself is
        # unaffected (its region masks are all-zero); the FORMAT head keeps
        # every row on purpose — discard's -1 vs wellformed's +1 IS the signal.
        # NOT a gdpo_reward_key (diagnostic-style batch key, like dcpo_phat),
        # so the FIVE-WAY SYNC key/weight lists are untouched.
        data.non_tensor_batch["dcpo_head_member"] = np.asarray(
            [0.0 if c == "discard" else 1.0 for c in _fmt_classes],
            dtype=np.float32)
        # v3m anti-collapse floor membership: 1.0 = TRUSTED meta row (region
        # routing reliable → eligible for the +dcpo_meta_floor emission bias on
        # its META_CONTENT tokens). discard/truncation/no_meta → 0.0 (no trusted
        # meta to lift; malformed meta must NOT farm the floor). Like
        # dcpo_head_member this is a diagnostic-style batch key (NOT a
        # gdpo_reward_key), so the FIVE-WAY SYNC key/weight lists are untouched.
        data.non_tensor_batch["dcpo_meta_floor_member"] = np.asarray(
            [1.0 if c in TRUSTED_META_CLASSES else 0.0 for c in _fmt_classes],
            dtype=np.float32)
        # open-meta-then-truncation membership (spec §3.3): 1.0 = the row opened
        # a <|meta|> then truncated before closing. compose applies the
        # un-centered -dcpo_trunc_open_penalty onto these rows' TRUNC_OPEN opener.
        # All-zero unless dcpo_trunc_open_penalty>0 -> default byte-identical.
        # Diagnostic-style batch key (NOT a gdpo_reward_key) -> SYNC lists intact.
        data.non_tensor_batch["dcpo_trunc_open_member"] = np.asarray(
            _heads.get("trunc_open_member", [0.0] * bs), dtype=np.float32)

    # ── TRIOBJ_DCPO_V4 R_meta SOURCE (ADDITIVE, mode+knob gated) ──────────────
    # dcpo_rmeta_source: 'cf' (EXPLICIT opt-in only — leave the
    # dcpo_region_rewards value, byte-identical to the v3 path) | 'pmi'
    # (overwrite meta_region_utility with the dense likelihood-delta head) |
    # 'none' (stage 1: hard-zero the head so the logged scalar cannot leak a
    # text-fallback CF signal at w_meta=0). Round 2 M-A: a MISSING/unreadable
    # knob RAISES — the old silent 'cf' default fell open onto the deprecated
    # regeneration path with plausible nonzero values, invisibly.
    # The overwrite happens HERE — after the authoritative head write above,
    # before compute_sdc_gdpo_advantage reads the key — so the FIVE-WAY SYNC
    # key/weight lists are untouched (same key, different source).
    if _ACTIVE_SDC_CONTEXT.get("mode", "") == "TRIOBJ_DCPO_V4":
        _algo_v4 = getattr(_config, "algorithm", None) if _config is not None else None

        def _v4_read(name, default):
            try:
                if _algo_v4 is not None and hasattr(_algo_v4, "get"):
                    return _algo_v4.get(name, default)
                return getattr(_algo_v4, name, default) if _algo_v4 is not None else default
            except Exception:
                return default

        _rmeta_src = _v4_rmeta_source_strict(_v4_read)
        if _rmeta_src == "pmi":
            _v4_prompt_texts = [
                _decode_prompt_only(
                    tokenizer,
                    data[i].batch["prompts"],
                    data[i].batch["attention_mask"],
                    prompt_length,
                )
                for i in range(bs)
            ]
            _r_meta_pmi, _rmeta_member = _compute_dcpo_v4_pmi_rmeta(
                tokenizer=tokenizer,
                trainer=trainer,
                prompt_texts=_v4_prompt_texts,
                response_texts=decoded_responses,
                fmt_classes=_fmt_classes,
                heads=_heads,
                read_knob=_v4_read,
                step=_step,
            )
            data.non_tensor_batch["meta_region_utility"] = _r_meta_pmi
            # R_meta-ONLY centering membership (review I2): 1.0 only for rows
            # whose PMI was actually computed (meta-emitting, splice-aligned,
            # guard-passed). NOT a gdpo_reward_key (diagnostic-style batch key
            # like dcpo_head_member) — FIVE-WAY SYNC lists untouched.
            data.non_tensor_batch["dcpo_rmeta_member"] = _rmeta_member
        elif _rmeta_src == "none":
            data.non_tensor_batch["meta_region_utility"] = np.zeros(bs, dtype=np.float32)
            data.non_tensor_batch["dcpo_rmeta_member"] = np.zeros(bs, dtype=np.float32)
        elif _rmeta_src == "cf_group":
            # GROUP-BRANCH COUNTERFACTUAL R_meta + SCoRe/AdaCoT (design 2026-06-21).
            # The without-meta sub-arm rows are REAL GRPO group members generated in
            # the MAIN rollout with the meta-open token banned (logit_bias). Their
            # standard c_with IS correct_without — no cf_texts, no second decode.
            # Arm membership comes from the gen-wrap stash dcpo_cf_with_meta (1.0
            # with / 0.0 without); it survives balance_batch reshuffle (per-row,
            # not ordinal). Fallback to the positional i%n split with a loud warn
            # if the wrap did not stash it (e.g. wrap not installed).
            _arm = data.non_tensor_batch.get("dcpo_cf_with_meta", None)
            if _arm is None:
                _n_roll = int(
                    getattr(getattr(getattr(_config, "actor_rollout_ref", None),
                                    "rollout", None), "n", 8) or 8
                ) if _config is not None else 8
                _frac = float(_v4_read("dcpo_cf_branch_frac", 0.5) or 0.5)
                _arm, _ = cf_group_arm_split(bs, n=_n_roll, branch_frac=_frac)
                _arm = np.asarray(_arm, dtype=np.float32)
                print(
                    "[DCPO-CFGROUP] WARN dcpo_cf_with_meta absent — falling back "
                    f"to positional i%{_n_roll} arm split (frac={_frac}). The "
                    "gen-wrap stash is missing; without-arm meta may NOT be banned.",
                    flush=True,
                )
            else:
                _arm = np.asarray(_arm, dtype=np.float32)
            _cfg = compute_cf_group_heads(
                c_with=_heads["c_with"],
                with_meta_flag=_arm,
                group_index=_uid,
                w_over=float(_v4_read("dcpo_w_over", 0.0) or 0.0),
                over_threshold=1.0,
                adaptthink_floor=True,
            )
            # cf_group routes R_meta onto ANSWER (locked decision), so the
            # META_CONTENT channel carries NOTHING: hard-zero the legacy R_meta key
            # + its centering membership so no stale value double-counts.
            data.non_tensor_batch["meta_region_utility"] = np.zeros(bs, dtype=np.float32)
            data.non_tensor_batch["dcpo_rmeta_member"] = np.zeros(bs, dtype=np.float32)
            # NEW diagnostic-style keys (NOT gdpo_reward_keys — FIVE-WAY SYNC lists
            # untouched, like dcpo_rmeta_member): the answer-routed counterfactual
            # delta R_meta + SCoRe R_trans + their with-arm member masks. Threaded
            # to compose by verl_sdc_utils._cfgroup_kwargs.
            data.non_tensor_batch["dcpo_ans_meta"] = np.asarray(
                _cfg["R_ans_meta"], dtype=np.float32)
            data.non_tensor_batch["dcpo_ans_member"] = np.asarray(
                _cfg["ans_meta_member"], dtype=np.float32)
            data.non_tensor_batch["dcpo_r_trans"] = np.asarray(
                _cfg["R_trans"], dtype=np.float32)
            data.non_tensor_batch["dcpo_trans_member"] = np.asarray(
                _cfg["trans_member"], dtype=np.float32)
            # AdaCoT over-trigger penalty folds onto R_corr's ANSWER routing (no
            # new GDPO key): subtract w_over from correctness for with-meta rows
            # whose without-arm was already correct.
            data.non_tensor_batch["correctness"] = (
                np.asarray(data.non_tensor_batch["correctness"], dtype=np.float32)
                - np.asarray(_cfg["over_penalty"], dtype=np.float32)
            ).astype(np.float32)
            # OBSERVABILITY (2026-06-22): the legacy dcpo/acc_without scalar reads
            # heads["c_without"] which cf_group never fills (NaN). Chart the TRUE
            # arm-split counterfactual instead — acc_with_arm/acc_without_arm/delta
            # + the mixed-group headroom rate — from the SAME _arm/_cfg the reward
            # uses (no recompute). Crash-proof: observability never kills training.
            try:
                import wandb as _wb
                if _wb.run is not None:
                    _summ = cfgroup_scalar_summary(
                        with_meta_flag=_arm, c_with=_heads["c_with"],
                        group_index=_uid, R_ans_meta=_cfg["R_ans_meta"],
                        ans_meta_member=_cfg["ans_meta_member"])
                    _wb.log(_summ, step=int(_step))
            except Exception as _e:  # pragma: no cover — diagnostics never raise
                print(f"[DCPO-CFGROUP] scalar-summary skipped: {_e}", flush=True)
        # 'cf' (explicit opt-in): no-op — the dcpo_region_rewards value stands.
        # Invalid values already raised inside _v4_rmeta_source_strict.
        if _rmeta_src in ("pmi", "none", "cf_group"):
            # Observability truth: the rollout table + trend scalars below must
            # chart the R_meta that actually ROUTES, not the stale CF/text-
            # fallback stash value. REASSIGN (not mutate): _DCPO_HEAD_STASH
            # still holds the original list, so the reward-func wrappers (which
            # feed the logging-only summed rm_scores) are untouched.
            _heads = dict(_heads)
            _heads["R_meta"] = [float(x) for x in data.non_tensor_batch["meta_region_utility"]]
            if _rmeta_src == "cf_group":
                # cf_group folded the AdaCoT over-penalty into correctness; chart
                # the R_corr that actually ROUTES (same observability rule).
                _heads["R_corr"] = [
                    float(x) for x in data.non_tensor_batch["correctness"]
                ]
        # w_meta warmup (review M4): linear 0 -> dcpo_w_meta over
        # dcpo_w_meta_warmup_steps; transported to the advantage stage via the
        # diagnostic-style key (absence-tolerant there; 0 steps -> scale 1.0).
        _warmup_steps = int(_v4_read("dcpo_w_meta_warmup_steps", 0) or 0)
        data.non_tensor_batch["dcpo_w_meta_scale"] = np.full(
            bs, dcpo_w_meta_warmup_scale(_step, _warmup_steps), dtype=np.float32)
        # Mild LENGTH COST (spec §2 emission-stability triad, third leg —
        # review round 1: in-scope, NOT deferred): subtract dcpo_len_cost *
        # (valid_response_len / max_response_len) * dcpo_w_meta_scale from the
        # R_corr scalar. Same 'correctness' key — no 6th GDPO key, FIVE-WAY
        # SYNC lists untouched; the M4 warmup couples it to w_meta per spec.
        # Knob default 0.0 keeps v4-off paths AND stage 1 byte-identical.
        _len_cost = float(_v4_read("dcpo_len_cost", 0.0) or 0.0)
        if _len_cost != 0.0:
            _valid_lens = (
                data.batch["attention_mask"][:, prompt_length:]
                .sum(dim=-1).cpu().numpy()
            )
            data.non_tensor_batch["correctness"] = (
                np.asarray(data.non_tensor_batch["correctness"], dtype=np.float32)
                - dcpo_length_cost(
                    _valid_lens, response_length, _len_cost,
                    data.non_tensor_batch["dcpo_w_meta_scale"])
            ).astype(np.float32)
            # Observability truth (same rule as the R_meta reassign above): the
            # rollout table / trend scalars must chart the R_corr that ROUTES.
            _heads = dict(_heads)
            _heads["R_corr"] = [float(x) for x in data.non_tensor_batch["correctness"]]

    # Diagnostics (wandb) — same as the synchronous __call__ block.
    data.non_tensor_batch["dcpo_phat"] = np.asarray(_heads["p_hat"], dtype=np.float32)
    data.non_tensor_batch["dcpo_group_acc"] = np.asarray(_heads["group_acc"], dtype=np.float32)
    data.non_tensor_batch["dcpo_canary_pass1_acc"] = np.asarray(
        _heads.get("canary_pass1_acc", [1.0] * bs), dtype=np.float32)
    data.non_tensor_batch["dcpo_sandbag_clamp"] = np.asarray(
        _heads.get("sandbag_clamp", [1.0] * bs), dtype=np.float32)

    # FULL-ROLLOUT wandb TABLE (observability: the v3b correlation-signal bug was
    # only visible by grepping node logs for one DBG sample; this puts EVERY
    # rollout — main text, CF text, per-head rewards, c_with/c_without — in the
    # wandb UI so "is the signal right?" is checkable per step).
    _log_dcpo_rollout_table(
        step=_step, uid=_uid, completions=completions, ground_truths=ground_truths,
        cf_texts=_cf_texts, heads=_heads,
        arm=data.non_tensor_batch.get("dcpo_cf_with_meta"),  # cf_group arm; None elsewhere
    )
    # INTENT-TREND scalars (one wandb chart each): emission rate, the R_meta
    # decomposition over meta-bearing rows, CF pipeline health, and the batch
    # causal effect acc_with - acc_without. These answer "is training moving
    # toward useful metacognition?" without grepping logs or opening the table.
    _log_dcpo_trend_scalars(step=_step, heads=_heads, cf_texts=_cf_texts)


# v3k §8 state: the populator-side old_log_prob consistency check runs on the
# FIRST step that carries replacements, then every N=50 steps (cheap but not free).
_DCPO_FMT_DBG_STATE = {"first_done": False}


def _dcpo_fmt_replace_runtime_check(data, step, every: int = 50):
    """v3k §8 runtime assertions at the ADVANTAGE stage (old_log_probs in batch).

    Validates Assumption A1 (verl recomputes old_log_prob on the tensors the CF
    wrap mutated — verl source absent locally, so this is checked AT RUNTIME):
      1. every recorded replacement survived fit()'s union:
         data.batch['responses'][row, pos] == new_id (HARD assert);
      2. old_log_probs[row, pos] is finite (HARD assert);
      3. heuristic (warn-only): the corrected tag was NOT sampled by the policy,
         so its old_log_prob should sit well below the sampled-token mean —
         if replaced_oldlp_mean > sampled_oldlp_mean - 0.5, print a LOUD
         [DCPO_DBG] OLD-LOGPROB-CONSISTENCY SUSPECT warning.
    Logs dcpo/replaced_oldlp_mean + dcpo/sampled_oldlp_mean. Never raises out
    (assertion failures print loudly + re-raise: silent stale-ratio training is
    the one failure mode this exists to prevent).
    """
    plans = data.non_tensor_batch.get("dcpo_fmt_replace_plan", None)
    if plans is None:
        return
    has_repl = any(len(p or []) > 0 for p in list(plans))
    if not has_repl:
        return
    if _DCPO_FMT_DBG_STATE["first_done"] and int(step) % every != 0:
        return
    _DCPO_FMT_DBG_STATE["first_done"] = True
    resp = data.batch["responses"]
    old_lp = data.batch.get("old_log_probs", None)
    repl_lps = []
    for row, plan in enumerate(list(plans)):
        for (pos, _old_id, new_id) in (plan or []):
            got = int(resp[row, pos])
            assert got == int(new_id), (
                f"[DCPO_DBG] REPLACEMENT LOST IN UNION: responses[{row},{pos}]="
                f"{got} != replaced id {int(new_id)} — the actor forward saw "
                f"different ids than the advantage stage (Assumption A1 broken)."
            )
            if old_lp is not None:
                lp = float(old_lp[row, pos])
                assert lp == lp and abs(lp) != float("inf"), (
                    f"[DCPO_DBG] old_log_probs[{row},{pos}]={lp} not finite at a "
                    f"replaced position."
                )
                repl_lps.append(lp)
    if old_lp is None or not repl_lps:
        return
    try:
        _rm = data.batch["attention_mask"][:, data.batch["prompts"].shape[-1]:]
        _rm = _rm[:, : old_lp.shape[-1]].bool()
        sampled_mean = float(old_lp[_rm].float().mean())
        replaced_mean = float(np.mean(repl_lps))
        if replaced_mean > sampled_mean - 0.5:
            print(
                f"[DCPO_DBG] OLD-LOGPROB-CONSISTENCY SUSPECT: replaced_oldlp_mean="
                f"{replaced_mean:.3f} vs sampled_oldlp_mean={sampled_mean:.3f} — "
                f"replaced (unsampled) tags should score well below sampled tokens; "
                f"check that the engine is NOT reusing rollout log-probs.", flush=True)
        import wandb
        if wandb.run is not None:
            wandb.log({"dcpo/replaced_oldlp_mean": replaced_mean,
                       "dcpo/sampled_oldlp_mean": sampled_mean}, step=int(step))
    except AssertionError:
        raise
    except Exception as _e:  # pragma: no cover — diagnostics never kill training
        print(f"[DCPO_DBG] oldlp-consistency scalar skipped: {_e}", flush=True)


def _log_dcpo_trend_scalars(*, step, heads, cf_texts):
    """Per-step intent-trend scalars under 'dcpo/' (crash-proof, never raises).

    dcpo/meta_emit_rate        fraction of rollouts emitting <|meta|> (v3b collapsed 42%->23%)
    dcpo/rmeta_pos_rate        fraction with R_meta=+1 (meta causally SAVED the answer)
    dcpo/rmeta_neg_rate        fraction with R_meta=-1 (meta causally HURT)
    dcpo/rmeta_mean_meta_rows  mean R_meta over meta-bearing rows ONLY (undiluted net utility)
    dcpo/cf_text_rate          CF regeneration success rate (pipeline health)
    dcpo/acc_with              batch accuracy of the main rollouts (c_with mean)
    dcpo/acc_without           batch accuracy of graded counterfactuals (c_without mean)
                               -> acc_with - acc_without = the batch-level CAUSAL effect of meta
    dcpo/cw_graded_rate        fraction of rows with a graded c_without (non-NaN)
    dcpo/meta_unclosed_rate    fraction with an UNCLOSED meta (continuity: textual unclosed = drift+truncation)
    dcpo/format_penalty_rate   fraction with format_penalty < 0 (v3k: drift + discard rows)
    v3k three-tier class rates (fmt_class present in the stash only under V3):
    dcpo/replaced_rate         tier-1 token-replaced rows (swapped/dup_open/reversed)
    dcpo/discard_rate          tier-2 rows (all heads zeroed, -1 on garbage delimiters)
    dcpo/drift_rate            tier-3 rows (recovered span plays R_meta, -1 on </think>)
    dcpo/wellformed_rate       originally-wellformed rows (+1 on the closer)
    """
    import os as _os
    if _os.environ.get("DCPO_WANDB_ROLLOUTS", "1") != "1":
        return
    try:
        import wandb  # noqa: F811
        if wandb.run is None:
            return
        hm = [bool(x) for x in heads["has_meta"]]
        rm = [float(x) for x in heads["R_meta"]]
        cw = [float(x) for x in heads["c_with"]]
        cwo = [float(x) for x in heads["c_without"]]   # NaN = no counterfactual
        B = max(1, len(rm))
        meta_rows = [i for i in range(len(rm)) if hm[i]]
        graded = [v for v in cwo if v == v]
        scal = {
            "dcpo/meta_emit_rate": sum(hm) / B,
            "dcpo/rmeta_pos_rate": sum(1 for v in rm if v > 0.5) / B,
            "dcpo/rmeta_neg_rate": sum(1 for v in rm if v < -0.5) / B,
            "dcpo/rmeta_mean_meta_rows": (
                sum(rm[i] for i in meta_rows) / len(meta_rows) if meta_rows else 0.0
            ),
            "dcpo/cf_text_rate": (
                sum(1 for t in (cf_texts or []) if t is not None) / B
            ),
            "dcpo/acc_with": sum(cw) / B,
            "dcpo/acc_without": (sum(graded) / len(graded)) if graded else float("nan"),
            "dcpo/cw_graded_rate": len(graded) / B,
            # unclosed/drift trends (.get-guarded: older stashes lack the keys).
            "dcpo/meta_unclosed_rate": (
                sum(1 for v in heads.get("meta_unclosed", []) if float(v) > 0.5) / B
            ),
            "dcpo/format_penalty_rate": (
                sum(1 for v in heads.get("format_penalty", []) if float(v) < 0.0) / B
            ),
        }
        # v3k class-rate scalars (heads["fmt_class"] is None pre-k / v2).
        fc = heads.get("fmt_class", None)
        if fc:
            _tier1 = ("swapped", "dup_open", "reversed")
            Bf = max(1, len(fc))
            scal["dcpo/replaced_rate"] = sum(1 for c in fc if c in _tier1) / Bf
            scal["dcpo/discard_rate"] = sum(1 for c in fc if c == "discard") / Bf
            scal["dcpo/drift_rate"] = sum(1 for c in fc if c == "drift") / Bf
            scal["dcpo/wellformed_rate"] = sum(1 for c in fc if c == "wellformed") / Bf
        # anchor-norm effective scales (spec 2026-06-15 HC1): is the weak PMI/meta
        # head riding at R_corr's scale (anchor working) or still buried? Read the
        # module-level EMA that compose updates. Guarded: empty dict pre-anchor /
        # anchor off -> no keys logged (byte-identical observability).
        try:
            from src.training.verl_sdc_utils import _ANCHOR_EMA_STATE as _AES
            _cs = float(_AES.get("corr", 0.0) or 0.0)
            if _cs > 0:
                for _h in ("corr", "meta", "cal", "format", "emit"):
                    if _h in _AES:
                        _v = float(_AES.get(_h, 0.0) or 0.0)
                        scal[f"dcpo/eff_scale_{_h}"] = _v
                        if _h != "corr":
                            scal[f"dcpo/eff_ratio_{_h}"] = _v / _cs
        except Exception:
            pass
        wandb.log(scal, step=int(step))
    except Exception as _e:  # pragma: no cover — observability never kills training
        print(f"[DCPO] trend-scalar log skipped: {type(_e).__name__}: {_e}", flush=True)


def _log_dcpo_rollout_table(*, step, uid, completions, ground_truths, cf_texts, heads,
                            arm=None):
    """Log the whole batch as a wandb Table under 'dcpo/rollouts'.

    Env knobs: DCPO_WANDB_ROLLOUTS=1 (default ON), DCPO_WANDB_ROLLOUTS_EVERY=5
    (log every Nth step; 1 = every step), DCPO_WANDB_TEXT_CHARS=1500 (tail chars
    of the main rollout; CF gets half). NEVER raises — observability must not
    kill training. No-op when wandb is absent / run not initialized (console-only).
    """
    import os as _os
    if _os.environ.get("DCPO_WANDB_ROLLOUTS", "1") != "1":
        return
    try:
        every = max(1, int(_os.environ.get("DCPO_WANDB_ROLLOUTS_EVERY", "5") or 5))
        if int(step) % every != 0:
            return
        import wandb  # noqa: F811
        if wandb.run is None:
            return
        nchars = max(200, int(_os.environ.get("DCPO_WANDB_TEXT_CHARS", "1500") or 1500))
        B = len(completions)
        _uid_l = list(uid.tolist() if hasattr(uid, "tolist") else (uid or range(B)))
        from src.training.rewards import _get_text as _gt_text
        cols = ["step", "row", "group", "arm", "gt", "answer", "c_with", "c_without",
                "R_corr", "R_meta", "R_cal", "conf", "has_meta", "unclosed",
                "fmt_class", "replaced", "main_tail", "cf_tail"]
        table = wandb.Table(columns=cols)
        # cf_group arm flag (1.0=with-meta / 0.0=without-meta arm); "" off cf_group.
        _arm_l = list(arm.tolist() if hasattr(arm, "tolist") else arm) if arm is not None else None
        _unc = heads.get("meta_unclosed", None)  # .get-guarded (older stashes)
        _fc = heads.get("fmt_class", None)       # v3k class column (None pre-k / v2)
        _tier1 = ("swapped", "dup_open", "reversed")
        for i in range(B):
            main = _gt_text(completions[i]) or ""
            cf = (cf_texts[i] if (cf_texts is not None and i < len(cf_texts)) else None) or ""
            _fci = str(_fc[i]) if (_fc is not None and i < len(_fc)) else ""
            _armi = ("" if _arm_l is None or i >= len(_arm_l)
                     else float(_arm_l[i]))
            table.add_data(
                int(step), i, str(_uid_l[i] if i < len(_uid_l) else i), _armi,
                str(ground_truths[i])[:80], str(heads["answer"][i])[:80],
                float(heads["c_with"][i]), float(heads["c_without"][i]),
                float(heads["R_corr"][i]), float(heads["R_meta"][i]), float(heads["R_cal"][i]),
                float(heads["conf"][i]), bool(heads["has_meta"][i]),
                bool(float(_unc[i]) > 0.5) if (_unc is not None and i < len(_unc)) else False,
                # fmt_class keeps the ORIGINAL tier-1 name for replaced rows;
                # `replaced` flags them (tier-1 names appear ONLY when replaced).
                _fci, _fci in _tier1,
                main[-nchars:], cf[-(nchars // 2):],
            )
        # step-keyed log; runs BEFORE the tracker's metric commit for this step,
        # so the explicit step stays monotonic (no grid step-collision clamp).
        wandb.log({"dcpo/rollouts": table}, step=int(step))
    except Exception as _e:  # pragma: no cover — observability never kills training
        print(f"[DCPO] rollout-table log skipped: {type(_e).__name__}: {_e}", flush=True)


def correctness_region_reward(completions, ground_truth=None, **kwargs):
    """TRIOBJ_DCPO_V2 R_corr head (reads the per-batch DCPO stash)."""
    r = _DCPO_HEAD_STASH.get("R_corr")
    return list(r) if r is not None else [0.0] * len(completions)


def meta_region_utility_reward(completions, ground_truth=None, **kwargs):
    """TRIOBJ_DCPO_V2 R_meta head (reads the per-batch DCPO stash)."""
    r = _DCPO_HEAD_STASH.get("R_meta")
    return list(r) if r is not None else [0.0] * len(completions)


def cal_region_reward(completions, ground_truth=None, **kwargs):
    """TRIOBJ_DCPO_V2 R_cal head (reads the per-batch DCPO stash)."""
    r = _DCPO_HEAD_STASH.get("R_cal")
    return list(r) if r is not None else [0.0] * len(completions)


def meta_emission_reward(completions, ground_truth=None, **kwargs):
    """OBSERVABILITY-ONLY (weight 0.0 in TRIOBJ_DCPO_V3): 1.0 iff the rollout
    emits a <|meta|> block. Contributes NOTHING to the reward (weight 0) — it
    rides the reward-key plumbing so val logs
    val-aux/<dataset>/meta_emission/mean@1 = per-benchmark META EMISSION RATE
    every test_freq steps (the v3b emission collapse 42%→23% was only visible
    by grepping node logs)."""
    from src.training.rewards import _get_text as _gt
    return [1.0 if "<|meta|>" in (_gt(c) or "") else 0.0 for c in completions]


def format_penalty_reward(completions, ground_truth=None, **kwargs):
    """FORMAT head (TRIOBJ_DCPO_V3, w 0.1; DeepSeek-R1-style separate format
    reward). STASH-FIRST (v3k five-way sync): when the per-batch DCPO head
    pre-pass ran (it always does on the region-routed paths, right before the
    reward-func loop), this returns the stashed per-class values — +1 wellformed
    / -1 drift+discard / 0 replaced+truncation+no_meta — so the sync __call__
    paths write the SAME format_penalty the async populator writes (identical
    gate/penalty/tier semantics both paths).

    TEXT FALLBACK (stash absent/stale, e.g. a bare val call): -1.0 iff the
    rollout opens a <|meta|> block, NEVER closes it, AND a </think> appears
    after the last open — i.e. format DRIFT (the model abandoned the tag
    mid-stream but kept generating). Text-level mirror of the mask-level
    meta_drift in build_dcpo_region_masks. TRUE TRUNCATION (no </think> after
    the open — cut at max length) scores 0.0: that is a length problem, not a
    format habit. Closed blocks / no-meta rollouts -> 0.0."""
    r = _DCPO_HEAD_STASH.get("format_penalty")
    if r is not None and len(r) == len(completions):
        return list(r)
    from src.training.rewards import _get_text as _gt
    out = []
    for c in completions:
        t = _gt(c) or ""
        pen = 0.0
        if "<|meta|>" in t and "<|/meta|>" not in t:
            _last_open = t.rfind("<|meta|>")
            if t.find("</think>", _last_open) != -1:
                pen = -1.0
        out.append(pen)
    return out


REWARD_CONFIGS = {
    "SDC_SHARED": {
        "funcs": [
            correctness_reward,
            outcome_calibration_reward,
            meta_structure_reward,
            meta_commit_shape_reward,
            postmeta_closure_reward,
        ],
        "weights": [1.0, 0.7, 0.25, 0.35, 0.45],
        "keys": [
            "correctness",
            "outcome_calibration",
            "meta_structure",
            "meta_commit_shape",
            "postmeta_closure",
        ],
    },
    # E21R-v2 + SDC contrastive: only correctness as the GDPO reward head.
    # Meta-format heads (saturated to ~95% by step 100, providing reward floor)
    # are removed because they push the policy toward "clean wrong" on hard tasks.
    # SDC teacher / anti-teacher contrastive (lambda_meta/shared/diff) remain as
    # auxiliary actor losses — they replace outcome_calibration as the second
    # signal source.
    "SDC_CORR_ONLY": {
        "funcs": [correctness_reward],
        "weights": [1.0],
        "keys": ["correctness"],
    },
    # Meta-only SDC RLSD: correctness + asymmetric meta penalty.
    # Penalty-only meta head (0 if meta present, -0.20 if missing) prevents
    # meta-scaffold collapse during RL without creating the +0.9 saturation
    # floor that the symmetric ±0.10 head caused. Pair with
    # sdc_lambda_shared/diff = 0 in the YAML to restrict contrastive teacher
    # to meta tokens only.
    "SDC_CORR_META_PEN": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ── RLSD ablation modes (arxiv 2604.03128) ───────────────────────────────
    # R0: vanilla GRPO baseline — no SDC teacher signal at all. Used to isolate
    # the contribution of meta-region teacher guidance over plain RLVR.
    "VANILLA_GRPO": {
        "funcs": [correctness_reward],
        "weights": [1.0],
        "keys": ["correctness"],
    },
    # R1: RLSD with single (gold-conditioned) teacher T+ on meta region only.
    # Matches the paper's RLSD formulation: factor = exp(sign × (T+ − student))
    # clipped to [1−ε, 1+ε], applied as multiplicative magnitude evaluator.
    # Decoy forward pass is SKIPPED at runtime (saves the 1 of ~3-4 forwards
    # used by the contrastive variants — roughly 25-33% wall-time saving).
    "RLSD_META_ATTR": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # R2: RLSD extended with contrastive component. Combined log-ratio
    #   combined = α × (T+ − student) + β × (T+ − T−)
    # On meta region, factor = clip(exp(sign × combined), 1−ε, 1+ε).
    # α (sdc_alpha_attr, default 0.5) weights attractive component.
    # β (sdc_beta_contrast, default 0.5) weights gold-vs-decoy contrast.
    # Both T+ and T− forward passes required.
    "RLSD_META_CONTRAST": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # R3: OPSD baseline — full distribution distillation KL(T+ ‖ student) on
    # meta tokens. Distillation loss path is NOT YET IMPLEMENTED (deferred);
    # this mode currently behaves like RLSD_META_ATTR with a marker for the
    # advantage path. Trainer must add an auxiliary KL loss term to fully
    # realize OPSD; tracked as phase-2 work.
    "OPSD_META": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # R5: RLSD with FORCED <|meta|> token on both student rollout and teacher
    # conditioning. Resolves the "meta empty 95%" pathology by guaranteeing
    # meta region presence (paper 2603.24472 epistemic suppression mitigation).
    # Verified S3 (inspect_forced_meta.py): V0_prefix + gold + force <|meta|>
    # → 71.4% gold commit, 33.3% AIME accuracy, valid meta + body + boxed.
    "RLSD_FORCED_META": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ROD-PT: R5 RLSD framework + position teacher amplify (Plan v5.17 FINAL).
    # Decoy T- replaced by T_position which measures log_prob(META | prompt+gold+response[:p])
    # at first META_START emit position p. Multiplicative on R5's w_meta:
    #   w_combined = w_attr * w_position (RLSD invariant 보존, sign 절대 안 바꿈).
    # Natural emit (forced_meta=False, V0_prefix unused).
    "ROD_PT": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ROD_PT_DEGEN (R16, 2026-05-12): ROD_PT + sample-level degeneration_penalty
    # via dedicated GDPO key, weight 0.3 vs correctness.
    # Stabilizes hard-regime generation (AIME tail collapse 57% → target <30%)
    # so R17 (control-field RLSD + follow-through) can be measured cleanly.
    # Spec: codex-locked V4 (4 rounds review, 1030-trace replay validated).
    # See src/training/rewards.py compute_degeneration_penalty for details.
    "ROD_PT_DEGEN": {
        "funcs": [correctness_reward, meta_penalty_reward, degeneration_penalty_reward],
        "weights": [1.0, 1.0, 0.3],
        "keys": ["correctness", "meta_penalty", "degeneration_penalty"],
    },
    # ROD_MQ (R18a, Plan v7.2.2): meta-quality verifiable signal — single
    # teacher T+ on EXTENDED meta region (meta block + post K=10 tokens).
    # Sign-preserving multiplicative factor:
    #   q_attr = mean over extended meta of clip(T+ − student, [-10,10])
    #   q_centered = q_attr − batch_median(q_attr)   (centered, RLVR invariant)
    #   w_meta_quality = clip(exp(sign × q_centered / τ), 1−ε, 1+ε)   per sequence
    #   w_meta = w_attr × w_meta_quality            (PRODUCT, sign preserved)
    # NO rubric, NO judge — verifiable signal only (RLVR + T+ logit).
    "ROD_MQ": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ROD_MQ_CONTRAST (R18b, Plan v7.2.2): R18a + decoy-contrast term.
    #   q_contrast = mean over extended meta of clip(T+ − T−, [-10,10])
    #   q_meta = α × q_attr + β × q_contrast        (α default 1.0, β default 0.0)
    # Same q_centered / w_meta_quality / w_meta product as R18a.
    "ROD_MQ_CONTRAST": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ROD_MQ_CONTRAST_INJECT (CTSD Phase C, Plan v5): identical reward heads to
    # ROD_MQ_CONTRAST (R18b) — the ONLY difference is rollout-time force-inject of
    # <|meta|> at the max-entropy pre-answer position (algorithm.sdc_force_inject).
    # R18b failed (70.9%) because contrastive reward had no good/bad variance to
    # act on in the model's decorative natural meta; force-inject creates the meta
    # region for the reward to shape. Inject core = src/training/meta_inject.py
    # (unit-tested); two-phase rollout wiring = SDCRayPPOTrainer (node-smoke-req).
    # Gated by A.3 PASS (force-inject shown causally helpful) before launch.
    "ROD_MQ_CONTRAST_INJECT": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # GFN_OPSD_CONTRAST (R18c, Plan v7.2.7 codex r12-r14 LOCK): GFN distribution
    # matching on meta token region. Listwise KL (target=softmax(logR/τ),
    # student=softmax(logP_S/τ)) as primary aux loss; pairwise cTB as diagnostic.
    # Verifiable signal only — T+/T- logit, no rubric/judge.
    # See `compute_sdc_gfn_actor_loss` below and `_patch_actor_loss_for_gfn`
    # for the ppo_loss hook injection.
    "GFN_OPSD_CONTRAST": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # RLSD_FAITHFUL_META (R20, Plan iter-3 SURVEY-GROUNDED LOCK, direction B):
    # "RLSD-faithful meta-token credit". Correctness-ONLY reward head — NO
    # meta_penalty: the asymmetric presence-only meta_penalty injects a
    # teacher/presence SIGN term into base_advantages (token_level_rewards),
    # which BREAKS the RLSD sign/magnitude separation on meta tokens (the
    # diagnosed −14pt cause vs E21Rv2). Here base_advantages sign = pure env
    # correctness; the teacher affects ONLY within-trajectory MAGNITUDE via an
    # UN-clipped sign-preserving w_meta in the advantage path (see
    # verl_sdc_utils sdc_mode=="RLSD_FAITHFUL_META" branch). Single teacher
    # (T+ only). Differentiation vs InT/RLSD/OPSD/Stable-GFN: the 4-part
    # invariant scoped to metacognitive control tokens specifically.
    "RLSD_FAITHFUL_META": {
        "funcs": [correctness_reward],
        "weights": [1.0],
        "keys": ["correctness"],
    },
    # STABLE_GFN (R21, Plan TWO-DIRECTION-SPLIT direction A — "Stable-GFlowNet"
    # signal DELIVERY; codex D1). NEW mode, additive, zero-touch to the 5
    # in-flight modes + RLSD_FAITHFUL_META. Reward head MIRRORS
    # GFN_OPSD_CONTRAST (correctness + meta_penalty) so the ONLY single-variable
    # delta vs the GFN baseline is the *delivery* of the teacher signal:
    #   • Advantage plane: sdc_lambda_meta/shared/diff = 0 in the YAML ⇒ the
    #     multiplicative w_meta throttle (the clipped ±20% no-op, diagnosed
    #     cause C1) is fully removed; the meta region receives the PURE env
    #     correctness advantage sign (RLSD invariant intact).
    #   • The teacher signal is delivered ENTIRELY through a Z-free pairwise
    #     contrastive Trajectory Balance aux loss on the actor-loss plane
    #     (sdc_gfn_objective=pairwise_ctb) + frozen_ref baseline +
    #     reward-temperature (target = logR / T_R, student NOT /τ).
    # Hypothesis HA: un-throttled delivery raises meta-conditioned accuracy IFF
    # the delivered signal is already correct (else it AMPLIFIES C3) — hence B
    # gates A; this mode is the A-only / A∘B delivery vehicle.
    "STABLE_GFN": {
        "funcs": [correctness_reward, meta_penalty_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty"],
    },
    # ── C1/C2 NEXT-WAVE ARMS (2026-05-18, EXPERIMENT_PLAN_ARMS.md "CODEX
    # PLAN-REVIEW CONVERGED"). All three entries below are NEW and ADDITIVE:
    # no entry above this comment is touched; existing §8/RLSD modes keep their
    # exact funcs/weights/keys (regression-asserted).
    #
    # ROD_PT2_E21CTRL  (Arm 2 = "Recipe X" additive primary)
    #   Intent     : combine the best self-distill ingredient (R10/ROD_PT
    #                content×position 2-teacher) with the best verifiable-RL
    #                ingredient (E21Rv2 control reward heads), un-throttled
    #                (C1 fix in verl_sdc_utils) + C2-fixed (the adaptive meta
    #                cost below replaces the presence-forced meta_penalty).
    #   Hypothesis : un-clip + C2-fix + E21Rv2 control + 2-teacher => accuracy
    #                >= B* (Arm 1) with adaptive (not forced ~100%) meta.
    #   Validation : 1-step numeric reward sanity + the
    #                ROD_PT2_E21CTRL w_meta branch (verl_sdc_utils) un-clipped
    #                to [1/w_max, w_max] (test_c1c2_arms_smoke.py).
    #   Reward head : correctness + the FOUR E21Rv2 control heads
    #                 (confidence_revision, redirect_execution,
    #                  verify_execution, meta_floor) + meta_count_bonus
    #                 + the C2-fix adaptive meta cost (meta_penalty_adaptive).
    #                 Weights mirror compute_score_confidence_centered's
    #                 E21Rv2 control ratios (corr 1.0, conf_rev 0.35,
    #                 redirect 0.30, verify 0.15, floor 0.5, count 1.0); the
    #                 C2-fix head carries weight 1.0 like the OLD meta_penalty
    #                 head it REPLACES (same magnitude envelope, adaptive sign).
    # 2026-05-22 ROOT-CAUSE FIX (replaces the 2026-05-20 emergency 2-key
    # collapse): the original 7-key design crashed with `AssertionError:
    # GDPO reward key 'confidence_revision' not found in non_tensor_batch`
    # because the agent_reward_loop path pre-fills `rm_scores` and the
    # MetaCotSDCRewardManager.__call__ early-return then dropped the
    # per-key emit (lines 1061-1064). Fix lives in __call__ now; reward
    # design restored to the documented E21Rv2 control × C2-fix product.
    "ROD_PT2_E21CTRL": {
        "funcs": [
            correctness_reward,
            confidence_revision_reward,
            redirect_execution_reward,
            verify_execution_reward,
            confidence_omission_floor,
            meta_count_bonus,
            meta_penalty_adaptive_reward,
        ],
        "weights": [1.0, 0.35, 0.30, 0.15, 0.5, 1.0, 1.0],
        "keys": [
            "correctness",
            "confidence_revision",
            "redirect_execution",
            "verify_execution",
            "meta_floor",
            "meta_count_bonus",
            "meta_penalty_adaptive",
        ],
    },
    # STABLE_GFN_C2FIX  (Arm 3 = STABLE_GFN PRIMARY, un-clip + C2-fix)
    #   Intent     : the SHIP'd STABLE_GFN mechanism (Z-free pairwise cTB,
    #                pairwise_ctb code byte-identical) but paper-grade primary:
    #                drop the presence-forced meta_penalty (C2) and apply the
    #                un-clip (C1) on the (λ=0) advantage plane for parity with
    #                Arm 2's magnitude bound.
    #   Hypothesis : cTB delivers the SD teacher without the C1 multiplicative
    #                throttle and, with C2-fix, reaches accuracy >= B* with
    #                adaptive meta; stability observational vs Arm 2.
    #   Validation : config-only diff vs STABLE_GFN (mechanism code
    #                byte-parity-proven in test_c1c2_arms_smoke.py); reward
    #                head = correctness + C2-fix adaptive (single-variable vs
    #                STABLE_GFN's correctness+meta_penalty: ONLY the meta cost
    #                head swaps presence-forced -> adaptive).
    "STABLE_GFN_C2FIX": {
        "funcs": [correctness_reward, meta_penalty_adaptive_reward],
        "weights": [1.0, 1.0],
        "keys": ["correctness", "meta_penalty_adaptive"],
    },
    # MATCHED_E21RV2  (Arm 1 = B* matched RLVR baseline — mandatory comparator)
    #   Intent     : a same-infra E21Rv2 re-run so any SD-vs-RLVR delta is
    #                attributable to METHOD, not nuisance (resolves C6 reward-
    #                path / C7 baseline-parity). It runs under the IDENTICAL
    #                SDC infra (verl_sdc_e21r_shared base, same SFT init / RL
    #                parquet / reward-path / decoding / eval) as Arms 2 & 3.
    #   Hypothesis : E21Rv2's reward, re-run here, yields baseline B* — MEASURE
    #                it; do NOT assume the stale 81.7. B* is THE comparator;
    #                all Arm-2/3 success is RELATIVE to B*.
    #   Validation : 1030-panel 16k overall+per-bench+meta-emission; this is
    #                the pre-registered non-inferiority anchor (parity fields in
    #                configs/verl_matched_e21rv2_arm1_h200_4x4k.yaml).
    #   Reward head : EXACTLY the E21Rv2 confidence-centered control set
    #                 (compute_score_confidence_centered: correctness +
    #                  confidence_revision + redirect_execution +
    #                  verify_execution + meta_floor + meta_count_bonus),
    #                 SAME funcs/weights as Arm-2's control block — but with
    #                 NO self-distill teacher (this mode is in _VANILLA_MODES:
    #                 _attach_teacher_signals returns early, NO T+/T-/position
    #                 forward) and NO C2-fix head: pure matched RLVR. The ONLY
    #                 reward-head difference vs Arm 2 is the absence of the
    #                 C2-fix adaptive head (Arm 2 = this + 2-teacher SD + C2-fix
    #                 + un-clip), so the bridge ablation is clean.
    # 2026-05-22 ROOT-CAUSE FIX (replaces 2026-05-20 emergency 2-key):
    # see ROD_PT2_E21CTRL note above. Arm 1 is the matched-RLVR comparator
    # using the E21Rv2 6-control reward set, no teacher forward.
    "MATCHED_E21RV2": {
        "funcs": [
            correctness_reward,
            confidence_revision_reward,
            redirect_execution_reward,
            verify_execution_reward,
            confidence_omission_floor,
            meta_count_bonus,
        ],
        "weights": [1.0, 0.35, 0.30, 0.15, 0.5, 1.0],
        "keys": [
            "correctness",
            "confidence_revision",
            "redirect_execution",
            "verify_execution",
            "meta_floor",
            "meta_count_bonus",
        ],
    },
    # ── E.9 BCI_RLVR (Binned-Confidence-Injection RLVR, 2026-06-05) ───────────
    # NEW + ADDITIVE: no entry above is touched. correctness (dominant head) +
    # outcome_calibration (proper-scoring Brier on the SEEDED confidence). The
    # binned conf seed is force-placed at response-start by the gated rollout
    # wrap (SDCRayPPOTrainer._bci_generate_sequences, flag
    # algorithm.sdc_force_inject_conf) so every problem is attempted across the
    # full confidence range and the proper-score always has signal to select.
    # GDPO per-reward normalization keeps correctness dominant (accuracy
    # preserved). Both funcs are already imported at the top of this file.
    # See docs/superpowers/specs/2026-06-05-e9-...-design.md.
    "BCI_RLVR": {
        "funcs": [correctness_reward, outcome_calibration_reward],
        "weights": [1.0, 0.5],
        "keys": ["correctness", "outcome_calibration"],
    },
    # TRIOBJ_META_V1 (ADDITIVE, env-reward-only): tri-objective GDPO multi-head,
    # sequence-level, mirrors the proven BCI_RLVR template (correctness dominant +
    # auxiliary heads, no teacher forward). Heads:
    #   1) correctness_reward (w=1.0) — final (last-boxed) answer; protects accuracy.
    #   2) meta_revision_utility_reward (w=0.5) — NEW; two-sided, OUTCOME-GATED
    #      credit for the CAUSAL effect of the preliminary->final revision
    #      (wrong->right +1, right->wrong -1, right->right+genuine-meta +0.15,
    #      over-check -0.1, both-wrong/one-box 0); clipped to [-1,1].
    #   3) meta_commit_shape_reward (w=0.3) — existing anti-decoherence (box/commit +
    #      decoherence penalty) to prevent the 16k LaTeX-spam truncation seen in inject.
    # NO sequence-level calibration head in v1 (it caused the inject gradient-conflict;
    # calibration-done-right = DCPO token-mask = v2). All three funcs are imported at
    # the top of this file. GDPO per-head normalization keeps correctness dominant.
    "TRIOBJ_META_V1": {
        "funcs": [correctness_reward, meta_revision_utility_reward, meta_commit_shape_reward],
        "weights": [1.0, 0.5, 0.3],
        "keys": ["correctness", "meta_revision_utility", "meta_commit_shape"],
    },
    # TRIOBJ_DCPO_V2 (ADDITIVE, env-reward-only, region-routed): EXACTLY 3 heads,
    # each group-normalized INDEPENDENTLY and masked onto its OWN token span by
    # _compute_dcpo_region_advantage (verl_sdc_utils). The "weights" here carry the
    # w_corr/w_meta/w_cal routing weights (1.0/0.5/0.3); the advantage path applies
    # them per-region rather than as a summed scalar. No teacher forward (joins
    # _REGION_ROUTED_MODES). KL/entropy disabled in the yaml (§2.6). The 3 funcs are
    # thin wrappers over dcpo_region_rewards (read the per-batch DCPO stash). See
    # docs/superpowers/specs/2026-06-09-dcpo-3region-design.md.
    "TRIOBJ_DCPO_V2": {
        "funcs": [correctness_region_reward, meta_region_utility_reward, cal_region_reward],
        "weights": [1.0, 0.5, 0.3],
        "keys": ["correctness", "meta_region_utility", "cal_region_reward"],
    },
    # TRIOBJ_DCPO_V3 (ADDITIVE, env-reward-only, region-routed): IDENTICAL wiring to
    # TRIOBJ_DCPO_V2 (same 3 region heads / same advantage routing). The ONLY change
    # is the DEFINITION of R_meta inside dcpo_region_rewards (transition-table proxy ->
    # causal counterfactual c_with-c_without) plus the `sdc_counterfactual` producer
    # that supplies cf_correct. The reward-head wrappers + masks are shared verbatim.
    # See docs/superpowers/specs/2026-06-09-dcpo-v3-counterfactual-design.md.
    "TRIOBJ_DCPO_V3": {
        # meta_emission is OBSERVABILITY-ONLY (weight 0.0): it never moves the
        # reward; it exists so val-aux/<ds>/meta_emission/mean@1 charts the
        # per-benchmark emission rate. It IS listed (weight 0.0) in the config's
        # gdpo_reward_keys/weights (boot validation requires len==len(funcs));
        # advantage routing reads the 3 region heads BY NAME, so it never routes.
        # format_penalty (w 0.1) is the 4th ROUTED head. v3k three-tier values:
        # +1 wellformed (routed onto FORMAT_OK at the closer) / -1 drift+discard
        # (routed onto FORMAT_VIOLATION) / 0 replaced+truncation+no_meta —
        # compose_dcpo_region_advantage centers ONE head and routes it onto the
        # per-row-disjoint FORMAT_OK ∪ FORMAT_VIOLATION union.
        # FIVE-WAY SYNC RULE (three prior boot/step-1 crashes!): (1) these
        # lists, (2) yaml algorithm.gdpo_reward_keys/gdpo_reward_weights,
        # (3) the populator + both sync __call__ non_tensor/mask writes,
        # (4) compose_dcpo_region_advantage ↔ _compute_dcpo_region_advantage
        # params, (5) build_dcpo_region_masks output keys ↔ the three
        # mask-stack sites MUST stay in lockstep (tests:
        # test_v3_yaml_reward_lists_match_reward_configs,
        # test_populate_writes_every_gdpo_reward_key,
        # test_v3_mask_stack_sites_in_lockstep).
        "funcs": [correctness_region_reward, meta_region_utility_reward, cal_region_reward,
                  meta_emission_reward, format_penalty_reward],
        "weights": [1.0, 0.5, 0.3, 0.0, 0.1],
        "keys": ["correctness", "meta_region_utility", "cal_region_reward", "meta_emission",
                 "format_penalty"],
    },
    # TRIOBJ_DCPO_V4 (ADDITIVE, env-reward-only, region-routed): IDENTICAL head
    # wiring to TRIOBJ_DCPO_V3 (same 5 funcs/keys/weights, same advantage
    # routing, same FIVE-WAY SYNC rule as documented on the V3 entry). The ONLY
    # change is the SOURCE of R_meta: instead of the CF-regeneration delta
    # (sdc_counterfactual=false in the v4 yamls — machinery dormant, NOT
    # deleted), the populator overwrites `meta_region_utility` with the dense
    # likelihood-delta (PMI) head when algorithm.dcpo_rmeta_source == 'pmi'
    # (sign-gated agg of logP_ref(C|prefix+meta) - logP_ref(C|prefix) over the
    # model's OWN post-meta continuation, frozen ref worker at T=1.0).
    # Stage 1 (format-only) sets dcpo_rmeta_source: none + dcpo_w_meta: 0.
    # See docs/superpowers/specs/2026-06-11-dcpo-v4-likelihood-rmeta-design.md.
    "TRIOBJ_DCPO_V4": {
        "funcs": [correctness_region_reward, meta_region_utility_reward, cal_region_reward,
                  meta_emission_reward, format_penalty_reward],
        "weights": [1.0, 0.5, 0.3, 0.0, 0.1],
        "keys": ["correctness", "meta_region_utility", "cal_region_reward", "meta_emission",
                 "format_penalty"],
    },
}

# Modes that do NOT compute teacher forward (env reward only).
# MATCHED_E21RV2 (Arm 1, ADDITIVE): a teacher-free matched-RLVR baseline —
# joins the no-teacher-forward set so _attach_teacher_signals returns early
# (no T+/T-/position forward), exactly like VANILLA_GRPO. The advantage path
# early-return in verl_sdc_utils was extended with a matching OR-clause.
# VANILLA_GRPO membership/behaviour is unchanged (set still contains it).
_VANILLA_MODES = {"VANILLA_GRPO", "MATCHED_E21RV2", "BCI_RLVR", "TRIOBJ_META_V1"}
# TRIOBJ_DCPO_V2 (ADDITIVE): region-routed, env-reward-only mode. It is NOT in
# _VANILLA_MODES (that set is left byte-identical) but it is teacher-FREE: the
# _attach_teacher_signals short-circuit and the verl_sdc_utils advantage branch
# both gate on this set, so no T+/T-/position forward runs and the per-region
# advantage path (_compute_dcpo_region_advantage) is used instead of the summed
# GDPO whiten. Membership of every pre-existing mode is unchanged.
_REGION_ROUTED_MODES = {"TRIOBJ_DCPO_V2", "TRIOBJ_DCPO_V3", "TRIOBJ_DCPO_V4"}
# TRIOBJ_DCPO_V4 (ADDITIVE): V4 reuses the v3/v3k/v3m format machinery VERBATIM
# (clamp/gate, three-tier classes, FORMAT_VIOLATION/OK stacks, head/floor
# membership) — this set gates all of it at the three mask-stack sites + the
# tier-1 replacement wrap. V2 stays outside (KARPATHY lock "v2 byte-identical");
# V3 membership keeps every v3 path byte-identical (the predicate is still true).
_DCPO_V3_FMT_MODES = {"TRIOBJ_DCPO_V3", "TRIOBJ_DCPO_V4"}
# BCI_RLVR (E.9, ADDITIVE): a NO-teacher env-reward-only mode (correctness +
# outcome_calibration; sdc_enabled=false). It joins the teacher-free set so
# _attach_teacher_signals returns early (no T+/T-/position forward) exactly like
# VANILLA_GRPO. Membership of every pre-existing mode is unchanged. The matching
# advantage-path early-return (verl_sdc_utils.compute_sdc_gdpo_advantage) carries
# the same additive OR-clause for sdc_mode=="BCI_RLVR".
# Modes that compute T+ forward only (single-teacher RLSD).
# ROD_PT: R5 + position teacher (decoy off, natural emit, multiplicative w_position)
# ROD_PT_DEGEN: ROD_PT + degeneration_penalty reward head (R16, 2026-05-12)
# ROD_MQ: meta-quality factor on extended meta region, T+ − student only (R18a)
# RLSD_FAITHFUL_META: R20 direction B. T+ only (gold-blind teacher → MAGNITUDE
#   only). NO meta_penalty head (see REWARD_CONFIGS). Restores the RLSD
#   sign/magnitude invariant on meta tokens that ROD_* break via clip+penalty.
# ROD_PT2_E21CTRL: Arm-2 additive primary. Same T+ + position-teacher forward
#   pair as ROD_PT (content×position 2-teacher), decoy OFF (single-teacher), so
#   it joins this set AND the ROD_PT position-teacher branch below. ADDITIVE:
#   does not change membership semantics of any pre-existing mode.
_SINGLE_TEACHER_MODES = {"RLSD_META_ATTR", "OPSD_META", "ROD_PT", "ROD_PT_DEGEN", "ROD_MQ", "RLSD_FAITHFUL_META", "ROD_PT2_E21CTRL"}
# Modes that compute T+ AND T− forward (contrastive RLSD).
# ROD_MQ_CONTRAST: R18a + T+ − T− contrast term mixed via α/β (R18b)
_CONTRASTIVE_MODES = {
    "SDC_SHARED",
    "SDC_CORR_ONLY",
    "SDC_CORR_META_PEN",
    "RLSD_META_CONTRAST",
    "ROD_MQ_CONTRAST",
    # ROD_MQ_CONTRAST_INJECT (CTSD Phase C): == ROD_MQ_CONTRAST advantage math,
    # so it also needs T+ AND T- forward for the q_contrast term.
    "ROD_MQ_CONTRAST_INJECT",
    # GFN_OPSD_CONTRAST (R18c, Plan v7.2.7 codex r12-r14 LOCK):
    # needs T+ AND T- forward to compute logR = α(T+−P_S.detach()) + β(T+−T-).
    "GFN_OPSD_CONTRAST",
    # STABLE_GFN (R21, direction A): same logR = α(T+−P_S.detach()) + β(T+−T-)
    # over the meta region as GFN_OPSD_CONTRAST → needs T+ AND T- forward.
    "STABLE_GFN",
    # STABLE_GFN_C2FIX (Arm 3, 2026-05-18): identical teacher-forward topology
    # to STABLE_GFN (same logR; needs T+ AND T-). The ONLY delta is the reward
    # head (C2-fix adaptive vs presence-forced meta_penalty) + the un-clip on
    # the λ=0 advantage plane — neither affects which teacher forwards run.
    "STABLE_GFN_C2FIX",
}
# Modes that prepend V0 student prefix + forced <|meta|> to teacher conditioning,
# AND require student rollout to start inside meta (via custom agent loop).
# These also do BOTH T+ and T− forwards (same as contrastive). See verl_sdc_utils
# build_sdc_region_masks for the started_inside_meta plumbing this enables.
_FORCED_META_MODES = {"RLSD_FORCED_META"}

_ACTIVE_SDC_CONTEXT = {"trainer": None, "tokenizer": None, "mode": "SDC_SHARED"}
# One-shot guard so the pairwise_ctb "0 usable uid groups" warning prints once
# per process instead of every degenerate microbatch (codex review pt.4).
_CTB_INACTIVE_WARNED = {"done": False}

# ── Arm-2 parameterized teacher-prompt slot (deliverable #2) ───────────────
# The gold-conditioned TEACHER prompt is the SD quality lever (codex R4 #2:
# prompt-strengthening, NOT a gameable reward head). It is a PARAMETERIZED
# slot keyed by `algorithm.sdc_teacher_prompt_set`, defaulting to
# "r10v2_baseline". The G2 probe (scripts/prompt_probe_g2.py) decides the
# FROZEN value later; until then the default is the live R10v2 system prompt
# (zero behavioural change vs ROD_PT — the baseline teacher prompt is the
# identity wrapper, so ROD_PT2_E21CTRL@baseline reproduces ROD_PT teacher
# conditioning exactly aside from the documented reward-head/un-clip deltas).
#
# Single source of truth = the G2 prompt JSONs under scripts/prompts/ (so a
# strengthened set cannot silently diverge between the probe and training).
# The slot ONLY ever applies to mode=ROD_PT2_E21CTRL; every pre-existing mode
# never reads it (guarded in _build_teacher_logprob_batch), so existing
# teacher conditioning is byte-identical.
#
# ── G2 PER-TEACHER SLOTS (deliverable #2, 2026-05-19) ──────────────────────
# The single shared slot above is REPLACED (additively) by TWO independent
# slots so the POSITION teacher and the CONTENT (T+) teacher can carry
# different gold-conditioned prompts (EXPERIMENT_PLAN_ARMS.md "G2 PER-TEACHER
# PROMPT SPEC — CODEX CONVERGED ... 2026-05-19"):
#   * algorithm.sdc_position_teacher_prompt_set  -> position-teacher logP(<|meta|>)
#   * algorithm.sdc_content_teacher_prompt_set   -> content-teacher T+ region
# Each accepts: "r10v2_baseline" (default -> "" identity prefix, byte-identical
# to the SHIP'd ROD_PT teacher conditioning) | "pos_teacher_v1" |
# "content_teacher_v1" (the frozen G2 spec prompts) | the legacy
# "strengthened_v*" sets (back-compat). PRECEDENCE: a per-teacher key, when
# explicitly set, overrides the legacy shared `sdc_teacher_prompt_set`; if a
# per-teacher key is left unset it INHERITS the legacy shared value (so the
# pre-existing single-slot configs keep working byte-identically). Both
# defaulting to "r10v2_baseline" => "" prefix for BOTH teachers => identical
# to current Arm-2 behaviour. Slots ONLY ever apply to mode=ROD_PT2_E21CTRL.
_TEACHER_PROMPT_SETS = (
    "r10v2_baseline",
    "strengthened_v1",
    "strengthened_v2",
    "pos_teacher_v1",
    "content_teacher_v1",
)

# ── E.4 self-distill contrast variants (plan_ctsd_E4_selfdistill_rl) ───────────
# `sdc_contrast_variant ∈ {decoy, stance, conf}` selects how the T+ / T- teacher
# CONTEXTS differ in ROD_MQ_CONTRAST. The contrast q = T+ − T- (over the meta
# region) and the β-mix are UNCHANGED (verl_sdc_utils:445-452) — only the two
# context strings differ:
#   decoy (DEFAULT, byte-identical to ship): T+ = prompt+gold, T- = prompt+decoy.
#   stance: BOTH sides gold (answer cancels in T+−T−); T+ gets the CAUTIOUS
#           suffix, T- the CONFIDENT suffix → isolates the verify-process axis.
#   conf:   BOTH sides gold; T+ "confidence: 0.15", T- "confidence: 0.95" →
#           isolates the verbalized-confidence axis (anti-overconfidence / ECE).
# CAUTIOUS_INSTR / CONFIDENT_INSTR are copied BYTE-IDENTICALLY from
# experiments/probes/e2_contrastive_steering.py:111-113 so the RL teacher uses
# the EXACT E.3-validated steering strings (context consistency). The join
# patterns below (" (answer is {g}) " gold marker + leading-space suffix join)
# also mirror e2's CONTRASTS registry (e2:135-139) verbatim.
_CONTRAST_VARIANTS = ("decoy", "stance", "conf", "conf_free")
CAUTIOUS_INSTR = ("Reason cautiously: question whether your current approach is right, verify each "
                  "step with an alternative method, avoid premature confidence.")
CONFIDENT_INSTR = ("Reason decisively: commit to your current approach with confidence and proceed.")


def _resolve_teacher_prompt_prefix(prompt_set: str) -> str:
    """Return the teacher-instruction PREFIX text for a prompt set.

    r10v2_baseline -> "" (identity: teacher conditioning unchanged vs ROD_PT;
                         the baseline prompt is the live system prompt the
                         policy already uses, NOT an extra teacher wrapper).
    strengthened_v* -> the literal `system_prompt` from the matching FROZEN
                       G2 prompt JSON, prepended (TRAINING-ONLY teacher
                       conditioning; never enters policy inference inputs —
                       guarded by being applied only inside the teacher
                       log-prob batch builder).
    """
    if prompt_set == "r10v2_baseline":
        return ""
    if prompt_set not in _TEACHER_PROMPT_SETS:
        raise ValueError(
            f"sdc_teacher_prompt_set={prompt_set!r} not in {_TEACHER_PROMPT_SETS}"
        )
    import json as _json

    pj = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "prompts"
        / f"{prompt_set}.json"
    )
    if not pj.exists():
        raise FileNotFoundError(
            f"teacher prompt set {prompt_set!r} -> {pj} missing (the G2 "
            "frozen prompt JSON is the single source of truth)"
        )
    spec = _json.loads(pj.read_text(encoding="utf-8"))
    sp = spec.get("system_prompt")
    if not isinstance(sp, str) or not sp.strip():
        raise ValueError(f"{pj.name}: empty/non-literal system_prompt")
    # Prepended as a teacher-only instruction block, clearly delimited.
    return f"{sp}\n\n"


def reward_loop_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Fallback scalar score for veRL agent-loop reward workers.

    Mode-aware: emits the GDPO reward keys appropriate to the active mode.
      • VANILLA_GRPO, SDC_CORR_ONLY  → correctness only
      • RLSD_META_ATTR / RLSD_META_CONTRAST / OPSD_META / SDC_CORR_META_PEN
                                     → correctness + meta_penalty
      • SDC_SHARED                   → all 5 legacy heads
                                       (correctness, outcome_calibration,
                                        meta_structure, meta_commit_shape,
                                        postmeta_closure)

    Always populates `correctness` and `meta_penalty` for backward compat with
    callers that read those keys directly; mode-specific extra heads are added
    for SDC_SHARED to avoid breaking the legacy 5-head config when async
    rollout is enabled.
    """
    completion = [[{"content": solution_str}]]
    gt = [ground_truth]

    def _safe_call(fn, with_gt=True):
        try:
            return float(fn(completion, gt)[0]) if with_gt else float(fn(completion)[0])
        except Exception:
            return 0.0

    correctness = _safe_call(correctness_reward, with_gt=True)
    meta_pen = _safe_call(meta_penalty_reward, with_gt=False)

    out = {
        "score": float(correctness + meta_pen),
        "correctness": correctness,
        "meta_penalty": meta_pen,
        "data_source": data_source or "",
    }

    mode = _ACTIVE_SDC_CONTEXT.get("mode", "SDC_SHARED")
    # R16 fix: Ray RewardLoopWorker actors do NOT inherit module-level
    # `_ACTIVE_SDC_CONTEXT["mode"]` from the trainer process. Module state is
    # per-actor. So the mode-conditional emit pattern below would silently skip
    # degeneration_penalty in async-rollout workers, which then fails the
    # GDPO assertion that demands the key exist in non_tensor_batch.
    # Fix: ALWAYS emit degeneration_penalty as a sample-level scalar. The
    # GDPO weight (configured in YAML) is 0 for non-R16 modes so this is a
    # safe no-op, and 0.3 for ROD_PT_DEGEN where it provides the intended signal.
    try:
        tok = _ACTIVE_SDC_CONTEXT.get("tokenizer")
        if tok is not None:
            try:
                length = len(tok.encode(solution_str, add_special_tokens=False))
            except Exception:
                length = len(solution_str.split())
        else:
            length = len(solution_str.split())
        from src.training.rewards import _extract_answer_fallback as _ext
        ans = _ext(solution_str)
        degen, _ = compute_degeneration_penalty(solution_str, length, ans)
        out["degeneration_penalty"] = float(degen)
        if mode == "ROD_PT_DEGEN":
            out["score"] = float(out["score"] + degen)
    except Exception:
        out["degeneration_penalty"] = 0.0

    # TRIOBJ_META_V1 (always-emit, same R16 robustness pattern as degeneration_penalty):
    # Ray RewardLoopWorker actors don't inherit the trainer's mode, so emit the
    # meta_revision_utility key unconditionally. Its GDPO weight is 0 for every other
    # mode (it's absent from their REWARD_CONFIGS keys), so this is a safe no-op there
    # and provides the signal for TRIOBJ_META_V1.
    out["meta_revision_utility"] = _safe_call(meta_revision_utility_reward, with_gt=True)

    # TRIOBJ_DCPO_V2 (always-emit placeholder, same R16 robustness pattern):
    # `meta_region_utility` / `cal_region_reward` are the DCPO GDPO reward keys.
    # `meta_region_utility` is GROUP-dependent (uses the group p_hat) so it CANNOT
    # be computed here per-rollout — emit 0.0 as a safety placeholder so the key is
    # never missing in any async path. The AUTHORITATIVE group-aware values are
    # written in the main process by `_populate_dcpo_region_keys` (called from
    # `_attach_teacher_signals`) and OVERWRITE these placeholders before the GDPO
    # advantage/assertion runs. `correctness` is already emitted above (group-free).
    # GDPO weight for these keys is 0 in every other mode (absent from their
    # REWARD_CONFIGS keys), so emitting them is a safe no-op everywhere else.
    out["meta_region_utility"] = 0.0
    out["cal_region_reward"] = 0.0

    if mode == "SDC_SHARED":
        # Restore the 5-head legacy contract so multi_turn / async rollout
        # paths don't crash on missing GDPO reward keys.
        out["outcome_calibration"] = _safe_call(outcome_calibration_reward, with_gt=True)
        out["meta_structure"] = _safe_call(meta_structure_reward, with_gt=False)
        out["meta_commit_shape"] = _safe_call(meta_commit_shape_reward, with_gt=False)
        from src.training.verl_sdc_utils import postmeta_closure_reward as _pcr
        out["postmeta_closure"] = _safe_call(_pcr, with_gt=False)

    # 2026-05-22: SAME R16 pattern (line 530-537) applied to the Arm-2
    # ROD_PT2_E21CTRL stabilizer set. Ray RewardLoopWorker actors do not
    # inherit `_ACTIVE_SDC_CONTEXT["mode"]` from the trainer process; the
    # in-actor mode defaults to "SDC_SHARED", so a mode-conditional emit for
    # ROD_PT2_E21CTRL would silently skip the 5 stabilizer keys (and
    # `meta_penalty_adaptive`) — that is the exact mechanism behind the
    # 2026-05-20 Arm-2 GDPO `AssertionError: GDPO reward key
    # 'confidence_revision' not found in non_tensor_batch` and the
    # downstream delimiter-spam collapse (50.9% / 108 empty meta blocks).
    # ALWAYS emit so async-rollout RewardLoopWorker actors honour the
    # Arm-2 reward-key contract regardless of in-actor module state. For
    # non-Arm-2 modes the GDPO weight is 0 (key not in gdpo_reward_keys),
    # so this is a safe no-op everywhere else.
    out["confidence_revision"] = _safe_call(confidence_revision_reward, with_gt=False)
    out["redirect_execution"] = _safe_call(redirect_execution_reward, with_gt=False)
    out["verify_execution"] = _safe_call(verify_execution_reward, with_gt=False)
    out["meta_floor"] = _safe_call(confidence_omission_floor, with_gt=False)
    out["meta_count_bonus"] = _safe_call(meta_count_bonus, with_gt=False)
    out["meta_penalty_adaptive"] = _safe_call(meta_penalty_adaptive_reward, with_gt=True)

    return out


def _is_gdpo_estimator(adv_estimator) -> bool:
    try:
        from verl.trainer.ppo.core_algos import AdvantageEstimator
    except Exception:
        AdvantageEstimator = None

    if adv_estimator == "gdpo":
        return True
    if AdvantageEstimator is not None and adv_estimator == AdvantageEstimator.GDPO:
        return True
    return False


def _decode_response(tokenizer, prompt_ids, response_ids, attention_mask, prompt_length: int) -> tuple[str, torch.Tensor]:
    # Decode ONLY the response tokens — never the prompt.
    # Why: reward heads pattern-match on \boxed{}, <|meta|>, "the answer is", etc.
    # If the prompt contains any such substring (few-shot example, retrieved
    # problem text, template boilerplate), returning prompt+response here leaks
    # that content into every reward and silently inflates/deflates signals.
    valid_response_length = attention_mask[prompt_length:].sum().item()
    valid_response_ids = response_ids[: int(valid_response_length)]
    text = tokenizer.decode(valid_response_ids, skip_special_tokens=False)
    return text, valid_response_ids


def _decode_prompt_only(tokenizer, prompt_ids, attention_mask, prompt_length: int) -> str:
    valid_prompt_length = attention_mask[:prompt_length].sum().item()
    valid_prompt_ids = prompt_ids[-int(valid_prompt_length):]
    return tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)


# ─── V0 prefix cache (RLSD_FORCED_META) ─────────────────────────────────────
# Why a module-level cache: every batch in a step calls _attach_teacher_signals
# once, but each unique prompt may appear under multiple uids. Caching by
# (global_steps, prompt_hash) means we generate V0 prefixes at most once per
# unique prompt per step. Cache is cleared whenever ``global_steps`` advances
# so we never serve a stale prefix from a previous policy.
_V0_PREFIX_CACHE: dict[tuple[int, str], str] = {}
_V0_PREFIX_LAST_STEP: list[int] = [-1]  # mutable holder, avoids `global` decl


def _generate_v0_prefixes(
    trainer,
    tokenizer,
    prompt_texts: list[str],
    global_steps: int,
    max_new_tokens: int = 256,
    max_chars: int = 1500,
) -> dict[str, str]:
    """Generate (or retrieve cached) V0 student-prefix strings for forced-meta mode.

    For RLSD_FORCED_META the teacher conditioning is augmented with the V0
    student's first attempt followed by the gold answer + ``<|meta|>``. This
    grounds the teacher distribution on a contextualized "rethink-after-attempt"
    rather than a synthetic continuation from prompt+answer alone.

    TODO(v0): The actual V0 generation path requires invoking the rollout
        manager mid-step (e.g., ``trainer.async_rollout_manager.generate_sequences``
        with ``meta_info["validate"]=True``). That is non-trivial because the
        rollout workers may be busy with the current generate call. To keep
        this plug-and-play and avoid deadlocks, this initial implementation
        falls back to a constant ``"(no prior attempt)"`` prefix for every
        prompt — which still validates the core hypothesis (force <|meta|> +
        gold context) without V0. Wire real V0 generation later as a follow-up
        after smoke-testing the forced <|meta|> + gold path end-to-end.

    Args:
        trainer: SDCRayPPOTrainer instance (used by future V0 generation).
        tokenizer: Tokenizer (currently unused; reserved for future stripping).
        prompt_texts: List of unique prompt strings to generate prefixes for.
        global_steps: Current trainer step. Cache is invalidated on step change
            so we don't reuse last-step prefixes after the policy has shifted.
        max_new_tokens: Future V0 generation cap (unused in fallback path).
        max_chars: Future cap on stripped prefix length (unused in fallback path).

    Returns:
        dict mapping each input prompt_text → its V0 prefix string. Always
        returns one entry per input.
    """
    # Step transition: drop stale entries to bound memory + avoid reusing
    # prefixes that no longer reflect the current policy's V0 behavior.
    if _V0_PREFIX_LAST_STEP[0] != global_steps:
        _V0_PREFIX_CACHE.clear()
        _V0_PREFIX_LAST_STEP[0] = global_steps

    out: dict[str, str] = {}
    missing: list[str] = []
    for pt in prompt_texts:
        # Hash by prompt prefix-16 of MD5 — collisions are vanishingly unlikely
        # within a single training step's unique prompt set.
        h = hashlib.md5(pt.encode("utf-8", errors="replace")).hexdigest()[:16]
        cache_key = (global_steps, h)
        if cache_key in _V0_PREFIX_CACHE:
            out[pt] = _V0_PREFIX_CACHE[cache_key]
        else:
            missing.append(pt)

    # ── Fallback path (TODO(v0)): no real generation; populate placeholder. ──
    # See module docstring above for rationale. This is intentional and safe:
    # the meta_info "(no prior attempt)" is non-empty, contains no <|meta|>
    # token, and is short — so downstream stripping/capping logic is a no-op
    # for it but still exercises the same code path that real V0 prefixes will.
    fallback_str = "(no prior attempt)"
    for pt in missing:
        h = hashlib.md5(pt.encode("utf-8", errors="replace")).hexdigest()[:16]
        cache_key = (global_steps, h)
        prefix = fallback_str

        # Apply the same hygiene we'll need once real V0 is wired up:
        # 1. Strip chat-template markers (in case generation includes them).
        for marker in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
            if marker in prefix:
                prefix = prefix.split(marker, 1)[0]
        # 2. Slice at first <|meta|> if present (real V0 may emit one).
        if META_START in prefix:
            prefix = prefix.split(META_START, 1)[0]
        # 3. Cap length from the END (most recent reasoning is most informative).
        if len(prefix) > max_chars:
            prefix = prefix[-max_chars:]
        prefix = prefix.strip()
        if not prefix:
            prefix = fallback_str

        _V0_PREFIX_CACHE[cache_key] = prefix
        out[pt] = prefix

    return out


def _build_teacher_logprob_batch(
    *,
    tokenizer,
    prompt_texts: list[str],
    answer_texts: list[str],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    v0_prefixes: dict[str, str] | None = None,
    forced_meta: bool = False,
    teacher_role: str = "content",
    contrast_side: str = "pos",
):
    # Arm-2 ONLY: a TRAINING-ONLY gold-conditioned teacher-prompt prefix
    # resolved from the PER-TEACHER parameterized slots (deliverable #2,
    # 2026-05-19). `teacher_role` selects which resolved prefix applies:
    #   "content"  -> the content-teacher (T+) slot
    #                 (_ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_prefix"])
    #   "position" -> the position-teacher logP(<|meta|>) slot
    #                 (_ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_prefix"])
    # For EVERY pre-existing mode this is the empty string (the context keys
    # are absent / "r10v2_baseline" -> ""), so the teacher prompt below is
    # byte-identical to the SHIP'd code path. Unknown roles fall back to "".
    _tp_prefix = ""
    if _ACTIVE_SDC_CONTEXT.get("mode") == "ROD_PT2_E21CTRL":
        if teacher_role == "position":
            _tp_prefix = str(
                _ACTIVE_SDC_CONTEXT.get(
                    "sdc_position_teacher_prompt_prefix", ""
                )
            )
        else:  # "content" (default; also covers forced-meta T+ path)
            _tp_prefix = str(
                _ACTIVE_SDC_CONTEXT.get(
                    "sdc_content_teacher_prompt_prefix", ""
                )
            )

    prompt_ids_list = []
    seq_lens = []
    for prompt_text, answer_text in zip(prompt_texts, answer_texts):
        if forced_meta:
            # RLSD_FORCED_META teacher conditioning: prepend the V0 student
            # prefix (when available) and a "(The correct answer is X.)" hint,
            # then force a <|meta|> opening so teacher log-prob is conditioned
            # on the same "start inside meta" state the student rollout sees.
            # Layout: <prompt> <V0_prefix>?\n(The correct answer is <gold>.)\n<|meta|>
            #
            # CRITICAL: prompt_text comes from _decode_prompt_only() of the
            # rollout's input_ids. The agent loop (ForcedMetaAgentLoop) has
            # already appended <|meta|> token id to prompt_ids → prompt_text
            # ends with META_START. Strip it before re-appending the suffix
            # so the final teacher prompt has EXACTLY ONE <|meta|> at the end
            # (matches inspect_forced_meta.py S3 verified format byte-for-byte).
            base = prompt_text
            if base.endswith(META_START):
                base = base[: -len(META_START)]
            v0 = (v0_prefixes.get(prompt_text, "") if v0_prefixes else "").rstrip()
            sep = "\n" if v0 else ""
            teacher_prompt = (
                f"{base}{v0}{sep}(The correct answer is {answer_text}.)\n{META_START}"
            )
        else:
            # Align teacher conditioning with what the actor actually sees:
            # prompt_text is already the chat-templated prompt (ending in the
            # assistant role marker), so we append the gold/decoy answer directly
            # instead of injecting a synthetic " Answer: " separator the actor
            # never produces. This keeps teacher log-prob on the same conditional
            # distribution the policy is optimizing against.
            #
            # Arm-2 (ROD_PT2_E21CTRL) ONLY: prepend the resolved strengthened
            # teacher-prompt prefix. `_tp_prefix == ""` for every other mode
            # AND for the r10v2_baseline set, so this is the IDENTITY
            # `f"{prompt_text}{answer_text}"` everywhere except a strengthened
            # Arm-2 run — preserving byte-parity for all existing modes.
            #
            # E.4 self-distill contrast variants: `decoy` is the UNCHANGED
            # f-string below (answer slot = gold for T+, decoy for T-, filled by
            # the caller). It is taken whenever `sdc_contrast_variant` is unset
            # or "decoy" → byte-identical for every pre-existing mode/config/test.
            # `stance`/`conf` append a side-specific suffix to a gold-on-BOTH-
            # sides answer marker (answer cancels in T+−T−).
            _cv = _ACTIVE_SDC_CONTEXT.get("sdc_contrast_variant", "decoy")
            if _cv == "decoy":
                teacher_prompt = f"{_tp_prefix}{prompt_text}{answer_text}"
            elif _cv == "stance":
                # e2_contrastive_steering CONTRASTS["gold_stance"] join pattern.
                _sfx = (" " + CAUTIOUS_INSTR) if contrast_side == "pos" else (" " + CONFIDENT_INSTR)
                teacher_prompt = f"{_tp_prefix}{prompt_text} (answer is {answer_text}){_sfx}"
            elif _cv == "conf":
                _sfx = "\nconfidence: 0.15\n" if contrast_side == "pos" else "\nconfidence: 0.95\n"
                teacher_prompt = f"{_tp_prefix}{prompt_text} (answer is {answer_text}){_sfx}"
            elif _cv == "conf_free":
                # E.8 GOLD-FREE conf-down: confidence suffix ONLY, NO answer injected (T+/T-
                # differ only in the confidence level, both gold-free). Combined with
                # mode=GFN_OPSD_CONTRAST this makes the teacher a DISTRIBUTION-MATCHING (listwise
                # KL) target pulling the policy's meta toward the low-conf-conditioned self —
                # genuinely != E.4 (RLSD_META_CONTRAST + conf = gold-conditioned MAGNITUDE
                # reshaping). Gold-free avoids the leakage that kills distribution-matching for
                # gold-conditioned teachers (Self-Distilled RLVR, arXiv 2604.03128).
                _sfx = "\nconfidence: 0.15\n" if contrast_side == "pos" else "\nconfidence: 0.95\n"
                teacher_prompt = f"{_tp_prefix}{prompt_text}{_sfx}"
            else:
                raise ValueError(
                    f"sdc_contrast_variant={_cv!r} not in {_CONTRAST_VARIANTS}"
                )
        ids = tokenizer(teacher_prompt, add_special_tokens=False)["input_ids"]
        prompt_ids_list.append(torch.tensor(ids, dtype=torch.long))
        seq_lens.append(len(ids))

    max_prompt_len = max(seq_lens) if seq_lens else 0
    response_len = responses.size(1)
    batch_size = responses.size(0)
    total_len = max_prompt_len + response_len

    input_ids = torch.zeros(batch_size, total_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long)
    position_ids = torch.zeros(batch_size, total_len, dtype=torch.long)
    response_mask_full = torch.zeros(batch_size, total_len, dtype=torch.long)
    # prompts/responses split required by verl 0.7.1 left_right_2_no_padding.
    prompts_padded = torch.zeros(batch_size, max_prompt_len, dtype=torch.long)
    prompts_attn = torch.zeros(batch_size, max_prompt_len, dtype=torch.long)

    for i in range(batch_size):
        p = prompt_ids_list[i]
        p_len = p.numel()
        r_mask = response_mask[i].long()
        r_ids = responses[i].long()
        input_ids[i, :p_len] = p
        attention_mask[i, :p_len] = 1
        valid_r = int(r_mask.sum().item())
        if valid_r > 0:
            input_ids[i, p_len : p_len + response_len] = r_ids
            attention_mask[i, p_len : p_len + valid_r] = 1
            response_mask_full[i, p_len : p_len + response_len] = r_mask
        position_ids[i] = torch.arange(total_len, dtype=torch.long)
        # left-pad each prompt to max_prompt_len so verl's prompt-side accessors work
        prompts_padded[i, max_prompt_len - p_len : max_prompt_len] = p
        prompts_attn[i, max_prompt_len - p_len : max_prompt_len] = 1

    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask_full,
            "position_ids": position_ids,
            "prompts": prompts_padded,
            "responses": responses.long(),
        }
    )


# ── TRIOBJ_DCPO_V4 dense likelihood-delta (PMI) R_meta scoring ───────────────
# Modeled on _build_teacher_logprob_batch (the ref-scoring custom-batch
# precedent) but with the verl-STANDARD tensor layout: the precedent writes
# input_ids LEFT-ALIGNED (prompt at cols [0,p_len), response at p_len) while
# verl 0.7.1's no_padding_2_padding computes prompt/response lengths from
# attention_mask split AT COLUMN P_max — so any row whose prompt is shorter
# than the batch max gets its response logprobs silently SHIFTED LEFT (latent
# C3-class bug, API-scout finding). Here the prompt is left-padded INTO the
# full tensor ([P_max-p_len, P_max)) and the response starts exactly at P_max,
# so ref_log_prob[i, t] aligns with responses[i, t] for EVERY row.
def _build_pmi_score_batches(prompt_ids_list, response_ids_list, pad_to_multiple: int = 1):
    """Build the ref-worker scoring tensors for the 2n PMI arm rows.

    Each row is one ARM of one scored rollout: prompt = everything before the
    shared C-span (prefix [+ meta]), response = the C-span token ids themselves
    (identical between the two arms of a rollout by splice_and_align's
    token-id-identity contract), so ref_log_prob[i, t] IS
    logP_ref(C_t | arm-context + C_<t) with no slicing arithmetic.

    Rows are padded to a multiple of `pad_to_multiple` (dp_size x ref
    micro-batch, verl dispatch divisibility — same duplicate-row-0 trick as the
    position-teacher subset batch); the caller reads only the first `real_n`
    rows of the result.

    Returns (tensors dict — input_ids / attention_mask / response_mask /
    position_ids / prompts / responses, all verl-standard — , real_n). The
    caller wraps DataProto.from_dict + meta_info; returning the plain dict
    keeps the layout unit-testable without verl.
    """
    real_n = len(prompt_ids_list)
    assert real_n == len(response_ids_list) and real_n > 0
    rows = list(zip(prompt_ids_list, response_ids_list))
    pad_n = (-real_n) % max(1, int(pad_to_multiple))
    rows += [rows[0]] * pad_n
    n = len(rows)
    p_max = max(len(p) for p, _ in rows)
    r_max = max(len(r) for _, r in rows)
    total = p_max + r_max

    input_ids = torch.zeros(n, total, dtype=torch.long)
    attention_mask = torch.zeros(n, total, dtype=torch.long)
    response_mask_full = torch.zeros(n, total, dtype=torch.long)
    for i, (p, r) in enumerate(rows):
        p_len, r_len = len(p), len(r)
        # prompt left-padded INTO the full tensor; response at column p_max;
        # attention contiguous across the p_max boundary (verl convention).
        input_ids[i, p_max - p_len : p_max] = torch.as_tensor(p, dtype=torch.long)
        input_ids[i, p_max : p_max + r_len] = torch.as_tensor(r, dtype=torch.long)
        attention_mask[i, p_max - p_len : p_max + r_len] = 1
        response_mask_full[i, p_max : p_max + r_len] = 1
    # verl position convention (NOT the precedent's arange, which is only valid
    # for its packed layout): positions count VALID tokens, pads clamp to 0.
    position_ids = torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

    return (
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask_full,
            "position_ids": position_ids,
            "prompts": input_ids[:, :p_max],
            "responses": input_ids[:, p_max:],
        },
        real_n,
    )


def _dcpo_v4_ref_logprobs(trainer, tensors):
    """Score the PMI arm rows on the FROZEN ref worker. T=1.0 HARDCODED (review
    M1): the precedent inherits rollout.temperature (0.6), which compresses the
    PMI delta by 1/T — the v4 scorer must NOT copy that line."""
    # M1 runtime guard (review round 1): the meta_info temperature below only
    # survives on the ENGINE worker path. verl 0.7.1's LEGACY fsdp worker
    # overwrites data.meta_info["temperature"] with rollout.temperature (0.6)
    # AFTER this caller sets it — a silent 1/T (~1.67x) PMI compression that is
    # invisible in the dcpo/pmi_* scalars. Both v4 yamls inherit
    # trainer.use_legacy_worker_impl: disable from verl_sdc_e21r_shared.yaml,
    # but a base-config change or a standalone yaml copy must CRASH step 1
    # here instead of training on compressed deltas (fail-closed: unreadable
    # config also raises).
    try:
        _legacy = str(trainer.config.trainer.use_legacy_worker_impl)
    except Exception:
        _legacy = "<unreadable>"
    assert _legacy == "disable", (
        f"v4 PMI requires the engine worker path "
        f"(trainer.use_legacy_worker_impl=disable, got {_legacy!r}): the legacy "
        f"fsdp worker overwrites meta_info['temperature'] with "
        f"rollout.temperature AFTER the caller's T=1.0 (review M1)")
    batch = DataProto.from_dict(tensors=tensors)
    batch.meta_info["temperature"] = 1.0
    out = trainer._compute_ref_log_prob(batch)
    # [i, t] = logP_ref(responses[i, t] | prompt + responses[i, :t]).
    return out.batch["ref_log_prob"]


def _log_pmi_wandb_scalars(step: int, *, attempted_rate: float, aligned_rate: float,
                           guard_hit_rate: float, member_rate: float,
                           nonfinite_rate: float,
                           placebo_fail_rate: float = 0.0) -> None:
    """One wandb point for the dcpo/pmi_* scalars (observability never kills
    training). Round 2 M-C: the early returns in _compute_dcpo_v4_pmi_rmeta call
    this too, so a no-aligned/ref-failure step charts as a ZERO, not a GAP —
    gaps are indistinguishable from logging outages."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "dcpo/pmi_attempted_rate": float(attempted_rate),
                "dcpo/pmi_aligned_rate": float(aligned_rate),
                "dcpo/pmi_guard_hit_rate": float(guard_hit_rate),
                "dcpo/pmi_member_rate": float(member_rate),
                "dcpo/pmi_nonfinite_rate": float(nonfinite_rate),
                "dcpo/pmi_placebo_fail_rate": float(placebo_fail_rate),
            }, step=int(step))
    except Exception:
        pass


def _compute_dcpo_v4_pmi_rmeta(
    *,
    tokenizer,
    trainer,
    prompt_texts: list,
    response_texts: list,
    fmt_classes: list,
    heads: dict,
    read_knob,
    step: int = 0,
):
    """TRIOBJ_DCPO_V4 dense R_meta (spec §2): per trusted meta-bearing row,
    Delta_t = logP_ref(C_t | prefix+meta+C_<t) - logP_ref(C_t | prefix+C_<t)
    over the model's OWN post-meta continuation C, aggregated + overlap-guarded
    + sign-gated by dcpo_pmi (the pure core shared with the offline probe).

    Row eligibility: fmt class TRUSTED (region routing reliable) AND a CLOSED
    first meta block in the text — drift rows (no <|/meta|>; the de-facto closer
    is </think>) are excluded rather than guessing a splice boundary, scoring 0
    with member 0 (conservative under-credit, never misalignment).

    CRASH-SAFE on the engine call (mirror of _dcpo_cf_generate_texts): a ref
    failure prints LOUDLY and returns all-zero R_meta + all-zero membership —
    training continues, and the dcpo/pmi_* scalars are still LOGGED as zeros
    on every early return (round 2 M-C: a zero-flatline is visible on the
    chart; a logging GAP is not).

    Returns (r_meta float32 [B], rmeta_member float32 [B]) — member 1.0 only
    for rows whose PMI was actually computed (aligned + guard-passed), the
    review-I2 centering population.
    """
    B = len(response_texts)
    r_meta = np.zeros(B, dtype=np.float32)
    member = np.zeros(B, dtype=np.float32)

    method = str(read_knob("dcpo_pmi_agg", "sum_clip"))
    topk_frac = float(read_knob("dcpo_pmi_topk_frac", 0.25))
    clip_c_token = float(read_knob("dcpo_pmi_clip_token", 2.0))
    clip_c_gate = float(read_knob("dcpo_pmi_clip_gate", 2.0))
    # RLT (2506.08388) worst-token coefficient for method='mean_min': agg =
    # mean(clip(delta)) + alpha*min(clip(delta)). 0.0 => clipped mean (no-op for
    # other methods). Default OFF so non-mean_min runs stay byte-identical.
    pmi_alpha = float(read_knob("dcpo_pmi_alpha", 0.0))
    ngram_n = int(read_knob("dcpo_pmi_ngram_n", 8))
    ngram_threshold = float(read_knob("dcpo_pmi_ngram_threshold", 0.25))
    # Cross-shuffle amendment (report 2026-06-11 §4.1): subtract the placebo
    # aggregate per row so the generic text-presence component (86% of raw
    # delta) cancels and only the CONTENT increment is rewarded. Third scored
    # arm (prefix + PLACEBO_META + continuation) => ref cost x1.5.
    placebo_correct = bool(read_knob("dcpo_pmi_placebo_correct", False))

    # 1) Select + splice. attempted = (batch_idx, row_dict, splice|None,
    #    placebo_splice|None); rows with splice=None are alignment failures
    #    (scored 0, counted in diag). A row whose PLACEBO splice fails (or
    #    whose placebo without-span diverges from the real one — boundary-drop
    #    divergence) FAILS CLOSED downstream via placebo_alignment_failed.
    attempted: list = []
    for i in range(B):
        if fmt_classes[i] not in TRUSTED_META_CLASSES:
            continue
        # Round 2 M-D: ONE split definition shared with the offline probe
        # (dcpo_pmi.split_first_meta) — None covers no-meta, drift (no
        # <|/meta|>) AND whitespace-only continuations (the stricter probe
        # semantics: nothing to score) — not attempted.
        parts = split_first_meta(response_texts[i])
        if parts is None:
            continue
        response_prefix, meta_text, continuation_text = parts
        prefix_text = (prompt_texts[i] or "") + response_prefix
        row = {
            "meta_text": meta_text,
            "continuation_text": continuation_text,
            "correct": bool(float(heads["c_with"][i]) > 0.5),
            "boxed_answer": (heads["answer"][i] or None),
        }
        try:
            sp = splice_and_align(tokenizer, prefix_text, meta_text, continuation_text)
        except SpliceAlignmentError:
            row["alignment_failed"] = True
            sp = None
        psp = None
        if sp is not None and placebo_correct:
            try:
                psp = splice_and_align(tokenizer, prefix_text, PLACEBO_META,
                                       continuation_text)
                # The placebo arm reuses the REAL without-arm logprobs, which
                # is only valid when both splices located the continuation at
                # the same without-span.
                if psp["c_span_without"] != sp["c_span_without"]:
                    psp = None
            except SpliceAlignmentError:
                psp = None
            if psp is None:
                row["placebo_alignment_failed"] = True
        attempted.append((i, row, sp, psp))
    aligned = [t for t in attempted if t[2] is not None]
    if not aligned:
        if attempted and os.environ.get("DCPO_DEBUG", "1") == "1":
            print(f"[DCPO-V4] pmi step={step}: {len(attempted)} attempted, 0 aligned "
                  f"— all R_meta 0 this batch.", flush=True)
        _log_pmi_wandb_scalars(step, attempted_rate=len(attempted) / max(1, B),
                               aligned_rate=0.0, guard_hit_rate=0.0,
                               member_rate=0.0, nonfinite_rate=0.0)
        return r_meta, member

    # 2) Score both arms on the frozen ref worker (with-arms rows [0, n),
    #    without-arms rows [n, 2n)). Same dispatch-divisibility padding as the
    #    position-teacher batch (dp_size x ref micro-batch, duplicate row 0).
    try:
        nnodes = int(trainer.config.trainer.nnodes)
    except Exception:
        nnodes = 1
    try:
        n_gpus_per_node = int(trainer.config.trainer.n_gpus_per_node)
    except Exception:
        n_gpus_per_node = 4
    try:
        micro_bs = int(trainer.config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu)
    except Exception:
        micro_bs = 4
    pad_unit = nnodes * n_gpus_per_node * micro_bs
    arm_prompts, arm_resps = [], []
    for (_i, _row, sp, _psp) in aligned:
        cs_w, _ = sp["c_span_with"]
        arm_prompts.append(sp["with_ids"][:cs_w])
        arm_resps.append(sp["with_ids"][cs_w:])
    for (_i, _row, sp, _psp) in aligned:
        cs_wo, _ = sp["c_span_without"]
        arm_prompts.append(sp["without_ids"][:cs_wo])
        arm_resps.append(sp["without_ids"][cs_wo:])
    # third arm (placebo_correct only): rows [2n, 2n + n_placebo) — only rows
    # whose placebo splice succeeded; the rest fail closed in compute_pmi_rows.
    placebo_idx: list = []
    if placebo_correct:
        for k, (_i, _row, _sp, psp) in enumerate(aligned):
            if psp is not None:
                placebo_idx.append(k)
                cs_p, _ = psp["c_span_with"]
                arm_prompts.append(psp["with_ids"][:cs_p])
                arm_resps.append(psp["with_ids"][cs_p:])
    tensors, real_n = _build_pmi_score_batches(arm_prompts, arm_resps, pad_unit)
    # bookkeeping invariant (review 2b83bf3 minor-1): the read ranges below
    # assume exactly 2 full arms + the placebo partial arm, in that order.
    assert real_n == 2 * len(aligned) + len(placebo_idx), (
        f"PMI arm bookkeeping broken: {real_n} scored rows != "
        f"2*{len(aligned)} + {len(placebo_idx)}")
    try:
        ref_lp = _dcpo_v4_ref_logprobs(trainer, tensors)
    except AssertionError:
        raise  # M1 config guard: deterministic misconfig must CRASH, not flatline
    except Exception as e:
        print(f"[DCPO-V4] PMI ref scoring FAILED ({type(e).__name__}: {e}) — "
              f"R_meta all-zero this batch (member 0; dcpo/pmi_* charts a zero).",
              flush=True)
        if os.environ.get("DCPO_DEBUG", "1") == "1":
            traceback.print_exc()
        _log_pmi_wandb_scalars(step, attempted_rate=len(attempted) / max(1, B),
                               aligned_rate=len(aligned) / max(1, B),
                               guard_hit_rate=0.0, member_rate=0.0,
                               nonfinite_rate=0.0)
        return r_meta, member
    n_al = len(aligned)
    for k, (_i, row, sp, _psp) in enumerate(aligned):
        L = len(arm_resps[k])  # == both arms' span length (token-id-identical)
        row["logp_with"] = ref_lp[k, :L].float().cpu().numpy()
        row["logp_without"] = ref_lp[n_al + k, :L].float().cpu().numpy()
    for j, k in enumerate(placebo_idx):
        row = aligned[k][1]
        Lp = len(arm_resps[2 * n_al + j])  # == without-span length (equality-checked)
        row["logp_placebo"] = ref_lp[2 * n_al + j, :Lp].float().cpu().numpy()
        # logp_placebo_without intentionally absent: compute_pmi_rows defaults
        # it to logp_without (valid — without-span equality enforced above).

    # 3) Aggregate + guard + sign-gate via the pure core; scatter back to B.
    rows = [row for (_i, row, _sp, _psp) in attempted]
    scored, diag = compute_pmi_rows(
        rows, method=method, topk_frac=topk_frac, clip_c_token=clip_c_token,
        clip_c_gate=clip_c_gate, ngram_n=ngram_n, ngram_threshold=ngram_threshold,
        placebo_correct=placebo_correct, alpha=pmi_alpha,
    )
    for j, (i, _row, _sp, _psp) in enumerate(attempted):
        r_meta[i] = scored[j]
        # IMPORTANT-3 (round 2): nonfinite rows are failed rows — R 0, member 0
        # — a NaN r_meta with member=1 would NaN every sibling's centered A_meta
        # in group_mean_subtract. Placebo-failed rows fail closed the same way
        # (no raw-delta fallback inside a centering group).
        member[i] = (
            0.0
            if (diag["alignment_failures"][j] or diag["nonfinite"][j]
                or diag["guard_hits"][j] or diag["placebo_failures"][j])
            else 1.0
        )

    n_guard = int(sum(bool(g) for g in diag["guard_hits"]))
    n_nonfinite = int(sum(bool(x) for x in diag["nonfinite"]))
    n_placebo_fail = int(sum(bool(x) for x in diag["placebo_failures"]))
    if n_nonfinite:
        print(f"[DCPO-V4] pmi step={step}: NON-FINITE arm logprobs on "
              f"{n_nonfinite}/{len(attempted)} attempted row(s) — those rows "
              f"score R_meta 0 / member 0 (poisoning guard, review round 2).",
              flush=True)
    if os.environ.get("DCPO_DEBUG", "1") == "1":
        _scored_vals = [float(r_meta[i]) for (i, _r, _s, _p) in attempted if member[i] > 0.5]
        print(f"[DCPO-V4] pmi step={step}: B={B} attempted={len(attempted)} "
              f"aligned={n_al} guard_hits={n_guard} nonfinite={n_nonfinite} "
              f"placebo_correct={placebo_correct} placebo_fails={n_placebo_fail} "
              f"rmeta_mean_scored={np.mean(_scored_vals) if _scored_vals else 0.0:.4f}",
              flush=True)
    _log_pmi_wandb_scalars(step, attempted_rate=len(attempted) / max(1, B),
                           aligned_rate=n_al / max(1, B),
                           guard_hit_rate=n_guard / max(1, B),
                           member_rate=float(member.mean()),
                           nonfinite_rate=n_nonfinite / max(1, B),
                           placebo_fail_rate=n_placebo_fail / max(1, B))
    return r_meta, member


def _attach_teacher_signals(data: DataProto):
    trainer = _ACTIVE_SDC_CONTEXT.get("trainer")
    tokenizer = _ACTIVE_SDC_CONTEXT.get("tokenizer")
    mode = _ACTIVE_SDC_CONTEXT.get("mode", "SDC_SHARED")
    if trainer is None or tokenizer is None:
        raise RuntimeError("SDC teacher context is not initialized")
    # R0 (VANILLA_GRPO): no teacher signal at all. Skip all forward passes
    # and return data unmodified — base GDPO advantage path takes over.
    if mode in _VANILLA_MODES:
        return data
    # TRIOBJ_DCPO_V2 (region-routed, ADDITIVE): env-reward-only — no teacher forward.
    # Short-circuit before T+/T-/position forward, exactly like _VANILLA_MODES. The
    # per-region advantage path reads only the stacked masks + head scalars.
    #
    # AUTHORITATIVE population (bugfix): `_compute_dcpo_region_advantage` reads BOTH
    # the 3 GDPO reward keys (correctness / meta_region_utility / cal_region_reward)
    # AND the 3 token masks (dcpo_answer_mask / dcpo_meta_content_mask /
    # dcpo_conf_mask) from data. In the async-rollout path the synchronous
    # `MetaCotSDCRewardManager.__call__` DCPO block is bypassed, and `reward_loop_score`
    # (running per-rollout in Ray actors, no group) cannot compute the GROUP-aware
    # R_meta (p_hat) — it only emits 0.0 placeholders. So write them here, in the
    # MAIN process with the full uid group + step, BEFORE `compute_sdc_gdpo_advantage`
    # runs the GDPO assertion (core_algos.compute_gdpo_outcome_advantage) and reads
    # the heads. This is the only place that has group structure AND runs pre-assertion.
    if mode in _REGION_ROUTED_MODES:
        _populate_dcpo_region_keys(data)
        return data
    # Both keys must exist for downstream compute_sdc_gdpo_advantage; an
    # interrupted attach (only one key set) must be recomputed, not cached.
    if (
        "sdc_teacher_pos_log_probs" in data.batch.keys()
        and "sdc_teacher_neg_log_probs" in data.batch.keys()
    ):
        return data

    prompt_tensor = data.batch["prompts"]
    response_tensor = data.batch["responses"]
    attention_mask = data.batch["attention_mask"]
    response_mask = data.batch["response_mask"]
    prompt_length = prompt_tensor.size(1)

    prompt_texts: list[str] = []
    gold_answers: list[str] = []
    decoy_answers: list[str] = []

    for i in range(response_tensor.size(0)):
        prompt_text = _decode_prompt_only(
            tokenizer,
            prompt_tensor[i],
            attention_mask[i],
            prompt_length,
        )
        prompt_texts.append(prompt_text)
        # reward_model may be absent on some DataProto views (codereview IMPORTANT-1:
        # `.get(..., [])[i]` raised IndexError when the key was missing) — per-row {}.
        _rm = data.non_tensor_batch.get("reward_model", None)
        gt = _rm[i] if _rm is not None else {}
        if isinstance(gt, dict):
            gt = gt.get("ground_truth", "")
        gold = str(gt)
        gold_answers.append(gold)
        decoy_answers.append(_rule_based_decoy(gold, seed=42))

    # GOLD is load-bearing for every teacher variant: an empty gold silently
    # conditions T+ on NO answer and T- on the absolute-fallback decoy " + 1",
    # producing a plausible-looking but content-free contrast (codereview
    # CRITICAL-1, same silent-empty class as the v3b gt="" bug). Fail fast when
    # the whole batch is goldless; count-and-warn on partial gaps.
    _n_empty_gold = sum(1 for g in gold_answers if not g.strip())
    if gold_answers and _n_empty_gold == len(gold_answers):
        raise RuntimeError(
            "[SDC] _attach_teacher_signals: ALL ground truths are empty — "
            "non_tensor 'reward_model'/'ground_truth' missing on this batch; "
            "the teacher would condition on no answer (silent no-op)."
        )
    if _n_empty_gold:
        print(f"[SDC] WARNING: {_n_empty_gold}/{len(gold_answers)} rows have EMPTY "
              f"gold — teacher contrast is content-free for those rows.", flush=True)

    # ── R5 (RLSD_FORCED_META) prep ─────────────────────────────────────────
    # Generate (or fetch cached) V0 student prefixes for unique prompts in this
    # batch. The forced-meta path threads these through to both T+ and T-
    # teacher conditioning so the teacher distribution is grounded on the same
    # "rethink-after-attempt" context the student sees post-rollout.
    v0_prefixes: dict[str, str] | None = None
    forced_meta_flag = mode in _FORCED_META_MODES
    if forced_meta_flag:
        global_steps = int(getattr(trainer, "global_steps", 0) or 0)
        unique_prompts = sorted(set(prompt_texts))
        v0_prefixes = _generate_v0_prefixes(
            trainer, tokenizer, unique_prompts, global_steps
        )

    # E.4 self-distill: for stance/conf the T- side conditions on GOLD (not the
    # decoy) so the answer CANCELS in T+−T− and the contrast isolates the
    # stance/confidence axis only. For `decoy` (default) neg_answers IS the
    # _rule_based_decoy output and contrast_side is ignored by the decoy branch
    # → the decoy teacher forward stays BYTE-IDENTICAL.
    contrast_variant = _ACTIVE_SDC_CONTEXT.get("sdc_contrast_variant", "decoy")

    pos_batch = _build_teacher_logprob_batch(
        tokenizer=tokenizer,
        prompt_texts=prompt_texts,
        answer_texts=gold_answers,
        responses=response_tensor,
        response_mask=response_mask,
        v0_prefixes=v0_prefixes,
        forced_meta=forced_meta_flag,
        teacher_role="content",  # gold-conditioned T+ (content teacher)
        contrast_side="pos",
    )
    # verl 0.7.1 engine_workers infer_batch reads micro_batch["temperature"];
    # the trainer's main fit() loop sets it on the rollout output, but our
    # freshly-built teacher batches don't inherit meta_info, so re-attach.
    # Tolerant of a config layout change / a test double without .config.
    try:
        rollout_temp = float(trainer.config.actor_rollout_ref.rollout.temperature)
    except Exception:
        rollout_temp = 1.0
    pos_batch.meta_info["temperature"] = rollout_temp
    pos_out = trainer._compute_ref_log_prob(pos_batch)
    target_device = response_tensor.device
    data.batch["sdc_teacher_pos_log_probs"] = pos_out.batch["ref_log_prob"].to(target_device)

    # R1 (RLSD_META_ATTR), OPSD_META, ROD_PT: skip decoy forward — saves the one
    # teacher pass dedicated to T- (roughly 25-33% wall-time vs full SDC).
    # Set teacher_neg = teacher_pos so any downstream contrastive computation
    # (delta = T+ − T−, w_shared) becomes a no-op (delta=0 → w_diff=1,
    # w_shared=w_attr). With λ_shared=λ_diff=0 in the yaml this is fully
    # equivalent to "no decoy".
    if mode in _SINGLE_TEACHER_MODES:
        data.batch["sdc_teacher_neg_log_probs"] = data.batch["sdc_teacher_pos_log_probs"].clone()

        # ROD_PT (Plan v5.17 FINAL): position teacher forward.
        # T_position input = prompt + gold + response[:p] where p = first META_START position.
        # Returns log_prob(META | prompt + gold + response[:p]) = position factor signal.
        # We reuse R5 _build_teacher_logprob_batch with truncated response_mask (valid only up to p).
        if mode in ("ROD_PT", "ROD_PT_DEGEN", "ROD_PT2_E21CTRL"):  # F1 codex r2 fix: ROD_PT_DEGEN needs position forward too (utils:308 expects it); ROD_PT2_E21CTRL (Arm 2) reuses the SAME content×position 2-teacher
            try:
                meta_start_id = int(tokenizer.convert_tokens_to_ids("<|meta|>"))
                # UNK-GUARD (codereview IMPORTANT-2): a tokenizer without <|meta|>
                # as a single token returns the unk id here — a positive int that
                # would pass `> 0` and scan for UNK tokens instead of meta openers
                # (silently misplacing/neutralizing the position teacher). Mirror
                # _meta_token_ids_safe's rejection.
                _unk = getattr(tokenizer, "unk_token_id", None)
                if _unk is not None and meta_start_id == int(_unk):
                    print("[SDC] WARNING: '<|meta|>' resolves to unk_token_id — "
                          "position teacher DISABLED for this run.", flush=True)
                    meta_start_id = -1
            except Exception:
                meta_start_id = -1
            target_device = response_tensor.device
            full_log_prob_meta = torch.zeros(response_tensor.size(0), device=target_device)

            if meta_start_id > 0:
                # Find first META_START position p per rollout
                rollout_ps: list[tuple[int, int]] = []
                for b in range(response_tensor.size(0)):
                    valid = (response_tensor[b] == meta_start_id) & response_mask[b].bool()
                    nz = valid.nonzero(as_tuple=True)[0]
                    if nz.numel() > 0:
                        rollout_ps.append((b, int(nz[0].item())))

                if rollout_ps:
                    real_N = len(rollout_ps)
                    # veRL dispatch requires N divisible by:
                    #   1. dp_size (chunk_tensordict in tensordict_utils.py:315)
                    #   2. force_group_size * micro_batch_size_per_gpu (ref_compute_ref_log_prob)
                    # dp_size is the data-parallel world size = nnodes * n_gpus_per_node
                    # (codex Round 3: previously used n_gpus_per_node only, broke on multi-node).
                    try:
                        nnodes = int(trainer.config.trainer.nnodes)
                    except Exception:
                        nnodes = 1
                    try:
                        n_gpus_per_node = int(trainer.config.trainer.n_gpus_per_node)
                    except Exception:
                        n_gpus_per_node = 4
                    dp_size = nnodes * n_gpus_per_node
                    try:
                        micro_bs = int(trainer.config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu)
                    except Exception:
                        try:
                            micro_bs = int(trainer.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu)
                        except Exception:
                            micro_bs = 4
                    pad_unit = dp_size * micro_bs  # safe LCM upper bound
                    pad_n = (-real_N) % pad_unit
                    rollout_ps_padded = list(rollout_ps)
                    for _ in range(pad_n):
                        rollout_ps_padded.append(rollout_ps[0])  # duplicate for padding
                    N = len(rollout_ps_padded)
                    T_resp = response_tensor.size(1)
                    # Build subset batch with truncated mask (valid only up to position p inclusive)
                    truncated_mask_subset = torch.zeros(
                        (N, T_resp), dtype=response_mask.dtype, device=response_mask.device
                    )
                    truncated_responses_subset = []
                    prompt_texts_subset = []
                    gold_subset = []
                    for i, (b, p) in enumerate(rollout_ps_padded):
                        truncated_mask_subset[i, : p + 1] = 1.0
                        truncated_responses_subset.append(response_tensor[b])
                        prompt_texts_subset.append(prompt_texts[b])
                        gold_subset.append(gold_answers[b])
                    truncated_responses = torch.stack(truncated_responses_subset, dim=0)

                    position_batch = _build_teacher_logprob_batch(
                        tokenizer=tokenizer,
                        prompt_texts=prompt_texts_subset,
                        answer_texts=gold_subset,
                        responses=truncated_responses,
                        response_mask=truncated_mask_subset,
                        v0_prefixes=None,
                        forced_meta=False,
                        teacher_role="position",  # logP(<|meta|>) position teacher
                    )
                    position_batch.meta_info["temperature"] = rollout_temp
                    pos_position_out = trainer._compute_ref_log_prob(position_batch)
                    # ref_log_prob[i, t] = log_prob of responses[i, t] given preceding context
                    # → ref_log_prob[i, p] = log_prob(META | prompt + gold + response[:p])
                    ref_log_probs_position = pos_position_out.batch["ref_log_prob"].to(target_device)

                    # IMPORTANT: only iterate up to real_N (skip padded duplicates)
                    for i, (b, p) in enumerate(rollout_ps[:real_N]):
                        # Bound check (in case T_resp_dim mismatch from padding)
                        if p < ref_log_probs_position.size(1):
                            full_log_prob_meta[b] = ref_log_probs_position[i, p]

            data.batch["sdc_position_log_prob_meta"] = full_log_prob_meta
    else:
        # Both contrastive (R2/SDC_*) and forced-meta (R5) modes run T- here.
        # Forced-meta passes the same v0_prefixes + forced_meta=True so the
        # decoy teacher conditioning matches the gold teacher format (only the
        # answer slot differs), preserving the contrast-on-answer-only
        # invariant the SDC contrastive math depends on.
        # T- decoy teacher. This branch is the `else` of
        # `if mode in _SINGLE_TEACHER_MODES` — ROD_PT2_E21CTRL is a
        # single-teacher mode so it NEVER reaches here (T-=T+ clone above).
        # Leaving teacher_role at its "content" default is byte-identical for
        # every mode that DOES run this (contrastive / forced-meta), because
        # the per-teacher prefix guard only fires for ROD_PT2_E21CTRL, which
        # by construction cannot enter this branch -> _tp_prefix stays "".
        neg_answers = (
            gold_answers if contrast_variant in ("stance", "conf") else decoy_answers
        )
        neg_batch = _build_teacher_logprob_batch(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            answer_texts=neg_answers,
            responses=response_tensor,
            response_mask=response_mask,
            v0_prefixes=v0_prefixes,
            forced_meta=forced_meta_flag,
            teacher_role="content",
            contrast_side="neg",
        )
        neg_batch.meta_info["temperature"] = rollout_temp
        neg_out = trainer._compute_ref_log_prob(neg_batch)
        data.batch["sdc_teacher_neg_log_probs"] = neg_out.batch["ref_log_prob"].to(target_device)

    # When agent_reward_loop pre-populates rm_scores asynchronously, the SDC
    # reward manager early-returns before computing region masks. compute_advantage
    # downstream still expects sdc_meta_mask / sdc_postmeta_*_mask / sdc_body_mask,
    # so populate them here from the response tokens we already have on hand.
    if "sdc_meta_mask" not in data.batch.keys():
        bs = response_tensor.size(0)
        response_length = response_tensor.size(1)
        meta_masks, post_shared, post_diff, body, fb = [], [], [], [], []
        for i in range(bs):
            r_ids = response_tensor[i].tolist()
            masks = build_sdc_region_masks(
                tokenizer,
                r_ids,
                tokenizer.decode(r_ids, skip_special_tokens=False),
            )
            def _pad(m: torch.Tensor) -> torch.Tensor:
                out = torch.zeros(response_length, dtype=torch.float32)
                usable = min(response_length, m.numel())
                out[:usable] = m[:usable]
                return out
            meta_masks.append(_pad(masks["meta_mask"]))
            post_shared.append(_pad(masks["postmeta_shared_mask"]))
            post_diff.append(_pad(masks["postmeta_diff_mask"]))
            body.append(_pad(masks["body_mask"]))
            fb.append(masks["fallback_triggered"])
        data.batch["sdc_meta_mask"] = torch.stack(meta_masks, dim=0).to(target_device)
        data.batch["sdc_postmeta_shared_mask"] = torch.stack(post_shared, dim=0).to(target_device)
        data.batch["sdc_postmeta_diff_mask"] = torch.stack(post_diff, dim=0).to(target_device)
        data.batch["sdc_body_mask"] = torch.stack(body, dim=0).to(target_device)
        data.non_tensor_batch["sdc_fallback_triggered"] = np.asarray(fb, dtype=np.float32)
    return data


class MetaCotSDCRewardManager:
    """SDC_SHARED reward aggregator.

    On each `__call__(batch)`:
      1. Computes SDC region masks (meta / postmeta_shared / postmeta_diff / body)
         for every response and writes them into `batch.batch`.  These masks
         are consumed downstream by `compute_sdc_gdpo_advantage`.
      2. Runs every reward head on decoded completions vs ground_truth, writes
         per-key scalar scores to `batch.non_tensor_batch[key]`, and accumulates
         a token-level reward tensor placed at the EOS position.
      3. Returns a DataProto carrying `rm_scores` + `reward_extra_keys` so that
         `RayPPOTrainer._compute_reward_colocate`'s output contract is honored
         (verl 0.7.1 fit() union's this back into the main batch and then
         `extract_reward(batch)` reads `batch.batch["rm_scores"]`).
    """

    def __init__(
        self,
        tokenizer,
        reward_funcs: List[Callable],
        reward_weights: List[float],
        reward_keys: List[str],
        num_examine: int = 0,
    ) -> None:
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.reward_weights = reward_weights
        self.reward_keys = reward_keys
        self.num_examine = num_examine
        assert len(reward_funcs) == len(reward_weights) == len(reward_keys)

    def __call__(self, data: DataProto, return_dict: bool = False):
        # return_dict (E.4 #2b, 2026-06-03): the base verl _validate path calls
        # reward_fn(batch, return_dict=True) and reads result['reward_tensor'] /
        # result['reward_extra_info'] (ray_trainer.py _compute_or_extract_reward).
        # Val batches come from generate_sequences with NO rm_scores pre-filled, so
        # they fall through to the main reward-compute body below (which already
        # computes `combined` unconditionally — there is NO NameError risk in this
        # file: the rm_scores branch returns early). When return_dict=True we return
        # the {'reward_tensor','reward_extra_info'} dict so reward_extra_info carries
        # `correctness` → process_validation_metrics emits
        # val-aux/<data_source>/correctness/mean@1 per benchmark. The default
        # return_dict=False (training) path stays BYTE-IDENTICAL (DataProto with
        # rm_scores + reward_extra_keys). Pairs with #2a.
        if "rm_scores" in data.batch.keys():
            # Already computed (e.g., agent_reward_loop path). 2026-05-22 fix:
            # the old pass-through returned `non_tensor_batch={}`, which dropped
            # the SDC-specific reward keys (confidence_revision,
            # redirect_execution, verify_execution, meta_floor, meta_count_bonus)
            # that compute_sdc_gdpo_advantage downstream reads. Result was
            # `AssertionError: GDPO reward key 'confidence_revision' not found
            # in non_tensor_batch` at training step 1 for Arm2 (ROD_PT2_E21CTRL).
            # Fix: emit per-key reward scores on this path too, by running the
            # configured reward funcs on decoded completions. rm_scores itself
            # is left untouched (pre-filled value preserved as the env reward).
            # See tests/test_arm2_reward_emit.py for the contract.
            bs = len(data)
            response_length = data.batch["responses"].shape[-1]
            prompt_length = data.batch["prompts"].shape[-1]
            decoded_responses, ground_truths = [], []
            for i in range(bs):
                item = data[i]
                text, _ids = _decode_response(
                    self.tokenizer,
                    item.batch["prompts"],
                    item.batch["responses"],
                    item.batch["attention_mask"],
                    prompt_length,
                )
                decoded_responses.append(text)
                gt = item.non_tensor_batch.get("reward_model", {})
                if isinstance(gt, dict):
                    gt = gt.get("ground_truth", "")
                ground_truths.append(str(gt))
            completions = [[{"content": t}] for t in decoded_responses]
            valid_response_length = (
                data.batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
            )
            completion_lengths_list = [
                int(valid_response_length[i].item()) + 1 for i in range(bs)
            ]
            from src.training.rewards import (
                _extract_answer_fallback as _extract_ans_for_degen,
            )
            answer_extracted_list = [
                _extract_ans_for_degen(t) for t in decoded_responses
            ]
            # ── TRIOBJ_DCPO_V2/V3 (ADDITIVE, mode-gated) ─────────────────────
            # rm_scores-PREFILLED path (agent-loop / async rollout): the reward
            # funcs below read `_DCPO_HEAD_STASH`, but the from-scratch branch
            # that normally runs the ONE-SHOT DCPO head pre-pass (masks + heads)
            # is NOT reached here — so without this block the region heads stay
            # the 0.0 placeholders and `dcpo_region_rewards` (its DCPO_DEBUG dump)
            # never runs. V3's CF wrap installs `agent_loop_config_path`, which
            # routes rollout through the prefilled path (V2 used the sync path,
            # so it ran the pre-pass at the from-scratch branch). Mirror the
            # from-scratch DCPO block here (masks + group-aware head stash, with
            # the producer's cf_correct), BEFORE the reward-func loop reads the
            # stash. Fires ONLY for the region-routed modes; every other mode's
            # prefilled path is byte-identical.
            _mode_pf = _ACTIVE_SDC_CONTEXT.get("mode", "")
            if _mode_pf in _REGION_ROUTED_MODES:
                # KARPATHY lock "v2 mode byte-identical": the v3 format-fix
                # pieces (clamp/gate, FORMAT_VIOLATION stack) are v3-only here
                # too — mirror of _populate_dcpo_region_keys (V4 joins via
                # _DCPO_V3_FMT_MODES, same machinery verbatim).
                _pf_v3 = _mode_pf in _DCPO_V3_FMT_MODES
                # v3k fmt machinery — EXACT mirror of _populate_dcpo_region_keys
                # (five-way sync: identical gate/penalty/tier semantics both
                # paths). Stash present = CF wrap replaced tier-1 tokens;
                # absent = classify here with tier1_to_discard.
                _pf_cls_stash = data.non_tensor_batch.get("dcpo_fmt_class", None) if _pf_v3 else None
                _pf_rep_stash = data.non_tensor_batch.get("dcpo_fmt_replaced", None) if _pf_v3 else None
                _pf_fmt_classes: list = []
                _pf_ans, _pf_meta_c, _pf_conf, _pf_fmt, _pf_fmt_ok = [], [], [], [], []
                _pf_trunc = []  # TRUNC_OPEN (spec §3.3)
                for i in range(bs):
                    _item = data[i]
                    _attn = _item.batch["attention_mask"]
                    _vlen = int(_attn[prompt_length:].sum().item())
                    _rids = _item.batch["responses"][:_vlen].tolist()
                    _rmask = [True] * len(_rids)
                    _decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)
                    if _pf_v3:
                        _rep = bool(
                            _pf_rep_stash is not None and i < len(_pf_rep_stash)
                            and float(_pf_rep_stash[i]) > 0.5)
                        _fmt = classify_dcpo_format(
                            _rids, _rmask, _decode, tier1_to_discard=not _rep)
                        if _rep and _pf_cls_stash is not None and i < len(_pf_cls_stash):
                            _pf_fmt_classes.append(str(_pf_cls_stash[i]))
                        else:
                            _pf_fmt_classes.append(_fmt["fmt_class"])
                        _rmasks = build_dcpo_region_masks(
                            _rids, _rmask, _decode, clamp_unclosed=True,
                            fmt=_fmt, fmt_replaced=_rep)
                    else:
                        _rmasks = build_dcpo_region_masks(
                            _rids, _rmask, _decode, clamp_unclosed=False)

                    def _pf_pad_bool(arr) -> torch.Tensor:
                        out = torch.zeros(response_length, dtype=torch.float32)
                        n = min(response_length, len(arr))
                        if n > 0:
                            out[:n] = torch.as_tensor(arr[:n], dtype=torch.float32)
                        return out

                    _pf_ans.append(_pf_pad_bool(_rmasks["ANSWER_REGION"]))
                    _pf_meta_c.append(_pf_pad_bool(_rmasks["META_CONTENT"]))
                    _pf_conf.append(_pf_pad_bool(_rmasks["CONF"]))
                    _pf_fmt.append(_pf_pad_bool(_rmasks["FORMAT_VIOLATION"]))
                    _pf_fmt_ok.append(_pf_pad_bool(_rmasks["FORMAT_OK"]))
                    _pf_trunc.append(_pf_pad_bool(_rmasks["TRUNC_OPEN"]))
                data.batch["dcpo_answer_mask"] = torch.stack(_pf_ans, dim=0)
                data.batch["dcpo_meta_content_mask"] = torch.stack(_pf_meta_c, dim=0)
                data.batch["dcpo_conf_mask"] = torch.stack(_pf_conf, dim=0)
                # 4th routed head's token spans (violation + v3k FORMAT_OK;
                # mirror of _populate_dcpo_region_keys — the async/sync paths
                # must agree). v3-ONLY: key presence arms the head downstream.
                if _pf_v3:
                    data.batch["dcpo_format_violation_mask"] = torch.stack(_pf_fmt, dim=0)
                    data.batch["dcpo_format_ok_mask"] = torch.stack(_pf_fmt_ok, dim=0)
                    data.batch["dcpo_trunc_open_mask"] = torch.stack(_pf_trunc, dim=0)

                _pf_uid = data.non_tensor_batch.get("uid", None)
                _pf_trainer = _ACTIVE_SDC_CONTEXT.get("trainer", None)
                _pf_step = int(getattr(_pf_trainer, "global_steps", 0) or 0)
                # TRIOBJ_DCPO_V3: consume producer cf_texts if present (graded here,
                # where real ground_truths exist); text fallback otherwise.
                _pf_cf = data.non_tensor_batch.get("cf_texts", None)
                if _pf_cf is not None:
                    _pf_cf = [None if t is None else str(t) for t in list(_pf_cf)]
                _pf_heads = _compute_dcpo_heads_stash(
                    completions, ground_truths, _pf_uid, _pf_step, self.config,
                    cf_completions=_pf_cf,
                    gate_unclosed=_pf_v3,   # v2 byte-identical: no gate/penalty
                    fmt_class=(_pf_fmt_classes if _pf_v3 else None),  # v3k tiers
                )
                data.non_tensor_batch["dcpo_phat"] = np.asarray(_pf_heads["p_hat"], dtype=np.float32)
                data.non_tensor_batch["dcpo_group_acc"] = np.asarray(_pf_heads["group_acc"], dtype=np.float32)
                data.non_tensor_batch["dcpo_canary_pass1_acc"] = np.asarray(
                    _pf_heads.get("canary_pass1_acc", [1.0] * bs), dtype=np.float32)
                data.non_tensor_batch["dcpo_sandbag_clamp"] = np.asarray(
                    _pf_heads.get("sandbag_clamp", [1.0] * bs), dtype=np.float32)
            for func_idx, reward_fn in enumerate(self.reward_funcs):
                key = self.reward_keys[func_idx]
                try:
                    scores = reward_fn(
                        completions=completions,
                        ground_truth=ground_truths,
                        completion_lengths=completion_lengths_list,
                        answer_extracted=answer_extracted_list,
                    )
                except Exception as exc:
                    print(f"[verl_sdc] reward {key} failed (pre-filled path): {exc}")
                    scores = [0.0] * bs
                if len(scores) != bs:
                    scores = (list(scores) + [0.0] * bs)[:bs]
                data.non_tensor_batch[key] = np.asarray(scores, dtype=np.float32)
            rm_td = TensorDict(
                {"rm_scores": data.batch["rm_scores"]}, batch_size=bs
            )
            non_tensor = {
                k: data.non_tensor_batch[k]
                for k in self.reward_keys
                if k in data.non_tensor_batch
            }
            if return_dict:
                # Defensive: the base val path extracts rm_scores directly before
                # ever calling reward_fn, so this branch is normally training-only;
                # honor the dict contract anyway for any caller that passes a
                # pre-scored batch with return_dict=True.
                return {
                    "reward_tensor": data.batch["rm_scores"],
                    "reward_extra_info": {k: np.asarray(v) for k, v in non_tensor.items()},
                }
            return DataProto(
                batch=rm_td,
                non_tensor_batch=non_tensor,
                meta_info={"reward_extra_keys": list(self.reward_keys)},
            )

        bs = len(data)
        response_length = data.batch["responses"].shape[-1]
        prompt_length = data.batch["prompts"].shape[-1]

        decoded_responses: list[str] = []
        ground_truths: list[str] = []
        meta_masks = []
        post_shared_masks = []
        post_diff_masks = []
        body_masks = []
        fallback_flags = []

        for i in range(bs):
            item = data[i]
            text, response_ids = _decode_response(
                self.tokenizer,
                item.batch["prompts"],
                item.batch["responses"],
                item.batch["attention_mask"],
                prompt_length,
            )
            decoded_responses.append(text)
            gt = item.non_tensor_batch.get("reward_model", {})
            if isinstance(gt, dict):
                gt = gt.get("ground_truth", "")
            ground_truths.append(str(gt))

            masks = build_sdc_region_masks(
                self.tokenizer,
                response_ids.tolist(),
                self.tokenizer.decode(response_ids, skip_special_tokens=False),
            )

            def _pad(mask: torch.Tensor) -> torch.Tensor:
                out = torch.zeros(response_length, dtype=torch.float32)
                usable = min(response_length, mask.numel())
                out[:usable] = mask[:usable]
                return out

            meta_masks.append(_pad(masks["meta_mask"]))
            post_shared_masks.append(_pad(masks["postmeta_shared_mask"]))
            post_diff_masks.append(_pad(masks["postmeta_diff_mask"]))
            body_masks.append(_pad(masks["body_mask"]))
            fallback_flags.append(masks["fallback_triggered"])

        data.batch["sdc_meta_mask"] = torch.stack(meta_masks, dim=0)
        data.batch["sdc_postmeta_shared_mask"] = torch.stack(post_shared_masks, dim=0)
        data.batch["sdc_postmeta_diff_mask"] = torch.stack(post_diff_masks, dim=0)
        data.batch["sdc_body_mask"] = torch.stack(body_masks, dim=0)
        data.non_tensor_batch["sdc_fallback_triggered"] = np.asarray(fallback_flags, dtype=np.float32)

        completions = [[{"content": text}] for text in decoded_responses]
        combined = torch.zeros(bs, response_length, dtype=torch.float32)
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1

        # ── TRIOBJ_DCPO_V2 (ADDITIVE, mode-gated) ────────────────────────────
        # Region-routed mode: stack the 3 DCPO token masks + the 2 per-rollout
        # group scalars, and run the ONE-SHOT head pre-pass that populates the
        # stash the three reward-func wrappers read. Fires ONLY for the DCPO mode;
        # every other mode's reward loop below is byte-identical.
        _mode = _ACTIVE_SDC_CONTEXT.get("mode", "")
        if _mode in _REGION_ROUTED_MODES:
            # KARPATHY lock "v2 mode byte-identical": v3 format-fix pieces
            # (clamp/gate, FORMAT_VIOLATION stack) are v3-only here too —
            # mirror of _populate_dcpo_region_keys (V4 joins via
            # _DCPO_V3_FMT_MODES, same machinery verbatim).
            _is_v3 = _mode in _DCPO_V3_FMT_MODES
            # v3k fmt machinery — EXACT mirror of _populate_dcpo_region_keys
            # (five-way sync: identical gate/penalty/tier semantics both paths).
            _sc_cls_stash = data.non_tensor_batch.get("dcpo_fmt_class", None) if _is_v3 else None
            _sc_rep_stash = data.non_tensor_batch.get("dcpo_fmt_replaced", None) if _is_v3 else None
            _sc_fmt_classes: list = []
            dcpo_ans, dcpo_meta_c, dcpo_conf, dcpo_fmt, dcpo_fmt_ok = [], [], [], [], []
            for i in range(bs):
                item = data[i]
                _resp_ids = item.batch["responses"]
                _attn = item.batch["attention_mask"]
                _vlen = int(_attn[prompt_length:].sum().item())
                _rids = _resp_ids[: _vlen].tolist()
                _rmask = [True] * len(_rids)
                _decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)
                if _is_v3:
                    _rep = bool(
                        _sc_rep_stash is not None and i < len(_sc_rep_stash)
                        and float(_sc_rep_stash[i]) > 0.5)
                    _fmt = classify_dcpo_format(
                        _rids, _rmask, _decode, tier1_to_discard=not _rep)
                    if _rep and _sc_cls_stash is not None and i < len(_sc_cls_stash):
                        _sc_fmt_classes.append(str(_sc_cls_stash[i]))
                    else:
                        _sc_fmt_classes.append(_fmt["fmt_class"])
                    rmasks = build_dcpo_region_masks(
                        _rids, _rmask, _decode, clamp_unclosed=True,
                        fmt=_fmt, fmt_replaced=_rep)
                else:
                    rmasks = build_dcpo_region_masks(
                        _rids, _rmask, _decode, clamp_unclosed=False)

                def _pad_bool(arr) -> torch.Tensor:
                    out = torch.zeros(response_length, dtype=torch.float32)
                    n = min(response_length, len(arr))
                    if n > 0:
                        out[:n] = torch.as_tensor(arr[:n], dtype=torch.float32)
                    return out

                dcpo_ans.append(_pad_bool(rmasks["ANSWER_REGION"]))
                dcpo_meta_c.append(_pad_bool(rmasks["META_CONTENT"]))
                dcpo_conf.append(_pad_bool(rmasks["CONF"]))
                dcpo_fmt.append(_pad_bool(rmasks["FORMAT_VIOLATION"]))
                dcpo_fmt_ok.append(_pad_bool(rmasks["FORMAT_OK"]))
            data.batch["dcpo_answer_mask"] = torch.stack(dcpo_ans, dim=0)
            data.batch["dcpo_meta_content_mask"] = torch.stack(dcpo_meta_c, dim=0)
            data.batch["dcpo_conf_mask"] = torch.stack(dcpo_conf, dim=0)
            # 4th routed head's token spans (violation + v3k FORMAT_OK; mirror
            # of _populate_dcpo_region_keys — the async/sync paths must agree).
            # v3-ONLY: presence of these keys arms the head downstream.
            if _is_v3:
                data.batch["dcpo_format_violation_mask"] = torch.stack(dcpo_fmt, dim=0)
                data.batch["dcpo_format_ok_mask"] = torch.stack(dcpo_fmt_ok, dim=0)

            _uid = data.non_tensor_batch.get("uid", None)
            _trainer = _ACTIVE_SDC_CONTEXT.get("trainer", None)
            _step = int(getattr(_trainer, "global_steps", 0) or 0)
            # TRIOBJ_DCPO_V3: consume producer cf_texts if present (graded here, where
            # real ground_truths exist); text fallback otherwise.
            _cf_texts = data.non_tensor_batch.get("cf_texts", None)
            if _cf_texts is not None:
                _cf_texts = [None if t is None else str(t) for t in list(_cf_texts)]
            _heads = _compute_dcpo_heads_stash(
                completions, ground_truths, _uid, _step, self.config,
                cf_completions=_cf_texts,
                gate_unclosed=_is_v3,   # v2 byte-identical: no gate/penalty
                fmt_class=(_sc_fmt_classes if _is_v3 else None),  # v3k tiers
            )
            data.non_tensor_batch["dcpo_phat"] = np.asarray(_heads["p_hat"], dtype=np.float32)
            data.non_tensor_batch["dcpo_group_acc"] = np.asarray(_heads["group_acc"], dtype=np.float32)
            # Sandbagging canary (batch pass-1 accuracy) + active clamp factor -> wandb.
            data.non_tensor_batch["dcpo_canary_pass1_acc"] = np.asarray(
                _heads.get("canary_pass1_acc", [1.0] * len(completions)), dtype=np.float32)
            data.non_tensor_batch["dcpo_sandbag_clamp"] = np.asarray(
                _heads.get("sandbag_clamp", [1.0] * len(completions)), dtype=np.float32)

        # Plumb completion_lengths + answer_extracted for the degeneration head
        # (codex round-5 review): without these, degeneration_penalty_reward
        # falls back to word-count and treats every short response as missing
        # an answer, falsely triggering the short-truncation penalty.
        # Other reward funcs accept **kwargs so the extras are no-ops for them.
        completion_lengths_list = [int(valid_response_length[i].item()) + 1 for i in range(bs)]
        from src.training.rewards import _extract_answer_fallback as _extract_ans_for_degen
        answer_extracted_list = [_extract_ans_for_degen(t) for t in decoded_responses]

        for func_idx, reward_fn in enumerate(self.reward_funcs):
            key = self.reward_keys[func_idx]
            try:
                scores = reward_fn(
                    completions=completions,
                    ground_truth=ground_truths,
                    completion_lengths=completion_lengths_list,
                    answer_extracted=answer_extracted_list,
                )
            except Exception as exc:
                print(f"[verl_sdc] reward {key} failed: {exc}")
                traceback.print_exc()
                scores = [0.0] * bs
            if len(scores) != bs:
                scores = (list(scores) + [0.0] * bs)[:bs]
            data.non_tensor_batch[key] = np.asarray(scores, dtype=np.float32)

            reward_tensor = torch.zeros(bs, response_length, dtype=torch.float32)
            for i in range(bs):
                eos_pos = max(0, int(valid_response_length[i].item()))
                reward_tensor[i, eos_pos] = float(scores[i]) * float(self.reward_weights[func_idx])
            combined += reward_tensor

        # Emit rm_scores + reward_extra_keys for verl 0.7.1 fit()/extract_reward contract.
        rm_td = TensorDict({"rm_scores": combined}, batch_size=bs)
        extra_keys = list(self.reward_keys) + ["sdc_fallback_triggered"]
        non_tensor = {k: data.non_tensor_batch[k] for k in extra_keys if k in data.non_tensor_batch}
        if return_dict:
            # Val path (#2b): return the dict the base _validate expects. Carry the
            # reward keys (notably `correctness`) so process_validation_metrics emits
            # val-aux/<data_source>/correctness/mean@1 per benchmark (gsm8k/math/aime).
            return {
                "reward_tensor": combined,
                "reward_extra_info": {
                    k: np.asarray(data.non_tensor_batch[k])
                    for k in self.reward_keys
                    if k in data.non_tensor_batch
                },
            }
        return DataProto(
            batch=rm_td,
            non_tensor_batch=non_tensor,
            meta_info={"reward_extra_keys": list(non_tensor.keys())},
        )


class SDCRayPPOTrainer(RayPPOTrainer):
    """Thin verl 0.7.1 trainer wrapper that injects an in-process reward manager.

    Why subclass: verl 0.7.1 removed `reward_fn`/`val_reward_fn` kwargs from
    `RayPPOTrainer.__init__`.  Reward now flows through `RewardLoopManager`.
    The SDC pipeline needs the reward call to SIDE-EFFECT the batch (meta
    masks, per-key scores, fallback flag) so that the downstream
    `compute_sdc_gdpo_advantage` can read them.  Routing SDC through the
    async reward_loop_workers would break those side effects.  Overriding
    `_compute_reward_colocate` keeps the contract: fit() still calls it,
    we just service it in-process without the reward_loop_manager.
    """

    def __init__(self, *args, reward_fn=None, val_reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sdc_reward_fn = reward_fn
        self._sdc_val_reward_fn = val_reward_fn if val_reward_fn is not None else reward_fn
        # LOGGING FIX (E.4 #2a, 2026-06-03): the base RayPPOTrainer gates BOTH the
        # initial validate (`if self.val_reward_fn is not None and ...val_before_train`)
        # and the periodic test_freq validate on self.val_reward_fn. This subclass
        # strips reward_fn/val_reward_fn out of kwargs (they are NOT forwarded to
        # super), so base self.val_reward_fn stays None → _validate() NEVER runs →
        # val-aux/<data_source>/correctness/mean@1 is produced for NO arm →
        # test_freq=25 is a silent no-op → the accuracy A/B (the verdict decider) is
        # unreadable for ALL 4 arms. Attach the managers onto the base attrs so eval
        # runs. INSEPARABLE from #2b (MetaCotSDCRewardManager.__call__ must honor
        # return_dict=True on the val path) — without #2b this would convert a silent
        # skip into a crash. Robust whether or not the deployed verl already accepts
        # these kwargs (idempotent re-assignment).
        self.reward_fn = self._sdc_reward_fn
        self.val_reward_fn = self._sdc_val_reward_fn
        # Fail-fast: sdc_force_inject is requested but the two-phase rollout repack
        # (_force_inject_rollout) is NODE-SMOKE-REQUIRED and not yet wired. Refuse
        # to launch rather than silently run a NON-inject experiment mislabeled as
        # inject (codex P1). Remove this guard once the repack is implemented +
        # 1-step smoke-tested on the node.
        _algo = getattr(self.config, "algorithm", {})
        _force_inject = bool(getattr(_algo, "sdc_force_inject", False))
        _sdc_mode = str(getattr(_algo, "sdc_mode", ""))
        # Consistency: the INJECT mode is meaningless without force-inject — refuse
        # to run it as a mislabeled non-inject R18b (codex follow-up #2).
        if _sdc_mode == "ROD_MQ_CONTRAST_INJECT" and not _force_inject:
            raise ValueError(
                "sdc_mode=ROD_MQ_CONTRAST_INJECT requires algorithm.sdc_force_inject=true "
                "(else it is just R18b mislabeled — use ROD_MQ_CONTRAST instead)."
            )
        if _force_inject:
            raise NotImplementedError(
                "sdc_force_inject=true but _force_inject_rollout repack is not yet "
                "wired/node-smoke-tested. Implement the DataProto repack against the "
                "live verl runtime (1-step smoke) before launching ROD_MQ_CONTRAST_INJECT."
            )

        # ─── E.9 BCI-RLVR gated binned-confidence-injection setup ─────────────
        # NEW flag `sdc_force_inject_conf` — DISTINCT from the legacy
        # `sdc_force_inject` hard-block above (which stays intact for
        # ROD_MQ_CONTRAST_INJECT). When this flag is FALSE (every existing mode)
        # nothing below installs a wrap → the rollout path is byte-identical.
        self._bci_inject_conf = bool(getattr(_algo, "sdc_force_inject_conf", False))
        self._bci_orig_generate = None
        self._bci_seed_ids = None  # list[list[int]] per bin, built lazily on tokenizer

        # ─── TRIOBJ_DCPO_V3 gated counterfactual 2nd-generation setup ─────────
        # NEW flag `sdc_counterfactual`. When FALSE (every existing mode) nothing
        # below installs a wrap → the rollout path is byte-identical. When TRUE we
        # install a generate_sequences wrap (init_workers) that, after the MAIN gen,
        # cuts each rollout at its first <|meta|>, regenerates a counterfactual with
        # <|meta|> SUPPRESSED, grades it, and stashes cf_correct onto the gen_output
        # BEFORE sleep_replicas() (spec §3.3-§3.5).
        self._dcpo_cf = bool(getattr(_algo, "sdc_counterfactual", False))
        self._dcpo_cf_orig_generate = None
        # v3k tier-1 token REPLACEMENT (yaml knob `dcpo_format_replace`, default
        # TRUE) — effective ONLY under sdc_mode==TRIOBJ_DCPO_V3/V4 (v2 and every
        # other mode byte-identical: the gate below can never arm for them).
        # Replacement happens inside the CF-wrap site (post-generation,
        # pre-old_log_prob) so verl recomputes old_log_prob on the REPLACED ids.
        # V4: the SAME wrap installs for replacement only — its CF regeneration
        # is independently gated on self._dcpo_cf (sdc_counterfactual=false in
        # the v4 yamls -> CF machinery dormant, NOT deleted).
        self._dcpo_fmt_replace = (
            _sdc_mode in ("TRIOBJ_DCPO_V3", "TRIOBJ_DCPO_V4")
            and bool(getattr(_algo, "dcpo_format_replace", True))
        )
        # GROUP-BRANCH COUNTERFACTUAL R_meta (design 2026-06-21): when
        # dcpo_rmeta_source=='cf_group' the main rollout is split into with-meta /
        # without-meta sub-arms (meta-open+close banned on the without-arm via
        # per-row logit_bias) so the counterfactual answer-delta is free. Gated +
        # default-OFF: every other source (cf/pmi/none) leaves this False and the
        # rollout path byte-identical. Only meaningful under TRIOBJ_DCPO_V4.
        self._dcpo_cf_group = (
            _sdc_mode == "TRIOBJ_DCPO_V4"
            and str(getattr(_algo, "dcpo_rmeta_source", "")) == "cf_group"
        )
        self._dcpo_cf_group_orig_generate = None
        self._dcpo_cf_branch_frac = float(
            getattr(_algo, "dcpo_cf_branch_frac", 0.5) or 0.5)
        # WITHOUT-ARM mechanism (design 2026-06-22): 'ban' (DEFAULT, byte-identical
        # to today = cf_groupban_agent meta-tag logit_bias) or 'placebo' (fix =
        # cf_placebo_agent forces a contentless placebo meta block so the without-
        # arm solves on-distribution; the ban degenerated to empty <think></think>
        # on the SFT init -> acc_without~0 -> invalid Δ). Only chooses placebo-vs-ban
        # WITHIN cf_group; whole-feature default-OFF is still guaranteed by
        # _dcpo_cf_group itself (TRIOBJ_DCPO_V4 + rmeta_source=='cf_group').
        self._dcpo_cf_without_mode = str(
            getattr(_algo, "dcpo_cf_without_mode", "ban") or "ban")
        if self._bci_inject_conf:
            from .meta_inject import default_conf_bins
            _n = int(self.config.actor_rollout_ref.rollout.n)
            _bins = getattr(_algo, "sdc_conf_bins", None)
            if _bins is None:
                self._bci_conf_bins = default_conf_bins(_n)
            else:
                self._bci_conf_bins = [float(x) for x in list(_bins)]
            if len(self._bci_conf_bins) != _n:
                raise ValueError(
                    f"sdc_conf_bins length {len(self._bci_conf_bins)} != rollout.n {_n}; "
                    "one confidence bin center is required per GRPO rollout."
                )
            print(
                f"[BCI-RLVR] binned-confidence-injection ENABLED: n={_n} "
                f"bins={self._bci_conf_bins} (wrap installed lazily in init_workers)"
            )

    def init_workers(self):
        """verl 0.7.1 creates `self.async_rollout_manager` inside the base
        init_workers (AgentLoopManager.create). The BCI wrap must replace its
        bound `generate_sequences` AFTER it exists, so we install the wrap here
        — ONLY under `algorithm.sdc_force_inject_conf` (else this override is a
        pure pass-through and the rollout path stays byte-identical)."""
        super().init_workers()
        if getattr(self, "_bci_inject_conf", False):
            mgr = getattr(self, "async_rollout_manager", None)
            if mgr is None:
                raise RuntimeError(
                    "BCI-RLVR: async_rollout_manager is None after init_workers — "
                    "cannot install the binned-confidence-injection wrap."
                )
            # Build per-bin seed token-ids now that the tokenizer is available, and
            # pad every bin's seed to a COMMON length so the prompt-tail slice and the
            # response-head splice are a single fixed width across the group.
            self._bci_build_seed_ids()
            self._bci_orig_generate = mgr.generate_sequences
            mgr.generate_sequences = self._bci_generate_sequences
            print("[BCI-RLVR] generate_sequences wrap INSTALLED on async_rollout_manager.")

        # ─── TRIOBJ_DCPO_V3 counterfactual wrap (ADDITIVE, gated) ─────────────
        # Install the CF 2nd-gen wrap under sdc_counterfactual OR the v3k tier-1
        # replacement knob (belt-and-suspenders: replacement must still run at
        # this post-generation/pre-old_log_prob site if CF were ever turned off;
        # the live v3 yaml has both true). Wrapping the SAME bound
        # generate_sequences (after any BCI wrap) keeps both additive: the CF
        # wrap calls the (possibly BCI-wrapped) main gen, replaces tier-1 format
        # tokens, then regenerates the meta-suppressed counterfactual.
        # Byte-identical when both flags are off.
        if getattr(self, "_dcpo_cf", False) or getattr(self, "_dcpo_fmt_replace", False):
            mgr = getattr(self, "async_rollout_manager", None)
            if mgr is None:
                raise RuntimeError(
                    "TRIOBJ_DCPO_V3: async_rollout_manager is None after init_workers — "
                    "cannot install the counterfactual generate_sequences wrap."
                )
            # Import the CF agent loop so its @register fires in this process too
            # (belt-and-suspenders; the Ray rollout workers resolve cf_prefix_agent via
            # configs/cf_prefix_agent.yaml on actor_rollout_ref.rollout.agent.agent_loop_config_path).
            try:
                import src.training.cf_prefix_agent  # noqa: F401  (registers cf_prefix_agent)
            except Exception as _e:  # pragma: no cover
                print(f"[DCPO-V3] cf_prefix_agent import warning: {_e}", flush=True)
            self._dcpo_cf_orig_generate = mgr.generate_sequences
            mgr.generate_sequences = self._dcpo_cf_generate_sequences
            print("[DCPO-V3] counterfactual generate_sequences wrap INSTALLED.")

        # ─── GROUP-BRANCH COUNTERFACTUAL wrap (ADDITIVE, gated) ───────────────
        # When dcpo_rmeta_source=='cf_group', split each GRPO group into with-meta
        # / without-meta sub-arms in the MAIN rollout (the without-arm bans the
        # meta-open+close tokens via per-row logit_bias). Additive-chained after
        # any BCI/CF wrap (wraps the SAME bound generate_sequences). Byte-identical
        # when the flag is off (every cf/pmi/none source).
        if getattr(self, "_dcpo_cf_group", False):
            mgr = getattr(self, "async_rollout_manager", None)
            if mgr is None:
                raise RuntimeError(
                    "cf_group: async_rollout_manager is None after init_workers — "
                    "cannot install the group-branch generate_sequences wrap."
                )
            # Import the agent loop so its @register fires in this process too
            # (the Ray rollout workers resolve cf_groupban_agent via its yaml on
            # actor_rollout_ref.rollout.agent.agent_loop_config_path).
            try:
                import src.training.cf_groupban_agent  # noqa: F401  (registers cf_groupban_agent)
                import src.training.cf_placebo_agent  # noqa: F401  (registers cf_placebo_agent, placebo without-mode)
            except Exception as _e:  # pragma: no cover
                print(f"[DCPO-CFGROUP] cf agent import warning: {_e}", flush=True)
            self._dcpo_cf_group_orig_generate = mgr.generate_sequences
            mgr.generate_sequences = self._dcpo_cf_group_generate_sequences
            print("[DCPO-CFGROUP] group-branch generate_sequences wrap INSTALLED.")

    def _bci_build_seed_ids(self):
        """Tokenize each bin's confidence seed into a list of token-id lists (one
        per bin). The agent-loop-native injection (BCIConfAgentLoop) prepends the
        seed to the response as plain token-id lists, so seeds need NOT be equal
        length — no fixed-width tensor slice, no pad/EOS hazard."""
        from .meta_inject import build_conf_seed_ids
        self._bci_seed_ids = [build_conf_seed_ids(self.tokenizer, c) for c in self._bci_conf_bins]

    def _bci_generate_sequences(self, gen_batch: "DataProto"):
        """Gated wrap of async_rollout_manager.generate_sequences (E.9).

        Agent-loop-native binned-confidence injection: instead of touching
        tensors, tag each rollout sample to use the custom BCIConfAgentLoop and
        hand it that sample's bin seed via non_tensor_batch. The agent loop
        prepends the seed to the response (response_mask=1, trained) and leaves
        the prompt original — the seeded confidence ends up in the trained
        response with no tensor repack. fit() passes gen_batch already repeated
        n× with interleave=True, so row r belongs to bin (r % n).

        No-op on validation: _validate() calls the same generate_sequences but
        does NOT repeat n×, so binning would inject an arbitrary confidence into
        every eval rollout and corrupt the acc/ECE gates (code-review C1). Val
        batches pass straight through to the default single_turn_agent loop.
        """
        if gen_batch.meta_info.get("validate", False):
            return self._bci_orig_generate(gen_batch)
        import numpy as _np
        n = int(self.config.actor_rollout_ref.rollout.n)
        B = len(gen_batch)
        seeds = self._bci_seed_ids
        # route every rollout sample to the BCI agent loop
        gen_batch.non_tensor_batch["agent_name"] = _np.array(
            ["bci_conf_agent"] * B, dtype=object
        )
        # per-sample bin seed (1-D object array of token-id lists; np.empty avoids
        # numpy collapsing equal-length lists into a 2-D array)
        seed_arr = _np.empty(B, dtype=object)
        for i in range(B):
            seed_arr[i] = list(seeds[i % n])
        gen_batch.non_tensor_batch["bci_conf_seed_ids"] = seed_arr
        return self._bci_orig_generate(gen_batch)

    def _dcpo_cf_group_generate_sequences(self, gen_batch: "DataProto"):
        """Gated wrap of generate_sequences for cf_group (design 2026-06-21).

        Split each GRPO group (n rollouts of one prompt) into a with-meta sub-arm
        (normal single_turn) and a without-meta sub-arm (meta-open+close banned
        via per-row logit_bias = the eval mechanism, inside the MAIN rollout). No
        second decode: the without-arm rows are real group members whose standard
        c_with is correct_without. fit() repeats gen_batch n× with interleave=True
        (verl_sdc:2946), so row r belongs to replica (r % n); a row is WITHOUT-META
        iff (r % n) >= n_with where n_with = round(n*(1 - branch_frac)).

        No-op on validation: _validate() does NOT repeat n×, so splitting would ban
        meta on half of eval and corrupt the acc/emission gates (same guard as BCI).
        """
        if gen_batch.meta_info.get("validate", False):
            return self._dcpo_cf_group_orig_generate(gen_batch)
        import numpy as _np
        n = int(self.config.actor_rollout_ref.rollout.n)
        B = len(gen_batch)
        arm, bias = cf_group_arm_split(
            B, n=n, branch_frac=float(getattr(self, "_dcpo_cf_branch_frac", 0.5)))
        # Route without-meta rows to the cf_groupban agent loop (keeps the NORMAL
        # chat-template path, only injects the meta-open+close logit_bias). With
        # rows keep the default single_turn agent.
        # Per-row routing. mode='ban' (DEFAULT) is byte-identical to today
        # (without-arm -> cf_groupban_agent + meta-tag logit_bias). mode='placebo'
        # (the 2026-06-22 fix) routes without-arm rows to cf_placebo_agent with NO
        # logit_bias — it forces a contentless placebo meta block as the trained
        # response prefix so the without-arm SOLVES on-distribution (the ban
        # degenerated to empty <think></think> on the SFT init -> invalid Δ).
        _mode = str(getattr(self, "_dcpo_cf_without_mode", "ban") or "ban")
        agent = _np.empty(B, dtype=object)
        bias_arr = _np.empty(B, dtype=object)
        with_meta = _np.empty(B, dtype=object)
        for i in range(B):
            agent[i], bias_arr[i], with_meta[i] = cf_group_route_row(
                arm[i], bias[i], mode=_mode)
        gen_batch.non_tensor_batch["agent_name"] = agent
        gen_batch.non_tensor_batch["cf_logit_bias"] = bias_arr
        # Stash the arm membership BEFORE generate (like BCI stashes seeds) so it
        # survives onto the returned batch for the populator to read directly
        # (per-row, survives balance_batch reshuffle).
        _with_meta_arr = _np.asarray([float(x) for x in with_meta], dtype=_np.float32)
        gen_batch.non_tensor_batch["dcpo_cf_with_meta"] = _with_meta_arr
        _out = self._dcpo_cf_group_orig_generate(gen_batch)
        # BUGFIX (0622): verl's generate output does NOT carry gen_batch's custom
        # non_tensor keys, so the arm-membership stash was lost and the reward
        # populator fell back to positional i%n — which balance_batch reshuffle
        # breaks, corrupting acc_without (the counterfactual). Re-attach the stash
        # onto the RETURNED batch (same B rows / order, pre-reshuffle) so it rides
        # per-row through union + balance_batch into the populator.
        try:
            if len(_out) == len(_with_meta_arr) and (
                "dcpo_cf_with_meta" not in _out.non_tensor_batch
            ):
                _out.non_tensor_batch["dcpo_cf_with_meta"] = _with_meta_arr
        except Exception:
            pass
        return _out

    # ─── TRIOBJ_DCPO_V3 counterfactual 2nd-generation (spec §3) ────────────────
    def _dcpo_cf_generate_sequences(self, gen_batch: "DataProto"):
        """Gated wrap of generate_sequences (TRIOBJ_DCPO_V3, spec §3.3).

        After the MAIN gen returns (replicas STILL awake; sleep_replicas() runs in
        ray_trainer only after this returns), build the counterfactual prefixes (cut
        at first <|meta|>), regenerate with <|meta|> id 151669 SUPPRESSED via
        logit_bias, and stash the decoded CF TEXTS (`cf_texts`, object array, length
        B, None for skipped/no-meta/failed rows) onto gen_output.non_tensor_batch.
        GRADING happens at the CONSUMER (_populate_dcpo_region_keys → dcpo_region_rewards
        cf_completions path) where the REAL ground truths are available — gen_batch/
        gen_output do NOT carry non_tensor 'reward_model', so grading here saw gt=""
        and judged every CF wrong (the v3b c_without≡0 bug). The 4 CF rollouts are
        inference-only — never placed in the GRPO group, never scored for advantage;
        they contribute exactly one scalar each to R_meta.

        No-op on validation (no GRPO, no reward routing) and on absent meta.
        """
        import numpy as _np

        gen_output = self._dcpo_cf_orig_generate(gen_batch)
        # Validation passes the same generate_sequences but does not train; skip
        # both the v3k format replacement and the CF.
        if gen_batch.meta_info.get("validate", False):
            return gen_output
        # ── v3k TIER-1 FORMAT REPLACEMENT (spec §6-2a) — runs FIRST: BEFORE the
        # CF prefix cut (the corrected opener is the cut point) and BEFORE verl
        # computes old_log_prob in its separate actor pass (Assumption A1), so
        # ratios are consistent on the REPLACED ids. CRASH-SAFE: on failure the
        # stash is absent and the populator demotes tier-1 rows to discard.
        if bool(getattr(self, "_dcpo_fmt_replace", False)):
            try:
                self._dcpo_format_classify_and_replace(gen_output)
            except Exception as e:  # pragma: no cover — defensive
                print(f"[DCPO-V3] format classify/replace FAILED "
                      f"({type(e).__name__}: {e}); tier-1 rows degrade to "
                      f"discard at the populator.", flush=True)
                if os.environ.get("DCPO_DEBUG", "1") == "1":
                    traceback.print_exc()
        if not bool(getattr(self, "_dcpo_cf", False)):
            return gen_output

        meta_open = int(getattr(self.config.algorithm, "dcpo_meta_open", 151669) or 151669)
        B = len(gen_output)
        try:
            resp = gen_output.batch["responses"]
            resp_mask = gen_output.batch.get("response_mask", None)
        except Exception as e:  # pragma: no cover — defensive
            print(f"[DCPO-V3] CF skipped: cannot read responses ({e}); cf_texts=None")
            _none = _np.empty(B, dtype=object)
            gen_output.non_tensor_batch["cf_texts"] = _none
            return gen_output

        # 1) Cut each rollout at its first <|meta|> → prefix ids (no-meta rows skipped).
        prefix_ids, skip = self._dcpo_cf_build_prefixes(gen_output, meta_open)

        # 2) Regenerate the counterfactuals with <|meta|> suppressed; decode TEXTS only.
        cf_texts = self._dcpo_cf_generate_texts(
            gen_batch, gen_output, prefix_ids, skip, meta_open
        )

        _arr = _np.empty(B, dtype=object)
        for _i in range(B):
            _arr[_i] = cf_texts[_i]
        gen_output.non_tensor_batch["cf_texts"] = _arr
        if os.environ.get("DCPO_DEBUG", "1") == "1":
            _n_cf = sum(1 for v in cf_texts if v is not None)
            print(f"[DCPO-V3] CF gen done: B={B} cf_texts={_n_cf} skipped(no-meta)={int(sum(skip))}",
                  flush=True)
        return gen_output

    def _dcpo_format_classify_and_replace(self, gen_output):
        """v3k TIER-1 token replacement + class stash (spec §3-tier-1 / §6-2a).

        Per row: `classify_dcpo_format` on the response ids (the ONE parser;
        tier-1 plans are §2.2-round-trip-validated INSIDE it). Tier-1 rows
        (swapped / dup_open / reversed) get their 1:1 SAME-LENGTH plan written
        into BOTH tensors the downstream consumers read (§1-V2):
          - gen_output.batch['responses'][row, pos]              (advantage /
            mask / reward decode + CF prefix cut)
          - gen_output.batch['input_ids'][row, prompt_len + pos] (actor
            log-prob forward), IF the key exists (defensive .get)
        attention_mask / position_ids / response_mask are untouched — same
        length, no re-pad, no position shift. After replacement the sequence IS
        wellformed → full normal routing; π(correct tag) rises with the row's
        routed advantage = token-local STaR-style correction.

        §8 runtime guards (verl source absent locally; Assumption A1 — verl
        recomputes old_log_prob AFTER this site — is validated at runtime):
          - HARD ABORT if the engine already returned log-probs
            ('old_log_probs' / 'rollout_log_probs' in gen_output.batch):
            replacement would invalidate them → skip ALL replacement (rows
            degrade to discard at the populator; never silently train on
            stale ratios).
          - per position: pre-write value must equal the plan's old_id in BOTH
            tensors (coherence guard); post-write re-read must equal new_id.

        Stash (flows through fit()'s union exactly like cf_texts):
          dcpo_fmt_class        [B] object  — parser class per row (ORIGINAL ids)
          dcpo_fmt_replaced     [B] float32 — 1.0 iff the row was replaced
          dcpo_fmt_replace_plan [B] object  — [(pos, old, new), ...] (else [])
        """
        import numpy as _np

        B = len(gen_output)
        resp = gen_output.batch["responses"]
        resp_mask = gen_output.batch.get("response_mask", None)
        attn = gen_output.batch.get("attention_mask", None)
        prompt_len = gen_output.batch["prompts"].shape[-1]
        input_ids = gen_output.batch.get("input_ids", None)

        # §8 hard abort: pre-existing log-probs would go stale under replacement.
        replace_ok = True
        for _k in ("old_log_probs", "rollout_log_probs"):
            if gen_output.batch.get(_k, None) is not None:
                print(f"[DCPO_DBG] FORMAT-REPLACE ABORT: gen_output.batch carries "
                      f"{_k!r} — the engine returned/precomputed log-probs that "
                      f"token replacement would invalidate. Skipping ALL "
                      f"replacement (rows degrade to discard).", flush=True)
                replace_ok = False

        meta_open = int(getattr(self.config.algorithm, "dcpo_meta_open", 151669) or 151669)
        # s3b §3.4 (flag, default False): widen tier-1 auto-correction to recover
        # the first valid meta pair from otherwise-discarded multi-open rows.
        recover_first_pair = bool(getattr(self.config.algorithm, "dcpo_recover_first_pair", False))
        _decode = lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False)

        classes = _np.empty(B, dtype=object)
        replaced = _np.zeros(B, dtype=_np.float32)
        plans = _np.empty(B, dtype=object)
        n_rep = 0
        for i in range(B):
            rids = resp[i]
            if resp_mask is not None:
                rm = resp_mask[i]
            elif attn is not None:
                rm = attn[i][prompt_len:]
            else:
                rm = None
            fmt = classify_dcpo_format(rids, rm, _decode, meta_open=meta_open,
                                       recover_first_pair=recover_first_pair)
            classes[i] = fmt["fmt_class"]
            plans[i] = []
            plan = fmt["replacement_plan"]
            if not plan or not replace_ok:
                continue
            # §8 coherence guard: pre-write values must match the plan's old_id
            # in BOTH tensors (a mismatch means input_ids is not the simple
            # prompt+response concat we verified — leave the row unreplaced; the
            # populator demotes it to discard via tier1_to_discard).
            coherent = all(
                int(resp[i, pos]) == int(old_id)
                and (input_ids is None
                     or int(input_ids[i, prompt_len + pos]) == int(old_id))
                for (pos, old_id, _new) in plan
            )
            if not coherent:
                print(f"[DCPO_DBG] FORMAT-REPLACE coherence FAIL row {i} "
                      f"(plan={plan}); row left unreplaced -> discard at the "
                      f"populator.", flush=True)
                continue
            for (pos, _old, new_id) in plan:
                resp[i, pos] = int(new_id)
                if input_ids is not None:
                    input_ids[i, prompt_len + pos] = int(new_id)
            # §8 post-write re-read.
            for (pos, _old, new_id) in plan:
                assert int(resp[i, pos]) == int(new_id)
                assert input_ids is None or int(input_ids[i, prompt_len + pos]) == int(new_id)
            replaced[i] = 1.0
            plans[i] = [(int(p), int(o), int(n)) for (p, o, n) in plan]
            n_rep += 1

        gen_output.non_tensor_batch["dcpo_fmt_class"] = classes
        gen_output.non_tensor_batch["dcpo_fmt_replaced"] = replaced
        gen_output.non_tensor_batch["dcpo_fmt_replace_plan"] = plans
        if os.environ.get("DCPO_DEBUG", "1") == "1":
            from collections import Counter as _Counter
            print(f"[DCPO-V3] fmt classify/replace: B={B} replaced={n_rep} "
                  f"classes={dict(_Counter(list(classes)))}", flush=True)
        return gen_output

    def _dcpo_cf_build_prefixes(self, gen_output, meta_open):
        """Per main rollout i: prefix_ids_i = prompt_ids_i + response_ids_i[:firstMeta]
        (left-pad stripped from prompt_ids). skip[i]=True when the rollout has no
        <|meta|> (cf_i ≈ r_i ⇒ R_meta 0). Returns (list[list[int]] | None, list[bool])."""
        import numpy as _np

        B = len(gen_output)
        resp = gen_output.batch["responses"]
        resp_mask = gen_output.batch.get("response_mask", None)
        prompts = gen_output.batch["prompts"]
        attn = gen_output.batch.get("attention_mask", None)
        prompt_len = prompts.shape[-1]

        prefix_ids = [None] * B
        skip = [False] * B
        for i in range(B):
            rids = resp[i]
            rmask = None if resp_mask is None else resp_mask[i]
            j = first_meta_token_index(rids, rmask, meta_open)
            if j is None:
                skip[i] = True
                continue
            # strip left-pad from the prompt (attention_mask over the prompt block).
            p_ids = prompts[i].tolist()
            if attn is not None:
                p_attn = attn[i][:prompt_len].tolist()
                p_ids = [tid for tid, a in zip(p_ids, p_attn) if a]
            r_ids = [int(t) for t in rids.tolist()[:j]]
            prefix_ids[i] = list(p_ids) + r_ids
        return prefix_ids, skip

    def _dcpo_cf_generate_texts(self, gen_batch, gen_output, prefix_ids, skip, meta_open):
        """Run the 2nd generate_sequences on the cut prefixes with <|meta|> suppressed
        and decode the CF continuation TEXTS. Returns a length-B list of (str | None);
        None = skipped/no-meta/failed → consumer falls back conservatively (R_meta 0
        for no-meta, pre-meta-prefix grade for failed rows).

        NO grading here: gen_output does NOT carry non_tensor 'reward_model', so any
        ground-truth read at this site is "" and every CF judges wrong (the v3b
        c_without≡0 bug). Grading lives in dcpo_region_rewards (cf_completions path),
        called from _populate_dcpo_region_keys where the full batch (with reward_model)
        is available.

        The verl 2nd-gen call is wired in `_dcpo_cf_call_engine` (cf_prefix_agent loop
        + per-call logit_bias suppression). CRASH-SAFE: any failure → all-None cf_texts
        so R_meta gracefully degrades (text fallback still supplies a conservative
        signal).
        """
        B = len(gen_output)

        # Filter to the rows that actually need a CF gen (have meta).
        active = [i for i in range(B) if not skip[i] and prefix_ids[i] is not None]

        # FORMAT-GATE skip (v3k §6-2c — NARROWED from the old "no <|/meta|>
        # anywhere" text check): CF is skipped ONLY for rows whose R_meta will
        # be zeroed/gated anyway — fmt_class ∈ {truncation, discard} plus
        # unreplaced tier-1 rows (the populator demotes those to discard).
        # DRIFT rows now RUN the CF: tier-3 plays R_meta over the recovered
        # span. Class source = the stashed parser output when the replacement
        # pass ran; otherwise classify here — same ONE parser, no duplicated
        # text logic.
        _cls_stash = gen_output.non_tensor_batch.get("dcpo_fmt_class", None)
        _rep_stash = gen_output.non_tensor_batch.get("dcpo_fmt_replaced", None)
        _resp = gen_output.batch["responses"]
        _resp_mask = gen_output.batch.get("response_mask", None)
        _attn = gen_output.batch.get("attention_mask", None)
        _plen = gen_output.batch["prompts"].shape[-1]
        _tier1 = ("swapped", "dup_open", "reversed")
        _gated = []
        for i in list(active):
            if _cls_stash is not None and i < len(_cls_stash):
                _cls = str(_cls_stash[i])
                _rep = bool(
                    _rep_stash is not None and i < len(_rep_stash)
                    and float(_rep_stash[i]) > 0.5)
            else:
                try:
                    _rm = (_resp_mask[i] if _resp_mask is not None
                           else (_attn[i][_plen:] if _attn is not None else None))
                    _cls = classify_dcpo_format(
                        _resp[i], _rm,
                        lambda ids: self.tokenizer.decode(ids, skip_special_tokens=False),
                        meta_open=meta_open,
                    )["fmt_class"]
                except Exception:
                    continue  # classify hiccup → keep the row (gates still hold downstream)
                _rep = False  # no stash = replacement never ran
            if _cls in ("truncation", "discard") or (_cls in _tier1 and not _rep):
                _gated.append(i)
        if _gated:
            active = [i for i in active if i not in set(_gated)]
            if os.environ.get("DCPO_DEBUG", "1") == "1":
                print(f"[DCPO-V3] CF skip (fmt gate truncation/discard/"
                      f"unreplaced-tier1): {len(_gated)} row(s) -> cf slot None "
                      f"(heads gated/zeroed anyway)", flush=True)

        cf_texts = [None] * B
        if not active:
            return cf_texts

        # The 2nd-gen call (spec §3.4/§3.5) is implemented in _dcpo_cf_call_engine:
        #   cf_batch = gen_batch.select_idxs(active)            # carry raw_prompt + meta_info
        #   route to cf_prefix_agent, attach prefix_ids + cf_logit_bias={meta_open:-100.0}
        #   cf_out = self._dcpo_cf_orig_generate(cf_batch)      # SAME engine, replicas awake
        #   decode cf_out.responses, assert 0 occurrences of meta_open, grade vs gts[active].
        # Fallbacks (spec §3.6) if cf_prefix_agent is unavailable: (1) chat-message prefix
        # via stock single_turn loop, (2) separate generate_sequences pass on a fresh batch.
        #
        # CRASH-SAFE: any failure in the verl 2nd-gen call → all-None cf_texts so
        # R_meta gracefully degrades (dcpo_region_rewards text-fallback still
        # supplies a conservative signal). Only the sdc_counterfactual-gated path runs
        # here; this whole method is unreachable when the flag is off.
        try:
            act_texts = self._dcpo_cf_call_engine(gen_batch, prefix_ids, active, meta_open)
        except Exception as e:  # pragma: no cover — verl/GPU only path
            print(f"[DCPO-V3] CF engine call FAILED ({type(e).__name__}: {e}); "
                  f"cf_texts=None (R_meta→text-fallback).", flush=True)
            if os.environ.get("DCPO_DEBUG", "1") == "1":
                traceback.print_exc()
            return [None] * B

        # `_dcpo_cf_call_engine` returns a parallel list of decoded CF response TEXTS
        # for `active` (<|meta|> suppressed). Map back to full-B slots; grading is the
        # consumer's job (real ground truths live there).
        for k, i in enumerate(active):
            txt = act_texts[k] if k < len(act_texts) else None
            cf_texts[i] = txt if (txt and txt.strip()) else None

        if os.environ.get("DCPO_DEBUG", "1") == "1":
            _n_txt = sum(1 for i in active if cf_texts[i] is not None)
            print(f"[DCPO-V3] CF texts: active={len(active)} non_empty={_n_txt} "
                  f"(grading deferred to consumer with real GTs)", flush=True)
        return cf_texts

    def _dcpo_cf_call_engine(self, gen_batch, prefix_ids, active, meta_open):
        """The verl 2nd-generation CALL (spec §3.4/§3.5). Build a DataProto of the
        `active` prefixes, route them to the `cf_prefix_agent` custom loop (ingests
        pre-tokenized `prefix_ids`, bypassing the chat template), suppress <|meta|>
        (id `meta_open`) for THAT call via per-row `cf_logit_bias = {meta_open: -100.0}`,
        run the SAME captured `generate_sequences`, decode the continuations, and
        return a parallel list of CF response TEXTS (one per index in `active`).

        verl API used (traced against verl source):
          - captured method   : AgentLoopManager.generate_sequences(DataProto)->DataProto
                                 (@auto_await → blocks, returns a materialized DataProto)
          - prompt source     : non_tensor_batch (tensor batch is NOT read for the prompt;
                                 agent_loop.py:523 splats per-row non_tensor into run() kwargs)
          - agent selection   : non_tensor_batch["agent_name"]="cf_prefix_agent"
                                 (agent_loop.py:491-493,552)
          - continuation prompt: non_tensor_batch["prefix_ids"] = [prompt+resp[:firstMeta]]
                                 → server_manager.generate(prompt_ids=...) → vLLM TokensPrompt
                                 (NO chat template; vllm_async_server.py:557)
          - meta suppression  : non_tensor_batch["cf_logit_bias"]={meta_open:-100.0}
                                 → SamplingParams(**sampling_params).logit_bias
                                 (verbatim splat, no key filtering; vllm_async_server.py:549)
          - return tensors    : cf_out.batch["responses"] (right-padded), stripped via
                                 attention_mask[:, prompt_len:] (agent_loop.py:808-820)
        `raw_prompt` is carried through by select_idxs (REQUIRED — _agent_loop_postprocess
        agent_loop.py:571 reads kwargs["raw_prompt"] unconditionally).
        """
        # CHUNK-DIVISIBILITY (blocker fix): AgentLoopManager.generate_sequences does
        # prompts.chunk(num_workers, strict=True) and DataProto.chunk asserts
        # len % num_workers == 0. The active (meta-bearing) count is arbitrary, so we
        # PAD the CF batch up to the MAIN batch size B (= len(gen_batch)), which is
        # divisible by the rollout-worker count by construction (the main gen of B rows
        # already chunked cleanly). The padding rows repeat active[0]'s prefix and are
        # DISCARDED after decode (return texts[:n_act]).
        n_act = len(active)
        B = len(prefix_ids)  # full main-batch size (one prefix slot per rollout, None=no-meta)
        padded = list(active) + [active[0]] * (B - n_act)  # length B, divisible
        cf_batch = gen_batch.select_idxs(padded)  # carries non_tensor (raw_prompt) + meta_info
        n_pad = len(padded)

        # per-row prefix ids (object array so numpy never collapses equal-length lists)
        pref = np.empty(n_pad, dtype=object)
        for k, i in enumerate(padded):
            pref[k] = [int(t) for t in list(prefix_ids[i])]

        cf_batch.non_tensor_batch["agent_name"] = np.array(["cf_prefix_agent"] * n_pad, dtype=object)
        cf_batch.non_tensor_batch["prefix_ids"] = pref
        bias = np.empty(n_pad, dtype=object)
        # Suppress BOTH meta tag ids: the swapped/reversed classes proved the model
        # can open meta content WITHOUT 151669 (e.g. "</think> content <|/meta|>"),
        # so banning only the opener leaves a CF leak path that contaminates
        # c_without. </think> (151668) stays ALLOWED — the CF must still close think.
        _meta_close_id = int(meta_open) + 1  # 151670 <|/meta|> (adjacent vocab id)
        # v3m CF signature suppression: banning only the TWO tag ids let the model
        # leak the reflection as PLAIN TEXT ("confidence: …"), which the leak guard
        # then ungrades — silencing R_meta (v3l: ~3/4 of CFs discarded, rmeta_pos→0).
        # Also down-bias the field-label first tokens so the CF answers directly
        # with no reflection block, raising the gradable-c_without rate. Config-
        # gated (default ON) + computed ONCE (cached). Absence-tolerant.
        _suppress_sig = bool(getattr(self.config.algorithm, "dcpo_cf_suppress_signature", True))
        _sig_ids = []
        if _suppress_sig:
            # Cache only a NON-EMPTY result: a transient first-call tokenizer
            # failure yields [] (signature_suppression_ids swallows exceptions);
            # caching [] would disable suppression for the WHOLE run (`[] is None`
            # is False). `if not _sig_ids` retries until it resolves real ids.
            _sig_ids = getattr(self, "_dcpo_cf_sig_ids", None)
            if not _sig_ids:
                _sig_ids = signature_suppression_ids(
                    lambda s: self.tokenizer.encode(s, add_special_tokens=False))
                if _sig_ids:
                    self._dcpo_cf_sig_ids = _sig_ids
        _base_bias = {int(meta_open): -100.0, _meta_close_id: -100.0}
        for _sid in _sig_ids:
            _base_bias[int(_sid)] = -100.0
        for k in range(n_pad):
            bias[k] = dict(_base_bias)
        cf_batch.non_tensor_batch["cf_logit_bias"] = bias

        # validate=False keeps it on the train sampling path (no val_kwargs override);
        # carry global_steps (read at agent_loop.py:517).
        base_meta = dict(getattr(gen_batch, "meta_info", {}) or {})
        base_meta["validate"] = False
        base_meta.setdefault("global_steps", base_meta.get("global_steps", -1))
        cf_batch.meta_info = base_meta

        cf_out = self._dcpo_cf_orig_generate(cf_batch)  # SAME engine, replicas awake
        texts = self._dcpo_cf_decode_texts(cf_out, meta_open)
        return texts[:n_act]  # discard padding rows; caller maps texts[k] for k in active

    def _dcpo_cf_decode_texts(self, cf_out, meta_open):
        """Decode CF continuations: strip right-pad via attention_mask[:, prompt_len:]
        (or response_mask), decode to text. Asserts <|meta|> did NOT leak (logit_bias)."""
        resp = cf_out.batch["responses"]
        attn = cf_out.batch.get("attention_mask", None)
        resp_mask = cf_out.batch.get("response_mask", None)
        prompt_len = cf_out.batch["prompts"].shape[-1]
        n = len(cf_out)
        texts = []
        for i in range(n):
            if resp_mask is not None:
                m = resp_mask[i].bool()
            elif attn is not None:
                m = attn[i][prompt_len:].bool()
            else:
                m = torch.ones(resp.shape[-1], dtype=torch.bool)
            ids = resp[i][m].tolist()
            ids = [int(t) for t in ids]
            if int(meta_open) in ids:
                # logit_bias should make this impossible; warn + strip rather than crash.
                print(f"[DCPO-V3] WARNING: meta_open={meta_open} leaked in CF row {i} "
                      f"despite logit_bias; stripping before grade.", flush=True)
                ids = [t for t in ids if t != int(meta_open)]
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return texts

    def _compute_reward_colocate(self, batch: DataProto) -> DataProto:
        fn = self._sdc_reward_fn
        if fn is None:
            return super()._compute_reward_colocate(batch)
        return fn(batch)

    # ─── CTSD force-inject (Plan v5 §3 H2, mode ROD_MQ_CONTRAST_INJECT) ────────
    # Two-phase rollout: generate → entropy → inject <|meta|> at the max-entropy
    # pre-answer position → regenerate. The DECISION logic is the unit-tested
    # src/training/meta_inject.plan_inject_prefixes; only the DataProto repack +
    # second generate_sequences call is verl-version-specific and lives here.
    #
    # OFF unless config.algorithm.sdc_force_inject is true → every existing mode
    # is byte-identical (Karpathy: surgical). Gated behind A.3 PASS before launch.
    #
    # NODE-SMOKE-REQUIRED: the repack below is written against verl 0.7.1's
    # DataProto/generate_sequences API but MUST be 1-step smoke-tested on the node
    # (per repo convention, task #123 pattern) before any real run — verl internals
    # cannot be exercised in the local CPU env where the core was unit-tested.
    #
    # TWO INVARIANTS the repack MUST satisfy (codex 2026-05-29), else the contrast
    # reward silently breaks:
    #  (1) MASK COHERENCE: the final `responses` tensor must include
    #      response[:p] + <|meta|> + model-generated-content + <|/meta|> so the
    #      existing find_meta_spans / meta_content_mask marks the injected block;
    #      otherwise q_contrast (T+ − T- over meta region) is empty/zero and this
    #      reduces to vanilla GRPO. (Marker-only mode = model writes the content;
    #      that content must be IN the scored response.)
    #  (2) CLOSE-RATE SAFETY: if the model fails to emit <|/meta|> the meta span
    #      runs to end-of-response (mask covers the answer). A.3 b_close≈0.68 →
    #      ~1/3 risk. Cap injected-meta length and log/alert close-rate (WandB
    #      train/inject_close_rate); drop or truncate samples whose forced block
    #      never closes within N tokens.
    def _force_inject_rollout(self, gen_batch, gen_output):
        """Return a regenerated gen_output with <|meta|> force-injected, or the
        original gen_output unchanged when force-inject is disabled."""
        algo = getattr(self.config, "algorithm", {})
        if not bool(getattr(algo, "sdc_force_inject", False)):
            return gen_output  # default path: no-op, identical to all other modes

        from .meta_inject import plan_inject_prefixes, MARKER_ONLY, GOOD_META
        tok = self.tokenizer
        meta_open = tok.convert_tokens_to_ids("<|meta|>")
        meta_close = tok.convert_tokens_to_ids("<|/meta|>")
        # inject mode (A.3 finding): "marker" (b-style, DEFAULT) injects only the
        # opening <|meta|> and lets the model fill content — the contrastive reward
        # (ROD_MQ_CONTRAST) shapes it during RL. "content" injects a fixed block.
        inject_mode = str(getattr(algo, "sdc_inject_mode", "marker"))
        template = MARKER_ONLY if inject_mode == "marker" else GOOD_META

        # (1) extract per-sample prompt ids, response ids, per-token entropy from
        #     phase-1 gen_output; (2) plan_inject_prefixes(...) → phase-2 prompts;
        #     (3) pack non-None prefixes into a DataProto and call
        #     self.actor_rollout_wg.generate_sequences(...) again; (4) merge the
        #     regenerated samples back, leaving None (no-valid-position) samples
        #     as their phase-1 rollout. See plan_inject_prefixes docstring.
        raise NotImplementedError(
            "force-inject repack is node-smoke-required — wire DataProto pack/"
            "unpack against the live verl runtime before enabling sdc_force_inject."
        )


# ─── GFN_OPSD_CONTRAST (R18c, Plan v7.2.7 codex r12-r14 LOCK) ───────────────
#
# Listwise KL distribution matching as PRIMARY aux loss; pairwise cTB stays
# diagnostic only. Verifiable signal — T+ / T- logit, no rubric / judge.
#
# Math (per sequence i in a uid-coherent group of size g = rollout.n):
#   logR_token_i = α (T+ − P_S.detach()) + β (T+ − T-)               (codex r5 + r12 detach)
#   logR_meta_i   = (logR_token_i × meta_content_mask).sum / mask.sum  (length-normalized; codex r13 D14)
#   logP_S_meta_i = (current_log_prob × meta_content_mask).sum / mask.sum
# Group softmax:
#   target_dist  = softmax(logR_meta.detach() / τ)                    (target stops gradient)
#   student_dist = softmax(logP_S_meta / τ)
# Listwise KL (forward, mode-covering — codex r12+r13 confirmed):
#   L_listwise = Σ target × (log target − log student)
# Ref-floor hinge on body (complement_mask = response_mask × (1 − meta_content_mask)):
#   L_hinge    = mean(max(0, current_log_prob − p_ref)^2)             (codex r13 D3)
# Final auxiliary loss:
#   aux        = λ × L_listwise + γ × L_hinge
def compute_sdc_gfn_actor_loss(
    current_log_prob: torch.Tensor,
    model_inputs: dict,
    config=None,
) -> torch.Tensor:
    """R18c GFN_OPSD_CONTRAST: listwise KL primary + ref-floor hinge diagnostic.

    Args
    ----
    current_log_prob : Tensor [B, T]
        Per-token log-prob from veRL's micro-batch forward — this is the only
        tensor that carries a gradient into the actor.
    model_inputs : dict
        Carries the SDC tensors (see `verl_sdc_utils.compute_sdc_gdpo_advantage`)
        plus the GFN hyperparams transported via `data.batch` (codex r13 D13):
          • sdc_teacher_pos_log_probs [B, T]   (no grad)
          • sdc_teacher_neg_log_probs [B, T]   (no grad)
          • sdc_meta_mask             [B, T]   (tag-inclusive; see note)
          • old_log_probs             [B, T]   (P_S used for attractive gain)
          • response_mask             [B, T]
          • uid                       [B]      np.ndarray / list of group ids
          • sdc_alpha_attr            float    default 0.5
          • sdc_beta_contrast         float    default 0.5
          • sdc_gfn_tau               float    default 1.0
          • sdc_gfn_lambda            float    default 0.1  (production fixed)
          • sdc_gfn_fluency_gamma     float    default 0.01 (ref-floor hinge)
          • sdc_log_ratio_clamp       float    default 10.0
          • sdc_ref_log_probs         [B, T]   optional; needed for hinge term

    Returns
    -------
    Tensor (0-dim) — scalar aux loss to add to `pg_loss` in the ppo_loss hook.

    Notes
    -----
    - `meta_content_mask` here uses `sdc_meta_mask` directly (tag-inclusive),
      matching the R18a/R18b fallback. The codex r7 exclusion of tag positions
      lives in `verl_sdc_utils._meta_mask_from_token_ids` and is consistent
      across both code paths.
    - All non-`current_log_prob` tensors are `.detach()`-ed inside `logR` per
      codex r12 — gradient flows ONLY through `current_log_prob` via
      `log_P_S_meta` and the body hinge.
    - On any non-finite aux, we fall back to a zero scalar so a single bad
      microbatch never poisons the optimizer.
    """
    # Tolerate either dict or TensorDict for `model_inputs`. TensorDicts do
    # not store arbitrary Python objects, so uid is typically transported in
    # `non_tensor_batch` and copied here for the test path. When uid is not
    # available at all, we fall back to "single group across the microbatch",
    # which is a no-op for L_listwise (one group = degenerate softmax).
    teacher_pos = model_inputs["sdc_teacher_pos_log_probs"]
    teacher_neg = model_inputs["sdc_teacher_neg_log_probs"]
    meta_mask = model_inputs["sdc_meta_mask"]
    old_log_probs = model_inputs["old_log_probs"]
    response_mask = model_inputs["response_mask"]
    try:
        uid = model_inputs["uid"]
    except KeyError:
        uid = ["__single_group__"] * int(current_log_prob.size(0))

    alpha = float(model_inputs.get("sdc_alpha_attr", 0.5))
    beta = float(model_inputs.get("sdc_beta_contrast", 0.5))
    tau = float(model_inputs.get("sdc_gfn_tau", 1.0))
    lambda_listwise = float(model_inputs.get("sdc_gfn_lambda", 0.1))
    gamma_hinge = float(model_inputs.get("sdc_gfn_fluency_gamma", 0.01))
    clamp = float(model_inputs.get("sdc_log_ratio_clamp", 10.0))

    # ── Objective dispatch (R21 direction A — additive, zero-touch) ────────
    # Transport mirrors the proven `mode` path: read the per-batch marker first
    # (test path supplies it via the dict model_inputs), then the module-level
    # `_ACTIVE_SDC_CONTEXT` cache populated deterministically in main_task
    # (the same mechanism `_sdc_mode` uses), defaulting to "listwise_kl".
    #   • "listwise_kl"  → GFN_OPSD_CONTRAST (R18c) — EXACT prior behavior,
    #                       numerically byte-identical (R18c never sets the key).
    #   • "pairwise_ctb" → STABLE_GFN (R21, A) — Z-free pairwise contrastive
    #                       Trajectory Balance + frozen_ref baseline +
    #                       reward-temperature.
    def _ctx(key, default):
        try:
            v = model_inputs.get(key, None)
        except Exception:
            v = None
        if v is None:
            try:
                v = _ACTIVE_SDC_CONTEXT.get(key, None)
            except Exception:
                v = None
        return default if v is None else v

    gfn_objective = str(_ctx("sdc_gfn_objective", "listwise_kl"))
    gfn_reward_baseline = str(_ctx("sdc_gfn_reward_baseline", "none"))
    # Reward-temperature T_R for pairwise cTB (target = logR / T_R; student is
    # NOT divided — per Plan direction-A spec). Reuses sdc_gfn_tau ONLY for the
    # listwise softmax; cTB uses its own T_R so the two objectives stay
    # independent and single-variable.
    reward_temp = float(_ctx("sdc_reward_temperature", 1.0))

    device = current_log_prob.device
    dtype = current_log_prob.dtype

    if tau <= 0:
        raise ValueError(f"sdc_gfn_tau must be > 0, got {tau}")
    if gfn_objective == "pairwise_ctb" and reward_temp <= 0:
        raise ValueError(
            f"sdc_reward_temperature must be > 0 for pairwise_ctb, got {reward_temp}"
        )

    # logR with both detach-able terms detached (codex r12 D2 + r5 sign).
    teacher_pos_d = teacher_pos.detach().to(device=device, dtype=dtype)
    teacher_neg_d = teacher_neg.detach().to(device=device, dtype=dtype)
    old_log_probs_d = old_log_probs.detach().to(device=device, dtype=dtype)

    gain_attr = (teacher_pos_d - old_log_probs_d).clamp(-clamp, clamp)
    gain_contrast = (teacher_pos_d - teacher_neg_d).clamp(-clamp, clamp)
    logR_token = alpha * gain_attr + beta * gain_contrast  # [B, T]

    # meta_content_mask: accept the tag-inclusive mask (see notes above).
    meta_content_mask = meta_mask.to(device=device, dtype=dtype)
    denom = meta_content_mask.sum(-1).clamp_min(1.0)

    # Length-normalized scores per rollout (codex r13 D14).
    logR_meta = (logR_token * meta_content_mask).sum(-1) / denom  # [B]
    log_P_S_meta = (current_log_prob * meta_content_mask).sum(-1) / denom  # [B], grad flows

    # ── Frozen-ref logprob (shared: cTB baseline + body hinge) ────────────
    # Fetched ONCE here. Try sdc-prefixed key first (legacy/test path), then
    # veRL native `ref_log_prob` (codex r2 #5). Listwise numerics are unchanged
    # — the hinge below reuses this exact tensor, same as before the move.
    p_ref = model_inputs.get("sdc_ref_log_probs", None)
    if p_ref is None:
        p_ref = model_inputs.get("ref_log_prob", None)
    if p_ref is not None:
        p_ref_t = p_ref.detach().to(device=device, dtype=dtype)
        log_P_ref_meta = (p_ref_t * meta_content_mask).sum(-1) / denom  # [B], no grad
    else:
        p_ref_t = None
        log_P_ref_meta = torch.zeros_like(log_P_S_meta)

    # ── Shared uid-group construction (codex r13 D13 + r12 D2) ────────────
    # Microbatch is uid-coherent only if rollout.n == ppo_micro_batch_size_per_gpu
    # AND balance_batch=False (codex r12 D2). YAML must enforce both.
    if hasattr(uid, "tolist"):
        uid_list = uid.tolist()
    else:
        uid_list = list(uid)

    groups: dict = {}
    for i, u in enumerate(uid_list):
        groups.setdefault(u, []).append(i)

    if gfn_objective == "pairwise_ctb":
        # ── R21 direction A: Z-free pairwise contrastive Trajectory Balance ──
        # GFlowNet TB over the meta region: logZ + logP_S(τ) = logR(τ). logZ is
        # intractable; for two rollouts i,j sharing a uid (⇒ same prompt ⇒ same
        # logZ) it CANCELS in the pairwise difference:
        #     (s_i − s_j) == (r_i − r_j)/T_R
        #   s_i = log_P_S_meta_i − [frozen_ref baseline]   (grad via current_lp)
        #   r_i = logR_meta_i                              (detached; reward-temp
        #                                                   applied to the target
        #                                                   difference only)
        # frozen_ref baseline = −log_P_ref_meta (detached). codex review
        # (gpt-5.5 NEEDS_WORK pt.1): this is NOT a pure variance-reduction
        # control variate — because (logP_ref_i − logP_ref_j) does NOT cancel
        # in the pairwise difference, it CHANGES the TB fixed point. That is
        # the INTENDED Stable-GFN relative-TB form: the constraint becomes
        #   (logP_S_i − logP_S_j) − (logP_ref_i − logP_ref_j) == (r_i−r_j)/T_R
        # i.e. the policy's *improvement over the frozen ref* (not its raw
        # log-flow) is matched to the reward difference — anchoring the
        # absolute scale to the SFT ref. With baseline="none" this reduces to
        # plain raw-policy pairwise TB. Student is NOT divided by T_R
        # (Plan direction-A spec: target = logR/T_R, student raw).
        use_ref_baseline = (gfn_reward_baseline == "frozen_ref") and (p_ref_t is not None)
        s_all = log_P_S_meta - (log_P_ref_meta if use_ref_baseline else 0.0)  # [B], grad
        r_all = (logR_meta.detach() / reward_temp)  # [B], no grad

        ctb_terms: list = []
        for u, idx in groups.items():
            if len(idx) < 2:
                continue  # need ≥ 2 rollouts for a pair
            # Same robust mask as listwise: drop empty-meta rollouts so the
            # pairwise residual is never formed on a zero-length meta region.
            idx_filtered = [i for i in idx if meta_content_mask[i].sum() > 0]
            if len(idx_filtered) < 2:
                continue
            idx_t = torch.tensor(idx_filtered, device=device, dtype=torch.long)
            s_g = s_all.index_select(0, idx_t)  # [g], grad flows
            r_g = r_all.index_select(0, idx_t)  # [g], no grad
            # Upper-triangular (i<j) pairwise residuals → mean squared error.
            ds = s_g.unsqueeze(1) - s_g.unsqueeze(0)  # [g,g]
            dr = r_g.unsqueeze(1) - r_g.unsqueeze(0)  # [g,g], no grad
            resid = ds - dr  # [g,g]
            g = resid.size(0)
            tri = torch.triu(torch.ones(g, g, device=device, dtype=torch.bool), diagonal=1)
            ctb_terms.append((resid[tri] ** 2).mean())

        if not ctb_terms:
            L_primary = torch.zeros((), device=device, dtype=dtype)
        else:
            L_primary = torch.stack(ctb_terms).mean()
        L_listwise = torch.zeros((), device=device, dtype=dtype)  # diag-only when cTB
        L_ctb = L_primary
        groups_used = len(ctb_terms)
    else:
        # ── listwise_kl: GFN_OPSD_CONTRAST (R18c) — UNCHANGED, byte-identical ─
        kl_terms: list = []
        for u, idx in groups.items():
            if len(idx) < 2:
                continue  # need ≥ 2 candidates for a listwise KL term
            # codex r2 #3: drop rollouts with empty meta from THIS listwise group
            # (previous version included them as score 0, contaminating target softmax).
            idx_filtered = [i for i in idx if meta_content_mask[i].sum() > 0]
            if len(idx_filtered) < 2:
                continue
            idx_t = torch.tensor(idx_filtered, device=device, dtype=torch.long)
            logR_group = logR_meta.index_select(0, idx_t)  # [g], no grad (logR is built from detached terms)
            log_P_S_group = log_P_S_meta.index_select(0, idx_t)  # [g], grad flows

            target_dist = torch.softmax(logR_group.detach() / tau, dim=-1)  # explicit .detach() safety
            student_log_dist = torch.log_softmax(log_P_S_group / tau, dim=-1)
            # Forward KL(target ‖ student) = Σ target × (log target − log student)
            kl = (target_dist * (target_dist.clamp_min(1e-9).log() - student_log_dist)).sum()
            kl_terms.append(kl)

        if not kl_terms:
            L_primary = torch.zeros((), device=device, dtype=dtype)
        else:
            L_primary = torch.stack(kl_terms).mean()
        L_listwise = L_primary
        L_ctb = torch.zeros((), device=device, dtype=dtype)  # diag-only when listwise
        groups_used = len(kl_terms)

    # ── Ref-floor hinge on body (codex r13 D3) — reuses the shared p_ref_t ──
    if p_ref_t is not None and gamma_hinge > 0:
        complement_mask = (response_mask.to(device=device, dtype=dtype)
                           * (1.0 - meta_content_mask))
        hinge_token = torch.clamp_max(current_log_prob - p_ref_t, 0.0) ** 2
        denom_body = complement_mask.sum().clamp_min(1.0)
        L_hinge = (hinge_token * complement_mask).sum() / denom_body
    else:
        L_hinge = torch.zeros((), device=device, dtype=dtype)

    # lambda_listwise (sdc_gfn_lambda) is the coefficient on the PRIMARY aux
    # loss for BOTH objectives (listwise KL or pairwise cTB) — single shared
    # delivery-strength knob so the objective swap stays single-variable.
    aux_loss = lambda_listwise * L_primary + gamma_hinge * L_hinge

    # codex review (gpt-5.5 NEEDS_WORK pt.4): a pairwise_ctb microbatch with
    # NO usable uid group (incomplete / singleton groups from a wrong
    # rollout.n / balance_batch) silently trains ZERO cTB signal while the run
    # still "works". Surface it loudly: a per-batch wandb flag + a one-time
    # process warning. (The startup config guard in main_task fail-fasts the
    # actual misconfig; this catches residual per-batch degeneracy.)
    ctb_inactive = bool(gfn_objective == "pairwise_ctb" and groups_used == 0)
    if ctb_inactive and not _CTB_INACTIVE_WARNED["done"]:
        _CTB_INACTIVE_WARNED["done"] = True
        print(
            "[SDC][GFN][WARN] pairwise_ctb produced 0 usable uid groups in a "
            "microbatch — cTB delivery is INACTIVE. Check rollout.n == "
            "ppo_micro_batch_size_per_gpu and trainer.balance_batch=False. "
            "(this warning prints once; watch wandb sdc_gfn_ctb_inactive)"
        )

    # codex r2 #7: surface primary / hinge / logR diagnostics for wandb.
    diag = {
        "sdc_gfn_objective": gfn_objective,
        "sdc_gfn_kl_listwise": float(L_listwise.detach()),
        "sdc_gfn_ctb_loss": float(L_ctb.detach()),
        "sdc_gfn_ctb_inactive": ctb_inactive,
        "sdc_gfn_ref_hinge": float(L_hinge.detach()),
        "sdc_gfn_logR_mean": float(logR_meta.detach().mean()) if logR_meta.numel() else 0.0,
        "sdc_gfn_logR_std": float(logR_meta.detach().std()) if logR_meta.numel() > 1 else 0.0,
        "sdc_gfn_groups_used": int(groups_used),
    }

    if not torch.isfinite(aux_loss):
        # Single bad microbatch must never poison the optimizer. Log via wandb
        # if available; otherwise silently zero-out (codex r13 finite check).
        return torch.zeros((), device=device, dtype=dtype), diag

    return aux_loss, diag


def _patch_actor_loss_for_gfn():
    """codex r13 D13: hook `ppo_loss` in both losses.py AND engine_workers.py.

    Active-path discovery: when `trainer.use_legacy_worker_impl: disable`, the
    new `engine_workers.py` imports `ppo_loss` once at module load. Patching
    `losses_mod.ppo_loss` alone does not retroactively rebind the symbol the
    engine workers already captured — we must patch both.

    The mode dispatch is read from `data.batch['_sdc_mode']` (transported via
    DataProto), NOT from actor config. This keeps the hook plug-and-play with
    every existing config and survives veRL's actor-config validators.
    """
    try:
        import verl.workers.utils.losses as losses_mod  # type: ignore
    except (ImportError, AttributeError) as exc:
        print(f"[SDC][GFN] skipped ppo_loss hook (losses module unavailable): "
              f"{type(exc).__name__}: {exc}")
        return

    try:
        import verl.workers.engine_workers as engine_mod  # type: ignore
    except (ImportError, AttributeError):
        engine_mod = None

    if getattr(losses_mod.ppo_loss, "_sdc_gfn_hook", False):
        return  # already patched

    original_ppo_loss = losses_mod.ppo_loss

    def sdc_gfn_ppo_loss(config, model_output, data, dp_group):
        policy_loss, metrics = original_ppo_loss(config, model_output, data, dp_group)
        # Mode dispatch (codex r13 D13): prefer the explicit per-batch marker
        # `_sdc_mode` if veRL forwarded one; fall back to the module-level
        # `_ACTIVE_SDC_CONTEXT["mode"]` cache populated in main_task. This
        # tolerates TensorDicts that disallow string entries while keeping
        # the plug-and-play marker semantics from the spec.
        sdc_mode = ""
        try:
            if hasattr(data, "batch") and "_sdc_mode" in getattr(data, "batch", {}):
                sdc_mode = str(data.batch["_sdc_mode"])
        except Exception:
            sdc_mode = ""
        if not sdc_mode:
            try:
                sdc_mode = _ACTIVE_SDC_CONTEXT.get("mode", "")
            except Exception:
                sdc_mode = ""
        # R21 direction A is additive: STABLE_GFN routes through the SAME
        # compute_sdc_gfn_actor_loss, which dispatches internally on
        # sdc_gfn_objective. GFN_OPSD_CONTRAST (R18c) runtime path is
        # byte-identical (objective defaults to "listwise_kl").
        if sdc_mode in ("GFN_OPSD_CONTRAST", "STABLE_GFN", "STABLE_GFN_C2FIX"):
            try:
                # codex r2 #4 fix: veRL 0.7.1 ppo_loss signature passes
                # `data` as TensorDict (NOT DataProto). The .batch attribute
                # does not exist; use `data` directly for subscript access.
                # Both TensorDict and DataProto support .get()/__getitem__.
                aux_data = data.batch if hasattr(data, "batch") else data
                # codex r2 #4 fix: pad model_output["log_probs"] to [B, T] form
                # to match how original_ppo_loss processes log_prob.
                try:
                    from verl.workers.utils.losses import no_padding_2_padding
                    current_lp = no_padding_2_padding(model_output["log_probs"], data)
                except (ImportError, AttributeError, KeyError):
                    current_lp = model_output["log_probs"]  # fallback for tests
                result = compute_sdc_gfn_actor_loss(
                    current_log_prob=current_lp,
                    model_inputs=aux_data,
                    config=config,
                )
                # codex r2 #7 fix: accept either tuple (aux, diag) or scalar aux.
                if isinstance(result, tuple):
                    aux, gfn_diag = result
                else:
                    aux, gfn_diag = result, {}
                policy_loss = policy_loss + aux
                metrics["sdc_gfn_aux_loss"] = float(aux.detach())
                for k, v in gfn_diag.items():
                    metrics[k] = v
            except Exception as e:
                # Never crash training on aux-loss failure; surface the diag.
                metrics["sdc_gfn_aux_loss_error"] = f"{type(e).__name__}: {e}"
        return policy_loss, metrics

    sdc_gfn_ppo_loss._sdc_gfn_hook = True  # type: ignore[attr-defined]
    losses_mod.ppo_loss = sdc_gfn_ppo_loss
    if engine_mod is not None and hasattr(engine_mod, "ppo_loss"):
        engine_mod.ppo_loss = sdc_gfn_ppo_loss
        # codex r13 strict assertion: refuse to proceed if either path is unhooked.
        assert getattr(engine_mod.ppo_loss, "_sdc_gfn_hook", False), (
            "GFN hook injection failed in engine_workers — refusing to start"
        )
    print("[SDC][GFN] ppo_loss hooked for GFN_OPSD_CONTRAST (R18c, Plan v7.2.7)")


def _patch_verl_for_sdc():
    import verl.trainer.ppo.ray_trainer as ray_trainer_module
    from verl.single_controller.ray import RayWorkerGroup
    original_compute_advantage = ray_trainer_module.compute_advantage

    def patched_compute_advantage(
        data: DataProto,
        adv_estimator,
        gamma=1.0,
        lam=1.0,
        num_repeat=1,
        norm_adv_by_std_in_grpo=True,
        config=None,
    ):
        # _REGION_ROUTED_MODES (TRIOBJ_DCPO_V2/V3) need the per-region advantage routing
        # (R_meta -> META_CONTENT tokens only) and the _populate producer — but they run
        # teacher-FREE (sdc_enabled=false). The original gate required sdc_enabled=true,
        # so region modes silently fell through to plain summed-GDPO (correctness broadcast
        # crushes meta = the v1 failure). Route region modes regardless of sdc_enabled;
        # _attach_teacher_signals short-circuits (no teacher forward) for these modes.
        try:
            _adv_sdc_mode = (config.get("sdc_mode", "") if config is not None else "") \
                or _ACTIVE_SDC_CONTEXT.get("mode", "")
        except Exception:
            _adv_sdc_mode = ""
        _adv_region = _adv_sdc_mode in _REGION_ROUTED_MODES
        if _is_gdpo_estimator(adv_estimator) and config is not None and \
           (config.get("sdc_enabled", False) or _adv_region):
            if "response_mask" not in data.batch.keys():
                data.batch["response_mask"] = ray_trainer_module.compute_response_mask(data)
            data = _attach_teacher_signals(data)
            advantages, returns = compute_sdc_gdpo_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=data.batch["response_mask"],
                index=data.non_tensor_batch["uid"],
                batch=data.batch,
                non_tensor_batch=data.non_tensor_batch,
                config=config,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
            return data
        return original_compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    ray_trainer_module.compute_advantage = patched_compute_advantage

    # GFN_OPSD_CONTRAST (R18c) ppo_loss hook — installed once per process.
    # Idempotent: subsequent calls early-return on the `_sdc_gfn_hook` marker.
    _patch_actor_loss_for_gfn()

    if not getattr(RayWorkerGroup, "_sdc_checkpoint_wrappers_applied", False):
        def _wg_update_weights(self, global_steps=None):
            return self.execute_all_async("update_weights", global_steps=global_steps)

        def _wg_execute_checkpoint_engine(self, methods, *args, **kwargs):
            return self.execute_all_async("execute_checkpoint_engine", methods, *args, **kwargs)

        RayWorkerGroup.update_weights = _wg_update_weights
        RayWorkerGroup.execute_checkpoint_engine = _wg_execute_checkpoint_engine
        RayWorkerGroup._sdc_checkpoint_wrappers_applied = True
        print("[SDC] patched RayWorkerGroup checkpoint wrappers for veRL 0.7.1")

    try:
        import verl.workers.rollout.vllm_rollout.vllm_async_server as vllm_async_server
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.v1.engine.async_llm import AsyncLLM as V1AsyncLLM

        if not getattr(vllm_async_server, "_sdc_asyncllm_patch_applied", False):
            class _CompatAsyncLLM:
                @staticmethod
                def from_vllm_config(*args, **kwargs):
                    try:
                        return V1AsyncLLM.from_vllm_config(*args, **kwargs)
                    except ValueError as exc:
                        if "VLLM_USE_V1=False" not in str(exc):
                            raise
                        return AsyncLLMEngine.from_vllm_config(*args, **kwargs)

            vllm_async_server.AsyncLLM = _CompatAsyncLLM
            vllm_async_server._sdc_asyncllm_patch_applied = True
            print("[SDC] patched vLLM AsyncLLM compatibility for vllm>=0.8 fallback")
    except Exception as exc:
        print(f"[SDC] skipped vLLM AsyncLLM patch: {type(exc).__name__}: {exc}")


import hydra


@hydra.main(config_path="../../configs", config_name="verl_sdc_e21r_shared", version_base=None)
def main(config):
    if not ray.is_initialized():
        # AMLT single-node jobs can expose a non-loopback pod IP that makes
        # Ray's default head bootstrap path hang while waiting for GCS.
        # For this veRL workload we only need a local head on the same node, so
        # pin Ray bootstrap to loopback and skip the dashboard to reduce
        # startup fragility.
        ray.init(
            include_dashboard=False,
            _node_ip_address="127.0.0.1",
            # propagate PYTHONPATH to Ray workers so hydra.utils.instantiate can
            # import custom _target_ classes (e.g. the E.9 BCIConfAgentLoop) by
            # FQDN inside the rollout workers. Harmless for every other mode (the
            # repo is already importable); removes the one registration unknown.
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "PYTHONPATH": os.environ.get("PYTHONPATH", "/scratch/metacognition"),
            }},
        )
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from omegaconf import OmegaConf, open_dict
    from pprint import pprint
    from verl.single_controller.ray import RayWorkerGroup
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
    from verl.experimental.reward_loop import migrate_legacy_reward_impl

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Migrate any legacy reward_model.* keys into the new reward.* layout so that
    # RayPPOTrainer internals (need_reward_model, reward_loop_manager) see a
    # consistent config tree.
    try:
        config = migrate_legacy_reward_impl(config)
    except Exception:
        # Migration is best-effort; config may already be in the new layout.
        pass

    logger_cfg = list(config.trainer.get("logger", []))
    has_wandb_key = bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY"))
    if "wandb" in logger_cfg and not has_wandb_key:
        filtered = [name for name in logger_cfg if name != "wandb"] or ["console"]
        with open_dict(config.trainer):
            config.trainer.logger = filtered
        print("[SDC] WANDB key absent; forcing trainer.logger=%s" % filtered)

    reward_fn_cfg = config.reward.get("custom_reward_function", None)
    if reward_fn_cfg is not None and not reward_fn_cfg.get("path"):
        with open_dict(config.reward.custom_reward_function):
            config.reward.custom_reward_function.path = os.path.abspath(__file__)
            config.reward.custom_reward_function.name = "reward_loop_score"
        print("[SDC] configured custom reward_loop fallback:", config.reward.custom_reward_function.path)

    _patch_verl_for_sdc()

    trust_remote_code = config.data.get("trust_remote_code", False)
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path,
        use_shm=config.actor_rollout_ref.model.get("use_shm", False),
    )
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    mode = config.get("mode", "SDC_SHARED")
    if mode not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown mode='{mode}'. Available: {sorted(REWARD_CONFIGS.keys())}"
        )
    if mode == "OPSD_META":
        # KL distillation auxiliary loss is NOT implemented yet. The advantage
        # path falls back to RLSD_META_ATTR (T+ only, attractive). Refusing to
        # silently run a half-implemented mode prevents accidental wasted runs.
        raise NotImplementedError(
            "mode=OPSD_META requires KL distillation loss in the actor — "
            "phase-2 work, not yet wired. Use RLSD_META_ATTR or "
            "RLSD_META_CONTRAST for now."
        )
    reward_cfg = REWARD_CONFIGS[mode]

    # FAIL-FAST gate-coherence check (codereview CRITICAL-2, the sdc_enabled-class
    # bug): patched_compute_advantage runs the SDC branch only when sdc_enabled is
    # truthy OR the mode is region-routed. A teacher-ON mode whose YAML omits/false
    # sdc_enabled would otherwise SILENTLY train as plain GDPO while labeled e.g.
    # ROD_MQ_CONTRAST — exactly how the v2/v3 region routing stayed off unnoticed.
    _teacher_on_modes = _SINGLE_TEACHER_MODES | _CONTRASTIVE_MODES | _FORCED_META_MODES
    _alg_for_gate = config.get("algorithm", {}) or {}
    if mode in _teacher_on_modes and not bool(_alg_for_gate.get("sdc_enabled", False)):
        raise ValueError(
            f"mode='{mode}' is a teacher-ON self-distill mode but "
            f"algorithm.sdc_enabled is not true — the teacher forward and SDC "
            f"advantage shaping would be SILENTLY skipped (plain GDPO). Set "
            f"algorithm.sdc_enabled: true in the YAML (or use a vanilla/region mode)."
        )

    # Make mode visible to runtime hooks (advantage compute & teacher attach).
    _ACTIVE_SDC_CONTEXT["mode"] = mode
    # Mirror mode into algorithm config so verl_sdc_utils.compute_sdc_gdpo_advantage
    # (which only receives algorithm config) can dispatch on it. Disable struct
    # mode locally so we can write a key that may not exist in legacy yamls.
    if "algorithm" in config:
        try:
            from omegaconf import OmegaConf as _OC
            _was_struct = _OC.is_struct(config.algorithm)
            _OC.set_struct(config.algorithm, False)
            config.algorithm.sdc_mode = mode
            if _was_struct:
                _OC.set_struct(config.algorithm, True)
        except Exception:
            # Last-resort dict-style assignment.
            try:
                config["algorithm"]["sdc_mode"] = mode
            except Exception:
                pass  # cannot inject; compute_sdc_gdpo_advantage will use default
    # Single source of truth: prefer config.algorithm.gdpo_reward_weights / gdpo_reward_keys.
    # REWARD_CONFIGS supplies the functions (which cannot live in YAML) and the
    # default weights/keys used when the YAML omits them.
    alg_cfg = config.get("algorithm", {}) or {}
    # R21 direction A (STABLE_GFN): stash the GFN-objective hyperparams into
    # the module-level context using the SAME deterministic transport `mode`
    # uses (the GFN aux path runs in a remote worker that does not receive the
    # hydra config; reading via _ACTIVE_SDC_CONTEXT avoids the murky batch
    # transport). Existing modes never set these keys in YAML → the defaults
    # ("listwise_kl"/"none"/1.0) preserve GFN_OPSD_CONTRAST byte-identically.
    _ACTIVE_SDC_CONTEXT["sdc_gfn_objective"] = str(
        alg_cfg.get("sdc_gfn_objective", "listwise_kl")
    )
    _ACTIVE_SDC_CONTEXT["sdc_gfn_reward_baseline"] = str(
        alg_cfg.get("sdc_gfn_reward_baseline", "none")
    )
    _ACTIVE_SDC_CONTEXT["sdc_reward_temperature"] = float(
        alg_cfg.get("sdc_reward_temperature", 1.0)
    )
    # E.4 self-distill contrast variant (plan_ctsd_E4_selfdistill_rl). Same
    # deterministic transport as sdc_gfn_objective: the teacher-attach hook runs
    # in a worker that does not receive the hydra config, so read it via
    # _ACTIVE_SDC_CONTEXT. Default "decoy" → byte-identical for every existing
    # mode/config/test (none of which set this key).
    _cv = str(alg_cfg.get("sdc_contrast_variant", "decoy"))
    if _cv not in _CONTRAST_VARIANTS:
        raise ValueError(
            f"algorithm.sdc_contrast_variant={_cv!r} not in {_CONTRAST_VARIANTS}"
        )
    _ACTIVE_SDC_CONTEXT["sdc_contrast_variant"] = _cv
    # Arm-2 parameterized teacher-prompt slot (deliverable #2). Resolved ONCE
    # at launch and stashed via the SAME deterministic transport. Default
    # "r10v2_baseline" -> "" prefix => byte-identical teacher conditioning for
    # every existing mode (none of which read this key) and for an Arm-2 run
    # that has not yet picked a strengthened set from the FROZEN G2 decision.
    _tp_set = str(alg_cfg.get("sdc_teacher_prompt_set", "r10v2_baseline"))
    _ACTIVE_SDC_CONTEXT["sdc_teacher_prompt_set"] = _tp_set
    # ── G2 PER-TEACHER SLOTS (deliverable #2, 2026-05-19) ──────────────────
    # PRECEDENCE (codex-converged): an explicitly-set per-teacher key wins;
    # otherwise it INHERITS the legacy shared `sdc_teacher_prompt_set` (so the
    # pre-existing single-slot config + every prior run is byte-identical).
    # `None` sentinel distinguishes "key absent -> inherit legacy" from "key
    # explicitly set to r10v2_baseline -> use baseline" (both yield "" prefix,
    # but the distinction keeps the back-compat semantics explicit & logged).
    _pos_set_raw = alg_cfg.get("sdc_position_teacher_prompt_set", None)
    _con_set_raw = alg_cfg.get("sdc_content_teacher_prompt_set", None)
    _pos_set = str(_pos_set_raw) if _pos_set_raw is not None else _tp_set
    _con_set = str(_con_set_raw) if _con_set_raw is not None else _tp_set
    _ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_set"] = _pos_set
    _ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_set"] = _con_set
    if mode == "ROD_PT2_E21CTRL":
        # Legacy shared prefix kept resolved for back-compat / logging.
        _ACTIVE_SDC_CONTEXT["sdc_teacher_prompt_prefix"] = (
            _resolve_teacher_prompt_prefix(_tp_set)
        )
        _ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_prefix"] = (
            _resolve_teacher_prompt_prefix(_pos_set)
        )
        _ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_prefix"] = (
            _resolve_teacher_prompt_prefix(_con_set)
        )
        print(
            "[SDC][Arm2] per-teacher prompt slots: position=%s "
            "(prefix_len=%d) content=%s (prefix_len=%d) "
            "[legacy shared=%s, precedence: explicit per-teacher > shared]"
            % (
                _pos_set,
                len(_ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_prefix"]),
                _con_set,
                len(_ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_prefix"]),
                _tp_set,
            )
        )
    else:
        _ACTIVE_SDC_CONTEXT["sdc_teacher_prompt_prefix"] = ""
        _ACTIVE_SDC_CONTEXT["sdc_position_teacher_prompt_prefix"] = ""
        _ACTIVE_SDC_CONTEXT["sdc_content_teacher_prompt_prefix"] = ""
    print(
        "[SDC][GFN] objective=%s reward_baseline=%s reward_temperature=%s"
        % (
            _ACTIVE_SDC_CONTEXT["sdc_gfn_objective"],
            _ACTIVE_SDC_CONTEXT["sdc_gfn_reward_baseline"],
            _ACTIVE_SDC_CONTEXT["sdc_reward_temperature"],
        )
    )
    # codex review (gpt-5.5 NEEDS_WORK pt.4): grouped pairwise cTB is only
    # well-posed when each microbatch is uid-coherent. FAIL FAST at launch on
    # the config invariant rather than silently training a degenerate (0-group)
    # objective for 300 steps. Mirrors the R18c listwise constraint (codex r12
    # D2) but now hard-enforced for the cTB delivery path.
    if mode in ("STABLE_GFN", "STABLE_GFN_C2FIX") and _ACTIVE_SDC_CONTEXT["sdc_gfn_objective"] == "pairwise_ctb":
        _arr = config.actor_rollout_ref
        _n = int(_arr.rollout.n)
        _micro = int(_arr.actor.ppo_micro_batch_size_per_gpu)
        _bal = bool(config.trainer.get("balance_batch", True))
        if _n != _micro or _bal:
            raise ValueError(
                "STABLE_GFN+pairwise_ctb requires uid-coherent microbatches: "
                f"rollout.n ({_n}) must equal "
                f"actor.ppo_micro_batch_size_per_gpu ({_micro}) AND "
                f"trainer.balance_batch must be False (got {_bal}). "
                "Otherwise grouped cTB silently degenerates to 0 usable groups."
            )
        print(
            "[SDC][GFN] STABLE_GFN+pairwise_ctb uid-coherence OK "
            f"(n={_n}==micro={_micro}, balance_batch=False)"
        )
    yaml_weights = alg_cfg.get("gdpo_reward_weights", None)
    yaml_keys = alg_cfg.get("gdpo_reward_keys", None)
    if yaml_weights is not None:
        resolved_weights = list(yaml_weights)
    else:
        resolved_weights = list(reward_cfg["weights"])
    if yaml_keys is not None:
        resolved_keys = list(yaml_keys)
    else:
        resolved_keys = list(reward_cfg["keys"])
    if len(resolved_weights) != len(reward_cfg["funcs"]):
        raise ValueError(
            f"gdpo_reward_weights length ({len(resolved_weights)}) does not match "
            f"number of reward funcs ({len(reward_cfg['funcs'])}) in mode={mode}"
        )
    if len(resolved_keys) != len(reward_cfg["funcs"]):
        raise ValueError(
            f"gdpo_reward_keys length ({len(resolved_keys)}) does not match "
            f"number of reward funcs ({len(reward_cfg['funcs'])}) in mode={mode}"
        )
    print(f"[SDC] reward weights: {resolved_keys} = {resolved_weights} (source={'yaml' if yaml_weights is not None else 'default'})")
    reward_fn = MetaCotSDCRewardManager(
        tokenizer=tokenizer,
        reward_funcs=reward_cfg["funcs"],
        reward_weights=resolved_weights,
        reward_keys=resolved_keys,
        num_examine=config.get("num_examine", 0),
    )
    val_reward_fn = MetaCotSDCRewardManager(
        tokenizer=tokenizer,
        reward_funcs=reward_cfg["funcs"],
        reward_weights=resolved_weights,
        reward_keys=resolved_keys,
        num_examine=1,
    )

    if config.actor_rollout_ref.actor.strategy not in ("fsdp", "fsdp2"):
        raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")
    # veRL 0.7.1 colocated checkpoint sync expects the actor/ref worker group to
    # expose async `update_weights()` / `execute_checkpoint_engine()` methods.
    # Those live on engine_workers.ActorRolloutRefWorker; the fsdp_workers base
    # class only provides them on a separate Async* subclass, which the current
    # RayPPOTrainer path does not instantiate here.
    from verl.workers.engine_workers import ActorRolloutRefWorker
    from verl.workers.fsdp_workers import CriticWorker
    ray_worker_group_cls = RayWorkerGroup

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = "global_pool"
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    train_dataset = create_rl_dataset(
        config.data.train_files,
        config.data,
        tokenizer,
        processor,
        is_train=True,
        max_samples=config.data.get("train_max_samples", -1),
    )
    val_dataset = create_rl_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=config.data.get("val_max_samples", -1),
    )
    train_sampler = create_rl_sampler(config.data, train_dataset)

    trainer = SDCRayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    _ACTIVE_SDC_CONTEXT["trainer"] = trainer
    _ACTIVE_SDC_CONTEXT["tokenizer"] = tokenizer
    trainer.init_workers()
    # verl 0.7.1 fit() only calls _compute_reward_colocate when use_rm=True.
    # We keep config.reward.reward_model.enable=False so init_workers does NOT
    # allocate an actual reward-model worker (we compute reward in-process), but
    # we flip use_rm AFTER init so the reward branch routes through our
    # SDCRayPPOTrainer._compute_reward_colocate override. Without this flip,
    # `extract_reward(batch)` raises KeyError for "rm_scores" since nothing
    # populates it.
    trainer.use_rm = True
    trainer.fit()


if __name__ == "__main__":
    main()
