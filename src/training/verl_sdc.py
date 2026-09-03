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

import math
import os
import re
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
    _check_correctness,
    compute_degeneration_penalty,
    correctness_reward,
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
    TRUSTED_META_CLASSES,
    signature_suppression_ids,
)
from src.training._decoy_utils import _rule_based_decoy

# ★검수(0831) countdown 원재료 컬럼의 «출처» 기억. 컬럼당 한 번만 찍기 위한 것.
#   값이 바뀌면(flat↔extra_info) 다시 찍는다 — 경로가 런 도중 갈리면 그게 사건이다.
_COUNTDOWN_COL_PROVENANCE: dict[str, str] = {}
# split_first_meta — the ONE prefix/meta/continuation splitter, shared by the
# live pmi_shift scorer and the always-on epistemic wandb block. (The dense-PMI
# generation this module was built for was removed 2026-08-03.)
from src.training.dcpo_pmi import (
    split_first_meta,
)
# Answer-string + divergent-token machinery — pure numpy. Built for the gm /
# RLSD generations (removed 2026-08-03); these two functions survive because the
# live pmi_shift scorer and src/eval/pmi_shift_signal.py both call them.
from src.training.dcpo_directional import (
    boxed_answer_string,
    divergent_token_mask,
)
# PMI-SHIFT-ACROSS-META (asymmetric sign-reversal) R_meta core — pure numpy
# sibling of dcpo_pmi/dcpo_directional. Used ONLY by the new pmi_shift branch +
# _compute_dcpo_v4_pmi_shift_rmeta below (default-OFF; existing arms unaffected).
from src.training.dcpo_pmi_shift import (
    compute_pmi_shift_reward,
    pmi_shift_reward,
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
        # R_cal repair knob (0812): "brier_neg" (default, legacy verbatim) or
        # "info_gain" (rewarding-doubt log score + meta-scoped conf parse).
        # Owner: rq3v2f_b3p3. See dcpo_region_rewards' cal_mode docstring.
        cal_mode=str(_read("dcpo_cal_mode", "brier_neg") or "brier_neg"),
    )
    _DCPO_HEAD_STASH.update(out)
    return out


# Round 2 M-A: under TRIOBJ_DCPO_V4 the R_meta source must be an EXPLICIT
# decision — the old `read("dcpo_rmeta_source", "cf")` default silently fell
# open onto the deprecated CF-regeneration path (plausible nonzero values, no
# log line) whenever the knob was missing OR the algorithm config was
# unreadable (the reader swallows exceptions into its default).
_V4_RMETA_SOURCES = ("cf", "none", "pmi_shift")
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

    # ACTIVATE resume-invariant anchor-EMA warmup (audit fix A): hand the restored
    # global step to compose_dcpo_region_advantage's warmup gate, which reads
    # _ANCHOR_EMA_STATE["global_step"]. Without this the gate falls back to the
    # process-local _n that resets on every preemption-resume, reopening the
    # anchor_warmup_steps un-normalized window (a spurious meta-arm / RQ2 bias).
    from src.training.verl_sdc_utils import _ANCHOR_EMA_STATE as _AES
    _AES["global_step"] = _step

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
    # dcpo_rmeta_source: 'pmi_shift' (THE live source — two-position teacher-
    # forcing, routed onto META) | 'cf' (EXPLICIT opt-in only — leave the
    # dcpo_region_rewards value, byte-identical to the v3 path) | 'none'
    # (stage 1: hard-zero the head so the logged scalar cannot leak a
    # text-fallback CF signal at w_meta=0). Round 2 M-A: a MISSING/unreadable
    # knob RAISES — the old silent 'cf' default fell open onto the deprecated
    # regeneration path with plausible nonzero values, invisibly.
    # Five further sources (pmi, cf_group, asym_cf, decoy_did_gm,
    # decoy_did_rlsd) were removed 2026-08-03; see
    # archive/reward_lineages_retired_0803/README.md.
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
        if _rmeta_src == "pmi_shift":
            # PMI-SHIFT-ACROSS-META (design 2026-06-25): TWO-position teacher-
            # forcing (gold/decoy at meta-OPEN and meta-CLOSE) → asymmetric sign-
            # reversal R_shift. ADDITIVE head routed onto META (independent head,
            # identical plumbing to pmi/decoy_did_gm), centered over the shift-
            # member population. decoy→gold reversal = +save, gold→decoy = −derail.
            _v4_prompt_texts = [
                _decode_prompt_only(
                    tokenizer,
                    data[i].batch["prompts"],
                    data[i].batch["attention_mask"],
                    prompt_length,
                )
                for i in range(bs)
            ]
            _r_shift, _shift_member, _shift_raw = _compute_dcpo_v4_pmi_shift_rmeta(
                tokenizer=tokenizer,
                trainer=trainer,
                prompt_texts=_v4_prompt_texts,
                response_texts=decoded_responses,
                ground_truths=ground_truths,
                fmt_classes=_fmt_classes,
                heads=_heads,
                read_knob=_v4_read,
                step=_step,
            )
            data.non_tensor_batch["meta_region_utility"] = _r_shift
            data.non_tensor_batch["dcpo_rmeta_member"] = _shift_member
        elif _rmeta_src == "none":
            data.non_tensor_batch["meta_region_utility"] = np.zeros(bs, dtype=np.float32)
            data.non_tensor_batch["dcpo_rmeta_member"] = np.zeros(bs, dtype=np.float32)
        # 'cf' (explicit opt-in): no-op — the dcpo_region_rewards value stands.
        # Invalid values already raised inside _v4_rmeta_source_strict.
        if _rmeta_src in ("none", "pmi_shift"):
            # Observability truth: the rollout table + trend scalars below must
            # chart the R_meta that actually ROUTES, not the stale CF/text-
            # fallback stash value. REASSIGN (not mutate): _DCPO_HEAD_STASH
            # still holds the original list, so the reward-func wrappers (which
            # feed the logging-only summed rm_scores) are untouched.
            _heads = dict(_heads)
            _heads["R_meta"] = [float(x) for x in data.non_tensor_batch["meta_region_utility"]]
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
    _log_dcpo_trend_scalars(step=_step, heads=_heads, cf_texts=_cf_texts,
                            decoded_responses=decoded_responses)


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


# Epistemic verbalization markers (spec 2026-06-24 §Epistemic measurement,
# adapted from SDPO check_epistemic_tokens.py): a direction that RAISES accuracy
# by SUPPRESSING these words (emission↓, epistemic-words↓) is REJECTED. Counted
# over the META block of meta-bearing rows so the rate measures genuine
# reflective verbalization, not generic solution prose.
_EPISTEMIC_WORDS = (
    "wait", "hmm", "perhaps", "maybe", "actually", "alternatively",
    "seems", "might", "likely", "check",
)
_EPISTEMIC_RE = re.compile(
    r"\b(" + "|".join(_EPISTEMIC_WORDS) + r")\b", re.IGNORECASE)


def _log_dcpo_trend_scalars(*, step, heads, cf_texts, decoded_responses=None):
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
        # ── cal/habit scalars (0812 R_cal repair ask). All .get-guarded: older
        # stashes (or reward paths that never fill the arrays) log nothing —
        # byte-identical observability for every pre-existing config.
        _cp = heads.get("conf_parsed", None)
        if _cp is not None and len(_cp) == B:
            scal["dcpo/conf_parse_rate"] = sum(float(v) for v in _cp) / B
            _meta_rows_cp = [i for i in range(B) if hm[i]]
            scal["dcpo/conf_parse_rate_meta_rows"] = (
                sum(float(_cp[i]) for i in _meta_rows_cp) / len(_meta_rows_cp)
                if _meta_rows_cp else 0.0
            )
        _cpos = heads.get("cal_positive", None)
        if _cpos is not None and len(_cpos) == B:
            scal["dcpo/cal_positive_rate"] = sum(float(v) for v in _cpos) / B
        for _key, _name in (("conf_gap", "dcpo/conf_gap_mean"),
                            ("cal_group_gap", "dcpo/cal_group_gap_mean")):
            _arr = heads.get(_key, None)
            if _arr is not None:
                _fin = [float(v) for v in _arr if float(v) == float(v)]
                scal[_name] = (sum(_fin) / len(_fin)) if _fin else float("nan")
        # habit sentinels: catch a meta-first relapse from the TRAINING rollouts
        # themselves (no eval round-trip). C-034 context: the repaired init
        # starts at meta_first 0.0 / think 1.0.
        _mf = heads.get("meta_first", None)
        if _mf is not None and len(_mf) == B:
            scal["dcpo/meta_first_rate"] = sum(float(v) for v in _mf) / B
        _ht = heads.get("has_think", None)
        if _ht is not None and len(_ht) == B:
            scal["dcpo/think_rate"] = sum(float(v) for v in _ht) / B
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
        # EPISTEMIC verbalization trend (spec 2026-06-24 §Epistemic measurement):
        # count epistemic words in the META block of meta-bearing rows + the
        # per-token epistemic-word density, so a direction that raises accuracy by
        # SUPPRESSING reflective language is visible (and REJECTED) on a chart.
        # decoded_responses=None (other callers) -> keys skipped (byte-identical).
        if decoded_responses is not None:
            try:
                _meta_texts = []
                for _i, _t in enumerate(decoded_responses):
                    if _i < len(hm) and hm[_i]:
                        _parts = split_first_meta(_t or "")
                        if _parts is not None:
                            _meta_texts.append(_parts[1])  # the meta block
                _n_meta = max(1, len(_meta_texts))
                _epi_counts = [len(_EPISTEMIC_RE.findall(_m)) for _m in _meta_texts]
                _tot_words = sum(max(1, len(_m.split())) for _m in _meta_texts) or 1
                scal["dcpo/epistemic_words_per_meta"] = sum(_epi_counts) / _n_meta
                scal["dcpo/epistemic_word_density"] = sum(_epi_counts) / _tot_words
                scal["dcpo/epistemic_meta_with_word_rate"] = (
                    sum(1 for c in _epi_counts if c > 0) / _n_meta
                )
                # meta-emission rate echoed under the epistemic namespace (the
                # decision pairs emission selectivity WITH word preservation).
                scal["dcpo/epistemic_meta_emit_rate"] = sum(hm) / B
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


# ── COUNTDOWN_6ARM reward-head wiring (ADDITIVE) ──────────────────────────────
# The countdown arms are GROUP-dependent (p_hat / sign(A_corr) need the rollout
# group) and TEACHER-dependent (the PMI-shift term needs a frozen-ref forward),
# but the reward manager calls each reward_fn per-key with no group structure and
# no trainer handle. Same shape as the DCPO wiring above: ONE mode-gated pre-pass
# (`_compute_countdown_arm_stash`, defined below the PMI helpers it reuses) runs
# per batch and stashes the per-rollout arm totals; the single thin wrapper here
# just reads the stash, so REWARD_CONFIGS['COUNTDOWN_6ARM'] keeps the standard
# funcs/keys contract. Pre-existing modes never touch this.
#
# NOTE ON WEIGHTS: `arm_reward` already multiplies each term by its ARM_SPECS
# weight AND the step warmup, so the head's manager-side weight MUST stay 1.0 —
# a second multiplication here would silently rescale every arm.
_COUNTDOWN_MODE = "COUNTDOWN_6ARM"
_COUNTDOWN_STASH: dict = {"step": None, "arm": None, "total": None,
                          "components": None, "n": 0}


def countdown_arm_reward(completions, **kwargs):
    """The ONE reward head of COUNTDOWN_6ARM — a thin reader of the pre-pass stash.

    There is deliberately NO text fallback and NO zero fill. A missing or stale
    stash means the pre-pass did not run for this batch, i.e. the arm is declared
    but not wired; returning 0.0 there is exactly the failure this mode exists to
    make impossible (eight arms silently trained on the same reward). We raise,
    and the countdown branch of the reward loop re-raises instead of swallowing.
    """
    n = len(completions)
    total = _COUNTDOWN_STASH.get("total")
    if total is None or len(total) != n:
        raise RuntimeError(
            "[COUNTDOWN] arm-reward stash is missing or stale (have "
            f"{'None' if total is None else len(total)} rows, batch has {n}): the "
            "pre-pass did not run for this batch. Refusing to emit a silent 0.0 "
            "reward — that is the 'declared lever, no wiring' failure mode.")
    return [float(v) for v in total]


def _log_countdown_rmeta_wandb(*, step, arm, mag, gvd=None, mq=None, phat_of=None, uid=None):
    """R_meta 크기 스칼라를 wandb 로. 없으면 조용히 no-op (콘솔 전용 런 지원).

    0818 인계 문서: "지표는 wandb 로 안 간다 — stdout 에만 찍히므로 로그를 판다.
    다음 판에 wandb 로깅을 붙이는 게 맞다." 이 함수가 그것이다.
    """
    try:
        import wandb  # noqa: F811
        if wandb.run is None:
            return
        scal = {
            f"cd/{arm}/meta_floor_abs_mean": float(mag["meta_floor_abs_mean"]),
            f"cd/{arm}/meta_abs_mean_total": float(mag["meta_abs_mean_total"]),
        }
        if "meta_share_of_total" in mag:
            scal[f"cd/{arm}/meta_share_of_total"] = float(mag["meta_share_of_total"])
        for k in ("std_total", "std_meta", "std_nonmeta", "std_corr",
                  "meta_var_share", "frac_corr_constant", "frac_tiny_std", "amp_p95"):
            if gvd and k in gvd:
                scal[f"cd/{arm}/gvar/{k}"] = float(gvd[k])
        for k in ("auc", "auc_total", "inversion_rate", "gap", "mean_pos", "mean_neg"):
            if mq and k in mq:
                scal[f"cd/{arm}/metaq/{k}"] = float(mq[k])
        for k, v in mag["terms"].items():
            scal[f"cd/{arm}/{k}/mean"] = float(v["mean"])
            scal[f"cd/{arm}/{k}/abs_mean"] = float(v["abs_mean"])
            scal[f"cd/{arm}/{k}/std"] = float(v["std"])
            scal[f"cd/{arm}/{k}/frac_zero"] = float(v["frac_zero"])
            scal[f"cd/{arm}/{k}/p95_abs"] = float(v["p95_abs"])
        # sign(A_corr)==0 비율 — 곱셈 팔(C·F)의 메타 항이 침묵하는 비율.
        # 이것이 높으면 C·F 의 널은 "곱하기가 안 통한다"가 아니라 "잴 수 없었다"이다.
        if phat_of is not None and uid is not None:
            ps = [phat_of[u] for u in uid if u in phat_of]
            if ps:
                scal[f"cd/{arm}/frac_sign_zero"] = (
                    sum(1.0 for p in ps if p in (0.0, 1.0)) / len(ps))
        wandb.log(scal, step=int(step))
    except Exception as _e:  # pragma: no cover — 관측이 학습을 죽이지 않는다
        print(f"[COUNTDOWN][RMETA] wandb log skipped: "
              f"{type(_e).__name__}: {_e}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# OSD 자동 강등 — «규칙에 손을 단다»
# ══════════════════════════════════════════════════════════════════════════════
# 지난 세대의 실패는 "규칙이 없었다" 가 아니다. `meta_outcome_discrimination` 의
# docstring 에 «AUC ~ 0.5 면 B/C/F 의 널은 신호 없음이다» 라고 **적혀 있었고**, AUC 를
# 매 스텝 재고 있었는데, 그 데이터에 **행동할 권한을 가진 코드가 없어서** 모니터링이
# 회로차단기가 아니라 사후 고고학이 됐다(0825 포스트모템). 이번엔 규칙이 손을 갖는다.
_OSD_AUC_HIST: list = []          # (step, auc)
_OSD_DEMOTED: dict = {"on": False, "step": None, "auc": None}
_OSD_AUC_FLOOR = 0.58             # 관문 통과선 0.60 보다 살짝 아래(학습 중 잡음 여유)
_OSD_AUC_MIN_STEP = 15            # 워밍업(0->20) 도중의 과민 반응을 막는다
_OSD_AUC_WINDOW = 10


def _osd_demote_check(step, auc, arm, *, n_pos=None, n_neg=None, n_unique=None) -> bool:
    """최근 10스텝 평균 AUC 가 바닥 밑이면 True — 호출자가 w_meta 를 0 으로 내린다."""
    if _OSD_DEMOTED["on"]:
        return True
    if auc is None or auc != auc:      # NaN = «못 쟀다» — 기록하지 않는다
        return False
    # ★«측정이 성립했는가» 게이트 (0826 06:10 추가).
    #   NaN 만 걸러서는 부족하다. 점수가 **전부 동률**이면 AUC 는 정의상 0.500 이 되고,
    #   그것은 «판별력이 없다» 가 아니라 «잴 수 없었다» 이다. 그 0.500 을 유효값으로
    #   받으면 회로차단기가 신호 품질과 무관하게 step 15 에서 무조건 자폭하고,
    #   그 순간 이 팔은 대조군과 비트 동일해진다 — 안전장치를 경유한 «배선 0».
    if n_unique is not None and int(n_unique) <= 1:
        print(f"[COUNTDOWN][OSD-AUC][SKIP] arm={arm} step={step} "
              f"점수 고유값 {n_unique}개 — 측정 불성립. 히스토리에 기록하지 않는다.",
              flush=True)
        return False
    if (n_pos is not None and int(n_pos) < 20) or (n_neg is not None and int(n_neg) < 20):
        print(f"[COUNTDOWN][OSD-AUC][SKIP] arm={arm} step={step} "
              f"표본 부족(pos={n_pos} neg={n_neg}) — 기록하지 않는다.", flush=True)
        return False
    _OSD_AUC_HIST.append((int(step), float(auc)))
    if int(step) < _OSD_AUC_MIN_STEP:
        return False
    win = [a for _s, a in _OSD_AUC_HIST[-_OSD_AUC_WINDOW:]]
    if len(win) < _OSD_AUC_WINDOW:
        return False
    roll = sum(win) / len(win)
    if roll < _OSD_AUC_FLOOR:
        _OSD_DEMOTED.update({"on": True, "step": int(step), "auc": roll})
        print(f"[COUNTDOWN][OSD-DEMOTE] arm={arm} step={step} "
              f"rolling{_OSD_AUC_WINDOW}_auc={roll:.3f} < floor={_OSD_AUC_FLOOR} "
              f"-- w_meta 를 0 으로 강등한다. 이후 스텝은 대조군(A)과 동등하므로 "
              f"사후 판정에서 처치로 세지 마라.", flush=True)
        return True
    return False



def _compute_countdown_arm_stash(self, data, decoded_responses, bs, prompt_length, step):
    r"""COUNTDOWN_6ARM 의 **유일한 발전기**. 배치당 한 번 돌며 `_COUNTDOWN_STASH` 를 채운다.

    위의 `countdown_arm_reward` 는 이 스태시를 읽기만 한다. 이 함수가 안 돌면 그쪽이
    RuntimeError 를 던지고, 아래 보상 루프의 countdown 분기가 그것을 **재던진다** —
    "선언된 레버, 배선 0" 실패(여덟 팔이 같은 보상으로 조용히 학습)를 불가능하게 만든다.

    학습·검증 두 배치 모두 `MetaCotSDCRewardManager.__call__` 을 지나므로, 보상 루프
    바로 앞에 두면 검증 배치도 같은 보상을 본다(팔 간 비교가 성립하려면 그래야 한다).
    """
    from src.training import countdown_inv as _cdi
    from src.training import countdown_pmi as _cdp
    from src.training import countdown_rewards as _cdr
    from src.training import countdown_task as _cdt

    arm = str(getattr(getattr(self.config, "algorithm", None),
                      "countdown_arm", "") or "").upper()
    if arm not in _cdr.ARM_SPECS:                      # fail-closed. 조용한 기본값 금지.
        raise ValueError(
            f"[COUNTDOWN] algorithm.countdown_arm={arm!r} 가 ARM_SPECS "
            f"{sorted(_cdr.ARM_SPECS)} 에 없다 — 팔이 배선되지 않았다. 런처가 "
            "++algorithm.countdown_arm 을 넘겼는지 확인하라.")

    nt = data.non_tensor_batch

    def _col(name):
        v = nt.get(name, None)
        _src = "flat"
        if v is None:
            _src = "extra_info"
            # ★2026-08-21: async(agent-loop) 경로는 parquet 의 **평평한** 컬럼을
            #   non_tensor_batch 로 넘기지 않는다. verl 이 항상 나르는 `extra_info`
            #   안에 같은 값이 들어 있으므로(build_records 가 양쪽에 넣는다) 거기서 읽는다.
            ei = nt.get("extra_info", None)
            if ei is not None:
                try:
                    cand = [(e or {}).get(name, None) for e in list(ei)]
                    if not all(x is None for x in cand):
                        v = cand
                except Exception:
                    v = None
        if v is None:
            raise RuntimeError(
                f"[COUNTDOWN] non_tensor_batch 에 '{name}' 컬럼이 없다(extra_info 폴백도 실패). "
                f"사용 가능한 키: {sorted(nt.keys())}. parquet 이 "
                "countdown_task.build_records(include_flat_cols=True) 로 빌드됐는지 "
                "확인하라 — PMI 원재료가 없으면 메타 팔 전부가 무효다.")
        # ★검수(0831) 출처 무음 제거. 실측: verl 0.7.1 의 async 경로는
        #   ray_trainer._get_gen_batch 가 flat 컬럼을 gen_batch 로 pop 하고,
        #   agent_loop._postprocess 는 reward_loop_worker_handles 가 None 일 때만
        #   되돌려 준다. SDC 는 init_workers 시점에 use_rm=False 이므로 핸들이
        #   «리스트»(None 아님)로 생성되고 → flat 은 **항상** 사라진다.
        #   즉 이 폴백은 예외가 아니라 상시 경로다. 어느 쪽을 읽었는지 남기지 않으면
        #   «어느 미끼로 학습했는가»가 사후 확정 불가능해진다(cd6_B_other50 사고).
        _seen = _COUNTDOWN_COL_PROVENANCE
        if _seen.get(name) != _src:
            _seen[name] = _src
            print(f"[COUNTDOWN][COL] {name} <- {_src}", flush=True)
        # flat 과 extra_info 가 둘 다 있으면서 어긋나면 즉사한다. 조용히 한쪽을
        # 고르면 «선언한 데이터»와 «학습한 데이터»가 갈라진다.
        if _src == "flat":
            _ei = nt.get("extra_info", None)
            if _ei is not None:
                try:
                    _alt = [(e or {}).get(name, None) for e in list(_ei)]
                except Exception:
                    _alt = None
                if _alt is not None and not all(x is None for x in _alt):
                    _bad = sum(1 for a, b in zip(_alt, list(v)) if str(a) != str(b))
                    if _bad:
                        raise RuntimeError(
                            f"[COUNTDOWN] '{name}' 이 flat 컬럼과 extra_info 에서 다르다 "
                            f"({_bad}/{len(_alt)} 행). 어느 쪽으로 학습했는지 사후 확정이 "
                            "불가능해지므로 즉사한다. parquet 을 다시 빌드하라 "
                            "(countdown_task.build_records 는 양쪽에 같은 값을 넣는다).")
        return list(v)

    witnesses, decoys = _col("witness"), _col("decoy")
    nums_col, target_col = _col("nums"), _col("target")

    prompt_texts = [
        _decode_prompt_only(self.tokenizer, data[i].batch["prompts"],
                            data[i].batch["attention_mask"], prompt_length)
        for i in range(bs)
    ]

    # ── PMI-shift. 평문 <meta> 토큰 스팬 + 증인식/연산자교체오답의 발산 토큰. ──────
    #    여기서만 GPU 를 쓴다(동결 ref forward). config 위반은 삼키지 않고 즉사한다.
    if "meta_pos_full" in _cdr.ARM_SPECS[arm]["terms"]:
        # ★P 팔: 같은 4n 팔 배치에서 «전체 스팬 평균» PMI 를 읽는 변형 스코어러.
        from src.training import countdown_pmi_full as _cdpf   # noqa: PLC0415
        rows, diag = _cdpf.score_pmi_shift_full(
            tokenizer=self.tokenizer,
            trainer=_ACTIVE_SDC_CONTEXT.get("trainer", None),
            prompt_texts=prompt_texts,
            response_texts=list(decoded_responses),
            witnesses=witnesses, decoys=decoys, step=step)
    else:
        rows, diag = _cdp.score_pmi_shift(
            tokenizer=self.tokenizer,
            trainer=_ACTIVE_SDC_CONTEXT.get("trainer", None),
            prompt_texts=prompt_texts,
            response_texts=list(decoded_responses),
            witnesses=witnesses, decoys=decoys, step=step)

    # ★B3(감사 0821): ref 스코어링이 실패하면 PMI 가 전부 NaN 이 되고, NaN 은
    #   `_pmi_shift_reward` 에서 fail-closed 로 0.0 이 된다 ⇒ 메타 항이 **조용히 사라져**
    #   B≡A · C≡A · F≡E 가 되고 `[COUNTDOWN][WIRED]` 는 정상으로 보인다. 이 모드가
    #   막으려던 바로 그 실패(«선언된 레버, 배선 0»)이므로 여기서 즉사시킨다.
    _pmi_terms = {"meta_pos", "meta_mul", "meta_ctx"} & set(_cdr.ARM_SPECS[arm]["terms"])
    if diag.get("ref_error") and _pmi_terms:
        raise RuntimeError(
            f"[COUNTDOWN] arm={arm} step={step}: PMI ref 스코어링 실패 "
            f"({diag['ref_error']}) — {sorted(_pmi_terms)} 항이 무음 0 이 되어 "
            f"이 팔이 A 팔과 같아진다. 조용히 진행하지 않는다.")

    _pmi_full_terms = {"meta_pos_full"} & set(_cdr.ARM_SPECS[arm]["terms"])
    if diag.get("ref_error") and _pmi_full_terms:
        raise RuntimeError(
            f"[COUNTDOWN] arm={arm} step={step}: PMI-FULL ref 스코어링 실패 "
            f"({diag['ref_error']}) — meta_pos_full 항이 무음 0 이 되어 이 팔이 "
            f"A 팔과 같아진다. 조용히 진행하지 않는다.")

    # ── 채점 · 형식 · 텔레메트리 원재료. gold 불필요 — target 이 프롬프트에 있다. ──
    def _parse_ok(t):
        # ★수리(0823): 원래 `extract_expr(t) is not None` 이었다. 그것은 문자집합·AST
        #   파싱을 건너뛰어 `\\boxed{(3+*)}` `\\boxed{abc}` 같은 **파싱 불가 문자열에
        #   w_format 0.35 를 지급**했다 — `countdown_task.parse_ok` 가 고치려던 결함 ④
        #   가 호출 지점에서 되살아나 있었다. 전원오답 조가 71% 인 구간에서 형식 항은
        #   조 내 분산의 큰 몫이므로 이 누수는 무해하지 않다.
        return int(_cdt.parse_ok(t))

    for i, r in enumerate(rows):
        text = decoded_responses[i]
        r["text"] = text
        r["r_corr"] = int(_cdt.grade(text, nums_col[i], int(target_col[i])))
        r["format_ok"] = _cdr.format_ok_row(text, arm, parse_expr_ok=_parse_ok)
        # ⚠`or ""` 를 지우지 마라. answer_leak 은 None 을 받으면 **예외를 던진다**
        #   (조용한 0 이 누출 중단조건을 무력화하는 것을 막는 의도적 설계다). 그런데
        #   \boxed 가 없는 행 — 절단되거나 답을 못 맺은 행 — 은 정말로 `None` 이 나오고,
        #   그런 행이 배치에 하나만 있어도 텔레메트리 전체가 터진다. 식을 안 썼으면
        #   누출도 없으므로 "" 가 정직한 값이다(빈 식은 어떤 메타에도 안 들어 있다).
        r["final_expr"] = _cdt.extract_expr(text) or ""
        r["arm"] = arm
        if "plan" in _cdr.ARM_SPECS[arm]["terms"]:      # ★0902 P 팔: next 첫수 해 생존 · 이행
            r["plan_ok"], r["plan_followed"] = _cdr.plan_next(text, nums_col[i], int(target_col[i]))

    # ── OSD (Outcome-Signed Surprisal Drop) — «메타 제거 문맥» Δcert. ─────────────
    #   PMI 경로와 **병렬**이다. 위의 score_pmi_shift 는 한 글자도 바뀌지 않았고,
    #   여기서는 별도 배치(행당 2팔) + 별도 ref 호출을 쓴다 — PMI 의 `base=4*k` 부기를
    #   건드리지 않기 위해서다(같은 배치에 섞으면 그 스트라이드가 조용히 깨진다).
    #   final_expr 이 필요하므로 채점 루프 **뒤**에 둔다(누출 가드 ②가 그걸 본다).
    #
    #   켜짐 조건 두 가지:
    #     · 팔이 meta_osd 항을 쓰면 **무조건** 켜지고, 실패는 fail-loud 다(PMI 와 같은
    #       이유 — 무음 0 은 그 팔을 A 팔로 만든다).
    #     · 아니면 `COUNTDOWN_OSD` 환경변수(기본 "1")로 켠다. 이때는 **측정 모드**다:
    #       기존 팔 A~H 의 보상은 delta_cert 를 읽지 않으므로 한 글자도 바뀌지 않고,
    #       [COUNTDOWN][OSD] 의 p90 만 쌓인다(정규화 상수 c 를 그 수로 정한다).
    #   ⚠비용: 발화 행마다 forward 2팔이 는다. 실측 수는 아래 로그의 fwd_* 에 찍는다.
    # ★항 이름을 여기서 문자열로 쓰지 않는다. 0825 적대검증에서 이 줄이 "meta_osd" 를
    #   보는데 실제 항은 "osd" 라, 이 가드가 **모든 팔에서 항상 빈 집합**이었고 OSD 팔이
    #   A 팔과 비트 동일한 보상을 냈다. 정의처는 countdown_rewards.OSD_TERM 하나다.
    _osd_terms = {_cdr.OSD_TERM} & set(_cdr.ARM_SPECS[arm].get("terms", ()))
    _osd_on = bool(_osd_terms) or os.environ.get("COUNTDOWN_OSD", "1") == "1"
    osd_diag: dict = {"enabled": bool(_osd_on)}
    if not _osd_on:
        for r in rows:
            r.update(_osd_empty_row("off"))
    else:
        try:
            osd_rows, _od = _compute_countdown_osd(
                tokenizer=self.tokenizer,
                trainer=_ACTIVE_SDC_CONTEXT.get("trainer", None),
                prompt_texts=prompt_texts,
                response_texts=list(decoded_responses),
                final_exprs=[r["final_expr"] for r in rows], step=step)
            osd_diag.update(_od)
            for i, r in enumerate(rows):
                r.update(osd_rows[i])
        except Exception as _oexc:
            # 항을 쓰는 팔이면 즉사한다 — 조용한 0 은 이 모드가 막으려는 실패다.
            if _osd_terms:
                raise
            # 아니면 학습을 죽이지 않는다. 단 **조용히 0 으로 채우지 않는다**:
            # delta_cert=None(«못 쟀다») 으로 두고 크게 남긴다.
            osd_diag["error"] = f"{type(_oexc).__name__}: {_oexc}"
            for r in rows:
                r.update(_osd_empty_row("error"))
            print(f"[COUNTDOWN][OSD][FAIL] step={step} arm={arm} "
                  f"{osd_diag['error']}", flush=True)
            # traceback 은 **첫 실패에만** — 150 스텝 내내 같은 스택을 쏟으면 로그가
            # 못 읽히고, 그러면 정작 다른 실패가 묻힌다. 위 [FAIL] 한 줄은 매 스텝 남는다.
            if _OSD_FAIL_SEEN["n"] == 0:
                traceback.print_exc()
            _OSD_FAIL_SEEN["n"] += 1
        # ★PMI 의 fail-loud 와 같은 규약: ref 가 실패했는데 팔이 그 항을 쓰면 즉사.
        if osd_diag.get("ref_error") and _osd_terms:
            raise RuntimeError(
                f"[COUNTDOWN] arm={arm} step={step}: OSD ref 스코어링 실패 "
                f"({osd_diag['ref_error']}) — meta_osd 항이 무음 0 이 되어 이 팔이 "
                f"A 팔과 같아진다. 조용히 진행하지 않는다.")
        # ── 텔레메트리 한 줄. p90 은 정규화 상수 c 를 정하는 값이다. ──────────────
        #   ⚠예외로 죽은 스텝에는 찍지 않는다 — 그 줄의 0 들은 「쟀는데 0」으로 읽히는데
        #     사실은 「못 쟀다」다. 그 스텝은 위 [OSD][FAIL] 한 줄이 대신한다.
        if "error" not in osd_diag:
            print(f"[COUNTDOWN][OSD] step={step} arm={arm} B={osd_diag.get('B', bs)} "
                  f"n_emitted={osd_diag.get('n_emitted', 0)} "
                  f"n_scored={osd_diag.get('scored', 0)} "
                  f"n_leak_blocked={osd_diag.get('leak_blocked', 0)} "
                  f"n_nan={osd_diag.get('nan_rows', 0)} "
                  f"no_boxed={osd_diag.get('no_boxed', 0)} "
                  f"meta_first={osd_diag.get('meta_first', 0)} "
                  f"dcert_mean={osd_diag.get('d_mean', float('nan')):+.4f} "
                  f"dcert_std={osd_diag.get('d_std', float('nan')):.4f} "
                  f"dcert_p90={osd_diag.get('d_p90', float('nan')):+.4f} "
                  f"dcert_abs_p90={osd_diag.get('d_abs_p90', float('nan')):.4f} "
                  f"pos_frac={osd_diag.get('d_pos_frac', float('nan')):.3f} "
                  f"w_len_mean={(osd_diag.get('w_len_sum', 0) / max(1, osd_diag.get('attempted', 0))):.1f} "
                  f"fwd_calls={osd_diag.get('fwd_calls', 0)} "
                  f"fwd_rows={osd_diag.get('fwd_rows', 0)}(+pad{osd_diag.get('fwd_rows_pad', 0)}) "
                  f"fwd_tokens={osd_diag.get('fwd_tokens', 0)} "
                  f"leak={osd_diag.get('leak_reasons', {})} "
                  f"terms_on={bool(_osd_terms)}", flush=True)

    # ── INV (도치 자) — «정답 힌트를 준 문맥 vs 안 준 문맥» 의 메타 프로즈 logp. ─────
    #   PMI·OSD 와 **병렬**이다. 셋 다 `_build_pmi_score_batches` + `_dcpo_v4_ref_logprobs`
    #   를 재사용하지만 **각자 별도 배치·별도 ref 호출**이다 — PMI 의 `base=4*k`, OSD 의
    #   `base=2*k`, INV 의 `base=2*k` 부기는 섞는 순간 조용히 깨진다(verl_sdc.py:2082 경고).
    #   final_expr 이 필요하므로(G5 누출 가드) 채점 루프 **뒤**에 둔다.
    #
    #   ★정답(witness)이 «채점용 문맥» 에만 들어가고 롤아웃에는 절대 안 들어간다는 격리:
    #     · 롤아웃 프롬프트는 parquet 의 `prompt` 컬럼이고 그것은
    #       `countdown_task.build_prompt` = [system, "Numbers: … / Target: …"] 뿐이다.
    #       witness 는 `extra_info`/평평한 컬럼에만 있다(`countdown_task.build_records`).
    #     · 여기서 만드는 힌트 프롬프트는 `countdown_inv.inv_hint_prompt` 가 돌려주는
    #       **새 문자열**이고, 그것이 가는 곳은 `_build_pmi_score_batches` 가 새로 만든
    #       텐서와 `DataProto.from_dict` 로 감싼 **새 DataProto** 뿐이다.
    #       `data.batch["prompts"]` 에는 한 글자도 안 쓴다(이 함수는 rows dict 만 채운다).
    #     · 그 forward 는 **동결 ref** 위의 teacher-forced 채점이고 생성이 아니다.
    #       정책이 witness 에 대해 배우는 것은 **행당 스칼라 1개**(보상)뿐이며, 그것은
    #       `r_corr`(정답 채점)이 이미 쓰는 통로와 같다.
    #   ⚠비용: 발화 행마다 forward 2팔. 실측 수는 아래 fwd_* 에 찍는다(OSD 와 같은 규모).
    _inv_terms = {_cdr.INV_TERM} & set(_cdr.ARM_SPECS[arm].get("terms", ()))
    _inv_on = bool(_inv_terms) or os.environ.get("COUNTDOWN_INV", "0") == "1"
    inv_diag: dict = {"enabled": bool(_inv_on)}
    if not _inv_on:
        for r in rows:
            r.update(_cdi.inv_empty_row("off"))
    else:
        try:
            inv_rows, _id = _cdi.score_inv(
                tokenizer=self.tokenizer,
                trainer=_ACTIVE_SDC_CONTEXT.get("trainer", None),
                prompt_texts=prompt_texts,
                response_texts=list(decoded_responses),
                witnesses=witnesses,
                targets=target_col,
                final_exprs=[r["final_expr"] for r in rows], step=step)
            inv_diag.update(_id)
            for i, r in enumerate(rows):
                r.update(inv_rows[i])
        except Exception as _iexc:
            if _inv_terms:
                raise                       # 항을 쓰는 팔이면 즉사(무음 0 = A 팔 위장)
            inv_diag["error"] = f"{type(_iexc).__name__}: {_iexc}"
            for r in rows:
                r.update(_cdi.inv_empty_row("error"))
            print(f"[COUNTDOWN][INV][FAIL] step={step} arm={arm} "
                  f"{inv_diag['error']}", flush=True)
            if _INV_FAIL_SEEN["n"] == 0:
                traceback.print_exc()
            _INV_FAIL_SEEN["n"] += 1
        if inv_diag.get("ref_error") and _inv_terms:
            raise RuntimeError(
                f"[COUNTDOWN] arm={arm} step={step}: INV ref 스코어링 실패 "
                f"({inv_diag['ref_error']}) — {_cdr.INV_TERM} 항이 무음 0 이 되어 이 팔이 "
                f"A 팔과 같아진다. 조용히 진행하지 않는다.")
        # ── 텔레메트리 한 줄. p50 이 τ 를, pen_p90 이 c 를 정하는 값이다. ───────────
        if "error" not in inv_diag:
            _att = max(1, inv_diag.get("attempted", 0))
            print(f"[COUNTDOWN][INV] step={step} arm={arm} B={inv_diag.get('B', bs)} "
                  f"ruler={_cdi.inv_signature()} "
                  f"n_emitted={inv_diag.get('n_emitted', 0)} "
                  f"n_scored={inv_diag.get('scored', 0)} "
                  f"n_leak_blocked={inv_diag.get('leak_blocked', 0)} "
                  f"n_false_claim={inv_diag.get('false_claim', 0)} "
                  f"n_nan={inv_diag.get('nan_rows', 0)} "
                  f"short_prose={inv_diag.get('short_prose', 0)} "
                  f"no_witness={inv_diag.get('no_witness', 0)} "
                  f"anchor_err={inv_diag.get('anchor_error', 0)} "
                  f"inv_mean={inv_diag.get('i_mean', float('nan')):+.4f} "
                  f"inv_std={inv_diag.get('i_std', float('nan')):.4f} "
                  f"inv_p25={inv_diag.get('i_p25', float('nan')):+.4f} "
                  f"inv_p50={inv_diag.get('i_p50', float('nan')):+.4f} "
                  f"inv_p75={inv_diag.get('i_p75', float('nan')):+.4f} "
                  f"pen_rate={inv_diag.get('i_pen_rate', float('nan')):.3f} "
                  f"pen_p90={inv_diag.get('i_pen_p90', float('nan')):.4f} "
                  f"prose_tok_mean={(inv_diag.get('prose_tok_sum', 0) / _att):.1f} "
                  f"fwd_calls={inv_diag.get('fwd_calls', 0)} "
                  f"fwd_rows={inv_diag.get('fwd_rows', 0)}(+pad{inv_diag.get('fwd_rows_pad', 0)}) "
                  f"fwd_tokens={inv_diag.get('fwd_tokens', 0)} "
                  f"leak={inv_diag.get('leak_reasons', {})} "
                  f"terms_on={bool(_inv_terms)}", flush=True)

    # ── 그룹 단위 두 수: p̂(자가검증률) 와 sign(A_corr). uid 없으면 계산 불가. ─────
    uid = nt.get("uid", None)
    if uid is None:
        raise RuntimeError(
            "[COUNTDOWN] non_tensor_batch 에 uid 가 없다 — p̂ 와 sign(A_corr) 는 "
            "롤아웃 그룹 단위라 계산할 수 없다. 곱하기·게이팅 팔이 전부 무효가 된다.")
    uid = [str(u) for u in uid]
    groups: dict = {}
    for i, u in enumerate(uid):
        groups.setdefault(u, []).append(i)
    phat_of, mean_of = {}, {}
    for u, ix in groups.items():
        phat_of[u] = _cdr.compute_phat([rows[i] for i in ix])
        mean_of[u] = sum(float(rows[i]["r_corr"]) for i in ix) / len(ix)
    for i, r in enumerate(rows):
        r["adv_corr"] = float(r["r_corr"]) - mean_of[uid[i]]
        r["phat"] = phat_of[uid[i]]
        r["group_id"] = uid[i]

    # ★강등이 걸려 있으면 osd 항을 0 으로 죽인다(이전 스텝의 판정이 이번 스텝부터 적용된다).
    #   지난 판정을 «다음 스텝부터» 적용하는 것이 옳다 — 이미 뽑은 롤아웃의 보상을
    #   소급해 바꾸면 그 스텝의 어드밴티지가 정책과 어긋난다.
    _osd_off = bool(_COUNTDOWN_STASH.get("osd_demoted"))
    totals, comps = [], []
    for i, r in enumerate(rows):
        _t, _c = _cdr.arm_reward(arm, r, step=step, phat=phat_of[uid[i]])
        if _osd_off and _cdr.OSD_TERM in _c:
            _t = float(_t) - float(_c[_cdr.OSD_TERM])
            _c = dict(_c); _c[_cdr.OSD_TERM] = 0.0
        totals.append(float(_t))
        comps.append(_c)

    _COUNTDOWN_STASH.update({"step": step, "arm": arm, "total": totals,
                             "components": comps, "n": len(totals), "rows": rows})

    # ★팔의 **정체 서명**을 런당 한 번 찍는다(검수 0831).
    #   `countdown_rewards.arm_signature` 의 docstring 은 "런처·로그·분석이 전부 이
    #   문자열을 찍으면 어떤 팔이 실제로 무엇을 켜고 돌았는지가 사후에 한 줄로 확인된다"
    #   고 선언하는데, **어디서도 찍지 않고 있었다**(로그 전수 grep 결과 0건).
    #   R 팔은 τ·c 가 잠정이면 서명에 '?' 가 박히므로, 이 줄이 없으면 «잠정값으로 돌았다»
    #   는 사실이 로그 어디에도 남지 않는다 — 서명을 둔 목적 그 자체가 사라진다.
    if not _ARM_SIG_SEEN.get(arm):
        _ARM_SIG_SEEN[arm] = True
        print(f"[COUNTDOWN][SIG] arm={arm} {_cdr.arm_signature(arm)}", flush=True)

    # ★배선의 유일한 증거. 런처 가드가 이 줄을 grep 해 없으면 창을 죽인다.
    #   반드시 계산 **뒤**에 찍는다 — import 시점에 찍으면 가드는 통과하고 아무것도
    #   증명하지 못한다(그게 이 모드가 존재하는 이유인 바로 그 실패다).
    print(f"[COUNTDOWN][WIRED] arm={arm} step={step} n={len(totals)} "
          f"mean={sum(totals) / max(1, len(totals)):.4f} "
          f"distinct={len({round(v, 6) for v in totals})} "
          f"phat_groups={len(groups)} "
          f"pmi_scored={diag.get('scored', 0)}/{diag.get('B', 0)} "
          f"osd_scored={osd_diag.get('scored', 0)}/{osd_diag.get('B', 0)} "
          f"inv_scored={inv_diag.get('scored', 0)}/{inv_diag.get('B', 0)}", flush=True)

    # ★0902 관측: 보상 구성 요소별 평균 · 발화율 · 계획 항(해 생존/이행) 비율 · 응답 표본 8개 → wandb (실패해도 학습은 계속)
    try:
        import wandb as _wb
        if _wb.run is not None:
            _keys = sorted({k for c in comps for k in c})
            _log = {f"cd/comp_{k}": sum(float(c.get(k, 0.0)) for c in comps) / max(1, len(comps)) for k in _keys}
            _log["cd/emit_rate"] = sum(int(r.get("emitted", 0)) for r in rows) / max(1, len(rows))
            if "plan" in _cdr.ARM_SPECS[arm]["terms"]:
                _em = [r for r in rows if int(r.get("emitted", 0))]
                _log["cd/plan_ok_rate"] = sum(int(r.get("plan_ok", 0)) for r in _em) / max(1, len(_em))
                _log["cd/plan_followed_rate"] = sum(int(r.get("plan_followed", 0)) for r in _em) / max(1, len(_em))
                _log["cd/plan_hit_rate"] = sum(int(r.get("plan_ok", 0)) and int(r.get("plan_followed", 0)) for r in _em) / max(1, len(_em))
            _log["cd/corr_rate"] = sum(int(r.get("r_corr", 0)) for r in rows) / max(1, len(rows))
            _log["cd/len_mean"] = sum(len(str(r.get("text", ""))) for r in rows) / max(1, len(rows))
            _wb.log(_log, step=step)
            print(f"[COUNTDOWN][CD] step={step} " + " ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in sorted(_log.items())), flush=True)
            if step % 5 == 0:   # 표본: 정답 4 · 오답 4 (메타 있는 행 우선)
                _pick = sorted(range(len(rows)), key=lambda i: (-int(rows[i].get("emitted", 0)), i))
                _c1 = [i for i in _pick if int(rows[i].get("r_corr", 0))][:4]; _c0 = [i for i in _pick if not int(rows[i].get("r_corr", 0))][:4]
                # 열: 메타 «앞» 응답 / 메타 블록 / 메타 «뒤» 응답 (사용자 요청 0902)
                _tbl = _wb.Table(columns=["step", "corr", "total", "emitted", "plan_ok", "followed", "before_meta", "meta", "after_meta"])
                for i in _c1 + _c0:
                    _txt = str(rows[i].get("text", "")); _pm = _cdr.parse_meta(_txt, "new")
                    if _pm.get("emitted") and _pm.get("start") is not None:
                        _b, _m, _a = _txt[:_pm["start"]], _txt[_pm["start"]:_pm["end"]], _txt[_pm["end"]:]
                    else:
                        _b, _m, _a = _txt, "", ""
                    _tbl.add_data(step, int(rows[i].get("r_corr", 0)), round(totals[i], 3), int(rows[i].get("emitted", 0)),
                                  int(rows[i].get("plan_ok", -1)), int(rows[i].get("plan_followed", -1)), _b[-1500:], _m[:800], _a[:1500])
                _wb.log({f"cd/samples": _tbl}, step=step)
    except Exception as _we:
        print(f"[COUNTDOWN][WANDB] 로깅 실패(무시): {_we}", flush=True)

    # ── R_meta 크기 계기 — **매 스텝**. C-012 가 만든 계기다. ────────────────────
    #   왜 텔레메트리(10스텝)와 따로 매 스텝인가: 붕괴는 스텝 사이에서 일어난다.
    #   base b3p 는 gs115 에 0.98 이었다가 gs135 에 0.744 였다 — 10스텝 격자로는
    #   내려가는 중간을 못 본다. 그리고 이 세 숫자(mean / abs_mean / std)는 계산이
    #   싸다(성분 dict 평균 몇 번).
    try:
        _mag = _cdr.rmeta_magnitude(comps, totals=totals,
                                   warmup=_cdr.warmup_scale(step))
        _parts = " ".join(
            f"{_k}[mean={_v['mean']:+.4f} abs={_v['abs_mean']:.4f} "
            f"std={_v['std']:.4f} zero={_v['frac_zero']:.2f}]"
            for _k, _v in _mag["terms"].items())
        print(f"[COUNTDOWN][RMETA] arm={arm} step={step} warmup={_mag['warmup']:.2f} {_parts} "
              f"floor_abs={_mag['meta_floor_abs_mean']:.4f} "
              f"floor_vs_meta={_mag['floor_vs_meta']:.3f} "
              f"meta_share={_mag.get('meta_share_of_total', float('nan')):.3f} "
              f"verdict={_mag['verdict']}", flush=True)
        # 그룹 내 분산 분해 — /σ 정규화가 켜져 있으면 이쪽이 실제 영향력이다.
        _gvd = _cdr.group_variance_decomposition(
            [[comps[i] for i in ix] for ix in groups.values()])
        print(f"[COUNTDOWN][GVAR] arm={arm} step={step} "
              f"std_total={_gvd['std_total']:.4f} std_meta={_gvd['std_meta']:.4f} "
              f"std_corr={_gvd['std_corr']:.4f} "
              f"meta_var_share={_gvd['meta_var_share']:.3f} "
              f"corr_const={_gvd['frac_corr_constant']:.2f} "
              f"tiny_std={_gvd['frac_tiny_std']:.2f} amp_p95={_gvd['amp_p95']:.1f} "
              f"verdict={_gvd['verdict']}", flush=True)

        # ★«좋은 메타인지인가» — R_meta 가 정답/오답을 가르는지. 수학 세대에서 이
        #   판별력이 가짜 대조군보다 낮았다(AUC 0.457 vs 0.598, 0817 §1.5).
        _mq = _cdr.meta_outcome_discrimination(rows, comps, group_ids=uid)
        # ★자동 강등 — OSD 팔에서만. unsigned Δcert 의 판별력이 바닥 밑이면 w_meta:=0.
        #   (osd 항은 y=r_corr 를 곱해 AUC 가 항진명제이므로, 판정은 **unsigned** 로 한다.)
        if _cdr.OSD_TERM in set(_cdr.ARM_SPECS[arm].get("terms", ())):
            # ★«발화한 행 전부» 를 센다 — 관문(scripts/osd_gate.py:228-247)의 정의다.
            #   관문은 형식위반·빈W·누출 행을 **Δ=0.0 으로 포함**한다. 두 이유:
            #     ① 그 행들은 실제로 R_osd=0 을 받는다. 감시는 «정책이 겪는 것» 을 재야 한다.
            #     ② 관문이 통과선을 정의했으므로, 감시가 다른 집단을 재면 두 수를 비교할 수 없다.
            #   ⚠0826 06:45~08:45 에 «채점 성공만» 으로 좁혔던 것은 오류다(내가 넣었다).
            #     같은 관문 스크립트로 학습 데이터를 재면 0.733 인데 학습 중 감시는 0.496 이었다.
            #     발화 2287행 중 358행(15.7%)을 뺀 것이 그 차이의 유력한 원인이다.
            #   미발화 행은 제외한다 — 관문의 `rows` 도 발화 행만이다.
            _keep = [i for i, r in enumerate(rows)
                     if int(r.get("emitted", 0)) == 1]
            _unsigned = [{"r_corr": rows[i].get("r_corr", 0)} for i in _keep]
            # ★키는 반드시 META_TERMS 에 있는 이름이어야 한다. 집계기가
            #   `sum(c.get(k,0) for k in META_TERMS)` 로만 읽기 때문이다.
            #   0826 검증에서 "osd_unsigned" 를 쓰다가 **완벽 판별 신호에도 AUC 0.500**
            #   이 나왔다 — 어제의 meta_osd/osd 키 불일치와 같은 버그가 그걸 막으라고
            #   만든 안전장치 안에서 재발했다.
            # 채점 못한 행(누출·답없음·형식위반)은 0.0 — 관문과 같은 규약.
            _uc = [{_cdr.OSD_TERM: float(rows[i].get("delta_cert") or 0.0)}
                   for i in _keep]
            _mq_u = _cdr.meta_outcome_discrimination(
                _unsigned, _uc, group_ids=[uid[i] for i in _keep])
            _uniq = len({round(float(c.get(_cdr.OSD_TERM, 0.0)), 9) for c in _uc})
            _demoted = _osd_demote_check(
                step, _mq_u.get("auc"), arm,
                n_pos=_mq_u.get("n_pos"), n_neg=_mq_u.get("n_neg"), n_unique=_uniq)
            _COUNTDOWN_STASH["osd_demoted"] = bool(_demoted)
            print(f"[COUNTDOWN][OSD-AUC] arm={arm} step={step} "
                  f"unsigned_auc={_mq_u.get('auc', float('nan')):.3f} "
                  f"n_pos={_mq_u.get('n_pos')} n_neg={_mq_u.get('n_neg')} "
                  f"n_kept={len(_keep)}/{len(rows)} demoted={_demoted}", flush=True)
        print(f"[COUNTDOWN][METAQ] arm={arm} step={step} scope={_mq['scope']} "
              f"auc={_mq['auc']:.3f} auc_total={_mq['auc_total']:.3f} "
              f"inversion={_mq['inversion_rate']:.1%} gap={_mq['gap']:+.4f} "
              f"mean_pos={_mq['mean_pos']:+.4f} mean_neg={_mq['mean_neg']:+.4f} "
              f"n_pos={_mq['n_pos']} n_neg={_mq['n_neg']} "
              f"verdict={_mq['verdict']}", flush=True)
        _log_countdown_rmeta_wandb(step=step, arm=arm, mag=_mag, gvd=_gvd, mq=_mq,
                                   phat_of=phat_of, uid=uid)
    except Exception as _mexc:      # 계기 실패로 학습을 죽이지 않는다
        print(f"[COUNTDOWN][RMETA] step={step} 실패: "
              f"{type(_mexc).__name__}: {_mexc}", flush=True)

    # ── 텔레메트리 + 중단 조건. ★수리(0823): 10스텝 격자 → **매 스텝**. ──────────
    #   왜: 붕괴는 스텝 사이에서 일어난다(이 파일 RMETA 주석이 이미 그렇게 적었다).
    #   그리고 2스텝 검증 런에서는 step%10==0 이 한 번도 참이 아니어서 **발화율을
    #   학습 중에 본 적이 한 번도 없었다** — 그 맹점이 이 실험을 두 번 헛돌게 했다.
    try:
        _rep = _cdr.telemetry_report(
            [[rows[i] for i in ix] for ix in groups.values()], components=comps)
        # ★수리(0823) 구조율 — 연구 의도("틀려도 점검해서 정답까지 간다")의 직접 지표.
        #   기존 보상 항 8종 중 이것을 재는 것이 하나도 없었다.
        _resc = _countdown_rescue_stats(rows, nums_col, target_col)
        _rep.update(_resc)
        _rep["arith_in_meta"] = _cdr.arithmetic_in_meta_rate(rows)
        print(f"[COUNTDOWN][TELEMETRY] step={step} arm={arm} {_rep}", flush=True)
        print(f"[COUNTDOWN][RESCUE] step={step} arm={arm} "
              f"rescue={_resc['rescue_rate']:.3f} pre_had={_resc['pre_had_rate']:.3f} "
              f"never={_resc['never_rate']:.3f} attempts={_resc['n_attempts_mean']:.1f} "
              f"emit={_rep.get('emit_rate', float('nan')):.3f} "
              f"arith={_rep['arith_in_meta']:.3f}", flush=True)
        _all = _cdr.check_abort(_rep)
        # ★두 상태를 절대 섞지 않는다.
        #   abort  = 지표를 쟀고 **선을 넘었다** → 연속 위반이면 학습을 죽인다.
        #   missing= 지표가 **안 찍혔다** → 죽이지 않는다. 계기 하나가 빠졌다고 정상
        #            학습을 죽이면 그게 더 큰 손실이다. 대신 [BLIND] 로 크게 남겨
        #            «못 보고 있다»가 조용히 지나가지 않게 한다(WIRED=0 사고의 교훈).
        # ★수리(0823) 실제 응답 표본 로깅 — 지표만 보면 «무엇이 오르는지»는 알아도
        #   «무엇을 쓰고 있는지»는 모른다. 판박이·편법은 숫자보다 원문에서 먼저 보인다.
        try:
            _ex = []
            for _i, _row in enumerate(rows):
                _m = _cdr.parse_meta(_row.get("text") or "", "new")
                if _m["emitted"]:
                    _ex.append((_i, _m, _row))
                if len(_ex) >= 2:
                    break
            for _i, _m, _row in _ex:
                _sh = (_row.get("pmi_close", float("nan")) - _row.get("pmi_open", float("nan")))
                print(f"[COUNTDOWN][SAMPLE] step={step} arm={arm} i={_i} "
                      f"corr={_row.get('r_corr')} conf={_m['confidence']} dec={_m['decision']} "
                      f"shift={_sh:+.3f} pos={(_m['start'] or 0)}/{len(_row.get('text') or '')} "
                      f"| {' '.join((_m['body'] or '').split())[:170]}", flush=True)
        except Exception as _sexc:
            print(f"[COUNTDOWN][SAMPLE] step={step} 실패: {_sexc}", flush=True)

        _hits = [h for h in _all if h.get("status") == "abort"]
        # ★0902: 메타 항이 없는 팔(N0 맨 GRPO)은 발화율 중단 규칙을 적용하지 않는다 — 발화 0 이 정상이다.
        if not any(t.startswith("meta") or t in ("gate", "osd", "plan") for t in _cdr.ARM_SPECS[arm]["terms"]):
            _hits = [h for h in _hits if h.get("metric") != "emit_rate"]
        for _hit in _hits:
            print(f"[COUNTDOWN][ABORT] step={step} {_hit}", flush=True)
        for _miss in [h for h in _all if h.get("status") != "abort"]:
            print(f"[COUNTDOWN][BLIND] step={step} {_miss['metric']} 미측정 — "
                  f"통과가 아니라 «못 봤다»다.", flush=True)
        # ★수리(0823): 중단 조건이 **출력만 하고 멈추지 않았다**(사전등록 §7 은 이것을
        #   '중단 조건'이라 부른다). 연속 위반이면 실제로 학습을 죽인다.
        _key = f"{arm}"
        if _hits:
            _ABORT_STREAK[_key] = _ABORT_STREAK.get(_key, 0) + 1
        else:
            _ABORT_STREAK[_key] = 0
        if _ABORT_STREAK.get(_key, 0) >= _ABORT_PATIENCE:
            raise _CountdownAbort(
                f"[COUNTDOWN][ABORT] arm={arm} step={step}: 중단 조건이 "
                f"{_ABORT_STREAK[_key]} 스텝 연속 위반 — {_hits}. 사전등록 §7 에 따라 정지한다.")
    except _CountdownAbort:
        raise                                         # ★중단은 삼키지 않는다
    except Exception as _texc:                        # 계기 실패로 학습을 죽이지는 않는다
        print(f"[COUNTDOWN][TELEMETRY] step={step} 실패: "
              f"{type(_texc).__name__}: {_texc}", flush=True)
    return totals


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
    # ── RLSD ablation modes (arxiv 2604.03128) ───────────────────────────────
    # R0: vanilla GRPO baseline — no SDC teacher signal at all. Used to isolate
    # the contribution of meta-region teacher guidance over plain RLVR.
    "VANILLA_GRPO": {
        "funcs": [correctness_reward],
        "weights": [1.0],
        "keys": ["correctness"],
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
    # PLAN-REVIEW CONVERGED"). Entries below are NEW and ADDITIVE:
    # no entry above this comment is touched; existing §8/RLSD modes keep their
    # exact funcs/weights/keys (regression-asserted).
    #
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
    # deleted), the populator overwrites `meta_region_utility` with the
    # PMI-SHIFT head when algorithm.dcpo_rmeta_source == 'pmi_shift' (two-
    # position teacher-forcing at meta-OPEN and meta-CLOSE; decoy->gold
    # reversal = +save, gold->decoy = -derail, frozen ref worker at T=1.0).
    # Stage 1 (format-only) sets dcpo_rmeta_source: none + dcpo_w_meta: 0.
    "TRIOBJ_DCPO_V4": {
        "funcs": [correctness_region_reward, meta_region_utility_reward, cal_region_reward,
                  meta_emission_reward, format_penalty_reward],
        "weights": [1.0, 0.5, 0.3, 0.0, 0.1],
        "keys": ["correctness", "meta_region_utility", "cal_region_reward", "meta_emission",
                 "format_penalty"],
    },
    # COUNTDOWN_6ARM (ADDITIVE, sequence-level GRPO, NO region routing): the
    # Countdown 6-arm family (A corr / B cur / C mul / E gate / F full / G neg).
    # EXACTLY ONE head. Every arm difference lives in
    # src/training/countdown_rewards.ARM_SPECS, selected by
    # algorithm.countdown_arm; the head below is a thin reader of the pre-pass
    # stash. Weight is 1.0 BY CONTRACT (arm_reward already applied the ARM_SPECS
    # weights and the 0->20 step warmup — see countdown_arm_reward's docstring).
    # This mode does NOT join _REGION_ROUTED_MODES or _VANILLA_MODES: with
    # algorithm.adv_estimator=grpo and sdc_enabled=false, patched_compute_advantage
    # falls straight through to verl's own GRPO advantage (no teacher forward, no
    # region routing, dcpo_region.py untouched). The mode-dispatch guard in
    # main_task fails the launch closed if either of those two is not true.
    "COUNTDOWN_6ARM": {
        "funcs": [countdown_arm_reward],
        "weights": [1.0],
        "keys": ["countdown_arm"],
    },
}

# Modes that do NOT compute teacher forward (env reward only).
_VANILLA_MODES = {"VANILLA_GRPO", "BCI_RLVR", "TRIOBJ_META_V1"}
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
# RLSD_FAITHFUL_META: R20 direction B. T+ only (gold-blind teacher → MAGNITUDE
#   only). NO meta_penalty head (see REWARD_CONFIGS). Restores the RLSD
#   sign/magnitude invariant on meta tokens that ROD_* break via clip+penalty.
_SINGLE_TEACHER_MODES = {"ROD_PT", "RLSD_FAITHFUL_META"}
# Modes that compute T+ AND T− forward (contrastive RLSD).
# ROD_MQ_CONTRAST: R18a + T+ − T− contrast term mixed via α/β (R18b)
_CONTRASTIVE_MODES = {
    "SDC_SHARED",
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
# ★배선 수정 2026-08-21 (A100 이식): `mode` 는 main_task 에서만 세팅되는 **모듈 변수**라
#   Ray 워커(별도 프로세스)에는 전달되지 않는다. 같은 함정이 R16 에 이미 기록돼 있다
#   ("Ray RewardLoopWorker actors do NOT inherit module-level _ACTIVE_SDC_CONTEXT").
#   COUNTDOWN 분기(`elif _mode_pf == _COUNTDOWN_MODE`)가 그 미수정 경로에 걸려
#   **여섯 팔 전부가 countdown 보상을 한 번도 못 받고** 150스텝을 돌 뻔했다
#   (실측: 5스텝 진행, [COUNTDOWN][WIRED] 0건).
#   ⇒ 환경변수로 초기화한다. 환경변수는 Ray 워커가 상속하므로 모든 프로세스가 같은 값을 본다.
_ACTIVE_SDC_CONTEXT = {
    "trainer": None,
    "tokenizer": None,
    "mode": os.environ.get("SDC_MODE_ENV", "") or "SDC_SHARED",
}
# One-shot guard so the pairwise_ctb "0 usable uid groups" warning prints once
# per process instead of every degenerate microbatch (codex review pt.4).
_CTB_INACTIVE_WARNED = {"done": False}

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


def reward_loop_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Fallback scalar score for veRL agent-loop reward workers.

    Mode-aware: emits the GDPO reward keys appropriate to the active mode.
      • VANILLA_GRPO                 → correctness only
      • RLSD_META_CONTRAST           → correctness + meta_penalty
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
    # GDPO weight (configured in YAML) is 0 for modes that do not list the
    # key, so this is a safe no-op everywhere else.
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


def _build_teacher_logprob_batch(
    *,
    tokenizer,
    prompt_texts: list[str],
    answer_texts: list[str],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    teacher_role: str = "content",
    contrast_side: str = "pos",
):
    prompt_ids_list = []
    seq_lens = []
    for prompt_text, answer_text in zip(prompt_texts, answer_texts):
        # Align teacher conditioning with what the actor actually sees:
        # prompt_text is already the chat-templated prompt (ending in the
        # assistant role marker), so we append the gold/decoy answer directly
        # instead of injecting a synthetic " Answer: " separator the actor
        # never produces. This keeps teacher log-prob on the same conditional
        # distribution the policy is optimizing against.
        #
        # E.4 self-distill contrast variants: `decoy` is the UNCHANGED
        # f-string below (answer slot = gold for T+, decoy for T-, filled by
        # the caller). It is taken whenever `sdc_contrast_variant` is unset
        # or "decoy" → byte-identical for every pre-existing mode/config/test.
        # `stance`/`conf` append a side-specific suffix to a gold-on-BOTH-
        # sides answer marker (answer cancels in T+−T−).
        _cv = _ACTIVE_SDC_CONTEXT.get("sdc_contrast_variant", "decoy")
        if _cv == "decoy":
            teacher_prompt = f"{prompt_text}{answer_text}"
        elif _cv == "stance":
            # e2_contrastive_steering CONTRASTS["gold_stance"] join pattern.
            _sfx = (" " + CAUTIOUS_INSTR) if contrast_side == "pos" else (" " + CONFIDENT_INSTR)
            teacher_prompt = f"{prompt_text} (answer is {answer_text}){_sfx}"
        elif _cv == "conf":
            _sfx = "\nconfidence: 0.15\n" if contrast_side == "pos" else "\nconfidence: 0.95\n"
            teacher_prompt = f"{prompt_text} (answer is {answer_text}){_sfx}"
        elif _cv == "conf_free":
            # E.8 GOLD-FREE conf-down: confidence suffix ONLY, NO answer injected (T+/T-
            # differ only in the confidence level, both gold-free). Combined with
            # mode=GFN_OPSD_CONTRAST this makes the teacher a DISTRIBUTION-MATCHING (listwise
            # KL) target pulling the policy's meta toward the low-conf-conditioned self —
            # genuinely != E.4 (RLSD_META_CONTRAST + conf = gold-conditioned MAGNITUDE
            # reshaping). Gold-free avoids the leakage that kills distribution-matching for
            # gold-conditioned teachers (Self-Distilled RLVR, arXiv 2604.03128).
            _sfx = "\nconfidence: 0.15\n" if contrast_side == "pos" else "\nconfidence: 0.95\n"
            teacher_prompt = f"{prompt_text}{_sfx}"
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


# ── TRIOBJ_DCPO_V4 ref-logprob batching, shared by the pmi_shift scorer ──────
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


# ══════════════════════════════════════════════════════════════════════════════
# OSD — Outcome-Signed Surprisal Drop. «메타 제거 문맥» teacher-forced logp.
# ══════════════════════════════════════════════════════════════════════════════
# 왜 여기에 있나 (추측 아님 — 읽고 확인한 것)
#   OSD 는 PMI-shift 를 **대체하지 않고 병렬로 얹는다**. PMI 경로(countdown_pmi.
#   score_pmi_shift)는 한 글자도 바뀌지 않는다. 재사용하는 것은 딱 두 개,
#   `_build_pmi_score_batches`(좌측패딩 정렬 수리 포함)와 `_dcpo_v4_ref_logprobs`
#   (T=1.0 + use_legacy_worker_impl fail-closed assert)다 — 둘 다 무수정.
#
#   ⚠PMI 배치와 **섞지 않는다**. `countdown_pmi.read_pmi_from_ref_logprobs` 는
#     `base = 4*k` 고정 스트라이드를 가정한다. OSD 는 행당 2팔이므로 같은 배치에
#     넣으면 그 부기가 조용히 깨진다. 그래서 별도 배치 + 별도 ref 호출이다.
#
# 정의 (사양 그대로)
#   t0 = span.open_start   (`ids[:t0]` 에 `<meta>` 가 한 글자도 없다)
#   t1+1 = span.close_end  (`ids[:close_end]` 는 `</meta>` 를 반드시 포함한다)
#   W = ids[close_end : close_end + L],  L = min(200, 마지막 \boxed 답 끝까지)
#   Δcert = mean logP(W | prompt ⊕ ids[:close_end])     # 메타 포함 문맥
#         − mean logP(W | prompt ⊕ ids[:open_start])    # 메타 구간만 제거한 문맥
#   두 문맥은 **메타 유무만** 다르고 W 는 양쪽에서 **같은 토큰열**이다. 샘플링 없음.
#
# 왜 with_meta 쪽을 재사용하지 못하나 (비용 감사 — verl 0.7.1 루프를 읽고 확인)
#   `ray_trainer.py:1343/1382` 에서 보상이 계산되고, `old_log_prob` 은 :1404,
#   `ref_log_prob`(KL 용) 은 그 뒤다. 즉 **보상 시점에는 응답 전체의 ref logp 가
#   아직 존재하지 않는다**. 게다가 PMI 4팔이 재는 것은 `\boxed{gold}`/`\boxed{decoy}`
#   합성 문자열이지 W 가 아니다. ⇒ 두 팔 다 새 forward 다. 추가 forward 는
#   **행당 2팔, 배치당 ref 호출 1회**이고, 그 수를 `[COUNTDOWN][OSD]` 에 찍는다.

_OSD_FAIL_SEEN = {"n": 0}      # fail-soft 경로에서 traceback 을 한 번만 찍기 위한 카운터
_INV_FAIL_SEEN = {"n": 0}      # 같은 이유(도치 자 스코어러)
_ARM_SIG_SEEN: dict = {}       # 팔 정체 서명을 런당 한 번만 찍기 위한 부기(검수 0831)
# ★자체 선언 금지 — arm_signature 가 서명에 새기는 값과 실제로 도는 값이 갈리면
#   서명이 거짓말한다. 정의처는 countdown_rewards 하나다(0825 적대검증).
#   `_cdr` 은 이 파일에서 함수 안에서만 import 되므로 여기서 한 번 더 가져온다.
from src.training.countdown_rewards import (  # noqa: E402
    OSD_W_MAX as OSD_W_MAX_TOK,
    OSD_LEAK_NGRAM,
)



def _osd_boxed_end_char(text: str):
    r"""마지막 `\boxed{…}` 의 **닫는 중괄호 다음** 문자 인덱스(배타). 없으면 None.

    `countdown_task._last_boxed` 와 **같은 균형괄호 스캔**이다(정규식판이 중첩 괄호에서
    실패한 전례가 그 함수를 그 형태로 만들었다). 그 함수를 부르지 않는 이유는 하나뿐이다
    — 그쪽은 **내용 문자열만** 돌려주고 우리는 **인덱스**가 필요하다. 채점(`grade`)이
    보는 것도 *마지막* boxed 이므로 여기서도 마지막을 쓴다(둘이 갈리면 안 된다).
    """
    t = text or ""
    end, i = None, 0
    while True:
        j = t.find("\\boxed{", i)
        if j < 0:
            break
        k, dep = j + 7, 1
        while k < len(t) and dep:
            dep += (t[k] == "{") - (t[k] == "}")
            k += 1
        if dep == 0:
            end = k          # 닫는 '}' 다음 (배타)
        i = j + 7
    return end


def _osd_encode_with_offsets(tokenizer, text: str, span):
    r"""`span` 이 서 있는 것과 **같은 토큰화**의 (ids, offsets).

    `MetaSpan` 은 offsets 를 보관하지 않는다(`countdown_pmi.MetaSpan`). W 의 끝을
    `\boxed` 문자 오프셋으로 잘라야 하므로 offsets 가 필요하고, 그래서 한 번 더
    인코딩한다(CPU 비용만). **다른 토큰화가 나오면 예외** — 두 토큰화를 섞어 문맥을
    만들면 Δcert 가 조용히 어긋난다(`find_meta_token_span` 의 `response_ids` 검사와
    같은 규약).
    """
    enc = tokenizer(text or "", add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    if ids != list(span.ids):
        raise ValueError(
            f"_osd_encode_with_offsets: 재인코딩이 MetaSpan 의 토큰화와 다르다 "
            f"({len(ids)} vs {len(span.ids)} 토큰).")
    return ids, [tuple(o) for o in enc["offset_mapping"]]


def _osd_window(span, offsets, boxed_end_char, max_len: int = OSD_W_MAX_TOK):
    r"""W 의 토큰 반열린 구간 `(w0, w1)`. 못 만들면 `(None, 사유)`.

    w0 = `span.close_end` — 메타 바로 뒤 토큰(사양의 t1+1).
    w1 = 마지막 `\boxed{…}` **끝을 넘지 않는** 마지막 토큰의 다음, 단 w0+max_len 이하.
         boxed 끝에 걸친 토큰은 **넣는다**(답을 잘라내지 않는다 — `find_meta_token_span`
         의 CLOSE 정책과 같은 방향).
    `\boxed` 가 없으면 잴 답이 없으므로 윈도를 만들지 않는다(사양이 L 을 boxed 끝으로
    정의한다). 그런 행은 r_corr=0 인 절단/미완 행이고, R_osd 는 0 이 된다.
    """
    n = len(span.ids)
    w0 = int(span.close_end)
    if w0 >= n:
        return None, "meta_at_end"          # 메타 뒤에 토큰이 없다 → |W|==0
    if boxed_end_char is None:
        return None, "no_boxed"
    w1 = n
    for idx, (a, b) in enumerate(offsets):
        if b <= a:                          # 길이 0 오프셋(특수토큰)은 건너뛴다
            continue
        if b >= boxed_end_char:
            w1 = idx + 1                    # 끝에 걸친 토큰까지 포함
            break
    w1 = min(w1, w0 + int(max_len), n)
    if w1 <= w0:
        return None, "boxed_before_meta"    # 답이 메타보다 앞에 있다 → |W|==0
    return (w0, w1), ""


def _osd_leak_guard(span, w_ids, meta_raw: str, final_expr: str,
                    *, ngram: int = OSD_LEAK_NGRAM) -> str:
    r"""누출 가드. 막으면 사유 문자열, 깨끗하면 "".

    막는 것 둘 (사양):
      ① 메타 본문과 W 가 **ngram 토큰 이상** 연속으로 겹친다 → 메타에 미래 토큰을 미리
         써 두고 Δcert 를 부풀리는 해킹.
      ② 메타가 최종 `\boxed` 식 문자열을 담고 있다 → `countdown_rewards.answer_leak`
         **그대로** 재사용한다(이 저장소가 정확히 이 용도로 이미 가진 함수다. 정규화
         규칙을 복제하면 두 곳이 갈린다).

    ⚠이 함수가 누출 가드의 **유일한 정의처**다. 토큰 id 기준 8-그램이 사양이다.
    ⚠(0825: countdown_rewards 에 있던 공백-단어 기준 판본은 2~3배 헐거워 제거했다.) 원문:
      바꿔라. 지금 그 이름은 저장소에 **없다**(`grep -r osd src/` → 0건) — 없는 함수의
      시그니처를 추측해 부르지 않는다. ①은 토큰 id 가 필요한데 `countdown_rewards` 는
      의존성 0 의 순수 텔레메트리 모듈이라 토크나이저를 모른다는 점도 함께 본다.
    """
    from src.training import countdown_rewards as _cdr      # noqa: PLC0415

    n = int(ngram)
    meta_ids = tuple(span.ids[span.inner_start:span.inner_end])
    w = tuple(w_ids)
    if n > 0 and len(meta_ids) >= n and len(w) >= n:
        grams = {meta_ids[i:i + n] for i in range(len(meta_ids) - n + 1)}
        for i in range(len(w) - n + 1):
            if w[i:i + n] in grams:
                return f"ngram{n}"
    if final_expr:
        # answer_leak 은 final_expr=None 에 **예외**를 던진다(조용한 0 금지 설계).
        # 빈 식은 어떤 메타에도 안 들어 있으므로 여기서 걸러 넘긴다.
        if _cdr.answer_leak(meta_raw or "", final_expr):
            return "answer_expr"
    return ""


# ★«못 쟀다» 와 «쟀는데 값이 0 으로 정의된다» 는 다른 사건이다 (0826 06:10 수리).
#   섞으면 둘 중 하나가 반드시 틀린다. 실제로 틀렸다 — 아래 W-정의불가 사유들이
#   delta_cert=None 을 받았고, r_osd 의 fail-loud 가 그걸 «배선 끊김» 으로 읽어
#   step 1 에서 학습을 죽였다. 이 사유는 512행 중 3~30행으로 **매 스텝 발생**한다.
_OSD_UNDEFINED_W = frozenset({
    "no_boxed",            # \boxed 가 없다 → 잴 답이 없다
    "meta_at_end",         # 메타 뒤에 토큰이 없다 → |W|=0
    "boxed_before_meta",   # 답이 메타보다 앞 → |W|=0
})
# 아래는 «스코어러가 안 돌았다» — 진짜 배선 사고이므로 None 을 유지해 fail-loud 한다.
_OSD_NOT_MEASURED = frozenset({"off", "ref_error", "span_error", "pending"})


def _osd_empty_row(status: str = "off") -> dict:
    """행 규약의 OSD 기본값.

    ★status 에 따라 delta_cert 가 갈린다:
      · W 가 **구조적으로 정의 불가**(no_boxed / meta_at_end / boxed_before_meta)
        → **0.0**. 이건 정상 행이다. 누출 차단 행(:2240)과 같은 규약이고,
          `_osd_window` 의 docstring 이 이미 «그런 행은 R_osd 는 0 이 된다» 고 선언한다.
      · **스코어러가 안 돌았다**(off / ref_error / span_error / pending)
        → **None**. 이건 배선 사고이므로 r_osd 가 즉사시켜야 한다.
    """
    d = 0.0 if status in _OSD_UNDEFINED_W else None
    return {"delta_cert": d, "osd_scored": 0, "osd_leak": 0,
            "osd_w_len": 0, "osd_status": status, "osd_meta_first": False}


class _OsdAttempt:
    __slots__ = ("row", "w_len")

    def __init__(self, row: int, w_len: int):
        self.row, self.w_len = int(row), int(w_len)


def _build_osd_arms(tokenizer, prompt_texts, response_texts, final_exprs):
    r"""점수를 매길 **2n 개** (문맥, W) 팔. GPU 를 잡지 않는다.

    행 하나당 팔 둘, **고정 순서**:  `W@close` (메타 포함), `W@open` (메타 제거).
    두 팔의 응답 토큰열은 **바이트 동일**하다 — 그것이 OSD 의 전제다.

    Returns (arm_prompts, arm_resps, attempts, per_row, diag).
      per_row[i]  그 행의 OSD 필드 초안(윈도·누출 판정까지 반영, Δcert 는 아직 없음)
    """
    from src.training import countdown_pmi as _cdp          # noqa: PLC0415

    B = len(response_texts)
    if not (len(prompt_texts) == len(final_exprs) == B):
        raise ValueError(
            f"_build_osd_arms: 길이 불일치 prompt={len(prompt_texts)} "
            f"resp={B} final_expr={len(final_exprs)}")

    arm_prompts, arm_resps, attempts = [], [], []
    per_row = [_osd_empty_row("no_meta") for _ in range(B)]
    diag = {"B": B, "no_meta": 0, "no_boxed": 0, "meta_at_end": 0,
            "boxed_before_meta": 0, "leak_blocked": 0, "span_error": 0,
            "meta_first": 0, "n_emitted": 0, "attempted": 0, "w_len_sum": 0,
            "fwd_tokens": 0, "leak_reasons": {}}

    for i in range(B):
        text = response_texts[i] or ""
        try:
            span = _cdp.find_meta_token_span(tokenizer, text)
        except TypeError:
            raise                                   # fast 토크나이저 없음 = 즉사(설계)
        except Exception as e:
            per_row[i] = _osd_empty_row(f"span_error:{type(e).__name__}")
            diag["span_error"] += 1
            continue
        if span is None:
            diag["no_meta"] += 1
            continue                                # per_row 는 이미 no_meta
        diag["n_emitted"] += 1

        try:
            _ids, offsets = _osd_encode_with_offsets(tokenizer, text, span)
        except Exception as e:
            per_row[i] = _osd_empty_row(f"span_error:{type(e).__name__}")
            diag["span_error"] += 1
            continue

        win, why = _osd_window(span, offsets, _osd_boxed_end_char(text))
        if win is None:
            per_row[i] = _osd_empty_row(why)
            diag[why] = diag.get(why, 0) + 1
            continue
        w0, w1 = win
        w_ids = list(span.ids[w0:w1])

        # 누출 행은 Δcert := 0 이 사양이므로 **forward 를 아예 아끼고** 0 을 채운다
        #   (R_osd = y*clip(0/c) = 0 으로 결과가 같다). 조용히 넘어가지 않게 개수와
        #   사유를 진단에 남긴다.
        meta_raw = text[span.char_open:span.char_close_end]
        reason = _osd_leak_guard(span, w_ids, meta_raw, final_exprs[i] or "")
        if reason:
            per_row[i] = {"delta_cert": 0.0, "osd_scored": 0, "osd_leak": 1,
                          "osd_w_len": len(w_ids), "osd_status": f"leak:{reason}",
                          "osd_meta_first": bool(span.meta_first)}
            diag["leak_blocked"] += 1
            diag["leak_reasons"][reason] = diag["leak_reasons"].get(reason, 0) + 1
            continue

        p_ids = list(tokenizer(prompt_texts[i] or "",
                               add_special_tokens=False)["input_ids"])
        ctx_close = p_ids + list(span.ids[:span.close_end])     # 메타 포함
        ctx_open = p_ids + list(span.ids[:span.open_start])     # 메타 구간만 제거
        arm_prompts.append(ctx_close); arm_resps.append(w_ids)
        arm_prompts.append(ctx_open);  arm_resps.append(list(w_ids))
        attempts.append(_OsdAttempt(row=i, w_len=len(w_ids)))
        # meta_first 행은 ctx_open 이 **프롬프트만** 이다 — Δcert 는 여전히 정의되지만
        # 「메타 앞 믿음」이 프롬프트만의 믿음이라 판정에서 갈라 봐야 한다
        # (`countdown_pmi.MetaSpan.meta_first` 주석과 같은 이유).
        per_row[i] = {"delta_cert": None, "osd_scored": 0, "osd_leak": 0,
                      "osd_w_len": len(w_ids), "osd_status": "pending",
                      "osd_meta_first": bool(span.meta_first)}
        diag["meta_first"] += int(bool(span.meta_first))
        diag["w_len_sum"] += len(w_ids)
        diag["fwd_tokens"] += len(ctx_close) + len(ctx_open) + 2 * len(w_ids)

    diag["attempted"] = len(attempts)
    return arm_prompts, arm_resps, attempts, per_row, diag


def _read_osd_from_ref_logprobs(ref_lp, attempts):
    r"""ref 토큰별 logp → 행별 Δcert.

    `base = 2*k`, 팔 순서 `W@close, W@open`. 각 팔의 `[:len(W)]` 를 **평균**한다
    (사양이 |W| 로 나눈다 — 메타 길이는 공식에 등장하지 않는다).
    유한하지 않으면 그 행만 NaN 으로 fail-closed 한다 — `countdown_pmi.
    read_pmi_from_ref_logprobs` 와 **같은 규약**이다. 행을 배치에서 빼지 않는 이유:
    GRPO 그룹에서 행을 진짜 빼면 그룹 크기가 달라져 센터링이 깨진다. 대신 그 행의
    항만 0 이 되고(`countdown_rewards._f`), 배치 전체가 실패한 경우에만 즉사한다.
    """
    from src.training.countdown_pmi import _row_sum          # noqa: PLC0415

    out = []
    for k, at in enumerate(attempts):
        base = 2 * k
        L = at.w_len
        try:
            close = _row_sum(ref_lp, base + 0, L, slice(0, L)) / L
            open_ = _row_sum(ref_lp, base + 1, L, slice(0, L)) / L
        except Exception:
            out.append(float("nan"))
            continue
        d = close - open_
        out.append(float(d) if math.isfinite(d) else float("nan"))
    return out


def _compute_countdown_osd(*, tokenizer, trainer, prompt_texts, response_texts,
                           final_exprs, step: int = 0, _ref_scorer=None):
    r"""행별 Δcert(`delta_cert`) + 진단. **여기서만 GPU 를 쓴다** (ref forward 1회).

    Returns (per_row, diag).
      per_row[i] = {"delta_cert", "osd_scored", "osd_leak", "osd_w_len", "osd_status"}
                   delta_cert 는 float | 0.0(누출) | NaN(비유한) | None(못 쟀다).
      diag       사유별 개수 + Δcert 요약통계 + **추가 forward 비용**.

    `_ref_scorer` 는 테스트 주입구다(주면 verl 을 안 부른다) — `score_pmi_shift` 와 같은 규약.
    """
    from src.training import countdown_pmi as _cdp          # noqa: PLC0415

    arm_prompts, arm_resps, attempts, per_row, diag = _build_osd_arms(
        tokenizer, prompt_texts, response_texts, final_exprs)
    diag["scored"] = 0
    diag["nan_rows"] = 0
    diag["ref_error"] = None
    diag["fwd_calls"] = 0
    diag["fwd_rows"] = 0
    diag["fwd_rows_pad"] = 0

    if not attempts:
        for r in per_row:
            if r["osd_status"] == "pending":
                r["osd_status"] = "unscored"
        diag.update(_osd_delta_stats([]))
        return per_row, diag

    if _ref_scorer is None:
        _cdp.assert_pmi_config(trainer)             # config 위반은 진입 즉시 깨진다
        tensors, real_n = _build_pmi_score_batches(
            arm_prompts, arm_resps, _cdp._pad_unit(trainer))
        if real_n != 2 * len(attempts):
            raise AssertionError(
                f"OSD 팔 부기가 깨졌다: {real_n} != 2*{len(attempts)}")
        diag["fwd_calls"] = 1
        diag["fwd_rows"] = real_n
        diag["fwd_rows_pad"] = int(tensors["input_ids"].shape[0]) - real_n
        try:
            ref_lp = _dcpo_v4_ref_logprobs(trainer, tensors)
        except AssertionError:
            raise                                   # config 위반은 절대 삼키지 않는다
        except Exception as e:
            diag["ref_error"] = f"{type(e).__name__}: {e}"
            for at in attempts:
                per_row[at.row]["osd_status"] = "ref_error"
            print(f"[COUNTDOWN][OSD][FAIL] step={step}: ref 스코어링 실패 "
                  f"({diag['ref_error']}) — 이 배치의 delta_cert 는 전부 None.",
                  flush=True)
            diag.update(_osd_delta_stats([]))
            return per_row, diag
    else:
        diag["fwd_calls"] = 1
        diag["fwd_rows"] = 2 * len(attempts)
        ref_lp = _ref_scorer(arm_prompts, arm_resps)

    deltas = _read_osd_from_ref_logprobs(ref_lp, attempts)
    good = []
    for at, d in zip(attempts, deltas):
        r = per_row[at.row]
        if math.isfinite(d):
            r["delta_cert"] = float(d)
            r["osd_scored"] = 1
            r["osd_status"] = "ok"
            good.append(float(d))
        else:
            r["delta_cert"] = float("nan")          # 조용한 0 이 아니다
            r["osd_scored"] = 0
            r["osd_status"] = "nan"
            diag["nan_rows"] += 1
    diag["scored"] = len(good)
    diag.update(_osd_delta_stats(good))
    return per_row, diag


def _osd_delta_stats(vals) -> dict:
    r"""Δcert 요약. **p90 은 정규화 상수 c 를 정하는 값이다**(사양: |Δcert| 의 90 퍼센타일).

    분위수는 `countdown_rewards._quantile`(선형보간, 의존성 0)을 쓴다 — 저장소에 이미
    있는 분위수를 복제하지 않는다.
    """
    from src.training.countdown_rewards import _quantile     # noqa: PLC0415

    xs = [float(v) for v in vals if math.isfinite(float(v))]
    n = len(xs)
    if not n:
        nan = float("nan")
        return {"d_mean": nan, "d_std": nan, "d_p90": nan, "d_abs_p90": nan,
                "d_abs_mean": nan, "d_pos_frac": nan, "n_delta": 0}
    m = sum(xs) / n
    var = sum((v - m) ** 2 for v in xs) / n
    return {
        "d_mean": m,
        "d_std": math.sqrt(var),
        "d_p90": _quantile(xs, 0.90),
        "d_abs_p90": _quantile([abs(v) for v in xs], 0.90),   # ★= 정규화 상수 c 후보
        "d_abs_mean": sum(abs(v) for v in xs) / n,
        "d_pos_frac": sum(1.0 for v in xs if v > 0) / n,
        "n_delta": n,
    }


def _pmi_position_scalar(logp_gold, logp_decoy, divergent_mask) -> float:
    r"""gold-minus-decoy summed logp over the DIVERGENT answer tokens at ONE
    teacher-forcing context (PMI_open or PMI_close).

    logp_gold/logp_decoy are per-token ref logprobs over the gold/decoy answer
    spans. The DiD is positionally over the GOLD span (where divergent_mask lives).

    LENGTH-MISMATCH (review 2026-06-25): the gold and decoy answer strings are both
    tokenizations of a `\boxed{...}` answer and SHOULD share length on the divergent
    span. The previous code zero-PADDED a shorter decoy span, which is WRONG: a
    missing decoy logprob is treated as logp=0 (P=1.0), inflating the gold-vs-decoy
    contrast by orders of magnitude (e.g. a 4-vs-2 mismatch turned 1.0 into 8.0) and
    corrupting the PMI-shift signal. We now FAIL CLOSED (return NaN) on any length
    mismatch so the caller drops the row instead of training on a tokenization
    accident. Only the masked (truly divergent) positions are summed. Empty/NaN ->
    NaN. This is the SUM (not mean_min) — a position-level belief log-odds that
    PMI_close − PMI_open then differences (matching the gm DiD locus)."""
    g = np.asarray(logp_gold, dtype=np.float64).reshape(-1)
    d = np.asarray(logp_decoy, dtype=np.float64).reshape(-1)
    n = g.size
    if n == 0:
        return float("nan")
    if d.size != n:
        # Misaligned gold/decoy spans: do NOT pad with 0 (that fabricates logp=0
        # i.e. P=1 for the missing decoy tokens). Fail closed.
        return float("nan")
    mask = (np.ones(n, dtype=bool) if divergent_mask is None
            else np.asarray(divergent_mask, dtype=bool).reshape(-1))
    if mask.size != n:
        return float("nan")
    diff = (g - d)[mask]
    if diff.size == 0 or not np.isfinite(diff).all():
        return float("nan")
    return float(diff.sum())


def _meta_body_token_jaccard(meta_inner: str, body_prefix: str) -> float:
    """Token-set Jaccard similarity between the meta-inner text and the body prefix.

    A near-1.0 value means the meta block merely RESTATES the body reasoning (no
    novel content) — the content-integrity guard then skips the row so a derivative
    meta cannot earn shift credit via mere presence (presence-as-confidence confound).
    Whitespace-tokenized, case-folded; empty meta -> 0.0 (caught upstream anyway)."""
    mt = set((meta_inner or "").lower().split())
    bt = set((body_prefix or "").lower().split())
    if not mt:
        return 0.0
    inter = len(mt & bt)
    union = len(mt | bt)
    return float(inter) / float(union) if union else 0.0


def _compute_dcpo_v4_pmi_shift_rmeta(
    *,
    tokenizer,
    trainer,
    prompt_texts: list,
    response_texts: list,
    ground_truths: list,
    fmt_classes: list,
    heads: dict,
    read_knob,
    step: int = 0,
):
    r"""TRIOBJ_DCPO_V4 PMI-SHIFT-ACROSS-META R_meta (design 2026-06-25).

    TWO-position teacher-forcing extension of the gm head. For each trusted
    meta-bearing row, score the GOLD and DECOY `\boxed{...}` answer strings at TWO
    contexts of the model's OWN rollout:
        OPEN  = body BEFORE <|meta|>          (belief before the meta block)
        CLOSE = body + <|meta|>...<|/meta|>   (belief after the meta block)
    and compute, over the divergent answer tokens,
        PMI_open  = Σ (logp(gold|OPEN)  − logp(decoy|OPEN))
        PMI_close = Σ (logp(gold|CLOSE) − logp(decoy|CLOSE))
    R_shift = pmi_shift_reward(PMI_open, PMI_close) — asymmetric sign-reversal
    (decoy→gold = +save, gold→decoy = −derail, derail>=save), default knobs.

    4 arms per row (gold@open, decoy@open, gold@close, decoy@close) batched into ONE
    frozen-ref forward, reusing _build_pmi_score_batches + _dcpo_v4_ref_logprobs.
    Reuses _rule_based_decoy for D. CRASH-SAFE: a ref failure prints LOUDLY and
    returns all-zero R_meta + member.

    Returns (r_meta float32[B], member float32[B], shift_raw float32[B]).
    """
    B = len(response_texts)
    r_meta = np.zeros(B, dtype=np.float32)
    member = np.zeros(B, dtype=np.float32)
    shift_raw = np.full(B, np.nan, dtype=np.float32)

    scale = float(read_knob("dcpo_pmishift_scale", 1.0))
    save_big = float(read_knob("dcpo_pmishift_reversal_save", 1.0))
    derail_big = float(read_knob("dcpo_pmishift_reversal_derail", 2.0))
    clip = float(read_knob("dcpo_pmishift_clip", 2.0))
    decoy_seed = int(read_knob("dcpo_pmishift_decoy_seed", 42))
    reversal_eps = float(read_knob("dcpo_pmishift_reversal_min_magnitude", 0.0))
    # meta-vs-body content-integrity: a meta that near-exactly duplicates the body
    # reasoning earns NO shift credit (presence-as-confidence confound guard). The
    # threshold is a token-Jaccard similarity over the meta-inner vs body-prefix
    # token sets; >= dup_thresh means derivative -> skip. dup_thresh>=1.0 disables.
    dup_thresh = float(read_knob("dcpo_pmishift_meta_body_dup_thresh", 1.0))

    # Config guard (review M1): the frozen-ref scorer requires the ENGINE worker
    # path (use_legacy_worker_impl=disable) so meta_info['temperature']=1.0 is NOT
    # overwritten by rollout.temperature (~1.67x silent PMI compression). We check
    # HERE before any work so a READABLE-but-WRONG config crashes loudly at the
    # pmi_shift entry rather than mid-batch. An UNREADABLE config (e.g. unit-test
    # trainer=object() with the ref monkeypatched away) is left to the hard
    # fail-closed assert inside _dcpo_v4_ref_logprobs, which fires on the real
    # scoring path — keeping the entry guard from breaking mock-ref tests.
    try:
        _legacy = str(trainer.config.trainer.use_legacy_worker_impl)
    except Exception:
        _legacy = None  # unreadable here -> deferred to the ref-scorer's hard assert
    if _legacy is not None:
        assert _legacy == "disable", (
            f"DCPO-V4 pmi_shift requires use_legacy_worker_impl=disable to preserve "
            f"T=1.0 ref scoring; got {_legacy!r} (the legacy fsdp worker overwrites "
            f"meta_info['temperature'] with rollout.temperature, compressing PMI ~1.67x)")

    # 1) Select trusted meta-bearing rows + build the 4 (context, answer) arms.
    attempted: list = []
    arm_prompts, arm_resps = [], []
    skip_empty_meta = 0       # meta block has no content between the tags
    skip_dup_meta = 0         # meta is a near-duplicate of body reasoning
    skip_decoy = 0            # gold==decoy or empty/mismatched decoy tokenization
    for i in range(B):
        if fmt_classes[i] not in TRUSTED_META_CLASSES:
            continue
        parts = split_first_meta(response_texts[i])
        if parts is None:
            continue
        gold = (ground_truths[i] or "").strip()
        if not gold:
            continue
        response_prefix, meta_text, _continuation = parts
        # meta_text is the TAG-INCLUSIVE block (<|meta|>...<|/meta|>). Require actual
        # content BETWEEN the tags: an empty/whitespace-only meta makes CLOSE==OPEN
        # and PMI_close==PMI_open by construction (a spurious null), and lets meta
        # PRESENCE (not content) earn credit. Skip such rows.
        meta_inner = meta_text[len(META_START):len(meta_text) - len(META_END)] \
            if meta_text.startswith(META_START) and meta_text.endswith(META_END) \
            else ""
        if not meta_inner.strip():
            skip_empty_meta += 1
            continue
        # Content-integrity (presence-as-confidence confound guard): if the meta is a
        # near-exact duplicate of the body reasoning it adds no novel reasoning — skip
        # so it cannot earn shift credit through mere presence. Token-set Jaccard.
        if dup_thresh < 1.0 and _meta_body_token_jaccard(
                meta_inner, response_prefix) >= dup_thresh:
            skip_dup_meta += 1
            continue
        # OPEN context = body strictly BEFORE <|meta|>; CLOSE = body through close.
        ctx_open_text = (prompt_texts[i] or "") + response_prefix
        ctx_close_text = ctx_open_text + meta_text
        try:
            decoy = _rule_based_decoy(gold, seed=decoy_seed, checker=_check_correctness)
        except Exception:
            decoy = _rule_based_decoy(gold, seed=decoy_seed)
        gold_str = boxed_answer_string(gold)
        decoy_str = boxed_answer_string(decoy)
        gold_ids = list(tokenizer.encode(gold_str, add_special_tokens=False))
        decoy_ids = list(tokenizer.encode(decoy_str, add_special_tokens=False))
        if not gold_ids:
            continue
        # Decoy integrity: an empty decoy tokenization would be treated as logp=0
        # (P=1) downstream; a gold==decoy decoy produces ZERO signal by construction;
        # a length-mismatched decoy would (post-fix) fail the row closed in
        # _pmi_position_scalar. Skip all three here and count them as a diagnostic.
        if (not decoy_ids) or (decoy_str == gold_str) or (len(decoy_ids) != len(gold_ids)):
            skip_decoy += 1
            if os.environ.get("DCPO_DEBUG", "1") == "1" and decoy_str == gold_str:
                print(f"[DCPO-V4] pmi_shift row {i}: gold==decoy "
                      f"({gold_str!r}) → zero signal, skipped.", flush=True)
            continue
        ctx_open = list(tokenizer.encode(ctx_open_text, add_special_tokens=False))
        ctx_close = list(tokenizer.encode(ctx_close_text, add_special_tokens=False))
        dmask = divergent_token_mask(gold_ids, decoy_ids)
        # 4 arms in fixed order: gold@open, decoy@open, gold@close, decoy@close.
        arm_prompts.append(ctx_open);  arm_resps.append(gold_ids)
        arm_prompts.append(ctx_open);  arm_resps.append(decoy_ids)
        arm_prompts.append(ctx_close); arm_resps.append(gold_ids)
        arm_prompts.append(ctx_close); arm_resps.append(decoy_ids)
        attempted.append((i, {"gold_ids": gold_ids, "decoy_ids": decoy_ids,
                              "divergent_mask": dmask}))

    if (skip_empty_meta or skip_dup_meta or skip_decoy) and \
            os.environ.get("DCPO_DEBUG", "1") == "1":
        print(f"[DCPO-V4] pmi_shift step={step}: skipped empty_meta={skip_empty_meta} "
              f"dup_meta={skip_dup_meta} decoy={skip_decoy}", flush=True)

    if not attempted:
        _log_pmi_shift_wandb_scalars(step, attempted_rate=0.0, member_rate=0.0,
                                     n_save=0, n_derail=0, rmeta_mean_scored=0.0)
        return r_meta, member, shift_raw

    # 2) Score ALL 4n arms on the frozen ref worker in ONE forward.
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
    tensors, real_n = _build_pmi_score_batches(arm_prompts, arm_resps, pad_unit)
    assert real_n == 4 * len(attempted), (
        f"pmi_shift arm bookkeeping broken: {real_n} != 4*{len(attempted)}")
    try:
        ref_lp = _dcpo_v4_ref_logprobs(trainer, tensors)
    except AssertionError:
        raise
    except Exception as e:
        print(f"[DCPO-V4] pmi_shift ref scoring FAILED ({type(e).__name__}: {e}) — "
              f"R_meta all-zero this batch (member 0).", flush=True)
        if os.environ.get("DCPO_DEBUG", "1") == "1":
            traceback.print_exc()
        _log_pmi_shift_wandb_scalars(step, attempted_rate=len(attempted) / max(1, B),
                                     member_rate=0.0, n_save=0, n_derail=0,
                                     rmeta_mean_scored=0.0)
        return r_meta, member, shift_raw

    # 3) Read back per-token logp, build PMI_open/PMI_close per rollout.
    rows: list = []
    for k, (_i, rmeta_row) in enumerate(attempted):
        Lg = len(rmeta_row["gold_ids"])
        Ld = len(rmeta_row["decoy_ids"])
        base = 4 * k
        gold_open = ref_lp[base + 0, :Lg].float().cpu().numpy()
        decoy_open = ref_lp[base + 1, :Ld].float().cpu().numpy()
        gold_close = ref_lp[base + 2, :Lg].float().cpu().numpy()
        decoy_close = ref_lp[base + 3, :Ld].float().cpu().numpy()
        dmask = rmeta_row["divergent_mask"]
        pmi_open = _pmi_position_scalar(gold_open, decoy_open, dmask)
        pmi_close = _pmi_position_scalar(gold_close, decoy_close, dmask)
        rows.append({"pmi_open": pmi_open, "pmi_close": pmi_close})

    scored, diag = compute_pmi_shift_reward(
        rows, scale=scale, reversal_save=save_big,
        reversal_derail=derail_big, clip=clip,
        reversal_min_magnitude=reversal_eps)

    for j, (i, _rmeta_row) in enumerate(attempted):
        r_meta[i] = scored[j]
        shift_raw[i] = diag["raw_shift"][j]
        member[i] = 0.0 if diag["failures"][j] else 1.0

    _scored_vals = [float(r_meta[i]) for (i, _r) in attempted if member[i] > 0.5]
    rmeta_mean = float(np.mean(_scored_vals)) if _scored_vals else 0.0
    if os.environ.get("DCPO_DEBUG", "1") == "1":
        print(f"[DCPO-V4] pmi_shift step={step}: B={B} attempted={len(attempted)} "
              f"n_save={diag['n_save']} n_derail={diag['n_derail']} "
              f"rmeta_mean_scored={rmeta_mean:.4f}", flush=True)
    _log_pmi_shift_wandb_scalars(step, attempted_rate=len(attempted) / max(1, B),
                                 member_rate=float(member.mean()),
                                 n_save=diag["n_save"], n_derail=diag["n_derail"],
                                 rmeta_mean_scored=rmeta_mean)
    return r_meta, member, shift_raw


def _log_pmi_shift_wandb_scalars(step: int, *, attempted_rate: float,
                                 member_rate: float, n_save: int, n_derail: int,
                                 rmeta_mean_scored: float) -> None:
    """One wandb point for the dcpo/pmishift_* scalars (never kills training)."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "dcpo/pmishift_attempted_rate": float(attempted_rate),
                "dcpo/pmishift_member_rate": float(member_rate),
                "dcpo/pmishift_n_save": float(n_save),
                "dcpo/pmishift_n_derail": float(n_derail),
                "dcpo/pmishift_rmeta_mean_scored": float(rmeta_mean_scored),
            }, step=int(step))
    except Exception:
        pass


# ══ ★수리(0823) 구조율(rescue) 계기 ═══════════════════════════════════════════
#   연구 의도: "풀이가 조금 틀리더라도 계속 점검·확인해서 정답까지 가이드되는가".
#   구조(rescue) = 메타 블록 **앞**에는 정답식이 없었는데 **뒤**에 나타난 롤아웃.
#   보상 항이 아니라 **계기**다 — 팔 정체를 바꾸지 않는다(사전등록 처치 불변).
_ABORT_STREAK: dict = {}
_ABORT_PATIENCE: int = 3


class _CountdownAbort(RuntimeError):
    """사전등록 §7 중단 조건. 계기 예외와 구분하기 위한 전용 타입."""


_RESCUE_EXPR = __import__("re").compile(r"[\d(][\d\s+\-*/().]{4,}")


def _countdown_solves(expr, nums, target) -> bool:
    from src.training import countdown_task as _t
    try:
        return (_t.eval_countdown(expr) == int(target)
                and _t.expr_numbers(expr) == sorted(int(x) for x in nums))
    except Exception:
        return False


def _countdown_rescue_stats(rows, nums_col, target_col) -> dict:
    """행별 구조 판정 → 집계. 메타 미발화 행은 분모에서 뺀다(구조가 정의 안 됨)."""
    from src.training import countdown_rewards as _r
    resc = pre = never = 0
    att = []
    n = 0
    for i, row in enumerate(rows):
        text = row.get("text") or ""
        att.append(len(_RESCUE_EXPR.findall(text)))
        m = _r.parse_meta(text, "new")
        if not m["emitted"]:
            continue
        n += 1
        nums, tgt = nums_col[i], target_col[i]
        def _found(seg):
            for mm in _RESCUE_EXPR.finditer(seg):
                if _countdown_solves(mm.group(0).strip().rstrip("."), nums, tgt):
                    return True
            return False
        before = _found(text[: m["start"] or 0])
        after = _found(text[m["end"] or 0 :])
        if before:
            pre += 1
        elif after:
            resc += 1
        else:
            never += 1
    d = float(n) if n else float("nan")
    return {"rescue_rate": resc / d, "pre_had_rate": pre / d, "never_rate": never / d,
            "rescue_n": resc, "rescue_denom": n,
            "n_attempts_mean": (sum(att) / len(att)) if att else float("nan")}


def _countdown_populate_token_rewards(data, algo_config):
    r"""★COUNTDOWN 메인프로세스 훅 (2026-08-21).

    verl 0.7.1 은 `rollout.mode=sync` 를 **제거**했고(ValueError), async 경로는
    동기 `MetaCotSDCRewardManager.__call__` 을 우회한다(이 파일 2111행 주석):
      "In the async-rollout path the synchronous __call__ DCPO block is bypassed,
       and reward_loop_score (running per-rollout in Ray actors, no group) cannot
       compute the GROUP-aware R_meta (p_hat) — it only emits 0.0 placeholders."
    그 결과 COUNTDOWN 보상이 **한 번도 계산되지 않은 채** 여섯 팔이 같은 보상으로
    돈다(실측: [COUNTDOWN][WIRED] 0건 x 4회, 그중 한 번은 5스텝 학습 완료).
    DCPO 계열은 `_populate_dcpo_region_keys` 로 같은 우회로를 이미 갖고 있다 —
    COUNTDOWN 판만 없었다. 이 함수가 그 자리를 채운다.

    여기는 **메인 프로세스**이고 `uid` 그룹이 온전하므로 p̂ 와 sign(A_corr) 를 셀 수 있다.
    """
    import torch as _t
    from types import SimpleNamespace as _NS

    tok = _ACTIVE_SDC_CONTEXT.get("tokenizer")
    trainer = _ACTIVE_SDC_CONTEXT.get("trainer")
    if tok is None:
        raise RuntimeError("[COUNTDOWN] tokenizer 가 컨텍스트에 없다 — 배선 버그다.")

    bs = len(data)
    prompt_length = data.batch["prompts"].shape[-1]
    decoded = []
    for i in range(bs):
        item = data[i]
        text, _ids = _decode_response(
            tok, item.batch["prompts"], item.batch["responses"],
            item.batch["attention_mask"], prompt_length)
        decoded.append(text)

    shim = _NS(tokenizer=tok, config=_NS(algorithm=algo_config))
    step = int(getattr(trainer, "global_steps", 0) or 0)
    totals = _compute_countdown_arm_stash(shim, data, decoded, bs, prompt_length, step)

    # 시퀀스 스칼라를 마지막 유효 토큰에 싣는다(verl 관례: token_level_rewards 합 = 시퀀스 보상).
    tlr = _t.zeros_like(data.batch["responses"], dtype=_t.float32)
    valid = data.batch["attention_mask"][:, prompt_length:].sum(dim=1) - 1
    for i in range(bs):
        j = int(valid[i])
        if 0 <= j < tlr.shape[1]:
            tlr[i, j] = float(totals[i])
    data.batch["token_level_rewards"] = tlr
    return data


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
        if mode == "ROD_PT":
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
        # Contrastive (R2/SDC_*) modes run the T- decoy teacher here. This
        # branch is the `else` of `if mode in _SINGLE_TEACHER_MODES`.
        neg_answers = (
            gold_answers if contrast_variant in ("stance", "conf") else decoy_answers
        )
        neg_batch = _build_teacher_logprob_batch(
            tokenizer=tokenizer,
            prompt_texts=prompt_texts,
            answer_texts=neg_answers,
            responses=response_tensor,
            response_mask=response_mask,
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
            elif _mode_pf == _COUNTDOWN_MODE:
                # COUNTDOWN_6ARM 의 발전기. 이 줄이 없으면 스태시가 비고 여섯 팔이
                # 전부 상수 0 보상으로 150스텝을 돈다(advantage 0 = 무학습).
                _compute_countdown_arm_stash(
                    self, data, decoded_responses, bs, prompt_length,
                    int(getattr(_ACTIVE_SDC_CONTEXT.get("trainer", None),
                                "global_steps", 0) or 0))
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
                    # countdown 은 삼키지 않는다 — 조용한 0 보상으로 150스텝을 도는
                    # 것이 이 모드가 존재해서 막으려는 바로 그 실패다.
                    if key == "countdown_arm":
                        raise
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

        # COUNTDOWN_6ARM 의 발전기 (동기 경로). 학습·검증 두 배치가 모두 여기를 지나므로
        # 검증도 같은 보상을 본다 — 그래야 팔 간 비교가 성립한다.
        if _ACTIVE_SDC_CONTEXT.get("mode", "") == _COUNTDOWN_MODE:
            _compute_countdown_arm_stash(
                self, data, decoded_responses, bs, prompt_length,
                int(getattr(_ACTIVE_SDC_CONTEXT.get("trainer", None),
                            "global_steps", 0) or 0))

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
                if key == "countdown_arm":            # 조용한 0 을 막는다 (위와 같은 이유)
                    raise
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
        # ★COUNTDOWN: async 우회로. 위 설명은 `_countdown_populate_token_rewards` 참조.
        if _adv_sdc_mode == _COUNTDOWN_MODE:
            data = _countdown_populate_token_rewards(data, config)
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
    # ★수리(0823) 영구 Ray 클러스터 연결 경로.
    #   실측: raylet 은 포트 파일을 15초 기다리는데(하드코딩), 공유 호스트 부하
    #   (load 36~45)에서 대시보드 에이전트 «프로세스 시작»만 23초 걸린다
    #   (18:05:25 raylet 시작 → 18:05:40 크래시 → 18:05:48 에이전트 시작).
    #   의존성을 다 깔아도(모듈 0→7개) 프로세스 기동 지연은 그대로다.
    #   ⇒ 팔마다 클러스터를 새로 띄우지 말고 한 번 띄운 것에 붙는다.
    _ray_addr = os.environ.get("RAY_ADDRESS")
    if _ray_addr and not ray.is_initialized():
        ray.init(address=_ray_addr, runtime_env={"env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "PYTHONPATH": os.environ.get("PYTHONPATH", "/scratch/metacognition"),
            "SDC_MODE_ENV": os.environ.get("SDC_MODE_ENV", ""),
            "VERL_DISABLE_FLASH_XENT": os.environ.get("VERL_DISABLE_FLASH_XENT", "1"),
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", ""),
            # ★검수 0831: 측정모드 스위치도 실어야 한다. Ray 워커는 드라이버의 임의
            #   환경변수를 상속하지 않는다(바로 아래 주석의 SDC_MODE_ENV 사고와 같은
            #   함정). 로컬 head 경로는 raylet 이 드라이버 env 를 물려받아 우연히
            #   통하지만, RAY_ADDRESS 로 **기존 클러스터에 붙는 이 경로**는 통하지
            #   않는다 — COUNTDOWN_INV=1 이 조용히 무시되어 τ·c 관문이 아무것도
            #   안 재고 통과한 것처럼 보인다.
            "COUNTDOWN_INV": os.environ.get("COUNTDOWN_INV", "0"),
        }})
        print(f"[SDC] 기존 Ray 클러스터에 연결: {_ray_addr}", flush=True)
    if not ray.is_initialized():
        # AMLT single-node jobs can expose a non-loopback pod IP that makes
        # Ray's default head bootstrap path hang while waiting for GCS.
        # For this veRL workload we only need a local head on the same node, so
        # pin Ray bootstrap to loopback and skip the dashboard to reduce
        # startup fragility.
        ray.init(
            include_dashboard=False,
            _node_ip_address="127.0.0.1",
            _system_config={"agent_register_timeout_ms": 600000},  # ★0823: raylet 이 15초만 기다리다 크래시(포트파일은 7초 뒤 생성). 공유호스트 부하로 에이전트 기동이 22초 지연됨. RAY_* 환경변수로는 안 바뀌어 여기서 직접 넘긴다.
            object_store_memory=20_000_000_000,  # ★0823: 기본은 /dev/shm 에 200GB mmap(store_runner.cc:50). 공유호스트 부하에서 이 mmap 이 기동을 지연시켜 raylet 의 하드코딩 30초 포트 대기를 넘긴다. verl 은 20GB 면 충분하다.
            # propagate PYTHONPATH to Ray workers so hydra.utils.instantiate can
            # import custom _target_ classes (e.g. the E.9 BCIConfAgentLoop) by
            # FQDN inside the rollout workers. Harmless for every other mode (the
            # repo is already importable); removes the one registration unknown.
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "PYTHONPATH": os.environ.get("PYTHONPATH", "/scratch/metacognition"),
                # ★2026-08-21: Ray 워커는 드라이버의 임의 환경변수를 **상속하지 않는다**.
                #   여기에 실은 것만 전달된다. `mode` 는 모듈 변수라 워커에 안 가고
                #   (R16 이 기록한 동일 함정), 그 결과 COUNTDOWN 분기가 통째로 죽어
                #   여섯 팔이 전부 countdown 보상 없이 돌 뻔했다(실측 5스텝, WIRED 0건).
                "SDC_MODE_ENV": os.environ.get("SDC_MODE_ENV", ""),
                "VERL_DISABLE_FLASH_XENT": os.environ.get("VERL_DISABLE_FLASH_XENT", "1"),
                "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", ""),
                # ★검수 0831: 측정모드 스위치(위 RAY_ADDRESS 경로와 같은 이유).
                "COUNTDOWN_INV": os.environ.get("COUNTDOWN_INV", "0"),
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

    # KNOB REGISTRY GATE 0803. A retired-lineage knob must not be reachable, and no
    # load-bearing knob may be inherited invisibly. Both failed once: dcpo_meta_floor
    # went 0.05 -> 0.0 on 2026-06-22 for cf_group, a pmi_shift run inherited it six
    # weeks later, and meta emission fell 1.00 -> 0.018 with every declared launch gate
    # still green. Opt out only for a mode that has no dcpo_* surface at all.
    if str(getattr(config, "mode", "")).upper() == _COUNTDOWN_MODE:
        # 팔 문자열은 load-bearing 이다 — 잘못되면 여섯 잡이 조용히 같아진다.
        # 프리패스 안에도 같은 검사가 있지만, 여기서 죽으면 GPU 를 40분 안 태운다.
        from src.training.countdown_rewards import ARM_SPECS as _CD_ARM_SPECS
        _cd_arm = str(getattr(getattr(config, "algorithm", None),
                              "countdown_arm", "") or "").upper()
        if _cd_arm not in _CD_ARM_SPECS:
            raise ValueError(
                f"COUNTDOWN_6ARM: algorithm.countdown_arm={_cd_arm!r} 가 미지정이거나 "
                f"미지의 팔이다. 가능한 값: {sorted(_CD_ARM_SPECS)}")
        print(f"[SDC] countdown arm = {_cd_arm}")
    # ⛔COUNTDOWN 을 이 게이트에 넣지 마라. 2026-08-19 에 넣었다가 7잡을 잃었다.
    #   core/KNOBS.yaml 은 **DCPO 세대의 계약**이다 — `dcpo_rmeta_source`(상호배타
    #   메타보상 세대 선택)와 `dcpo_ack_load_bearing` 을 요구하는데, COUNTDOWN_6ARM 은
    #   시퀀스 수준 GRPO 라 그 노브를 하나도 안 쓴다. 그래서 통과가 원리적으로 불가능하고,
    #   실패는 부팅 시 KnobRegistryError 로 나온다.
    #   COUNTDOWN 의 팔 검증은 위의 fail-closed(`countdown_arm not in ARM_SPECS -> ValueError`)와
    #   배치마다 찍는 `[COUNTDOWN][WIRED]` 가 담당한다 — 레지스트리보다 강한 검사다.
    if str(getattr(config, "mode", "")).upper().startswith("TRIOBJ"):
        from src.training.knob_registry import validate as _validate_knobs
        _resolved = _validate_knobs(getattr(config, "algorithm", None))
        print("[SDC] knob registry OK — %d live knobs resolved:" % len(_resolved))
        for _name, _value in _resolved:
            print("[SDC]   %-40s = %r" % (_name, _value))

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
    # ★0902: 키가 없어도 WANDB_MODE=offline 이면 로컬 run 디렉터리에 기록한다 (나중에 `wandb sync`).
    if os.environ.get("WANDB_MODE", "") == "offline":
        has_wandb_key = True
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
    reward_cfg = REWARD_CONFIGS[mode]

    # FAIL-FAST gate-coherence check (codereview CRITICAL-2, the sdc_enabled-class
    # bug): patched_compute_advantage runs the SDC branch only when sdc_enabled is
    # truthy OR the mode is region-routed. A teacher-ON mode whose YAML omits/false
    # sdc_enabled would otherwise SILENTLY train as plain GDPO while labeled e.g.
    # ROD_MQ_CONTRAST — exactly how the v2/v3 region routing stayed off unnoticed.
    _teacher_on_modes = _SINGLE_TEACHER_MODES | _CONTRASTIVE_MODES
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
