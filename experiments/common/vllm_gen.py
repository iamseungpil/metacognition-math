"""Thin vLLM generation wrapper for the B.1/B.2 probes (CTSD Phase C).

WHY this exists (efficiency, NOT statistics):
  Under sampling, E20a force-opened with <|meta|> rarely emits EOS, so the HF
  `model.generate` path (a3.gen_batch) runs the FULL max_new every single call
  (~19 generate calls/problem → intractable: 0/24 in 5.5h). vLLM respects
  EOS/stop and batches all prompts in one call, giving ~10-50x speedup. ONLY the
  generation MECHANISM changes — every statistical / experimental block in the
  probes (cross-fit, discovery/confirmation split, oriented-AUC, the 5 metrics,
  gates, power-guard, status field, teacher metrics) stays byte-for-byte identical.

Coexistence note: the B.1/B.2 probes ALSO need HF forward passes (entropy,
taken-token logp, pause-ref, teacher KL) of ~8B models (~16GB each, run one at a
time). vLLM is therefore created with a MODEST gpu_memory_utilization (0.45 ≈
36GB on an 80GB A100) so an HF scoring model can coexist. The probe calls
VllmGen.free() before its HF teacher/ref phases to release the vLLM allocation.

API (vllm 0.10.2):
  - SamplingParams supports n, max_tokens, temperature, top_p, seed, stop.
  - LLM.generate accepts prompt_token_ids=List[List[int]].
  - each result r has r.outputs (list of len n), each o has o.token_ids.
"""
from __future__ import annotations
import gc

# E20a tokenizer path: identical Qwen3 vocab (<|meta|>=151669, <|/meta|>=151670).
# The v8 checkpoint tokenizer FAILS to load under transformers 4.57.6, so the
# caller substitutes this path for the vLLM tokenizer when on the v8 substrate.
E20A_TOKENIZER_PATH = "/home/v-seungplee/sft_e20a_local"
META_OPEN_ID = 151669      # <|meta|>
META_CLOSE_ID = 151670     # <|/meta|>


def safe_tokenizer_path(model_path: str) -> str:
    """Return a tokenizer path vLLM can load. The v8 checkpoint tokenizer raises
    under transformers 4.57.6 (AttributeError in _set_model_specific_special_tokens);
    fall back to the E20a tokenizer, which has an IDENTICAL Qwen3 vocab. ASSERT the
    meta-token IDs so a silent vocab mismatch can never slip through (same idea as
    the probes' load_tokenizer)."""
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        path = model_path
    except Exception as e:
        print(f"[vllm_gen] tokenizer {model_path} failed "
              f"({type(e).__name__}: {str(e)[:80]}); falling back to E20a tokenizer "
              f"(identical vocab)")
        tok = AutoTokenizer.from_pretrained(E20A_TOKENIZER_PATH)
        path = E20A_TOKENIZER_PATH
    assert tok.convert_tokens_to_ids("<|meta|>") == META_OPEN_ID, "META_OPEN_ID mismatch"
    assert tok.convert_tokens_to_ids("<|/meta|>") == META_CLOSE_ID, "META_CLOSE_ID mismatch"
    return path


class VllmGen:
    """Batched vLLM generation over pre-tokenized prompts.

    The whole point is BATCHING: one VllmGen.generate call serves many prompts at
    once (vs the old per-(position,arm) HF loop), which combined with EOS-respecting
    early stop is the speed win. gpu_memory_utilization is kept modest (0.45) so an
    HF scoring model (~16GB) can live on the same 80GB A100.
    """

    def __init__(self, model_path: str, tokenizer_path: str | None = None,
                 gpu_memory_utilization: float = 0.45, max_model_len: int = 4096,
                 seed: int = 0):
        from vllm import LLM
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.llm = LLM(
            model=model_path,
            tokenizer=self.tokenizer_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,
            trust_remote_code=True,
            seed=seed,
        )

    def generate(self, prompt_token_ids: list[list[int]], n: int, max_tokens: int,
                 temperature: float = 0.7, top_p: float = 0.95,
                 seed: int | None = None,
                 stop: list[str] | None = None) -> list[list[list[int]]]:
        """ONE batched llm.generate over all prompts. Returns, per prompt, a list of
        n continuations, each a list[int] of generated token ids (response-relative
        to the prompt — vLLM's o.token_ids excludes the prompt). BATCH everything in
        a single call (do NOT loop per-prompt — that defeats the speed win)."""
        from vllm import SamplingParams
        sp = SamplingParams(
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop=stop,
        )
        # vLLM 0.10.x: pass token-id prompts as TokensPrompt dicts (the top-level
        # prompt_token_ids= kwarg was removed).
        prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        results = self.llm.generate(prompts, sampling_params=sp)
        return [[list(o.token_ids) for o in r.outputs] for r in results]

    def free(self):
        """Release the vLLM GPU allocation so the HF scoring phase has room. Deletes
        the LLM, runs gc + empties the CUDA cache, and tears down vLLM's model-parallel
        state if that helper is available in this vllm version."""
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
        except Exception:
            destroy_model_parallel = None
        try:
            del self.llm
        except Exception:
            pass
        self.llm = None
        gc.collect()
        if destroy_model_parallel is not None:
            try:
                destroy_model_parallel()
            except Exception:
                pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
