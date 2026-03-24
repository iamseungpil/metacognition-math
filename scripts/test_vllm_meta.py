"""Test whether vLLM generates <|meta|> tokens."""
import sys
sys.path.insert(0, "/scratch/metacognition")
from vllm import LLM, SamplingParams

llm = LLM(
    model="checkpoints/meta_sft",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    max_model_len=1024,
    dtype="bfloat16",
    trust_remote_code=True,
)
tok = llm.get_tokenizer()
model_vocab = llm.llm_engine.model_config.hf_config.vocab_size
print(f"tokenizer vocab={len(tok)}, model config vocab={model_vocab}")

meta_token = "<|meta|>"
meta_id = tok.convert_tokens_to_ids(meta_token)
print(f"meta token id={meta_id}")

# Check if model embedding matches tokenizer
if model_vocab < len(tok):
    print(f"WARNING: model vocab ({model_vocab}) < tokenizer vocab ({len(tok)})")
    print("vLLM may not generate tokens beyond model vocab!")

sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=300)
msgs = [{"role": "user", "content": "What is 2+3?"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
out = llm.generate([prompt], sampling_params=sp)
text = out[0].outputs[0].text

has_meta = meta_token in text
print(f"has_meta={has_meta}")
print(f"Output: {text[:300]}")
