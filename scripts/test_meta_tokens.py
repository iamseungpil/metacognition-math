import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("checkpoints/qwen3_metacot_v2_sft", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("checkpoints/qwen3_metacot_v2_sft", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

print("META in vocab:", "<|meta|>" in tok.get_vocab())
print("Is special:", "<|meta|>" in (tok.additional_special_tokens or []))
print("Additional special tokens:", tok.additional_special_tokens)

msgs = [{"role": "user", "content": "What is 15 + 27?"}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.7)

g_full = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)
g_skip = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("\n=== skip_special_tokens=False ===")
print(g_full[:500])
print("\nHas <|meta|>:", "<|meta|>" in g_full)

print("\n=== skip_special_tokens=True ===")
print(g_skip[:500])
print("\nHas <|meta|>:", "<|meta|>" in g_skip)

# This is what TRL uses for reward computation
print("\n=== VERDICT ===")
if "<|meta|>" in g_skip:
    print("META_VISIBLE=True - reward functions CAN see meta tokens")
else:
    print("META_VISIBLE=False - reward functions CANNOT see meta tokens - BUG STILL EXISTS")
