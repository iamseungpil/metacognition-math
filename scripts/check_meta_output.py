"""Check what Meta SFT model actually generates."""
import torch
import re
import sys
sys.path.insert(0, "/scratch/metacognition")

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.metacot.prompt import META_START, META_END, parse_meta_blocks

tokenizer = AutoTokenizer.from_pretrained("checkpoints/meta_sft")
tokenizer.add_special_tokens({"additional_special_tokens": [META_START, META_END]})
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/meta_sft", torch_dtype=torch.bfloat16, device_map="cuda:1"
)
model.resize_token_embeddings(len(tokenizer))

problems = [
    "Find the remainder when 2^100 is divided by 7.",
    "What is 15 + 27?",
    "Solve x^2 - 5x + 6 = 0.",
]

for q in problems:
    msgs = [{"role": "user", "content": q}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").to("cuda:1")
    out = model.generate(**ids, max_new_tokens=600, temperature=0.7, do_sample=True)
    text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
    parsed = parse_meta_blocks(text)

    print(f"Q: {q}")
    print(f"  blocks={parsed['num_blocks']}, confidences={parsed['confidences']}, valid={parsed['valid']}")

    pat = re.escape(META_START) + "(.*?)" + re.escape(META_END)
    blocks = re.findall(pat, text, re.DOTALL)
    if blocks:
        for i, b in enumerate(blocks[:3]):
            print(f"  [meta {i}]: {b[:200].strip()}")
    else:
        print(f"  [no meta] output: {text[:300].strip()}")
    print()
