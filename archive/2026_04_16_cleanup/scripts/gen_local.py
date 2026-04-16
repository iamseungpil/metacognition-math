"""Generate Meta-CoT v2 data from this VM (has az cli for TRAPI auth)."""
import json, os, sys, time, random, re, math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.metacot.prompt_v2 import META_COT_V2_SYSTEM_PROMPT, build_metacot_v2_prompt, META_START, META_END

from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider

provider = get_bearer_token_provider(AzureCliCredential(), "api://trapi/.default")

def get_client():
    return AzureOpenAI(
        azure_endpoint="https://trapi.research.microsoft.com/gcr/shared",
        api_key=provider(), api_version="2025-04-01-preview"
    )

questions = json.load(open("/tmp/questions_clean.json"))
random.shuffle(questions)
N = min(5000, len(questions))
print(f"Generating {N} chains with gpt-5.4-mini...")

results = []
failed = 0
client = get_client()
client_refresh = time.time()

def gen_one(idx):
    global client, client_refresh
    # Refresh token every 30 min
    if time.time() - client_refresh > 1800:
        client = get_client()
        client_refresh = time.time()

    q = questions[idx]
    prompt = build_metacot_v2_prompt(q["q"], q["pr"])

    for attempt in range(5):
        try:
            resp = client.responses.create(
                model="gpt-5.4-mini_2026-03-17",
                input=[
                    {"role": "system", "content": META_COT_V2_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            text = resp.output_text
            if META_START in text and "boxed" in text:
                return {"q": q["q"], "text": text, "gt": q.get("gt", "")}
        except Exception as e:
            if "429" in str(e):
                time.sleep(5 * (2**attempt) + random.random()*3)
            else:
                time.sleep(3)
    return None

with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(gen_one, i): i for i in range(N)}
    for i, f in enumerate(as_completed(futures)):
        r = f.result()
        if r:
            results.append(r)
        else:
            failed += 1
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{N}: {len(results)} valid, {failed} failed", flush=True)

# Save
import pandas as pd
records = []
for r in results:
    messages = json.dumps([
        {"role": "user", "content": r["q"]},
        {"role": "assistant", "content": r["text"]},
    ])
    records.append({"messages": messages, "source": "metacot_v2_trapi"})

df = pd.DataFrame(records)
df.to_parquet("/tmp/metacot_v2_trapi.parquet")

# Stats
all_confs = []
error_fix = 0
for r in results:
    confs = re.findall(r'(?:probability|confidence)[\s\w:]*?(\d+\.\d+)', r["text"], re.I)
    for c in confs:
        v = float(c)
        if 0 < v <= 1: all_confs.append(v)
    if re.search(r'\b(wait|wrong|fix|actually|mistake)\b', r["text"], re.I):
        error_fix += 1

print(f"\n=== DONE ===")
print(f"Valid: {len(results)}/{N} ({len(results)/N:.1%})")
print(f"Conf: mean={sum(all_confs)/len(all_confs):.3f}, >0.95={sum(1 for c in all_confs if c>0.95)/len(all_confs):.1%}")
print(f"Error-fix: {error_fix}/{len(results)} ({error_fix/len(results):.1%})")
print(f"Saved: /tmp/metacot_v2_trapi.parquet")
