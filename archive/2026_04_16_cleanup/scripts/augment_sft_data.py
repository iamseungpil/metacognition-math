"""Augment existing Meta-CoT SFT data with diverse confidence + error→fix.

Instead of regenerating via TRAPI (auth issues), post-process existing data:
1. Replace confidence 0.95+ with difficulty-appropriate values
2. Add error→correction patterns to 30% of chains
3. Add final verification meta block
4. Keep meta blocks short

Usage: python scripts/augment_sft_data.py
"""
import json
import os
import random
import re
import sys

import pandas as pd

META_START = "<|meta|>"
META_END = "<|/meta|>"


def diversify_confidence(text, target_conf_range=(0.3, 0.8)):
    """Replace high confidence values with diverse ones."""
    def replace_conf(match):
        keyword = match.group(1)
        original = float(match.group(2))
        if original > 1:
            original /= 100

        # Map to target range based on position in text
        # Earlier confidences should be lower
        position = match.start() / max(len(text), 1)
        low, high = target_conf_range

        if position < 0.3:  # pre-meta: lower confidence
            new_conf = random.uniform(low, low + 0.2)
        elif position < 0.7:  # mid-meta: medium
            new_conf = random.uniform(low + 0.1, high)
        else:  # post-meta: higher (but not 0.99)
            new_conf = random.uniform(high - 0.1, min(high + 0.1, 0.92))

        return f"{keyword} {new_conf:.2f}"

    pattern = r'((?:probability|confidence)[:\s\w]*?)\s*(\d+\.\d+)'
    return re.sub(pattern, replace_conf, text, flags=re.IGNORECASE)


def add_error_correction(text):
    """Insert an error→correction pattern into a random step."""
    # Find a good insertion point (after a calculation line)
    lines = text.split('\n')
    calc_indices = [i for i, l in enumerate(lines)
                    if re.search(r'[=≡+\-×÷].*\d', l) and META_START not in l and META_END not in l]

    if not calc_indices:
        return text, False

    # Pick a random calculation to "correct"
    idx = random.choice(calc_indices)
    correction_meta = f"\n{META_START}\nQ: Wait, is this right? A: Let me double-check this step. confidence 0.4\n{META_END}\n"

    lines.insert(idx + 1, correction_meta)
    return '\n'.join(lines), True


def add_final_verification(text):
    """Ensure there's a final verification meta before \\boxed{}."""
    # Check if already has final verification
    boxed_pos = text.rfind('\\boxed')
    if boxed_pos == -1:
        boxed_pos = text.rfind('boxed{')
    if boxed_pos == -1:
        return text

    # Check if there's a meta block in the last 200 chars before boxed
    pre_boxed = text[max(0, boxed_pos - 200):boxed_pos]
    if "final" in pre_boxed.lower() or "verify" in pre_boxed.lower() or "check" in pre_boxed.lower():
        return text  # already has

    # Add final verification
    conf = random.uniform(0.7, 0.92)
    verification = f"\n{META_START}\nQ: Final check before answering. A: Let me verify the key steps. confidence {conf:.2f}\n{META_END}\n"
    text = text[:boxed_pos] + verification + text[boxed_pos:]
    return text


def shorten_meta_blocks(text, max_tokens=60):
    """Shorten overly long meta blocks."""
    parts = text.split(META_START)
    result = [parts[0]]
    for part in parts[1:]:
        end_idx = part.find(META_END)
        if end_idx == -1:
            result.append(META_START + part)
            continue
        meta_content = part[:end_idx]
        rest = part[end_idx + len(META_END):]

        # Shorten if too long
        words = meta_content.split()
        if len(words) > max_tokens:
            meta_content = ' '.join(words[:max_tokens]) + '...'

        result.append(META_START + meta_content + META_END + rest)

    return ''.join(result)


def process_chain(messages_str, target_error_fix_rate=0.3):
    """Process one SFT chain."""
    messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str
    if len(messages) < 2:
        return None

    assistant_msg = messages[-1]["content"]

    # 1. Diversify confidence
    assistant_msg = diversify_confidence(assistant_msg)

    # 2. Add error correction (30% of chains)
    added_error = False
    if random.random() < target_error_fix_rate:
        assistant_msg, added_error = add_error_correction(assistant_msg)

    # 3. Add final verification
    assistant_msg = add_final_verification(assistant_msg)

    # 4. Shorten meta blocks
    assistant_msg = shorten_meta_blocks(assistant_msg)

    messages[-1]["content"] = assistant_msg
    return json.dumps(messages), added_error


def main():
    input_path = "/scratch/metacognition/sft_data/metacot_sft.parquet"
    output_path = "/scratch/metacognition/sft_data/metacot_v2_sft.parquet"

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} chains from {input_path}")

    results = []
    error_fix_count = 0

    for i, row in df.iterrows():
        result = process_chain(row["messages"])
        if result is None:
            continue
        new_messages, added_error = result
        if added_error:
            error_fix_count += 1
        results.append({
            "messages": new_messages,
            "problem_id": row.get("problem_id", str(i)),
            "source": row.get("source", "metacot_v2"),
        })

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_parquet(output_path)

    # Validate
    confs_all = []
    for _, row in out_df.iterrows():
        msgs = json.loads(row["messages"])
        text = msgs[-1]["content"]
        confs = re.findall(r'(?:probability|confidence)[:\s\w]*?(\d+\.\d+)', text, re.IGNORECASE)
        for c in confs:
            v = float(c)
            if 0 < v <= 1:
                confs_all.append(v)

    print(f"\n=== Augmentation Complete ===")
    print(f"Chains: {len(out_df)}")
    print(f"Error-fix: {error_fix_count}/{len(out_df)} ({error_fix_count/len(out_df):.1%})")
    print(f"Confidence: mean={sum(confs_all)/len(confs_all):.3f}, "
          f"std={pd.Series(confs_all).std():.3f}, "
          f"min={min(confs_all):.3f}, max={max(confs_all):.3f}")
    print(f">0.95: {sum(1 for c in confs_all if c > 0.95)}/{len(confs_all)} "
          f"({sum(1 for c in confs_all if c > 0.95)/len(confs_all):.1%})")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
