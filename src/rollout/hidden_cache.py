"""Cache hidden states from model for probe training."""
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.rollout.vllm_rollout import build_chat_messages


def cache_hidden_states(
    rollouts_path: str,
    output_dir: str,
    model_path: str,
    num_samples: int = 30000,
    max_len: int = 2048,
    device: str = "cuda",
):
    """Cache last-layer hidden states for probe training.

    Samples a balanced set of correct/incorrect rollouts and saves
    their hidden states as .pt files with a manifest.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hs_dir = output_path / "hidden_states"
    hs_dir.mkdir(exist_ok=True)

    # Load rollouts
    df = pd.read_parquet(rollouts_path)

    # Balance correct/incorrect
    correct = df[df["is_correct"]].sample(
        n=min(num_samples // 2, df["is_correct"].sum()),
        random_state=42,
    )
    incorrect = df[~df["is_correct"]].sample(
        n=min(num_samples // 2, (~df["is_correct"]).sum()),
        random_state=42,
    )
    selected = pd.concat([correct, incorrect]).sample(frac=1, random_state=42)
    print(f"Selected {len(selected)} samples ({len(correct)} correct, {len(incorrect)} incorrect)")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    ).to(device).eval()

    # Process samples
    manifest = []
    for idx, (_, row) in enumerate(tqdm(selected.iterrows(), total=len(selected))):
        messages = build_chat_messages(row["question"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt + row["completion"]

        enc = tokenizer(
            full_text, return_tensors="pt",
            truncation=True, max_length=max_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Last layer hidden state, full sequence (probe does its own pooling)
            last_hidden = outputs.hidden_states[-1][0].cpu().to(torch.float16)

        del outputs, input_ids, attention_mask
        if (idx + 1) % 500 == 0:
            torch.cuda.empty_cache()

        # Save hidden state
        hs_file = f"hs_{idx:06d}.pt"
        torch.save(last_hidden, hs_dir / hs_file)

        manifest.append({
            "idx": idx,
            "problem_id": row["problem_id"],
            "is_correct": bool(row["is_correct"]),
            "category": row["category"],
            "difficulty": row["difficulty"],
            "hidden_state_file": f"hidden_states/{hs_file}",
            "seq_len": last_hidden.shape[0],
        })

        if (idx + 1) % 1000 == 0:
            print(f"  Cached {idx + 1}/{len(selected)} hidden states")

    # Split into train/val
    split_idx = int(len(manifest) * 0.9)
    train_manifest = manifest[:split_idx]
    val_manifest = manifest[split_idx:]

    with open(output_path / "train_manifest.json", "w") as f:
        json.dump(train_manifest, f)
    with open(output_path / "val_manifest.json", "w") as f:
        json.dump(val_manifest, f)

    print(f"Done! Train: {len(train_manifest)}, Val: {len(val_manifest)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-samples", type=int, default=30000)
    args = parser.parse_args()

    cache_hidden_states(
        rollouts_path=args.rollouts_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        num_samples=args.num_samples,
    )
