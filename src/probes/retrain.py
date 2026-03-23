"""Re-train Gnosis probe on SFT model's hidden states.

After SFT changes the model, hidden state distribution shifts.
Probe must be re-trained on the new model's representations.

Also supports extracting hidden states at <|meta|> positions
for step-level probe training.
"""
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.metacot.prompt import META_START, META_END
from src.probes.simple_probe import SimpleCorrectnessProbe, train_simple_probe


def extract_hidden_states_from_sft_model(
    model_path: str,
    rollouts_path: str,
    output_dir: str,
    num_samples: int = 5000,
    max_len: int = 2048,
    device: str = "cuda",
):
    """Extract hidden states from SFT model for probe re-training.

    Extracts at two levels:
    1. Full-sequence pooled (for overall correctness prediction)
    2. At <|meta|> positions (for step-level prediction during RL)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hs_dir = output_path / "hidden_states"
    hs_dir.mkdir(exist_ok=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [META_START, META_END]
    })

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    ).to(device).eval()

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Load rollouts
    df = pd.read_parquet(rollouts_path)
    correct = df[df["is_correct"]].sample(
        n=min(num_samples // 2, df["is_correct"].sum()), random_state=42,
    )
    incorrect = df[~df["is_correct"]].sample(
        n=min(num_samples // 2, (~df["is_correct"]).sum()), random_state=42,
    )
    selected = pd.concat([correct, incorrect]).sample(frac=1, random_state=42)
    print(f"Extracting from {len(selected)} samples ({len(correct)} correct, {len(incorrect)} incorrect)")

    manifest = []
    for idx, (_, row) in enumerate(tqdm(selected.iterrows(), total=len(selected))):
        full_text = row["question"] + "\n" + row["completion"]
        enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            last_hidden = outputs.hidden_states[-1][0].cpu().to(torch.float16)

        del outputs, input_ids, attention_mask
        if (idx + 1) % 500 == 0:
            torch.cuda.empty_cache()

        hs_file = f"hs_{idx:06d}.pt"
        torch.save(last_hidden, hs_dir / hs_file)

        manifest.append({
            "idx": idx,
            "problem_id": row["problem_id"],
            "is_correct": bool(row["is_correct"]),
            "category": row["category"],
            "hidden_state_file": f"hidden_states/{hs_file}",
            "seq_len": last_hidden.shape[0],
        })

    # Split train/val
    split_idx = int(len(manifest) * 0.9)
    with open(output_path / "train_manifest.json", "w") as f:
        json.dump(manifest[:split_idx], f)
    with open(output_path / "val_manifest.json", "w") as f:
        json.dump(manifest[split_idx:], f)

    print(f"Done! Train: {split_idx}, Val: {len(manifest) - split_idx}")
    del model
    torch.cuda.empty_cache()


def retrain_probe(
    model_path: str,
    rollouts_path: str,
    output_dir: str,
    hidden_dim: int = 3584,
    num_samples: int = 5000,
    epochs: int = 10,
    device: str = "cuda",
):
    """Full pipeline: extract hidden states → train probe."""
    hs_dir = Path(output_dir) / "hidden_states_cache"

    print("Step 1: Extracting hidden states from SFT model...")
    extract_hidden_states_from_sft_model(
        model_path=model_path,
        rollouts_path=rollouts_path,
        output_dir=str(hs_dir),
        num_samples=num_samples,
        device=device,
    )

    print("Step 2: Training probe...")
    import wandb
    run = wandb.init(project="metacot-math", name="probe-retrain", reinit=True)
    auroc = train_simple_probe(
        data_dir=str(hs_dir),
        output_dir=output_dir,
        hidden_dim=hidden_dim,
        epochs=epochs,
        wandb_run=run,
    )
    wandb.finish()
    print(f"Probe re-trained. AUROC: {auroc:.4f}")
    return auroc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rollouts-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=3584)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    retrain_probe(
        model_path=args.model_path,
        rollouts_path=args.rollouts_path,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_samples=args.num_samples,
        epochs=args.epochs,
    )
