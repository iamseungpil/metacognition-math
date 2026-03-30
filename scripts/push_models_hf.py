"""Push model checkpoints to HuggingFace Hub.

Uploads a local model directory (full checkpoint or merged LoRA) to the
iamseungpil/metacot repository on HuggingFace.

Usage:
  python scripts/push_models_hf.py \
      --model_path checkpoints/qwen3_metacot_v2_sft \
      --model_name qwen3_metacot_v2_sft

  python scripts/push_models_hf.py \
      --model_path checkpoints/grpo_v2_E3/final \
      --model_name grpo_v2_E3_final

  # Push multiple models at once:
  python scripts/push_models_hf.py \
      --model_path checkpoints/qwen3_base_sft checkpoints/qwen3_metacot_v2_sft \
      --model_name qwen3_base_sft qwen3_metacot_v2_sft
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO_ID = "iamseungpil/metacot"
HF_TOKEN = os.environ.get("HF_TOKEN", "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")

# Files to skip when uploading (training artifacts, not needed for inference)
IGNORE_PATTERNS = [
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "*.safetensors.index.json.tmp",
    "samples/*",
    "responses/*",
    "runs/*",
    "__pycache__/*",
    "*.pyc",
]


# ---------------------------------------------------------------------------
# Upload logic
# ---------------------------------------------------------------------------
def push_model(
    model_path: str,
    model_name: str,
    repo_id: str = HF_REPO_ID,
    token: str = HF_TOKEN,
) -> None:
    """Upload a model checkpoint directory to HuggingFace Hub.

    Args:
        model_path: Local path to the model directory.
        model_name: Name for the model subdirectory in the repo.
        repo_id: HuggingFace repository ID (default: iamseungpil/metacot).
        token: HuggingFace API token.
    """
    from huggingface_hub import HfApi

    model_dir = Path(model_path).resolve()
    if not model_dir.is_dir():
        print(f"  ERROR: {model_dir} is not a directory. Skipping.")
        return

    # Verify essential model files exist
    has_model_files = (
        any(model_dir.glob("*.safetensors"))
        or any(model_dir.glob("*.bin"))
        or (model_dir / "adapter_config.json").exists()
    )
    if not has_model_files:
        print(f"  WARNING: No model weight files found in {model_dir}")
        print(f"           Expected *.safetensors, *.bin, or adapter_config.json")
        response = input("  Continue anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("  Skipping.")
            return

    # Upload path in repo: models/<model_name>/
    path_in_repo = f"models/{model_name}"

    print(f"  Uploading: {model_dir}")
    print(f"       -> hf://{repo_id}/{path_in_repo}/")

    api = HfApi(token=token)

    # Ensure the repo exists (create if needed, type=model for model files)
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        # Repo might be a dataset type; try dataset
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except Exception:
            print(f"  Creating new repo: {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Upload the folder
    commit_info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        path_in_repo=path_in_repo,
        commit_message=f"Upload model: {model_name}",
        ignore_patterns=IGNORE_PATTERNS,
    )

    print(f"  Done. Commit: {commit_info.commit_url}")


def list_model_files(model_path: str) -> None:
    """Print model files that would be uploaded."""
    model_dir = Path(model_path).resolve()
    if not model_dir.is_dir():
        print(f"  ERROR: {model_dir} is not a directory.")
        return

    print(f"\n  Files in {model_dir}:")
    total_size = 0
    for f in sorted(model_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            # Mark files that would be skipped
            rel = f.relative_to(model_dir)
            skip = any(rel.match(pat) for pat in IGNORE_PATTERNS)
            marker = " [SKIP]" if skip else ""
            print(f"    {rel}  ({size_mb:.1f} MB){marker}")
    print(f"  Total: {total_size:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Push model checkpoints to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_path",
        nargs="+",
        required=True,
        help="Path(s) to model checkpoint directories",
    )
    parser.add_argument(
        "--model_name",
        nargs="+",
        required=True,
        help="Name(s) for the model(s) in HF repo (must match --model_path count)",
    )
    parser.add_argument(
        "--repo_id",
        default=HF_REPO_ID,
        help=f"HuggingFace repo ID (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List files that would be uploaded without uploading",
    )
    args = parser.parse_args()

    if len(args.model_path) != len(args.model_name):
        print(f"ERROR: --model_path ({len(args.model_path)} items) and "
              f"--model_name ({len(args.model_name)} items) must have the same count.")
        sys.exit(1)

    for model_path, model_name in zip(args.model_path, args.model_name):
        print(f"\n{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        if args.dry_run:
            list_model_files(model_path)
        else:
            push_model(
                model_path=model_path,
                model_name=model_name,
                repo_id=args.repo_id,
            )

    print(f"\n{'='*60}")
    print(f"  All done. Repo: https://huggingface.co/{args.repo_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
