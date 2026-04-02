"""Upload dataset artifacts into the HF dataset repo used for experiments."""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="iamseungpil/metacot")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--path-in-repo-prefix", default="data")
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    api = HfApi()
    for file_path in args.files:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        path_in_repo = f"{args.path_in_repo_prefix}/{path.name}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload {path.name}",
        )
        print(f"uploaded {path} -> {args.repo_id}:{path_in_repo}")


if __name__ == "__main__":
    main()
