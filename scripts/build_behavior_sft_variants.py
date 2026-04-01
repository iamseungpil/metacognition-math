"""Build behavior-SFT variants from the TRAPI-generated dataset."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "behavior_all_sft.parquet": df,
        "behavior_verify_sft.parquet": df[df["scenario"].isin(["straight", "verify"])].copy(),
        "behavior_redirect_sft.parquet": df[df["scenario"].isin(["straight", "redirect"])].copy(),
    }
    for name, frame in variants.items():
        path = out_dir / name
        frame.to_parquet(path, index=False)
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
