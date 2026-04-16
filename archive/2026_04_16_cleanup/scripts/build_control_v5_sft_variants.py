"""Build v5 control SFT variants from the generated control dataset."""
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
        "control_v5_all_sft.parquet": df,
        "control_v5_verify_sft.parquet": df[df["scenario"].isin(["straight", "verify"])].copy(),
        "control_v5_redirect_sft.parquet": df[df["scenario"].isin(["straight", "redirect"])].copy(),
    }

    for name, frame in variants.items():
        path = out_dir / name
        frame.to_parquet(path, index=False)
        print(f"{name}: {len(frame)} rows")
        if len(frame) and "scenario" in frame.columns:
            print(frame["scenario"].value_counts().to_string())


if __name__ == "__main__":
    main()
