"""Print stratified control-v4 samples for manual QC."""
import argparse
import json
import random

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--samples-per-bucket", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    df = pd.read_parquet(args.input)

    print("=== COUNTS ===")
    if len(df):
        print(df.groupby(["scenario", "difficulty"]).size().to_string())
    print()

    for (scenario, difficulty), group in df.groupby(["scenario", "difficulty"]):
        print(f"=== {scenario} / {difficulty} ===")
        sample_n = min(args.samples_per_bucket, len(group))
        sample_df = group.sample(n=sample_n, random_state=args.seed) if sample_n else group
        for _, row in sample_df.iterrows():
            messages = json.loads(row["messages"])
            answer = messages[-1]["content"]
            print(
                {
                    "source": row.get("source"),
                    "topic": row.get("topic"),
                    "meta_count": row.get("meta_count"),
                    "repeated_intervention": row.get("repeated_intervention"),
                    "has_verify": row.get("has_verify"),
                    "has_switch": row.get("has_switch"),
                    "has_conf_drop": row.get("has_conf_drop"),
                    "has_trigger": row.get("has_trigger"),
                    "has_diagnosis": row.get("has_diagnosis"),
                    "has_decomposition": row.get("has_decomposition"),
                    "has_next_strategy": row.get("has_next_strategy"),
                }
            )
            print(answer[:1200].replace("\n", " "))
            print()


if __name__ == "__main__":
    main()
