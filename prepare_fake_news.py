from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", required=True, help="Path to Fake.csv")
    parser.add_argument("--true", required=True, help="Path to True.csv")
    parser.add_argument("--output", default="data/fake_news.csv")
    parser.add_argument(
        "--text-column",
        default="text",
        help="Article text column present in both files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fake = pd.read_csv(args.fake)
    true = pd.read_csv(args.true)

    for name, df in (("Fake.csv", fake), ("True.csv", true)):
        if args.text_column not in df.columns:
            raise ValueError(
                f"{name} does not contain column '{args.text_column}'. "
                f"Available columns: {list(df.columns)}"
            )

    fake_out = pd.DataFrame(
        {
            "text": fake[args.text_column].astype(str),
            "label": 1,
        }
    )
    true_out = pd.DataFrame(
        {
            "text": true[args.text_column].astype(str),
            "label": 0,
        }
    )

    combined = pd.concat([fake_out, true_out], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    print(f"Saved {len(combined):,} rows to {output}")
    print("Label convention: 0 = real, 1 = fake")


if __name__ == "__main__":
    main()
