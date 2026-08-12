"""Ad performance: CTR & CPC analysis.

Computes click-through rate and cost per click from a campaign CSV,
guarding against divide-by-zero rows (zero impressions or zero clicks).

Run with: python ads_performance.py --data dataset.csv --out processed_dataset.csv
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = ["clicks", "impressions", "cost"]


def add_ctr_cpc(df: pd.DataFrame) -> pd.DataFrame:
    """Add CTR (%) and CPC columns; rows with zero denominators get NaN."""
    df = df.copy()
    df["CTR"] = np.where(
        df["impressions"] > 0, df["clicks"] / df["impressions"] * 100, np.nan
    )
    df["CPC"] = np.where(df["clicks"] > 0, df["cost"] / df["clicks"], np.nan)
    return df


def main():
    parser = argparse.ArgumentParser(description="Ad performance: CTR & CPC analysis")
    parser.add_argument("--data", default="dataset.csv",
                        help="Path to a CSV with columns: clicks, impressions, cost")
    parser.add_argument("--out", default="processed_dataset.csv",
                        help="Path to write the processed CSV with CTR/CPC columns")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    print(df.head())
    print(df.describe())

    df = add_ctr_cpc(df)

    plt.figure(figsize=(10, 5))
    sns.histplot(df["CTR"].dropna(), bins=30, kde=True)
    plt.title("Distribution of Click-Through Rate (CTR)")
    plt.xlabel("CTR (%)")
    plt.ylabel("Frequency")
    plt.show()

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
