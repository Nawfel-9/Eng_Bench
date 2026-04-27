from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INDEX_FILE = Path("EEE_Bench/benchmark_index.csv")
DEV_FILE = Path("EEE_Bench/dev_split.csv")
TEST_FILE = Path("EEE_Bench/test_split.csv")


def main() -> None:
    df = pd.read_csv(INDEX_FILE)
    counts = df["answer_format"].value_counts()
    rare_formats = counts[counts < 2].index.tolist()

    rare_df = df[df["answer_format"].isin(rare_formats)].copy()
    regular_df = df[~df["answer_format"].isin(rare_formats)].copy()

    dev, test = train_test_split(
        regular_df,
        test_size=0.8,
        random_state=42,
        stratify=regular_df["answer_format"],
    )

    if not rare_df.empty:
        # Keep rare answer formats in the final test set so full-benchmark coverage is preserved.
        test = pd.concat([test, rare_df], ignore_index=True)

    dev.to_csv(DEV_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    print(f"Dev rows: {len(dev)}")
    print(f"Test rows: {len(test)}")
    if rare_formats:
        print(f"Rare formats assigned directly to test: {rare_formats}")
    print("\nDev distribution:")
    print((dev["answer_format"].value_counts(normalize=True) * 100).round(2).sort_index().to_string())
    print("\nTest distribution:")
    print((test["answer_format"].value_counts(normalize=True) * 100).round(2).sort_index().to_string())


if __name__ == "__main__":
    main()
