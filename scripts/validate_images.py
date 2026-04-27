from pathlib import Path

import pandas as pd
from PIL import Image


INDEX_FILE = Path("EEE_Bench/benchmark_index.csv")


def main() -> None:
    df = pd.read_csv(INDEX_FILE)
    bad = []

    for _, row in df.iterrows():
        try:
            with Image.open(row["abs_image_path"]) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001
            bad.append((row["abs_image_path"], str(exc)))

    print(f"Validated images: {len(df)}")
    print(f"Corrupt or missing images: {len(bad)}")
    for path, err in bad[:50]:
        print(f"{path}: {err}")


if __name__ == "__main__":
    main()
