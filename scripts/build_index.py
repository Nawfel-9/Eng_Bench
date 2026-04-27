import json
from pathlib import Path

import pandas as pd

try:
    from scripts.benchmark_utils import infer_answer_format
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from scripts.benchmark_utils import infer_answer_format


DATA_ROOT = Path("EEE_Bench")
IMAGE_DIR = DATA_ROOT / "images"
QA_FILE = DATA_ROOT / "qa" / "eee_bench_qa.json"
OUT_FILE = DATA_ROOT / "benchmark_index.csv"


def main() -> None:
    items = json.loads(QA_FILE.read_text(encoding="utf-8"))
    records = []
    missing_images = 0

    for item in items:
        image_path = IMAGE_DIR / item["image"]
        if not image_path.exists():
            missing_images += 1
            continue

        records.append(
            {
                "image_id": item["id"],
                "raw_split": item.get("split", ""),
                "image_path": item["image"],
                "abs_image_path": str(image_path.resolve()),
                "question": item["question"],
                "answer": str(item["answer"]).strip(),
                "answer_format": infer_answer_format(item["question"]),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(OUT_FILE, index=False)

    print(f"Indexed rows: {len(df)}")
    print(f"Missing images skipped: {missing_images}")
    print(df["answer_format"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
