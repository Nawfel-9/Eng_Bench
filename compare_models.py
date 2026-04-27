from pathlib import Path

import pandas as pd

from evaluate_results import evaluate


def main() -> None:
    results_dir = Path("results")
    summaries = []

    for result_file in sorted(results_dir.glob("*_results.json")):
        df = evaluate(str(result_file))
        summaries.append(
            {
                "model": result_file.stem.replace("_results", ""),
                "final_answer_accuracy": round(df["final_answer_accuracy"].mean() * 100, 2),
                "EM": round(df["EM"].mean() * 100, 2),
                "Contains": round(df["Contains"].mean() * 100, 2),
                "F1": round(df["F1"].mean() * 100, 2),
                "BLEU": round(df["BLEU"].mean() * 100, 2),
                "ROUGE-L": round(df["ROUGE-L"].mean() * 100, 2),
                "avg_latency_s": round(df["latency_s"].mean(), 3),
                "error_rate": round(df["is_error"].mean() * 100, 2),
                "N": int(len(df)),
            }
        )

    if not summaries:
        raise SystemExit("No result JSON files found in results/")

    comparison = pd.DataFrame(summaries).sort_values(
        by=["final_answer_accuracy", "F1"],
        ascending=[False, False],
    )
    out_file = results_dir / "model_comparison.csv"
    comparison.to_csv(out_file, index=False)
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison CSV to {out_file}")


if __name__ == "__main__":
    main()
