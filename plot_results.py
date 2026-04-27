from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    comparison_file = Path("results/model_comparison.csv")
    df = pd.read_csv(comparison_file)
    df = df.sort_values("final_answer_accuracy", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.barplot(
        data=df,
        y="model",
        x="final_answer_accuracy",
        ax=axes[0],
        palette="crest",
    )
    axes[0].set_title("Final Answer Accuracy by Model")
    axes[0].set_xlabel("Final Answer Accuracy (%)")
    axes[0].set_ylabel("Model")

    sns.scatterplot(
        data=df,
        x="avg_latency_s",
        y="final_answer_accuracy",
        hue="model",
        s=140,
        ax=axes[1],
    )
    axes[1].set_title("Accuracy vs Latency")
    axes[1].set_xlabel("Average Latency per Sample (s)")
    axes[1].set_ylabel("Final Answer Accuracy (%)")

    plt.tight_layout()
    out_file = Path("results/model_comparison.png")
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to {out_file}")


if __name__ == "__main__":
    main()
