import argparse
import json
import re
from pathlib import Path

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from scripts.benchmark_utils import extract_final_answer, normalize_text


def exact_match(pred: str, gt: str) -> int:
    return int(normalize_text(pred) == normalize_text(gt))


def contains_match(pred: str, gt: str) -> int:
    gt_norm = normalize_text(gt)
    return int(bool(gt_norm) and gt_norm in normalize_text(pred))


def token_f1(pred: str, gt: str) -> float:
    pred_tokens = set(normalize_text(pred).split())
    gt_tokens = set(normalize_text(gt).split())
    if not pred_tokens or not gt_tokens:
        return 0.0

    overlap = len(pred_tokens & gt_tokens)
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu(pred: str, gt: str) -> float:
    reference = [normalize_text(gt).split()]
    hypothesis = normalize_text(pred).split()
    if not hypothesis or not reference[0]:
        return 0.0
    return sentence_bleu(reference, hypothesis, smoothing_function=SmoothingFunction().method4)


def compute_rouge_l(pred: str, gt: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(normalize_text(gt), normalize_text(pred))["rougeL"].fmeasure


def final_answer_match(pred_final: str, gt: str, answer_format: str) -> int:
    gt_norm = str(gt).strip()
    pred_norm = str(pred_final).strip()
    return int(pred_norm == gt_norm)


def summarize(df: pd.DataFrame) -> dict:
    return {
        "EM": round(df["EM"].mean() * 100, 2),
        "Contains": round(df["Contains"].mean() * 100, 2),
        "F1": round(df["F1"].mean() * 100, 2),
        "BLEU": round(df["BLEU"].mean() * 100, 2),
        "ROUGE-L": round(df["ROUGE-L"].mean() * 100, 2),
        "final_answer_accuracy": round(df["final_answer_accuracy"].mean() * 100, 2),
        "avg_latency_s": round(df["latency_s"].mean(), 3),
        "error_rate": round(df["is_error"].mean() * 100, 2),
        "N": int(len(df)),
    }


def evaluate(results_file: str) -> pd.DataFrame:
    with open(results_file, encoding="utf-8") as handle:
        rows = json.load(handle)

    scored_rows = []
    for row in rows:
        prediction = row.get("prediction", "")
        gt_answer = str(row.get("gt_answer", "")).strip()
        answer_format = row.get("answer_format", "other")
        final_answer = row.get("final_answer") or extract_final_answer(prediction, answer_format)
        is_error = int(str(prediction).startswith("ERROR:"))

        if is_error:
            scored_rows.append(
                {
                    **row,
                    "final_answer": final_answer,
                    "EM": 0,
                    "Contains": 0,
                    "F1": 0.0,
                    "BLEU": 0.0,
                    "ROUGE-L": 0.0,
                    "final_answer_accuracy": 0,
                    "is_error": 1,
                }
            )
            continue

        scored_rows.append(
            {
                **row,
                "final_answer": final_answer,
                "EM": exact_match(prediction, gt_answer),
                "Contains": contains_match(prediction, gt_answer),
                "F1": token_f1(prediction, gt_answer),
                "BLEU": compute_bleu(prediction, gt_answer),
                "ROUGE-L": compute_rouge_l(prediction, gt_answer),
                "final_answer_accuracy": final_answer_match(final_answer, gt_answer, answer_format),
                "is_error": 0,
            }
        )

    df = pd.DataFrame(scored_rows)
    summary = summarize(df)

    print(f"\n{'=' * 70}")
    print(f"Scored results: {Path(results_file).stem}")
    print(f"{'=' * 70}")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if "answer_format" in df.columns:
        format_summary = (
            df.groupby("answer_format")[["final_answer_accuracy", "EM", "F1"]]
            .mean()
            .mul(100)
            .round(2)
        )
        print("\nBreakdown by answer_format:")
        print(format_summary.to_string())

    out_file = re.sub(r"\.json$", "_scored.csv", results_file)
    df.to_csv(out_file, index=False)
    print(f"\nSaved scored CSV to {out_file}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    evaluate(args.results)


if __name__ == "__main__":
    main()
