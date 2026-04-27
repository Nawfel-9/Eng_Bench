"""EEE_Bench VLM Benchmark Runner — llama.cpp backend.

Runs vision-language model inference on the EEE_Bench dataset using
local GGUF models via ``llama-cpp-python``.  Every model in the
registry is loaded through the same ``LlamaCppRunner`` class.

Usage
-----
    python run_benchmark.py --model qwen25vl --split EEE_Bench/test_split.csv
    python run_benchmark.py --model phi35v --quant Q8_0 --max 50
"""

import argparse
import base64
import json
import mimetypes
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from models.registry import (
    BENCHMARK_ORDER,
    DEFAULT_QUANT,
    MODEL_REGISTRY,
    QUANT_OPTIONS,
    resolve_model_paths,
)
from scripts.benchmark_utils import build_prompt, extract_final_answer


# ─── Helpers ──────────────────────────────────────────────────────────

def _image_to_data_uri(image_path: str) -> str:
    """Read a local image file and return a ``data:`` URI with base64 payload."""
    path = Path(image_path)
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _get_chat_handler(handler_name: str, mmproj_path: Path):
    """Dynamically import and instantiate a llama-cpp-python chat handler."""
    import llama_cpp.llama_chat_format as fmt  # type: ignore # noqa: E402

    handler_cls = getattr(fmt, handler_name, None)
    if handler_cls is None:
        # Fallback: use the generic LLaVA-1.5 handler which works for most
        # vision architectures that accept the standard image_url input.
        handler_cls = getattr(fmt, "Llava15ChatHandler")
        print(
            f"  ⚠  Chat handler '{handler_name}' not found in llama-cpp-python, "
            f"falling back to Llava15ChatHandler.",
            file=sys.stderr,
        )
    return handler_cls(clip_model_path=str(mmproj_path))


# ─── Runner ───────────────────────────────────────────────────────────

class LlamaCppRunner:
    """Unified inference runner backed by llama-cpp-python.

    Parameters
    ----------
    model_key : str
        Key from :data:`MODEL_REGISTRY`.
    quant : str
        Quantisation level (``Q4_K_M``, ``Q5_K_M``, ``Q8_0``).
    max_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Sampling temperature.  ``0.0`` for greedy decoding.
    """

    def __init__(
        self,
        model_key: str,
        quant: str = DEFAULT_QUANT,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        from llama_cpp import Llama  # noqa: E402

        spec = MODEL_REGISTRY[model_key]
        paths = resolve_model_paths(model_key, quant)

        self.model_key = model_key
        self.max_tokens = max_tokens
        self.temperature = temperature

        chat_handler = _get_chat_handler(spec["chat_handler"], paths["mmproj_path"])

        self.llm = Llama(
            model_path=str(paths["model_path"]),
            chat_handler=chat_handler,
            n_gpu_layers=-1,          # offload all layers to GPU
            n_ctx=spec.get("n_ctx", 8192),
            logits_all=True,
            verbose=False,
        )

    def infer(self, image_path: str, prompt: str) -> str:
        """Run a single multimodal inference and return the generated text."""
        data_uri = _image_to_data_uri(image_path)

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri},
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        choices = response.get("choices", [])
        if not choices:
            return ""
        return (choices[0].get("message", {}).get("content") or "").strip()


# ─── Benchmark loop ──────────────────────────────────────────────────

def run_benchmark(
    model_key: str,
    split_csv: str,
    output_dir: str,
    quant: str = DEFAULT_QUANT,
    max_samples: int | None = None,
) -> Path:
    """Run the full benchmark for a single model and save results to JSON.

    Parameters
    ----------
    model_key : str
        Key from :data:`MODEL_REGISTRY`.
    split_csv : str
        Path to the CSV split file (``dev_split.csv`` or ``test_split.csv``).
    output_dir : str
        Directory where result JSON files are written.
    quant : str
        GGUF quantisation level.
    max_samples : int or None
        Cap on number of samples (for debugging / smoke tests).

    Returns
    -------
    pathlib.Path
        Path to the written JSON results file.
    """
    spec = MODEL_REGISTRY[model_key]
    df = pd.read_csv(split_csv)
    if max_samples:
        df = df.sample(min(max_samples, len(df)), random_state=42)

    runner = LlamaCppRunner(model_key, quant=quant)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{model_key}_results.json"

    records: list[dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=model_key):
        prompt = build_prompt(row["question"], row["answer_format"])
        start = time.perf_counter()
        error = ""
        try:
            prediction = runner.infer(row["abs_image_path"], prompt)
        except Exception as exc:  # noqa: BLE001
            prediction = f"ERROR: {exc}"
            error = str(exc)
        latency = round(time.perf_counter() - start, 3)
        final_answer = extract_final_answer(prediction, row["answer_format"])

        records.append(
            {
                "model_key": model_key,
                "model_id": spec["display_name"],
                "runner": "llama.cpp",
                "quant": quant,
                "image_id": row["image_id"],
                "raw_split": row["raw_split"],
                "image_path": row["image_path"],
                "abs_image_path": row["abs_image_path"],
                "question": row["question"],
                "answer_format": row["answer_format"],
                "gt_answer": str(row["answer"]).strip(),
                "prediction": prediction,
                "final_answer": final_answer,
                "latency_s": latency,
                "error": error,
            }
        )

    with result_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} rows to {result_file}")
    return result_file


# ─── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the EEE_Bench VLM benchmark using local GGUF models."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_REGISTRY),
        help="Model key from the registry.",
    )
    parser.add_argument(
        "--split",
        default="EEE_Bench/test_split.csv",
        help="Path to the CSV split file (default: test_split.csv).",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for result JSON files.",
    )
    parser.add_argument(
        "--quant",
        default=DEFAULT_QUANT,
        choices=QUANT_OPTIONS,
        help=f"GGUF quantisation level (default: {DEFAULT_QUANT}).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for debugging).",
    )
    args = parser.parse_args()

    run_benchmark(args.model, args.split, args.output, args.quant, args.max)


if __name__ == "__main__":
    main()
