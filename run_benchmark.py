"""EEE_Bench VLM Benchmark Runner — llama.cpp backends.

Runs vision-language model inference on the EEE_Bench dataset using
local GGUF models via ``llama-cpp-python`` or an already-running
OpenAI-compatible ``llama-server``.

Usage
-----
    python run_benchmark.py --model qwen3vl8b --split EEE_Bench/test_split.csv
    python run_benchmark.py --model qwen3vl4b --quant Q8_0 --max 50
    python run_benchmark.py --model qwen3vl8b --backend llama-server
"""

import argparse
import base64
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

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
from prompts.task_prompts import SYSTEM_PROMPT

BACKEND_CHOICES = ["llama-cpp-python", "llama-server"]
DEFAULT_BACKEND = "llama-cpp-python"
DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_N_BATCH = 1024
DEFAULT_N_UBATCH = 256


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


def _chat_messages(image_path: str, prompt: str) -> list[dict]:
    """Build OpenAI-style multimodal chat messages for both backends."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_uri(image_path)},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
    ]


def _extract_chat_content(response: dict) -> str:
    """Extract text from an OpenAI-compatible chat completion response."""
    choices = response.get("choices", [])
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content") or ""
    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()


def _record_key(row: pd.Series | dict) -> str:
    """Return a stable key for resume de-duplication."""
    image_id = str(row["image_id"])
    question = str(row["question"])
    return f"{image_id}::{question}"


def _load_existing_records(result_file: Path) -> list[dict]:
    """Load existing result records for resumable benchmarking."""
    if not result_file.exists():
        return []
    try:
        with result_file.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"  ⚠  Could not read existing results at {result_file}: {exc}. "
            "Starting a fresh result list.",
            file=sys.stderr,
        )
        return []
    if not isinstance(rows, list):
        print(
            f"  ⚠  Existing results at {result_file} are not a JSON list. "
            "Starting a fresh result list.",
            file=sys.stderr,
        )
        return []
    return [row for row in rows if isinstance(row, dict)]


def _write_results_atomic(result_file: Path, records: list[dict]) -> None:
    """Write result JSON via a temporary file, then replace the target."""
    tmp_file = result_file.with_suffix(result_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    tmp_file.replace(result_file)


class BenchmarkRunner(Protocol):
    """Common interface for benchmark inference backends."""

    backend_name: str

    def infer(self, image_path: str, prompt: str) -> str:
        """Run a single multimodal inference and return generated text."""


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
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        logits_all: bool = False,
        flash_attn: bool = True,
        n_batch: int = DEFAULT_N_BATCH,
        n_ubatch: int = DEFAULT_N_UBATCH,
    ):
        from llama_cpp import Llama  # noqa: E402

        spec = MODEL_REGISTRY[model_key]
        paths = resolve_model_paths(model_key, quant)

        self.model_key = model_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.backend_name = "llama-cpp-python"

        chat_handler = _get_chat_handler(spec["chat_handler"], paths["mmproj_path"])

        self.llm = Llama(
            model_path=str(paths["model_path"]),
            chat_handler=chat_handler,
            n_gpu_layers=-1,          # offload all layers to GPU
            n_ctx=spec.get("n_ctx", 8192),
            n_batch=n_batch,
            n_ubatch=n_ubatch,
            flash_attn=flash_attn,
            logits_all=logits_all,
            verbose=False,
        )

    def infer(self, image_path: str, prompt: str) -> str:
        """Run a single multimodal inference and return the generated text."""
        response = self.llm.create_chat_completion(
            messages=_chat_messages(image_path, prompt),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return _extract_chat_content(response)


class LlamaServerRunner:
    """Inference runner for an OpenAI-compatible llama-server endpoint."""

    def __init__(
        self,
        model_key: str,
        server_url: str = DEFAULT_SERVER_URL,
        server_model: str | None = None,
        api_key: str = "sk-no-key",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = 300.0,
    ):
        self.model_key = model_key
        self.model = server_model or model_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.backend_name = "llama-server"
        self.chat_url = server_url.rstrip("/") + "/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def infer(self, image_path: str, prompt: str) -> str:
        """Run a single multimodal inference through llama-server."""
        payload = {
            "model": self.model,
            "messages": _chat_messages(image_path, prompt),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.chat_url,
            data=data,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama-server returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach llama-server at {self.chat_url}: {exc}") from exc

        return _extract_chat_content(json.loads(body))


def _make_runner(
    backend: str,
    model_key: str,
    quant: str,
    max_tokens: int,
    temperature: float,
    logits_all: bool,
    flash_attn: bool,
    n_batch: int,
    n_ubatch: int,
    server_url: str,
    server_model: str | None,
    server_api_key: str,
    server_timeout: float,
) -> BenchmarkRunner:
    if backend == "llama-cpp-python":
        return LlamaCppRunner(
            model_key,
            quant=quant,
            max_tokens=max_tokens,
            temperature=temperature,
            logits_all=logits_all,
            flash_attn=flash_attn,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
        )
    if backend == "llama-server":
        return LlamaServerRunner(
            model_key,
            server_url=server_url,
            server_model=server_model,
            api_key=server_api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=server_timeout,
        )
    raise ValueError(f"Unknown backend: {backend}")


# ─── Benchmark loop ──────────────────────────────────────────────────

def run_benchmark(
    model_key: str,
    split_csv: str,
    output_dir: str,
    quant: str = DEFAULT_QUANT,
    max_samples: int | None = None,
    backend: str = DEFAULT_BACKEND,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    logits_all: bool = False,
    flash_attn: bool = True,
    n_batch: int = DEFAULT_N_BATCH,
    n_ubatch: int = DEFAULT_N_UBATCH,
    server_url: str = DEFAULT_SERVER_URL,
    server_model: str | None = None,
    server_api_key: str = "sk-no-key",
    server_timeout: float = 300.0,
    resume: bool = True,
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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = "" if backend == "llama-cpp-python" else "_server"
    result_file = output_path / f"{model_key}{suffix}_results.json"

    runner = _make_runner(
        backend=backend,
        model_key=model_key,
        quant=quant,
        max_tokens=max_tokens,
        temperature=temperature,
        logits_all=logits_all,
        flash_attn=flash_attn,
        n_batch=n_batch,
        n_ubatch=n_ubatch,
        server_url=server_url,
        server_model=server_model,
        server_api_key=server_api_key,
        server_timeout=server_timeout,
    )

    records = _load_existing_records(result_file) if resume else []
    completed_keys = {_record_key(row) for row in records if "image_id" in row and "question" in row}
    if completed_keys:
        print(f"Resuming {result_file}: {len(completed_keys)} completed rows found.")

    pending_df = df[~df.apply(_record_key, axis=1).isin(completed_keys)]
    if pending_df.empty:
        _write_results_atomic(result_file, records)
        print(f"No pending rows. Results already complete at {result_file}")
        return result_file

    for _, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc=model_key):
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
                "runner": runner.backend_name,
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
        completed_keys.add(_record_key(row))
        _write_results_atomic(result_file, records)

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
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=BACKEND_CHOICES,
        help=f"Inference backend (default: {DEFAULT_BACKEND}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum generated tokens per sample (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE}).",
    )
    parser.add_argument(
        "--logits-all",
        action="store_true",
        help="Enable logits_all for llama-cpp-python. Disabled by default to save memory.",
    )
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention for llama-cpp-python.",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=DEFAULT_N_BATCH,
        help=f"llama-cpp-python prompt batch size (default: {DEFAULT_N_BATCH}).",
    )
    parser.add_argument(
        "--n-ubatch",
        type=int,
        default=DEFAULT_N_UBATCH,
        help=f"llama-cpp-python physical micro-batch size (default: {DEFAULT_N_UBATCH}).",
    )
    parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=f"OpenAI-compatible llama-server base URL (default: {DEFAULT_SERVER_URL}).",
    )
    parser.add_argument(
        "--server-model",
        default=None,
        help="Model name sent to llama-server (default: registry model key).",
    )
    parser.add_argument(
        "--server-api-key",
        default="sk-no-key",
        help="Bearer token for llama-server if configured (default: sk-no-key).",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds for llama-server requests (default: 300).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing result JSON and start from scratch.",
    )
    args = parser.parse_args()

    run_benchmark(
        args.model,
        args.split,
        args.output,
        args.quant,
        args.max,
        backend=args.backend,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        logits_all=args.logits_all,
        flash_attn=not args.no_flash_attn,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        server_url=args.server_url,
        server_model=args.server_model,
        server_api_key=args.server_api_key,
        server_timeout=args.server_timeout,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
