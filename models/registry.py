r"""Model registry — maps benchmark keys to local GGUF model files.

All models are stored under  F:\Models\LLAMA\<model_dir>\  and loaded
via llama-cpp-python.  Each entry specifies the GGUF filename pattern
(with a ``{quant}`` placeholder), the multimodal-projector filename,
the llama-cpp-python chat-handler class to use, and the HuggingFace
repo the files were downloaded from.
"""

from pathlib import Path

# ── Root directory for all GGUF model files ──────────────────────────
MODELS_ROOT = Path(r"F:\Models\LLAMA")

# ── Canonical benchmark evaluation order ─────────────────────────────
BENCHMARK_ORDER = [
    "phi35v",
    "gemma4e4b",
    "minicpmv",
    "qwen25vl",
    "internvl3",
    "glm4v",
    "llama32v",
    "pixtral",
]

# ── Available quantisation levels ────────────────────────────────────
QUANT_OPTIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]
DEFAULT_QUANT = "Q8_0"  # Best quality that fits within 16 GB VRAM for most models

# ── Model definitions ────────────────────────────────────────────────
MODEL_REGISTRY = {
    "qwen25vl": {
        "display_name": "Qwen2.5-VL 7B Instruct",
        "model_dir": "Qwen2.5-VL-7B-Instruct",
        "gguf_file": "Qwen2.5-VL-7B-Instruct-{quant}.gguf",
        "mmproj_file": "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
        "chat_handler": "Qwen25VLChatHandler",
        "default_quant": "Q8_0",   # 7B: ~7.5 GB + ~1.5 GB mmproj ≈ 9 GB — fits 16 GB
        "n_ctx": 8192,
        "hf_repo": "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
    },
    "llama32v": {
        "display_name": "Llama 3.2 Vision 11B",
        "model_dir": "Llama-3.2-11B-Vision-Instruct",
        "gguf_file": "Llama-3.2-11B-Vision-Instruct-{quant}.gguf",
        "mmproj_file": "mmproj-Llama-3.2-11B-Vision-Instruct-f16.gguf",
        "chat_handler": "Llava15ChatHandler",
        "default_quant": "Q5_K_M", # 11B: Q8 ~11.5 GB too tight; Q5 ~7.6 GB + ~1.5 GB mmproj ≈ 9 GB
        "n_ctx": 8192,
        "hf_repo": "ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF",
    },
    "minicpmv": {
        "display_name": "MiniCPM-V 2.6 8B",
        "model_dir": "MiniCPM-V-2_6",
        "gguf_file": "MiniCPM-V-2_6-{quant}.gguf",
        "mmproj_file": "mmproj-MiniCPM-V-2_6-f16.gguf",
        "chat_handler": "MiniCPMv26ChatHandler",
        "default_quant": "Q8_0",   # 8B: ~8.5 GB + ~1.5 GB mmproj ≈ 10 GB — fits 16 GB
        "n_ctx": 8192,
        "hf_repo": "openbmb/MiniCPM-V-2_6-gguf",
    },
    "gemma4e4b": {
        "display_name": "Gemma 4 E4B",
        "model_dir": "gemma-4-E4B-it",
        "gguf_file": "gemma-4-E4B-it-{quant}.gguf",
        "mmproj_file": "mmproj-gemma-4-E4B-it-bf16.gguf",
        "chat_handler": "Gemma4ChatHandler",
        "default_quant": "Q8_0",   # ~4B eff: ~4.4 GB + ~1.5 GB mmproj ≈ 6 GB — fits 16 GB
        "n_ctx": 8192,
        "hf_repo": "ggml-org/gemma-4-E4B-it-GGUF",
    },
    "internvl3": {
        "display_name": "InternVL3 8B Instruct",
        "model_dir": "InternVL3-8B-Instruct",
        "gguf_file": "InternVL3-8B-Instruct-{quant}.gguf",
        "mmproj_file": "mmproj-InternVL3-8B-Instruct-f16.gguf",
        "chat_handler": "Llava15ChatHandler",
        "default_quant": "Q8_0",   # 8B: ~8.5 GB + ~1.5 GB mmproj ≈ 10 GB — fits 16 GB
        "n_ctx": 8192,
        "hf_repo": "ggml-org/InternVL3-8B-Instruct-GGUF",
    },
    "glm4v": {
        "display_name": "GLM-4.1V-9B-Thinking",
        "model_dir": "GLM-4.1V-9B-Thinking",
        "gguf_file": "GLM-4.1V-9B-Thinking-{quant}.gguf",
        "mmproj_file": "mmproj-GLM-4.1V-9B-Thinking-f16.gguf",
        "chat_handler": "Llava15ChatHandler",
        "default_quant": "Q8_0",   # 9B: ~9.5 GB + ~1.5 GB mmproj ≈ 11 GB — fits 16 GB
        "n_ctx": 8192,
        "hf_repo": "mradermacher/GLM-4.1V-9B-Thinking-GGUF",
    },
    "phi35v": {
        "display_name": "Phi-3.5-Vision-Instruct",
        "model_dir": "Phi-3.5-vision-instruct",
        "gguf_file": "Phi-3.5-vision-instruct-{quant}.gguf",
        "mmproj_file": "mmproj-Phi-3.5-vision-instruct-f16.gguf",
        "chat_handler": "Llava15ChatHandler",
        "default_quant": "Q8_0",   # 3.8B: ~4.1 GB + ~1 GB mmproj ≈ 5 GB — fits 16 GB
        "n_ctx": 8192,
        "hf_repo": "abetlen/Phi-3.5-vision-instruct-gguf",
    },
    "pixtral": {
        "display_name": "Pixtral 12B",
        "model_dir": "pixtral-12b",
        "gguf_file": "pixtral-12b-{quant}.gguf",
        "mmproj_file": "mmproj-pixtral-12b-f16.gguf",
        "chat_handler": "Llava15ChatHandler",
        "default_quant": "Q5_K_M", # 12B: Q8 ~12.5 GB too tight; Q5 ~8.3 GB + ~1.5 GB mmproj ≈ 10 GB
        "n_ctx": 8192,
        "hf_repo": "ggml-org/pixtral-12b-GGUF",
    },
}


def resolve_model_paths(model_key: str, quant: str | None = None) -> dict:
    """Return a dict with resolved ``model_path`` and ``mmproj_path``.

    Parameters
    ----------
    model_key:
        One of the keys in :data:`MODEL_REGISTRY`.
    quant:
        Quantisation level (``Q4_K_M``, ``Q5_K_M``, ``Q8_0``).
        Falls back to the model's ``default_quant`` if *None*.

    Returns
    -------
    dict
        ``{"model_path": Path, "mmproj_path": Path}``

    Raises
    ------
    KeyError
        If *model_key* is not in the registry.
    FileNotFoundError
        If either resolved path does not exist on disk.
    """
    spec = MODEL_REGISTRY[model_key]
    quant = quant or spec["default_quant"]
    base = MODELS_ROOT / spec["model_dir"]

    model_path = base / spec["gguf_file"].format(quant=quant)
    mmproj_path = base / spec["mmproj_file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"GGUF model file not found: {model_path}\n"
            f"Run the download script first:\n"
            f"  hf download {spec['hf_repo']} --local-dir {base}"
        )
    if not mmproj_path.exists():
        raise FileNotFoundError(
            f"Multimodal projector not found: {mmproj_path}\n"
            f"Run the download script first:\n"
            f"  hf download {spec['hf_repo']} --local-dir {base}"
        )

    return {"model_path": model_path, "mmproj_path": mmproj_path}
