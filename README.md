# EEE_Bench — Vision-Language Model Benchmark for Technical Document Analysis

A benchmarking framework for evaluating **local vision-language models (VLMs)** on the **EEE_Bench** dataset — a collection of electrical and electronics engineering technical diagrams paired with question-answer items.  Inference runs through **llama.cpp** either in-process via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or over HTTP via an OpenAI-compatible `llama-server`, using locally stored GGUF model files.

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Repository Structure](#repository-structure)
4. [Supported Models](#supported-models)
5. [Environment Setup](#environment-setup)
6. [Downloading Models](#downloading-models)
7. [Dataset Preparation](#dataset-preparation)
8. [Running the Benchmark](#running-the-benchmark)
9. [Evaluating Results](#evaluating-results)
10. [Comparing Models](#comparing-models)
11. [Plotting Results](#plotting-results)
12. [Prompt Construction Pipeline](#prompt-construction-pipeline)
13. [Configuration Reference](#configuration-reference)

---

## Overview

The benchmark evaluates how well different VLMs can understand and reason about technical engineering diagrams.  Each sample consists of an image (circuit schematic, block diagram, waveform, etc.) and a structured question.  The system:

1. Infers the expected **answer format** from the question text (multiple-choice, float, integer, array, or free-form).
2. Wraps the question in a **controlled prompt** with an expert-role instruction and a strict final-line output rule.
3. Sends the image + prompt to a **local GGUF model** via `llama-cpp-python` or an OpenAI-compatible `llama-server`.
4. Extracts a **clean final answer** from the model's output using deterministic parsing.
5. Scores the answer against the ground truth using multiple metrics (exact match, contains, token F1, BLEU, ROUGE-L, and the primary **final-answer accuracy**).

---

## Hardware Requirements

| Component       | Minimum         | Recommended      |
|-----------------|-----------------|------------------|
| **GPU**         | 8 GB VRAM       | 16 GB VRAM       |
| **RAM**         | 32 GB           | 64 GB            |
| **Storage**     | 50 GB free      | 200 GB free      |
| **OS**          | Windows 10/11   | Windows 11       |
| **Python**      | 3.10            | 3.11+            |

> **Note**: All 7 supported models at their default quantisation fit within 16 GB VRAM.

---

## Repository Structure

```
Prototype/
├── run_benchmark.py            # Main entry point — runs inference for one model
├── evaluate_results.py         # Scores a results JSON against ground truth
├── compare_models.py           # Aggregates & ranks all model results
├── plot_results.py             # Generates accuracy/latency charts
│
├── models/
│   └── registry.py             # Model registry — GGUF paths, handlers, metadata
│
├── prompts/
│   └── task_prompts.py         # BASE_PROMPT + format-specific suffixes
│
├── scripts/
│   ├── benchmark_utils.py      # Shared helpers: prompt builder, answer parser, normaliser
│   ├── build_index.py          # One-time: indexes QA JSON → benchmark_index.csv
│   ├── split_dataset.py        # One-time: stratified dev/test split
│   ├── validate_images.py      # Checks image integrity
│   ├── download_models.py      # Generates huggingface-cli download commands
│   └── download_all_models.ps1 # PowerShell script to download all GGUF models
│
├── EEE_Bench/
│   ├── images/                 # Technical diagram images
│   ├── qa/                     # Raw QA JSON files
│   ├── benchmark_index.csv     # Full indexed dataset
│   ├── dev_split.csv           # 20% development split
│   └── test_split.csv          # 80% test split
│
├── results/                    # Per-model result JSONs and scored CSVs
│
├── LLM_Question_Pipeline.md   # Detailed documentation of prompt construction
├── VLM_Benchmark_Guide.md     # Full technical guide
└── README.md                  # This file
```

### External: Model Storage

All GGUF model files are stored at:

```
F:\Models\LLAMA\
├── Qwen3-VL-4B-Instruct/
├── Qwen3-VL-8B-Instruct/
├── Llama-3.2-11B-Vision-Instruct/
├── MiniCPM-V-2_6/
├── gemma-4-E4B-it/
├── InternVL3_5-8B/
└── pixtral-12b/
```

Each model directory contains:
- One or more **quantised GGUF** files (Q4\_K\_M, Q5\_K\_M, Q8\_0)
- A **multimodal projector** file (`mmproj-*-f16.gguf`)

---

## Supported Models

All models can run through either `llama-cpp-python` or an OpenAI-compatible `llama-server`. The default quantization levels shown below are optimized for a **16 GB VRAM GPU** by choosing the highest published GGUF quant that leaves comfortable headroom for the model, multimodal projector, and runtime overhead.

| Key         | Display Name              | Parameters | Default Quant | Est. GGUF + mmproj | Chat Handler             | HuggingFace GGUF Repo                                          |
|-------------|---------------------------|------------|---------------|--------------------|--------------------------|------------------------------------------------------------------|
| `gemma4e4b` | Gemma 4 E4B               | ~4 B eff.  | Q8\_0         | ~8.40 GiB          | Gemma4ChatHandler        | `ggml-org/gemma-4-E4B-it-GGUF`                                  |
| `minicpmv`  | MiniCPM-V 2.6 8B          | 8 B        | Q8\_0         | ~8.51 GiB          | MiniCPMv26ChatHandler    | `openbmb/MiniCPM-V-2_6-gguf`                                    |
| `qwen3vl8b` | Qwen3-VL 8B Instruct      | 8 B        | Q8\_0         | ~9.19 GiB          | Qwen3VLChatHandler       | `Qwen/Qwen3-VL-8B-Instruct-GGUF`                                |
| `internvl35`| InternVL3.5 8B Instruct   | 8.5 B      | Q8\_0         | ~8.74 GiB          | Llava15ChatHandler       | `lmstudio-community/InternVL3_5-8B-GGUF`                        |
| `llama32v`  | Llama 3.2 Vision 11B      | 11 B       | Q8\_0         | ~11.49 GiB         | Llava15ChatHandler       | `leafspark/Llama-3.2-11B-Vision-Instruct-GGUF`                  |
| `pixtral`   | Pixtral 12B               | 12 B       | Q8\_0         | ~12.94 GiB         | Llava15ChatHandler       | `ggml-org/pixtral-12b-GGUF`                                     |

### Quantisation Options

| Level     | Description                | Typical Size (7B) | Quality    |
|-----------|----------------------------|--------------------|------------|
| `Q4_K_M`  | 4-bit, medium quality      | ~4.4 GB            | Good       |
| `Q5_K_M`  | 5-bit, medium quality      | ~5.1 GB            | Better     |
| `Q8_0`    | 8-bit                      | ~7.5 GB            | Near-lossless |

---

## Environment Setup

### 1. Create a Python environment

```bash
conda create -n vlm_bench python=3.11 -y
conda activate vlm_bench
```

### 2. Install llama-cpp-python with CUDA support

```bash
# For NVIDIA GPUs (CUDA 12.x):
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# On Windows (PowerShell):
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python
```

### 3. Install remaining dependencies

```bash
pip install pandas tqdm pillow scikit-learn matplotlib seaborn rouge-score nltk
```

---

## Downloading Models

GGUF model files are downloaded from HuggingFace and stored at `F:\Models\LLAMA\`.

> **Note:** If you want to download models to a different directory, you must update the `MODELS_ROOT` constant in `models/registry.py` before running the download script or the benchmark.

### Option A: Run the master download script

```powershell
.\scripts\download_all_models.ps1
```

This downloads all supported models at their published quantisation levels plus their multimodal projectors.  Comment out any models or quants you don't need.

### Option B: Download a specific model

```bash
# Example: download only Qwen3-VL 8B at Q8_0
hf download Qwen/Qwen3-VL-8B-Instruct-GGUF \
    Qwen3VL-8B-Instruct-Q8_0.gguf \
    mmproj-Qwen3VL-8B-Instruct-F16.gguf \
    --local-dir "F:\Models\LLAMA\Qwen3-VL-8B-Instruct"
```

### Option C: Generate commands programmatically

```bash
python scripts/download_models.py             # print to stdout
python scripts/download_models.py --output scripts/download_all_models.ps1  # write .ps1
```

---

## Dataset Preparation

**FOR NOW**, the benchmark uses data hosted on Hugging Face at [https://huggingface.co/datasets/afdsafas/EEE-Bench](https://huggingface.co/datasets/afdsafas/EEE-Bench). You can download it by running:

```bash
python scripts/download_EEE_Bench.py
```

These one-time scripts prepare the EEE\_Bench dataset for benchmarking.

### 1. Build the unified index

Reads `EEE_Bench/qa/eee_bench_qa.json`, resolves image paths, infers answer formats, and writes `EEE_Bench/benchmark_index.csv`.

```bash
python scripts/build_index.py
```

### 2. Validate images

Checks that every referenced image file exists and can be opened.

```bash
python scripts/validate_images.py
```

### 3. Create dev/test splits

Produces stratified `dev_split.csv` (20%) and `test_split.csv` (80%).

```bash
python scripts/split_dataset.py
```

---

## Running the Benchmark

### Basic usage

```bash
python run_benchmark.py --model qwen3vl8b --split EEE_Bench/test_split.csv
```

### OpenAI-compatible llama-server backend

Start `llama-server` separately with the same GGUF model and multimodal projector, then point the benchmark at its OpenAI-compatible `/v1/chat/completions` endpoint:

```powershell
llama-server `
    -m "F:\Models\LLAMA\Qwen3-VL-8B-Instruct\Qwen3VL-8B-Instruct-Q8_0.gguf" `
    --mmproj "F:\Models\LLAMA\Qwen3-VL-8B-Instruct\mmproj-Qwen3VL-8B-Instruct-F16.gguf" `
    -ngl -1 -c 8192 --flash-attn --batch-size 1024 --ubatch-size 256 `
    --host 127.0.0.1 --port 8080

python run_benchmark.py --model qwen3vl8b --backend llama-server --server-url http://127.0.0.1:8080/v1
```

### All CLI options

| Argument   | Default                      | Description                                      |
|------------|------------------------------|--------------------------------------------------|
| `--model`  | *(required)*                 | Model key from the registry                      |
| `--split`  | `EEE_Bench/test_split.csv`   | Path to the CSV split file                       |
| `--output` | `results`                    | Output directory for result JSONs                |
| `--quant`  | `Q8_0`                       | Quantisation level: `Q4_K_M`, `Q5_K_M`, `Q8_0`  |
| `--max`    | *(all samples)*              | Limit sample count (for debugging / smoke tests) |
| `--backend`| `llama-cpp-python`           | `llama-cpp-python` or `llama-server`             |
| `--max-tokens` | `1024`                   | Maximum generated tokens per sample              |
| `--temperature` | `0.0`                  | Sampling temperature                             |
| `--logits-all` | `false`                 | Enable `logits_all`; disabled by default to save memory |
| `--no-flash-attn` | `false`              | Disable flash attention for `llama-cpp-python`   |
| `--n-batch` | `1024`                    | Prompt batch size for `llama-cpp-python`         |
| `--n-ubatch` | `256`                   | Physical micro-batch size for `llama-cpp-python` |
| `--server-url` | `http://127.0.0.1:8080/v1` | OpenAI-compatible llama-server base URL      |
| `--server-model` | *(model key)*         | Model name sent to llama-server                  |
| `--server-api-key` | `sk-no-key`         | Bearer token if llama-server requires one        |
| `--server-timeout` | `300`               | HTTP timeout in seconds                          |
| `--no-resume` | `false`                  | Ignore existing JSON and start from scratch      |

By default, benchmark runs are resumable: existing rows in `results/<model>_results.json` are skipped, and progress is written after each sample. The `llama-server` backend writes to `results/<model>_server_results.json` so local and HTTP runs do not overwrite each other.

### Smoke test on the dev set

```bash
python run_benchmark.py --model qwen3vl8b --split EEE_Bench/dev_split.csv --max 20
```

### Full benchmark run (all models, lightest → heaviest)

```bash
python run_benchmark.py --model gemma4e4b --split EEE_Bench/test_split.csv
python run_benchmark.py --model minicpmv  --split EEE_Bench/test_split.csv
python run_benchmark.py --model qwen3vl8b --split EEE_Bench/test_split.csv
python run_benchmark.py --model internvl35 --split EEE_Bench/test_split.csv
python run_benchmark.py --model llama32v  --split EEE_Bench/test_split.csv
python run_benchmark.py --model pixtral   --split EEE_Bench/test_split.csv
```

> **Tip**: Only one model is loaded into VRAM at a time.  Each `run_benchmark.py` invocation loads, runs, and then releases the model when the process exits.

---

## Evaluating Results

Score a single model's results JSON:

```bash
python evaluate_results.py --results results/qwen3vl8b_results.json
```

**Output**:
- Prints per-sample and aggregate metrics to stdout
- Writes `results/qwen3vl8b_results_scored.csv`

### Metrics computed

| Metric                   | Description                                                           |
|--------------------------|-----------------------------------------------------------------------|
| **final\_answer\_accuracy** | Primary metric — parsed answer vs. ground truth (exact string match)  |
| **EM** (Exact Match)     | Full prediction text matches ground truth after normalisation         |
| **Contains**             | Ground truth appears anywhere in the prediction                       |
| **Token F1**             | Token-level F1 between prediction and ground truth                   |
| **BLEU**                 | Smoothed sentence-level BLEU (method 4)                              |
| **ROUGE-L**              | Longest common subsequence F-measure                                 |
| **Error Rate**           | Percentage of samples that returned an error                         |

---

## Comparing Models

After evaluating all models, generate a ranked comparison table:

```bash
python compare_models.py
```

This reads every `results/*_results.json`, runs evaluation, and writes `results/model_comparison.csv` with all models sorted by final-answer accuracy.

---

## Plotting Results

Generate visual charts from the comparison table:

```bash
python plot_results.py
```

Produces `results/model_comparison.png` with:
- **Bar chart**: Final-answer accuracy by model
- **Scatter plot**: Accuracy vs. average latency per sample

---

## Prompt Construction Pipeline

Each question goes through a structured pipeline before being sent to the model:

1. **Load** the question and image path from the dataset CSV.
2. **Infer answer format** from keyword hints in the question text:
   - `"correct option letter"` → `multiple_choice`
   - `"floating-point number with N decimal"` → `float_1dp` / `float_2dp` / `float_3dp`
   - `"requiring an integer"` → `integer`
   - `"data array"` / `"array or tuple"` → `array`
   - Otherwise → `other`
3. **Build the prompt** as:
   - `BASE_PROMPT` — expert role instruction ("You are an expert electrical and electronics engineer…")
   - The raw question text
   - A format-specific suffix ("On the last line, write only the final option letter.")
4. **Encode the image** as a base64 data URI.
5. **Send** both to the selected backend: `llm.create_chat_completion()` for `llama-cpp-python`, or `/v1/chat/completions` for `llama-server`.
6. **Parse** the model's output to extract a clean final answer using deterministic regex rules.

See [LLM_Question_Pipeline.md](LLM_Question_Pipeline.md) for the full technical breakdown and flowchart, and [VLM_Benchmark_Guide.md](VLM_Benchmark_Guide.md) for the complete end-to-end project guide (environment setup, model selection, experiment execution, and results analysis).

---

## Configuration Reference

### Model Registry (`models/registry.py`)

The `MODEL_REGISTRY` dict maps short model keys to their configuration:

```python
MODEL_REGISTRY["qwen3vl8b"] = {
    "display_name": "Qwen3-VL 8B Instruct",
    "model_dir":    "Qwen3-VL-8B-Instruct",        # subdirectory under F:\Models\LLAMA
    "gguf_file":    "Qwen3VL-8B-Instruct-{quant}.gguf",
    "mmproj_file":  "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
    "chat_handler": "Qwen3VLChatHandler",          # llama-cpp-python handler class
    "default_quant":"Q8_0",
    "n_ctx":        8192,
    "hf_repo":      "Qwen/Qwen3-VL-8B-Instruct-GGUF",
}
```

### Key constants

| Constant         | Value               | Description                           |
|------------------|---------------------|---------------------------------------|
| `MODELS_ROOT`    | `F:\Models\LLAMA`   | Root directory for all GGUF files     |
| `DEFAULT_QUANT`  | `Q8_0`              | Default quantisation when not specified |
| `QUANT_OPTIONS`  | `Q4_K_M`, `Q5_K_M`, `Q8_0` | Supported quantisation levels |
| `BENCHMARK_ORDER`| *(7 model keys)*    | Canonical ordering for benchmark runs |

### Prompt templates (`prompts/task_prompts.py`)

| Format           | Suffix behaviour                                               |
|------------------|----------------------------------------------------------------|
| `multiple_choice`| "On the last line, write only the final option letter."        |
| `float_1dp`      | "…the final numeric answer with exactly one decimal place."    |
| `float_2dp`      | "…the final numeric answer with exactly two decimal places."   |
| `float_3dp`      | "…the final numeric answer with exactly three decimal places." |
| `integer`        | "…the final integer value."                                    |
| `array`          | "…the final answer in the requested array or tuple style."     |
| `other`          | "…the final answer."                                           |
