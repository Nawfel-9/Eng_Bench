# EEE_Bench — Vision-Language Model Benchmark for Technical Document Analysis

A benchmarking framework for evaluating **local vision-language models (VLMs)** on the **EEE_Bench** dataset — a collection of electrical and electronics engineering technical diagrams paired with question-answer items.  All inference runs through **llama.cpp** via the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) library using locally stored GGUF model files.

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
3. Sends the image + prompt to a **local GGUF model** via `llama-cpp-python`.
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

> **Note**: All 8 models at Q4\_K\_M fit within 16 GB VRAM.  Larger quantisations (Q8\_0) for the 11–12 B models may require swapping between runs.

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
├── Qwen2.5-VL-7B-Instruct/
├── Llama-3.2-11B-Vision-Instruct/
├── MiniCPM-V-2_6/
├── gemma-4-E4B-it/
├── InternVL3-8B-Instruct/
├── GLM-4.1V-9B-Thinking/
├── Phi-3.5-vision-instruct/
└── pixtral-12b/
```

Each model directory contains:
- One or more **quantised GGUF** files (Q4\_K\_M, Q5\_K\_M, Q8\_0)
- A **multimodal projector** file (`mmproj-*-f16.gguf`)

---

## Supported Models

All models are loaded through a single `LlamaCppRunner` using `llama-cpp-python`. The default quantization levels shown below are optimized specifically for an **RTX 5060 Ti with 16 GB VRAM**, maximizing quality while safely avoiding out-of-memory errors.

| Key         | Display Name              | Parameters | Default Quant | Chat Handler             | HuggingFace GGUF Repo                                          |
|-------------|---------------------------|------------|---------------|--------------------------|------------------------------------------------------------------|
| `phi35v`    | Phi-3.5-Vision-Instruct   | 3.8 B      | Q8\_0         | Llava15ChatHandler       | `abetlen/Phi-3.5-vision-instruct-gguf`                          |
| `gemma4e4b` | Gemma 4 E4B               | ~4 B eff.  | Q8\_0         | Gemma4ChatHandler        | `ggml-org/gemma-4-E4B-it-GGUF`                                  |
| `minicpmv`  | MiniCPM-V 2.6 8B          | 8 B        | Q8\_0         | MiniCPMv26ChatHandler    | `openbmb/MiniCPM-V-2_6-gguf`                                    |
| `qwen25vl`  | Qwen2.5-VL 7B Instruct    | 7 B        | Q8\_0         | Qwen25VLChatHandler      | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF`                          |
| `internvl3` | InternVL3 8B Instruct     | 8 B        | Q8\_0         | Llava15ChatHandler       | `ggml-org/InternVL3-8B-Instruct-GGUF`                           |
| `glm4v`     | GLM-4.1V-9B-Thinking      | 9 B        | Q8\_0         | Llava15ChatHandler       | `mradermacher/GLM-4.1V-9B-Thinking-GGUF`                        |
| `llama32v`  | Llama 3.2 Vision 11B      | 11 B       | Q5\_K\_M      | Llava15ChatHandler       | `ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF`                   |
| `pixtral`   | Pixtral 12B               | 12 B       | Q5\_K\_M      | Llava15ChatHandler       | `ggml-org/pixtral-12b-GGUF`                                     |

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

This downloads **all 8 models** at **all 3 quantisation levels** plus their multimodal projectors.  Comment out any models or quants you don't need.

### Option B: Download a specific model

```bash
# Example: download only Qwen2.5-VL at Q4_K_M
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
    Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
    mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
    --local-dir "F:\Models\LLAMA\Qwen2.5-VL-7B-Instruct"
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
python run_benchmark.py --model qwen25vl --split EEE_Bench/test_split.csv
```

### All CLI options

| Argument   | Default                      | Description                                      |
|------------|------------------------------|--------------------------------------------------|
| `--model`  | *(required)*                 | Model key from the registry                      |
| `--split`  | `EEE_Bench/test_split.csv`   | Path to the CSV split file                       |
| `--output` | `results`                    | Output directory for result JSONs                |
| `--quant`  | `Q4_K_M`                     | Quantisation level: `Q4_K_M`, `Q5_K_M`, `Q8_0`  |
| `--max`    | *(all samples)*              | Limit sample count (for debugging / smoke tests) |

### Smoke test on the dev set

```bash
python run_benchmark.py --model phi35v --split EEE_Bench/dev_split.csv --max 20 --quant Q4_K_M
```

### Full benchmark run (all models, lightest → heaviest)

```bash
python run_benchmark.py --model phi35v    --split EEE_Bench/test_split.csv
python run_benchmark.py --model gemma4e4b --split EEE_Bench/test_split.csv
python run_benchmark.py --model minicpmv  --split EEE_Bench/test_split.csv
python run_benchmark.py --model qwen25vl  --split EEE_Bench/test_split.csv
python run_benchmark.py --model internvl3 --split EEE_Bench/test_split.csv
python run_benchmark.py --model glm4v     --split EEE_Bench/test_split.csv
python run_benchmark.py --model llama32v  --split EEE_Bench/test_split.csv
python run_benchmark.py --model pixtral   --split EEE_Bench/test_split.csv
```

> **Tip**: Only one model is loaded into VRAM at a time.  Each `run_benchmark.py` invocation loads, runs, and then releases the model when the process exits.

---

## Evaluating Results

Score a single model's results JSON:

```bash
python evaluate_results.py --results results/qwen25vl_results.json
```

**Output**:
- Prints per-sample and aggregate metrics to stdout
- Writes `results/qwen25vl_results_scored.csv`

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
5. **Send** both to the model via `llm.create_chat_completion()`.
6. **Parse** the model's output to extract a clean final answer using deterministic regex rules.

See [LLM_Question_Pipeline.md](LLM_Question_Pipeline.md) for the full technical breakdown and flowchart, and [VLM_Benchmark_Guide.md](VLM_Benchmark_Guide.md) for the complete end-to-end project guide (environment setup, model selection, experiment execution, and results analysis).

---

## Configuration Reference

### Model Registry (`models/registry.py`)

The `MODEL_REGISTRY` dict maps short model keys to their configuration:

```python
MODEL_REGISTRY["qwen25vl"] = {
    "display_name": "Qwen2.5-VL 7B Instruct",
    "model_dir":    "Qwen2.5-VL-7B-Instruct",       # subdirectory under F:\Models\LLAMA
    "gguf_file":    "Qwen2.5-VL-7B-Instruct-{quant}.gguf",
    "mmproj_file":  "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
    "chat_handler": "Qwen25VLChatHandler",            # llama-cpp-python handler class
    "default_quant":"Q4_K_M",
    "n_ctx":        8192,
    "hf_repo":      "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
}
```

### Key constants

| Constant         | Value               | Description                           |
|------------------|---------------------|---------------------------------------|
| `MODELS_ROOT`    | `F:\Models\LLAMA`   | Root directory for all GGUF files     |
| `DEFAULT_QUANT`  | `Q8_0`              | Default quantisation when not specified |
| `QUANT_OPTIONS`  | `Q4_K_M`, `Q5_K_M`, `Q8_0` | Supported quantisation levels |
| `BENCHMARK_ORDER`| *(8 model keys)*    | Canonical ordering for benchmark runs |

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
