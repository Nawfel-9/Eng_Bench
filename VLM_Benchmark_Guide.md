# VLM Benchmark Guide — Technical Document Analysis (EEE_Bench)
> Based on: *"LLM Vision-Langage pour l'analyse automatique de documents techniques et schémas d'ingénierie"*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware & OS Prerequisites](#2-hardware--os-prerequisites)
3. [Environment Setup](#3-environment-setup)
4. [Dataset Preparation (EEE_Bench)](#4-dataset-preparation-eee_bench)
5. [Model Selection for 16 GB VRAM](#5-model-selection-for-16-gb-vram)
6. [Downloading Models (GGUF)](#6-downloading-models-gguf)
7. [Benchmark Pipeline](#7-benchmark-pipeline)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Running the Experiments](#9-running-the-experiments)
10. [Results & Analysis](#10-results--analysis)
11. [Optional: Fine-tuning Qwen2.5-VL on EEE_Bench](#11-optional-fine-tuning-qwen25-vl-on-eee_bench)
12. [Project File Structure](#12-project-file-structure)

---

## 1. Project Overview

The goal is to systematically evaluate **local vision-language models (VLMs)** on **EEE_Bench** — a dataset of electrical/electronic engineering technical documents — across tasks such as:

| Task | Description |
|------|-------------|
| **Symbol Recognition** | Identify electrical/engineering symbols in diagrams |
| **Component Relationship Extraction** | Understand how components connect or relate |
| **Specification Q&A** | Answer questions about specs, values, and annotations |
| **Anomaly Detection** | Spot errors, inconsistencies, or missing elements |

All inference runs through **llama.cpp** via the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) library using locally stored GGUF model files.

Dataset structure:
```
EEE_Bench/
├── images/      ← technical diagram images (.png / .jpg)
├── qa/          ← question-answer pairs (.json)
├── benchmark_index.csv
├── dev_split.csv
└── test_split.csv
```

---

## 2. Hardware & OS Prerequisites

### Recommended Setup
- **GPU**: 16 GB VRAM (e.g., RTX 5060 Ti, RTX 3090, RTX 4080)
- **OS**: Windows 10/11 (native) or Ubuntu 22.04 LTS

### Minimum system requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 16 GB |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB free | 200 GB free |
| CUDA | 12.x | 12.4+ |
| Python | 3.10 | 3.11+ |

> **Note**: All 8 models at their per-model optimal quant fit within 16 GB VRAM. Only one model is loaded at a time — each `run_benchmark.py` invocation loads, runs, and releases the model when the process exits.

---

## 3. Environment Setup

### 3.1 Create a Python environment

```bash
conda create -n vlm_bench python=3.11 -y
conda activate vlm_bench
```

### 3.2 Install llama-cpp-python with CUDA support

```bash
# For NVIDIA GPUs (CUDA 12.x):
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# On Windows (PowerShell):
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python
```

### 3.3 Install remaining dependencies

```bash
pip install pandas tqdm pillow scikit-learn matplotlib seaborn rouge-score nltk
```

---

## 4. Dataset Preparation (EEE_Bench)

**FOR NOW**, the benchmark uses data hosted on Hugging Face at [https://huggingface.co/datasets/afdsafas/EEE-Bench](https://huggingface.co/datasets/afdsafas/EEE-Bench). You can download it by running:

```bash
python scripts/download_EEE_Bench.py
```

These one-time scripts build and validate the dataset index.

### 4.1 Build the unified index

Reads `EEE_Bench/qa/eee_bench_qa.json`, resolves image paths, infers answer formats, and writes `EEE_Bench/benchmark_index.csv`.

```bash
python scripts/build_index.py
```

### 4.2 Validate image integrity

Checks that every referenced image file exists and can be opened by Pillow.

```bash
python scripts/validate_images.py
```

### 4.3 Create dev / test splits

Produces a stratified `dev_split.csv` (20%) and `test_split.csv` (80%), stratified by `answer_format`.

```bash
python scripts/split_dataset.py
```

> Use **dev_split.csv** for prompt engineering / sanity checks, and **test_split.csv** for final evaluation.

---

## 5. Model Selection for 16 GB VRAM

All models are loaded through a single `LlamaCppRunner` using `llama-cpp-python`. Each model directory (under `F:\Models\LLAMA\`) contains one or more quantised GGUF files and a multimodal projector (`mmproj-*-f16.gguf`).

The `default_quant` per model is chosen as the highest quality level whose total VRAM footprint (model GGUF + mmproj ≈ 1–2 GB) fits safely within 16 GB:

| Key | Display Name | Params | Default Quant | Est. VRAM | Chat Handler |
|-----|--------------|--------|---------------|-----------|--------------|
| `phi35v` | Phi-3.5-Vision-Instruct | 3.8B | **Q8_0** | ~5 GB | Llava15ChatHandler |
| `gemma4e4b` | Gemma 4 E4B | ~4B eff. | **Q8_0** | ~6 GB | Gemma4ChatHandler |
| `minicpmv` | MiniCPM-V 2.6 8B | 8B | **Q8_0** | ~10 GB | MiniCPMv26ChatHandler |
| `qwen25vl` | Qwen2.5-VL 7B Instruct | 7B | **Q8_0** | ~9 GB | Qwen25VLChatHandler |
| `internvl3` | InternVL3 8B Instruct | 8B | **Q8_0** | ~10 GB | Llava15ChatHandler |
| `glm4v` | GLM-4.1V-9B-Thinking | 9B | **Q8_0** | ~11 GB | Llava15ChatHandler |
| `llama32v` | Llama 3.2 Vision 11B | 11B | **Q5_K_M** | ~9 GB | Llava15ChatHandler |
| `pixtral` | Pixtral 12B | 12B | **Q5_K_M** | ~10 GB | Llava15ChatHandler |

> **Why Q5_K_M for the two largest models?** Llama 11B and Pixtral 12B at Q8_0 would consume ~13–14 GB, leaving no headroom for the OS and KV cache. Q5_K_M keeps them safely under 11 GB total.

### Quantisation levels available

| Level | Description | Typical Size (7B) | Quality |
|-------|-------------|-------------------|---------|
| `Q4_K_M` | 4-bit, medium | ~4.4 GB | Good |
| `Q5_K_M` | 5-bit, medium | ~5.1 GB | Better |
| `Q8_0` | 8-bit | ~7.5 GB | Near-lossless |

---

## 6. Downloading Models (GGUF)

GGUF files are downloaded from HuggingFace and stored at `F:\Models\LLAMA\`.

> **Note:** If you want to download models to a different directory, you must update the `MODELS_ROOT` constant in `models/registry.py` before running the download script or the benchmark.

### Option A: Run the master download script

```powershell
.\scripts\download_all_models.ps1
```

This script uses the `hf` CLI to download all 8 models at all 3 quantisation levels plus their multimodal projectors. Comment out any models or quants you don't need.

### Option B: Download a specific model

```bash
# Example: download only Qwen2.5-VL at Q8_0
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF \
    Qwen2.5-VL-7B-Instruct-Q8_0.gguf \
    mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
    --local-dir "F:\Models\LLAMA\Qwen2.5-VL-7B-Instruct"
```

### Option C: Regenerate the download script from the registry

```bash
python scripts/download_models.py             # print to stdout
python scripts/download_models.py --output scripts/download_all_models.ps1  # write .ps1
```

### Model storage layout

```
F:\Models\LLAMA\
├── Qwen2.5-VL-7B-Instruct\
│   ├── Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
│   ├── Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf
│   ├── Qwen2.5-VL-7B-Instruct-Q8_0.gguf
│   └── mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf
├── Llama-3.2-11B-Vision-Instruct\
├── MiniCPM-V-2_6\
├── gemma-4-E4B-it\
├── InternVL3-8B-Instruct\
├── GLM-4.1V-9B-Thinking\
├── Phi-3.5-vision-instruct\
└── pixtral-12b\
```

---

## 7. Benchmark Pipeline

### 7.1 Model registry (`models/registry.py`)

The `MODEL_REGISTRY` dict maps short model keys to their GGUF configuration:

```python
MODEL_REGISTRY["qwen25vl"] = {
    "display_name": "Qwen2.5-VL 7B Instruct",
    "model_dir":    "Qwen2.5-VL-7B-Instruct",       # subdirectory under F:\Models\LLAMA
    "gguf_file":    "Qwen2.5-VL-7B-Instruct-{quant}.gguf",
    "mmproj_file":  "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
    "chat_handler": "Qwen25VLChatHandler",            # llama-cpp-python handler class
    "default_quant": "Q8_0",
    "n_ctx":        8192,
    "hf_repo":      "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
}
```

`resolve_model_paths(model_key, quant)` returns the resolved `model_path` and `mmproj_path`, falling back to `default_quant` when no quant is supplied.

### 7.2 Prompt construction (`prompts/task_prompts.py`, `scripts/benchmark_utils.py`)

Each question goes through a structured pipeline:

1. **Infer answer format** from keyword hints in the question text:
   - `"correct option letter"` → `multiple_choice`
   - `"floating-point number with N decimal"` → `float_1dp` / `float_2dp` / `float_3dp`
   - `"requiring an integer"` → `integer`
   - `"data array"` / `"array or tuple"` → `array`
   - Otherwise → `other`

2. **Build the prompt** as:
   - `BASE_PROMPT` — expert role instruction (*"You are an expert electrical and electronics engineer…"*)
   - The raw question text
   - A format-specific suffix (*"On the last line, write only the final option letter."*)

### 7.3 Inference runner (`run_benchmark.py`)

`LlamaCppRunner` handles model loading and inference:

```python
class LlamaCppRunner:
    def __init__(self, model_key, quant=DEFAULT_QUANT, max_tokens=1024, temperature=0.0):
        spec = MODEL_REGISTRY[model_key]
        paths = resolve_model_paths(model_key, quant)

        chat_handler = _get_chat_handler(spec["chat_handler"], paths["mmproj_path"])
        # chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))

        self.llm = Llama(
            model_path=str(paths["model_path"]),
            chat_handler=chat_handler,
            n_gpu_layers=-1,    # offload all layers to GPU
            n_ctx=spec.get("n_ctx", 8192),
            logits_all=True,
            verbose=False,
        )
```

For each sample, the image is encoded as a base64 data URI and sent alongside the prompt:

```python
def infer(self, image_path: str, prompt: str) -> str:
    data_uri = _image_to_data_uri(image_path)   # base64 encoded
    response = self.llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=self.max_tokens,
        temperature=self.temperature,
    )
```

### 7.4 Answer extraction (`scripts/benchmark_utils.py`)

`extract_final_answer(prediction, answer_format)` applies deterministic regex rules:

| Format | Extraction rule |
|--------|----------------|
| `multiple_choice` | Last standalone capital letter in the full output |
| `float_1dp/2dp/3dp` | Last number found; quantised to the required decimal places |
| `integer` | Last integer in the final line |
| `array` | Last `(…)` tuple expression |
| `other` | Final non-empty line |

---

## 8. Evaluation Metrics

Computed by `evaluate_results.py`:

| Metric | Description |
|--------|-------------|
| **final_answer_accuracy** | Primary metric — parsed `final_answer` vs. ground truth (exact string match) |
| **EM** (Exact Match) | Full prediction text matches ground truth after normalisation |
| **Contains** | Ground truth appears anywhere in the prediction |
| **Token F1** | Token-level F1 between prediction and ground truth |
| **BLEU** | Smoothed sentence-level BLEU (method 4) |
| **ROUGE-L** | Longest common subsequence F-measure |
| **Error Rate** | Percentage of samples that returned an error |

Results per sample are saved to `results/<model_key>_results.json`, and a scored CSV is written to `results/<model_key>_results_scored.csv`.

---

## 9. Running the Experiments

### 9.1 Step-by-step execution order

```bash
# Step 1 — activate environment
conda activate vlm_bench

# Step 2 — prepare dataset (one-time)
python scripts/build_index.py
python scripts/validate_images.py
python scripts/split_dataset.py

# Step 3 — smoke test on 20 samples with the fastest model
python run_benchmark.py --model phi35v --split EEE_Bench/dev_split.csv --max 20
python evaluate_results.py --results results/phi35v_results.json

# Step 4 — full benchmark, one model at a time (lightest to heaviest)
python run_benchmark.py --model phi35v    --split EEE_Bench/test_split.csv
python run_benchmark.py --model gemma4e4b --split EEE_Bench/test_split.csv
python run_benchmark.py --model minicpmv  --split EEE_Bench/test_split.csv
python run_benchmark.py --model qwen25vl  --split EEE_Bench/test_split.csv
python run_benchmark.py --model internvl3 --split EEE_Bench/test_split.csv
python run_benchmark.py --model glm4v     --split EEE_Bench/test_split.csv
python run_benchmark.py --model llama32v  --split EEE_Bench/test_split.csv
python run_benchmark.py --model pixtral   --split EEE_Bench/test_split.csv

# Step 5 — aggregate and rank
python compare_models.py   # reads all results/*_results.json, writes results/model_comparison.csv

# Step 6 — visualise
python plot_results.py     # reads results/model_comparison.csv, writes results/model_comparison.png
```

### 9.2 CLI options for `run_benchmark.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *(required)* | Model key from the registry |
| `--split` | `EEE_Bench/test_split.csv` | Path to the CSV split file |
| `--output` | `results` | Output directory for result JSONs |
| `--quant` | *(model's default_quant)* | Override quantisation: `Q4_K_M`, `Q5_K_M`, `Q8_0` |
| `--max` | *(all samples)* | Limit sample count (for debugging / smoke tests) |

### 9.3 Monitor VRAM

Only one model is in VRAM at a time. Each `run_benchmark.py` process exits cleanly, releasing all GPU memory before the next run begins. You can confirm with:

```bash
nvidia-smi
```

### 9.4 If a model runs out of VRAM

Try a lower quantisation level explicitly:

```bash
python run_benchmark.py --model llama32v --quant Q4_K_M --split EEE_Bench/test_split.csv
```

Or reduce the context window in the registry (`n_ctx`) to lower KV cache usage.

---

## 10. Results & Analysis

### 10.1 Aggregate comparison (`compare_models.py`)

Reads every `results/*_results.json`, re-evaluates each, and writes `results/model_comparison.csv` sorted by `final_answer_accuracy` descending.

```bash
python compare_models.py
```

The output CSV contains: `model`, `final_answer_accuracy`, `EM`, `Contains`, `F1`, `BLEU`, `ROUGE-L`, `avg_latency_s`, `error_rate`, `N`.

### 10.2 Visualisation (`plot_results.py`)

Reads `results/model_comparison.csv` and writes `results/model_comparison.png` with:
- **Bar chart**: Final-answer accuracy by model
- **Scatter plot**: Accuracy vs. average latency per sample

```bash
python plot_results.py
```

---

## 11. Optional: Fine-tuning Qwen2.5-VL on EEE_Bench

As shown in the reference paper, fine-tuning a VLM on ~800 domain-specific samples closes most of the gap to GPT-4o at a fraction of the cost. Qwen2.5-VL-7B is the recommended candidate — it is the strongest 7B VLM and has good QLoRA support. Fine-tuning is done via **transformers + PEFT** (not llama.cpp, which is inference-only).

### 11.1 Install fine-tuning dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.49.0 accelerate peft trl bitsandbytes qwen-vl-utils
```

### 11.2 Prepare fine-tuning data

```python
# scripts/prepare_finetune_data.py
import json, pandas as pd

df = pd.read_csv("EEE_Bench/dev_split.csv")   # use dev set for fine-tuning
train_data = []

for _, row in df.iterrows():
    train_data.append({
        "id": row.get("image_id", str(_)),
        "image": row["abs_image_path"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{row['question']}"},
            {"from": "gpt",   "value": row["answer"]}
        ]
    })

with open("EEE_Bench/finetune_train.json", "w") as f:
    json.dump(train_data, f, indent=2)
print(f"Fine-tuning samples: {len(train_data)}")
```

### 11.3 Fine-tune with QLoRA

```python
# finetune_qwen25vl.py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
import torch

# Load in 4-bit for fine-tuning (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=bnb_config,
    device_map="cuda"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# LoRA — target attention and MLP projection layers
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir="checkpoints/qwen25vl-eee",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    remove_unused_columns=False,
)

# Full implementation reference:
# https://github.com/huggingface/trl/blob/main/examples/scripts/vsft_qwen2_vl.py
```

> After fine-tuning, export the merged model to GGUF and load it through the same llama-cpp-python pipeline:
> ```bash
> python llama.cpp/convert_hf_to_gguf.py checkpoints/qwen25vl-eee --outtype q8_0
> ```
> Then add a new entry to `models/registry.py` and run the benchmark normally.

---

## 12. Project File Structure

```
Prototype/
├── run_benchmark.py            # Main entry point — runs inference for one model
├── evaluate_results.py         # Scores a results JSON against ground truth
├── compare_models.py           # Aggregates & ranks all model results
├── plot_results.py             # Generates accuracy/latency charts
│
├── models/
│   └── registry.py             # Model registry — GGUF paths, handlers, metadata, default_quant
│
├── prompts/
│   └── task_prompts.py         # BASE_PROMPT + FORMAT_SUFFIXES (per answer format)
│
├── scripts/
│   ├── benchmark_utils.py      # Shared helpers: prompt builder, answer parser, normaliser
│   ├── build_index.py          # One-time: indexes QA JSON -> benchmark_index.csv
│   ├── split_dataset.py        # One-time: stratified dev/test split
│   ├── validate_images.py      # Checks image integrity
│   ├── download_models.py      # Generates hf download commands
│   └── download_all_models.ps1 # PowerShell script to download all GGUF models
│
├── EEE_Bench/
│   ├── images/                 # Technical diagram images
│   ├── qa/                     # Raw QA JSON files
│   ├── benchmark_index.csv     # Full indexed dataset
│   ├── dev_split.csv           # 20% development split
│   └── test_split.csv          # 80% test split
│
├── results/                    # Per-model result JSONs, scored CSVs, comparison table & chart
│
├── LLM_Question_Pipeline.md   # Detailed documentation of prompt construction
├── VLM_Benchmark_Guide.md     # This file
└── README.md                  # Quick-start guide
```

### External: Model Storage

All GGUF model files are stored at:
```
F:\Models\LLAMA\
├── Qwen2.5-VL-7B-Instruct\
├── Llama-3.2-11B-Vision-Instruct\
├── MiniCPM-V-2_6\
├── gemma-4-E4B-it\
├── InternVL3-8B-Instruct\
├── GLM-4.1V-9B-Thinking\
├── Phi-3.5-vision-instruct\
└── pixtral-12b\
```

---

## Quick-Start Summary

```bash
# 1. Create environment and install dependencies
conda create -n vlm_bench python=3.11 -y && conda activate vlm_bench
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python
pip install pandas tqdm pillow scikit-learn matplotlib seaborn rouge-score nltk

# 2. Download models (edit the script to comment out quants you don't need)
.\scripts\download_all_models.ps1

# 3. Prepare dataset
python scripts/build_index.py
python scripts/split_dataset.py

# 4. Sanity check (20 samples, fastest model)
python run_benchmark.py --model phi35v --max 20
python evaluate_results.py --results results/phi35v_results.json

# 5. Full benchmark (lightest to heaviest)
python run_benchmark.py --model phi35v    --split EEE_Bench/test_split.csv
python run_benchmark.py --model gemma4e4b --split EEE_Bench/test_split.csv
python run_benchmark.py --model minicpmv  --split EEE_Bench/test_split.csv
python run_benchmark.py --model qwen25vl  --split EEE_Bench/test_split.csv
python run_benchmark.py --model internvl3 --split EEE_Bench/test_split.csv
python run_benchmark.py --model glm4v     --split EEE_Bench/test_split.csv
python run_benchmark.py --model llama32v  --split EEE_Bench/test_split.csv
python run_benchmark.py --model pixtral   --split EEE_Bench/test_split.csv

# 6. Compare and visualise
python compare_models.py
python plot_results.py
```

---

*Start with `phi35v --max 20` to verify the full pipeline end-to-end, then run all 8 models in order from lightest to heaviest.*