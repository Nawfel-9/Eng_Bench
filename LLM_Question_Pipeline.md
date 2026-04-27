# LLM Question Construction Pipeline (EEE_Bench)

This document explains how each model query is built in this benchmark.
It has two parts:
- Detailed pipeline (technical)
- Simplified pipeline (easy version)

## 1) Detailed Pipeline

### Step A: Load raw QA items
- Source file: `EEE_Bench/qa/eee_bench_qa.json`
- Each item includes: `id`, `split`, `image`, `question`, `answer`
- Loader script: `scripts/build_index.py`

What happens:
1. Read all JSON items.
2. Resolve the image path using `EEE_Bench/images/<image>`.
3. Skip rows with missing images.

### Step B: Infer `answer_format` from hint text
- Function: `infer_answer_format(question)`
- File: `scripts/benchmark_utils.py`

Classification rules (keyword-based):
- Contains "correct option letter" -> `multiple_choice`
- Contains "floating-point number with one decimal" -> `float_1dp`
- Contains "floating-point number with two decimal" -> `float_2dp`
- Contains "floating-point number with three decimal" -> `float_3dp`
- Contains "requiring an integer" -> `integer`
- Contains "data array" or "array or tuple" -> `array`
- Otherwise -> `other`

Output row written to `EEE_Bench/benchmark_index.csv` includes:
- `question`
- `answer`
- `answer_format`
- image paths (`image_path`, `abs_image_path`)

### Step C: Create dev/test split
- Script: `scripts/split_dataset.py`
- Uses `answer_format` for stratification (for non-rare formats).

Result files:
- `EEE_Bench/dev_split.csv`
- `EEE_Bench/test_split.csv`

### Step D: Build the actual model prompt
- Function: `build_prompt(question, answer_format)`
- Files:
  - `prompts/task_prompts.py` (prompt templates)
  - `scripts/benchmark_utils.py` (prompt builder)

Prompt is constructed as:
1. `BASE_PROMPT` (role + constraints)
2. Raw dataset `question` text
3. `FORMAT_SUFFIXES[answer_format]` instruction

This means the model sees not just the raw question, but:
- Role instruction (engineering expert)
- The original question body and choices
- A strict output-format instruction for the last line

### Step E: Send image + prompt to the model
- Main runner: `run_benchmark.py`

Each call goes through `LlamaCppRunner.infer()` in `run_benchmark.py`, which:
- Encodes the image as a base64 data URI
- Sends it alongside the constructed prompt via `llm.create_chat_completion()`
- Uses `temperature=0` (greedy decoding) and `max_tokens=1024`

So each query is multimodal: diagram image + constructed text prompt.

### Step F: Parse model output into `final_answer`
- Function: `extract_final_answer(prediction, answer_format)`
- File: `scripts/benchmark_utils.py`

Parsing behavior by format:
- `multiple_choice`: extract last standalone capital letter
- `float_1dp/2dp/3dp`: extract last number and quantize to required decimals
- `integer`: extract last integer
- `array`: extract tuple-like expression if present
- `other`: use final non-empty line

Saved in results JSON per sample:
- `prediction` (full model text)
- `final_answer` (parsed answer used for main accuracy)

### Step G: Score answers
- Script: `evaluate_results.py`
- Main metric used for comparison: `final_answer_accuracy`

This compares:
- parsed `final_answer`
- ground-truth `gt_answer`

Also computes text-level metrics (EM, Contains, F1, BLEU, ROUGE-L).

---

## 2) Simplified Pipeline (Easy Version)

Think of it as a 5-step assembly line:
1. Read one question + image from dataset.
2. Detect expected answer type from the hint text (MCQ, float, integer, array).
3. Wrap the question with:
   - expert role instruction at top
   - strict final-line format rule at bottom
4. Send image + wrapped prompt to model.
5. Extract one clean final answer from model output and score it.

In short:
- Raw question is not sent alone.
- It is converted into a controlled prompt with formatting rules.
- The benchmark score mainly depends on the parsed final answer.

---

## 3) Flowchart (Mermaid)

```mermaid
flowchart TD
    A[EEE_Bench/qa/eee_bench_qa.json] --> B[scripts/build_index.py]
    B --> C[infer_answer_format(question)]
    C --> D[EEE_Bench/benchmark_index.csv]
    D --> E[scripts/split_dataset.py]
    E --> F[EEE_Bench/dev_split.csv]
    E --> G[EEE_Bench/test_split.csv]

    F --> H[run_benchmark.py]
    G --> H

    H --> I[build_prompt(question, answer_format)]
    I --> J[BASE_PROMPT + question + format suffix]
    J --> K[Runner sends image + prompt]
    K --> L[Model prediction]
    L --> M[extract_final_answer(prediction, answer_format)]
    M --> N[results/<model>_results.json]
    N --> O[evaluate_results.py]
    O --> P[results/<model>_results_scored.csv]
    P --> Q[final_answer_accuracy + text metrics]
```

---

## 4) Quick Example of Prompt Composition

If a row is:
- `question`: includes choices A/B/C/D and says "provide the correct option letter"
- inferred `answer_format`: `multiple_choice`

Final prompt becomes:
1. Engineering role instruction (`BASE_PROMPT`)
2. The full original question text (with choices)
3. Suffix: "On the last line, write only the final option letter."

This structure is why final-answer extraction can be deterministic and scoreable.
