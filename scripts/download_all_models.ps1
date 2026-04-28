# ============================================================
# EEE_Bench — Download all GGUF models from HuggingFace
# ============================================================
#
# Prerequisites:
#   pip install huggingface-hub
#
# Usage:
#   .\scripts\download_all_models.ps1
#
# To download only a specific quantisation, comment out the
# lines you do not need.
# ============================================================

# Ensure the target directory exists
New-Item -ItemType Directory -Force -Path "F:\Models\LLAMA" | Out-Null

# ── Qwen2.5-VL 7B Instruct (qwen25vl) ──
# HuggingFace repo: https://huggingface.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf --local-dir "F:\Models\LLAMA\Qwen2.5-VL-7B-Instruct"
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --local-dir "F:\Models\LLAMA\Qwen2.5-VL-7B-Instruct"
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-Q5_K_M.gguf --local-dir "F:\Models\LLAMA\Qwen2.5-VL-7B-Instruct"
hf download ggml-org/Qwen2.5-VL-7B-Instruct-GGUF Qwen2.5-VL-7B-Instruct-Q8_0.gguf --local-dir "F:\Models\LLAMA\Qwen2.5-VL-7B-Instruct"

# ── Llama 3.2 Vision 11B (llama32v) ──
# HuggingFace repo: https://huggingface.co/ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF
hf download ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF mmproj-Llama-3.2-11B-Vision-Instruct-f16.gguf --local-dir "F:\Models\LLAMA\Llama-3.2-11B-Vision-Instruct"
hf download ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf --local-dir "F:\Models\LLAMA\Llama-3.2-11B-Vision-Instruct"
hf download ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF Llama-3.2-11B-Vision-Instruct-Q5_K_M.gguf --local-dir "F:\Models\LLAMA\Llama-3.2-11B-Vision-Instruct"
hf download ggml-org/Llama-3.2-11B-Vision-Instruct-GGUF Llama-3.2-11B-Vision-Instruct-Q8_0.gguf --local-dir "F:\Models\LLAMA\Llama-3.2-11B-Vision-Instruct"

# ── MiniCPM-V 2.6 8B (minicpmv) ──
# HuggingFace repo: https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf
hf download openbmb/MiniCPM-V-2_6-gguf mmproj-MiniCPM-V-2_6-f16.gguf --local-dir "F:\Models\LLAMA\MiniCPM-V-2_6"
hf download openbmb/MiniCPM-V-2_6-gguf MiniCPM-V-2_6-Q4_K_M.gguf --local-dir "F:\Models\LLAMA\MiniCPM-V-2_6"
hf download openbmb/MiniCPM-V-2_6-gguf MiniCPM-V-2_6-Q5_K_M.gguf --local-dir "F:\Models\LLAMA\MiniCPM-V-2_6"
hf download openbmb/MiniCPM-V-2_6-gguf MiniCPM-V-2_6-Q8_0.gguf --local-dir "F:\Models\LLAMA\MiniCPM-V-2_6"

# ── Gemma 4 E4B (gemma4e4b) ──
# HuggingFace repo: https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-bf16.gguf --local-dir "F:\Models\LLAMA\gemma-4-E4B-it"
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf --local-dir "F:\Models\LLAMA\gemma-4-E4B-it"
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q5_K_M.gguf --local-dir "F:\Models\LLAMA\gemma-4-E4B-it"
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir "F:\Models\LLAMA\gemma-4-E4B-it"

# ── InternVL3 8B Instruct (internvl3) ──
# HuggingFace repo: https://huggingface.co/ggml-org/InternVL3-8B-Instruct-GGUF
hf download ggml-org/InternVL3-8B-Instruct-GGUF mmproj-InternVL3-8B-Instruct-f16.gguf --local-dir "F:\Models\LLAMA\InternVL3-8B-Instruct"
hf download ggml-org/InternVL3-8B-Instruct-GGUF InternVL3-8B-Instruct-Q4_K_M.gguf --local-dir "F:\Models\LLAMA\InternVL3-8B-Instruct"
hf download ggml-org/InternVL3-8B-Instruct-GGUF InternVL3-8B-Instruct-Q5_K_M.gguf --local-dir "F:\Models\LLAMA\InternVL3-8B-Instruct"
hf download ggml-org/InternVL3-8B-Instruct-GGUF InternVL3-8B-Instruct-Q8_0.gguf --local-dir "F:\Models\LLAMA\InternVL3-8B-Instruct"

# ── GLM-4.1V-9B-Thinking (glm4v) ──
# HuggingFace repo: https://huggingface.co/mradermacher/GLM-4.1V-9B-Thinking-GGUF
hf download mradermacher/GLM-4.1V-9B-Thinking-GGUF mmproj-GLM-4.1V-9B-Thinking-f16.gguf --local-dir "F:\Models\LLAMA\GLM-4.1V-9B-Thinking"
hf download mradermacher/GLM-4.1V-9B-Thinking-GGUF GLM-4.1V-9B-Thinking-Q4_K_M.gguf --local-dir "F:\Models\LLAMA\GLM-4.1V-9B-Thinking"
hf download mradermacher/GLM-4.1V-9B-Thinking-GGUF GLM-4.1V-9B-Thinking-Q5_K_M.gguf --local-dir "F:\Models\LLAMA\GLM-4.1V-9B-Thinking"
hf download mradermacher/GLM-4.1V-9B-Thinking-GGUF GLM-4.1V-9B-Thinking-Q8_0.gguf --local-dir "F:\Models\LLAMA\GLM-4.1V-9B-Thinking"

# ── Phi-3.5-Vision-Instruct (phi35v) ──
# HuggingFace repo: https://huggingface.co/abetlen/Phi-3.5-vision-instruct-gguf
hf download abetlen/Phi-3.5-vision-instruct-gguf mmproj-Phi-3.5-vision-instruct-f16.gguf --local-dir "F:\Models\LLAMA\Phi-3.5-vision-instruct"
hf download abetlen/Phi-3.5-vision-instruct-gguf Phi-3.5-vision-instruct-Q4_K_M.gguf --local-dir "F:\Models\LLAMA\Phi-3.5-vision-instruct"
hf download abetlen/Phi-3.5-vision-instruct-gguf Phi-3.5-vision-instruct-Q5_K_M.gguf --local-dir "F:\Models\LLAMA\Phi-3.5-vision-instruct"
hf download abetlen/Phi-3.5-vision-instruct-gguf Phi-3.5-vision-instruct-Q8_0.gguf --local-dir "F:\Models\LLAMA\Phi-3.5-vision-instruct"

# ── Pixtral 12B (pixtral) ──
# HuggingFace repo: https://huggingface.co/ggml-org/pixtral-12b-GGUF
hf download ggml-org/pixtral-12b-GGUF mmproj-pixtral-12b-f16.gguf --local-dir "F:\Models\LLAMA\pixtral-12b"
hf download ggml-org/pixtral-12b-GGUF pixtral-12b-Q4_K_M.gguf --local-dir "F:\Models\LLAMA\pixtral-12b"
hf download ggml-org/pixtral-12b-GGUF pixtral-12b-Q5_K_M.gguf --local-dir "F:\Models\LLAMA\pixtral-12b"
hf download ggml-org/pixtral-12b-GGUF pixtral-12b-Q8_0.gguf --local-dir "F:\Models\LLAMA\pixtral-12b"
