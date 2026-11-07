# 📦 Models Directory

Place your `.gguf` model weights and tokenizer binary tables in this directory (or point to any custom path via `--model` and `--draft` CLI flags).

---

## 🌐 Supported Architectural Families

HyperStream supports **any GGUF model** belonging to the following architectural families:

1. **Qwen2 & Qwen2.5 Family:**
   * All parameter sizes: `0.5B`, `1.5B`, `3B`, `7B`, `14B`, `32B`, `72B`.
   * Variants: `Qwen2.5-Coder`, `Qwen2.5-Math`, `Instruct`, and base checkpoints.
2. **DeepSeek-R1 Distilled Reasoning Family:**
   * `DeepSeek-R1-Distill-Qwen` (1.5B, 7B, 14B, 32B).
   * `DeepSeek-R1-Distill-Llama-8B` / `70B`.
3. **Llama 3 / 3.1 / 3.2 Family:**
   * `Llama-3.2` (1B, 3B), `Llama-3.1` (8B, 70B), and fine-tunes like `Hermes 3`.
4. **Community Reasoning & Math Derivatives:**
   * `Marco-o1`, `QwQ-32B-Preview`, `OpenCodeInterpreter`, etc.

---

## 🎯 Recommended Out-of-the-Box Setup (Dual-Resident on 8GB GPU)

For a standard **8GB VRAM GPU (e.g. AMD RX 580)**, the recommended combination for maximum speed and speculative acceleration is:

### 1. Primary Target Engine (7B Coder)
* **Model:** `Qwen2.5-Coder-7B-Instruct-Q4_0.gguf` (or `DeepSeek-R1-Distill-Qwen-7B`)
* **VRAM Footprint:** ~4.09 GB
* **Download:**
  ```bash
  huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_0.gguf --local-dir models/
  ```

### 2. Draft Lookahead Engine (0.5B Speculative)
* **Model:** `Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf`
* **VRAM Footprint:** ~0.64 GB
* **Download:**
  ```bash
  huggingface-cli download Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF qwen2.5-coder-0.5b-instruct-q8_0.gguf --local-dir models/
  ```

---

## 📂 Expected Directory Layout

```
models/
├── qwen2.5-coder-7b-instruct-q4_0.gguf   # Primary Target Engine (~4.09 GB)
├── qwen2.5-coder-0.5b-instruct-q8_0.gguf # Draft Speculative Engine (~0.64 GB)
├── qwen_vocab_tokens.bin                 # BPE vocab strings & offsets
├── qwen_vocab_map.bin                    # BPE string-to-token map
└── qwen_merges_table.bin                 # BPE merge rules
```

> **Note:** Generate tokenizer binary tables in 3 seconds:
> ```bash
> python scripts/export_qwen_merges.py
> ```
