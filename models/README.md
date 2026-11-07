# 📦 Models Directory

Place your `.gguf` model weights and tokenizer binary tables in this directory (or specify a custom path via CLI flags).

---

## 🎯 Recommended Models

### 1. Target Engine (Primary Model - 7B)
* **Model:** `Qwen2.5-Coder-7B-Instruct-Q4_0.gguf`
* **File Name:** `qwen2.5-coder-7b-instruct-q4_0.gguf`
* **VRAM Footprint:** ~4.09 GB
* **Download:**
  ```bash
  # Via huggingface-cli
  huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-GGUF qwen2.5-coder-7b-instruct-q4_0.gguf --local-dir models/
  ```

### 2. Draft Engine (Speculative Lookahead - 0.5B)
* **Model:** `Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf`
* **File Name:** `qwen2.5-coder-0.5b-instruct-q8_0.gguf`
* **VRAM Footprint:** ~0.64 GB
* **Download:**
  ```bash
  # Via huggingface-cli
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

> **Note:** You can generate the tokenizer binary tables in 3 seconds using:
> ```bash
> python scripts/export_qwen_merges.py
> ```
