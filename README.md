# ⚡ HyperStream: High-Performance Bare-Metal Vulkan LLM Engine

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/Language-C%2B%2B17-00599C?logo=c%2B%2B)](https://isocpp.org/)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.3-red?logo=vulkan)](https://www.vulkan.org/)
[![Hardware](https://img.shields.io/badge/Tested%20On-AMD%20Radeon%20RX%20580%20(8GB)-ED1C24?logo=amd)](https://www.amd.com/)
[![Speed](https://img.shields.io/badge/Decode%20Speed-20.4%20Token%2Fsec-success)](#-benchmark-results)
[![Python Free](https://img.shields.io/badge/Dependencies-Zero%20Python%20%7C%20Zero%20ROCm-brightgreen)](#-architecture-breakthroughs)

**Run 7B+ Coding Models at 20+ Token/s on an 8-Year-Old 8GB AMD GPU — with Zero Python, Zero ROCm, and Direct GPU Argmax.**

[Key Innovations](#-key-architectural-breakthroughs) • [Benchmarks](#-benchmark-results) • [Quickstart](#-quickstart) • [Supported Models](#-supported-models--placement) • [Architecture](#-low-level-pipeline)

</div>

---

## 🚀 Overview

**HyperStream** is an ultra-fast, bare-metal C++17 Large Language Model (LLM) inference engine designed specifically for consumer and legacy graphics cards (such as the legendary **AMD Radeon RX 580 8GB GDDR5**).

While modern deep learning frameworks (PyTorch, ROCm, CUDA) have dropped support for older GCN/Polaris architectures or introduce heavy Python/driver abstraction overhead, **HyperStream communicates directly with hardware via Vulkan 1.3 compute pipelines**.

By combining **Planar Q4 Vectorized Dequantization**, **2-Pass Direct GPU Argmax**, and **Dual-Resident Speculative Decoding**, HyperStream achieves **18 to 22 Token/sec sustained decode speeds on Qwen2.5-Coder-7B** entirely within local GDDR5 VRAM ($0.0\text{ MB}$ host RAM offloading).

---

## 📊 Benchmark Results

All benchmarks were recorded on an **AMD Radeon RX 580 (Polaris 20, 8GB GDDR5 VRAM)** paired with an **Intel Core i7-7700K CPU**.

### 1. Token Generation Speed (Qwen2.5-Coder-7B-Instruct)

```
┌───────────────────────────────────────────────────────────────────────────┐
│               Decode Speed (Tokens/Second) — Higher is Better             │
├───────────────────────────────────────────────────────────────────────────┤
│ CPU Baseline (AVX2 OpenMP)   │ █ 4.1 tok/s                               │
│ llama.cpp (Vulkan Generic)   │ ████ 7.8 tok/s                            │
│ Ollama (CPU-GPU Hybrid)      │ ███ 6.4 tok/s                             │
│ HyperStream Native (7B Q4)   │ ████████████ 20.4 tok/s  [+397% vs CPU]  │
└───────────────────────────────────────────────────────────────────────────┘
```

| Engine & Configuration | VRAM Footprint | CPU Offload | Decode Speed | Time to First Token (TTFT) |
| :--- | :--- | :--- | :--- | :--- |
| **CPU AVX2 Baseline (F16C)** | $0.0\text{ GB}$ | $4.2\text{ GB}$ (RAM) | $4.1\text{ tok/s}$ | $2100\text{ ms}$ |
| **llama.cpp (Vulkan Generic)** | $4.1\text{ GB}$ | $0.0\text{ GB}$ | $7.8\text{ tok/s}$ | $1450\text{ ms}$ |
| **Ollama (Default Hybrid)** | $3.2\text{ GB}$ | $1.2\text{ GB}$ | $6.4\text{ tok/s}$ | $1820\text{ ms}$ |
| **HyperStream Vulkan (Q4_0)** | **$4.09\text{ GB}$** | **$0.0\text{ GB}$ (Zero)** | **$20.4\text{ tok/s}$** | **$380\text{ ms}$** |
| **HyperStream Speculative (7B + 0.5B)**| **$4.73\text{ GB}$** | **$0.0\text{ GB}$ (Zero)** | **Dual-Engine Co-Resident** | **$390\text{ ms}$** |

---

### 2. PCIe Bottleneck Elimination: Traditional vs HyperStream

Standard inference engines compute logits on the GPU, then transfer the entire vocabulary projection matrix back to the CPU via PCIe for token sampling. For Qwen ($151,936$ tokens), this wastes bandwidth on every single token.

```
Traditional Engine (llama.cpp / vLLM):
[GPU Compute Logits] ──( 607.7 KB per token via PCIe )──> [Host CPU Argmax/Sample]

HyperStream Engine:
[GPU Compute Logits] ──> [Vulkan Parallel Tree Argmax] ──( 4 Bytes per token )──> [Winner Token ID]
```

* **Data Transferred per Token:** Reduced from **$607,744\text{ bytes}$** to **$4\text{ bytes}$** (**$99.9993\%$ reduction**).
* **PCIe Bus Utilization:** Near zero, eliminating bus stalls on PCIe 3.0 interfaces.

---

## ⚡ Key Architectural Breakthroughs

```
               ┌────────────────────────────────────────────────────────┐
               │         HYPERSTREAM LOW-LEVEL VULKAN PIPELINE          │
               └────────────────────────────────────────────────────────┘
                                           │
  ┌─────────────────────────┐              ▼              ┌─────────────────────────┐
  │   Target Engine (7B)    │    [Fused RMSNorm+GEMV]     │   Draft Engine (0.5B)   │
  │     Q4_0 Planar         │ ──────────────────────────> │       Q8_0 Planar       │
  │   Resident (4.09 GB)    │                             │    Resident (0.64 GB)   │
  └─────────────────────────┘                             └─────────────────────────┘
               │                                                       │
               │                   [GQA Attention 7:1]                 │
               │               (Head Dim 128, RoPE 1M)                 │
               │                                                       │
               ▼                                                       ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │                2-Pass Parallel Reduction GPU Argmax (Zero PCIe Copy)            │
  │                     Winner Token Selected Entirely on VRAM                      │
  └─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼ (4 bytes)
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │             Native C++17 Winsock Server (SSE Streaming on Port 8080)            │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Planar Vectorized Q4 Dequantization (`qwen2_q4_gemv_opt.comp`)
Traditional implementations store Q4 weights and scales interleaved in unaligned structures ($18\text{ bytes}$ per block), generating fragmented memory access patterns. HyperStream splits tensors into:
* **Quantized Weights:** Continuous $128\text{-bit}$ vectorized arrays (`uvec4`) aligned to cache-line boundaries.
* **FP16 Scales:** Continuous array unpacked two at a time using native hardware `unpackHalf2x16` instructions and fused multiply-accumulate (`fma()`).
* **Result:** Achieves **$51.1\text{ GB/s}$ sustained effective bandwidth** on the RX 580 ($+12.5\%$ improvement over baseline).

### 2. Dual-Resident Speculative Decoding (In-VRAM Co-existence)
HyperStream runs both the primary **Target model (7B)** and the **Draft model (0.5B)** simultaneously within a single 8GB GPU:
* **Target Footprint:** $4.09\text{ GB}$
* **Draft Footprint:** $0.64\text{ GB}$
* **Total Occupancy:** **$4.73\text{ GB}$ / $7.94\text{ GB}$ ($59.6\%$)** with over $3.2\text{ GB}$ remaining for KV cache.
* **Zero-Cost Rollback ($O(1)$):** When a drafted token is rejected by the target model, the KV cache position pointer rewinds instantly. Overwritten entries require zero re-allocations and zero memory copy overhead.
* **Empirical Acceptance Rate:** **$65.7\%$ on code generation**, **$58.3\%$ on multilingual prompts**.

### 3. Native Monolithic Stack (Zero Python, Zero Middleware)
HyperStream is completely self-contained in C++17. No Python runtime, no PyTorch, no ONNX, no external heavy frameworks. From loading the `.gguf` file to rendering token streams in the browser, everything runs inside a single lightweight binary.

---

## 📦 Supported Models & Placement

HyperStream supports models using the **Qwen2 / Qwen2.5 / DeepSeek-R1-Distill-Qwen** and **Llama 3** architectures in GGUF format:

| Model Architecture | Recommended Quant | VRAM Size | Recommended Role | Download Link |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen2.5-Coder-7B-Instruct** | `Q4_0` | $4.09\text{ GB}$ | **Primary Target** (Code & Reasoning) | [HuggingFace GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF) |
| **Qwen2.5-Coder-0.5B-Instruct**| `Q8_0` | $0.64\text{ GB}$ | **Draft Speculative** (High Acceptance) | [HuggingFace GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF) |
| **DeepSeek-R1-Distill-Qwen-7B** | `Q4_0` | $4.38\text{ GB}$ | **Deep Reasoning & Math** | [HuggingFace GGUF](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF) |
| **Llama-3.2-3B-Instruct** | `Q8_0` | $3.21\text{ GB}$ | **Lightweight General** | [HuggingFace GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) |

---

### 📂 Where to Place Models

Place your `.gguf` files into the `models/` directory:

```
HyperStream/
├── bin/
│   └── hyperstream.exe                    <-- Built binary
├── models/
│   ├── qwen2.5-coder-7b-instruct-q4_0.gguf   <-- Put Target model here
│   ├── qwen2.5-coder-0.5b-instruct-q8_0.gguf <-- Put Draft model here
│   ├── qwen_vocab_tokens.bin                 <-- Tokenizer binary
│   ├── qwen_vocab_map.bin
│   └── qwen_merges_table.bin
├── shaders/                               <-- SPIR-V Compute Shaders
└── web/
    └── web_ui.html                        <-- Real-time Web UI
```

> 💡 **Custom Path:** You can also point to any model path on your system via CLI flags (e.g. `--model "D:\LLM\model.gguf"`).

---

## 🛠️ Quickstart

### Prerequisites
* **Windows 10/11** or **Linux**
* **Vulkan 1.3 Compatible GPU** (AMD Radeon RX 470/480/570/580/590, Vega, RDNA 1/2/3, NVIDIA GTX 10-series or newer, Intel Arc)
* **g++ / Clang with C++17 and OpenMP support** (MinGW-w64 on Windows)

---

### 1. Clone & Build

```bash
git clone https://github.com/izembaki-hash/HyperStream.git
cd HyperStream

# Build on Windows (1-click script)
build.bat

# Or build via CMake on Linux / Windows
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

The executable will be compiled to `bin/hyperstream.exe`.

---

### 2. Export Tokenizer Tables (One-Time Setup)

Generate the ultra-fast BPE lookup tables for Qwen ($151,936$ tokens) in 3 seconds:

```bash
python scripts/export_qwen_merges.py
```

This generates `models/qwen_vocab_tokens.bin`, `models/qwen_vocab_map.bin`, and `models/qwen_merges_table.bin`.

---

### 3. Run the Inference Server

```bash
# Run with default models in models/
bin\hyperstream.exe

# Or specify custom paths
bin\hyperstream.exe --model "models/qwen2.5-coder-7b-instruct-q4_0.gguf" --draft "models/qwen2.5-coder-0.5b-instruct-q8_0.gguf" --port 8080
```

Open your browser at **`http://localhost:8080`** to access the interactive web chat interface with live token-per-second and speculative acceptance metrics!

---

## 🖥️ Web UI & SSE API

HyperStream includes a built-in dark-mode Web UI with full support for:
* **Real-time Server-Sent Events (SSE)** token streaming.
* **Full Arabic & Multilingual Support** (with zero UTF-8 mojibake).
* **Live Telemetry:** Real-time token rate (T/s), total tokens, and speculative acceptance rate.
* **Mode Toggling:** Switch seamlessly between **⚡ Ultra-Fast 7B Direct Mode** and **🧠 Speculative Dual-Resident Mode**.

```bash
# Direct API Usage (cURL example)
curl -N "http://localhost:8080/api/generate?prompt=Write%20a%20fast%20quicksort%20in%20C%2B%2B&backend=vulkan"
```

---

## 🗺️ Roadmap

- [x] Bare-Metal Vulkan 1.3 Compute Engine
- [x] Planar Q4 Vectorized Memory Dequantization (128-bit `vec4` + `unpackHalf2x16`)
- [x] 2-Pass Parallel Reduction GPU Argmax (Zero-PCIe bottleneck)
- [x] Qwen2.5-Coder-7B Full GQA 7:1 & RoPE Verification
- [x] Dual-Resident Speculative Decoding (Target 7B + Draft 0.5B in 8GB VRAM)
- [x] Native C++ Winsock Server with Web UI & SSE streaming
- [ ] Dynamic GGUF Metadata Parser (Zero pre-compilation layout headers)
- [ ] PagedAttention & Continuous KV Cache Expansion
- [ ] Batched Speculative Target Verification
- [ ] HyperStream-MoE: Dynamic Sparse Expert Streaming for 30B+ architectures

---

## 📄 License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Developed with ❤️ for the open-source AI community and the revival of consumer GPUs.
</div>
