# HyperStream E2E Test Infrastructure Specification

**Document Identifier**: `TEST_INFRA.md`  
**Status**: Authoritative Master Test Architecture  
**Scope**: Requirements R1 through R7, Features 1 through 17  
**Engine**: HyperStream Bare-Metal Vulkan LLM Inference Engine  
**Integrity Standard**: Production Systems Engineering (Zero-Emoji, Strict Technical Precision)

---

## 1. Test Philosophy & Opaque-Box Methodology

The HyperStream End-to-End (E2E) test suite is architected as an authoritative, requirement-driven, opaque-box verification system. Its primary mission is to provide rigorous mathematical, structural, and behavioral guarantees across all seven core requirements (R1–R7) without coupling test logic to internal implementation minutiae.

### 1.1 Core Principles
1. **Opaque-Box Specification Verification**:
   Tests interact exclusively with public interface contracts defined in `PROJECT.md`, standard network protocols (HTTP/1.1 and Server-Sent Events), standard binary file formats (GGUF v3), and observable GPU execution properties (dispatch counts, specialization constants, buffer allocations).
2. **Authoritative Expected Output Derivation**:
   Every expected output is derived from first principles:
   - *GGUF & Tokenizer (R1)*: Official GGUF v3 binary specification and HuggingFace/GPT-2 BPE byte-fallback merge standards, cross-verified against real model checkpoints (`qwen2.5-coder-0.5b-instruct-q8_0.gguf` and `qwen2.5-coder-7b-instruct-q4_0.gguf`).
   - *Model Topology & Specialization (R2)*: Exact architectural dimensions derived from GGUF metadata key-value pairs (`general.architecture`, `*.block_count`, `*.embedding_length`, `*.attention.head_count`) mapped to standard `VkSpecializationInfo` data layouts.
   - *Mega-Kernel Execution Graph (R3)*: Mathematical equivalence between sequential operations (RMSNorm -> QKV GEMV -> RoPE, and PostNorm -> SwiGLU -> Down GEMV) and fused single-dispatch equivalents, combined with invariant checking of the 84-dispatch execution graph (3 dispatches per layer * 28 layers).
   - *LDS Wavefront FlashAttention (R4)*: Numerical convergence and numerical stability of FlashAttention-2 online softmax vs. standard two-pass softmax, with verification of 64-thread wavefront LDS layouts and zero VRAM score buffer roundtrips.
   - *Speculative Tree Decoding (R5)*: Graph-theoretical correctness of the 7-candidate speculation tree (depth 3, 2-2-1-1-1 topology) and boolean invariant verification of the 7x7 2D tree-attention mask matrix.
   - *Paged KV Cache (R6)*: Allocation invariants, block-table mapping correctness, fragmentation resistance, and memory footprint bounds (8K context within 8GB VRAM).
   - *Cross-Platform System Layer (R7)*: Socket loopback communication roundtrips, memory-mapped file integrity, and zero-emoji compliance.
3. **Deterministic & Isolated Execution**:
   Each test case sets up its own state, executes deterministically, and cleans up temporary files or memory mappings upon completion. Tests run without cross-test execution order dependencies.
4. **Adversarial & Fault Injection Hardening**:
   The suite systematically probes boundary extremes, corrupted file headers, truncated streams, buffer overruns, socket disconnects, and malformed inputs to ensure bulletproof crash-resilience.

---

## 2. Feature Inventory Mapping to Test Tiers

The 17 architectural features and 7 requirements specified in `PROJECT.md` are mapped across four hierarchical testing tiers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        E2E TESTING TIERS                               │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 1: FEATURE COVERAGE (Unit/E2E Functional Invariants, >=35 Tests)  │
│   R1: GGUF v3 Parser & In-Engine BPE Tokenizer (5+ Tests)              │
│   R2: Dynamic Model Topology & VkSpecializationInfo (5+ Tests)         │
│   R3: Fused Pre-Attn & MLP Mega-Kernels, 84 Dispatches (5+ Tests)      │
│   R4: LDS Wavefront FlashAttention & Online Softmax (5+ Tests)         │
│   R5: 7-Candidate Spec-Tree & 2D Mask Verification (5+ Tests)          │
│   R6: Paged KV Cache Dynamic Allocation & 8K Limit (5+ Tests)          │
│   R7: Cross-Platform Sockets, Mmap & ASCII Logging (5+ Tests)          │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 2: BOUNDARY & CORNER CASES (Extreme Conditions, >=35 Tests)       │
│   Corrupted GGUF, zero/max dimensions, exp overflow, tree edge cases,  │
│   cache exhaustion, network disconnects, emoji fuzzing (5+ per R)      │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 3: CROSS-FEATURE COMBINATIONS (Pairwise Interactions, >=10 Tests) │
│   R1+R2: GGUF to dynamic topology pipeline                             │
│   R2+R3: Dynamic topology to mega-kernel push constants                │
│   R4+R6: FlashAttention through paged KV block table translation       │
│   R5+R6: Speculative tree branch rollback and cache block fork         │
│   R1+R7: Network prompt ingestion to tokenization to streaming         │
├────────────────────────────────────────────────────────────────────────┤
│ TIER 4: REAL-WORLD APPLICATION SCENARIOS (E2E Workflows, >=8 Tests)    │
│   Multi-turn ChatML conversations, streaming HTTP/SSE generation,      │
│   real GGUF model ingestion, speculative decode throughput benchmark   │
└────────────────────────────────────────────────────────────────────────┘
```

### Detailed Feature-to-Tier Matrix

| Feature ID | Description | Source Req | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **F1** | Native GGUF v3 Parser | R1 | T1.1 | T2.1 | T3.1 | T4.3 |
| **F2** | In-Engine BPE Tokenizer | R1 | T1.2 | T2.2 | T3.7 | T4.1 |
| **F3** | Elimination of Python/Bin | R1 | T1.3 | T2.3 | T3.1 | T4.3 |
| **F4** | Cross-Platform Socket Layer | R7 | T1.7 | T2.7 | T3.7 | T4.2 |
| **F5** | Cross-Platform Mmap File | R7 | T1.7 | T2.7 | T3.1 | T4.3 |
| **F6** | Zero-Emoji Logging Standard | R7 | T1.7 | T2.7 | T3.9 | T4.8 |
| **F7** | Dynamic Model Topology | R2 | T1.4 | T2.4 | T3.1 | T4.3 |
| **F8** | Vulkan Specialization Constants | R2 | T1.5 | T2.5 | T3.2 | T4.7 |
| **F9** | Fused Pre-Attention Mega-Kernel | R3 | T1.8 | T2.8 | T3.2 | T4.7 |
| **F10** | Fused MLP Mega-Kernel | R3 | T1.9 | T2.9 | T3.6 | T4.7 |
| **F11** | Dispatch Count Reduction (84) | R3 | T1.10 | T2.10 | T3.6 | T4.4 |
| **F12** | LDS Wavefront FlashAttention | R4 | T1.11 | T2.11 | T3.4 | T4.7 |
| **F13** | Paged KV Cache (8K Context) | R6 | T1.12 | T2.12 | T3.4 | T4.5 |
| **F14** | 7-Candidate Spec-Tree Generation | R5 | T1.13 | T2.13 | T3.5 | T4.4 |
| **F15** | 2D Tree-Attention Mask Verification| R5 | T1.14 | T2.14 | T3.5 | T4.4 |
| **F16** | Modern CMake & CTest Build | R7 | T1.15 | T2.15 | T3.9 | T4.8 |
| **F17** | 100% E2E Pass & Hardening | R7 | ALL | ALL | ALL | ALL |

---

## 3. Test Architecture & Directory Layout

The E2E test framework is housed completely under `H:\HyperStream\tests\e2e\`:

```
H:\HyperStream\tests\e2e\
├── common\
│   ├── e2e_test_framework.hpp      # High-precision test harness & assertion macros
│   ├── gguf_test_utils.hpp         # Standalone synthetic GGUF v3 binary generator
│   └── test_constants.hpp          # Common tokens, topologies, and tolerances
├── tier1_feature_coverage\
│   ├── test_tier1_r1_gguf_tokenizer.cpp  # >=5 tests for GGUF & BPE
│   ├── test_tier1_r2_dynamic_topology.cpp# >=5 tests for dynamic model topology & spec constants
│   ├── test_tier1_r3_mega_kernels.cpp    # >=5 tests for mega-kernels & 84 dispatches
│   ├── test_tier1_r4_flash_attention.cpp # >=5 tests for LDS wavefront FlashAttention
│   ├── test_tier1_r5_spec_tree.cpp       # >=5 tests for 7-candidate tree & 2D mask
│   ├── test_tier1_r6_paged_kv.cpp        # >=5 tests for paged KV cache allocation
│   └── test_tier1_r7_platform_build.cpp  # >=5 tests for sockets, mmap & zero-emoji
├── tier2_boundary_corner\
│   ├── test_tier2_r1_gguf_boundary.cpp   # Corrupted magic, bad alignment, unknown types
│   ├── test_tier2_r2_topology_boundary.cpp# Zero layers, tied weights, odd head counts
│   ├── test_tier2_r3_mega_boundary.cpp   # Epsilon underflow, extreme Q4 scales
│   ├── test_tier2_r4_flash_boundary.cpp  # Length=1, length=8192, extreme logits
│   ├── test_tier2_r5_spec_boundary.cpp   # Zero accepted, all accepted, cyclic branches
│   ├── test_tier2_r6_paged_boundary.cpp  # OOM block exhaustion, free thrashing
│   └── test_tier2_r7_platform_boundary.cpp# Socket collision, abrupt EOF, emoji fuzzing
├── tier3_cross_feature\
│   └── test_tier3_cross_feature.cpp      # Pairwise cross-feature interaction suite (10 tests)
├── tier4_real_world\
│   └── test_tier4_real_world.cpp         # Real-world chat, SSE, GGUF models, throughput (8 tests)
└── run_all_e2e.ps1                       # Master test runner script
```

---

## 4. Test Harness Execution Instructions

### 4.1 Prerequisites
- C++17 compliant compiler: MinGW-w64 `g++` (v11+), Clang, or MSVC.
- PowerShell 5.1+ or PowerShell 7+ on Windows (or bash on Linux).
- Optional: Real GGUF models under `H:\ai model\` or `H:\HyperStream\models\` for deep validation.

### 4.2 Executing All E2E Tests
To execute the complete E2E test suite with full reporting:

```powershell
pwsh -ExecutionPolicy Bypass -File tests\e2e\run_all_e2e.ps1
```

### 4.3 Selective Tier & Feature Execution
The runner provides flexible switches for targeted execution:

```powershell
# Run only Tier 1 Feature Coverage
pwsh tests\e2e\run_all_e2e.ps1 -Tier 1

# Run only Tier 2 Boundary & Corner Cases
pwsh tests\e2e\run_all_e2e.ps1 -Tier 2

# Run only Tier 3 Cross-Feature Combinations
pwsh tests\e2e\run_all_e2e.ps1 -Tier 3

# Run only Tier 4 Real-World Application Scenarios
pwsh tests\e2e\run_all_e2e.ps1 -Tier 4

# Run with verbose diagnostic output
pwsh tests\e2e\run_all_e2e.ps1 -Verbose

# Filter by keyword / feature name (e.g. FlashAttention or SpecTree)
pwsh tests\e2e\run_all_e2e.ps1 -Filter "flash_attention"
```

### 4.4 Exit Codes & Automated Gating
- **Exit Code 0**: 100% of executed test cases passed with zero errors or integrity violations.
- **Exit Code 1**: One or more test assertions failed, a crash occurred, or compilation failed.
- The runner emits structured ASCII logs adhering strictly to `[TEST]`, `[PASS]`, and `[FAIL]` prefixes.
