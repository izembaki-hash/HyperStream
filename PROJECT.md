# Project: HyperStream — High-Performance Bare-Metal Vulkan LLM Inference Engine

## Architecture
HyperStream is a high-performance bare-metal Vulkan LLM inference engine executing quantized GGUF models with zero external Python dependencies, dynamic topology adaptation, fused mega-kernels, LDS wavefront FlashAttention, paged KV caching, speculative tree decoding, and cross-platform socket serving.

### Core Architecture Components
1. **Model & Token Ingestion Layer**: Native C++ GGUF v3 parser and in-engine BPE tokenizer extracting vocabularies, merges, and tensor tables directly from memory-mapped GGUF files.
2. **Topology & Pipeline Specialization Layer**: Dynamic model loader (`model_topology.hpp`) configuring model geometry at runtime and compiling Vulkan compute pipelines with `VkSpecializationInfo` constants.
3. **Mega-Kernel Execution Graph**: Fused compute shaders reducing transformer layer dispatches from 8 down to 3 per layer (84 dispatches total across 28 layers).
4. **Attention & Memory Subsystem**: LDS Wavefront FlashAttention with online softmax and Paged KV Cache managing dynamic blocks for up to 8K context within 8GB VRAM.
5. **Speculative Decoding Engine**: Spec-Tree generator producing a 7-candidate draft tree verified in a single batched target pass via a 2D tree-attention mask.
6. **Cross-Platform Server & Build**: Abstracted socket layer (Winsock2 / POSIX), clean CMake build system, zero-emoji production systems logging, and automated testing suite.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Native GGUF v3 Parser | C++17 parser reading header, metadata KV, tensor headers, and data offsets directly from memory-mapped GGUF | M1 | R1 |
| 2 | In-Engine BPE Tokenizer | Tokenizer parsing vocab strings and merge pairs from GGUF metadata directly into lookup tables | M1 | R1 |
| 3 | Elimination of Python/Bin | Removal of Python scripts (`export_qwen_merges.py`, `generate_qwen_layout_header.py`) and `.bin` lookup files | M1 | R1 |
| 4 | Cross-Platform Socket Layer | Abstracted socket class supporting Windows Winsock2 and Linux POSIX BSD sockets (`src/net/socket.hpp`) | M1 | R7 |
| 5 | Cross-Platform Mmap & Loader | Abstracted memory mapping (`src/util/mmap_file.hpp`) and dynamic Vulkan loading (`dlopen`/`LoadLibrary`) | M1 | R7 |
| 6 | Zero-Emoji Logging Standard | Elimination of all emojis in `src/main.cpp` and `web/web_ui.html`, standardizing on ASCII log prefixes | M1 | R7 |
| 7 | Dynamic Model Topology | Dynamic model loader (`src/model_topology.hpp`) replacing hardcoded layout headers | M2 | R2 |
| 8 | Vulkan Specialization Constants | Pipeline compilation injecting `VkSpecializationInfo` for `HIDDEN_DIM`, `HEAD_DIM`, `NUM_HEADS`, `NUM_KV_HEADS` | M2 | R2 |
| 9 | Fused Pre-Attention Mega-Kernel | RMSNorm + QKV GEMV + RoPE fused into a single dispatch per transformer layer | M3 | R3 |
| 10 | Fused MLP Mega-Kernel | Post-Norm + SwiGLU Gate/Up + Down GEMV fused into a single dispatch per transformer layer | M3 | R3 |
| 11 | Dispatch Count Reduction | Execution graph dispatch count per token reduced from 224 down to 84 dispatches | M3 | R3 |
| 12 | LDS Wavefront FlashAttention | Online softmax tiled across 64-thread wavefronts using 64KB/32KB LDS, eliminating VRAM roundtrips | M4 | R4 |
| 13 | Paged KV Cache | Dynamic block allocation table for KV cache enabling up to 8K context within 8GB VRAM footprint | M4 | R6 |
| 14 | 7-Candidate Spec-Tree Generation | Draft runner generating a 7-candidate speculation tree (depth 3, 2-2-1-1-1 branching) | M5 | R5 |
| 15 | 2D Tree-Attention Mask Verification | Single-pass batched target verification utilizing a 2D tree-attention mask matrix | M5 | R5 |
| 16 | Modern CMake & CTest Build | Complete CMake build configuration supporting MSVC/Clang on Windows and GCC/Clang on Linux | M6 | R7 |
| 17 | 100% E2E Pass & Adversarial Hardening | Verification against all E2E test tiers (Tiers 1-4) followed by Tier 5 adversarial coverage hardening | M6 | R7 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | GGUF Parser, Tokenizer & Platform Abstraction | Native GGUF v3 parser, BPE tokenizer from metadata, cross-platform socket/mmap, emoji sanitation | None | PLANNED |
| M2 | Dynamic Model Topology & Specialization Constants | Dynamic model loader (`model_topology.hpp`), `VkSpecializationInfo` in pipeline creation | M1 | PLANNED |
| M3 | Fused Mega-Kernel Graph Architecture | Pre-Attention mega-kernel, MLP mega-kernel, reduction to 84 dispatches per token | M2 | PLANNED |
| M4 | LDS Wavefront FlashAttention & Paged KV Cache | Online softmax tiled over 64-thread wavefronts with LDS, paged KV cache table (8K context) | M2 | PLANNED |
| M5 | Speculative Tree Decoding & 2D Mask Verification | 7-candidate tree generation in draft runner, 2D tree-attention mask, batched single-pass verification | M3, M4 | PLANNED |
| M6 | Cross-Platform Build, Integration & E2E Verification | Modern CMake build, unified engine execution, pass 100% of E2E test suite (Tiers 1-5) | M1, M2, M3, M4, M5, TEST_READY | PLANNED |

---

## Code Layout
```
H:\HyperStream\
├── CMakeLists.txt                  # Top-level modern CMake build script
├── include/                        # Public engine headers
├── src/
│   ├── main.cpp                    # Server entry point, HTTP/SSE loop, speculative loop
│   ├── gguf_parser.hpp             # Native C++ GGUF v3 parser
│   ├── bpe_tokenizer.hpp           # Native in-engine BPE tokenizer interface
│   ├── bpe_tokenizer.cpp           # Native BPE tokenizer implementation
│   ├── model_topology.hpp          # Dynamic model topology configuration
│   ├── net/
│   │   └── socket.hpp              # Cross-platform socket abstraction (Winsock2 / POSIX)
│   ├── util/
│   │   └── mmap_file.hpp           # Cross-platform memory mapping abstraction
│   └── vulkan/
│       ├── vk_context.hpp/cpp      # Vulkan instance, physical/logical device, queue management
│       ├── vk_pipeline.hpp/cpp     # Pipeline creation with VkSpecializationInfo
│       ├── vk_buffer.hpp/cpp       # GPU buffer management and staging
│       ├── vk_dispatch.hpp/cpp     # Dynamic Vulkan loader (Windows / Linux)
│       ├── paged_kv_cache.hpp/cpp  # Paged KV Cache dynamic block table allocator
│       ├── spec_tree.hpp/cpp       # Spec-Tree 7-candidate tree generation and mask builder
│       └── vk_engine_unified.hpp/cpp # Unified dynamic forward inference engine
├── shaders/
│   ├── qwen2_mega_pre_attn.comp    # Fused RMSNorm + QKV GEMV + RoPE mega-kernel
│   ├── qwen2_mega_mlp.comp         # Fused Post-Norm + SwiGLU Gate/Up + Down GEMV mega-kernel
│   ├── qwen2_flash_attention.comp  # LDS Wavefront FlashAttention with online softmax
│   └── qwen2_tree_attention.comp   # 2D Tree-Attention batched verification shader
├── tests/
│   ├── test_gguf_parser.cpp        # Unit test for GGUF v3 parsing
│   ├── test_bpe_tokenizer.cpp      # Unit test for BPE encoding/decoding
│   ├── test_model_topology.cpp     # Unit test for topology configuration
│   ├── test_sockets.cpp            # Unit test for cross-platform sockets
│   ├── test_paged_kv.cpp           # Unit test for paged KV cache allocation
│   ├── test_spec_tree.cpp          # Unit test for 7-candidate tree & 2D mask
│   └── e2e/                        # Comprehensive opaque-box E2E test suite
└── web/
    └── web_ui.html                 # Web chat interface (emoji-free, clean styling)
```

---

## Interface Contracts

### 1. GGUF Parser ↔ Model Topology & Tokenizer
```cpp
namespace hyperstream {
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t type; // GGML type (Q4_0, Q8_0, F32, etc.)
    uint64_t offset;
};

class GGUFParser {
public:
    bool open(const std::string& path);
    bool get_metadata_string(const std::string& key, std::string& val) const;
    bool get_metadata_u32(const std::string& key, uint32_t& val) const;
    bool get_metadata_u64(const std::string& key, uint64_t& val) const;
    bool get_metadata_array_string(const std::string& key, std::vector<std::string>& val) const;
    const GGUFTensorInfo* find_tensor(const std::string& name) const;
    const uint8_t* get_tensor_data(const GGUFTensorInfo& info) const;
    uint64_t get_tensor_data_offset(const GGUFTensorInfo& info) const;
    uint64_t get_data_start_offset() const;
    const uint8_t* data() const;
    size_t size() const;
};
}
```

### 2. Tokenizer Contract
```cpp
namespace hyperstream {
class BPETokenizer {
public:
    bool load_from_gguf(const GGUFParser& parser);
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false) const;
    std::string decode(int32_t token_id) const;
    std::string decode(const std::vector<int32_t>& tokens) const;
    int32_t eos_token() const;
    int32_t bos_token() const;
    int32_t im_start_token() const;
    int32_t im_end_token() const;
    size_t vocab_size() const;
};
}
```

### 3. Model Topology Contract
```cpp
namespace hyperstream {
struct ModelTopology {
    std::string architecture;
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t num_layers = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_freq_base = 10000.0f;
    uint64_t data_start_offset = 0;
    
    // Per-layer tensor offsets relative to data_start_offset
    struct LayerOffsets {
        uint64_t attn_norm = 0;
        uint64_t qkv_weight = 0;
        uint64_t o_weight = 0;
        uint64_t ffn_norm = 0;
        uint64_t gate_up_weight = 0;
        uint64_t down_weight = 0;
    };
    std::vector<LayerOffsets> layers;
    uint64_t token_embd_offset = 0;
    uint64_t output_norm_offset = 0;
    uint64_t output_weight_offset = 0;

    static bool build_from_gguf(const GGUFParser& parser, ModelTopology& topo);
};
}
```

### 4. Paged KV Cache Contract
```cpp
namespace hyperstream {
struct PagedKVCacheConfig {
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t block_size = 16;     // tokens per block
    uint32_t max_blocks = 512;    // up to 8192 tokens
};

class PagedKVCache {
public:
    bool init(VkDevice device, VkPhysicalDeviceMemoryProperties mem_props, const PagedKVCacheConfig& config);
    int32_t allocate_block();
    void free_block(int32_t block_id);
    void append_token(uint32_t seq_id, uint32_t pos);
    VkBuffer get_physical_k_buffer() const;
    VkBuffer get_physical_v_buffer() const;
    VkBuffer get_block_table_buffer() const; // mapping seq_pos -> physical_block_id
};
}
```

### 5. Speculative Tree Contract
```cpp
namespace hyperstream {
struct SpecTreeNode {
    int32_t token_id;
    int32_t parent_idx; // -1 for root
    int32_t depth;
    std::vector<int32_t> children;
};

struct SpecTree {
    static constexpr int TREE_SIZE = 7;
    std::array<SpecTreeNode, TREE_SIZE> nodes;
    std::array<float, TREE_SIZE * TREE_SIZE> attention_mask; // 1.0f where i can attend to j, else 0.0f
    std::vector<std::vector<int32_t>> paths; // distinct generation paths
    
    void build_default_7_tree(const std::vector<int32_t>& drafted_candidates);
};
}
```

### 6. Cross-Platform Socket Contract
```cpp
namespace hyperstream {
class Socket {
public:
    Socket();
    ~Socket();
    bool listen(uint16_t port);
    bool accept(Socket& client);
    int send(const void* data, size_t len);
    int recv(void* data, size_t len);
    void close();
    static bool init_networking();
    static void cleanup_networking();
};
}
```
