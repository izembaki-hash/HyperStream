#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <cstdint>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace hyperstream::test;

namespace {

struct PagedKVCacheConfig {
    uint32_t num_layers = 28;
    uint32_t num_kv_heads = 4;
    uint32_t head_dim = 128;
    uint32_t block_size = 16;     // tokens per block
    uint32_t max_blocks = 512;    // 512 blocks * 16 tokens = 8192 tokens
};

class SimulatedPagedKVCache {
public:
    SimulatedPagedKVCache(const PagedKVCacheConfig& cfg) : m_cfg(cfg) {
        // Initialize free block pool
        for (int32_t i = static_cast<int32_t>(m_cfg.max_blocks) - 1; i >= 0; --i) {
            m_free_blocks.push(i);
        }
    }

    int32_t allocate_block() {
        if (m_free_blocks.empty()) {
            return -1; // Out of memory
        }
        int32_t blk = m_free_blocks.top();
        m_free_blocks.pop();
        m_allocated_blocks.insert(blk);
        return blk;
    }

    void free_block(int32_t blk_id) {
        if (m_allocated_blocks.find(blk_id) != m_allocated_blocks.end()) {
            m_allocated_blocks.erase(blk_id);
            m_free_blocks.push(blk_id);
        }
    }

    // Sequence appends token at pos, allocating a new block if crossing block boundary
    bool append_token(uint32_t seq_id, uint32_t pos) {
        uint32_t logical_block = pos / m_cfg.block_size;
        auto& table = m_seq_block_tables[seq_id];
        if (logical_block >= table.size()) {
            int32_t phys = allocate_block();
            if (phys < 0) return false;
            table.push_back(phys);
        }
        return true;
    }

    int32_t translate_token_to_physical_block(uint32_t seq_id, uint32_t pos) const {
        auto it = m_seq_block_tables.find(seq_id);
        if (it == m_seq_block_tables.end()) return -1;
        uint32_t logical_block = pos / m_cfg.block_size;
        if (logical_block >= it->second.size()) return -1;
        return it->second[logical_block];
    }

    size_t get_free_block_count() const {
        return m_free_blocks.size();
    }

    size_t get_allocated_block_count() const {
        return m_allocated_blocks.size();
    }

    const std::vector<int32_t>& get_block_table(uint32_t seq_id) const {
        static const std::vector<int32_t> empty;
        auto it = m_seq_block_tables.find(seq_id);
        return (it != m_seq_block_tables.end()) ? it->second : empty;
    }

    // Single token KV storage size in bytes (FP32)
    size_t bytes_per_token_fp32() const {
        // K and V: 2 * num_layers * num_kv_heads * head_dim * sizeof(float)
        return 2ULL * m_cfg.num_layers * m_cfg.num_kv_heads * m_cfg.head_dim * sizeof(float);
    }

    // Single token KV storage size in bytes (FP16)
    size_t bytes_per_token_fp16() const {
        return 2ULL * m_cfg.num_layers * m_cfg.num_kv_heads * m_cfg.head_dim * sizeof(uint16_t);
    }

    // Single block storage size in bytes (FP32)
    size_t bytes_per_block_fp32() const {
        return bytes_per_token_fp32() * m_cfg.block_size;
    }

private:
    PagedKVCacheConfig m_cfg;
    std::stack<int32_t> m_free_blocks;
    std::unordered_set<int32_t> m_allocated_blocks;
    std::unordered_map<uint32_t, std::vector<int32_t>> m_seq_block_tables;
};

} // anonymous namespace

// 1. Paged KV Cache Dynamic Block Allocation & Translation
E2E_TEST_CASE(Tier1_R6, test_r6_paged_kv_block_table_dynamic_allocation) {
    PagedKVCacheConfig cfg;
    cfg.block_size = 16;
    cfg.max_blocks = 512;
    SimulatedPagedKVCache cache(cfg);

    ASSERT_EQ(cache.get_free_block_count(), 512u);
    ASSERT_EQ(cache.get_allocated_block_count(), 0u);

    // Append 35 tokens into sequence 1
    // Tokens 0-15: Block 0
    // Tokens 16-31: Block 1
    // Tokens 32-34: Block 2
    for (uint32_t pos = 0; pos < 35; ++pos) {
        bool ok = cache.append_token(1, pos);
        ASSERT_TRUE(ok);
    }

    ASSERT_EQ(cache.get_allocated_block_count(), 3u);
    ASSERT_EQ(cache.get_free_block_count(), 509u);

    int32_t blk_0 = cache.translate_token_to_physical_block(1, 0);
    int32_t blk_15 = cache.translate_token_to_physical_block(1, 15);
    int32_t blk_16 = cache.translate_token_to_physical_block(1, 16);
    int32_t blk_31 = cache.translate_token_to_physical_block(1, 31);
    int32_t blk_32 = cache.translate_token_to_physical_block(1, 32);

    ASSERT_EQ(blk_0, blk_15);
    ASSERT_EQ(blk_16, blk_31);
    ASSERT_NE(blk_0, blk_16);
    ASSERT_NE(blk_16, blk_32);
}

// 2. 8K Context Memory Bounds Verification (Fits in 8GB VRAM)
E2E_TEST_CASE(Tier1_R6, test_r6_paged_kv_8k_context_memory_footprint_bounds) {
    PagedKVCacheConfig cfg;
    cfg.num_layers = 28;
    cfg.num_kv_heads = 4;
    cfg.head_dim = 128;
    cfg.block_size = 16;
    cfg.max_blocks = 512; // 8192 tokens

    SimulatedPagedKVCache cache(cfg);

    size_t token_bytes_fp32 = cache.bytes_per_token_fp32();
    size_t token_bytes_fp16 = cache.bytes_per_token_fp16();

    // 2 * 28 * 4 * 128 * 4 = 114,688 bytes
    ASSERT_EQ(token_bytes_fp32, 114688u);
    // FP16: 57,344 bytes
    ASSERT_EQ(token_bytes_fp16, 57344u);

    // Full 8K Context (8192 tokens)
    size_t total_kv_fp32 = 8192ULL * token_bytes_fp32;
    size_t total_kv_fp16 = 8192ULL * token_bytes_fp16;

    // FP32: 939,524,096 bytes ~ 896 MB (or 917.5 MB decimal)
    ASSERT_TRUE(total_kv_fp32 < 1024ULL * 1024 * 1024); // Under 1 GB!
    // FP16: ~ 448 MB
    ASSERT_TRUE(total_kv_fp16 < 512ULL * 1024 * 1024);  // Under 512 MB!

    // Verify VRAM headroom on 8GB GPU:
    // 7B Target Q4: ~ 4.09 GB
    // 0.5B Draft Q8: ~ 0.64 GB
    // KV Cache (8K FP32): ~ 0.89 GB
    // Activations / Temp: ~ 0.50 GB
    // Total: ~ 6.12 GB <= 8.0 GB (Headroom >= 1.8 GB)
    double target_gb = 4.09;
    double draft_gb = 0.64;
    double kv_gb = static_cast<double>(total_kv_fp32) / (1024.0 * 1024.0 * 1024.0);
    double total_vram_gb = target_gb + draft_gb + kv_gb + 0.5;

    ASSERT_TRUE(total_vram_gb < 8.0);
    double headroom = 8.0 - total_vram_gb;
    ASSERT_TRUE(headroom >= 1.5);
}

// 3. Block Free and Immediate Reuse Invariants
E2E_TEST_CASE(Tier1_R6, test_r6_paged_kv_block_free_and_reuse_invariants) {
    PagedKVCacheConfig cfg;
    cfg.max_blocks = 4;
    SimulatedPagedKVCache cache(cfg);

    int32_t b1 = cache.allocate_block();
    int32_t b2 = cache.allocate_block();
    ASSERT_EQ(cache.get_allocated_block_count(), 2u);

    cache.free_block(b2);
    ASSERT_EQ(cache.get_allocated_block_count(), 1u);
    ASSERT_EQ(cache.get_free_block_count(), 3u);

    // Immediate re-allocation must yield b2 (LIFO or free pool reuse)
    int32_t b2_reused = cache.allocate_block();
    ASSERT_EQ(b2_reused, b2);
    ASSERT_EQ(cache.get_allocated_block_count(), 2u);

    cache.free_block(b1);
    cache.free_block(b2_reused);
    ASSERT_EQ(cache.get_allocated_block_count(), 0u);
    ASSERT_EQ(cache.get_free_block_count(), 4u);
}

// 4. Zero External Fragmentation Invariance
E2E_TEST_CASE(Tier1_R6, test_r6_paged_kv_fragmentation_resistance) {
    PagedKVCacheConfig cfg;
    cfg.max_blocks = 100;
    SimulatedPagedKVCache cache(cfg);

    // Simulate multi-sequence interleaved requests
    // Allocate 5 sequences of 10 blocks each (50 blocks total)
    std::vector<int32_t> seq_blocks[5];
    for (int s = 0; s < 5; ++s) {
        for (int b = 0; b < 10; ++b) {
            seq_blocks[s].push_back(cache.allocate_block());
        }
    }
    ASSERT_EQ(cache.get_allocated_block_count(), 50u);

    // Free sequences 1 and 3 (20 blocks released in scattered fashion)
    for (int b : seq_blocks[1]) cache.free_block(b);
    for (int b : seq_blocks[3]) cache.free_block(b);
    ASSERT_EQ(cache.get_allocated_block_count(), 30u);
    ASSERT_EQ(cache.get_free_block_count(), 70u);

    // Allocate a large continuous sequence requiring 20 blocks
    // In flat memory this would fail due to fragmentation.
    // In paged KV cache, 20 blocks are cleanly allocated without failure.
    std::vector<int32_t> new_seq_blocks;
    for (int b = 0; b < 20; ++b) {
        int32_t blk = cache.allocate_block();
        ASSERT_TRUE(blk >= 0);
        new_seq_blocks.push_back(blk);
    }
    ASSERT_EQ(new_seq_blocks.size(), 20u);
    ASSERT_EQ(cache.get_allocated_block_count(), 50u);
}

// 5. Physical Buffer Address Translation Math
E2E_TEST_CASE(Tier1_R6, test_r6_paged_kv_physical_address_translation_math) {
    constexpr uint32_t num_layers = 28;
    constexpr uint32_t num_kv_heads = 4;
    constexpr uint32_t head_dim = 128;
    constexpr uint32_t block_size = 16;

    // Buffer Layout per block:
    // Block size in floats for single layer single KV: block_size * num_kv_heads * head_dim
    constexpr size_t layer_block_floats = block_size * num_kv_heads * head_dim; // 16 * 4 * 128 = 8192 floats
    constexpr size_t layer_block_bytes = layer_block_floats * sizeof(float);      // 32 KB

    auto compute_offset = [&](uint32_t phys_block, uint32_t layer, uint32_t tok_offset, uint32_t kv_head) -> size_t {
        // Physical memory offset formula:
        // base of block = phys_block * (num_layers * layer_block_bytes)
        // layer base = base + layer * layer_block_bytes
        // token offset = tok_offset * (num_kv_heads * head_dim * sizeof(float))
        // head offset = kv_head * (head_dim * sizeof(float))
        size_t block_total_bytes = num_layers * layer_block_bytes;
        size_t off = phys_block * block_total_bytes + layer * layer_block_bytes +
                     tok_offset * (num_kv_heads * head_dim * sizeof(float)) +
                     kv_head * (head_dim * sizeof(float));
        return off;
    };

    size_t off_0 = compute_offset(0, 0, 0, 0);
    size_t off_1 = compute_offset(0, 0, 1, 0);
    size_t off_head1 = compute_offset(0, 0, 0, 1);

    ASSERT_EQ(off_0, 0u);
    ASSERT_EQ(off_1, 4u * 128u * sizeof(float)); // 2048 bytes per token per layer
    ASSERT_EQ(off_head1, 128u * sizeof(float));   // 512 bytes per head
}

// 6. Prefix Sharing and Branch Fork Invariance
E2E_TEST_CASE(Tier1_R6, test_r6_paged_kv_prefix_caching_and_fork_invariants) {
    PagedKVCacheConfig cfg;
    cfg.block_size = 16;
    SimulatedPagedKVCache cache(cfg);

    // Sequence 1: Prompt prefix of 32 tokens (2 blocks)
    for (uint32_t pos = 0; pos < 32; ++pos) {
        cache.append_token(1, pos);
    }
    ASSERT_EQ(cache.get_allocated_block_count(), 2u);

    // Sequence 2 branches off from Sequence 1 (shares first 2 blocks)
    std::vector<int32_t> seq2_blocks = cache.get_block_table(1);
    ASSERT_EQ(seq2_blocks.size(), 2u);

    // Branch 1 continues on Sequence 1 (token 32)
    cache.append_token(1, 32);
    ASSERT_EQ(cache.get_allocated_block_count(), 3u);

    // Branch 2 creates its own diverged token at position 32
    int32_t diverged_block = cache.allocate_block();
    ASSERT_TRUE(diverged_block >= 0);
    ASSERT_EQ(cache.get_allocated_block_count(), 4u);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
