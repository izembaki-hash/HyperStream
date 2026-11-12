#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <cstdint>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

using namespace hyperstream::test;

namespace {

class RobustPagedKVCache {
public:
    RobustPagedKVCache(uint32_t block_size = 16, uint32_t max_blocks = 512)
        : m_block_size(block_size), m_max_blocks(max_blocks) 
    {
        for (int32_t i = static_cast<int32_t>(m_max_blocks) - 1; i >= 0; --i) {
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

    bool free_block(int32_t blk_id) {
        // Validation: Block ID within bounds and currently allocated
        if (blk_id < 0 || blk_id >= static_cast<int32_t>(m_max_blocks)) {
            return false; // Out of range
        }
        if (m_allocated_blocks.find(blk_id) == m_allocated_blocks.end()) {
            return false; // Double free or freeing unallocated block
        }
        m_allocated_blocks.erase(blk_id);
        m_free_blocks.push(blk_id);
        return true;
    }

    bool append_token(uint32_t seq_id, uint32_t pos) {
        // Enforce max sequence length (e.g. 8192 tokens = max_blocks * block_size)
        if (pos >= m_max_blocks * m_block_size) {
            return false; // Exceeds context window limit
        }
        uint32_t logical_block = pos / m_block_size;
        auto& table = m_seq_block_tables[seq_id];
        if (logical_block >= table.size()) {
            int32_t phys = allocate_block();
            if (phys < 0) return false;
            table.push_back(phys);
        }
        return true;
    }

    size_t get_free_count() const { return m_free_blocks.size(); }
    size_t get_allocated_count() const { return m_allocated_blocks.size(); }
    uint32_t get_max_blocks() const { return m_max_blocks; }

private:
    uint32_t m_block_size;
    uint32_t m_max_blocks;
    std::stack<int32_t> m_free_blocks;
    std::unordered_set<int32_t> m_allocated_blocks;
    std::unordered_map<uint32_t, std::vector<int32_t>> m_seq_block_tables;
};

} // anonymous namespace

// 1. OOM Block Pool Exhaustion Detection
E2E_TEST_CASE(Tier2_R6, test_r6_oom_block_exhaustion_detection) {
    RobustPagedKVCache cache(16, 5); // Small pool of 5 blocks
    ASSERT_EQ(cache.get_free_count(), 5u);

    std::vector<int32_t> allocated;
    for (int i = 0; i < 5; ++i) {
        int32_t blk = cache.allocate_block();
        ASSERT_TRUE(blk >= 0);
        allocated.push_back(blk);
    }
    ASSERT_EQ(cache.get_free_count(), 0u);
    ASSERT_EQ(cache.get_allocated_count(), 5u);

    // 6th allocation must fail gracefully with -1
    int32_t oom_blk = cache.allocate_block();
    ASSERT_EQ(oom_blk, -1);
    ASSERT_EQ(cache.get_allocated_count(), 5u);
}

// 2. Double Free Protection
E2E_TEST_CASE(Tier2_R6, test_r6_double_free_protection) {
    RobustPagedKVCache cache(16, 4);
    int32_t blk = cache.allocate_block();
    ASSERT_TRUE(blk >= 0);
    ASSERT_EQ(cache.get_allocated_count(), 1u);

    // First free succeeds
    bool ok1 = cache.free_block(blk);
    ASSERT_TRUE(ok1);
    ASSERT_EQ(cache.get_allocated_count(), 0u);
    ASSERT_EQ(cache.get_free_count(), 4u);

    // Second free of the same block must be safely rejected
    bool ok2 = cache.free_block(blk);
    ASSERT_FALSE(ok2);
    // Free pool count must not exceed maximum capacity
    ASSERT_EQ(cache.get_free_count(), 4u);
}

// 3. Invalid Block ID Free Rejection
E2E_TEST_CASE(Tier2_R6, test_r6_invalid_block_id_free_rejection) {
    RobustPagedKVCache cache(16, 10);
    ASSERT_FALSE(cache.free_block(-1));
    ASSERT_FALSE(cache.free_block(-999));
    ASSERT_FALSE(cache.free_block(10));  // Out of bounds (0-9)
    ASSERT_FALSE(cache.free_block(100)); // Out of bounds
}

// 4. Token Append Across Block Boundaries
E2E_TEST_CASE(Tier2_R6, test_r6_token_append_across_block_boundaries) {
    RobustPagedKVCache cache(16, 10);

    // Append tokens 0 to 15 (First block: 16 tokens)
    for (uint32_t i = 0; i <= 15; ++i) {
        ASSERT_TRUE(cache.append_token(1, i));
    }
    ASSERT_EQ(cache.get_allocated_count(), 1u);

    // Crossing boundary to token 16 allocates second block
    ASSERT_TRUE(cache.append_token(1, 16));
    ASSERT_EQ(cache.get_allocated_count(), 2u);

    // Fill second block (17..31)
    for (uint32_t i = 17; i <= 31; ++i) {
        ASSERT_TRUE(cache.append_token(1, i));
    }
    ASSERT_EQ(cache.get_allocated_count(), 2u);

    // Crossing boundary to token 32 allocates third block
    ASSERT_TRUE(cache.append_token(1, 32));
    ASSERT_EQ(cache.get_allocated_count(), 3u);
}

// 5. Rapid Thrashing Allocation & Deallocation
E2E_TEST_CASE(Tier2_R6, test_r6_rapid_thrashing_allocation_free) {
    RobustPagedKVCache cache(16, 64);

    for (int cycle = 0; cycle < 1000; ++cycle) {
        int32_t blk1 = cache.allocate_block();
        int32_t blk2 = cache.allocate_block();
        ASSERT_TRUE(blk1 >= 0);
        ASSERT_TRUE(blk2 >= 0);

        ASSERT_TRUE(cache.free_block(blk1));
        ASSERT_TRUE(cache.free_block(blk2));
    }

    ASSERT_EQ(cache.get_allocated_count(), 0u);
    ASSERT_EQ(cache.get_free_count(), 64u);
}

// 6. Context Limit Saturation (Exact 8192 Tokens)
E2E_TEST_CASE(Tier2_R6, test_r6_exact_8192_token_limit_saturation) {
    RobustPagedKVCache cache(16, 512); // Exactly 8192 tokens capacity

    // Fill sequence up to 8192 tokens (0 to 8191)
    for (uint32_t pos = 0; pos < 8192; pos += 16) {
        ASSERT_TRUE(cache.append_token(1, pos));
    }
    ASSERT_EQ(cache.get_allocated_count(), 512u);
    ASSERT_EQ(cache.get_free_count(), 0u);

    // Token 8192 (exceeding 8K context limit) must be rejected
    ASSERT_FALSE(cache.append_token(1, 8192));
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
