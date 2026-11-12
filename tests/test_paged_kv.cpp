#include "e2e/common/e2e_test_framework.hpp"
#include "vulkan/paged_kv_cache.hpp"
#include <iostream>

using namespace hyperstream;
using namespace hyperstream::test;

E2E_TEST_CASE(PagedKVTest, test_allocation_and_free) {
    PagedKVCacheConfig cfg;
    cfg.max_blocks = 16;
    cfg.block_size = 16;
    PagedKVCache cache(cfg);

    ASSERT_EQ(cache.get_free_block_count(), 16u);
    ASSERT_EQ(cache.get_allocated_block_count(), 0u);

    int32_t b0 = cache.allocate_block();
    ASSERT_TRUE(b0 >= 0);
    ASSERT_EQ(cache.get_free_block_count(), 15u);
    ASSERT_EQ(cache.get_allocated_block_count(), 1u);

    int32_t b1 = cache.allocate_block();
    ASSERT_TRUE(b1 >= 0);
    ASSERT_NE(b0, b1);

    cache.free_block(b0);
    ASSERT_EQ(cache.get_free_block_count(), 15u);
    ASSERT_EQ(cache.get_allocated_block_count(), 1u);

    cache.free_block(b1);
    ASSERT_EQ(cache.get_free_block_count(), 16u);
    ASSERT_EQ(cache.get_allocated_block_count(), 0u);
}

E2E_TEST_CASE(PagedKVTest, test_append_token_and_translation) {
    PagedKVCacheConfig cfg;
    cfg.block_size = 16;
    cfg.max_blocks = 64;
    PagedKVCache cache(cfg);

    uint32_t seq_id = 1;
    for (uint32_t pos = 0; pos < 40; ++pos) {
        ASSERT_TRUE(cache.append_token(seq_id, pos));
    }

    // 40 tokens / 16 block_size = 3 blocks needed (tokens 0..15, 16..31, 32..39)
    ASSERT_EQ(cache.get_allocated_block_count(), 3u);

    int32_t phys_block_0 = cache.translate_token_to_physical_block(seq_id, 0);
    int32_t phys_block_15 = cache.translate_token_to_physical_block(seq_id, 15);
    int32_t phys_block_16 = cache.translate_token_to_physical_block(seq_id, 16);
    int32_t phys_block_32 = cache.translate_token_to_physical_block(seq_id, 32);

    ASSERT_EQ(phys_block_0, phys_block_15);
    ASSERT_NE(phys_block_0, phys_block_16);
    ASSERT_NE(phys_block_16, phys_block_32);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
