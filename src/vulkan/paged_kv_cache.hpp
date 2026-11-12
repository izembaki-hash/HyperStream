#ifndef HYPERSTREAM_PAGED_KV_CACHE_HPP
#define HYPERSTREAM_PAGED_KV_CACHE_HPP

#include <cstdint>
#include <vector>
#include <stack>
#include <unordered_map>
#include <unordered_set>

namespace hyperstream {

struct PagedKVCacheConfig {
    uint32_t num_layers = 28;
    uint32_t num_kv_heads = 4;
    uint32_t head_dim = 128;
    uint32_t block_size = 16;     // tokens per block
    uint32_t max_blocks = 512;    // 512 blocks * 16 tokens = 8192 tokens max context
};

class PagedKVCache {
public:
    explicit PagedKVCache(const PagedKVCacheConfig& cfg = PagedKVCacheConfig()) : m_cfg(cfg) {
        reset();
    }

    void reset() {
        while (!m_free_blocks.empty()) m_free_blocks.pop();
        for (int32_t i = static_cast<int32_t>(m_cfg.max_blocks) - 1; i >= 0; --i) {
            m_free_blocks.push(i);
        }
        m_allocated_blocks.clear();
        m_seq_block_tables.clear();
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

    const PagedKVCacheConfig& get_config() const {
        return m_cfg;
    }

    // Calculate total VRAM consumed by physical blocks in bytes
    // (num_layers * 2 * num_kv_heads * head_dim * sizeof(fp16) * block_size * num_blocks)
    size_t calculate_allocated_bytes() const {
        size_t block_bytes = static_cast<size_t>(m_cfg.num_layers) * 2ULL *
                             static_cast<size_t>(m_cfg.num_kv_heads) *
                             static_cast<size_t>(m_cfg.head_dim) * sizeof(uint16_t) *
                             static_cast<size_t>(m_cfg.block_size);
        return block_bytes * m_allocated_blocks.size();
    }

private:
    PagedKVCacheConfig m_cfg;
    std::stack<int32_t> m_free_blocks;
    std::unordered_set<int32_t> m_allocated_blocks;
    std::unordered_map<uint32_t, std::vector<int32_t>> m_seq_block_tables;
};

} // namespace hyperstream

#endif // HYPERSTREAM_PAGED_KV_CACHE_HPP
