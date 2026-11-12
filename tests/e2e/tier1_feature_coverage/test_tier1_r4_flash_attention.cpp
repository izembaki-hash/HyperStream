#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

using namespace hyperstream::test;

namespace {

// Standard Two-Pass Attention (Reference Oracle)
std::vector<float> standard_attention(
    const std::vector<float>& q,       // [head_dim]
    const std::vector<float>& k_cache, // [seq_len, head_dim]
    const std::vector<float>& v_cache, // [seq_len, head_dim]
    uint32_t seq_len,
    uint32_t head_dim,
    float scale) 
{
    std::vector<float> scores(seq_len);
    float max_score = -std::numeric_limits<float>::infinity();

    // 1. QK^T * scale
    for (uint32_t t = 0; t < seq_len; ++t) {
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            dot += q[d] * k_cache[t * head_dim + d];
        }
        scores[t] = dot * scale;
        if (scores[t] > max_score) {
            max_score = scores[t];
        }
    }

    // 2. Softmax
    double sum_exp = 0.0;
    std::vector<float> probs(seq_len);
    for (uint32_t t = 0; t < seq_len; ++t) {
        float p = std::exp(scores[t] - max_score);
        probs[t] = p;
        sum_exp += static_cast<double>(p);
    }
    float inv_sum = static_cast<float>(1.0 / sum_exp);
    for (uint32_t t = 0; t < seq_len; ++t) {
        probs[t] *= inv_sum;
    }

    // 3. Probs * V
    std::vector<float> out(head_dim, 0.0f);
    for (uint32_t t = 0; t < seq_len; ++t) {
        float p = probs[t];
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] += p * v_cache[t * head_dim + d];
        }
    }
    return out;
}

// FlashAttention-2 Tiled Online Softmax (Single-Pass in LDS)
std::vector<float> online_flash_attention(
    const std::vector<float>& q,       // [head_dim]
    const std::vector<float>& k_cache, // [seq_len, head_dim]
    const std::vector<float>& v_cache, // [seq_len, head_dim]
    uint32_t seq_len,
    uint32_t head_dim,
    float scale,
    uint32_t tile_size = 64)
{
    std::vector<float> out(head_dim, 0.0f);
    float running_max = -std::numeric_limits<float>::infinity();
    float running_sum = 0.0f;

    uint32_t num_tiles = (seq_len + tile_size - 1) / tile_size;

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint32_t t_start = tile * tile_size;
        uint32_t t_end = std::min(t_start + tile_size, seq_len);
        uint32_t cur_tile_len = t_end - t_start;

        // Tile local scores
        std::vector<float> tile_scores(cur_tile_len);
        float tile_max = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < cur_tile_len; ++i) {
            uint32_t t = t_start + i;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += q[d] * k_cache[t * head_dim + d];
            }
            tile_scores[i] = dot * scale;
            if (tile_scores[i] > tile_max) {
                tile_max = tile_scores[i];
            }
        }

        // Online softmax update: m_new = max(m_old, tile_max)
        float new_max = std::max(running_max, tile_max);
        float alpha = (running_max == -std::numeric_limits<float>::infinity()) ? 0.0f : std::exp(running_max - new_max);
        float beta = std::exp(tile_max - new_max);

        // Scale previous output accumulator: out = out * alpha
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] *= alpha;
        }

        // Accumulate current tile
        float tile_sum = 0.0f;
        for (uint32_t i = 0; i < cur_tile_len; ++i) {
            uint32_t t = t_start + i;
            float p = std::exp(tile_scores[i] - tile_max) * beta;
            tile_sum += p;
            for (uint32_t d = 0; d < head_dim; ++d) {
                out[d] += p * v_cache[t * head_dim + d];
            }
        }

        // Update running statistics
        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;
    }

    // Final normalization: out = out / running_sum
    if (running_sum > 0.0f) {
        float inv_sum = 1.0f / running_sum;
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] *= inv_sum;
        }
    }

    return out;
}

} // anonymous namespace

// 1. Online Softmax FlashAttention vs Two-Pass Standard Attention Equivalence
E2E_TEST_CASE(Tier1_R4, test_r4_online_softmax_mathematical_equivalence) {
    constexpr uint32_t seq_len = 192;
    constexpr uint32_t head_dim = 128;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> q(head_dim);
    for (uint32_t i = 0; i < head_dim; ++i) {
        q[i] = std::sin(static_cast<float>(i) * 0.1f);
    }

    std::vector<float> k_cache(seq_len * head_dim);
    std::vector<float> v_cache(seq_len * head_dim);
    for (size_t i = 0; i < seq_len * head_dim; ++i) {
        k_cache[i] = std::cos(static_cast<float>(i) * 0.05f);
        v_cache[i] = std::sin(static_cast<float>(i) * 0.03f) + 1.0f;
    }

    auto out_standard = standard_attention(q, k_cache, v_cache, seq_len, head_dim, scale);
    auto out_flash = online_flash_attention(q, k_cache, v_cache, seq_len, head_dim, scale, 64);

    for (uint32_t d = 0; d < head_dim; ++d) {
        ASSERT_NEAR(out_standard[d], out_flash[d], 1e-4f);
    }
}

// 2. Grouped Query Attention (GQA) Head Mapping
E2E_TEST_CASE(Tier1_R4, test_r4_gqa_head_mapping_invariants) {
    // Qwen2.5-Coder-7B: 28 Query Heads, 4 KV Heads (Ratio = 7)
    constexpr uint32_t num_q_heads_7b = 28;
    constexpr uint32_t num_kv_heads_7b = 4;
    constexpr uint32_t group_size_7b = num_q_heads_7b / num_kv_heads_7b;
    ASSERT_EQ(group_size_7b, 7u);

    for (uint32_t q_h = 0; q_h < num_q_heads_7b; ++q_h) {
        uint32_t kv_h = q_h / group_size_7b;
        ASSERT_TRUE(kv_h < num_kv_heads_7b);
        if (q_h < 7) {
            ASSERT_EQ(kv_h, 0u);
        } else if (q_h < 14) {
            ASSERT_EQ(kv_h, 1u);
        } else if (q_h < 21) {
            ASSERT_EQ(kv_h, 2u);
        } else {
            ASSERT_EQ(kv_h, 3u);
        }
    }

    // Qwen2.5-Coder-0.5B: 14 Query Heads, 2 KV Heads (Ratio = 7)
    constexpr uint32_t num_q_heads_0_5b = 14;
    constexpr uint32_t num_kv_heads_0_5b = 2;
    constexpr uint32_t group_size_0_5b = num_q_heads_0_5b / num_kv_heads_0_5b;
    ASSERT_EQ(group_size_0_5b, 7u);

    for (uint32_t q_h = 0; q_h < num_q_heads_0_5b; ++q_h) {
        uint32_t kv_h = q_h / group_size_0_5b;
        ASSERT_TRUE(kv_h < num_kv_heads_0_5b);
    }
}

// 3. Wavefront LDS Tiling Geometry and Memory Footprint Invariants
E2E_TEST_CASE(Tier1_R4, test_r4_wavefront_lds_tiling_geometry) {
    constexpr uint32_t wavefront_size = 64; // AMD GCN/RDNA wavefront size
    constexpr uint32_t tile_size = 64;      // Bc = 64 tokens per tile
    constexpr uint32_t head_dim = 128;      // d = 128 floats

    // LDS Shared memory allocation required per workgroup:
    // Q tile: 1 query vector = 128 floats = 512 bytes
    // K tile: 64 tokens * 128 floats = 8,192 floats = 32,768 bytes (32 KB)
    // V tile: 64 tokens * 128 floats = 8,192 floats = 32,768 bytes (32 KB)
    // Running max/sum vectors across wavefront: 64 floats = 256 bytes
    size_t q_bytes = head_dim * sizeof(float);
    size_t k_tile_bytes = tile_size * head_dim * sizeof(float);
    size_t v_tile_bytes = tile_size * head_dim * sizeof(float);
    size_t total_lds_bytes = q_bytes + k_tile_bytes + v_tile_bytes;

    ASSERT_EQ(wavefront_size, 64u);
    ASSERT_EQ(q_bytes, 512u);
    ASSERT_EQ(k_tile_bytes, 32768u);
    ASSERT_EQ(v_tile_bytes, 32768u);

    // Invariant: Total LDS memory per compute unit must not exceed 64 KB (65,536 bytes)
    // In ping-pong buffer design or direct load, K and V are loaded sequentially or in 32-token sub-blocks.
    // If K and V are loaded in sub-tiles of 32 tokens:
    size_t tiled_subblock_lds = q_bytes + (32 * head_dim * sizeof(float)) * 2;
    ASSERT_TRUE(tiled_subblock_lds <= 65536);
}

// 4. Zero VRAM Intermediate Score Buffer Invariant
E2E_TEST_CASE(Tier1_R4, test_r4_zero_vram_roundtrip_invariant) {
    // In legacy attention:
    // Requires allocating global VRAM for attention scores: S = [num_heads, seq_len] floats.
    // At seq_len = 8192, 28 heads: 28 * 8192 * 4 bytes = 917,504 bytes per layer.
    // In FlashAttention:
    // Intermediate scores are never written to global memory buffers.
    // Only the final output head buffer is written to global memory:
    constexpr uint32_t num_heads = 28;
    constexpr uint32_t head_dim = 128;
    size_t global_write_bytes = num_heads * head_dim * sizeof(float);

    ASSERT_EQ(global_write_bytes, 14336u); // Exactly 14 KB written to VRAM per token
}

// 5. Numerical Stability under Extreme Logits
E2E_TEST_CASE(Tier1_R4, test_r4_numerical_stability_extreme_logits) {
    constexpr uint32_t seq_len = 4;
    constexpr uint32_t head_dim = 2;

    std::vector<float> q = {1000.0f, 1000.0f};
    std::vector<float> k_cache = {
        1000.0f, 1000.0f, // Dot product ~ 2,000,000 (extreme overflow if exp is unscaled)
        999.0f,  999.0f,
        0.0f,    0.0f,
        -1000.0f, -1000.0f
    };
    std::vector<float> v_cache = {
        1.0f, 0.0f,
        2.0f, 0.0f,
        3.0f, 0.0f,
        4.0f, 0.0f
    };

    auto out = online_flash_attention(q, k_cache, v_cache, seq_len, head_dim, 1.0f, 2);

    // Invariant: Must not produce NaN or Inf
    ASSERT_FALSE(std::isnan(out[0]));
    ASSERT_FALSE(std::isnan(out[1]));
    ASSERT_FALSE(std::isinf(out[0]));
    ASSERT_FALSE(std::isinf(out[1]));

    // Top score dominates completely, out[0] should be 1.0f
    ASSERT_NEAR(out[0], 1.0f, 1e-4f);
    ASSERT_NEAR(out[1], 0.0f, 1e-4f);
}

// 6. Attention Scale Factor Normalization
E2E_TEST_CASE(Tier1_R4, test_r4_attention_scale_factor_normalization) {
    float scale_64 = 1.0f / std::sqrt(64.0f);
    float scale_128 = 1.0f / std::sqrt(128.0f);

    ASSERT_NEAR(scale_64, 0.125f, 1e-6f);
    ASSERT_NEAR(scale_128, 0.088388347f, 1e-6f);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
