#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

using namespace hyperstream::test;

namespace {

std::vector<float> online_flash_attention_masked(
    const std::vector<float>& q,       // [head_dim]
    const std::vector<float>& k_cache, // [seq_len, head_dim]
    const std::vector<float>& v_cache, // [seq_len, head_dim]
    const std::vector<float>& mask,    // [seq_len], 0.0f = attend, -10000.0f = mask out
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

        std::vector<float> tile_scores(cur_tile_len);
        float tile_max = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < cur_tile_len; ++i) {
            uint32_t t = t_start + i;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += q[d] * k_cache[t * head_dim + d];
            }
            float m = mask.empty() ? 0.0f : mask[t];
            tile_scores[i] = dot * scale + m;
            if (tile_scores[i] > tile_max) {
                tile_max = tile_scores[i];
            }
        }

        float new_max = std::max(running_max, tile_max);
        float alpha = (running_max == -std::numeric_limits<float>::infinity()) ? 0.0f : std::exp(running_max - new_max);
        float beta = std::exp(tile_max - new_max);

        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] *= alpha;
        }

        float tile_sum = 0.0f;
        for (uint32_t i = 0; i < cur_tile_len; ++i) {
            uint32_t t = t_start + i;
            float p = std::exp(tile_scores[i] - tile_max) * beta;
            tile_sum += p;
            for (uint32_t d = 0; d < head_dim; ++d) {
                out[d] += p * v_cache[t * head_dim + d];
            }
        }

        running_sum = running_sum * alpha + tile_sum;
        running_max = new_max;
    }

    if (running_sum > 0.0f) {
        float inv_sum = 1.0f / running_sum;
        for (uint32_t d = 0; d < head_dim; ++d) {
            out[d] *= inv_sum;
        }
    }

    return out;
}

} // anonymous namespace

// 1. Single Token First Step Context (Length = 1)
E2E_TEST_CASE(Tier2_R4, test_r4_single_token_first_step_context) {
    constexpr uint32_t head_dim = 64;
    std::vector<float> q(head_dim, 2.5f);
    std::vector<float> k(head_dim, 1.0f);
    std::vector<float> v(head_dim, 42.0f);

    auto out = online_flash_attention_masked(q, k, v, {}, 1, head_dim, 0.125f, 64);

    // Invariant: At seq_len = 1, attention probability is identically 1.0f, out must equal v exactly
    for (uint32_t d = 0; d < head_dim; ++d) {
        ASSERT_NEAR(out[d], v[d], 1e-5f);
    }
}

// 2. Full Context Length = 8192 Tokens Online Softmax Stability
E2E_TEST_CASE(Tier2_R4, test_r4_full_context_length_8192_online_softmax) {
    constexpr uint32_t seq_len = 8192;
    constexpr uint32_t head_dim = 64;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> q(head_dim, 0.5f);
    std::vector<float> k_cache(seq_len * head_dim);
    std::vector<float> v_cache(seq_len * head_dim);

    // Populate with small regular values
    for (size_t i = 0; i < seq_len * head_dim; ++i) {
        k_cache[i] = static_cast<float>((i % 13) - 6) * 0.1f;
        v_cache[i] = static_cast<float>((i % 7) + 1);
    }

    auto out = online_flash_attention_masked(q, k_cache, v_cache, {}, seq_len, head_dim, scale, 64);

    for (uint32_t d = 0; d < head_dim; ++d) {
        ASSERT_FALSE(std::isnan(out[d]));
        ASSERT_FALSE(std::isinf(out[d]));
        ASSERT_TRUE(out[d] >= 1.0f && out[d] <= 7.0f); // Convex combination of values in [1, 7]
    }
}

// 3. Identical Dot Products Uniform Distribution Mean Invariance
E2E_TEST_CASE(Tier2_R4, test_r4_identical_dot_products_uniform_distribution) {
    constexpr uint32_t seq_len = 100;
    constexpr uint32_t head_dim = 16;

    std::vector<float> q(head_dim, 1.0f);
    std::vector<float> k_cache(seq_len * head_dim, 1.0f); // All identical dot products
    std::vector<float> v_cache(seq_len * head_dim);

    // v_t = t for all elements
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            v_cache[t * head_dim + d] = static_cast<float>(t);
        }
    }

    auto out = online_flash_attention_masked(q, k_cache, v_cache, {}, seq_len, head_dim, 0.25f, 32);

    // Expected: Arithmetic mean of 0..99 = 49.5f
    for (uint32_t d = 0; d < head_dim; ++d) {
        ASSERT_NEAR(out[d], 49.5f, 1e-4f);
    }
}

// 4. Zero Query Vector Invariant
E2E_TEST_CASE(Tier2_R4, test_r4_zero_query_vector_invariance) {
    constexpr uint32_t seq_len = 64;
    constexpr uint32_t head_dim = 32;

    std::vector<float> q(head_dim, 0.0f); // Zero query
    std::vector<float> k_cache(seq_len * head_dim, 1.5f);
    std::vector<float> v_cache(seq_len * head_dim, 3.14f);

    auto out = online_flash_attention_masked(q, k_cache, v_cache, {}, seq_len, head_dim, 0.125f, 64);

    for (uint32_t d = 0; d < head_dim; ++d) {
        ASSERT_NEAR(out[d], 3.14f, 1e-4f);
    }
}

// 5. Causal Mask Attenuation (-10000.0f)
E2E_TEST_CASE(Tier2_R4, test_r4_infinite_causal_mask_rejection) {
    constexpr uint32_t seq_len = 4;
    constexpr uint32_t head_dim = 16;

    std::vector<float> q(head_dim, 1.0f);
    std::vector<float> k(seq_len * head_dim, 1.0f);
    std::vector<float> v(seq_len * head_dim);

    // Token 0: value 100.0f (valid)
    // Tokens 1, 2, 3: value 999.0f (masked out with -10000.0f)
    for (uint32_t d = 0; d < head_dim; ++d) {
        v[0 * head_dim + d] = 100.0f;
        v[1 * head_dim + d] = 999.0f;
        v[2 * head_dim + d] = 999.0f;
        v[3 * head_dim + d] = 999.0f;
    }

    std::vector<float> mask = {0.0f, -10000.0f, -10000.0f, -10000.0f};

    auto out = online_flash_attention_masked(q, k, v, mask, seq_len, head_dim, 0.25f, 64);

    // Invariant: Output must match token 0 value (100.0f), masked tokens must have zero influence
    for (uint32_t d = 0; d < head_dim; ++d) {
        ASSERT_NEAR(out[d], 100.0f, 1e-4f);
    }
}

// 6. Single Non-Masked Token Exact Reconstruction
E2E_TEST_CASE(Tier2_R4, test_r4_single_non_masked_token_reconstruction) {
    constexpr uint32_t seq_len = 8;
    constexpr uint32_t head_dim = 8;

    std::vector<float> q(head_dim, 1.0f);
    std::vector<float> k(seq_len * head_dim, 1.0f);
    std::vector<float> v(seq_len * head_dim, 0.0f);

    // Set token 5 as the only unmasked token
    for (uint32_t d = 0; d < head_dim; ++d) {
        v[5 * head_dim + d] = 77.0f;
    }

    std::vector<float> mask(seq_len, -10000.0f);
    mask[5] = 0.0f; // Only token 5 is visible

    auto out = online_flash_attention_masked(q, k, v, mask, seq_len, head_dim, 0.25f, 4);

    for (uint32_t d = 0; d < head_dim; ++d) {
        ASSERT_NEAR(out[d], 77.0f, 1e-4f);
    }
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
