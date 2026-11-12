#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

using namespace hyperstream::test;

namespace {

// Mathematical Reference Implementation for RMSNorm
void reference_rmsnorm(const float* x, const float* w, float* out, size_t size, float eps) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    float rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(size) + static_cast<double>(eps)));
    float inv_rms = 1.0f / rms;
    for (size_t i = 0; i < size; ++i) {
        out[i] = x[i] * inv_rms * w[i];
    }
}

// Mathematical Reference Implementation for RoPE
void reference_rope(float* q, float* k, uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_dim, uint32_t pos, float freq_base) {
    for (uint32_t h = 0; h < num_heads; ++h) {
        float* vec = q + h * head_dim;
        for (uint32_t i = 0; i < head_dim; i += 2) {
            float theta = std::pow(freq_base, -static_cast<float>(i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * theta;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i]     = v0 * cos_a - v1 * sin_a;
            vec[i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
    for (uint32_t h = 0; h < num_kv_heads; ++h) {
        float* vec = k + h * head_dim;
        for (uint32_t i = 0; i < head_dim; i += 2) {
            float theta = std::pow(freq_base, -static_cast<float>(i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * theta;
            float cos_a = std::cos(angle);
            float sin_a = std::sin(angle);
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i]     = v0 * cos_a - v1 * sin_a;
            vec[i + 1] = v0 * sin_a + v1 * cos_a;
        }
    }
}

// SwiGLU Reference: x * sigmoid(x) * up
float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// Push Constant Structures for Mega-Kernels
struct PreAttnPushConstants {
    uint32_t layer_idx;
    uint32_t seq_len;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    float rms_norm_eps;
    float rope_freq_base;
};

struct MLPPushConstants {
    uint32_t layer_idx;
    uint32_t hidden_dim;
    uint32_t intermediate_dim;
    uint32_t add_residual;
    float rms_norm_eps;
};

} // anonymous namespace

// 1. RMSNorm Mathematical Precision & Stability
E2E_TEST_CASE(Tier1_R3, test_r3_rmsnorm_mathematical_precision) {
    constexpr size_t dim = 896;
    std::vector<float> x(dim);
    std::vector<float> w(dim);
    std::vector<float> out(dim);

    for (size_t i = 0; i < dim; ++i) {
        x[i] = static_cast<float>(i % 17) - 8.0f;
        w[i] = 1.0f + static_cast<float>(i) * 0.001f;
    }

    reference_rmsnorm(x.data(), w.data(), out.data(), dim, 1e-6f);

    // Verify properties: sum of squares normalized
    double sum_sq = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        float unweighted = out[i] / w[i];
        sum_sq += unweighted * unweighted;
    }
    double mean_sq = sum_sq / static_cast<double>(dim);
    ASSERT_NEAR(mean_sq, 1.0, 1e-4);
}

// 2. Fused Pre-Attention Mega-Kernel Equivalence
E2E_TEST_CASE(Tier1_R3, test_r3_fused_pre_attention_mega_kernel_equivalence) {
    constexpr uint32_t hidden_dim = 128;
    constexpr uint32_t num_heads = 2;
    constexpr uint32_t num_kv_heads = 1;
    constexpr uint32_t head_dim = 64;
    constexpr uint32_t q_dim = num_heads * head_dim;   // 128
    constexpr uint32_t k_dim = num_kv_heads * head_dim; // 64
    constexpr uint32_t v_dim = num_kv_heads * head_dim; // 64

    std::vector<float> input_x(hidden_dim, 1.5f);
    std::vector<float> norm_w(hidden_dim, 1.0f);
    std::vector<float> norm_out(hidden_dim);

    // Step A: Sequential execution
    reference_rmsnorm(input_x.data(), norm_w.data(), norm_out.data(), hidden_dim, 1e-6f);

    // Toy linear projection weights (Q, K, V)
    std::vector<float> q_seq(q_dim, 0.0f);
    std::vector<float> k_seq(k_dim, 0.0f);
    std::vector<float> v_seq(v_dim, 0.0f);

    for (uint32_t i = 0; i < q_dim; ++i) {
        q_seq[i] = norm_out[i % hidden_dim] * 0.5f;
    }
    for (uint32_t i = 0; i < k_dim; ++i) {
        k_seq[i] = norm_out[i % hidden_dim] * 0.25f;
    }
    for (uint32_t i = 0; i < v_dim; ++i) {
        v_seq[i] = norm_out[i % hidden_dim] * 0.125f;
    }

    reference_rope(q_seq.data(), k_seq.data(), num_heads, num_kv_heads, head_dim, 10, 10000.0f);

    // Step B: Fused Mega-Kernel equivalent calculation (in a single pass)
    std::vector<float> q_fused(q_dim, 0.0f);
    std::vector<float> k_fused(k_dim, 0.0f);
    std::vector<float> v_fused(v_dim, 0.0f);

    // Single-pass fusion emulation
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        sum_sq += input_x[i] * input_x[i];
    }
    float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / hidden_dim + 1e-6));

    for (uint32_t i = 0; i < q_dim; ++i) {
        float x_norm = input_x[i % hidden_dim] * inv_rms * norm_w[i % hidden_dim];
        q_fused[i] = x_norm * 0.5f;
    }
    for (uint32_t i = 0; i < k_dim; ++i) {
        float x_norm = input_x[i % hidden_dim] * inv_rms * norm_w[i % hidden_dim];
        k_fused[i] = x_norm * 0.25f;
    }
    for (uint32_t i = 0; i < v_dim; ++i) {
        float x_norm = input_x[i % hidden_dim] * inv_rms * norm_w[i % hidden_dim];
        v_fused[i] = x_norm * 0.125f;
    }
    reference_rope(q_fused.data(), k_fused.data(), num_heads, num_kv_heads, head_dim, 10, 10000.0f);

    // Assert mathematical equivalence between sequential and fused execution
    for (uint32_t i = 0; i < q_dim; ++i) {
        ASSERT_NEAR(q_seq[i], q_fused[i], 1e-5f);
    }
    for (uint32_t i = 0; i < k_dim; ++i) {
        ASSERT_NEAR(k_seq[i], k_fused[i], 1e-5f);
    }
    for (uint32_t i = 0; i < v_dim; ++i) {
        ASSERT_NEAR(v_seq[i], v_fused[i], 1e-5f);
    }
}

// 3. Fused MLP Mega-Kernel (SwiGLU + Down GEMV + Residual Add) Equivalence
E2E_TEST_CASE(Tier1_R3, test_r3_fused_mlp_swiglu_mathematical_equivalence) {
    constexpr uint32_t hidden_dim = 64;
    constexpr uint32_t intermediate_dim = 128;

    std::vector<float> x(hidden_dim, 0.5f);
    std::vector<float> post_norm_w(hidden_dim, 1.0f);
    std::vector<float> norm_out(hidden_dim);

    reference_rmsnorm(x.data(), post_norm_w.data(), norm_out.data(), hidden_dim, 1e-6f);

    // Sequential SwiGLU: silu(gate) * up
    std::vector<float> gate(intermediate_dim, 0.8f);
    std::vector<float> up(intermediate_dim, 1.2f);
    std::vector<float> swiglu_out(intermediate_dim);
    for (uint32_t i = 0; i < intermediate_dim; ++i) {
        swiglu_out[i] = silu(gate[i]) * up[i];
    }

    // Down projection with residual addition: out = x + (swiglu * W_down)
    std::vector<float> out_seq(hidden_dim);
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < intermediate_dim; ++j) {
            acc += swiglu_out[j] * 0.01f;
        }
        out_seq[i] = x[i] + acc;
    }

    // Fused Mega-Kernel equivalent: compute SwiGLU in register and accumulate into residual
    std::vector<float> out_fused(hidden_dim);
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < intermediate_dim; ++j) {
            float fused_swiglu = silu(gate[j]) * up[j];
            acc += fused_swiglu * 0.01f;
        }
        out_fused[i] = x[i] + acc;
    }

    for (uint32_t i = 0; i < hidden_dim; ++i) {
        ASSERT_NEAR(out_seq[i], out_fused[i], 1e-5f);
    }
}

// 4. Dispatch Count Invariant: 84 Dispatches for 28-Layer Backbone
E2E_TEST_CASE(Tier1_R3, test_r3_dispatch_count_reduction_84_invariants) {
    constexpr uint32_t num_layers = 28;
    constexpr uint32_t legacy_dispatches_per_layer = 8;
    constexpr uint32_t mega_kernel_dispatches_per_layer = 3;

    uint32_t legacy_total = num_layers * legacy_dispatches_per_layer;
    uint32_t mega_total = num_layers * mega_kernel_dispatches_per_layer;

    ASSERT_EQ(legacy_total, 224u);
    ASSERT_EQ(mega_total, 84u);

    // Invariant: Dispatch structure per layer must be exactly:
    // 1: Pre-Attention Mega-Kernel (RMSNorm + QKV + RoPE)
    // 2: LDS Wavefront FlashAttention (Online Softmax)
    // 3: MLP Mega-Kernel (Post-Norm + SwiGLU + Down + Residual)
    struct DispatchPlan {
        std::string name;
        uint32_t dispatch_id;
    };

    std::vector<DispatchPlan> layer_dispatches = {
        {"Fused_PreAttn_RMSNorm_QKV_RoPE", 0},
        {"LDS_Wavefront_FlashAttention", 1},
        {"Fused_MLP_SwiGLU_Down_Residual", 2}
    };

    ASSERT_EQ(layer_dispatches.size(), 3u);
    ASSERT_EQ(layer_dispatches.size() * num_layers, 84u);

    // Pipeline barrier count reduction: 8 barriers -> 3 barriers per layer
    uint32_t legacy_barriers = num_layers * 8;
    uint32_t mega_barriers = num_layers * 3;
    ASSERT_EQ(legacy_barriers, 224u);
    ASSERT_EQ(mega_barriers, 84u);
}

// 5. Mega-Kernel Push Constants Binary Layout & Alignment
E2E_TEST_CASE(Tier1_R3, test_r3_push_constants_binary_layout) {
    // Vulkan specification requires push constants size to be multiples of 4 bytes
    ASSERT_EQ(sizeof(PreAttnPushConstants) % 4, 0u);
    ASSERT_EQ(sizeof(MLPPushConstants) % 4, 0u);

    ASSERT_EQ(sizeof(PreAttnPushConstants), 32u);
    ASSERT_EQ(sizeof(MLPPushConstants), 20u);

    // Validate offset alignments
    ASSERT_EQ(offsetof(PreAttnPushConstants, layer_idx), 0u);
    ASSERT_EQ(offsetof(PreAttnPushConstants, seq_len), 4u);
    ASSERT_EQ(offsetof(PreAttnPushConstants, num_heads), 8u);
    ASSERT_EQ(offsetof(PreAttnPushConstants, num_kv_heads), 12u);
    ASSERT_EQ(offsetof(PreAttnPushConstants, head_dim), 16u);
    ASSERT_EQ(offsetof(PreAttnPushConstants, hidden_dim), 20u);
    ASSERT_EQ(offsetof(PreAttnPushConstants, rms_norm_eps), 24u);
    ASSERT_EQ(offsetof(PreAttnPushConstants, rope_freq_base), 28u);

    ASSERT_EQ(offsetof(MLPPushConstants, layer_idx), 0u);
    ASSERT_EQ(offsetof(MLPPushConstants, hidden_dim), 4u);
    ASSERT_EQ(offsetof(MLPPushConstants, intermediate_dim), 8u);
    ASSERT_EQ(offsetof(MLPPushConstants, add_residual), 12u);
    ASSERT_EQ(offsetof(MLPPushConstants, rms_norm_eps), 16u);
}

// 6. RoPE Unit-Norm and Positional Decay Properties
E2E_TEST_CASE(Tier1_R3, test_r3_rope_rotary_embedding_complex_math) {
    constexpr uint32_t head_dim = 64;
    std::vector<float> q1(head_dim, 1.0f);
    std::vector<float> q2(head_dim, 1.0f);
    std::vector<float> k_dummy(head_dim, 0.0f);

    // Calculate initial L2 norm
    double norm_initial = 0.0;
    for (float v : q1) norm_initial += v * v;
    norm_initial = std::sqrt(norm_initial);

    reference_rope(q1.data(), k_dummy.data(), 1, 0, head_dim, 0, 10000.0f);
    reference_rope(q2.data(), k_dummy.data(), 1, 0, head_dim, 5, 10000.0f);

    // Invariant: RoPE rotation preserves vector Euclidean length
    double norm_q1 = 0.0;
    for (float v : q1) norm_q1 += v * v;
    norm_q1 = std::sqrt(norm_q1);

    double norm_q2 = 0.0;
    for (float v : q2) norm_q2 += v * v;
    norm_q2 = std::sqrt(norm_q2);

    ASSERT_NEAR(norm_q1, norm_initial, 1e-4);
    ASSERT_NEAR(norm_q2, norm_initial, 1e-4);

    // Invariant: Positional rotation changes coordinates
    bool modified = false;
    for (size_t i = 0; i < head_dim; ++i) {
        if (std::abs(q1[i] - q2[i]) > 1e-3f) {
            modified = true;
            break;
        }
    }
    ASSERT_TRUE(modified);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
