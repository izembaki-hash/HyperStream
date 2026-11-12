#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>

using namespace hyperstream::test;

namespace {

void rmsnorm_boundary(const float* x, const float* w, float* out, size_t size, float eps) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }
    double mean_sq = sum_sq / static_cast<double>(size);
    float inv_rms = static_cast<float>(1.0 / std::sqrt(mean_sq + static_cast<double>(eps)));
    for (size_t i = 0; i < size; ++i) {
        out[i] = x[i] * inv_rms * w[i];
    }
}

float silu_safe(float x) {
    if (x > 20.0f) return x; // e^-x ~ 0, silu(x) ~ x
    if (x < -20.0f) return 0.0f; // e^-x ~ large, silu(x) ~ 0
    return x / (1.0f + std::exp(-x));
}

} // anonymous namespace

// 1. RMSNorm Near-Zero Inputs & Epsilon Underflow Stability
E2E_TEST_CASE(Tier2_R3, test_r3_rmsnorm_near_zero_inputs_epsilon_underflow) {
    constexpr size_t dim = 128;
    std::vector<float> x(dim, 0.0f); // All zeros
    std::vector<float> w(dim, 1.0f);
    std::vector<float> out(dim, -999.0f);

    rmsnorm_boundary(x.data(), w.data(), out.data(), dim, 1e-6f);

    for (size_t i = 0; i < dim; ++i) {
        ASSERT_FALSE(std::isnan(out[i]));
        ASSERT_FALSE(std::isinf(out[i]));
        ASSERT_EQ(out[i], 0.0f); // 0 / sqrt(eps) * 1 = 0
    }
}

// 2. RMSNorm Extreme Large Magnitude Input Stability
E2E_TEST_CASE(Tier2_R3, test_r3_rmsnorm_extreme_magnitude_overflow_resistance) {
    constexpr size_t dim = 64;
    std::vector<float> x(dim, 1e18f); // Huge values
    std::vector<float> w(dim, 1.0f);
    std::vector<float> out(dim);

    rmsnorm_boundary(x.data(), w.data(), out.data(), dim, 1e-6f);

    // Invariant: Since all elements are identical, normalized values must be identically 1.0f
    for (size_t i = 0; i < dim; ++i) {
        ASSERT_FALSE(std::isnan(out[i]));
        ASSERT_FALSE(std::isinf(out[i]));
        ASSERT_NEAR(out[i], 1.0f, 1e-4f);
    }
}

// 3. SwiGLU Extreme Negative Gate (Asymptotic Convergence to Zero)
E2E_TEST_CASE(Tier2_R3, test_r3_swiglu_extreme_negative_gate) {
    float gate = -100.0f;
    float up = 50.0f;
    float swiglu = silu_safe(gate) * up;

    ASSERT_FALSE(std::isnan(swiglu));
    ASSERT_FALSE(std::isinf(swiglu));
    ASSERT_NEAR(swiglu, 0.0f, 1e-6f);
}

// 4. SwiGLU Extreme Positive Gate (Linear Asymptote)
E2E_TEST_CASE(Tier2_R3, test_r3_swiglu_extreme_positive_gate) {
    float gate = 50.0f;
    float up = 2.0f;
    float swiglu = silu_safe(gate) * up;

    ASSERT_FALSE(std::isnan(swiglu));
    ASSERT_FALSE(std::isinf(swiglu));
    ASSERT_NEAR(swiglu, 100.0f, 1e-3f); // 50 * 2 = 100
}

// 5. Residual Addition with Negative / Opposing Vectors
E2E_TEST_CASE(Tier2_R3, test_r3_residual_addition_opposing_cancellation) {
    constexpr size_t dim = 128;
    std::vector<float> residual(dim, 10.5f);
    std::vector<float> mlp_down(dim, -10.5f);
    std::vector<float> out(dim);

    for (size_t i = 0; i < dim; ++i) {
        out[i] = residual[i] + mlp_down[i];
    }

    for (size_t i = 0; i < dim; ++i) {
        ASSERT_NEAR(out[i], 0.0f, 1e-6f);
    }
}

// 6. RoPE at Extreme Long Context Position (e.g. 131,072)
E2E_TEST_CASE(Tier2_R3, test_r3_rope_extreme_sequence_positions) {
    constexpr uint32_t head_dim = 128;
    constexpr uint32_t extreme_pos = 131072;
    constexpr float freq_base = 1000000.0f;

    std::vector<float> q(head_dim, 1.0f);

    for (uint32_t i = 0; i < head_dim; i += 2) {
        float theta = std::pow(freq_base, -static_cast<float>(i) / static_cast<float>(head_dim));
        float angle = static_cast<float>(extreme_pos) * theta;
        float cos_a = std::cos(angle);
        float sin_a = std::sin(angle);

        ASSERT_FALSE(std::isnan(cos_a));
        ASSERT_FALSE(std::isnan(sin_a));
        ASSERT_FALSE(std::isinf(cos_a));
        ASSERT_FALSE(std::isinf(sin_a));

        float v0 = q[i];
        float v1 = q[i + 1];
        q[i]     = v0 * cos_a - v1 * sin_a;
        q[i + 1] = v0 * sin_a + v1 * cos_a;
    }

    // Verify Euclidean norm is preserved
    double norm_sq = 0.0;
    for (float v : q) norm_sq += v * v;
    ASSERT_NEAR(std::sqrt(norm_sq), std::sqrt(static_cast<double>(head_dim)), 1e-4);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
