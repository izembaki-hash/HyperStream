#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

using namespace hyperstream::test;

namespace {

struct LayerOffsets {
    uint64_t attn_norm = 0;
    uint64_t qkv_weight = 0;
    uint64_t o_weight = 0;
    uint64_t ffn_norm = 0;
    uint64_t gate_up_weight = 0;
    uint64_t down_weight = 0;
};

struct DynamicModelTopologySafe {
    std::string architecture;
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t num_layers = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_freq_base = 1000000.0f;

    std::vector<LayerOffsets> layers;
    uint64_t token_embd_offset = 0;
    uint64_t output_norm_offset = 0;
    uint64_t output_weight_offset = 0;

    static bool build_from_metadata(
        const std::unordered_map<std::string, uint32_t>& u32s,
        const std::unordered_map<std::string, float>& f32s,
        const std::unordered_map<std::string, std::string>& strings,
        const std::unordered_map<std::string, uint64_t>& tensor_offsets,
        DynamicModelTopologySafe& topo) 
    {
        auto it_arch = strings.find("general.architecture");
        topo.architecture = (it_arch != strings.end()) ? it_arch->second : "qwen2";

        std::string pfx = topo.architecture + ".";
        auto get_u32 = [&](const std::string& key, uint32_t def = 0) {
            auto it = u32s.find(pfx + key);
            return (it != u32s.end()) ? it->second : def;
        };

        topo.num_layers = get_u32("block_count", 0);
        topo.hidden_size = get_u32("embedding_length", 0);
        topo.intermediate_size = get_u32("feed_forward_length", 0);
        topo.num_heads = get_u32("attention.head_count", 0);

        // Validation against zero or invalid parameters
        if (topo.num_layers == 0 || topo.hidden_size == 0 || topo.num_heads == 0) {
            return false;
        }

        // KV heads defaults to num_heads if absent (Standard MHA fallback)
        topo.num_kv_heads = get_u32("attention.head_count_kv", topo.num_heads);
        if (topo.num_kv_heads == 0) {
            topo.num_kv_heads = topo.num_heads;
        }

        topo.head_dim = topo.hidden_size / topo.num_heads;
        if (topo.head_dim == 0 || (topo.hidden_size % topo.num_heads != 0)) {
            return false; // Non-divisible geometry
        }

        auto it_eps = f32s.find(pfx + "attention.layer_norm_rms_epsilon");
        if (it_eps != f32s.end()) topo.rms_norm_eps = it_eps->second;

        auto it_rope = f32s.find(pfx + "rope.freq_base");
        if (it_rope != f32s.end()) topo.rope_freq_base = it_rope->second;

        auto it_embd = tensor_offsets.find("token_embd.weight");
        if (it_embd != tensor_offsets.end()) {
            topo.token_embd_offset = it_embd->second;
        } else {
            return false; // token_embd.weight is mandatory
        }

        auto it_fnorm = tensor_offsets.find("output_norm.weight");
        if (it_fnorm != tensor_offsets.end()) topo.output_norm_offset = it_fnorm->second;

        auto it_lm = tensor_offsets.find("output.weight");
        if (it_lm != tensor_offsets.end()) {
            topo.output_weight_offset = it_lm->second;
        } else {
            topo.output_weight_offset = topo.token_embd_offset; // Tied weights
        }

        topo.layers.resize(topo.num_layers);
        for (uint32_t l = 0; l < topo.num_layers; ++l) {
            std::string blk = "blk." + std::to_string(l) + ".";
            auto& lo = topo.layers[l];
            auto get_to = [&](const std::string& name) -> uint64_t {
                auto it = tensor_offsets.find(blk + name);
                return (it != tensor_offsets.end()) ? it->second : 0;
            };
            lo.attn_norm = get_to("attn_norm.weight");
            lo.qkv_weight = get_to("attn_q.weight");
            lo.o_weight = get_to("attn_output.weight");
            lo.ffn_norm = get_to("ffn_norm.weight");
            lo.gate_up_weight = get_to("ffn_gate.weight");
            lo.down_weight = get_to("ffn_down.weight");
        }

        return true;
    }
};

} // anonymous namespace

// 1. Zero Block Count Rejection
E2E_TEST_CASE(Tier2_R2, test_r2_zero_layers_rejection) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"qwen2.block_count", 0}, // Invalid 0 layers
        {"qwen2.embedding_length", 896},
        {"qwen2.attention.head_count", 14}
    };
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"token_embd.weight", 100ULL}
    };

    DynamicModelTopologySafe topo;
    bool ok = DynamicModelTopologySafe::build_from_metadata(u32s, {}, {}, toffsets, topo);
    ASSERT_FALSE(ok);
}

// 2. Zero Hidden Size Rejection
E2E_TEST_CASE(Tier2_R2, test_r2_zero_hidden_dimension_rejection) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"qwen2.block_count", 24},
        {"qwen2.embedding_length", 0}, // Invalid 0 hidden dim
        {"qwen2.attention.head_count", 14}
    };
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"token_embd.weight", 100ULL}
    };

    DynamicModelTopologySafe topo;
    bool ok = DynamicModelTopologySafe::build_from_metadata(u32s, {}, {}, toffsets, topo);
    ASSERT_FALSE(ok);
}

// 3. Zero Head Count Prevent Division by Zero
E2E_TEST_CASE(Tier2_R2, test_r2_zero_head_count_prevent_division_by_zero) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"qwen2.block_count", 24},
        {"qwen2.embedding_length", 896},
        {"qwen2.attention.head_count", 0} // Would cause div-by-zero
    };
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"token_embd.weight", 100ULL}
    };

    DynamicModelTopologySafe topo;
    bool ok = DynamicModelTopologySafe::build_from_metadata(u32s, {}, {}, toffsets, topo);
    ASSERT_FALSE(ok);
}

// 4. Non-Divisible Hidden Size and Head Count Rejection
E2E_TEST_CASE(Tier2_R2, test_r2_non_divisible_geometry_rejection) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"qwen2.block_count", 24},
        {"qwen2.embedding_length", 897}, // 897 not divisible by 14
        {"qwen2.attention.head_count", 14}
    };
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"token_embd.weight", 100ULL}
    };

    DynamicModelTopologySafe topo;
    bool ok = DynamicModelTopologySafe::build_from_metadata(u32s, {}, {}, toffsets, topo);
    ASSERT_FALSE(ok);
}

// 5. Missing Mandatory Embedding Tensor Rejection
E2E_TEST_CASE(Tier2_R2, test_r2_missing_critical_tensors_detection) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"qwen2.block_count", 24},
        {"qwen2.embedding_length", 896},
        {"qwen2.attention.head_count", 14}
    };
    // No "token_embd.weight" provided
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"output_norm.weight", 200ULL}
    };

    DynamicModelTopologySafe topo;
    bool ok = DynamicModelTopologySafe::build_from_metadata(u32s, {}, {}, toffsets, topo);
    ASSERT_FALSE(ok);
}

// 6. Default MHA Fallback when head_count_kv is Missing
E2E_TEST_CASE(Tier2_R2, test_r2_head_count_kv_defaults_to_head_count) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"llama.block_count", 32},
        {"llama.embedding_length", 4096},
        {"llama.attention.head_count", 32}
        // attention.head_count_kv omitted intentionally
    };
    std::unordered_map<std::string, std::string> strings = {
        {"general.architecture", "llama"}
    };
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"token_embd.weight", 1000ULL}
    };

    DynamicModelTopologySafe topo;
    bool ok = DynamicModelTopologySafe::build_from_metadata(u32s, {}, strings, toffsets, topo);
    ASSERT_TRUE(ok);
    ASSERT_EQ(topo.num_heads, 32u);
    ASSERT_EQ(topo.num_kv_heads, 32u); // MHA fallback verified
    ASSERT_EQ(topo.head_dim, 128u);
}

// 7. Extreme 70B Dimension Scaling Boundary
E2E_TEST_CASE(Tier2_R2, test_r2_extreme_70b_dimension_scaling) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"llama.block_count", 80},
        {"llama.embedding_length", 8192},
        {"llama.feed_forward_length", 28672},
        {"llama.attention.head_count", 64},
        {"llama.attention.head_count_kv", 8}
    };
    std::unordered_map<std::string, std::string> strings = {
        {"general.architecture", "llama"}
    };
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"token_embd.weight", 5000ULL}
    };

    DynamicModelTopologySafe topo;
    bool ok = DynamicModelTopologySafe::build_from_metadata(u32s, {}, strings, toffsets, topo);
    ASSERT_TRUE(ok);
    ASSERT_EQ(topo.num_layers, 80u);
    ASSERT_EQ(topo.hidden_size, 8192u);
    ASSERT_EQ(topo.intermediate_size, 28672u);
    ASSERT_EQ(topo.num_heads, 64u);
    ASSERT_EQ(topo.num_kv_heads, 8u);
    ASSERT_EQ(topo.head_dim, 128u);
    ASSERT_EQ(topo.layers.size(), 80u);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
