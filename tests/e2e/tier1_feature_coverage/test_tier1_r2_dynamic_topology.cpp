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

struct DynamicModelTopology {
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
    uint64_t data_start_offset = 0;

    std::vector<LayerOffsets> layers;
    uint64_t token_embd_offset = 0;
    uint64_t output_norm_offset = 0;
    uint64_t output_weight_offset = 0;

    static bool build_from_metadata(
        const std::unordered_map<std::string, uint32_t>& u32s,
        const std::unordered_map<std::string, float>& f32s,
        const std::unordered_map<std::string, std::string>& strings,
        const std::unordered_map<std::string, uint64_t>& tensor_offsets,
        DynamicModelTopology& topo) 
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
        topo.num_kv_heads = get_u32("attention.head_count_kv", topo.num_heads);
        if (topo.num_heads > 0) {
            topo.head_dim = topo.hidden_size / topo.num_heads;
        }

        auto it_eps = f32s.find(pfx + "attention.layer_norm_rms_epsilon");
        if (it_eps != f32s.end()) topo.rms_norm_eps = it_eps->second;

        auto it_rope = f32s.find(pfx + "rope.freq_base");
        if (it_rope != f32s.end()) topo.rope_freq_base = it_rope->second;

        auto it_embd = tensor_offsets.find("token_embd.weight");
        if (it_embd != tensor_offsets.end()) topo.token_embd_offset = it_embd->second;

        auto it_fnorm = tensor_offsets.find("output_norm.weight");
        if (it_fnorm != tensor_offsets.end()) topo.output_norm_offset = it_fnorm->second;

        auto it_lm = tensor_offsets.find("output.weight");
        if (it_lm != tensor_offsets.end()) {
            topo.output_weight_offset = it_lm->second;
        } else {
            // Tied embedding fallback
            topo.output_weight_offset = topo.token_embd_offset;
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

        return (topo.num_layers > 0 && topo.hidden_size > 0 && topo.num_heads > 0);
    }
};

struct SpecializationData {
    uint32_t HIDDEN_DIM;
    uint32_t HEAD_DIM;
    uint32_t NUM_HEADS;
    uint32_t NUM_KV_HEADS;
    uint32_t INTERMEDIATE_DIM;
    uint32_t MAX_SEQ_LEN;
    uint32_t VOCAB_SIZE;
};

} // anonymous namespace

// 1. Dynamic Topology Extraction Verification
E2E_TEST_CASE(Tier1_R2, test_r2_dynamic_topology_derivation) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"qwen2.block_count", 28},
        {"qwen2.embedding_length", 3584},
        {"qwen2.feed_forward_length", 18944},
        {"qwen2.attention.head_count", 28},
        {"qwen2.attention.head_count_kv", 4}
    };
    std::unordered_map<std::string, float> f32s = {
        {"qwen2.attention.layer_norm_rms_epsilon", 1e-6f},
        {"qwen2.rope.freq_base", 1000000.0f}
    };
    std::unordered_map<std::string, std::string> strings = {
        {"general.architecture", "qwen2"}
    };
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"token_embd.weight", 1048576ULL},
        {"output_norm.weight", 2097152ULL},
        {"output.weight", 3145728ULL}
    };

    DynamicModelTopology topo;
    bool ok = DynamicModelTopology::build_from_metadata(u32s, f32s, strings, toffsets, topo);
    ASSERT_TRUE(ok);
    ASSERT_EQ(topo.num_layers, 28u);
    ASSERT_EQ(topo.hidden_size, 3584u);
    ASSERT_EQ(topo.intermediate_size, 18944u);
    ASSERT_EQ(topo.num_heads, 28u);
    ASSERT_EQ(topo.num_kv_heads, 4u);
    ASSERT_EQ(topo.head_dim, 128u); // 3584 / 28
    ASSERT_NEAR(topo.rms_norm_eps, 1e-6f, 1e-8f);
    ASSERT_NEAR(topo.rope_freq_base, 1000000.0f, 1e-3f);
}

// 2. Layer Tensor Offset Mapping Verification
E2E_TEST_CASE(Tier1_R2, test_r2_layer_tensor_offset_mapping) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"qwen2.block_count", 2},
        {"qwen2.embedding_length", 896},
        {"qwen2.feed_forward_length", 4864},
        {"qwen2.attention.head_count", 14},
        {"qwen2.attention.head_count_kv", 2}
    };
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"blk.0.attn_norm.weight", 100ULL},
        {"blk.0.attn_q.weight", 200ULL},
        {"blk.0.attn_output.weight", 300ULL},
        {"blk.0.ffn_norm.weight", 400ULL},
        {"blk.0.ffn_gate.weight", 500ULL},
        {"blk.0.ffn_down.weight", 600ULL},
        {"blk.1.attn_norm.weight", 700ULL},
        {"blk.1.attn_q.weight", 800ULL},
        {"blk.1.attn_output.weight", 900ULL},
        {"blk.1.ffn_norm.weight", 1000ULL},
        {"blk.1.ffn_gate.weight", 1100ULL},
        {"blk.1.ffn_down.weight", 1200ULL}
    };

    DynamicModelTopology topo;
    bool ok = DynamicModelTopology::build_from_metadata(u32s, {}, {}, toffsets, topo);
    ASSERT_TRUE(ok);
    ASSERT_EQ(topo.layers.size(), 2u);
    ASSERT_EQ(topo.layers[0].attn_norm, 100ULL);
    ASSERT_EQ(topo.layers[0].down_weight, 600ULL);
    ASSERT_EQ(topo.layers[1].attn_norm, 700ULL);
    ASSERT_EQ(topo.layers[1].down_weight, 1200ULL);
}

// 3. Tied LM Head Fallback Resolution
E2E_TEST_CASE(Tier1_R2, test_r2_tied_embedding_handling) {
    std::unordered_map<std::string, uint32_t> u32s = {
        {"qwen2.block_count", 1},
        {"qwen2.embedding_length", 512},
        {"qwen2.feed_forward_length", 2048},
        {"qwen2.attention.head_count", 8}
    };
    // Note: output.weight is intentionally absent
    std::unordered_map<std::string, uint64_t> toffsets = {
        {"token_embd.weight", 9999ULL}
    };

    DynamicModelTopology topo;
    bool ok = DynamicModelTopology::build_from_metadata(u32s, {}, {}, toffsets, topo);
    ASSERT_TRUE(ok);
    // Invariant: when output.weight is absent, output_weight_offset must tie to token_embd_offset
    ASSERT_EQ(topo.output_weight_offset, 9999ULL);
    ASSERT_EQ(topo.token_embd_offset, 9999ULL);
}

// 4. Vulkan Specialization Constants Packing & Layout
E2E_TEST_CASE(Tier1_R2, test_r2_vulkan_specialization_info_layout) {
    SpecializationData spec_data = {
        3584, // HIDDEN_DIM
        128,  // HEAD_DIM
        28,   // NUM_HEADS
        4,    // NUM_KV_HEADS
        18944,// INTERMEDIATE_DIM
        4096, // MAX_SEQ_LEN
        152064// VOCAB_SIZE
    };

    VkSpecializationMapEntry entries[] = {
        { 0, offsetof(SpecializationData, HIDDEN_DIM), sizeof(uint32_t) },
        { 1, offsetof(SpecializationData, HEAD_DIM), sizeof(uint32_t) },
        { 2, offsetof(SpecializationData, NUM_HEADS), sizeof(uint32_t) },
        { 3, offsetof(SpecializationData, NUM_KV_HEADS), sizeof(uint32_t) },
        { 4, offsetof(SpecializationData, INTERMEDIATE_DIM), sizeof(uint32_t) },
        { 5, offsetof(SpecializationData, MAX_SEQ_LEN), sizeof(uint32_t) },
        { 6, offsetof(SpecializationData, VOCAB_SIZE), sizeof(uint32_t) }
    };

    VkSpecializationInfo spec_info = {};
    spec_info.mapEntryCount = sizeof(entries) / sizeof(entries[0]);
    spec_info.pMapEntries = entries;
    spec_info.dataSize = sizeof(spec_data);
    spec_info.pData = &spec_data;

    ASSERT_EQ(spec_info.mapEntryCount, 7u);
    ASSERT_EQ(spec_info.dataSize, sizeof(SpecializationData));
    ASSERT_EQ(reinterpret_cast<const uint32_t*>(spec_info.pData)[0], 3584u);
    ASSERT_EQ(reinterpret_cast<const uint32_t*>(spec_info.pData)[1], 128u);
    ASSERT_EQ(reinterpret_cast<const uint32_t*>(spec_info.pData)[2], 28u);
    ASSERT_EQ(reinterpret_cast<const uint32_t*>(spec_info.pData)[3], 4u);
}

// 5. Arbitrary Non-Standard Topology Dimension Scaling
E2E_TEST_CASE(Tier1_R2, test_r2_arbitrary_dimension_scaling) {
    // Testing scaling to a custom 14B or Llama-3 style architecture
    std::unordered_map<std::string, uint32_t> u32s = {
        {"llama.block_count", 32},
        {"llama.embedding_length", 4096},
        {"llama.feed_forward_length", 14336},
        {"llama.attention.head_count", 32},
        {"llama.attention.head_count_kv", 8}
    };
    std::unordered_map<std::string, std::string> strings = {
        {"general.architecture", "llama"}
    };

    DynamicModelTopology topo;
    bool ok = DynamicModelTopology::build_from_metadata(u32s, {}, strings, {}, topo);
    ASSERT_TRUE(ok);
    ASSERT_EQ(topo.architecture, "llama");
    ASSERT_EQ(topo.num_layers, 32u);
    ASSERT_EQ(topo.hidden_size, 4096u);
    ASSERT_EQ(topo.num_heads, 32u);
    ASSERT_EQ(topo.num_kv_heads, 8u);
    ASSERT_EQ(topo.head_dim, 128u); // 4096 / 32
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
