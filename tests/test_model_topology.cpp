#include "../tests/e2e/common/e2e_test_framework.hpp"
#include "../tests/e2e/common/gguf_test_utils.hpp"
#include "model_topology.hpp"
#include <iostream>

using namespace hyperstream;
using namespace hyperstream::test;

E2E_TEST_CASE(ModelTopologyTest, test_topology_build_from_synthetic_gguf) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.architecture", "qwen2");
    builder.add_metadata_u32("qwen2.block_count", 28);
    builder.add_metadata_u32("qwen2.embedding_length", 3584);
    builder.add_metadata_u32("qwen2.feed_forward_length", 18944);
    builder.add_metadata_u32("qwen2.attention.head_count", 28);
    builder.add_metadata_u32("qwen2.attention.head_count_kv", 4);
    builder.add_metadata_f32("qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
    builder.add_metadata_f32("qwen2.rope.freq_base", 1000000.0f);

    uint64_t cur_offset = 1048576ULL;
    auto next_offset = [&]() -> uint64_t {
        uint64_t o = cur_offset;
        cur_offset += 65536ULL;
        return o;
    };

    builder.add_tensor("token_embd.weight", {3584, 151936}, 2, next_offset()); // Q4_0
    builder.add_tensor("output_norm.weight", {3584, 1}, 0, next_offset());     // F32
    builder.add_tensor("output.weight", {3584, 151936}, 2, next_offset());      // Q4_0

    for (uint32_t l = 0; l < 28; ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        builder.add_tensor(p + "attn_norm.weight", {3584, 1}, 0, next_offset());
        builder.add_tensor(p + "attn_q.weight", {3584, 3584}, 2, next_offset());
        builder.add_tensor(p + "attn_k.weight", {3584, 512}, 2, next_offset());
        builder.add_tensor(p + "attn_v.weight", {3584, 512}, 2, next_offset());
        builder.add_tensor(p + "attn_output.weight", {3584, 3584}, 2, next_offset());
        builder.add_tensor(p + "ffn_norm.weight", {3584, 1}, 0, next_offset());
        builder.add_tensor(p + "ffn_gate.weight", {3584, 18944}, 2, next_offset());
        builder.add_tensor(p + "ffn_up.weight", {3584, 18944}, 2, next_offset());
        builder.add_tensor(p + "ffn_down.weight", {18944, 3584}, 2, next_offset());
    }

    auto buf = builder.build();

    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    ModelTopology topo;
    ASSERT_TRUE(ModelTopology::build_from_gguf(parser, topo));

    ASSERT_EQ(topo.architecture, "qwen2");
    ASSERT_EQ(topo.num_layers, 28u);
    ASSERT_EQ(topo.hidden_size, 3584u);
    ASSERT_EQ(topo.intermediate_size, 18944u);
    ASSERT_EQ(topo.num_heads, 28u);
    ASSERT_EQ(topo.num_kv_heads, 4u);
    ASSERT_EQ(topo.head_dim, 128u);
    ASSERT_EQ(topo.vocab_size, 151936u);
    ASSERT_EQ(topo.layers.size(), 28u);

    // Verify per-layer offsets are non-zero and populated
    for (uint32_t l = 0; l < 28; ++l) {
        ASSERT_NE(topo.layers[l].attn_norm, 0ULL);
        ASSERT_NE(topo.layers[l].qkv_weight, 0ULL);
        ASSERT_NE(topo.layers[l].o_weight, 0ULL);
        ASSERT_NE(topo.layers[l].ffn_norm, 0ULL);
        ASSERT_NE(topo.layers[l].gate_up_weight, 0ULL);
        ASSERT_NE(topo.layers[l].down_weight, 0ULL);
    }

    auto spec = topo.get_specialization_data(4096);
    ASSERT_EQ(spec.HIDDEN_DIM, 3584u);
    ASSERT_EQ(spec.HEAD_DIM, 128u);
    ASSERT_EQ(spec.NUM_HEADS, 28u);
    ASSERT_EQ(spec.NUM_KV_HEADS, 4u);
    ASSERT_EQ(spec.INTERMEDIATE_DIM, 18944u);
    ASSERT_EQ(spec.MAX_SEQ_LEN, 4096u);
    ASSERT_EQ(spec.VOCAB_SIZE, 151936u);
}

int main(int argc, char** argv) {
    return hyperstream::test::TestRegistry::instance().run_all(argc, argv);
}
