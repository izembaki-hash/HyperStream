#include "e2e/common/e2e_test_framework.hpp"
#include "e2e/common/gguf_test_utils.hpp"
#include "gguf_parser.hpp"
#include <filesystem>

using namespace hyperstream;
using namespace hyperstream::test;

// 1. Header Magic & Version 3 Verification
E2E_TEST_CASE(GGUFParserTest, test_header_magic_and_version) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.architecture", "qwen2");
    auto buf = builder.build();

    GGUFParser parser;
    bool ok = parser.parse_from_memory(buf.data(), buf.size());
    ASSERT_TRUE(ok);
    ASSERT_EQ(parser.get_version(), 3u);
    ASSERT_STREQ(parser.get_string("general.architecture"), "qwen2");
}

// 2. Corrupt Magic Rejection
E2E_TEST_CASE(GGUFParserTest, test_corrupt_magic_rejected) {
    SyntheticGGUFBuilder builder;
    auto buf = builder.build(true); // corrupt magic = 0xDEADBEEF

    GGUFParser parser;
    bool ok = parser.parse_from_memory(buf.data(), buf.size());
    ASSERT_FALSE(ok);
}

// 3. Invalid Version Rejection
E2E_TEST_CASE(GGUFParserTest, test_invalid_version_rejected) {
    SyntheticGGUFBuilder builder;
    auto buf = builder.build(false, 99); // custom invalid version = 99

    GGUFParser parser;
    bool ok = parser.parse_from_memory(buf.data(), buf.size());
    ASSERT_FALSE(ok);
}

// 4. Metadata Types Extraction (String, U32, U64, F32, Bool, String Array)
E2E_TEST_CASE(GGUFParserTest, test_metadata_extraction_types) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.name", "Qwen2.5-Coder-0.5B");
    builder.add_metadata_u32("qwen2.block_count", 24);
    builder.add_metadata_u64("qwen2.context_length", 32768ULL);
    builder.add_metadata_f32("qwen2.rope.freq_base", 1000000.0f);
    builder.add_metadata_array_string("tokenizer.tags", {"code", "vulkan", "high-performance"});

    auto buf = builder.build();
    GGUFParser parser;
    bool ok = parser.parse_from_memory(buf.data(), buf.size());
    ASSERT_TRUE(ok);

    std::string name;
    ASSERT_TRUE(parser.get_metadata_string("general.name", name));
    ASSERT_STREQ(name, "Qwen2.5-Coder-0.5B");

    uint32_t blocks = 0;
    ASSERT_TRUE(parser.get_metadata_u32("qwen2.block_count", blocks));
    ASSERT_EQ(blocks, 24u);

    uint64_t ctx = 0;
    ASSERT_TRUE(parser.get_metadata_u64("qwen2.context_length", ctx));
    ASSERT_EQ(ctx, 32768ULL);

    float rope = 0.0f;
    ASSERT_TRUE(parser.get_metadata_f32("qwen2.rope.freq_base", rope));
    ASSERT_NEAR(rope, 1000000.0f, 1e-3f);

    std::vector<std::string> tags;
    ASSERT_TRUE(parser.get_metadata_array_string("tokenizer.tags", tags));
    ASSERT_EQ(tags.size(), 3u);
    ASSERT_STREQ(tags[0], "code");
    ASSERT_STREQ(tags[1], "vulkan");
    ASSERT_STREQ(tags[2], "high-performance");

    // Missing key check
    std::string missing_str;
    ASSERT_FALSE(parser.get_metadata_string("non_existent_key", missing_str));
    ASSERT_EQ(parser.get_u32("non_existent_key", 999), 999u);
}

// 5. Alignment Handling & Data Start Offset
E2E_TEST_CASE(GGUFParserTest, test_alignment_and_data_start_offset) {
    SyntheticGGUFBuilder builder64(64);
    builder64.add_metadata_u32("general.alignment", 64);
    builder64.add_tensor("test_tensor_1", {128, 64}, 8, 0, {0xAA, 0xBB, 0xCC, 0xDD});

    auto buf64 = builder64.build();
    GGUFParser parser;
    bool ok = parser.parse_from_memory(buf64.data(), buf64.size());
    ASSERT_TRUE(ok);
    ASSERT_EQ(parser.get_alignment(), 64u);
    ASSERT_EQ(parser.get_data_start_offset() % 64, 0u);
}

// 6. Tensor Descriptors and Data Pointer Resolution
E2E_TEST_CASE(GGUFParserTest, test_tensor_descriptors_and_data_access) {
    SyntheticGGUFBuilder builder(32);
    std::vector<uint8_t> payload1 = {0x01, 0x02, 0x03, 0x04};
    std::vector<uint8_t> payload2 = {0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80};

    builder.add_tensor("token_embd.weight", {896, 151936}, 8, 0, payload1);
    builder.add_tensor("output_norm.weight", {896}, 0, 4, payload2);

    auto buf = builder.build();
    GGUFParser parser;
    bool ok = parser.parse_from_memory(buf.data(), buf.size());
    ASSERT_TRUE(ok);
    ASSERT_EQ(parser.get_tensor_count(), 2u);

    const auto* t1 = parser.find_tensor("token_embd.weight");
    ASSERT_TRUE(t1 != nullptr);
    ASSERT_STREQ(t1->name, "token_embd.weight");
    ASSERT_EQ(t1->n_dims, 2u);
    ASSERT_EQ(t1->ne[0], 896ULL);
    ASSERT_EQ(t1->ne[1], 151936ULL);
    ASSERT_EQ(t1->type, 8u); // Q8_0
    ASSERT_EQ(t1->offset, 0ULL);

    const uint8_t* data1 = parser.get_tensor_data(*t1);
    ASSERT_TRUE(data1 != nullptr);
    ASSERT_EQ(data1[0], 0x01);
    ASSERT_EQ(data1[1], 0x02);
    ASSERT_EQ(data1[2], 0x03);
    ASSERT_EQ(data1[3], 0x04);

    const auto* t2 = parser.find_tensor("output_norm.weight");
    ASSERT_TRUE(t2 != nullptr);
    ASSERT_EQ(t2->n_dims, 1u);
    ASSERT_EQ(t2->ne[0], 896ULL);
    ASSERT_EQ(t2->type, 0u); // F32
    ASSERT_EQ(t2->offset, 4ULL);

    const uint8_t* data2 = parser.get_tensor_data(*t2);
    ASSERT_TRUE(data2 != nullptr);
    ASSERT_EQ(data2[0], 0x10);
    ASSERT_EQ(data2[1], 0x20);

    // Non-existent tensor lookup returns nullptr
    ASSERT_TRUE(parser.find_tensor("non_existent_tensor") == nullptr);
}

// 7. File-Based Open & Memory Mapping
E2E_TEST_CASE(GGUFParserTest, test_file_open_via_mmap) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.architecture", "llama");
    builder.add_metadata_u32("llama.block_count", 32);
    builder.add_tensor("blk.0.attn_q.weight", {4096, 4096}, 2, 0, {0xCA, 0xFE});

    std::string temp_path = "bin/temp_test_model.gguf";
    bool written = builder.write_to_file(temp_path);
    ASSERT_TRUE(written);

    GGUFParser parser;
    bool ok = parser.open(temp_path);
    ASSERT_TRUE(ok);
    ASSERT_STREQ(parser.get_string("general.architecture"), "llama");
    ASSERT_EQ(parser.get_u32("llama.block_count"), 32u);

    const auto* t = parser.find_tensor("blk.0.attn_q.weight");
    ASSERT_TRUE(t != nullptr);
    ASSERT_EQ(t->type, 2u); // Q4_0
    const uint8_t* d = parser.get_tensor_data(*t);
    ASSERT_TRUE(d != nullptr);
    ASSERT_EQ(d[0], 0xCA);
    ASSERT_EQ(d[1], 0xFE);

    parser.close();
    std::filesystem::remove(temp_path);
}

// 8. Real Model GGUF Parsing (if present on disk)
E2E_TEST_CASE(GGUFParserTest, test_real_model_gguf_metadata_inspection) {
    std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(real_path)) {
        real_path = "models/qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    }

    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        bool ok = parser.open(real_path);
        ASSERT_TRUE(ok);
        ASSERT_EQ(parser.get_version(), 3u);

        std::string arch = parser.get_string("general.architecture");
        ASSERT_STREQ(arch, "qwen2");

        uint32_t block_count = parser.get_u32("qwen2.block_count");
        ASSERT_EQ(block_count, 24u);

        uint32_t embed_dim = parser.get_u32("qwen2.embedding_length");
        ASSERT_EQ(embed_dim, 896u);

        uint32_t heads = parser.get_u32("qwen2.attention.head_count");
        ASSERT_EQ(heads, 14u);

        uint32_t kv_heads = parser.get_u32("qwen2.attention.head_count_kv");
        ASSERT_EQ(kv_heads, 2u);

        const auto* t_embd = parser.find_tensor("token_embd.weight");
        ASSERT_TRUE(t_embd != nullptr);
        ASSERT_EQ(t_embd->ne[0], 896ULL);
        ASSERT_EQ(t_embd->ne[1], 151936ULL);
        ASSERT_EQ(t_embd->type, 8u); // Q8_0

        const auto* t_norm = parser.find_tensor("output_norm.weight");
        ASSERT_TRUE(t_norm != nullptr);
        ASSERT_EQ(t_norm->ne[0], 896ULL);
        ASSERT_EQ(t_norm->type, 0u); // F32
    }
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
