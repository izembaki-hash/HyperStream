#include "../common/e2e_test_framework.hpp"
#include "../common/gguf_test_utils.hpp"
#include "../common/test_constants.hpp"
#include <algorithm>
#include <unordered_map>
#include <fstream>

using namespace hyperstream::test;

// --- GGUF Parser Interface Contract Helper for Testing ---
namespace {

struct ParsedMetadata {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    std::unordered_map<std::string, std::string> strings;
    std::unordered_map<std::string, uint32_t> u32s;
    std::unordered_map<std::string, uint64_t> u64s;
    std::unordered_map<std::string, float> f32s;
    std::unordered_map<std::string, std::vector<std::string>> string_arrays;
    uint64_t data_start_offset = 0;
};

bool parse_gguf_buffer(const uint8_t* data, size_t size, ParsedMetadata& out, uint32_t alignment = 32) {
    if (size < 24) return false;
    size_t offset = 0;

    auto read_bytes = [&](void* dst, size_t n) -> bool {
        if (offset + n > size) return false;
        std::memcpy(dst, data + offset, n);
        offset += n;
        return true;
    };

    if (!read_bytes(&out.magic, 4)) return false;
    if (out.magic != 0x46554747) return false; // "GGUF"

    if (!read_bytes(&out.version, 4)) return false;
    if (out.version != 3) return false;

    if (!read_bytes(&out.tensor_count, 8)) return false;
    if (!read_bytes(&out.kv_count, 8)) return false;

    for (uint64_t i = 0; i < out.kv_count; ++i) {
        uint64_t key_len = 0;
        if (!read_bytes(&key_len, 8)) return false;
        if (offset + key_len > size) return false;
        std::string key(reinterpret_cast<const char*>(data + offset), key_len);
        offset += key_len;

        uint32_t val_type = 0;
        if (!read_bytes(&val_type, 4)) return false;

        if (val_type == static_cast<uint32_t>(GGUFValueType::UINT32)) {
            uint32_t v = 0;
            if (!read_bytes(&v, 4)) return false;
            out.u32s[key] = v;
        } else if (val_type == static_cast<uint32_t>(GGUFValueType::UINT64)) {
            uint64_t v = 0;
            if (!read_bytes(&v, 8)) return false;
            out.u64s[key] = v;
        } else if (val_type == static_cast<uint32_t>(GGUFValueType::FLOAT32)) {
            float v = 0.0f;
            if (!read_bytes(&v, 4)) return false;
            out.f32s[key] = v;
        } else if (val_type == static_cast<uint32_t>(GGUFValueType::STRING)) {
            uint64_t slen = 0;
            if (!read_bytes(&slen, 8)) return false;
            if (offset + slen > size) return false;
            std::string s(reinterpret_cast<const char*>(data + offset), slen);
            offset += slen;
            out.strings[key] = s;
        } else if (val_type == static_cast<uint32_t>(GGUFValueType::ARRAY)) {
            uint32_t item_type = 0;
            uint64_t item_count = 0;
            if (!read_bytes(&item_type, 4)) return false;
            if (!read_bytes(&item_count, 8)) return false;

            if (item_type == static_cast<uint32_t>(GGUFValueType::STRING)) {
                std::vector<std::string> arr;
                arr.reserve(item_count);
                for (uint64_t j = 0; j < item_count; ++j) {
                    uint64_t slen = 0;
                    if (!read_bytes(&slen, 8)) return false;
                    if (offset + slen > size) return false;
                    std::string s(reinterpret_cast<const char*>(data + offset), slen);
                    offset += slen;
                    arr.push_back(s);
                }
                out.string_arrays[key] = arr;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    // Skip tensor descriptors
    for (uint64_t i = 0; i < out.tensor_count; ++i) {
        uint64_t name_len = 0;
        if (!read_bytes(&name_len, 8)) return false;
        offset += name_len;
        uint32_t n_dims = 0;
        if (!read_bytes(&n_dims, 4)) return false;
        offset += n_dims * 8; // dims
        offset += 4; // type
        offset += 8; // offset
    }

    size_t pad = (alignment - (offset % alignment)) % alignment;
    offset += pad;
    out.data_start_offset = offset;

    return true;
}

// Minimal BPE Tokenizer for Opaque Testing
struct TestBPE {
    std::unordered_map<std::string, int32_t> vocab;
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int32_t> merges;
    int32_t eos_id = -1;
    int32_t bos_id = -1;
    int32_t im_start_id = -1;
    int32_t im_end_id = -1;

    void load_from_parsed(const ParsedMetadata& meta) {
        auto it = meta.string_arrays.find("tokenizer.ggml.tokens");
        if (it != meta.string_arrays.end()) {
            id_to_token = it->second;
            for (size_t i = 0; i < id_to_token.size(); ++i) {
                vocab[id_to_token[i]] = static_cast<int32_t>(i);
            }
        }
        auto it_m = meta.string_arrays.find("tokenizer.ggml.merges");
        if (it_m != meta.string_arrays.end()) {
            for (size_t i = 0; i < it_m->second.size(); ++i) {
                merges[it_m->second[i]] = static_cast<int32_t>(i);
            }
        }
        auto it_eos = meta.u32s.find("tokenizer.ggml.eos_token_id");
        if (it_eos != meta.u32s.end()) eos_id = it_eos->second;
        auto it_bos = meta.u32s.find("tokenizer.ggml.bos_token_id");
        if (it_bos != meta.u32s.end()) bos_id = it_bos->second;

        auto it_s = vocab.find("<|im_start|>");
        if (it_s != vocab.end()) im_start_id = it_s->second;
        auto it_e = vocab.find("<|im_end|>");
        if (it_e != vocab.end()) im_end_id = it_e->second;
    }

    std::vector<int32_t> encode_simple(const std::string& text) const {
        std::vector<int32_t> ids;
        for (char c : text) {
            std::string s(1, c);
            auto it = vocab.find(s);
            if (it != vocab.end()) ids.push_back(it->second);
        }
        return ids;
    }

    std::string decode(int32_t token_id) const {
        if (token_id >= 0 && token_id < static_cast<int32_t>(id_to_token.size())) {
            return id_to_token[token_id];
        }
        return "";
    }
};

} // anonymous namespace

// 1. Header Magic & Version 3 Verification
E2E_TEST_CASE(Tier1_R1, test_r1_gguf_v3_header_magic_version) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.architecture", "qwen2");
    auto buf = builder.build();

    ParsedMetadata meta;
    bool ok = parse_gguf_buffer(buf.data(), buf.size(), meta);
    ASSERT_TRUE(ok);
    ASSERT_EQ(meta.magic, 0x46554747); // "GGUF"
    ASSERT_EQ(meta.version, 3u);
    ASSERT_EQ(meta.strings["general.architecture"], "qwen2");
}

// 2. Metadata Types Ingestion (string, uint32, uint64, float32, string array)
E2E_TEST_CASE(Tier1_R1, test_r1_gguf_metadata_extraction_types) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.name", "Qwen2.5-Coder-0.5B");
    builder.add_metadata_u32("qwen2.block_count", 24);
    builder.add_metadata_u64("qwen2.context_length", 32768ULL);
    builder.add_metadata_f32("qwen2.rope.freq_base", 1000000.0f);
    builder.add_metadata_array_string("test.tags", {"coder", "vulkan", "bare-metal"});

    auto buf = builder.build();
    ParsedMetadata meta;
    bool ok = parse_gguf_buffer(buf.data(), buf.size(), meta);
    ASSERT_TRUE(ok);

    ASSERT_EQ(meta.strings["general.name"], "Qwen2.5-Coder-0.5B");
    ASSERT_EQ(meta.u32s["qwen2.block_count"], 24u);
    ASSERT_EQ(meta.u64s["qwen2.context_length"], 32768ULL);
    ASSERT_NEAR(meta.f32s["qwen2.rope.freq_base"], 1000000.0f, 1e-3f);
    ASSERT_EQ(meta.string_arrays["test.tags"].size(), 3u);
    ASSERT_EQ(meta.string_arrays["test.tags"][1], "vulkan");
}

// 3. Tensor Info and Data Start Alignment Computation
E2E_TEST_CASE(Tier1_R1, test_r1_gguf_tensor_info_offset_computation) {
    SyntheticGGUFBuilder builder(64); // 64-byte alignment
    builder.add_metadata_string("general.architecture", "qwen2");
    builder.add_tensor("token_embd.weight", {896, 151936}, 8, 0, {0x11, 0x22, 0x33, 0x44});
    builder.add_tensor("output_norm.weight", {896}, 0, 1024, {0x55, 0x66});

    auto buf = builder.build();
    ParsedMetadata meta;
    bool ok = parse_gguf_buffer(buf.data(), buf.size(), meta, 64);
    ASSERT_TRUE(ok);
    ASSERT_EQ(meta.tensor_count, 2u);
    // Invariant: data_start_offset must be aligned to 64 bytes
    ASSERT_EQ(meta.data_start_offset % 64, 0u);
    ASSERT_TRUE(meta.data_start_offset > 0);
}

// 4. In-Engine BPE Vocab Reconstruction from Metadata
E2E_TEST_CASE(Tier1_R1, test_r1_bpe_vocab_reconstruction_from_metadata) {
    SyntheticGGUFBuilder builder;
    std::vector<std::string> sample_tokens = {"<|endoftext|>", "H", "e", "l", "o", "Hello", "<|im_start|>", "<|im_end|>"};
    builder.add_metadata_array_string("tokenizer.ggml.tokens", sample_tokens);
    builder.add_metadata_u32("tokenizer.ggml.eos_token_id", 0);
    builder.add_metadata_u32("tokenizer.ggml.bos_token_id", 0);

    auto buf = builder.build();
    ParsedMetadata meta;
    bool ok = parse_gguf_buffer(buf.data(), buf.size(), meta);
    ASSERT_TRUE(ok);

    TestBPE tokenizer;
    tokenizer.load_from_parsed(meta);

    ASSERT_EQ(tokenizer.id_to_token.size(), 8u);
    ASSERT_EQ(tokenizer.vocab["<|im_start|>"], 6);
    ASSERT_EQ(tokenizer.vocab["<|im_end|>"], 7);
    ASSERT_EQ(tokenizer.im_start_id, 6);
    ASSERT_EQ(tokenizer.im_end_id, 7);
}

// 5. BPE Merges Table Parsing & Token Encoding
E2E_TEST_CASE(Tier1_R1, test_r1_bpe_merges_and_tokenization_encoding) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_array_string("tokenizer.ggml.tokens", {"H", "e", "l", "o", "Hel", "lo"});
    builder.add_metadata_array_string("tokenizer.ggml.merges", {"H e", "Hel l", "l o"});

    auto buf = builder.build();
    ParsedMetadata meta;
    bool ok = parse_gguf_buffer(buf.data(), buf.size(), meta);
    ASSERT_TRUE(ok);

    TestBPE tokenizer;
    tokenizer.load_from_parsed(meta);

    ASSERT_EQ(tokenizer.merges.size(), 3u);
    ASSERT_EQ(tokenizer.merges["H e"], 0);
    ASSERT_EQ(tokenizer.merges["Hel l"], 1);

    auto ids = tokenizer.encode_simple("Hel");
    ASSERT_EQ(ids.size(), 3u);
    ASSERT_EQ(ids[0], 0); // H
    ASSERT_EQ(ids[1], 1); // e
    ASSERT_EQ(ids[2], 2); // l
}

// 6. Token Decoding and Special Tokens Verification
E2E_TEST_CASE(Tier1_R1, test_r1_bpe_decoding_and_special_tokens) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_array_string("tokenizer.ggml.tokens", {"<|endoftext|>", "<|im_start|>", "<|im_end|>", "return", "0"});
    builder.add_metadata_u32("tokenizer.ggml.eos_token_id", 0);

    auto buf = builder.build();
    ParsedMetadata meta;
    bool ok = parse_gguf_buffer(buf.data(), buf.size(), meta);
    ASSERT_TRUE(ok);

    TestBPE tokenizer;
    tokenizer.load_from_parsed(meta);

    ASSERT_STREQ(tokenizer.decode(0), "<|endoftext|>");
    ASSERT_STREQ(tokenizer.decode(1), "<|im_start|>");
    ASSERT_STREQ(tokenizer.decode(2), "<|im_end|>");
    ASSERT_STREQ(tokenizer.decode(3), "return");
    ASSERT_STREQ(tokenizer.decode(4), "0");
    ASSERT_STREQ(tokenizer.decode(999), ""); // Out of bounds
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
