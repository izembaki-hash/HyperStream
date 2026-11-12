#include "../common/e2e_test_framework.hpp"
#include "../common/gguf_test_utils.hpp"
#include "../common/test_constants.hpp"
#include <vector>
#include <string>
#include <cstring>
#include <unordered_map>

using namespace hyperstream::test;

namespace {

struct ParsedMetadataBoundary {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    std::unordered_map<std::string, std::string> strings;
    std::unordered_map<std::string, uint32_t> u32s;
    std::unordered_map<std::string, std::vector<std::string>> string_arrays;
};

bool parse_gguf_safe(const uint8_t* data, size_t size, ParsedMetadataBoundary& out) {
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
    if (out.version != 3) return false; // Must be v3

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
            return false; // Unknown or unsupported type
        }
    }
    return true;
}

} // anonymous namespace

// 1. Corrupted Magic Rejection
E2E_TEST_CASE(Tier2_R1, test_r1_corrupted_magic_rejection) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.architecture", "qwen2");
    auto buf = builder.build(true); // corrupt magic to 0xDEADBEEF

    ParsedMetadataBoundary meta;
    bool ok = parse_gguf_safe(buf.data(), buf.size(), meta);
    ASSERT_FALSE(ok);
}

// 2. Invalid Version (Not Version 3) Rejection
E2E_TEST_CASE(Tier2_R1, test_r1_invalid_version_rejection) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.architecture", "qwen2");
    auto buf_v2 = builder.build(false, 2); // Version 2
    auto buf_v4 = builder.build(false, 4); // Version 4

    ParsedMetadataBoundary meta;
    ASSERT_FALSE(parse_gguf_safe(buf_v2.data(), buf_v2.size(), meta));
    ASSERT_FALSE(parse_gguf_safe(buf_v4.data(), buf_v4.size(), meta));
}

// 3. Truncated Header (Under 24 bytes) Rejection
E2E_TEST_CASE(Tier2_R1, test_r1_truncated_file_header_rejection) {
    std::vector<uint8_t> truncated_buf = {0x47, 0x55, 0x46, 0x46, 0x03, 0x00, 0x00, 0x00}; // Only 8 bytes

    ParsedMetadataBoundary meta;
    bool ok = parse_gguf_safe(truncated_buf.data(), truncated_buf.size(), meta);
    ASSERT_FALSE(ok);
}

// 4. Truncated Metadata Value Rejection (Prevent Buffer Overrun)
E2E_TEST_CASE(Tier2_R1, test_r1_truncated_metadata_key_or_val) {
    SyntheticGGUFBuilder builder(1); // 1-byte alignment: no trailing padding
    builder.add_metadata_string("general.name", "A_very_long_model_name_that_will_be_cut_off");
    auto buf = builder.build();

    // Truncate buffer by 10 bytes at the end
    ASSERT_TRUE(buf.size() > 10);
    buf.resize(buf.size() - 10);

    ParsedMetadataBoundary meta;
    bool ok = parse_gguf_safe(buf.data(), buf.size(), meta);
    ASSERT_FALSE(ok);
}

// 5. Unknown Metadata Type Rejection
E2E_TEST_CASE(Tier2_R1, test_r1_unknown_metadata_type_handling) {
    std::vector<uint8_t> buf;
    // Magic "GGUF", version 3, tensor_count 0, kv_count 1
    uint32_t magic = 0x46554747;
    uint32_t version = 3;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 1;

    auto append_u32 = [&](uint32_t v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + 4);
    };
    auto append_u64 = [&](uint64_t v) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
        buf.insert(buf.end(), p, p + 8);
    };

    append_u32(magic);
    append_u32(version);
    append_u64(tensor_count);
    append_u64(kv_count);

    // Key: "bad_type"
    std::string key = "bad_type";
    append_u64(key.size());
    buf.insert(buf.end(), key.begin(), key.end());

    // Type: 999 (invalid)
    append_u32(999);

    ParsedMetadataBoundary meta;
    bool ok = parse_gguf_safe(buf.data(), buf.size(), meta);
    ASSERT_FALSE(ok);
}

// 6. Zero Tensors and Zero Metadata KV Valid Boundary
E2E_TEST_CASE(Tier2_R1, test_r1_zero_tensors_and_zero_kv_valid_boundary) {
    SyntheticGGUFBuilder builder;
    auto buf = builder.build();

    ParsedMetadataBoundary meta;
    bool ok = parse_gguf_safe(buf.data(), buf.size(), meta);
    ASSERT_TRUE(ok);
    ASSERT_EQ(meta.tensor_count, 0u);
    ASSERT_EQ(meta.kv_count, 0u);
}

// 7. Empty Strings and Zero-Length Arrays
E2E_TEST_CASE(Tier2_R1, test_r1_empty_strings_and_empty_arrays) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("empty.string", "");
    builder.add_metadata_array_string("empty.tokens", {});

    auto buf = builder.build();
    ParsedMetadataBoundary meta;
    bool ok = parse_gguf_safe(buf.data(), buf.size(), meta);
    ASSERT_TRUE(ok);
    ASSERT_EQ(meta.strings["empty.string"], "");
    ASSERT_EQ(meta.string_arrays["empty.tokens"].size(), 0u);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
