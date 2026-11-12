#include "e2e/common/e2e_test_framework.hpp"
#include "e2e/common/gguf_test_utils.hpp"
#include "gguf_parser.hpp"
#include "bpe_tokenizer.hpp"
#include <filesystem>
#include <iostream>
#include <unordered_set>

using namespace hyperstream;
using namespace hyperstream::test;

// 1. All 151,936 vocab tokens completeness, matching expected strings and anchor points
E2E_TEST_CASE(RealModelVerificationTest, test_real_model_vocab_completeness_and_anchor_tokens) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    ASSERT_TRUE(std::filesystem::exists(real_path));

    GGUFParser parser;
    ASSERT_TRUE(parser.open(real_path));
    ASSERT_EQ(parser.get_version(), 3u);

    const auto& tokens = parser.get_string_array("tokenizer.ggml.tokens");
    // Verify exact vocab size: 151,936 tokens
    ASSERT_EQ(tokens.size(), 151936ULL);

    // In Qwen2 / GPT-2 BPE vocabularies, token 0 is "!"
    ASSERT_STREQ(tokens[0], "!");
    ASSERT_STREQ(tokens[1], "\"");
    ASSERT_STREQ(tokens[2], "#");

    // Verify critical anchor tokens in Qwen2.5-Coder vocabulary
    ASSERT_STREQ(tokens[151643], "<|endoftext|>");
    ASSERT_STREQ(tokens[151644], "<|im_start|>");
    ASSERT_STREQ(tokens[151645], "<|im_end|>");
    ASSERT_STREQ(tokens[151646], "<|object_ref_start|>");
    ASSERT_STREQ(tokens[151647], "<|object_ref_end|>");
    ASSERT_STREQ(tokens[151648], "<|box_start|>");
    ASSERT_STREQ(tokens[151649], "<|box_end|>");
    ASSERT_STREQ(tokens[151650], "<|quad_start|>");
    ASSERT_STREQ(tokens[151651], "<|quad_end|>");
    ASSERT_STREQ(tokens[151652], "<|vision_start|>");
    ASSERT_STREQ(tokens[151653], "<|vision_end|>");
    ASSERT_STREQ(tokens[151654], "<|vision_pad|>");
    ASSERT_STREQ(tokens[151655], "<|image_pad|>");
    ASSERT_STREQ(tokens[151656], "<|video_pad|>");
    ASSERT_STREQ(tokens[151657], "<tool_call>");
    ASSERT_STREQ(tokens[151658], "</tool_call>");
    ASSERT_STREQ(tokens[151659], "<|fim_prefix|>");
    ASSERT_STREQ(tokens[151660], "<|fim_middle|>");
    ASSERT_STREQ(tokens[151661], "<|fim_suffix|>");
    ASSERT_STREQ(tokens[151662], "<|fim_pad|>");
    ASSERT_STREQ(tokens[151663], "<|repo_name|>");
    ASSERT_STREQ(tokens[151664], "<|file_sep|>");

    // Ingest into BPETokenizer
    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));
    ASSERT_EQ(tokenizer.vocab_size(), 151936ULL);

    // Verify all 151,936 tokens can be decoded without crashing
    for (size_t i = 0; i < 151936; ++i) {
        std::string piece = tokenizer.decode(static_cast<int32_t>(i));
        (void)piece;
    }

    // Verify token reverse map fidelity
    const auto& id_to_tok = tokenizer.id_to_token();
    ASSERT_EQ(id_to_tok.size(), 151936ULL);
    ASSERT_STREQ(id_to_tok[151644], "<|im_start|>");
    ASSERT_STREQ(id_to_tok[151645], "<|im_end|>");
    ASSERT_STREQ(id_to_tok[151643], "<|endoftext|>");
}

// 2. All 151,387 merges load without errors
E2E_TEST_CASE(RealModelVerificationTest, test_real_model_merges_loading_and_rank_fidelity) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    GGUFParser parser;
    ASSERT_TRUE(parser.open(real_path));

    const auto& raw_merges = parser.get_string_array("tokenizer.ggml.merges");
    // Verify exact raw merges count in GGUF: 151,387
    ASSERT_EQ(raw_merges.size(), 151387ULL);

    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));

    const auto& merges = tokenizer.merges();
    // Verify all 151,387 merges loaded cleanly into internal merge map
    ASSERT_EQ(merges.size(), 151387ULL);

    // Verify first and last merge rules
    const std::string& first_merge = raw_merges[0];
    size_t sp1 = first_merge.find(' ');
    ASSERT_TRUE(sp1 != std::string::npos);
    std::string p1_a = first_merge.substr(0, sp1);
    std::string p1_b = first_merge.substr(sp1 + 1);
    auto it_first = merges.find({p1_a, p1_b});
    ASSERT_TRUE(it_first != merges.end());
    ASSERT_EQ(it_first->second, 0);

    const std::string& last_merge = raw_merges[151386];
    size_t sp_last = last_merge.find(' ');
    ASSERT_TRUE(sp_last != std::string::npos);
    std::string plast_a = last_merge.substr(0, sp_last);
    std::string plast_b = last_merge.substr(sp_last + 1);
    auto it_last = merges.find({plast_a, plast_b});
    ASSERT_TRUE(it_last != merges.end());
    ASSERT_EQ(it_last->second, 151386);
}

// 3. Dynamic Special Tokens Parsing Without Hardcoding
E2E_TEST_CASE(RealModelVerificationTest, test_special_tokens_dynamic_parsing_adversarial) {
    // 3a: Verify real model dynamic resolution
    {
        const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
        GGUFParser parser;
        ASSERT_TRUE(parser.open(real_path));

        BPETokenizer tokenizer;
        ASSERT_TRUE(tokenizer.load_from_gguf(parser));
        ASSERT_EQ(tokenizer.im_start_token(), 151644);
        ASSERT_EQ(tokenizer.im_end_token(), 151645);
        ASSERT_TRUE(tokenizer.is_special_token(151644));
        ASSERT_TRUE(tokenizer.is_special_token(151645));
    }

    // 3b: Adversarial test - synthetic model with non-standard IDs:
    // im_start at ID 7, im_end at ID 8, eos at ID 12
    {
        SyntheticGGUFBuilder builder;
        std::vector<std::string> custom_tokens = {
            "token_0", "token_1", "token_2", "token_3", "token_4",
            "token_5", "token_6", "<|im_start|>", "<|im_end|>", "token_9",
            "token_10", "token_11", "</s>", "token_13"
        };
        builder.add_metadata_array_string("tokenizer.ggml.tokens", custom_tokens);
        builder.add_metadata_u32("tokenizer.ggml.eos_token_id", 12);
        builder.add_metadata_u32("tokenizer.ggml.bos_token_id", 7);

        auto buf = builder.build();
        GGUFParser custom_parser;
        ASSERT_TRUE(custom_parser.parse_from_memory(buf.data(), buf.size()));

        BPETokenizer custom_tok;
        ASSERT_TRUE(custom_tok.load_from_gguf(custom_parser));

        // Must dynamically discover <|im_start|> is token 7, NOT 151644!
        ASSERT_EQ(custom_tok.im_start_token(), 7);
        // Must dynamically discover <|im_end|> is token 8, NOT 151645!
        ASSERT_EQ(custom_tok.im_end_token(), 8);
        // Must dynamically resolve eos token from metadata (12)
        ASSERT_EQ(custom_tok.eos_token(), 12);
        ASSERT_EQ(custom_tok.bos_token(), 7);

        ASSERT_TRUE(custom_tok.is_special_token(7));
        ASSERT_TRUE(custom_tok.is_special_token(8));
        ASSERT_FALSE(custom_tok.is_special_token(0));
    }
}

// 4. Prompt Encoding/Decoding Multi-Line C++ Roundtrip Fidelity (100% Exact Match)
E2E_TEST_CASE(RealModelVerificationTest, test_multiline_cpp_prompt_roundtrip_fidelity) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    GGUFParser parser;
    ASSERT_TRUE(parser.open(real_path));

    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));

    // Multi-line C++ code with templates, pointers, bitwise shifts, quotes, comments
    const std::string cpp_code = 
        "#include <iostream>\n"
        "#include <vector>\n"
        "#include <string>\n"
        "#include <memory>\n"
        "\n"
        "// Fast reciprocal square root implementation\n"
        "float Q_rsqrt(float number) {\n"
        "    long i;\n"
        "    float x2, y;\n"
        "    const float threehalfs = 1.5F;\n"
        "    x2 = number * 0.5F;\n"
        "    y  = number;\n"
        "    i  = * ( long * ) &y;\n"
        "    i  = 0x5f3759df - ( i >> 1 );\n"
        "    y  = * ( float * ) &i;\n"
        "    y  = y * ( threehalfs - ( x2 * y * y ) );\n"
        "    return y;\n"
        "}\n"
        "\n"
        "template<typename T, size_t N>\n"
        "class MatrixBuffer {\n"
        "public:\n"
        "    MatrixBuffer() : m_data(std::make_unique<T[]>(N)) {}\n"
        "    T& operator[](size_t idx) { return m_data[idx]; }\n"
        "    constexpr size_t size() const noexcept { return N; }\n"
        "private:\n"
        "    std::unique_ptr<T[]> m_data;\n"
        "};\n"
        "\n"
        "int main(int argc, char** argv) {\n"
        "    std::cout << \"[INFO] Running Q_rsqrt tests...\" << std::endl;\n"
        "    MatrixBuffer<float, 64> mat;\n"
        "    for (size_t i = 0; i < mat.size(); ++i) {\n"
        "        mat[i] = Q_rsqrt(static_cast<float>(i + 1));\n"
        "    }\n"
        "    std::cout << \"[SUCCESS] Matrix filled!\" << std::endl;\n"
        "    return 0;\n"
        "}\n";

    auto encoded = tokenizer.encode(cpp_code);
    ASSERT_TRUE(!encoded.empty());

    std::string decoded = tokenizer.decode(encoded);

    // 100% Roundtrip Fidelity Assertion
    ASSERT_STREQ(decoded, cpp_code);
    ASSERT_EQ(decoded.size(), cpp_code.size());

    // Also test ChatML conversation wrapper roundtrip
    const std::string chatml_prompt =
        "<|im_start|>system\n"
        "You are HyperStream's C++17 inference engine.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Explain std::memory_order_relaxed vs std::memory_order_seq_cst.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto chatml_encoded = tokenizer.encode(chatml_prompt);
    ASSERT_TRUE(!chatml_encoded.empty());
    // First token must be <|im_start|> (151644)
    ASSERT_EQ(chatml_encoded[0], 151644);

    std::string chatml_decoded = tokenizer.decode(chatml_encoded);
    ASSERT_STREQ(chatml_decoded, chatml_prompt);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
