#include "e2e/common/e2e_test_framework.hpp"
#include "e2e/common/gguf_test_utils.hpp"
#include "gguf_parser.hpp"
#include "bpe_tokenizer.hpp"
#include <filesystem>

using namespace hyperstream;
using namespace hyperstream::test;

// 1. Vocabulary & Merges Loading from GGUF Metadata
E2E_TEST_CASE(BPETokenizerTest, test_load_from_gguf_metadata) {
    SyntheticGGUFBuilder builder;
    std::vector<std::string> tokens = {
        "<|endoftext|>", "H", "e", "l", "o", "w", "r", "d", " ", ",", "!",
        "He", "ll", "lo", "Hello", "world", "Hello,", " Hello"
    };
    std::vector<std::string> merges = {
        "H e", "l l", "l o", "He ll", "Hell o", "w o", "wo r", "wor l", "worl d", "  H", " H e"
    };

    builder.add_metadata_array_string("tokenizer.ggml.tokens", tokens);
    builder.add_metadata_array_string("tokenizer.ggml.merges", merges);
    builder.add_metadata_u32("tokenizer.ggml.eos_token_id", 0);
    builder.add_metadata_u32("tokenizer.ggml.bos_token_id", 0);

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));
    ASSERT_EQ(tokenizer.vocab_size(), tokens.size());
    ASSERT_EQ(tokenizer.eos_token(), 0);
    ASSERT_EQ(tokenizer.bos_token(), 0);
}

// 2. Special Tokens Discovery
E2E_TEST_CASE(BPETokenizerTest, test_special_tokens_discovery) {
    SyntheticGGUFBuilder builder;
    std::vector<std::string> tokens = {
        "<|endoftext|>", "a", "b", "<|im_start|>", "<|im_end|>", "<|pad|>"
    };
    builder.add_metadata_array_string("tokenizer.ggml.tokens", tokens);
    builder.add_metadata_u32("tokenizer.ggml.eos_token_id", 0);

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));

    ASSERT_EQ(tokenizer.im_start_token(), 3);
    ASSERT_EQ(tokenizer.im_end_token(), 4);
    ASSERT_EQ(tokenizer.eos_token(), 0);
    ASSERT_TRUE(tokenizer.is_special_token(3));
    ASSERT_TRUE(tokenizer.is_special_token(4));
    ASSERT_FALSE(tokenizer.is_special_token(1));
}

// 3. BPE Token Encoding with Merge Ranks
E2E_TEST_CASE(BPETokenizerTest, test_bpe_encoding_merges) {
    SyntheticGGUFBuilder builder;
    std::vector<std::string> tokens = {
        "H", "e", "l", "o", "He", "lo", "Hel", "lo"
    };
    std::vector<std::string> merges = {
        "H e", "l o"
    };
    builder.add_metadata_array_string("tokenizer.ggml.tokens", tokens);
    builder.add_metadata_array_string("tokenizer.ggml.merges", merges);

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));

    // "He" should merge H + e -> He (token 4)
    auto ids = tokenizer.encode("He");
    ASSERT_EQ(ids.size(), 1u);
    ASSERT_EQ(ids[0], 4);
}

// 4. Token Decoding (Single & Batch)
E2E_TEST_CASE(BPETokenizerTest, test_token_decoding) {
    SyntheticGGUFBuilder builder;
    std::vector<std::string> tokens = {
        "Hello", "world", "!", "<|im_start|>", "<|im_end|>"
    };
    builder.add_metadata_array_string("tokenizer.ggml.tokens", tokens);

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));

    ASSERT_STREQ(tokenizer.decode(0), "Hello");
    ASSERT_STREQ(tokenizer.decode(1), "world");
    ASSERT_STREQ(tokenizer.decode(2), "!");
    ASSERT_STREQ(tokenizer.decode(3), "<|im_start|>");
    ASSERT_STREQ(tokenizer.decode(4), "<|im_end|>");

    // Batch decoding
    std::vector<int32_t> batch = {0, 1, 2};
    std::string decoded = tokenizer.decode(batch);
    ASSERT_STREQ(decoded, "Helloworld!");
}

// 5. Special Tokens Preservation in Prompts
E2E_TEST_CASE(BPETokenizerTest, test_special_tokens_preservation) {
    SyntheticGGUFBuilder builder;
    std::vector<std::string> tokens = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "system", "user", "hi"
    };
    builder.add_metadata_array_string("tokenizer.ggml.tokens", tokens);

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));

    std::string prompt = "<|im_start|>userhi<|im_end|>";
    auto ids = tokenizer.encode(prompt);
    ASSERT_EQ(ids.size(), 4u);
    ASSERT_EQ(ids[0], 1); // <|im_start|>
    ASSERT_EQ(ids[1], 4); // user
    ASSERT_EQ(ids[2], 5); // hi
    ASSERT_EQ(ids[3], 2); // <|im_end|>

    std::string roundtrip = tokenizer.decode(ids);
    ASSERT_STREQ(roundtrip, prompt);
}

// 6. Out of Bounds Token Handling
E2E_TEST_CASE(BPETokenizerTest, test_out_of_bounds_handling) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_array_string("tokenizer.ggml.tokens", {"a", "b"});

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    BPETokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load_from_gguf(parser));

    ASSERT_STREQ(tokenizer.decode(-1), "");
    ASSERT_STREQ(tokenizer.decode(100), "");
    ASSERT_STREQ(tokenizer.decode(999999), "");
}

// 7. Real Model Tokenizer Inspection (if available)
E2E_TEST_CASE(BPETokenizerTest, test_real_model_tokenizer_loading_and_encoding) {
    std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(real_path)) {
        real_path = "models/qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    }

    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        ASSERT_TRUE(parser.open(real_path));

        BPETokenizer tokenizer;
        ASSERT_TRUE(tokenizer.load_from_gguf(parser));
        ASSERT_TRUE(tokenizer.vocab_size() >= 151936);

        int32_t im_start = tokenizer.im_start_token();
        int32_t im_end = tokenizer.im_end_token();
        ASSERT_EQ(im_start, 151644);
        ASSERT_EQ(im_end, 151645);

        // Test encoding and decoding
        std::string sample = "int main() { return 0; }";
        auto tokens = tokenizer.encode(sample);
        ASSERT_TRUE(!tokens.empty());

        std::string decoded = tokenizer.decode(tokens);
        ASSERT_STREQ(decoded, sample);
    }
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
