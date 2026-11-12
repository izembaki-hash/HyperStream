#include "e2e/common/e2e_test_framework.hpp"
#include "e2e/common/gguf_test_utils.hpp"
#include "gguf_parser.hpp"
#include "bpe_tokenizer.hpp"
#include "net/socket.hpp"
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <random>
#include <algorithm>
#include <filesystem>
#include <numeric>

using namespace hyperstream;
using namespace hyperstream::test;

// =========================================================================
// SUITE 1: GGUF PARSER ADVERSARIAL STRESS
// =========================================================================

// 1.1 Corrupted Magic Numbers
E2E_TEST_CASE(GGUF_Challenger, test_corrupted_magic_numbers) {
    uint32_t bad_magics[] = {
        0x00000000, // All zeros
        0xFFFFFFFF, // All ones
        0x4C4D4747, // "GGML"
        0x46554746, // "GGUF" with last byte modified
        0xDEADBEEF,
        0x01020304
    };

    for (uint32_t magic : bad_magics) {
        SyntheticGGUFBuilder builder;
        builder.add_metadata_string("general.architecture", "qwen2");
        auto buf = builder.build();
        // Overwrite magic
        std::memcpy(buf.data(), &magic, sizeof(magic));

        GGUFParser parser;
        bool ok = parser.parse_from_memory(buf.data(), buf.size());
        ASSERT_FALSE(ok);
        ASSERT_EQ(parser.get_version(), 0u);
        ASSERT_EQ(parser.get_tensor_count(), 0u);
    }
}

// 1.2 Truncated Buffers & Boundary Underflow
E2E_TEST_CASE(GGUF_Challenger, test_truncated_buffers_and_underflow) {
    GGUFParser parser;

    // A. Null pointer or zero size
    ASSERT_FALSE(parser.parse_from_memory(nullptr, 0));
    uint8_t dummy = 0;
    ASSERT_FALSE(parser.parse_from_memory(&dummy, 0));

    // B. Sizes from 1 to 23 bytes (under minimum 24-byte header)
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("test.key", "test.val");
    auto full_buf = builder.build();

    for (size_t s = 1; s < 24; ++s) {
        ASSERT_FALSE(parser.parse_from_memory(full_buf.data(), s));
    }

    // C. Truncated at various points in metadata (1-byte alignment to avoid padding)
    SyntheticGGUFBuilder builder_nopad(1);
    builder_nopad.add_metadata_string("test.key", "test.val");
    auto nopad_buf = builder_nopad.build();

    size_t test_cutoffs[] = {24, 25, 30, 35, nopad_buf.size() - 5, nopad_buf.size() - 1};
    for (size_t cutoff : test_cutoffs) {
        if (cutoff < nopad_buf.size()) {
            ASSERT_FALSE(parser.parse_from_memory(nopad_buf.data(), cutoff));
        }
    }

    // D. Huge string length that could wrap size_t addition
    std::vector<uint8_t> wrap_buf(64, 0);
    uint32_t magic = 0x46554747;
    uint32_t version = 3;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 1;
    std::memcpy(wrap_buf.data() + 0, &magic, 4);
    std::memcpy(wrap_buf.data() + 4, &version, 4);
    std::memcpy(wrap_buf.data() + 8, &tensor_count, 8);
    std::memcpy(wrap_buf.data() + 16, &kv_count, 8);

    // Key length = 0xFFFFFFFFFFFFFFFE (huge near UINT64_MAX)
    uint64_t huge_len = 0xFFFFFFFFFFFFFFFEULL;
    std::memcpy(wrap_buf.data() + 24, &huge_len, 8);

    // Parsing must fail cleanly without crashing or throwing bad_alloc
    bool wrap_ok = false;
    try {
        wrap_ok = parser.parse_from_memory(wrap_buf.data(), wrap_buf.size());
    } catch (...) {
        wrap_ok = false;
    }
    ASSERT_FALSE(wrap_ok);
}

// 1.3 Out-of-Bounds Offsets
E2E_TEST_CASE(GGUF_Challenger, test_out_of_bounds_tensor_offsets) {
    SyntheticGGUFBuilder builder;
    // Tensor offset pointing far beyond the buffer
    builder.add_tensor("oob_tensor_1", {128, 64}, 8, 999999999ULL);
    builder.add_tensor("oob_tensor_2", {64}, 0, 0xFFFFFFFFFFFFFF00ULL);

    auto buf = builder.build();
    GGUFParser parser;
    bool ok = parser.parse_from_memory(buf.data(), buf.size());
    ASSERT_TRUE(ok);

    const auto* t1 = parser.find_tensor("oob_tensor_1");
    ASSERT_TRUE(t1 != nullptr);
    // Invariant: get_tensor_data MUST return nullptr for out-of-bounds offset
    ASSERT_EQ(parser.get_tensor_data(*t1), nullptr);

    const auto* t2 = parser.find_tensor("oob_tensor_2");
    ASSERT_TRUE(t2 != nullptr);
    ASSERT_EQ(parser.get_tensor_data(*t2), nullptr);

    // Invariant: non-existent tensor returns nullptr
    ASSERT_EQ(parser.find_tensor("does_not_exist"), nullptr);
}

// 1.4 Zero Tensor Count and Zero KV Count Boundaries
E2E_TEST_CASE(GGUF_Challenger, test_zero_tensor_and_kv_counts) {
    // A. 0 tensors, 0 KVs
    {
        SyntheticGGUFBuilder builder;
        auto buf = builder.build();
        GGUFParser parser;
        ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));
        ASSERT_EQ(parser.get_tensor_count(), 0u);
        ASSERT_EQ(parser.find_tensor("any"), nullptr);
    }

    // B. 0 tensors, 5 KVs
    {
        SyntheticGGUFBuilder builder;
        for (int i = 0; i < 5; ++i) {
            builder.add_metadata_u32("key_" + std::to_string(i), i * 10);
        }
        auto buf = builder.build();
        GGUFParser parser;
        ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));
        ASSERT_EQ(parser.get_tensor_count(), 0u);
        for (int i = 0; i < 5; ++i) {
            ASSERT_EQ(parser.get_u32("key_" + std::to_string(i)), static_cast<uint32_t>(i * 10));
        }
    }

    // C. 5 tensors, 0 KVs
    {
        SyntheticGGUFBuilder builder;
        for (int i = 0; i < 5; ++i) {
            builder.add_tensor("tensor_" + std::to_string(i), {64}, 0, i * 64, {0x01, 0x02});
        }
        auto buf = builder.build();
        GGUFParser parser;
        ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));
        ASSERT_EQ(parser.get_tensor_count(), 5u);
        for (int i = 0; i < 5; ++i) {
            ASSERT_TRUE(parser.find_tensor("tensor_" + std::to_string(i)) != nullptr);
        }
    }
}

// 1.5 Huge and Boundary Alignment Values
E2E_TEST_CASE(GGUF_Challenger, test_boundary_alignment_values) {
    uint32_t test_alignments[] = {0, 1, 16, 64, 128, 4096, 65536, 1048576};

    for (uint32_t align : test_alignments) {
        SyntheticGGUFBuilder builder(align == 0 ? 32 : align);
        builder.add_metadata_u32("general.alignment", align);
        builder.add_tensor("t1", {16}, 0, 0, {0x10, 0x20});

        auto buf = builder.build();
        GGUFParser parser;
        ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

        uint32_t effective_align = parser.get_alignment();
        if (align == 0) {
            ASSERT_EQ(effective_align, 32u); // Fallback to 32
        } else {
            ASSERT_EQ(effective_align, align);
        }

        uint64_t data_start = parser.get_data_start_offset();
        ASSERT_EQ(data_start % effective_align, 0u);
    }
}

// 1.6 Non-Existent and Type-Mismatched Metadata Keys
E2E_TEST_CASE(GGUF_Challenger, test_nonexistent_and_type_mismatched_keys) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("meta.str", "hello_world");
    builder.add_metadata_u32("meta.u32", 42);
    builder.add_metadata_f32("meta.f32", 3.14159f);
    builder.add_metadata_array_string("meta.arr", {"a", "b", "c"});

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    // Non-existent keys
    std::string s_out;
    uint32_t u32_out = 0;
    uint64_t u64_out = 0;
    float f32_out = 0.0f;
    bool b_out = false;
    std::vector<std::string> arr_out;

    ASSERT_FALSE(parser.get_metadata_string("missing.str", s_out));
    ASSERT_FALSE(parser.get_metadata_u32("missing.u32", u32_out));
    ASSERT_FALSE(parser.get_metadata_u64("missing.u64", u64_out));
    ASSERT_FALSE(parser.get_metadata_f32("missing.f32", f32_out));
    ASSERT_FALSE(parser.get_metadata_bool("missing.bool", b_out));
    ASSERT_FALSE(parser.get_metadata_array_string("missing.arr", arr_out));

    // Default return accessors
    ASSERT_STREQ(parser.get_string("missing.str", "default_val"), "default_val");
    ASSERT_EQ(parser.get_u32("missing.u32", 999), 999u);
    ASSERT_EQ(parser.get_u64("missing.u64", 8888ULL), 8888ULL);
    ASSERT_NEAR(parser.get_f32("missing.f32", 1.23f), 1.23f, 1e-4f);
    ASSERT_EQ(parser.get_bool("missing.bool", true), true);
    ASSERT_TRUE(parser.get_string_array("missing.arr").empty());

    // Type mismatch queries: getting a string key as an array
    ASSERT_FALSE(parser.get_metadata_array_string("meta.str", arr_out));
    // Getting an array key as a string
    ASSERT_FALSE(parser.get_metadata_string("meta.arr", s_out));
}

// 1.7 Real Model Torture & Tensor Lookup Stress
E2E_TEST_CASE(GGUF_Challenger, test_real_model_torture) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(real_path)) return;

    GGUFParser parser;
    ASSERT_TRUE(parser.open(real_path));
    ASSERT_EQ(parser.get_version(), 3u);

    // Stress lookup of 200 non-existent keys
    for (int i = 0; i < 200; ++i) {
        std::string dummy_key = "nonexistent.key.adversarial." + std::to_string(i);
        std::string s_val;
        ASSERT_FALSE(parser.get_metadata_string(dummy_key, s_val));
        ASSERT_EQ(parser.get_u32(dummy_key, 12345), 12345u);
        ASSERT_EQ(parser.find_tensor(dummy_key), nullptr);
    }

    // Verify all legitimate tensors have valid data pointers
    const auto& tensors = parser.get_tensors();
    ASSERT_TRUE(tensors.size() > 50);
    for (const auto& t : tensors) {
        const uint8_t* ptr = parser.get_tensor_data(t);
        ASSERT_TRUE(ptr != nullptr);
        // Offset sanity check
        ASSERT_TRUE(parser.get_tensor_data_offset(t) >= parser.get_data_start_offset());
    }

    parser.close();
}


// =========================================================================
// SUITE 2: BPE TOKENIZER ADVERSARIAL STRESS
// =========================================================================

// Helper to build a rich tokenizer with multilingual symbols and merges
static BPETokenizer make_adversarial_tokenizer() {
    SyntheticGGUFBuilder builder;
    std::vector<std::string> tokens = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<s>", "</s>",
        " ", "a", "b", "c", "d", "e", "h", "i", "l", "m", "n", "o", "r", "s", "t", "u", "w",
        "th", "he", "the", "in", "an", "is",
        // Multi-byte characters
        "\xD8\xB9", // Arabic Ayn 'ع'
        "\xD8\xB1", // Arabic Reh 'ر'
        "\xD8\xA8", // Arabic Beh 'ب'
        "\xE4\xBD\xA0", // Chinese Ni '你'
        "\xE5\xA5\xBD", // Chinese Hao '好'
        "\xF0\x9F\x9A\x80", // Emoji Rocket 🚀
        "\xF0\x9F\x94\xA5", // Emoji Fire 🔥
        // Control tokens
        "\n", "\r", "\t", " "
    };
    std::vector<std::string> merges = {
        "t h", "h e", "th e", "i n", "a n", "i s"
    };

    builder.add_metadata_array_string("tokenizer.ggml.tokens", tokens);
    builder.add_metadata_array_string("tokenizer.ggml.merges", merges);
    builder.add_metadata_u32("tokenizer.ggml.eos_token_id", 0);
    builder.add_metadata_u32("tokenizer.ggml.bos_token_id", 3);

    auto buf = builder.build();
    GGUFParser parser;
    parser.parse_from_memory(buf.data(), buf.size());

    BPETokenizer tok;
    tok.load_from_gguf(parser);
    return tok;
}

// 2.1 Non-ASCII UTF-8 Sequences (Arabic, Chinese, Emojis)
E2E_TEST_CASE(BPE_Challenger, test_non_ascii_multilingual_encoding) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    BPETokenizer tokenizer;
    bool has_real = false;

    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        if (parser.open(real_path)) {
            has_real = tokenizer.load_from_gguf(parser);
        }
    }
    if (!has_real) {
        tokenizer = make_adversarial_tokenizer();
    }

    // A. Arabic
    std::string arabic_sample = "مرحبا بالعالم ، هذا نموذج ذكاء اصطناعي";
    auto ar_ids = tokenizer.encode(arabic_sample);
    ASSERT_FALSE(ar_ids.empty());
    std::string ar_dec = tokenizer.decode(ar_ids);
    ASSERT_STREQ(ar_dec, arabic_sample);

    // B. Chinese
    std::string chinese_sample = "你好世界，这是高并发推理引擎测试。";
    auto zh_ids = tokenizer.encode(chinese_sample);
    ASSERT_FALSE(zh_ids.empty());
    std::string zh_dec = tokenizer.decode(zh_ids);
    ASSERT_STREQ(zh_dec, chinese_sample);

    // C. Emojis
    std::string emoji_sample = "HyperStream 🚀⚡🧠🔥 Bare-Metal Vulkan";
    auto em_ids = tokenizer.encode(emoji_sample);
    ASSERT_FALSE(em_ids.empty());
    std::string em_dec = tokenizer.decode(em_ids);
    ASSERT_STREQ(em_dec, emoji_sample);

    // D. Mixed Multilingual
    std::string mixed_sample = "Hello مرحبا 你好 🚀 Code: 42";
    auto mx_ids = tokenizer.encode(mixed_sample);
    ASSERT_FALSE(mx_ids.empty());
    std::string mx_dec = tokenizer.decode(mx_ids);
    ASSERT_STREQ(mx_dec, mixed_sample);
}

// 2.2 Control Characters Handling
E2E_TEST_CASE(BPE_Challenger, test_control_characters_handling) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    BPETokenizer tokenizer;
    bool loaded = false;
    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        if (parser.open(real_path)) {
            loaded = tokenizer.load_from_gguf(parser);
        }
    }
    if (!loaded) tokenizer = make_adversarial_tokenizer();

    // Strings with tabs, newlines, carriage returns, and control chars
    std::string ctrl_str = "Line 1\r\nLine 2\tTabbed\n\x1B[31mRed\x1B[0m\x7F";
    auto ids = tokenizer.encode(ctrl_str);
    ASSERT_FALSE(ids.empty());
    std::string decoded = tokenizer.decode(ids);
    ASSERT_STREQ(decoded, ctrl_str);
}

// 2.3 Adjacent Special Tokens
E2E_TEST_CASE(BPE_Challenger, test_adjacent_special_tokens) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    BPETokenizer tokenizer;
    bool loaded = false;
    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        if (parser.open(real_path)) {
            loaded = tokenizer.load_from_gguf(parser);
        }
    }
    if (!loaded) tokenizer = make_adversarial_tokenizer();

    // Adjacent special tokens with zero intermediate characters
    std::string prompt1 = "<|im_start|><|im_end|>";
    auto ids1 = tokenizer.encode(prompt1);
    ASSERT_EQ(ids1.size(), 2u);
    ASSERT_EQ(ids1[0], tokenizer.im_start_token());
    ASSERT_EQ(ids1[1], tokenizer.im_end_token());
    ASSERT_STREQ(tokenizer.decode(ids1), prompt1);

    // Repeated special tokens
    std::string prompt2 = "<|im_start|><|im_start|><|im_end|><|im_end|>";
    auto ids2 = tokenizer.encode(prompt2);
    ASSERT_EQ(ids2.size(), 4u);
    ASSERT_EQ(ids2[0], tokenizer.im_start_token());
    ASSERT_EQ(ids2[1], tokenizer.im_start_token());
    ASSERT_EQ(ids2[2], tokenizer.im_end_token());
    ASSERT_EQ(ids2[3], tokenizer.im_end_token());
    ASSERT_STREQ(tokenizer.decode(ids2), prompt2);

    // Multi-turn ChatML prompt
    std::string chatml = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    auto chat_ids = tokenizer.encode(chatml);
    ASSERT_FALSE(chat_ids.empty());
    std::string chat_dec = tokenizer.decode(chat_ids);
    ASSERT_STREQ(chat_dec, chatml);
}

// 2.4 Empty Strings and Boundary Inputs
E2E_TEST_CASE(BPE_Challenger, test_empty_strings_and_single_characters) {
    auto tokenizer = make_adversarial_tokenizer();

    // Empty string without BOS
    auto empty_ids = tokenizer.encode("", false);
    ASSERT_TRUE(empty_ids.empty());

    // Empty string with BOS
    auto bos_ids = tokenizer.encode("", true);
    ASSERT_EQ(bos_ids.size(), 1u);
    ASSERT_EQ(bos_ids[0], tokenizer.bos_token());

    // Single spaces
    auto space_ids = tokenizer.encode(" ");
    ASSERT_FALSE(space_ids.empty());

    // Single characters
    std::vector<std::string> single_chars = {"a", "b", "c", "\n", "\t"};
    for (const auto& ch : single_chars) {
        auto ch_ids = tokenizer.encode(ch);
        ASSERT_FALSE(ch_ids.empty());
        ASSERT_STREQ(tokenizer.decode(ch_ids), ch);
    }
}

// 2.5 Repeated Patterns & Deep Merge Loops
E2E_TEST_CASE(BPE_Challenger, test_repeated_patterns_and_deep_merges) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    BPETokenizer tokenizer;
    bool loaded = false;
    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        if (parser.open(real_path)) {
            loaded = tokenizer.load_from_gguf(parser);
        }
    }
    if (!loaded) tokenizer = make_adversarial_tokenizer();

    // 2,000 repetitive characters: "aaaaaaaa..."
    std::string repeat_a(2000, 'a');
    auto t0 = std::chrono::high_resolution_clock::now();
    auto ids_a = tokenizer.encode(repeat_a);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    ASSERT_FALSE(ids_a.empty());
    // Invariant: Must encode within reasonable time (< 500ms) without infinite looping
    ASSERT_TRUE(ms < 500.0);
    std::string dec_a = tokenizer.decode(ids_a);
    ASSERT_STREQ(dec_a, repeat_a);

    // 1,000 repetitions of word pattern: "the the the the..."
    std::string repeat_words;
    for (int i = 0; i < 500; ++i) repeat_words += "the ";
    auto ids_words = tokenizer.encode(repeat_words);
    ASSERT_FALSE(ids_words.empty());
    std::string dec_words = tokenizer.decode(ids_words);
    ASSERT_STREQ(dec_words, repeat_words);
}

// 2.6 Out of Vocabulary Token Decoding
E2E_TEST_CASE(BPE_Challenger, test_out_of_vocab_token_decoding) {
    auto tokenizer = make_adversarial_tokenizer();
    int32_t v_size = static_cast<int32_t>(tokenizer.vocab_size());

    int32_t bad_ids[] = {-1, -100, -999999, v_size, v_size + 1, v_size + 100000, 2147483647};
    for (int32_t bad_id : bad_ids) {
        ASSERT_STREQ(tokenizer.decode(bad_id), "");
        ASSERT_FALSE(tokenizer.is_special_token(bad_id));
    }

    // Vector with mixed valid and invalid tokens
    std::vector<int32_t> mixed_tokens = {-1, tokenizer.im_start_token(), v_size + 50, tokenizer.im_end_token(), -999};
    std::string dec_mixed = tokenizer.decode(mixed_tokens);
    std::string expected = std::string("<|im_start|>") + "<|im_end|>";
    ASSERT_STREQ(dec_mixed, expected);
}


// =========================================================================
// SUITE 3: SOCKET LAYER ADVERSARIAL STRESS
// =========================================================================

// 3.1 Rapid Connect / Close Cycles (Zero Leaks)
E2E_TEST_CASE(Socket_Challenger, test_rapid_connect_close_cycles) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28100;

    Socket server;
    ASSERT_TRUE(server.listen(port));

    constexpr int CYCLES = 50;
    std::thread client_thread([&]() {
        for (int i = 0; i < CYCLES; ++i) {
            Socket client;
            bool ok = client.connect("127.0.0.1", port);
            if (ok) {
                uint32_t ping = static_cast<uint32_t>(i);
                client.send(&ping, sizeof(ping));
                uint32_t pong = 0;
                client.recv(&pong, sizeof(pong));
            }
            client.close();
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    });

    for (int i = 0; i < CYCLES; ++i) {
        Socket accepted;
        bool ok = server.accept(accepted);
        ASSERT_TRUE(ok);
        uint32_t ping = 0;
        int n = accepted.recv(&ping, sizeof(ping));
        ASSERT_EQ(n, static_cast<int>(sizeof(ping)));
        ASSERT_EQ(ping, static_cast<uint32_t>(i));
        accepted.send(&ping, sizeof(ping));
        accepted.close();
    }

    client_thread.join();
    server.close();
}

// 3.2 Large Payload Transmission (2MB Exact Byte Integrity)
E2E_TEST_CASE(Socket_Challenger, test_large_payload_transmission_2mb) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28101;

    Socket server;
    ASSERT_TRUE(server.listen(port));

    constexpr size_t PAYLOAD_SIZE = 2 * 1024 * 1024; // 2 Megabytes
    std::vector<uint8_t> send_buf(PAYLOAD_SIZE);
    std::mt19937 rng(42);
    for (size_t i = 0; i < PAYLOAD_SIZE; ++i) {
        send_buf[i] = static_cast<uint8_t>(rng() & 0xFF);
    }

    std::vector<uint8_t> recv_buf(PAYLOAD_SIZE, 0);

    std::thread client_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Socket client;
        if (client.connect("127.0.0.1", port)) {
            int sent = client.send(send_buf.data(), send_buf.size());
            ASSERT_EQ(sent, static_cast<int>(PAYLOAD_SIZE));
            client.close();
        }
    });

    Socket accepted;
    ASSERT_TRUE(server.accept(accepted));

    size_t total_received = 0;
    while (total_received < PAYLOAD_SIZE) {
        int n = accepted.recv(recv_buf.data() + total_received, PAYLOAD_SIZE - total_received);
        if (n <= 0) break;
        total_received += static_cast<size_t>(n);
    }

    client_thread.join();
    accepted.close();
    server.close();

    ASSERT_EQ(total_received, PAYLOAD_SIZE);
    ASSERT_TRUE(std::memcmp(send_buf.data(), recv_buf.data(), PAYLOAD_SIZE) == 0);
}

// 3.3 Zero Byte Send & Boundary Calls
E2E_TEST_CASE(Socket_Challenger, test_zero_byte_send) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28102;

    Socket server;
    ASSERT_TRUE(server.listen(port));

    std::thread client_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Socket client;
        if (client.connect("127.0.0.1", port)) {
            // Send 0 bytes with non-null pointer
            char dummy = 'x';
            int res1 = client.send(&dummy, 0);
            ASSERT_EQ(res1, 0);

            // Send 0 bytes with null pointer
            int res2 = client.send(nullptr, 0);
            ASSERT_EQ(res2, 0);

            // Followed by valid payload
            client.send("ACK", 3);
            client.close();
        }
    });

    Socket accepted;
    ASSERT_TRUE(server.accept(accepted));
    char buf[16] = {0};
    int n = accepted.recv(buf, sizeof(buf));
    ASSERT_EQ(n, 3);
    ASSERT_STREQ(std::string(buf, 3), "ACK");

    client_thread.join();
    accepted.close();
    server.close();
}

// 3.4 Recv on Closed Socket & Abrupt Disconnect
E2E_TEST_CASE(Socket_Challenger, test_recv_on_closed_socket) {
    ASSERT_TRUE(Socket::init_networking());

    // A. Recv on unopened socket must return -1
    Socket uninit;
    char buf[16];
    ASSERT_EQ(uninit.recv(buf, sizeof(buf)), -1);

    // B. Recv on closed socket must return -1
    Socket s;
    s.listen(28103);
    s.close();
    ASSERT_EQ(s.recv(buf, sizeof(buf)), -1);

    // C. Client abruptly closes after sending 1 byte -> next recv returns 0 (EOF)
    uint16_t port = 28104;
    Socket server;
    ASSERT_TRUE(server.listen(port));

    std::thread client_thread([&]() {
        Socket client;
        if (client.connect("127.0.0.1", port)) {
            client.send("A", 1);
            client.close(); // Abrupt close
        }
    });

    Socket accepted;
    ASSERT_TRUE(server.accept(accepted));
    char r_buf[16] = {0};
    int r1 = accepted.recv(r_buf, 1);
    ASSERT_EQ(r1, 1);
    ASSERT_EQ(r_buf[0], 'A');

    // Next recv must immediately return 0 (graceful EOF) or -1, and NOT hang
    int r2 = accepted.recv(r_buf, 1);
    ASSERT_TRUE(r2 <= 0);

    client_thread.join();
    accepted.close();
    server.close();
}

// 3.5 Port Collision Handling
E2E_TEST_CASE(Socket_Challenger, test_port_collision_handling) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28105;

    Socket server1;
    ASSERT_TRUE(server1.listen(port));
    ASSERT_TRUE(server1.is_valid());

    // Attempting to listen on the exact same port
    Socket server2;
    // On POSIX/Linux, two binds to the same port without SO_REUSEPORT fail.
    // On Windows with SO_REUSEADDR, Winsock may allow binding or reject depending on OS security policy.
    bool ok2 = server2.listen(port);
    // Even if it succeeds or fails, neither socket may crash or segfault!
    if (!ok2) {
        ASSERT_FALSE(server2.is_valid());
    } else {
        server2.close();
    }

    server1.close();
    Socket::cleanup_networking();
}

// 3.6 Broken Pipe / Send After Peer Abrupt Close
E2E_TEST_CASE(Socket_Challenger, test_send_after_peer_abrupt_close) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28106;

    Socket server;
    ASSERT_TRUE(server.listen(port));

    std::thread client_thread([&]() {
        Socket client;
        if (client.connect("127.0.0.1", port)) {
            // Read 1 byte then immediately close socket
            char b = 0;
            client.recv(&b, 1);
            client.close();
        }
    });

    Socket accepted;
    ASSERT_TRUE(server.accept(accepted));
    // Send 1 byte so client unblocks
    accepted.send("X", 1);

    // Wait for client to close
    client_thread.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Sending to closed peer should return -1 or partial, but MUST NEVER CRASH
    char dummy_payload[1024];
    std::memset(dummy_payload, 'A', sizeof(dummy_payload));
    int res = accepted.send(dummy_payload, sizeof(dummy_payload));
    // Must return <= 0 (connection reset / broken pipe)
    ASSERT_TRUE(res <= 0);

    accepted.close();
    server.close();
}

// 1.8 Invalid GGUF Versions (v1, v4, v99)
E2E_TEST_CASE(GGUF_Challenger, test_invalid_gguf_versions) {
    uint32_t bad_versions[] = {0, 1, 4, 5, 99, 0xFFFFFFFF};
    for (uint32_t v : bad_versions) {
        SyntheticGGUFBuilder builder;
        builder.add_metadata_string("arch", "qwen2");
        auto buf = builder.build(false, v);

        GGUFParser parser;
        bool ok = parser.parse_from_memory(buf.data(), buf.size());
        ASSERT_FALSE(ok);
        ASSERT_EQ(parser.get_version(), 0u);
    }
}

// 2.7 Malformed UTF-8 Sequences and Massive Text
E2E_TEST_CASE(BPE_Challenger, test_malformed_utf8_and_massive_text) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    BPETokenizer tokenizer;
    bool loaded = false;
    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        if (parser.open(real_path)) {
            loaded = tokenizer.load_from_gguf(parser);
        }
    }
    if (!loaded) tokenizer = make_adversarial_tokenizer();

    // A. Malformed UTF-8 byte sequences
    std::string malformed;
    malformed += "\xC0\xAF"; // Overlong encoding
    malformed += "\xED\xA0\x80"; // Surrogate half
    malformed += "\xF4\x90\x80\x80"; // Beyond U+10FFFF
    malformed += "\x80\x81\x82"; // Stray continuation bytes
    malformed += "\xFE\xFF"; // Invalid bytes

    // Invariant: Must not crash, hang, or throw
    auto mal_ids = tokenizer.encode(malformed);
    std::string mal_dec = tokenizer.decode(mal_ids);

    // B. Massive 100KB input string
    std::string massive;
    massive.reserve(102400);
    const std::string seed_phrase = "HyperStream inference kernel online softmax LDS 64KB tokens; ";
    while (massive.size() < 100000) {
        massive += seed_phrase;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto massive_ids = tokenizer.encode(massive);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    ASSERT_FALSE(massive_ids.empty());
    // Encode 100KB in reasonable time (< 2000 ms)
    ASSERT_TRUE(ms < 2000.0);
    std::string massive_dec = tokenizer.decode(massive_ids);
    ASSERT_STREQ(massive_dec, massive);
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
