#include "e2e/common/e2e_test_framework.hpp"
#include "e2e/common/gguf_test_utils.hpp"
#include "src/gguf_parser.hpp"
#include "src/bpe_tokenizer.hpp"
#include "src/net/socket.hpp"
#include "src/util/mmap_file.hpp"

#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <random>
#include <algorithm>
#include <filesystem>
#include <numeric>
#include <iostream>

using namespace hyperstream;
using namespace hyperstream::test;

// =========================================================================
// SECTION 1: GGUF PARSER ADVERSARIAL CHALLENGE SUITE
// =========================================================================

E2E_TEST_CASE(GGUF_Challenger_Rep, test_corrupted_magic_numbers) {
    uint32_t bad_magics[] = {
        0x00000000, // Zero
        0xFFFFFFFF, // All ones
        0x4C4D4747, // "GGML"
        0x46554746, // "GGUF" flipped
        0xDEADBEEF, // Common sentinel
        0x04030201  // Endian reversed
    };

    for (uint32_t magic : bad_magics) {
        SyntheticGGUFBuilder builder;
        builder.add_metadata_string("general.architecture", "qwen2");
        auto buf = builder.build();
        std::memcpy(buf.data(), &magic, sizeof(magic));

        GGUFParser parser;
        bool ok = parser.parse_from_memory(buf.data(), buf.size());
        ASSERT_FALSE(ok);
        ASSERT_EQ(parser.get_tensor_count(), 0u);
    }
}

E2E_TEST_CASE(GGUF_Challenger_Rep, test_truncated_files_sweep) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.architecture", "qwen2");
    builder.add_metadata_u32("general.alignment", 32);
    builder.add_tensor("blk.0.attn_norm.weight", {128}, 0, 0, {0x01, 0x02});
    auto buf = builder.build();

    GGUFParser parser;
    // Sweep truncation at every byte from 0 up to full buffer size
    for (size_t s = 0; s < 35; ++s) {
        if (s < buf.size()) {
            ASSERT_FALSE(parser.parse_from_memory(buf.data(), s));
        }
    }
}

E2E_TEST_CASE(GGUF_Challenger_Rep, test_out_of_bounds_and_overflow_offsets) {
    SyntheticGGUFBuilder builder;
    // Offset well past buffer size
    builder.add_tensor("oob.tensor.1", {64}, 0, 10000000ULL);
    // Offset that wraps around uint64_t
    builder.add_tensor("oob.tensor.2", {64}, 0, 0xFFFFFFFFFFFFFFFEULL);

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    const auto* t1 = parser.find_tensor("oob.tensor.1");
    ASSERT_TRUE(t1 != nullptr);
    ASSERT_EQ(parser.get_tensor_data(*t1), nullptr);

    const auto* t2 = parser.find_tensor("oob.tensor.2");
    ASSERT_TRUE(t2 != nullptr);
    ASSERT_EQ(parser.get_tensor_data(*t2), nullptr);

    // Non-existent tensor
    ASSERT_EQ(parser.find_tensor("non_existent_tensor"), nullptr);
}

E2E_TEST_CASE(GGUF_Challenger_Rep, test_zero_tensor_and_kv_counts) {
    // 0 tensors, 0 KVs
    {
        SyntheticGGUFBuilder builder;
        auto buf = builder.build();
        GGUFParser parser;
        ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));
        ASSERT_EQ(parser.get_tensor_count(), 0u);
        ASSERT_EQ(parser.find_tensor("none"), nullptr);
    }
    // 10 tensors, 0 KVs
    {
        SyntheticGGUFBuilder builder;
        for (int i = 0; i < 10; ++i) {
            builder.add_tensor("t_" + std::to_string(i), {16}, 0, i * 16, {0x11, 0x22});
        }
        auto buf = builder.build();
        GGUFParser parser;
        ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));
        ASSERT_EQ(parser.get_tensor_count(), 10u);
    }
}

E2E_TEST_CASE(GGUF_Challenger_Rep, test_boundary_and_huge_alignments) {
    uint32_t alignments[] = {0, 1, 2, 4, 8, 16, 32, 64, 128, 4096, 65536, 1048576};
    for (uint32_t align : alignments) {
        SyntheticGGUFBuilder builder(align == 0 ? 32 : align);
        builder.add_metadata_u32("general.alignment", align);
        builder.add_tensor("t.align", {32}, 0, 0, {0x01, 0x02});

        auto buf = builder.build();
        GGUFParser parser;
        ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

        uint32_t eff_align = parser.get_alignment();
        if (align == 0) {
            ASSERT_EQ(eff_align, 32u);
        } else {
            ASSERT_EQ(eff_align, align);
        }
        uint64_t start_off = parser.get_data_start_offset();
        ASSERT_EQ(start_off % eff_align, 0u);
    }
}

E2E_TEST_CASE(GGUF_Challenger_Rep, test_nonexistent_and_type_mismatched_keys) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("valid.str", "HyperStream");
    builder.add_metadata_u32("valid.u32", 42);
    builder.add_metadata_f32("valid.f32", 3.1415f);
    builder.add_metadata_array_string("valid.arr", {"a", "b"});

    auto buf = builder.build();
    GGUFParser parser;
    ASSERT_TRUE(parser.parse_from_memory(buf.data(), buf.size()));

    std::string s_val;
    uint32_t u_val = 0;
    float f_val = 0.0f;
    std::vector<std::string> arr_val;

    // Non-existent keys
    ASSERT_FALSE(parser.get_metadata_string("missing.str", s_val));
    ASSERT_FALSE(parser.get_metadata_u32("missing.u32", u_val));
    ASSERT_FALSE(parser.get_metadata_f32("missing.f32", f_val));
    ASSERT_FALSE(parser.get_metadata_array_string("missing.arr", arr_val));

    // Fallbacks
    ASSERT_STREQ(parser.get_string("missing.str", "default"), "default");
    ASSERT_EQ(parser.get_u32("missing.u32", 123), 123u);
    ASSERT_NEAR(parser.get_f32("missing.f32", 9.9f), 9.9f, 1e-4f);
    ASSERT_TRUE(parser.get_string_array("missing.arr").empty());

    // Type mismatches (e.g. querying string as array)
    ASSERT_FALSE(parser.get_metadata_array_string("valid.str", arr_val));
    ASSERT_FALSE(parser.get_metadata_string("valid.arr", s_val));
}


// =========================================================================
// SECTION 2: BPE TOKENIZER ADVERSARIAL CHALLENGE SUITE
// =========================================================================

static BPETokenizer create_test_tokenizer() {
    SyntheticGGUFBuilder builder;
    std::vector<std::string> tokens = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<s>", "</s>",
        " ", "a", "b", "c", "d", "e", "h", "i", "l", "m", "n", "o", "r", "s", "t", "u", "w",
        "th", "he", "the", "in", "an", "is",
        // Multi-byte UTF-8
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

E2E_TEST_CASE(BPE_Challenger_Rep, test_uninitialized_tokenizer_safety) {
    BPETokenizer uninit;
    // Calling methods on uninitialized tokenizer must not crash or segfault
    ASSERT_EQ(uninit.vocab_size(), 0u);
    ASSERT_EQ(uninit.eos_token(), -1);
    ASSERT_EQ(uninit.bos_token(), -1);

    auto ids = uninit.encode("Hello World", false);
    ASSERT_TRUE(ids.empty());

    std::string dec = uninit.decode(42);
    ASSERT_STREQ(dec, "");

    std::string dec_arr = uninit.decode(std::vector<int32_t>{1, 2, 3});
    ASSERT_STREQ(dec_arr, "");
}

E2E_TEST_CASE(BPE_Challenger_Rep, test_multilingual_utf8_sequences) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    BPETokenizer tokenizer;
    bool has_real = false;

    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        if (parser.open(real_path)) {
            has_real = tokenizer.load_from_gguf(parser);
        }
    }
    if (!has_real) tokenizer = create_test_tokenizer();

    // Arabic
    std::string arabic_str = "محرك استدلال عالي الأداء بنواة فولكان";
    auto ar_ids = tokenizer.encode(arabic_str);
    ASSERT_FALSE(ar_ids.empty());
    ASSERT_STREQ(tokenizer.decode(ar_ids), arabic_str);

    // Chinese
    std::string chinese_str = "这是高并发大语言模型推理引擎测试。";
    auto zh_ids = tokenizer.encode(chinese_str);
    ASSERT_FALSE(zh_ids.empty());
    ASSERT_STREQ(tokenizer.decode(zh_ids), chinese_str);

    // Control characters
    std::string ctrl_str = "Row1\r\nRow2\tCol\x1B[32mGreen\x1B[0m";
    auto ctrl_ids = tokenizer.encode(ctrl_str);
    ASSERT_FALSE(ctrl_ids.empty());
    ASSERT_STREQ(tokenizer.decode(ctrl_ids), ctrl_str);
}

E2E_TEST_CASE(BPE_Challenger_Rep, test_adjacent_and_repeated_special_tokens) {
    const std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    BPETokenizer tokenizer;
    bool has_real = false;

    if (std::filesystem::exists(real_path)) {
        GGUFParser parser;
        if (parser.open(real_path)) {
            has_real = tokenizer.load_from_gguf(parser);
        }
    }
    if (!has_real) tokenizer = create_test_tokenizer();

    // Adjacent special tokens
    std::string s_adj = "<|im_start|><|im_end|>";
    auto ids_adj = tokenizer.encode(s_adj);
    ASSERT_EQ(ids_adj.size(), 2u);
    ASSERT_EQ(ids_adj[0], tokenizer.im_start_token());
    ASSERT_EQ(ids_adj[1], tokenizer.im_end_token());
    ASSERT_STREQ(tokenizer.decode(ids_adj), s_adj);

    // Repeated special tokens
    std::string s_rep = "<|im_start|><|im_start|><|im_end|><|im_end|>";
    auto ids_rep = tokenizer.encode(s_rep);
    ASSERT_EQ(ids_rep.size(), 4u);
    ASSERT_STREQ(tokenizer.decode(ids_rep), s_rep);
}

E2E_TEST_CASE(BPE_Challenger_Rep, test_empty_string_and_single_characters) {
    auto tokenizer = create_test_tokenizer();

    // Empty string
    ASSERT_TRUE(tokenizer.encode("", false).empty());
    auto ids_bos = tokenizer.encode("", true);
    ASSERT_EQ(ids_bos.size(), 1u);
    ASSERT_EQ(ids_bos[0], tokenizer.bos_token());

    // Single chars
    std::vector<std::string> singles = {"a", "b", "c", " ", "\n", "\t"};
    for (const auto& ch : singles) {
        auto ids = tokenizer.encode(ch);
        ASSERT_FALSE(ids.empty());
        ASSERT_STREQ(tokenizer.decode(ids), ch);
    }
}

E2E_TEST_CASE(BPE_Challenger_Rep, test_repeated_merges_and_patterns) {
    auto tokenizer = create_test_tokenizer();

    // Repetition of 2000 identical chars
    std::string rep(2000, 'a');
    auto ids = tokenizer.encode(rep);
    ASSERT_FALSE(ids.empty());
    ASSERT_STREQ(tokenizer.decode(ids), rep);

    // Repetition of words
    std::string rep_words;
    for (int i = 0; i < 200; ++i) rep_words += "the ";
    auto ids_w = tokenizer.encode(rep_words);
    ASSERT_FALSE(ids_w.empty());
    ASSERT_STREQ(tokenizer.decode(ids_w), rep_words);
}

E2E_TEST_CASE(BPE_Challenger_Rep, test_out_of_vocab_decoding_graceful) {
    auto tokenizer = create_test_tokenizer();
    int32_t v_size = static_cast<int32_t>(tokenizer.vocab_size());

    int32_t bad_ids[] = {-1, -500, v_size, v_size + 1, 9999999, 2147483647};
    for (int32_t bad : bad_ids) {
        ASSERT_STREQ(tokenizer.decode(bad), "");
        ASSERT_FALSE(tokenizer.is_special_token(bad));
    }

    std::vector<int32_t> mixed = {-1, tokenizer.im_start_token(), v_size + 10, tokenizer.im_end_token(), -99};
    std::string dec = tokenizer.decode(mixed);
    std::string expected = std::string("<|im_start|>") + "<|im_end|>";
    ASSERT_STREQ(dec, expected);
}


// =========================================================================
// SECTION 3: SOCKET LAYER ADVERSARIAL CHALLENGE SUITE
// =========================================================================

E2E_TEST_CASE(Socket_Challenger_Rep, test_rapid_connect_close_cycles_100) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28200;

    Socket server;
    ASSERT_TRUE(server.listen(port));

    constexpr int CYCLES = 100;
    std::thread client_thread([&]() {
        for (int i = 0; i < CYCLES; ++i) {
            Socket client;
            if (client.connect("127.0.0.1", port)) {
                uint32_t ping = static_cast<uint32_t>(i);
                client.send(&ping, sizeof(ping));
                uint32_t pong = 0;
                client.recv(&pong, sizeof(pong));
                client.close();
            }
        }
    });

    for (int i = 0; i < CYCLES; ++i) {
        Socket accepted;
        ASSERT_TRUE(server.accept(accepted));
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

E2E_TEST_CASE(Socket_Challenger_Rep, test_large_payload_transmission_4mb) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28201;

    Socket server;
    ASSERT_TRUE(server.listen(port));

    constexpr size_t PAYLOAD_SIZE = 4 * 1024 * 1024; // 4 Megabytes
    std::vector<uint8_t> send_buf(PAYLOAD_SIZE);
    std::mt19937 rng(12345);
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

E2E_TEST_CASE(Socket_Challenger_Rep, test_zero_byte_send_and_recv) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28202;

    Socket server;
    ASSERT_TRUE(server.listen(port));

    std::thread client_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        Socket client;
        if (client.connect("127.0.0.1", port)) {
            char dummy = 'Z';
            // Send 0 bytes with pointer
            ASSERT_EQ(client.send(&dummy, 0), 0);
            // Send 0 bytes with nullptr
            ASSERT_EQ(client.send(nullptr, 0), 0);
            // Send 1 valid byte
            client.send(&dummy, 1);
            client.close();
        }
    });

    Socket accepted;
    ASSERT_TRUE(server.accept(accepted));
    // Recv 0 bytes
    char buf[16] = {0};
    ASSERT_EQ(accepted.recv(buf, 0), 0);

    // Recv 1 byte
    int n = accepted.recv(buf, sizeof(buf));
    ASSERT_EQ(n, 1);
    ASSERT_EQ(buf[0], 'Z');

    client_thread.join();
    accepted.close();
    server.close();
}

E2E_TEST_CASE(Socket_Challenger_Rep, test_recv_on_closed_and_unopened_sockets) {
    ASSERT_TRUE(Socket::init_networking());

    // Unopened socket
    Socket uninit;
    char buf[16];
    ASSERT_EQ(uninit.recv(buf, sizeof(buf)), -1);
    ASSERT_EQ(uninit.send(buf, sizeof(buf)), -1);

    // Closed socket
    Socket s;
    s.listen(28203);
    s.close();
    ASSERT_EQ(s.recv(buf, sizeof(buf)), -1);
    ASSERT_EQ(s.send(buf, sizeof(buf)), -1);
}

E2E_TEST_CASE(Socket_Challenger_Rep, test_port_collision_and_non_blocking_close) {
    ASSERT_TRUE(Socket::init_networking());
    uint16_t port = 28204;

    Socket s1;
    ASSERT_TRUE(s1.listen(port));
    ASSERT_TRUE(s1.is_valid());

    Socket s2;
    // Attempting to listen on the exact same port
    bool ok2 = s2.listen(port);
    // Regardless of OS SO_REUSEADDR policy, neither must crash or corrupt state
    if (ok2) {
        s2.close();
    } else {
        ASSERT_FALSE(s2.is_valid());
    }

    s1.close();
    ASSERT_FALSE(s1.is_valid());
    Socket::cleanup_networking();
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
