#include "e2e/common/e2e_test_framework.hpp"
#include "e2e/common/gguf_test_utils.hpp"
#include "util/mmap_file.hpp"
#include "gguf_parser.hpp"
#include <filesystem>
#include <vector>
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#include <psapi.h>
#endif

using namespace hyperstream;
using namespace hyperstream::test;

static uint32_t get_current_handle_count() {
#if defined(_WIN32) || defined(_WIN64)
    DWORD handle_count = 0;
    if (GetProcessHandleCount(GetCurrentProcess(), &handle_count)) {
        return static_cast<uint32_t>(handle_count);
    }
#endif
    return 0;
}

static size_t get_current_memory_working_set() {
#if defined(_WIN32) || defined(_WIN64)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
#endif
    return 0;
}

// 1. MmapFile Handle Churn (5,000 rapid open/close cycles)
E2E_TEST_CASE(MemorySafetyTest, test_mmap_file_handle_churn) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("test.key", "handle_churn_verification");
    std::string temp_path = "bin/temp_churn_mmap.bin";
    ASSERT_TRUE(builder.write_to_file(temp_path));

    uint32_t h_initial = get_current_handle_count();

    for (int i = 0; i < 5000; ++i) {
        MmapFile mfile;
        bool ok = mfile.open(temp_path);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(mfile.is_open());
        ASSERT_TRUE(mfile.data() != nullptr);
        ASSERT_TRUE(mfile.size() > 0);
        mfile.close();
        ASSERT_FALSE(mfile.is_open());
    }

    uint32_t h_final = get_current_handle_count();
    std::filesystem::remove(temp_path);

    // Verify zero handle leak
    ASSERT_EQ(h_final, h_initial);
}

// 2. MmapFile Move Semantics Churn & Self-Move Robustness
E2E_TEST_CASE(MemorySafetyTest, test_mmap_file_move_semantics_and_self_move) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("move.key", "move_churn_test");
    std::string temp_path = "bin/temp_move_mmap.bin";
    ASSERT_TRUE(builder.write_to_file(temp_path));

    uint32_t h_initial = get_current_handle_count();

    for (int i = 0; i < 3000; ++i) {
        MmapFile f1;
        ASSERT_TRUE(f1.open(temp_path));
        const uint8_t* ptr1 = f1.data();
        size_t sz1 = f1.size();

        // Move constructor
        MmapFile f2(std::move(f1));
        ASSERT_FALSE(f1.is_open());
        ASSERT_TRUE(f1.data() == nullptr);
        ASSERT_EQ(f1.size(), 0u);
        ASSERT_TRUE(f2.is_open());
        ASSERT_EQ(f2.data(), ptr1);
        ASSERT_EQ(f2.size(), sz1);

        // Move assignment
        MmapFile f3;
        f3 = std::move(f2);
        ASSERT_FALSE(f2.is_open());
        ASSERT_TRUE(f3.is_open());
        ASSERT_EQ(f3.data(), ptr1);

        // Self-move assignment
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wself-move"
        f3 = std::move(f3);
        #pragma GCC diagnostic pop
        ASSERT_TRUE(f3.is_open());
        ASSERT_EQ(f3.data(), ptr1);
        ASSERT_EQ(f3.size(), sz1);

        f3.close();
    }

    uint32_t h_final = get_current_handle_count();
    std::filesystem::remove(temp_path);

    ASSERT_EQ(h_final, h_initial);
}

// 3. MmapFile Re-open & Error Paths Zero Leaks
E2E_TEST_CASE(MemorySafetyTest, test_mmap_reopen_and_error_paths) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("key", "val");
    std::string temp1 = "bin/temp_reopen1.bin";
    std::string temp2 = "bin/temp_reopen2.bin";
    std::string empty_file = "bin/temp_empty.bin";
    ASSERT_TRUE(builder.write_to_file(temp1));
    ASSERT_TRUE(builder.write_to_file(temp2));

    // Create a 0-byte file
    {
        std::ofstream ofs(empty_file, std::ios::binary);
    }

    uint32_t h_initial = get_current_handle_count();

    // Reopen without close 2000 times
    MmapFile f;
    for (int i = 0; i < 2000; ++i) {
        ASSERT_TRUE(f.open(i % 2 == 0 ? temp1 : temp2));
        ASSERT_TRUE(f.is_open());
    }
    f.close();

    // Open non-existent and empty files repeatedly
    for (int i = 0; i < 1000; ++i) {
        MmapFile f_fail;
        ASSERT_FALSE(f_fail.open("non_existent_path_xyz.bin"));
        ASSERT_FALSE(f_fail.is_open());

        ASSERT_FALSE(f_fail.open(empty_file));
        ASSERT_FALSE(f_fail.is_open());
    }

    uint32_t h_final = get_current_handle_count();
    std::filesystem::remove(temp1);
    std::filesystem::remove(temp2);
    std::filesystem::remove(empty_file);

    ASSERT_EQ(h_final, h_initial);
}

// 4. GGUFParser Memory & Handle Churn (1,000 parses)
E2E_TEST_CASE(MemorySafetyTest, test_gguf_parser_handle_and_memory_churn) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.architecture", "qwen2");
    builder.add_metadata_u32("qwen2.block_count", 24);
    builder.add_metadata_array_string("tokenizer.ggml.tokens", {"<|endoftext|>", "hello", "world"});
    builder.add_tensor("token_embd.weight", {128, 3}, 8, 0, {0x01, 0x02, 0x03, 0x04});

    std::string temp_path = "bin/temp_parser_churn.gguf";
    ASSERT_TRUE(builder.write_to_file(temp_path));

    uint32_t h_initial = get_current_handle_count();

    for (int i = 0; i < 1000; ++i) {
        GGUFParser parser;
        bool ok = parser.open(temp_path);
        ASSERT_TRUE(ok);
        ASSERT_EQ(parser.get_version(), 3u);
        ASSERT_STREQ(parser.get_string("general.architecture"), "qwen2");

        const auto* t = parser.find_tensor("token_embd.weight");
        ASSERT_TRUE(t != nullptr);
        const uint8_t* ptr = parser.get_tensor_data(*t);
        ASSERT_TRUE(ptr != nullptr);
        ASSERT_EQ(ptr[0], 0x01);
    }

    uint32_t h_final = get_current_handle_count();
    std::filesystem::remove(temp_path);

    ASSERT_EQ(h_final, h_initial);
}

// 5. GGUFParser Move Semantics & Dangling Handle Verification
E2E_TEST_CASE(MemorySafetyTest, test_gguf_parser_move_semantics_and_lifetime) {
    SyntheticGGUFBuilder builder;
    builder.add_metadata_string("general.name", "QwenModel");
    builder.add_tensor("test_tensor", {16, 16}, 0, 0, {0x11, 0x22, 0x33, 0x44});
    std::string temp_path = "bin/temp_parser_move.gguf";
    ASSERT_TRUE(builder.write_to_file(temp_path));

    uint32_t h_initial = get_current_handle_count();

    {
        GGUFParser p3;
        {
            GGUFParser p1;
            ASSERT_TRUE(p1.open(temp_path));
            const auto* t1 = p1.find_tensor("test_tensor");
            ASSERT_TRUE(t1 != nullptr);
            const uint8_t* data_orig = p1.get_tensor_data(*t1);
            ASSERT_TRUE(data_orig != nullptr);

            // Move construct to p2
            GGUFParser p2(std::move(p1));
            // p1 is moved from; its destruction should be harmless

            // Move assign to p3
            p3 = std::move(p2);
            // p2 is moved from; its destruction should be harmless
        }
        // At this point p1 and p2 have gone out of scope and were destroyed.
        // p3 must still hold valid mappings and data!
        std::string name;
        ASSERT_TRUE(p3.get_metadata_string("general.name", name));
        ASSERT_STREQ(name, "QwenModel");

        const auto* t3 = p3.find_tensor("test_tensor");
        ASSERT_TRUE(t3 != nullptr);
        const uint8_t* data_p3 = p3.get_tensor_data(*t3);
        ASSERT_TRUE(data_p3 != nullptr);
        ASSERT_EQ(data_p3[0], 0x11);
        ASSERT_EQ(data_p3[1], 0x22);
    }
    // Now p3 is destroyed.

    uint32_t h_final = get_current_handle_count();
    std::filesystem::remove(temp_path);

    ASSERT_EQ(h_final, h_initial);
}

// 6. Real Model GGUF Churn (Open/Close 50 times on 644MB file)
E2E_TEST_CASE(MemorySafetyTest, test_real_model_open_close_churn) {
    std::string real_path = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    if (std::filesystem::exists(real_path)) {
        uint32_t h_initial = get_current_handle_count();
        size_t mem_initial = get_current_memory_working_set();

        for (int i = 0; i < 50; ++i) {
            GGUFParser parser;
            ASSERT_TRUE(parser.open(real_path));
            ASSERT_EQ(parser.get_version(), 3u);
            ASSERT_TRUE(parser.size() > 600000000ULL);
            const auto* embd = parser.find_tensor("token_embd.weight");
            ASSERT_TRUE(embd != nullptr);
            const uint8_t* data = parser.get_tensor_data(*embd);
            ASSERT_TRUE(data != nullptr);
            // Read 1 byte to ensure mapped page access
            volatile uint8_t b = data[0];
            (void)b;
            parser.close();
        }

        uint32_t h_final = get_current_handle_count();
        size_t mem_final = get_current_memory_working_set();

        // Handle count must be identical
        ASSERT_EQ(h_final, h_initial);

        // Working set memory increase should be less than 50MB after unmapping
        size_t mem_diff = (mem_final > mem_initial) ? (mem_final - mem_initial) : 0;
        ASSERT_TRUE(mem_diff < 50 * 1024 * 1024);
    }
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
