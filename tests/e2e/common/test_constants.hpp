#pragma once

#include <string>
#include <cstdint>

namespace hyperstream::test {

struct TestConstants {
    // Standard Qwen2 Token IDs
    static constexpr int32_t QWEN2_IM_START_TOKEN = 151644;
    static constexpr int32_t QWEN2_IM_END_TOKEN   = 151645;
    static constexpr int32_t QWEN2_ENDOFTEXT_TOKEN = 151643;
    static constexpr int32_t QWEN2_NEWLINE_TOKEN   = 198;

    // Tolerance
    static constexpr float MATH_EPSILON = 1e-4f;
    static constexpr float ATTN_EPSILON = 1e-4f;

    // Model Paths
    inline static const std::string DRAFT_MODEL_PATH = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    inline static const std::string TARGET_MODEL_PATH = "H:\\ai model\\qwen2.5-coder-7b-instruct-q4_0.gguf";

    // Standard Qwen2.5-Coder-0.5B Dimensions
    static constexpr uint32_t DRAFT_HIDDEN_DIM = 896;
    static constexpr uint32_t DRAFT_INTERMEDIATE_DIM = 4864;
    static constexpr uint32_t DRAFT_NUM_LAYERS = 24;
    static constexpr uint32_t DRAFT_NUM_HEADS = 14;
    static constexpr uint32_t DRAFT_NUM_KV_HEADS = 2;
    static constexpr uint32_t DRAFT_HEAD_DIM = 64;
    static constexpr uint32_t DRAFT_VOCAB_SIZE = 151936;

    // Standard Qwen2.5-Coder-7B Dimensions
    static constexpr uint32_t TARGET_HIDDEN_DIM = 3584;
    static constexpr uint32_t TARGET_INTERMEDIATE_DIM = 18944;
    static constexpr uint32_t TARGET_NUM_LAYERS = 28;
    static constexpr uint32_t TARGET_NUM_HEADS = 28;
    static constexpr uint32_t TARGET_NUM_KV_HEADS = 4;
    static constexpr uint32_t TARGET_HEAD_DIM = 128;
    static constexpr uint32_t TARGET_VOCAB_SIZE = 152064;

    // Standard Speculative Tree Size
    static constexpr int SPEC_TREE_SIZE = 7;

    // Dispatch counts
    static constexpr uint32_t TARGET_DISPATCH_COUNT_PER_LAYER = 3;
    static constexpr uint32_t TARGET_TOTAL_DISPATCHES_28_LAYERS = 84;
    static constexpr uint32_t LEGACY_DISPATCH_COUNT_PER_LAYER = 8;
    static constexpr uint32_t LEGACY_TOTAL_DISPATCHES_28_LAYERS = 224;
};

} // namespace hyperstream::test
