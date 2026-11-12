#ifndef HYPERSTREAM_MODEL_TOPOLOGY_HPP
#define HYPERSTREAM_MODEL_TOPOLOGY_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <vulkan/vulkan.h>
#include "gguf_parser.hpp"

namespace hyperstream {

struct VulkanSpecializationData {
    uint32_t HIDDEN_DIM;
    uint32_t HEAD_DIM;
    uint32_t NUM_HEADS;
    uint32_t NUM_KV_HEADS;
    uint32_t INTERMEDIATE_DIM;
    uint32_t MAX_SEQ_LEN;
    uint32_t VOCAB_SIZE;
};

struct SpecializationInfoHelper {
    VulkanSpecializationData data{};
    std::array<VkSpecializationMapEntry, 7> entries{};
    VkSpecializationInfo info{};

    SpecializationInfoHelper() {
        setup_entries();
    }

    explicit SpecializationInfoHelper(const VulkanSpecializationData& spec_data) {
        init(spec_data);
    }

    void init(const VulkanSpecializationData& spec_data) {
        data = spec_data;
        setup_entries();
    }

    void setup_entries() {
        entries[0] = { 0, static_cast<uint32_t>(offsetof(VulkanSpecializationData, HIDDEN_DIM)), sizeof(uint32_t) };
        entries[1] = { 1, static_cast<uint32_t>(offsetof(VulkanSpecializationData, HEAD_DIM)), sizeof(uint32_t) };
        entries[2] = { 2, static_cast<uint32_t>(offsetof(VulkanSpecializationData, NUM_HEADS)), sizeof(uint32_t) };
        entries[3] = { 3, static_cast<uint32_t>(offsetof(VulkanSpecializationData, NUM_KV_HEADS)), sizeof(uint32_t) };
        entries[4] = { 4, static_cast<uint32_t>(offsetof(VulkanSpecializationData, INTERMEDIATE_DIM)), sizeof(uint32_t) };
        entries[5] = { 5, static_cast<uint32_t>(offsetof(VulkanSpecializationData, MAX_SEQ_LEN)), sizeof(uint32_t) };
        entries[6] = { 6, static_cast<uint32_t>(offsetof(VulkanSpecializationData, VOCAB_SIZE)), sizeof(uint32_t) };

        info.mapEntryCount = static_cast<uint32_t>(entries.size());
        info.pMapEntries = entries.data();
        info.dataSize = sizeof(VulkanSpecializationData);
        info.pData = &data;
    }

    const VkSpecializationInfo* get_info() const { return &info; }
    const VkSpecializationInfo* get() const { return &info; }
    operator const VkSpecializationInfo*() const { return &info; }
};

struct ModelTopology {
    std::string architecture = "qwen2";
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t num_layers = 0;
    uint32_t num_heads = 0;
    uint32_t num_kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t vocab_size = 0;
    float rms_norm_eps = 1e-6f;
    float rope_freq_base = 1000000.0f;
    uint64_t data_start_offset = 0;

    struct LayerOffsets {
        uint64_t attn_norm = 0;
        uint64_t qkv_weight = 0;
        uint64_t o_weight = 0;
        uint64_t ffn_norm = 0;
        uint64_t gate_up_weight = 0;
        uint64_t down_weight = 0;

        // Individual tensor offsets (used when QKV or Gate/Up are unfused in GGUF)
        uint64_t q_weight = 0;
        uint64_t k_weight = 0;
        uint64_t v_weight = 0;
        uint64_t gate_weight = 0;
        uint64_t up_weight = 0;

        // Bias offsets
        uint64_t q_bias = 0;
        uint64_t k_bias = 0;
        uint64_t v_bias = 0;
    };

    std::vector<LayerOffsets> layers;
    uint64_t token_embd_offset = 0;
    uint64_t output_norm_offset = 0;
    uint64_t output_weight_offset = 0;

    VulkanSpecializationData get_specialization_data(uint32_t max_seq_len = 8192) const {
        VulkanSpecializationData spec{};
        spec.HIDDEN_DIM = hidden_size;
        spec.HEAD_DIM = head_dim;
        spec.NUM_HEADS = num_heads;
        spec.NUM_KV_HEADS = num_kv_heads;
        spec.INTERMEDIATE_DIM = intermediate_size;
        spec.MAX_SEQ_LEN = max_seq_len;
        spec.VOCAB_SIZE = vocab_size;
        return spec;
    }

    SpecializationInfoHelper create_specialization_info(uint32_t max_seq_len = 8192) const {
        return SpecializationInfoHelper(get_specialization_data(max_seq_len));
    }

    static void create_specialization_info(const VulkanSpecializationData& spec_data,
                                          VkSpecializationInfo& out_info,
                                          std::array<VkSpecializationMapEntry, 7>& out_entries) {
        out_entries[0] = { 0, static_cast<uint32_t>(offsetof(VulkanSpecializationData, HIDDEN_DIM)), sizeof(uint32_t) };
        out_entries[1] = { 1, static_cast<uint32_t>(offsetof(VulkanSpecializationData, HEAD_DIM)), sizeof(uint32_t) };
        out_entries[2] = { 2, static_cast<uint32_t>(offsetof(VulkanSpecializationData, NUM_HEADS)), sizeof(uint32_t) };
        out_entries[3] = { 3, static_cast<uint32_t>(offsetof(VulkanSpecializationData, NUM_KV_HEADS)), sizeof(uint32_t) };
        out_entries[4] = { 4, static_cast<uint32_t>(offsetof(VulkanSpecializationData, INTERMEDIATE_DIM)), sizeof(uint32_t) };
        out_entries[5] = { 5, static_cast<uint32_t>(offsetof(VulkanSpecializationData, MAX_SEQ_LEN)), sizeof(uint32_t) };
        out_entries[6] = { 6, static_cast<uint32_t>(offsetof(VulkanSpecializationData, VOCAB_SIZE)), sizeof(uint32_t) };

        out_info.mapEntryCount = 7;
        out_info.pMapEntries = out_entries.data();
        out_info.dataSize = sizeof(VulkanSpecializationData);
        out_info.pData = &spec_data;
    }

    static bool build_from_gguf(const GGUFParser& parser, ModelTopology& topo) {
        // 1. Architecture
        if (!parser.get_metadata_string("general.architecture", topo.architecture)) {
            topo.architecture = "qwen2"; // default architecture fallback
        }

        std::string pfx = topo.architecture + ".";

        // 2. Geometry hyperparameters
        if (!parser.get_metadata_u32(pfx + "block_count", topo.num_layers) || topo.num_layers == 0) {
            return false;
        }
        if (!parser.get_metadata_u32(pfx + "embedding_length", topo.hidden_size) || topo.hidden_size == 0) {
            return false;
        }
        if (!parser.get_metadata_u32(pfx + "feed_forward_length", topo.intermediate_size)) {
            return false;
        }
        if (!parser.get_metadata_u32(pfx + "attention.head_count", topo.num_heads) || topo.num_heads == 0) {
            return false;
        }
        if (!parser.get_metadata_u32(pfx + "attention.head_count_kv", topo.num_kv_heads)) {
            topo.num_kv_heads = topo.num_heads; // Default to MHA if GQA not specified
        }
        if (topo.num_kv_heads == 0) {
            topo.num_kv_heads = topo.num_heads;
        }

        if (topo.hidden_size % topo.num_heads != 0) {
            return false; // Non-divisible geometry rejection
        }
        topo.head_dim = topo.hidden_size / topo.num_heads;
        if (topo.head_dim == 0) {
            return false;
        }

        // 3. Normalization and RoPE
        if (!parser.get_metadata_f32(pfx + "attention.layer_norm_rms_epsilon", topo.rms_norm_eps)) {
            topo.rms_norm_eps = 1e-6f;
        }
        if (!parser.get_metadata_f32(pfx + "rope.freq_base", topo.rope_freq_base)) {
            topo.rope_freq_base = 1000000.0f;
        }

        topo.data_start_offset = parser.get_data_start_offset();

        // 4. Global tensors
        const auto* embd_tensor = parser.find_tensor("token_embd.weight");
        if (!embd_tensor) {
            return false; // token_embd.weight is mandatory
        }
        topo.token_embd_offset = embd_tensor->offset;
        topo.vocab_size = static_cast<uint32_t>(embd_tensor->ne[1]);

        const auto* out_norm_tensor = parser.find_tensor("output_norm.weight");
        if (out_norm_tensor) {
            topo.output_norm_offset = out_norm_tensor->offset;
        }

        const auto* lm_head_tensor = parser.find_tensor("output.weight");
        if (lm_head_tensor) {
            topo.output_weight_offset = lm_head_tensor->offset;
            if (topo.vocab_size == 0) {
                topo.vocab_size = static_cast<uint32_t>(lm_head_tensor->ne[1]);
            }
        } else {
            // Tied embedding architecture fallback
            topo.output_weight_offset = topo.token_embd_offset;
        }

        // 5. Per-layer tensors
        topo.layers.resize(topo.num_layers);
        for (uint32_t l = 0; l < topo.num_layers; ++l) {
            std::string blk = "blk." + std::to_string(l) + ".";
            auto& lo = topo.layers[l];

            auto get_offset = [&](const std::string& name) -> uint64_t {
                const auto* t = parser.find_tensor(blk + name);
                return t ? t->offset : 0ULL;
            };

            lo.attn_norm       = get_offset("attn_norm.weight");
            lo.qkv_weight      = get_offset("attn_qkv.weight");
            lo.q_weight        = get_offset("attn_q.weight");
            lo.k_weight        = get_offset("attn_k.weight");
            lo.v_weight        = get_offset("attn_v.weight");
            lo.q_bias          = get_offset("attn_q.bias");
            lo.k_bias          = get_offset("attn_k.bias");
            lo.v_bias          = get_offset("attn_v.bias");
            lo.o_weight        = get_offset("attn_output.weight");
            lo.ffn_norm        = get_offset("ffn_norm.weight");
            lo.gate_up_weight  = get_offset("ffn_gate_up.weight");
            lo.gate_weight     = get_offset("ffn_gate.weight");
            lo.up_weight       = get_offset("ffn_up.weight");
            lo.down_weight     = get_offset("ffn_down.weight");

            // If fused weights are not present, fallback to individual Q/Gate offsets
            if (lo.qkv_weight == 0) {
                lo.qkv_weight = lo.q_weight;
            }
            if (lo.q_weight == 0) {
                lo.q_weight = lo.qkv_weight;
            }
            if (lo.gate_up_weight == 0) {
                lo.gate_up_weight = lo.gate_weight;
            }
            if (lo.gate_weight == 0) {
                lo.gate_weight = lo.gate_up_weight;
            }
        }

        return (topo.num_layers > 0 && topo.hidden_size > 0 && topo.num_heads > 0);
    }
};

} // namespace hyperstream

#endif // HYPERSTREAM_MODEL_TOPOLOGY_HPP
