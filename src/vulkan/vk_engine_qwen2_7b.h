#ifndef VK_ENGINE_QWEN2_7B_H
#define VK_ENGINE_QWEN2_7B_H

#include <vector>
#include <string>
#include <memory>
#include "vk_context.h"
#include "vk_memory.h"
#include "vk_pipeline.h"
#include "qwen2_7b_layout.h"

// Q8_0 Block definition (for LM Head)
struct BlockQ8Qwen2_7B {
    float scale;
    int8_t qs[32];
};

struct VulkanQwen2_7BLayerWeights {
    VulkanBuffer input_layernorm;     // [3584] floats

    // Fused QKV (Planar Q4_0: Quants + Packed Scales + F32 Biases)
    VulkanBuffer qkv_quants;          // [4608 * 112] uvec4
    VulkanBuffer qkv_scales;          // [(4608 * 112 + 1) / 2] uint
    VulkanBuffer qkv_bias;            // [4608] floats

    // Attn Out (Planar Q4_0: Quants + Packed Scales)
    VulkanBuffer o_proj_quants;       // [3584 * 112] uvec4
    VulkanBuffer o_proj_scales;       // [(3584 * 112 + 1) / 2] uint

    VulkanBuffer post_attn_layernorm; // [3584] floats

    // Fused SwiGLU: Gate & Up (Planar Q4_0)
    VulkanBuffer gate_quants;         // [18944 * 112] uvec4
    VulkanBuffer gate_scales;         // [(18944 * 112 + 1) / 2] uint
    VulkanBuffer up_quants;           // [18944 * 112] uvec4
    VulkanBuffer up_scales;           // [(18944 * 112 + 1) / 2] uint

    // Down Projection (Planar Q4_0)
    VulkanBuffer down_quants;         // [3584 * 592] uvec4
    VulkanBuffer down_scales;         // [(3584 * 592 + 1) / 2] uint

    // Static Descriptor Sets
    VkDescriptorSet ds_input_norm = VK_NULL_HANDLE;
    VkDescriptorSet ds_fused_qkv = VK_NULL_HANDLE;
    VkDescriptorSet ds_rope_cache = VK_NULL_HANDLE;
    VkDescriptorSet ds_attention = VK_NULL_HANDLE;
    VkDescriptorSet ds_o_proj = VK_NULL_HANDLE;
    VkDescriptorSet ds_post_norm = VK_NULL_HANDLE;
    VkDescriptorSet ds_fused_swiglu = VK_NULL_HANDLE;
    VkDescriptorSet ds_down_proj = VK_NULL_HANDLE;
};

class VulkanInferenceEngineQwen2_7B {
public:
    VulkanInferenceEngineQwen2_7B();
    ~VulkanInferenceEngineQwen2_7B();

    bool init();
    bool upload_model_weights(const void *raw_mmap_base, const Qwen2Offsets &offsets);
    void reset_kv_cache();

    // Forward pass on AMD RX 580: returns logits in out_logits [QWEN2_VOCAB_SIZE]
    bool forward_token(int token_id, int pos, float *out_logits);

    // Ultra-fast GPU Argmax forward pass: returns winner token_id directly with 4-byte readback
    int forward_token_gpu_argmax(int token_id, int pos);

    void cleanup();

    const VulkanContext& get_context() const { return m_ctx; }
    bool is_initialized() const { return m_initialized; }

private:
    bool init_pipelines();
    bool allocate_buffers();
    void record_forward_command_buffer();

    VulkanContext m_ctx;
    bool m_initialized = false;

    // Compute Pipelines
    ComputePipeline m_pipe_q4_fused_qkv;
    ComputePipeline m_pipe_q4_gemv;
    ComputePipeline m_pipe_q4_fused_swiglu;
    ComputePipeline m_pipe_q8_gemv; // for LM Head
    ComputePipeline m_pipe_rmsnorm;
    ComputePipeline m_pipe_rope_cache;
    ComputePipeline m_pipe_attention;
    ComputePipeline m_pipe_argmax_p1;
    ComputePipeline m_pipe_argmax_p2;

    // Model Weights in VRAM
    VulkanBuffer m_w_lm_head;      // [152064 * 112] Q8_0 blocks (output.weight dequantized to Q8_0)
    VulkanBuffer m_w_final_norm;   // [3584] floats (output_norm.weight)
    std::vector<VulkanQwen2_7BLayerWeights> m_layers;

    // Raw file pointer for direct token embedding dequantization
    const uint8_t *m_raw_embed_base = nullptr;
    uint64_t m_embed_offset = 0;

    // KV Cache in VRAM (28 layers * 1024 seq * 512 kv_dim * 4 bytes ≈ 58.7 MB each)
    VulkanBuffer m_buf_k_cache;
    VulkanBuffer m_buf_v_cache;

    // Intermediate Activations in VRAM
    VulkanBuffer m_act_x;            // [3584] (Host-visible & mapped)
    VulkanBuffer m_act_norm_x;       // [3584]
    VulkanBuffer m_act_q;            // [3584]
    VulkanBuffer m_act_k;            // [512]
    VulkanBuffer m_act_v;            // [512]
    VulkanBuffer m_act_head_outputs; // [3584] Attention heads output
    VulkanBuffer m_act_attn_out;     // [3584] O-proj output
    VulkanBuffer m_act_post_norm_x;  // [3584]
    VulkanBuffer m_act_mlp_hidden;   // [18944] Fused SwiGLU output
    VulkanBuffer m_act_mlp_out;      // [3584] Down-proj output
    VulkanBuffer m_act_logits;       // [152064] (Host-visible & mapped)
    VulkanBuffer m_buf_pos;          // [1] uint32_t (Host-visible & mapped)

    // GPU Argmax Buffers
    VulkanBuffer m_buf_arg_part_vals;   // [128] floats
    VulkanBuffer m_buf_arg_part_ids;    // [128] uint32_t
    VulkanBuffer m_buf_arg_token_result;// [2] uint32_t (Host-visible & mapped)

    // Pre-allocated static Descriptor Sets
    VkDescriptorSet m_ds_final_norm = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_lm_head = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_arg_p1 = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_arg_p2 = VK_NULL_HANDLE;

    // Command execution
    VkCommandBuffer m_cmd_buffer = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd_buffer_argmax = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
};

#endif // VK_ENGINE_QWEN2_7B_H
