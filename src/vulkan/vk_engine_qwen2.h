#ifndef VK_ENGINE_QWEN2_H
#define VK_ENGINE_QWEN2_H

#include <vector>
#include <string>
#include <memory>
#include "vk_context.h"
#include "vk_memory.h"
#include "vk_pipeline.h"
#include "model_topology.hpp"

// Q8_0 Block definition (4-byte aligned for Vulkan std430)
struct BlockQ8Qwen2 {
    float scale;
    int8_t qs[32];
};

struct VulkanQwen2LayerWeights {
    VulkanBuffer input_layernorm;     // [896] floats
    VulkanBuffer q_proj;              // [896, 28] blocks
    VulkanBuffer q_bias;              // [896] floats
    VulkanBuffer k_proj;              // [128, 28] blocks
    VulkanBuffer k_bias;              // [128] floats
    VulkanBuffer v_proj;              // [128, 28] blocks
    VulkanBuffer v_bias;              // [128] floats
    VulkanBuffer o_proj;              // [896, 28] blocks
    VulkanBuffer post_attn_layernorm; // [896] floats
    VulkanBuffer gate_proj;           // [4864, 28] blocks
    VulkanBuffer up_proj;             // [4864, 28] blocks
    VulkanBuffer down_proj;           // [896, 152] blocks

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

class VulkanInferenceEngineQwen2 {
public:
    VulkanInferenceEngineQwen2();
    ~VulkanInferenceEngineQwen2();

    bool init(const hyperstream::ModelTopology &topo = {});
    bool upload_model_weights(const void *raw_mmap_base, const hyperstream::ModelTopology &topo);
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
    hyperstream::ModelTopology m_topo;

    // Compute Pipelines
    ComputePipeline m_pipe_q8_gemv;
    ComputePipeline m_pipe_rmsnorm;
    ComputePipeline m_pipe_rope_cache;
    ComputePipeline m_pipe_attention;
    ComputePipeline m_pipe_fused_qkv;
    ComputePipeline m_pipe_fused_swiglu;
    ComputePipeline m_pipe_residual_add;
    ComputePipeline m_pipe_argmax_p1;
    ComputePipeline m_pipe_argmax_p2;

    // Model Weights in VRAM
    VulkanBuffer m_w_embed_tokens; // [151936, 28] blocks (token_embd.weight)
    VulkanBuffer m_w_lm_head;      // [151936, 28] blocks (output.weight)
    VulkanBuffer m_w_final_norm;   // [896] floats (output_norm.weight)
    std::vector<VulkanQwen2LayerWeights> m_layers;

    // Base pointer for direct token embedding dequantization
    const uint8_t *m_raw_embed_base = nullptr;

    // KV Cache in VRAM
    VulkanBuffer m_buf_k_cache; // [24 layers * 1024 seq * 128 kv_dim] floats (~12.5 MB)
    VulkanBuffer m_buf_v_cache; // [24 layers * 1024 seq * 128 kv_dim] floats (~12.5 MB)

    // Intermediate Activations
    VulkanBuffer m_act_x;            // [896] (Host-visible & mapped)
    VulkanBuffer m_act_norm_x;       // [896]
    VulkanBuffer m_act_q;            // [896]
    VulkanBuffer m_act_k;            // [128]
    VulkanBuffer m_act_v;            // [128]
    VulkanBuffer m_act_head_outputs; // [896] Attention heads output
    VulkanBuffer m_act_attn_out;     // [896] O-proj output
    VulkanBuffer m_act_post_norm_x;  // [896]
    VulkanBuffer m_act_mlp_hidden;   // [4864] Fused SwiGLU output
    VulkanBuffer m_act_mlp_out;      // [896] Down-proj output
    VulkanBuffer m_act_logits;       // [151936] (Host-visible & mapped)
    VulkanBuffer m_buf_pos;          // [1] uint32_t (Host-visible & mapped)

    // GPU Argmax Buffers
    VulkanBuffer m_buf_arg_part_vals;   // [128] floats
    VulkanBuffer m_buf_arg_part_ids;    // [128] uint32_t
    VulkanBuffer m_buf_arg_token_result;// [2] uint32_t (Host-visible & mapped)

    // Pre-allocated static Descriptor Sets
    VkDescriptorSet m_ds_final_norm = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_lm_head = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_res_attn = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_res_mlp = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_arg_p1 = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_arg_p2 = VK_NULL_HANDLE;

    // Command execution
    VkCommandBuffer m_cmd_buffer = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd_buffer_argmax = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
};

#endif // VK_ENGINE_QWEN2_H
