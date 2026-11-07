#ifndef VK_ENGINE_H
#define VK_ENGINE_H

#include <vector>
#include <string>
#include <memory>
#include "vk_context.h"
#include "vk_memory.h"
#include "vk_pipeline.h"
#include "smol_layout.h"

// Q8_0 Block definition
struct BlockQ8 {
    float scale;
    int8_t qs[32];
};

struct VulkanLayerWeights {
    VulkanBuffer input_layernorm;     // [960] floats
    VulkanBuffer q_proj;              // [960, 30] blocks
    VulkanBuffer k_proj;              // [320, 30] blocks
    VulkanBuffer v_proj;              // [320, 30] blocks
    VulkanBuffer o_proj;              // [960, 30] blocks
    VulkanBuffer post_attn_layernorm; // [960] floats
    VulkanBuffer gate_proj;           // [2560, 30] blocks
    VulkanBuffer up_proj;             // [2560, 30] blocks
    VulkanBuffer down_proj;           // [960, 80] blocks

    // Pre-allocated Descriptor Sets for GEMVs
    VkDescriptorSet ds_q_proj = VK_NULL_HANDLE;
    VkDescriptorSet ds_k_proj = VK_NULL_HANDLE;
    VkDescriptorSet ds_v_proj = VK_NULL_HANDLE;
    VkDescriptorSet ds_o_proj = VK_NULL_HANDLE;
    VkDescriptorSet ds_gate_proj = VK_NULL_HANDLE;
    VkDescriptorSet ds_up_proj = VK_NULL_HANDLE;
    VkDescriptorSet ds_down_proj = VK_NULL_HANDLE;

    VkDescriptorSet ds_input_norm = VK_NULL_HANDLE;
    VkDescriptorSet ds_post_norm = VK_NULL_HANDLE;

    // Fused Pipeline Descriptors
    VkDescriptorSet ds_fused_qkv = VK_NULL_HANDLE;
    VkDescriptorSet ds_fused_swiglu = VK_NULL_HANDLE;
};

class VulkanInferenceEngine {
public:
    VulkanInferenceEngine();
    ~VulkanInferenceEngine();

    bool init();
    bool upload_model_weights(const void *raw_mmap_base, const ModelOffsets &offsets);
    void reset_kv_cache();

    // Forward pass on AMD RX 580: returns logits in out_logits [VOCAB_SIZE]
    bool forward_token(int token_id, int pos, float *out_logits);

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
    ComputePipeline m_pipe_q8_gemv;
    ComputePipeline m_pipe_rmsnorm;
    ComputePipeline m_pipe_rope_cache;
    ComputePipeline m_pipe_attention;
    ComputePipeline m_pipe_swiglu;
    ComputePipeline m_pipe_residual_add;

    // Fused Kernels
    ComputePipeline m_pipe_fused_qkv;
    ComputePipeline m_pipe_fused_swiglu;

    // Model Weights in VRAM
    VulkanBuffer m_w_embed_tokens; // [49152, 30] blocks
    VulkanBuffer m_w_final_norm;   // [960] floats
    std::vector<VulkanLayerWeights> m_layers;

    // KV Cache in VRAM
    VulkanBuffer m_buf_k_cache; // [32 * 1024 * 320] floats (~40 MB)
    VulkanBuffer m_buf_v_cache; // [32 * 1024 * 320] floats (~40 MB)

    // Intermediate Activations in VRAM
    VulkanBuffer m_act_x;            // [960]
    VulkanBuffer m_act_norm_x;       // [960]
    VulkanBuffer m_act_q;            // [960]
    VulkanBuffer m_act_k;            // [320]
    VulkanBuffer m_act_v;            // [320]
    VulkanBuffer m_act_head_outputs; // [960] Attention heads output
    VulkanBuffer m_act_attn_out;     // [960] O-proj output
    VulkanBuffer m_act_post_norm_x;  // [960]
    VulkanBuffer m_act_gate;         // [2560]
    VulkanBuffer m_act_up;           // [2560]
    VulkanBuffer m_act_mlp_hidden;   // [2560]
    VulkanBuffer m_act_mlp_out;      // [960]
    VulkanBuffer m_act_logits;       // [49152]
    VulkanBuffer m_buf_pos;          // [1] uint32_t current_pos mapped buffer

    // Embedding weights table for host lookup
    std::vector<float> m_host_embed_table; // [VOCAB_SIZE * HIDDEN_SIZE]

    // Pre-allocated static Descriptor Sets
    VkDescriptorSet m_ds_final_norm = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_lm_head = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_rope_cache = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_attention = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_swiglu = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_res_attn = VK_NULL_HANDLE;
    VkDescriptorSet m_ds_res_mlp = VK_NULL_HANDLE;

    // Command execution
    VkCommandBuffer m_cmd_buffer = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
};

#endif // VK_ENGINE_H
