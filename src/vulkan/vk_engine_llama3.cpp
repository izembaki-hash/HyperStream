#include "vk_engine_llama3.h"
#include "shaders/compiled_llama3_shaders.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>

#pragma pack(push, 1)
struct GGUFBlockQ8 {
    uint16_t d; // fp16
    int8_t qs[32];
};
#pragma pack(pop)

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    if (exp == 0) {
        if (mant == 0) {
            union { uint32_t u; float f; } val;
            val.u = sign;
            return val.f;
        }
        while (!(mant & 0x0400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        union { uint32_t u; float f; } val;
        val.u = sign | 0x7F800000 | (mant << 13);
        return val.f;
    }
    exp = exp + (127 - 15);
    union { uint32_t u; float f; } val;
    val.u = sign | (exp << 23) | (mant << 13);
    return val.f;
}

static void unpack_gguf_to_vulkan_blocks(std::vector<BlockQ8Llama3> &out_blocks, const GGUFBlockQ8 *src_blocks, size_t total_blocks) {
    out_blocks.resize(total_blocks);
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < total_blocks; b++) {
        out_blocks[b].scale = fp16_to_fp32(src_blocks[b].d);
        memcpy(out_blocks[b].qs, src_blocks[b].qs, 32);
    }
}

VulkanInferenceEngineLlama3::VulkanInferenceEngineLlama3() {
}

VulkanInferenceEngineLlama3::~VulkanInferenceEngineLlama3() {
    cleanup();
}

bool VulkanInferenceEngineLlama3::init() {
    if (!m_ctx.init()) {
        std::cerr << "[Vulkan Llama3 Engine] Context initialization failed.\n";
        return false;
    }

    if (!init_pipelines()) {
        std::cerr << "[Vulkan Llama3 Engine] Pipeline creation failed.\n";
        return false;
    }

    if (!allocate_buffers()) {
        std::cerr << "[Vulkan Llama3 Engine] Buffer allocation failed.\n";
        return false;
    }

    // Allocate persistent command buffers (normal and argmax) & fence
    VkCommandBufferAllocateInfo cmd_alloc = {};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = m_ctx.command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 2;

    VkCommandBuffer cmds[2];
    VkResult res = g_pfn_vkAllocateCommandBuffers(m_ctx.device, &cmd_alloc, cmds);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Llama3 Engine] Failed to allocate command buffer: " << res << "\n";
        return false;
    }
    m_cmd_buffer = cmds[0];
    m_cmd_buffer_argmax = cmds[1];

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    res = g_pfn_vkCreateFence(m_ctx.device, &fence_info, nullptr, &m_fence);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Llama3 Engine] Failed to create fence: " << res << "\n";
        return false;
    }

    m_initialized = true;
    std::cout << "[Vulkan Llama3 Engine] Initialized successfully on " << m_ctx.device_name << "\n";
    return true;
}

bool VulkanInferenceEngineLlama3::init_pipelines() {
    // 1. RMSNorm (push consts: uint size, float eps = 8 bytes; 3 buffers: in, w, out)
    if (!m_pipe_rmsnorm.init_from_spv(m_ctx, spv_llama3_rmsnorm_data, spv_llama3_rmsnorm_size, 3, 8)) {
        return false;
    }

    // 2. Fused QKV GEMV (push consts: 16 bytes; 7 buffers: in, wq, wk, wv, out_q, out_k, out_v)
    if (!m_pipe_fused_qkv.init_from_spv(m_ctx, spv_llama3_fused_qkv_data, spv_llama3_fused_qkv_size, 7, 16)) {
        return false;
    }

    // 3. RoPE & KV Cache (push consts: 8 bytes; 6 buffers: q, k, v, k_cache, v_cache, pos)
    if (!m_pipe_rope_cache.init_from_spv(m_ctx, spv_llama3_rope_cache_data, spv_llama3_rope_cache_size, 6, 8)) {
        return false;
    }

    // 4. Attention (push consts: 8 bytes; 5 buffers: q, k_cache, v_cache, out, pos)
    if (!m_pipe_attention.init_from_spv(m_ctx, spv_llama3_attention_data, spv_llama3_attention_size, 5, 8)) {
        return false;
    }

    // 5. Q8 GEMV (push consts: 8 bytes; 3 buffers: in, w, out)
    if (!m_pipe_q8_gemv.init_from_spv(m_ctx, spv_llama3_q8_gemv_data, spv_llama3_q8_gemv_size, 3, 8)) {
        return false;
    }

    // 6. Fused SwiGLU GEMV (push consts: 8 bytes; 4 buffers: in, w_gate, w_up, out_activated)
    if (!m_pipe_fused_swiglu.init_from_spv(m_ctx, spv_llama3_fused_swiglu_data, spv_llama3_fused_swiglu_size, 4, 8)) {
        return false;
    }

    // 7. Residual Add (push consts: 4 bytes; 2 buffers: x, delta)
    if (!m_pipe_residual_add.init_from_spv(m_ctx, spv_llama3_residual_add_data, spv_llama3_residual_add_size, 2, 4)) {
        return false;
    }

    // 8. Argmax Pass 1 (push consts: 4 bytes; 3 buffers: in_logits, part_vals, part_ids)
    if (!m_pipe_argmax_p1.init_from_spv(m_ctx, spv_llama3_argmax_pass1_data, spv_llama3_argmax_pass1_size, 3, 4)) {
        return false;
    }

    // 9. Argmax Pass 2 (push consts: 4 bytes; 3 buffers: part_vals, part_ids, result_id)
    if (!m_pipe_argmax_p2.init_from_spv(m_ctx, spv_llama3_argmax_pass2_data, spv_llama3_argmax_pass2_size, 3, 4)) {
        return false;
    }

    std::cout << "[Vulkan Llama3 Engine] All Compute Pipelines compiled and loaded.\n";
    return true;
}

bool VulkanInferenceEngineLlama3::allocate_buffers() {
    VkBufferUsageFlags storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags vram_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // Intermediate activations
    m_act_x.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_act_x.memory, 0, LLAMA3_HIDDEN_SIZE * sizeof(float), 0, &m_act_x.mapped_ptr);

    m_act_norm_x.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_q.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_k.allocate(m_ctx, LLAMA3_NUM_KV_HEADS * LLAMA3_HEAD_DIM * sizeof(float), storage_usage, vram_flags);
    m_act_v.allocate(m_ctx, LLAMA3_NUM_KV_HEADS * LLAMA3_HEAD_DIM * sizeof(float), storage_usage, vram_flags);
    m_act_head_outputs.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_attn_out.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_post_norm_x.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_mlp_hidden.allocate(m_ctx, LLAMA3_INTERMEDIATE_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_mlp_out.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);

    m_act_logits.allocate(m_ctx, LLAMA3_VOCAB_SIZE * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_act_logits.memory, 0, LLAMA3_VOCAB_SIZE * sizeof(float), 0, &m_act_logits.mapped_ptr);

    // Dynamic sequence position buffer
    m_buf_pos.allocate(m_ctx, sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_buf_pos.memory, 0, sizeof(uint32_t), 0, &m_buf_pos.mapped_ptr);
    *(uint32_t*)m_buf_pos.mapped_ptr = 0;

    // KV Cache in VRAM: [28 layers * 1024 seq * 1024 dim] floats = ~117 MB each
    size_t kv_cache_bytes = (size_t)LLAMA3_NUM_LAYERS * 1024 * (LLAMA3_NUM_KV_HEADS * LLAMA3_HEAD_DIM) * sizeof(float);
    m_buf_k_cache.allocate(m_ctx, kv_cache_bytes, storage_usage, vram_flags);
    m_buf_v_cache.allocate(m_ctx, kv_cache_bytes, storage_usage, vram_flags);

    // GPU Argmax reduction buffers
    m_buf_arg_part_vals.allocate(m_ctx, 128 * sizeof(float), storage_usage, vram_flags);
    m_buf_arg_part_ids.allocate(m_ctx, 128 * sizeof(uint32_t), storage_usage, vram_flags);
    m_buf_arg_token_result.allocate(m_ctx, 2 * sizeof(uint32_t), storage_usage | host_flags, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_buf_arg_token_result.memory, 0, 2 * sizeof(uint32_t), 0, &m_buf_arg_token_result.mapped_ptr);

    return true;
}

void VulkanInferenceEngineLlama3::reset_kv_cache() {
    if (m_buf_pos.mapped_ptr) {
        *(uint32_t*)m_buf_pos.mapped_ptr = 0;
    }
}

bool VulkanInferenceEngineLlama3::upload_model_weights(const void *raw_mmap_base, const Llama3Offsets &offsets) {
    const uint8_t *base = (const uint8_t*)raw_mmap_base + LLAMA3_DATA_START_OFFSET;
    m_raw_embed_base = base + offsets.embed_tokens;

    std::cout << "[Vulkan Llama3 Engine] Uploading 3.19 GB model weights into AMD RX 580 GDDR5 VRAM...\n";

    // 1. LM Head (Tied to embed_tokens): [128256, 96] blocks
    size_t embed_num_blocks = (size_t)LLAMA3_VOCAB_SIZE * (LLAMA3_HIDDEN_SIZE / 32);
    const GGUFBlockQ8 *embed_src = (const GGUFBlockQ8*)(base + offsets.embed_tokens);
    std::vector<BlockQ8Llama3> embed_blocks;
    unpack_gguf_to_vulkan_blocks(embed_blocks, embed_src, embed_num_blocks);

    size_t embed_bytes = embed_blocks.size() * sizeof(BlockQ8Llama3);
    m_w_embed_tokens.allocate(m_ctx, embed_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_w_embed_tokens.upload_staging(m_ctx, embed_blocks.data(), embed_bytes);

    // LM Head Descriptor Set: (in: norm_x, w: embed_tokens, out: logits)
    m_ds_lm_head = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_norm_x.buffer, m_w_embed_tokens.buffer, m_act_logits.buffer }, {});

    // 2. Final Layernorm (FP32)
    const float *final_norm_src = (const float*)(base + offsets.final_norm);
    m_w_final_norm.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_w_final_norm.upload_staging(m_ctx, final_norm_src, LLAMA3_HIDDEN_SIZE * sizeof(float));

    // Final Norm Descriptor Set: (in: x, w: final_norm, out: norm_x)
    m_ds_final_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, m_w_final_norm.buffer, m_act_norm_x.buffer }, {});

    // 3. Static Residual Add Descriptor Sets
    m_ds_res_attn = m_pipe_residual_add.create_descriptor_set(m_ctx, { m_act_x.buffer, m_act_attn_out.buffer }, {});
    m_ds_res_mlp  = m_pipe_residual_add.create_descriptor_set(m_ctx, { m_act_x.buffer, m_act_mlp_out.buffer }, {});

    // 4. Static Argmax Descriptor Sets
    m_ds_arg_p1 = m_pipe_argmax_p1.create_descriptor_set(m_ctx, { m_act_logits.buffer, m_buf_arg_part_vals.buffer, m_buf_arg_part_ids.buffer }, {});
    m_ds_arg_p2 = m_pipe_argmax_p2.create_descriptor_set(m_ctx, { m_buf_arg_part_vals.buffer, m_buf_arg_part_ids.buffer, m_buf_arg_token_result.buffer }, {});

    // 5. Layers (28 Layers)
    m_layers.resize(LLAMA3_NUM_LAYERS);
    VkBufferUsageFlags w_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags vram_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    for (int l = 0; l < LLAMA3_NUM_LAYERS; l++) {
        const Llama3LayerOffsets &lo = offsets.layers[l];
        VulkanLlama3LayerWeights &lw = m_layers[l];

        // Layernorms (FP32)
        const float *in_norm_src = (const float*)(base + lo.input_layernorm);
        const float *post_norm_src = (const float*)(base + lo.post_attn_layernorm);

        lw.input_layernorm.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), w_usage, vram_flags);
        lw.input_layernorm.upload_staging(m_ctx, in_norm_src, LLAMA3_HIDDEN_SIZE * sizeof(float));

        lw.post_attn_layernorm.allocate(m_ctx, LLAMA3_HIDDEN_SIZE * sizeof(float), w_usage, vram_flags);
        lw.post_attn_layernorm.upload_staging(m_ctx, post_norm_src, LLAMA3_HIDDEN_SIZE * sizeof(float));

        // Unpack Projections
        std::vector<BlockQ8Llama3> q_blocks, k_blocks, v_blocks, o_blocks, gate_blocks, up_blocks, down_blocks;
        unpack_gguf_to_vulkan_blocks(q_blocks, (const GGUFBlockQ8*)(base + lo.q_proj), (size_t)LLAMA3_HIDDEN_SIZE * (LLAMA3_HIDDEN_SIZE / 32));
        unpack_gguf_to_vulkan_blocks(k_blocks, (const GGUFBlockQ8*)(base + lo.k_proj), (size_t)(LLAMA3_NUM_KV_HEADS * LLAMA3_HEAD_DIM) * (LLAMA3_HIDDEN_SIZE / 32));
        unpack_gguf_to_vulkan_blocks(v_blocks, (const GGUFBlockQ8*)(base + lo.v_proj), (size_t)(LLAMA3_NUM_KV_HEADS * LLAMA3_HEAD_DIM) * (LLAMA3_HIDDEN_SIZE / 32));
        unpack_gguf_to_vulkan_blocks(o_blocks, (const GGUFBlockQ8*)(base + lo.o_proj), (size_t)LLAMA3_HIDDEN_SIZE * (LLAMA3_HIDDEN_SIZE / 32));
        unpack_gguf_to_vulkan_blocks(gate_blocks, (const GGUFBlockQ8*)(base + lo.gate_proj), (size_t)LLAMA3_INTERMEDIATE_SIZE * (LLAMA3_HIDDEN_SIZE / 32));
        unpack_gguf_to_vulkan_blocks(up_blocks,   (const GGUFBlockQ8*)(base + lo.up_proj),   (size_t)LLAMA3_INTERMEDIATE_SIZE * (LLAMA3_HIDDEN_SIZE / 32));
        unpack_gguf_to_vulkan_blocks(down_blocks, (const GGUFBlockQ8*)(base + lo.down_proj), (size_t)LLAMA3_HIDDEN_SIZE * (LLAMA3_INTERMEDIATE_SIZE / 32));

        lw.q_proj.allocate(m_ctx, q_blocks.size() * sizeof(BlockQ8Llama3), w_usage, vram_flags);
        lw.q_proj.upload_staging(m_ctx, q_blocks.data(), q_blocks.size() * sizeof(BlockQ8Llama3));

        lw.k_proj.allocate(m_ctx, k_blocks.size() * sizeof(BlockQ8Llama3), w_usage, vram_flags);
        lw.k_proj.upload_staging(m_ctx, k_blocks.data(), k_blocks.size() * sizeof(BlockQ8Llama3));

        lw.v_proj.allocate(m_ctx, v_blocks.size() * sizeof(BlockQ8Llama3), w_usage, vram_flags);
        lw.v_proj.upload_staging(m_ctx, v_blocks.data(), v_blocks.size() * sizeof(BlockQ8Llama3));

        lw.o_proj.allocate(m_ctx, o_blocks.size() * sizeof(BlockQ8Llama3), w_usage, vram_flags);
        lw.o_proj.upload_staging(m_ctx, o_blocks.data(), o_blocks.size() * sizeof(BlockQ8Llama3));

        lw.gate_proj.allocate(m_ctx, gate_blocks.size() * sizeof(BlockQ8Llama3), w_usage, vram_flags);
        lw.gate_proj.upload_staging(m_ctx, gate_blocks.data(), gate_blocks.size() * sizeof(BlockQ8Llama3));

        lw.up_proj.allocate(m_ctx, up_blocks.size() * sizeof(BlockQ8Llama3), w_usage, vram_flags);
        lw.up_proj.upload_staging(m_ctx, up_blocks.data(), up_blocks.size() * sizeof(BlockQ8Llama3));

        lw.down_proj.allocate(m_ctx, down_blocks.size() * sizeof(BlockQ8Llama3), w_usage, vram_flags);
        lw.down_proj.upload_staging(m_ctx, down_blocks.data(), down_blocks.size() * sizeof(BlockQ8Llama3));

        // Create Static Descriptor Sets for this layer
        lw.ds_input_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, lw.input_layernorm.buffer, m_act_norm_x.buffer }, {});
        lw.ds_fused_qkv = m_pipe_fused_qkv.create_descriptor_set(m_ctx, {
            m_act_norm_x.buffer, lw.q_proj.buffer, lw.k_proj.buffer, lw.v_proj.buffer,
            m_act_q.buffer, m_act_k.buffer, m_act_v.buffer
        }, {});

        lw.ds_rope_cache = m_pipe_rope_cache.create_descriptor_set(m_ctx, {
            m_act_q.buffer, m_act_k.buffer, m_act_v.buffer,
            m_buf_k_cache.buffer, m_buf_v_cache.buffer, m_buf_pos.buffer
        }, {});

        lw.ds_attention = m_pipe_attention.create_descriptor_set(m_ctx, {
            m_act_q.buffer, m_buf_k_cache.buffer, m_buf_v_cache.buffer,
            m_act_head_outputs.buffer, m_buf_pos.buffer
        }, {});

        lw.ds_o_proj = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_head_outputs.buffer, lw.o_proj.buffer, m_act_attn_out.buffer }, {});
        lw.ds_post_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, lw.post_attn_layernorm.buffer, m_act_post_norm_x.buffer }, {});

        lw.ds_fused_swiglu = m_pipe_fused_swiglu.create_descriptor_set(m_ctx, {
            m_act_post_norm_x.buffer, lw.gate_proj.buffer, lw.up_proj.buffer, m_act_mlp_hidden.buffer
        }, {});

        lw.ds_down_proj = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_mlp_hidden.buffer, lw.down_proj.buffer, m_act_mlp_out.buffer }, {});

        if ((l + 1) % 7 == 0 || l == LLAMA3_NUM_LAYERS - 1) {
            std::cout << "  [Layer " << (l + 1) << "/" << LLAMA3_NUM_LAYERS << "] Uploaded and Descriptors Bound.\n";
        }
    }

    // Pre-record static command buffer
    record_forward_command_buffer();
    std::cout << "[Vulkan Llama3 Engine] All weights uploaded & Static Command Buffer recorded!\n";
    return true;
}

void VulkanInferenceEngineLlama3::record_forward_command_buffer() {
    auto record_transformer = [&](VkCommandBuffer cmd) {
        VkMemoryBarrier comp_barrier = {};
        comp_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        comp_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        comp_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        // Loop through all 28 layers
        for (int l = 0; l < LLAMA3_NUM_LAYERS; l++) {
            const VulkanLlama3LayerWeights &lw = m_layers[l];

            // 1. Input RMSNorm: x -> norm_x
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &lw.ds_input_norm, 0, nullptr);
            struct { uint32_t size; float eps; } pc_norm = { LLAMA3_HIDDEN_SIZE, LLAMA3_RMS_NORM_EPS };
            g_pfn_vkCmdPushConstants(cmd, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_norm), &pc_norm);
            g_pfn_vkCmdDispatch(cmd, 1, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 2. Fused QKV GEMV: norm_x -> q, k, v (5120 workgroups, 1 wavefront per row, coalesced)
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_qkv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_qkv.pipeline_layout, 0, 1, &lw.ds_fused_qkv, 0, nullptr);
            struct { uint32_t q_rows; uint32_t k_rows; uint32_t v_rows; uint32_t cols; } pc_qkv = {
                LLAMA3_HIDDEN_SIZE, LLAMA3_NUM_KV_HEADS * LLAMA3_HEAD_DIM, LLAMA3_NUM_KV_HEADS * LLAMA3_HEAD_DIM, LLAMA3_HIDDEN_SIZE
            };
            g_pfn_vkCmdPushConstants(cmd, m_pipe_fused_qkv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_qkv), &pc_qkv);
            g_pfn_vkCmdDispatch(cmd, (LLAMA3_HIDDEN_SIZE + 2 * LLAMA3_NUM_KV_HEADS * LLAMA3_HEAD_DIM), 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 3. RoPE & KV Cache: q, k, v -> rot_q, rot_k, and caches
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rope_cache.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rope_cache.pipeline_layout, 0, 1, &lw.ds_rope_cache, 0, nullptr);
            struct { uint32_t layer_idx; uint32_t max_seq_len; } pc_rope = { (uint32_t)l, 1024 };
            g_pfn_vkCmdPushConstants(cmd, m_pipe_rope_cache.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_rope), &pc_rope);
            g_pfn_vkCmdDispatch(cmd, 24, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 4. Attention Heads: q, k_cache, v_cache -> head_outputs
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_attention.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_attention.pipeline_layout, 0, 1, &lw.ds_attention, 0, nullptr);
            g_pfn_vkCmdPushConstants(cmd, m_pipe_attention.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_rope), &pc_rope);
            g_pfn_vkCmdDispatch(cmd, LLAMA3_NUM_HEADS, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 5. O-proj GEMV: head_outputs -> attn_out (3072 workgroups, coalesced)
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout, 0, 1, &lw.ds_o_proj, 0, nullptr);
            struct { uint32_t rows; uint32_t cols; } pc_o = { LLAMA3_HIDDEN_SIZE, LLAMA3_HIDDEN_SIZE };
            g_pfn_vkCmdPushConstants(cmd, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_o), &pc_o);
            g_pfn_vkCmdDispatch(cmd, LLAMA3_HIDDEN_SIZE, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 6. Residual Add: x = x + attn_out
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline_layout, 0, 1, &m_ds_res_attn, 0, nullptr);
            uint32_t pc_res = LLAMA3_HIDDEN_SIZE;
            g_pfn_vkCmdPushConstants(cmd, m_pipe_residual_add.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_res), &pc_res);
            g_pfn_vkCmdDispatch(cmd, (LLAMA3_HIDDEN_SIZE + 255) / 256, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 7. Post-Attn RMSNorm: x -> post_norm_x
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &lw.ds_post_norm, 0, nullptr);
            g_pfn_vkCmdPushConstants(cmd, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_norm), &pc_norm);
            g_pfn_vkCmdDispatch(cmd, 1, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 8. Fused SwiGLU GEMV: post_norm_x -> mlp_hidden (8192 workgroups across all CUs, coalesced)
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_swiglu.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_swiglu.pipeline_layout, 0, 1, &lw.ds_fused_swiglu, 0, nullptr);
            struct { uint32_t rows; uint32_t cols; } pc_swiglu = { LLAMA3_INTERMEDIATE_SIZE, LLAMA3_HIDDEN_SIZE };
            g_pfn_vkCmdPushConstants(cmd, m_pipe_fused_swiglu.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_swiglu), &pc_swiglu);
            g_pfn_vkCmdDispatch(cmd, LLAMA3_INTERMEDIATE_SIZE, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 9. Down-proj GEMV: mlp_hidden -> mlp_out (3072 workgroups, coalesced)
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout, 0, 1, &lw.ds_down_proj, 0, nullptr);
            struct { uint32_t rows; uint32_t cols; } pc_down = { LLAMA3_HIDDEN_SIZE, LLAMA3_INTERMEDIATE_SIZE };
            g_pfn_vkCmdPushConstants(cmd, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_down), &pc_down);
            g_pfn_vkCmdDispatch(cmd, LLAMA3_HIDDEN_SIZE, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

            // 10. Residual Add: x = x + mlp_out
            g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline_layout, 0, 1, &m_ds_res_mlp, 0, nullptr);
            g_pfn_vkCmdPushConstants(cmd, m_pipe_residual_add.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_res), &pc_res);
            g_pfn_vkCmdDispatch(cmd, (LLAMA3_HIDDEN_SIZE + 255) / 256, 1, 1);
            g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);
        }

        // Final RMSNorm: x -> norm_x
        g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
        g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &m_ds_final_norm, 0, nullptr);
        struct { uint32_t size; float eps; } pc_norm = { LLAMA3_HIDDEN_SIZE, LLAMA3_RMS_NORM_EPS };
        g_pfn_vkCmdPushConstants(cmd, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_norm), &pc_norm);
        g_pfn_vkCmdDispatch(cmd, 1, 1, 1);
        g_pfn_vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &comp_barrier, 0, nullptr, 0, nullptr);

        // LM Head GEMV: norm_x -> logits (128256 workgroups, coalesced)
        g_pfn_vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
        g_pfn_vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout, 0, 1, &m_ds_lm_head, 0, nullptr);
        struct { uint32_t rows; uint32_t cols; } pc_lm = { LLAMA3_VOCAB_SIZE, LLAMA3_HIDDEN_SIZE };
        g_pfn_vkCmdPushConstants(cmd, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_lm), &pc_lm);
        g_pfn_vkCmdDispatch(cmd, LLAMA3_VOCAB_SIZE, 1, 1);
    };

    // 1. Record Standard Command Buffer (full logits in m_act_logits)
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    g_pfn_vkBeginCommandBuffer(m_cmd_buffer, &begin_info);
    record_transformer(m_cmd_buffer);
    g_pfn_vkEndCommandBuffer(m_cmd_buffer);

    // 2. Record GPU Argmax Command Buffer (runs reduction and writes winner token ID)
    g_pfn_vkBeginCommandBuffer(m_cmd_buffer_argmax, &begin_info);
    record_transformer(m_cmd_buffer_argmax);

    // Barrier: LM Head -> Argmax Pass 1
    VkMemoryBarrier b_arg = {};
    b_arg.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b_arg.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b_arg.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    g_pfn_vkCmdPipelineBarrier(m_cmd_buffer_argmax, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b_arg, 0, nullptr, 0, nullptr);

    // Argmax Pass 1: 128 workgroups of 256 threads
    g_pfn_vkCmdBindPipeline(m_cmd_buffer_argmax, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p1.pipeline);
    g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer_argmax, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p1.pipeline_layout, 0, 1, &m_ds_arg_p1, 0, nullptr);
    struct { uint32_t v_sz; } pc_a1 = { LLAMA3_VOCAB_SIZE };
    g_pfn_vkCmdPushConstants(m_cmd_buffer_argmax, m_pipe_argmax_p1.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_a1), &pc_a1);
    g_pfn_vkCmdDispatch(m_cmd_buffer_argmax, 128, 1, 1);

    // Barrier: Pass 1 -> Pass 2
    g_pfn_vkCmdPipelineBarrier(m_cmd_buffer_argmax, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b_arg, 0, nullptr, 0, nullptr);

    // Argmax Pass 2: 1 workgroup of 128 threads
    g_pfn_vkCmdBindPipeline(m_cmd_buffer_argmax, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p2.pipeline);
    g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer_argmax, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p2.pipeline_layout, 0, 1, &m_ds_arg_p2, 0, nullptr);
    struct { uint32_t cands; } pc_a2 = { 128 };
    g_pfn_vkCmdPushConstants(m_cmd_buffer_argmax, m_pipe_argmax_p2.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_a2), &pc_a2);
    g_pfn_vkCmdDispatch(m_cmd_buffer_argmax, 1, 1, 1);

    g_pfn_vkEndCommandBuffer(m_cmd_buffer_argmax);
}

int VulkanInferenceEngineLlama3::forward_token_gpu_argmax(int token_id, int pos) {
    if (!m_initialized || token_id < 0 || token_id >= LLAMA3_VOCAB_SIZE) return -1;

    // 1. Direct on-the-fly dequantization of input token embedding row
    float *dst_x = (float*)m_act_x.mapped_ptr;
    const GGUFBlockQ8 *row_blocks = (const GGUFBlockQ8*)(m_raw_embed_base + (size_t)token_id * (LLAMA3_HIDDEN_SIZE / 32) * sizeof(GGUFBlockQ8));
    for (int b = 0; b < LLAMA3_HIDDEN_SIZE / 32; b++) {
        float scale = fp16_to_fp32(row_blocks[b].d);
        float *dst = dst_x + b * 32;
        const int8_t *qs = row_blocks[b].qs;
        for (int i = 0; i < 32; i++) {
            dst[i] = (float)qs[i] * scale;
        }
    }

    // 2. Update sequence position
    *(uint32_t*)m_buf_pos.mapped_ptr = (uint32_t)pos;

    // 3. Submit GPU Argmax command buffer
    g_pfn_vkResetFences(m_ctx.device, 1, &m_fence);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_cmd_buffer_argmax;

    VkResult res = g_pfn_vkQueueSubmit(m_ctx.compute_queue, 1, &submit_info, m_fence);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Llama3 Engine] Queue submit failed: " << res << "\n";
        return -1;
    }

    // 4. Wait for GPU completion
    g_pfn_vkWaitForFences(m_ctx.device, 1, &m_fence, VK_TRUE, UINT64_MAX);

    // 5. Read back single winner token ID (4 bytes)
    return (int)*(const uint32_t*)m_buf_arg_token_result.mapped_ptr;
}

bool VulkanInferenceEngineLlama3::forward_token(int token_id, int pos, float *out_logits) {
    if (!m_initialized || token_id < 0 || token_id >= LLAMA3_VOCAB_SIZE) return false;

    // 1. Direct on-the-fly dequantization of input token embedding row into m_act_x.mapped_ptr
    float *dst_x = (float*)m_act_x.mapped_ptr;
    const GGUFBlockQ8 *row_blocks = (const GGUFBlockQ8*)(m_raw_embed_base + (size_t)token_id * (LLAMA3_HIDDEN_SIZE / 32) * sizeof(GGUFBlockQ8));
    for (int b = 0; b < LLAMA3_HIDDEN_SIZE / 32; b++) {
        float scale = fp16_to_fp32(row_blocks[b].d);
        float *dst = dst_x + b * 32;
        const int8_t *qs = row_blocks[b].qs;
        for (int i = 0; i < 32; i++) {
            dst[i] = (float)qs[i] * scale;
        }
    }

    // 2. Update sequence position
    *(uint32_t*)m_buf_pos.mapped_ptr = (uint32_t)pos;

    // 3. Submit pre-recorded static command buffer
    g_pfn_vkResetFences(m_ctx.device, 1, &m_fence);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_cmd_buffer;

    VkResult res = g_pfn_vkQueueSubmit(m_ctx.compute_queue, 1, &submit_info, m_fence);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Llama3 Engine] Queue submit failed: " << res << "\n";
        return false;
    }

    // 4. Wait for GPU completion
    g_pfn_vkWaitForFences(m_ctx.device, 1, &m_fence, VK_TRUE, UINT64_MAX);

    // 5. Copy logits to caller output (only if requested)
    if (out_logits) {
        memcpy(out_logits, m_act_logits.mapped_ptr, LLAMA3_VOCAB_SIZE * sizeof(float));
    }

    return true;
}

void VulkanInferenceEngineLlama3::cleanup() {
    if (m_ctx.device != VK_NULL_HANDLE) {
        g_pfn_vkQueueWaitIdle(m_ctx.compute_queue);

        if (m_fence != VK_NULL_HANDLE) {
            g_pfn_vkDestroyFence(m_ctx.device, m_fence, nullptr);
            m_fence = VK_NULL_HANDLE;
        }
    }
    m_initialized = false;
}
