#include "vk_engine_qwen2.h"
#include "shaders/compiled_qwen2_shaders.h"
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

static void unpack_gguf_to_vulkan_blocks(std::vector<BlockQ8Qwen2> &out_blocks, const GGUFBlockQ8 *src_blocks, size_t total_blocks) {
    out_blocks.resize(total_blocks);
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < total_blocks; b++) {
        out_blocks[b].scale = fp16_to_fp32(src_blocks[b].d);
        memcpy(out_blocks[b].qs, src_blocks[b].qs, 32);
    }
}

VulkanInferenceEngineQwen2::VulkanInferenceEngineQwen2() {
}

VulkanInferenceEngineQwen2::~VulkanInferenceEngineQwen2() {
    cleanup();
}

bool VulkanInferenceEngineQwen2::init() {
    if (!m_ctx.init()) {
        std::cerr << "[Vulkan Qwen2 Engine] Context initialization failed.\n";
        return false;
    }

    if (!init_pipelines()) {
        std::cerr << "[Vulkan Qwen2 Engine] Pipeline creation failed.\n";
        return false;
    }

    if (!allocate_buffers()) {
        std::cerr << "[Vulkan Qwen2 Engine] Buffer allocation failed.\n";
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
        std::cerr << "[Vulkan Qwen2 Engine] Failed to allocate command buffer: " << res << "\n";
        return false;
    }
    m_cmd_buffer = cmds[0];
    m_cmd_buffer_argmax = cmds[1];

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    res = g_pfn_vkCreateFence(m_ctx.device, &fence_info, nullptr, &m_fence);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Qwen2 Engine] Failed to create fence: " << res << "\n";
        return false;
    }

    m_initialized = true;
    std::cout << "[Vulkan Qwen2 Engine] Initialized successfully on " << m_ctx.device_name << "\n";
    return true;
}

bool VulkanInferenceEngineQwen2::init_pipelines() {
    // 1. RMSNorm (push consts: uint size, float eps = 8 bytes; 3 buffers: in, w, out)
    if (!m_pipe_rmsnorm.init_from_spv(m_ctx, spv_qwen2_rmsnorm_data, spv_qwen2_rmsnorm_size, 3, 8)) {
        return false;
    }

    // 2. Fused QKV GEMV (push consts: 16 bytes; 10 buffers: in, wq, wk, wv, out_q, out_k, out_v, bq, bk, bv)
    if (!m_pipe_fused_qkv.init_from_spv(m_ctx, spv_qwen2_fused_qkv_data, spv_qwen2_fused_qkv_size, 10, 16)) {
        return false;
    }

    // 3. RoPE & KV Cache (push consts: 24 bytes; 6 buffers: q, k, v, k_cache, v_cache, pos)
    if (!m_pipe_rope_cache.init_from_spv(m_ctx, spv_qwen2_rope_cache_data, spv_qwen2_rope_cache_size, 6, 24)) {
        return false;
    }

    // 4. Attention (push consts: 24 bytes; 5 buffers: q, k_cache, v_cache, out, pos)
    if (!m_pipe_attention.init_from_spv(m_ctx, spv_qwen2_attention_data, spv_qwen2_attention_size, 5, 24)) {
        return false;
    }

    // 5. Q8 GEMV (push consts: 8 bytes; 3 buffers: in, w, out)
    if (!m_pipe_q8_gemv.init_from_spv(m_ctx, spv_qwen2_q8_gemv_data, spv_qwen2_q8_gemv_size, 3, 8)) {
        return false;
    }

    // 6. Fused SwiGLU GEMV (push consts: 8 bytes; 4 buffers: in, w_gate, w_up, out_activated)
    if (!m_pipe_fused_swiglu.init_from_spv(m_ctx, spv_qwen2_fused_swiglu_data, spv_qwen2_fused_swiglu_size, 4, 8)) {
        return false;
    }

    // 7. Residual Add (push consts: 4 bytes; 2 buffers: x, delta)
    if (!m_pipe_residual_add.init_from_spv(m_ctx, spv_qwen2_residual_add_data, spv_qwen2_residual_add_size, 2, 4)) {
        return false;
    }

    // 8. Argmax Pass 1 (push consts: 4 bytes; 3 buffers: in_logits, part_vals, part_ids)
    if (!m_pipe_argmax_p1.init_from_spv(m_ctx, spv_qwen2_argmax_pass1_data, spv_qwen2_argmax_pass1_size, 3, 4)) {
        return false;
    }

    // 9. Argmax Pass 2 (push consts: 4 bytes; 3 buffers: part_vals, part_ids, result_id)
    if (!m_pipe_argmax_p2.init_from_spv(m_ctx, spv_qwen2_argmax_pass2_data, spv_qwen2_argmax_pass2_size, 3, 4)) {
        return false;
    }

    std::cout << "[Vulkan Qwen2 Engine] All Compute Pipelines compiled and loaded.\n";
    return true;
}

bool VulkanInferenceEngineQwen2::allocate_buffers() {
    VkBufferUsageFlags storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags vram_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // Intermediate activations
    m_act_x.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_act_x.memory, 0, QWEN2_HIDDEN_SIZE * sizeof(float), 0, &m_act_x.mapped_ptr);

    m_act_norm_x.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_q.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_k.allocate(m_ctx, QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM * sizeof(float), storage_usage, vram_flags);
    m_act_v.allocate(m_ctx, QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM * sizeof(float), storage_usage, vram_flags);
    m_act_head_outputs.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_attn_out.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_post_norm_x.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_mlp_hidden.allocate(m_ctx, QWEN2_INTERMEDIATE_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_mlp_out.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);

    // Host-visible Logits buffer for standard CPU readback
    m_act_logits.allocate(m_ctx, QWEN2_VOCAB_SIZE * sizeof(float), storage_usage, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_act_logits.memory, 0, QWEN2_VOCAB_SIZE * sizeof(float), 0, &m_act_logits.mapped_ptr);

    // Sequence position buffer
    m_buf_pos.allocate(m_ctx, sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_buf_pos.memory, 0, sizeof(uint32_t), 0, &m_buf_pos.mapped_ptr);

    // KV Cache in VRAM: [24 layers * 1024 seq * 128 kv_dim] floats (~12.5 MB each)
    size_t kv_cache_size = (size_t)QWEN2_NUM_LAYERS * 1024 * (QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) * sizeof(float);
    m_buf_k_cache.allocate(m_ctx, kv_cache_size, storage_usage, vram_flags);
    m_buf_v_cache.allocate(m_ctx, kv_cache_size, storage_usage, vram_flags);

    // GPU Argmax Buffers
    m_buf_arg_part_vals.allocate(m_ctx, 128 * sizeof(float), storage_usage, vram_flags);
    m_buf_arg_part_ids.allocate(m_ctx, 128 * sizeof(uint32_t), storage_usage, vram_flags);
    m_buf_arg_token_result.allocate(m_ctx, 2 * sizeof(uint32_t), storage_usage, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_buf_arg_token_result.memory, 0, 2 * sizeof(uint32_t), 0, &m_buf_arg_token_result.mapped_ptr);

    reset_kv_cache();

    std::cout << "[Vulkan Qwen2 Engine] Activations and KV Cache allocated.\n";
    return true;
}

void VulkanInferenceEngineQwen2::reset_kv_cache() {
    size_t kv_cache_size = (size_t)QWEN2_NUM_LAYERS * 1024 * (QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) * sizeof(float);
    std::vector<float> zeros(kv_cache_size / sizeof(float), 0.0f);
    m_buf_k_cache.upload_staging(m_ctx, zeros.data(), kv_cache_size);
    m_buf_v_cache.upload_staging(m_ctx, zeros.data(), kv_cache_size);
}

bool VulkanInferenceEngineQwen2::upload_model_weights(const void *raw_mmap_base, const Qwen2Offsets &offsets) {
    const uint8_t *base = (const uint8_t*)raw_mmap_base + QWEN2_DATA_START_OFFSET;
    m_raw_embed_base = base + offsets.embed_tokens;

    std::cout << "[Vulkan Qwen2 Engine] Uploading weights to AMD RX 580 VRAM...\n";

    VkBufferUsageFlags w_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags vram_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // 1. Final Output Norm
    m_w_final_norm.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), w_usage, vram_flags);
    m_w_final_norm.upload_staging(m_ctx, base + offsets.final_norm, QWEN2_HIDDEN_SIZE * sizeof(float));

    // 2. LM Head (output.weight): [151936, 28] Q8 blocks
    size_t lm_head_blocks = (size_t)QWEN2_VOCAB_SIZE * (QWEN2_HIDDEN_SIZE / 32);
    std::vector<BlockQ8Qwen2> unpacked_lm_head;
    unpack_gguf_to_vulkan_blocks(unpacked_lm_head, (const GGUFBlockQ8*)(base + offsets.lm_head), lm_head_blocks);

    m_w_lm_head.allocate(m_ctx, lm_head_blocks * sizeof(BlockQ8Qwen2), w_usage, vram_flags);
    m_w_lm_head.upload_staging(m_ctx, unpacked_lm_head.data(), lm_head_blocks * sizeof(BlockQ8Qwen2));

    // Static LM Head & Final Norm Descriptor Sets
    m_ds_final_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, m_w_final_norm.buffer, m_act_norm_x.buffer }, {});
    m_ds_lm_head    = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_norm_x.buffer, m_w_lm_head.buffer, m_act_logits.buffer }, {});

    // Residual Add Sets
    m_ds_res_attn = m_pipe_residual_add.create_descriptor_set(m_ctx, { m_act_x.buffer, m_act_attn_out.buffer }, {});
    m_ds_res_mlp  = m_pipe_residual_add.create_descriptor_set(m_ctx, { m_act_x.buffer, m_act_mlp_out.buffer }, {});

    // Argmax Sets
    m_ds_arg_p1 = m_pipe_argmax_p1.create_descriptor_set(m_ctx, { m_act_logits.buffer, m_buf_arg_part_vals.buffer, m_buf_arg_part_ids.buffer }, {});
    m_ds_arg_p2 = m_pipe_argmax_p2.create_descriptor_set(m_ctx, { m_buf_arg_part_vals.buffer, m_buf_arg_part_ids.buffer, m_buf_arg_token_result.buffer }, {});

    // 3. 24 Transformer Layers
    m_layers.resize(QWEN2_NUM_LAYERS);
    uint32_t hidden_blocks = QWEN2_HIDDEN_SIZE / 32;                  // 28
    uint32_t kv_blocks = (QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) / 32; // 4
    uint32_t mlp_blocks = QWEN2_INTERMEDIATE_SIZE / 32;               // 152

    std::vector<BlockQ8Qwen2> unpacked_buf;

    for (int l = 0; l < QWEN2_NUM_LAYERS; l++) {
        const auto &lo = offsets.layers[l];
        auto &lw = m_layers[l];

        // Input Norm
        lw.input_layernorm.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), w_usage, vram_flags);
        lw.input_layernorm.upload_staging(m_ctx, base + lo.input_layernorm, QWEN2_HIDDEN_SIZE * sizeof(float));

        // Q Proj & Bias
        size_t q_blocks = (size_t)QWEN2_HIDDEN_SIZE * hidden_blocks;
        unpack_gguf_to_vulkan_blocks(unpacked_buf, (const GGUFBlockQ8*)(base + lo.q_proj), q_blocks);
        lw.q_proj.allocate(m_ctx, q_blocks * sizeof(BlockQ8Qwen2), w_usage, vram_flags);
        lw.q_proj.upload_staging(m_ctx, unpacked_buf.data(), q_blocks * sizeof(BlockQ8Qwen2));

        lw.q_bias.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), w_usage, vram_flags);
        lw.q_bias.upload_staging(m_ctx, base + lo.q_bias, QWEN2_HIDDEN_SIZE * sizeof(float));

        // K Proj & Bias
        size_t k_blocks = (size_t)(QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) * hidden_blocks;
        unpack_gguf_to_vulkan_blocks(unpacked_buf, (const GGUFBlockQ8*)(base + lo.k_proj), k_blocks);
        lw.k_proj.allocate(m_ctx, k_blocks * sizeof(BlockQ8Qwen2), w_usage, vram_flags);
        lw.k_proj.upload_staging(m_ctx, unpacked_buf.data(), k_blocks * sizeof(BlockQ8Qwen2));

        lw.k_bias.allocate(m_ctx, (QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) * sizeof(float), w_usage, vram_flags);
        lw.k_bias.upload_staging(m_ctx, base + lo.k_bias, (QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) * sizeof(float));

        // V Proj & Bias
        size_t v_blocks = (size_t)(QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) * hidden_blocks;
        unpack_gguf_to_vulkan_blocks(unpacked_buf, (const GGUFBlockQ8*)(base + lo.v_proj), v_blocks);
        lw.v_proj.allocate(m_ctx, v_blocks * sizeof(BlockQ8Qwen2), w_usage, vram_flags);
        lw.v_proj.upload_staging(m_ctx, unpacked_buf.data(), v_blocks * sizeof(BlockQ8Qwen2));

        lw.v_bias.allocate(m_ctx, (QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) * sizeof(float), w_usage, vram_flags);
        lw.v_bias.upload_staging(m_ctx, base + lo.v_bias, (QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM) * sizeof(float));

        // O Proj
        size_t o_blocks = (size_t)QWEN2_HIDDEN_SIZE * hidden_blocks;
        unpack_gguf_to_vulkan_blocks(unpacked_buf, (const GGUFBlockQ8*)(base + lo.o_proj), o_blocks);
        lw.o_proj.allocate(m_ctx, o_blocks * sizeof(BlockQ8Qwen2), w_usage, vram_flags);
        lw.o_proj.upload_staging(m_ctx, unpacked_buf.data(), o_blocks * sizeof(BlockQ8Qwen2));

        // Post Attention Norm
        lw.post_attn_layernorm.allocate(m_ctx, QWEN2_HIDDEN_SIZE * sizeof(float), w_usage, vram_flags);
        lw.post_attn_layernorm.upload_staging(m_ctx, base + lo.post_attn_layernorm, QWEN2_HIDDEN_SIZE * sizeof(float));

        // Gate Proj
        size_t gate_blocks = (size_t)QWEN2_INTERMEDIATE_SIZE * hidden_blocks;
        unpack_gguf_to_vulkan_blocks(unpacked_buf, (const GGUFBlockQ8*)(base + lo.gate_proj), gate_blocks);
        lw.gate_proj.allocate(m_ctx, gate_blocks * sizeof(BlockQ8Qwen2), w_usage, vram_flags);
        lw.gate_proj.upload_staging(m_ctx, unpacked_buf.data(), gate_blocks * sizeof(BlockQ8Qwen2));

        // Up Proj
        size_t up_blocks = (size_t)QWEN2_INTERMEDIATE_SIZE * hidden_blocks;
        unpack_gguf_to_vulkan_blocks(unpacked_buf, (const GGUFBlockQ8*)(base + lo.up_proj), up_blocks);
        lw.up_proj.allocate(m_ctx, up_blocks * sizeof(BlockQ8Qwen2), w_usage, vram_flags);
        lw.up_proj.upload_staging(m_ctx, unpacked_buf.data(), up_blocks * sizeof(BlockQ8Qwen2));

        // Down Proj
        size_t down_blocks = (size_t)QWEN2_HIDDEN_SIZE * mlp_blocks;
        unpack_gguf_to_vulkan_blocks(unpacked_buf, (const GGUFBlockQ8*)(base + lo.down_proj), down_blocks);
        lw.down_proj.allocate(m_ctx, down_blocks * sizeof(BlockQ8Qwen2), w_usage, vram_flags);
        lw.down_proj.upload_staging(m_ctx, unpacked_buf.data(), down_blocks * sizeof(BlockQ8Qwen2));

        // Static Descriptor Sets for Layer l
        lw.ds_input_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, lw.input_layernorm.buffer, m_act_norm_x.buffer }, {});

        lw.ds_fused_qkv = m_pipe_fused_qkv.create_descriptor_set(m_ctx, {
            m_act_norm_x.buffer, lw.q_proj.buffer, lw.k_proj.buffer, lw.v_proj.buffer,
            m_act_q.buffer, m_act_k.buffer, m_act_v.buffer,
            lw.q_bias.buffer, lw.k_bias.buffer, lw.v_bias.buffer
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

        if ((l + 1) % 6 == 0 || l == QWEN2_NUM_LAYERS - 1) {
            std::cout << "  [Layer " << (l + 1) << "/" << QWEN2_NUM_LAYERS << "] Uploaded and Descriptors Bound.\n";
        }
    }

    std::cout << "[Vulkan Qwen2 Engine] Weights uploaded and descriptor sets bound successfully.\n";

    // Pre-record command buffers
    record_forward_command_buffer();
    return true;
}

void VulkanInferenceEngineQwen2::record_forward_command_buffer() {
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    for (int mode = 0; mode < 2; mode++) {
        VkCommandBuffer cb = (mode == 0) ? m_cmd_buffer : m_cmd_buffer_argmax;

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        g_pfn_vkBeginCommandBuffer(cb, &begin_info);

        for (int l = 0; l < QWEN2_NUM_LAYERS; l++) {
            const auto &ly = m_layers[l];

            // 1. Input RMSNorm
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &ly.ds_input_norm, 0, nullptr);
            struct { uint32_t size; float eps; } norm_pc = { QWEN2_HIDDEN_SIZE, QWEN2_RMS_NORM_EPS };
            g_pfn_vkCmdPushConstants(cb, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(norm_pc), &norm_pc);
            g_pfn_vkCmdDispatch(cb, 1, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 2. Fused QKV GEMV (with Bias)
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_qkv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_qkv.pipeline_layout, 0, 1, &ly.ds_fused_qkv, 0, nullptr);
            struct { uint32_t nq, nk, nv, cols; } qkv_pc = {
                QWEN2_HIDDEN_SIZE,
                QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM,
                QWEN2_NUM_KV_HEADS * QWEN2_HEAD_DIM,
                QWEN2_HIDDEN_SIZE
            };
            g_pfn_vkCmdPushConstants(cb, m_pipe_fused_qkv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(qkv_pc), &qkv_pc);
            g_pfn_vkCmdDispatch(cb, qkv_pc.nq + qkv_pc.nk + qkv_pc.nv, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 3. RoPE & KV Cache
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rope_cache.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rope_cache.pipeline_layout, 0, 1, &ly.ds_rope_cache, 0, nullptr);
            struct {
                uint32_t layer_idx;
                uint32_t max_seq_len;
                uint32_t num_q_heads;
                uint32_t num_kv_heads;
                uint32_t head_dim;
                float theta;
            } rope_pc = {
                (uint32_t)l, 1024, QWEN2_NUM_HEADS, QWEN2_NUM_KV_HEADS, QWEN2_HEAD_DIM, QWEN2_ROPE_THETA
            };
            g_pfn_vkCmdPushConstants(cb, m_pipe_rope_cache.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rope_pc), &rope_pc);
            uint32_t total_pairs = (QWEN2_NUM_HEADS > QWEN2_NUM_KV_HEADS ? QWEN2_NUM_HEADS : QWEN2_NUM_KV_HEADS) * (QWEN2_HEAD_DIM / 2);
            g_pfn_vkCmdDispatch(cb, (total_pairs + 63) / 64, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 4. Attention
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_attention.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_attention.pipeline_layout, 0, 1, &ly.ds_attention, 0, nullptr);
            struct {
                uint32_t layer_idx;
                uint32_t max_seq_len;
                uint32_t num_q_heads;
                uint32_t num_kv_heads;
                uint32_t head_dim;
                float inv_sqrt_d;
            } attn_pc = {
                (uint32_t)l, 1024, QWEN2_NUM_HEADS, QWEN2_NUM_KV_HEADS, QWEN2_HEAD_DIM, 1.0f / sqrtf((float)QWEN2_HEAD_DIM)
            };
            g_pfn_vkCmdPushConstants(cb, m_pipe_attention.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(attn_pc), &attn_pc);
            g_pfn_vkCmdDispatch(cb, QWEN2_NUM_HEADS, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 5. O Projection
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout, 0, 1, &ly.ds_o_proj, 0, nullptr);
            struct { uint32_t rows, cols; } o_pc = { QWEN2_HIDDEN_SIZE, QWEN2_HIDDEN_SIZE };
            g_pfn_vkCmdPushConstants(cb, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(o_pc), &o_pc);
            g_pfn_vkCmdDispatch(cb, QWEN2_HIDDEN_SIZE, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 6. Residual Add (x += attn_out)
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline_layout, 0, 1, &m_ds_res_attn, 0, nullptr);
            uint32_t res_size = QWEN2_HIDDEN_SIZE;
            g_pfn_vkCmdPushConstants(cb, m_pipe_residual_add.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(res_size), &res_size);
            g_pfn_vkCmdDispatch(cb, (QWEN2_HIDDEN_SIZE + 255) / 256, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 7. Post-Attention RMSNorm
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &ly.ds_post_norm, 0, nullptr);
            g_pfn_vkCmdPushConstants(cb, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(norm_pc), &norm_pc);
            g_pfn_vkCmdDispatch(cb, 1, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 8. Fused SwiGLU GEMV
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_swiglu.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_swiglu.pipeline_layout, 0, 1, &ly.ds_fused_swiglu, 0, nullptr);
            struct { uint32_t rows, cols; } swiglu_pc = { QWEN2_INTERMEDIATE_SIZE, QWEN2_HIDDEN_SIZE };
            g_pfn_vkCmdPushConstants(cb, m_pipe_fused_swiglu.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(swiglu_pc), &swiglu_pc);
            g_pfn_vkCmdDispatch(cb, QWEN2_INTERMEDIATE_SIZE, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 9. Down Projection
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout, 0, 1, &ly.ds_down_proj, 0, nullptr);
            struct { uint32_t rows, cols; } down_pc = { QWEN2_HIDDEN_SIZE, QWEN2_INTERMEDIATE_SIZE };
            g_pfn_vkCmdPushConstants(cb, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(down_pc), &down_pc);
            g_pfn_vkCmdDispatch(cb, QWEN2_HIDDEN_SIZE, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 10. Residual Add (x += mlp_out)
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline_layout, 0, 1, &m_ds_res_mlp, 0, nullptr);
            g_pfn_vkCmdPushConstants(cb, m_pipe_residual_add.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(res_size), &res_size);
            g_pfn_vkCmdDispatch(cb, (QWEN2_HIDDEN_SIZE + 255) / 256, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // Final RMSNorm
        g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
        g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &m_ds_final_norm, 0, nullptr);
        struct { uint32_t size; float eps; } fnorm_pc = { QWEN2_HIDDEN_SIZE, QWEN2_RMS_NORM_EPS };
        g_pfn_vkCmdPushConstants(cb, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fnorm_pc), &fnorm_pc);
        g_pfn_vkCmdDispatch(cb, 1, 1, 1);

        g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // LM Head GEMV
        g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
        g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout, 0, 1, &m_ds_lm_head, 0, nullptr);
        struct { uint32_t rows, cols; } lm_pc = { QWEN2_VOCAB_SIZE, QWEN2_HIDDEN_SIZE };
        g_pfn_vkCmdPushConstants(cb, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lm_pc), &lm_pc);
        g_pfn_vkCmdDispatch(cb, QWEN2_VOCAB_SIZE, 1, 1);

        if (mode == 1) {
            // GPU Argmax Pass 1 & 2
            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p1.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p1.pipeline_layout, 0, 1, &m_ds_arg_p1, 0, nullptr);
            uint32_t vocab_size = QWEN2_VOCAB_SIZE;
            g_pfn_vkCmdPushConstants(cb, m_pipe_argmax_p1.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(vocab_size), &vocab_size);
            g_pfn_vkCmdDispatch(cb, 128, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p2.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p2.pipeline_layout, 0, 1, &m_ds_arg_p2, 0, nullptr);
            uint32_t num_partials = 128;
            g_pfn_vkCmdPushConstants(cb, m_pipe_argmax_p2.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(num_partials), &num_partials);
            g_pfn_vkCmdDispatch(cb, 1, 1, 1);
        }

        g_pfn_vkEndCommandBuffer(cb);
    }
}

bool VulkanInferenceEngineQwen2::forward_token(int token_id, int pos, float *out_logits) {
    if (!m_initialized) return false;

    // 1. Dequantize token embedding directly to m_act_x in host-visible VRAM
    uint32_t blocks_per_row = QWEN2_HIDDEN_SIZE / 32;
    const GGUFBlockQ8 *row = (const GGUFBlockQ8*)(m_raw_embed_base + (size_t)token_id * blocks_per_row * sizeof(GGUFBlockQ8));
    float *x_dst = (float*)m_act_x.mapped_ptr;
    for (uint32_t b = 0; b < blocks_per_row; b++) {
        float scale = fp16_to_fp32(row[b].d);
        for (int i = 0; i < 32; i++) {
            x_dst[b * 32 + i] = float(row[b].qs[i]) * scale;
        }
    }

    // 2. Set current sequence position
    *(uint32_t*)m_buf_pos.mapped_ptr = (uint32_t)pos;

    // 3. Submit pre-recorded forward pass
    g_pfn_vkResetFences(m_ctx.device, 1, &m_fence);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_cmd_buffer;

    VkResult res = g_pfn_vkQueueSubmit(m_ctx.compute_queue, 1, &submit_info, m_fence);
    if (res != VK_SUCCESS) return false;

    res = g_pfn_vkWaitForFences(m_ctx.device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) return false;

    if (out_logits) {
        memcpy(out_logits, m_act_logits.mapped_ptr, QWEN2_VOCAB_SIZE * sizeof(float));
    }
    return true;
}

int VulkanInferenceEngineQwen2::forward_token_gpu_argmax(int token_id, int pos) {
    if (!m_initialized) return -1;

    // 1. Dequantize token embedding directly to m_act_x
    uint32_t blocks_per_row = QWEN2_HIDDEN_SIZE / 32;
    const GGUFBlockQ8 *row = (const GGUFBlockQ8*)(m_raw_embed_base + (size_t)token_id * blocks_per_row * sizeof(GGUFBlockQ8));
    float *x_dst = (float*)m_act_x.mapped_ptr;
    for (uint32_t b = 0; b < blocks_per_row; b++) {
        float scale = fp16_to_fp32(row[b].d);
        for (int i = 0; i < 32; i++) {
            x_dst[b * 32 + i] = float(row[b].qs[i]) * scale;
        }
    }

    // 2. Set position
    *(uint32_t*)m_buf_pos.mapped_ptr = (uint32_t)pos;

    // 3. Submit pre-recorded forward pass with fused GPU Argmax
    g_pfn_vkResetFences(m_ctx.device, 1, &m_fence);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_cmd_buffer_argmax;

    VkResult res = g_pfn_vkQueueSubmit(m_ctx.compute_queue, 1, &submit_info, m_fence);
    if (res != VK_SUCCESS) return -1;

    res = g_pfn_vkWaitForFences(m_ctx.device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) return -1;

    // 4. Read back single 4-byte winner token ID directly
    uint32_t winner_id = *(const uint32_t*)m_buf_arg_token_result.mapped_ptr;
    return (int)winner_id;
}

void VulkanInferenceEngineQwen2::cleanup() {
    if (m_ctx.device != VK_NULL_HANDLE) {
        g_pfn_vkQueueWaitIdle(m_ctx.compute_queue);

        if (m_fence != VK_NULL_HANDLE) {
            g_pfn_vkDestroyFence(m_ctx.device, m_fence, nullptr);
            m_fence = VK_NULL_HANDLE;
        }

        m_act_x.cleanup(m_ctx);
        m_act_norm_x.cleanup(m_ctx);
        m_act_q.cleanup(m_ctx);
        m_act_k.cleanup(m_ctx);
        m_act_v.cleanup(m_ctx);
        m_act_head_outputs.cleanup(m_ctx);
        m_act_attn_out.cleanup(m_ctx);
        m_act_post_norm_x.cleanup(m_ctx);
        m_act_mlp_hidden.cleanup(m_ctx);
        m_act_mlp_out.cleanup(m_ctx);
        m_act_logits.cleanup(m_ctx);
        m_buf_pos.cleanup(m_ctx);

        m_buf_arg_part_vals.cleanup(m_ctx);
        m_buf_arg_part_ids.cleanup(m_ctx);
        m_buf_arg_token_result.cleanup(m_ctx);

        m_buf_k_cache.cleanup(m_ctx);
        m_buf_v_cache.cleanup(m_ctx);

        m_w_lm_head.cleanup(m_ctx);
        m_w_final_norm.cleanup(m_ctx);

        for (auto &ly : m_layers) {
            ly.input_layernorm.cleanup(m_ctx);
            ly.q_proj.cleanup(m_ctx);
            ly.q_bias.cleanup(m_ctx);
            ly.k_proj.cleanup(m_ctx);
            ly.k_bias.cleanup(m_ctx);
            ly.v_proj.cleanup(m_ctx);
            ly.v_bias.cleanup(m_ctx);
            ly.o_proj.cleanup(m_ctx);
            ly.post_attn_layernorm.cleanup(m_ctx);
            ly.gate_proj.cleanup(m_ctx);
            ly.up_proj.cleanup(m_ctx);
            ly.down_proj.cleanup(m_ctx);
        }
        m_layers.clear();

        m_pipe_rmsnorm.cleanup(m_ctx);
        m_pipe_fused_qkv.cleanup(m_ctx);
        m_pipe_rope_cache.cleanup(m_ctx);
        m_pipe_attention.cleanup(m_ctx);
        m_pipe_q8_gemv.cleanup(m_ctx);
        m_pipe_fused_swiglu.cleanup(m_ctx);
        m_pipe_residual_add.cleanup(m_ctx);
        m_pipe_argmax_p1.cleanup(m_ctx);
        m_pipe_argmax_p2.cleanup(m_ctx);

        m_ctx.cleanup();
    }
    m_initialized = false;
}
