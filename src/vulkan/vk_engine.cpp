#include "vk_engine.h"
#include "shaders/compiled_shaders.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>

static inline float bf16_to_fp32(uint16_t b) {
    union { uint32_t u; float f; } val;
    val.u = ((uint32_t)b) << 16;
    return val.f;
}


static void quantize_bf16_to_q8_blocks(std::vector<BlockQ8> &out_blocks, const uint16_t *src, int rows, int cols) {
    int num_blocks = cols / 32;
    out_blocks.resize((size_t)rows * num_blocks);

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint16_t *src_row = src + (int64_t)r * cols;
        BlockQ8 *out_row = out_blocks.data() + r * num_blocks;

        for (int b = 0; b < num_blocks; b++) {
            const uint16_t *b_src = src_row + b * 32;
            float max_val = 0.0f;
            float f_vals[32];

            for (int i = 0; i < 32; i++) {
                float v = std::fabs(bf16_to_fp32(b_src[i]));
                f_vals[i] = bf16_to_fp32(b_src[i]);
                if (v > max_val) max_val = v;
            }

            float scale = max_val / 127.0f;
            out_row[b].scale = scale;
            float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;

            for (int i = 0; i < 32; i++) {
                int q = (int)std::round(f_vals[i] * inv_scale);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                out_row[b].qs[i] = (int8_t)q;
            }
        }
    }
}

VulkanInferenceEngine::VulkanInferenceEngine() {
}

VulkanInferenceEngine::~VulkanInferenceEngine() {
    cleanup();
}

bool VulkanInferenceEngine::init() {
    if (!m_ctx.init()) {
        std::cerr << "[Vulkan Engine] Context initialization failed.\n";
        return false;
    }

    if (!init_pipelines()) {
        std::cerr << "[Vulkan Engine] Pipeline creation failed.\n";
        return false;
    }

    if (!allocate_buffers()) {
        std::cerr << "[Vulkan Engine] Buffer allocation failed.\n";
        return false;
    }

    // Allocate persistent command buffer & fence
    VkCommandBufferAllocateInfo cmd_alloc = {};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = m_ctx.command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;

    VkResult res = g_pfn_vkAllocateCommandBuffers(m_ctx.device, &cmd_alloc, &m_cmd_buffer);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Engine] vkAllocateCommandBuffers failed: " << res << "\n";
        return false;
    }

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    g_pfn_vkCreateFence(m_ctx.device, &fence_info, NULL, &m_fence);

    m_initialized = true;
    std::cout << "[Vulkan Engine] Engine initialized successfully on " << m_ctx.device_name << "!\n";
    return true;
}

bool VulkanInferenceEngine::init_pipelines() {
    // 1. Q8 GEMV (push consts: uint rows, uint cols = 8 bytes; 3 buffers: in, w, out)
    if (!m_pipe_q8_gemv.init_from_spv(m_ctx, spv_q8_gemv_data, spv_q8_gemv_size, 3, 8)) {
        return false;
    }

    // 2. RMSNorm (push consts: uint size, float eps = 8 bytes; 3 buffers: in, w, out)
    if (!m_pipe_rmsnorm.init_from_spv(m_ctx, spv_rmsnorm_data, spv_rmsnorm_size, 3, 8)) {
        return false;
    }

    // 3. RoPE & Cache (push consts: uint layer_idx, uint max_seq_len = 8 bytes; 6 buffers: q, k, v, k_cache, v_cache, pos)
    if (!m_pipe_rope_cache.init_from_spv(m_ctx, spv_rope_and_cache_data, spv_rope_and_cache_size, 6, 8)) {
        return false;
    }

    // 4. Attention (push consts: uint layer_idx, uint max_seq_len = 8 bytes; 5 buffers: q, k_cache, v_cache, out, pos)
    if (!m_pipe_attention.init_from_spv(m_ctx, spv_attention_data, spv_attention_size, 5, 8)) {
        return false;
    }

    // 5. SwiGLU (push consts: uint size = 4 bytes; 3 buffers: gate, up, out)
    if (!m_pipe_swiglu.init_from_spv(m_ctx, spv_swiglu_data, spv_swiglu_size, 3, 4)) {
        return false;
    }

    // 6. Residual Add (push consts: uint size = 4 bytes; 2 buffers: x, delta)
    if (!m_pipe_residual_add.init_from_spv(m_ctx, spv_residual_add_data, spv_residual_add_size, 2, 4)) {
        return false;
    }

    // 7. Fused QKV GEMV (push consts: 16 bytes; 7 buffers: in, wq, wk, wv, out_q, out_k, out_v)
    if (!m_pipe_fused_qkv.init_from_spv(m_ctx, spv_fused_qkv_gemv_data, spv_fused_qkv_gemv_size, 7, 16)) {
        return false;
    }

    // 8. Fused SwiGLU GEMV (push consts: 8 bytes; 4 buffers: in, w_gate, w_up, out_activated)
    if (!m_pipe_fused_swiglu.init_from_spv(m_ctx, spv_fused_swiglu_gemv_data, spv_fused_swiglu_gemv_size, 4, 8)) {
        return false;
    }

    std::cout << "[Vulkan Engine] All Compute Pipelines (including Fused QKV & SwiGLU) compiled and loaded.\n";
    return true;
}

bool VulkanInferenceEngine::allocate_buffers() {
    // 1. Allocate Intermediate Activations in VRAM
    VkBufferUsageFlags storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags vram_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    m_act_x.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_act_x.memory, 0, HIDDEN_SIZE * sizeof(float), 0, &m_act_x.mapped_ptr);

    m_act_norm_x.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_q.allocate(m_ctx, NUM_HEADS * HEAD_DIM * sizeof(float), storage_usage, vram_flags);
    m_act_k.allocate(m_ctx, NUM_KV_HEADS * HEAD_DIM * sizeof(float), storage_usage, vram_flags);
    m_act_v.allocate(m_ctx, NUM_KV_HEADS * HEAD_DIM * sizeof(float), storage_usage, vram_flags);
    m_act_head_outputs.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_attn_out.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_post_norm_x.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_gate.allocate(m_ctx, INTERMEDIATE_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_up.allocate(m_ctx, INTERMEDIATE_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_mlp_hidden.allocate(m_ctx, INTERMEDIATE_SIZE * sizeof(float), storage_usage, vram_flags);
    m_act_mlp_out.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), storage_usage, vram_flags);

    m_act_logits.allocate(m_ctx, VOCAB_SIZE * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_act_logits.memory, 0, VOCAB_SIZE * sizeof(float), 0, &m_act_logits.mapped_ptr);

    // Sequence position buffer for dynamic tracking without command re-recording
    m_buf_pos.allocate(m_ctx, sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_buf_pos.memory, 0, sizeof(uint32_t), 0, &m_buf_pos.mapped_ptr);
    *(uint32_t*)m_buf_pos.mapped_ptr = 0;

    // 2. Allocate KV Cache in VRAM (32 layers * 1024 seq * 320 floats = ~40 MB each)
    size_t kv_cache_bytes = (size_t)NUM_LAYERS * 1024 * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    m_buf_k_cache.allocate(m_ctx, kv_cache_bytes, storage_usage, vram_flags);
    m_buf_v_cache.allocate(m_ctx, kv_cache_bytes, storage_usage, vram_flags);

    // 3. Pre-create Static Descriptor Sets for shared intermediate activations
    // RoPE & Cache (includes m_buf_pos)
    std::vector<VkBuffer> rope_bufs = { m_act_q.buffer, m_act_k.buffer, m_act_v.buffer, m_buf_k_cache.buffer, m_buf_v_cache.buffer, m_buf_pos.buffer };
    m_ds_rope_cache = m_pipe_rope_cache.create_descriptor_set(m_ctx, rope_bufs, {});

    // Attention (includes m_buf_pos)
    std::vector<VkBuffer> attn_bufs = { m_act_q.buffer, m_buf_k_cache.buffer, m_buf_v_cache.buffer, m_act_head_outputs.buffer, m_buf_pos.buffer };
    m_ds_attention = m_pipe_attention.create_descriptor_set(m_ctx, attn_bufs, {});

    // SwiGLU
    std::vector<VkBuffer> swiglu_bufs = { m_act_gate.buffer, m_act_up.buffer, m_act_mlp_hidden.buffer };
    m_ds_swiglu = m_pipe_swiglu.create_descriptor_set(m_ctx, swiglu_bufs, {});

    // Residual additions
    std::vector<VkBuffer> res_attn_bufs = { m_act_x.buffer, m_act_attn_out.buffer };
    m_ds_res_attn = m_pipe_residual_add.create_descriptor_set(m_ctx, res_attn_bufs, {});

    std::vector<VkBuffer> res_mlp_bufs = { m_act_x.buffer, m_act_mlp_out.buffer };
    m_ds_res_mlp = m_pipe_residual_add.create_descriptor_set(m_ctx, res_mlp_bufs, {});

    return true;
}

bool VulkanInferenceEngine::upload_model_weights(const void *raw_mmap_base, const ModelOffsets &offsets) {
    std::cout << "[Vulkan Engine] Quantizing and uploading SmolLM2-360M weights to RX 580 VRAM...\n";
    const uint8_t *base = (const uint8_t*)raw_mmap_base + DATA_BASE_OFFSET;

    // 1. Host Embed Table for rapid token embedding
    const uint16_t *embed_src = (const uint16_t*)(base + offsets.embed_tokens);
    m_host_embed_table.resize((size_t)VOCAB_SIZE * HIDDEN_SIZE);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < VOCAB_SIZE * HIDDEN_SIZE; i++) {
        m_host_embed_table[i] = bf16_to_fp32(embed_src[i]);
    }

    // 2. LM Head (Tied to embed_tokens): Quantize to Q8 and upload to VRAM
    std::vector<BlockQ8> q8_embed;
    quantize_bf16_to_q8_blocks(q8_embed, embed_src, VOCAB_SIZE, HIDDEN_SIZE);
    size_t embed_bytes = q8_embed.size() * sizeof(BlockQ8);
    m_w_embed_tokens.allocate(m_ctx, embed_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_w_embed_tokens.upload_staging(m_ctx, q8_embed.data(), embed_bytes);

    // LM Head Descriptor Set: (in: norm_x, w: embed_tokens, out: logits)
    std::vector<VkBuffer> lm_head_bufs = { m_act_norm_x.buffer, m_w_embed_tokens.buffer, m_act_logits.buffer };
    m_ds_lm_head = m_pipe_q8_gemv.create_descriptor_set(m_ctx, lm_head_bufs, {});

    // 3. Final Layernorm
    const uint16_t *final_norm_src = (const uint16_t*)(base + offsets.final_norm);
    std::vector<float> final_norm_fp32(HIDDEN_SIZE);
    for (int i = 0; i < HIDDEN_SIZE; i++) final_norm_fp32[i] = bf16_to_fp32(final_norm_src[i]);
    m_w_final_norm.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_w_final_norm.upload_staging(m_ctx, final_norm_fp32.data(), HIDDEN_SIZE * sizeof(float));

    // Final Norm Descriptor Set: (in: x, w: final_norm, out: norm_x)
    std::vector<VkBuffer> final_norm_bufs = { m_act_x.buffer, m_w_final_norm.buffer, m_act_norm_x.buffer };
    m_ds_final_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, final_norm_bufs, {});

    // 4. Layers (32 Layers)
    m_layers.resize(NUM_LAYERS);
    VkBufferUsageFlags w_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags vram_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    for (int l = 0; l < NUM_LAYERS; l++) {
        const LayerOffsets &lo = offsets.layers[l];
        VulkanLayerWeights &lw = m_layers[l];

        // Layernorms
        const uint16_t *in_norm_src = (const uint16_t*)(base + lo.input_layernorm);
        const uint16_t *post_norm_src = (const uint16_t*)(base + lo.post_attn_layernorm);
        std::vector<float> in_norm_fp32(HIDDEN_SIZE), post_norm_fp32(HIDDEN_SIZE);
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            in_norm_fp32[i] = bf16_to_fp32(in_norm_src[i]);
            post_norm_fp32[i] = bf16_to_fp32(post_norm_src[i]);
        }
        lw.input_layernorm.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), w_usage, vram_flags);
        lw.input_layernorm.upload_staging(m_ctx, in_norm_fp32.data(), HIDDEN_SIZE * sizeof(float));

        lw.post_attn_layernorm.allocate(m_ctx, HIDDEN_SIZE * sizeof(float), w_usage, vram_flags);
        lw.post_attn_layernorm.upload_staging(m_ctx, post_norm_fp32.data(), HIDDEN_SIZE * sizeof(float));

        // Quantize Projections
        std::vector<BlockQ8> q_blocks, k_blocks, v_blocks, o_blocks, gate_blocks, up_blocks, down_blocks;
        quantize_bf16_to_q8_blocks(q_blocks, (const uint16_t*)(base + lo.q_proj), NUM_HEADS * HEAD_DIM, HIDDEN_SIZE);
        quantize_bf16_to_q8_blocks(k_blocks, (const uint16_t*)(base + lo.k_proj), NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE);
        quantize_bf16_to_q8_blocks(v_blocks, (const uint16_t*)(base + lo.v_proj), NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE);
        quantize_bf16_to_q8_blocks(o_blocks, (const uint16_t*)(base + lo.o_proj), HIDDEN_SIZE, HIDDEN_SIZE);
        quantize_bf16_to_q8_blocks(gate_blocks, (const uint16_t*)(base + lo.gate_proj), INTERMEDIATE_SIZE, HIDDEN_SIZE);
        quantize_bf16_to_q8_blocks(up_blocks,   (const uint16_t*)(base + lo.up_proj),   INTERMEDIATE_SIZE, HIDDEN_SIZE);
        quantize_bf16_to_q8_blocks(down_blocks, (const uint16_t*)(base + lo.down_proj), HIDDEN_SIZE, INTERMEDIATE_SIZE);

        lw.q_proj.allocate(m_ctx, q_blocks.size() * sizeof(BlockQ8), w_usage, vram_flags);
        lw.q_proj.upload_staging(m_ctx, q_blocks.data(), q_blocks.size() * sizeof(BlockQ8));

        lw.k_proj.allocate(m_ctx, k_blocks.size() * sizeof(BlockQ8), w_usage, vram_flags);
        lw.k_proj.upload_staging(m_ctx, k_blocks.data(), k_blocks.size() * sizeof(BlockQ8));

        lw.v_proj.allocate(m_ctx, v_blocks.size() * sizeof(BlockQ8), w_usage, vram_flags);
        lw.v_proj.upload_staging(m_ctx, v_blocks.data(), v_blocks.size() * sizeof(BlockQ8));

        lw.o_proj.allocate(m_ctx, o_blocks.size() * sizeof(BlockQ8), w_usage, vram_flags);
        lw.o_proj.upload_staging(m_ctx, o_blocks.data(), o_blocks.size() * sizeof(BlockQ8));

        lw.gate_proj.allocate(m_ctx, gate_blocks.size() * sizeof(BlockQ8), w_usage, vram_flags);
        lw.gate_proj.upload_staging(m_ctx, gate_blocks.data(), gate_blocks.size() * sizeof(BlockQ8));

        lw.up_proj.allocate(m_ctx, up_blocks.size() * sizeof(BlockQ8), w_usage, vram_flags);
        lw.up_proj.upload_staging(m_ctx, up_blocks.data(), up_blocks.size() * sizeof(BlockQ8));

        lw.down_proj.allocate(m_ctx, down_blocks.size() * sizeof(BlockQ8), w_usage, vram_flags);
        lw.down_proj.upload_staging(m_ctx, down_blocks.data(), down_blocks.size() * sizeof(BlockQ8));

        // Create Static Descriptor Sets for this layer
        lw.ds_input_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, lw.input_layernorm.buffer, m_act_norm_x.buffer }, {});
        lw.ds_q_proj = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_norm_x.buffer, lw.q_proj.buffer, m_act_q.buffer }, {});
        lw.ds_k_proj = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_norm_x.buffer, lw.k_proj.buffer, m_act_k.buffer }, {});
        lw.ds_v_proj = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_norm_x.buffer, lw.v_proj.buffer, m_act_v.buffer }, {});

        lw.ds_o_proj = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_head_outputs.buffer, lw.o_proj.buffer, m_act_attn_out.buffer }, {});
        lw.ds_post_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, lw.post_attn_layernorm.buffer, m_act_post_norm_x.buffer }, {});

        lw.ds_gate_proj = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_post_norm_x.buffer, lw.gate_proj.buffer, m_act_gate.buffer }, {});
        lw.ds_up_proj   = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_post_norm_x.buffer, lw.up_proj.buffer,   m_act_up.buffer }, {});
        lw.ds_down_proj = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_mlp_hidden.buffer, lw.down_proj.buffer, m_act_mlp_out.buffer }, {});

        // Fused Descriptors
        lw.ds_fused_qkv = m_pipe_fused_qkv.create_descriptor_set(m_ctx, {
            m_act_norm_x.buffer, lw.q_proj.buffer, lw.k_proj.buffer, lw.v_proj.buffer,
            m_act_q.buffer, m_act_k.buffer, m_act_v.buffer
        }, {});
        lw.ds_fused_swiglu = m_pipe_fused_swiglu.create_descriptor_set(m_ctx, {
            m_act_post_norm_x.buffer, lw.gate_proj.buffer, lw.up_proj.buffer, m_act_mlp_hidden.buffer
        }, {});
    }

    std::cout << "[Vulkan Engine] All 32 Layers & LM Head resident in RX 580 VRAM (~407 MB)!\n";
    std::cout << "[Vulkan Engine] Pre-recording static forward pass command buffer on RX 580...\n";
    record_forward_command_buffer();
    std::cout << "[Vulkan Engine] Static forward command buffer pre-recorded successfully!\n";
    return true;
}

void VulkanInferenceEngine::reset_kv_cache() {
    // KV Cache reset is handled automatically by indexing pos from 0
}

void VulkanInferenceEngine::record_forward_command_buffer() {
    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    g_pfn_vkBeginCommandBuffer(m_cmd_buffer, &begin_info);

    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    auto insert_barrier = [&]() {
        g_pfn_vkCmdPipelineBarrier(m_cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    };

    struct PushConstsGEMV { uint32_t rows; uint32_t cols; };
    struct PushConstsNorm { uint32_t size; float eps; };
    struct PushConstsSeq { uint32_t layer_idx; uint32_t max_seq_len; };
    struct PushConstsSize { uint32_t size; };

    PushConstsNorm pc_norm = { HIDDEN_SIZE, RMS_NORM_EPS };
    PushConstsGEMV pc_q = { NUM_HEADS * HEAD_DIM, HIDDEN_SIZE };
    PushConstsGEMV pc_k = { NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE };
    PushConstsGEMV pc_v = { NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE };
    PushConstsGEMV pc_o = { HIDDEN_SIZE, NUM_HEADS * HEAD_DIM };
    PushConstsGEMV pc_gate_up = { INTERMEDIATE_SIZE, HIDDEN_SIZE };
    PushConstsGEMV pc_down = { HIDDEN_SIZE, INTERMEDIATE_SIZE };
    PushConstsSize pc_swiglu = { INTERMEDIATE_SIZE };
    PushConstsSize pc_res_x = { HIDDEN_SIZE };

    // 32 Transformer Layers
    for (uint32_t l = 0; l < (uint32_t)NUM_LAYERS; l++) {
        VulkanLayerWeights &lw = m_layers[l];
        PushConstsSeq pc_seq = { l, 1024 };

        // 1. Input RMSNorm
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout,
                                      0, 1, &lw.ds_input_norm, 0, NULL);
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_norm), &pc_norm);
        g_pfn_vkCmdDispatch(m_cmd_buffer, 1, 1, 1);
        insert_barrier();

        // 2. Fused Q, K, V Projection
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_qkv.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_qkv.pipeline_layout,
                                      0, 1, &lw.ds_fused_qkv, 0, NULL);
        struct { uint32_t num_q; uint32_t num_k; uint32_t num_v; uint32_t cols; } pc_qkv = {
            (uint32_t)HIDDEN_SIZE, (uint32_t)(NUM_KV_HEADS * HEAD_DIM), (uint32_t)(NUM_KV_HEADS * HEAD_DIM), (uint32_t)HIDDEN_SIZE
        };
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_fused_qkv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_qkv), &pc_qkv);
        uint32_t total_qkv = pc_qkv.num_q + pc_qkv.num_k + pc_qkv.num_v;
        g_pfn_vkCmdDispatch(m_cmd_buffer, (total_qkv + 63) / 64, 1, 1);
        insert_barrier();

        // 3. RoPE & KV Cache update
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rope_cache.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rope_cache.pipeline_layout,
                                      0, 1, &m_ds_rope_cache, 0, NULL);
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_rope_cache.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_seq), &pc_seq);
        g_pfn_vkCmdDispatch(m_cmd_buffer, 8, 1, 1);
        insert_barrier();

        // 4. Attention
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_attention.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_attention.pipeline_layout,
                                      0, 1, &m_ds_attention, 0, NULL);
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_attention.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_seq), &pc_seq);
        g_pfn_vkCmdDispatch(m_cmd_buffer, 15, 1, 1); // 15 workgroups for 15 heads
        insert_barrier();

        // 5. O Proj
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout,
                                      0, 1, &lw.ds_o_proj, 0, NULL);
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_o), &pc_o);
        g_pfn_vkCmdDispatch(m_cmd_buffer, (pc_o.rows + 63) / 64, 1, 1);
        insert_barrier();

        // 6. Residual Add (x += attn_out)
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline_layout,
                                      0, 1, &m_ds_res_attn, 0, NULL);
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_residual_add.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_res_x), &pc_res_x);
        g_pfn_vkCmdDispatch(m_cmd_buffer, (HIDDEN_SIZE + 255) / 256, 1, 1);
        insert_barrier();

        // 7. Post Layernorm
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout,
                                      0, 1, &lw.ds_post_norm, 0, NULL);
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_norm), &pc_norm);
        g_pfn_vkCmdDispatch(m_cmd_buffer, 1, 1, 1);
        insert_barrier();

        // 8. Fused SwiGLU (Gate + Up + SiLU directly into mlp_hidden)
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_swiglu.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_fused_swiglu.pipeline_layout,
                                      0, 1, &lw.ds_fused_swiglu, 0, NULL);
        struct { uint32_t rows; uint32_t cols; } pc_fused_swiglu = { (uint32_t)INTERMEDIATE_SIZE, (uint32_t)HIDDEN_SIZE };
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_fused_swiglu.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_fused_swiglu), &pc_fused_swiglu);
        g_pfn_vkCmdDispatch(m_cmd_buffer, (INTERMEDIATE_SIZE + 63) / 64, 1, 1);
        insert_barrier();

        // 10. Down Proj
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout,
                                      0, 1, &lw.ds_down_proj, 0, NULL);
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_down), &pc_down);
        g_pfn_vkCmdDispatch(m_cmd_buffer, (pc_down.rows + 63) / 64, 1, 1);
        insert_barrier();

        // 11. Residual Add (x += mlp_out)
        g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline);
        g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_residual_add.pipeline_layout,
                                      0, 1, &m_ds_res_mlp, 0, NULL);
        g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_residual_add.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(pc_res_x), &pc_res_x);
        g_pfn_vkCmdDispatch(m_cmd_buffer, (HIDDEN_SIZE + 255) / 256, 1, 1);
        insert_barrier();
    }

    // Final Norm
    g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
    g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout,
                                  0, 1, &m_ds_final_norm, 0, NULL);
    g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                             0, sizeof(pc_norm), &pc_norm);
    g_pfn_vkCmdDispatch(m_cmd_buffer, 1, 1, 1);
    insert_barrier();

    // LM Head (tied to embed_tokens: 49152 x 960)
    PushConstsGEMV pc_lm_head = { (uint32_t)VOCAB_SIZE, (uint32_t)HIDDEN_SIZE };
    g_pfn_vkCmdBindPipeline(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
    g_pfn_vkCmdBindDescriptorSets(m_cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout,
                                  0, 1, &m_ds_lm_head, 0, NULL);
    g_pfn_vkCmdPushConstants(m_cmd_buffer, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                             0, sizeof(pc_lm_head), &pc_lm_head);
    g_pfn_vkCmdDispatch(m_cmd_buffer, (VOCAB_SIZE + 63) / 64, 1, 1);

    g_pfn_vkEndCommandBuffer(m_cmd_buffer);
}

bool VulkanInferenceEngine::forward_token(int token_id, int pos, float *out_logits) {
    if (!m_initialized) return false;

    // 1. Host Lookup of token embedding vector x [960]
    const float *tok_embed = m_host_embed_table.data() + (size_t)token_id * HIDDEN_SIZE;
    memcpy(m_act_x.mapped_ptr, tok_embed, HIDDEN_SIZE * sizeof(float));

    // 2. Set dynamic sequence position in mapped buffer (Zero command recording overhead!)
    *(uint32_t*)m_buf_pos.mapped_ptr = (uint32_t)pos;

    // 3. Dispatch Pre-recorded Static Command Buffer on AMD RX 580
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_cmd_buffer;

    g_pfn_vkResetFences(m_ctx.device, 1, &m_fence);
    g_pfn_vkQueueSubmit(m_ctx.compute_queue, 1, &submit_info, m_fence);
    g_pfn_vkWaitForFences(m_ctx.device, 1, &m_fence, VK_TRUE, UINT64_MAX);

    // 4. Direct Read Logits [VOCAB_SIZE] from mapped buffer to Host
    memcpy(out_logits, m_act_logits.mapped_ptr, VOCAB_SIZE * sizeof(float));
    return true;
}

void VulkanInferenceEngine::cleanup() {
    if (!m_initialized) return;

    if (m_fence) {
        g_pfn_vkDestroyFence(m_ctx.device, m_fence, NULL);
        m_fence = VK_NULL_HANDLE;
    }
    if (m_cmd_buffer) {
        g_pfn_vkFreeCommandBuffers(m_ctx.device, m_ctx.command_pool, 1, &m_cmd_buffer);
        m_cmd_buffer = VK_NULL_HANDLE;
    }

    m_act_x.cleanup(m_ctx);
    m_act_norm_x.cleanup(m_ctx);
    m_act_q.cleanup(m_ctx);
    m_act_k.cleanup(m_ctx);
    m_act_v.cleanup(m_ctx);
    m_act_head_outputs.cleanup(m_ctx);
    m_act_attn_out.cleanup(m_ctx);
    m_act_post_norm_x.cleanup(m_ctx);
    m_act_gate.cleanup(m_ctx);
    m_act_up.cleanup(m_ctx);
    m_act_mlp_hidden.cleanup(m_ctx);
    m_act_mlp_out.cleanup(m_ctx);
    m_act_logits.cleanup(m_ctx);
    m_buf_pos.cleanup(m_ctx);

    m_buf_k_cache.cleanup(m_ctx);
    m_buf_v_cache.cleanup(m_ctx);

    m_w_embed_tokens.cleanup(m_ctx);
    m_w_final_norm.cleanup(m_ctx);

    for (auto &lw : m_layers) {
        lw.input_layernorm.cleanup(m_ctx);
        lw.q_proj.cleanup(m_ctx);
        lw.k_proj.cleanup(m_ctx);
        lw.v_proj.cleanup(m_ctx);
        lw.o_proj.cleanup(m_ctx);
        lw.post_attn_layernorm.cleanup(m_ctx);
        lw.gate_proj.cleanup(m_ctx);
        lw.up_proj.cleanup(m_ctx);
        lw.down_proj.cleanup(m_ctx);
    }
    m_layers.clear();

    m_pipe_q8_gemv.cleanup(m_ctx);
    m_pipe_rmsnorm.cleanup(m_ctx);
    m_pipe_rope_cache.cleanup(m_ctx);
    m_pipe_attention.cleanup(m_ctx);
    m_pipe_swiglu.cleanup(m_ctx);
    m_pipe_residual_add.cleanup(m_ctx);
    m_pipe_fused_qkv.cleanup(m_ctx);
    m_pipe_fused_swiglu.cleanup(m_ctx);

    m_ctx.cleanup();
    m_initialized = false;
}
