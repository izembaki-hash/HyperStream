#include "vk_engine_qwen2_7b.h"
#include "shaders/compiled_qwen2_shaders.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <immintrin.h>

#define QK4_0 32
#define QK_K 256
#define QK8_0 32

typedef uint16_t ggml_fp16_t;

#pragma pack(push, 1)
struct GGUFBlockQ4_0 {
    ggml_fp16_t d; // fp16
    uint8_t qs[16];
};

struct GGUFBlockQ6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    ggml_fp16_t d;
};
#pragma pack(pop)

struct uvec4 {
    uint32_t x, y, z, w;
};

static inline float fp16_to_fp32(ggml_fp16_t h) {
    __m128i vh = _mm_cvtsi32_si128((int)h);
    __m128 vf = _mm_cvtph_ps(vh);
    return _mm_cvtss_f32(vf);
}

static inline ggml_fp16_t fp32_to_fp16(float f) {
    __m128 vf = _mm_set_ss(f);
    __m128i vh = _mm_cvtps_ph(vf, 0);
    return (ggml_fp16_t)_mm_cvtsi128_si32(vh);
}

// Convert GGUF Q4_0 blocks to Planar (uvec4 quants + packed uint scales)
static void split_q4_planar(const GGUFBlockQ4_0 *src, size_t num_blocks,
                            std::vector<uvec4> &out_quants,
                            std::vector<uint32_t> &out_scales) {
    out_quants.resize(num_blocks);
    out_scales.resize((num_blocks + 1) / 2);

    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < num_blocks; b++) {
        memcpy(&out_quants[b], src[b].qs, 16);
    }

    size_t num_pairs = (num_blocks + 1) / 2;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_pairs; i++) {
        uint16_t d0 = src[i * 2].d;
        uint16_t d1 = (i * 2 + 1 < num_blocks) ? src[i * 2 + 1].d : 0;
        out_scales[i] = ((uint32_t)d1 << 16) | (uint32_t)d0;
    }
}

// Dequantize Q6_K row directly to Q8_0 blocks
static void dequantize_q6_k_to_q8_0_row(const GGUFBlockQ6_K *x, BlockQ8Qwen2_7B *y_q8, int k) {
    int nb = k / QK_K;
    float row_f[QK_K];

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;

        float *y = row_f;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0]  = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }

        // Quantize row_f (256 floats = 8 blocks of 32) into y_q8
        BlockQ8Qwen2_7B *out_block = y_q8 + (i * 8);
        for (int b = 0; b < 8; b++) {
            const float *src = row_f + b * 32;
            float amax = 0.0f;
            for (int j = 0; j < 32; j++) {
                float v = fabsf(src[j]);
                if (v > amax) amax = v;
            }
            float d_scale = amax / 127.0f;
            float id_scale = (d_scale > 0.0f) ? (1.0f / d_scale) : 0.0f;
            out_block[b].scale = d_scale;
            for (int j = 0; j < 32; j++) {
                int q = (int)roundf(src[j] * id_scale);
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                out_block[b].qs[j] = (int8_t)q;
            }
        }
    }
}

VulkanInferenceEngineQwen2_7B::VulkanInferenceEngineQwen2_7B() {
}

VulkanInferenceEngineQwen2_7B::~VulkanInferenceEngineQwen2_7B() {
    cleanup();
}

bool VulkanInferenceEngineQwen2_7B::init(const hyperstream::ModelTopology &topo) {
    if (topo.hidden_size > 0) {
        m_topo = topo;
    } else {
        // Fallback default for 7B
        m_topo.num_layers = 28;
        m_topo.hidden_size = 3584;
        m_topo.intermediate_size = 18944;
        m_topo.num_heads = 28;
        m_topo.num_kv_heads = 4;
        m_topo.head_dim = 128;
        m_topo.vocab_size = 152064;
        m_topo.rms_norm_eps = 1e-6f;
        m_topo.rope_freq_base = 1000000.0f;
    }

    if (!m_ctx.init()) {
        std::cerr << "[Vulkan Qwen2 7B Engine] Context initialization failed.\n";
        return false;
    }

    if (!init_pipelines()) {
        std::cerr << "[Vulkan Qwen2 7B Engine] Pipeline creation failed.\n";
        return false;
    }

    if (!allocate_buffers()) {
        std::cerr << "[Vulkan Qwen2 7B Engine] Buffer allocation failed.\n";
        return false;
    }

    // Allocate persistent command buffers & fence
    VkCommandBufferAllocateInfo cmd_alloc = {};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = m_ctx.command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 2;

    VkCommandBuffer cmds[2];
    VkResult res = g_pfn_vkAllocateCommandBuffers(m_ctx.device, &cmd_alloc, cmds);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Qwen2 7B Engine] Failed to allocate command buffer: " << res << "\n";
        return false;
    }
    m_cmd_buffer = cmds[0];
    m_cmd_buffer_argmax = cmds[1];

    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    res = g_pfn_vkCreateFence(m_ctx.device, &fence_info, nullptr, &m_fence);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Qwen2 7B Engine] Failed to create fence: " << res << "\n";
        return false;
    }

    m_initialized = true;
    std::cout << "[Vulkan Qwen2 7B Engine] Initialized successfully on " << m_ctx.device_name << "\n";
    return true;
}

bool VulkanInferenceEngineQwen2_7B::init_pipelines() {
    hyperstream::SpecializationInfoHelper spec_helper = m_topo.create_specialization_info(1024);
    const VkSpecializationInfo *spec_info = spec_helper.get();

    // 1. RMSNorm (push consts: 8 bytes; 3 buffers: in, w, out)
    if (!m_pipe_rmsnorm.init_from_spv(m_ctx, spv_qwen2_rmsnorm_data, spv_qwen2_rmsnorm_size, 3, 8, spec_info)) {
        return false;
    }

    // 2. Q4 Fused QKV GEMV (push consts: 16 bytes; 7 buffers: in, quants, scales, out_q, out_k, out_v, bias)
    if (!m_pipe_q4_fused_qkv.init_from_spv(m_ctx, spv_qwen2_q4_fused_qkv_data, spv_qwen2_q4_fused_qkv_size, 7, 16, spec_info)) {
        return false;
    }

    // 3. RoPE & KV Cache (push consts: 24 bytes; 6 buffers: q, k, v, k_cache, v_cache, pos)
    if (!m_pipe_rope_cache.init_from_spv(m_ctx, spv_qwen2_rope_cache_data, spv_qwen2_rope_cache_size, 6, 24, spec_info)) {
        return false;
    }

    // 4. Attention (push consts: 24 bytes; 5 buffers: q, k_cache, v_cache, out, pos)
    if (!m_pipe_attention.init_from_spv(m_ctx, spv_qwen2_attention_data, spv_qwen2_attention_size, 5, 24, spec_info)) {
        return false;
    }

    // 5. Q4 GEMV with optional residual add (push consts: 12 bytes; 4 buffers: in, quants, scales, out)
    if (!m_pipe_q4_gemv.init_from_spv(m_ctx, spv_qwen2_q4_gemv_data, spv_qwen2_q4_gemv_size, 4, 12, spec_info)) {
        return false;
    }

    // 6. Q4 Fused SwiGLU GEMV (push consts: 8 bytes; 6 buffers: in, q_gate, s_gate, q_up, s_up, out)
    if (!m_pipe_q4_fused_swiglu.init_from_spv(m_ctx, spv_qwen2_q4_fused_swiglu_data, spv_qwen2_q4_fused_swiglu_size, 6, 8, spec_info)) {
        return false;
    }

    // 7. Q8 GEMV (for LM Head; push consts: 8 bytes; 3 buffers: in, w, out)
    if (!m_pipe_q8_gemv.init_from_spv(m_ctx, spv_qwen2_q8_gemv_data, spv_qwen2_q8_gemv_size, 3, 8, spec_info)) {
        return false;
    }

    // 8. Argmax Pass 1 (push consts: 4 bytes; 3 buffers: in_logits, part_vals, part_ids)
    if (!m_pipe_argmax_p1.init_from_spv(m_ctx, spv_qwen2_argmax_pass1_data, spv_qwen2_argmax_pass1_size, 3, 4, spec_info)) {
        return false;
    }

    // 9. Argmax Pass 2 (push consts: 4 bytes; 3 buffers: part_vals, part_ids, result_id)
    if (!m_pipe_argmax_p2.init_from_spv(m_ctx, spv_qwen2_argmax_pass2_data, spv_qwen2_argmax_pass2_size, 3, 4, spec_info)) {
        return false;
    }

    std::cout << "[Vulkan Qwen2 7B Engine] All Compute Pipelines compiled and specialized with VkSpecializationInfo.\n";
    return true;
}

bool VulkanInferenceEngineQwen2_7B::allocate_buffers() {
    VkBufferUsageFlags storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkMemoryPropertyFlags vram_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // Intermediate activations
    m_act_x.allocate(m_ctx, m_topo.hidden_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_act_x.memory, 0, m_topo.hidden_size * sizeof(float), 0, &m_act_x.mapped_ptr);

    m_act_norm_x.allocate(m_ctx, m_topo.hidden_size * sizeof(float), storage_usage, vram_flags);
    m_act_q.allocate(m_ctx, m_topo.hidden_size * sizeof(float), storage_usage, vram_flags);
    m_act_k.allocate(m_ctx, m_topo.num_kv_heads * m_topo.head_dim * sizeof(float), storage_usage, vram_flags);
    m_act_v.allocate(m_ctx, m_topo.num_kv_heads * m_topo.head_dim * sizeof(float), storage_usage, vram_flags);
    m_act_head_outputs.allocate(m_ctx, m_topo.hidden_size * sizeof(float), storage_usage, vram_flags);
    m_act_attn_out.allocate(m_ctx, m_topo.hidden_size * sizeof(float), storage_usage, vram_flags);
    m_act_post_norm_x.allocate(m_ctx, m_topo.hidden_size * sizeof(float), storage_usage, vram_flags);
    m_act_mlp_hidden.allocate(m_ctx, m_topo.intermediate_size * sizeof(float), storage_usage, vram_flags);
    m_act_mlp_out.allocate(m_ctx, m_topo.hidden_size * sizeof(float), storage_usage, vram_flags);

    // Host-visible Logits buffer
    m_act_logits.allocate(m_ctx, m_topo.vocab_size * sizeof(float), storage_usage, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_act_logits.memory, 0, m_topo.vocab_size * sizeof(float), 0, &m_act_logits.mapped_ptr);

    // Sequence position buffer
    m_buf_pos.allocate(m_ctx, sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_buf_pos.memory, 0, sizeof(uint32_t), 0, &m_buf_pos.mapped_ptr);

    // KV Cache in VRAM: [layers * 1024 seq * kv_dim] floats
    size_t kv_cache_size = (size_t)m_topo.num_layers * 1024 * (m_topo.num_kv_heads * m_topo.head_dim) * sizeof(float);
    m_buf_k_cache.allocate(m_ctx, kv_cache_size, storage_usage, vram_flags);
    m_buf_v_cache.allocate(m_ctx, kv_cache_size, storage_usage, vram_flags);

    // GPU Argmax Buffers
    m_buf_arg_part_vals.allocate(m_ctx, 128 * sizeof(float), storage_usage, vram_flags);
    m_buf_arg_part_ids.allocate(m_ctx, 128 * sizeof(uint32_t), storage_usage, vram_flags);
    m_buf_arg_token_result.allocate(m_ctx, 2 * sizeof(uint32_t), storage_usage, host_flags);
    g_pfn_vkMapMemory(m_ctx.device, m_buf_arg_token_result.memory, 0, 2 * sizeof(uint32_t), 0, &m_buf_arg_token_result.mapped_ptr);

    reset_kv_cache();

    std::cout << "[Vulkan Qwen2 7B Engine] Activations and KV Cache allocated.\n";
    return true;
}

void VulkanInferenceEngineQwen2_7B::reset_kv_cache() {
    size_t kv_cache_size = (size_t)m_topo.num_layers * 1024 * (m_topo.num_kv_heads * m_topo.head_dim) * sizeof(float);
    std::vector<float> zeros(kv_cache_size / sizeof(float), 0.0f);
    m_buf_k_cache.upload_staging(m_ctx, zeros.data(), kv_cache_size);
    m_buf_v_cache.upload_staging(m_ctx, zeros.data(), kv_cache_size);
}

bool VulkanInferenceEngineQwen2_7B::upload_model_weights(const void *raw_mmap_base, const hyperstream::ModelTopology &topo) {
    m_topo = topo;
    const uint8_t *base = (const uint8_t*)raw_mmap_base + m_topo.data_start_offset;
    m_raw_embed_base = base + m_topo.token_embd_offset;

    std::cout << "[Vulkan Qwen2 7B Engine] Uploading weights to AMD RX 580 VRAM (Q4_0 Planar + Q8_0 LM Head)...\n";

    VkBufferUsageFlags w_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags vram_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    // 1. Final Output Norm
    m_w_final_norm.allocate(m_ctx, m_topo.hidden_size * sizeof(float), w_usage, vram_flags);
    m_w_final_norm.upload_staging(m_ctx, base + m_topo.output_norm_offset, m_topo.hidden_size * sizeof(float));

    // 2. LM Head (output.weight): [vocab_size, hidden_size] Q6_K -> converted on CPU to Q8_0
    size_t total_q8_blocks = (size_t)m_topo.vocab_size * (m_topo.hidden_size / QK8_0);
    std::vector<BlockQ8Qwen2_7B> q8_lm_head(total_q8_blocks);
    const GGUFBlockQ6_K *q6_base = (const GGUFBlockQ6_K*)(base + m_topo.output_weight_offset);
    int q6_k_blocks_per_row = m_topo.hidden_size / QK_K;

    std::cout << "  Dequantizing LM Head (Q6_K -> Q8_0) across " << m_topo.vocab_size << " rows...\n";
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < (int)m_topo.vocab_size; r++) {
        dequantize_q6_k_to_q8_0_row(q6_base + (size_t)r * q6_k_blocks_per_row,
                                    q8_lm_head.data() + (size_t)r * (m_topo.hidden_size / QK8_0),
                                    m_topo.hidden_size);
    }
    std::cout << "  LM Head dequantization done! Uploading to VRAM...\n";

    m_w_lm_head.allocate(m_ctx, total_q8_blocks * sizeof(BlockQ8Qwen2_7B), w_usage, vram_flags);
    m_w_lm_head.upload_staging(m_ctx, q8_lm_head.data(), total_q8_blocks * sizeof(BlockQ8Qwen2_7B));

    // Static LM Head & Final Norm Descriptor Sets
    m_ds_final_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, m_w_final_norm.buffer, m_act_norm_x.buffer }, {});
    m_ds_lm_head    = m_pipe_q8_gemv.create_descriptor_set(m_ctx, { m_act_norm_x.buffer, m_w_lm_head.buffer, m_act_logits.buffer }, {});

    // Argmax Sets
    m_ds_arg_p1 = m_pipe_argmax_p1.create_descriptor_set(m_ctx, { m_act_logits.buffer, m_buf_arg_part_vals.buffer, m_buf_arg_part_ids.buffer }, {});
    m_ds_arg_p2 = m_pipe_argmax_p2.create_descriptor_set(m_ctx, { m_buf_arg_part_vals.buffer, m_buf_arg_part_ids.buffer, m_buf_arg_token_result.buffer }, {});

    // 3. Transformer Layers
    m_layers.resize(m_topo.num_layers);
    uint32_t hidden_blocks = m_topo.hidden_size / 32;
    uint32_t kv_blocks = (m_topo.num_kv_heads * m_topo.head_dim) / 32;
    uint32_t mlp_blocks = m_topo.intermediate_size / 32;

    std::vector<uvec4> quants_buf;
    std::vector<uint32_t> scales_buf;

    // Fast Staging Buffer: 140 MB reusable host-visible buffer for batched layer uploads
    const size_t STAGING_CAP = 140 * 1024 * 1024;
    VulkanBuffer staging_buf;
    if (!staging_buf.allocate(m_ctx, STAGING_CAP, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        std::cerr << "Failed to allocate 140 MB staging buffer!\n";
        return false;
    }
    void *staging_mapped = nullptr;
    g_pfn_vkMapMemory(m_ctx.device, staging_buf.memory, 0, STAGING_CAP, 0, &staging_mapped);

    VkCommandBuffer transfer_cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cmd_alloc = {};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = m_ctx.command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    g_pfn_vkAllocateCommandBuffers(m_ctx.device, &cmd_alloc, &transfer_cmd);

    VkFence transfer_fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    g_pfn_vkCreateFence(m_ctx.device, &fence_info, NULL, &transfer_fence);

    for (uint32_t l = 0; l < m_topo.num_layers; l++) {
        const auto &lo = m_topo.layers[l];
        auto &lw = m_layers[l];

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        g_pfn_vkBeginCommandBuffer(transfer_cmd, &begin_info);

        size_t stage_offset = 0;
        auto stage_and_record = [&](VulkanBuffer &dst_buf, const void *src_data, size_t sz) {
            stage_offset = (stage_offset + 255) & ~255; // 256-byte align
            memcpy((uint8_t*)staging_mapped + stage_offset, src_data, sz);
            VkBufferCopy region = {};
            region.srcOffset = stage_offset;
            region.dstOffset = 0;
            region.size = sz;
            g_pfn_vkCmdCopyBuffer(transfer_cmd, staging_buf.buffer, dst_buf.buffer, 1, &region);
            stage_offset += sz;
        };

        // Input Norm
        lw.input_layernorm.allocate(m_ctx, m_topo.hidden_size * sizeof(float), w_usage, vram_flags);
        stage_and_record(lw.input_layernorm, base + lo.attn_norm, m_topo.hidden_size * sizeof(float));

        // Fused QKV (concat Q, K, V rows into a single planar matrix)
        uint32_t total_qkv_rows = m_topo.hidden_size + (m_topo.num_kv_heads * m_topo.head_dim) * 2;
        size_t total_qkv_blocks = (size_t)total_qkv_rows * hidden_blocks;
        std::vector<GGUFBlockQ4_0> fused_qkv_blocks(total_qkv_blocks);

        // Copy Q
        size_t q_blocks = (size_t)m_topo.hidden_size * hidden_blocks;
        memcpy(fused_qkv_blocks.data(), base + lo.q_weight, q_blocks * sizeof(GGUFBlockQ4_0));
        // Copy K
        size_t k_blocks = (size_t)(m_topo.num_kv_heads * m_topo.head_dim) * hidden_blocks;
        memcpy(fused_qkv_blocks.data() + q_blocks, base + lo.k_weight, k_blocks * sizeof(GGUFBlockQ4_0));
        // Copy V
        size_t v_blocks = k_blocks;
        memcpy(fused_qkv_blocks.data() + q_blocks + k_blocks, base + lo.v_weight, v_blocks * sizeof(GGUFBlockQ4_0));

        // Split to planar
        split_q4_planar(fused_qkv_blocks.data(), total_qkv_blocks, quants_buf, scales_buf);

        lw.qkv_quants.allocate(m_ctx, quants_buf.size() * sizeof(uvec4), w_usage, vram_flags);
        stage_and_record(lw.qkv_quants, quants_buf.data(), quants_buf.size() * sizeof(uvec4));

        lw.qkv_scales.allocate(m_ctx, scales_buf.size() * sizeof(uint32_t), w_usage, vram_flags);
        stage_and_record(lw.qkv_scales, scales_buf.data(), scales_buf.size() * sizeof(uint32_t));

        // Concat QKV Biases
        std::vector<float> fused_bias(total_qkv_rows);
        memcpy(fused_bias.data(), base + lo.q_bias, m_topo.hidden_size * sizeof(float));
        memcpy(fused_bias.data() + m_topo.hidden_size, base + lo.k_bias, (m_topo.num_kv_heads * m_topo.head_dim) * sizeof(float));
        memcpy(fused_bias.data() + m_topo.hidden_size + (m_topo.num_kv_heads * m_topo.head_dim), base + lo.v_bias, (m_topo.num_kv_heads * m_topo.head_dim) * sizeof(float));

        lw.qkv_bias.allocate(m_ctx, total_qkv_rows * sizeof(float), w_usage, vram_flags);
        stage_and_record(lw.qkv_bias, fused_bias.data(), total_qkv_rows * sizeof(float));

        // O Proj
        size_t o_blocks = (size_t)m_topo.hidden_size * hidden_blocks;
        split_q4_planar((const GGUFBlockQ4_0*)(base + lo.o_weight), o_blocks, quants_buf, scales_buf);

        lw.o_proj_quants.allocate(m_ctx, quants_buf.size() * sizeof(uvec4), w_usage, vram_flags);
        stage_and_record(lw.o_proj_quants, quants_buf.data(), quants_buf.size() * sizeof(uvec4));

        lw.o_proj_scales.allocate(m_ctx, scales_buf.size() * sizeof(uint32_t), w_usage, vram_flags);
        stage_and_record(lw.o_proj_scales, scales_buf.data(), scales_buf.size() * sizeof(uint32_t));

        // Post Attention Norm
        lw.post_attn_layernorm.allocate(m_ctx, m_topo.hidden_size * sizeof(float), w_usage, vram_flags);
        stage_and_record(lw.post_attn_layernorm, base + lo.ffn_norm, m_topo.hidden_size * sizeof(float));

        // Gate Proj
        size_t gate_blocks = (size_t)m_topo.intermediate_size * hidden_blocks;
        split_q4_planar((const GGUFBlockQ4_0*)(base + lo.gate_weight), gate_blocks, quants_buf, scales_buf);

        lw.gate_quants.allocate(m_ctx, quants_buf.size() * sizeof(uvec4), w_usage, vram_flags);
        stage_and_record(lw.gate_quants, quants_buf.data(), quants_buf.size() * sizeof(uvec4));

        lw.gate_scales.allocate(m_ctx, scales_buf.size() * sizeof(uint32_t), w_usage, vram_flags);
        stage_and_record(lw.gate_scales, scales_buf.data(), scales_buf.size() * sizeof(uint32_t));

        // Up Proj
        size_t up_blocks = (size_t)m_topo.intermediate_size * hidden_blocks;
        split_q4_planar((const GGUFBlockQ4_0*)(base + lo.up_weight), up_blocks, quants_buf, scales_buf);

        lw.up_quants.allocate(m_ctx, quants_buf.size() * sizeof(uvec4), w_usage, vram_flags);
        stage_and_record(lw.up_quants, quants_buf.data(), quants_buf.size() * sizeof(uvec4));

        lw.up_scales.allocate(m_ctx, scales_buf.size() * sizeof(uint32_t), w_usage, vram_flags);
        stage_and_record(lw.up_scales, scales_buf.data(), scales_buf.size() * sizeof(uint32_t));

        // Down Proj
        size_t down_blocks = (size_t)m_topo.hidden_size * mlp_blocks;
        split_q4_planar((const GGUFBlockQ4_0*)(base + lo.down_weight), down_blocks, quants_buf, scales_buf);

        lw.down_quants.allocate(m_ctx, quants_buf.size() * sizeof(uvec4), w_usage, vram_flags);
        stage_and_record(lw.down_quants, quants_buf.data(), quants_buf.size() * sizeof(uvec4));

        lw.down_scales.allocate(m_ctx, scales_buf.size() * sizeof(uint32_t), w_usage, vram_flags);
        stage_and_record(lw.down_scales, scales_buf.data(), scales_buf.size() * sizeof(uint32_t));

        g_pfn_vkEndCommandBuffer(transfer_cmd);

        // Submit transfer command buffer for this entire layer
        VkSubmitInfo submit_info = {};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &transfer_cmd;
        g_pfn_vkResetFences(m_ctx.device, 1, &transfer_fence);
        g_pfn_vkQueueSubmit(m_ctx.compute_queue, 1, &submit_info, transfer_fence);
        g_pfn_vkWaitForFences(m_ctx.device, 1, &transfer_fence, VK_TRUE, UINT64_MAX);

        // Static Descriptor Sets for Layer l
        lw.ds_input_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, lw.input_layernorm.buffer, m_act_norm_x.buffer }, {});

        lw.ds_fused_qkv = m_pipe_q4_fused_qkv.create_descriptor_set(m_ctx, {
            m_act_norm_x.buffer, lw.qkv_quants.buffer, lw.qkv_scales.buffer,
            m_act_q.buffer, m_act_k.buffer, m_act_v.buffer, lw.qkv_bias.buffer
        }, {});

        lw.ds_rope_cache = m_pipe_rope_cache.create_descriptor_set(m_ctx, {
            m_act_q.buffer, m_act_k.buffer, m_act_v.buffer,
            m_buf_k_cache.buffer, m_buf_v_cache.buffer, m_buf_pos.buffer
        }, {});

        lw.ds_attention = m_pipe_attention.create_descriptor_set(m_ctx, {
            m_act_q.buffer, m_buf_k_cache.buffer, m_buf_v_cache.buffer,
            m_act_head_outputs.buffer, m_buf_pos.buffer
        }, {});

        lw.ds_o_proj = m_pipe_q4_gemv.create_descriptor_set(m_ctx, {
            m_act_head_outputs.buffer, lw.o_proj_quants.buffer, lw.o_proj_scales.buffer, m_act_x.buffer
        }, {});

        lw.ds_post_norm = m_pipe_rmsnorm.create_descriptor_set(m_ctx, { m_act_x.buffer, lw.post_attn_layernorm.buffer, m_act_post_norm_x.buffer }, {});

        lw.ds_fused_swiglu = m_pipe_q4_fused_swiglu.create_descriptor_set(m_ctx, {
            m_act_post_norm_x.buffer,
            lw.gate_quants.buffer, lw.gate_scales.buffer,
            lw.up_quants.buffer, lw.up_scales.buffer,
            m_act_mlp_hidden.buffer
        }, {});

        lw.ds_down_proj = m_pipe_q4_gemv.create_descriptor_set(m_ctx, {
            m_act_mlp_hidden.buffer, lw.down_quants.buffer, lw.down_scales.buffer, m_act_x.buffer
        }, {});

        if ((l + 1) % 4 == 0 || l == m_topo.num_layers - 1) {
            std::cout << "  [Layer " << (l + 1) << "/" << m_topo.num_layers << "] Uploaded and Descriptors Bound.\n";
        }
    }

    // Cleanup transfer staging resources
    g_pfn_vkUnmapMemory(m_ctx.device, staging_buf.memory);
    staging_buf.cleanup(m_ctx);
    g_pfn_vkDestroyFence(m_ctx.device, transfer_fence, NULL);
    g_pfn_vkFreeCommandBuffers(m_ctx.device, m_ctx.command_pool, 1, &transfer_cmd);

    std::cout << "[Vulkan Qwen2 7B Engine] Weights uploaded and descriptor sets bound successfully.\n";

    // Pre-record command buffers
    record_forward_command_buffer();
    return true;
}

void VulkanInferenceEngineQwen2_7B::record_forward_command_buffer() {
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    for (int mode = 0; mode < 2; mode++) {
        VkCommandBuffer cb = (mode == 0) ? m_cmd_buffer : m_cmd_buffer_argmax;

        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        g_pfn_vkBeginCommandBuffer(cb, &begin_info);

        for (uint32_t l = 0; l < m_topo.num_layers; l++) {
            const auto &ly = m_layers[l];

            // 1. Input RMSNorm (m_act_x -> m_act_norm_x)
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &ly.ds_input_norm, 0, nullptr);
            struct { uint32_t size; float eps; } norm_pc = { m_topo.hidden_size, m_topo.rms_norm_eps };
            g_pfn_vkCmdPushConstants(cb, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(norm_pc), &norm_pc);
            g_pfn_vkCmdDispatch(cb, 1, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 2. Q4 Fused QKV GEMV (m_act_norm_x -> m_act_q, m_act_k, m_act_v)
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q4_fused_qkv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q4_fused_qkv.pipeline_layout, 0, 1, &ly.ds_fused_qkv, 0, nullptr);
            struct { uint32_t nq, nk, nv, cols; } qkv_pc = {
                m_topo.hidden_size,
                m_topo.num_kv_heads * m_topo.head_dim,
                m_topo.num_kv_heads * m_topo.head_dim,
                m_topo.hidden_size
            };
            g_pfn_vkCmdPushConstants(cb, m_pipe_q4_fused_qkv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(qkv_pc), &qkv_pc);
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
                l, 1024, m_topo.num_heads, m_topo.num_kv_heads, m_topo.head_dim, m_topo.rope_freq_base
            };
            g_pfn_vkCmdPushConstants(cb, m_pipe_rope_cache.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rope_pc), &rope_pc);
            uint32_t max_heads = (m_topo.num_heads > m_topo.num_kv_heads ? m_topo.num_heads : m_topo.num_kv_heads);
            uint32_t total_pairs = max_heads * (m_topo.head_dim / 2);
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
                l, 1024, m_topo.num_heads, m_topo.num_kv_heads, m_topo.head_dim, 1.0f / sqrtf((float)m_topo.head_dim)
            };
            g_pfn_vkCmdPushConstants(cb, m_pipe_attention.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(attn_pc), &attn_pc);
            g_pfn_vkCmdDispatch(cb, m_topo.num_heads, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 5. O Projection
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q4_gemv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q4_gemv.pipeline_layout, 0, 1, &ly.ds_o_proj, 0, nullptr);
            struct { uint32_t rows, cols, add_residual; } o_pc = { m_topo.hidden_size, m_topo.hidden_size, 1u };
            g_pfn_vkCmdPushConstants(cb, m_pipe_q4_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(o_pc), &o_pc);
            g_pfn_vkCmdDispatch(cb, m_topo.hidden_size, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 6. Post-Attention RMSNorm
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &ly.ds_post_norm, 0, nullptr);
            g_pfn_vkCmdPushConstants(cb, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(norm_pc), &norm_pc);
            g_pfn_vkCmdDispatch(cb, 1, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 7. Fused SwiGLU GEMV
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q4_fused_swiglu.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q4_fused_swiglu.pipeline_layout, 0, 1, &ly.ds_fused_swiglu, 0, nullptr);
            struct { uint32_t rows, cols; } swiglu_pc = { m_topo.intermediate_size, m_topo.hidden_size };
            g_pfn_vkCmdPushConstants(cb, m_pipe_q4_fused_swiglu.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(swiglu_pc), &swiglu_pc);
            g_pfn_vkCmdDispatch(cb, m_topo.intermediate_size, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            // 8. Down Projection
            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q4_gemv.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q4_gemv.pipeline_layout, 0, 1, &ly.ds_down_proj, 0, nullptr);
            struct { uint32_t rows, cols, add_residual; } down_pc = { m_topo.hidden_size, m_topo.intermediate_size, 1u };
            g_pfn_vkCmdPushConstants(cb, m_pipe_q4_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(down_pc), &down_pc);
            g_pfn_vkCmdDispatch(cb, m_topo.hidden_size, 1, 1);

            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // Final RMSNorm
        g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline);
        g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_rmsnorm.pipeline_layout, 0, 1, &m_ds_final_norm, 0, nullptr);
        struct { uint32_t size; float eps; } fnorm_pc = { m_topo.hidden_size, m_topo.rms_norm_eps };
        g_pfn_vkCmdPushConstants(cb, m_pipe_rmsnorm.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fnorm_pc), &fnorm_pc);
        g_pfn_vkCmdDispatch(cb, 1, 1, 1);

        g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        // LM Head Q8 GEMV (m_act_norm_x -> m_act_logits)
        g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline);
        g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_q8_gemv.pipeline_layout, 0, 1, &m_ds_lm_head, 0, nullptr);
        struct { uint32_t rows, cols; } lm_pc = { m_topo.vocab_size, m_topo.hidden_size };
        g_pfn_vkCmdPushConstants(cb, m_pipe_q8_gemv.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lm_pc), &lm_pc);
        g_pfn_vkCmdDispatch(cb, m_topo.vocab_size, 1, 1);

        if (mode == 1) {
            // GPU Argmax Pass 1 & 2
            g_pfn_vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

            g_pfn_vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p1.pipeline);
            g_pfn_vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_argmax_p1.pipeline_layout, 0, 1, &m_ds_arg_p1, 0, nullptr);
            uint32_t vocab_size = m_topo.vocab_size;
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

bool VulkanInferenceEngineQwen2_7B::forward_token(int token_id, int pos, float *out_logits) {
    if (!m_initialized) return false;

    // 1. Dequantize Q4_0 token embedding directly to m_act_x in host-visible VRAM
    uint32_t blocks_per_row = m_topo.hidden_size / 32;
    const GGUFBlockQ4_0 *row = (const GGUFBlockQ4_0*)(m_raw_embed_base + (size_t)token_id * blocks_per_row * sizeof(GGUFBlockQ4_0));
    float *dst = (float*)m_act_x.mapped_ptr;

    for (uint32_t b = 0; b < blocks_per_row; b++) {
        float d = fp16_to_fp32(row[b].d);
        float *bdst = dst + b * 32;
        for (int j = 0; j < 16; j++) {
            uint8_t byte_val = row[b].qs[j];
            bdst[j]      = (float)((int)(byte_val & 0x0F) - 8) * d;
            bdst[j + 16] = (float)((int)(byte_val >> 4) - 8) * d;
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
        memcpy(out_logits, m_act_logits.mapped_ptr, m_topo.vocab_size * sizeof(float));
    }
    return true;
}

int VulkanInferenceEngineQwen2_7B::forward_token_gpu_argmax(int token_id, int pos) {
    if (!m_initialized) return -1;

    // 1. Dequantize Q4_0 token embedding directly to m_act_x in host-visible VRAM
    uint32_t blocks_per_row = m_topo.hidden_size / 32;
    const GGUFBlockQ4_0 *row = (const GGUFBlockQ4_0*)(m_raw_embed_base + (size_t)token_id * blocks_per_row * sizeof(GGUFBlockQ4_0));
    float *dst = (float*)m_act_x.mapped_ptr;

    for (uint32_t b = 0; b < blocks_per_row; b++) {
        float d = fp16_to_fp32(row[b].d);
        float *bdst = dst + b * 32;
        for (int j = 0; j < 16; j++) {
            uint8_t byte_val = row[b].qs[j];
            bdst[j]      = (float)((int)(byte_val & 0x0F) - 8) * d;
            bdst[j + 16] = (float)((int)(byte_val >> 4) - 8) * d;
        }
    }

    // 2. Set current sequence position
    *(uint32_t*)m_buf_pos.mapped_ptr = (uint32_t)pos;

    // 3. Submit pre-recorded forward pass with GPU Argmax
    g_pfn_vkResetFences(m_ctx.device, 1, &m_fence);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &m_cmd_buffer_argmax;

    VkResult res = g_pfn_vkQueueSubmit(m_ctx.compute_queue, 1, &submit_info, m_fence);
    if (res != VK_SUCCESS) return -1;

    res = g_pfn_vkWaitForFences(m_ctx.device, 1, &m_fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS) return -1;

    // 4. Read back single 4-byte token ID
    uint32_t winner_token = *(const uint32_t*)m_buf_arg_token_result.mapped_ptr;
    return (int)winner_token;
}

void VulkanInferenceEngineQwen2_7B::cleanup() {
    if (!m_initialized) return;

    if (m_ctx.compute_queue != VK_NULL_HANDLE) {
        g_pfn_vkQueueWaitIdle(m_ctx.compute_queue);
    }

    if (m_fence != VK_NULL_HANDLE) {
        g_pfn_vkDestroyFence(m_ctx.device, m_fence, nullptr);
        m_fence = VK_NULL_HANDLE;
    }

    // Clean layer buffers
    for (auto &l : m_layers) {
        l.input_layernorm.cleanup(m_ctx);
        l.qkv_quants.cleanup(m_ctx);
        l.qkv_scales.cleanup(m_ctx);
        l.qkv_bias.cleanup(m_ctx);
        l.o_proj_quants.cleanup(m_ctx);
        l.o_proj_scales.cleanup(m_ctx);
        l.post_attn_layernorm.cleanup(m_ctx);
        l.gate_quants.cleanup(m_ctx);
        l.gate_scales.cleanup(m_ctx);
        l.up_quants.cleanup(m_ctx);
        l.up_scales.cleanup(m_ctx);
        l.down_quants.cleanup(m_ctx);
        l.down_scales.cleanup(m_ctx);
    }
    m_layers.clear();

    m_w_lm_head.cleanup(m_ctx);
    m_w_final_norm.cleanup(m_ctx);

    m_buf_k_cache.cleanup(m_ctx);
    m_buf_v_cache.cleanup(m_ctx);

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

    m_pipe_q4_fused_qkv.cleanup(m_ctx);
    m_pipe_q4_gemv.cleanup(m_ctx);
    m_pipe_q4_fused_swiglu.cleanup(m_ctx);
    m_pipe_q8_gemv.cleanup(m_ctx);
    m_pipe_rmsnorm.cleanup(m_ctx);
    m_pipe_rope_cache.cleanup(m_ctx);
    m_pipe_attention.cleanup(m_ctx);
    m_pipe_argmax_p1.cleanup(m_ctx);
    m_pipe_argmax_p2.cleanup(m_ctx);

    m_ctx.cleanup();
    m_initialized = false;
}
