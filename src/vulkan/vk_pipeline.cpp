#include "vk_pipeline.h"
#include <iostream>

bool ComputePipeline::init_from_spv(const VulkanContext &ctx,
                                   const uint32_t *spv_code, size_t spv_size_bytes,
                                   uint32_t num_storage_buffers,
                                   uint32_t push_constant_size_bytes) {
    // 1. Create Shader Module
    VkShaderModuleCreateInfo sm_info = {};
    sm_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm_info.codeSize = spv_size_bytes;
    sm_info.pCode = spv_code;

    VkResult res = g_pfn_vkCreateShaderModule(ctx.device, &sm_info, NULL, &shader_module);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Pipeline] vkCreateShaderModule failed: " << res << "\n";
        return false;
    }

    // 2. Create Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_buffers);
    for (uint32_t i = 0; i < num_storage_buffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = NULL;
    }

    VkDescriptorSetLayoutCreateInfo dsl_info = {};
    dsl_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsl_info.bindingCount = num_storage_buffers;
    dsl_info.pBindings = bindings.data();

    res = g_pfn_vkCreateDescriptorSetLayout(ctx.device, &dsl_info, NULL, &desc_layout);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Pipeline] vkCreateDescriptorSetLayout failed: " << res << "\n";
        return false;
    }

    // 3. Create Pipeline Layout
    VkPushConstantRange pc_range = {};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = push_constant_size_bytes;

    VkPipelineLayoutCreateInfo pl_info = {};
    pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pl_info.setLayoutCount = 1;
    pl_info.pSetLayouts = &desc_layout;
    if (push_constant_size_bytes > 0) {
        pl_info.pushConstantRangeCount = 1;
        pl_info.pPushConstantRanges = &pc_range;
    }

    res = g_pfn_vkCreatePipelineLayout(ctx.device, &pl_info, NULL, &pipeline_layout);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Pipeline] vkCreatePipelineLayout failed: " << res << "\n";
        return false;
    }

    // 4. Create Compute Pipeline
    VkComputePipelineCreateInfo cp_info = {};
    cp_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp_info.stage.module = shader_module;
    cp_info.stage.pName = "main";
    cp_info.layout = pipeline_layout;

    res = g_pfn_vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cp_info, NULL, &pipeline);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Pipeline] vkCreateComputePipelines failed: " << res << "\n";
        return false;
    }

    // 5. Create Descriptor Pool
    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = num_storage_buffers * 2048;

    VkDescriptorPoolCreateInfo dp_info = {};
    dp_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dp_info.maxSets = 2048;
    dp_info.poolSizeCount = 1;
    dp_info.pPoolSizes = &pool_size;

    res = g_pfn_vkCreateDescriptorPool(ctx.device, &dp_info, NULL, &desc_pool);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Pipeline] vkCreateDescriptorPool failed: " << res << "\n";
        return false;
    }

    return true;
}

VkDescriptorSet ComputePipeline::create_descriptor_set(const VulkanContext &ctx,
                                                      const std::vector<VkBuffer> &buffers,
                                                      const std::vector<VkDeviceSize> &ranges) {
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = desc_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &desc_layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult res = g_pfn_vkAllocateDescriptorSets(ctx.device, &alloc_info, &set);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Pipeline] vkAllocateDescriptorSets failed: " << res << "\n";
        return VK_NULL_HANDLE;
    }

    std::vector<VkDescriptorBufferInfo> buf_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());

    for (size_t i = 0; i < buffers.size(); i++) {
        buf_infos[i].buffer = buffers[i];
        buf_infos[i].offset = 0;
        buf_infos[i].range = (i < ranges.size()) ? ranges[i] : VK_WHOLE_SIZE;

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buf_infos[i];
    }

    g_pfn_vkUpdateDescriptorSets(ctx.device, (uint32_t)writes.size(), writes.data(), 0, NULL);
    return set;
}

void ComputePipeline::cleanup(const VulkanContext &ctx) {
    if (desc_pool) {
        g_pfn_vkDestroyDescriptorPool(ctx.device, desc_pool, NULL);
        desc_pool = VK_NULL_HANDLE;
    }
    if (pipeline) {
        g_pfn_vkDestroyPipeline(ctx.device, pipeline, NULL);
        pipeline = VK_NULL_HANDLE;
    }
    if (pipeline_layout) {
        g_pfn_vkDestroyPipelineLayout(ctx.device, pipeline_layout, NULL);
        pipeline_layout = VK_NULL_HANDLE;
    }
    if (desc_layout) {
        g_pfn_vkDestroyDescriptorSetLayout(ctx.device, desc_layout, NULL);
        desc_layout = VK_NULL_HANDLE;
    }
    if (shader_module) {
        g_pfn_vkDestroyShaderModule(ctx.device, shader_module, NULL);
        shader_module = VK_NULL_HANDLE;
    }
}
