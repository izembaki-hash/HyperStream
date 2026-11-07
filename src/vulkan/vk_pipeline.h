#ifndef VK_PIPELINE_H
#define VK_PIPELINE_H

#include "vk_context.h"
#include <vector>

struct ComputePipeline {
    VkShaderModule shader_module = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;

    bool init_from_spv(const VulkanContext &ctx,
                       const uint32_t *spv_code, size_t spv_size_bytes,
                       uint32_t num_storage_buffers,
                       uint32_t push_constant_size_bytes);

    VkDescriptorSet create_descriptor_set(const VulkanContext &ctx,
                                          const std::vector<VkBuffer> &buffers,
                                          const std::vector<VkDeviceSize> &ranges);

    void cleanup(const VulkanContext &ctx);
};

#endif // VK_PIPELINE_H
