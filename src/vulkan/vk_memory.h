#ifndef VK_MEMORY_H
#define VK_MEMORY_H

#include "vk_context.h"
#include <cstdint>

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    size_t size = 0;
    VkBufferUsageFlags usage = 0;
    VkMemoryPropertyFlags properties = 0;
    void *mapped_ptr = nullptr;

    bool allocate(const VulkanContext &ctx, size_t num_bytes,
                  VkBufferUsageFlags usage_flags,
                  VkMemoryPropertyFlags prop_flags);

    bool upload_direct(const void *data, size_t num_bytes);
    bool upload_staging(const VulkanContext &ctx, const void *data, size_t num_bytes);
    bool download(const VulkanContext &ctx, void *dest, size_t num_bytes);

    void cleanup(const VulkanContext &ctx);
};

#endif // VK_MEMORY_H
