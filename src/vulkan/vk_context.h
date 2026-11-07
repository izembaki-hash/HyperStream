#ifndef VK_CONTEXT_H
#define VK_CONTEXT_H

#include "vk_dispatch.h"
#include <string>
#include <vector>

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties device_properties = {};
    VkPhysicalDeviceMemoryProperties memory_properties = {};
    VkDevice device = VK_NULL_HANDLE;
    uint32_t compute_queue_family_index = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;

    // Discovered GPU specifications
    std::string device_name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint64_t vram_heap_size = 0;

    int find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
        for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
            if ((type_filter & (1 << i)) &&
                (memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return (int)i;
            }
        }
        return -1;
    }

    bool init();
    void cleanup();
};

#endif // VK_CONTEXT_H
