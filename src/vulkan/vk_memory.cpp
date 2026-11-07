#include "vk_memory.h"
#include <iostream>
#include <cstring>

bool VulkanBuffer::allocate(const VulkanContext &ctx, size_t num_bytes,
                            VkBufferUsageFlags usage_flags,
                            VkMemoryPropertyFlags prop_flags) {
    size = num_bytes;
    usage = usage_flags;
    properties = prop_flags;

    VkBufferCreateInfo buf_info = {};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = num_bytes;
    buf_info.usage = usage_flags;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = g_pfn_vkCreateBuffer(ctx.device, &buf_info, NULL, &buffer);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Memory] vkCreateBuffer failed: " << res << "\n";
        return false;
    }

    VkMemoryRequirements mem_reqs;
    g_pfn_vkGetBufferMemoryRequirements(ctx.device, buffer, &mem_reqs);

    int mem_type = ctx.find_memory_type(mem_reqs.memoryTypeBits, prop_flags);
    if (mem_type == -1) {
        std::cerr << "[Vulkan Memory] Failed to find matching memory type for properties: " << prop_flags << "\n";
        return false;
    }

    VkMemoryAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = (uint32_t)mem_type;

    res = g_pfn_vkAllocateMemory(ctx.device, &alloc_info, NULL, &memory);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Memory] vkAllocateMemory failed: " << res << "\n";
        return false;
    }

    res = g_pfn_vkBindBufferMemory(ctx.device, buffer, memory, 0);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Memory] vkBindBufferMemory failed: " << res << "\n";
        return false;
    }

    return true;
}

bool VulkanBuffer::upload_direct(const void *data, size_t num_bytes) {
    if (!mapped_ptr) {
        std::cerr << "[Vulkan Memory] upload_direct called on unmapped buffer.\n";
        return false;
    }
    memcpy(mapped_ptr, data, num_bytes);
    return true;
}

bool VulkanBuffer::upload_staging(const VulkanContext &ctx, const void *data, size_t num_bytes) {
    // 1. Allocate host-visible staging buffer
    VulkanBuffer staging;
    if (!staging.allocate(ctx, num_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        return false;
    }

    // 2. Map and copy data
    void *map = nullptr;
    g_pfn_vkMapMemory(ctx.device, staging.memory, 0, num_bytes, 0, &map);
    memcpy(map, data, num_bytes);
    g_pfn_vkUnmapMemory(ctx.device, staging.memory);

    // 3. Allocate command buffer
    VkCommandBufferAllocateInfo cmd_alloc = {};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = ctx.command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    g_pfn_vkAllocateCommandBuffers(ctx.device, &cmd_alloc, &cmd);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    g_pfn_vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = num_bytes;
    g_pfn_vkCmdCopyBuffer(cmd, staging.buffer, buffer, 1, &copy_region);

    g_pfn_vkEndCommandBuffer(cmd);

    // 4. Submit and wait
    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    g_pfn_vkCreateFence(ctx.device, &fence_info, NULL, &fence);

    g_pfn_vkQueueSubmit(ctx.compute_queue, 1, &submit_info, fence);
    g_pfn_vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);

    // 5. Cleanup
    g_pfn_vkDestroyFence(ctx.device, fence, NULL);
    g_pfn_vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
    staging.cleanup(ctx);

    return true;
}

bool VulkanBuffer::download(const VulkanContext &ctx, void *dest, size_t num_bytes) {
    VulkanBuffer staging;
    if (!staging.allocate(ctx, num_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        return false;
    }

    VkCommandBufferAllocateInfo cmd_alloc = {};
    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.commandPool = ctx.command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    g_pfn_vkAllocateCommandBuffers(ctx.device, &cmd_alloc, &cmd);

    VkCommandBufferBeginInfo begin_info = {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    g_pfn_vkBeginCommandBuffer(cmd, &begin_info);

    VkBufferCopy copy_region = {};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = num_bytes;
    g_pfn_vkCmdCopyBuffer(cmd, buffer, staging.buffer, 1, &copy_region);

    g_pfn_vkEndCommandBuffer(cmd);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fence_info = {};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    g_pfn_vkCreateFence(ctx.device, &fence_info, NULL, &fence);

    g_pfn_vkQueueSubmit(ctx.compute_queue, 1, &submit_info, fence);
    g_pfn_vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX);

    void *map = nullptr;
    g_pfn_vkMapMemory(ctx.device, staging.memory, 0, num_bytes, 0, &map);
    memcpy(dest, map, num_bytes);
    g_pfn_vkUnmapMemory(ctx.device, staging.memory);

    g_pfn_vkDestroyFence(ctx.device, fence, NULL);
    g_pfn_vkFreeCommandBuffers(ctx.device, ctx.command_pool, 1, &cmd);
    staging.cleanup(ctx);

    return true;
}

void VulkanBuffer::cleanup(const VulkanContext &ctx) {
    if (mapped_ptr && memory) {
        g_pfn_vkUnmapMemory(ctx.device, memory);
        mapped_ptr = nullptr;
    }
    if (buffer) {
        g_pfn_vkDestroyBuffer(ctx.device, buffer, NULL);
        buffer = VK_NULL_HANDLE;
    }
    if (memory) {
        g_pfn_vkFreeMemory(ctx.device, memory, NULL);
        memory = VK_NULL_HANDLE;
    }
    size = 0;
}
