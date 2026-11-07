#include "vk_dispatch.h"

// Define function pointers
PFN_vkCreateInstance g_pfn_vkCreateInstance = NULL;
PFN_vkDestroyInstance g_pfn_vkDestroyInstance = NULL;
PFN_vkEnumeratePhysicalDevices g_pfn_vkEnumeratePhysicalDevices = NULL;
PFN_vkGetPhysicalDeviceProperties g_pfn_vkGetPhysicalDeviceProperties = NULL;
PFN_vkGetPhysicalDeviceFeatures g_pfn_vkGetPhysicalDeviceFeatures = NULL;
PFN_vkGetPhysicalDeviceMemoryProperties g_pfn_vkGetPhysicalDeviceMemoryProperties = NULL;
PFN_vkGetPhysicalDeviceQueueFamilyProperties g_pfn_vkGetPhysicalDeviceQueueFamilyProperties = NULL;
PFN_vkCreateDevice g_pfn_vkCreateDevice = NULL;
PFN_vkDestroyDevice g_pfn_vkDestroyDevice = NULL;
PFN_vkGetDeviceQueue g_pfn_vkGetDeviceQueue = NULL;
PFN_vkCreateCommandPool g_pfn_vkCreateCommandPool = NULL;
PFN_vkDestroyCommandPool g_pfn_vkDestroyCommandPool = NULL;
PFN_vkAllocateCommandBuffers g_pfn_vkAllocateCommandBuffers = NULL;
PFN_vkFreeCommandBuffers g_pfn_vkFreeCommandBuffers = NULL;
PFN_vkBeginCommandBuffer g_pfn_vkBeginCommandBuffer = NULL;
PFN_vkEndCommandBuffer g_pfn_vkEndCommandBuffer = NULL;
PFN_vkQueueSubmit g_pfn_vkQueueSubmit = NULL;
PFN_vkQueueWaitIdle g_pfn_vkQueueWaitIdle = NULL;
PFN_vkCreateFence g_pfn_vkCreateFence = NULL;
PFN_vkDestroyFence g_pfn_vkDestroyFence = NULL;
PFN_vkWaitForFences g_pfn_vkWaitForFences = NULL;
PFN_vkResetFences g_pfn_vkResetFences = NULL;
PFN_vkCreateBuffer g_pfn_vkCreateBuffer = NULL;
PFN_vkDestroyBuffer g_pfn_vkDestroyBuffer = NULL;
PFN_vkGetBufferMemoryRequirements g_pfn_vkGetBufferMemoryRequirements = NULL;
PFN_vkAllocateMemory g_pfn_vkAllocateMemory = NULL;
PFN_vkFreeMemory g_pfn_vkFreeMemory = NULL;
PFN_vkBindBufferMemory g_pfn_vkBindBufferMemory = NULL;
PFN_vkMapMemory g_pfn_vkMapMemory = NULL;
PFN_vkUnmapMemory g_pfn_vkUnmapMemory = NULL;
PFN_vkFlushMappedMemoryRanges g_pfn_vkFlushMappedMemoryRanges = NULL;
PFN_vkInvalidateMappedMemoryRanges g_pfn_vkInvalidateMappedMemoryRanges = NULL;
PFN_vkCreateShaderModule g_pfn_vkCreateShaderModule = NULL;
PFN_vkDestroyShaderModule g_pfn_vkDestroyShaderModule = NULL;
PFN_vkCreateDescriptorSetLayout g_pfn_vkCreateDescriptorSetLayout = NULL;
PFN_vkDestroyDescriptorSetLayout g_pfn_vkDestroyDescriptorSetLayout = NULL;
PFN_vkCreatePipelineLayout g_pfn_vkCreatePipelineLayout = NULL;
PFN_vkDestroyPipelineLayout g_pfn_vkDestroyPipelineLayout = NULL;
PFN_vkCreateComputePipelines g_pfn_vkCreateComputePipelines = NULL;
PFN_vkDestroyPipeline g_pfn_vkDestroyPipeline = NULL;
PFN_vkCreateDescriptorPool g_pfn_vkCreateDescriptorPool = NULL;
PFN_vkDestroyDescriptorPool g_pfn_vkDestroyDescriptorPool = NULL;
PFN_vkAllocateDescriptorSets g_pfn_vkAllocateDescriptorSets = NULL;
PFN_vkUpdateDescriptorSets g_pfn_vkUpdateDescriptorSets = NULL;
PFN_vkCmdBindPipeline g_pfn_vkCmdBindPipeline = NULL;
PFN_vkCmdBindDescriptorSets g_pfn_vkCmdBindDescriptorSets = NULL;
PFN_vkCmdPushConstants g_pfn_vkCmdPushConstants = NULL;
PFN_vkCmdDispatch g_pfn_vkCmdDispatch = NULL;
PFN_vkCmdPipelineBarrier g_pfn_vkCmdPipelineBarrier = NULL;
PFN_vkCmdCopyBuffer g_pfn_vkCmdCopyBuffer = NULL;
PFN_vkCreateQueryPool g_pfn_vkCreateQueryPool = NULL;
PFN_vkDestroyQueryPool g_pfn_vkDestroyQueryPool = NULL;
PFN_vkCmdResetQueryPool g_pfn_vkCmdResetQueryPool = NULL;
PFN_vkCmdWriteTimestamp g_pfn_vkCmdWriteTimestamp = NULL;
PFN_vkGetQueryPoolResults g_pfn_vkGetQueryPoolResults = NULL;
PFN_vkGetPhysicalDeviceProperties2 g_pfn_vkGetPhysicalDeviceProperties2 = NULL;

static HMODULE g_hVulkanDll = NULL;

#define LOAD_VK_ENTRY(name) \
    g_pfn_##name = (PFN_##name)GetProcAddress(g_hVulkanDll, #name); \
    if (!g_pfn_##name) { \
        printf("[Vulkan Error] Failed to load symbol: %s\n", #name); \
        return 0; \
    }

int init_vulkan_dispatch(void) {
    if (g_hVulkanDll) return 1; // already loaded

    g_hVulkanDll = LoadLibraryA("vulkan-1.dll");
    if (!g_hVulkanDll) {
        printf("[Vulkan Error] Could not load vulkan-1.dll from system path!\n");
        return 0;
    }

    LOAD_VK_ENTRY(vkCreateInstance);
    LOAD_VK_ENTRY(vkDestroyInstance);
    LOAD_VK_ENTRY(vkEnumeratePhysicalDevices);
    LOAD_VK_ENTRY(vkGetPhysicalDeviceProperties);
    LOAD_VK_ENTRY(vkGetPhysicalDeviceFeatures);
    LOAD_VK_ENTRY(vkGetPhysicalDeviceMemoryProperties);
    LOAD_VK_ENTRY(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_VK_ENTRY(vkCreateDevice);
    LOAD_VK_ENTRY(vkDestroyDevice);
    LOAD_VK_ENTRY(vkGetDeviceQueue);
    LOAD_VK_ENTRY(vkCreateCommandPool);
    LOAD_VK_ENTRY(vkDestroyCommandPool);
    LOAD_VK_ENTRY(vkAllocateCommandBuffers);
    LOAD_VK_ENTRY(vkFreeCommandBuffers);
    LOAD_VK_ENTRY(vkBeginCommandBuffer);
    LOAD_VK_ENTRY(vkEndCommandBuffer);
    LOAD_VK_ENTRY(vkQueueSubmit);
    LOAD_VK_ENTRY(vkQueueWaitIdle);
    LOAD_VK_ENTRY(vkCreateFence);
    LOAD_VK_ENTRY(vkDestroyFence);
    LOAD_VK_ENTRY(vkWaitForFences);
    LOAD_VK_ENTRY(vkResetFences);
    LOAD_VK_ENTRY(vkCreateBuffer);
    LOAD_VK_ENTRY(vkDestroyBuffer);
    LOAD_VK_ENTRY(vkGetBufferMemoryRequirements);
    LOAD_VK_ENTRY(vkAllocateMemory);
    LOAD_VK_ENTRY(vkFreeMemory);
    LOAD_VK_ENTRY(vkBindBufferMemory);
    LOAD_VK_ENTRY(vkMapMemory);
    LOAD_VK_ENTRY(vkUnmapMemory);
    LOAD_VK_ENTRY(vkFlushMappedMemoryRanges);
    LOAD_VK_ENTRY(vkInvalidateMappedMemoryRanges);
    LOAD_VK_ENTRY(vkCreateShaderModule);
    LOAD_VK_ENTRY(vkDestroyShaderModule);
    LOAD_VK_ENTRY(vkCreateDescriptorSetLayout);
    LOAD_VK_ENTRY(vkDestroyDescriptorSetLayout);
    LOAD_VK_ENTRY(vkCreatePipelineLayout);
    LOAD_VK_ENTRY(vkDestroyPipelineLayout);
    LOAD_VK_ENTRY(vkCreateComputePipelines);
    LOAD_VK_ENTRY(vkDestroyPipeline);
    LOAD_VK_ENTRY(vkCreateDescriptorPool);
    LOAD_VK_ENTRY(vkDestroyDescriptorPool);
    LOAD_VK_ENTRY(vkAllocateDescriptorSets);
    LOAD_VK_ENTRY(vkUpdateDescriptorSets);
    LOAD_VK_ENTRY(vkCmdBindPipeline);
    LOAD_VK_ENTRY(vkCmdBindDescriptorSets);
    LOAD_VK_ENTRY(vkCmdPushConstants);
    LOAD_VK_ENTRY(vkCmdDispatch);
    LOAD_VK_ENTRY(vkCmdPipelineBarrier);
    LOAD_VK_ENTRY(vkCmdCopyBuffer);
    LOAD_VK_ENTRY(vkCreateQueryPool);
    LOAD_VK_ENTRY(vkDestroyQueryPool);
    LOAD_VK_ENTRY(vkCmdResetQueryPool);
    LOAD_VK_ENTRY(vkCmdWriteTimestamp);
    LOAD_VK_ENTRY(vkGetQueryPoolResults);
    g_pfn_vkGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)GetProcAddress(g_hVulkanDll, "vkGetPhysicalDeviceProperties2");

    return 1;
}
