#include "vk_context.h"
#include <iostream>
#include <stdexcept>

bool VulkanContext::init() {
    if (!init_vulkan_dispatch()) {
        std::cerr << "[Vulkan Context] Failed to initialize dispatch table.\n";
        return false;
    }

    // 1. Create Instance
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "HyperStream-Vulkan";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "HyperStream";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo inst_info = {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;

    VkResult res = g_pfn_vkCreateInstance(&inst_info, NULL, &instance);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Context] vkCreateInstance failed: " << res << "\n";
        return false;
    }

    // 2. Enumerate Physical Devices & Find AMD RX 580
    uint32_t dev_count = 0;
    g_pfn_vkEnumeratePhysicalDevices(instance, &dev_count, NULL);
    if (dev_count == 0) {
        std::cerr << "[Vulkan Context] No Vulkan-compatible physical devices found.\n";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(dev_count);
    g_pfn_vkEnumeratePhysicalDevices(instance, &dev_count, devices.data());

    physical_device = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties props;
        g_pfn_vkGetPhysicalDeviceProperties(devices[i], &props);

        // Check if AMD Discrete GPU (0x1002)
        if (props.vendorID == 0x1002 && props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physical_device = devices[i];
            device_properties = props;
            device_name = props.deviceName;
            vendor_id = props.vendorID;
            device_id = props.deviceID;
            break;
        }
    }

    // Fallback to first discrete GPU if specific AMD check did not match
    if (!physical_device) {
        for (uint32_t i = 0; i < dev_count; i++) {
            VkPhysicalDeviceProperties props;
            g_pfn_vkGetPhysicalDeviceProperties(devices[i], &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical_device = devices[i];
                device_properties = props;
                device_name = props.deviceName;
                vendor_id = props.vendorID;
                device_id = props.deviceID;
                break;
            }
        }
    }

    if (!physical_device) {
        std::cerr << "[Vulkan Context] Could not find a suitable discrete GPU (RX 580).\n";
        return false;
    }

    g_pfn_vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (uint32_t i = 0; i < memory_properties.memoryHeapCount; i++) {
        if (memory_properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            vram_heap_size = memory_properties.memoryHeaps[i].size;
            break;
        }
    }

    std::cout << "[Vulkan Context] Target GPU Selected: " << device_name
              << " (Vendor: 0x" << std::hex << vendor_id << ", Device: 0x" << device_id << std::dec
              << ", VRAM: " << (vram_heap_size / (1024 * 1024)) << " MB)\n";

    // 3. Find Compute Queue Family
    uint32_t q_count = 0;
    g_pfn_vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &q_count, NULL);
    std::vector<VkQueueFamilyProperties> q_props(q_count);
    g_pfn_vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &q_count, q_props.data());

    int selected_q = -1;
    // Prefer dedicated compute queue (Compute YES, Graphics NO)
    for (uint32_t i = 0; i < q_count; i++) {
        if ((q_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(q_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            selected_q = i;
            break;
        }
    }
    // Otherwise fallback to universal compute queue
    if (selected_q == -1) {
        for (uint32_t i = 0; i < q_count; i++) {
            if (q_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                selected_q = i;
                break;
            }
        }
    }

    if (selected_q == -1) {
        std::cerr << "[Vulkan Context] No compute-capable queue family found.\n";
        return false;
    }

    compute_queue_family_index = (uint32_t)selected_q;
    std::cout << "[Vulkan Context] Using Compute Queue Family: " << compute_queue_family_index << "\n";

    // 4. Create Logical Device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo q_info = {};
    q_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q_info.queueFamilyIndex = compute_queue_family_index;
    q_info.queueCount = 1;
    q_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features = {};

    VkDeviceCreateInfo dev_info = {};
    dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &q_info;
    dev_info.pEnabledFeatures = &device_features;

    res = g_pfn_vkCreateDevice(physical_device, &dev_info, NULL, &device);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Context] vkCreateDevice failed: " << res << "\n";
        return false;
    }

    g_pfn_vkGetDeviceQueue(device, compute_queue_family_index, 0, &compute_queue);

    // 5. Create Command Pool
    VkCommandPoolCreateInfo cp_info = {};
    cp_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cp_info.queueFamilyIndex = compute_queue_family_index;
    cp_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    res = g_pfn_vkCreateCommandPool(device, &cp_info, NULL, &command_pool);
    if (res != VK_SUCCESS) {
        std::cerr << "[Vulkan Context] vkCreateCommandPool failed: " << res << "\n";
        return false;
    }

    std::cout << "[Vulkan Context] Vulkan Device & Compute Queue successfully created!\n";
    return true;
}

void VulkanContext::cleanup() {
    if (device) {
        g_pfn_vkQueueWaitIdle(compute_queue);
        if (command_pool) {
            g_pfn_vkDestroyCommandPool(device, command_pool, NULL);
            command_pool = VK_NULL_HANDLE;
        }
        g_pfn_vkDestroyDevice(device, NULL);
        device = VK_NULL_HANDLE;
    }
    if (instance) {
        g_pfn_vkDestroyInstance(instance, NULL);
        instance = VK_NULL_HANDLE;
    }
}
