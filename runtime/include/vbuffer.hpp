/** @file vbuffer.hpp
 *  Vulkan buffer with staging — device-local (or host-visible fallback).
 */
#pragma once

#include <vector>
#include <cstring>
#include <vulkan/vulkan_core.h>

// ── Helper: find memory type index with required properties ────────────

static inline uint32_t find_mem_type(VkPhysicalDevice pdev, uint32_t type_filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return 0xFFFFFFFFu;
}

class VBuffer
{
public:
    void create(VkDevice dev, VkPhysicalDevice pdev, VkDeviceSize size_bytes)
    {
        device_   = dev;
        phys_dev_ = pdev;

        VkBufferCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size          = size_bytes;
        ci.usage         = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT  |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check_vk(vkCreateBuffer(dev, &ci, nullptr, &buf_), "VBuffer::create");

        VkMemoryRequirements mem_req{};
        vkGetBufferMemoryRequirements(dev, buf_, &mem_req);

        /* Prefer device-local memory */
        mem_type_idx_ = find_mem_type(phys_dev_, mem_req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mem_type_idx_ == 0xFFFFFFFFu) {
            /* fallback: host-visible + coherent for direct access */
            mem_type_idx_ = find_mem_type(phys_dev_, mem_req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }

        VkMemoryAllocateInfo ai{};
        ai.sType               = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize      = mem_req.size;
        ai.memoryTypeIndex     = mem_type_idx_;
        check_vk(vkAllocateMemory(dev, &ai, nullptr, &mem_), "VBuffer::alloc");

        check_vk(vkBindBufferMemory(dev, buf_, mem_, 0), "VBuffer::bind");
        size_ = size_bytes;
    }

    void destroy(VkDevice dev)
    {
        if (mem_ != VK_NULL_HANDLE) vkFreeMemory(dev, mem_, nullptr);
        if (buf_  != VK_NULL_HANDLE) vkDestroyBuffer(dev, buf_,  nullptr);
        mem_ = static_cast<VkDeviceMemory>(VK_NULL_HANDLE);
        buf_  = static_cast<VkBuffer>(VK_NULL_HANDLE);
    }

    VkBuffer         get()          const { return buf_; }
    VkDeviceSize     size()         const { return size_; }
    uint32_t         mem_type_idx() const { return mem_type_idx_; }

    void write(const void* src, VkDeviceSize offset = 0, VkDeviceSize count = VK_WHOLE_SIZE)
    {
        void* mapped;
        check_vk(vkMapMemory(device_, mem_, offset, count, 0, &mapped), "VBuffer::write map");
        std::memcpy(static_cast<char*>(mapped) + offset, src, static_cast<size_t>(count));
        vkUnmapMemory(device_, mem_);
    }

    std::vector<uint8_t> read(VkDeviceSize offset = 0, VkDeviceSize count = VK_WHOLE_SIZE)
    {
        void* mapped;
        check_vk(vkMapMemory(device_, mem_, offset, count, 0, &mapped), "VBuffer::read map");
        std::vector<uint8_t> data(static_cast<size_t>(count));
        std::memcpy(data.data(), static_cast<const char*>(mapped) + offset,
                    static_cast<size_t>(count));
        vkUnmapMemory(device_, mem_);
        return data;
    }

private:
    VkBuffer         buf_   = VK_NULL_HANDLE;
    VkDeviceMemory   mem_   = VK_NULL_HANDLE;
    VkDeviceSize     size_  = 0;
    uint32_t         mem_type_idx_ = 0xFFFFFFFFu;
    VkDevice         device_;
    VkPhysicalDevice phys_dev_;
};
