/** @file test_vulkan_helpers.h
 *  Test-only Vulkan infrastructure for kernel compiler unit tests.
 *  Provides VulkanComputeContext (wrapper around VkDescriptorSetLayout,
 *  VkPipelineLayout, vkDescriptorPool*, vkCommandPool*) to reduce per-test overhead,
 *  plus a VulkanTestBase fixture for device creation and initialization.
 *
 *  Depends on runtime/src/test_vulkan_helpers.cc which provides the
 *  vk_result_str / check_vk implementations (link against it).
 */

#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>

#include <vulkan/vulkan_core.h>
#include <gtest/gtest.h>

// Forward declarations for functions implemented in test_vulkan_helpers.cc
std::string vk_result_str(VkResult rc);
void check_vk(VkResult rc, const char *label);

struct VulkanDimension
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};


// ───────────── VulkanTestBase fixture (creates instance + device) ───

class VulkanSession
{
public:
    explicit VulkanSession(bool enable_cooperative_matrix2 = false);

    bool has_device() const { return m_instance != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE; }
    VkDevice get_device() const { return m_device; }
    VkQueue get_queue() const { return m_queue; }
    uint32_t get_queue_family_index() const { return m_queue_fi; }
    VkPhysicalDevice get_phys_device() const { return m_phys_dev; }
    bool shader_buffer_float32_atomic_add_enabled() const { return m_shader_buffer_float32_atomic_add_enabled; }
    bool cooperative_matrix2_enabled() const { return m_coopmat2_enabled; }
    const std::string& cooperative_matrix2_unavailable_reason() const { return m_coopmat2_unavailable_reason; }

    ~VulkanSession()
    {
        if (m_device)
        {
            vkDestroyDevice(m_device, nullptr);
            m_device = VK_NULL_HANDLE;
        }
        if (m_instance)
        {
            vkDestroyInstance(m_instance, nullptr);
            m_instance = VK_NULL_HANDLE;
        }
    }

protected:
    static bool device_name_contains(const char* device_name, const char* needle);
    bool has_device_extension(const char* name) const;

    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_phys_dev = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queue_fi = 0xFFFFFFFFu;
    bool m_shader_buffer_float32_atomic_add_enabled = false;
    bool m_coopmat2_enabled = false;
    std::string m_coopmat2_unavailable_reason;
};

// ───────────── VulkanComputeContext: bundles device ptr for RAII ────

class VulkanComputeKernel
{
public:
    VulkanComputeKernel(VulkanSession &session, const std::string &path, uint32_t push_size_bytes, uint32_t num_ssbo)
        : m_session(session)
    {
        init_descriptor_layout(num_ssbo);        /* for example: 3 SSBOs */
        init_pipeline_layout(push_size_bytes); /* for example: 5 × int32 = 20 bytes */
        create_pipeline(path);
    }

    ~VulkanComputeKernel()
    {
        /* Destroy in reverse order of creation */
        if (m_pipeline != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(get_device(), m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }
        if (m_pipe_layout != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(get_device(), m_pipe_layout, nullptr);
            m_pipe_layout = VK_NULL_HANDLE;
        }
        if (m_shader_module != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(get_device(), m_shader_module, nullptr);
            m_shader_module = VK_NULL_HANDLE;
        }
        if (m_desc_pool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(get_device(), m_desc_pool, nullptr);
            m_desc_pool = VK_NULL_HANDLE;
        }
        if (m_dsl != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(get_device(), m_dsl, nullptr);
            m_dsl = VK_NULL_HANDLE;
        }
    }

private:
    /* Create pipeline layout (push constants) */
    void init_pipeline_layout(uint32_t push_size_bytes)
    {
        m_push_const_offset = 0;

        /* Only create push constant range if size > 0 */
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        if (m_dsl != VK_NULL_HANDLE)
        {
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &m_dsl;
        }
        
        if (push_size_bytes > 0)
        {
            VkPushConstantRange pcr{};
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.offset = m_push_const_offset;
            pcr.size = push_size_bytes;

            plci.pushConstantRangeCount = 1;
            plci.pPushConstantRanges = &pcr;
        }
        else
        {
            plci.pushConstantRangeCount = 0;
            plci.pPushConstantRanges = nullptr;
        }

        check_vk(vkCreatePipelineLayout(get_device(), &plci, nullptr, &m_pipe_layout), "VkComputeSession pipeline layout");
    }

    /* Create descriptor set layout (n SSBO bindings) */
    void init_descriptor_layout(uint32_t num_ssbo)
    {
        if (num_ssbo == 0)
            return;

        std::vector<VkDescriptorSetLayoutBinding> bindings(num_ssbo);
        for (uint32_t i = 0; i < num_ssbo; ++i)
        {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo dslici{};
        dslici.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslici.bindingCount = static_cast<uint32_t>(bindings.size());
        dslici.pBindings = bindings.data();

        check_vk(vkCreateDescriptorSetLayout(get_device(), &dslici, nullptr, &m_dsl), "VkComputeSession DSL");

        /* Allocate descriptor pool and descriptor set */
        VkDescriptorPoolSize pool_size{};
        pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = num_ssbo;

        VkDescriptorPoolCreateInfo dpici{};
        dpici.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpici.maxSets                 = 1;
        dpici.poolSizeCount           = 1;
        dpici.pPoolSizes              = &pool_size;
        check_vk(vkCreateDescriptorPool(get_device(), &dpici, nullptr, &m_desc_pool), "VkComputeSession descriptor pool");

        VkDescriptorSetLayout layouts[] = {m_dsl};
        VkDescriptorSetAllocateInfo dasai{};
        dasai.sType                   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dasai.descriptorPool          = m_desc_pool;
        dasai.descriptorSetCount      = 1;
        dasai.pSetLayouts             = layouts;
        check_vk(vkAllocateDescriptorSets(get_device(), &dasai, &m_desc_set), "VkComputeSession allocate descriptor set");
    }

    /* Compile GLSL -> SPIR-V and create compute pipeline */
    void create_pipeline(const std::string &glsl_file);

public:
    VkDescriptorSet desc_set() const { return m_desc_set; }
    VkDescriptorSetLayout desc_set_layout() const { return m_dsl; }
    VkPipelineLayout pipeline_layout() const { return m_pipe_layout; }
    VkPipeline pipeline() const { return m_pipeline; }

    /* Dispatch the compute kernel. Bind a descriptor set if provided; otherwise push-constants only. */
    void dispatch(VkCommandBuffer cb, void* constants, size_t num_bytes, const VulkanDimension& dims, VkDescriptorSet desc_set = VK_NULL_HANDLE)
    {
        if (num_bytes > 0 && constants)
        {
            vkCmdPushConstants(cb, m_pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, m_push_const_offset, num_bytes, constants);
        }

        VkDescriptorSet set_to_bind = desc_set != VK_NULL_HANDLE ? desc_set : m_desc_set;
        if (set_to_bind != VK_NULL_HANDLE)
        {
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipe_layout, 0, 1, &set_to_bind, 0, nullptr);
        }
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
        vkCmdDispatch(cb, dims.x, dims.y, dims.z);
    }

private:
    static constexpr size_t MAX_ARGS = 128;

    VulkanSession &m_session;

    uint32_t m_push_const_offset = 0;
    VkPipelineLayout m_pipe_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkShaderModule m_shader_module = VK_NULL_HANDLE;
    VkDescriptorSet m_desc_set = VK_NULL_HANDLE;
    VkDescriptorSetLayout   m_dsl      = VK_NULL_HANDLE;
    VkDescriptorPool        m_desc_pool = VK_NULL_HANDLE;

    VkDevice get_device() const { return m_session.get_device(); }
};

class VulkanComputeContext
{
public:
    VulkanComputeContext(VulkanSession &session)
        : m_session(session)
    {
        init_command_pool();
        allocate_descriptor_set(32);  /* default: up to 32 SSBOs */
    }

    ~VulkanComputeContext()
    {
        if (m_desc_set != VK_NULL_HANDLE)
        {
            vkFreeDescriptorSets(get_device(), m_desc_pool, 1, &m_desc_set);
            m_desc_set = VK_NULL_HANDLE;
        }
        if (m_desc_pool != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(get_device(), m_desc_pool, nullptr);
            m_desc_pool = VK_NULL_HANDLE;
        }
        if (m_cmd_buf != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(get_device(), m_cmd_pool, 1, &m_cmd_buf);
            m_cmd_buf = VK_NULL_HANDLE;
        }
        if (m_cmd_pool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(get_device(), m_cmd_pool, nullptr);
            m_cmd_pool = VK_NULL_HANDLE;
        }
    }

    /* Return the allocated descriptor set for descriptor writes. */
    VkDescriptorSet desc_set() const { return m_desc_set; }

public:
    VkDevice get_device() const { return m_session.get_device(); }
    VkQueue get_queue() const { return m_session.get_queue(); }

private:
    /* Create command pool and allocate one primary command buffer */
    void init_command_pool()
    {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = m_session.get_queue_family_index();

        check_vk(vkCreateCommandPool(get_device(), &cpci, nullptr, &m_cmd_pool), "VkComputeSession cmd pool");

        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = m_cmd_pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check_vk(vkAllocateCommandBuffers(get_device(), &cai, &m_cmd_buf), "VkComputeSession cmd buf alloc");
    }

public:
    /* Begin the command buffer (caller fills it in between begin/end) */
    VkCommandBuffer begin_command_buffer()
    {
        VkCommandBufferBeginInfo bbci{};
        bbci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check_vk(vkBeginCommandBuffer(m_cmd_buf, &bbci), "VkComputeSession cmd buf begin");
        return m_cmd_buf;
    }

    /* Submit the command buffer and wait for completion. */
    void submit_and_wait()
    {
        vkEndCommandBuffer(m_cmd_buf);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &m_cmd_buf;
        check_vk(vkQueueSubmit(get_queue(), 1, &si, VK_NULL_HANDLE), "VkComputeSession submit");
        check_vk(vkDeviceWaitIdle(get_device()), "VkComputeSession wait idle");
    }

private:
    /* Allocate a descriptor set from a per-context pool. */
    void allocate_descriptor_set(uint32_t max_sets)
    {
        VkDescriptorPoolSize pool_size{};
        pool_size.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = max_sets;

        VkDescriptorPoolCreateInfo dpici{};
        dpici.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpici.maxSets                 = max_sets;
        dpici.poolSizeCount           = 1;
        dpici.pPoolSizes              = &pool_size;
        check_vk(vkCreateDescriptorPool(get_device(), &dpici, nullptr, &m_desc_pool), "VulkanComputeContext descriptor pool");

        /* We need a DSL for allocation — create one with storage buffer bindings. */
        VkDescriptorSetLayoutBinding lsb{};
        lsb.binding         = 0;
        lsb.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lsb.descriptorCount = max_sets;
        lsb.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslici{};
        dslici.sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslici.bindingCount   = 1;
        dslici.pBindings      = &lsb;

        VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
        check_vk(vkCreateDescriptorSetLayout(get_device(), &dslici, nullptr, &dsl), "VulkanComputeContext DSL");

        VkDescriptorSetAllocateInfo dasai{};
        dasai.sType                   = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dasai.descriptorPool          = m_desc_pool;
        dasai.descriptorSetCount      = 1;
        dasai.pSetLayouts             = &dsl;
        check_vk(vkAllocateDescriptorSets(get_device(), &dasai, &m_desc_set), "VulkanComputeContext allocate descriptor set");

        vkDestroyDescriptorSetLayout(get_device(), dsl, nullptr);
    }

    VulkanSession &m_session;
    VkCommandPool   m_cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer m_cmd_buf  = VK_NULL_HANDLE;
    VkDescriptorPool m_desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet  m_desc_set  = VK_NULL_HANDLE;
};

static inline uint32_t find_mem_type(VkPhysicalDevice pdev, uint32_t type_filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    {
        if ((type_filter & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return 0xFFFFFFFFu;
}

static inline void create_bound_buffer(
    VulkanSession& session,
    VkDeviceSize size_bytes,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags preferred_props,
    VkBuffer& buffer,
    VkDeviceMemory& memory,
    uint32_t& mem_type_idx,
    const char* label)
{
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size_bytes;
    ci.usage = usage;
    check_vk(vkCreateBuffer(session.get_device(), &ci, nullptr, &buffer), label);

    VkMemoryRequirements mem_req{};
    vkGetBufferMemoryRequirements(session.get_device(), buffer, &mem_req);

    mem_type_idx = find_mem_type(session.get_phys_device(), mem_req.memoryTypeBits, preferred_props);
    if (mem_type_idx == 0xFFFFFFFFu && (preferred_props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        mem_type_idx = find_mem_type(session.get_phys_device(), mem_req.memoryTypeBits, 0);
    if (mem_type_idx == 0xFFFFFFFFu &&
        (preferred_props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (preferred_props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
    {
        mem_type_idx = find_mem_type(
            session.get_phys_device(),
            mem_req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }
    if (mem_type_idx == 0xFFFFFFFFu)
        throw std::runtime_error(std::string(label) + ": no suitable memory type");

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = mem_req.size;
    ai.memoryTypeIndex = mem_type_idx;
    check_vk(vkAllocateMemory(session.get_device(), &ai, nullptr, &memory), label);
    check_vk(vkBindBufferMemory(session.get_device(), buffer, memory, 0), label);
}

class VBaseHostBuffer
{
public:
    VBaseHostBuffer(VulkanSession &session, VkDeviceSize size_bytes)
        : m_session(session)
    {
        size_ = size_bytes;
        create_bound_buffer(
            m_session,
            size_bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            buf_,
            mem_,
            mem_type_idx_,
            "VBaseHostBuffer");
        void *mapped = nullptr;
        check_vk(vkMapMemory(m_session.get_device(), mem_, 0, size_, 0, &mapped), "VBaseHostBuffer::map");
        mapped_ = static_cast<uint8_t *>(mapped);
    }

    VBaseHostBuffer(const VBaseHostBuffer&) = delete;
    VBaseHostBuffer& operator=(const VBaseHostBuffer&) = delete;

    VBaseHostBuffer(VBaseHostBuffer&& other)
        : m_session(other.m_session),
          buf_(other.buf_),
          mem_(other.mem_),
          mapped_(other.mapped_),
          size_(other.size_),
          mem_type_idx_(other.mem_type_idx_)
    {
        other.buf_ = VK_NULL_HANDLE;
        other.mem_ = VK_NULL_HANDLE;
        other.mapped_ = nullptr;
        other.size_ = 0;
        other.mem_type_idx_ = 0xFFFFFFFFu;
    }

    VBaseHostBuffer& operator=(VBaseHostBuffer&& other)
    {
        if (this != &other)
        {
            destroy_resources();
            m_session = other.m_session;
            buf_ = other.buf_;
            mem_ = other.mem_;
            mapped_ = other.mapped_;
            size_ = other.size_;
            mem_type_idx_ = other.mem_type_idx_;

            other.buf_ = VK_NULL_HANDLE;
            other.mem_ = VK_NULL_HANDLE;
            other.mapped_ = nullptr;
            other.size_ = 0;
            other.mem_type_idx_ = 0xFFFFFFFFu;
        }
        return *this;
    }

    ~VBaseHostBuffer()
    {
        destroy_resources();
    }

    uint8_t* get() { return mapped_; }
    const uint8_t* get() const { return mapped_; }
    VkDeviceSize size() const { return size_; }
    uint32_t mem_type_idx() const { return mem_type_idx_; }

private:
    friend class VBaseDeviceBuffer;

    VkBuffer vk_buffer() const { return buf_; }

    bool host_memory_is_coherent() const
    {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(m_session.get_phys_device(), &mp);
        return mem_type_idx_ < mp.memoryTypeCount &&
            (mp.memoryTypes[mem_type_idx_].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    void flush()
    {
        if (host_memory_is_coherent())
            return;
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = mem_;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        check_vk(vkFlushMappedMemoryRanges(m_session.get_device(), 1, &range), "VBaseHostBuffer::write flush");
    }

    void invalidate()
    {
        if (host_memory_is_coherent())
            return;
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = mem_;
        range.offset = 0;
        range.size = VK_WHOLE_SIZE;
        check_vk(vkInvalidateMappedMemoryRanges(m_session.get_device(), 1, &range), "VBaseHostBuffer::read invalidate");
    }

    void destroy_resources()
    {
        if (mapped_)
            vkUnmapMemory(m_session.get_device(), mem_);
        if (buf_ != VK_NULL_HANDLE)
            vkDestroyBuffer(m_session.get_device(), buf_, nullptr);
        if (mem_ != VK_NULL_HANDLE)
            vkFreeMemory(m_session.get_device(), mem_, nullptr);
        mapped_ = nullptr;
        mem_ = VK_NULL_HANDLE;
        buf_ = VK_NULL_HANDLE;
    }

    VulkanSession &m_session;

    VkBuffer buf_ = VK_NULL_HANDLE;
    VkDeviceMemory mem_ = VK_NULL_HANDLE;
    uint8_t *mapped_ = nullptr;
    VkDeviceSize size_ = 0;
    uint32_t mem_type_idx_ = 0xFFFFFFFFu;
};

template <typename T>
class VHostBuffer : public VBaseHostBuffer
{
public:
    VHostBuffer(VulkanSession& session, size_t count)
        : VBaseHostBuffer(session, static_cast<VkDeviceSize>(count) * sizeof(T)),
          count_(count)
    {
    }

    T* get() { return reinterpret_cast<T*>(VBaseHostBuffer::get()); }
    const T* get() const { return reinterpret_cast<const T*>(VBaseHostBuffer::get()); }
    T get(size_t index) const { return get()[index]; }
    void set(size_t index, T value) { get()[index] = value; }
    void fill(T value) { std::fill_n(get(), count_, value); }
    size_t count() const { return count_; }

private:
    size_t count_ = 0;
};

class VBaseDeviceBuffer
{
public:
    VBaseDeviceBuffer(VulkanSession &session, VkDeviceSize size_bytes)
        : m_session(session)
    {
        size_ = size_bytes;
        create_bound_buffer(
            m_session,
            size_bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            buf_,
            mem_,
            mem_type_idx_,
            "VBaseDeviceBuffer");
    }

    VBaseDeviceBuffer(const VBaseDeviceBuffer&) = delete;
    VBaseDeviceBuffer& operator=(const VBaseDeviceBuffer&) = delete;

    VBaseDeviceBuffer(VBaseDeviceBuffer&& other)
        : m_session(other.m_session),
          buf_(other.buf_),
          mem_(other.mem_),
          size_(other.size_),
          mem_type_idx_(other.mem_type_idx_)
    {
        other.buf_ = VK_NULL_HANDLE;
        other.mem_ = VK_NULL_HANDLE;
        other.size_ = 0;
        other.mem_type_idx_ = 0xFFFFFFFFu;
    }

    VBaseDeviceBuffer& operator=(VBaseDeviceBuffer&& other)
    {
        if (this != &other)
        {
            destroy_resources();
            m_session = other.m_session;
            buf_ = other.buf_;
            mem_ = other.mem_;
            size_ = other.size_;
            mem_type_idx_ = other.mem_type_idx_;

            other.buf_ = VK_NULL_HANDLE;
            other.mem_ = VK_NULL_HANDLE;
            other.size_ = 0;
            other.mem_type_idx_ = 0xFFFFFFFFu;
        }
        return *this;
    }

    ~VBaseDeviceBuffer()
    {
        destroy_resources();
    }

    VkBuffer get() const { return buf_; }
    VkDeviceSize size() const { return size_; }
    uint32_t mem_type_idx() const { return mem_type_idx_; }

    void write(VBaseHostBuffer& src, VkDeviceSize count = VK_WHOLE_SIZE, VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0)
    {
        if (count == VK_WHOLE_SIZE)
            count = src.size() - src_offset;
        src.flush();
        copy_buffer(src.vk_buffer(), buf_, src_offset, dst_offset, count);
    }

    void read(VBaseHostBuffer& dst, VkDeviceSize count = VK_WHOLE_SIZE, VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0)
    {
        if (count == VK_WHOLE_SIZE)
            count = dst.size() - dst_offset;
        copy_buffer(buf_, dst.vk_buffer(), src_offset, dst_offset, count);
        dst.invalidate();
    }

private:
    void copy_buffer(
        VkBuffer src,
        VkBuffer dst,
        VkDeviceSize src_offset,
        VkDeviceSize dst_offset,
        VkDeviceSize count)
    {
        VkCommandPool cmd_pool = VK_NULL_HANDLE;
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        cpci.queueFamilyIndex = m_session.get_queue_family_index();
        check_vk(vkCreateCommandPool(m_session.get_device(), &cpci, nullptr, &cmd_pool), "VBaseDeviceBuffer::copy cmd pool");

        VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo cai{};
        cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool = cmd_pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check_vk(vkAllocateCommandBuffers(m_session.get_device(), &cai, &cmd_buf), "VBaseDeviceBuffer::copy cmd buf alloc");

        VkCommandBufferBeginInfo bbci{};
        bbci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bbci.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check_vk(vkBeginCommandBuffer(cmd_buf, &bbci), "VBaseDeviceBuffer::copy cmd buf begin");

        if (src == buf_)
        {
            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = src;
            barrier.offset = src_offset;
            barrier.size = count;
            vkCmdPipelineBarrier(
                cmd_buf,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr,
                1, &barrier,
                0, nullptr);
        }

        VkBufferCopy region{};
        region.srcOffset = src_offset;
        region.dstOffset = dst_offset;
        region.size = count;
        vkCmdCopyBuffer(cmd_buf, src, dst, 1, &region);

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dst;
        barrier.offset = dst_offset;
        barrier.size = count;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        if (dst == buf_)
        {
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
        else
        {
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            dst_stage = VK_PIPELINE_STAGE_HOST_BIT;
        }
        vkCmdPipelineBarrier(
            cmd_buf,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            dst_stage,
            0,
            0, nullptr,
            1, &barrier,
            0, nullptr);

        check_vk(vkEndCommandBuffer(cmd_buf), "VBaseDeviceBuffer::copy cmd buf end");

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_buf;
        check_vk(vkQueueSubmit(m_session.get_queue(), 1, &si, VK_NULL_HANDLE), "VBaseDeviceBuffer::copy submit");
        check_vk(vkDeviceWaitIdle(m_session.get_device()), "VBaseDeviceBuffer::copy wait idle");

        vkFreeCommandBuffers(m_session.get_device(), cmd_pool, 1, &cmd_buf);
        vkDestroyCommandPool(m_session.get_device(), cmd_pool, nullptr);
    }

    void destroy_resources()
    {
        if (buf_ != VK_NULL_HANDLE)
            vkDestroyBuffer(m_session.get_device(), buf_, nullptr);
        if (mem_ != VK_NULL_HANDLE)
            vkFreeMemory(m_session.get_device(), mem_, nullptr);
        mem_ = VK_NULL_HANDLE;
        buf_ = VK_NULL_HANDLE;
    }

    VulkanSession &m_session;

    VkBuffer buf_ = VK_NULL_HANDLE;
    VkDeviceMemory mem_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    uint32_t mem_type_idx_ = 0xFFFFFFFFu;
};

template <typename T>
class VDeviceBuffer : public VBaseDeviceBuffer
{
public:
    explicit VDeviceBuffer(VulkanSession& session)
        : VBaseDeviceBuffer(session, sizeof(T))
    {
    }

    VDeviceBuffer(VulkanSession& session, VkDeviceSize size_bytes)
        : VBaseDeviceBuffer(session, size_bytes)
    {
    }
};
