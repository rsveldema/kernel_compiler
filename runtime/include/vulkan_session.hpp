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

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
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
    VulkanSession()
    {
        VkApplicationInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.pApplicationName = "kernel_compiler_tests";
        ai.apiVersion = VK_API_VERSION_1_0;
        ai.pEngineName = "kernel_compiler";
        ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);

        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &ai;
        ici.enabledLayerCount = 0;
        ici.ppEnabledLayerNames = nullptr;
        ici.enabledExtensionCount = 0;
        ici.ppEnabledExtensionNames = nullptr;

        VkResult rc = vkCreateInstance(&ici, nullptr, &m_instance);
        if (rc != VK_SUCCESS)
        {
            fprintf(stderr, "init_vulkan_instance: vkCreateInstance failed: %s\n", vk_result_str(rc).c_str());
            m_instance = VK_NULL_HANDLE;
            return;
        }

        /* Enumerate physical devices */
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devs(count);
        if (count > 0)
            vkEnumeratePhysicalDevices(m_instance, &count, devs.data());
        if (count == 0)
        {
            fprintf(stderr, "init_vulkan_instance: no Vulkan physical devices found\n");
            m_instance = VK_NULL_HANDLE;
            return;
        }

        // Select device by VULKAN_DEVICE env var (default: "llvmpipe")
        m_phys_dev = devs[0]; /* fallback */
        const char *chosen = getenv("VULKAN_DEVICE");
        if (!chosen)
            chosen = "llvmpipe";
        for (auto &dev : devs)
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(dev, &props);
            if (strstr(props.deviceName, chosen))
            {
                m_phys_dev = dev;
                fprintf(stderr, "init_vulkan_instance: selecting device \"%s\"\n", props.deviceName);
                break;
            }
        }

        /* Get queue family */
        uint32_t qfc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(m_phys_dev, &qfc, nullptr);
        std::vector<VkQueueFamilyProperties> qfps(qfc > 0 ? qfc : 1);
        if (qfc > 0)
            vkGetPhysicalDeviceQueueFamilyProperties(m_phys_dev, &qfc, qfps.data());
        for (uint32_t i = 0; i < qfc; ++i)
        {
            if (qfps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                m_queue_fi = i;
                break;
            }
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo dqi{};
        dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqi.queueFamilyIndex = m_queue_fi;
        dqi.queueCount = 1;
        dqi.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &dqi;

        rc = vkCreateDevice(m_phys_dev, &dci, nullptr, &m_device);
        if (rc != VK_SUCCESS)
        {
            fprintf(stderr, "init_vulkan_instance: vkCreateDevice failed: %s\n", vk_result_str(rc).c_str());
            m_instance = static_cast<VkInstance>(VK_NULL_HANDLE);
            m_device = static_cast<VkDevice>(VK_NULL_HANDLE);
            return;
        }

        vkGetDeviceQueue(m_device, m_queue_fi, 0, &m_queue);
    }

    bool has_device() const { return m_instance != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE; }
    VkDevice get_device() const { return m_device; }
    VkQueue get_queue() const { return m_queue; }
    uint32_t get_queue_family_index() const { return m_queue_fi; }
    VkPhysicalDevice get_phys_device() const { return m_phys_dev; }

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
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPhysicalDevice m_phys_dev = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queue_fi = 0xFFFFFFFFu;
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
        VkDescriptorSetLayoutBinding lsb{};
        lsb.binding = 0;
        lsb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lsb.descriptorCount = num_ssbo;
        lsb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslici{};
        dslici.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslici.bindingCount = 1;
        dslici.pBindings = &lsb;

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

    /* Compile GLSL → SPIR-V and create compute pipeline */
    void create_pipeline(const std::string &glsl_file)
    {
        /* Read GLSL source */
        std::ifstream ifs(glsl_file);
        if (!ifs.is_open())
            throw std::runtime_error("Cannot open GLSL file: " + glsl_file);

        /* Compile via glslc and pipe to temp file */
        char tmp_spv[] = "/tmp/spv_gen_XXXXXX";
        int fd = mkstemp(tmp_spv);
        if (fd < 0)
            throw std::runtime_error("mkstemp failed");

        FILE *fp = popen(("glslc -x glsl -O -fshader-stage=compute " + glsl_file + " -o -").c_str(), "r");
        if (!fp)
        {
            close(fd);
            throw std::runtime_error("glslc failed for: " + glsl_file);
        }

        char buf[4096];
        while (auto n = fread(buf, 1, sizeof(buf), fp))
            write(fd, buf, static_cast<size_t>(n));
        int rc_p = pclose(fp);
        if (rc_p != 0)
        {
            close(fd);
            throw std::runtime_error("glslc error for: " + glsl_file);
        }
        close(fd);

        /* Read back SPIR-V */
        std::ifstream ifs_spv(tmp_spv, std::ios::binary);
        if (!ifs_spv.is_open())
            throw std::runtime_error("Cannot read generated SPIR-V");
        std::vector<uint8_t> spirv((std::istreambuf_iterator<char>(ifs_spv)),
                                    std::istreambuf_iterator<char>());

        /* Clean up temp file */
        remove(tmp_spv);

        if (spirv.empty())
            throw std::runtime_error("Empty SPIR-V from glslc for: " + glsl_file);

        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = spirv.size();
        smci.pCode = reinterpret_cast<const uint32_t*>(spirv.data());

        check_vk(vkCreateShaderModule(get_device(), &smci, nullptr, &m_shader_module), "VkComputeSession shader module");

        VkPipelineShaderStageCreateInfo psci{};
        psci.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        psci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        psci.module = m_shader_module;
        psci.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = psci;
        cpci.layout = m_pipe_layout;

        check_vk(vkCreateComputePipelines(get_device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &m_pipeline), "VkComputeSession pipeline");
    }

public:
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

        VkDescriptorSetLayout dsl;
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

class VBuffer
{
public:
    VBuffer(VulkanSession &session, VkDeviceSize size_bytes)
        : m_session(session)
    {
        VkBufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        ci.size = size_bytes;
        ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        check_vk(vkCreateBuffer(m_session.get_device(), &ci, nullptr, &buf_), "VBuffer::create");

        VkMemoryRequirements mem_req{};
        vkGetBufferMemoryRequirements(m_session.get_device(), buf_, &mem_req);

        /* Prefer device-local memory */
        mem_type_idx_ = find_mem_type(m_session.get_phys_device(), mem_req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mem_type_idx_ == 0xFFFFFFFFu)
        {
            /* fallback: host-visible + coherent for direct access */
            mem_type_idx_ = find_mem_type(m_session.get_phys_device(), mem_req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }

        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = mem_req.size;
        ai.memoryTypeIndex = mem_type_idx_;
        check_vk(vkAllocateMemory(m_session.get_device(), &ai, nullptr, &mem_), "VBuffer::alloc");

        check_vk(vkBindBufferMemory(m_session.get_device(), buf_, mem_, 0), "VBuffer::bind");
        size_ = size_bytes;
    }


    VBuffer(const VBuffer&)             = delete;
    VBuffer& operator=(const VBuffer&)   = delete;
    /* Move constructor — transfers Vulkan resource ownership from |other|. */
    VBuffer(VBuffer&& other)
        : m_session(other.m_session),
          buf_(other.buf_), mem_(other.mem_), size_(other.size_), mem_type_idx_(other.mem_type_idx_)
    {
        other.buf_     = VK_NULL_HANDLE;
        other.mem_     = VK_NULL_HANDLE;
        other.size_    = 0;
        other.mem_type_idx_ = 0xFFFFFFFFu;
    }

    VBuffer& operator=(VBuffer&& other)
    {
        if (this != &other)
        {
            /* Release our own resources */
            if (mem_ != VK_NULL_HANDLE)
                vkFreeMemory(m_session.get_device(), mem_, nullptr);
            if (buf_ != VK_NULL_HANDLE)
                vkDestroyBuffer(m_session.get_device(), buf_, nullptr);

            /* Adopt |other|'s resources */
            m_session         = other.m_session;
            buf_              = other.buf_;
            mem_              = other.mem_;
            size_             = other.size_;
            mem_type_idx_     = other.mem_type_idx_;

            other.buf_       = VK_NULL_HANDLE;
            other.mem_       = VK_NULL_HANDLE;
            other.size_      = 0;
            other.mem_type_idx_ = 0xFFFFFFFFu;
        }
        return *this;
    }

    ~VBuffer()
    {
        if (mem_ != VK_NULL_HANDLE)
            vkFreeMemory(m_session.get_device(), mem_, nullptr);
        if (buf_ != VK_NULL_HANDLE)
            vkDestroyBuffer(m_session.get_device(), buf_, nullptr);
        mem_ = static_cast<VkDeviceMemory>(VK_NULL_HANDLE);
        buf_ = static_cast<VkBuffer>(VK_NULL_HANDLE);
    }

    VkBuffer get() const { return buf_; }
    VkDeviceSize size() const { return size_; }
    uint32_t mem_type_idx() const { return mem_type_idx_; }

    void write(const void *src, VkDeviceSize offset = 0, VkDeviceSize count = VK_WHOLE_SIZE)
    {
        void *mapped;
        check_vk(vkMapMemory(m_session.get_device(), mem_, offset, count, 0, &mapped), "VBuffer::write map");
        std::memcpy(static_cast<char *>(mapped) + offset, src, static_cast<size_t>(count));
        vkUnmapMemory(m_session.get_device(), mem_);
    }

    std::vector<uint8_t> read(VkDeviceSize offset = 0, VkDeviceSize count = VK_WHOLE_SIZE)
    {
        void *mapped;
        check_vk(vkMapMemory(m_session.get_device(), mem_, offset, count, 0, &mapped), "VBuffer::read map");
        std::vector<uint8_t> data(static_cast<size_t>(count));
        std::memcpy(data.data(), static_cast<const char *>(mapped) + offset,
                    static_cast<size_t>(count));
        vkUnmapMemory(m_session.get_device(), mem_);
        return data;
    }

private:
    VulkanSession &m_session;

    VkBuffer buf_ = VK_NULL_HANDLE;
    VkDeviceMemory mem_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    uint32_t mem_type_idx_ = 0xFFFFFFFFu;
};
