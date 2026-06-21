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
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstdlib>
#include <cstdio>

#include <vulkan/vulkan_core.h>

// Forward declarations for functions implemented in test_vulkan_helpers.cc
std::string vk_result_str(VkResult rc);
void check_vk(VkResult rc, const char *label);

struct VulkanDimension
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

struct VStatistics
{
    std::string kernel_name;
    size_t host_to_device = 0;
    size_t device_to_host = 0;
    size_t host_to_device_bytes = 0;
    size_t device_to_host_bytes = 0;
};

class AbstractKernel;

class ComputeKernelRegistry
{
public:
    class ScopedActiveKernel
    {
    public:
        explicit ScopedActiveKernel(AbstractKernel& kernel);
        ~ScopedActiveKernel();

        ScopedActiveKernel(const ScopedActiveKernel&) = delete;
        ScopedActiveKernel& operator=(const ScopedActiveKernel&) = delete;

    private:
        AbstractKernel* m_previous = nullptr;
    };

    static ComputeKernelRegistry& instance()
    {
        static ComputeKernelRegistry registry;
        return registry;
    }

    void registerKernel(AbstractKernel& kernel)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (std::find(m_kernels.begin(), m_kernels.end(), &kernel) == m_kernels.end())
        {
            m_kernels.push_back(&kernel);
            logKernelRegistrationLocked(kernel);
        }
    }

    void recordHostToDevice(AbstractKernel* kernel, size_t bytes);
    void recordDeviceToHost(AbstractKernel* kernel, size_t bytes);

    void recordHostToDevice(std::string_view kernel_name, size_t bytes);
    void recordDeviceToHost(std::string_view kernel_name, size_t bytes);
    void resetStatistics();
    void enableRegistrationLog(const std::string& filename);

    std::vector<VStatistics> getStatistics(int top_users)
    {
        std::vector<VStatistics> result;
        std::lock_guard<std::mutex> lock(m_mutex);
        result.reserve(m_kernels.size());
        for (const auto* kernel : m_kernels)
            result.push_back(statisticsFor(*kernel));

        std::sort(result.begin(), result.end(), [](const VStatistics& lhs, const VStatistics& rhs) {
            const size_t lhs_count = lhs.host_to_device + lhs.device_to_host;
            const size_t rhs_count = rhs.host_to_device + rhs.device_to_host;
            if (lhs_count != rhs_count)
                return lhs_count > rhs_count;

            const size_t lhs_bytes = lhs.host_to_device_bytes + lhs.device_to_host_bytes;
            const size_t rhs_bytes = rhs.host_to_device_bytes + rhs.device_to_host_bytes;
            if (lhs_bytes != rhs_bytes)
                return lhs_bytes > rhs_bytes;

            return lhs.kernel_name < rhs.kernel_name;
        });

        if (top_users >= 0 && static_cast<size_t>(top_users) < result.size())
            result.resize(static_cast<size_t>(top_users));
        return result;
    }

    static AbstractKernel* activeKernel()
    {
        return activeKernelSlot();
    }

    static std::string_view activeKernelName();

private:
    static AbstractKernel*& activeKernelSlot()
    {
        thread_local AbstractKernel* kernel = nullptr;
        return kernel;
    }

    VStatistics statisticsFor(const AbstractKernel& kernel) const;
    void logKernelRegistrationLocked(const AbstractKernel& kernel);
    bool registrationWasLoggedLocked(const AbstractKernel& kernel) const;

    std::mutex m_mutex;
    std::vector<AbstractKernel*> m_kernels;
    std::vector<std::string> m_logged_kernel_names;
    std::string m_registration_log_filename;
};

enum class KernelDimension { OneD, TwoD, ThreeD };
enum class KernelType { Vector, Matrix, Triangular };

class AbstractKernel
{
public:
    explicit AbstractKernel(
        std::string kernel_name,
        KernelDimension dimension = KernelDimension::OneD,
        KernelType type = KernelType::Vector,
        std::string generated_descriptor = {})
        : m_kernel_name(std::move(kernel_name))
        , m_dimension(dimension)
        , m_type(type)
        , m_generated_descriptor(std::move(generated_descriptor))
    {
        ComputeKernelRegistry::instance().registerKernel(*this);
    }

    virtual ~AbstractKernel() = 0;

    const std::string& kernelName() const { return m_kernel_name; }
    const std::string& generatedDescriptor() const { return m_generated_descriptor; }

    KernelDimension dimension() const { return m_dimension; }
    KernelType type() const { return m_type; }

    static const char* dimensionStr(KernelDimension dim) {
        switch (dim) {
            case KernelDimension::OneD:   return "1D";
            case KernelDimension::TwoD:   return "2D";
            case KernelDimension::ThreeD: return "3D";
            default: return "?";
        }
    }

    static const char* typeStr(KernelType type) {
        switch (type) {
            case KernelType::Vector:      return "vector";
            case KernelType::Matrix:      return "matrix";
            case KernelType::Triangular:  return "triangular";
            default: return "?";
        }
    }

    // Demangle the kernel name into "{class}:{lineno}" format for display.
    // Mangled names have the format: "{namespace}::{class}|{display}"
    // where |{display} is an optional appended '{class}:{lineno}'.
    static std::string demangleKernelName(const std::string& mangled) {
        size_t sep = mangled.find('|');
        if (sep != std::string::npos)
            return mangled.substr(sep + 1);
        // Fallback: extract "{class}:{lineno}" from the class name portion
        size_t double_colon = mangled.find("::");
        if (double_colon == std::string::npos)
            return mangled;
        std::string cls = mangled.substr(double_colon + 2);
        // Find last digit-run suffix as lineno
        size_t last_digit_start = std::string::npos;
        for (size_t i = cls.size(); i > 0; --i) {
            if (!isdigit(cls[i - 1])) {
                last_digit_start = i;
                break;
            }
        }
        if (last_digit_start == std::string::npos || last_digit_start >= cls.size())
            return cls + ":?";
        // Strip trailing class name prefix that matches the stem
        size_t stem_end = 0;
        for (size_t i = last_digit_start; i > 0; --i) {
            if (isupper(cls[i - 1]) || isdigit(cls[i - 1])) {
                stem_end = i;
            } else {
                break;
            }
        }
        std::string name_part = cls.substr(stem_end, last_digit_start - stem_end);
        std::string line_part = cls.substr(last_digit_start);
        return name_part + ":" + line_part;
    }

    void recordHostToDevice(size_t bytes)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_statistics.host_to_device;
        m_statistics.host_to_device_bytes += bytes;
    }

    void recordDeviceToHost(size_t bytes)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_statistics.device_to_host;
        m_statistics.device_to_host_bytes += bytes;
    }

    void resetStatistics()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_statistics = {};
    }

    VStatistics statistics() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        VStatistics stats = m_statistics;
        stats.kernel_name = m_kernel_name;
        return stats;
    }

private:
    std::string m_kernel_name;
    KernelDimension m_dimension;
    KernelType m_type;
    std::string m_generated_descriptor;
    mutable std::mutex m_mutex;
    VStatistics m_statistics;
};

inline AbstractKernel::~AbstractKernel() = default;

inline ComputeKernelRegistry::ScopedActiveKernel::ScopedActiveKernel(AbstractKernel& kernel)
    : m_previous(activeKernelSlot())
{
    activeKernelSlot() = &kernel;
}

inline ComputeKernelRegistry::ScopedActiveKernel::~ScopedActiveKernel()
{
    activeKernelSlot() = m_previous;
}

inline void ComputeKernelRegistry::recordHostToDevice(AbstractKernel* kernel, size_t bytes)
{
    if (kernel != nullptr)
        kernel->recordHostToDevice(bytes);
}

inline void ComputeKernelRegistry::recordDeviceToHost(AbstractKernel* kernel, size_t bytes)
{
    if (kernel != nullptr)
        kernel->recordDeviceToHost(bytes);
}

inline void ComputeKernelRegistry::recordHostToDevice(std::string_view kernel_name, size_t bytes)
{
    if (kernel_name.empty())
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto* kernel : m_kernels)
    {
        if (kernel->kernelName() == kernel_name)
        {
            kernel->recordHostToDevice(bytes);
            return;
        }
    }
}

inline void ComputeKernelRegistry::recordDeviceToHost(std::string_view kernel_name, size_t bytes)
{
    if (kernel_name.empty())
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto* kernel : m_kernels)
    {
        if (kernel->kernelName() == kernel_name)
        {
            kernel->recordDeviceToHost(bytes);
            return;
        }
    }
}

inline void ComputeKernelRegistry::resetStatistics()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto* kernel : m_kernels)
        kernel->resetStatistics();
}

inline void ComputeKernelRegistry::enableRegistrationLog(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_registration_log_filename = filename;
    m_logged_kernel_names.clear();

    std::ofstream out(m_registration_log_filename, std::ios::trunc);
    out << "compute kernels\n";
    out << "name\ttiled\n";

    for (const auto* kernel : m_kernels)
        logKernelRegistrationLocked(*kernel);
}

inline std::string_view ComputeKernelRegistry::activeKernelName()
{
    const auto* kernel = activeKernel();
    return kernel == nullptr ? std::string_view{} : std::string_view{kernel->kernelName()};
}

inline VStatistics ComputeKernelRegistry::statisticsFor(const AbstractKernel& kernel) const
{
    return kernel.statistics();
}

inline bool ComputeKernelRegistry::registrationWasLoggedLocked(const AbstractKernel& kernel) const
{
    return std::find(m_logged_kernel_names.begin(), m_logged_kernel_names.end(), kernel.kernelName()) !=
        m_logged_kernel_names.end();
}

inline void ComputeKernelRegistry::logKernelRegistrationLocked(const AbstractKernel& kernel)
{
    if (m_registration_log_filename.empty() || registrationWasLoggedLocked(kernel))
        return;

    const std::string_view descriptor{kernel.generatedDescriptor()};
    std::ofstream out(m_registration_log_filename, std::ios::app);
    out << AbstractKernel::demangleKernelName(kernel.kernelName()) << '\t' << descriptor << '\n';
    m_logged_kernel_names.push_back(kernel.kernelName());
}


// ───────────── VulkanTestBase fixture (creates instance + device) ───

class VulkanSession
{
public:
    explicit VulkanSession(
        bool enable_cooperative_matrix2 = false,
        const char* preferred_device_name = nullptr);

    bool has_device() const { return m_instance != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE; }
    VkDevice get_device() const { return m_device; }
    VkQueue get_queue() const { return m_queue; }
    uint32_t get_queue_family_index() const { return m_queue_fi; }
    VkPhysicalDevice get_phys_device() const { return m_phys_dev; }
    VkDeviceSize maxMemoryAllocationSize() const { return m_max_memory_allocation_size; }
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
    VkDeviceSize m_max_memory_allocation_size = 0;
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
    VkCommandPool command_pool() const { return m_cmd_pool; }

private:
    /* Create command pool and allocate one primary command buffer */
    void init_command_pool()
    {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
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
        m_mutex.lock();
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
        check_vk(vkResetCommandPool(get_device(), m_cmd_pool, 0), "VkComputeSession reset cmd pool");
        m_mutex.unlock();
    }

    std::recursive_mutex& mutex() { return m_mutex; }

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
    std::recursive_mutex m_mutex;
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
    {
        std::fprintf(stderr, "%s: no suitable memory type\n", label);
        std::abort();
    }

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

    virtual ~VBaseHostBuffer()
    {
        destroy_resources();
    }

    virtual const char* typed_buffer_name() const = 0;

    uint8_t* get() { return mapped_; }
    const uint8_t* get() const { return mapped_; }
    VkDeviceSize size() const { return size_; }
    uint32_t mem_type_idx() const { return mem_type_idx_; }
    VulkanSession& session() { return m_session; }
    const VulkanSession& session() const { return m_session; }

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
    using value_type = std::conditional_t<std::is_array_v<T>, std::remove_extent_t<T>, T>;

    explicit VHostBuffer(VulkanSession& session)
        : VBaseHostBuffer(session, sizeof(T))
    {
    }

    const char* typed_buffer_name() const override { return "VHostBuffer"; }
    value_type* get() { return reinterpret_cast<value_type*>(VBaseHostBuffer::get()); }
    const value_type* get() const { return reinterpret_cast<const value_type*>(VBaseHostBuffer::get()); }
    value_type get(size_t index) const { return get()[index]; }
    void set(size_t index, value_type value) { get()[index] = value; }
    template <typename U>
    void fill(U value)
    {
        if constexpr (std::is_array_v<T>)
        {
            std::fill_n(get(), count(), value);
        }
        else if constexpr (requires(T& item) { item.data; })
        {
            using data_type = std::remove_reference_t<decltype(std::declval<T&>().data)>;
            std::fill_n(get()->data, std::extent_v<data_type>, value);
        }
        else
        {
            *get() = value;
        }
    }
    template <typename U>
    void fill(U value, size_t count)
    {
        fill_at(value, 0, count);
    }
    template <typename U>
    void fill_at(U value, size_t offset, size_t count)
    {
        if constexpr (std::is_array_v<T>)
        {
            assert(offset <= this->count());
            assert(count <= this->count() - offset);
            std::fill_n(get() + offset, count, value);
        }
        else if constexpr (requires(T& item) { item.data; })
        {
            using data_type = std::remove_reference_t<decltype(std::declval<T&>().data)>;
            using element_type = std::remove_extent_t<data_type>;
            const size_t capacity = static_cast<size_t>(size() / sizeof(element_type));
            if (offset > capacity || count > capacity - offset)
            {
                std::fprintf(stderr, "VHostBuffer::fill_at out of bounds\n");
                std::abort();
            }
            std::fill_n(get()->data + offset, count, value);
        }
        else
        {
            assert(offset <= 1);
            assert(count <= 1 - offset);
            if (count == 1)
                *get() = value;
        }
    }
    size_t count() const { return std::is_array_v<T> ? std::extent_v<T> : 1; }
};

class VDynamicHostBuffer : public VBaseHostBuffer
{
public:
    VDynamicHostBuffer(VulkanSession& session, VkDeviceSize size_bytes)
        : VBaseHostBuffer(session, size_bytes)
    {}

    const char* typed_buffer_name() const override { return "VDynamicHostBuffer"; }
    uint8_t* bytes() { return VBaseHostBuffer::get(); }
    const uint8_t* bytes() const { return VBaseHostBuffer::get(); }
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

    virtual ~VBaseDeviceBuffer()
    {
        destroy_resources();
    }

    virtual const char* typed_buffer_name() const = 0;

    VkBuffer get() const { return buf_; }
    VkDeviceSize size() const { return size_; }
    uint32_t mem_type_idx() const { return mem_type_idx_; }

    void write(VulkanComputeContext& context, VBaseHostBuffer& src, VkDeviceSize count = VK_WHOLE_SIZE, VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0)
    {
        std::lock_guard<std::recursive_mutex> lock(context.mutex());
        assert(src_offset <= src.size());
        assert(dst_offset <= size_);
        if (count == VK_WHOLE_SIZE)
            count = src.size() - src_offset;
        else
            assert(count <= src.size() - src_offset);
        assert(count <= size_ - dst_offset);
        src.flush();
        copy_buffer(context.command_pool(), src.vk_buffer(), buf_, src_offset, dst_offset, count);
    }

    void read(VulkanComputeContext& context, VBaseHostBuffer& dst, VkDeviceSize count = VK_WHOLE_SIZE, VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0)
    {
        std::lock_guard<std::recursive_mutex> lock(context.mutex());
        assert(src_offset <= size_);
        assert(dst_offset <= dst.size());
        if (count == VK_WHOLE_SIZE)
            count = dst.size() - dst_offset;
        else
            assert(count <= size_ - src_offset);
        assert(count <= dst.size() - dst_offset);
        copy_buffer(context.command_pool(), buf_, dst.vk_buffer(), src_offset, dst_offset, count);
        dst.invalidate();
    }

private:
    void copy_buffer(
        VkCommandPool cmd_pool,
        VkBuffer src,
        VkBuffer dst,
        VkDeviceSize src_offset,
        VkDeviceSize dst_offset,
        VkDeviceSize count)
    {
        const VkDeviceSize max_copy_bytes = copy_chunk_size_bytes();
        VkDeviceSize copied = 0;
        while (copied < count)
        {
            const VkDeviceSize chunk = std::min(max_copy_bytes, count - copied);
            copy_buffer_chunk(
                cmd_pool,
                src,
                dst,
                src_offset + copied,
                dst_offset + copied,
                chunk);
            copied += chunk;
        }
    }

    VkDeviceSize copy_chunk_size_bytes() const
    {
        constexpr VkDeviceSize default_copy_chunk_bytes = 16ull * 1024ull * 1024ull;
        const char* env = std::getenv("RLLM_VULKAN_COPY_CHUNK_MB");
        if (!env || !*env)
            return default_copy_chunk_bytes;

        char* end = nullptr;
        const auto mb = std::strtoull(env, &end, 10);
        if (end == env || mb == 0)
            return default_copy_chunk_bytes;
        return static_cast<VkDeviceSize>(mb) * 1024ull * 1024ull;
    }

    void copy_buffer_chunk(
        VkCommandPool cmd_pool,
        VkBuffer src,
        VkBuffer dst,
        VkDeviceSize src_offset,
        VkDeviceSize dst_offset,
        VkDeviceSize count)
    {
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
    explicit VDeviceBuffer(VHostBuffer<T>& host)
        : VBaseDeviceBuffer(host.session(), host.size())
    {
    }

    const char* typed_buffer_name() const override { return "VDeviceBuffer"; }

    void write(VulkanComputeContext& context, VHostBuffer<T>& src, VkDeviceSize count = VK_WHOLE_SIZE, VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0)
    {
        VBaseDeviceBuffer::write(context, src, count, src_offset, dst_offset);
    }

    void read(VulkanComputeContext& context, VHostBuffer<T>& dst, VkDeviceSize count = VK_WHOLE_SIZE, VkDeviceSize src_offset = 0, VkDeviceSize dst_offset = 0)
    {
        VBaseDeviceBuffer::read(context, dst, count, src_offset, dst_offset);
    }
};

class VDynamicDeviceBuffer : public VBaseDeviceBuffer
{
public:
    VDynamicDeviceBuffer(VulkanSession& session, VkDeviceSize size_bytes)
        : VBaseDeviceBuffer(session, size_bytes)
    {}

    const char* typed_buffer_name() const override { return "VDynamicDeviceBuffer"; }
};
