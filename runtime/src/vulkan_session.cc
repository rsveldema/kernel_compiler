/** @file test_vulkan_helpers.cc
 *  Test-only helper utilities — implementations for vk_result_str and
 *  check_vk used across the Vulkan test infrastructure.
 */

#include <string>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vulkan/vulkan_core.h>
#include <unistd.h>


#include <logging.hpp>


#if defined(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME) && \
    defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV)
#define RLLM_HAS_VK_COOPMAT2 1
#else
#define RLLM_HAS_VK_COOPMAT2 0
#endif

#include <vulkan_session.hpp>

// Forward declarations from test_vulkan_helpers.h
std::string vk_result_str(VkResult rc);
void        check_vk(VkResult rc, const char* label);

std::ofstream s_nn_log;

void set_nn_log_file(const std::string& filename)
{
    if (s_nn_log.is_open())
        s_nn_log.close();
    s_nn_log.open(filename);
}


VulkanQueue::VulkanQueue(VulkanSession &session, size_t index)
    : m_session(session)
{
    vkGetDeviceQueue(get_device(), session.get_queue_family_index(), index, &m_queue);
    init_command_pool();
}


void VulkanQueue::init_command_pool()
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


VulkanSession::VulkanSession(bool enable_cooperative_matrix2, const char* preferred_device_name)
{
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "kernel_compiler_tests";
#ifdef VK_API_VERSION_1_4
    ai.apiVersion = enable_cooperative_matrix2 ? VK_API_VERSION_1_4 : VK_API_VERSION_1_1;
#else
    ai.apiVersion = VK_API_VERSION_1_1;
#endif
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

    // Select device by explicit preference, then VULKAN_DEVICE env var. By default, prefer a non-llvmpipe
    // device when the loader exposes one, but still fall back to llvmpipe-only setups.
    m_phys_dev = devs[0]; /* fallback */
    const char *chosen = preferred_device_name ? preferred_device_name : getenv("VULKAN_DEVICE");
    if (chosen)
    {
        for (auto &dev : devs)
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(dev, &props);
            if (strstr(props.deviceName, chosen))
            {
                m_phys_dev = dev;
                LOG_INFO("init_vulkan_instance: selecting device \"%s\"\n", props.deviceName);
                break;
            }
        }
    }
    else
    {
        for (auto &dev : devs)
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(dev, &props);
            if (!device_name_contains(props.deviceName, "llvmpipe"))
            {
                LOG_INFO("USING DEV: {}", props.deviceName);
                m_phys_dev = dev;
                break;
            }
        }
    }

    VkPhysicalDeviceMaintenance3Properties maintenance3_props{};
    maintenance3_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &maintenance3_props;
    vkGetPhysicalDeviceProperties2(m_phys_dev, &props2);
    m_max_memory_allocation_size = maintenance3_props.maxMemoryAllocationSize;

    /* Get queue family */
    uint32_t qfc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_phys_dev, &qfc, nullptr);
    std::vector<VkQueueFamilyProperties> qfps(qfc > 0 ? qfc : 1);
    if (qfc > 0)
        vkGetPhysicalDeviceQueueFamilyProperties(m_phys_dev, &qfc, qfps.data());
    bool found = false;
    for (uint32_t i = 0; i < qfc; ++i)
    {
        if (qfps[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            found = true;
            m_queue_fi = i;
            m_queue_count = std::min<size_t>(MAX_QUEUES, std::max<uint32_t>(1, qfps[i].queueCount));
            break;
        }
    }

    if (! found)
    {
        LOG_ERROR("failed to find a queue family for compute!");
        std::abort();
    }

    LOG_INFO("NUM queues supported = {}", m_queue_count);

    std::vector<float> prios(m_queue_count, 1.0f);
    VkDeviceQueueCreateInfo dqi{};
    dqi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dqi.queueFamilyIndex = m_queue_fi;
    dqi.queueCount = static_cast<uint32_t>(m_queue_count);
    dqi.pQueuePriorities = prios.data();

    std::vector<const char*> device_extensions;
    void* device_pnext = nullptr;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
    VkPhysicalDevice16BitStorageFeatures storage_16bit_features{};
    VkPhysicalDeviceShaderFloat16Int8Features shader_float16_int8_features{};
#if RLLM_HAS_VK_COOPMAT2
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features{};
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features{};
#endif

    if (has_device_extension(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME))
    {
        atomic_float_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &atomic_float_features;
        vkGetPhysicalDeviceFeatures2(m_phys_dev, &features2);

        if (atomic_float_features.shaderBufferFloat32AtomicAdd)
        {
            device_extensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
            atomic_float_features.pNext = device_pnext;
            device_pnext = &atomic_float_features;
            m_shader_buffer_float32_atomic_add_enabled = true;
        }
    }

    if (has_device_extension(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME))
    {
        shader_float16_int8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &shader_float16_int8_features;
        vkGetPhysicalDeviceFeatures2(m_phys_dev, &features2);

        if (shader_float16_int8_features.shaderFloat16)
        {
            device_extensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
            shader_float16_int8_features.pNext = device_pnext;
            device_pnext = &shader_float16_int8_features;
        }
    }

    if (has_device_extension(VK_KHR_16BIT_STORAGE_EXTENSION_NAME))
    {
        storage_16bit_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &storage_16bit_features;
        vkGetPhysicalDeviceFeatures2(m_phys_dev, &features2);

        if (storage_16bit_features.storageBuffer16BitAccess)
        {
            device_extensions.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
            storage_16bit_features.pNext = device_pnext;
            device_pnext = &storage_16bit_features;
        }
    }

    if (enable_cooperative_matrix2)
    {
#if !RLLM_HAS_VK_COOPMAT2
        m_coopmat2_unavailable_reason =
            "Vulkan headers do not provide VK_NV_cooperative_matrix2 symbols";
#else
        if (!has_device_extension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) ||
            !has_device_extension(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME))
        {
            m_coopmat2_unavailable_reason =
                "device does not expose VK_KHR_cooperative_matrix and VK_NV_cooperative_matrix2";
        }
        else
        {
            coopmat2_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
            coopmat_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
            coopmat_features.pNext = &coopmat2_features;

            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &coopmat_features;
            vkGetPhysicalDeviceFeatures2(m_phys_dev, &features2);

            if (!coopmat_features.cooperativeMatrix ||
                !coopmat2_features.cooperativeMatrixWorkgroupScope ||
                !coopmat2_features.cooperativeMatrixTensorAddressing ||
                !coopmat2_features.cooperativeMatrixBlockLoads)
            {
                m_coopmat2_unavailable_reason =
                    "device exposes cooperative matrix extensions but not the required coopmat2 features";
                coopmat_features = {};
                coopmat2_features = {};
            }
            else
            {
                device_extensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
                device_extensions.push_back(VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
                coopmat2_features.pNext = device_pnext;
                device_pnext = &coopmat_features;
                m_coopmat2_enabled = true;
            }
        }
#endif
    }

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dqi;
    dci.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    dci.ppEnabledExtensionNames = device_extensions.empty() ? nullptr : device_extensions.data();
    dci.pNext = device_pnext;

    rc = vkCreateDevice(m_phys_dev, &dci, nullptr, &m_device);
    if (rc != VK_SUCCESS)
    {
        fprintf(stderr, "init_vulkan_instance: vkCreateDevice failed: %s\n", vk_result_str(rc).c_str());
        m_instance = static_cast<VkInstance>(VK_NULL_HANDLE);
        m_device = static_cast<VkDevice>(VK_NULL_HANDLE);
        return;
    }

    m_queues.reserve(m_queue_count);
    for (size_t i = 0; i < m_queue_count; ++i)
        m_queues.emplace_back(*this, i);
}


bool VulkanSession::device_name_contains(const char* device_name, const char* needle)
{
    if (!device_name || !needle)
        return false;

    const size_t needle_len = strlen(needle);
    if (needle_len == 0)
        return true;

    for (const char* pos = device_name; *pos; ++pos)
    {
        size_t i = 0;
        for (; i < needle_len && pos[i]; ++i)
        {
            const auto lhs = static_cast<unsigned char>(pos[i]);
            const auto rhs = static_cast<unsigned char>(needle[i]);
            if (std::tolower(lhs) != std::tolower(rhs))
                break;
        }
        if (i == needle_len)
            return true;
    }
    return false;
}

bool VulkanSession::has_device_extension(const char* name) const
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(m_phys_dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    if (count > 0)
        vkEnumerateDeviceExtensionProperties(m_phys_dev, nullptr, &count, extensions.data());
    for (const auto& ext : extensions)
    {
        if (strcmp(ext.extensionName, name) == 0)
            return true;
    }
    return false;
}

void VulkanComputeKernel::create_pipeline(const std::string &glsl_file)
{
    std::vector<uint32_t> spirv;
    if (glsl_file.size() >= 4 && glsl_file.ends_with(".spv"))
    {
        std::ifstream ifs_spv(glsl_file, std::ios::binary);
        if (!ifs_spv.is_open())
            {
                std::fprintf(stderr, "Cannot open SPIR-V file: %s\n", glsl_file.c_str());
                std::abort();
            }
        std::vector<char> bytes{
            std::istreambuf_iterator<char>(ifs_spv),
            std::istreambuf_iterator<char>()};
        if (bytes.size() % sizeof(uint32_t) != 0)
        {
            std::fprintf(stderr, "Invalid SPIR-V byte size for: %s\n", glsl_file.c_str());
            std::abort();
        }
        spirv.resize(bytes.size() / sizeof(uint32_t));
        std::memcpy(spirv.data(), bytes.data(), bytes.size());
    }
    else
    {
        /* Compile GLSL via glslc and pipe to temp file. */
        char tmp_spv[] = "/tmp/spv_gen_XXXXXX";
        int fd = mkstemp(tmp_spv);
        if (fd < 0)
        {
            std::fprintf(stderr, "mkstemp failed\n");
            std::abort();
        }

        std::string glslc_cmd = "glslc -x glsl -O -fshader-stage=compute ";
        if (glsl_file.find("coopmat2") != std::string::npos)
            glslc_cmd += "--target-env=vulkan1.4 ";
        glslc_cmd += glsl_file + " -o -";
        FILE *fp = popen(glslc_cmd.c_str(), "r");
        if (!fp)
        {
            close(fd);
            std::fprintf(stderr, "glslc failed for: %s\n", glsl_file.c_str());
            std::abort();
        }

        char buf[4096];
        while (auto n = fread(buf, 1, sizeof(buf), fp)) {
            const auto wrote = write(fd, buf, static_cast<size_t>(n));
            if (wrote != static_cast<ssize_t>(n))
            {
                close(fd);
                pclose(fp);
                std::fprintf(stderr, "failed to write temporary SPIR-V file\n");
                std::abort();
            }
        }

        int rc_p = pclose(fp);
        if (rc_p != 0)
        {
            close(fd);
            std::fprintf(stderr, "glslc error for: %s\n", glsl_file.c_str());
            std::abort();
        }
        close(fd);

        std::ifstream ifs_spv(tmp_spv, std::ios::binary);
        if (!ifs_spv.is_open())
            {
                std::fprintf(stderr, "Cannot read generated SPIR-V\n");
                std::abort();
            }
        std::vector<char> bytes{
            std::istreambuf_iterator<char>(ifs_spv),
            std::istreambuf_iterator<char>()};
        if (bytes.size() % sizeof(uint32_t) != 0)
        {
            close(fd);
            std::fprintf(stderr, "Invalid generated SPIR-V byte size\n");
            std::abort();
        }
        spirv.resize(bytes.size() / sizeof(uint32_t));
        std::memcpy(spirv.data(), bytes.data(), bytes.size());

        remove(tmp_spv);
    }

    if (spirv.empty())
        {
            std::fprintf(stderr, "Empty SPIR-V from glslc for: %s\n", glsl_file.c_str());
            std::abort();
        }

    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode = spirv.data();

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

// ── Implementation of vk_result_str ────────────────────────────────

std::string vk_result_str(VkResult rc)
{
    switch (rc) {
#define X(v) case v: return #v;
        X(VK_SUCCESS)                    X(VK_NOT_READY)       X(VK_TIMEOUT)
        X(VK_EVENT_SET)                  X(VK_INCOMPLETE)      X(VK_ERROR_OUT_OF_HOST_MEMORY)
        X(VK_ERROR_OUT_OF_DEVICE_MEMORY) X(VK_ERROR_INITIALIZATION_FAILED)
        X(VK_ERROR_DEVICE_LOST)          X(VK_ERROR_MEMORY_MAP_FAILED)
        X(VK_ERROR_LAYER_NOT_PRESENT)    X(VK_ERROR_EXTENSION_NOT_PRESENT)
        X(VK_ERROR_FEATURE_NOT_PRESENT)  X(VK_ERROR_INCOMPATIBLE_DRIVER)
        X(VK_ERROR_TOO_MANY_OBJECTS)     X(VK_ERROR_FORMAT_NOT_SUPPORTED)
        X(VK_ERROR_FRAGMENTED_POOL)      X(VK_ERROR_UNKNOWN)
#undef X
        default: return "VK_RESULT(" + std::to_string(rc) + ")";
    }
}

// ── Implementation of check_vk ─────────────────────────────────────

void check_vk(VkResult rc, const char* label)
{
    if (rc != VK_SUCCESS)
    {
        std::fprintf(stderr, "%s: %s\n", label, vk_result_str(rc).c_str());
        std::abort();
    }
}
