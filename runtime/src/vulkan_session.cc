/** @file test_vulkan_helpers.cc
 *  Test-only helper utilities — implementations for vk_result_str and
 *  check_vk used across the Vulkan test infrastructure.
 */

#include <string>
#include <stdexcept>
#include <cctype>
#include <fstream>
#include <iterator>
#include <vulkan/vulkan_core.h>
#include <unistd.h>

#include <vulkan_session.hpp>

// Forward declarations from test_vulkan_helpers.h
std::string vk_result_str(VkResult rc);
void        check_vk(VkResult rc, const char* label);

VulkanSession::VulkanSession(bool enable_cooperative_matrix2, const char* preferred_device_name)
{
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "kernel_compiler_tests";
    ai.apiVersion = enable_cooperative_matrix2 ? VK_API_VERSION_1_4 : VK_API_VERSION_1_1;
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
                fprintf(stderr, "init_vulkan_instance: selecting device \"%s\"\n", props.deviceName);
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

    std::vector<const char*> device_extensions;
    void* device_pnext = nullptr;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float_features{};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat_features{};
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV coopmat2_features{};

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

    if (enable_cooperative_matrix2)
    {
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

    vkGetDeviceQueue(m_device, m_queue_fi, 0, &m_queue);
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
    std::vector<uint8_t> spirv;
    if (glsl_file.size() >= 4 && glsl_file.ends_with(".spv"))
    {
        std::ifstream ifs_spv(glsl_file, std::ios::binary);
        if (!ifs_spv.is_open())
            throw std::runtime_error("Cannot open SPIR-V file: " + glsl_file);
        spirv.assign(
            std::istreambuf_iterator<char>(ifs_spv),
            std::istreambuf_iterator<char>());
    }
    else
    {
        /* Compile GLSL via glslc and pipe to temp file. */
        char tmp_spv[] = "/tmp/spv_gen_XXXXXX";
        int fd = mkstemp(tmp_spv);
        if (fd < 0)
            throw std::runtime_error("mkstemp failed");

        std::string glslc_cmd = "glslc -x glsl -O -fshader-stage=compute ";
        if (glsl_file.find("coopmat2") != std::string::npos)
            glslc_cmd += "--target-env=vulkan1.4 ";
        glslc_cmd += glsl_file + " -o -";
        FILE *fp = popen(glslc_cmd.c_str(), "r");
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

        std::ifstream ifs_spv(tmp_spv, std::ios::binary);
        if (!ifs_spv.is_open())
            throw std::runtime_error("Cannot read generated SPIR-V");
        spirv.assign(
            std::istreambuf_iterator<char>(ifs_spv),
            std::istreambuf_iterator<char>());

        remove(tmp_spv);
    }

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
        throw std::runtime_error(std::string(label) + ": " + vk_result_str(rc));
}
