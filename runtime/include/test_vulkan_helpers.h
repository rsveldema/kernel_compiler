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

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

#include <vulkan/vulkan_core.h>
#include <gtest/gtest.h>

// Forward declarations for functions implemented in test_vulkan_helpers.cc
std::string vk_result_str(VkResult rc);
void        check_vk(VkResult rc, const char* label);

// ───────────── VBuffer (include from vbuffer.hpp) ────────────
#include "vbuffer.hpp"

// ───────────── VulkanComputeSession: descriptor set layout, pipeline layout,
//               descriptor pool/set, command pool/buffer lifecycle ─────

class VulkanComputeSession
{
public:
    VkPipeline          pipeline  = VK_NULL_HANDLE;
    VkDescriptorSet     desc_set  = VK_NULL_HANDLE;
    VkPipelineLayout    pipe_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl       = VK_NULL_HANDLE;
    uint32_t            push_const_offset_ = 0;

    /* Create descriptor set layout (n SSBO bindings) */
    void init_descriptor_layout(VkDevice dev, uint32_t num_ssbo)
    {
        VkDescriptorSetLayoutBinding lsb{};
        lsb.binding     = 0;
        lsb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lsb.descriptorCount = num_ssbo;
        lsb.stageFlags  = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslici{};
        dslici.sType          = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslici.bindingCount   = 1;
        dslici.pBindings      = &lsb;

        check_vk(vkCreateDescriptorSetLayout(dev, &dslici, nullptr, &dsl), "VkComputeSession DSL");
    }

    /* Create pipeline layout (push constants) */
    void init_pipeline_layout(VkDevice dev, uint32_t push_size_bytes)
    {
        push_const_offset_ = 0;

        VkPushConstantRange pcr{};
        pcr.stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT;
        pcr.offset         = push_const_offset_;
        pcr.size           = push_size_bytes;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType               = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges  = &pcr;

        check_vk(vkCreatePipelineLayout(dev, &plci, nullptr, &pipe_layout_), "VkComputeSession pipeline layout");
    }

    /* Compile GLSL → SPIR-V and create compute pipeline */
    void create_pipeline(VkDevice dev, const std::string& glsl_file)
    {
        /* Read GLSL source */
        std::ifstream ifs(glsl_file);
        if (!ifs.is_open()) throw std::runtime_error("Cannot open GLSL file: " + glsl_file);

        /* Compile via glslc and pipe to temp file */
        const std::string tmp_spv = "/tmp/spv_gen_XXXXXX";
        int fd = mkstemp(const_cast<char*>(tmp_spv.data()) + 14);
        if (fd < 0) throw std::runtime_error("mkstemp failed");

        FILE* fp = popen(("glslc -x glsl-compute -O -fshader-stage=compute " + glsl_file + " -o -").c_str(), "r");
        if (!fp) { close(fd); throw std::runtime_error("glslc failed for: " + glsl_file); }

        char buf[4096];
        while (auto n = fread(buf, 1, sizeof(buf), fp))
            write(fd, buf, static_cast<size_t>(n));
        int rc_p = pclose(fp);
        if (rc_p != 0) { close(fd); throw std::runtime_error("glslc error for: " + glsl_file); }
        close(fd);

        /* Read back SPIR-V */
        std::ifstream ifs_spv(tmp_spv, std::ios::binary);
        if (!ifs_spv.is_open()) throw std::runtime_error("Cannot read generated SPIR-V");
        std::vector<uint32_t> spirv((std::istreambuf_iterator<char>(ifs_spv)),
                                     std::istreambuf_iterator<char>());

        /* Clean up temp file */
        remove(tmp_spv.c_str());

        if (spirv.empty()) throw std::runtime_error("Empty SPIR-V from glslc for: " + glsl_file);

        VkShaderModuleCreateInfo smci{};
        smci.sType             = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize          = spirv.size() * sizeof(uint32_t);
        smci.pCode             = spirv.data();

        check_vk(vkCreateShaderModule(dev, &smci, nullptr, &shader_module_), "VkComputeSession shader module");

        VkPipelineShaderStageCreateInfo psci{};
        psci.sType     = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        psci.stage     = VK_SHADER_STAGE_COMPUTE_BIT;
        psci.module    = shader_module_;
        psci.pName     = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType       = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage       = psci;
        cpci.layout      = pipe_layout_;

        check_vk(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline), "VkComputeSession pipeline");
    }

    /* Create command pool and allocate one primary command buffer */
    void init_command_pool(VkDevice dev, uint32_t queue_family_index)
    {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType         = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = queue_family_index;

        check_vk(vkCreateCommandPool(dev, &cpci, nullptr, &cmd_pool), "VkComputeSession cmd pool");

        VkCommandBufferAllocateInfo cai{};
        cai.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cai.commandPool      = cmd_pool;
        cai.level            = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check_vk(vkAllocateCommandBuffers(dev, &cai, &cmd_buf), "VkComputeSession cmd buf alloc");
    }

    /* Begin the command buffer (caller fills it in between begin/end) */
    VkCommandBuffer begin_command_buffer()
    {
        VkCommandBufferBeginInfo bbci{};
        bbci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        check_vk(vkBeginCommandBuffer(cmd_buf, &bbci), "VkComputeSession cmd buf begin");
        return cmd_buf;
    }

    /* Submit the command buffer and wait for completion. */
    void submit_and_wait(VkQueue queue, VkDevice dev)
    {
        vkEndCommandBuffer(cmd_buf);

        VkSubmitInfo si{};
        si.sType           = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers  = &cmd_buf;
        check_vk(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE), "VkComputeSession submit");
        check_vk(vkDeviceWaitIdle(dev), "VkComputeSession wait idle");
    }

    void destroy(VkDevice dev)
    {
        if (pipeline   != VK_NULL_HANDLE) { vkDestroyPipeline(dev, pipeline,      nullptr); pipeline   = VK_NULL_HANDLE; }
        if (cmd_buf    != VK_NULL_HANDLE) { vkFreeCommandBuffers(dev, cmd_pool, 1, &cmd_buf); cmd_buf   = VK_NULL_HANDLE; }
        if (cmd_pool   != VK_NULL_HANDLE) { vkDestroyCommandPool(dev, cmd_pool,  nullptr);     cmd_pool  = VK_NULL_HANDLE; }
        if (dsl        != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, dsl,         nullptr); dsl       = VK_NULL_HANDLE; }
        if (pipe_layout_ != VK_NULL_HANDLE){vkDestroyPipelineLayout(dev, pipe_layout_,  nullptr); pipe_layout_ = VK_NULL_HANDLE; }
    }

private:
    VkCommandPool     cmd_pool   = VK_NULL_HANDLE;
    VkCommandBuffer   cmd_buf    = VK_NULL_HANDLE;
    VkShaderModule    shader_module_ = VK_NULL_HANDLE;
};

// ───────────── VulkanComputeContext: bundles device ptr for RAII ────

class VulkanComputeContext
{
public:
    VulkanComputeContext(VkDevice dev, VkPhysicalDevice pdev, VkQueue q, uint32_t qi)
        : device_(dev), phys_dev_(pdev), queue_(q), queue_fi_(qi) {}

    void init_descriptor_layout(uint32_t num_ssbo) { session_.init_descriptor_layout(device_, num_ssbo); }
    void init_pipeline_layout(uint32_t push_size_bytes) { session_.init_pipeline_layout(device_, push_size_bytes); }
    void create_pipeline(const std::string& glsl_file) { session_.create_pipeline(device_, glsl_file); }
    void init_command_pool() { session_.init_command_pool(device_, queue_fi_); }

    void submit_and_wait() { session_.submit_and_wait(queue_, device_); }

    VulkanComputeSession& session() { return session_; }
    const VulkanComputeSession& session() const { return session_; }

    VkDescriptorSet& desc_set() { return desc_set_; }

private:
    VkDevice         device_;
    VkPhysicalDevice phys_dev_;
    VkQueue          queue_;
    uint32_t         queue_fi_;
    VulkanComputeSession session_;
    VkDescriptorSet  desc_set_ = VK_NULL_HANDLE;
};

// ───────────── VulkanTestBase fixture (creates instance + device) ───

class VulkanTestBase : public ::testing::Test
{
protected:
    void SetUp() override { init_vulkan_instance(); ASSERT_TRUE(has_device()) << "Vulkan device unavailable"; }
    void TearDown() override { destroy_vulkan(); }    bool has_device() const { return instance_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE; }

protected:
    VkInstance        instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice  phys_dev_ = VK_NULL_HANDLE;
    VkDevice          device_   = VK_NULL_HANDLE;
    VkQueue           queue_    = VK_NULL_HANDLE;
    uint32_t          queue_fi_ = 0xFFFFFFFFu;

private:
    void init_vulkan_instance()
    {
        const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
        const char* exts[]   = {
            VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        };

        VkApplicationInfo ai{};
        ai.sType           = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        ai.pApplicationName    = "kernel_compiler_tests";
        ai.apiVersion        = VK_API_VERSION_1_0;
        ai.pEngineName       = "kernel_compiler";
        ai.engineVersion     = VK_MAKE_VERSION(1, 0, 0);

        VkInstanceCreateInfo ici{};
        ici.sType                    = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo         = &ai;
        ici.enabledLayerCount        = 1;
        ici.ppEnabledLayerNames      = layers;
        ici.enabledExtensionCount    = 1;
        ici.ppEnabledExtensionNames  = exts;

        VkResult rc = vkCreateInstance(&ici, nullptr, &instance_);
        if (rc != VK_SUCCESS) {
            fprintf(stderr, "init_vulkan_instance: vkCreateInstance failed: %s\n", vk_result_str(rc).c_str());
            instance_ = VK_NULL_HANDLE; return;
        }

        /* Enumerate physical devices */
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance_, &count, nullptr);
        std::vector<VkPhysicalDevice> devs(count);
        if (count > 0) vkEnumeratePhysicalDevices(instance_, &count, devs.data());
        if (count == 0) {
            fprintf(stderr, "init_vulkan_instance: no Vulkan physical devices found\n");
            instance_ = VK_NULL_HANDLE; return;
        }
        phys_dev_ = devs[0];

        /* Get queue family */
        uint32_t qfc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_dev_, &qfc, nullptr);
        std::vector<VkQueueFamilyProperties> qfps(qfc > 0 ? qfc : 1);
        if (qfc > 0) vkGetPhysicalDeviceQueueFamilyProperties(phys_dev_, &qfc, qfps.data());
        for (uint32_t i = 0; i < qfc; ++i) {
            if (qfps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_fi_ = i;
                break;
            }
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo dqi{};
        dqi.sType              = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqi.queueFamilyIndex   = queue_fi_;
        dqi.queueCount         = 1;
        dqi.pQueuePriorities   = &prio;

        VkDeviceCreateInfo dci{};
        dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos    = &dqi;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = exts;

        rc = vkCreateDevice(phys_dev_, &dci, nullptr, &device_);
        if (rc != VK_SUCCESS) {
            fprintf(stderr, "init_vulkan_instance: vkCreateDevice failed: %s\n", vk_result_str(rc).c_str());
            instance_ = static_cast<VkInstance>(VK_NULL_HANDLE); device_ = static_cast<VkDevice>(VK_NULL_HANDLE); return;
        }

        vkGetDeviceQueue(device_, queue_fi_, 0, &queue_);
    }

    void destroy_vulkan()
    {
        if (device_)   vkDestroyDevice(device_,   nullptr); device_   = VK_NULL_HANDLE;
        if (instance_) vkDestroyInstance(instance_,nullptr); instance_ = VK_NULL_HANDLE;
    }
};
