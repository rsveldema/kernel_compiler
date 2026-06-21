/** @file test_vulkan_stubs.cc
 *  Unit tests that call all generated C++ stub dispatch functions using
 *  real Vulkan compute shaders, buffer allocations, and descriptor sets —
 *  with actual numerical verification of kernel outputs.
 */

#include <cstring>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

// Shared Vulkan test infrastructure
#include "vulkan_test_helpers.hpp"


// ───────────── Test: multi-arg (3 independent matmuls + accumulate) ──


#if 0
static void write_multi_arg_descriptors(
    VulkanComputeKernel& kernel,
    VkDevice device,
    const std::vector<VBaseDeviceBuffer*>& bufs)
{
    std::vector<VkDescriptorBufferInfo> dbis(bufs.size());
    for (uint32_t i = 0; i < bufs.size(); ++i) {
        dbis[i].buffer = bufs[i]->get();
        dbis[i].offset = 0;
        dbis[i].range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet wds{};
        wds.sType             = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet            = kernel.desc_set();
        wds.dstBinding        = i;
        wds.descriptorCount   = 1;
        wds.descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds.pBufferInfo       = &dbis[i];
        vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
    }
}
#endif

#if 0
static double dispatch_multi_arg_once(
    VulkanComputeKernel& kernel,
    VulkanSession& session,
    VBaseDeviceBuffer& c_buf,
    VBaseHostBuffer& zero_c,
    const VulkanDimension& dims)
{
    VulkanComputeContext ctx(session);
    c_buf.write(ctx, zero_c);

    auto cb = ctx.begin_command_buffer();
    const auto start = std::chrono::steady_clock::now();
    kernel.dispatch(cb, nullptr, 0, dims);
    ctx.submit_and_wait();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}
#endif

#if 0
static double dispatch_raw_kernel_once(
    VulkanComputeKernel& kernel,
    VulkanSession& session,
    VBaseDeviceBuffer& c_buf,
    VBaseHostBuffer& initial_c,
    const VulkanDimension& dims,
    void* push_constants,
    size_t push_constants_size)
{
    VulkanComputeContext ctx(session);
    c_buf.write(ctx, initial_c);

    auto cb = ctx.begin_command_buffer();
    const auto start = std::chrono::steady_clock::now();
    kernel.dispatch(cb, push_constants, push_constants_size, dims);
    ctx.submit_and_wait();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}
#endif

template <typename T>
static std::vector<float> read_float_buffer(VulkanComputeContext& context, VDeviceBuffer<T>& buf, VHostBuffer<T>& host)
{
    const VkDeviceSize size_bytes = host.size();
    std::vector<float> actual(static_cast<size_t>(size_bytes / sizeof(float)));
    buf.read(context, host);
    memcpy(actual.data(), host.get(), static_cast<size_t>(size_bytes));
    return actual;
}

#if 0
static bool selected_device_is_llvmpipe(VulkanSession& session)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(session.get_phys_device(), &props);
    return strstr(props.deviceName, "llvmpipe") != nullptr;
}

static bool selected_device_is_dzn(VulkanSession& session)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(session.get_phys_device(), &props);
    return strstr(props.deviceName, "dzn") != nullptr ||
           strstr(props.deviceName, "Direct3D12") != nullptr;
}

static std::string read_text_file(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
#endif

TEST_F(VulkanTestBase, host_device_buffer_copy_bandwidth)
{
    constexpr size_t size_bytes = 64ull * 1024ull * 1024ull;
    constexpr int iterations = 5;

    VHostBuffer<uint8_t[size_bytes]> upload(get_session());
    VHostBuffer<uint8_t[size_bytes]> download(get_session());
    VDeviceBuffer<uint8_t[size_bytes]> device(upload);
    VulkanComputeContext copy_context(get_session());

    for (VkDeviceSize i = 0; i < size_bytes; ++i) {
        upload.get()[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
    }
    std::memset(download.get(), 0, static_cast<size_t>(size_bytes));

    device.write(copy_context, upload);
    device.read(copy_context, download);
    ASSERT_EQ(std::memcmp(upload.get(), download.get(), static_cast<size_t>(size_bytes)), 0);

    double upload_ms = 1.0e30;
    double download_ms = 1.0e30;
    double roundtrip_ms = 1.0e30;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        device.write(copy_context, upload);
        auto mid = std::chrono::steady_clock::now();
        device.read(copy_context, download);
        auto end = std::chrono::steady_clock::now();

        upload_ms = std::min(upload_ms, std::chrono::duration<double, std::milli>(mid - start).count());
        download_ms = std::min(download_ms, std::chrono::duration<double, std::milli>(end - mid).count());
        roundtrip_ms = std::min(roundtrip_ms, std::chrono::duration<double, std::milli>(end - start).count());
    }

    ASSERT_EQ(std::memcmp(upload.get(), download.get(), static_cast<size_t>(size_bytes)), 0);

    const double gib = static_cast<double>(size_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double upload_gibs = gib / (upload_ms / 1000.0);
    const double download_gibs = gib / (download_ms / 1000.0);
    const double roundtrip_gibs = (2.0 * gib) / (roundtrip_ms / 1000.0);

    RecordProperty("size_bytes", static_cast<long long>(size_bytes));
    RecordProperty("upload_gibs", upload_gibs);
    RecordProperty("download_gibs", download_gibs);
    RecordProperty("roundtrip_gibs", roundtrip_gibs);
    std::cerr << "host/device buffer copy bandwidth: "
              << "upload=" << upload_gibs << " GiB/s"
              << ", download=" << download_gibs << " GiB/s"
              << ", roundtrip=" << roundtrip_gibs << " GiB/s"
              << " (" << size_bytes << " bytes, best of " << iterations << ")\n";
}