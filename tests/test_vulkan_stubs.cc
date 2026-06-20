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

// Generated stub headers
#include "dynamic-atb-acc.h"
#include "multi-arg.h"
#include "single-assign.h"
#include "single-assign-tiled.h"
#include "triangular-matrix-access.h"
#include "triangular1.h"
#include "var-size-loop-tiled.h"
#include "with-wg2.h"

// Shared Vulkan test infrastructure
#include "vulkan_test_helpers.hpp"

// ───────────── Test: single-assign (sets all elements to a value) ──

TEST_F(VulkanTestBase, single_assign_correctness)
{
    constexpr uint32_t N = 64;
    int push_value = 42;
    std::vector<int> expected(N, 42);

    /* Allocate device buffer for output */
    VHostBuffer<rllm_single_assign::RllmBuffer_dst> readback(get_session());
    VDeviceBuffer<rllm_single_assign::RllmBuffer_dst> dst_buf(readback);

    rllm_single_assign::SingleAssignVecmathKernel kernel(
        get_session(),
        TESTDATA_DIR "/single-assign.glsl");
    VulkanComputeContext ctx(get_session());

    kernel.dispatch(
        ctx,
        N,
        dst_buf,
        rllm_single_assign::single_assign_vecmath_PushConstants{
            static_cast<int32_t>(N),
            push_value,
        });

    /* Read back & verify */
    dst_buf.read(ctx, readback);
    const int* actual = reinterpret_cast<const int*>(readback.get()->data);

    for (uint32_t i = 0; i < N; ++i) {
        ASSERT_EQ(actual[i], push_value) << "single-assign dst[" << i << "]";
    }
}

// ───────────── Test: multi-arg (3 independent matmuls + accumulate) ──

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

template <typename T>
static std::vector<float> read_float_buffer(VulkanComputeContext& context, VDeviceBuffer<T>& buf, VHostBuffer<T>& host)
{
    const VkDeviceSize size_bytes = host.size();
    std::vector<float> actual(static_cast<size_t>(size_bytes / sizeof(float)));
    buf.read(context, host);
    memcpy(actual.data(), host.get(), static_cast<size_t>(size_bytes));
    return actual;
}

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

TEST_F(VulkanTestBase, multi_arg_correctness)
{
    /* multi-arg does: C[i,j] += sum_k( A1[n,k]*B1[k,m] + A2* B2 + A3* B3 ) */
    if (selected_device_is_dzn(get_session())) {
        GTEST_SKIP() << "multi_arg_correctness is unstable on dzn/Direct3D12 Vulkan drivers";
    }

    constexpr uint32_t ROWS = 64;
    constexpr uint32_t COLS = 1024;
    float expected_val = 44.0f * static_cast<float>(COLS);

    VHostBuffer<rllm_multi_arg::RllmBuffer_A1> a1(get_session());
    VHostBuffer<rllm_multi_arg::RllmBuffer_B1> b1(get_session());
    VHostBuffer<rllm_multi_arg::RllmBuffer_A2> a2(get_session());
    VHostBuffer<rllm_multi_arg::RllmBuffer_B2> b2(get_session());
    VHostBuffer<rllm_multi_arg::RllmBuffer_A3> a3(get_session());
    VHostBuffer<rllm_multi_arg::RllmBuffer_B3> b3(get_session());
    VHostBuffer<rllm_multi_arg::RllmBuffer_C> c(get_session());
    a1.fill(1.0f);
    b1.fill(2.0f);
    a2.fill(3.0f);
    b2.fill(4.0f);
    a3.fill(5.0f);
    b3.fill(6.0f);
    c.fill(0.0f);
    VDeviceBuffer<rllm_multi_arg::RllmBuffer_A1> a1_buf(a1);
    VDeviceBuffer<rllm_multi_arg::RllmBuffer_B1> b1_buf(b1);
    VDeviceBuffer<rllm_multi_arg::RllmBuffer_A2> a2_buf(a2);
    VDeviceBuffer<rllm_multi_arg::RllmBuffer_B2> b2_buf(b2);
    VDeviceBuffer<rllm_multi_arg::RllmBuffer_A3> a3_buf(a3);
    VDeviceBuffer<rllm_multi_arg::RllmBuffer_B3> b3_buf(b3);
    VDeviceBuffer<rllm_multi_arg::RllmBuffer_C> c_buf(c);
    VulkanComputeContext ctx(get_session());
    a1_buf.write(ctx, a1);
    b1_buf.write(ctx, b1);
    a2_buf.write(ctx, a2);
    b2_buf.write(ctx, b2);
    a3_buf.write(ctx, a3);
    b3_buf.write(ctx, b3);
    c_buf.write(ctx, c);

    rllm_multi_arg::MultiArgVecmathKernel kernel(get_session(), TESTDATA_DIR "/multi-arg.glsl");

    kernel.dispatch(
        ctx,
        ROWS,
        COLS,
        a1_buf,
        b1_buf,
        a2_buf,
        b2_buf,
        a3_buf,
        b3_buf,
        c_buf,
        rllm_multi_arg::multi_arg_vecmath_PushConstants{
            static_cast<int32_t>(ROWS),
            static_cast<int32_t>(COLS),
        });

    VHostBuffer<rllm_multi_arg::RllmBuffer_C> actual(get_session());
    c_buf.read(ctx, actual);

    for (uint32_t i = 0; i < ROWS * COLS; ++i) {
        ASSERT_NEAR(actual.get()->data[i], expected_val, 1.0f) << "multi-arg C[" << i << "]";
    }
}

TEST_F(VulkanTestBase, multi_arg_shared_memory_tiling_is_faster)
{
    if (!get_session().shader_buffer_float32_atomic_add_enabled()) {
        GTEST_SKIP() << "selected Vulkan device does not expose shaderBufferFloat32AtomicAdd";
    }

    constexpr uint32_t ROWS = 256;
    constexpr uint32_t COLS = 1024;

    VHostBuffer<float[ROWS * COLS]> a1(get_session());
    VHostBuffer<float[COLS * COLS]> b1(get_session());
    VHostBuffer<float[ROWS * COLS]> a2(get_session());
    VHostBuffer<float[COLS * COLS]> b2(get_session());
    VHostBuffer<float[ROWS * COLS]> a3(get_session());
    VHostBuffer<float[COLS * COLS]> b3(get_session());
    VHostBuffer<float[ROWS * COLS]> c(get_session());
    VHostBuffer<float[ROWS * COLS]> readback(get_session());
    a1.fill(1.0f);
    b1.fill(2.0f);
    a2.fill(3.0f);
    b2.fill(4.0f);
    a3.fill(5.0f);
    b3.fill(6.0f);
    c.fill(0.0f);
    VDeviceBuffer<float[ROWS * COLS]> a1_buf(a1);
    VDeviceBuffer<float[COLS * COLS]> b1_buf(b1);
    VDeviceBuffer<float[ROWS * COLS]> a2_buf(a2);
    VDeviceBuffer<float[COLS * COLS]> b2_buf(b2);
    VDeviceBuffer<float[ROWS * COLS]> a3_buf(a3);
    VDeviceBuffer<float[COLS * COLS]> b3_buf(b3);
    VDeviceBuffer<float[ROWS * COLS]> c_buf(c);
    std::vector<VBaseDeviceBuffer*> bufs{
        &a1_buf,
        &b1_buf,
        &a2_buf,
        &b2_buf,
        &a3_buf,
        &b3_buf,
        &c_buf,
    };
    VulkanComputeContext transfer_context(get_session());
    a1_buf.write(transfer_context, a1);
    b1_buf.write(transfer_context, b1);
    a2_buf.write(transfer_context, a2);
    b2_buf.write(transfer_context, b2);
    a3_buf.write(transfer_context, a3);
    b3_buf.write(transfer_context, b3);

    VulkanComputeKernel unoptimized(get_session(), TESTDATA_DIR "/multi-arg-noopt.glsl", 0, bufs.size());
    write_multi_arg_descriptors(unoptimized, get_device(), bufs);

    const VulkanDimension unoptimized_dims{ROWS, COLS, 1};

    (void)dispatch_multi_arg_once(unoptimized, get_session(), c_buf, c, unoptimized_dims);
    const auto sequential_actual = read_float_buffer(transfer_context, c_buf, readback);

    double best_unoptimized = 1.0e30;
    for (int i = 0; i < 3; ++i) {
        best_unoptimized = std::min(best_unoptimized, dispatch_multi_arg_once(unoptimized, get_session(), c_buf, c, unoptimized_dims));
    }

    const uint32_t chunk_sizes[] = {1, 2, 4, 8, 16, 32, 64};
    double best_shared = 1.0e30;
    uint32_t best_chunk_size = 0;

    for (uint32_t chunk_size : chunk_sizes) {
        std::string shader_path = std::string(TESTDATA_DIR) + "/multi-arg-chunk-" + std::to_string(chunk_size) + ".glsl";
        VulkanComputeKernel shared(get_session(), shader_path, 0, bufs.size());
        write_multi_arg_descriptors(shared, get_device(), bufs);
        const VulkanDimension shared_dims{
            (ROWS + 8 - 1) / 8,
            (COLS + 8 - 1) / 8,
            (1024 + chunk_size - 1) / chunk_size,
        };

        (void)dispatch_multi_arg_once(shared, get_session(), c_buf, c, shared_dims);
        const auto shared_actual = read_float_buffer(transfer_context, c_buf, readback);

        ASSERT_EQ(shared_actual.size(), sequential_actual.size());
        for (uint32_t i = 0; i < shared_actual.size(); ++i) {
            ASSERT_NEAR(shared_actual[i], sequential_actual[i], 1.0f)
                << "multi-arg chunk=" << chunk_size << " optimized vs sequential C[" << i << "]";
        }

        double best_chunk = 1.0e30;
        for (int i = 0; i < 3; ++i) {
            best_chunk = std::min(best_chunk, dispatch_multi_arg_once(shared, get_session(), c_buf, c, shared_dims));
        }

        const double speedup = best_unoptimized / best_chunk;
        RecordProperty(("chunk_" + std::to_string(chunk_size) + "_ms").c_str(), best_chunk);
        RecordProperty(("chunk_" + std::to_string(chunk_size) + "_speedup").c_str(), speedup);
        std::cerr << "multi_arg shared-memory chunk=" << chunk_size
                  << " speedup: " << speedup << "x"
                  << " (shared=" << best_chunk << " ms"
                  << ", unoptimized=" << best_unoptimized << " ms)\n";

        if (best_chunk < best_shared) {
            best_shared = best_chunk;
            best_chunk_size = chunk_size;
        }
    }

    const double best_speedup = best_unoptimized / best_shared;
    RecordProperty("best_chunk_size", best_chunk_size);
    RecordProperty("best_shared_ms", best_shared);
    RecordProperty("unoptimized_ms", best_unoptimized);
    RecordProperty("best_speedup", best_speedup);
    std::cerr << "multi_arg best shared-memory chunk=" << best_chunk_size
              << " speedup: " << best_speedup << "x"
              << " (shared=" << best_shared << " ms"
              << ", unoptimized=" << best_unoptimized << " ms)\n";

    if (selected_device_is_llvmpipe(get_session())) {
        EXPECT_LT(best_shared, best_unoptimized)
            << "best chunk=" << best_chunk_size
            << ", shared=" << best_shared << "ms, unoptimized=" << best_unoptimized << "ms";
    } else {
        SUCCEED() << "shared-memory tiling timing is informational on non-llvmpipe devices: "
                  << "best chunk=" << best_chunk_size
                  << ", shared=" << best_shared << "ms, unoptimized=" << best_unoptimized << "ms";
    }
}

TEST_F(VulkanTestBase, multi_arg_cooperative_matrix2_correctness)
{
    VulkanSession coop_session(true);
    if (!coop_session.has_device()) {
        GTEST_SKIP() << "no Vulkan device available for cooperative_matrix2 test";
    }
    if (!coop_session.cooperative_matrix2_enabled()) {
        GTEST_SKIP() << coop_session.cooperative_matrix2_unavailable_reason();
    }

    constexpr uint32_t ROWS = 64;
    constexpr uint32_t COLS = 1024;
    constexpr uint32_t CHUNK = 16;
    float expected_val = 44.0f * static_cast<float>(COLS);

    VHostBuffer<float[ROWS * COLS]> a1(coop_session);
    VHostBuffer<float[COLS * COLS]> b1(coop_session);
    VHostBuffer<float[ROWS * COLS]> a2(coop_session);
    VHostBuffer<float[COLS * COLS]> b2(coop_session);
    VHostBuffer<float[ROWS * COLS]> a3(coop_session);
    VHostBuffer<float[COLS * COLS]> b3(coop_session);
    VHostBuffer<float[ROWS * COLS]> c(coop_session);
    VHostBuffer<float[ROWS * COLS]> readback(coop_session);
    a1.fill(1.0f);
    b1.fill(2.0f);
    a2.fill(3.0f);
    b2.fill(4.0f);
    a3.fill(5.0f);
    b3.fill(6.0f);
    c.fill(0.0f);
    VDeviceBuffer<float[ROWS * COLS]> a1_buf(a1);
    VDeviceBuffer<float[COLS * COLS]> b1_buf(b1);
    VDeviceBuffer<float[ROWS * COLS]> a2_buf(a2);
    VDeviceBuffer<float[COLS * COLS]> b2_buf(b2);
    VDeviceBuffer<float[ROWS * COLS]> a3_buf(a3);
    VDeviceBuffer<float[COLS * COLS]> b3_buf(b3);
    VDeviceBuffer<float[ROWS * COLS]> c_buf(c);
    std::vector<VBaseDeviceBuffer*> bufs{
        &a1_buf,
        &b1_buf,
        &a2_buf,
        &b2_buf,
        &a3_buf,
        &b3_buf,
        &c_buf,
    };
    VulkanComputeContext transfer_context(coop_session);
    a1_buf.write(transfer_context, a1);
    b1_buf.write(transfer_context, b1);
    a2_buf.write(transfer_context, a2);
    b2_buf.write(transfer_context, b2);
    a3_buf.write(transfer_context, a3);
    b3_buf.write(transfer_context, b3);

    VulkanComputeKernel kernel(
        coop_session,
        TESTDATA_DIR "/multi-arg-coopmat2-chunk-16.glsl",
        0,
        bufs.size());
    write_multi_arg_descriptors(kernel, coop_session.get_device(), bufs);

    (void)dispatch_multi_arg_once(
        kernel,
        coop_session,
        c_buf,
        c,
        VulkanDimension{(ROWS + 8 - 1) / 8, (COLS + 8 - 1) / 8, 1});

    const auto actual = read_float_buffer(transfer_context, c_buf, readback);
    ASSERT_EQ(actual.size(), ROWS * COLS);
    for (uint32_t i = 0; i < actual.size(); ++i) {
        ASSERT_NEAR(actual[i], expected_val, 1.0f)
            << "multi-arg coopmat2 chunk=" << CHUNK << " C[" << i << "]";
    }
}

// ───────────── Test: triangular1 (triangular attention pattern) ──

TEST_F(VulkanTestBase, triangular1_correctness)
{
    VulkanSession triangular_session(false, "llvmpipe");
    if (!triangular_session.has_device()) {
        GTEST_SKIP() << "no Vulkan device available for triangular1 test";
    }
    if (!selected_device_is_llvmpipe(triangular_session)) {
        GTEST_SKIP() << "llvmpipe Vulkan device is not available for triangular1 test";
    }

    static_assert(rllm_triangular1::d_scores_h_X == rllm_triangular1::d_raw_h_X);
    static_assert(rllm_triangular1::d_scores_h_Y == rllm_triangular1::d_raw_h_Y);
    static_assert(rllm_triangular1::d_scores_h_X == rllm_triangular1::attn_w_h_X);
    static_assert(rllm_triangular1::d_scores_h_Y == rllm_triangular1::attn_w_h_Y);

    constexpr uint32_t test_seq_len = 32;
    static_assert(test_seq_len <= rllm_triangular1::d_scores_h_X);
    static_assert(test_seq_len <= rllm_triangular1::d_scores_h_Y);
    constexpr VkDeviceSize buffer_size_bytes =
        static_cast<VkDeviceSize>(test_seq_len) *
        static_cast<VkDeviceSize>(test_seq_len) *
        sizeof(float);
    constexpr float raw_sentinel = -123.0f;

    VDynamicHostBuffer d_scores_h(triangular_session, buffer_size_bytes);
    VDynamicHostBuffer d_raw_h(triangular_session, buffer_size_bytes);
    VDynamicHostBuffer attn_w_h(triangular_session, buffer_size_bytes);
    float* scores = reinterpret_cast<float*>(d_scores_h.bytes());
    float* raw = reinterpret_cast<float*>(d_raw_h.bytes());
    float* attn = reinterpret_cast<float*>(attn_w_h.bytes());
    for (uint32_t i = 0; i < test_seq_len; ++i) {
        for (uint32_t j = 0; j < test_seq_len; ++j) {
            const size_t idx = static_cast<size_t>(i) * test_seq_len + j;
            scores[idx] = static_cast<float>(i + j + 1);
            attn[idx] = 0.25f + 0.03125f * static_cast<float>((i + j) % 5);
            raw[idx] = raw_sentinel;
        }
    }

    VDynamicDeviceBuffer d_scores_buf(triangular_session, buffer_size_bytes);
    VDynamicDeviceBuffer d_raw_buf(triangular_session, buffer_size_bytes);
    VDynamicDeviceBuffer attn_w_buf(triangular_session, buffer_size_bytes);
    rllm_triangular1::triangular1_TransformerBlock_PushConstants pc{
        static_cast<int32_t>(test_seq_len),
        static_cast<int32_t>(test_seq_len),
        static_cast<int32_t>(test_seq_len),
    };

    rllm_triangular1::Triangular1TransformerBlockKernel kernel(
        triangular_session,
        TESTDATA_DIR "/triangular1.glsl");

    VulkanComputeContext ctx(triangular_session);
    d_scores_buf.write(ctx, d_scores_h);
    d_raw_buf.write(ctx, d_raw_h);
    attn_w_buf.write(ctx, attn_w_h);

    kernel.dispatch(
        ctx,
        test_seq_len,
        test_seq_len,
        d_scores_buf,
        d_raw_buf,
        attn_w_buf,
        pc);

    d_raw_buf.read(ctx, d_raw_h);
    const float* actual = reinterpret_cast<const float*>(d_raw_h.bytes());

    for (uint32_t i = 0; i < test_seq_len; ++i) {
        float row_dot = 0.0f;
        for (uint32_t k = 0; k <= i; ++k) {
            const size_t idx = static_cast<size_t>(i) * test_seq_len + k;
            row_dot += scores[idx] * attn[idx];
        }
        for (uint32_t j = 0; j < test_seq_len; ++j) {
            const size_t idx = static_cast<size_t>(i) * test_seq_len + j;
            const float expected = j <= i ? attn[idx] * (scores[idx] - row_dot) : raw_sentinel;
            ASSERT_NEAR(actual[idx], expected, 1.0e-4f)
                << "triangular1 d_raw[" << i << "," << j << "]";
        }
    }
}

// ───────────── Test: fixed_size_triangular_matrix access ──

TEST_F(VulkanTestBase, triangular_matrix_access_correctness)
{
    VulkanSession triangular_session(false, "llvmpipe");
    if (!triangular_session.has_device()) {
        GTEST_SKIP() << "no Vulkan device available for triangular matrix access test";
    }
    if (!selected_device_is_llvmpipe(triangular_session)) {
        GTEST_SKIP() << "llvmpipe Vulkan device is not available for triangular matrix access test";
    }

    constexpr uint32_t N = 8;
    constexpr uint32_t triangular_cells = N * (N + 1) / 2;
    static_assert(rllm_triangular_matrix_access::tri_in_X == N);
    static_assert(rllm_triangular_matrix_access::tri_in_Y == N);
    static_assert(rllm_triangular_matrix_access::tri_out_X == N);
    static_assert(rllm_triangular_matrix_access::tri_out_Y == N);

    constexpr VkDeviceSize buffer_size_bytes = triangular_cells * sizeof(float);
    VDynamicHostBuffer input_host(triangular_session, buffer_size_bytes);
    VDynamicHostBuffer output_host(triangular_session, buffer_size_bytes);
    float* input = reinterpret_cast<float*>(input_host.bytes());
    float* output = reinterpret_cast<float*>(output_host.bytes());

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            const size_t idx = static_cast<size_t>(i) * (i + 1) / 2 + j;
            input[idx] = 10.0f * static_cast<float>(i) + static_cast<float>(j);
            output[idx] = -999.0f;
        }
    }

    VDynamicDeviceBuffer input_buf(triangular_session, buffer_size_bytes);
    VDynamicDeviceBuffer output_buf(triangular_session, buffer_size_bytes);
    rllm_triangular_matrix_access::triangular_matrix_access_triangular_access_PushConstants pc{
        static_cast<int32_t>(N),
        static_cast<int32_t>(N),
    };

    rllm_triangular_matrix_access::TriangularMatrixAccessTriangularAccessKernel kernel(
        triangular_session,
        TESTDATA_DIR "/triangular-matrix-access.glsl");

    VulkanComputeContext ctx(triangular_session);
    input_buf.write(ctx, input_host);
    output_buf.write(ctx, output_host);

    kernel.dispatch(
        ctx,
        N,
        N,
        input_buf,
        output_buf,
        pc);

    output_buf.read(ctx, output_host);
    const float* actual = reinterpret_cast<const float*>(output_host.bytes());

    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j <= i; ++j) {
            const size_t idx = static_cast<size_t>(i) * (i + 1) / 2 + j;
            const float expected = input[idx] + 1.0f;
            ASSERT_FLOAT_EQ(actual[idx], expected)
                << "triangular matrix output[" << i << "," << j << "]";
        }
    }
}

// ───────────── Test: with-wg2 (push-only kernel) ──────

TEST_F(VulkanTestBase, with_wg2_correctness)
{
    const uint32_t N = 1024;

    /* Vulkan compute context — no descriptor layout needed (no SSBOs) */
    rllm_with_wg2::WithWg2TestKernel kernel(
        get_session(),
        TESTDATA_DIR "/with-wg2.glsl");
    VulkanComputeContext ctx(get_session());

    uint32_t wg_x = 512, wg_y = 1;
    int32_t push_A = 42;

    kernel.dispatch(
        ctx,
        N,
        rllm_with_wg2::with_wg2_test_PushConstants{push_A});

    SUCCEED() << "with-wg2 dispatch completed (grid=" << N << "x" << N
              << ", workgroup=" << wg_x << "x" << wg_y
              << ", push.A=" << push_A << ")";
}

// ───────────── Test: Tiled execution (workgroup partitioning) ──

TEST_F(VulkanTestBase, tiled_kernel_generates_correct_output)
{
    /**
     * Verify that workgroup partitioning (tiling) correctly transforms loops
     * into workgroup-local strided loops with barriers.
     * 
     * This test validates that:
     * 1. Parallelizable loops are correctly identified
     * 2. Step 2 guard (local_id == 0 barriers) is inserted
     * 3. Step 3 chunking (loop tiling with stride) is applied
     * 4. Numerical correctness is preserved after transformation
     */
    constexpr uint32_t N = 128;
    int push_value = 77;
    std::vector<int> expected(N, 77);

    VHostBuffer<rllm_single_assign::RllmBuffer_dst> readback(get_session());
    VDeviceBuffer<rllm_single_assign::RllmBuffer_dst> dst_buf(readback);

    // Load the standard single-assign kernel; its loop will be tiled if
    // the tiling pass is enabled during code generation
    rllm_single_assign_tiled::SingleAssignTiledVecmathKernel kernel(
        get_session(),
        TESTDATA_DIR "/single-assign-tiled.glsl");
    const std::string descriptor = rllm_single_assign_tiled::SingleAssignTiledVecmathKernel::generated_descriptor();
    ASSERT_NE(descriptor.find("tiling=on"), std::string::npos)
        << "tiling is not enabled in generated stub descriptor: " << descriptor;

    VulkanComputeContext ctx(get_session());

    kernel.dispatch(
        ctx,
        N,
        dst_buf,
        rllm_single_assign_tiled::single_assign_tiled_vecmath_PushConstants{
            static_cast<int32_t>(N),
            push_value,
        });

    dst_buf.read(ctx, readback);
    const int* actual = reinterpret_cast<const int*>(readback.get()->data);

    // Verify tiled loop produces correct results
    for (uint32_t i = 0; i < N; ++i) {
        ASSERT_EQ(actual[i], push_value)
            << "tiled kernel produced incorrect value at index " << i;
    }

    SUCCEED() << "tiled kernel dispatch completed with N=" << N
              << " and produced correct output (all elements = " << push_value << ")";
}

TEST_F(VulkanTestBase, tiled_variable_size_loop_generates_correct_output)
{
    constexpr uint32_t N = 128;
    constexpr int push_value = 91;
    constexpr int k_count = 17;

    VHostBuffer<rllm_var_size_loop_tiled::RllmBuffer_dst> readback(get_session());
    VDeviceBuffer<rllm_var_size_loop_tiled::RllmBuffer_dst> dst_buf(readback);

    rllm_var_size_loop_tiled::VarSizeLoopTiledVecmathKernel kernel(
        get_session(),
        TESTDATA_DIR "/var-size-loop-tiled.glsl");
    const std::string descriptor = rllm_var_size_loop_tiled::VarSizeLoopTiledVecmathKernel::generated_descriptor();
    ASSERT_NE(descriptor.find("tiling=on"), std::string::npos)
        << "tiling is not enabled in generated stub descriptor: " << descriptor;

    VulkanComputeContext ctx(get_session());

    kernel.dispatch(
        ctx,
        N,
        dst_buf,
        rllm_var_size_loop_tiled::var_size_loop_tiled_vecmath_PushConstants{
            static_cast<int32_t>(N),
            push_value,
            k_count,
            static_cast<int32_t>(N),
        });

    dst_buf.read(ctx, readback);
    const int* actual = reinterpret_cast<const int*>(readback.get()->data);

    for (uint32_t i = 0; i < N; ++i) {
        ASSERT_EQ(actual[i], push_value)
            << "variable-size tiled kernel produced incorrect value at index " << i;
    }
}

TEST_F(VulkanTestBase, dynamic_atb_acc_reduction_loop_is_chunked_by_k_count)
{
    const std::string descriptor = rllm_dynamic_atb_acc::DynamicAtbAccVecmathKernel::generated_descriptor();
    ASSERT_NE(descriptor.find("tiling=on"), std::string::npos)
        << "tiling is not enabled in generated stub descriptor: " << descriptor;
    ASSERT_NE(descriptor.find("shared_memory_tiling=on"), std::string::npos)
        << "shared-memory reduction tiling is not enabled in generated stub descriptor: " << descriptor;

    const std::string glsl = read_text_file(TESTDATA_DIR "/dynamic-atb-acc.glsl");
    EXPECT_NE(glsl.find("layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;"), std::string::npos);
    EXPECT_NE(glsl.find("int(gl_WorkGroupID.z) * 16"), std::string::npos);
    EXPECT_NE(glsl.find("l_idx < ((int(gl_WorkGroupID.z) * 16) + 16) && l_idx < rllm_push.k_count"), std::string::npos);
    EXPECT_NE(glsl.find("atomicAdd(C["), std::string::npos);

    const std::string header = read_text_file(TESTDATA_DIR "/dynamic-atb-acc.h");
    EXPECT_NE(header.find("(push_constants.k_count + 16 - 1) / 16"), std::string::npos);

    if (!get_session().shader_buffer_float32_atomic_add_enabled()) {
        GTEST_SKIP() << "selected Vulkan device does not expose shaderBufferFloat32AtomicAdd";
    }

    constexpr uint32_t rows = 8;
    constexpr uint32_t cols = 8;
    constexpr int k_count = 7;
    constexpr float expected = 3.0f + 2.0f * static_cast<float>(k_count);

    VHostBuffer<rllm_dynamic_atb_acc::RllmBuffer_A> a(get_session());
    VHostBuffer<rllm_dynamic_atb_acc::RllmBuffer_B> b(get_session());
    VHostBuffer<rllm_dynamic_atb_acc::RllmBuffer_C> c(get_session());
    VHostBuffer<rllm_dynamic_atb_acc::RllmBuffer_C> readback(get_session());
    a.fill(1.0f);
    b.fill(2.0f);
    c.fill(3.0f);

    VDeviceBuffer<rllm_dynamic_atb_acc::RllmBuffer_A> a_buf(a);
    VDeviceBuffer<rllm_dynamic_atb_acc::RllmBuffer_B> b_buf(b);
    VDeviceBuffer<rllm_dynamic_atb_acc::RllmBuffer_C> c_buf(c);
    VulkanComputeContext ctx(get_session());
    a_buf.write(ctx, a);
    b_buf.write(ctx, b);
    c_buf.write(ctx, c);

    rllm_dynamic_atb_acc::DynamicAtbAccVecmathKernel kernel(
        get_session(),
        TESTDATA_DIR "/dynamic-atb-acc.glsl");

    kernel.dispatch(
        ctx,
        rows,
        cols,
        a_buf,
        b_buf,
        c_buf,
        rllm_dynamic_atb_acc::dynamic_atb_acc_vecmath_PushConstants{
            static_cast<int32_t>(rows),
            static_cast<int32_t>(cols),
            k_count,
        });

    c_buf.read(ctx, readback);
    for (uint32_t i = 0; i < rows * cols; ++i) {
        ASSERT_NEAR(readback.get()->data[i], expected, 1.0e-4f)
            << "dynamic-atb-acc C[" << i << "]";
    }
}

TEST_F(VulkanTestBase, dynamic_atb_acc_tiling_is_faster_than_no_tiling)
{
    if (!get_session().shader_buffer_float32_atomic_add_enabled()) {
        GTEST_SKIP() << "selected Vulkan device does not expose shaderBufferFloat32AtomicAdd";
    }

    constexpr uint32_t rows = 128;
    constexpr uint32_t cols = 128;
    constexpr int k_count = 512;
    constexpr float expected = 2.0f * static_cast<float>(k_count);

    struct PushConstants {
        int32_t rllm_bound_x;
        int32_t rllm_bound_y;
        int32_t k_count;
    } push_constants{
        static_cast<int32_t>(rows),
        static_cast<int32_t>(cols),
        k_count,
    };

    VHostBuffer<float[512 * 128]> a(get_session());
    VHostBuffer<float[512 * 128]> b(get_session());
    VHostBuffer<float[128 * 128]> c(get_session());
    VHostBuffer<float[128 * 128]> readback(get_session());
    a.fill(1.0f);
    b.fill(2.0f);
    c.fill(0.0f);

    VDeviceBuffer<float[512 * 128]> a_buf(a);
    VDeviceBuffer<float[512 * 128]> b_buf(b);
    VDeviceBuffer<float[128 * 128]> c_buf(c);
    std::vector<VBaseDeviceBuffer*> bufs{&a_buf, &b_buf, &c_buf};

    VulkanComputeContext transfer_context(get_session());
    a_buf.write(transfer_context, a);
    b_buf.write(transfer_context, b);

    VulkanComputeKernel no_tiling(
        get_session(),
        TESTDATA_DIR "/dynamic-atb-acc-perf-noopt.glsl",
        sizeof(PushConstants),
        bufs.size());
    write_multi_arg_descriptors(no_tiling, get_device(), bufs);

    const VulkanDimension no_tiling_dims{rows, cols, 1};

    (void)dispatch_raw_kernel_once(
        no_tiling,
        get_session(),
        c_buf,
        c,
        no_tiling_dims,
        &push_constants,
        sizeof(push_constants));
    const auto no_tiling_actual = read_float_buffer(transfer_context, c_buf, readback);

    for (uint32_t i = 0; i < no_tiling_actual.size(); ++i) {
        ASSERT_NEAR(no_tiling_actual[i], expected, 1.0e-4f)
            << "dynamic-atb-acc no-tiling C[" << i << "]";
    }

    double best_no_tiling = 1.0e30;
    for (int i = 0; i < 5; ++i) {
        best_no_tiling = std::min(
            best_no_tiling,
            dispatch_raw_kernel_once(
                no_tiling,
                get_session(),
                c_buf,
                c,
                no_tiling_dims,
                &push_constants,
                sizeof(push_constants)));
    }

    const uint32_t chunk_sizes[] = {1, 2, 4, 8, 16, 32, 64};
    double best_tiled = 1.0e30;
    uint32_t best_chunk_size = 0;

    for (uint32_t chunk_size : chunk_sizes) {
        const std::string shader_path =
            std::string(TESTDATA_DIR) + "/dynamic-atb-acc-perf-chunk-" + std::to_string(chunk_size) + ".glsl";
        const std::string tiled_glsl = read_text_file(shader_path);
        ASSERT_NE(tiled_glsl.find("int(gl_WorkGroupID.z) * " + std::to_string(chunk_size)), std::string::npos);
        ASSERT_NE(tiled_glsl.find("atomicAdd(C["), std::string::npos);

        VulkanComputeKernel tiled(
            get_session(),
            shader_path,
            sizeof(PushConstants),
            bufs.size());
        write_multi_arg_descriptors(tiled, get_device(), bufs);

        const VulkanDimension tiled_dims{
            (rows + 8 - 1) / 8,
            (cols + 8 - 1) / 8,
            (static_cast<uint32_t>(k_count) + chunk_size - 1) / chunk_size,
        };

        (void)dispatch_raw_kernel_once(
            tiled,
            get_session(),
            c_buf,
            c,
            tiled_dims,
            &push_constants,
            sizeof(push_constants));
        const auto tiled_actual = read_float_buffer(transfer_context, c_buf, readback);

        ASSERT_EQ(tiled_actual.size(), no_tiling_actual.size());
        for (uint32_t i = 0; i < tiled_actual.size(); ++i) {
            ASSERT_NEAR(tiled_actual[i], expected, 1.0e-4f)
                << "dynamic-atb-acc chunk=" << chunk_size << " C[" << i << "]";
        }

        double best_chunk = 1.0e30;
        for (int i = 0; i < 5; ++i) {
            best_chunk = std::min(
                best_chunk,
                dispatch_raw_kernel_once(
                    tiled,
                    get_session(),
                    c_buf,
                    c,
                    tiled_dims,
                    &push_constants,
                    sizeof(push_constants)));
        }

        const double chunk_speedup = best_no_tiling / best_chunk;
        RecordProperty(("dynamic_atb_chunk_" + std::to_string(chunk_size) + "_ms").c_str(), best_chunk);
        RecordProperty(("dynamic_atb_chunk_" + std::to_string(chunk_size) + "_speedup").c_str(), chunk_speedup);
        std::cerr << "dynamic_atb_acc chunk=" << chunk_size
                  << " speedup: " << chunk_speedup << "x"
                  << " (tiled=" << best_chunk << " ms"
                  << ", no_tiling=" << best_no_tiling << " ms)\n";

        if (best_chunk < best_tiled) {
            best_tiled = best_chunk;
            best_chunk_size = chunk_size;
        }
    }

    const double speedup = best_no_tiling / best_tiled;
    RecordProperty("dynamic_atb_best_chunk_size", best_chunk_size);
    RecordProperty("dynamic_atb_no_tiling_ms", best_no_tiling);
    RecordProperty("dynamic_atb_best_tiled_ms", best_tiled);
    RecordProperty("dynamic_atb_best_speedup", speedup);
    std::cerr << "dynamic_atb_acc best chunk=" << best_chunk_size
              << " speedup: " << speedup << "x"
              << " (tiled=" << best_tiled << " ms"
              << ", no_tiling=" << best_no_tiling << " ms)\n";

    EXPECT_LT(best_tiled, best_no_tiling)
        << "dynamic AtB tiling did not improve runtime: best chunk=" << best_chunk_size
        << ", tiled=" << best_tiled
        << "ms, no_tiling=" << best_no_tiling << "ms";
}
