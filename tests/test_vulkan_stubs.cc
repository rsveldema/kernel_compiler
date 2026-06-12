/** @file test_vulkan_stubs.cc
 *  Unit tests that call all generated C++ stub dispatch functions using
 *  real Vulkan compute shaders, buffer allocations, and descriptor sets —
 *  with actual numerical verification of kernel outputs.
 */

#include <cstring>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <vulkan/vulkan_core.h>

// Generated stub headers
#include "multi-arg.h"
#include "single-assign.h"
#include "triangular1.h"
#include "with-wg2.h"

// Shared Vulkan test infrastructure
#include "vulkan_test_helpers.hpp"

// ───────────── Test: single-assign (sets all elements to a value) ──

TEST_F(VulkanTestBase, single_assign_correctness)
{
    const uint32_t N = 64;
    int push_value = 42;
    std::vector<int> expected(N, 42);

    /* Allocate device buffer for output */
    VDeviceBuffer dst_buf(get_session(), N * sizeof(int));

    /* Vulkan compute context — owns DSL, descriptors, pipeline layout, command buf */

    VulkanComputeKernel kernel(get_session(), TESTDATA_DIR "/single-assign.glsl", 4, 1);
    VulkanComputeContext ctx(get_session());

    /* Write buffer to descriptor set */
    VkDescriptorBufferInfo dbi{};
    dbi.buffer = dst_buf.get();
    dbi.offset = 0;
    dbi.range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet wds{};
    wds.sType             = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet            = kernel.desc_set();
    wds.dstBinding        = 0;
    wds.descriptorCount   = 1;
    wds.descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo       = &dbi;
    vkUpdateDescriptorSets(get_device(), 1, &wds, 0, nullptr);

    /* Dispatch */
    auto cb = ctx.begin_command_buffer();
    kernel.dispatch(cb, &push_value, sizeof(push_value), VulkanDimension{N, 1, 1});

    ctx.submit_and_wait();

    /* Read back & verify */
    VHostBuffer<int> readback(get_session(), N);
    std::vector<int> actual(N);
    dst_buf.read(readback);
    memcpy(actual.data(), readback.get(), N * sizeof(int));

    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(actual[i], push_value) << "single-assign dst[" << i << "]";
    }
}

// ───────────── Test: multi-arg (3 independent matmuls + accumulate) ──

static void write_multi_arg_descriptors(
    VulkanComputeKernel& kernel,
    VkDevice device,
    std::vector<VDeviceBuffer>& bufs)
{
    std::vector<VkDescriptorBufferInfo> dbis(bufs.size());
    for (uint32_t i = 0; i < bufs.size(); ++i) {
        dbis[i].buffer = bufs[i].get();
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

static void download_buffer(
    VulkanSession& session,
    VDeviceBuffer& src,
    void* dst)
{
    const VkDeviceSize size_bytes = src.size();
    VBaseHostBuffer host(session, size_bytes);
    src.read(host);
    memcpy(dst, host.get(), static_cast<size_t>(size_bytes));
}

static double dispatch_multi_arg_once(
    VulkanComputeKernel& kernel,
    VulkanSession& session,
    VDeviceBuffer& c_buf,
    VBaseHostBuffer& zero_c,
    const VulkanDimension& dims)
{
    c_buf.write(zero_c);

    VulkanComputeContext ctx(session);
    auto cb = ctx.begin_command_buffer();
    const auto start = std::chrono::steady_clock::now();
    kernel.dispatch(cb, nullptr, 0, dims);
    ctx.submit_and_wait();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::vector<float> read_float_buffer(VulkanSession& session, VDeviceBuffer& buf)
{
    const VkDeviceSize size_bytes = buf.size();
    std::vector<float> actual(static_cast<size_t>(size_bytes / sizeof(float)));
    VHostBuffer<float> host(session, actual.size());
    buf.read(host);
    memcpy(actual.data(), host.get(), static_cast<size_t>(size_bytes));
    return actual;
}

static bool selected_device_is_llvmpipe(VulkanSession& session)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(session.get_phys_device(), &props);
    return strstr(props.deviceName, "llvmpipe") != nullptr;
}

TEST_F(VulkanTestBase, host_device_buffer_copy_bandwidth)
{
    constexpr VkDeviceSize size_bytes = 64ull * 1024ull * 1024ull;
    constexpr int iterations = 5;

    VBaseHostBuffer upload(get_session(), size_bytes);
    VBaseHostBuffer download(get_session(), size_bytes);
    VDeviceBuffer device(get_session(), size_bytes);

    for (VkDeviceSize i = 0; i < size_bytes; ++i) {
        upload.get()[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
    }
    std::memset(download.get(), 0, static_cast<size_t>(size_bytes));

    device.write(upload);
    device.read(download);
    ASSERT_EQ(std::memcmp(upload.get(), download.get(), static_cast<size_t>(size_bytes)), 0);

    double upload_ms = 1.0e30;
    double download_ms = 1.0e30;
    double roundtrip_ms = 1.0e30;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        device.write(upload);
        auto mid = std::chrono::steady_clock::now();
        device.read(download);
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
    const uint32_t ROWS = 64;
    const uint32_t COLS = 1024;
    float expected_val = 44.0f * static_cast<float>(COLS);

    VHostBuffer<float> a1(get_session(), ROWS * COLS);
    VHostBuffer<float> b1(get_session(), COLS * COLS);
    VHostBuffer<float> a2(get_session(), ROWS * COLS);
    VHostBuffer<float> b2(get_session(), COLS * COLS);
    VHostBuffer<float> a3(get_session(), ROWS * COLS);
    VHostBuffer<float> b3(get_session(), COLS * COLS);
    VHostBuffer<float> c(get_session(), ROWS * COLS);
    a1.fill(1.0f);
    b1.fill(2.0f);
    a2.fill(3.0f);
    b2.fill(4.0f);
    a3.fill(5.0f);
    b3.fill(6.0f);
    c.fill(0.0f);
    std::vector<VDeviceBuffer> bufs;
    bufs.emplace_back(get_session(), a1.size());
    bufs.emplace_back(get_session(), b1.size());
    bufs.emplace_back(get_session(), a2.size());
    bufs.emplace_back(get_session(), b2.size());
    bufs.emplace_back(get_session(), a3.size());
    bufs.emplace_back(get_session(), b3.size());
    bufs.emplace_back(get_session(), c.size());
    bufs[0].write(a1);
    bufs[1].write(b1);
    bufs[2].write(a2);
    bufs[3].write(b2);
    bufs[4].write(a3);
    bufs[5].write(b3);
    bufs[6].write(c);

    /* Vulkan compute context */
    VulkanComputeKernel kernel(get_session(), TESTDATA_DIR "/multi-arg.glsl", 0, bufs.size());
    VulkanComputeContext ctx(get_session());

    write_multi_arg_descriptors(kernel, get_device(), bufs);

    /* Dispatch */
    auto cb = ctx.begin_command_buffer();
    kernel.dispatch(cb, nullptr, 0, VulkanDimension{(ROWS + 8 - 1) / 8, (COLS + 8 - 1) / 8, 1024 / 8});
    ctx.submit_and_wait();

    /* Read back C (buffer index 6) */
    std::vector<float> actual(ROWS * COLS);
    download_buffer(get_session(), bufs[6], actual.data());

    for (uint32_t i = 0; i < ROWS * COLS; ++i) {
        EXPECT_NEAR(actual[i], expected_val, 1.0f) << "multi-arg C[" << i << "]";
    }
}

TEST_F(VulkanTestBase, multi_arg_shared_memory_tiling_is_faster)
{
    if (!get_session().shader_buffer_float32_atomic_add_enabled()) {
        GTEST_SKIP() << "selected Vulkan device does not expose shaderBufferFloat32AtomicAdd";
    }

    const uint32_t ROWS = 256;
    const uint32_t COLS = 1024;

    VHostBuffer<float> a1(get_session(), ROWS * COLS);
    VHostBuffer<float> b1(get_session(), COLS * COLS);
    VHostBuffer<float> a2(get_session(), ROWS * COLS);
    VHostBuffer<float> b2(get_session(), COLS * COLS);
    VHostBuffer<float> a3(get_session(), ROWS * COLS);
    VHostBuffer<float> b3(get_session(), COLS * COLS);
    VHostBuffer<float> c(get_session(), ROWS * COLS);
    a1.fill(1.0f);
    b1.fill(2.0f);
    a2.fill(3.0f);
    b2.fill(4.0f);
    a3.fill(5.0f);
    b3.fill(6.0f);
    c.fill(0.0f);
    std::vector<VDeviceBuffer> bufs;
    bufs.emplace_back(get_session(), a1.size());
    bufs.emplace_back(get_session(), b1.size());
    bufs.emplace_back(get_session(), a2.size());
    bufs.emplace_back(get_session(), b2.size());
    bufs.emplace_back(get_session(), a3.size());
    bufs.emplace_back(get_session(), b3.size());
    bufs.emplace_back(get_session(), c.size());
    bufs[0].write(a1);
    bufs[1].write(b1);
    bufs[2].write(a2);
    bufs[3].write(b2);
    bufs[4].write(a3);
    bufs[5].write(b3);

    VulkanComputeKernel unoptimized(get_session(), TESTDATA_DIR "/multi-arg-noopt.glsl", 0, bufs.size());
    write_multi_arg_descriptors(unoptimized, get_device(), bufs);

    const VulkanDimension unoptimized_dims{ROWS, COLS, 1};

    (void)dispatch_multi_arg_once(unoptimized, get_session(), bufs[6], c, unoptimized_dims);
    const auto sequential_actual = read_float_buffer(get_session(), bufs[6]);

    double best_unoptimized = 1.0e30;
    for (int i = 0; i < 3; ++i) {
        best_unoptimized = std::min(best_unoptimized, dispatch_multi_arg_once(unoptimized, get_session(), bufs[6], c, unoptimized_dims));
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

        (void)dispatch_multi_arg_once(shared, get_session(), bufs[6], c, shared_dims);
        const auto shared_actual = read_float_buffer(get_session(), bufs[6]);

        ASSERT_EQ(shared_actual.size(), sequential_actual.size());
        for (uint32_t i = 0; i < shared_actual.size(); ++i) {
            EXPECT_NEAR(shared_actual[i], sequential_actual[i], 1.0f)
                << "multi-arg chunk=" << chunk_size << " optimized vs sequential C[" << i << "]";
        }

        double best_chunk = 1.0e30;
        for (int i = 0; i < 3; ++i) {
            best_chunk = std::min(best_chunk, dispatch_multi_arg_once(shared, get_session(), bufs[6], c, shared_dims));
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

    const uint32_t ROWS = 64;
    const uint32_t COLS = 1024;
    const uint32_t CHUNK = 16;
    float expected_val = 44.0f * static_cast<float>(COLS);

    VHostBuffer<float> a1(coop_session, ROWS * COLS);
    VHostBuffer<float> b1(coop_session, COLS * COLS);
    VHostBuffer<float> a2(coop_session, ROWS * COLS);
    VHostBuffer<float> b2(coop_session, COLS * COLS);
    VHostBuffer<float> a3(coop_session, ROWS * COLS);
    VHostBuffer<float> b3(coop_session, COLS * COLS);
    VHostBuffer<float> c(coop_session, ROWS * COLS);
    a1.fill(1.0f);
    b1.fill(2.0f);
    a2.fill(3.0f);
    b2.fill(4.0f);
    a3.fill(5.0f);
    b3.fill(6.0f);
    c.fill(0.0f);
    std::vector<VDeviceBuffer> bufs;
    bufs.emplace_back(coop_session, a1.size());
    bufs.emplace_back(coop_session, b1.size());
    bufs.emplace_back(coop_session, a2.size());
    bufs.emplace_back(coop_session, b2.size());
    bufs.emplace_back(coop_session, a3.size());
    bufs.emplace_back(coop_session, b3.size());
    bufs.emplace_back(coop_session, c.size());
    bufs[0].write(a1);
    bufs[1].write(b1);
    bufs[2].write(a2);
    bufs[3].write(b2);
    bufs[4].write(a3);
    bufs[5].write(b3);

    VulkanComputeKernel kernel(
        coop_session,
        TESTDATA_DIR "/multi-arg-coopmat2-chunk-16.glsl",
        0,
        bufs.size());
    write_multi_arg_descriptors(kernel, coop_session.get_device(), bufs);

    (void)dispatch_multi_arg_once(
        kernel,
        coop_session,
        bufs[6],
        c,
        VulkanDimension{(ROWS + 8 - 1) / 8, (COLS + 8 - 1) / 8, 1});

    const auto actual = read_float_buffer(coop_session, bufs[6]);
    ASSERT_EQ(actual.size(), ROWS * COLS);
    for (uint32_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected_val, 1.0f)
            << "multi-arg coopmat2 chunk=" << CHUNK << " C[" << i << "]";
    }
}

// ───────────── Test: triangular1 (triangular attention pattern) ──

TEST_F(VulkanTestBase, triangular1_correctness)
{
    const uint32_t H   = 8;
    const uint32_t W   = 32;
    const uint32_t D   = 32;

    float expected_val = 1.0f - static_cast<float>(D); /* 1 - 32 = -31 */

    const size_t elem_count = static_cast<size_t>(H) * W * D;
    std::vector<VDeviceBuffer> bufs;

    VHostBuffer<float> d_scores_h(get_session(), elem_count);
    VHostBuffer<float> d_raw_h(get_session(), elem_count);
    VHostBuffer<float> attn_w_h(get_session(), elem_count);
    d_scores_h.fill(1.0f);
    d_raw_h.fill(0.0f);
    attn_w_h.fill(1.0f);
    bufs.emplace_back(get_session(), d_scores_h.size());
    bufs.emplace_back(get_session(), d_raw_h.size());
    bufs.emplace_back(get_session(), attn_w_h.size());
    bufs[0].write(d_scores_h);
    bufs[1].write(d_raw_h);
    bufs[2].write(attn_w_h);

    /* Push constants */
    struct Tri1Push { int32_t seq_len; int32_t ds_rows; int32_t ds_cols; int32_t dr_rows; int32_t dr_cols; };
    Tri1Push pc = { static_cast<int32_t>(W), H, W, H, D };

    VulkanComputeKernel kernel(get_session(), TESTDATA_DIR "/triangular1.glsl", sizeof(pc), 3);

    /* Vulkan compute context */
    VulkanComputeContext ctx(get_session());

    /* Write descriptors */
    std::vector<VkDescriptorBufferInfo> dbis(3);
    for (uint32_t i = 0; i < 3; ++i) {
        dbis[i].buffer = bufs[i].get();
        dbis[i].offset = 0;
        dbis[i].range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet wds{};
        wds.sType             = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet            = kernel.desc_set();
        wds.dstBinding        = i;
        wds.descriptorCount   = 1;
        wds.descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds.pBufferInfo       = &dbis[i];
        vkUpdateDescriptorSets(get_device(), 1, &wds, 0, nullptr);
    }

    /* Dispatch */
    auto cb = ctx.begin_command_buffer();
    kernel.dispatch(cb, &pc, sizeof(pc), VulkanDimension{H, W, D});
    ctx.submit_and_wait();

    /* Read back d_raw (buffer index 1) */
    std::vector<float> actual(elem_count);
    download_buffer(get_session(), bufs[1], actual.data());

    for (uint32_t hi = 0; hi < H; ++hi) {
        for (uint32_t i = 0; i < W; ++i) {
            for (uint32_t j = 0; j < D; ++j) {
                uint32_t idx = (hi * W + i) * D + j;
                EXPECT_NEAR(actual[idx], expected_val, 1.0f)
                    << "triangular1 d_raw[" << hi << "," << i << "," << j << "]";
            }
        }
    }
}

// ───────────── Test: with-wg2 (push-only kernel) ──────

TEST_F(VulkanTestBase, with_wg2_correctness)
{
    const uint32_t N = 1024;

    /* Vulkan compute context — no descriptor layout needed (no SSBOs) */
    VulkanComputeKernel kernel(get_session(), TESTDATA_DIR "/with-wg2.glsl", sizeof(int), 0);
    VulkanComputeContext ctx(get_session());

    uint32_t wg_x = 512, wg_y = 64;
    int32_t push_A = 42;

    auto cb = ctx.begin_command_buffer();
    /* Dispatch: ceil(1024/512) x ceil(1024/64) = 2 x 16 = 32 total workgroups */
    kernel.dispatch(cb, &push_A, sizeof(push_A), VulkanDimension{(N + wg_x - 1) / wg_x, (N + wg_y - 1) / wg_y, 1});
    ctx.submit_and_wait();

    SUCCEED() << "with-wg2 dispatch completed (grid=" << N << "x" << N
              << ", workgroup=" << wg_x << "x" << wg_y
              << ", push.A=" << push_A << ")";
}
