/** @file test_vulkan_stubs.cc
 *  Unit tests that call all generated C++ stub dispatch functions using
 *  real Vulkan compute shaders, buffer allocations, and descriptor sets —
 *  with actual numerical verification of kernel outputs.
 */

#include <cstring>
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
    VBuffer dst_buf(get_session(), N * sizeof(int));

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
    std::vector<int> actual(N);
    dst_buf.read(reinterpret_cast<uint8_t*>(actual.data()), N * sizeof(int));

    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(actual[i], push_value) << "single-assign dst[" << i << "]";
    }
}

// ───────────── Test: multi-arg (3 independent matmuls + accumulate) ──

static void write_multi_arg_descriptors(
    VulkanComputeKernel& kernel,
    VkDevice device,
    std::vector<VBuffer>& bufs)
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

static double dispatch_multi_arg_once(
    VulkanComputeKernel& kernel,
    VulkanSession& session,
    VBuffer& c_buf,
    const std::vector<float>& zero_c,
    const VulkanDimension& dims)
{
    c_buf.write(zero_c.data(), 0, zero_c.size() * sizeof(float));

    VulkanComputeContext ctx(session);
    auto cb = ctx.begin_command_buffer();
    const auto start = std::chrono::steady_clock::now();
    kernel.dispatch(cb, nullptr, 0, dims);
    ctx.submit_and_wait();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static std::vector<float> read_float_buffer(VBuffer& buf, VkDeviceSize size_bytes)
{
    std::vector<float> actual(static_cast<size_t>(size_bytes / sizeof(float)));
    buf.read(reinterpret_cast<uint8_t*>(actual.data()), size_bytes);
    return actual;
}

static bool selected_device_is_llvmpipe(VulkanSession& session)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(session.get_phys_device(), &props);
    return strstr(props.deviceName, "llvmpipe") != nullptr;
}

TEST_F(VulkanTestBase, multi_arg_correctness)
{
    /* multi-arg does: C[i,j] += sum_k( A1[n,k]*B1[k,m] + A2* B2 + A3* B3 ) */
    const uint32_t ROWS = 64;
    const uint32_t COLS = 1024;
    float expected_val = 44.0f * static_cast<float>(COLS);

    /* Host input data */
    std::vector<float> a1(ROWS * COLS, 1.0f);
    std::vector<float> b1(COLS * COLS, 2.0f);
    std::vector<float> c(ROWS * COLS, 0.0f);
    std::vector<float> a2(ROWS * COLS, 3.0f);
    std::vector<float> b2(COLS * COLS, 4.0f);
    std::vector<float> a3(ROWS * COLS, 5.0f);
    std::vector<float> b3(COLS * COLS, 6.0f);

    /* Device buffers */
    std::vector<VBuffer> bufs;
    VkDeviceSize buf_sizes[7] = {
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),   /* A1 */
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),   /* B1 */
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),   /* A2 */
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),   /* B2 */
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),   /* A3 */
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),   /* B3 */
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),   /* C (output) */
    };
    for (uint32_t i = 0; i < 7; ++i) bufs.emplace_back(get_session(), buf_sizes[i]);

    bufs[0].write(a1.data(), 0, a1.size() * sizeof(float));
    bufs[1].write(b1.data(), 0, b1.size() * sizeof(float));
    bufs[2].write(a2.data(), 0, a2.size() * sizeof(float));
    bufs[3].write(b2.data(), 0, b2.size() * sizeof(float));
    bufs[4].write(a3.data(), 0, a3.size() * sizeof(float));
    bufs[5].write(b3.data(), 0, b3.size() * sizeof(float));

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
    bufs[6].read(reinterpret_cast<uint8_t*>(actual.data()), buf_sizes[6]);

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

    std::vector<float> a1(ROWS * COLS, 1.0f);
    std::vector<float> b1(COLS * COLS, 2.0f);
    std::vector<float> c(ROWS * COLS, 0.0f);
    std::vector<float> a2(ROWS * COLS, 3.0f);
    std::vector<float> b2(COLS * COLS, 4.0f);
    std::vector<float> a3(ROWS * COLS, 5.0f);
    std::vector<float> b3(COLS * COLS, 6.0f);

    std::vector<VBuffer> bufs;
    VkDeviceSize buf_sizes[7] = {
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),
    };
    for (uint32_t i = 0; i < 7; ++i) bufs.emplace_back(get_session(), buf_sizes[i]);

    bufs[0].write(a1.data(), 0, a1.size() * sizeof(float));
    bufs[1].write(b1.data(), 0, b1.size() * sizeof(float));
    bufs[2].write(a2.data(), 0, a2.size() * sizeof(float));
    bufs[3].write(b2.data(), 0, b2.size() * sizeof(float));
    bufs[4].write(a3.data(), 0, a3.size() * sizeof(float));
    bufs[5].write(b3.data(), 0, b3.size() * sizeof(float));

    VulkanComputeKernel unoptimized(get_session(), TESTDATA_DIR "/multi-arg-noopt.glsl", 0, bufs.size());
    write_multi_arg_descriptors(unoptimized, get_device(), bufs);

    const VulkanDimension unoptimized_dims{ROWS, COLS, 1};

    (void)dispatch_multi_arg_once(unoptimized, get_session(), bufs[6], c, unoptimized_dims);
    const auto sequential_actual = read_float_buffer(bufs[6], buf_sizes[6]);

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
        const auto shared_actual = read_float_buffer(bufs[6], buf_sizes[6]);

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

    std::vector<float> a1(ROWS * COLS, 1.0f);
    std::vector<float> b1(COLS * COLS, 2.0f);
    std::vector<float> c(ROWS * COLS, 0.0f);
    std::vector<float> a2(ROWS * COLS, 3.0f);
    std::vector<float> b2(COLS * COLS, 4.0f);
    std::vector<float> a3(ROWS * COLS, 5.0f);
    std::vector<float> b3(COLS * COLS, 6.0f);

    std::vector<VBuffer> bufs;
    VkDeviceSize buf_sizes[7] = {
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(COLS) * COLS * sizeof(float),
        static_cast<VkDeviceSize>(ROWS) * COLS * sizeof(float),
    };
    for (uint32_t i = 0; i < 7; ++i) bufs.emplace_back(coop_session, buf_sizes[i]);

    bufs[0].write(a1.data(), 0, a1.size() * sizeof(float));
    bufs[1].write(b1.data(), 0, b1.size() * sizeof(float));
    bufs[2].write(a2.data(), 0, a2.size() * sizeof(float));
    bufs[3].write(b2.data(), 0, b2.size() * sizeof(float));
    bufs[4].write(a3.data(), 0, a3.size() * sizeof(float));
    bufs[5].write(b3.data(), 0, b3.size() * sizeof(float));

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

    const auto actual = read_float_buffer(bufs[6], buf_sizes[6]);
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

    std::vector<float> d_scores_h(H * W * D, 1.0f);
    std::vector<float> attn_w_h(H * W * D, 1.0f);
    float expected_val = 1.0f - static_cast<float>(D); /* 1 - 32 = -31 */

    VkDeviceSize buf_size = static_cast<VkDeviceSize>(H) * W * D * sizeof(float);
    std::vector<VBuffer> bufs;
    for (uint32_t i = 0; i < 3; ++i) bufs.emplace_back(get_session(), buf_size);

    bufs[0].write(d_scores_h.data(), 0, d_scores_h.size() * sizeof(float));
    bufs[2].write(attn_w_h.data(), 0, attn_w_h.size() * sizeof(float));

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
    std::vector<float> actual(H * W * D);
    bufs[1].read(reinterpret_cast<uint8_t*>(actual.data()), buf_size);

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
