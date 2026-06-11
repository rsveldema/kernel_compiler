/** @file test_vulkan_stubs.cc
 *  Unit tests that call all generated C++ stub dispatch functions using
 *  real Vulkan compute shaders, buffer allocations, and descriptor sets —
 *  with actual numerical verification of kernel outputs.
 */

#include <cstring>
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
    wds.dstSet            = ctx.desc_set();
    wds.dstBinding        = 0;
    wds.descriptorCount   = 1;
    wds.descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo       = &dbi;
    vkUpdateDescriptorSets(get_device(), 1, &wds, 0, nullptr);

    /* Dispatch */
    auto cb = ctx.begin_command_buffer();
    kernel.dispatch(cb, &push_value, sizeof(push_value), {(N + 256 - 1) / 256, 1, 1});

    ctx.submit_and_wait();

    /* Read back & verify */
    auto data = dst_buf.read(0, N * sizeof(int));
    std::vector<int> actual(N);
    memcpy(actual.data(), data.data(), N * sizeof(int));

    for (uint32_t i = 0; i < N; ++i) {
        EXPECT_EQ(actual[i], push_value) << "single-assign dst[" << i << "]";
    }
}

// ───────────── Test: multi-arg (3 independent matmuls + accumulate) ──

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

    /* Write descriptors for all 7 buffers */
    std::vector<VkDescriptorBufferInfo> dbis(7);
    for (uint32_t i = 0; i < 7; ++i) {
        dbis[i].buffer = bufs[i].get();
        dbis[i].offset = 0;
        dbis[i].range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet wds{};
        wds.sType             = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet            = ctx.desc_set();
        wds.dstBinding        = i;
        wds.descriptorCount   = 1;
        wds.descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds.pBufferInfo       = &dbis[i];
        vkUpdateDescriptorSets(get_device(), 1, &wds, 0, nullptr);
    }

    /* Dispatch */
    auto cb = ctx.begin_command_buffer();
    kernel.dispatch(cb, 0, 0, {(ROWS + 256 - 1) / 256, COLS, 1});
    ctx.submit_and_wait();

    /* Read back C (buffer index 6) */
    auto data = bufs[6].read(0, buf_sizes[6]);
    std::vector<float> actual(ROWS * COLS);
    memcpy(actual.data(), data.data(), static_cast<size_t>(buf_sizes[6]));

    for (uint32_t i = 0; i < ROWS * COLS; ++i) {
        EXPECT_NEAR(actual[i], expected_val, 1.0f) << "multi-arg C[" << i << "]";
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
    bufs[1].write(attn_w_h.data(), 0, attn_w_h.size() * sizeof(float));

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
        wds.dstSet            = ctx.desc_set();
        wds.dstBinding        = i;
        wds.descriptorCount   = 1;
        wds.descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds.pBufferInfo       = &dbis[i];
        vkUpdateDescriptorSets(get_device(), 1, &wds, 0, nullptr);
    }

    /* Dispatch — triangular1 workgroup: x=8, y=16, z=16 */
    auto cb = ctx.begin_command_buffer();
    kernel.dispatch(cb, &pc, sizeof(pc), {(H + 8 - 1) / 8, (W + 16 - 1) / 16, (D + 16 - 1) / 16});
    ctx.submit_and_wait();

    /* Read back d_raw (buffer index 2) */
    auto data = bufs[2].read(0, buf_size);
    std::vector<float> actual(H * W * D);
    memcpy(actual.data(), data.data(), static_cast<size_t>(buf_size));

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
    kernel.dispatch(cb, &push_A, sizeof(push_A), { (N + wg_x - 1) / wg_x, (N + wg_y - 1) / wg_y, 1});
    ctx.submit_and_wait();

    SUCCEED() << "with-wg2 dispatch completed (grid=" << N << "x" << N
              << ", workgroup=" << wg_x << "x" << wg_y
              << ", push.A=" << push_A << ")";
}
