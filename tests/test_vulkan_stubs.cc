/** @file test_vulkan_stubs.cc
 *  Unit tests that call all generated C++ stub dispatch functions using the
 *  real Vulkan types from <vulkan/vulkan_core.h>.
 *
 *  Each kernel produces a header with inline struct definitions and a dispatch
 *  function. We include those headers alongside <vulkan/vulkan_core.h> to verify
 *  correct type wiring — VkDevice, VkCommandBuffer, etc. come from the real
 *  Vulkan SDK (libvulkan), not mocks.
 */

#include <gtest/gtest.h>
#include <memory>

// Real Vulkan types and vkCmdDispatch declaration from the system vulkan package.
#include <vulkan/vulkan_core.h>

// Generated stub headers — each brings in its own inline struct definitions.
#include "multi-arg.h"
#include "single-assign.h"
#include "triangular1.h"
#include "with-wg2.h"

// ───────────── helpers: fake Vulkan handles ────────────

static VkDevice     make_device()       { return reinterpret_cast<VkDevice>(0x1); }
static VkPipelineLayout make_ppl()      { return reinterpret_cast<VkPipelineLayout>(0x1); }
static VkDescriptorSetLayout make_dsl() { return reinterpret_cast<VkDescriptorSetLayout>(0x1); }
static VkCommandBuffer make_cb()        { return reinterpret_cast<VkCommandBuffer>(0x1); }

// ───────────── multi-arg (2D tiled, 8 buffers, no push constants) ──

TEST(test_vulkan_stubs, multi_arg_dispatch_calls)
{
    // multi_arg_vecmath from multi-arg.h takes: VkDevice, VkPipelineLayout,
    //   VkDescriptorSetLayout, VkCommandBuffer, VkDescriptorSet, uint32_t rows,
    //   uint32_t cols, RllmBuffer_A1&, ..., RllmBuffer_C&

    float _buf[4];
    __builtin_memset(_buf, 0, sizeof(_buf));

    multi_arg_vecmath(
        make_device(), make_ppl(), make_dsl(), make_cb(),
        reinterpret_cast<VkDescriptorSet>(0x2),
        /*dispatch_rows=*/64, /*dispatch_cols=*/64,
        reinterpret_cast<RllmBuffer_A1&>(_buf),
        reinterpret_cast<RllmBuffer_B1&>(_buf),
        reinterpret_cast<RllmBuffer_A2&>(_buf),
        reinterpret_cast<RllmBuffer_B2&>(_buf),
        reinterpret_cast<RllmBuffer_A3&>(_buf),
        reinterpret_cast<RllmBuffer_B3&>(_buf),
        reinterpret_cast<RllmBuffer_C&>(_buf));

    SUCCEED() << "multi-arg multi_arg_vecmath called with real Vulkan types";
}

// ───────────── single-assign (1D + push constants) ─────

TEST(test_vulkan_stubs, single_assign_dispatch_calls)
{
    // single_assign_vecmath from single-assign.h: VkDevice, ..., uint32_t dispatch_rows,
    //   RllmBuffer_dst&, const single_assign_vecmath_PushConstants&
    float _buf[4];
    __builtin_memset(_buf, 0, sizeof(_buf));

    single_assign_vecmath_PushConstants pc{42};

    single_assign_vecmath(
        make_device(), make_ppl(), make_dsl(), make_cb(),
        reinterpret_cast<VkDescriptorSet>(0x2),
        /*dispatch_rows=*/64,
        reinterpret_cast<RllmBuffer_dst&>(_buf),
        pc);

    SUCCEED() << "single-assign single_assign_vecmath called with real Vulkan types";
}

// ───────────── triangular1 (2D + push constants) ──

TEST(test_vulkan_stubs, triangular1_dispatch_calls)
{
    // RllmBuffer_d_scores has float data[8*16384*16384] — huge. Use dynamic allocation.
    auto d_scores = std::make_unique<RllmBuffer_d_scores>();
    auto d_raw    = std::make_unique<RllmBuffer_d_raw>();
    auto attn_w   = std::make_unique<RllmBuffer_attn_w>();

    triangular1_TransformerBlock_PushConstants pc{
        /*seq_len*/512, /*d_scores_rows*/8, /*d_scores_cols*/512,
        /*d_raw_rows*/8,  /*d_raw_cols*/512};

    triangular1_TransformerBlock(
        make_device(), make_ppl(), make_dsl(), make_cb(),
        reinterpret_cast<VkDescriptorSet>(0x2),
        /*dispatch_rows=*/64, /*dispatch_cols=*/64,
        *d_scores, *d_raw, *attn_w, pc);

    SUCCEED() << "triangular1 triangular1_TransformerBlock called with real Vulkan types";
}

// ───────────── with-wg2 (no buffers, only push constants) ─────

TEST(test_vulkan_stubs, with_wg2_dispatch_calls)
{
    with_wg2_test_PushConstants pc{42};  // 'A' field

    with_wg2_test(
        make_device(), make_ppl(), make_dsl(), make_cb(),
        reinterpret_cast<VkDescriptorSet>(0x2),
        /*dispatch_rows=*/16, pc);

    SUCCEED() << "with-wg2 with_wg2_test called with real Vulkan types";
}
