#pragma once

// Minimal Vulkan type definitions for testing generated C++ stubs.
// These provide just enough structure to verify compilation without
// requiring an actual Vulkan runtime or ICD.

#include <cstdint>
#include <string_view>
#include <utility>

// Opaque handles (just integers, sufficient for header-only tests)
struct VkDevice_T {};
using VkDevice = VkDevice_T*;

struct VkPipelineLayout_T {};
using VkPipelineLayout = VkPipelineLayout_T*;

struct VkDescriptorSetLayout_T {};
using VkDescriptorSetLayout = VkDescriptorSetLayout_T*;

struct VkCommandBuffer_T {};
using VkCommandBuffer = VkCommandBuffer_T*;

struct VkDescriptorSet_T {};
using VkDescriptorSet = VkDescriptorSet_T*;

// vkCmdDispatch stub (empty function, just needs to link)
inline void vkCmdDispatch(
    VkCommandBuffer /*command_buffer*/,
    uint32_t /*groupCountX*/,
    uint32_t /*groupCountY*/,
    uint32_t /*groupCountZ*/
) {}

// ---- Helper: construct typed matrix/vector structs for test data ----

template<typename T>
struct FRM_float {
    T*         data  = nullptr;
    uint32_t   rows  = 0;
    uint32_t   cols  = 0;
};

template<typename T>
struct SRM_float {
    T*                data = nullptr;
    static constexpr uint32_t rows = 1;
    static constexpr uint32_t cols = 1;
};

// ---- Helper: build a test descriptor set for the stub function ----

// Each generated kernel has its own DescriptorSet struct type.
// We provide a small utility to zero-initialize them for tests.
