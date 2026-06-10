/** @file test_vulkan_helpers.cc
 *  Test-only helper utilities — implementations for vk_result_str and
 *  check_vk used across the Vulkan test infrastructure.
 */

#include <string>
#include <stdexcept>
#include <vulkan/vulkan_core.h>

// Forward declarations from test_vulkan_helpers.h
std::string vk_result_str(VkResult rc);
void        check_vk(VkResult rc, const char* label);

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
