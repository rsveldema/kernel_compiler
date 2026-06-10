add_library(libvulkanruntime STATIC
    src/test_vulkan_helpers.cc
)
target_include_directories(libvulkanruntime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
