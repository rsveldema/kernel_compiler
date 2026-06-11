add_library(libvulkanruntime STATIC
    src/vulkan_session.cc
)

target_compile_options(libvulkanruntime PUBLIC -Wall -Werror -Og)

target_include_directories(libvulkanruntime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
