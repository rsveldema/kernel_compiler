include(FetchContent)
set(KERNEL_COMPILER_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

find_package(Vulkan REQUIRED)

enable_testing()

file(GLOB OFFLOAD_PARFOR_PARSE_TESTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/test_offload_parfor_*_cc.py")
foreach(parse_test IN LISTS OFFLOAD_PARFOR_PARSE_TESTS)
    get_filename_component(parse_test_name "${parse_test}" NAME_WE)
    add_test(
        NAME "${parse_test_name}"
        COMMAND ${Python3_EXECUTABLE} "${parse_test}"
        WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    )
    set_tests_properties("${parse_test_name}" PROPERTIES
        ENVIRONMENT "PYTHONPATH=${KERNEL_COMPILER_SOURCE_DIR}:$ENV{PYTHONPATH}"
    )
endforeach()


# Vulkan kernels: generate headers and GLSL via compile.py
set(GENERATED_KERNEL_SRCS)
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated/vulkanized/kernel_compiler_tests)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(VECMATH225_KERNEL "${KERNEL_COMPILER_SOURCE_DIR}/testdata/offload_parfor_vecmath_225.kernel")
set(VECMATH225_DUMP "${GENERATED_DIR}/offload_parfor_vecmath_225.kernel")
set(VECMATH225_GLSL "${GENERATED_DIR}/vecmath.L225.glsl")
set(VECMATH225_HEADER "${GENERATED_DIR}/vecmath.L225.h")
set(VECMATH225_SPV "${GENERATED_DIR}/vecmath.L225.spv")
set(VECMATH225_TYPE_ALIASES "${GENERATED_DIR}/_type_aliases.hpp")

add_custom_command(
    OUTPUT "${VECMATH225_DUMP}"
    COMMAND "${Python3_EXECUTABLE}" -c
        "import sys; sys.path.insert(0, '${KERNEL_COMPILER_SOURCE_DIR}/..'); from pathlib import Path; from offloadize_common import write_parfor_dump_from_kernel; write_parfor_dump_from_kernel(Path('${VECMATH225_KERNEL}'), Path('${GENERATED_DIR}'))"
    DEPENDS "${VECMATH225_KERNEL}" "${CMAKE_SOURCE_DIR}/offloadize_common.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating parfor dump for vecmath:225"
    VERBATIM
)

add_custom_command(
    OUTPUT
        "${VECMATH225_HEADER}"
        "${VECMATH225_SPV}"
        "${VECMATH225_TYPE_ALIASES}"
    BYPRODUCTS
        "${VECMATH225_GLSL}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${VECMATH225_DUMP}"
            --vulkan "${VECMATH225_GLSL}"
    COMMAND Vulkan::glslc -fshader-stage=compute --target-env=vulkan1.1
            "${VECMATH225_GLSL}" -o "${VECMATH225_SPV}"
    DEPENDS
        "${VECMATH225_DUMP}"
        "${VECMATH225_KERNEL}"
        "${KERNEL_COMPILER_SOURCE_DIR}/transforms/optimize_vecmath225.tkernel"
        "${CMAKE_SOURCE_DIR}/offloadize_common.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/tree_rewriter.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/vulkan_kernel_visitor.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/vulkan_cpp_stub_visitor.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating vecmath:225 Vulkan test stub"
    VERBATIM
)
list(APPEND GENERATED_KERNEL_SRCS
    "${VECMATH225_HEADER}"
    "${VECMATH225_SPV}"
    "${VECMATH225_TYPE_ALIASES}"
)

add_custom_target(all_generated_stubs ALL DEPENDS ${GENERATED_KERNEL_SRCS})


# test_vulkan_stubs executable (uses real <vulkan/vulkan_core.h>)
add_executable(test_vulkan_stubs
    test_vulkan_stubs.cc
)
add_dependencies(test_vulkan_stubs all_generated_stubs)
target_link_libraries(test_vulkan_stubs PRIVATE GTest::gtest_main Vulkan::Vulkan libvulkanruntime)
target_compile_options(test_vulkan_stubs PRIVATE )
target_compile_definitions(test_vulkan_stubs PRIVATE TESTDATA_DIR="${GENERATED_DIR}")
target_include_directories(test_vulkan_stubs PRIVATE
    ${KERNEL_COMPILER_SOURCE_DIR}/include
    ${KERNEL_COMPILER_SOURCE_DIR}/tests
    ${CMAKE_VULKAN_INCLUDE_DIRS}
    ${GENERATED_DIR}
)
gtest_discover_tests(test_vulkan_stubs WORKING_DIRECTORY ${KERNEL_COMPILER_SOURCE_DIR})
