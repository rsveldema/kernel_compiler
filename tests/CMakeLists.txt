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

foreach(kernel_name dynamic-atb-acc dynamic-atb-acc-perf multi-arg single-assign single-assign-tiled triangular-matrix-access triangular1 var-size-loop-tiled with-wg2)
    set(kernel_file "${KERNEL_COMPILER_SOURCE_DIR}/testdata/${kernel_name}.kernel")
    set(output_header "${GENERATED_DIR}/${kernel_name}.h")
    set(output_glsl "${GENERATED_DIR}/${kernel_name}.glsl")
    list(APPEND GENERATED_KERNEL_SRCS ${output_header} ${output_glsl})

    set(extra_codegen_args "")
    if(kernel_name STREQUAL "single-assign" OR kernel_name STREQUAL "single-assign-tiled" OR kernel_name STREQUAL "var-size-loop-tiled")
        set(extra_codegen_args --parallelize)
    endif()

    add_custom_command(
        OUTPUT "${output_header}" "${output_glsl}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
        COMMAND env PYTHONPATH="${KERNEL_COMPILER_SOURCE_DIR}:/tmp/pylib:${PYTHONPATH}"
            ${Python3_EXECUTABLE}
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            ${extra_codegen_args}
            --vulkan "${output_glsl}"
            "${kernel_file}"
        WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
        DEPENDS
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/vulkan_cpp_stub_visitor.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/vulkan_kernel_visitor.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/visitor.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/tree_rewriter.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/resolve_array_indices.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/kast/__init__.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/kast/program.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/kast/type.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/kast/expression.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/kast/statement.py"
            "${KERNEL_COMPILER_SOURCE_DIR}/codegen/kast/ast_node.py"
            "${kernel_file}"
        COMMENT "Generating Vulkan artifacts for ${kernel_name}.kernel"
    )
endforeach()
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
