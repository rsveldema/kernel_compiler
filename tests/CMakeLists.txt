include(FetchContent)
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


# Vulkan kernels: generate headers and GLSL via compile.py
set(GENERATED_KERNEL_SRCS)
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated/vulkanized/kernels)
file(MAKE_DIRECTORY ${GENERATED_DIR})

foreach(kernel_name multi-arg single-assign triangular1 with-wg2)
    set(kernel_file "${CMAKE_SOURCE_DIR}/testdata/${kernel_name}.kernel")
    set(output_header "${GENERATED_DIR}/${kernel_name}.h")
    set(output_glsl "${GENERATED_DIR}/${kernel_name}.glsl")
    list(APPEND GENERATED_KERNEL_SRCS ${output_header} ${output_glsl})

    add_custom_command(
        OUTPUT "${output_header}" "${output_glsl}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
        COMMAND env PYTHONPATH="${CMAKE_SOURCE_DIR}:/tmp/pylib:${PYTHONPATH}"
            ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            --cpp-stub "${output_header}"
            "${kernel_file}"
        COMMAND env PYTHONPATH="${CMAKE_SOURCE_DIR}:/tmp/pylib:${PYTHONPATH}"
            ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            --vulkan "${output_glsl}"
            "${kernel_file}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DEPENDS
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            "${CMAKE_SOURCE_DIR}/codegen/parser.py"
            "${CMAKE_SOURCE_DIR}/codegen/transforms.py"
            "${CMAKE_SOURCE_DIR}/codegen/optim.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/vulkan_cpp_stub_visitor.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/vulkan_kernel_visitor.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/visitor.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/resolve_array_indices.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/__init__.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/program.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/type.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/expression.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/statement.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/ast_node.py"
            "${kernel_file}"
        COMMENT "Generating Vulkan artifacts for ${kernel_name}.kernel"
    )
endforeach()

set(MULTI_ARG_NOOPT_GLSL "${GENERATED_DIR}/multi-arg-noopt.glsl")
list(APPEND GENERATED_KERNEL_SRCS ${MULTI_ARG_NOOPT_GLSL})
add_custom_command(
    OUTPUT "${MULTI_ARG_NOOPT_GLSL}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
    COMMAND env PYTHONPATH="${CMAKE_SOURCE_DIR}:/tmp/pylib:${PYTHONPATH}"
        ${Python3_EXECUTABLE}
        "${CMAKE_SOURCE_DIR}/codegen/compile.py"
        --no-optimize
        --vulkan "${MULTI_ARG_NOOPT_GLSL}"
        "${CMAKE_SOURCE_DIR}/testdata/multi-arg.kernel"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    DEPENDS
        "${CMAKE_SOURCE_DIR}/codegen/compile.py"
        "${CMAKE_SOURCE_DIR}/codegen/parser.py"
        "${CMAKE_SOURCE_DIR}/codegen/transforms.py"
        "${CMAKE_SOURCE_DIR}/codegen/optim.py"
        "${CMAKE_SOURCE_DIR}/codegen/visitors/vulkan_kernel_visitor.py"
        "${CMAKE_SOURCE_DIR}/codegen/visitors/visitor.py"
        "${CMAKE_SOURCE_DIR}/codegen/visitors/resolve_array_indices.py"
        "${CMAKE_SOURCE_DIR}/codegen/ast/__init__.py"
        "${CMAKE_SOURCE_DIR}/codegen/ast/program.py"
        "${CMAKE_SOURCE_DIR}/codegen/ast/type.py"
        "${CMAKE_SOURCE_DIR}/codegen/ast/expression.py"
        "${CMAKE_SOURCE_DIR}/codegen/ast/statement.py"
        "${CMAKE_SOURCE_DIR}/codegen/ast/ast_node.py"
        "${CMAKE_SOURCE_DIR}/testdata/multi-arg.kernel"
    COMMENT "Generating unoptimized Vulkan shader for multi-arg.kernel"
)

foreach(chunk_size 1 2 4 8 16 32 64)
    set(MULTI_ARG_CHUNK_GLSL "${GENERATED_DIR}/multi-arg-chunk-${chunk_size}.glsl")
    list(APPEND GENERATED_KERNEL_SRCS ${MULTI_ARG_CHUNK_GLSL})
    add_custom_command(
        OUTPUT "${MULTI_ARG_CHUNK_GLSL}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
        COMMAND env PYTHONPATH="${CMAKE_SOURCE_DIR}:/tmp/pylib:${PYTHONPATH}"
            ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            --chunk-size "${chunk_size}"
            --vulkan "${MULTI_ARG_CHUNK_GLSL}"
            "${CMAKE_SOURCE_DIR}/testdata/multi-arg.kernel"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DEPENDS
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            "${CMAKE_SOURCE_DIR}/codegen/parser.py"
            "${CMAKE_SOURCE_DIR}/codegen/transforms.py"
            "${CMAKE_SOURCE_DIR}/codegen/optim.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/vulkan_kernel_visitor.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/visitor.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/resolve_array_indices.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/__init__.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/program.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/type.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/expression.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/statement.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/ast_node.py"
            "${CMAKE_SOURCE_DIR}/testdata/multi-arg.kernel"
        COMMENT "Generating Vulkan shader for multi-arg.kernel chunk size ${chunk_size}"
    )
endforeach()

foreach(chunk_size 8 16 32)
    set(MULTI_ARG_COOPMAT2_GLSL "${GENERATED_DIR}/multi-arg-coopmat2-chunk-${chunk_size}.glsl")
    list(APPEND GENERATED_KERNEL_SRCS ${MULTI_ARG_COOPMAT2_GLSL})
    add_custom_command(
        OUTPUT "${MULTI_ARG_COOPMAT2_GLSL}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
        COMMAND env PYTHONPATH="${CMAKE_SOURCE_DIR}:/tmp/pylib:${PYTHONPATH}"
            ${Python3_EXECUTABLE}
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            --coopmat2
            --chunk-size "${chunk_size}"
            --vulkan "${MULTI_ARG_COOPMAT2_GLSL}"
            "${CMAKE_SOURCE_DIR}/testdata/multi-arg.kernel"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        DEPENDS
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            "${CMAKE_SOURCE_DIR}/codegen/parser.py"
            "${CMAKE_SOURCE_DIR}/codegen/transforms.py"
            "${CMAKE_SOURCE_DIR}/codegen/optim.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/vulkan_kernel_visitor.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/visitor.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/resolve_array_indices.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/__init__.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/program.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/type.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/expression.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/statement.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/ast_node.py"
            "${CMAKE_SOURCE_DIR}/testdata/multi-arg.kernel"
        COMMENT "Generating VK_NV_cooperative_matrix2 shader for multi-arg.kernel chunk size ${chunk_size}"
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
target_compile_definitions(test_vulkan_stubs PRIVATE VK_STUB_NOOP TESTDATA_DIR="${GENERATED_DIR}")
target_include_directories(test_vulkan_stubs PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_VULKAN_INCLUDE_DIRS}
    ${GENERATED_DIR}
)
gtest_discover_tests(test_vulkan_stubs WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
