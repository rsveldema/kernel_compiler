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


# Vulkan stubs: generate headers via compile.py --cpp-stub
set(STUB_SRCS)
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated/vulkanized/kernels)
file(MAKE_DIRECTORY ${GENERATED_DIR})

foreach(kernel_name multi-arg single-assign triangular1 with-wg2)
    set(kernel_file "${CMAKE_SOURCE_DIR}/testdata/${kernel_name}.kernel")
    set(output_header "${GENERATED_DIR}/${kernel_name}.h")
    list(APPEND STUB_SRCS ${output_header})

    add_custom_command(
        OUTPUT "${output_header}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${GENERATED_DIR}"
        COMMAND env PYTHONPATH="/tmp/pylib:${PYTHONPATH}" ${Python3_EXECUTABLE} -S
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            --cpp-stub "${output_header}"
            "${kernel_file}"
        DEPENDS
            "${CMAKE_SOURCE_DIR}/codegen/compile.py"
            "${CMAKE_SOURCE_DIR}/codegen/parser.py"
            "${CMAKE_SOURCE_DIR}/codegen/visitors/vulkan_cpp_stub_visitor.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/__init__.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/program.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/type.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/expression.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/statement.py"
            "${CMAKE_SOURCE_DIR}/codegen/ast/ast_node.py"
            "${kernel_file}"
        COMMENT "Generating C++ stub for ${kernel_name}.kernel"
    )
endforeach()

add_custom_target(all_generated_stubs ALL DEPENDS ${STUB_SRCS})


# test_vulkan_stubs executable (uses real <vulkan/vulkan_core.h>)
add_executable(test_vulkan_stubs test_vulkan_stubs.cc)
add_dependencies(test_vulkan_stubs all_generated_stubs)
target_link_libraries(test_vulkan_stubs PRIVATE GTest::gtest_main Vulkan::Vulkan)
target_compile_options(test_vulkan_stubs PRIVATE -march=native -O3)
target_include_directories(test_vulkan_stubs PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_VULKAN_INCLUDE_DIRS}
    ${GENERATED_DIR}
)
gtest_discover_tests(test_vulkan_stubs WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
