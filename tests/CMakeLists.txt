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

find_package(hip REQUIRED CONFIG)
find_program(RLLM_TEST_HIPCC NAMES hipcc REQUIRED)

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


# Hip kernels: generate headers and GLSL via compile.py
set(GENERATED_KERNEL_SRCS)
set(GENERATED_DIR ${CMAKE_BINARY_DIR}/generated/hipized/kernel_compiler_tests)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(VECMATH231_KERNEL "${KERNEL_COMPILER_SOURCE_DIR}/testdata/offload_parfor_vecmath_231.kernel")
set(VECMATH231_DUMP "${GENERATED_DIR}/offload_parfor_vecmath_231.kernel")
set(VECMATH231_GLSL "${GENERATED_DIR}/vecmath.L231.hip")
set(VECMATH231_HEADER "${GENERATED_DIR}/vecmath.L231.h")
set(VECMATH231_SPV "${GENERATED_DIR}/vecmath.L231.hsaco")
set(VECMATH231_NOOPT_DUMP "${GENERATED_DIR}/offload_parfor_vecmath_231_noopt.kernel")
set(VECMATH231_NOOPT_GLSL "${GENERATED_DIR}/vecmath.L231.noopt.hip")
set(VECMATH231_NOOPT_HEADER "${GENERATED_DIR}/vecmath.L231.noopt.h")
set(VECMATH231_NOOPT_SPV "${GENERATED_DIR}/vecmath.L231.noopt.hsaco")
set(VECMATH231_TYPE_ALIASES "${GENERATED_DIR}/_type_aliases.hpp")
set(VECMATH304_KERNEL "${KERNEL_COMPILER_SOURCE_DIR}/testdata/offload_parfor_vecmath_304.kernel")
set(VECMATH304_DUMP "${GENERATED_DIR}/offload_parfor_vecmath_304.kernel")
set(VECMATH304_GLSL "${GENERATED_DIR}/vecmath.L304.hip")
set(VECMATH304_HEADER "${GENERATED_DIR}/vecmath.L304.h")
set(VECMATH304_SPV "${GENERATED_DIR}/vecmath.L304.hsaco")
set(VECMATH304_NOOPT_DUMP "${GENERATED_DIR}/offload_parfor_vecmath_304_noopt.kernel")
set(VECMATH304_NOOPT_GLSL "${GENERATED_DIR}/vecmath.L304.noopt.hip")
set(VECMATH304_NOOPT_HEADER "${GENERATED_DIR}/vecmath.L304.noopt.h")
set(VECMATH304_NOOPT_SPV "${GENERATED_DIR}/vecmath.L304.noopt.hsaco")
set(VECMATH999_KERNEL "${KERNEL_COMPILER_SOURCE_DIR}/testdata/offload_parfor_vecmath_999.kernel")
set(VECMATH999_DUMP "${GENERATED_DIR}/offload_parfor_vecmath_999.kernel")
set(VECMATH999_HIP "${GENERATED_DIR}/vecmath.L999.hip")
set(VECMATH999_HEADER "${GENERATED_DIR}/vecmath.L999.h")
set(VECMATH999_HSACO "${GENERATED_DIR}/vecmath.L999.hsaco")
set(VECMATH999_BLAS_DUMP "${GENERATED_DIR}/offload_parfor_vecmath_999_blas.kernel")
set(VECMATH999_BLAS_HIP "${GENERATED_DIR}/vecmath.L999.blas.hip")
set(VECMATH999_BLAS_HEADER "${GENERATED_DIR}/vecmath.L999.blas.h")
set(VECMATH999_BLAS_HSACO "${GENERATED_DIR}/vecmath.L999.blas.hsaco")
set(VECMATH999_NOOPT_DUMP "${GENERATED_DIR}/offload_parfor_vecmath_999_noopt.kernel")
set(VECMATH999_NOOPT_HIP "${GENERATED_DIR}/vecmath.L999.noopt.hip")
set(VECMATH999_NOOPT_HEADER "${GENERATED_DIR}/vecmath.L999.noopt.h")
set(VECMATH999_NOOPT_HSACO "${GENERATED_DIR}/vecmath.L999.noopt.hsaco")

add_custom_command(
    OUTPUT "${VECMATH231_DUMP}"
    COMMAND "${Python3_EXECUTABLE}" -c
        "import sys; sys.path.insert(0, '${KERNEL_COMPILER_SOURCE_DIR}/..'); from pathlib import Path; from offloadize_common import write_parfor_dump_from_kernel; write_parfor_dump_from_kernel(Path('${VECMATH231_KERNEL}'), Path('${GENERATED_DIR}'))"
    DEPENDS "${VECMATH231_KERNEL}" "${CMAKE_SOURCE_DIR}/offloadize_common.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating parfor dump for vecmath:231"
    VERBATIM
)

add_custom_command(
    OUTPUT
        "${VECMATH231_HEADER}"
        "${VECMATH231_SPV}"
        "${VECMATH231_TYPE_ALIASES}"
    BYPRODUCTS
        "${VECMATH231_GLSL}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${VECMATH231_DUMP}"
            --hip "${VECMATH231_GLSL}"
    COMMAND "${RLLM_TEST_HIPCC}" --genco -O3
            "${VECMATH231_GLSL}" -o "${VECMATH231_SPV}"
    DEPENDS
        "${VECMATH231_DUMP}"
        "${VECMATH231_KERNEL}"
        "${KERNEL_COMPILER_SOURCE_DIR}/transforms/optimize_vecmath225.tkernel"
        "${CMAKE_SOURCE_DIR}/offloadize_common.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/tree_rewriter.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_kernel_visitor.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_cpp_stub_visitor.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating vecmath:231 Hip test stub"
    VERBATIM
)
list(APPEND GENERATED_KERNEL_SRCS
    "${VECMATH231_HEADER}"
    "${VECMATH231_SPV}"
    "${VECMATH231_TYPE_ALIASES}"
)

add_custom_command(
    OUTPUT "${VECMATH231_NOOPT_DUMP}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${VECMATH231_DUMP}" "${VECMATH231_NOOPT_DUMP}"
    DEPENDS "${VECMATH231_DUMP}"
    COMMENT "Copying vecmath:231 parfor dump for no-optimization baseline"
    VERBATIM
)

add_custom_command(
    OUTPUT
        "${VECMATH231_NOOPT_HEADER}"
        "${VECMATH231_NOOPT_SPV}"
    BYPRODUCTS
        "${VECMATH231_NOOPT_GLSL}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${VECMATH231_NOOPT_DUMP}"
            --hip "${VECMATH231_NOOPT_GLSL}"
            --no-optimizations
    COMMAND "${RLLM_TEST_HIPCC}" --genco -O3
            "${VECMATH231_NOOPT_GLSL}" -o "${VECMATH231_NOOPT_SPV}"
    DEPENDS
        "${VECMATH231_NOOPT_DUMP}"
        "${VECMATH231_KERNEL}"
        "${CMAKE_SOURCE_DIR}/offloadize_common.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/tree_rewriter.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_kernel_visitor.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_cpp_stub_visitor.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating vecmath:231 no-optimization Hip test stub"
    VERBATIM
)
list(APPEND GENERATED_KERNEL_SRCS
    "${VECMATH231_NOOPT_HEADER}"
    "${VECMATH231_NOOPT_SPV}"
)

add_custom_command(
    OUTPUT "${VECMATH304_DUMP}"
    COMMAND "${Python3_EXECUTABLE}" -c
        "import sys; sys.path.insert(0, '${KERNEL_COMPILER_SOURCE_DIR}/..'); from pathlib import Path; from offloadize_common import write_parfor_dump_from_kernel; write_parfor_dump_from_kernel(Path('${VECMATH304_KERNEL}'), Path('${GENERATED_DIR}'))"
    DEPENDS "${VECMATH304_KERNEL}" "${CMAKE_SOURCE_DIR}/offloadize_common.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating parfor dump for vecmath:304"
    VERBATIM
)

add_custom_command(
    OUTPUT
        "${VECMATH304_HEADER}"
        "${VECMATH304_SPV}"
    BYPRODUCTS
        "${VECMATH304_GLSL}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${VECMATH304_DUMP}"
            --hip "${VECMATH304_GLSL}"
    COMMAND "${RLLM_TEST_HIPCC}" --genco -O3
            "${VECMATH304_GLSL}" -o "${VECMATH304_SPV}"
    DEPENDS
        "${VECMATH304_DUMP}"
        "${VECMATH304_KERNEL}"
        "${KERNEL_COMPILER_SOURCE_DIR}/transforms/optimize_vecmath296.tkernel"
        "${CMAKE_SOURCE_DIR}/offloadize_common.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/tree_rewriter.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_kernel_visitor.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_cpp_stub_visitor.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating vecmath:304 Hip test stub"
    VERBATIM
)
list(APPEND GENERATED_KERNEL_SRCS
    "${VECMATH304_HEADER}"
    "${VECMATH304_SPV}"
)

add_custom_command(
    OUTPUT "${VECMATH304_NOOPT_DUMP}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${VECMATH304_DUMP}" "${VECMATH304_NOOPT_DUMP}"
    DEPENDS "${VECMATH304_DUMP}"
    COMMENT "Copying vecmath:304 parfor dump for no-optimization baseline"
    VERBATIM
)

add_custom_command(
    OUTPUT
        "${VECMATH304_NOOPT_HEADER}"
        "${VECMATH304_NOOPT_SPV}"
    BYPRODUCTS
        "${VECMATH304_NOOPT_GLSL}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${VECMATH304_NOOPT_DUMP}"
            --hip "${VECMATH304_NOOPT_GLSL}"
            --no-optimizations
    COMMAND "${RLLM_TEST_HIPCC}" --genco -O3
            "${VECMATH304_NOOPT_GLSL}" -o "${VECMATH304_NOOPT_SPV}"
    DEPENDS
        "${VECMATH304_NOOPT_DUMP}"
        "${VECMATH304_KERNEL}"
        "${CMAKE_SOURCE_DIR}/offloadize_common.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_kernel_visitor.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_cpp_stub_visitor.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating vecmath:304 no-optimization Hip test stub"
    VERBATIM
)
list(APPEND GENERATED_KERNEL_SRCS
    "${VECMATH304_NOOPT_HEADER}"
    "${VECMATH304_NOOPT_SPV}"
)

add_custom_command(
    OUTPUT "${VECMATH999_DUMP}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${VECMATH999_KERNEL}" "${VECMATH999_DUMP}"
    DEPENDS "${VECMATH999_KERNEL}"
    COMMENT "Copying FFN activation matmul-and-clip regression fixture"
    VERBATIM
)

add_custom_command(
    OUTPUT "${VECMATH999_HEADER}" "${VECMATH999_HSACO}"
    BYPRODUCTS "${VECMATH999_HIP}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${VECMATH999_DUMP}" --hip "${VECMATH999_HIP}"
    COMMAND "${RLLM_TEST_HIPCC}" --genco -O3 "${VECMATH999_HIP}" -o "${VECMATH999_HSACO}"
    DEPENDS
        "${VECMATH999_DUMP}"
        "${KERNEL_COMPILER_SOURCE_DIR}/transforms/blas_clip_abt.tkernel"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/tree_rewriter.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/blas.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_kernel_visitor.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_cpp_stub_visitor.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating BLAS plus clip FFN regression kernel"
    VERBATIM
)

add_custom_command(
    OUTPUT "${VECMATH999_NOOPT_DUMP}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${VECMATH999_KERNEL}" "${VECMATH999_NOOPT_DUMP}"
    DEPENDS "${VECMATH999_KERNEL}"
    COMMENT "Copying sequential FFN regression fixture"
    VERBATIM
)

add_custom_command(
    OUTPUT "${VECMATH999_BLAS_DUMP}"
    COMMAND "${CMAKE_COMMAND}" -E copy "${VECMATH999_KERNEL}" "${VECMATH999_BLAS_DUMP}"
    DEPENDS "${VECMATH999_KERNEL}"
    COMMENT "Copying forced BLAS-plus-clip regression fixture"
    VERBATIM
)

add_custom_command(
    OUTPUT "${VECMATH999_BLAS_HEADER}" "${VECMATH999_BLAS_HSACO}"
    BYPRODUCTS "${VECMATH999_BLAS_HIP}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${VECMATH999_BLAS_DUMP}" --hip "${VECMATH999_BLAS_HIP}"
            --test-enable-clip-blas-epilogue
    COMMAND "${RLLM_TEST_HIPCC}" --genco -O3 "${VECMATH999_BLAS_HIP}" -o "${VECMATH999_BLAS_HSACO}"
    DEPENDS
        "${VECMATH999_BLAS_DUMP}"
        "${KERNEL_COMPILER_SOURCE_DIR}/transforms/blas_clip_abt.tkernel"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/blas.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_kernel_visitor.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_cpp_stub_visitor.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating forced BLAS plus clip regression kernel"
    VERBATIM
)

add_custom_command(
    OUTPUT "${VECMATH999_NOOPT_HEADER}" "${VECMATH999_NOOPT_HSACO}"
    BYPRODUCTS "${VECMATH999_NOOPT_HIP}"
    COMMAND "${Python3_EXECUTABLE}" "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
            "${VECMATH999_NOOPT_DUMP}" --hip "${VECMATH999_NOOPT_HIP}" --no-optimizations
    COMMAND "${RLLM_TEST_HIPCC}" --genco -O3 "${VECMATH999_NOOPT_HIP}" -o "${VECMATH999_NOOPT_HSACO}"
    DEPENDS
        "${VECMATH999_NOOPT_DUMP}"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/compile.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/parser.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/transforms.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_kernel_visitor.py"
        "${KERNEL_COMPILER_SOURCE_DIR}/codegen/visitors/hip_cpp_stub_visitor.py"
    WORKING_DIRECTORY "${KERNEL_COMPILER_SOURCE_DIR}"
    COMMENT "Generating sequential FFN regression kernel"
    VERBATIM
)
list(APPEND GENERATED_KERNEL_SRCS
    "${VECMATH999_HEADER}"
    "${VECMATH999_HSACO}"
    "${VECMATH999_BLAS_HEADER}"
    "${VECMATH999_BLAS_HSACO}"
    "${VECMATH999_NOOPT_HEADER}"
    "${VECMATH999_NOOPT_HSACO}"
)

add_custom_target(all_generated_stubs ALL DEPENDS ${GENERATED_KERNEL_SRCS})


# test_hip_stubs executable (uses real <hip/hip_runtime_api.h>)
add_executable(test_hip_stubs
    test_hip_stubs.cc
)
add_dependencies(test_hip_stubs all_generated_stubs)
target_link_libraries(test_hip_stubs PRIVATE GTest::gtest_main hip::host libhipruntime)
target_compile_options(test_hip_stubs PRIVATE )
target_compile_definitions(test_hip_stubs PRIVATE TESTDATA_DIR="${GENERATED_DIR}")
target_include_directories(test_hip_stubs PRIVATE
    ${KERNEL_COMPILER_SOURCE_DIR}/include
    ${KERNEL_COMPILER_SOURCE_DIR}/tests
    ${CMAKE_HIP_INCLUDE_DIRS}
    ${GENERATED_DIR}
)
gtest_discover_tests(test_hip_stubs WORKING_DIRECTORY ${KERNEL_COMPILER_SOURCE_DIR})
