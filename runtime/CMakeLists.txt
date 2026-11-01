add_library(libhipruntime STATIC
    src/hip_session.cc
)
find_package(hip REQUIRED CONFIG)
find_package(hipblas REQUIRED CONFIG)
target_link_libraries(libhipruntime PUBLIC hip::host roc::hipblas)

target_compile_options(libhipruntime PUBLIC -Wall -Werror -Og)

target_include_directories(libhipruntime PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
