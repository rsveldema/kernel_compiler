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

enable_testing()


# ---- Vulkan stubs unit test (header-only, links against GTest) ----
add_executable(test_vulkan_stubs
    test_vulkan_stubs.cc
)
target_link_libraries(test_vulkan_stubs PRIVATE GTest::gtest_main)
target_compile_options(test_vulkan_stubs PRIVATE -march=native -O3)
target_include_directories(test_vulkan_stubs PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests)
gtest_discover_tests(test_vulkan_stubs WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
