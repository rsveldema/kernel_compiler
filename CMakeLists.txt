cmake_minimum_required(VERSION 3.20)
project(rllm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(CTest)

add_subdirectory(runtime)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
