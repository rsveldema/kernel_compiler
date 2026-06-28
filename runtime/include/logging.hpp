#pragma once

#include <print>

#ifndef LOG_INFO
#define LOG_INFO(...) std::println(stderr, __VA_ARGS__)
#endif

#ifndef LOG_ERROR
#define LOG_ERROR(...) std::println(stderr, __VA_ARGS__)
#endif