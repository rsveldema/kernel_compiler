#pragma once

#include <print>

#define LOG_INFO_EVERY_N(...) \
    do \
    { \
        static int counter = 0; \
        if (counter++ % 100 == 0) \
        { \
            LOG_INFO(__VA_ARGS__); \
        } \
    } while (0)

extern std::ofstream s_nn_log;


#define LOG_INFO(...)  { if(s_nn_log.is_open()) { \
     s_nn_log << std::format(__VA_ARGS__) << '\n' << std::flush; \
} else { \
     std::println(stderr, __VA_ARGS__); \
} }



#define LOG_ERROR(...)  { if(s_nn_log.is_open()) { \
     s_nn_log << std::format(__VA_ARGS__) << '\n' << std::flush; \
} else { \
     std::println(stderr, __VA_ARGS__); \
} }


void set_nn_log_file(const std::string& filename);
