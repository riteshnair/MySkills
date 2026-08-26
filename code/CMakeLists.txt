cmake_minimum_required(VERSION 3.16)
project(tiny_lm LANGUAGES C CXX)

option(LM_BUILD_TESTS "Build focused core tests" ON)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(lm_core STATIC core/lm.cpp core/model.cpp core/kv.cpp)
target_include_directories(lm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(lm_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

add_executable(tiny-lm cli/main.cpp)
target_link_libraries(tiny-lm PRIVATE lm_core)
target_compile_options(tiny-lm PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(LM_BUILD_TESTS)
    enable_testing()
    add_executable(lm-core-tests tests/test_core.cpp)
    target_link_libraries(lm-core-tests PRIVATE lm_core)
    target_compile_options(lm-core-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME lm-core-tests COMMAND lm-core-tests)
endif()
