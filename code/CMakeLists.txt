cmake_minimum_required(VERSION 3.16)
project(tiny_lm LANGUAGES C CXX)

option(LM_BUILD_TESTS "Build focused core tests" ON)
option(LM_ENABLE_VULKAN "Build Vulkan device discovery" ON)


set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(lm_core STATIC core/lm.cpp core/model.cpp core/kv.cpp core/kernel.cpp core/tensor.cpp)
target_include_directories(lm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(lm_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

if(LM_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
endif()
if(Vulkan_FOUND)
    target_sources(lm_core PRIVATE vulkan/device.cpp vulkan/dp4.cpp)
    target_link_libraries(lm_core PUBLIC Vulkan::Vulkan)
    target_compile_definitions(lm_core PUBLIC LM_HAS_VULKAN=1)
else()
    target_sources(lm_core PRIVATE vulkan/device_stub.cpp vulkan/dp4_stub.cpp)
    target_compile_definitions(lm_core PUBLIC LM_HAS_VULKAN=0)
endif()

add_executable(tiny-lm cli/main.cpp)
target_link_libraries(tiny-lm PRIVATE lm_core)
target_compile_options(tiny-lm PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)

find_program(GLSLANG_VALIDATOR glslangValidator)
if(GLSLANG_VALIDATOR)
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dot_i8_dp4.comp.spv
        COMMAND ${GLSLANG_VALIDATOR} -V --target-env vulkan1.1
                ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/dot_i8_dp4.comp
                -o ${CMAKE_CURRENT_BINARY_DIR}/dot_i8_dp4.comp.spv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vulkan/shaders/dot_i8_dp4.comp
        VERBATIM)
    add_custom_target(lm-vulkan-shaders ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dot_i8_dp4.comp.spv)
endif()

if(LM_BUILD_TESTS)
    enable_testing()
    add_executable(lm-core-tests tests/test_core.cpp)
    target_link_libraries(lm-core-tests PRIVATE lm_core)
    target_compile_options(lm-core-tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    add_test(NAME lm-core-tests COMMAND lm-core-tests)
endif()
