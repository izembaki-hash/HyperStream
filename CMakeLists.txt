cmake_minimum_required(VERSION 3.16)
project(HyperStream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(SOURCES
    src/main.cpp
    src/vulkan/draft_qwen2_runner.cpp
    src/vulkan/target_qwen2_runner.cpp
    src/vulkan/vk_engine_qwen2.cpp
    src/vulkan/vk_engine_qwen2_7b.cpp
    src/vulkan/vk_dispatch.cpp
    src/vulkan/vk_context.cpp
    src/vulkan/vk_memory.cpp
    src/vulkan/vk_pipeline.cpp
)

add_executable(hyperstream ${SOURCES})

target_compile_options(hyperstream PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -mavx2 -mf16c>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /arch:AVX2>
)

target_link_libraries(hyperstream PRIVATE OpenMP::OpenMP_CXX)

if(WIN32)
    target_link_libraries(hyperstream PRIVATE ws2_32)
endif()
