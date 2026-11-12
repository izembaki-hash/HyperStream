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
    src/bpe_tokenizer.cpp
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

# Tests
option(BUILD_TESTING "Build HyperStream test suite" ON)
if(BUILD_TESTING)
    enable_testing()

    add_executable(test_gguf_parser tests/test_gguf_parser.cpp)
    target_include_directories(test_gguf_parser PRIVATE include src)
    add_test(NAME test_gguf_parser COMMAND test_gguf_parser)

    add_executable(test_bpe_tokenizer tests/test_bpe_tokenizer.cpp src/bpe_tokenizer.cpp)
    target_include_directories(test_bpe_tokenizer PRIVATE include src)
    add_test(NAME test_bpe_tokenizer COMMAND test_bpe_tokenizer)

    add_executable(test_sockets tests/test_sockets.cpp)
    target_include_directories(test_sockets PRIVATE include src)
    if(WIN32)
        target_link_libraries(test_sockets PRIVATE ws2_32)
    endif()
    add_test(NAME test_sockets COMMAND test_sockets)

    add_executable(test_model_topology tests/test_model_topology.cpp)
    target_include_directories(test_model_topology PRIVATE include src tests)
    add_test(NAME test_model_topology COMMAND test_model_topology)

    add_executable(test_paged_kv tests/test_paged_kv.cpp)
    target_include_directories(test_paged_kv PRIVATE include src tests)
    add_test(NAME test_paged_kv COMMAND test_paged_kv)

    add_executable(test_spec_tree tests/test_spec_tree.cpp)
    target_include_directories(test_spec_tree PRIVATE include src tests)
    add_test(NAME test_spec_tree COMMAND test_spec_tree)
endif()

