@echo off
echo =========================================================================
echo   Compiling HyperStream Native Vulkan LLM Inference Engine...
echo =========================================================================

if not exist bin mkdir bin

g++ -O3 -std=c++17 -fopenmp -mavx2 -mf16c ^
    -I. -Iinclude -Isrc ^
    src/main.cpp ^
    src/bpe_tokenizer.cpp ^
    src/vulkan/draft_qwen2_runner.cpp ^
    src/vulkan/target_qwen2_runner.cpp ^
    src/vulkan/vk_engine_qwen2.cpp ^
    src/vulkan/vk_engine_qwen2_7b.cpp ^
    src/vulkan/vk_dispatch.cpp ^
    src/vulkan/vk_context.cpp ^
    src/vulkan/vk_memory.cpp ^
    src/vulkan/vk_pipeline.cpp ^
    -o bin/hyperstream.exe -lws2_32

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed!
    exit /b 1
)

echo [SUCCESS] Build complete! Executable located at: bin\hyperstream.exe
