#define _WIN32_WINNT 0x0600
#include <windows.h>
#include <iostream>
#include "target_qwen2_runner.h"
#include "model_topology.hpp"
#include "gguf_parser.hpp"
#include "vk_engine_qwen2_7b.h"

TargetQwen2Runner::TargetQwen2Runner() {}

TargetQwen2Runner::~TargetQwen2Runner() {
    cleanup();
}

bool TargetQwen2Runner::init(const char *model_path) {
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[Target Runner] Failed to open model: " << model_path << "\n";
        return false;
    }
    m_h_file = (void*)hFile;

    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        CloseHandle(hFile);
        m_h_file = nullptr;
        return false;
    }
    size_t file_size = static_cast<size_t>(liSize.QuadPart);

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        m_h_file = nullptr;
        return false;
    }
    m_h_map = (void*)hMap;

    const uint8_t *file_base = (const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!file_base) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        m_h_map = nullptr;
        m_h_file = nullptr;
        return false;
    }
    m_file_base = (void*)file_base;

    hyperstream::GGUFParser parser;
    if (!parser.parse_from_memory(file_base, file_size)) {
        std::cerr << "[Target Runner] Failed to parse GGUF model: " << model_path << "\n";
        return false;
    }

    hyperstream::ModelTopology topo;
    if (!hyperstream::ModelTopology::build_from_gguf(parser, topo)) {
        std::cerr << "[Target Runner] Failed to build ModelTopology from GGUF\n";
        return false;
    }

    auto *engine = new VulkanInferenceEngineQwen2_7B();
    if (!engine->init(topo)) {
        delete engine;
        return false;
    }

    if (!engine->upload_model_weights(file_base, topo)) {
        delete engine;
        return false;
    }

    m_impl = (void*)engine;
    return true;
}

void TargetQwen2Runner::reset_kv_cache() {
    if (m_impl) {
        ((VulkanInferenceEngineQwen2_7B*)m_impl)->reset_kv_cache();
    }
}

int TargetQwen2Runner::forward_token_gpu_argmax(int token_id, int pos) {
    if (!m_impl) return -1;
    return ((VulkanInferenceEngineQwen2_7B*)m_impl)->forward_token_gpu_argmax(token_id, pos);
}

void TargetQwen2Runner::cleanup() {
    if (m_impl) {
        delete (VulkanInferenceEngineQwen2_7B*)m_impl;
        m_impl = nullptr;
    }
    if (m_file_base) {
        UnmapViewOfFile(m_file_base);
        m_file_base = nullptr;
    }
    if (m_h_map) {
        CloseHandle((HANDLE)m_h_map);
        m_h_map = nullptr;
    }
    if (m_h_file) {
        CloseHandle((HANDLE)m_h_file);
        m_h_file = nullptr;
    }
}
