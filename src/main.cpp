#define _WIN32_WINNT 0x0600
#include <winsock2.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
#include <shellapi.h>

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

#include "vulkan/draft_qwen2_runner.h"
#include "vulkan/target_qwen2_runner.h"
#include "bpe_tokenizer.hpp"

static const uint32_t *g_vocab_offsets = NULL;
static const char *g_vocab_strings = NULL;
static BPETokenizer g_tokenizer;
static std::string g_html_content;

static TargetQwen2Runner g_target_engine;
static DraftQwen2Runner  g_draft_engine;
static bool g_draft_loaded = false;

void load_vocab(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t*)malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);
    g_vocab_offsets = (const uint32_t*)buf;
    g_vocab_strings = (const char*)(buf + 151936 * sizeof(uint32_t));
}

const char* get_token_str(int token_id) {
    if (!g_vocab_strings || token_id < 0 || token_id >= 151936) return "";
    return g_vocab_strings + g_vocab_offsets[token_id];
}

std::string url_decode(const std::string &src) {
    std::string ret;
    char ch;
    int i, ii;
    for (i = 0; i < (int)src.length(); i++) {
        if (src[i] == '%') {
            sscanf(src.substr(i + 1, 2).c_str(), "%x", &ii);
            ch = static_cast<char>(ii);
            ret += ch;
            i = i + 2;
        } else if (src[i] == '+') {
            ret += ' ';
        } else {
            ret += src[i];
        }
    }
    return ret;
}

std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
                break;
        }
    }
    return out;
}

bool file_exists(const std::string &path) {
    DWORD dwAttrib = GetFileAttributesA(path.c_str());
    return (dwAttrib != INVALID_FILE_ATTRIBUTES && !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

int main(int argc, char **argv) {
    SetConsoleOutputCP(CP_UTF8);

    int port = 8080;
    std::string model_target = "models/qwen2.5-coder-7b-instruct-q4_0.gguf";
    std::string model_draft  = "models/qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    std::string web_path     = "web/web_ui.html";

    // Auto-detect common alternate paths
    if (!file_exists(model_target) && file_exists("H:\\ai model\\qwen2.5-coder-7b-instruct-q4_0.gguf")) {
        model_target = "H:\\ai model\\qwen2.5-coder-7b-instruct-q4_0.gguf";
    }
    if (!file_exists(model_draft) && file_exists("H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf")) {
        model_draft = "H:\\ai model\\qwen2.5-coder-0.5b-instruct-q8_0.gguf";
    }
    if (!file_exists(web_path) && file_exists("web_ui.html")) {
        web_path = "web_ui.html";
    }

    // CLI Argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_target = argv[++i];
        else if (arg == "--draft" && i + 1 < argc) model_draft = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = atoi(argv[++i]);
        else if (arg == "--web" && i + 1 < argc) web_path = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: hyperstream [options]\n"
                      << "  --model <path>  Target model (.gguf) [default: models/qwen2.5-coder-7b-instruct-q4_0.gguf]\n"
                      << "  --draft <path>  Draft model (.gguf)  [default: models/qwen2.5-coder-0.5b-instruct-q8_0.gguf]\n"
                      << "  --port <num>    Web server port      [default: 8080]\n"
                      << "  --web <path>    Web UI HTML file     [default: web/web_ui.html]\n";
            return 0;
        }
    }

    std::cout << "=========================================================================================\n";
    std::cout << "   ⚡ HYPERSTREAM: HIGH-PERFORMANCE BARE-METAL VULKAN INFERENCE ENGINE\n";
    std::cout << "   Dual-Resident LLM Acceleration for AMD Radeon & Vulkan-Compatible GPUs\n";
    std::cout << "=========================================================================================\n\n";

    // 1. Load HTML
    std::ifstream html_file(web_path);
    if (html_file.is_open()) {
        std::stringstream buffer;
        buffer << html_file.rdbuf();
        g_html_content = buffer.str();
    } else {
        std::cerr << "Warning: Could not open web UI at: " << web_path << "\n";
    }

    // 2. Load Vocab & Tokenizer
    std::cout << "[1/3] Loading Tokenizer Tables...\n";
    std::string vocab_tok = file_exists("models/qwen_vocab_tokens.bin") ? "models/qwen_vocab_tokens.bin" : "qwen_vocab_tokens.bin";
    std::string vocab_map = file_exists("models/qwen_vocab_map.bin") ? "models/qwen_vocab_map.bin" : "qwen_vocab_map.bin";
    std::string vocab_mrg = file_exists("models/qwen_merges_table.bin") ? "models/qwen_merges_table.bin" : "qwen_merges_table.bin";

    if (!file_exists(vocab_tok) && file_exists("H:\\asurrini\\qwen_vocab_tokens.bin")) {
        vocab_tok = "H:\\asurrini\\qwen_vocab_tokens.bin";
        vocab_map = "H:\\asurrini\\qwen_vocab_map.bin";
        vocab_mrg = "H:\\asurrini\\qwen_merges_table.bin";
    }

    load_vocab(vocab_tok.c_str());
    if (!g_tokenizer.load(vocab_map.c_str(), vocab_mrg.c_str())) {
        std::cerr << "Failed to load tokenizer binary tables!\n"
                  << "Please run 'python scripts/export_qwen_merges.py' to generate them.\n";
        return 1;
    }
    std::cout << "      Tokenizer Ready: " << g_tokenizer.vocab.size() << " tokens loaded.\n\n";

    // 3. Initialize Target Model
    std::cout << "[2/3] Uploading Target Model to VRAM: " << model_target << " ...\n";
    if (!g_target_engine.init(model_target.c_str())) {
        std::cerr << "Error: Target model failed to initialize! Check file path and VRAM capacity.\n";
        return 1;
    }

    // 4. Initialize Draft Model (Optional)
    if (file_exists(model_draft)) {
        std::cout << "[3/3] Uploading Draft Model to VRAM: " << model_draft << " ...\n";
        if (g_draft_engine.init(model_draft.c_str())) {
            g_draft_loaded = true;
            std::cout << "      ✅ Both Models Resident Concurrently in GPU VRAM (Speculative Mode Active)!\n\n";
        } else {
            std::cout << "      ⚠️ Draft model initialization skipped. Standalone mode active.\n\n";
        }
    } else {
        std::cout << "[3/3] Draft model not found (" << model_draft << "). Running in standalone Target mode.\n\n";
    }

    // 5. Initialize Winsock HTTP Server
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2,2), &wsa) != 0) {
        std::cerr << "WSAStartup failed!\n";
        return 1;
    }

    SOCKET server_sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port);

    if (bind(server_sock, (sockaddr*)&server_addr, sizeof(server_addr)) == SOCKET_ERROR) {
        std::cerr << "Bind failed on port " << port << "!\n";
        return 1;
    }

    listen(server_sock, 10);
    std::cout << "=========================================================================================\n";
    std::cout << "🌐 HYPERSTREAM SERVER READY at: http://localhost:" << port << "\n";
    std::cout << "   Open your web browser to start real-time GPU inference.\n";
    std::cout << "=========================================================================================\n\n";

    while (true) {
        sockaddr_in client_addr;
        int client_len = sizeof(client_addr);
        SOCKET client_sock = accept(server_sock, (sockaddr*)&client_addr, &client_len);
        if (client_sock == INVALID_SOCKET) continue;

        char req_buf[4096] = {0};
        int bytes_read = recv(client_sock, req_buf, sizeof(req_buf) - 1, 0);
        if (bytes_read <= 0) {
            closesocket(client_sock);
            continue;
        }

        std::string request(req_buf);
        if (request.rfind("GET / ", 0) == 0 || request.rfind("GET /index.html", 0) == 0) {
            std::ifstream cur_html_file(web_path);
            std::string cur_html = g_html_content;
            if (cur_html_file.is_open()) {
                std::stringstream buffer;
                buffer << cur_html_file.rdbuf();
                cur_html = buffer.str();
            }
            std::string resp = "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: " +
                               std::to_string(cur_html.size()) + "\r\nConnection: close\r\n\r\n" + cur_html;
            send(client_sock, resp.c_str(), (int)resp.size(), 0);
            closesocket(client_sock);
        } else if (request.rfind("GET /api/generate", 0) == 0) {
            std::string prompt = "";
            size_t p_pos = request.find("prompt=");
            if (p_pos != std::string::npos) {
                p_pos += 7;
                size_t amp_pos = request.find("&", p_pos);
                size_t spc_pos = request.find(" ", p_pos);
                size_t p_end = (amp_pos < spc_pos) ? amp_pos : spc_pos;
                if (p_end == std::string::npos) p_end = request.size();
                prompt = url_decode(request.substr(p_pos, p_end - p_pos));
            }

            std::string backend = "vulkan";
            size_t b_pos = request.find("backend=");
            if (b_pos != std::string::npos) {
                b_pos += 8;
                size_t amp_pos = request.find("&", b_pos);
                size_t spc_pos = request.find(" ", b_pos);
                size_t b_end = (amp_pos < spc_pos) ? amp_pos : spc_pos;
                if (b_end == std::string::npos) b_end = request.size();
                backend = request.substr(b_pos, b_end - b_pos);
            }

            if (backend == "speculative" && !g_draft_loaded) {
                backend = "vulkan";
            }

            std::cout << "📩 Request: \"" << prompt << "\" [Mode: " << backend << "]\n";

            std::string sse_hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
            send(client_sock, sse_hdr.c_str(), (int)sse_hdr.size(), 0);

            // Construct Qwen ChatML Format
            std::vector<int> prompt_tokens;
            prompt_tokens.push_back(151644); // <|im_start|>
            for (int t : g_tokenizer.tokenize("system\nYou are a helpful and concise coding assistant.")) prompt_tokens.push_back(t);
            prompt_tokens.push_back(151645); // <|im_end|>
            prompt_tokens.push_back(198);    // \n

            prompt_tokens.push_back(151644); // <|im_start|>
            for (int t : g_tokenizer.tokenize("user\n" + prompt)) prompt_tokens.push_back(t);
            prompt_tokens.push_back(151645); // <|im_end|>
            prompt_tokens.push_back(198);    // \n

            prompt_tokens.push_back(151644); // <|im_start|>
            for (int t : g_tokenizer.tokenize("assistant\n")) prompt_tokens.push_back(t);

            int prompt_len = (int)prompt_tokens.size();

            // Reset KV Caches
            g_target_engine.reset_kv_cache();
            if (backend == "speculative") {
                g_draft_engine.reset_kv_cache();
            }

            // Prompt Ingestion
            int cur_token = prompt_tokens[0];
            for (int pos = 0; pos < prompt_len; pos++) {
                cur_token = g_target_engine.forward_token_gpu_argmax(prompt_tokens[pos], pos);
                if (backend == "speculative") {
                    g_draft_engine.forward_token_gpu_argmax(prompt_tokens[pos], pos);
                }
            }

            auto gen_t0 = std::chrono::high_resolution_clock::now();
            int tokens_generated = 0;
            const int MAX_GEN = 512;
            int cur_pos = prompt_len;

            if (backend == "speculative") {
                const int K = 3;
                int total_drafted = 0;
                int total_accepted = 0;

                if (cur_token != 151645 && cur_token != 151643) {
                    const char *piece = get_token_str(cur_token);
                    tokens_generated++;
                    std::string sse_evt = "data: {\"token\": \"" + json_escape(piece) + "\", \"tok_sec\": 18.0, \"accept_rate\": 0.0}\n\n";
                    send(client_sock, sse_evt.c_str(), (int)sse_evt.size(), 0);
                }

                while (tokens_generated < MAX_GEN) {
                    if (cur_token == 151645 || cur_token == 151643) break;

                    std::vector<int> drafted(K);
                    int d_tok = cur_token;
                    for (int k = 0; k < K; k++) {
                        drafted[k] = g_draft_engine.forward_token_gpu_argmax(d_tok, cur_pos + k);
                        d_tok = drafted[k];
                        total_drafted++;
                    }

                    int tokens_advanced = 0;
                    int verify_tok = cur_token;
                    int next_seq_tok = cur_token;
                    bool client_disconnected = false;

                    for (int k = 0; k < K; k++) {
                        int target_pred = g_target_engine.forward_token_gpu_argmax(verify_tok, cur_pos + k);

                        if (target_pred == drafted[k]) {
                            total_accepted++;
                            tokens_generated++;
                            tokens_advanced++;

                            const char *piece = get_token_str(drafted[k]);
                            auto now = std::chrono::high_resolution_clock::now();
                            double el_s = std::chrono::duration<double>(now - gen_t0).count();
                            double tps = (el_s > 0) ? (tokens_generated / el_s) : 0.0;
                            double acc = (total_drafted > 0) ? ((double)total_accepted / total_drafted * 100.0) : 0.0;

                            std::string sse_evt = "data: {\"token\": \"" + json_escape(piece) + "\", \"tok_sec\": " + std::to_string(tps) + ", \"accept_rate\": " + std::to_string(acc) + "}\n\n";
                            if (send(client_sock, sse_evt.c_str(), (int)sse_evt.size(), 0) == SOCKET_ERROR) {
                                client_disconnected = true;
                                break;
                            }

                            verify_tok = drafted[k];
                            if (drafted[k] == 151645 || drafted[k] == 151643) {
                                next_seq_tok = drafted[k];
                                break;
                            }

                            if (k == K - 1) {
                                int bonus_tok = g_target_engine.forward_token_gpu_argmax(drafted[k], cur_pos + K);
                                tokens_generated++;
                                tokens_advanced++;
                                const char *b_piece = get_token_str(bonus_tok);
                                std::string sse_bonus = "data: {\"token\": \"" + json_escape(b_piece) + "\", \"tok_sec\": " + std::to_string(tps) + ", \"accept_rate\": " + std::to_string(acc) + "}\n\n";
                                send(client_sock, sse_bonus.c_str(), (int)sse_bonus.size(), 0);
                                next_seq_tok = bonus_tok;
                            }
                        } else {
                            tokens_generated++;
                            tokens_advanced++;
                            const char *piece = get_token_str(target_pred);
                            auto now = std::chrono::high_resolution_clock::now();
                            double el_s = std::chrono::duration<double>(now - gen_t0).count();
                            double tps = (el_s > 0) ? (tokens_generated / el_s) : 0.0;
                            double acc = (total_drafted > 0) ? ((double)total_accepted / total_drafted * 100.0) : 0.0;

                            std::string sse_evt = "data: {\"token\": \"" + json_escape(piece) + "\", \"tok_sec\": " + std::to_string(tps) + ", \"accept_rate\": " + std::to_string(acc) + "}\n\n";
                            if (send(client_sock, sse_evt.c_str(), (int)sse_evt.size(), 0) == SOCKET_ERROR) {
                                client_disconnected = true;
                            }
                            next_seq_tok = target_pred;
                            break;
                        }
                    }

                    if (client_disconnected) break;
                    cur_pos += tokens_advanced;
                    cur_token = next_seq_tok;
                }
            } else {
                for (int step = 0; step < MAX_GEN; step++) {
                    if (cur_token == 151645 || cur_token == 151643) break;

                    const char *piece = get_token_str(cur_token);
                    tokens_generated++;

                    auto now = std::chrono::high_resolution_clock::now();
                    double el_s = std::chrono::duration<double>(now - gen_t0).count();
                    double tps = (el_s > 0) ? (tokens_generated / el_s) : 0.0;

                    std::string sse_evt = "data: {\"token\": \"" + json_escape(piece) + "\", \"tok_sec\": " + std::to_string(tps) + ", \"accept_rate\": 100.0}\n\n";
                    if (send(client_sock, sse_evt.c_str(), (int)sse_evt.size(), 0) == SOCKET_ERROR) {
                        break;
                    }

                    cur_token = g_target_engine.forward_token_gpu_argmax(cur_token, cur_pos);
                    cur_pos++;
                }
            }

            std::string done_evt = "data: [DONE]\n\n";
            send(client_sock, done_evt.c_str(), (int)done_evt.size(), 0);
            closesocket(client_sock);

            auto gen_t1 = std::chrono::high_resolution_clock::now();
            double total_time_ms = std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count();
            if (tokens_generated > 0) {
                std::cout << "Done: " << tokens_generated << " tokens in " << total_time_ms << " ms ("
                          << std::fixed << std::setprecision(2) << (tokens_generated / (total_time_ms / 1000.0)) << " Token/sec)\n\n";
            }
        } else {
            std::string resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_sock, resp.c_str(), (int)resp.size(), 0);
            closesocket(client_sock);
        }
    }

    closesocket(server_sock);
    WSACleanup();
    g_target_engine.cleanup();
    if (g_draft_loaded) g_draft_engine.cleanup();
    return 0;
}
