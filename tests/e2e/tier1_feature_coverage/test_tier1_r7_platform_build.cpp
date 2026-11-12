#include "../common/e2e_test_framework.hpp"
#include "../common/test_constants.hpp"
#include "../../../src/util/mmap_file.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include <thread>
#include <chrono>

#if defined(_WIN32) || defined(_WIN64)
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    #define IS_VALID_SOCKET(s) ((s) != INVALID_SOCKET)
    #define CLOSE_SOCKET(s) closesocket(s)
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    typedef int socket_t;
    #define INVALID_SOCKET (-1)
    #define IS_VALID_SOCKET(s) ((s) >= 0)
    #define CLOSE_SOCKET(s) close(s)
#endif

using namespace hyperstream::test;

namespace {

class TestSocket {
public:
    TestSocket() : m_sock(INVALID_SOCKET) {}
    ~TestSocket() { close(); }

    static bool init_networking() {
#if defined(_WIN32) || defined(_WIN64)
        WSADATA wsa;
        return (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
#else
        return true;
#endif
    }

    static void cleanup_networking() {
#if defined(_WIN32) || defined(_WIN64)
        WSACleanup();
#endif
    }

    bool bind_and_listen(uint16_t port) {
        close();
        m_sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (!IS_VALID_SOCKET(m_sock)) return false;

        int opt = 1;
#if defined(_WIN32) || defined(_WIN64)
        setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
        setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // 127.0.0.1
        addr.sin_port = htons(port);

        if (::bind(m_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            close();
            return false;
        }

        if (::listen(m_sock, 5) != 0) {
            close();
            return false;
        }

        return true;
    }

    bool connect(const std::string& host, uint16_t port) {
        close();
        m_sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (!IS_VALID_SOCKET(m_sock)) return false;

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

        if (::connect(m_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
            close();
            return false;
        }
        return true;
    }

    bool accept(TestSocket& client) {
        client.close();
        sockaddr_in client_addr{};
#if defined(_WIN32) || defined(_WIN64)
        int len = sizeof(client_addr);
#else
        socklen_t len = sizeof(client_addr);
#endif
        socket_t s = ::accept(m_sock, reinterpret_cast<sockaddr*>(&client_addr), &len);
        if (!IS_VALID_SOCKET(s)) return false;
        client.m_sock = s;
        return true;
    }

    int send(const void* data, size_t len) {
        if (!IS_VALID_SOCKET(m_sock)) return -1;
#if defined(_WIN32) || defined(_WIN64)
        return ::send(m_sock, reinterpret_cast<const char*>(data), static_cast<int>(len), 0);
#else
        return ::send(m_sock, data, len, 0);
#endif
    }

    int recv(void* data, size_t len) {
        if (!IS_VALID_SOCKET(m_sock)) return -1;
#if defined(_WIN32) || defined(_WIN64)
        return ::recv(m_sock, reinterpret_cast<char*>(data), static_cast<int>(len), 0);
#else
        return ::recv(m_sock, data, len, 0);
#endif
    }

    void close() {
        if (IS_VALID_SOCKET(m_sock)) {
            CLOSE_SOCKET(m_sock);
            m_sock = INVALID_SOCKET;
        }
    }

private:
    socket_t m_sock;
};

// Zero-Emoji Detector: checks if string contains UTF-8 emoji sequences
bool contains_emoji(const std::string& str) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(str.data());
    size_t len = str.size();
    size_t i = 0;
    while (i < len) {
        if (p[i] < 0x80) {
            // ASCII character: strictly valid
            i++;
        } else if ((p[i] & 0xE0) == 0xC0 && i + 1 < len) {
            // 2-byte UTF-8
            i += 2;
        } else if ((p[i] & 0xF0) == 0xE0 && i + 2 < len) {
            // 3-byte UTF-8: check Dingbats / Miscellaneous Symbols (0x2600-0x27BF)
            uint32_t cp = ((p[i] & 0x0F) << 12) | ((p[i+1] & 0x3F) << 6) | (p[i+2] & 0x3F);
            if ((cp >= 0x2600 && cp <= 0x27BF) || (cp >= 0x2300 && cp <= 0x23FF)) {
                return true;
            }
            i += 3;
        } else if ((p[i] & 0xF8) == 0xF0 && i + 3 < len) {
            // 4-byte UTF-8: emojis reside primarily in 0x1F000 - 0x1FAFF
            uint32_t cp = ((p[i] & 0x07) << 18) | ((p[i+1] & 0x3F) << 12) | ((p[i+2] & 0x3F) << 6) | (p[i+3] & 0x3F);
            if (cp >= 0x1F000 && cp <= 0x1FAFF) {
                return true;
            }
            i += 4;
        } else {
            i++;
        }
    }
    return false;
}

} // anonymous namespace

// 1. Socket Loopback Roundtrip Communication
E2E_TEST_CASE(Tier1_R7, test_r7_socket_loopback_roundtrip) {
    ASSERT_TRUE(TestSocket::init_networking());

    constexpr uint16_t port = 19876;
    TestSocket server;
    ASSERT_TRUE(server.bind_and_listen(port));

    const std::string payload = "HYPERSTREAM_E2E_PACKET_VERIFICATION";
    std::string received;
    bool client_ok = false;

    std::thread client_thread([&]() {
        TestSocket client;
        if (client.connect("127.0.0.1", port)) {
            client.send(payload.data(), payload.size());
            client_ok = true;
        }
    });

    TestSocket accepted;
    ASSERT_TRUE(server.accept(accepted));

    char buf[128] = {0};
    int bytes = accepted.recv(buf, sizeof(buf) - 1);
    ASSERT_TRUE(bytes > 0);
    received = std::string(buf, bytes);

    client_thread.join();
    ASSERT_TRUE(client_ok);
    ASSERT_STREQ(received, payload);

    accepted.close();
    server.close();
    TestSocket::cleanup_networking();
}

// 2. Socket Binary Buffer Invariance
E2E_TEST_CASE(Tier1_R7, test_r7_socket_blocking_and_buffer_invariants) {
    ASSERT_TRUE(TestSocket::init_networking());

    constexpr uint16_t port = 19877;
    TestSocket server;
    ASSERT_TRUE(server.bind_and_listen(port));

    std::vector<uint8_t> send_buf(4096);
    for (size_t i = 0; i < send_buf.size(); ++i) {
        send_buf[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
    }

    std::vector<uint8_t> recv_buf(4096, 0);

    std::thread client_thread([&]() {
        TestSocket client;
        if (client.connect("127.0.0.1", port)) {
            client.send(send_buf.data(), send_buf.size());
        }
    });

    TestSocket accepted;
    ASSERT_TRUE(server.accept(accepted));

    size_t total_read = 0;
    while (total_read < send_buf.size()) {
        int r = accepted.recv(recv_buf.data() + total_read, send_buf.size() - total_read);
        if (r <= 0) break;
        total_read += r;
    }

    client_thread.join();
    ASSERT_EQ(total_read, send_buf.size());
    ASSERT_EQ(std::memcmp(send_buf.data(), recv_buf.data(), send_buf.size()), 0);

    accepted.close();
    server.close();
    TestSocket::cleanup_networking();
}

// 3. Cross-Platform Memory Mapping (MmapFile) Invariant
E2E_TEST_CASE(Tier1_R7, test_r7_mmap_file_creation_and_memory_access) {
    const std::string tmp_path = "mmap_test_fixture.tmp";

    // Write deterministic binary payload
    std::vector<uint8_t> payload = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};
    {
        std::ofstream ofs(tmp_path, std::ios::binary);
        ASSERT_TRUE(ofs.is_open());
        ofs.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    }

    // Map file with hyperstream::MmapFile
    hyperstream::MmapFile mmap;
    bool ok = mmap.open(tmp_path);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(mmap.is_open());
    ASSERT_EQ(mmap.size(), payload.size());
    ASSERT_TRUE(mmap.data() != nullptr);

    // Verify content
    for (size_t i = 0; i < payload.size(); ++i) {
        ASSERT_EQ(mmap.data()[i], payload[i]);
    }

    mmap.close();
    ASSERT_FALSE(mmap.is_open());
    ASSERT_EQ(mmap.data(), nullptr);
    ASSERT_EQ(mmap.size(), 0u);

    // Cleanup temp file
    std::remove(tmp_path.c_str());
}

// 4. Zero-Emoji Logging Standard Compliance Verification
E2E_TEST_CASE(Tier1_R7, test_r7_zero_emoji_logging_standard_compliance) {
    // Valid technical ASCII messages
    std::vector<std::string> valid_logs = {
        "[INIT] HyperStream Bare-Metal Vulkan LLM Inference Engine starting...",
        "[LOAD] Model checkpoint loaded: Qwen2.5-Coder-7B",
        "[VULKAN] Physical Device: AMD Radeon RX 580 (Polaris 20, 8GB GDDR5)",
        "[VULKAN] Wavefront size: 64, Local Data Share: 64KB",
        "[E2E] Tier 1 Feature Coverage: PASS",
        "[DECODE] Throughput: 20.4 tok/s, Speculative acceptance rate: 78.5%"
    };

    for (const auto& log : valid_logs) {
        ASSERT_FALSE(contains_emoji(log));
    }

    // Adversarial tests with emojis: must be detected
    std::string emoji_rocket = "[INIT] Launching \xF0\x9F\x9A\x80 HyperStream"; // 🚀
    std::string emoji_fire   = "[PERF] Super fast \xF0\x9F\x94\xA5 decode";    // 🔥
    std::string emoji_robot  = "[AI] Assistant \xF0\x9F\xA4\x96 ready";         // 🤖

    ASSERT_TRUE(contains_emoji(emoji_rocket));
    ASSERT_TRUE(contains_emoji(emoji_fire));
    ASSERT_TRUE(contains_emoji(emoji_robot));
}

// 5. CMake Build Configuration Audit
E2E_TEST_CASE(Tier1_R7, test_r7_cmake_build_configuration_audit) {
    std::ifstream ifs("CMakeLists.txt");
    if (!ifs.is_open()) {
        // Try relative to workspace root
        ifs.open("H:\\HyperStream\\CMakeLists.txt");
    }
    ASSERT_TRUE(ifs.is_open());

    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    // Verify standard CMake commands are present
    ASSERT_TRUE(content.find("cmake_minimum_required") != std::string::npos);
    ASSERT_TRUE(content.find("project(HyperStream") != std::string::npos || content.find("project(hyperstream") != std::string::npos);
    ASSERT_TRUE(content.find("CXX_STANDARD 17") != std::string::npos || content.find("CMAKE_CXX_STANDARD 17") != std::string::npos);

    // Invariant: Zero-Python mandate in runtime pipeline
    ASSERT_TRUE(content.find("export_qwen_merges.py") == std::string::npos);
    ASSERT_TRUE(content.find("generate_qwen_layout_header.py") == std::string::npos);
}

// 6. Socket Re-Initialization and Multiple Open/Close Cycles
E2E_TEST_CASE(Tier1_R7, test_r7_clean_socket_shutdown_and_cleanup) {
    ASSERT_TRUE(TestSocket::init_networking());

    for (int cycle = 0; cycle < 5; ++cycle) {
        TestSocket s;
        bool ok = s.bind_and_listen(20000 + cycle);
        ASSERT_TRUE(ok);
        s.close();
    }

    TestSocket::cleanup_networking();
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
