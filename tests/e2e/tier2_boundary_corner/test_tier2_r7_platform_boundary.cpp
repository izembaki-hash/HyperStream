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

class RobustSocket {
public:
    RobustSocket() : m_sock(INVALID_SOCKET) {}
    ~RobustSocket() { close(); }

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

    bool bind_and_listen(uint16_t port, bool allow_reuse = true) {
        close();
        m_sock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (!IS_VALID_SOCKET(m_sock)) return false;

        if (allow_reuse) {
            int opt = 1;
#if defined(_WIN32) || defined(_WIN64)
            setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
            setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
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

    bool accept(RobustSocket& client) {
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

bool scan_contains_emoji(const std::string& str) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(str.data());
    size_t len = str.size();
    size_t i = 0;
    while (i < len) {
        if (p[i] < 0x80) {
            i++;
        } else if ((p[i] & 0xE0) == 0xC0 && i + 1 < len) {
            i += 2;
        } else if ((p[i] & 0xF0) == 0xE0 && i + 2 < len) {
            uint32_t cp = ((p[i] & 0x0F) << 12) | ((p[i+1] & 0x3F) << 6) | (p[i+2] & 0x3F);
            // Dingbats, Miscellaneous Symbols, Miscellaneous Technical
            if ((cp >= 0x2600 && cp <= 0x27BF) || (cp >= 0x2300 && cp <= 0x23FF)) {
                return true;
            }
            i += 3;
        } else if ((p[i] & 0xF8) == 0xF0 && i + 3 < len) {
            uint32_t cp = ((p[i] & 0x07) << 18) | ((p[i+1] & 0x3F) << 12) | ((p[i+2] & 0x3F) << 6) | (p[i+3] & 0x3F);
            // Emoticons, Symbols, Pictographs, Supplemental Symbols
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

// 1. Socket Port Collision / Binding Rejection
E2E_TEST_CASE(Tier2_R7, test_r7_socket_port_collision_binding_failure) {
    ASSERT_TRUE(RobustSocket::init_networking());
    constexpr uint16_t port = 21980;

    RobustSocket s1;
    ASSERT_TRUE(s1.bind_and_listen(port, false));

    // Second socket attempting to bind to the exact same port must fail
    RobustSocket s2;
    bool ok2 = s2.bind_and_listen(port, false);
    ASSERT_FALSE(ok2);

    s1.close();
    RobustSocket::cleanup_networking();
}

// 2. Client Connection Refused to Unopened Port
E2E_TEST_CASE(Tier2_R7, test_r7_socket_client_connection_refused) {
    ASSERT_TRUE(RobustSocket::init_networking());
    constexpr uint16_t dead_port = 59998;

    RobustSocket client;
    bool ok = client.connect("127.0.0.1", dead_port);
    ASSERT_FALSE(ok); // Must fail cleanly without hanging

    RobustSocket::cleanup_networking();
}

// 3. Mmap Non-Existent File Rejection
E2E_TEST_CASE(Tier2_R7, test_r7_mmap_non_existent_file_rejection) {
    hyperstream::MmapFile mmap;
    bool ok = mmap.open("this_file_does_not_exist_at_all.gguf");
    ASSERT_FALSE(ok);
    ASSERT_FALSE(mmap.is_open());
    ASSERT_EQ(mmap.data(), nullptr);
    ASSERT_EQ(mmap.size(), 0u);
}

// 4. Mmap Zero-Byte File Rejection
E2E_TEST_CASE(Tier2_R7, test_r7_mmap_zero_byte_empty_file_rejection) {
    const std::string empty_file = "empty_zero_byte.tmp";
    {
        std::ofstream ofs(empty_file, std::ios::binary);
    } // Closes file immediately with 0 bytes

    hyperstream::MmapFile mmap;
    bool ok = mmap.open(empty_file);
    ASSERT_FALSE(ok); // Zero-byte mmap must be rejected
    ASSERT_FALSE(mmap.is_open());
    ASSERT_EQ(mmap.data(), nullptr);

    std::remove(empty_file.c_str());
}

// 5. Emoji and Control Character Adversarial Fuzzing
E2E_TEST_CASE(Tier2_R7, test_r7_emoji_and_control_character_fuzzing) {
    // Pure ASCII technical log with ANSI escape sequences: ALLOWED
    std::string ansi_log = "\033[32m[PASS]\033[0m Model loaded cleanly in 12.4 ms";
    ASSERT_FALSE(scan_contains_emoji(ansi_log));

    // ASCII symbols: ALLOWED
    std::string ascii_symbols = "[STATUS] | / - \\ ~ ! @ # $ % ^ & * ( ) _ + = { } [ ] ; : , . < > ?";
    ASSERT_FALSE(scan_contains_emoji(ascii_symbols));

    // Diverse Emoji Fuzzing: MUST ALL BE DETECTED
    std::vector<std::string> emoji_fuzz = {
        "Thumbs up \xF0\x9F\x91\x8D",         // 👍 U+1F44D
        "Sparkles \xE2\x9C\xA8",             // ✨ U+2728 (Dingbats)
        "Warning sign \xE2\x9A\xA0",         // ⚠️ U+26A0 (Misc Symbols)
        "Brain \xF0\x9F\xA7\xA0",            // 🧠 U+1F9E0
        "Check mark button \xE2\x9C\x85",    // ✅ U+2705
        "Heart \xE2\x9D\xA4",                // ❤️ U+2764
        "Smiling face \xF0\x9F\x98\x80"      // 😀 U+1F600
    };

    for (const auto& s : emoji_fuzz) {
        ASSERT_TRUE(scan_contains_emoji(s));
    }
}

// 6. Socket Abrupt Disconnect (EOF Handling)
E2E_TEST_CASE(Tier2_R7, test_r7_socket_abrupt_disconnect_handling) {
    ASSERT_TRUE(RobustSocket::init_networking());
    constexpr uint16_t port = 21981;

    RobustSocket server;
    ASSERT_TRUE(server.bind_and_listen(port));

    std::thread client_thread([&]() {
        RobustSocket client;
        if (client.connect("127.0.0.1", port)) {
            client.send("HI", 2);
            // Immediately close without reading response
            client.close();
        }
    });

    RobustSocket accepted;
    ASSERT_TRUE(server.accept(accepted));

    char buf[16] = {0};
    int r1 = accepted.recv(buf, sizeof(buf));
    ASSERT_EQ(r1, 2);

    // Next recv must return 0 or negative immediately (EOF), not hang!
    int r2 = accepted.recv(buf, sizeof(buf));
    ASSERT_TRUE(r2 <= 0);

    client_thread.join();
    accepted.close();
    server.close();
    RobustSocket::cleanup_networking();
}

int main(int argc, char** argv) {
    return TestRegistry::instance().run_all(argc, argv);
}
