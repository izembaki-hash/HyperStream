#ifndef HYPERSTREAM_NET_SOCKET_HPP
#define HYPERSTREAM_NET_SOCKET_HPP

#include <cstdint>
#include <cstddef>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    using hs_socket_t = SOCKET;
    constexpr hs_socket_t HS_INVALID_SOCKET = INVALID_SOCKET;
    constexpr int HS_SOCKET_ERROR = SOCKET_ERROR;
#else
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>
    using hs_socket_t = int;
    constexpr hs_socket_t HS_INVALID_SOCKET = -1;
    constexpr int HS_SOCKET_ERROR = -1;
#endif

namespace hyperstream {

class Socket {
public:
    Socket() : m_sock(HS_INVALID_SOCKET) {}
    explicit Socket(hs_socket_t sock) : m_sock(sock) {}
    ~Socket() { close(); }

    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;

    Socket(Socket&& other) noexcept : m_sock(other.m_sock) {
        other.m_sock = HS_INVALID_SOCKET;
    }

    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            close();
            m_sock = other.m_sock;
            other.m_sock = HS_INVALID_SOCKET;
        }
        return *this;
    }

    bool listen(uint16_t port) {
        close();
        m_sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (m_sock == HS_INVALID_SOCKET) {
            return false;
        }

        int opt = 1;
#if defined(_WIN32) || defined(_WIN64)
        ::setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
        ::setsockopt(m_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (::bind(m_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == HS_SOCKET_ERROR) {
            close();
            return false;
        }

        if (::listen(m_sock, 128) == HS_SOCKET_ERROR) {
            close();
            return false;
        }

        return true;
    }

    bool connect(const std::string& host, uint16_t port) {
        close();
        m_sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (m_sock == HS_INVALID_SOCKET) {
            return false;
        }

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
            close();
            return false;
        }

        if (::connect(m_sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == HS_SOCKET_ERROR) {
            close();
            return false;
        }

        return true;
    }

    bool accept(Socket& client) {
        if (m_sock == HS_INVALID_SOCKET) {
            return false;
        }

        sockaddr_in client_addr{};
#if defined(_WIN32) || defined(_WIN64)
        int client_len = sizeof(client_addr);
#else
        socklen_t client_len = sizeof(client_addr);
#endif
        hs_socket_t client_fd = ::accept(m_sock, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        if (client_fd == HS_INVALID_SOCKET) {
            return false;
        }

        client.close();
        client.m_sock = client_fd;
        return true;
    }

    int send(const void* data, size_t len) {
        if (m_sock == HS_INVALID_SOCKET) return -1;
        const char* ptr = reinterpret_cast<const char*>(data);
        size_t total = 0;
        while (total < len) {
#if defined(_WIN32) || defined(_WIN64)
            int n = ::send(m_sock, ptr + total, static_cast<int>(len - total), 0);
#else
            int n = ::send(m_sock, ptr + total, len - total, MSG_NOSIGNAL);
#endif
            if (n <= 0) {
                return (total > 0) ? static_cast<int>(total) : n;
            }
            total += static_cast<size_t>(n);
        }
        return static_cast<int>(total);
    }

    int recv(void* data, size_t len) {
        if (m_sock == HS_INVALID_SOCKET) return -1;
#if defined(_WIN32) || defined(_WIN64)
        return ::recv(m_sock, reinterpret_cast<char*>(data), static_cast<int>(len), 0);
#else
        return ::recv(m_sock, data, len, 0);
#endif
    }

    void close() {
        if (m_sock != HS_INVALID_SOCKET) {
#if defined(_WIN32) || defined(_WIN64)
            ::closesocket(m_sock);
#else
            ::close(m_sock);
#endif
            m_sock = HS_INVALID_SOCKET;
        }
    }

    bool is_valid() const {
        return m_sock != HS_INVALID_SOCKET;
    }

    hs_socket_t native_handle() const {
        return m_sock;
    }

    static bool init_networking() {
#if defined(_WIN32) || defined(_WIN64)
        WSADATA wsa;
        return ::WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#else
        return true;
#endif
    }

    static void cleanup_networking() {
#if defined(_WIN32) || defined(_WIN64)
        ::WSACleanup();
#endif
    }

private:
    hs_socket_t m_sock = HS_INVALID_SOCKET;
};

} // namespace hyperstream

#endif // HYPERSTREAM_NET_SOCKET_HPP
