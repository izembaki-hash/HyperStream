#include "e2e/common/e2e_test_framework.hpp"
#include "net/socket.hpp"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iostream>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#endif

using namespace hyperstream;
using namespace hyperstream::test;

static uint32_t get_current_handle_count() {
#if defined(_WIN32) || defined(_WIN64)
    DWORD handle_count = 0;
    if (GetProcessHandleCount(GetCurrentProcess(), &handle_count)) {
        return static_cast<uint32_t>(handle_count);
    }
#endif
    return 0;
}

// 1. Simultaneous Burst Connections: 25 threads connect concurrently on gate trigger
E2E_TEST_CASE(SocketConcurrencyTest, test_simultaneous_burst_connections) {
    const uint16_t port = 28100;
    Socket server;
    ASSERT_TRUE(server.listen(port));

    const int NUM_CLIENTS = 25;
    std::atomic<bool> gate{false};
    std::atomic<int> success_count{0};
    std::atomic<int> failure_count{0};

    std::vector<std::thread> client_threads;
    client_threads.reserve(NUM_CLIENTS);

    for (int i = 0; i < NUM_CLIENTS; ++i) {
        client_threads.emplace_back([i, port, &gate, &success_count, &failure_count]() {
            // Wait for simultaneous release
            while (!gate.load(std::memory_order_relaxed)) {
                std::this_thread::yield();
            }

            Socket client;
            if (!client.connect("127.0.0.1", port)) {
                failure_count.fetch_add(1);
                return;
            }

            std::string msg = "CLIENT_BURST_" + std::to_string(i) + "\n";
            if (client.send(msg.data(), msg.size()) != static_cast<int>(msg.size())) {
                failure_count.fetch_add(1);
                return;
            }

            char buf[128] = {0};
            int n = client.recv(buf, sizeof(buf) - 1);
            if (n > 0) {
                std::string ack(buf, n);
                std::string expected = "ACK_" + std::to_string(i) + "\n";
                if (ack == expected) {
                    success_count.fetch_add(1);
                } else {
                    failure_count.fetch_add(1);
                }
            } else {
                failure_count.fetch_add(1);
            }
            client.close();
        });
    }

    // Thread to accept and handle connections on server
    std::thread server_thread([&]() {
        // Release client gate
        gate.store(true, std::memory_order_release);

        for (int i = 0; i < NUM_CLIENTS; ++i) {
            Socket client_sock;
            if (server.accept(client_sock)) {
                char buf[128] = {0};
                int n = client_sock.recv(buf, sizeof(buf) - 1);
                if (n > 0) {
                    std::string req(buf, n);
                    // Extract client id from "CLIENT_BURST_<id>\n"
                    size_t prefix_len = std::string("CLIENT_BURST_").size();
                    if (req.size() > prefix_len) {
                        std::string id_str = req.substr(prefix_len);
                        // Strip newline
                        while (!id_str.empty() && (id_str.back() == '\n' || id_str.back() == '\r')) {
                            id_str.pop_back();
                        }
                        std::string ack = "ACK_" + id_str + "\n";
                        client_sock.send(ack.data(), ack.size());
                    }
                }
                client_sock.close();
            }
        }
    });

    server_thread.join();
    for (auto& t : client_threads) {
        t.join();
    }
    server.close();

    ASSERT_EQ(success_count.load(), NUM_CLIENTS);
    ASSERT_EQ(failure_count.load(), 0);
}

// 2. High-Frequency Concurrent Data Streaming: 8 clients * 15 round-trips
E2E_TEST_CASE(SocketConcurrencyTest, test_concurrent_multi_client_data_streaming) {
    const uint16_t port = 28101;
    Socket server;
    ASSERT_TRUE(server.listen(port));

    const int NUM_CLIENTS = 8;
    const int ROUNDS_PER_CLIENT = 15;
    std::atomic<bool> server_running{true};
    std::atomic<int> total_completed_rounds{0};

    // Server accepts connections and spawns lightweight handlers
    std::vector<std::thread> server_handlers;
    std::thread server_listener([&]() {
        while (server_running.load()) {
            Socket client_conn;
            if (server.accept(client_conn)) {
                server_handlers.emplace_back([sock = std::move(client_conn)]() mutable {
                    while (true) {
                        uint32_t val = 0;
                        int n = sock.recv(&val, sizeof(val));
                        if (n <= 0) break;
                        uint32_t reply = val ^ 0xAAAAAAAA;
                        if (sock.send(&reply, sizeof(reply)) <= 0) break;
                    }
                    sock.close();
                });
            }
        }
    });

    // Run client threads
    std::vector<std::thread> client_threads;
    client_threads.reserve(NUM_CLIENTS);
    for (int i = 0; i < NUM_CLIENTS; ++i) {
        client_threads.emplace_back([i, port, ROUNDS_PER_CLIENT, &total_completed_rounds]() {
            Socket client;
            if (!client.connect("127.0.0.1", port)) return;

            for (int r = 0; r < ROUNDS_PER_CLIENT; ++r) {
                uint32_t send_val = static_cast<uint32_t>((i << 16) | r);
                int sent = client.send(&send_val, sizeof(send_val));
                if (sent != sizeof(send_val)) break;

                uint32_t recv_val = 0;
                int recvd = client.recv(&recv_val, sizeof(recv_val));
                if (recvd != sizeof(recv_val)) break;

                if (recv_val == (send_val ^ 0xAAAAAAAA)) {
                    total_completed_rounds.fetch_add(1);
                }
            }
            client.close();
        });
    }

    for (auto& t : client_threads) {
        t.join();
    }

    // Stop server
    server_running.store(false);
    // Unblock accept by connecting a dummy socket
    Socket dummy;
    dummy.connect("127.0.0.1", port);
    dummy.close();
    server.close();

    server_listener.join();
    for (auto& h : server_handlers) {
        if (h.joinable()) h.join();
    }

    ASSERT_EQ(total_completed_rounds.load(), NUM_CLIENTS * ROUNDS_PER_CLIENT);
}

// 3. Socket Handle Churn: 300 rapid connect/disconnect cycles verify ZERO handle leak
E2E_TEST_CASE(SocketConcurrencyTest, test_socket_handle_churn_and_cleanup) {
    const uint16_t port = 28102;
    Socket server;
    ASSERT_TRUE(server.listen(port));

    // Warm-up 5 cycles so Winsock initializes its thread/handle pool
    for (int i = 0; i < 5; ++i) {
        Socket c;
        if (c.connect("127.0.0.1", port)) {
            Socket s;
            server.accept(s);
            s.close();
            c.close();
        }
    }

    uint32_t h_baseline = get_current_handle_count();

    const int ITERATIONS = 300;
    for (int i = 0; i < ITERATIONS; ++i) {
        Socket client;
        ASSERT_TRUE(client.connect("127.0.0.1", port));
        Socket accepted;
        ASSERT_TRUE(server.accept(accepted));
        accepted.close();
        client.close();
    }

    server.close();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    uint32_t h_after_300 = get_current_handle_count();
    // Verify zero handle leak after 300 rapid connections (handles at end must be <= baseline)
    ASSERT_TRUE(h_after_300 <= h_baseline);
}

// 4. Abrupt Client Disconnect and Closed Port Resilience
E2E_TEST_CASE(SocketConcurrencyTest, test_abrupt_disconnect_and_closed_port_resilience) {
    const uint16_t port = 28103;
    Socket server;
    ASSERT_TRUE(server.listen(port));

    // Test client immediately closing connection after sending 1 byte
    std::thread client_thread([port]() {
        Socket client;
        if (client.connect("127.0.0.1", port)) {
            char b = 'X';
            client.send(&b, 1);
            client.close(); // Abrupt disconnect
        }
    });

    Socket incoming;
    ASSERT_TRUE(server.accept(incoming));
    client_thread.join();

    // Server reading after client closed should return <= 0 gracefully
    char buf[16];
    int n = incoming.recv(buf, sizeof(buf));
    if (n > 0) {
        int n2 = incoming.recv(buf, sizeof(buf));
        ASSERT_TRUE(n2 <= 0); // Second recv must cleanly hit EOF
    }
    incoming.close();
    server.close();

    // 10 concurrent threads connecting to closed port
    std::atomic<int> failed_connects{0};
    std::vector<std::thread> fail_threads;
    for (int i = 0; i < 10; ++i) {
        fail_threads.emplace_back([&failed_connects]() {
            Socket s;
            if (!s.connect("127.0.0.1", 29998)) {
                failed_connects.fetch_add(1);
            }
        });
    }
    for (auto& t : fail_threads) {
        t.join();
    }
    ASSERT_EQ(failed_connects.load(), 10);
}

int main(int argc, char** argv) {
    Socket::init_networking();
    int res = TestRegistry::instance().run_all(argc, argv);
    Socket::cleanup_networking();
    return res;
}
