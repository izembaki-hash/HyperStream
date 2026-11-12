#include "e2e/common/e2e_test_framework.hpp"
#include "net/socket.hpp"
#include <thread>
#include <chrono>

using namespace hyperstream;
using namespace hyperstream::test;

// 1. Networking Lifecycle Initialization & Cleanup
E2E_TEST_CASE(SocketTest, test_networking_init_cleanup) {
    bool ok = Socket::init_networking();
    ASSERT_TRUE(ok);
    Socket::cleanup_networking();
    // Re-initialize for subsequent test cases
    ASSERT_TRUE(Socket::init_networking());
}

// 2. Server Socket Listen on Ephemeral Port
E2E_TEST_CASE(SocketTest, test_socket_listen) {
    Socket server;
    ASSERT_FALSE(server.is_valid());

    uint16_t test_port = 28081;
    bool ok = server.listen(test_port);
    ASSERT_TRUE(ok);
    ASSERT_TRUE(server.is_valid());

    server.close();
    ASSERT_FALSE(server.is_valid());
}

// 3. Socket Loopback Bidirectional Echo
E2E_TEST_CASE(SocketTest, test_socket_loopback_bidirectional_echo) {
    uint16_t test_port = 28082;
    Socket server;
    ASSERT_TRUE(server.listen(test_port));

    const std::string client_msg = "HYPERSTREAM_SOCKET_VERIFICATION_PAYLOAD";
    const std::string server_reply = "SERVER_ACK_PAYLOAD_CONFIRMED";
    std::string received_by_server;
    std::string received_by_client;

    // Run client connection in a separate thread
    std::thread client_thread([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Socket client;
        if (!client.connect("127.0.0.1", test_port)) {
            return;
        }

        // Send payload to server
        client.send(client_msg.data(), client_msg.size());

        // Receive response from server
        char buf[256] = {0};
        int n = client.recv(buf, sizeof(buf) - 1);
        if (n > 0) {
            received_by_client.assign(buf, static_cast<size_t>(n));
        }
        client.close();
    });

    // Accept on server
    Socket client_conn;
    bool accepted = server.accept(client_conn);
    ASSERT_TRUE(accepted);
    ASSERT_TRUE(client_conn.is_valid());

    // Receive client payload
    char s_buf[256] = {0};
    int s_n = client_conn.recv(s_buf, sizeof(s_buf) - 1);
    ASSERT_TRUE(s_n > 0);
    received_by_server.assign(s_buf, static_cast<size_t>(s_n));

    // Send reply to client
    int sent = client_conn.send(server_reply.data(), server_reply.size());
    ASSERT_EQ(sent, static_cast<int>(server_reply.size()));

    client_thread.join();
    client_conn.close();
    server.close();

    ASSERT_STREQ(received_by_server, client_msg);
    ASSERT_STREQ(received_by_client, server_reply);
}

// 4. Move Semantics & Handle Transfer
E2E_TEST_CASE(SocketTest, test_socket_move_semantics) {
    uint16_t test_port = 28083;
    Socket server1;
    ASSERT_TRUE(server1.listen(test_port));
    ASSERT_TRUE(server1.is_valid());

    // Move construct
    Socket server2(std::move(server1));
    ASSERT_FALSE(server1.is_valid());
    ASSERT_TRUE(server2.is_valid());

    // Move assign
    Socket server3;
    server3 = std::move(server2);
    ASSERT_FALSE(server2.is_valid());
    ASSERT_TRUE(server3.is_valid());

    server3.close();
    ASSERT_FALSE(server3.is_valid());
}

// 5. Graceful Close & Idempotency
E2E_TEST_CASE(SocketTest, test_socket_close_idempotent) {
    Socket s;
    ASSERT_FALSE(s.is_valid());
    // Closing uninitialized socket is safe no-op
    s.close();
    ASSERT_FALSE(s.is_valid());

    uint16_t test_port = 28084;
    ASSERT_TRUE(s.listen(test_port));
    ASSERT_TRUE(s.is_valid());

    s.close();
    ASSERT_FALSE(s.is_valid());
    // Second close is safe no-op
    s.close();
    ASSERT_FALSE(s.is_valid());
}

// 6. Connect Failure Handling
E2E_TEST_CASE(SocketTest, test_socket_connect_failure) {
    Socket s;
    // Connect to port where nothing is listening
    bool ok = s.connect("127.0.0.1", 29999);
    ASSERT_FALSE(ok);
    ASSERT_FALSE(s.is_valid());
}

int main(int argc, char** argv) {
    int res = TestRegistry::instance().run_all(argc, argv);
    Socket::cleanup_networking();
    return res;
}
