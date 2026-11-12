#include "src/net/socket.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

using namespace hyperstream;

int main() {
    Socket::init_networking();
    uint16_t port = 29991;

    Socket server;
    if (!server.listen(port)) {
        std::cerr << "listen failed\n";
        return 1;
    }

    std::thread client_thread([&]() {
        Socket client;
        if (client.connect("127.0.0.1", port)) {
            char b = 0;
            client.recv(&b, 1);
            // Abrupt hard close with RST via SO_LINGER
            struct linger sl = {1, 0};
            setsockopt(client.native_handle(), SOL_SOCKET, SO_LINGER, reinterpret_cast<const char*>(&sl), sizeof(sl));
            client.close();
        }
    });

    Socket accepted;
    if (!server.accept(accepted)) {
        std::cerr << "accept failed\n";
        return 1;
    }

    // Unblock client
    accepted.send("X", 1);
    client_thread.join();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::vector<char> dummy(1024, 'A');
    int r1 = accepted.send(dummy.data(), dummy.size());
    std::cout << "With RST: Send 1 returned: " << r1 << std::endl;

    accepted.close();
    server.close();
    Socket::cleanup_networking();
    return 0;
}
