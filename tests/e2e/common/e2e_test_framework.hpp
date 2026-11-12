#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <sstream>
#include <functional>
#include <stdexcept>
#include <iomanip>
#include <cstdlib>

namespace hyperstream::test {

struct TestCase {
    std::string name;
    std::string suite;
    std::function<void()> func;
};

class TestRegistry {
public:
    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void register_test(const std::string& suite, const std::string& name, std::function<void()> func) {
        m_tests.push_back({name, suite, func});
    }

    int run_all(int argc = 0, char** argv = nullptr) {
        std::string filter = "";
        bool verbose = false;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--filter=", 0) == 0) {
                filter = arg.substr(9);
            } else if (arg == "-v" || arg == "--verbose") {
                verbose = true;
            }
        }

        int passed = 0;
        int failed = 0;
        int skipped = 0;

        std::cout << "[TEST SUITE] Running " << m_tests.size() << " test cases...\n";
        auto suite_start = std::chrono::high_resolution_clock::now();

        for (const auto& tc : m_tests) {
            std::string full_name = tc.suite + "::" + tc.name;
            if (!filter.empty() && full_name.find(filter) == std::string::npos) {
                skipped++;
                continue;
            }

            std::cout << "[TEST] " << std::left << std::setw(60) << full_name << " ... ";
            std::cout.flush();

            auto t0 = std::chrono::high_resolution_clock::now();
            try {
                tc.func();
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "[PASS] (" << std::fixed << std::setprecision(2) << ms << " ms)\n";
                passed++;
            } catch (const std::exception& e) {
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "[FAIL] (" << std::fixed << std::setprecision(2) << ms << " ms)\n";
                std::cout << "       Reason: " << e.what() << "\n";
                failed++;
            } catch (...) {
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                std::cout << "[FAIL] (" << std::fixed << std::setprecision(2) << ms << " ms)\n";
                std::cout << "       Reason: Unknown exception caught.\n";
                failed++;
            }
        }

        auto suite_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(suite_end - suite_start).count();

        std::cout << "=========================================================================\n";
        std::cout << "[SUMMARY] Total: " << m_tests.size()
                  << " | Passed: " << passed
                  << " | Failed: " << failed
                  << " | Skipped: " << skipped
                  << " | Duration: " << std::fixed << std::setprecision(2) << total_ms << " ms\n";
        std::cout << "=========================================================================\n";

        return (failed == 0) ? 0 : 1;
    }

private:
    std::vector<TestCase> m_tests;
};

struct TestRegistrar {
    TestRegistrar(const std::string& suite, const std::string& name, std::function<void()> func) {
        TestRegistry::instance().register_test(suite, name, func);
    }
};

#define E2E_TEST_CASE(suite, name) \
    static void test_##suite##_##name(); \
    static ::hyperstream::test::TestRegistrar reg_##suite##_##name(#suite, #name, test_##suite##_##name); \
    static void test_##suite##_##name()

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " #cond " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

#define ASSERT_FALSE(cond) \
    do { \
        if (cond) { \
            std::ostringstream oss; \
            oss << "Assertion failed (expected false): " #cond " at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

#define ASSERT_EQ(val1, val2) \
    do { \
        if (!((val1) == (val2))) { \
            std::ostringstream oss; \
            oss << "Assertion failed: (" #val1 " == " #val2 ") [" << (val1) << " != " << (val2) << "] at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

#define ASSERT_NE(val1, val2) \
    do { \
        if ((val1) == (val2)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: (" #val1 " != " #val2 ") [" << (val1) << " == " << (val2) << "] at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

#define ASSERT_NEAR(val1, val2, eps) \
    do { \
        double diff = std::fabs(static_cast<double>(val1) - static_cast<double>(val2)); \
        if (diff > (eps)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: |" #val1 " - " #val2 "| <= " #eps " [" << (val1) << " vs " << (val2) << ", diff=" << diff << "] at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

#define ASSERT_STREQ(s1, s2) \
    do { \
        std::string str1 = (s1); \
        std::string str2 = (s2); \
        if (str1 != str2) { \
            std::ostringstream oss; \
            oss << "Assertion failed: (" #s1 " == " #s2 ") [\"" << str1 << "\" != \"" << str2 << "\"] at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

} // namespace hyperstream::test
