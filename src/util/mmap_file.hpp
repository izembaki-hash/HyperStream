#ifndef HYPERSTREAM_MMAP_FILE_HPP
#define HYPERSTREAM_MMAP_FILE_HPP

#include <cstdint>
#include <cstddef>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace hyperstream {

class MmapFile {
public:
    MmapFile() = default;
    ~MmapFile() { close(); }

    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    MmapFile(MmapFile&& other) noexcept 
        : m_data(other.m_data), m_size(other.m_size)
#if defined(_WIN32) || defined(_WIN64)
        , m_file_handle(other.m_file_handle), m_map_handle(other.m_map_handle)
#else
        , m_fd(other.m_fd)
#endif
    {
        other.m_data = nullptr;
        other.m_size = 0;
#if defined(_WIN32) || defined(_WIN64)
        other.m_file_handle = INVALID_HANDLE_VALUE;
        other.m_map_handle = nullptr;
#else
        other.m_fd = -1;
#endif
    }

    MmapFile& operator=(MmapFile&& other) noexcept {
        if (this != &other) {
            close();
            m_data = other.m_data;
            m_size = other.m_size;
#if defined(_WIN32) || defined(_WIN64)
            m_file_handle = other.m_file_handle;
            m_map_handle = other.m_map_handle;
            other.m_file_handle = INVALID_HANDLE_VALUE;
            other.m_map_handle = nullptr;
#else
            m_fd = other.m_fd;
            other.m_fd = -1;
#endif
            other.m_data = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    bool open(const std::string& path) {
        close();

#if defined(_WIN32) || defined(_WIN64)
        m_file_handle = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (m_file_handle == INVALID_HANDLE_VALUE) {
            return false;
        }

        LARGE_INTEGER file_size;
        if (!GetFileSizeEx(m_file_handle, &file_size)) {
            close();
            return false;
        }
        m_size = static_cast<size_t>(file_size.QuadPart);
        if (m_size == 0) {
            close();
            return false;
        }

        m_map_handle = CreateFileMappingA(m_file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m_map_handle) {
            close();
            return false;
        }

        m_data = static_cast<const uint8_t*>(MapViewOfFile(m_map_handle, FILE_MAP_READ, 0, 0, 0));
        if (!m_data) {
            close();
            return false;
        }
#else
        m_fd = ::open(path.c_str(), O_RDONLY);
        if (m_fd < 0) {
            return false;
        }

        struct stat st;
        if (fstat(m_fd, &st) < 0) {
            close();
            return false;
        }
        m_size = static_cast<size_t>(st.st_size);
        if (m_size == 0) {
            close();
            return false;
        }

        void* mapped = ::mmap(nullptr, m_size, PROT_READ, MAP_SHARED, m_fd, 0);
        if (mapped == MAP_FAILED) {
            m_data = nullptr;
            close();
            return false;
        }
        m_data = static_cast<const uint8_t*>(mapped);
#endif
        return true;
    }

    void close() {
#if defined(_WIN32) || defined(_WIN64)
        if (m_data) {
            UnmapViewOfFile(m_data);
            m_data = nullptr;
        }
        if (m_map_handle) {
            CloseHandle(m_map_handle);
            m_map_handle = nullptr;
        }
        if (m_file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(m_file_handle);
            m_file_handle = INVALID_HANDLE_VALUE;
        }
#else
        if (m_data) {
            ::munmap(const_cast<uint8_t*>(m_data), m_size);
            m_data = nullptr;
        }
        if (m_fd >= 0) {
            ::close(m_fd);
            m_fd = -1;
        }
#endif
        m_size = 0;
    }

    bool is_open() const { return m_data != nullptr; }
    const uint8_t* data() const { return m_data; }
    size_t size() const { return m_size; }

private:
    const uint8_t* m_data = nullptr;
    size_t m_size = 0;

#if defined(_WIN32) || defined(_WIN64)
    HANDLE m_file_handle = INVALID_HANDLE_VALUE;
    HANDLE m_map_handle = nullptr;
#else
    int m_fd = -1;
#endif
};

} // namespace hyperstream

#endif // HYPERSTREAM_MMAP_FILE_HPP
