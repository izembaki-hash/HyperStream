#ifndef HYPERSTREAM_GGUF_PARSER_HPP
#define HYPERSTREAM_GGUF_PARSER_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "util/mmap_file.hpp"

namespace hyperstream {

enum class GGMLType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
};

enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims = 0;
    uint64_t ne[4] = {1, 1, 1, 1};
    uint32_t type = 0; // GGML type (Q4_0, Q8_0, F32, etc.)
    uint64_t offset = 0; // Relative to tensor data start offset
};

class GGUFParser {
public:
    struct MetadataValue {
        GGUFValueType type = GGUFValueType::UINT32;
        uint64_t u64_val = 0;
        int64_t  i64_val = 0;
        double   f64_val = 0.0;
        std::string str_val;
        std::vector<std::string> str_arr;
        std::vector<int32_t>     i32_arr;
        std::vector<uint32_t>    u32_arr;
        std::vector<float>       f32_arr;
    };

    GGUFParser() = default;
    ~GGUFParser() { close(); }

    GGUFParser(const GGUFParser&) = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;

    GGUFParser(GGUFParser&& other) noexcept = default;
    GGUFParser& operator=(GGUFParser&& other) noexcept = default;

    bool open(const std::string& path) {
        close();
        if (!m_mmap.open(path)) {
            return false;
        }
        return parse_from_memory(m_mmap.data(), m_mmap.size());
    }

    bool parse_from_memory(const uint8_t* buffer, size_t buffer_size) {
        m_data = buffer;
        m_size = buffer_size;
        m_metadata.clear();
        m_tensors.clear();
        m_tensors_list.clear();
        m_data_start_offset = 0;
        m_alignment = 32;

        if (!m_data || m_size < 24) {
            return false;
        }

        size_t cursor = 0;
        auto read_bytes = [&](void* dst, size_t n) -> bool {
            if (cursor + n > m_size) return false;
            std::memcpy(dst, m_data + cursor, n);
            cursor += n;
            return true;
        };

        // 1. Header
        uint32_t magic = 0;
        if (!read_bytes(&magic, sizeof(magic))) return false;
        if (magic != 0x46554747) { // "GGUF" in Little Endian
            return false;
        }

        if (!read_bytes(&m_version, sizeof(m_version))) return false;
        if (m_version != 2 && m_version != 3) {
            return false;
        }

        uint64_t tensor_count = 0;
        uint64_t kv_count = 0;
        if (!read_bytes(&tensor_count, sizeof(tensor_count))) return false;
        if (!read_bytes(&kv_count, sizeof(kv_count))) return false;

        // 2. Metadata Key-Value pairs
        auto read_string = [&](std::string& out) -> bool {
            uint64_t slen = 0;
            if (!read_bytes(&slen, sizeof(slen))) return false;
            if (cursor + slen > m_size) return false;
            out.assign(reinterpret_cast<const char*>(m_data + cursor), static_cast<size_t>(slen));
            cursor += static_cast<size_t>(slen);
            return true;
        };

        for (uint64_t i = 0; i < kv_count; ++i) {
            std::string key;
            if (!read_string(key)) return false;

            uint32_t val_type_raw = 0;
            if (!read_bytes(&val_type_raw, sizeof(val_type_raw))) return false;
            if (val_type_raw > 12) return false;

            GGUFValueType val_type = static_cast<GGUFValueType>(val_type_raw);
            MetadataValue val;
            val.type = val_type;

            switch (val_type) {
                case GGUFValueType::UINT8: {
                    uint8_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.u64_val = v;
                    val.i64_val = v;
                    break;
                }
                case GGUFValueType::INT8: {
                    int8_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.i64_val = v;
                    val.u64_val = static_cast<uint64_t>(v);
                    break;
                }
                case GGUFValueType::UINT16: {
                    uint16_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.u64_val = v;
                    val.i64_val = v;
                    break;
                }
                case GGUFValueType::INT16: {
                    int16_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.i64_val = v;
                    val.u64_val = static_cast<uint64_t>(v);
                    break;
                }
                case GGUFValueType::UINT32: {
                    uint32_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.u64_val = v;
                    val.i64_val = v;
                    break;
                }
                case GGUFValueType::INT32: {
                    int32_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.i64_val = v;
                    val.u64_val = static_cast<uint64_t>(v);
                    break;
                }
                case GGUFValueType::FLOAT32: {
                    float v = 0.0f;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.f64_val = v;
                    break;
                }
                case GGUFValueType::BOOL: {
                    uint8_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.u64_val = (v != 0);
                    val.i64_val = (v != 0);
                    break;
                }
                case GGUFValueType::STRING: {
                    if (!read_string(val.str_val)) return false;
                    break;
                }
                case GGUFValueType::ARRAY: {
                    uint32_t item_type_raw = 0;
                    uint64_t item_count = 0;
                    if (!read_bytes(&item_type_raw, sizeof(item_type_raw))) return false;
                    if (!read_bytes(&item_count, sizeof(item_count))) return false;

                    GGUFValueType item_type = static_cast<GGUFValueType>(item_type_raw);
                    if (item_type == GGUFValueType::STRING) {
                        val.str_arr.reserve(static_cast<size_t>(item_count));
                        for (uint64_t j = 0; j < item_count; ++j) {
                            std::string s;
                            if (!read_string(s)) return false;
                            val.str_arr.push_back(std::move(s));
                        }
                    } else if (item_type == GGUFValueType::INT32) {
                        val.i32_arr.resize(static_cast<size_t>(item_count));
                        if (!read_bytes(val.i32_arr.data(), static_cast<size_t>(item_count) * sizeof(int32_t))) return false;
                    } else if (item_type == GGUFValueType::UINT32) {
                        val.u32_arr.resize(static_cast<size_t>(item_count));
                        if (!read_bytes(val.u32_arr.data(), static_cast<size_t>(item_count) * sizeof(uint32_t))) return false;
                    } else if (item_type == GGUFValueType::FLOAT32) {
                        val.f32_arr.resize(static_cast<size_t>(item_count));
                        if (!read_bytes(val.f32_arr.data(), static_cast<size_t>(item_count) * sizeof(float))) return false;
                    } else {
                        // For other array types, skip elements safely
                        size_t elem_size = 1;
                        if (item_type == GGUFValueType::UINT16 || item_type == GGUFValueType::INT16) elem_size = 2;
                        else if (item_type == GGUFValueType::UINT32 || item_type == GGUFValueType::INT32 || item_type == GGUFValueType::FLOAT32) elem_size = 4;
                        else if (item_type == GGUFValueType::UINT64 || item_type == GGUFValueType::INT64 || item_type == GGUFValueType::FLOAT64) elem_size = 8;
                        if (cursor + item_count * elem_size > m_size) return false;
                        cursor += static_cast<size_t>(item_count * elem_size);
                    }
                    break;
                }
                case GGUFValueType::UINT64: {
                    uint64_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.u64_val = v;
                    val.i64_val = static_cast<int64_t>(v);
                    break;
                }
                case GGUFValueType::INT64: {
                    int64_t v = 0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.i64_val = v;
                    val.u64_val = static_cast<uint64_t>(v);
                    break;
                }
                case GGUFValueType::FLOAT64: {
                    double v = 0.0;
                    if (!read_bytes(&v, sizeof(v))) return false;
                    val.f64_val = v;
                    break;
                }
            }

            m_metadata[key] = std::move(val);
        }

        // Check alignment override in metadata
        auto it_align = m_metadata.find("general.alignment");
        if (it_align != m_metadata.end()) {
            m_alignment = static_cast<uint32_t>(it_align->second.u64_val);
            if (m_alignment == 0) m_alignment = 32;
        }

        // 3. Tensor Info Table
        m_tensors_list.reserve(static_cast<size_t>(tensor_count));
        for (uint64_t i = 0; i < tensor_count; ++i) {
            GGUFTensorInfo info;
            if (!read_string(info.name)) return false;

            if (!read_bytes(&info.n_dims, sizeof(info.n_dims))) return false;
            for (uint32_t d = 0; d < 4; ++d) {
                if (d < info.n_dims) {
                    if (!read_bytes(&info.ne[d], sizeof(uint64_t))) return false;
                } else {
                    info.ne[d] = 1;
                }
            }
            // Skip any dimensions beyond 4 if present in file
            if (info.n_dims > 4) {
                for (uint32_t d = 4; d < info.n_dims; ++d) {
                    uint64_t extra_dim = 0;
                    if (!read_bytes(&extra_dim, sizeof(extra_dim))) return false;
                }
            }

            if (!read_bytes(&info.type, sizeof(info.type))) return false;
            if (!read_bytes(&info.offset, sizeof(info.offset))) return false;

            m_tensors[info.name] = info;
            m_tensors_list.push_back(std::move(info));
        }

        // 4. Calculate Data Start Offset with alignment
        size_t header_end_offset = cursor;
        size_t pad = (m_alignment - (header_end_offset % m_alignment)) % m_alignment;
        m_data_start_offset = header_end_offset + pad;

        return true;
    }

    void close() {
        m_mmap.close();
        m_data = nullptr;
        m_size = 0;
        m_version = 0;
        m_metadata.clear();
        m_tensors.clear();
        m_tensors_list.clear();
        m_data_start_offset = 0;
        m_alignment = 32;
    }

    // --- Interface Contract Methods ---
    bool get_metadata_string(const std::string& key, std::string& val) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.type == GGUFValueType::STRING) {
            val = it->second.str_val;
            return true;
        }
        return false;
    }

    bool get_metadata_u32(const std::string& key, uint32_t& val) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end()) {
            val = static_cast<uint32_t>(it->second.u64_val);
            return true;
        }
        return false;
    }

    bool get_metadata_u64(const std::string& key, uint64_t& val) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end()) {
            val = it->second.u64_val;
            return true;
        }
        return false;
    }

    bool get_metadata_f32(const std::string& key, float& val) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end()) {
            val = static_cast<float>(it->second.f64_val);
            return true;
        }
        return false;
    }

    bool get_metadata_bool(const std::string& key, bool& val) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end()) {
            val = (it->second.u64_val != 0);
            return true;
        }
        return false;
    }

    bool get_metadata_array_string(const std::string& key, std::vector<std::string>& val) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.type == GGUFValueType::ARRAY) {
            val = it->second.str_arr;
            return true;
        }
        return false;
    }

    bool get_metadata_array_i32(const std::string& key, std::vector<int32_t>& val) const {
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.type == GGUFValueType::ARRAY) {
            val = it->second.i32_arr;
            return true;
        }
        return false;
    }

    const GGUFTensorInfo* find_tensor(const std::string& name) const {
        auto it = m_tensors.find(name);
        if (it != m_tensors.end()) {
            return &it->second;
        }
        return nullptr;
    }

    const uint8_t* get_tensor_data(const GGUFTensorInfo& info) const {
        if (!m_data) return nullptr;
        uint64_t full_offset = m_data_start_offset + info.offset;
        if (full_offset >= m_size) return nullptr;
        return m_data + full_offset;
    }

    uint64_t get_tensor_data_offset(const GGUFTensorInfo& info) const {
        return m_data_start_offset + info.offset;
    }

    uint64_t get_data_start_offset() const {
        return m_data_start_offset;
    }

    const uint8_t* data() const {
        return m_data;
    }

    size_t size() const {
        return m_size;
    }

    // --- Convenience Accessors ---
    std::string get_string(const std::string& key, const std::string& def = "") const {
        std::string val;
        return get_metadata_string(key, val) ? val : def;
    }

    uint32_t get_u32(const std::string& key, uint32_t def = 0) const {
        uint32_t val = 0;
        return get_metadata_u32(key, val) ? val : def;
    }

    uint64_t get_u64(const std::string& key, uint64_t def = 0) const {
        uint64_t val = 0;
        return get_metadata_u64(key, val) ? val : def;
    }

    float get_f32(const std::string& key, float def = 0.0f) const {
        float val = 0.0f;
        return get_metadata_f32(key, val) ? val : def;
    }

    bool get_bool(const std::string& key, bool def = false) const {
        bool val = false;
        return get_metadata_bool(key, val) ? val : def;
    }

    const std::vector<std::string>& get_string_array(const std::string& key) const {
        static const std::vector<std::string> empty_arr;
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.type == GGUFValueType::ARRAY) {
            return it->second.str_arr;
        }
        return empty_arr;
    }

    const std::vector<int32_t>& get_i32_array(const std::string& key) const {
        static const std::vector<int32_t> empty_arr;
        auto it = m_metadata.find(key);
        if (it != m_metadata.end() && it->second.type == GGUFValueType::ARRAY) {
            return it->second.i32_arr;
        }
        return empty_arr;
    }

    uint32_t get_version() const { return m_version; }
    uint32_t get_alignment() const { return m_alignment; }
    size_t get_tensor_count() const { return m_tensors_list.size(); }
    const std::vector<GGUFTensorInfo>& get_tensors() const { return m_tensors_list; }
    const std::unordered_map<std::string, MetadataValue>& get_metadata() const { return m_metadata; }

private:
    MmapFile m_mmap;
    const uint8_t* m_data = nullptr;
    size_t m_size = 0;
    uint32_t m_version = 0;
    uint32_t m_alignment = 32;
    uint64_t m_data_start_offset = 0;

    std::unordered_map<std::string, MetadataValue> m_metadata;
    std::unordered_map<std::string, GGUFTensorInfo> m_tensors;
    std::vector<GGUFTensorInfo> m_tensors_list;
};

} // namespace hyperstream

#endif // HYPERSTREAM_GGUF_PARSER_HPP
