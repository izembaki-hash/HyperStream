#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace hyperstream::test {

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

class SyntheticGGUFBuilder {
public:
    SyntheticGGUFBuilder(uint32_t alignment = 32) : m_alignment(alignment) {}

    void add_metadata_u32(const std::string& key, uint32_t val) {
        KVEntry e;
        e.key = key;
        e.type = GGUFValueType::UINT32;
        e.u32_val = val;
        m_kvs.push_back(std::move(e));
    }

    void add_metadata_u64(const std::string& key, uint64_t val) {
        KVEntry e;
        e.key = key;
        e.type = GGUFValueType::UINT64;
        e.u64_val = val;
        m_kvs.push_back(std::move(e));
    }

    void add_metadata_f32(const std::string& key, float val) {
        KVEntry e;
        e.key = key;
        e.type = GGUFValueType::FLOAT32;
        e.f32_val = val;
        m_kvs.push_back(std::move(e));
    }

    void add_metadata_string(const std::string& key, const std::string& val) {
        KVEntry e;
        e.key = key;
        e.type = GGUFValueType::STRING;
        e.str_val = val;
        m_kvs.push_back(std::move(e));
    }

    void add_metadata_array_string(const std::string& key, const std::vector<std::string>& arr) {
        KVEntry e;
        e.key = key;
        e.type = GGUFValueType::ARRAY;
        e.arr_item_type = GGUFValueType::STRING;
        e.str_arr = arr;
        m_kvs.push_back(std::move(e));
    }

    void add_metadata_array_i32(const std::string& key, const std::vector<int32_t>& arr) {
        KVEntry e;
        e.key = key;
        e.type = GGUFValueType::ARRAY;
        e.arr_item_type = GGUFValueType::INT32;
        e.i32_arr = arr;
        m_kvs.push_back(std::move(e));
    }

    void add_tensor(const std::string& name, const std::vector<uint64_t>& dims, uint32_t type, uint64_t offset, const std::vector<uint8_t>& payload = {}) {
        m_tensors.push_back({name, dims, type, offset, payload});
    }

    std::vector<uint8_t> build(bool corrupt_magic = false, uint32_t custom_version = 3) const {
        std::vector<uint8_t> out;

        // 1. Header
        uint32_t magic = corrupt_magic ? 0xDEADBEEF : 0x46554747; // "GGUF" in Little Endian
        append_val(out, magic);
        append_val(out, custom_version);
        uint64_t tensor_count = static_cast<uint64_t>(m_tensors.size());
        uint64_t kv_count = static_cast<uint64_t>(m_kvs.size());
        append_val(out, tensor_count);
        append_val(out, kv_count);

        // 2. Metadata Key-Value pairs
        for (const auto& kv : m_kvs) {
            append_string(out, kv.key);
            append_val(out, static_cast<uint32_t>(kv.type));

            switch (kv.type) {
                case GGUFValueType::UINT32:
                    append_val(out, kv.u32_val);
                    break;
                case GGUFValueType::UINT64:
                    append_val(out, kv.u64_val);
                    break;
                case GGUFValueType::FLOAT32:
                    append_val(out, kv.f32_val);
                    break;
                case GGUFValueType::STRING:
                    append_string(out, kv.str_val);
                    break;
                case GGUFValueType::ARRAY:
                    append_val(out, static_cast<uint32_t>(kv.arr_item_type));
                    if (kv.arr_item_type == GGUFValueType::STRING) {
                        append_val(out, static_cast<uint64_t>(kv.str_arr.size()));
                        for (const auto& s : kv.str_arr) {
                            append_string(out, s);
                        }
                    } else {
                        append_val(out, static_cast<uint64_t>(kv.i32_arr.size()));
                        for (int32_t v : kv.i32_arr) {
                            append_val(out, v);
                        }
                    }
                    break;
                default:
                    break;
            }
        }

        // 3. Tensor Info Table
        for (const auto& t : m_tensors) {
            append_string(out, t.name);
            uint32_t n_dims = static_cast<uint32_t>(t.dims.size());
            append_val(out, n_dims);
            for (uint64_t d : t.dims) {
                append_val(out, d);
            }
            append_val(out, t.type);
            append_val(out, t.offset);
        }

        // 4. Alignment Padding
        size_t header_end = out.size();
        size_t pad = (m_alignment - (header_end % m_alignment)) % m_alignment;
        for (size_t i = 0; i < pad; ++i) {
            out.push_back(0);
        }

        // 5. Raw Tensor Payloads
        for (const auto& t : m_tensors) {
            if (!t.payload.empty()) {
                out.insert(out.end(), t.payload.begin(), t.payload.end());
            }
        }

        return out;
    }

    bool write_to_file(const std::string& path, bool corrupt_magic = false) const {
        auto buf = build(corrupt_magic);
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) return false;
        ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        return ofs.good();
    }

private:
    uint32_t m_alignment;

    struct KVEntry {
        std::string key;
        GGUFValueType type;
        GGUFValueType arr_item_type;
        std::vector<std::string> str_arr;
        std::vector<int32_t> i32_arr;
        uint32_t u32_val;
        uint64_t u64_val;
        float f32_val;
        std::string str_val;
    };

    struct TensorEntry {
        std::string name;
        std::vector<uint64_t> dims;
        uint32_t type;
        uint64_t offset;
        std::vector<uint8_t> payload;
    };

    std::vector<KVEntry> m_kvs;
    std::vector<TensorEntry> m_tensors;

    template <typename T>
    static void append_val(std::vector<uint8_t>& buf, const T& val) {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
        buf.insert(buf.end(), p, p + sizeof(T));
    }

    static void append_string(std::vector<uint8_t>& buf, const std::string& s) {
        uint64_t len = static_cast<uint64_t>(s.size());
        append_val(buf, len);
        buf.insert(buf.end(), s.begin(), s.end());
    }
};

} // namespace hyperstream::test
