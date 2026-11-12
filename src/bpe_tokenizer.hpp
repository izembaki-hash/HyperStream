#ifndef HYPERSTREAM_BPE_TOKENIZER_HPP
#define HYPERSTREAM_BPE_TOKENIZER_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "gguf_parser.hpp"

namespace hyperstream {

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string>& p) const {
        size_t h1 = std::hash<std::string>()(p.first);
        size_t h2 = std::hash<std::string>()(p.second);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer() = default;

    // --- Interface Contract (PROJECT.md) ---
    bool load_from_gguf(const GGUFParser& parser);
    std::vector<int32_t> encode(const std::string& text, bool add_bos = false) const;
    std::string decode(int32_t token_id) const;
    std::string decode(const std::vector<int32_t>& tokens) const;
    int32_t eos_token() const;
    int32_t bos_token() const;
    int32_t im_start_token() const;
    int32_t im_end_token() const;
    size_t vocab_size() const;

    // --- Backwards Compatibility & Utility Accessors ---
    std::vector<int> tokenize(const std::string& text) const;
    std::vector<int32_t> encode_simple(const std::string& text) const;
    bool load(const std::string& vocab_path, const std::string& merges_path);

    int32_t eos_token_id() const { return eos_token(); }
    int32_t bos_token_id() const { return bos_token(); }
    int32_t im_start_id() const { return im_start_token(); }
    int32_t im_end_id() const { return im_end_token(); }
    int32_t pad_token_id() const { return m_pad_id; }

    const std::vector<std::string>& id_to_token() const { return m_id_to_token; }
    const std::unordered_map<std::string, int32_t>& vocab() const { return m_vocab; }
    const std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash>& merges() const { return m_merges; }

    bool is_special_token(int32_t token_id) const;

private:
    void init_byte_encoder();
    std::vector<std::string> gpt2_split(const std::string& text) const;
    std::vector<int32_t> bpe_encode_chunk(const std::string& chunk) const;

    std::unordered_map<std::string, int32_t> m_vocab;
    std::vector<std::string> m_id_to_token;
    std::unordered_map<std::pair<std::string, std::string>, int32_t, PairHash> m_merges;

    std::unordered_map<uint8_t, std::string> m_byte_to_unicode;
    std::unordered_map<std::string, uint8_t> m_unicode_to_byte;
    std::unordered_set<std::string> m_special_tokens;

    int32_t m_bos_id = -1;
    int32_t m_eos_id = -1;
    int32_t m_pad_id = -1;
    int32_t m_im_start_id = -1;
    int32_t m_im_end_id = -1;
};

} // namespace hyperstream

#endif // HYPERSTREAM_BPE_TOKENIZER_HPP
