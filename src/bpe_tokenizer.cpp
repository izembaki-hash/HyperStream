#include "bpe_tokenizer.hpp"
#include <fstream>
#include <algorithm>
#include <cctype>

namespace hyperstream {

BPETokenizer::BPETokenizer() {
    init_byte_encoder();
}

void BPETokenizer::init_byte_encoder() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; b++) bs.push_back(b);
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    m_byte_to_unicode.clear();
    m_unicode_to_byte.clear();

    for (size_t i = 0; i < bs.size(); i++) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        int codepoint = cs[i];
        std::string utf8_char;
        if (codepoint < 0x80) {
            utf8_char += static_cast<char>(codepoint);
        } else if (codepoint < 0x800) {
            utf8_char += static_cast<char>(0xC0 | (codepoint >> 6));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else {
            utf8_char += static_cast<char>(0xE0 | (codepoint >> 12));
            utf8_char += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        m_byte_to_unicode[byte_val] = utf8_char;
        m_unicode_to_byte[utf8_char] = byte_val;
    }
}

bool BPETokenizer::load_from_gguf(const GGUFParser& parser) {
    const auto& tokens = parser.get_string_array("tokenizer.ggml.tokens");
    if (tokens.empty()) {
        return false;
    }

    m_id_to_token = tokens;
    m_vocab.clear();
    m_vocab.reserve(tokens.size());
    m_special_tokens.clear();

    for (size_t i = 0; i < tokens.size(); i++) {
        m_vocab[tokens[i]] = static_cast<int32_t>(i);
        // Identify special tokens like <|...|>, <s>, </s>, <unk>, <pad>
        const std::string& tok = tokens[i];
        if ((tok.size() >= 4 && tok.substr(0, 2) == "<|" && tok.substr(tok.size() - 2) == "|>") ||
            tok == "<s>" || tok == "</s>" || tok == "<unk>" || tok == "<pad>") {
            m_special_tokens.insert(tok);
        }
    }

    const auto& merges = parser.get_string_array("tokenizer.ggml.merges");
    m_merges.clear();
    m_merges.reserve(merges.size());

    for (size_t rank = 0; rank < merges.size(); rank++) {
        const std::string& rule = merges[rank];
        size_t sp = rule.find(' ');
        if (sp != std::string::npos) {
            std::string p1 = rule.substr(0, sp);
            std::string p2 = rule.substr(sp + 1);
            m_merges[{p1, p2}] = static_cast<int32_t>(rank);
        }
    }

    // Special token resolution
    uint32_t eos_u32 = 0;
    if (parser.get_metadata_u32("tokenizer.ggml.eos_token_id", eos_u32)) {
        m_eos_id = static_cast<int32_t>(eos_u32);
    } else {
        auto it = m_vocab.find("<|im_end|>");
        if (it != m_vocab.end()) m_eos_id = it->second;
        else {
            auto it2 = m_vocab.find("<|endoftext|>");
            if (it2 != m_vocab.end()) m_eos_id = it2->second;
            else {
                auto it3 = m_vocab.find("</s>");
                if (it3 != m_vocab.end()) m_eos_id = it3->second;
            }
        }
    }

    uint32_t bos_u32 = 0;
    if (parser.get_metadata_u32("tokenizer.ggml.bos_token_id", bos_u32)) {
        m_bos_id = static_cast<int32_t>(bos_u32);
    } else {
        auto it = m_vocab.find("<|im_start|>");
        if (it != m_vocab.end()) m_bos_id = it->second;
        else {
            auto it2 = m_vocab.find("<s>");
            if (it2 != m_vocab.end()) m_bos_id = it2->second;
        }
    }

    uint32_t pad_u32 = 0;
    if (parser.get_metadata_u32("tokenizer.ggml.padding_token_id", pad_u32)) {
        m_pad_id = static_cast<int32_t>(pad_u32);
    } else {
        m_pad_id = m_eos_id;
    }

    auto it_start = m_vocab.find("<|im_start|>");
    m_im_start_id = (it_start != m_vocab.end()) ? it_start->second : 151644;
    if (it_start != m_vocab.end()) m_special_tokens.insert("<|im_start|>");

    auto it_end = m_vocab.find("<|im_end|>");
    m_im_end_id = (it_end != m_vocab.end()) ? it_end->second : 151645;
    if (it_end != m_vocab.end()) m_special_tokens.insert("<|im_end|>");

    auto it_eot = m_vocab.find("<|endoftext|>");
    if (it_eot != m_vocab.end()) m_special_tokens.insert("<|endoftext|>");

    return true;
}

bool BPETokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    std::ifstream fv(vocab_path, std::ios::binary);
    if (!fv.is_open()) return false;
    uint32_t num_vocab = 0;
    fv.read(reinterpret_cast<char*>(&num_vocab), 4);
    m_vocab.clear();
    m_vocab.reserve(num_vocab);
    m_id_to_token.resize(num_vocab);
    m_special_tokens.clear();

    for (uint32_t i = 0; i < num_vocab; i++) {
        uint32_t token_id = 0;
        uint16_t len = 0;
        fv.read(reinterpret_cast<char*>(&token_id), 4);
        fv.read(reinterpret_cast<char*>(&len), 2);
        std::string s(len, '\0');
        fv.read(&s[0], len);
        m_vocab[s] = static_cast<int32_t>(token_id);
        if (token_id < m_id_to_token.size()) {
            m_id_to_token[token_id] = s;
        }
        if ((s.size() >= 4 && s.substr(0, 2) == "<|" && s.substr(s.size() - 2) == "|>") ||
            s == "<s>" || s == "</s>") {
            m_special_tokens.insert(s);
        }
    }
    fv.close();

    std::ifstream fm(merges_path, std::ios::binary);
    if (!fm.is_open()) return false;
    uint32_t num_merges = 0;
    fm.read(reinterpret_cast<char*>(&num_merges), 4);
    m_merges.clear();
    m_merges.reserve(num_merges);

    for (uint32_t i = 0; i < num_merges; i++) {
        uint8_t len1 = 0, len2 = 0;
        fm.read(reinterpret_cast<char*>(&len1), 1);
        std::string p1(len1, '\0');
        fm.read(&p1[0], len1);

        fm.read(reinterpret_cast<char*>(&len2), 1);
        std::string p2(len2, '\0');
        fm.read(&p2[0], len2);

        uint32_t rank = 0;
        fm.read(reinterpret_cast<char*>(&rank), 4);
        m_merges[{p1, p2}] = static_cast<int32_t>(rank);
    }
    fm.close();

    auto it_s = m_vocab.find("<|im_start|>");
    m_im_start_id = (it_s != m_vocab.end()) ? it_s->second : 151644;
    auto it_e = m_vocab.find("<|im_end|>");
    m_im_end_id = (it_e != m_vocab.end()) ? it_e->second : 151645;
    m_eos_id = m_im_end_id;
    m_bos_id = m_im_start_id;

    return true;
}

std::vector<std::string> BPETokenizer::gpt2_split(const std::string& text) const {
    std::vector<std::string> chunks;
    size_t i = 0;
    size_t len = text.length();

    while (i < len) {
        size_t start = i;

        // Check leading space
        if (text[i] == ' ') {
            i++;
        }

        if (i < len && (std::isalnum(static_cast<unsigned char>(text[i])) || static_cast<unsigned char>(text[i]) >= 128)) {
            while (i < len && (std::isalnum(static_cast<unsigned char>(text[i])) || static_cast<unsigned char>(text[i]) >= 128)) {
                i++;
            }
        } else if (i < len && std::isspace(static_cast<unsigned char>(text[i]))) {
            while (i < len && std::isspace(static_cast<unsigned char>(text[i]))) {
                i++;
            }
        } else if (i < len) {
            i++;
        }

        if (i > start) {
            chunks.push_back(text.substr(start, i - start));
        } else {
            i++;
        }
    }
    return chunks;
}

std::vector<int32_t> BPETokenizer::bpe_encode_chunk(const std::string& chunk) const {
    std::vector<int32_t> token_ids;
    if (chunk.empty()) return token_ids;

    // 1. Direct vocabulary match
    auto it_direct = m_vocab.find(chunk);
    if (it_direct != m_vocab.end()) {
        token_ids.push_back(it_direct->second);
        return token_ids;
    }

    // 2. Standard BPE with merges if merge table is available
    if (!m_merges.empty()) {
        std::string bpe_chunk;
        for (unsigned char c : chunk) {
            auto it = m_byte_to_unicode.find(c);
            if (it != m_byte_to_unicode.end()) {
                bpe_chunk += it->second;
            } else {
                bpe_chunk += static_cast<char>(c);
            }
        }

        // Split bpe_chunk into individual UTF-8 characters
        std::vector<std::string> word;
        for (size_t j = 0; j < bpe_chunk.length(); ) {
            unsigned char lead = static_cast<unsigned char>(bpe_chunk[j]);
            size_t char_len = 1;
            if ((lead & 0xE0) == 0xC0) char_len = 2;
            else if ((lead & 0xF0) == 0xE0) char_len = 3;
            else if ((lead & 0xF8) == 0xF0) char_len = 4;
            if (j + char_len > bpe_chunk.length()) char_len = bpe_chunk.length() - j;
            word.push_back(bpe_chunk.substr(j, char_len));
            j += char_len;
        }

        // Iteratively apply lowest-rank merges
        while (word.size() > 1) {
            int32_t min_rank = 1000000000;
            int best_idx = -1;
            std::pair<std::string, std::string> best_pair;

            for (size_t k = 0; k < word.size() - 1; k++) {
                auto it = m_merges.find({word[k], word[k+1]});
                if (it != m_merges.end() && it->second < min_rank) {
                    min_rank = it->second;
                    best_idx = static_cast<int>(k);
                    best_pair = it->first;
                }
            }

            if (best_idx == -1) break;

            std::vector<std::string> new_word;
            new_word.reserve(word.size());
            for (size_t k = 0; k < word.size(); ) {
                if (k < word.size() - 1 && word[k] == best_pair.first && word[k+1] == best_pair.second) {
                    new_word.push_back(word[k] + word[k+1]);
                    k += 2;
                } else {
                    new_word.push_back(word[k]);
                    k++;
                }
            }
            word = std::move(new_word);
        }

        // Map merged subwords to vocabulary IDs
        bool all_found = true;
        std::vector<int32_t> bpe_ids;
        for (const auto& subword : word) {
            auto it = m_vocab.find(subword);
            if (it != m_vocab.end()) {
                bpe_ids.push_back(it->second);
            } else {
                all_found = false;
                break;
            }
        }

        if (all_found && !bpe_ids.empty()) {
            return bpe_ids;
        }
    }

    // 3. Fallback: Greedy longest-prefix matching against vocabulary
    size_t pos = 0;
    while (pos < chunk.size()) {
        bool found = false;
        for (size_t l = chunk.size() - pos; l > 0; --l) {
            std::string sub = chunk.substr(pos, l);
            auto it = m_vocab.find(sub);
            if (it != m_vocab.end()) {
                token_ids.push_back(it->second);
                pos += l;
                found = true;
                break;
            }
        }
        if (!found) {
            pos++;
        }
    }

    return token_ids;
}

std::vector<int32_t> BPETokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int32_t> token_ids;
    if (add_bos && m_bos_id >= 0) {
        token_ids.push_back(m_bos_id);
    }

    if (text.empty()) {
        return token_ids;
    }

    // Scan text and preserve special tokens as atomic units
    size_t i = 0;
    size_t text_len = text.size();
    size_t segment_start = 0;

    auto flush_segment = [&](size_t seg_end) {
        if (seg_end > segment_start) {
            std::string seg = text.substr(segment_start, seg_end - segment_start);
            std::vector<std::string> chunks = gpt2_split(seg);
            for (const auto& chunk : chunks) {
                auto chunk_ids = bpe_encode_chunk(chunk);
                token_ids.insert(token_ids.end(), chunk_ids.begin(), chunk_ids.end());
            }
        }
        segment_start = seg_end;
    };

    while (i < text_len) {
        bool matched_special = false;
        // Check for special token match at position i
        for (const auto& sp : m_special_tokens) {
            if (i + sp.size() <= text_len && text.compare(i, sp.size(), sp) == 0) {
                flush_segment(i);
                auto it = m_vocab.find(sp);
                if (it != m_vocab.end()) {
                    token_ids.push_back(it->second);
                }
                i += sp.size();
                segment_start = i;
                matched_special = true;
                break;
            }
        }
        if (!matched_special) {
            i++;
        }
    }
    flush_segment(text_len);

    return token_ids;
}

std::vector<int32_t> BPETokenizer::encode_simple(const std::string& text) const {
    std::vector<int32_t> ids;
    for (char c : text) {
        std::string s(1, c);
        auto it = m_vocab.find(s);
        if (it != m_vocab.end()) {
            ids.push_back(it->second);
        }
    }
    return ids;
}

std::vector<int> BPETokenizer::tokenize(const std::string& text) const {
    auto ids = encode(text, false);
    return std::vector<int>(ids.begin(), ids.end());
}

std::string BPETokenizer::decode(int32_t token_id) const {
    if (token_id < 0 || token_id >= static_cast<int32_t>(m_id_to_token.size())) {
        return "";
    }

    const std::string& piece = m_id_to_token[token_id];
    // Special tokens are returned verbatim
    if (m_special_tokens.find(piece) != m_special_tokens.end() ||
        (piece.size() >= 4 && piece.substr(0, 2) == "<|" && piece.substr(piece.size() - 2) == "|>")) {
        return piece;
    }

    // Decode byte-level unicode characters back to raw bytes
    std::string out;
    out.reserve(piece.size());

    for (size_t i = 0; i < piece.size(); ) {
        unsigned char lead = static_cast<unsigned char>(piece[i]);
        size_t char_len = 1;
        if ((lead & 0xE0) == 0xC0) char_len = 2;
        else if ((lead & 0xF0) == 0xE0) char_len = 3;
        else if ((lead & 0xF8) == 0xF0) char_len = 4;
        if (i + char_len > piece.size()) char_len = piece.size() - i;

        std::string u_char = piece.substr(i, char_len);
        auto it = m_unicode_to_byte.find(u_char);
        if (it != m_unicode_to_byte.end()) {
            out += static_cast<char>(it->second);
        } else {
            out += u_char;
        }
        i += char_len;
    }

    return out;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string out;
    for (int32_t id : tokens) {
        out += decode(id);
    }
    return out;
}

bool BPETokenizer::is_special_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= static_cast<int32_t>(m_id_to_token.size())) {
        return false;
    }
    return m_special_tokens.find(m_id_to_token[token_id]) != m_special_tokens.end();
}

int32_t BPETokenizer::eos_token() const { return m_eos_id; }
int32_t BPETokenizer::bos_token() const { return m_bos_id; }
int32_t BPETokenizer::im_start_token() const { return m_im_start_id; }
int32_t BPETokenizer::im_end_token() const { return m_im_end_id; }
size_t BPETokenizer::vocab_size() const { return m_id_to_token.size(); }

} // namespace hyperstream
