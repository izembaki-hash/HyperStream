#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>
#include <iostream>
#include <algorithm>

struct PairHash {
    size_t operator()(const std::pair<std::string, std::string> &p) const {
        return std::hash<std::string>()(p.first) ^ (std::hash<std::string>()(p.second) << 1);
    }
};

class BPETokenizer {
public:
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merges;
    std::unordered_map<uint8_t, std::string> byte_to_unicode;

    BPETokenizer() {
        init_byte_encoder();
    }

    void init_byte_encoder() {
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
        for (size_t i = 0; i < bs.size(); i++) {
            uint8_t byte_val = (uint8_t)bs[i];
            int codepoint = cs[i];
            std::string utf8_char;
            if (codepoint < 0x80) {
                utf8_char += (char)codepoint;
            } else if (codepoint < 0x800) {
                utf8_char += (char)(0xC0 | (codepoint >> 6));
                utf8_char += (char)(0x80 | (codepoint & 0x3F));
            } else {
                utf8_char += (char)(0xE0 | (codepoint >> 12));
                utf8_char += (char)(0x80 | ((codepoint >> 6) & 0x3F));
                utf8_char += (char)(0x80 | (codepoint & 0x3F));
            }
            byte_to_unicode[byte_val] = utf8_char;
        }
    }

    bool load(const std::string &vocab_path, const std::string &merges_path) {
        std::ifstream fv(vocab_path, std::ios::binary);
        if (!fv.is_open()) return false;
        uint32_t num_vocab = 0;
        fv.read((char*)&num_vocab, 4);
        for (uint32_t i = 0; i < num_vocab; i++) {
            uint32_t token_id = 0;
            uint16_t len = 0;
            fv.read((char*)&token_id, 4);
            fv.read((char*)&len, 2);
            std::string s(len, '\0');
            fv.read(&s[0], len);
            vocab[s] = (int)token_id;
        }
        fv.close();

        std::ifstream fm(merges_path, std::ios::binary);
        if (!fm.is_open()) return false;
        uint32_t num_merges = 0;
        fm.read((char*)&num_merges, 4);
        for (uint32_t i = 0; i < num_merges; i++) {
            uint8_t len1 = 0, len2 = 0;
            fm.read((char*)&len1, 1);
            std::string p1(len1, '\0');
            fm.read(&p1[0], len1);

            fm.read((char*)&len2, 1);
            std::string p2(len2, '\0');
            fm.read(&p2[0], len2);

            uint32_t rank = 0;
            fm.read((char*)&rank, 4);
            merges[{p1, p2}] = (int)rank;
        }
        fm.close();
        return true;
    }

    // GPT-2 pattern matching: ' ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+'
    std::vector<std::string> gpt2_split(const std::string &text) {
        std::vector<std::string> chunks;
        size_t i = 0;
        while (i < text.length()) {
            size_t start = i;
            bool has_space = false;
            if (text[i] == ' ') {
                has_space = true;
                i++;
            }

            if (i < text.length() && (isalnum((unsigned char)text[i]) || (unsigned char)text[i] >= 128)) {
                // Word or alphanumeric
                while (i < text.length() && (isalnum((unsigned char)text[i]) || (unsigned char)text[i] >= 128)) {
                    i++;
                }
            } else if (i < text.length() && isspace((unsigned char)text[i])) {
                // Multiple consecutive spaces
                while (i < text.length() && isspace((unsigned char)text[i])) i++;
            } else if (i < text.length()) {
                // Punctuation
                i++;
            }
            chunks.push_back(text.substr(start, i - start));
        }
        return chunks;
    }

    std::vector<int> tokenize(const std::string &text) {
        std::vector<int> token_ids;
        std::vector<std::string> chunks = gpt2_split(text);

        for (const auto &chunk : chunks) {
            std::string bpe_chunk = "";
            for (unsigned char c : chunk) {
                bpe_chunk += byte_to_unicode[c];
            }

            std::vector<std::string> word;
            for (size_t j = 0; j < bpe_chunk.length(); ) {
                unsigned char lead = (unsigned char)bpe_chunk[j];
                int char_len = 1;
                if ((lead & 0xE0) == 0xC0) char_len = 2;
                else if ((lead & 0xF0) == 0xE0) char_len = 3;
                else if ((lead & 0xF8) == 0xF0) char_len = 4;
                word.push_back(bpe_chunk.substr(j, char_len));
                j += char_len;
            }

            while (word.size() > 1) {
                int min_rank = 1000000000;
                int best_idx = -1;
                std::pair<std::string, std::string> best_pair;

                for (size_t k = 0; k < word.size() - 1; k++) {
                    auto it = merges.find({word[k], word[k+1]});
                    if (it != merges.end() && it->second < min_rank) {
                        min_rank = it->second;
                        best_idx = (int)k;
                        best_pair = it->first;
                    }
                }

                if (best_idx == -1) break;

                std::vector<std::string> new_word;
                for (size_t k = 0; k < word.size(); ) {
                    if (k < word.size() - 1 && word[k] == best_pair.first && word[k+1] == best_pair.second) {
                        new_word.push_back(word[k] + word[k+1]);
                        k += 2;
                    } else {
                        new_word.push_back(word[k]);
                        k++;
                    }
                }
                word = new_word;
            }

            for (const auto &p : word) {
                auto it = vocab.find(p);
                if (it != vocab.end()) {
                    token_ids.push_back(it->second);
                }
            }
        }
        return token_ids;
    }
};
