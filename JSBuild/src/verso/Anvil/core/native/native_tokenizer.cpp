#include "unicode.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2> & value) const noexcept {
        return std::hash<T1>{}(value.first) ^ (std::hash<T2>{}(value.second) << 1);
    }
};

struct Symbol {
    using Index = int;

    Index prev = -1;
    Index next = -1;
    const char * text = nullptr;
    std::size_t n = 0;
};

struct Bigram {
    struct Comparator {
        bool operator()(const Bigram & left, const Bigram & right) const noexcept {
            return left.rank > right.rank ||
                   (left.rank == right.rank && left.left > right.left);
        }
    };

    Symbol::Index left = -1;
    Symbol::Index right = -1;
    std::string text;
    int rank = -1;
};

enum class PreType {
    kGPT2,
    kDBRX,
    kQwen35,
};

class NativeBPETokenizer {
public:
    NativeBPETokenizer(
        std::vector<std::string> vocab_tokens,
        std::vector<std::string> merges,
        int bos_id,
        int eos_id,
        int pad_id,
        int unk_id,
        std::string tokenizer_model,
        std::string tokenizer_pre
    ) : vocab_tokens_(std::move(vocab_tokens)),
        bos_id_(bos_id),
        eos_id_(eos_id),
        pad_id_(pad_id),
        unk_id_(unk_id),
        tokenizer_model_(std::move(tokenizer_model)),
        tokenizer_pre_(std::move(tokenizer_pre)) {
        if (tokenizer_model_ != "gpt2") {
            throw std::runtime_error(
                "Native tokenizer currently supports tokenizer.ggml.model=gpt2 only");
        }

        for (std::size_t i = 0; i < vocab_tokens_.size(); ++i) {
            token_to_id_.emplace(vocab_tokens_[i], static_cast<int>(i));
        }

        for (std::size_t i = 0; i < merges.size(); ++i) {
            const std::string & merge = merges[i];
            const std::size_t pos = merge.find(' ', 1);
            if (pos == std::string::npos) {
                continue;
            }
            bpe_ranks_.emplace(
                std::make_pair(merge.substr(0, pos), merge.substr(pos + 1)),
                static_cast<int>(i)
            );
        }

        pre_type_ = resolve_pre_type(tokenizer_pre_);
        regex_exprs_ = build_regex_exprs(pre_type_);
        infer_control_tokens();
    }

    std::vector<int> encode(const std::string & text, bool add_bos, bool add_eos) const {
        std::vector<int> output;
        if (add_bos && valid_id(bos_id_)) {
            output.push_back(bos_id_);
        }

        std::size_t pos = 0;
        while (pos < text.size()) {
            int special_id = -1;
            std::size_t special_len = 0;
            if (match_special_token(text, pos, special_id, special_len)) {
                output.push_back(special_id);
                pos += special_len;
                continue;
            }

            std::size_t next_special = find_next_special(text, pos);
            tokenize_segment(text.substr(pos, next_special - pos), output);
            pos = next_special;
        }

        if (add_eos && valid_id(eos_id_)) {
            output.push_back(eos_id_);
        }
        return output;
    }

    std::string decode(const std::vector<int> & token_ids, bool skip_special) const {
        std::string encoded;
        for (int token_id : token_ids) {
            if (!valid_id(token_id)) {
                continue;
            }
            if (skip_special && decode_skip_ids_.count(token_id) > 0) {
                continue;
            }
            encoded.append(vocab_tokens_[static_cast<std::size_t>(token_id)]);
        }

        std::string result;
        result.reserve(encoded.size());
        for (std::size_t offset = 0; offset < encoded.size();) {
            const std::size_t char_len = unicode_len_utf8(encoded[offset]);
            const std::string piece = encoded.substr(offset, char_len);
            try {
                result.push_back(static_cast<char>(unicode_utf8_to_byte(piece)));
            } catch (...) {
                result.append(piece);
            }
            offset += char_len;
        }
        return result;
    }

    const std::unordered_set<int> & suppressed_token_ids() const {
        return decode_skip_ids_;
    }

private:
    static PreType resolve_pre_type(const std::string & tokenizer_pre) {
        if (tokenizer_pre == "qwen35") {
            return PreType::kQwen35;
        }
        if (tokenizer_pre == "dbrx" || tokenizer_pre == "smaug-bpe") {
            return PreType::kDBRX;
        }
        return PreType::kGPT2;
    }

    static std::vector<std::string> build_regex_exprs(PreType pre_type) {
        switch (pre_type) {
            case PreType::kDBRX:
                return {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
            case PreType::kQwen35:
                return {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
            case PreType::kGPT2:
            default:
                return {
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
        }
    }

    void infer_control_tokens() {
        static const std::unordered_set<std::string> exact_markers = {
            "<|im_start|>",
            "<|im_end|>",
            "<|start_of_role|>",
            "<|end_of_role|>",
            "<|end_of_text|>",
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            "<bos>",
            "<eos>",
            "<pad>",
            "<unk>",
            "<s>",
            "</s>",
        };

        auto add_special = [this](int token_id) {
            if (valid_id(token_id)) {
                decode_skip_ids_.insert(token_id);
            }
        };
        add_special(bos_id_);
        add_special(eos_id_);
        add_special(pad_id_);
        add_special(unk_id_);

        for (std::size_t token_id = 0; token_id < vocab_tokens_.size(); ++token_id) {
            const std::string & token = vocab_tokens_[token_id];
            const bool is_control =
                exact_markers.count(token) > 0 ||
                (token.size() >= 4 && token.rfind("<|", 0) == 0 &&
                 token.substr(token.size() - 2) == "|>");
            if (!is_control) {
                continue;
            }
            const int token_id_int = static_cast<int>(token_id);
            control_token_ids_.insert(token_id_int);
            decode_skip_ids_.insert(token_id_int);
            special_token_texts_.push_back(token);
            special_token_text_to_id_.emplace(token, token_id_int);
        }

        std::sort(
            special_token_texts_.begin(),
            special_token_texts_.end(),
            [](const std::string & left, const std::string & right) {
                return left.size() > right.size();
            }
        );
    }

    bool valid_id(int token_id) const {
        return token_id >= 0 && static_cast<std::size_t>(token_id) < vocab_tokens_.size();
    }

    bool match_special_token(
        const std::string & text,
        std::size_t pos,
        int & token_id,
        std::size_t & token_len
    ) const {
        for (const std::string & candidate : special_token_texts_) {
            if (candidate.empty() || pos + candidate.size() > text.size()) {
                continue;
            }
            if (text.compare(pos, candidate.size(), candidate) != 0) {
                continue;
            }
            token_id = special_token_text_to_id_.at(candidate);
            token_len = candidate.size();
            return true;
        }
        return false;
    }

    std::size_t find_next_special(const std::string & text, std::size_t pos) const {
        std::size_t next = text.size();
        for (const std::string & candidate : special_token_texts_) {
            if (candidate.empty()) {
                continue;
            }
            const std::size_t found = text.find(candidate, pos);
            if (found != std::string::npos) {
                next = std::min(next, found);
            }
        }
        return next;
    }

    void tokenize_segment(const std::string & text, std::vector<int> & output) const {
        if (text.empty()) {
            return;
        }
        const auto word_collection = unicode_regex_split(text, regex_exprs_);
        std::vector<Symbol> symbols_final;
        int final_prev_index = -1;

        for (const auto & word : word_collection) {
            std::priority_queue<Bigram, std::vector<Bigram>, Bigram::Comparator> queue;
            std::vector<Symbol> symbols;
            int index = 0;
            std::size_t offset = 0;

            while (offset < word.size()) {
                Symbol symbol;
                symbol.text = word.c_str() + offset;
                symbol.n = std::min(word.size() - offset, unicode_len_utf8(word[offset]));
                offset += symbol.n;
                symbol.prev = index - 1;
                symbol.next = offset == word.size() ? -1 : index + 1;
                ++index;
                symbols.push_back(symbol);
            }

            auto add_bigram = [&](int left, int right) {
                if (left < 0 || right < 0) {
                    return;
                }
                const std::string left_token(symbols[left].text, symbols[left].n);
                const std::string right_token(symbols[right].text, symbols[right].n);
                const auto it = bpe_ranks_.find(std::make_pair(left_token, right_token));
                if (it == bpe_ranks_.end()) {
                    return;
                }
                Bigram bigram;
                bigram.left = left;
                bigram.right = right;
                bigram.text = left_token + right_token;
                bigram.rank = it->second;
                queue.push(std::move(bigram));
            };

            for (int i = 1; i < static_cast<int>(symbols.size()); ++i) {
                add_bigram(i - 1, i);
            }

            while (!queue.empty()) {
                Bigram bigram = queue.top();
                queue.pop();

                Symbol & left_symbol = symbols[bigram.left];
                Symbol & right_symbol = symbols[bigram.right];
                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }

                const std::string left_token(left_symbol.text, left_symbol.n);
                const std::string right_token(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;
                }

                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_bigram(left_symbol.prev, bigram.left);
                add_bigram(bigram.left, left_symbol.next);
            }

            for (auto & symbol : symbols) {
                if (symbol.n == 0) {
                    continue;
                }
                symbol.prev = final_prev_index;
                symbol.next = -1;
                if (final_prev_index >= 0) {
                    symbols_final[final_prev_index].next =
                        static_cast<int>(symbols_final.size());
                }
                symbols_final.push_back(symbol);
                final_prev_index = static_cast<int>(symbols_final.size()) - 1;
            }
        }

        for (int i = symbols_final.empty() ? -1 : 0; i != -1; i = symbols_final[i].next) {
            const Symbol & symbol = symbols_final[i];
            const std::string piece(symbol.text, symbol.n);
            const auto token_it = token_to_id_.find(piece);
            if (token_it != token_to_id_.end()) {
                output.push_back(token_it->second);
                continue;
            }

            for (char ch : piece) {
                const std::string byte_str(1, ch);
                const auto byte_it = token_to_id_.find(byte_str);
                if (byte_it != token_to_id_.end()) {
                    output.push_back(byte_it->second);
                } else if (valid_id(unk_id_)) {
                    output.push_back(unk_id_);
                }
            }
        }
    }

    std::vector<std::string> vocab_tokens_;
    std::unordered_map<std::string, int> token_to_id_;
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpe_ranks_;
    std::unordered_set<int> control_token_ids_;
    std::unordered_set<int> decode_skip_ids_;
    std::unordered_map<std::string, int> special_token_text_to_id_;
    std::vector<std::string> special_token_texts_;
    std::vector<std::string> regex_exprs_;
    int bos_id_ = -1;
    int eos_id_ = -1;
    int pad_id_ = -1;
    int unk_id_ = -1;
    std::string tokenizer_model_;
    std::string tokenizer_pre_;
    PreType pre_type_ = PreType::kGPT2;
};

std::vector<std::string> copy_c_string_array(const char ** values, int count) {
    std::vector<std::string> out;
    if (values == nullptr || count <= 0) {
        return out;
    }
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        out.emplace_back(values[i] == nullptr ? "" : values[i]);
    }
    return out;
}

}  // namespace

extern "C" {

void * anvil_native_tokenizer_create(
    const char ** vocab_tokens,
    int vocab_count,
    const char ** merges,
    int merge_count,
    int bos_id,
    int eos_id,
    int pad_id,
    int unk_id,
    const char * tokenizer_model,
    const char * tokenizer_pre
) {
    try {
        return new NativeBPETokenizer(
            copy_c_string_array(vocab_tokens, vocab_count),
            copy_c_string_array(merges, merge_count),
            bos_id,
            eos_id,
            pad_id,
            unk_id,
            tokenizer_model == nullptr ? "" : tokenizer_model,
            tokenizer_pre == nullptr ? "" : tokenizer_pre
        );
    } catch (...) {
        return nullptr;
    }
}

void anvil_native_tokenizer_destroy(void * handle) {
    delete static_cast<NativeBPETokenizer *>(handle);
}

int anvil_native_tokenizer_encode(
    void * handle,
    const char * text,
    int add_bos,
    int add_eos,
    int * out_tokens,
    int max_tokens
) {
    if (handle == nullptr) {
        return -1;
    }
    try {
        const auto tokens = static_cast<NativeBPETokenizer *>(handle)->encode(
            text == nullptr ? "" : text,
            add_bos != 0,
            add_eos != 0
        );
        const int required = static_cast<int>(tokens.size());
        if (out_tokens == nullptr || max_tokens < required) {
            return required;
        }
        std::memcpy(out_tokens, tokens.data(), sizeof(int) * tokens.size());
        return required;
    } catch (...) {
        return -1;
    }
}

int anvil_native_tokenizer_decode(
    void * handle,
    const int * token_ids,
    int token_count,
    int skip_special,
    char * out_text,
    int max_bytes
) {
    if (handle == nullptr) {
        return -1;
    }
    try {
        std::vector<int> ids;
        if (token_ids != nullptr && token_count > 0) {
            ids.assign(token_ids, token_ids + token_count);
        }
        const std::string text = static_cast<NativeBPETokenizer *>(handle)->decode(
            ids,
            skip_special != 0
        );
        const int required = static_cast<int>(text.size());
        if (out_text == nullptr || max_bytes <= required) {
            return required;
        }
        std::memcpy(out_text, text.data(), text.size());
        out_text[required] = '\0';
        return required;
    } catch (...) {
        return -1;
    }
}

int anvil_native_tokenizer_get_suppressed_token_count(void * handle) {
    if (handle == nullptr) {
        return -1;
    }
    const auto & ids =
        static_cast<NativeBPETokenizer *>(handle)->suppressed_token_ids();
    return static_cast<int>(ids.size());
}

int anvil_native_tokenizer_get_suppressed_tokens(
    void * handle,
    int * out_ids,
    int max_ids
) {
    if (handle == nullptr) {
        return -1;
    }
    const auto & ids =
        static_cast<NativeBPETokenizer *>(handle)->suppressed_token_ids();
    const int required = static_cast<int>(ids.size());
    if (out_ids == nullptr || max_ids < required) {
        return required;
    }
    int idx = 0;
    for (int token_id : ids) {
        out_ids[idx++] = token_id;
    }
    return required;
}

}
