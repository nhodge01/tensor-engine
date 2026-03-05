#pragma once
#include "tokenizer_base.hpp"
#include <algorithm>
#include <vector>
#include <string>

namespace acrelab {

class ByT5Tokenizer : public ITokenizer {
public:
    static constexpr int32_t PAD_ID = 0;
    static constexpr int32_t EOS_ID = 1;
    static constexpr int32_t OFFSET = 3;

    // Added override here for safety
    std::vector<int32_t> Encode(const std::string& text) override {
        std::vector<int32_t> ids;
        ids.reserve(text.size() + 1);
        for (unsigned char b : text) ids.push_back(static_cast<int32_t>(b) + OFFSET);
        ids.push_back(EOS_ID);
        return ids;
    }

    // Added override here as well
    void EncodeToBuffer(std::string_view text, int32_t* ids, int32_t* mask, int max_len) override {
        int i = 0;
        // 1. Encode bytes
        for (; i < (int)text.size() && i < (max_len - 1); ++i) {
            ids[i] = static_cast<unsigned char>(text[i]) + OFFSET;
            mask[i] = 1;
        }
        // 2. Add EOS
        if (i < max_len) {
            ids[i] = EOS_ID;
            mask[i] = 1;
            i++;
        }
        // 3. Pad remainder
        for (; i < max_len; ++i) {
            ids[i] = PAD_ID;
            mask[i] = 0;
        }
    }
};

} // namespace acrelab
