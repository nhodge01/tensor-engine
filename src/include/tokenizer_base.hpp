#pragma once
#include <string_view>
#include <vector>
#include <string> // Added to support std::string in the Encode signature
#include <cstdint>

namespace acrelab {

class ITokenizer {
public:
    virtual ~ITokenizer() = default;

    virtual std::vector<int32_t> Encode(const std::string& text) = 0;

    // Fixed the typo: string_view_text -> string_view text
    virtual void EncodeToBuffer(std::string_view text,
                                int32_t* input_ids,
                                int32_t* mask,
                                int max_len) = 0;
};

} // namespace acrelab
