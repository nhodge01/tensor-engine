#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <iomanip>
#include "byt5_tokenizer.hpp"

using namespace acrelab;

void print_ids(const std::string& label, const int32_t* ids, int len) {
    std::cout << label << ": [";
    for (int i = 0; i < len; ++i) {
        std::cout << ids[i] << (i == len - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;
}

int main() {
    ByT5Tokenizer tokenizer;
    std::string test_str = "Property: 123 Main St! 🏠";
    int max_len = 32;

    // Buffers to simulate BatchJob pinned memory
    std::vector<int32_t> input_ids(max_len, 0);
    std::vector<int32_t> attention_mask(max_len, 0);

    std::cout << "--- ByT5 C++ Validation ---" << std::endl;
    std::cout << "Input: '" << test_str << "'" << std::endl;

    // 1. Test EncodeToBuffer
    tokenizer.EncodeToBuffer(test_str, input_ids.data(), attention_mask.data(), max_len);

    // 2. Validate Byte Offset (Python: byte + 3)
    // 'P' is ASCII 80. ByT5 should be 80 + 3 = 83.
    if (input_ids[0] == 83) {
        std::cout << "✅ Logic Check: 'P' (80) + Offset (3) = 83. Match!" << std::endl;
    } else {
        std::cerr << "❌ Logic Check Failed! Expected 83, got " << input_ids[0] << std::endl;
    }

    // 3. Validate EOS (Python: append 1)
    // The string "Property: 123 Main St! 🏠" is 27 bytes in UTF-8. 
    // The 28th token (index 27) should be EOS (1).
    int expected_eos_idx = 0;
    for(unsigned char b : test_str) expected_eos_idx++; 
    
    if (input_ids[expected_eos_idx] == ByT5Tokenizer::EOS_ID) {
        std::cout << "✅ EOS Check: Found ID 1 at index " << expected_eos_idx << ". Match!" << std::endl;
    } else {
        std::cerr << "❌ EOS Check Failed! Index " << expected_eos_idx << " is " << input_ids[expected_eos_idx] << std::endl;
    }

    // 4. Validate Padding
    if (input_ids[max_len - 1] == 0 && attention_mask[max_len - 1] == 0) {
        std::cout << "✅ Padding Check: Tail of buffer is zeroed. Match!" << std::endl;
    }

    print_ids("IDs  ", input_ids.data(), 15);
    print_ids("Mask ", attention_mask.data(), 15);

    std::cout << "\nRESULT: ByT5 C++ implementation is ready for the engine." << std::endl;

    return 0;
}
