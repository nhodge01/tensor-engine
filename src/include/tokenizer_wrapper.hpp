#ifndef ACRELAB_GAUSS_TOKENIZER_WRAPPER_HPP
#define ACRELAB_GAUSS_TOKENIZER_WRAPPER_HPP

#include <memory>
#include <vector>
#include <string>
#include <string_view>
#include <fstream>
#include <mutex>
#include <algorithm>
#include "tokenizers_cpp.h"
#include "tokenizer_base.hpp" // Ensure this exists in your include path

namespace acrelab {

/**
 * @brief Thread-safe wrapper around the Rust tokenizer library, 
 * now implementing the ITokenizer interface for polymorphic switching.
 */
class TokenizerWrapper : public ITokenizer {
public:
    /**
     * @brief Factory: Load tokenizer from a JSON file path
     */
    static std::unique_ptr<TokenizerWrapper> FromFile(const std::string& json_path) {
        std::ifstream ifs(json_path);
        if (!ifs.is_open()) return nullptr;
        
        std::string json_blob((std::istreambuf_iterator<char>(ifs)),
                              std::istreambuf_iterator<char>());
        return FromJSON(json_blob);
    }

    /**
     * @brief Factory: Load tokenizer from a JSON string blob
     */
    static std::unique_ptr<TokenizerWrapper> FromJSON(const std::string& json_blob) {
        auto rust_tokenizer = tokenizers::Tokenizer::FromBlobJSON(json_blob);
        if (!rust_tokenizer) return nullptr;
        
        // Using new because constructor is private
        return std::unique_ptr<TokenizerWrapper>(
            new TokenizerWrapper(std::move(rust_tokenizer))
        );
    }

    // --- ITokenizer Interface Implementation ---

    /**
     * @brief Standard vector-based encoding (Overrides ITokenizer)
     */
    std::vector<int32_t> Encode(const std::string& text) override {
        return tokenizer_->Encode(text);
    }

    /**
     * @brief High-performance direct buffer write (Overrides ITokenizer)
     * Note: Currently performs a copy for FFI, but satisfies the interface
     */
    void EncodeToBuffer(std::string_view text, int32_t* ids, int32_t* mask, int max_len) override {
        // Rust FFI requires std::string
        auto vec = tokenizer_->Encode(std::string(text));
        
        int write_len = std::min(static_cast<int>(vec.size()), max_len);
        
        // Write tokens and mask
        for (int i = 0; i < write_len; ++i) {
            ids[i] = vec[i];
            mask[i] = 1;
        }
        
        // Fill padding
        for (int i = write_len; i < max_len; ++i) {
            ids[i] = 0; 
            mask[i] = 0;
        }
    }

    // --- Legacy and Helper Methods ---

    std::vector<std::vector<int32_t>> EncodeBatch(const std::vector<std::string>& texts) const {
        return tokenizer_->EncodeBatch(texts);
    }

    std::string Decode(const std::vector<int32_t>& ids) {
        return tokenizer_->Decode(ids);
    }

private:
    explicit TokenizerWrapper(std::unique_ptr<tokenizers::Tokenizer> tokenizer)
        : tokenizer_(std::move(tokenizer)) {}

    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

/**
 * @brief Simple pool for multi-threaded tokenization scenarios
 */
class TokenizerPool {
public:
    explicit TokenizerPool(const std::string& json_path, size_t pool_size = 1) {
        auto wrapper = TokenizerWrapper::FromFile(json_path);
        if (!wrapper) throw std::runtime_error("Failed to load tokenizer: " + json_path);
        
        tokenizers_.push_back(std::move(wrapper));
        // Note: Logic for cloning instances would go here if needed
    }

    TokenizerWrapper* GetTokenizer(int thread_id = 0) {
        return tokenizers_[thread_id % tokenizers_.size()].get();
    }

private:
    std::vector<std::unique_ptr<TokenizerWrapper>> tokenizers_;
};

} // namespace acrelab

#endif // ACRELAB_GAUSS_TOKENIZER_WRAPPER_HPP
