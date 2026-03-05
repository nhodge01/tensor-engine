#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "json.hpp"

using json = nlohmann::json;

struct ModelConfig {
    int embedding_dim;
    int max_tokens;
    int batch_size;
    std::string engine_path;
    std::string vocab_path;
    std::string tokenizer_type; // <-- NEW: Added this field
    std::vector<std::string> input_columns; // Multi-vector column names

    // Output configuration
    bool enable_s3_output;
    std::string s3_output_path;

    static ModelConfig load_from_json(const std::string& config_path) {
        ModelConfig config;

        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }

        json j;
        config_file >> j;

        // Parse the JSON values
        config.embedding_dim = j["embedding_dim"];
        config.max_tokens = j["max_tokens"];
        config.batch_size = j["batch_size"];
        config.engine_path = j["engine_path"];
        config.vocab_path = j["vocab_path"];
        
        // NEW: Parse tokenizer type with a safe default
        config.tokenizer_type = j.value("tokenizer_type", "huggingface");

        // Parse input columns for multi-vector support
        if (j.contains("input_columns")) {
            for (const auto& col : j["input_columns"]) {
                config.input_columns.push_back(col.get<std::string>());
            }
        }

        // Parse output configuration - now at root level
        config.enable_s3_output = j.value("enable_s3_output", false);
        config.s3_output_path = j.value("s3_output_path", "");

        // Validate config
        if (config.embedding_dim <= 0 || config.max_tokens <= 0 || config.batch_size <= 0) {
            throw std::runtime_error("Invalid config values: dimensions must be positive");
        }

        std::cout << "Loaded model config:" << std::endl;
        std::cout << "  - Tokenizer type: " << config.tokenizer_type << std::endl;
        std::cout << "  - Embedding dim: " << config.embedding_dim << std::endl;
        std::cout << "  - Max tokens: " << config.max_tokens << std::endl;
        std::cout << "  - Batch size: " << config.batch_size << std::endl;
        std::cout << "  - S3 output enabled: " << (config.enable_s3_output ? "yes" : "no") << std::endl;
        if (config.enable_s3_output) {
            std::cout << "  - S3 output path: " << config.s3_output_path << std::endl;
        }
        std::cout << "  - Input columns: " << config.input_columns.size() << std::endl;

        return config;
    }
};
