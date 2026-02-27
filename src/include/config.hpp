#pragma once

#include <string>
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

        // Validate config
        if (config.embedding_dim <= 0 || config.max_tokens <= 0 || config.batch_size <= 0) {
            throw std::runtime_error("Invalid config values: dimensions must be positive");
        }

        std::cout << "Loaded model config:" << std::endl;
        std::cout << "  - Embedding dim: " << config.embedding_dim << std::endl;
        std::cout << "  - Max tokens: " << config.max_tokens << std::endl;
        std::cout << "  - Batch size: " << config.batch_size << std::endl;
        std::cout << "  - Engine path: " << config.engine_path << std::endl;
        std::cout << "  - Vocab path: " << config.vocab_path << std::endl;

        return config;
    }
};