#ifndef ACRELAB_GAUSS_ENV_LOADER_HPP
#define ACRELAB_GAUSS_ENV_LOADER_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <cstdlib>
#include <algorithm>

namespace acrelab {

/**
 * @brief Simple .env file loader for configuration
 *
 * Reads key=value pairs from a .env file and sets them as environment variables.
 * Supports:
 * - Comments starting with #
 * - Empty lines
 * - Quoted values (single or double quotes)
 * - Trimming whitespace
 *
 * Example .env file:
 * ```
 * # Database Configuration
 * DB_HOST=localhost
 * DB_PORT=5432
 * DB_NAME=mydb
 *
 * # MinIO Configuration
 * MINIO_ENDPOINT=your-minio-endpoint:9000
 * MINIO_ACCESS_KEY="your-access-key"
 * MINIO_SECRET_KEY='your-secret-key'
 *
 * # S3 Output Configuration
 * ENABLE_S3_OUTPUT=1
 * S3_OUTPUT_PATH=your-bucket/path/to/output.parquet
 * ```
 */
class EnvLoader {
public:
    /**
     * @brief Load environment variables from a .env file
     *
     * @param filepath Path to the .env file (default: ".env" in current directory)
     * @param override_existing If true, override existing environment variables
     * @return true if file was loaded successfully, false otherwise
     */
    static bool LoadEnvFile(const std::string& filepath = ".env", bool override_existing = true) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            // Try parent directory
            std::ifstream parent_file("../" + filepath);
            if (!parent_file.is_open()) {
                std::cerr << "Warning: Could not open env file: " << filepath << std::endl;
                std::cerr << "Falling back to system environment variables." << std::endl;
                return false;
            }
            file = std::move(parent_file);
        }

        std::cout << "Loading environment from: " << filepath << std::endl;

        std::string line;
        int loaded_count = 0;

        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // Find the = separator
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) {
                continue;
            }

            // Extract key and value
            std::string key = line.substr(0, eq_pos);
            std::string value = line.substr(eq_pos + 1);

            // Trim whitespace from key
            key.erase(0, key.find_first_not_of(" \t\r\n"));
            key.erase(key.find_last_not_of(" \t\r\n") + 1);

            // Trim whitespace from value
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);

            // Remove quotes if present
            if ((value.front() == '"' && value.back() == '"') ||
                (value.front() == '\'' && value.back() == '\'')) {
                value = value.substr(1, value.length() - 2);
            }

            // Check if should override
            if (!override_existing && std::getenv(key.c_str()) != nullptr) {
                continue;
            }

            // Set the environment variable
            if (setenv(key.c_str(), value.c_str(), 1) == 0) {
                loaded_count++;
            }
        }

        std::cout << "Loaded " << loaded_count << " environment variables from " << filepath << std::endl;
        return true;
    }

    /**
     * @brief Get an environment variable with a default value
     */
    static std::string GetEnv(const std::string& key, const std::string& default_value = "") {
        const char* value = std::getenv(key.c_str());
        return value ? std::string(value) : default_value;
    }

    /**
     * @brief Get an environment variable or throw if not found
     */
    static std::string GetEnvOrThrow(const std::string& key) {
        const char* value = std::getenv(key.c_str());
        if (!value) {
            throw std::runtime_error("Environment variable " + key + " not found");
        }
        return std::string(value);
    }

    /**
     * @brief Print all environment variables that match a prefix (for debugging)
     */
    static void PrintEnvVars(const std::string& prefix = "") {
        std::cout << "Environment variables";
        if (!prefix.empty()) {
            std::cout << " (prefix: " << prefix << ")";
        }
        std::cout << ":" << std::endl;

        extern char** environ;
        for (char** env = environ; *env != nullptr; ++env) {
            std::string var(*env);
            if (prefix.empty() || var.find(prefix) == 0) {
                std::cout << "  " << var << std::endl;
            }
        }
    }
};

/**
 * @brief RAII helper to automatically load .env file on program start
 *
 * Usage: Just create a static instance in main.cpp:
 * ```cpp
 * static acrelab::AutoEnvLoader env_loader(".env.ducklake");
 * ```
 */
class AutoEnvLoader {
public:
    explicit AutoEnvLoader(const std::string& filepath = ".env", bool override_existing = true) {
        EnvLoader::LoadEnvFile(filepath, override_existing);
    }
};

} // namespace acrelab

#endif // ACRELAB_GAUSS_ENV_LOADER_HPP