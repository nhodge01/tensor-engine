#pragma once

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <duckdb.hpp>

// Define our types
using hr_clock = std::chrono::high_resolution_clock;

struct TextBatch {
    std::vector<int64_t> gids;
    std::vector<std::string> texts;
    size_t num_rows = 0;
};

// Inline helper function
inline std::string get_env_or_throw(const char* key) {
    if (const char* val = std::getenv(key)) {
        return std::string(val);
    }
    throw std::runtime_error(std::string(key) + " missing from environment...");
}

class DuckLake {
public:
    duckdb::DuckDB db;
    duckdb::Connection conn;

    // Constructor handles the attachment
    DuckLake() : db(":memory:"), conn(db) {
        auto start = hr_clock::now();

        std::string db_host = get_env_or_throw("DB_HOST");
        std::string db_port = get_env_or_throw("DB_PORT"); // Note: You need this in your .env now!
        std::string db_name = get_env_or_throw("DB_NAME");
        std::string db_user = get_env_or_throw("DB_USER");
        std::string db_pass = get_env_or_throw("DB_PASS");

        std::string minio_endpoint = get_env_or_throw("MINIO_ENDPOINT");
        std::string minio_access_key = get_env_or_throw("MINIO_ACCESS_KEY");
        std::string minio_secret_key = get_env_or_throw("MINIO_SECRET_KEY");

        // Lambda Checker
        auto execute = [this](const std::string& query, const std::string& step_name) {
            auto result = conn.Query(query);
            if (result->HasError()) {
                throw std::runtime_error(step_name + " failed: " + result->GetError());
            } 
        };

        std::cout << "DuckLake Connection Loading...\n";
        
        // Setup base config
        execute("SET memory_limit='8GB';", "Memory Limit Config");
        execute("SET threads=16;", "Thread Config");

        execute(
            "INSTALL ducklake; INSTALL httpfs; INSTALL postgres;"
            "LOAD ducklake; LOAD httpfs; LOAD postgres;",
            "Extension Load"
        );

        std::string minio_query = 
            "CREATE OR REPLACE SECRET minio_secret ("
            "    TYPE s3,"
            "    KEY_ID '" + minio_access_key + "',"
            "    SECRET '" + minio_secret_key + "',"
            "    ENDPOINT '" + minio_endpoint + "',"
            "    URL_STYLE 'path',"
            "    USE_SSL false"
            ");";
        execute(minio_query, "MinIO Secret Creation");
        
        std::string ducklake_query = 
            "CREATE OR REPLACE SECRET acrelake ("
            "    TYPE ducklake,"
            "    METADATA_PATH 'postgres:host=" + db_host + " port=" + db_port + 
            " dbname=" + db_name + " user=" + db_user + " password=" + db_pass + " sslmode=require',"
            "    DATA_PATH 's3://REDACTED_BUCKET'"
            ");";
        execute(ducklake_query, "DuckLake Secret Creation");

        std::cout << "Attaching DuckLake...\n";
        execute("ATTACH 'ducklake:acrelake' AS lake;", "DuckLake Attachment");

        auto end = hr_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        std::cout << "DuckDB DuckLake Connection Established! (" << ms.count() << " ms)\n";
    }

    // Inline the extraction method
    inline TextBatch extract_text_batch(const std::string& query, size_t reserve_size) {
        auto start_total = hr_clock::now();
        TextBatch batch;
        batch.gids.reserve(reserve_size);
        batch.texts.reserve(reserve_size);

        auto start_query = hr_clock::now();
        auto result = conn.SendQuery(query);
        auto end_query = hr_clock::now();

        if (result->HasError()) {
            std::cerr << "Query Error: " << result->GetError() << std::endl;
            return batch;
        }
        
        auto start_fetch = hr_clock::now();
        while (true) {
            auto chunk = result->Fetch();
            if (!chunk || chunk->size() == 0) break;

            chunk->Flatten();
            size_t chunk_size = chunk->size();

            auto& gid_validity = duckdb::FlatVector::Validity(chunk->data[0]);
            auto& text_validity = duckdb::FlatVector::Validity(chunk->data[1]);
            auto gid_data = duckdb::FlatVector::GetData<int64_t>(chunk->data[0]);
            auto text_data = duckdb::FlatVector::GetData<duckdb::string_t>(chunk->data[1]);

            for (size_t i = 0; i < chunk_size; i++) {
                int64_t gid = gid_validity.RowIsValid(i) ? gid_data[i] : 0;
                batch.gids.emplace_back(gid);
                
                if (text_validity.RowIsValid(i)) {
                    batch.texts.emplace_back(text_data[i].GetData(), text_data[i].GetSize());
                } else {
                    batch.texts.emplace_back("");
                }
            }
            batch.num_rows += chunk_size;
        }
        
        auto end_total = hr_clock::now();

        std::chrono::duration<double, std::milli> query_ms = end_query - start_query;
        std::chrono::duration<double, std::milli> fetch_ms = end_total - start_fetch;
        std::chrono::duration<double, std::milli> total_ms = end_total - start_total;

        std::cout << "[Lake] Fetched " << batch.num_rows << " rows.\n";
        std::cout << "  -> Query execution: " << query_ms.count() << " ms\n";
        std::cout << "  -> Data copy to RAM: " << fetch_ms.count() << " ms\n";
        std::cout << "  -> Total Time: " << total_ms.count() << " ms\n";
        std::cout << "  -> Throughput: " << (batch.num_rows / (total_ms.count() / 1000.0)) << " rows/sec\n";

        return batch; 
    }
};
