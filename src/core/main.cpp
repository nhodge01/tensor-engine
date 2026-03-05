#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <thread>
#include <atomic>
#include <string_view>
#include <cstring>
#include "tokenizer_wrapper.hpp"
#include "tokenizer_worker.hpp"
#include "harvester_worker.hpp"
#include "tensor_engine.hpp"      
#include "threadsafequeue.hpp"
#include "batch_job.hpp"
#include "lake.hpp"               // <--- Using the new header-only library!
#include "env_loader.hpp"
#include "config.hpp"

using namespace std::chrono;

// ZERO-COPY SPIGOT WORKER - Extracts at 4M+ RPS
void spigot_worker(duckdb::unique_ptr<duckdb::MaterializedQueryResult>& result,
                   std::vector<duckdb::unique_ptr<duckdb::DataChunk>>& chunk_vault,
                   ThreadSafeQueue<BatchJob*>& free_queue,
                   ThreadSafeQueue<BatchJob*>& tokenizer_queue,
                   const ModelConfig& config) {
    BatchJob* job = nullptr;
    int job_idx = 0;
    size_t total_extracted = 0;
    int num_vectors = config.input_columns.size();

    // Mapping phase: Find indices of columns defined in config
    std::vector<int> col_indices;

    // DuckDB columns are in order of SELECT statement
    // Column 0 is gid, columns 1-N are the input columns
    for (size_t i = 0; i < config.input_columns.size(); i++) {
        col_indices.push_back(i + 1);  // +1 because gid is at index 0
    }

    std::cout << "Spigot: Starting MULTI-VECTOR extraction (" << num_vectors << " vectors/row)..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    while (auto chunk = result->Fetch()) {
        if (!chunk || chunk->size() == 0) break;
        if (!chunk->AllConstant()) chunk->Flatten();

        auto gid_data = duckdb::FlatVector::GetData<int64_t>(chunk->data[0]);
        auto& gid_validity = duckdb::FlatVector::Validity(chunk->data[0]);
        size_t count = chunk->size();

        for (size_t i = 0; i < count; i++) {
            if (!job) {
                if (!free_queue.wait_and_loop(job)) return;
                job_idx = 0;
            }

            // Set GID
            job->row_ids[job_idx] = gid_validity.RowIsValid(i) ? gid_data[i] : 0;

            // Extract all N versions for this GID
            for (int v = 0; v < num_vectors; ++v) {
                int col_idx = col_indices[v];
                auto& text_validity = duckdb::FlatVector::Validity(chunk->data[col_idx]);
                auto text_data = duckdb::FlatVector::GetData<duckdb::string_t>(chunk->data[col_idx]);

                int slot = (job_idx * num_vectors) + v;
                if (text_validity.RowIsValid(i)) {
                    job->raw_texts[slot] = std::string_view(text_data[i].GetData(), text_data[i].GetSize());
                } else {
                    job->raw_texts[slot] = std::string_view();
                }
            }

            job_idx++;
            total_extracted++;

            if (job_idx == job->max_batch_size) {
                job->valid_items = job_idx;
                tokenizer_queue.push(job);
                job = nullptr; 
            }
        }
        chunk_vault.push_back(std::move(chunk));
    }

    if (job && job_idx > 0) {
        job->valid_items = job_idx;
        tokenizer_queue.push(job);
    }

    tokenizer_queue.shutdown();
}


int main() {
    // Load environment variables
    acrelab::EnvLoader::LoadEnvFile(".env", true);

    auto program_start = std::chrono::high_resolution_clock::now();

    std::cout << "=== ZERO-COPY DUAL-GPU PIPELINE ===" << std::endl;
    std::cout << "4M+ RPS DuckLake -> Tokenizers -> Dual 4090s -> MinIO\n" << std::endl;

    // Load model configuration
    ModelConfig config = ModelConfig::load_from_json("src/config.json");
    std::cout << std::endl;

    // 1. Initialize DuckLake connection (The beautiful 1-liner)
    DuckLake lake;

    // 2. Check S3 output configuration
    if (config.enable_s3_output && config.s3_output_path.empty()) {
        std::cerr << "ERROR: S3 output is enabled but no path specified in config!" << std::endl;
        return 1;
    }
    if (config.enable_s3_output) {
        std::cout << "S3 output enabled. Will write to: " << config.s3_output_path << std::endl;
    } else {
        std::cout << "S3 output disabled." << std::endl;
    }

    // 3. Load Tokenizer
    // auto tokenizer = acrelab::TokenizerWrapper::FromFile(config.vocab_path);
    // if (!tokenizer) {
    //     std::cerr << "Failed to load tokenizer!" << std::endl;
    //     return 1;
    // }

    
    // 4. Initialize the Queues
    ThreadSafeQueue<BatchJob*> free_queue;
    ThreadSafeQueue<BatchJob*> tokenizer_queue;
    ThreadSafeQueue<BatchJob*> inference_queue;
    ThreadSafeQueue<BatchJob*> harvester_queue;
    std::atomic<int> total_queries_processed{0};


    const int num_tokenizers = 16;
    std::cout << "Initializing Tokenizer Pool: (" << config.tokenizer_type <<") with " << num_tokenizers << " threads..." << std::endl;
    acrelab::TokenizerWorkerPool tokenizer_pool(
        num_tokenizers,
        config,
        tokenizer_queue,
        inference_queue
    );

    

    int batch_size = config.batch_size;
    int seq_len = config.max_tokens;
    int hidden_dim = config.embedding_dim;
    int num_jobs_in_pool = 1500; 

    

    // 5. Pre-Allocate the Pinned Memory Pool
    int num_vectors = config.input_columns.size();
    std::cout << "Allocating " << num_jobs_in_pool << " pinned memory structs..." << std::endl;
    std::vector<BatchJob*> job_pool;
    for (int i = 0; i < num_jobs_in_pool; ++i) {
        // Updated Constructor call!
        auto job = new BatchJob(i, config.batch_size, num_vectors, config.max_tokens, config.embedding_dim);
        job_pool.push_back(job);
        free_queue.push(job);
    }
    // 6. Updated SQL Query - MUST match config.input_columns exactly!
        std::string query = R"(
            SELECT
                gid,
                COALESCE(standard_owner_1_first_last_name, '') as raw_owner_name_1,
                CASE
                    WHEN standard_owner_2_first_last_name IS NOT NULL AND standard_owner_2_first_last_name != ''
                    THEN 'Property Owner: ' || standard_owner_2_first_last_name
                    ELSE 'Property Owner: '
                END as context_owner_name_2,
                CASE
                    WHEN standard_owner_2_first_last_name IS NOT NULL AND standard_owner_2_first_last_name != ''
                    THEN COALESCE(standard_owner_1_first_last_name, '') || ', ' || standard_owner_2_first_last_name
                    ELSE COALESCE(standard_owner_1_first_last_name, '')
                END as raw_owner_names_concatenated,
                COALESCE(standard_mail_address, '') as raw_address,
                'Property Owners: ' ||
                CASE
                    WHEN standard_owner_2_first_last_name IS NOT NULL AND standard_owner_2_first_last_name != ''
                    THEN COALESCE(standard_owner_1_first_last_name, '') || ', ' || standard_owner_2_first_last_name
                    ELSE COALESCE(standard_owner_1_first_last_name, '')
                END || ' | Property Mailing Address: ' || COALESCE(standard_mail_address, '') as context_both
            FROM lake.desoto_jrrtolkein.consolidated_parcel
            WHERE state IN ('RI', 'AR')
        )";

        std::cout << "Executing query..." << std::endl;
        auto result = lake.conn.Query(query);

        if (!result->HasError()) {
            size_t total_rows_in_db = result->RowCount();
        std::cout << "Query returned " << total_rows_in_db << " rows" << std::endl;

        std::vector<duckdb::unique_ptr<duckdb::DataChunk>> chunk_vault;
        chunk_vault.reserve(total_rows_in_db / batch_size + 100);

        // 7. Start Harvester (Dynamic column names from config)
        std::cout << "Spawning Harvester thread..." << std::endl;
        std::thread harvester_thread = acrelab::CreateHarvesterThread(
            harvester_queue, free_queue, total_queries_processed,
            config.s3_output_path, config.embedding_dim, config.input_columns
        );

        // 8. Start Dual GPU Engines
        std::cout << "Spawning GPU workers..." << std::endl;
        TensorEngine engine0(0, config.engine_path, config, inference_queue, harvester_queue);
        TensorEngine engine1(1, config.engine_path, config, inference_queue, harvester_queue);
        engine0.start();
        engine1.start();

        // 9. Start Tokenizer Pool
        tokenizer_pool.Start();

        // 10. Start the pipeline
        std::cout << "\n--- PIPELINE LIVE: Processing " << total_rows_in_db << " rows ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Blocks main thread until DB is drained. Passing config so it knows columns.
        spigot_worker(result, chunk_vault, free_queue, tokenizer_queue, config);

        // --- Shutdown Sequence ---
        tokenizer_pool.Join();
        inference_queue.shutdown();
        engine0.join();
        engine1.join();

        harvester_queue.shutdown();
        if (harvester_thread.joinable()) harvester_thread.join();

        // ... [Final Statistics Print] ...
        // 11. Print final statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time_sec = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        std::cout << "\n=== PIPELINE COMPLETE ===" << std::endl;
        std::cout << "Total rows processed: " << total_queries_processed.load() << std::endl;
        std::cout << "Total time: " << total_time_sec << " seconds" << std::endl;
        std::cout << "Overall throughput: " << (total_queries_processed.load() / (total_time_sec + 0.001))
                  << " rows/second" << std::endl;

        // 12. Cleanup Memory Pool
        for (auto job : job_pool) {
            delete job;
        }

    } else {
        std::cerr << "Query failed: " << result->GetError() << std::endl;
        return 1;
    }

    return 0;
}
