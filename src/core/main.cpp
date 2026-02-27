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
                   ThreadSafeQueue<BatchJob*>& tokenizer_queue) {
    BatchJob* job = nullptr;
    int job_idx = 0;
    size_t total_extracted = 0;

    std::cout << "Spigot: Starting ZERO-COPY extraction..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    while (auto chunk = result->Fetch()) {
        if (!chunk || chunk->size() == 0) break;

        if (!chunk->AllConstant()) {
            chunk->Flatten();
        }

        auto& gid_validity = duckdb::FlatVector::Validity(chunk->data[0]);
        auto& text_validity = duckdb::FlatVector::Validity(chunk->data[1]);
        auto gid_data = duckdb::FlatVector::GetData<int64_t>(chunk->data[0]);
        auto text_data = duckdb::FlatVector::GetData<duckdb::string_t>(chunk->data[1]);

        size_t count = chunk->size();

        for (size_t i = 0; i < count; i++) {
            if (!job) {
                if (!free_queue.wait_and_loop(job)) return;
                job_idx = 0;
            }

            job->row_ids[job_idx] = gid_validity.RowIsValid(i) ? gid_data[i] : 0;

            if (text_validity.RowIsValid(i)) {
                job->raw_texts[job_idx] = std::string_view(text_data[i].GetData(), text_data[i].GetSize());
            } else {
                job->raw_texts[job_idx] = std::string_view();
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

    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::high_resolution_clock::now() - start).count();

    std::cout << "Spigot: Extracted " << total_extracted << " rows in " << elapsed
              << " seconds (" << (total_extracted / (elapsed + 0.001)) << " RPS)" << std::endl;

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

    // 2. Check required S3 output path
    const char* s3_path_env = std::getenv("S3_OUTPUT_PATH");
    if (!s3_path_env || strlen(s3_path_env) == 0) {
        std::cerr << "ERROR: S3_OUTPUT_PATH environment variable is required!" << std::endl;
        return 1;
    }
    std::string s3_output_path(s3_path_env);
    std::cout << "Output will be written to: " << s3_output_path << std::endl;

    // 3. Load Tokenizer
    auto tokenizer = acrelab::TokenizerWrapper::FromFile(config.vocab_path);
    if (!tokenizer) {
        std::cerr << "Failed to load tokenizer!" << std::endl;
        return 1;
    }

    int batch_size = config.batch_size;
    int seq_len = config.max_tokens;
    int hidden_dim = config.embedding_dim;
    int num_jobs_in_pool = 1500; 

    // 4. Initialize the Queues
    ThreadSafeQueue<BatchJob*> free_queue;
    ThreadSafeQueue<BatchJob*> tokenizer_queue;
    ThreadSafeQueue<BatchJob*> inference_queue;
    ThreadSafeQueue<BatchJob*> harvester_queue;
    std::atomic<int> total_queries_processed{0};

    // 5. Pre-Allocate the Pinned Memory Pool
    std::cout << "Allocating " << num_jobs_in_pool << " pinned memory structs..." << std::endl;
    std::vector<BatchJob*> job_pool;
    for (int i = 0; i < num_jobs_in_pool; ++i) {
        auto job = new BatchJob(i, batch_size, seq_len, hidden_dim);
        job_pool.push_back(job);
        free_queue.push(job);
    }

    // 6. Query the database
    std::string query = R"(
        SELECT gid, concat_ws(' ', owner1_name, owner2_name, mail_address) as text
        FROM lake.desoto_jrrtolkein.consolidated_parcel
        WHERE state IN ('RI', 'AR')
    )";

    std::cout << "Executing query..." << std::endl;
    
    // Execute using the connection from our new DuckLake object!
    auto result = lake.conn.Query(query);

    if (!result->HasError()) {
        size_t total_rows_in_db = result->RowCount();
        std::cout << "Query returned " << total_rows_in_db << " rows" << std::endl;

        std::vector<duckdb::unique_ptr<duckdb::DataChunk>> chunk_vault;
        chunk_vault.reserve(total_rows_in_db / batch_size + 100);

        // 7. Start Harvester
        std::cout << "Spawning Harvester thread with S3/MinIO output..." << std::endl;
        acrelab::HarvesterWorker harvester(harvester_queue, free_queue, total_queries_processed, s3_output_path, hidden_dim);
        std::thread harvester_thread([&harvester]() { harvester.Run(); });

        // 8. Start Dual GPU Engines
        std::cout << "Spawning GPU workers..." << std::endl;
        TensorEngine engine0(0, config.engine_path, config, inference_queue, harvester_queue);
        TensorEngine engine1(1, config.engine_path, config, inference_queue, harvester_queue);
        
        engine0.start();
        engine1.start();

        // 9. Start Tokenizers (Set to 16/24 based on your previous tuning!)
        const int num_tokenizers = 16;
        std::cout << "Spawning " << num_tokenizers << " tokenizer threads..." << std::endl;
        std::vector<std::unique_ptr<acrelab::TokenizerWorker>> tokenizer_workers;
        std::vector<std::thread> tokenizer_threads;

        for (int i = 0; i < num_tokenizers; i++) {
            auto worker = std::make_unique<acrelab::TokenizerWorker>(
                i, tokenizer.get(), tokenizer_queue, inference_queue
            );
            tokenizer_workers.push_back(std::move(worker));
        }

        for (auto& worker : tokenizer_workers) {
            tokenizer_threads.emplace_back([&worker]() { worker->Run(); });
        }

        // 10. Start the pipeline
        std::cout << "\n--- PIPELINE LIVE: Processing " << total_rows_in_db << " rows ---" << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        // Blocks main thread
        spigot_worker(result, chunk_vault, free_queue, tokenizer_queue);

        // Wait for tokenizers to finish
        for (auto& t : tokenizer_threads) {
            t.join();
        }

        // Signal GPU workers to stop and wait
        inference_queue.shutdown();
        engine0.join();
        engine1.join();

        // Signal harvester to stop
        harvester_queue.shutdown();
        harvester_thread.join();

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
