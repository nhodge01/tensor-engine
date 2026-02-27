#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <thread>
#include <atomic>
#include <string_view>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "tokenizer_wrapper.hpp"
#include "tokenizer_worker.hpp"
#include "cuda_kernels.cuh"
#include "harvester_worker_fwd.hpp"
#include "threadsafequeue.hpp"
#include "batch_job.hpp"
#include <duckdb.hpp>
#include "lake.h"
#include "env_loader.hpp"



using namespace nvinfer1;
using namespace std::chrono;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) std::cout << "[TRT ERROR] " << msg << std::endl;
    }
} gLogger;

std::string load_file_contents(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

std::vector<char> load_engine_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

// CUDA kernels moved to include/acrelab-gauss/cuda_kernels.cuh

// Global counters for both GPUs
std::atomic<int> gpu0_iterations{0};
std::atomic<int> gpu1_iterations{0};
std::atomic<bool> keep_running{true};

// Timing metrics (in microseconds for clean atomic operations)
std::atomic<uint64_t> total_gpu_compute_time_us{0};
std::atomic<int> total_gpu_batches{0};



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

        for (size_t i = 0; i < chunk->size(); i++) {
            if (!job) {
                if (!free_queue.wait_and_loop(job)) return;
                job_idx = 0;
            }

            job->row_ids[job_idx] = gid_validity.RowIsValid(i) ? gid_data[i] : 0;

            // ZERO-COPY: Just store pointers!
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

        // KEEP THE MEMORY ALIVE!
        // Move the chunk into the vault instead of letting it die
        chunk_vault.push_back(std::move(chunk));
    }

    if (job && job_idx > 0) {
        job->valid_items = job_idx;
        tokenizer_queue.push(job);
    }

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "Spigot: Extracted " << total_extracted << " rows in " << elapsed
              << " ms (" << (total_extracted * 1000 / elapsed) << " RPS)" << std::endl;
}

// Tokenizer worker function moved to include/acrelab-gauss/tokenizer_worker.hpp


// GPU worker function
void gpu_worker(int gpu_id, const std::vector<char>& engine_data,
                ThreadSafeQueue<BatchJob*>& inference_queue,
                ThreadSafeQueue<BatchJob*>& harvester_queue) {

    cudaSetDevice(gpu_id);
    std::cout << "GPU " << gpu_id << ": Initializing..." << std::endl;

    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    auto engine = std::unique_ptr<ICudaEngine>(
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
    );

    if (!engine) {
        std::cerr << "GPU " << gpu_id << ": Failed to load engine!" << std::endl;
        return;
    }

    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    
    // We assume all BatchJobs passing through here have the same max dimensions
    int batch_size = 256; 
    int seq_len = 32;
    int hidden_dim = 384;
    int total_tokens = batch_size * seq_len;
    int pooled_floats = batch_size * hidden_dim;

    void *d_ids, *d_mask, *d_type, *d_out;
    float *d_pooled; // Notice we removed h_pooled! The destination is now inside the BatchJob

    cudaMalloc(&d_ids, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_mask, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_type, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_out, total_tokens * hidden_dim * sizeof(float));
    cudaMalloc(&d_pooled, pooled_floats * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    context->setInputShape("input_ids", Dims2{batch_size, seq_len});
    context->setInputShape("attention_mask", Dims2{batch_size, seq_len});
    context->setInputShape("token_type_ids", Dims2{batch_size, seq_len});

    context->setTensorAddress("input_ids", d_ids);
    context->setTensorAddress("attention_mask", d_mask);
    context->setTensorAddress("token_type_ids", d_type);
    context->setTensorAddress("last_hidden_state", d_out);

    // 1. Warmup execution to force TensorRT to allocate all memory (OUTSIDE the graph)
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    std::cout << "GPU " << gpu_id << ": Creating COMPUTE-ONLY CUDA graph..." << std::endl;

    // 2. Capture the graph using THREAD LOCAL mode
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal); // <--- THE FIX

    context->enqueueV3(stream);

    // Use the kernel from the header with proper namespace
    acrelab::cuda::launch_mean_pooling(
        (float*)d_out, (int*)d_mask, d_pooled,
        batch_size, seq_len, hidden_dim, stream
    );

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    std::cout << "GPU " << gpu_id << ": Ready and waiting for Tokenizer!" << std::endl;

    BatchJob* job;
    uint64_t gpu_time_us = 0;
    int batch_count = 0;

    // --- THE PRODUCTION PIPELINE LOOP ---
    // Thread sleeps here using zero CPU until inference_queue.push() is called
    while (inference_queue.wait_and_loop(job)) {
        auto gpu_start = std::chrono::high_resolution_clock::now();

        // 1. Dynamic H2D Copy (Pulling straight from the Queue's Payload)
        cudaMemcpyAsync(d_ids, job->pinned_input_ids, total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mask, job->pinned_attention_mask, total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_type, job->pinned_token_type_ids, total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

        // 2. Launch the pre-recorded math
        cudaGraphLaunch(graphExec, stream);

        // 3. Dynamic D2H Copy (Writing straight back into the Queue's Payload)
        cudaMemcpyAsync(job->pinned_embeddings, d_pooled, pooled_floats * sizeof(float), cudaMemcpyDeviceToHost, stream);

        // 4. Wait for the PCIe bus to finish writing
        cudaStreamSynchronize(stream);

        auto gpu_end = std::chrono::high_resolution_clock::now();
        gpu_time_us += std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count();
        batch_count++;

        // 5. Send the finished pointers to the Harvester
        harvester_queue.push(job);
    }

    // Update global timing metrics - fetch_add works flawlessly with uint64_t!
    total_gpu_compute_time_us.fetch_add(gpu_time_us);
    total_gpu_batches.fetch_add(batch_count);

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_ids); cudaFree(d_mask); cudaFree(d_type); cudaFree(d_out); cudaFree(d_pooled);
    
    std::cout << "GPU " << gpu_id << ": Shutting down smoothly." << std::endl;
}


// Original memory-only harvester (for benchmarking without I/O)
void harvester_worker_memory_only(ThreadSafeQueue<BatchJob*>& harvester_queue,
                                  ThreadSafeQueue<BatchJob*>& free_queue,
                                  std::atomic<int>& total_queries_processed) {
    BatchJob* job;

    while (harvester_queue.wait_and_loop(job)) {
        total_queries_processed += job->valid_items;

        memset(job->pinned_input_ids, 0, job->max_batch_size * job->seq_len * sizeof(int32_t));
        memset(job->pinned_attention_mask, 0, job->max_batch_size * job->seq_len * sizeof(int32_t));

        free_queue.push(job);
    }
}

// S3-enabled harvester worker using the new header
void harvester_worker(ThreadSafeQueue<BatchJob*>& harvester_queue,
                      ThreadSafeQueue<BatchJob*>& free_queue,
                      std::atomic<int>& total_queries_processed) {

    // Check if we should write to S3
    const char* enable_s3 = std::getenv("ENABLE_S3_OUTPUT");

    // Hardcoded output path for now
    const std::string output_path = "REDACTED_BUCKET/REDACTED_PATH";

    if (enable_s3 && std::string(enable_s3) == "1") {
        // Use S3 output
        std::cout << "Harvester: S3 output enabled, writing to " << output_path << std::endl;
        acrelab::harvester_worker_s3(harvester_queue, free_queue, total_queries_processed, output_path);
    } else {
        // Fallback to memory-only mode
        std::cout << "Harvester: Running in memory-only mode (set ENABLE_S3_OUTPUT=1 for S3)" << std::endl;
        harvester_worker_memory_only(harvester_queue, free_queue, total_queries_processed);
    }
}



int main() {
    // Load environment from .env.ducklake file
    acrelab::EnvLoader::LoadEnvFile(".env.ducklake", true);

    auto program_start = std::chrono::high_resolution_clock::now();

    std::cout << "=== ZERO-COPY DUAL-GPU PIPELINE ===" << std::endl;
    std::cout << "4M+ RPS DuckLake -> 8 Tokenizers -> Dual 4090s -> 330k+ RPS\n" << std::endl;

    // 1. Initialize DuckLake connection
    std::cout << "Connecting to DuckLake..." << std::endl;
    duckdb::DuckDB db(":memory:");
    duckdb::Connection conn(db);

    // Configure DuckDB
    conn.Query("SET memory_limit='8GB';");
    conn.Query("SET threads=4;");
    conn.Query("INSTALL ducklake; INSTALL httpfs; INSTALL postgres;");
    conn.Query("LOAD ducklake; LOAD httpfs; LOAD postgres;");

    // Setup credentials
    std::string minio_endpoint = std::getenv("MINIO_ENDPOINT") ? std::getenv("MINIO_ENDPOINT") : "";
    std::string minio_access_key = std::getenv("MINIO_ACCESS_KEY") ? std::getenv("MINIO_ACCESS_KEY") : "";
    std::string minio_secret_key = std::getenv("MINIO_SECRET_KEY") ? std::getenv("MINIO_SECRET_KEY") : "";
    std::string db_host = std::getenv("DB_HOST") ? std::getenv("DB_HOST") : "";
    std::string db_name = std::getenv("DB_NAME") ? std::getenv("DB_NAME") : "";
    std::string db_user = std::getenv("DB_USER") ? std::getenv("DB_USER") : "";
    std::string db_pass = std::getenv("DB_PASS") ? std::getenv("DB_PASS") : "";

    // Create secrets
    conn.Query("CREATE OR REPLACE SECRET minio_secret (TYPE s3, KEY_ID '" + minio_access_key +
               "', SECRET '" + minio_secret_key + "', ENDPOINT '" + minio_endpoint +
               "', URL_STYLE 'path', USE_SSL false);");

    conn.Query("CREATE OR REPLACE SECRET acrelake (TYPE ducklake, "
               "METADATA_PATH 'postgres:host=" + db_host + " port=5432 dbname=" + db_name +
               " user=" + db_user + " password=" + db_pass + " sslmode=require', "
               "DATA_PATH 's3://REDACTED_BUCKET');");

    conn.Query("ATTACH 'ducklake:acrelake' AS lake;");

    // 2. Load tokenizer and engine
    auto tokenizer = acrelab::TokenizerWrapper::FromFile("onnx/tokenizer.json");
    auto engine_data = load_engine_binary("engines/model_batch256_len32.engine");

    if (!tokenizer || engine_data.empty()) {
        std::cerr << "Failed to load tokenizer or engine!" << std::endl;
        return 1;
    }

    int batch_size = 256;
    int seq_len = 32;
    int hidden_dim = 384;
    int num_jobs_in_pool = 200; // The size of our rotating buffer

    // 3. Initialize the Queues
    ThreadSafeQueue<BatchJob*> free_queue;
    ThreadSafeQueue<BatchJob*> tokenizer_queue;  // New queue for spigot->tokenizer
    ThreadSafeQueue<BatchJob*> inference_queue;
    ThreadSafeQueue<BatchJob*> harvester_queue;
    std::atomic<int> total_queries_processed{0};

    // 3. Pre-Allocate the Pinned Memory Pool (The Depot)
    std::cout << "Allocating " << num_jobs_in_pool << " pinned memory structs..." << std::endl;
    std::vector<BatchJob*> job_pool;
    
    for (int i = 0; i < num_jobs_in_pool; ++i) {
        auto job = new BatchJob(i, batch_size, seq_len, hidden_dim);
        job_pool.push_back(job);
        free_queue.push(job);
    }

    // 4. CRITICAL: Materialize the query BEFORE starting threads
    std::cout << "\nMaterializing DuckLake query..." << std::endl;
    auto query_start = std::chrono::high_resolution_clock::now();

    auto result = conn.Query(R"(
        WITH state_filter AS (
            -- STEP 1: Scan ONLY the state column to find the right rows
            SELECT gid,
                   standard_owner_1_first_last_name,
                   standard_owner_2_first_last_name,
                   standard_mail_address
            FROM lake.desoto_jrrtolkein.consolidated_parcel
            WHERE state IN ('RI', 'AR')
        )
        -- STEP 2: Only do the heavy string concatenation on the 2.5M surviving rows
        SELECT
            CAST(gid as BIGINT) AS gid,
            concat_ws(' ',
                standard_owner_1_first_last_name,
                standard_owner_2_first_last_name,
                standard_mail_address
            ) AS text
        FROM state_filter
        WHERE (standard_owner_1_first_last_name IS NOT NULL
            OR standard_owner_2_first_last_name IS NOT NULL
            OR standard_mail_address IS NOT NULL)
    )");

    double query_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - query_start).count();
    std::cout << "Query materialized in " << query_ms << " ms\n" << std::endl;

    if (result->HasError()) {
        std::cerr << "Query failed: " << result->GetError() << std::endl;
        return 1;
    }

    // Get the total row count from the materialized result
    size_t total_rows_in_db = result->RowCount();
    std::cout << "Total rows to process: " << total_rows_in_db << std::endl;

    // Chunk vault - keeps DuckDB memory alive for the entire pipeline duration
    std::vector<duckdb::unique_ptr<duckdb::DataChunk>> chunk_vault;

    // 5. Launch the Threads!
    std::cout << "\nLAUNCHING ZERO-COPY PIPELINE!" << std::endl;

    // Spigot thread - ZERO-COPY extraction at 4M+ RPS
    std::thread spigot_thread(spigot_worker, std::ref(result), std::ref(chunk_vault),
                              std::ref(free_queue), std::ref(tokenizer_queue));

    // Tokenizer threads - Convert string_views to tokens
    int num_tokenizer_threads = 8;
    std::vector<std::thread> tokenizer_threads;
    std::cout << "Spawning " << num_tokenizer_threads << " Tokenizer threads..." << std::endl;

    // Create tokenizer workers using the new wrapper
    std::vector<std::unique_ptr<acrelab::TokenizerWorker>> tokenizer_workers;
    for (int i = 0; i < num_tokenizer_threads; ++i) {
        tokenizer_workers.push_back(std::make_unique<acrelab::TokenizerWorker>(
            i, tokenizer.get(), tokenizer_queue, inference_queue
        ));
    }

    // Launch threads
    for (auto& worker : tokenizer_workers) {
        tokenizer_threads.emplace_back([&worker]() { worker->Run(); });
    }

    std::cout << "Spawning Harvester thread..." << std::endl;
    std::thread harvester_thread(harvester_worker, std::ref(harvester_queue), std::ref(free_queue), std::ref(total_queries_processed));

    std::cout << "Spawning GPU workers..." << std::endl;
    std::vector<std::thread> gpu_threads;
    gpu_threads.emplace_back(gpu_worker, 0, std::ref(engine_data), std::ref(inference_queue), std::ref(harvester_queue));
    gpu_threads.emplace_back(gpu_worker, 1, std::ref(engine_data), std::ref(inference_queue), std::ref(harvester_queue));

    // 5. Start the Stopwatch
    std::cout << "\n--- PIPELINE LIVE: Processing " << total_rows_in_db << " rows ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Progress tracking variables
    auto last_print_time = start_time;
    size_t last_count = 0;

    // THE NEW LOOP: Wait until the Harvester has processed every single row
    while (total_queries_processed.load() < total_rows_in_db) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Print progress every second
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count() >= 1) {
            size_t current_count = total_queries_processed.load();
            size_t rows_per_sec = current_count - last_count;
            double progress = (current_count * 100.0) / total_rows_in_db;

            std::cout << "\rProgress: " << current_count << "/" << total_rows_in_db
                      << " (" << std::fixed << std::setprecision(1) << progress << "%) "
                      << "- Current RPS: " << rows_per_sec << " " << std::flush;

            last_print_time = now;
            last_count = current_count;
        }
    }

    std::cout << "\nAll rows processed! Shutting down pipeline..." << std::endl;

    // Wait for spigot to finish cleanly
    spigot_thread.join();

    // Then shutdown the queues in order
    tokenizer_queue.shutdown();
    inference_queue.shutdown();
    harvester_queue.shutdown();

    // 7. Wait for everyone to pack up and go home
    for (auto& t : tokenizer_threads) t.join();
    for (auto& t : gpu_threads) t.join();
    harvester_thread.join();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // 8. The Final Math
    double actual_time = elapsed_seconds.count();
    double qps = total_queries_processed.load() / actual_time;

    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         END-TO-END PIPELINE RESULTS                        ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ Total Duration:     " << std::fixed << std::setprecision(2) << std::setw(8)
              << actual_time << " seconds                      ║" << std::endl;
    std::cout << "║ Total Embedded:     " << std::setw(8) << total_queries_processed.load()
              << " strings                      ║" << std::endl;
    std::cout << "║ Overall Throughput: " << std::fixed << std::setprecision(0) << std::setw(8)
              << qps << " RPS                          ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║                 GPU COMPUTE METRICS                        ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════╣" << std::endl;

    double total_gpu_compute_ms = total_gpu_compute_time_us.load() / 1000.0;
    double avg_gpu_time_ms = total_gpu_compute_ms / total_gpu_batches.load();
    double gpu_utilization = (total_gpu_compute_ms / 2.0) / (actual_time * 1000) * 100; // 2 GPUs

    std::cout << "║ Total GPU Batches:  " << std::setw(8) << total_gpu_batches.load()
              << "                              ║" << std::endl;
    std::cout << "║ Avg GPU Time/Batch: " << std::fixed << std::setprecision(2) << std::setw(8)
              << avg_gpu_time_ms << " ms                         ║" << std::endl;
    std::cout << "║ GPU Utilization:    " << std::fixed << std::setprecision(1) << std::setw(7)
              << gpu_utilization << "%                            ║" << std::endl;
    std::cout << "║ GPU Compute RPS:    " << std::fixed << std::setprecision(0) << std::setw(8)
              << (total_gpu_batches.load() * 256) / (total_gpu_compute_ms / 2000.0)
              << "                              ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;

    auto program_end = std::chrono::high_resolution_clock::now();
    double total_program_time = std::chrono::duration<double>(program_end - program_start).count();

    std::cout << "\nTotal Program Runtime: " << std::fixed << std::setprecision(2)
              << total_program_time << " seconds (including " << std::fixed << std::setprecision(1)
              << query_ms/1000.0 << "s query materialization)\n" << std::endl;

    // 9. Clean up the Depot
    for (auto job : job_pool) {
        delete job;
    }

    return 0;
}
