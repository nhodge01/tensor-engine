#ifndef ACRELAB_GAUSS_HARVESTER_WORKER_HPP
#define ACRELAB_GAUSS_HARVESTER_WORKER_HPP

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include "threadsafequeue.hpp"
#include "batch_job.hpp"
#include "arrow_parquet_writer.hpp"

namespace acrelab {

/**
 * @brief Harvester worker that collects GPU output and writes to S3/MinIO
 *
 * This worker:
 * 1. Receives completed BatchJobs from GPU workers
 * 2. Writes embeddings to Parquet file on S3/MinIO (zero-copy)
 * 3. Clears the BatchJob memory
 * 4. Returns the empty BatchJob to free_queue for reuse
 *
 * Performance: ~700 MB/s to object storage with Snappy compression
 */
class HarvesterWorker {
public:
    /**
     * @brief Construct harvester with S3 output
     */
    HarvesterWorker(ThreadSafeQueue<BatchJob*>& harvester_queue,
                    ThreadSafeQueue<BatchJob*>& free_queue,
                    std::atomic<int>& total_queries_processed,
                    const std::string& output_path,
                    int embedding_dim = 384)
        : harvester_queue_(harvester_queue),
          free_queue_(free_queue),
          total_queries_processed_(total_queries_processed),
          output_path_(output_path),
          embedding_dim_(embedding_dim) {}

    /**
     * @brief Main harvester loop - writes to S3
     */
    void Run() {
        std::cout << "Harvester: Initializing S3/MinIO writer..." << std::endl;

        // Create the Arrow/Parquet writer
        auto writer = CreateS3WriterFromEnv(output_path_, embedding_dim_);
        writer->Open();

        std::cout << "Harvester: S3 stream open. Ready to catch embeddings!" << std::endl;

        BatchJob* job;
        auto start_time = std::chrono::high_resolution_clock::now();
        size_t batches_written = 0;

        // Main processing loop
        while (harvester_queue_.wait_and_loop(job)) {
            if (job->valid_items > 0) {
                // Write to S3 (zero-copy)
                writer->WriteBatchJob(job);

                // Update counters
                total_queries_processed_ += job->valid_items;
                batches_written++;

                // Print progress every 100 batches
                if (batches_written % 100 == 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - start_time).count();
                    double rate = total_queries_processed_.load() / static_cast<double>(elapsed);
                    std::cout << "Harvester: Wrote " << writer->GetTotalRowsWritten()
                              << " embeddings (" << rate << " rows/sec)" << std::endl;
                }

                // Clear the pinned memory for reuse
                ClearBatchJob(job);
            }

            // Return empty bucket to the spigot
            free_queue_.push(job);
        }

        // Close the Parquet file
        writer->Close();

        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();

        std::cout << "Harvester: Complete. Wrote " << writer->GetTotalRowsWritten()
                  << " embeddings in " << total_time << " seconds ("
                  << writer->GetTotalRowsWritten() / total_time << " rows/sec)" << std::endl;
    }

    /**
     * @brief Alternative run method without S3 (memory only)
     */
    void RunMemoryOnly() {
        std::cout << "Harvester: Running in memory-only mode (no S3 output)" << std::endl;

        BatchJob* job;
        while (harvester_queue_.wait_and_loop(job)) {
            total_queries_processed_ += job->valid_items;

            // Clear the memory
            ClearBatchJob(job);

            // Return to free queue
            free_queue_.push(job);
        }

        std::cout << "Harvester: Complete. Processed " << total_queries_processed_.load()
                  << " embeddings (memory only)" << std::endl;
    }

private:
    /**
     * @brief Clear the BatchJob memory for reuse
     */
    void ClearBatchJob(BatchJob* job) {
        // Clear the input buffers
        memset(job->pinned_input_ids, 0,
               job->max_batch_size * job->seq_len * sizeof(int32_t));
        memset(job->pinned_attention_mask, 0,
               job->max_batch_size * job->seq_len * sizeof(int32_t));

        // Note: We don't clear output embeddings as they'll be overwritten
        // Note: We don't clear row_ids/raw_texts as spigot will overwrite them
    }

    ThreadSafeQueue<BatchJob*>& harvester_queue_;
    ThreadSafeQueue<BatchJob*>& free_queue_;
    std::atomic<int>& total_queries_processed_;
    std::string output_path_;
    int embedding_dim_;
};

// Forward declarations - implementations in harvester_worker.cpp
std::thread CreateHarvesterThread(
    ThreadSafeQueue<BatchJob*>& harvester_queue,
    ThreadSafeQueue<BatchJob*>& free_queue,
    std::atomic<int>& total_queries_processed,
    const std::string& output_path = "",
    bool use_s3 = true);

void harvester_worker(ThreadSafeQueue<BatchJob*>& harvester_queue,
                     ThreadSafeQueue<BatchJob*>& free_queue,
                     std::atomic<int>& total_queries_processed);

void harvester_worker_s3(ThreadSafeQueue<BatchJob*>& harvester_queue,
                        ThreadSafeQueue<BatchJob*>& free_queue,
                        std::atomic<int>& total_queries_processed,
                        const std::string& output_path);

} // namespace acrelab

#endif // ACRELAB_GAUSS_HARVESTER_WORKER_HPP