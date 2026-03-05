#ifndef ACRELAB_GAUSS_HARVESTER_WORKER_HPP
#define ACRELAB_GAUSS_HARVESTER_WORKER_HPP

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>
#include <string>
#include "threadsafequeue.hpp"
#include "batch_job.hpp"
#include "arrow_parquet_writer.hpp"

namespace acrelab {

/**
 * @brief Harvester worker: Universal Multi-Vector Edition
 * Handles N-concats per row dynamically based on BatchJob metadata.
 */
class HarvesterWorker {
public:
    HarvesterWorker(ThreadSafeQueue<BatchJob*>& harvester_queue,
                    ThreadSafeQueue<BatchJob*>& free_queue,
                    std::atomic<int>& total_queries_processed,
                    const std::string& output_path,
                    int embedding_dim,
                    const std::vector<std::string>& column_names)
        : harvester_queue_(harvester_queue),
          free_queue_(free_queue),
          total_queries_processed_(total_queries_processed),
          output_path_(output_path),
          embedding_dim_(embedding_dim),
          column_names_(column_names) {}

    void Run() {
        std::cout << "Harvester: Initializing Multi-Vector S3 Writer..." << std::endl;

        // The writer is now initialized with the list of names (v1, v2, v3...)
        auto writer = CreateS3WriterFromEnv(output_path_, embedding_dim_, column_names_);
        writer->Open();

        BatchJob* job;
        auto start_time = std::chrono::high_resolution_clock::now();
        size_t batches_written = 0;

        while (harvester_queue_.wait_and_loop(job)) {
            if (job->valid_items > 0) {
                // Write the wide row (1 GID + N Vectors)
                writer->WriteBatchJob(job);

                total_queries_processed_ += job->valid_items;
                batches_written++;

                if (batches_written % 100 == 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::high_resolution_clock::now() - start_time).count();
                    double rate = total_queries_processed_.load() / (elapsed + 0.001);
                    std::cout << "Harvester: Wrote " << writer->GetTotalRowsWritten()
                              << " physical rows (" << rate << " rows/sec)" << std::endl;
                }

                ClearBatchJob(job);
            }
            free_queue_.push(job);
        }

        writer->Close();
        std::cout << "Harvester: Shutdown complete." << std::endl;
    }

private:
    /**
     * @brief Dynamic memory clearing
     * Scales based on num_concats to ensure no 'ghost' data remains.
     */
    void ClearBatchJob(BatchJob* job) {
        // total_sequences = (e.g. 256 rows * 6 concats)
        size_t total_sequences = job->max_batch_size * job->num_concats;
        size_t bytes_to_clear = total_sequences * job->seq_len * sizeof(int32_t);

        memset(job->pinned_input_ids, 0, bytes_to_clear);
        memset(job->pinned_attention_mask, 0, bytes_to_clear);
        
        // No need to clear pinned_token_type_ids if ByT5, 
        // but for safety if switching back to BERT:
        if (job->pinned_token_type_ids) {
            memset(job->pinned_token_type_ids, 0, bytes_to_clear);
        }
    }

    ThreadSafeQueue<BatchJob*>& harvester_queue_;
    ThreadSafeQueue<BatchJob*>& free_queue_;
    std::atomic<int>& total_queries_processed_;
    std::string output_path_;
    int embedding_dim_;
    std::vector<std::string> column_names_;
};

// --- HELPER FOR MAIN.CPP ---

inline std::thread CreateHarvesterThread(
    ThreadSafeQueue<BatchJob*>& harvester_queue,
    ThreadSafeQueue<BatchJob*>& free_queue,
    std::atomic<int>& total_queries_processed,
    const std::string& output_path,
    int embedding_dim,
    const std::vector<std::string>& column_names) {
    
    return std::thread([=, &harvester_queue, &free_queue, &total_queries_processed]() {
        HarvesterWorker harvester(harvester_queue, free_queue, total_queries_processed, 
                                 output_path, embedding_dim, column_names);
        harvester.Run();
    });
}

} // namespace acrelab

#endif
