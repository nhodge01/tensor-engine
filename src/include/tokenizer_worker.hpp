#ifndef ACRELAB_GAUSS_TOKENIZER_WORKER_HPP
#define ACRELAB_GAUSS_TOKENIZER_WORKER_HPP

#include <string_view>
#include <vector>
#include <cstring>
#include <thread>
#include <memory>
#include "tokenizer_base.hpp"
#include "byt5_tokenizer.hpp"
#include "tokenizer_wrapper.hpp"
#include "threadsafequeue.hpp"
#include "batch_job.hpp"
#include "config.hpp"

namespace acrelab {

/**
 * @brief Worker thread for tokenizing text batches.
 * Now agnostic to the underlying tokenizer implementation (ByT5 or HuggingFace).
 */
class TokenizerWorker {
public:
    TokenizerWorker(int thread_id,
                    ITokenizer* tokenizer,
                    ThreadSafeQueue<BatchJob*>& input_queue,
                    ThreadSafeQueue<BatchJob*>& output_queue)
        : thread_id_(thread_id),
          tokenizer_(tokenizer),
          input_queue_(input_queue),
          output_queue_(output_queue) {}

    /**
     * @brief Main processing loop
     */
    void Run() {
        BatchJob* job;
        while (input_queue_.wait_and_loop(job)) {
            ProcessBatch(job);
            output_queue_.push(job);
        }
    }

    /**
     * @brief Process a single batch using the polymorphic EncodeToBuffer.
     * This handles the direct write from DuckDB string_views to CUDA Pinned Memory.
     */
    void ProcessBatch(BatchJob* job) {
        // CRITICAL FIX: Process ALL vectors (valid_items * num_concats)
        // Each physical row has num_concats vectors (e.g., 5 text perspectives)
        const size_t total_sequences = job->valid_items * job->num_concats;

        for (size_t i = 0; i < total_sequences; ++i) {
            // EncodeToBuffer handles empty strings, truncation, and padding internally
            tokenizer_->EncodeToBuffer(
                job->raw_texts[i],  // Now correctly iterates through ALL vectors
                &job->pinned_input_ids[i * job->seq_len],
                &job->pinned_attention_mask[i * job->seq_len],
                job->seq_len
            );
        }
    }

private:
    int thread_id_;
    ITokenizer* tokenizer_;
    ThreadSafeQueue<BatchJob*>& input_queue_;
    ThreadSafeQueue<BatchJob*>& output_queue_;
};

/**
 * @brief Manages a pool of tokenizer threads and the lifecycle of the Tokenizer instance.
 */
class TokenizerWorkerPool {
public:
    TokenizerWorkerPool(int num_threads,
                        const ModelConfig& config,
                        ThreadSafeQueue<BatchJob*>& input_queue,
                        ThreadSafeQueue<BatchJob*>& output_queue)
        : input_queue_(input_queue),
          output_queue_(output_queue) {

        // --- THE GRACEFUL SWITCH ---
        if (config.tokenizer_type == "byt5") {
            tokenizer_ = std::make_unique<ByT5Tokenizer>();
            std::cout << "[Pool] Initialized High-Performance ByT5 (C++)" << std::endl;
        } else {
            auto wrapper = TokenizerWrapper::FromFile(config.vocab_path);
            if (!wrapper) {
                throw std::runtime_error("Failed to load HuggingFace tokenizer: " + config.vocab_path);
            }
            tokenizer_ = std::move(wrapper);
            std::cout << "[Pool] Initialized HuggingFace/Rust Tokenizer" << std::endl;
        }

        // Create workers using the common interface
        workers_.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            workers_.emplace_back(
                i, tokenizer_.get(), input_queue_, output_queue_
            );
        }
    }

    void Start() {
        threads_.reserve(workers_.size());
        for (auto& worker : workers_) {
            threads_.emplace_back([&worker]() { worker.Run(); });
        }
    }

    void Join() {
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    size_t NumWorkers() const { return workers_.size(); }

private:
    std::unique_ptr<ITokenizer> tokenizer_; // Polymorphic pointer
    std::vector<TokenizerWorker> workers_;
    std::vector<std::thread> threads_;
    ThreadSafeQueue<BatchJob*>& input_queue_;
    ThreadSafeQueue<BatchJob*>& output_queue_;
};

} // namespace acrelab

#endif // ACRELAB_GAUSS_TOKENIZER_WORKER_HPP
