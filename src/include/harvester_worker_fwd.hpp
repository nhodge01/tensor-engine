#ifndef ACRELAB_GAUSS_HARVESTER_WORKER_FWD_HPP
#define ACRELAB_GAUSS_HARVESTER_WORKER_FWD_HPP

#include <atomic>
#include <string>
#include <thread>
#include "threadsafequeue.hpp"
#include "batch_job.hpp"

namespace acrelab {

// Forward declarations only - implementations in harvester_worker.cpp
void harvester_worker_s3(ThreadSafeQueue<BatchJob*>& harvester_queue,
                        ThreadSafeQueue<BatchJob*>& free_queue,
                        std::atomic<int>& total_queries_processed,
                        const std::string& output_path,
                        int embedding_dim = 384);

void harvester_worker(ThreadSafeQueue<BatchJob*>& harvester_queue,
                     ThreadSafeQueue<BatchJob*>& free_queue,
                     std::atomic<int>& total_queries_processed,
                     int embedding_dim = 384);

std::thread CreateHarvesterThread(
    ThreadSafeQueue<BatchJob*>& harvester_queue,
    ThreadSafeQueue<BatchJob*>& free_queue,
    std::atomic<int>& total_queries_processed,
    const std::string& output_path = "",
    bool use_s3 = true,
    int embedding_dim = 384);

} // namespace acrelab

#endif // ACRELAB_GAUSS_HARVESTER_WORKER_FWD_HPP