#include "acrelab-gauss/harvester_worker.hpp"

namespace acrelab {

// Provide non-inline implementations for functions called from CUDA code
void harvester_worker_s3(ThreadSafeQueue<BatchJob*>& harvester_queue,
                        ThreadSafeQueue<BatchJob*>& free_queue,
                        std::atomic<int>& total_queries_processed,
                        const std::string& output_path) {
    HarvesterWorker worker(harvester_queue, free_queue, total_queries_processed, output_path);
    worker.Run();
}

void harvester_worker(ThreadSafeQueue<BatchJob*>& harvester_queue,
                     ThreadSafeQueue<BatchJob*>& free_queue,
                     std::atomic<int>& total_queries_processed) {
    HarvesterWorker worker(harvester_queue, free_queue, total_queries_processed, "");
    worker.RunMemoryOnly();
}

std::thread CreateHarvesterThread(
    ThreadSafeQueue<BatchJob*>& harvester_queue,
    ThreadSafeQueue<BatchJob*>& free_queue,
    std::atomic<int>& total_queries_processed,
    const std::string& output_path,
    bool use_s3) {
    if (use_s3 && !output_path.empty()) {
        return std::thread([&harvester_queue, &free_queue, &total_queries_processed, output_path]() {
            HarvesterWorker harvester(harvester_queue, free_queue, total_queries_processed, output_path);
            harvester.Run();
        });
    } else {
        return std::thread([&harvester_queue, &free_queue, &total_queries_processed]() {
            HarvesterWorker harvester(harvester_queue, free_queue, total_queries_processed, "");
            harvester.RunMemoryOnly();
        });
    }
}

} // namespace acrelab