#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "tokenizers_cpp.h"
#include "include/threadsafequeue.hpp"
#include "include/batch_job.hpp"



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

__global__ void mean_pooling_kernel(
    const float* hidden_states, const int* attention_mask, float* pooled_output,
    int batch_size, int seq_len, int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_dim;

    if (idx < total_elements) {
        int b = idx / hidden_dim;
        int d = idx % hidden_dim;

        float sum = 0.0f;
        float mask_sum = 0.0f;

        for (int s = 0; s < seq_len; ++s) {
            int mask_val = attention_mask[b * seq_len + s];
            sum += hidden_states[(b * seq_len + s) * hidden_dim + d] * mask_val;
            mask_sum += mask_val;
        }

        pooled_output[idx] = mask_sum > 0.0f ? (sum / mask_sum) : 0.0f;
    }
}

// Global counters for both GPUs
std::atomic<int> gpu0_iterations{0};
std::atomic<int> gpu1_iterations{0};
std::atomic<bool> keep_running{true};



void tokenizer_worker(int thread_id, tokenizers::Tokenizer* tokenizer,
                    ThreadSafeQueue<BatchJob*>& free_queue,
                    ThreadSafeQueue<BatchJob*>& inference_queue) {
    BatchJob* job;

    while (free_queue.wait_and_loop(job)) {
        for (int i=0; i < job->valid_items; ++i) {
            std::vector<int> ids = tokenizer->Encode(job->raw_texts[i]);

            int write_len = (ids.size() < job->seq_len) ? ids.size() : job->seq_len;
            int offset = i * job->seq_len;

            for (int j = 0; j < write_len; ++j) {
                job->pinned_input_ids[offset + j] = ids[j];
                job->pinned_attention_mask[offset + j] = 1;
            }
        }

        inference_queue.push(job);
    }    
}


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

    int threads_per_block = 256;
    int num_blocks = (pooled_floats + threads_per_block - 1) / threads_per_block;

    mean_pooling_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        (float*)d_out, (int*)d_mask, d_pooled, batch_size, seq_len, hidden_dim
    );

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

    std::cout << "GPU " << gpu_id << ": Ready and waiting for Tokenizer!" << std::endl;

    BatchJob* job;
    
    // --- THE PRODUCTION PIPELINE LOOP ---
    // Thread sleeps here using zero CPU until inference_queue.push() is called
    while (inference_queue.wait_and_loop(job)) {
        
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
        
        // 5. Send the finished pointers to the Harvester
        harvester_queue.push(job);
    }

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_ids); cudaFree(d_mask); cudaFree(d_type); cudaFree(d_out); cudaFree(d_pooled);
    
    std::cout << "GPU " << gpu_id << ": Shutting down smoothly." << std::endl;
}


void harvester_worker(ThreadSafeQueue<BatchJob*>& harvester_queue,
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



int main() {
    std::cout << "=== INITIALIZING DUAL-GPU MPMC QUEUE PIPELINE ===" << std::endl;

    // 1. Load the raw bytes (so each GPU can build its own context)
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(load_file_contents("onnx/tokenizer.json"));
    auto engine_data = load_engine_binary("engines/model_batch256_len32.engine");

    if (!tokenizer || engine_data.empty()) {
        std::cerr << "Failed to load tokenizer or engine!" << std::endl;
        return 1;
    }

    int batch_size = 256;
    int seq_len = 32;
    int hidden_dim = 384;
    int num_jobs_in_pool = 200; // The size of our rotating buffer

    // 2. Initialize the Queues
    ThreadSafeQueue<BatchJob*> free_queue;
    ThreadSafeQueue<BatchJob*> inference_queue;
    ThreadSafeQueue<BatchJob*> harvester_queue;
    std::atomic<int> total_queries_processed{0};

    // 3. Pre-Allocate the Pinned Memory Pool (The Depot)
    std::cout << "Allocating " << num_jobs_in_pool << " pinned memory structs..." << std::endl;
    std::vector<BatchJob*> job_pool;
    
    for (int i = 0; i < num_jobs_in_pool; ++i) {
        auto job = new BatchJob(i, batch_size, seq_len, hidden_dim);
        
        // For this synthetic benchmark, we pretend DuckDB just filled the raw texts
        job->valid_items = batch_size;
        for(int j = 0; j < batch_size; ++j) {
            job->raw_texts[j] = "JOHN DOE 1234 FAKE STREET ROGERS AR 72758";
            job->row_ids[j] = j; 
        }
        
        job_pool.push_back(job);
        free_queue.push(job); // Put it in the queue for the tokenizers to grab
    }

    // 4. Launch the Threads!
    int num_tokenizer_threads = 8; // The sweet spot we found earlier
    std::vector<std::thread> tokenizer_threads;
    std::vector<std::thread> gpu_threads;

    std::cout << "Spawning " << num_tokenizer_threads << " Tokenizer threads..." << std::endl;
    for (int i = 0; i < num_tokenizer_threads; ++i) {
        tokenizer_threads.emplace_back(tokenizer_worker, i, tokenizer.get(), std::ref(free_queue), std::ref(inference_queue));
    }

    std::cout << "Spawning Harvester thread..." << std::endl;
    std::thread harvester_thread(harvester_worker, std::ref(harvester_queue), std::ref(free_queue), std::ref(total_queries_processed));

    std::cout << "Spawning GPU workers..." << std::endl;
    gpu_threads.emplace_back(gpu_worker, 0, std::ref(engine_data), std::ref(inference_queue), std::ref(harvester_queue));
    gpu_threads.emplace_back(gpu_worker, 1, std::ref(engine_data), std::ref(inference_queue), std::ref(harvester_queue));

    // 5. Start the Stopwatch
    std::cout << "\n--- PIPELINE LIVE: Running 10-Second Benchmark ---" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // The Main thread just chills here while the Threadripper and 4090s go to war
    std::this_thread::sleep_for(std::chrono::seconds(10));

    // 6. Graceful Shutdown
    std::cout << "Time's up! Broadcasting shutdown signal to all queues..." << std::endl;
    
    // Dropping the poison pill wakes up any sleeping threads and tells them to exit their while loops
    free_queue.shutdown();
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

    std::cout << "\n=== END-TO-END PIPELINE RESULTS ===" << std::endl;
    std::cout << "Duration:        " << std::fixed << std::setprecision(2) << actual_time << " seconds" << std::endl;
    std::cout << "Total Embedded:  " << total_queries_processed.load() << " strings" << std::endl;
    std::cout << "Throughput:      " << std::fixed << std::setprecision(0) << qps << " RPS" << std::endl;
    std::cout << "===================================\n" << std::endl;

    // 9. Clean up the Depot
    for (auto job : job_pool) {
        delete job;
    }

    return 0;
}
