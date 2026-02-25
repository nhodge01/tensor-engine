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

// GPU worker function
void gpu_worker(int gpu_id, const std::vector<char>& engine_data,
                const int* pinned_ids, const int* pinned_mask, const int* pinned_type,
                int batch_size, int seq_len, int hidden_dim,
                std::atomic<int>& iteration_counter) {

    // Set this thread to use the specified GPU
    cudaSetDevice(gpu_id);

    std::cout << "GPU " << gpu_id << ": Initializing..." << std::endl;

    // Create TensorRT runtime and engine for this GPU
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    auto engine = std::unique_ptr<ICudaEngine>(
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
    );

    if (!engine) {
        std::cerr << "GPU " << gpu_id << ": Failed to load engine!" << std::endl;
        return;
    }

    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        std::cerr << "GPU " << gpu_id << ": Failed to create context!" << std::endl;
        return;
    }

    int total_tokens = batch_size * seq_len;
    int pooled_floats = batch_size * hidden_dim;

    // Allocate device memory on this GPU
    void *d_ids, *d_mask, *d_type, *d_out;
    float *d_pooled, *h_pooled;

    cudaMalloc(&d_ids, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_mask, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_type, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_out, total_tokens * hidden_dim * sizeof(float));
    cudaMalloc(&d_pooled, pooled_floats * sizeof(float));
    cudaHostAlloc(&h_pooled, pooled_floats * sizeof(float), cudaHostAllocDefault);

    // Create stream for this GPU
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Set up context
    context->setInputShape("input_ids", Dims2{batch_size, seq_len});
    context->setInputShape("attention_mask", Dims2{batch_size, seq_len});
    context->setInputShape("token_type_ids", Dims2{batch_size, seq_len});

    context->setTensorAddress("input_ids", d_ids);
    context->setTensorAddress("attention_mask", d_mask);
    context->setTensorAddress("token_type_ids", d_type);
    context->setTensorAddress("last_hidden_state", d_out);

    // Copy input data to GPU
    cudaMemcpyAsync(d_ids, pinned_ids, total_tokens * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mask, pinned_mask, total_tokens * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_type, pinned_type, total_tokens * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Initial enqueueV3 to set up TensorRT
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    std::cout << "GPU " << gpu_id << ": Creating CUDA graph..." << std::endl;

    // Capture CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    context->enqueueV3(stream);

    int threads_per_block = 256;
    int num_blocks = (pooled_floats + threads_per_block - 1) / threads_per_block;

    mean_pooling_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        (float*)d_out, (int*)d_mask, d_pooled, batch_size, seq_len, hidden_dim
    );

    cudaMemcpyAsync(h_pooled, d_pooled, pooled_floats * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamEndCapture(stream, &graph);

    cudaError_t result = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
    if (result != cudaSuccess) {
        std::cerr << "GPU " << gpu_id << ": Graph instantiation failed!" << std::endl;
        return;
    }

    // Warmup
    std::cout << "GPU " << gpu_id << ": Warming up..." << std::endl;
    for (int i = 0; i < 100; ++i) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }

    std::cout << "GPU " << gpu_id << ": Running benchmark..." << std::endl;

    // Benchmark loop
    while (keep_running.load()) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
        iteration_counter++;
    }

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_ids);
    cudaFree(d_mask);
    cudaFree(d_type);
    cudaFree(d_out);
    cudaFree(d_pooled);
    cudaFreeHost(h_pooled);

    std::cout << "GPU " << gpu_id << ": Finished" << std::endl;
}

int main() {
    std::cout << "=== DUAL GPU BENCHMARK (2x RTX 4090) ===" << std::endl;

    // Check available GPUs
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " CUDA devices" << std::endl;

    if (deviceCount < 2) {
        std::cerr << "This benchmark requires 2 GPUs!" << std::endl;
        return 1;
    }

    // Print GPU info
    for (int i = 0; i < 2; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU " << i << ": " << prop.name
                  << " (" << prop.totalGlobalMem / (1024*1024*1024) << " GB)" << std::endl;
    }

    // Load tokenizer and prepare data
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(
        load_file_contents("onnx/tokenizer.json")
    );

    // Load engine once (will be copied to both GPUs)
    auto engine_data = load_engine_binary("engines/model_batch256_len32.engine");

    int batch_size = 256;
    int seq_len = 32;
    int hidden_dim = 384;
    int total_tokens = batch_size * seq_len;

    // Prepare input data
    std::string text = "JOHN DOE 1234 FAKE STREET ROGERS AR 72758";
    std::vector<int> single_ids = tokenizer->Encode(text);
    single_ids.resize(seq_len, 0);

    // Allocate pinned host memory (shared by both GPUs)
    int *pinned_ids, *pinned_mask, *pinned_type;
    cudaHostAlloc(&pinned_ids, total_tokens * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&pinned_mask, total_tokens * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&pinned_type, total_tokens * sizeof(int), cudaHostAllocDefault);

    // Fill input data
    for(int i = 0; i < batch_size; ++i) {
        std::copy(single_ids.begin(), single_ids.end(), pinned_ids + (i * seq_len));
        for(int j = 0; j < seq_len; ++j) {
            pinned_mask[i * seq_len + j] = (single_ids[j] != 0) ? 1 : 0;
            pinned_type[i * seq_len + j] = 0;
        }
    }

    std::cout << "\nStarting dual-GPU benchmark for 10 seconds..." << std::endl;

    auto start_time = high_resolution_clock::now();

    // Launch worker threads for both GPUs
    std::thread gpu0_thread(gpu_worker, 0, std::ref(engine_data),
                            pinned_ids, pinned_mask, pinned_type,
                            batch_size, seq_len, hidden_dim,
                            std::ref(gpu0_iterations));

    std::thread gpu1_thread(gpu_worker, 1, std::ref(engine_data),
                            pinned_ids, pinned_mask, pinned_type,
                            batch_size, seq_len, hidden_dim,
                            std::ref(gpu1_iterations));

    // Run for 10 seconds
    std::this_thread::sleep_for(std::chrono::seconds(10));
    keep_running = false;

    // Wait for threads to finish
    gpu0_thread.join();
    gpu1_thread.join();

    auto end_time = high_resolution_clock::now();
    duration<double> elapsed = end_time - start_time;

    // Calculate results
    int total_iterations = gpu0_iterations.load() + gpu1_iterations.load();
    int total_queries = total_iterations * batch_size;
    double qps = total_queries / elapsed.count();

    double gpu0_qps = (gpu0_iterations.load() * batch_size) / elapsed.count();
    double gpu1_qps = (gpu1_iterations.load() * batch_size) / elapsed.count();

    std::cout << "\n=== DUAL GPU RESULTS ===" << std::endl;
    std::cout << "Configuration:   2x RTX 4090" << std::endl;
    std::cout << "Batch Size:      " << batch_size << " per GPU" << std::endl;
    std::cout << "Duration:        " << std::fixed << std::setprecision(2)
              << elapsed.count() << " seconds" << std::endl;
    std::cout << "\nPer-GPU Performance:" << std::endl;
    std::cout << "  GPU 0: " << gpu0_iterations.load() << " batches, "
              << std::fixed << std::setprecision(0) << gpu0_qps << " RPS" << std::endl;
    std::cout << "  GPU 1: " << gpu1_iterations.load() << " batches, "
              << std::fixed << std::setprecision(0) << gpu1_qps << " RPS" << std::endl;
    std::cout << "\nCombined Performance:" << std::endl;
    std::cout << "  Total Batches:  " << total_iterations << std::endl;
    std::cout << "  Total Queries:  " << total_queries << std::endl;
    std::cout << "  Throughput:     " << std::fixed << std::setprecision(0)
              << qps << " RPS" << std::endl;

    // Compare to single GPU
    double speedup = qps / 141094.0;
    std::cout << "\nSpeedup vs Single GPU:" << std::endl;
    std::cout << "  Single GPU: 141,094 RPS" << std::endl;
    std::cout << "  Dual GPU:   " << std::fixed << std::setprecision(0) << qps << " RPS" << std::endl;
    std::cout << "  Speedup:    " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;

    if (qps >= 280000) {
        std::cout << "\nDOUBLE TARGET ACHIEVED! ✓✓" << std::endl;
    } else if (qps >= 140000) {
        std::cout << "\nTARGET ACHIEVED! ✓" << std::endl;
    }

    // Cleanup
    cudaFreeHost(pinned_ids);
    cudaFreeHost(pinned_mask);
    cudaFreeHost(pinned_type);

    return 0;
}