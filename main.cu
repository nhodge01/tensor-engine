#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <iomanip>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "tokenizers_cpp.h"

using namespace nvinfer1;

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

int main() {
    std::cout << "--- Initializing SVP-Proof Benchmark Pipeline ---" << std::endl;
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(load_file_contents("onnx/tokenizer.json"));
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
    auto engine_data = load_engine_binary("model_batch512_len32.engine"); 
    auto engine = std::unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

    int batch_size = 512;
    int seq_len = 32;
    int hidden_dim = 384;
    int total_tokens = batch_size * seq_len;
    int pooled_floats = batch_size * hidden_dim; 

    std::string text = "JOHN DOE 1234 FAKE STREET ROGERS AR 72758";
    std::vector<int> single_ids = tokenizer->Encode(text);
    single_ids.resize(seq_len, 0);

    std::vector<int> input_ids(total_tokens);
    for(int i = 0; i < batch_size; ++i) {
        std::copy(single_ids.begin(), single_ids.end(), input_ids.begin() + (i * seq_len));
    }
    
    std::vector<int> attention_mask(total_tokens, 0);
    for(int i = 0; i < batch_size; ++i) {
        for(int j = 0; j < seq_len; ++j) {
            if (single_ids[j] != 0) attention_mask[i * seq_len + j] = 1;
        }
    }
    std::vector<int> token_type_ids(total_tokens, 0);

    void *d_ids, *d_mask, *d_type, *d_out, *d_pooled_out;
    cudaMalloc(&d_ids, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_mask, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_type, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_out, total_tokens * hidden_dim * sizeof(float)); 
    cudaMalloc(&d_pooled_out, pooled_floats * sizeof(float)); 

    context->setInputShape("input_ids", Dims2{batch_size, seq_len});
    context->setInputShape("attention_mask", Dims2{batch_size, seq_len});
    context->setInputShape("token_type_ids", Dims2{batch_size, seq_len});
    context->setTensorAddress("input_ids", d_ids);
    context->setTensorAddress("attention_mask", d_mask);
    context->setTensorAddress("token_type_ids", d_type);
    context->setTensorAddress("last_hidden_state", d_out);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threads_per_block = 256;
    int num_blocks = (pooled_floats + threads_per_block - 1) / threads_per_block;

    // A destination buffer for the final vectors in Host RAM
    std::vector<float> final_buffer(pooled_floats);

    // Warmup
    std::cout << "Warming up PCIe Bus and Tensor Cores..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        cudaMemcpyAsync(d_ids, input_ids.data(), total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mask, attention_mask.data(), total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_type, token_type_ids.data(), total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        context->enqueueV3(stream);
        mean_pooling_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            (float*)d_out, (int*)d_mask, (float*)d_pooled_out, batch_size, seq_len, hidden_dim
        );
        cudaMemcpyAsync(final_buffer.data(), d_pooled_out, pooled_floats * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    std::cout << "Starting 10-second END-TO-END throughput benchmark..." << std::endl;
    
    int iterations = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto current_time = start_time;
    
    while (std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count() < 10) {
        
        // 1. Send data to GPU
        cudaMemcpyAsync(d_ids, input_ids.data(), total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mask, attention_mask.data(), total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_type, token_type_ids.data(), total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        
        // 2. Compute embeddings
        context->enqueueV3(stream);
        
        // 3. Pool embeddings
        mean_pooling_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            (float*)d_out, (int*)d_mask, (float*)d_pooled_out, batch_size, seq_len, hidden_dim
        );
        
        // 4. Pull data back to CPU
        cudaMemcpyAsync(final_buffer.data(), d_pooled_out, pooled_floats * sizeof(float), cudaMemcpyDeviceToHost, stream);
        
        // 5. Force CPU to wait until the floats are actually in RAM
        cudaStreamSynchronize(stream); 
        
        iterations++;
        current_time = std::chrono::high_resolution_clock::now();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    double total_time_ms = elapsed_seconds.count() * 1000.0;
    double avg_latency_ms = total_time_ms / iterations;
    int total_queries_processed = iterations * batch_size;
    double queries_per_second = total_queries_processed / elapsed_seconds.count();

    std::cout << "\n=== RTX 4080 END-TO-END RESULTS ===" << std::endl;
    std::cout << "Includes H2D Copy, TRT Compute, Kernel Pool, and D2H Copy" << std::endl;
    std::cout << "Batch Size:      " << batch_size << std::endl;
    std::cout << "Sequence Length: " << seq_len << " tokens" << std::endl;
    std::cout << "Total Queries:   " << total_queries_processed << std::endl;
    std::cout << "Batch Latency:   " << std::fixed << std::setprecision(4) << avg_latency_ms << " ms" << std::endl;
    std::cout << "Throughput:      " << std::fixed << std::setprecision(0) << queries_per_second << " RPS" << std::endl;
    std::cout << "===================================\n" << std::endl;

    std::cout << "--- OUTPUT ---" << std::endl;
    std::cout << "Text Input: \"" << text << "\"" << std::endl;
    std::cout << "First 10 dimensions of the vector safely back in CPU RAM:" << std::endl;
    std::cout << "[ ";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::fixed << std::setprecision(6) << final_buffer[i] << (i < 9 ? ", " : "");
    }
    std::cout << " ... ]\n" << std::endl;

    cudaStreamDestroy(stream);
    cudaFree(d_ids); cudaFree(d_mask); cudaFree(d_type); 
    cudaFree(d_out); cudaFree(d_pooled_out);
    return 0;
}
