#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "threadsafequeue.hpp"
#include "batch_job.hpp"
#include "config.hpp"

using namespace nvinfer1;

class EngineLogger : public ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kERROR) std::cout << "[TRT ERROR] " << msg << std::endl;
  }
};

class TensorEngine {
private:
  int gpu_id;
  std::string engine_path;
  ModelConfig config;
  ThreadSafeQueue<BatchJob*>& inference_queue;
  ThreadSafeQueue<BatchJob*>& harvester_queue;

  std::thread worker_thread;
  EngineLogger logger;

  std::vector<char> load_engine_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open engine file: " + path);
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
  }

  void worker_loop() {
    cudaSetDevice(gpu_id);
    std::cout << "GPU " << gpu_id << ": Initializing Engine..." << std::endl;

    auto engine_data = load_engine_binary(engine_path);
    auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(logger));
    auto engine = std::unique_ptr<ICudaEngine>(
      runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())
    );

    if (!engine) {
      std::cerr << "GPU " << gpu_id << ": Failed to Deserialize Engine!" << std::endl;
      return;
    }

    auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

    int batch_size = config.batch_size;
    int seq_len = config.max_tokens;
    int embedding_dim = config.embedding_dim;
    int total_tokens = batch_size * seq_len;
    int total_embeddings = batch_size * embedding_dim;

    void *d_ids, *d_mask, *d_type, *d_embeddings;
    cudaMalloc(&d_ids, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_mask, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_type, total_tokens * sizeof(int32_t));
    cudaMalloc(&d_embeddings, total_embeddings * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    context->setInputShape("input_ids", Dims2{batch_size, seq_len});
    context->setInputShape("attention_mask", Dims2{batch_size, seq_len});
    context->setInputShape("token_type_ids", Dims2{batch_size, seq_len});

    context->setTensorAddress("input_ids", d_ids);
    context->setTensorAddress("attention_mask", d_mask);
    context->setTensorAddress("token_type_ids", d_type);
    context->setTensorAddress("embeddings", d_embeddings);

    std::cout << "GPU " << gpu_id << ": Ready for inference!" << std::endl;
    
    BatchJob* job;
    int batch_count = 0;

    while (inference_queue.wait_and_loop(job)) {
      cudaMemcpyAsync(d_ids, job->pinned_input_ids, total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_mask, job->pinned_attention_mask, total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
      cudaMemcpyAsync(d_type, job->pinned_token_type_ids, total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

      context->enqueueV3(stream);

      cudaMemcpyAsync(job->pinned_embeddings, d_embeddings, total_embeddings * sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      batch_count++;

      harvester_queue.push(job);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_ids);
    cudaFree(d_mask);
    cudaFree(d_type);
    cudaFree(d_embeddings);

    std::cout << "GPU " << gpu_id << ": Processed " << batch_count << " batches. Shutting down." << std::endl;
  }

public:
  TensorEngine(int id, const std::string& path, const ModelConfig& cfg,
    ThreadSafeQueue<BatchJob*>& in_q, ThreadSafeQueue<BatchJob*>& out_q)
    : gpu_id(id), engine_path(path), config(cfg), inference_queue(in_q), harvester_queue(out_q) {}

  // Spawn the background thread
  void start() {
    worker_thread = std::thread(&TensorEngine::worker_loop, this);
  } 

  // Wait for the thread to finish
  void join() {
    if (worker_thread.joinable()) {
      worker_thread.join();
    }
  }
};
