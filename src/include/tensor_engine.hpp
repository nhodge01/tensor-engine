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
using namespace acrelab;

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

      // 1. DISCOVERY PHASE
      std::vector<std::string> active_inputs;
      const char* potential_inputs[] = {"input_ids", "attention_mask", "token_type_ids"};
    
      for (const char* name : potential_inputs) {
          if (engine->getTensorIOMode(name) != TensorIOMode::kNONE) {
              active_inputs.push_back(name);
              std::cout << "GPU " << gpu_id << ": Active Input found -> " << name << std::endl;
          }
      }

      // CRITICAL: Scale allocations by the number of concats/versions
      // This assumes all jobs have the same num_concats as defined in the config
      int num_v = config.input_columns.size();
      int max_gpu_batch = config.batch_size * num_v;
      int max_tokens = max_gpu_batch * config.max_tokens;
      int max_embeddings = max_gpu_batch * config.embedding_dim;

      // 2. ALLOCATION (Sized for the "Wide" Batch)
      void *d_ids, *d_mask, *d_type, *d_embeddings;
      cudaMalloc(&d_ids, max_tokens * sizeof(int32_t));
      cudaMalloc(&d_mask, max_tokens * sizeof(int32_t));
      cudaMalloc(&d_type, max_tokens * sizeof(int32_t));
      cudaMalloc(&d_embeddings, max_embeddings * sizeof(float));

      cudaStream_t stream;
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

      // 3. BINDING (Use the d_ pointers)
      // We set the max shape here; the loop will refine it with setInputShape per batch
      for (const auto& name : active_inputs) {
          if (name == "input_ids") context->setTensorAddress(name.c_str(), d_ids);
          if (name == "attention_mask") context->setTensorAddress(name.c_str(), d_mask);
          if (name == "token_type_ids") context->setTensorAddress(name.c_str(), d_type);
      }
      context->setTensorAddress("embeddings", d_embeddings);

      std::cout << "GPU " << gpu_id << ": Ready for inference!" << std::endl;
    
      BatchJob* job;
      int batch_count = 0;

      // 4. INFERENCE LOOP
      while (inference_queue.wait_and_loop(job)) {
          // Calculate ACTIVE count (num_concats comes from the job struct)
          int total_active_sequences = job->valid_items * job->num_concats;
          size_t active_token_count = (size_t)total_active_sequences * config.max_tokens;
          size_t active_float_count = (size_t)total_active_sequences * config.embedding_dim;

          // Set the GPU shapes dynamically
          context->setInputShape("input_ids", Dims2{total_active_sequences, config.max_tokens});
          context->setInputShape("attention_mask", Dims2{total_active_sequences, config.max_tokens});

          // Dynamic PCIe Upload
          for (const auto& name : active_inputs) {
              if (name == "input_ids") 
                  cudaMemcpyAsync(d_ids, job->pinned_input_ids, active_token_count * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
              else if (name == "attention_mask") 
                  cudaMemcpyAsync(d_mask, job->pinned_attention_mask, active_token_count * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
              else if (name == "token_type_ids") 
                  cudaMemcpyAsync(d_type, job->pinned_token_type_ids, active_token_count * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
          }

          // Run Inference
          context->enqueueV3(stream);

          // Dynamic PCIe Download
          cudaMemcpyAsync(job->pinned_embeddings, d_embeddings, active_float_count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
          cudaStreamSynchronize(stream);
          batch_count++;
          harvester_queue.push(job);
      }

      // Cleanup
      cudaStreamDestroy(stream);
      cudaFree(d_ids); cudaFree(d_mask); cudaFree(d_type); cudaFree(d_embeddings);
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
