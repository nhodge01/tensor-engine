#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cuda_runtime_api.h>

struct BatchJob {
  int batch_id;
  int valid_items;
  std::vector<int64_t> row_ids;
  std::vector<std::string> raw_texts;

  int32_t* pinned_input_ids;
  int32_t* pinned_attention_mask;
  int32_t* pinned_token_type_ids;
  float* pinned_embeddings;

  int max_batch_size;
  int seq_len;
  int hidden_dim;

  BatchJob(
    int id, int batch_size, int sequence_length, int hidden_dimension
  ) : batch_id(id), valid_items(0), max_batch_size(batch_size), seq_len(sequence_length), hidden_dim(hidden_dimension)
  {
    row_ids.resize(max_batch_size);
    raw_texts.resize(max_batch_size);

    int total_tokens = max_batch_size * seq_len;
    int total_floats = max_batch_size * hidden_dim;

    if (cudaHostAlloc(&pinned_input_ids, total_tokens * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&pinned_attention_mask, total_tokens * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&pinned_token_type_ids, total_tokens * sizeof(int32_t), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(&pinned_embeddings, total_floats * sizeof(float), cudaHostAllocDefault) != cudaSuccess)
    {
      std::cerr << "[FATAL] Failed to allocate pinned memory for BatchJob " << batch_id << std::endl;
      exit(1);
    }

    memset(pinned_token_type_ids, 0, total_tokens * sizeof(int32_t));
  }

  ~BatchJob() {
    cudaFreeHost(pinned_input_ids);
    cudaFreeHost(pinned_attention_mask);
    cudaFreeHost(pinned_token_type_ids);
    cudaFreeHost(pinned_embeddings);
  }
  
  BatchJob(const BatchJob&) = delete;
  BatchJob& operator=(const BatchJob&) = delete;
  BatchJob(BatchJob&&) = delete;
  BatchJob& operator=(BatchJob&&) = delete;
};
