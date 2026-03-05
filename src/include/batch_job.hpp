#ifndef ACRELAB_GAUSS_BATCH_JOB_HPP
#define ACRELAB_GAUSS_BATCH_JOB_HPP

#include <vector>
#include <string>
#include <string_view>
#include <iostream>
#include <cuda_runtime_api.h>

namespace acrelab {

/**
 * @brief Universal Multi-Vector BatchJob
 * Designed to hold N versions/concatenations of the same physical row.
 */
struct BatchJob {
    int batch_id;
    int valid_items;      // Physical rows (e.g., 256 GIDs)
    int num_concats;      // Versions per row (e.g., 5 or 6)
    
    std::vector<int64_t> row_ids;
    // Flat vector holding strings for all versions: batch_size * num_concats
    std::vector<std::string_view> raw_texts; 

    int32_t* pinned_input_ids;
    int32_t* pinned_attention_mask;
    int32_t* pinned_token_type_ids;
    float* pinned_embeddings;

    int max_batch_size;
    int seq_len;
    int hidden_dim;

    BatchJob(int id, int batch_size, int n_vectors, int sequence_length, int hidden_dimension) 
        : batch_id(id), valid_items(0), num_concats(n_vectors), 
          max_batch_size(batch_size), seq_len(sequence_length), hidden_dim(hidden_dimension)
    {
        row_ids.resize(max_batch_size);
        raw_texts.resize(max_batch_size * num_concats);

        int total_slots = max_batch_size * num_concats;
        size_t token_bytes = total_slots * seq_len * sizeof(int32_t);
        size_t float_bytes = total_slots * hidden_dim * sizeof(float);

        // Standard Pinned Allocation
        if (cudaHostAlloc((void**)&pinned_input_ids, token_bytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**)&pinned_attention_mask, token_bytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**)&pinned_token_type_ids, token_bytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**)&pinned_embeddings, float_bytes, cudaHostAllocDefault) != cudaSuccess) 
        {
            std::cerr << "[FATAL] Failed to allocate pinned memory for BatchJob " << batch_id << std::endl;
            exit(1);
        }

        memset(pinned_token_type_ids, 0, token_bytes);
    }

    ~BatchJob() {
        cudaFreeHost(pinned_input_ids);
        cudaFreeHost(pinned_attention_mask);
        cudaFreeHost(pinned_token_type_ids);
        cudaFreeHost(pinned_embeddings);
    }

    // Move-only semantics to prevent accidental copies of massive pinned buffers
    BatchJob(const BatchJob&) = delete;
    BatchJob& operator=(const BatchJob&) = delete;
};

} // namespace acrelab

#endif // ACRELAB_GAUSS_BATCH_JOB_HPP
