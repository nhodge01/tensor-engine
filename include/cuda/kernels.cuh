#ifndef ACRELAB_GAUSS_CUDA_KERNELS_CUH
#define ACRELAB_GAUSS_CUDA_KERNELS_CUH

#include <cuda_runtime.h>

namespace acrelab {
namespace cuda {

/**
 * @brief CUDA kernel for mean pooling over sequence dimension
 *
 * This kernel performs mean pooling of BERT/transformer hidden states
 * over the sequence dimension, respecting the attention mask.
 *
 * Grid/Block configuration:
 * - Total threads needed: batch_size * hidden_dim
 * - Recommended: 256 threads per block
 * - Blocks: (total_elements + 255) / 256
 *
 * @param hidden_states Input tensor [batch_size, seq_len, hidden_dim]
 * @param attention_mask Binary mask [batch_size, seq_len]
 * @param pooled_output Output tensor [batch_size, hidden_dim]
 * @param batch_size Number of samples in batch
 * @param seq_len Sequence length (e.g., 32)
 * @param hidden_dim Hidden dimension size (e.g., 384 for MiniLM)
 */
__global__ void mean_pooling_kernel(
    const float* hidden_states,
    const int* attention_mask,
    float* pooled_output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_dim;

    if (idx < total_elements) {
        int b = idx / hidden_dim;  // Batch index
        int d = idx % hidden_dim;  // Hidden dimension index

        float sum = 0.0f;
        float mask_sum = 0.0f;

        // Accumulate over sequence dimension
        for (int s = 0; s < seq_len; ++s) {
            int mask_val = attention_mask[b * seq_len + s];
            sum += hidden_states[(b * seq_len + s) * hidden_dim + d] * mask_val;
            mask_sum += mask_val;
        }

        // Avoid division by zero
        pooled_output[idx] = mask_sum > 0.0f ? (sum / mask_sum) : 0.0f;
    }
}

/**
 * @brief Helper function to launch mean pooling kernel
 */
inline void launch_mean_pooling(
    const float* hidden_states,
    const int* attention_mask,
    float* pooled_output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    cudaStream_t stream = 0
) {
    int total_elements = batch_size * hidden_dim;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    mean_pooling_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        hidden_states, attention_mask, pooled_output,
        batch_size, seq_len, hidden_dim
    );
}

/**
 * @brief L2 normalization kernel for embeddings
 * Optional: Add this if you need normalized embeddings
 */
__global__ void l2_normalize_kernel(
    float* embeddings,
    int batch_size,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size) {
        float* emb = embeddings + idx * hidden_dim;

        // Compute L2 norm
        float norm = 0.0f;
        for (int i = 0; i < hidden_dim; ++i) {
            norm += emb[i] * emb[i];
        }
        norm = sqrtf(norm + 1e-12f);  // Add epsilon for numerical stability

        // Normalize
        for (int i = 0; i < hidden_dim; ++i) {
            emb[i] /= norm;
        }
    }
}

} // namespace cuda
} // namespace acrelab

#endif // ACRELAB_GAUSS_CUDA_KERNELS_CUH