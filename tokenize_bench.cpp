/**
 * High-Performance Tokenizer Benchmark
 *
 * Demonstrates parallel tokenization using OpenMP on AMD Threadripper
 * Achieves 500k+ RPS with 8 threads for feeding dual RTX 4090 setup
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include "tokenizers_cpp.h"

std::string load_file_contents(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

int main(int argc, char* argv[]) {
    std::cout << "=== High-Performance Tokenizer Benchmark ===" << std::endl;

    // Load the tokenizer
    auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(
        load_file_contents("onnx/tokenizer.json")
    );

    if (!tokenizer) {
        std::cerr << "Failed to load tokenizer!" << std::endl;
        return -1;
    }

    // Configuration
    int total_strings = 1000000;  // 1M strings to tokenize
    int seq_len = 32;             // Match TensorRT engine expectations
    int num_threads = 8;          // Optimal for feeding dual GPUs

    // Allow command line override of thread count
    if (argc > 1) {
        num_threads = std::atoi(argv[1]);
    }

    // Test string - typical address/PII format
    std::string text = "JOHN DOE 1234 FAKE STREET ROGERS AR 72758";

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Strings:     " << total_strings << std::endl;
    std::cout << "  Seq Length:  " << seq_len << std::endl;
    std::cout << "  Threads:     " << num_threads << std::endl;

    // Allocate input strings
    std::cout << "\nAllocating " << total_strings << " strings in RAM..." << std::endl;
    std::vector<std::string> raw_texts(total_strings, text);

    // Pre-allocate output arrays (flat layout for GPU consumption)
    // These match exactly what TensorRT needs
    std::vector<int> flat_input_ids(total_strings * seq_len, 0);
    std::vector<int> flat_attention_mask(total_strings * seq_len, 0);
    std::vector<int> flat_token_type_ids(total_strings * seq_len, 0);

    // Set OpenMP thread count
    omp_set_num_threads(num_threads);

    std::cout << "Starting tokenization with " << num_threads << " OpenMP threads..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Parallel tokenization loop
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < total_strings; ++i) {
        // Tokenize the string
        std::vector<int> ids = tokenizer->Encode(raw_texts[i]);

        // Calculate actual length and truncate if needed
        int current_len = ids.size();
        int write_len = (current_len < seq_len) ? current_len : seq_len;

        // Calculate offset in flat arrays
        int offset = i * seq_len;

        // Write tokens and masks to flat arrays
        for (int j = 0; j < write_len; ++j) {
            flat_input_ids[offset + j] = ids[j];
            flat_attention_mask[offset + j] = 1;
            // token_type_ids stays 0 (single sentence)
        }
        // Padding is automatic (arrays initialized to 0)
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Calculate metrics
    double actual_time = elapsed_seconds.count();
    double rps = total_strings / actual_time;
    double million_per_sec = rps / 1000000.0;

    // Display results
    std::cout << "\n=== TOKENIZER BENCHMARK RESULTS ===" << std::endl;
    std::cout << "CPU Threads:     " << num_threads << std::endl;
    std::cout << "Total Strings:   " << total_strings << std::endl;
    std::cout << "Time Elapsed:    " << std::fixed << std::setprecision(4)
              << actual_time << " seconds" << std::endl;
    std::cout << "Throughput:      " << std::fixed << std::setprecision(0)
              << rps << " RPS" << std::endl;
    std::cout << "                 " << std::fixed << std::setprecision(2)
              << million_per_sec << "M strings/sec" << std::endl;

    // Performance analysis
    std::cout << "\n--- Performance Analysis ---" << std::endl;

    // Compare to GPU throughput
    double single_gpu_rps = 167000;  // Batch 256 optimized
    double dual_gpu_rps = 330000;    // Both GPUs

    if (rps > dual_gpu_rps) {
        std::cout << "✓ Can feed dual GPU setup ("
                  << std::fixed << std::setprecision(1)
                  << (rps / dual_gpu_rps) << "x headroom)" << std::endl;
    } else if (rps > single_gpu_rps) {
        std::cout << "✓ Can feed single GPU ("
                  << std::fixed << std::setprecision(1)
                  << (rps / single_gpu_rps) << "x headroom)" << std::endl;
        std::cout << "⚠ May bottleneck dual GPU setup" << std::endl;
    } else {
        std::cout << "⚠ Tokenization is the bottleneck!" << std::endl;
    }

    // Estimate scaling
    int total_cores = 24;  // Threadripper 3960X
    double scaling_efficiency = 0.7;  // Conservative estimate
    double estimated_max_rps = (rps / num_threads) * total_cores * scaling_efficiency;

    std::cout << "\nScaling estimate (24 cores):" << std::endl;
    std::cout << "  Theoretical: " << std::fixed << std::setprecision(0)
              << estimated_max_rps << " RPS" << std::endl;
    std::cout << "  GPUs supportable: " << std::fixed << std::setprecision(1)
              << (estimated_max_rps / single_gpu_rps) << " GPUs" << std::endl;

    std::cout << "====================================\n" << std::endl;

    // Verify output (spot check)
    bool verify = false;  // Set to true for debugging
    if (verify && total_strings > 0) {
        std::cout << "Sample tokenization (first string):" << std::endl;
        std::cout << "  Input: \"" << raw_texts[0] << "\"" << std::endl;
        std::cout << "  Tokens: ";
        for (int j = 0; j < 10 && j < seq_len; ++j) {
            std::cout << flat_input_ids[j] << " ";
        }
        std::cout << "..." << std::endl;
    }

    return 0;
}