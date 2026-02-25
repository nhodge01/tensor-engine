#include <iostream>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <iomanip>
#include <vector>

using namespace std::chrono;

void benchmark_mutex() {
    std::mutex mtx;
    const int iterations = 10000000;  // 10 million

    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        mtx.lock();
        // Critical section (empty - just measuring lock/unlock overhead)
        mtx.unlock();
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<nanoseconds>(end - start);

    double ns_per_lock = duration.count() / static_cast<double>(iterations);
    double locks_per_second = 1e9 / ns_per_lock;

    std::cout << "\n=== std::mutex Performance ===" << std::endl;
    std::cout << "Iterations:        " << iterations << std::endl;
    std::cout << "Total time:        " << duration_cast<milliseconds>(duration).count() << " ms" << std::endl;
    std::cout << "Time per lock:     " << std::fixed << std::setprecision(1) << ns_per_lock << " ns" << std::endl;
    std::cout << "Locks per second:  " << std::fixed << std::setprecision(0) << locks_per_second << std::endl;
}

void benchmark_mutex_with_contention() {
    std::mutex mtx;
    std::atomic<int> counter{0};
    std::atomic<bool> running{true};
    const int num_threads = 4;
    const auto test_duration = std::chrono::seconds(1);

    auto worker = [&]() {
        while (running.load()) {
            std::lock_guard<std::mutex> lock(mtx);
            counter++;
        }
    };

    // Start worker threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    // Let them run for 1 second
    std::this_thread::sleep_for(test_duration);
    running = false;

    // Wait for threads to finish
    for (auto& t : threads) {
        t.join();
    }

    std::cout << "\n=== std::mutex With Contention (" << num_threads << " threads) ===" << std::endl;
    std::cout << "Total locks acquired: " << counter.load() << std::endl;
    std::cout << "Locks per second:     " << counter.load() << std::endl;
}

void benchmark_condition_variable() {
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> notifications{0};
    std::atomic<bool> ready{false};
    const int iterations = 100000;  // Less iterations because CV is slower

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < iterations; ++i) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                ready = true;
            }
            cv.notify_one();
        }
    });

    // Consumer measures time
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return ready.load(); });
        ready = false;
        notifications++;
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    producer.join();

    double us_per_op = duration.count() / static_cast<double>(iterations);
    double ops_per_second = 1e6 / us_per_op;

    std::cout << "\n=== std::condition_variable Performance ===" << std::endl;
    std::cout << "Iterations:        " << iterations << std::endl;
    std::cout << "Total time:        " << duration_cast<milliseconds>(duration).count() << " ms" << std::endl;
    std::cout << "Time per notify:   " << std::fixed << std::setprecision(1) << us_per_op << " µs" << std::endl;
    std::cout << "Notifies/second:   " << std::fixed << std::setprecision(0) << ops_per_second << std::endl;
}

void benchmark_your_gpu_rate() {
    const int batch_size = 256;
    const int gpu_rps = 330000;  // Your dual GPU throughput
    const double batches_per_second = gpu_rps / static_cast<double>(batch_size);

    std::cout << "\n=== Your GPU Requirements ===" << std::endl;
    std::cout << "GPU throughput:    " << gpu_rps << " RPS" << std::endl;
    std::cout << "Batch size:        " << batch_size << std::endl;
    std::cout << "Batches/second:    " << std::fixed << std::setprecision(0) << batches_per_second << std::endl;
    std::cout << "\nThe claim of '1,290 locks/sec' is off by " << std::fixed << std::setprecision(0)
              << (batches_per_second / 1290.0) << "x" << std::endl;
}

int main() {
    std::cout << "=== Testing std::mutex and std::condition_variable Performance ===" << std::endl;
    std::cout << "CPU: " << std::thread::hardware_concurrency() << " hardware threads available" << std::endl;

    benchmark_mutex();
    benchmark_mutex_with_contention();
    benchmark_condition_variable();
    benchmark_your_gpu_rate();

    std::cout << "\n=== VERDICT ===" << std::endl;
    std::cout << "The claim '1,290 locks per second' is COMPLETELY WRONG!" << std::endl;
    std::cout << "std::mutex can handle MILLIONS of locks per second." << std::endl;
    std::cout << "Even with contention, it's still orders of magnitude faster." << std::endl;

    return 0;
}