#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadSafeQueue {
private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_var_;
  bool shutdown_flag_ = false;

public:
  ThreadSafeQueue() = default;
  ~ThreadSafeQueue() = default;

  ThreadSafeQueue(const ThreadSafeQueue&) = delete;
  ThreadSafeQueue& operator=(const ThreadSafeQueue&) = delete;

  void push(T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(value));
    cond_var_.notify_one();
  }

  bool wait_and_loop(T& value) {
    std::unique_lock<std::mutex> lock(mutex_);

    cond_var_.wait(lock, [this] { return !queue_.empty() || shutdown_flag_; });

    if (shutdown_flag_ && queue_.empty()) {
      return false;
    }

    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool try_pop(T& value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_flag_ = true;
    cond_var_.notify_all();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

};
