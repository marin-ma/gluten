/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ReaderThreadPool.h"

namespace gluten {

ReaderThreadPool::ReaderThreadPool(size_t numThreads) : numThreads_(numThreads) {
  workers_.reserve(numThreads);
  for (size_t i = 0; i < numThreads; ++i) {
    workers_.emplace_back([this]() { workerThread(); });
  }
}

ReaderThreadPool::~ReaderThreadPool() {
  shutdown();
}

void ReaderThreadPool::submit(Task task) {
  {
    std::lock_guard<std::mutex> lock(queueMutex_);
    if (stop_.load(std::memory_order_acquire)) {
      return; // Don't accept new tasks after shutdown
    }
    tasks_.push(std::move(task));
  }
  condition_.notify_one();
}

void ReaderThreadPool::shutdown() {
  if (!isShutdown()) {
    stop_.store(true, std::memory_order_release);
    condition_.notify_all();

    // Wait for all worker threads to finish their current tasks and join.
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }
}

void ReaderThreadPool::workerThread() {
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(queueMutex_);
      condition_.wait(lock, [this]() { return stop_.load(std::memory_order_acquire) || !tasks_.empty(); });

      if (stop_.load(std::memory_order_acquire)) {
        // Discard remaining tasks and exit the thread.
        return;
      }

      task = std::move(tasks_.front());
      tasks_.pop();
    }

    if (task) {
      task();
    }
  }
}

} // namespace gluten

// Made with Bob
