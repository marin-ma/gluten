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

// This File includes common helper functions with Arrow dependency.

#pragma once

#include <arrow/memory_pool.h>
#include <arrow/type.h>
#include <arrow/util/logging.h>
#include <sys/mman.h>
#include <numeric>
#include "memory/ArrowMemoryPool.h"
#include "memory/ColumnarBatch.h"
#include "utils/macros.h"

#include <iostream>

namespace gluten {

arrow::Result<std::shared_ptr<ColumnarBatch>> recordBatch2VeloxColumnarBatch(const arrow::RecordBatch& rb);

/**
 * arrow::MemoryPool instance used by tests and benchmarks
 */
class MyMemoryPool final : public arrow::MemoryPool {
 public:
  explicit MyMemoryPool() : capacity_(std::numeric_limits<int64_t>::max()) {}
  explicit MyMemoryPool(int64_t capacity) : capacity_(capacity) {}

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
    if (bytes_allocated() + size > capacity_) {
      return arrow::Status::OutOfMemory("malloc of size ", size, " failed");
    }
    RETURN_NOT_OK(pool_->Allocate(size, out));
    stats_.UpdateAllocatedBytes(size);
    return arrow::Status::OK();
  }

  arrow::Status Reallocate(int64_t oldSize, int64_t newSize, int64_t alignment, uint8_t** ptr) override {
    if (newSize > capacity_) {
      return arrow::Status::OutOfMemory("malloc of size ", newSize, " failed");
    }
    // auto old_ptr = *ptr;
    RETURN_NOT_OK(pool_->Reallocate(oldSize, newSize, ptr));
    stats_.UpdateAllocatedBytes(newSize - oldSize);
    return arrow::Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override {
    pool_->Free(buffer, size);
    stats_.UpdateAllocatedBytes(-size);
  }

  int64_t bytes_allocated() const override {
    return stats_.bytes_allocated();
  }

  int64_t max_memory() const override {
    return pool_->max_memory();
  }

  int64_t total_bytes_allocated() const override {
    return pool_->total_bytes_allocated();
  }

  int64_t num_allocations() const override {
    throw pool_->num_allocations();
  }

  std::string backend_name() const override {
    return pool_->backend_name();
  }

 private:
  arrow::MemoryPool* pool_ = arrow::default_memory_pool();
  int64_t capacity_;
  arrow::internal::MemoryPoolStats stats_;
};

class LargeMemoryPool : public arrow::MemoryPool {
 public:
  constexpr static uint64_t kHugePageSize = 1 << 21;
  constexpr static uint64_t kLargeBufferSize = 4 << 21;

  explicit LargeMemoryPool(MemoryPool* defaultMemoryPool = arrow::default_memory_pool()) : pool_(defaultMemoryPool) {}

  ~LargeMemoryPool() {
    std::for_each(
        freedBuffers_.begin(), freedBuffers_.end(), [this](BufferAllocated& buf) { doFree(buf.startAddr, buf.size); });
    ARROW_CHECK(buffers_.size() == 0);
  }

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
    if (size == 0) {
      return pool_->Allocate(0, alignment, out);
    }
    // make sure the size is cache line size aligned
    size = ROUND_TO_LINE(size, alignment);
    if (buffers_.empty() || size > buffers_.back().size - buffers_.back().allocated) {
      // Search in freed_buffers
      auto freedIt = std::find_if(
          freedBuffers_.begin(), freedBuffers_.end(), [size](BufferAllocated& buf) { return size <= buf.size; });
      if (freedIt != freedBuffers_.end()) {
        buffers_.push_back({freedIt->startAddr, freedIt->startAddr, freedIt->size, 0, 0});
        freedBuffers_.erase(freedIt);
      } else {
        // Allocate new page. Align to kHugePageSize.
        uint8_t* allocAddr;
        uint64_t allocSize = size > kLargeBufferSize ? ROUND_TO_LINE(size, kHugePageSize) : kLargeBufferSize;

        RETURN_NOT_OK(doAlloc(allocSize, kHugePageSize, &allocAddr));
        madvise(allocAddr, size, MADV_WILLNEED);
        buffers_.push_back({allocAddr, allocAddr, allocSize, 0, 0});
      }
    }
    auto& last = buffers_.back();
    *out = last.startAddr + last.allocated;
    last.lastAllocAddr = *out;
    last.allocated += size;
    return arrow::Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override {
    if (size == 0) {
      return;
    }
    // make sure the size is cache line size aligned
    size = ROUND_TO_LINE(size, alignment);

    auto its = std::find_if(buffers_.begin(), buffers_.end(), [buffer](BufferAllocated& buf) {
      return buffer >= buf.startAddr && buffer < buf.startAddr + buf.size;
    });
    ARROW_CHECK_NE(its, buffers_.end());
    its->freed += size;
    if (its->freed && its->freed == its->allocated) {
      freedBuffers_.push_back(*its);
      buffers_.erase(its);
    }
  }

  arrow::Status Reallocate(int64_t oldSize, int64_t newSize, int64_t alignment, uint8_t** ptr) override {
    if (oldSize == 0) {
      return arrow::Status::Invalid("Cannot call reallocated on oldSize == 0");
    }
    if (newSize == 0) {
      return arrow::Status::Invalid("Cannot call reallocated on newSize == 0");
    }
    auto* oldPtr = *ptr;
    auto& lastBuffer = buffers_.back();
    if (!(oldPtr >= lastBuffer.lastAllocAddr && oldPtr < lastBuffer.startAddr + lastBuffer.size)) {
      return arrow::Status::Invalid("Reallocate can only be called for the last buffer");
    }

    // shrink-to-fit
    if (newSize <= oldSize) {
      lastBuffer.allocated -= (oldSize - newSize);
      return arrow::Status::OK();
    }

    if (newSize - oldSize > lastBuffer.size - lastBuffer.allocated) {
      RETURN_NOT_OK(Allocate(newSize, alignment, ptr));
      memcpy(*ptr, oldPtr, std::min(oldSize, newSize));
      Free(oldPtr, oldSize, alignment);
    } else {
      lastBuffer.allocated += (newSize - oldSize);
    }
    return arrow::Status::OK();
  }

  int64_t bytes_allocated() const override {
    auto used = std::accumulate(buffers_.begin(), buffers_.end(), 0LL, [](uint64_t size, const BufferAllocated& buf) {
      return size + buf.size;
    });
    return std::accumulate(
        freedBuffers_.begin(), freedBuffers_.end(), used, [](uint64_t size, const BufferAllocated& buf) {
          return size + buf.size;
        });
  }

  int64_t max_memory() const override {
    return pool_->max_memory();
  }

  std::string backend_name() const override {
    return "LargeMemoryPool";
  }

  int64_t total_bytes_allocated() const override {
    return pool_->total_bytes_allocated();
  }

  int64_t num_allocations() const override {
    return pool_->num_allocations();
  }

 protected:
  virtual arrow::Status doAlloc(int64_t size, int64_t alignment, uint8_t** out) {
    return pool_->Allocate(size, alignment, out);
  }
  virtual void doFree(uint8_t* buffer, int64_t size) {
    pool_->Free(buffer, size);
  }

  struct BufferAllocated {
    uint8_t* startAddr;
    uint8_t* lastAllocAddr;
    uint64_t size;
    uint64_t allocated;
    uint64_t freed;
  };

  std::vector<BufferAllocated> buffers_;
  std::vector<BufferAllocated> freedBuffers_;

  MemoryPool* pool_;
};

class MMapMemoryPool : public LargeMemoryPool {
 public:
  explicit MMapMemoryPool() : LargeMemoryPool() {}

  ~MMapMemoryPool() override {
    std::for_each(
        freedBuffers_.begin(), freedBuffers_.end(), [this](BufferAllocated& buf) { doFree(buf.startAddr, buf.size); });
    ARROW_CHECK(buffers_.size() == 0);
  }

 protected:
  arrow::Status doAlloc(int64_t size, int64_t alignment, uint8_t** out) override {
    *out = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (*out == MAP_FAILED) {
      return arrow::Status::OutOfMemory(" mmap error ", size);
    } else {
      madvise(*out, size, MADV_WILLNEED);
      return arrow::Status::OK();
    }
  }

  void doFree(uint8_t* buffer, int64_t size) override {
    munmap((void*)(buffer), size);
  }
};

} // namespace gluten
