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
#include "memory/ColumnarBatch.h"
#include "utils/macros.h"

#include <boost/stacktrace.hpp>
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

namespace {
#define ALIGNMENT 2 * 1024 * 1024
#define LARGE_BUFFER_SIZE 16 * 1024 * 1024
#define CAPACITY 1LL * 1024 * 1024 * 1024
} // namespace

class LargeMemoryPool : public arrow::MemoryPool {
 public:
  constexpr static uint64_t huge_page_size = 1 << 21;

  explicit LargeMemoryPool() : capacity_(std::numeric_limits<int64_t>::max()) {}
  explicit LargeMemoryPool(int64_t capacity) : capacity_(capacity) {}

  ~LargeMemoryPool() override = default;

  void SetSpillFunc(std::function<arrow::Status(int64_t, int64_t*)> f_spill) {
    f_spill_ = f_spill;
  }

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override {
    if (size == 0) {
      return pool_->Allocate(0, alignment, out);
    }
    // make sure the size is cache line size aligned
    size = ROUND_TO_LINE(size, alignment);
    uint64_t alloc_size = size > LARGE_BUFFER_SIZE ? size : LARGE_BUFFER_SIZE;
    alloc_size = ROUND_TO_LINE(alloc_size, huge_page_size);
    // std::cout << " allocated " << size << std::endl;
    auto its = std::find_if(buffers_.begin(), buffers_.end(), [size](BufferAllocated& buf) {
      return buf.allocated + size <= buf.alloc_size;
    });

    if (its == buffers_.end()) {
      uint8_t* alloc_addr;
      auto total_alloc1 = bytes_allocated();
      auto total_alloc2 = 0;
      if (total_alloc1 > capacity_ - alloc_size) {
        if (f_spill_) {
          int64_t act_free = 0;
          RETURN_NOT_OK(f_spill_(size, &act_free));
        }
        total_alloc2 = bytes_allocated();
        if (total_alloc2 > capacity_ - alloc_size)
          return arrow::Status::OutOfMemory("malloc of size ", size, " failed");
      }

      RETURN_NOT_OK(do_alloc(alloc_size, &alloc_addr));

      buffers_.push_back({alloc_addr, 0, 0, alloc_size});
      // std::cout << "alloc before = " << (total_alloc1 / 1024 / 1024) << " after = " << (total_alloc2 / 1024 / 1024)
      //           << " alloc size = " << alloc_size << " buffer size = " << buffers_.size() << std::endl;
      its = std::prev(buffers_.end());
    }

    BufferAllocated& lastalloc = *its;

    *out = lastalloc.start_addr + lastalloc.allocated;
    lastalloc.allocated += size;
    stats_.UpdateAllocatedBytes(size);
    return arrow::Status::OK();
  }

  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override {
    if (size == 0) {
      return pool_->Free(buffer, 0, alignment);
    }
    // make sure the size is cache line size aligned
    size = ROUND_TO_LINE(size, alignment);

    auto its = std::find_if(buffers_.begin(), buffers_.end(), [buffer](BufferAllocated& buf) {
      return buffer >= buf.start_addr && buffer < buf.start_addr + buf.alloc_size;
    });
    ARROW_CHECK_NE(its, buffers_.end());
    Free0(its, size);
    stats_.UpdateAllocatedBytes(-size);
  }

  arrow::Status Reallocate(int64_t oldSize, int64_t newSize, int64_t alignment, uint8_t** ptr) override {
    //    // No shrink-to-fit
    //    if (newSize <= oldSize) {
    //      return arrow::Status::OK();
    //    }
    Free(*ptr, oldSize, alignment);
    return Allocate(newSize, alignment, ptr);
  }

  int64_t bytes_allocated() const override {
    return std::accumulate(buffers_.begin(), buffers_.end(), 0LL, [](uint64_t a, const BufferAllocated& buf) {
      return a + buf.alloc_size;
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
  virtual arrow::Status do_alloc(int64_t size, uint8_t** out) {
    return pool_->Allocate(size, out);
  }
  virtual void do_free(uint8_t* buffer, int64_t size) {
    pool_->Free(buffer, size);
  }

  struct BufferAllocated {
    uint8_t* start_addr;
    uint64_t allocated;
    uint64_t freed;
    uint64_t alloc_size;
  };

  void Free0(std::vector<BufferAllocated>::iterator to_free, int64_t size) {
    // print_trace();
    to_free->freed += size;
    if (to_free->freed /*> (LARGE_BUFFER_SIZE >> 1)*/ && to_free->freed == to_free->allocated) {
      do_free(to_free->start_addr, to_free->alloc_size);
      buffers_.erase(to_free);
      // std::cout << "free " << std::hex << (uint64_t)to_free.start_addr << std::dec
      //           << " buffer size = " << buffers_.size() << std::endl;
    }
  }

  std::vector<BufferAllocated> buffers_;
  MemoryPool* pool_ = arrow::default_memory_pool();
  std::function<arrow::Status(int64_t, int64_t*)> f_spill_ = nullptr;
  uint64_t capacity_;
  arrow::internal::MemoryPoolStats stats_;
};

class LargePageMemoryPool : public LargeMemoryPool {
 public:
  explicit LargePageMemoryPool() : LargeMemoryPool() {}
  explicit LargePageMemoryPool(int64_t capacity) : LargeMemoryPool(capacity) {}

 protected:
  virtual arrow::Status do_alloc(int64_t size, uint8_t** out) {
    int rst = posix_memalign((void**)out, 1 << 21, size);
    madvise(*out, size, MADV_HUGEPAGE);
    madvise(*out, size, MADV_WILLNEED);
    if (rst != 0 || *out == nullptr) {
      return arrow::Status::OutOfMemory(" posix_memalign error ");
    } else {
      return arrow::Status::OK();
    }
  }
  virtual void do_free(uint8_t* buffer, int64_t size) {
    std::free((void*)(buffer));
  }
};

class MMapMemoryPool : public LargeMemoryPool {
 public:
  explicit MMapMemoryPool() : LargeMemoryPool() {}
  explicit MMapMemoryPool(int64_t capacity) : LargeMemoryPool(capacity) {}

 protected:
  virtual arrow::Status do_alloc(int64_t size, uint8_t** out) {
    *out = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (*out == MAP_FAILED) {
      std::cout << "stack: " << boost::stacktrace::stacktrace() << std::endl;
      return arrow::Status::OutOfMemory(" mmap error ", size);
    } else {
      madvise(*out, size, MADV_HUGEPAGE);
      madvise(*out, size, MADV_WILLNEED);
      return arrow::Status::OK();
    }
  }
  virtual void do_free(uint8_t* buffer, int64_t size) {
    munmap((void*)(buffer), size);
  }
};

} // namespace gluten
