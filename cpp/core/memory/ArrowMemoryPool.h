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

#pragma once

#include <arrow/util/logging.h>
#include <sys/mman.h>
#include <numeric>
#include "MemoryAllocator.h"
#include "utils/macros.h"

namespace gluten {

std::shared_ptr<arrow::MemoryPool> asArrowMemoryPool(MemoryAllocator* allocator);

std::shared_ptr<arrow::MemoryPool> defaultArrowMemoryPool();

class ArrowMemoryPool final : public arrow::MemoryPool {
 public:
  explicit ArrowMemoryPool(MemoryAllocator* allocator) : allocator_(allocator) {}

  arrow::Status Allocate(int64_t size, int64_t alignment, uint8_t** out) override;

  arrow::Status Reallocate(int64_t oldSize, int64_t newSize, int64_t alignment, uint8_t** ptr) override;

  void Free(uint8_t* buffer, int64_t size, int64_t alignment) override;

  int64_t bytes_allocated() const override;

  int64_t total_bytes_allocated() const override;

  int64_t num_allocations() const override;

  std::string backend_name() const override;

 private:
  MemoryAllocator* allocator_;
};

namespace {
#define ALIGNMENT 2 * 1024 * 1024
#define LARGE_BUFFER_SIZE 16 * 1024 * 1024
#define CAPACITY 1LL * 1024 * 1024 * 1024
} // namespace

class LargeMemoryPool : public arrow::MemoryPool {
 public:
  constexpr static uint64_t kHugePageSize = 1 << 21;
  constexpr static int64_t kMaximumCapacity = std::numeric_limits<int64_t>::max();

  explicit LargeMemoryPool() : capacity_(kMaximumCapacity) {}
  explicit LargeMemoryPool(std::shared_ptr<AllocationListener> listener)
      : capacity_(kMaximumCapacity), listener_(std::move(listener)) {}
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
    // std::cout << " allocated " << size << std::endl;
    auto its = std::find_if(buffers_.begin(), buffers_.end(), [size](BufferAllocated& buf) {
      return buf.allocated + size <= buf.alloc_size;
    });

    if (its == buffers_.end()) {
      uint64_t alloc_size = size > LARGE_BUFFER_SIZE ? size : LARGE_BUFFER_SIZE;
      alloc_size = ROUND_TO_LINE(alloc_size, kHugePageSize);

      uint8_t* alloc_addr;
      auto total_alloc1 = bytes_allocated();
      if (alloc_size > capacity_ - total_alloc1) {
        if (f_spill_) {
          int64_t act_free = 0;
          RETURN_NOT_OK(f_spill_(size, &act_free));
        }
        auto total_alloc2 = 0;
        total_alloc2 = bytes_allocated();
        if (alloc_size > capacity_ - total_alloc2) {
          return arrow::Status::OutOfMemory("malloc of size ", size, " failed");
        }
      }

      listener_->allocationChanged(alloc_size);
      auto allocStatus = do_alloc(alloc_size, &alloc_addr);
      if (!allocStatus.ok()) {
        listener_->allocationChanged(-alloc_size);
        return allocStatus;
      }

      buffers_.push_back({alloc_addr, 0, 0, alloc_size});
      // std::cout << "alloc before = " << (total_alloc1 / 1024 / 1024) << " after = " << (total_alloc2 / 1024 / 1024)
      //           << " alloc size = " << alloc_size << " buffer size = " << buffers_.size() << std::endl;
      its = std::prev(buffers_.end());
    }

    BufferAllocated& lastalloc = *its;

    *out = lastalloc.start_addr + lastalloc.allocated;
    lastalloc.allocated += size;

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
    its->freed += size;
    if (its->freed /*> (LARGE_BUFFER_SIZE >> 1)*/ && its->freed == its->allocated) {
      do_free(its->start_addr, its->alloc_size);
      buffers_.erase(its);
      listener_->allocationChanged(-(its->alloc_size));
      // std::cout << "free " << std::hex << (uint64_t)to_free.start_addr << std::dec
      //           << " buffer size = " << buffers_.size() << std::endl;
    }
  }

  arrow::Status Reallocate(int64_t oldSize, int64_t newSize, int64_t alignment, uint8_t** ptr) override {
    // No shrink-to-fit
    // if (newSize <= oldSize) {
    //  return arrow::Status::OK();
    // }
    auto* oldPtr = *ptr;
    RETURN_NOT_OK(Allocate(newSize, alignment, ptr));
    memcpy(*ptr, oldPtr, std::min(oldSize, newSize));
    Free(oldPtr, oldSize, alignment);
    return arrow::Status::OK();
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

  std::vector<BufferAllocated> buffers_;
  MemoryPool* pool_ = defaultArrowMemoryPool().get();
  std::function<arrow::Status(int64_t, int64_t*)> f_spill_ = nullptr;
  uint64_t capacity_;
  std::shared_ptr<AllocationListener> listener_;
};

class LargePageMemoryPool : public LargeMemoryPool {
 public:
  explicit LargePageMemoryPool() : LargeMemoryPool() {}
  explicit LargePageMemoryPool(int64_t capacity) : LargeMemoryPool(capacity) {}

 protected:
  arrow::Status do_alloc(int64_t size, uint8_t** out) override {
    int rst = posix_memalign((void**)out, kHugePageSize, size);
    madvise(*out, size, MADV_HUGEPAGE);
    madvise(*out, size, MADV_WILLNEED);
    if (rst != 0 || *out == nullptr) {
      return arrow::Status::OutOfMemory(" posix_memalign error ");
    } else {
      return arrow::Status::OK();
    }
  }
  void do_free(uint8_t* buffer, int64_t size) override {
    std::free((void*)(buffer));
  }
};

class MMapMemoryPool : public LargeMemoryPool {
 public:
  explicit MMapMemoryPool() : LargeMemoryPool() {}
  explicit MMapMemoryPool(std::shared_ptr<AllocationListener> listener) : LargeMemoryPool(std::move(listener)) {}
  explicit MMapMemoryPool(int64_t capacity) : LargeMemoryPool(capacity) {}

 protected:
  arrow::Status do_alloc(int64_t size, uint8_t** out) override {
    *out = static_cast<uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (*out == MAP_FAILED) {
      return arrow::Status::OutOfMemory(" mmap error ", size);
    } else {
      madvise(*out, size, MADV_WILLNEED);
      return arrow::Status::OK();
    }
  }

  void do_free(uint8_t* buffer, int64_t size) override {
    munmap((void*)(buffer), size);
  }
};
} // namespace gluten
