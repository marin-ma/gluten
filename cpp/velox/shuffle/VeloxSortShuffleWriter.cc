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

#include "shuffle/VeloxSortShuffleWriter.h"
#include <arrow/io/memory.h>
#include "memory/ArrowMemory.h"
#include "memory/VeloxColumnarBatch.h"
#include "utils/Common.h"

#include "velox/row/CompactRow.h"

namespace gluten {

namespace {
uint64_t toCompactRowId(uint32_t partitionId, uint16_t inputIndex, uint32_t rowIndex) {
  // Make sure inputIdex < 4k (0 ~ 4095)
  return (uint64_t)partitionId << 32 | (uint32_t)inputIndex << 20 | rowIndex;
}

uint32_t extractPartitionId(uint64_t compactRowId) {
  return (uint32_t)(compactRowId >> 32);
}
} // namespace

arrow::Result<std::shared_ptr<VeloxShuffleWriter>> VeloxSortShuffleWriter::create(
    uint32_t numPartitions,
    std::unique_ptr<PartitionWriter> partitionWriter,
    ShuffleWriterOptions options,
    std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
    arrow::MemoryPool* arrowPool) {
  std::shared_ptr<VeloxSortShuffleWriter> writer(new VeloxSortShuffleWriter(
      numPartitions, std::move(partitionWriter), std::move(options), std::move(veloxPool), arrowPool));
  writer->init();
  return writer;
}

VeloxSortShuffleWriter::VeloxSortShuffleWriter(
    uint32_t numPartitions,
    std::unique_ptr<PartitionWriter> partitionWriter,
    ShuffleWriterOptions options,
    std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
    arrow::MemoryPool* pool)
    : VeloxShuffleWriter(numPartitions, std::move(partitionWriter), std::move(options), std::move(veloxPool), pool) {
  VELOX_CHECK(options_.partitioning != Partitioning::kSingle)
}

void VeloxSortShuffleWriter::insert(const facebook::velox::RowVectorPtr& vector) {
  auto numRows = vector->size();
  VELOX_DCHECK_GT(numRows, 0);

  facebook::velox::row::CompactRow row(vector);

  size_t totalSize = 0;
  if (auto fixedRowSize = facebook::velox::row::CompactRow::fixedRowSize(rowType_)) {
    totalSize = fixedRowSize.value() * numRows;
  } else {
    for (auto i = 0; i < numRows; ++i) {
      totalSize += row.rowSize(i);
    }
  }

  totalRows_ += numRows;
  totalBytes_ += totalSize;
  std::cout << "Allocating " << totalSize << " bytes, totalBytes: " << totalBytes_ << ", totalRows" << totalRows_
            << ", numInputs: " << numInputs_ << std::endl;
  // Buffer allocation can trigger self-spill. Separate buffer allocation and rowBuffer_.emplace_back().
  auto buffer = facebook::velox::AlignedBuffer::allocate<char>(totalSize, veloxPool_.get(), 0);
  auto* rawBuffer = buffer->asMutable<char>();
  size_t offset = 0;

  data_.reserve(data_.size() + numRows);
  for (auto i = 0; i < numRows; ++i) {
    auto size = row.serialize(i, rawBuffer + offset);
    data_.emplace_back(toCompactRowId(row2Partition_[i], numInputs_, i), std::string_view(rawBuffer + offset, size));
    offset += size;
  }

  rowBuffer_.emplace_back(std::move(buffer));
  ++numInputs_;
}

arrow::Status VeloxSortShuffleWriter::write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) {
  ARROW_ASSIGN_OR_RAISE(auto rv, getPeeledRowVector(cb));
  initRowType(rv);
  insert(rv);
  RETURN_NOT_OK(spillIfNeeded(memLimit));
  return arrow::Status::OK();
}

arrow::Status VeloxSortShuffleWriter::reclaimFixedSize(int64_t size, int64_t* actual) {
  if (evictState_ == EvictState::kUnevictable) {
    *actual = 0;
    return arrow::Status::OK();
  }
  auto beforeReclaim = veloxPool_->usedBytes();
  RETURN_NOT_OK(evictAllPartitions());
  *actual = beforeReclaim - veloxPool_->usedBytes();
  return arrow::Status::OK();
}

arrow::Status VeloxSortShuffleWriter::stop() {
  RETURN_NOT_OK(evictAllPartitions());
  sortedBuffer_ = nullptr;
  RETURN_NOT_OK(partitionWriter_->stop(&metrics_));
  return arrow::Status::OK();
}

arrow::Status VeloxSortShuffleWriter::evictPartition(uint32_t partitionId, size_t begin, size_t end) {
  // Serialize [begin, end)
  uint32_t numRows = end - begin;
  uint64_t rawSize = numRows * sizeof(size_t);
  for (auto i = begin; i < end; ++i) {
    rawSize += data_[i].second.length();
  }

  // numRows(4) | rawSize(8) | buffer
  uint64_t actualSize = sizeof(uint32_t) + sizeof(rawSize) + rawSize;
  if (sortedBuffer_ == nullptr) {
    sortedBuffer_ = facebook::velox::AlignedBuffer::allocate<char>(actualSize, veloxPool_.get());
  } else if (sortedBuffer_->size() < actualSize) {
    facebook::velox::AlignedBuffer::reallocate<char>(&sortedBuffer_, actualSize);
  }
  auto* rawBuffer = sortedBuffer_->asMutable<char>();

  uint64_t offset = 0;
  memcpy(rawBuffer, &numRows, sizeof(numRows));
  offset += sizeof(numRows);
  memcpy(rawBuffer + offset, &rawSize, sizeof(rawSize));
  offset += sizeof(rawSize);

  for (auto i = begin; i < end; ++i) {
    // size(size_t) | bytes
    auto size = data_[i].second.size();
    memcpy(rawBuffer + offset, &size, sizeof(size_t));
    offset += sizeof(size_t);
    memcpy(rawBuffer + offset, data_[i].second.data(), size);
    offset += size;
  }
  VELOX_CHECK_EQ(offset, actualSize);

  RETURN_NOT_OK(partitionWriter_->evict(partitionId, actualSize, sortedBuffer_->as<char>(), actualSize));
  return arrow::Status::OK();
}

arrow::Status VeloxSortShuffleWriter::evictAllPartitions() {
  if (evictState_ == EvictState::kUnevictable || data_.empty()) {
    return arrow::Status::OK();
  }
  EvictGuard evictGuard{evictState_};

  std::sort(data_.begin(), data_.end());

  size_t begin = 0;
  size_t cur = 0;
  auto pid = extractPartitionId(data_[begin].first);
  while (++cur < data_.size()) {
    auto curPid = extractPartitionId(data_[cur].first);
    if (curPid != pid) {
      RETURN_NOT_OK(evictPartition(pid, begin, cur));
      pid = curPid;
      begin = cur;
    }
  }
  RETURN_NOT_OK(evictPartition(pid, begin, cur));

  data_.clear();
  rowBuffer_.clear();
  numInputs_ = 0;
  totalBytes_ = 0;
  totalRows_ = 0;
  return arrow::Status::OK();
}

void VeloxSortShuffleWriter::init() {
  partition2RowCount_.resize(numPartitions_, 0);
}

void VeloxSortShuffleWriter::initRowType(const facebook::velox::RowVectorPtr& rv) {
  if (UNLIKELY(!rowType_)) {
    rowType_ = facebook::velox::asRowType(rv->type());
  }
}

arrow::Result<facebook::velox::RowVectorPtr> VeloxSortShuffleWriter::getPeeledRowVector(
    const std::shared_ptr<ColumnarBatch>& cb) {
  if (options_.partitioning == Partitioning::kRange) {
    auto compositeBatch = std::dynamic_pointer_cast<CompositeColumnarBatch>(cb);
    VELOX_CHECK_NOT_NULL(compositeBatch);
    auto batches = compositeBatch->getBatches();
    VELOX_CHECK_EQ(batches.size(), 2);

    auto pidBatch = VeloxColumnarBatch::from(veloxPool_.get(), batches[0]);
    auto pidArr = getFirstColumn(*(pidBatch->getRowVector()));
    RETURN_NOT_OK(partitioner_->compute(pidArr, pidBatch->numRows(), row2Partition_, partition2RowCount_));

    auto rvBatch = VeloxColumnarBatch::from(veloxPool_.get(), batches[1]);
    return rvBatch->getFlattenedRowVector();
  } else {
    auto veloxColumnBatch = VeloxColumnarBatch::from(veloxPool_.get(), cb);
    VELOX_CHECK_NOT_NULL(veloxColumnBatch);
    auto rv = veloxColumnBatch->getFlattenedRowVector();
    if (partitioner_->hasPid()) {
      auto pidArr = getFirstColumn(*rv);
      RETURN_NOT_OK(partitioner_->compute(pidArr, rv->size(), row2Partition_, partition2RowCount_));
      return getStrippedRowVector(*rv);
    } else {
      RETURN_NOT_OK(partitioner_->compute(nullptr, rv->size(), row2Partition_, partition2RowCount_));
      return rv;
    }
  }
}

arrow::Status VeloxSortShuffleWriter::spillIfNeeded(int64_t memLimit) {
  if (totalBytes_ >= memLimit >> 2 || totalRows_ / numPartitions_ >= 10) {
    RETURN_NOT_OK(evictAllPartitions());
  }
  return arrow::Status::OK();
}
} // namespace gluten
