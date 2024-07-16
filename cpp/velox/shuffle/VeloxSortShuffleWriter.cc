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
#include "utils/Timer.h"

namespace gluten {

namespace {
constexpr uint32_t kMaskLower27Bits = (1 << 27) - 1;
constexpr uint64_t kMaskLower40Bits = (1UL << 40) - 1;

uint64_t toCompactRowId(uint32_t partitionId, uint32_t pageNumber, uint32_t offsetInPage) {
  // |63 partitionId(24) |39 inputIndex(13) |26 rowIndex(27) |
  return (uint64_t)partitionId << 40 | (uint64_t)pageNumber << 27 | offsetInPage;
}

uint32_t extractPartitionId(uint64_t compactRowId) {
  return (uint32_t)(compactRowId >> 40);
}

std::pair<uint32_t, uint32_t> extractPageNumberAndOffset(uint64_t compactRowId) {
  return {(compactRowId & kMaskLower40Bits) >> 27, compactRowId & kMaskLower27Bits};
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
  VELOX_CHECK(options_.partitioning != Partitioning::kSingle);
  // FIXME: Use configuration to replace hardcode.
  data_.resize(4096);
}

void VeloxSortShuffleWriter::insertRows(facebook::velox::row::CompactRow& row, uint32_t offset, uint32_t rows) {
  auto dataSize = data_.size();
  while (dataSize < totalRows_ + rows) {
    dataSize <<= 1;
  }
  if (dataSize != data_.size()) {
    std::cout << "resize data_: " << data_.size() << " " << dataSize << std::endl;
    data_.resize(dataSize);
  }

  for (auto i = offset; i < offset + rows; ++i) {
    auto size = row.serialize(i, currentPage_ + pageCursor_);
    data_[totalRows_++] = {toCompactRowId(row2Partition_[i], pageNumber_, pageCursor_), size};
    pageCursor_ += size;
  }
}

uint32_t VeloxSortShuffleWriter::maxRowsToInsert(uint32_t offset, uint32_t rows) {
  // Check how many rows can be handled.
  if (pages_.empty()) {
    return 0;
  }
  auto remainingBytes = pages_.back()->size() - pageCursor_;
  if (fixedRowSize_) {
    return std::min((uint32_t)(remainingBytes / (fixedRowSize_.value())), rows);
  }
  auto beginIter = rowSizes_.begin() + 1 + offset;
  auto iter = std::upper_bound(beginIter, rowSizes_.end(), remainingBytes);
  return iter - beginIter;
}

void VeloxSortShuffleWriter::acquireNewBuffer(int64_t memLimit, uint64_t minSizeRequired) {
  auto size = std::max(std::min((uint64_t)memLimit >> 2, 64UL * 1024 * 1024), minSizeRequired);
  auto newBuffer = facebook::velox::AlignedBuffer::allocate<char>(size, veloxPool_.get(), 0);
  std::cout << "Allocate new buffer: " << size << std::endl;
  // Allocating new buffer can trigger spill.
  pages_.emplace_back(std::move(newBuffer));
  pageCursor_ = 0;
  pageNumber_ = pages_.size() - 1;
  currentPage_ = pages_.back()->asMutable<char>();
  memset(currentPage_, 0, size);
  pageAddresses_.emplace_back(currentPage_);
}

arrow::Status VeloxSortShuffleWriter::insert(const facebook::velox::RowVectorPtr& vector, int64_t memLimit) {
  ScopedTimer timer(&convertTime_);
  auto inputRows = vector->size();
  VELOX_DCHECK_GT(inputRows, 0);

  facebook::velox::row::CompactRow row(vector);

  if (!fixedRowSize_) {
    rowSizes_.resize(inputRows + 1);
    rowSizes_[0] = 0;
    for (auto i = 0; i < inputRows; ++i) {
      const size_t rowSize = row.rowSize(i);
      rowSizes_[i + 1] = rowSizes_[i] + rowSize;
    }
  }

  uint32_t rowOffset = 0;
  while (rowOffset < inputRows) {
    auto remainingRows = inputRows - rowOffset;
    auto rows = maxRowsToInsert(rowOffset, remainingRows);
    if (rows == 0) {
      auto minSizeRequired = fixedRowSize_ ? fixedRowSize_.value() : rowSizes_[rowOffset + 1] - rowSizes_[rowOffset];
      acquireNewBuffer(memLimit, minSizeRequired);
      rows = maxRowsToInsert(rowOffset, remainingRows);
      ARROW_RETURN_IF(
          rows == 0, arrow::Status::Invalid("Failed to insert rows. Remaining rows: " + std::to_string(remainingRows)));
    }
    //    std::cout << rowOffset << " " << rows << " " << inputRows << std::endl;
    insertRows(row, rowOffset, rows);
    rowOffset += rows;
  }
  ++numInputs_;
  //  std::cout << "writeOffset: " << writeOffset_ << ", totalRows: " << totalRows_ << ", numInputs: " << numInputs_
  //            << std::endl;
  return arrow::Status::OK();
}

arrow::Status VeloxSortShuffleWriter::write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) {
  //  RETURN_NOT_OK(spillIfNeeded(memLimit, cb->numRows()));
  ARROW_ASSIGN_OR_RAISE(auto rv, getPeeledRowVector(cb));
  initRowType(rv);
  RETURN_NOT_OK(insert(rv, memLimit));
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
  std::cout << "convert time: " << convertTime_ << std::endl;
  stopped_ = true;
  RETURN_NOT_OK(evictAllPartitions());
  sortedBuffer_ = nullptr;
  RETURN_NOT_OK(partitionWriter_->stop(&metrics_));
  return arrow::Status::OK();
}

arrow::Status VeloxSortShuffleWriter::evictPartition(uint32_t partitionId, size_t begin, size_t end) {
  // Serialize [begin, end)
  uint32_t numRows = end - begin;
  uint64_t rawSize = numRows * sizeof(RowSize);
  for (auto i = begin; i < end; ++i) {
    rawSize += data_[i].second;
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
    auto size = data_[i].second;
    memcpy(rawBuffer + offset, &size, sizeof(RowSize));
    offset += sizeof(RowSize);
    auto index = extractPageNumberAndOffset(data_[i].first);
    memcpy(rawBuffer + offset, pageAddresses_[index.first] + index.second, size);
    offset += size;
  }
  VELOX_CHECK_EQ(offset, actualSize);

  std::unique_ptr<BlockPayload> payload;
  auto rawData = sortedBuffer_->as<uint8_t>();
  if (codec_) {
    auto maxCompressedLength = codec_->MaxCompressedLen(actualSize, rawData);
    std::shared_ptr<arrow::ResizableBuffer> compressed;
    ARROW_ASSIGN_OR_RAISE(compressed, arrow::AllocateResizableBuffer(maxCompressedLength + 2 * sizeof(int64_t), pool_));
    // Compress.
    ARROW_ASSIGN_OR_RAISE(
        auto compressedLength,
        codec_->Compress(actualSize, rawData, maxCompressedLength, compressed->mutable_data() + 2 * sizeof(int64_t)));
    memcpy(compressed->mutable_data(), &compressedLength, sizeof(int64_t));
    memcpy(compressed->mutable_data() + sizeof(int64_t), &actualSize, sizeof(int64_t));
    auto sliced = arrow::SliceBuffer(compressed, 0, compressedLength + 2 * sizeof(int64_t));
    ARROW_ASSIGN_OR_RAISE(
        payload, BlockPayload::fromBuffers(Payload::kRaw, 0, {std::move(sliced)}, nullptr, nullptr, nullptr));
  } else {
    auto buffer = std::make_shared<arrow::Buffer>(rawData, actualSize);
    ARROW_ASSIGN_OR_RAISE(
        payload, BlockPayload::fromBuffers(Payload::kRaw, 0, {std::move(buffer)}, nullptr, nullptr, nullptr));
  }

  RETURN_NOT_OK(partitionWriter_->evict(partitionId, std::move(payload), stopped_));
  return arrow::Status::OK();
}

arrow::Status VeloxSortShuffleWriter::evictAllPartitions() {
  if (evictState_ == EvictState::kUnevictable || totalRows_ == 0) {
    return arrow::Status::OK();
  }
  EvictGuard evictGuard{evictState_};

  // TODO: Add radix sort/tim sort to align with Spark.
  std::sort(data_.begin(), data_.begin() + totalRows_);

  size_t begin = 0;
  size_t cur = 0;
  auto pid = extractPartitionId(data_[begin].first);
  while (++cur < totalRows_) {
    auto curPid = extractPartitionId(data_[cur].first);
    if (curPid != pid) {
      RETURN_NOT_OK(evictPartition(pid, begin, cur));
      pid = curPid;
      begin = cur;
    }
  }
  RETURN_NOT_OK(evictPartition(pid, begin, cur));

  pages_.clear();
  pageAddresses_.clear();
  numInputs_ = 0;
  pageCursor_ = 0;
  totalRows_ = 0;
  return arrow::Status::OK();
}

void VeloxSortShuffleWriter::init() {
  partition2RowCount_.resize(numPartitions_, 0);
  const auto& partitionWriterOptions = partitionWriter_->options();
  // TODO: Compress in partitionWriter_.
  codec_ = createArrowIpcCodec(
      partitionWriterOptions.compressionType,
      partitionWriterOptions.codecBackend,
      partitionWriterOptions.compressionLevel);
}

void VeloxSortShuffleWriter::initRowType(const facebook::velox::RowVectorPtr& rv) {
  if (UNLIKELY(!rowType_)) {
    rowType_ = facebook::velox::asRowType(rv->type());
    fixedRowSize_ = facebook::velox::row::CompactRow::fixedRowSize(rowType_);
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

arrow::Status VeloxSortShuffleWriter::spillIfNeeded(int64_t memLimit, int32_t nextRows) {
  if (pageCursor_ >= memLimit >> 2 || totalRows_ + nextRows > 8UL * 1024 * 1024) {
    std::cout << "spill triggered: totalRows: " << totalRows_ << std::endl;
    RETURN_NOT_OK(evictAllPartitions());
  }
  return arrow::Status::OK();
}
} // namespace gluten
