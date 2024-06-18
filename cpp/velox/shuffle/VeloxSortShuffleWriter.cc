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
#include "memory/VeloxColumnarBatch.h"
#include "velox/vector/TypeAliases.h"

namespace gluten {

namespace {

BufferPtr
makeIndices(vector_size_t size, std::function<vector_size_t(vector_size_t)> indexAt, memory::MemoryPool* pool) {
  BufferPtr indices = AlignedBuffer::allocate<vector_size_t>(size, pool);
  auto rawIndices = indices->asMutable<vector_size_t>();

  for (vector_size_t i = 0; i < size; i++) {
    rawIndices[i] = indexAt(i);
  }

  return indices;
}
} // namespace

arrow::Result<std::shared_ptr<VeloxShuffleWriter>> VeloxSortShuffleWriter::create(
    uint32_t numPartitions,
    std::unique_ptr<PartitionWriter> partitionWriter,
    gluten::ShuffleWriterOptions options,
    std::shared_ptr<facebook::velox::memory::MemoryPool> veloxPool,
    arrow::MemoryPool* arrowPool) {
  auto writer =
      std::make_shared<VeloxShuffleWriter>(numPartitions, std::move(partitionWriter), options, veloxPool, arrowPool);
  RETURN_NOT_OK(writer->init());
  return writer;
}

arrow::Status VeloxSortShuffleWriter::write(std::shared_ptr<ColumnarBatch> cb, int64_t memLimit) {
  if (options_.partitioning == Partitioning::kSingle) {
    auto veloxColumnBatch = VeloxColumnarBatch::from(veloxPool_.get(), cb);
    VELOX_CHECK_NOT_NULL(veloxColumnBatch);
    auto& rv = *veloxColumnBatch->getFlattenedRowVector();
    RETURN_NOT_OK(initFromRowVector(rv));
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;
    std::vector<facebook::velox::VectorPtr> complexChildren;
    for (auto& child : rv.children()) {
      if (child->encoding() == facebook::velox::VectorEncoding::Simple::FLAT) {
        auto status = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
            collectFlatVectorBuffer, child->typeKind(), child.get(), buffers, partitionBufferPool_.get());
        RETURN_NOT_OK(status);
      } else {
        complexChildren.emplace_back(child);
      }
    }
    if (complexChildren.size() > 0) {
      auto rowVector = std::make_shared<facebook::velox::RowVector>(
          veloxPool_.get(),
          complexWriteType_,
          facebook::velox::BufferPtr(nullptr),
          rv.size(),
          std::move(complexChildren));
      buffers.emplace_back();
      ARROW_ASSIGN_OR_RAISE(buffers.back(), generateComplexTypeBuffers(rowVector));
    }
    RETURN_NOT_OK(evictBuffers(0, rv.size(), std::move(buffers), false));
  } else if (options_.partitioning == Partitioning::kRange) {
    auto compositeBatch = std::dynamic_pointer_cast<CompositeColumnarBatch>(cb);
    VELOX_CHECK_NOT_NULL(compositeBatch);
    auto batches = compositeBatch->getBatches();
    VELOX_CHECK_EQ(batches.size(), 2);
    auto pidBatch = VeloxColumnarBatch::from(veloxPool_.get(), batches[0]);
    auto pidArr = getFirstColumn(*(pidBatch->getRowVector()));
    RETURN_NOT_OK(partitioner_->compute(pidArr, pidBatch->numRows(), row2Partition_, partition2RowCount_));
    auto rvBatch = VeloxColumnarBatch::from(veloxPool_.get(), batches[1]);
    auto& rv = *rvBatch->getFlattenedRowVector();
    RETURN_NOT_OK(initFromRowVector(rv));
    RETURN_NOT_OK(doSplit(rv, memLimit));
  } else {
    auto veloxColumnBatch = VeloxColumnarBatch::from(veloxPool_.get(), cb);
    VELOX_CHECK_NOT_NULL(veloxColumnBatch);
    facebook::velox::RowVectorPtr rv;
    rv = veloxColumnBatch->getFlattenedRowVector();
    if (isExtremelyLargeBatch(rv)) {
      auto numRows = rv->size();
      int32_t offset = 0;
      do {
        auto length = std::min(maxBatchSize_, numRows);
        auto slicedBatch = std::dynamic_pointer_cast<facebook::velox::RowVector>(rv->slice(offset, length));
        RETURN_NOT_OK(partitioningAndEvict(std::move(slicedBatch), memLimit));
        offset += length;
        numRows -= length;
      } while (numRows);
    } else {
      RETURN_NOT_OK(partitioningAndEvict(std::move(rv), memLimit));
    }
  }
  return arrow::Status::OK();
}

void collectBuffersFromRowVector(const facebook::velox::RowVector& rv) {
  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  std::vector<facebook::velox::VectorPtr> complexChildren;
  for (auto& child : rv.children()) {
    if (child->encoding() == facebook::velox::VectorEncoding::Simple::FLAT) {
      auto status = VELOX_DYNAMIC_SCALAR_TYPE_DISPATCH_ALL(
          collectFlatVectorBuffer, child->typeKind(), child.get(), buffers, partitionBufferPool_.get());
      RETURN_NOT_OK(status);
    } else {
      complexChildren.emplace_back(child);
    }
  }
  if (complexChildren.size() > 0) {
    auto rowVector = std::make_shared<facebook::velox::RowVector>(
        veloxPool_.get(),
        complexWriteType_,
        facebook::velox::BufferPtr(nullptr),
        rv.size(),
        std::move(complexChildren));
    buffers.emplace_back();
    ARROW_ASSIGN_OR_RAISE(buffers.back(), generateComplexTypeBuffers(rowVector));
  }
  RETURN_NOT_OK(evictBuffers(0, rv.size(), std::move(buffers), false));
}

arrow::Status localSort(
    const facebook::velox::VectorPtr& vector,
    facebook::velox::memory::MemoryPool* pool) {
  std::vector<std::pair<uint32_t, size_t>> pidWithRowId;
  pidWithRowId.reserve(row2Partition_.size());
  for (auto i = 0; i < row2Partition.size(); ++i) {
    auto pid = row2Partition[i];
    pidWithRowId.emplace_back(pid, i);
  }

  std::sort(pidWithRowId.begin(), pidWithRowId.end());

  // Build dictionary index for partition id.
  facebook::velox::BufferPtr indices =
      facebook::velox::AlignedBuffer::allocate<facebook::velox::vector_size_t>(row2Partition.size(), pool);
  auto rawIndices = indices->asMutable<facebook::velox::vector_size_t>();
  for (facebook::velox::vector_size_t i = 0; i < row2Partition.size(); i++) {
    rawIndices[i] = pidWithRowId[i].second;
  }

  auto sortedVector = facebook::velox::BaseVector::wrapInDictionary(
      facebook::velox::BufferPtr(nullptr), indices, row2Partition.size(), vector);
  facebook::velox::BaseVector::flattenVector(sortedVector);

  // Build pid to RowVector index + row range.
  for (auto i = 0; i < row2Partition.size(); ++i) {
    auto pid = pidWithRowId[i].first;
    auto rowId = pidWithRowId[i].second;
    partition2RowVector_[pid].emplace_back(rowId, folly::Range<uint32_t>(i, 1));
  }

  return sortedVector;
}
} // namespace gluten
