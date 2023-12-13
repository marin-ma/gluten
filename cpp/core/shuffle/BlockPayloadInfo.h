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

namespace gluten {
class MergableBlockPayloadInfo : public Mergable {
 public:
  MergableBlockPayloadInfo(std::unique_ptr<arrow::io::InputStream>& is, uint32_t numRows)
      : BlockPayloadInfo(is), Mergable(numRows) {}

  arrow::Status merge(uint8_t* source, int64_t rawSize) override {
    ARROW_ASSIGN_OR_RAISE(auto bytesRead, is_->Read(rawSize, source));
    (void)bytesRead;
    return arrow::Status::OK();
  }

  arrow::Status mergeBitmap(uint8_t* source, int64_t offset, int64_t rawSize) override {
    ARROW_ASSIGN_OR_RAISE(auto buffer, is_->Read(rawSize));
    for (auto row = 0; row < numRows_; ++row) {
      if (arrow::bit_util::GetBit(buffer->data(), row)) {
        arrow::bit_util::SetBit(source, offset + row);
      }
    }
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> nextRawSize() override {
    int64_t size;
    ARROW_ASSIGN_OR_RAISE(auto bytesRead, is_->Read(sizeof(int64_t), &size));
    (void)bytesRead;
    return size;
  }
};

arrow::Status stop() {
  std::vector<Spill> spills;
  for (auto pid = 0; pid < numPartitions_; ++pid) {
    auto spillIter = spills.begin();
    std::vector<Mergable*> toBeMerged;
    uint32_t totalRows = 0;
    while (spillIter != spills.end()) {
      while (spillIter->hasNextPayload(pid)) {
        auto nextPayload = spillIter->nextPayload(pid);
        if (nextPayload->mergable()) {
          if (totalRows + nextPayload->getPayload()->numRows_ > kMaxBatchSize) {
            mergeAndClear(&toBeMerged, &totalRows, outputStream, codec, pool);
          }
          toBeMerged.push_back(dynamic_cast<Mergable*>(nextPayload));
          totalRows += nextPayload->getPayload()->numRows_;
          if (totalRows >= kBatchSize) {
            mergeAndClear(&toBeMerged, &totalRows, outputStream, codec, pool);
          }
          continue;
        }
        if (!toBeMerged.empty()) {
          // mergeAndClear should specially handle if toBeMerged.size == 1
          mergeAndClear(&toBeMerged, &totalRows, outputStream, codec, pool);
        }
        flush(nextPayload, os);
      }
      ++spillIter;
    }
  }
}

arrow::Status mergeAndClear(
    uint32_t& totalRows,
    std::vector<Mergable*>& mergables,
    const std::vector<bool>& isValidityBuffer,
    arrow::io::OutputStream* outputStream,
    arrow::util::Codec* codec,
    arrow::MemoryPool* pool) {
  auto validityBufferSize = arrow::bit_util::BytesForBits(totalRows);
  ARROW_ASSIGN_OR_RAISE(auto uncompressed, arrow::AllocateResizableBuffer(validityBufferSize, pool));
  for (bool isValidity : isValidityBuffer) {
    uint32_t rows = 0;
    if (isValidity) {
      memset(uncompressed->mutable_data(), 0, validityBufferSize);
      auto isAllNull = true;
      for (auto mergable : mergables) {
        ARROW_ASSIGN_OR_RAISE(auto rawSize, mergable->nextRawSize());
        if (!isAllNull && rawSize == kNullBuffer) {
          // Append all true of mergable->numRows.
          arrow::bit_util::SetBitsTo(uncompressed->mutable_data(), rows, mergable->numRows(), true);
        } else if (rawSize != kNullBuffer) {
          if (isAllNull) {
            isAllNull = false;
            // appendAllTrue of rows; handle bits;
            arrow::bit_util::SetBitsTo(uncompressed->mutable_data(), 0, rows, true);
          }
          RETURN_NOT_OK(mergable->mergeBitmap(uncompressed->mutable_data(), rows, rawSize));
        }
        rows += mergable->numRows();
      }
      RETURN_NOT_OK(uncompressed->Resize(validityBufferSize, false));
      RETURN_NOT_OK(compressAndFlush(uncompressed.get(), outputStream, codec, pool));
    } else {
      std::vector<int64_t> rawSizes;
      rawSizes.reserve(mergables.size());
      auto uncompressedSize = 0;
      for (auto mergable : mergables) {
        // rawSize cannot be 0.
        ARROW_ASSIGN_OR_RAISE(auto rawSize, mergable->nextRawSize());
        rawSizes.push_back(rawSize);
        uncompressedSize += rawSize;
      }
      RETURN_NOT_OK(uncompressed->Resize(uncompressedSize, false));
      auto writePos = uncompressed->mutable_data();
      for (size_t i = 0; i < mergables.size(); ++i) {
        RETURN_NOT_OK(mergables[i]->merge(writePos, rawSizes[i]));
        writePos += rawSizes[i];
      }
      RETURN_NOT_OK(compressAndFlush(uncompressed.get(), outputStream, codec, pool));
    }
  }
  totalRows = 0;
  mergables.clear();
}
} // namespace gluten