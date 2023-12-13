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

#include <arrow/buffer.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>

#include "shuffle/Options.h"

namespace gluten {

static constexpr int64_t kZeroLengthBuffer = 0;
static constexpr int64_t kNullBuffer = -1;
static constexpr int64_t kUncompressedBuffer = -2;

class Payload {
 public:
  enum Type : int32_t { kCompressed, kUncompressed };
  Payload(
      Type type,
      uint32_t numRows,
      const std::vector<bool>* isValidityBuffer,
      arrow::MemoryPool* pool)
      : type_(type),
        numRows_(numRows),
        isValidityBuffer_(isValidityBuffer),
        pool_(pool) {}

  virtual ~Payload() = default;

  virtual arrow::Status serialize(arrow::io::OutputStream* outputStream) = 0;

  virtual arrow::Result<std::shared_ptr<arrow::Buffer>> readBufferAt(uint32_t index) = 0;

  static arrow::Result<std::pair<int32_t, uint32_t>> readTypeAndRows(arrow::io::InputStream* inputStream);

  Type type() const {
    return type_;
  }

  uint32_t numRows() const {
    return numRows_;
  }

  uint32_t numBuffers() {
    return isValidityBuffer_->size();
  }

  const std::vector<bool>* isValidityBuffer() const {
    return isValidityBuffer_;
  }

  arrow::MemoryPool* pool() const {
    return pool_;
  }

 protected:
  Type type_;
  uint32_t numRows_;
  bool hasComplexType_;
  const std::vector<bool>* isValidityBuffer_;
  arrow::MemoryPool* pool_;
};

// A block represents data to be cached in-memory.
// Can be compressed or uncompressed.
class BlockPayload : public Payload {
 public:
  BlockPayload(
      Type type,
      uint32_t numRows,
      const std::vector<bool>* isValidityBuffer,
      arrow::MemoryPool* pool,
      arrow::util::Codec* codec,
      std::vector<std::shared_ptr<arrow::Buffer>> buffers);

  arrow::Result<std::shared_ptr<arrow::Buffer>> readBufferAt(uint32_t pos) override;

  static arrow::Result<std::unique_ptr<BlockPayload>> fromBuffers(
      Payload::Type type,
      uint32_t numRows,
      std::vector<std::shared_ptr<arrow::Buffer>> buffers,
      const std::vector<bool>* isValidityBuffer,
      arrow::MemoryPool* pool,
      arrow::util::Codec* codec,
      bool reuseBuffers);

  arrow::Status serialize(arrow::io::OutputStream* outputStream) override;

  static arrow::Result<std::vector<std::shared_ptr<arrow::Buffer>>> deserialize(
      arrow::io::InputStream* inputStream,
      const std::shared_ptr<arrow::Schema>& schema,
      const std::shared_ptr<arrow::util::Codec>& codec,
      arrow::MemoryPool* pool,
      uint32_t& numRows);

  static arrow::Result<std::shared_ptr<arrow::Buffer>> readUncompressedBuffer(arrow::io::InputStream* inputStream);

  static arrow::Result<std::shared_ptr<arrow::Buffer>> readCompressedBuffer(
      arrow::io::InputStream* inputStream,
      const std::shared_ptr<arrow::util::Codec>& codec,
      arrow::MemoryPool* pool);

  static arrow::Status mergeCompressed(
      arrow::io::InputStream* inputStream,
      arrow::io::OutputStream* outputStream,
      uint32_t numRows,
      int64_t totalLength) {
    static const Type kType = Type::kUncompressed;
    RETURN_NOT_OK(outputStream->Write(&kType, sizeof(Type)));
    RETURN_NOT_OK(outputStream->Write(&numRows, sizeof(uint32_t)));
    ARROW_ASSIGN_OR_RAISE(auto buffer, inputStream->Read(totalLength));
    RETURN_NOT_OK(outputStream->Write(buffer));
    return arrow::Status::OK();
  }

  static arrow::Status mergeUncompressed(arrow::io::InputStream* inputStream, arrow::ResizableBuffer* output) {
    ARROW_ASSIGN_OR_RAISE(auto input, readUncompressedBuffer(inputStream));
    auto data = output->mutable_data() + output->size();
    auto newSize = output->size() + input->size();
    RETURN_NOT_OK(output->Resize(newSize));
    memcpy(data, input->data(), input->size());
    return arrow::Status::OK();
  }

 private:
  arrow::util::Codec* codec_;
  std::vector<std::shared_ptr<arrow::Buffer>> buffers_;
};

class GroupPayload final : public Payload {
 public:
  GroupPayload(
      Type type,
      uint32_t numRows,
      const std::vector<bool>* isValidityBuffer,
      arrow::MemoryPool* pool,
      arrow::util::Codec* codec,
      std::vector<std::unique_ptr<Payload>> payloads);

  arrow::Status serialize(arrow::io::OutputStream* outputStream) override;

  arrow::Result<std::shared_ptr<arrow::Buffer>> readBufferAt(uint32_t index) override;

 private:
  arrow::util::Codec* codec_;
  std::vector<std::vector<std::shared_ptr<arrow::Buffer>>> buffers_;
  std::vector<uint32_t> bufferNumRows_;
  std::vector<bool> isValidityAllNull_;

  int64_t rawSizeAt(uint32_t index);

  const arrow::Buffer* validityBufferAllTrue();

  arrow::Status writeValidityBuffer(arrow::io::OutputStream* outputStream, uint32_t index);

  arrow::Status writeBuffer(arrow::io::OutputStream* outputStream, uint32_t index);

  arrow::Status serializeUncompressed(arrow::io::OutputStream* outputStream);

  arrow::Status serializeCompressed(arrow::io::OutputStream* outputStream);
};

class UncompressedDiskBlockPayload : public Payload {
 public:
  UncompressedDiskBlockPayload(
      Type type,
      uint32_t numRows,
      const std::vector<bool>* isValidityBuffer,
      arrow::MemoryPool* pool,
      arrow::io::InputStream*& inputStream,
      uint64_t rawSize,
      arrow::util::Codec* codec);

  arrow::Result<std::shared_ptr<arrow::Buffer>> readBufferAt(uint32_t index) override;

  arrow::Status serialize(arrow::io::OutputStream* outputStream) override;

 private:
  arrow::io::InputStream*& inputStream_;
  uint64_t rawSize_;
  arrow::util::Codec* codec_;
  uint32_t readPos_{0};

  arrow::Result<std::shared_ptr<arrow::Buffer>> readUncompressedBuffer();
};

class CompressedDiskBlockPayload : public Payload {
 public:
  CompressedDiskBlockPayload(
      uint32_t numRows,
      const std::vector<bool>* isValidityBuffer,
      arrow::MemoryPool* pool,
      arrow::io::InputStream*& inputStream,
      uint64_t rawSize);

  arrow::Status serialize(arrow::io::OutputStream* outputStream) override;

  arrow::Result<std::shared_ptr<arrow::Buffer>> readBufferAt(uint32_t index) override;

 private:
  arrow::io::InputStream*& inputStream_;
  uint64_t rawSize_;
};
} // namespace gluten