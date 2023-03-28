// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <arrow/util/compression.h>
#include <arrow/util/logging.h>
#include <utils/qpl/qpl_codec.h>
#include <utils/qpl/qpl_job.h>
#include <iostream>
#include <map>

namespace gluten {
namespace qpl {

class QplGzipCodec final : public arrow::util::Codec {
 public:
  explicit QplGzipCodec(qpl_compression_levels compressionLevel)
      : dedicated_(std::make_unique<IAADeflateCodec>(compressionLevel)) {}

  arrow::Result<int64_t>
  Compress(int64_t input_len, const uint8_t* input, int64_t output_buffer_len, uint8_t* output_buffer) override {
    return dedicated_->doCompressData(input, input_len, output_buffer, output_buffer_len);
  }

  arrow::Result<int64_t>
  Decompress(int64_t input_len, const uint8_t* input, int64_t output_buffer_len, uint8_t* output_buffer) override {
    return dedicated_->doDecompressData(input, input_len, output_buffer, output_buffer_len);
  }

  int64_t MaxCompressedLen(int64_t input_len, const uint8_t* ARROW_ARG_UNUSED(input)) override {
    ARROW_DCHECK_GE(input_len, 0);
    /// Aligned with ZLIB
    return ((input_len) + ((input_len) >> 12) + ((input_len) >> 14) + ((input_len) >> 25) + 13);
  }

  arrow::Result<std::shared_ptr<arrow::util::Compressor>> MakeCompressor() override {
    return arrow::Status::NotImplemented("Streaming compression unsupported with QAT");
  }

  arrow::Result<std::shared_ptr<arrow::util::Decompressor>> MakeDecompressor() override {
    return arrow::Status::NotImplemented("Streaming decompression unsupported with QAT");
  }

  arrow::Compression::type compression_type() const override {
    return arrow::Compression::CUSTOM;
  }

  int minimum_compression_level() const override {
    return qpl_level_1;
  }
  int maximum_compression_level() const override {
    return qpl_high_level;
  }
  int default_compression_level() const override {
    return qpl_default_level;
  }

 private:
  std::unique_ptr<IAADeflateCodec> dedicated_;
};

bool SupportsCodec(const std::string& codec) {
  if (std::any_of(qpl_supported_codec.begin(), qpl_supported_codec.end(), [&](const auto& qat_codec) {
        return qat_codec == codec;
      })) {
    return true;
  }
  return false;
}

void EnsureQplCodecRegistered(const std::string& codec) {
  if (codec == "gzip") {
    arrow::util::RegisterCustomCodec([](int) { return MakeDefaultQplGZipCodec(); });
  }
}

std::unique_ptr<arrow::util::Codec> MakeQplGZipCodec(int compressionLevel) {
  auto qplCompressionLevel = static_cast<qpl_compression_levels>(compressionLevel);
  return std::unique_ptr<arrow::util::Codec>(new QplGzipCodec(qplCompressionLevel));
}

std::unique_ptr<arrow::util::Codec> MakeDefaultQplGZipCodec() {
  return MakeQplGZipCodec(qpl_default_level);
}

} // namespace qpl
} // namespace gluten