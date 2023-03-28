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

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include <qpl/qpl.h>
#include "utils/exception.h"

namespace gluten {
namespace qpl {

class IAAJob;

class IAADeflateCodec {
 public:
  explicit IAADeflateCodec(qpl_compression_levels compressionLevel) : compressionLevel_(compressionLevel){};
  uint32_t doCompressData(const uint8_t* source, uint32_t source_size, uint8_t* dest, uint32_t dest_size);
  uint32_t doDecompressData(const uint8_t* source, uint32_t source_size, uint8_t* dest, uint32_t uncompressed_size);

 private:
  qpl_compression_levels compressionLevel_ = qpl_default_level;

  static thread_local IAAJob iaaJob_;
  static constexpr auto paths = {qpl_path_hardware, qpl_path_software};
};

} // namespace qpl
} // namespace gluten