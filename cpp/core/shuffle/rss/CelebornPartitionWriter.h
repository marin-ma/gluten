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

#include <arrow/io/api.h>
#include <arrow/memory_pool.h>

#include "shuffle/rss/RemotePartitionWriter.h"
#include "shuffle/rss/RssClient.h"
#include "utils/macros.h"

namespace gluten {

class CelebornPartitionWriter final : public RemotePartitionWriter {
 public:
  CelebornPartitionWriter(
      uint32_t numPartitions,
      PartitionWriterOptions options,
      arrow::MemoryPool* pool,
      std::shared_ptr<RssClient> celebornClient)
      : RemotePartitionWriter(numPartitions, std::move(options), pool), celebornClient_(celebornClient) {
    init();
  }

  arrow::Status evict(
      uint32_t partitionId,
      uint32_t numRows,
      std::vector<std::shared_ptr<arrow::Buffer>> buffers,
      const std::vector<bool>* isValidityBuffer,
      bool reuseBuffers, /* unused */
      Evict::type evictType /* unused */,
      bool hasComplexType /* unused */) override;

  arrow::Status reclaimFixedSize(int64_t size, int64_t* actual) override;

  arrow::Status stop(ShuffleWriterMetrics* metrics) override;

 private:
  void init();

  std::shared_ptr<RssClient> celebornClient_;

  std::vector<int64_t> bytesEvicted_;
  std::vector<int64_t> rawPartitionLengths_;
};
} // namespace gluten
