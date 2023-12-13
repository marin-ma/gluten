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

#include "shuffle/Spill.h"

namespace gluten {

bool GroupSpill::hasNextPayload(uint32_t partitionId) {
  return partitionToPayloads_.find(partitionId) != partitionToPayloads_.end() &&
      !partitionToPayloads_[partitionId].empty();
}

std::unique_ptr<Payload> GroupSpill::nextPayload(uint32_t partitionId) {
  if (!hasNextPayload(partitionId)) {
    return nullptr;
  }
  auto front = std::move(partitionToPayloads_[partitionId].front());
  partitionToPayloads_[partitionId].pop_front();
  return front;
}

GroupSpill::GroupSpill(
    uint32_t numPartitions,
    uint32_t batchSize,
    uint32_t compressionThreshold,
    arrow::util::Codec* codec,
    std::unordered_map<uint32_t, std::list<std::unique_ptr<Payload>>> partitionToPayloads)
    : Spill(numPartitions),
      partitionToPayloads_(std::move(partitionToPayloads)),
      batchSize_(batchSize),
      compressionThreshold_(compressionThreshold),
      codec_(codec) {}

void GroupSpill::grouping(Payload::Type groupPayloadType) {
  std::unordered_map<uint32_t, std::list<std::unique_ptr<Payload>>> partitionToGroupPayloads;
  for (auto pid = 0; pid < numPartitions_; ++pid) {
    if (partitionToPayloads_.find(pid) == partitionToPayloads_.end()) {
      continue;
    }
    partitionToGroupPayloads[pid] = std::list<std::unique_ptr<Payload>>{};
    auto& payloads = partitionToPayloads_[pid];
    uint32_t rows = 0;
    std::vector<std::unique_ptr<Payload>> toBeMerged{};
    while (!payloads.empty()) {
      auto payload = std::move(payloads.front());
      payloads.pop_front();
      if (payload->type() == Payload::Type::kUncompressed) {
        // If payload is uncompressed, check whether it can be appended to the last group.
        // If total rows exceeds configured batch size, create a new group from toBeMerged.
        if (!toBeMerged.empty() && rows + payload->numRows() > batchSize_) {
          // TODO: Add compression threshold to force uncompress.
          partitionToGroupPayloads[pid].push_back(createGroupPayload(groupPayloadType, rows, std::move(toBeMerged)));
          toBeMerged.clear();
        }
        rows += payload->numRows();
        toBeMerged.push_back(std::move(payload));
        continue;
      }
      // Current payload is compressed, which means it cannot be merged with previous ones.
      // Create a new group from toBeMerged.
      if (!toBeMerged.empty()) {
        partitionToGroupPayloads[pid].push_back(createGroupPayload(groupPayloadType, rows, std::move(toBeMerged)));
        toBeMerged.clear();
      }
      partitionToGroupPayloads[pid].push_back(std::move(payload));
    }
    // Create a new group for the remaining payloads in toBeMerged, if any.
    // TODO: The last payload can be merged with next spill/partition buffers.
    if (!toBeMerged.empty()) {
      partitionToGroupPayloads[pid].push_back(createGroupPayload(groupPayloadType, rows, std::move(toBeMerged)));
    }
  }
  partitionToPayloads_ = std::move(partitionToGroupPayloads);
}

std::unique_ptr<Payload> GroupSpill::createGroupPayload(
    Payload::Type groupPayloadType,
    uint32_t& rows,
    std::vector<std::unique_ptr<Payload>> toBeMerged) {
  // If there's only one payload in toBeMerged, return it.
  if (toBeMerged.size() == 1) {
    rows = 0;
    auto payload = std::move(toBeMerged.back());
    toBeMerged.pop_back();
    return payload;
  }
  auto isValidityBuffer = toBeMerged.back()->isValidityBuffer();
  auto pool = toBeMerged.back()->pool();
  auto type = rows < compressionThreshold_ ? Payload::Type::kUncompressed : groupPayloadType;
  auto payload = std::make_unique<GroupPayload>(type, rows, isValidityBuffer, pool, codec_, std::move(toBeMerged));
  toBeMerged.clear();
  rows = 0;
  return payload;
}

DiskSpill::DiskSpill(uint32_t numPartitions, DiskSpill::SpillType type, const std::string& spillFile)
    : Spill(numPartitions), type_(type), spillFile_(spillFile) {}

DiskSpill::~DiskSpill() {
  if (is_) {
    (void)is_->Close();
  }
}

DiskSpill::DiskSpill(uint32_t numPartitions) : Spill(numPartitions) {}

bool DiskSpill::hasNextPayload(uint32_t partitionId) {
  return !partitionPayloads_.empty() && partitionPayloads_.front().partitionId == partitionId;
}

std::unique_ptr<Payload> DiskSpill::nextPayload(uint32_t partitionId) {
  openSpillFile();
  if (!hasNextPayload(partitionId)) {
    return nullptr;
  }
  auto payload = std::move(partitionPayloads_.front().payload);
  partitionPayloads_.pop_front();
  return payload;
}

void DiskSpill::insertPayload(
    uint32_t partitionId,
    Payload::Type payloadType,
    uint32_t numRows,
    const std::vector<bool>* isValidityBuffer,
    uint64_t rawSize,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec) {
  // TODO: Support merging uncompressed payloads when codec is null.
  if (payloadType == Payload::Type::kCompressed) {
    partitionPayloads_.push_back(
        {partitionId, std::make_unique<CompressedDiskBlockPayload>(numRows, isValidityBuffer, pool, rawIs_, rawSize)});
  } else {
    partitionPayloads_.push_back(
        {partitionId,
         std::make_unique<UncompressedDiskBlockPayload>(
             payloadType, numRows, isValidityBuffer, pool, rawIs_, rawSize, codec)});
  }
}

void DiskSpill::openSpillFile() {
  if (!is_) {
    GLUTEN_ASSIGN_OR_THROW(is_, arrow::io::MemoryMappedFile::Open(spillFile_, arrow::io::FileMode::READ));
    rawIs_ = is_.get();
  }
}
} // namespace gluten