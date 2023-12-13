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

#include <filesystem>
#include <random>
#include <thread>

#include "shuffle/LocalPartitionWriter.h"
#include "shuffle/Payload.h"
#include "shuffle/Utils.h"
#include "utils/DebugOut.h"
#include "utils/Timer.h"

namespace gluten {

class LocalPartitionWriter::LocalEvictor : public Evictor {
 public:
  LocalEvictor(
      uint32_t numPartitions,
      ShuffleWriterOptions* options,
      arrow::MemoryPool* pool,
      arrow::util::Codec* codec,
      const std::string& spillFile)
      : Evictor(options), numPartitions_(numPartitions), pool_(pool), codec_(codec), spillFile_(spillFile) {}

  static arrow::Result<std::unique_ptr<LocalEvictor>> create(
      uint32_t numPartitions,
      ShuffleWriterOptions* options,
      arrow::MemoryPool* pool,
      arrow::util::Codec* codec,
      const std::string& spillFile,
      Evictor::Type evictType);

  virtual Type evictType() = 0;

  virtual arrow::Status spill() = 0;

 protected:
  uint32_t numPartitions_;
  arrow::MemoryPool* pool_;
  arrow::util::Codec* codec_;
  std::string spillFile_;
};

class CacheEvictor final : public LocalPartitionWriter::LocalEvictor {
 public:
  CacheEvictor(
      uint32_t numPartitions,
      ShuffleWriterOptions* options,
      arrow::MemoryPool* pool,
      arrow::util::Codec* codec,
      const std::string& spillFile)
      : LocalPartitionWriter::LocalEvictor(numPartitions, options, pool, codec, spillFile) {}

  arrow::Status evict(uint32_t partitionId, std::unique_ptr<Payload> payload) override {
    if (partitionCachedPayload_.find(partitionId) == partitionCachedPayload_.end()) {
      partitionCachedPayload_.emplace(partitionId, std::list<std::unique_ptr<Payload>>{});
    }
    partitionCachedPayload_[partitionId].push_back(std::move(payload));
    return arrow::Status::OK();
  }

  arrow::Status spill() override {
    ScopedTimer timer(evictTime_);

    ARROW_ASSIGN_OR_RAISE(auto groupSpill, createGroupSpill(Payload::Type::kUncompressed));

    ARROW_ASSIGN_OR_RAISE(auto os, arrow::io::FileOutputStream::Open(spillFile_, true));
    ARROW_ASSIGN_OR_RAISE(auto start, os->Tell());
    for (uint32_t pid = 0; pid < numPartitions_; ++pid) {
      while (auto payload = groupSpill->nextPayload(pid)) {
        RETURN_NOT_OK(payload->serialize(os.get()));
        ARROW_ASSIGN_OR_RAISE(auto end, os->Tell());

        if (!diskSpill_) {
          diskSpill_ = std::make_unique<DiskSpill>(numPartitions_, DiskSpill::SpillType::kBatchedSpill, spillFile_);
        }
        diskSpill_->insertPayload(
            pid,
            payload->type(),
            payload->numRows(),
            payload->isValidityBuffer(),
            end - start,
            pool_,
            codec_);
        start = end;
      }
    }
    RETURN_NOT_OK(os->Close());
    if (!diskSpill_) {
      return arrow::Status::Invalid("CacheEvictor no cached data spilled to disk.");
    }
    return arrow::Status::OK();
  }

  arrow::Result<std::unique_ptr<Spill>> finish() override {
    if (finished_) {
      return arrow::Status::Invalid("Calling finish() on a finished CacheEvictor.");
    }
    finished_ = true;

    if (diskSpill_) {
      return std::move(diskSpill_);
    }
    // No spill on disk. Delete the empty spill file.
    std::filesystem::remove(spillFile_);

    ARROW_ASSIGN_OR_RAISE(auto groupSpill, createGroupSpill(Payload::Type::kCompressed));
    return std::move(groupSpill);
  }

  Type evictType() override {
    return Type::kCache;
  }

 private:
  bool finished_{false};
  std::unique_ptr<DiskSpill> diskSpill_{nullptr};
  std::unordered_map<uint32_t, std::list<std::unique_ptr<Payload>>> partitionCachedPayload_;

  arrow::Result<std::unique_ptr<GroupSpill>> createGroupSpill(Payload::Type groupPayloadType) {
    if (partitionCachedPayload_.empty()) {
      return arrow::Status::Invalid("CacheEvictor has empty cached payloads.");
    }
    auto spill = std::make_unique<GroupSpill>(
        numPartitions_,
        options_->buffer_size,
        options_->compression_threshold,
        codec_,
        std::move(partitionCachedPayload_));
    partitionCachedPayload_.clear();

    if (options_->supportsMerging) {
      spill->grouping(Payload::Type::kUncompressed);
    }
    return spill;
  }
};

class SpillEvictor final : public LocalPartitionWriter::LocalEvictor {
 public:
  SpillEvictor(
      uint32_t numPartitions,
      ShuffleWriterOptions* options,
      arrow::MemoryPool* pool,
      arrow::util::Codec* codec,
      const std::string& spillFile)
      : LocalPartitionWriter::LocalEvictor(numPartitions, options, pool, codec, spillFile) {}

  arrow::Status evict(uint32_t partitionId, std::unique_ptr<Payload> payload) override {
    ScopedTimer timer(evictTime_);
    if (!opened_) {
      opened_ = true;
      ARROW_ASSIGN_OR_RAISE(os_, arrow::io::FileOutputStream::Open(spillFile_, true));
      spill_ = std::make_unique<DiskSpill>(numPartitions_, DiskSpill::SpillType::kSequentialSpill, spillFile_);
    }

    ARROW_ASSIGN_OR_RAISE(auto start, os_->Tell());
    RETURN_NOT_OK(payload->serialize(os_.get()));
    ARROW_ASSIGN_OR_RAISE(auto end, os_->Tell());
    DEBUG_OUT << "Spilled partition " << partitionId << " file start: " << start << ", file end: " << end
              << ", file: " << spillFile_ << std::endl;
    spill_->insertPayload(
        partitionId,
        payload->type(),
        payload->numRows(),
        payload->isValidityBuffer(),
        end - start,
        pool_,
        codec_);
    return arrow::Status::OK();
  }

  arrow::Result<std::unique_ptr<Spill>> finish() override {
    if (finished_) {
      return arrow::Status::Invalid("Calling finish() on a finished SpillEvictor.");
    }
    finished_ = true;

    if (!opened_) {
      return arrow::Status::Invalid("SpillEvictor has no data spilled.");
    }
    RETURN_NOT_OK(os_->Close());
    return std::move(spill_);
  }

  arrow::Status spill() override {
    return arrow::Status::OK();
  }

  Type evictType() override {
    return Type::kSpill;
  }

 private:
  bool opened_{false};
  bool finished_{false};
  std::unique_ptr<DiskSpill> spill_{nullptr};
  std::shared_ptr<arrow::io::FileOutputStream> os_;
};

arrow::Result<std::unique_ptr<LocalPartitionWriter::LocalEvictor>> LocalPartitionWriter::LocalEvictor::create(
    uint32_t numPartitions,
    ShuffleWriterOptions* options,
    arrow::MemoryPool* pool,
    arrow::util::Codec* codec,
    const std::string& spillFile,
    Evictor::Type evictType) {
  switch (evictType) {
    case Evictor::Type::kSpill:
      return std::make_unique<SpillEvictor>(numPartitions, options, pool, codec, spillFile);
    case Evictor::Type::kCache:
      return std::make_unique<CacheEvictor>(numPartitions, options, pool, codec, spillFile);
    default:
      return arrow::Status::Invalid("Cannot create Evictor from type Evictor::Type::kStop.");
  }
}

LocalPartitionWriter::LocalPartitionWriter(
    uint32_t numPartitions,
    const std::string& dataFile,
    const std::vector<std::string>& localDirs,
    ShuffleWriterOptions* options)
    : PartitionWriter(numPartitions, options), dataFile_(dataFile), localDirs_(localDirs) {
  init();
}

std::string LocalPartitionWriter::nextSpilledFileDir() {
  auto spilledFileDir = getSpilledShuffleFileDir(localDirs_[dirSelection_], subDirSelection_[dirSelection_]);
  subDirSelection_[dirSelection_] = (subDirSelection_[dirSelection_] + 1) % options_->num_sub_dirs;
  dirSelection_ = (dirSelection_ + 1) % localDirs_.size();
  return spilledFileDir;
}

arrow::Status LocalPartitionWriter::openDataFile() {
  // open data file output stream
  std::shared_ptr<arrow::io::FileOutputStream> fout;
  ARROW_ASSIGN_OR_RAISE(fout, arrow::io::FileOutputStream::Open(dataFile_));
  if (options_->buffered_write) {
    // Output stream buffer is neither partition buffer memory nor ipc memory.
    ARROW_ASSIGN_OR_RAISE(dataFileOs_, arrow::io::BufferedOutputStream::Create(16384, options_->memory_pool, fout));
  } else {
    dataFileOs_ = fout;
  }
  return arrow::Status::OK();
}

arrow::Status LocalPartitionWriter::clearResource() {
  RETURN_NOT_OK(dataFileOs_->Close());
  // When buffered_write = true, dataFileOs_->Close doesn't release underlying buffer.
  dataFileOs_.reset();
  return arrow::Status::OK();
}

void LocalPartitionWriter::init() {
  partitionLengths_.resize(numPartitions_, 0);
  rawPartitionLengths_.resize(numPartitions_, 0);
  fs_ = std::make_shared<arrow::fs::LocalFileSystem>();

  // Shuffle the configured local directories. This prevents each task from using the same directory for spilled files.
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::shuffle(localDirs_.begin(), localDirs_.end(), engine);
  subDirSelection_.assign(localDirs_.size(), 0);
}

arrow::Status LocalPartitionWriter::mergeSpills(uint32_t partitionId) {
  for (const auto& spill : spillResults_) {
    // Read if partition exists in the spilled file and write to the final file.
    while (auto payload = spill->nextPayload(partitionId)) {
      RETURN_NOT_OK(payload->serialize(dataFileOs_.get()));
    }
  }
  return arrow::Status::OK();
}

arrow::Status LocalPartitionWriter::stop(ShuffleWriterMetrics* metrics) {
  if (stopped_) {
    return arrow::Status::OK();
  }
  stopped_ = true;

  // Open final file.
  // If options_.buffered_write is set, it will acquire 16KB memory that might trigger spill.
  RETURN_NOT_OK(openDataFile());

  auto writeTimer = Timer();
  writeTimer.start();

  int64_t endInFinalFile = 0;
  // Write cached batches.
  if (evictor_) {
    ARROW_RETURN_IF(evictor_->evictType() != LocalEvictor::Type::kCache, arrow::Status::Invalid("Unclosed evictor."));
    ARROW_ASSIGN_OR_RAISE(auto groupingSpill, evictor_->finish());
    spillResults_.push_back(std::move(groupingSpill));
  }
  // Iterator over pid.
  for (auto pid = 0; pid < numPartitions_; ++pid) {
    // Record start offset.
    auto startInFinalFile = endInFinalFile;
    // Iterator over all spilled files.
    RETURN_NOT_OK(mergeSpills(pid));
    ARROW_ASSIGN_OR_RAISE(endInFinalFile, dataFileOs_->Tell());
    if (endInFinalFile != startInFinalFile && options_->write_eos) {
      // Write EOS if any payload written.
      int64_t bytes;
      RETURN_NOT_OK(writeEos(dataFileOs_.get(), &bytes));
      endInFinalFile += bytes;
    }
    partitionLengths_[pid] = endInFinalFile - startInFinalFile;
  }

  for (const auto& spill : spillResults_) {
    for (auto pid = 0; pid < numPartitions_; ++pid) {
      if (spill->hasNextPayload(pid)) {
        return arrow::Status::Invalid("Merging from spill is not exhausted.");
      }
    }
  }

  writeTimer.stop();
  writeTime_ = writeTimer.realTimeUsed();
  ARROW_ASSIGN_OR_RAISE(totalBytesWritten_, dataFileOs_->Tell());

  // Close Final file, Clear buffered resources.
  RETURN_NOT_OK(clearResource());
  // Populate shuffle writer metrics.
  RETURN_NOT_OK(populateMetrics(metrics));
  return arrow::Status::OK();
}

arrow::Status LocalPartitionWriter::requestEvict(Evictor::Type evictType) {
  if (evictor_ && evictor_->evictType() == evictType) {
    return arrow::Status::OK();
  }
  RETURN_NOT_OK(spill());

  ARROW_ASSIGN_OR_RAISE(auto spilledFile, createTempShuffleFile(nextSpilledFileDir()));
  ARROW_ASSIGN_OR_RAISE(
      evictor_,
      LocalEvictor::create(numPartitions_, options_, payloadPool_.get(), codec_.get(), spilledFile, evictType));
  return arrow::Status::OK();
}

arrow::Status LocalPartitionWriter::spill() {
  if (evictor_) {
    spillResults_.emplace_back();
    RETURN_NOT_OK(evictor_->spill());
    ARROW_ASSIGN_OR_RAISE(spillResults_.back(), evictor_->finish());
    evictTime_ += evictor_->getEvictTime();
    evictor_ = nullptr;
  }
  return arrow::Status::OK();
}

arrow::Status LocalPartitionWriter::evict(
    uint32_t partitionId,
    uint32_t numRows,
    std::vector<std::shared_ptr<arrow::Buffer>> buffers,
    const std::vector<bool>* isValidityBuffer,
    bool reuseBuffers,
    Evictor::Type evictType) {
  rawPartitionLengths_[partitionId] += getBufferSize(buffers);
  Payload::Type payloadType = (codec_ && evictType == Evictor::kCache && numRows >= options_->compression_threshold)
      ? Payload::Type::kCompressed
      : Payload::Type::kUncompressed;
  ARROW_ASSIGN_OR_RAISE(
      auto payload,
      BlockPayload::fromBuffers(
          payloadType,
          numRows,
          std::move(buffers),
          isValidityBuffer,
          payloadPool_.get(),
          codec_ ? codec_.get() : nullptr,
          reuseBuffers));
  RETURN_NOT_OK(requestEvict(evictType));
  RETURN_NOT_OK(evictor_->evict(partitionId, std::move(payload)));
  return arrow::Status::OK();
}

arrow::Status LocalPartitionWriter::populateMetrics(ShuffleWriterMetrics* metrics) {
  metrics->totalCompressTime += compressTime_;
  metrics->totalEvictTime += evictTime_;
  metrics->totalWriteTime += writeTime_;
  metrics->totalBytesEvicted += totalBytesEvicted_;
  metrics->totalBytesWritten += totalBytesWritten_;
  metrics->partitionLengths = std::move(partitionLengths_);
  metrics->rawPartitionLengths = std::move(rawPartitionLengths_);
  return arrow::Status::OK();
}
} // namespace gluten
