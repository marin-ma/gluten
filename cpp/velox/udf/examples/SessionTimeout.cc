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

#include <array>
#include <optional>
#include <vector>

#include "UdfCommon.h"
#include "velox/exec/WindowFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/type/Timestamp.h"

#include "udf/Udwf.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;

namespace {

static const char* kBigInt = "bigint";
static const char* kInteger = "int";
static const char* kTimestamp = "timestamp";

class SessionTimeoutFunction : public exec::WindowFunction {
 public:
  explicit SessionTimeoutFunction(
      const std::vector<exec::WindowFunctionArg>& args,
      const TypePtr& resultType,
      memory::MemoryPool* pool)
      : WindowFunction(resultType, pool, nullptr) {
    VELOX_CHECK_EQ(args.size(), 2, "SessionTimeout window function requires exactly 2 arguments");
    VELOX_CHECK(args[0].index.has_value(), "timestamp argument must reference an input column");
    VELOX_CHECK(args[1].index.has_value(), "gap_seconds argument must reference an input column");

    timestampIndex_ = args[0].index.value();
    gapSecondsIndex_ = args[1].index.value();
    timestampType_ = args[0].type;
    gapSecondsType_ = args[1].type;
  }

  void resetPartition(const exec::WindowPartition* partition) override {
    partition_ = partition;
    sessionNumber_ = 0;
    lastTimestamp_.reset();
  }

  void apply(
      const BufferPtr& /*peerGroupStarts*/,
      const BufferPtr& /*peerGroupEnds*/,
      const BufferPtr& /*frameStarts*/,
      const BufferPtr& /*frameEnds*/,
      const SelectivityVector& validRows,
      vector_size_t resultOffset,
      const VectorPtr& result) override {
    VELOX_CHECK_NOT_NULL(partition_, "Partition must be initialized before apply");

    const auto numRows = validRows.end();
    auto flatResult = result->asFlatVector<int64_t>();
    auto* rawValues = flatResult->mutableRawValues();

    // Mark invalid rows (e.g. empty frame rows) as null.
    for (vector_size_t i = 0; i < numRows; ++i) {
      if (!validRows.isValid(i)) {
        flatResult->setNull(resultOffset + i, true);
      }
    }

    std::vector<vector_size_t> rowNumbers(numRows);
    for (vector_size_t i = 0; i < numRows; ++i) {
      rowNumbers[i] = resultOffset + i;
    }
    auto rowRange = folly::Range<const vector_size_t*>(rowNumbers.data(), rowNumbers.size());

    auto timestampVector = BaseVector::create(timestampType_, numRows, pool());
    auto gapSecondsVector = BaseVector::create(gapSecondsType_, numRows, pool());

    partition_->extractColumn(timestampIndex_, rowRange, 0, timestampVector);
    partition_->extractColumn(gapSecondsIndex_, rowRange, 0, gapSecondsVector);

    validRows.applyToSelected([&](vector_size_t i) {
      if (timestampVector->isNullAt(i) || gapSecondsVector->isNullAt(i)) {
        flatResult->setNull(resultOffset + i, true);
        return;
      }

      int64_t currentTimestamp = 0;
      if (timestampVector->typeKind() == TypeKind::BIGINT) {
        currentTimestamp = timestampVector->as<SimpleVector<int64_t>>()->valueAt(i);
      } else if (timestampVector->typeKind() == TypeKind::TIMESTAMP) {
        currentTimestamp = timestampVector->as<SimpleVector<Timestamp>>()->valueAt(i).toMillis();
      } else {
        VELOX_FAIL("Unsupported timestamp argument type: {}", timestampVector->type()->toString());
      }

      int64_t gapSeconds = 0;
      if (gapSecondsVector->typeKind() == TypeKind::INTEGER) {
        gapSeconds = gapSecondsVector->as<SimpleVector<int32_t>>()->valueAt(i);
      } else if (gapSecondsVector->typeKind() == TypeKind::BIGINT) {
        gapSeconds = gapSecondsVector->as<SimpleVector<int64_t>>()->valueAt(i);
      } else {
        VELOX_FAIL("Unsupported gap_seconds argument type: {}", gapSecondsVector->type()->toString());
      }

      if (!lastTimestamp_.has_value()) {
        sessionNumber_ = 1;
      } else {
        const int64_t diffMillis = currentTimestamp - lastTimestamp_.value();
        if (diffMillis > gapSeconds * 1000) {
          ++sessionNumber_;
        }
      }

      lastTimestamp_ = currentTimestamp;
      flatResult->setNull(resultOffset + i, false);
      rawValues[resultOffset + i] = sessionNumber_;
    });
  }

 private:
  const exec::WindowPartition* partition_{nullptr};
  column_index_t timestampIndex_;
  column_index_t gapSecondsIndex_;
  TypePtr timestampType_;
  TypePtr gapSecondsType_;
  int64_t sessionNumber_{0};
  std::optional<int64_t> lastTimestamp_;
};

class SessionTimeoutWindowRegisterer final : public gluten::UdwfRegisterer {
 public:
  int getNumUdwf() override {
    return 4;
  }

  void populateUdwfEntries(int& index, gluten::UdwfEntry* windowEntries) override {
    for (const auto& args : argTypeSets_) {
      windowEntries[index++] = {name_.c_str(), kBigInt, 2, args, false, true};
    }
  }

  void registerSignatures() override {
    if (registered_) {
      return;
    }

    std::vector<exec::FunctionSignaturePtr> signatures{
        exec::FunctionSignatureBuilder().returnType("bigint").argumentType("bigint").argumentType("integer").build(),
        exec::FunctionSignatureBuilder().returnType("bigint").argumentType("bigint").argumentType("bigint").build(),
        exec::FunctionSignatureBuilder().returnType("bigint").argumentType("timestamp").argumentType("integer").build(),
        exec::FunctionSignatureBuilder().returnType("bigint").argumentType("timestamp").argumentType("bigint").build(),
    };

    auto windowFunctionFactory = [](const std::vector<exec::WindowFunctionArg>& args,
                                    const TypePtr& resultType,
                                    bool /*ignoreNulls*/,
                                    memory::MemoryPool* pool,
                                    HashStringAllocator* /*stringAllocator*/,
                                    const core::QueryConfig& /*queryConfig*/) -> std::unique_ptr<exec::WindowFunction> {
      return std::make_unique<SessionTimeoutFunction>(args, resultType, pool);
    };

    exec::registerWindowFunction(
        name_,
        std::move(signatures),
        {exec::WindowFunction::ProcessMode::kRows, false},
        std::move(windowFunctionFactory));

    registered_ = true;
  }

 private:
  const std::string name_{"com.adobe.platform.query.spark.sql.helpers.SessionTimeout"};
  const char* argBigIntInt_[2] = {kBigInt, kInteger};
  const char* argBigIntBigInt_[2] = {kBigInt, kBigInt};
  const char* argTimestampInt_[2] = {kTimestamp, kInteger};
  const char* argTimestampBigInt_[2] = {kTimestamp, kBigInt};
  std::array<const char**, 4> argTypeSets_{argBigIntInt_, argBigIntBigInt_, argTimestampInt_, argTimestampBigInt_};

  bool registered_{false};
};

std::vector<std::shared_ptr<gluten::UdwfRegisterer>>& globalRegisters() {
  static std::vector<std::shared_ptr<gluten::UdwfRegisterer>> registerers;
  return registerers;
}

void setupRegisterers() {
  static bool inited = false;
  if (inited) {
    return;
  }
  auto& registerers = globalRegisters();
  registerers.push_back(std::make_shared<SessionTimeoutWindowRegisterer>());
  inited = true;
}

} // namespace

DEFINE_GET_NUM_UDWF {
  setupRegisterers();

  int numUdf = 0;
  for (const auto& registerer : globalRegisters()) {
    numUdf += registerer->getNumUdwf();
  }
  return numUdf;
}

DEFINE_GET_UDWF_ENTRIES {
  setupRegisterers();

  int index = 0;
  for (const auto& registerer : globalRegisters()) {
    registerer->populateUdwfEntries(index, udwfEntries);
  }
}

DEFINE_REGISTER_UDWF {
  setupRegisterers();

  for (const auto& registerer : globalRegisters()) {
    registerer->registerSignatures();
  }
}
