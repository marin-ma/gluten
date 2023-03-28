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

#include "utils/qpl/qpl_job.h"

#include <arrow/util/logging.h>
#include <iostream>

namespace gluten {
namespace qpl {

class IAAJob {
 public:
  ~IAAJob() {
    for (const auto& job : jobs_) {
      if (job != nullptr) {
        qpl_fini_job(job);
        delete[] job;
      }
    }
  }

  [[nodiscard]] bool Initialized() const {
    return initialized_;
  }

  void InitJobs() {
    if (!initialized_) {
      jobs_.resize(2);
      InitJob(qpl_path_hardware);
      InitJob(qpl_path_software);
      initialized_ = true;
    }
  }

  qpl_job* GetJob(qpl_path_t execution_path) {
    return jobs_[execution_path];
  }

 private:
  struct JobDeleter {
    void operator()(qpl_job* ptr) const {
      delete[] reinterpret_cast<char*>(ptr);
    }
  };

  void InitJob(qpl_path_t execution_path) {
    uint32_t size;
    qpl_status status = qpl_get_job_size(execution_path, &size);
    if (status != QPL_STS_OK) {
      jobs_[execution_path] = nullptr;
      return;
    }

    try {
      jobs_[execution_path] = reinterpret_cast<qpl_job*>(new char[size]);
    } catch (std::bad_alloc& e) {
      jobs_[execution_path] = nullptr;
      return;
    }
    status = qpl_init_job(execution_path, jobs_[execution_path]);
    if (status != QPL_STS_OK) {
      jobs_[execution_path] = nullptr;
    }
  }

  bool initialized_{false};
  std::vector<qpl_job*> jobs_;
};

int64_t
IAADeflateCodec::doCompressData(const uint8_t* source, uint32_t source_size, uint8_t* dest, uint32_t dest_size) {
  iaaJob_.InitJobs();
  int64_t status = -1;
  for (const auto& path : paths) {
    qpl_job* job = iaaJob_.GetJob(path);

    job->op = qpl_op_compress;
    job->next_in_ptr = const_cast<uint8_t*>(source);
    job->next_out_ptr = dest;
    job->available_in = source_size;
    job->level = compressionLevel_;
    job->available_out = dest_size;
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_DYNAMIC_HUFFMAN | QPL_FLAG_LAST | QPL_FLAG_OMIT_VERIFY;

    try {
      status = qpl_submit_job(job);
      if (QPL_STS_OK == status) {
        do {
          status = qpl_check_job(job);
        } while (QPL_STS_BEING_PROCESSED == status);
        if (QPL_STS_OK != status) {
          throw GlutenException("qpl_check_job() failed with status " + std::to_string(status));
        }
      } else {
        throw GlutenException("qpl_submit_job() failed with status " + std::to_string(status));
      }
      status = qpl_wait_job(job);
      if (status == QPL_STS_OK) {
        auto compressed_size = job->total_out;
        if (qpl_fini_job(job) != QPL_STS_OK) {
          throw GlutenException("qpl_fini_job() failed with status " + std::to_string(status));
        }
        return compressed_size;
      } else {
        throw GlutenException("qpl_wait_job() failed with status " + std::to_string(status));
      }
    } catch (const GlutenException& e) {
      if (path == qpl_path_hardware) {
        ARROW_LOG(WARNING)
            << "DeflateQpl HW codec failed, falling back to SW codec. (Details: doCompressData->qpl_execute_job with error code: "
            << status << " - please refer to qpl_status in ./contrib/qpl/include/qpl/c_api/status.h)";
      }
      qpl_fini_job(job);
    }
  }
  return -1;
}

int64_t IAADeflateCodec::doDecompressData(
    const uint8_t* source,
    uint32_t source_size,
    uint8_t* dest,
    uint32_t uncompressed_size) {
  iaaJob_.InitJobs();
  int64_t status = -1;
  for (const auto& path : paths) {
    qpl_job* job = iaaJob_.GetJob(path);
    // Performing a decompression operation
    job->op = qpl_op_decompress;
    job->next_in_ptr = const_cast<uint8_t*>(source);
    job->next_out_ptr = dest;
    job->available_in = source_size;
    job->available_out = uncompressed_size;
    job->flags = QPL_FLAG_FIRST | QPL_FLAG_LAST;

    status = qpl_execute_job(job);
    if (status == QPL_STS_OK) {
      auto decompressed_size = job->total_out;
      return decompressed_size;
    }
    if (path == qpl_path_hardware) {
      ARROW_LOG(WARNING)
          << "DeflateQpl HW codec failed, falling back to SW codec. (Details: doDeCompressData->qpl_execute_job with error code: "
          << status << " - please refer to qpl_status in ./contrib/qpl/include/qpl/c_api/status.h)";
    }
  }
  return status;
}

thread_local IAAJob IAADeflateCodec::iaaJob_;

} // namespace qpl
} // namespace gluten
