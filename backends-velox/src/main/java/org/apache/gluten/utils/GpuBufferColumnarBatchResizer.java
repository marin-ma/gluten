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
package org.apache.gluten.utils;

import org.apache.gluten.backendsapi.BackendsApiManager;
import org.apache.gluten.runtime.Runtime;
import org.apache.gluten.runtime.Runtimes;
import org.apache.gluten.vectorized.ColumnarBatchInIterator;
import org.apache.gluten.vectorized.ColumnarBatchOutIterator;

import org.apache.spark.sql.execution.metric.SQLMetric;
import org.apache.spark.sql.vectorized.ColumnarBatch;

import java.util.Iterator;

public final class GpuBufferColumnarBatchResizer {
  private static class ResizeBatchOutIterator extends ColumnarBatchOutIterator {
    GpuBufferBatchResizerJniWrapper jniWrapper;
    SQLMetric blockingTime;
    SQLMetric resizeTime;

    ResizeBatchOutIterator(
        GpuBufferBatchResizerJniWrapper jniWrapper,
        Runtime runtime,
        long iterHandle,
        SQLMetric blockingTime,
        SQLMetric resizeTime) {
      super(runtime, iterHandle);
      this.jniWrapper = jniWrapper;
      this.blockingTime = blockingTime;
      this.resizeTime = resizeTime;
    }

    @Override
    public void close0() {
      long[] blockingAndResizeTime = jniWrapper.getBlockingAndResizeTime(itrHandle());
      this.blockingTime.add(blockingAndResizeTime[0]);
      this.resizeTime.add(blockingAndResizeTime[1]);
      super.close0();
    }
  }

  public static ColumnarBatchOutIterator create(
      int minOutputBatchSize,
      Iterator<ColumnarBatch> in,
      SQLMetric blockingTime,
      SQLMetric resizeTime) {
    final Runtime runtime =
        Runtimes.contextInstance(
            BackendsApiManager.getBackendName(), "GpuBufferColumnarBatchResizer");
    GpuBufferBatchResizerJniWrapper jniWrapper = GpuBufferBatchResizerJniWrapper.create(runtime);
    long outHandle =
        jniWrapper.create(
            minOutputBatchSize,
            new ColumnarBatchInIterator(BackendsApiManager.getBackendName(), in));
    return new ResizeBatchOutIterator(jniWrapper, runtime, outHandle, blockingTime, resizeTime);
  }
}
