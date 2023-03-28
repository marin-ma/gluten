#!/bin/bash
####################################################################################################
#  The main function of this script is to allow developers to build the environment with one click #
#  Recommended commands for first-time installation:                                               #
#  ./dev/buildbundle-veloxbe.sh                                                            #
####################################################################################################
set -exu

CURRENT_DIR=$(cd "$(dirname "$BASH_SOURCE")"; pwd)
GLUTEN_DIR="$CURRENT_DIR/.."
BUILD_TYPE=Release
BUILD_TESTS=OFF
BUILD_BENCHMARKS=OFF
BUILD_JEMALLOC=ON
BUILD_PROTOBUF=ON
ENABLE_QAT=OFF
ENABLE_IAA=OFF
ENABLE_HBM=OFF
ENABLE_S3=OFF
ENABLE_HDFS=OFF
ENABLE_EP_CACHE=OFF
ARROW_ENABLE_CUSTOM_CODEC=OFF
for arg in "$@"
do
    case $arg in
        --build_type=*)
        BUILD_TYPE=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --build_tests=*)
        BUILD_TESTS=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --build_benchmarks=*)
        BUILD_BENCHMARKS=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --build_jemalloc=*)
        BUILD_JEMALLOC=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --enable_qat=*)
        ENABLE_QAT=("${arg#*=}")
        ARROW_ENABLE_CUSTOM_CODEC=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --enable_iaa=*)
        ENABLE_IAA=("${arg#*=}")
        ARROW_ENABLE_CUSTOM_CODEC=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --enable_hbm=*)
        ENABLE_HBM=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --build_protobuf=*)
        BUILD_PROTOBUF=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --enable_s3=*)
        ENABLE_S3=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --enable_hdfs=*)
        ENABLE_HDFS=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
        --enable_ep_cache=*)
        ENABLE_EP_CACHE=("${arg#*=}")
        shift # Remove argument name from processing
        ;;
	      *)
        OTHER_ARGUMENTS+=("$1")
        shift # Remove generic argument from processing
        ;;
    esac
done

##install arrow
cd $GLUTEN_DIR/ep/build-arrow/src
./get_arrow.sh --enable_custom_codec=$ARROW_ENABLE_CUSTOM_CODEC
./build_arrow.sh --build_type=$BUILD_TYPE --build_tests=$BUILD_TESTS --build_benchmarks=$BUILD_BENCHMARKS \
                         --enable_ep_cache=$ENABLE_EP_CACHE

##install velox
cd $GLUTEN_DIR/ep/build-velox/src
./get_velox.sh --enable_hdfs=$ENABLE_HDFS --build_protobuf=$BUILD_PROTOBUF --enable_s3=$ENABLE_S3
./build_velox.sh --enable_s3=$ENABLE_S3 --build_type=$BUILD_TYPE --enable_hdfs=$ENABLE_HDFS \
               --enable_ep_cache=$ENABLE_EP_CACHE

## compile gluten cpp
cd $GLUTEN_DIR/cpp
rm -rf build
mkdir build
cd build
cmake -DBUILD_VELOX_BACKEND=ON -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DBUILD_TESTS=$BUILD_TESTS -DBUILD_BENCHMARKS=$BUILD_BENCHMARKS -DBUILD_JEMALLOC=$BUILD_JEMALLOC \
      -DENABLE_HBM=$ENABLE_HBM -DENABLE_QAT=$ENABLE_QAT -DENABLE_IAA=$ENABLE_IAA -DVELOX_ENABLE_S3=$ENABLE_S3 -DVELOX_ENABLE_HDFS=$ENABLE_HDFS ..
make -j


