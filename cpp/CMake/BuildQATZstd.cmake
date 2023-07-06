# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

include(ExternalProject)

macro(build_qatzstd)
  # Find ZSTD
  include(FindZstd)

  message(STATUS "Building QAT-ZSTD from source")
  set(QATZSTD_SOURCE_URL
      "https://github.com/marin-ma/QAT-ZSTD-Plugin.git")
  set(QATZSTD_SOURCE_BRANCH "fix-duplicate-symbol")
  set(QATZSTD_LIB_NAME "qatseqprod")

  set(QATZSTD_PREFIX
      "${CMAKE_CURRENT_BINARY_DIR}/qatzstd_ep-install")
  set(QATZSTD_SOURCE_DIR "${QATZSTD_PREFIX}/src/qatzstd_ep")
  set(QATZSTD_INCLUDE_DIR "${QATZSTD_SOURCE_DIR}/src")
  set(QATZSTD_STATIC_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${QATZSTD_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(QATZSTD_STATIC_LIB_TARGETS "${QATZSTD_SOURCE_DIR}/src/${QATZSTD_STATIC_LIB_NAME}")
  set(QATZSTD_MAKE_ARGS "ENABLE_USDM_DRV=1" "ZSTDLIB=${ZSTD_INCLUDE_DIR}")

  ExternalProject_Add(qatzstd_ep
      PREFIX ${QATZSTD_PREFIX}
      GIT_REPOSITORY ${QATZSTD_SOURCE_URL}
      GIT_TAG ${QATZSTD_SOURCE_BRANCH}
      SOURCE_DIR ${QATZSTD_SOURCE_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND $(MAKE) ${QATZSTD_MAKE_ARGS}
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS ${QATZSTD_STATIC_LIB_TARGETS}
      BUILD_IN_SOURCE 1)

  add_library(qatzstd::qatzstd STATIC IMPORTED)

  if(ZSTD_USE_BUNDLED)
    add_dependencies(qatzstd_ep qatzstd_zstd_ep)
  endif()
  set(QATZSTD_INCLUDE_DIRS
      "${QATZSTD_INCLUDE_DIR}"
      "${ZSTD_INCLUDE_DIR}")

  # The include directory must exist before it is referenced by a target.
  file(MAKE_DIRECTORY "${QATZSTD_INCLUDE_DIR}")

  set_target_properties(qatzstd::qatzstd
      PROPERTIES IMPORTED_LOCATION
      "${QATZSTD_STATIC_LIB_TARGETS}"
      INTERFACE_INCLUDE_DIRECTORIES
      "${QATZSTD_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES
      "${ZSTD_STATIC_LIB}")

  add_dependencies(qatzstd::qatzstd qatzstd_ep)
endmacro()

build_qatzstd()
