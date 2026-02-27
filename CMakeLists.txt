cmake_minimum_required(VERSION 3.18)
project(MiniLM_Inference LANGUAGES CXX)  # No more CUDA!

set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA Runtime (still needed for TensorRT)
find_package(CUDAToolkit REQUIRED)

# Arrow/Parquet - use local build from deps
set(ARROW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/arrow-cpp/cpp/build")
set(ARROW_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/arrow-cpp/cpp/src")
set(ARROW_BUILD_INCLUDE_DIR "${ARROW_DIR}/src")
set(ARROW_LIB_DIR "${ARROW_DIR}/release")

# Check if Arrow is built locally
if(EXISTS "${ARROW_LIB_DIR}/libarrow.a")
    set(Arrow_FOUND TRUE)
    set(Parquet_FOUND TRUE)
    message(STATUS "Using local Arrow build from deps/arrow-cpp")
    include_directories(${ARROW_INCLUDE_DIR})
    include_directories(${ARROW_BUILD_INCLUDE_DIR})
    link_directories(${ARROW_LIB_DIR})

    # Add thrift dependency
    set(THRIFT_LIB_DIR "${ARROW_DIR}/thrift_ep-install/lib")
    link_directories(${THRIFT_LIB_DIR})

    # Add Snappy dependency
    set(SNAPPY_LIB_DIR "${ARROW_DIR}/snappy_ep/src/snappy_ep-install/lib")
    link_directories(${SNAPPY_LIB_DIR})

    # Add AWS SDK dependencies for S3
    set(AWS_SDK_DIR "${ARROW_DIR}/awssdk_ep-install/lib")
    link_directories(${AWS_SDK_DIR})
else()
    set(Arrow_FOUND FALSE)
    set(Parquet_FOUND FALSE)
    message(STATUS "Arrow not built in deps/arrow-cpp. Run build instructions first.")
endif()

# TensorRT
find_library(NVINFER_LIB nvinfer)
find_path(NVINFER_INC NvInfer.h)

# Tokenizer
set(TOK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/tokenizers-cpp")
include_directories("${TOK_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src/include")
link_directories("${TOK_DIR}/build")
link_directories("${TOK_DIR}/build/tokenizers")
link_directories("${TOK_DIR}/target/release")

# DuckDB
set(DUCKDB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../acrelab-gauss-cpp/build/_deps/duckdb_lib-src")
include_directories(${DUCKDB_DIR})
link_directories(${DUCKDB_DIR})

# Main executable - now using main.cpp instead of main.cu
add_executable(embed_test
    src/core/main.cpp  # Changed from .cu to .cpp
    # arrow_parquet_writer and harvester_worker are now header-only
)

target_include_directories(embed_test PRIVATE
    ${NVINFER_INC}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(embed_test
    ${NVINFER_LIB}
    tokenizers_cpp
    tokenizers_c
    duckdb
    CUDA::cudart  # CUDA runtime for TensorRT
    dl
    pthread
    OpenMP::OpenMP_CXX
    # Arrow/Parquet/S3 dependencies
    parquet
    arrow
    thrift
    snappy
    aws-cpp-sdk-s3
    aws-cpp-sdk-core
    aws-cpp-sdk-identity-management
    aws-cpp-sdk-sts
    aws-cpp-sdk-cognito-identity
    aws-crt-cpp
    aws-c-s3
    aws-c-auth
    aws-c-event-stream
    aws-c-http
    aws-c-io
    aws-c-mqtt
    aws-c-cal
    aws-c-compression
    aws-c-sdkutils
    aws-checksums
    aws-c-common
    s2n
    crypto
    ssl
    curl
    z
)

# Benchmarks removed - source files no longer exist

# Note: Removed test_cuda_kernels since we no longer have custom CUDA kernels
