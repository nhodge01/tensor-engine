cmake_minimum_required(VERSION 3.18)
project(MiniLM_Inference LANGUAGES CXX CUDA) # Enable CUDA language!

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

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

find_library(NVINFER_LIB nvinfer)
find_path(NVINFER_INC NvInfer.h)

set(TOK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/tokenizers-cpp")
include_directories("${TOK_DIR}/include")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/acrelab-gauss")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
link_directories("${TOK_DIR}/build")
link_directories("${TOK_DIR}/build/tokenizers")
link_directories("${TOK_DIR}/target/release")
link_directories(/opt/cuda/lib64)

# DuckDB
set(DUCKDB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../acrelab-gauss-cpp/build/_deps/duckdb_lib-src")
include_directories(${DUCKDB_DIR})
link_directories(${DUCKDB_DIR})

add_executable(embed_test
    src/core/main.cu
    src/core/lake.cpp
    src/core/arrow_parquet_writer.cpp
    src/core/harvester_worker.cpp
) # Include all core components

target_include_directories(embed_test PRIVATE ${NVINFER_INC})

target_link_libraries(embed_test
    ${NVINFER_LIB}
    tokenizers_cpp
    tokenizers_c
    duckdb
    cudart
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

# Link Arrow/Parquet if found (commented out - using static libs instead)
# if(Arrow_FOUND AND Parquet_FOUND)
#     target_link_libraries(embed_test Arrow::arrow_shared Parquet::parquet_shared)
# endif()

# Tokenizer benchmark executable
add_executable(tokenize_bench src/benchmarks/tokenize_bench.cpp)

target_link_libraries(tokenize_bench
    tokenizers_cpp
    tokenizers_c
    OpenMP::OpenMP_CXX
    pthread
)

# CUDA kernels test executable
add_executable(test_cuda_kernels src/benchmarks/test_cuda_kernels.cu)

target_link_libraries(test_cuda_kernels
    cudart
    pthread
)

# Sink benchmark executable (only if Arrow is available)
if(Arrow_FOUND AND Parquet_FOUND)
    add_executable(sink_benchmark src/benchmarks/sink_benchmark.cpp)

    target_link_libraries(sink_benchmark
        pthread
        OpenMP::OpenMP_CXX
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
    message(STATUS "sink_benchmark will be built (Arrow found)")
else()
    message(STATUS "sink_benchmark will NOT be built (Arrow not found)")
endif()
