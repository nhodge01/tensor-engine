cmake_minimum_required(VERSION 3.18)
project(Acrelab_Gauss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# CUDA Runtime (for TensorRT)
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIB nvinfer)
find_path(NVINFER_INC NvInfer.h)

# --- 1. Tell CMake where to find Arrow's hidden config files ---
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/deps/arrow-cpp/cpp/build/install")
# Now CMake will magically map out all the AWS/Crypto dependencies for us!
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# --- 2. DuckDB (Local Build) ---
set(DUCKDB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../acrelab-gauss-cpp/build/_deps/duckdb_lib-src")
include_directories(${DUCKDB_DIR})
link_directories(${DUCKDB_DIR})

# --- 3. Tokenizers (Local Build) ---
set(TOK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/tokenizers-cpp")
include_directories("${TOK_DIR}/include")
link_directories("${TOK_DIR}/build" "${TOK_DIR}/build/tokenizers" "${TOK_DIR}/target/release")

# --- 4. Build The Engine ---
add_executable(embed_test src/core/main.cpp)

target_include_directories(embed_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/deps/arrow-cpp/cpp/build/install/include
    ${NVINFER_INC}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# --- 5. Clean Linking ---
target_link_libraries(embed_test PRIVATE
    ${NVINFER_LIB}
    CUDA::cudart
    duckdb
    tokenizers_cpp
    tokenizers_c
    dl
    pthread
    OpenMP::OpenMP_CXX
    Arrow::arrow_shared      # <--- Arrow handles all its own transitive deps now!
    Parquet::parquet_shared  # <--- Same here!
)
