cmake_minimum_required(VERSION 3.18)
project(MiniLM_Inference LANGUAGES CXX CUDA) # Enable CUDA language!

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

find_library(NVINFER_LIB nvinfer)
find_path(NVINFER_INC NvInfer.h)

set(TOK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/tokenizers-cpp")
include_directories("${TOK_DIR}/include")
link_directories("${TOK_DIR}/build")
link_directories("${TOK_DIR}/build/tokenizers") 
link_directories(/opt/cuda/lib64)

add_executable(embed_test main.cu) # Point to the new .cu file

target_include_directories(embed_test PRIVATE ${NVINFER_INC})

target_link_libraries(embed_test 
    ${NVINFER_LIB}
    tokenizers_cpp
    tokenizers_c
    cudart
    dl
    pthread
    OpenMP::OpenMP_CXX
)
