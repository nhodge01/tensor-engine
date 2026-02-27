
#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Setting up Acrelab Gauss Dependencies ==="

# Create the deps directory
mkdir -p deps
cd deps

# --- 1. APACHE ARROW ---
echo "--- Building Apache Arrow (14.0.2) ---"
if [ ! -d "arrow-cpp" ]; then
    git clone --depth 1 --branch apache-arrow-14.0.2 https://github.com/apache/arrow.git arrow-cpp
fi

cd arrow-cpp/cpp
mkdir -p build && cd build

# The exact flags we bled for
cmake .. \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/install" \
  -DCMAKE_BUILD_TYPE=Release \
  -DARROW_FILESYSTEM=ON \
  -DARROW_PARQUET=ON \
  -DARROW_S3=ON \
  -DARROW_WITH_SNAPPY=ON \
  -DARROW_BUILD_SHARED=ON \
  -DARROW_BUILD_STATIC=OFF \
  -DARROW_DEPENDENCY_SOURCE=BUNDLED \
  -DARROW_JEMALLOC=OFF

make -j$(nproc)
make install
cd ../../../


# --- 2. TOKENIZERS-CPP ---
echo "--- Building Tokenizers-cpp ---"
if [ ! -d "tokenizers-cpp" ]; then
    # Added --recursive to fetch msgpack and sentencepiece!
    git clone --recursive https://github.com/mlc-ai/tokenizers-cpp.git
fi


cd tokenizers-cpp
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ../../

echo "=== Dependencies built successfully! ==="
