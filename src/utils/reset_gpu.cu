#include <cuda_runtime.h>
#include <iostream>

int main() {
    // Set device and reset it
    cudaSetDevice(0);

    // This will destroy all allocations and reset the GPU context
    cudaDeviceReset();

    std::cout << "GPU 0 context reset completed" << std::endl;

    return 0;
}