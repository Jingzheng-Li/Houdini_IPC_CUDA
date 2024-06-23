#include "cuda_kernels.cuh"
#include <math.h>
#include <iostream>

__global__ void transformLissajous(double3* positions, int numPoints, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numPoints) return;

    float A = 2.0f + 0.5f * (idx % 5); // Amplitude in x direction
    float B = 2.0f + 0.5f * ((idx + 2) % 5); // Amplitude in y direction
    float C = 2.0f + 0.5f * ((idx + 4) % 5); // Amplitude in z direction
    float a = 3.0f + 0.1f * (idx % 7); // Frequency in x direction
    float b = 2.0f + 0.1f * ((idx + 3) % 7); // Frequency in y direction
    float c = 1.0f + 0.1f * ((idx + 5) % 7); // Frequency in z direction
    float delta = M_PI_2; // Phase shift

    // Calculate new position based on Lissajous curve
    float newX = A * sinf(a * time + delta * idx);
    float newY = B * sinf(b * time + delta * idx);
    float newZ = C * sinf(c * time + delta * idx);

    positions[idx] = make_double3(newX, newY, newZ);

}


void transformLissajousCUDA(double3* cudaTetPos, int numPoints, float time) {

    int blockSize = 256;
    int numBlocks = (numPoints + blockSize - 1) / blockSize;
    transformLissajous<<<numBlocks, blockSize>>>(cudaTetPos, numPoints, time);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(error) << std::endl;
    }

    error = cudaDeviceSynchronize(); // Ensure the kernel has finished executing
    if (error != cudaSuccess) {
        std::cerr << "CUDA synchronize error: " << cudaGetErrorString(error) << std::endl;
    }

    printDeviceProperties();
}

void printDeviceProperties() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total global memory: " << deviceProp.totalGlobalMem << std::endl;
    }
}
