#pragma once

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

const static int default_threads = 256;

#define CUDA_SAFE_CALL(err) cuda_safe_call_(err, __FILE__, __LINE__)

inline void cuda_safe_call_(cudaError_t err, const char* file_name, const int num_line) {
    if (cudaSuccess != err) {
        std::cerr << file_name << "[" << num_line << "]: "
                  << "CUDA Runtime API error[" << static_cast<int>(err) << "]: "
                  << cudaGetErrorString(err) << std::endl;

        exit(EXIT_FAILURE);
    }
}

template<typename T>
static void freeCUDASafe(T*& cudaData) {
    if (cudaData) {
        CUDA_SAFE_CALL(cudaFree(cudaData));
        cudaData = nullptr;
    }
}