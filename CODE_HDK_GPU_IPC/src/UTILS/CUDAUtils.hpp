#pragma once

#include <iostream>
#include <vector>

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

template <typename T>
void CUDAMallocSafe(T*& cudaData, int size) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaData, size * sizeof(T)));
}


template <typename EigenType, typename CudaType>
static void copyToCUDASafe(const EigenType& eigenData, CudaType* cudaData) {
    std::vector<CudaType> temp(eigenData.rows());
    for (int i = 0; i < eigenData.rows(); ++i) {
        if constexpr (std::is_same<CudaType, double3>::value) {
            temp[i] = make_double3(eigenData(i, 0), eigenData(i, 1), eigenData(i, 2));
        } else if constexpr (std::is_same<CudaType, int4>::value) {
            temp[i] = make_int4(eigenData(i, 0), eigenData(i, 1), eigenData(i, 2), eigenData(i, 3));
        } else if constexpr (std::is_same<CudaType, int3>::value) {
            temp[i] = make_int3(eigenData(i, 0), eigenData(i, 1), eigenData(i, 2));
        } else if constexpr (std::is_same<CudaType, int2>::value) {
            temp[i] = make_int2(eigenData(i, 0), eigenData(i, 1));
        } else if constexpr (std::is_same<CudaType, double>::value) {
            temp[i] = eigenData(i);
        }
    }
    CUDA_SAFE_CALL(cudaMemcpy(cudaData, temp.data(), eigenData.rows() * sizeof(CudaType), cudaMemcpyHostToDevice));
}


template <typename EigenType, typename CudaType>
static void copyFromCUDASafe(EigenType& eigenData, CudaType* cudaData) {
    std::vector<CudaType> temp(eigenData.rows());
    CUDA_SAFE_CALL(cudaMemcpy(temp.data(), cudaData, eigenData.rows() * sizeof(CudaType), cudaMemcpyDeviceToHost));
    for (int i = 0; i < eigenData.rows(); ++i) {
        if constexpr (std::is_same<CudaType, double3>::value) {
            eigenData(i, 0) = temp[i].x;
            eigenData(i, 1) = temp[i].y;
            eigenData(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, int4>::value) {
            eigenData(i, 0) = temp[i].x;
            eigenData(i, 1) = temp[i].y;
            eigenData(i, 2) = temp[i].z;
            eigenData(i, 3) = temp[i].w;
        } else if constexpr (std::is_same<CudaType, int3>::value) {
            eigenData(i, 0) = temp[i].x;
            eigenData(i, 1) = temp[i].y;
            eigenData(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, int2>::value) {
            eigenData(i, 0) = temp[i].x;
            eigenData(i, 1) = temp[i].y;
        } else if constexpr (std::is_same<CudaType, double>::value) {
            eigenData(i) = temp[i];
        }
    }
}