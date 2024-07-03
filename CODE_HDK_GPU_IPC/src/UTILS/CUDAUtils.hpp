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
void allocateCUDASafe(T*& cudaData, int size) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaData, size * sizeof(T)));
}


template <typename EigenType, typename CudaType>
static void copyToCUDASafe(const EigenType& data, CudaType* cudaData) {
    std::vector<CudaType> temp(data.rows());
    for (int i = 0; i < data.rows(); ++i) {
        if constexpr (std::is_same<CudaType, double3>::value) {
            temp[i] = make_double3(data(i, 0), data(i, 1), data(i, 2));
        } else if constexpr (std::is_same<CudaType, int4>::value) {
            temp[i] = make_int4(data(i, 0), data(i, 1), data(i, 2), data(i, 3));
        } else if constexpr (std::is_same<CudaType, int3>::value) {
            temp[i] = make_int3(data(i, 0), data(i, 1), data(i, 2));
        } else if constexpr (std::is_same<CudaType, int2>::value) {
            temp[i] = make_int2(data(i, 0), data(i, 1));
        } else if constexpr (std::is_same<CudaType, double>::value) {
            temp[i] = data(i);
        }
    }
    CUDA_SAFE_CALL(cudaMemcpy(cudaData, temp.data(), data.rows() * sizeof(CudaType), cudaMemcpyHostToDevice));
}


template <typename EigenType, typename CudaType>
static void copyFromCUDASafe(EigenType& data, CudaType* cudaData) {
    std::vector<CudaType> temp(data.rows());
    CUDA_SAFE_CALL(cudaMemcpy(temp.data(), cudaData, data.rows() * sizeof(CudaType), cudaMemcpyDeviceToHost));
    for (int i = 0; i < data.rows(); ++i) {
        if constexpr (std::is_same<CudaType, double3>::value) {
            data(i, 0) = temp[i].x;
            data(i, 1) = temp[i].y;
            data(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, int4>::value) {
            data(i, 0) = temp[i].x;
            data(i, 1) = temp[i].y;
            data(i, 2) = temp[i].z;
            data(i, 3) = temp[i].w;
        } else if constexpr (std::is_same<CudaType, int3>::value) {
            data(i, 0) = temp[i].x;
            data(i, 1) = temp[i].y;
            data(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, int2>::value) {
            data(i, 0) = temp[i].x;
            data(i, 1) = temp[i].y;
        } else if constexpr (std::is_same<CudaType, double>::value) {
            data(i) = temp[i];
        }
    }
}