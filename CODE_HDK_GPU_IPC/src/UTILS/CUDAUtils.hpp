#pragma once

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

const static int default_threads = 256;

#define CHECK_ERROR(correctcond, msg) \
    if (!(correctcond)) { \
        std::cerr << msg << std::endl; \
        return; \
    }

#define CUDA_SAFE_CALL(err) cuda_safe_call_(err, __FILE__, __LINE__)
inline void cuda_safe_call_(cudaError_t err, const char* file_name, const int num_line) {
    if (cudaSuccess != err) {
        std::cerr << file_name << "[" << num_line << "]: "
                  << "CUDA Runtime API error[" << static_cast<int>(err) << "]: "
                  << cudaGetErrorString(err) << std::endl;

        exit(EXIT_FAILURE);
    }
}

#define CUDA_KERNEL_CHECK() cuda_kernel_check_(__FILE__, __LINE__)
inline void cuda_kernel_check_(const char* file_name, const int num_line) {
    cudaError_t err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        std::cerr << file_name << "[" << num_line << "]: "
                  << "CUDA Kernel execution error[" << static_cast<int>(err) << "]: "
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_INDEX_BOUNDS(idx, limit) check_index_bounds_(idx, limit, __FILE__, __LINE__)
__device__ inline void check_index_bounds_(uint32_t idx, uint32_t limit, const char* file_name, int num_line) {
    if (idx >= limit) {
        printf("%s[%d]: Error: index %d out of bounds (>= %d)\n", file_name, num_line, idx, limit);
        asm("trap;");
    }
}

template<typename T>
static void CUDAFreeSafe(T*& cudaData) {
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
static void CUDAMemcpyHToDSafe(const EigenType& eigenData, CudaType* cudaData) {
    std::vector<CudaType> temp(eigenData.rows());
    for (int i = 0; i < eigenData.rows(); ++i) {
        if constexpr (std::is_same<CudaType, double3>::value) {
            temp[i] = make_double3(eigenData(i, 0), eigenData(i, 1), eigenData(i, 2));
        } else if constexpr (std::is_same<CudaType, int4>::value) {
            temp[i] = make_int4(eigenData(i, 0), eigenData(i, 1), eigenData(i, 2), eigenData(i, 3));
        } else if constexpr (std::is_same<CudaType, uint3>::value) {
            temp[i] = make_uint3(eigenData(i, 0), eigenData(i, 1), eigenData(i, 2));
        } else if constexpr (std::is_same<CudaType, uint4>::value) {
            temp[i] = make_uint4(eigenData(i, 0), eigenData(i, 1), eigenData(i, 2), eigenData(i, 3));
        } else if constexpr (std::is_same<CudaType, int2>::value) {
            temp[i] = make_int2(eigenData(i, 0), eigenData(i, 1));
        } else if constexpr (std::is_same<CudaType, uint2>::value) {
            temp[i] = make_uint2(eigenData(i, 0), eigenData(i, 1));
        } else if constexpr (std::is_same<CudaType, double>::value) {
            temp[i] = eigenData(i);
        } else if constexpr (std::is_same<CudaType, uint32_t>::value) {
            temp[i] = static_cast<uint32_t>(eigenData(i));
        }
    }
    CUDA_SAFE_CALL(cudaMemcpy(cudaData, temp.data(), eigenData.rows() * sizeof(CudaType), cudaMemcpyHostToDevice));
}


template <typename EigenType, typename CudaType>
static void CUDAMemcpyDToHSafe(EigenType& eigenData, CudaType* cudaData) {
    std::vector<CudaType> temp(eigenData.rows());
    CUDA_SAFE_CALL(cudaMemcpy(temp.data(), cudaData, eigenData.rows() * sizeof(CudaType), cudaMemcpyDeviceToHost));
    for (int i = 0; i < eigenData.rows(); ++i) {
        if constexpr (std::is_same<CudaType, double3>::value) {
            eigenData(i, 0) = temp[i].x;
            eigenData(i, 1) = temp[i].y;
            eigenData(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, uint4>::value) {
            eigenData(i, 0) = temp[i].x;
            eigenData(i, 1) = temp[i].y;
            eigenData(i, 2) = temp[i].z;
            eigenData(i, 3) = temp[i].w;
        } else if constexpr (std::is_same<CudaType, uint3>::value) {
            eigenData(i, 0) = temp[i].x;
            eigenData(i, 1) = temp[i].y;
            eigenData(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, uint2>::value) {
            eigenData(i, 0) = temp[i].x;
            eigenData(i, 1) = temp[i].y;
        } else if constexpr (std::is_same<CudaType, double>::value) {
            eigenData(i) = temp[i];
        } else if constexpr (std::is_same<CudaType, uint32_t>::value) {
            eigenData(i) = temp[i];
        }
    }
}