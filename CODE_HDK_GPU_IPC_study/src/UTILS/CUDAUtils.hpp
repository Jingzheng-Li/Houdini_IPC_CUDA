#pragma once

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "MathUtils.hpp"

const static int default_threads = 256;


#define CHECK_ERROR_CUDA(cond, msg, error) \
    if (!(cond)) { \
        printf("\033[1;31mCUDA Error: %s\033[0m\n", msg); \
        error = true; \
        return; \
    }

#define CUDA_SAFE_CALL(err) cuda_safe_call_(err, __FILE__, __LINE__)
inline void cuda_safe_call_(cudaError_t err, const char* file_name, const int num_line) {
    if (cudaSuccess != err) {
        std::cerr << "\033[1;31m" << file_name << "[" << num_line << "]: "
                  << "CUDA Runtime API error[" << static_cast<int>(err) << "]: "
                  << cudaGetErrorString(err) << "\033[0m" << std::endl;

        exit(EXIT_FAILURE);
    }
}

#define CUDA_KERNEL_CHECK() cuda_kernel_check_(__FILE__, __LINE__)
inline void cuda_kernel_check_(const char* file_name, const int num_line) {
    cudaError_t err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        std::cerr << "\033[1;31m" << file_name << "[" << num_line << "]: "
                  << "CUDA Kernel execution error[" << static_cast<int>(err) << "]: "
                  << cudaGetErrorString(err) << "\033[0m" << std::endl;
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
static void CUDAMemcpyHToDSafe(CudaType* dstCudaData, const EigenType& srcEigenData) {
    std::vector<CudaType> temp(srcEigenData.rows());
    for (int i = 0; i < srcEigenData.rows(); ++i) {
        if constexpr (std::is_same<CudaType, double>::value || 
                      std::is_same<CudaType, int>::value ||
                      std::is_same<CudaType, uint32_t>::value) {
            temp[i] = srcEigenData(i);
        } else if constexpr (std::is_same<CudaType, double2>::value) {
            temp[i] = make_double2(srcEigenData(i, 0), srcEigenData(i, 1));
        } else if constexpr (std::is_same<CudaType, double3>::value) {
            temp[i] = make_double3(srcEigenData(i, 0), srcEigenData(i, 1), srcEigenData(i, 2));
        } else if constexpr (std::is_same<CudaType, double4>::value) {
            temp[i] = make_double4(srcEigenData(i, 0), srcEigenData(i, 1), srcEigenData(i, 2), srcEigenData(i, 3));
        } else if constexpr (std::is_same<CudaType, int2>::value) {
            temp[i] = make_int2(srcEigenData(i, 0), srcEigenData(i, 1));
        } else if constexpr (std::is_same<CudaType, int3>::value) {
            temp[i] = make_int3(srcEigenData(i, 0), srcEigenData(i, 1), srcEigenData(i, 2));
        } else if constexpr (std::is_same<CudaType, int4>::value) {
            temp[i] = make_int4(srcEigenData(i, 0), srcEigenData(i, 1), srcEigenData(i, 2), srcEigenData(i, 3));
        } else if constexpr (std::is_same<CudaType, uint2>::value) {
            temp[i] = make_uint2(srcEigenData(i, 0), srcEigenData(i, 1));
        } else if constexpr (std::is_same<CudaType, uint3>::value) {
            temp[i] = make_uint3(srcEigenData(i, 0), srcEigenData(i, 1), srcEigenData(i, 2));
        } else if constexpr (std::is_same<CudaType, uint4>::value) {
            temp[i] = make_uint4(srcEigenData(i, 0), srcEigenData(i, 1), srcEigenData(i, 2), srcEigenData(i, 3));
        } else if constexpr (std::is_same<CudaType, MATHUTILS::Matrix3x3d>::value || 
                             std::is_same<CudaType, MATHUTILS::Matrix2x2d>::value || 
                             std::is_same<CudaType, MATHUTILS::Matrix3x2d>::value) {
            temp[i] = srcEigenData(i);
        } else {
            std::cerr << "\033[1;31m" << "we do not support such type of data in CUDAMemcpyHToDSafe right now" << "\033[0m" << std::endl;
        }
    }
    CUDA_SAFE_CALL(cudaMemcpy(dstCudaData, temp.data(), srcEigenData.rows() * sizeof(CudaType), cudaMemcpyHostToDevice));
}


template <typename EigenType, typename CudaType>
static void CUDAMemcpyDToHSafe(EigenType& dstEigenData, const CudaType* srcCudaData) {
    std::vector<CudaType> temp(dstEigenData.rows());
    CUDA_SAFE_CALL(cudaMemcpy(temp.data(), srcCudaData, dstEigenData.rows() * sizeof(CudaType), cudaMemcpyDeviceToHost));

    for (int i = 0; i < dstEigenData.rows(); ++i) {
        if constexpr (std::is_same<CudaType, double>::value || 
                      std::is_same<CudaType, int>::value ||
                      std::is_same<CudaType, uint32_t>::value) {
            dstEigenData(i) = temp[i];
        } else if constexpr (std::is_same<CudaType, double2>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
        } else if constexpr (std::is_same<CudaType, double3>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
            dstEigenData(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, double4>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
            dstEigenData(i, 2) = temp[i].z;
            dstEigenData(i, 3) = temp[i].w;
        } else if constexpr (std::is_same<CudaType, int2>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
        } else if constexpr (std::is_same<CudaType, int3>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
            dstEigenData(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, int4>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
            dstEigenData(i, 2) = temp[i].z;
            dstEigenData(i, 3) = temp[i].w;
        } else if constexpr (std::is_same<CudaType, uint2>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
        } else if constexpr (std::is_same<CudaType, uint3>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
            dstEigenData(i, 2) = temp[i].z;
        } else if constexpr (std::is_same<CudaType, uint4>::value) {
            dstEigenData(i, 0) = temp[i].x;
            dstEigenData(i, 1) = temp[i].y;
            dstEigenData(i, 2) = temp[i].z;
            dstEigenData(i, 3) = temp[i].w;
        } else if constexpr (std::is_same<CudaType, MATHUTILS::Matrix3x3d>::value || 
                             std::is_same<CudaType, MATHUTILS::Matrix2x2d>::value || 
                             std::is_same<CudaType, MATHUTILS::Matrix3x2d>::value) {
            dstEigenData(i) = temp[i];
        } else {
            std::cerr << "\033[1;31m" << "Unsupported type in CUDAMemcpyDToHSafe" << "\033[0m" << std::endl; 
        }
    }
}
