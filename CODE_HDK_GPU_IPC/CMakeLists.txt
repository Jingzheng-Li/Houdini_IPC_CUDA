cmake_minimum_required(VERSION 3.20)

project(CODE_HDK_GPU_IPC)

list(APPEND CMAKE_PREFIX_PATH "/opt/hfs19.5/toolkit/cmake")

find_package(Houdini REQUIRED)
find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

if (EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else (EIGEN3_FOUND)
    message(FATAL_ERROR "Cannot find Eigen3")
endif (EIGEN3_FOUND)

# CUDA architectures
set(CUDA_ARCHITECTURES "86")
# Clear previous CUDA NVCC flags to avoid duplication
set(CUDA_NVCC_FLAGS "")
# Add C++17 standard flag
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
# Add architecture-specific code generation flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Add combined library #######################################
add_subdirectory(src)
