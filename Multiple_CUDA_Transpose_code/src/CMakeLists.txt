cmake_minimum_required(VERSION 3.20)

set(library_name HDK_DOP_Multiple)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 设置CUDA架构为sm_86
set(CUDA_ARCHITECTURES "86")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -gencode arch=compute_86,code=sm_86")

cuda_add_library(${library_name} SHARED
    main_readBuffer.cpp
    main_readBuffer.hpp
    main_writeBuffer.cpp
    main_writeBuffer.hpp
    main_transformLissajous.cpp
    main_transformLissajous.hpp
    public_functions.cpp
    public_functions.hpp
    common_init.cpp
    cuda_kernels.cu
)

target_link_libraries(${library_name} ${CUDA_LIBRARIES} Houdini)

houdini_configure_target(${library_name})
