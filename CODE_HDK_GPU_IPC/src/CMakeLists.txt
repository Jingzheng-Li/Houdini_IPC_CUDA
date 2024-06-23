cmake_minimum_required(VERSION 3.20)

set(library_name HDK_GPU_IPC)

find_package(CUDA REQUIRED)

set(CUDA_ARCHITECTURES "86")
set(CUDA_NVCC_FLAGS "")
# Add C++17 standard flag
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
# Add architecture-specific code generation flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")

# Gather source files
set(SOURCES
    RWBuffer/main_readBuffer.cpp
    RWBuffer/main_writeBuffer.cpp
    # UTILS/main_transformLissajous.cpp
    common_init.cpp
)

# Gather header files
set(HEADERS
    RWBuffer/main_readBuffer.hpp
    RWBuffer/main_writeBuffer.hpp
    UTILS/HoudiniUtils.hpp
    UTILS/GeometryManager.hpp
    UTILS/cuda_kernels.cuh
    # UTILS/main_transformLissajous.hpp
)

# Gather CUDA files
set(CUDA_SOURCES
    UTILS/cuda_kernels.cu
)

# Add library
cuda_add_library(${library_name} SHARED ${SOURCES} ${CUDA_SOURCES})

# Set properties for CUDA files
set_target_properties(${library_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link libraries
target_link_libraries(${library_name} ${CUDA_LIBRARIES} Houdini)

# Configure Houdini target
houdini_configure_target(${library_name})

# Add include directories for the target
target_include_directories(${library_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/RWBuffer
    ${CMAKE_SOURCE_DIR}/src/UTILS
)
