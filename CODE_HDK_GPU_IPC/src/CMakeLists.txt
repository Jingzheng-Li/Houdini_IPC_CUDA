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
# set(SOURCES
#     RWBuffer/main_readBuffer.cpp
#     RWBuffer/main_writeBuffer.cpp
#     # CCD/main_LBVH.cpp

#     common_init.cpp
# )

# # Gather header files
# set(HEADERS
#     RWBuffer/main_readBuffer.hpp
#     RWBuffer/main_writeBuffer.hpp
#     # CCD/main_LBVH.hpp
#     # CCD/LBVH.cuh

#     UTILS/CUDAUtils.hpp
#     UTILS/HoudiniUtils.hpp
#     UTILS/GeometryManager.hpp
#     UTILS/MathUtils.hpp
#     UTILS/MathUtils.cuh
    
# )

# # Gather CUDA files
# set(CUDA_SOURCES
#     # CCD/LBVH.cu
#     UTILS/MathUtils.cu
# )

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS 
    "CCD/*.cpp"
    # "Energy/*.cpp"
    # "IPC/*.cpp"
    # "PCG/*.cpp"
    "RWBuffer/*.cpp"
    "UTILS/*.cpp"

    "common_init.cpp"
)

file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS 
    "CCD/*.hpp"
    # "Energy/*.hpp"
    # "IPC/*.hpp"
    # "PCG/*.hpp"
    "RWBuffer/*.hpp"
    "UTILS/*.hpp"

    # "zensim/*.hpp"
    # "zensim/*.h"
    # "zensim/**/*.hpp"
    # "zensim/**/*.h"
    # "zensim/**/**/*.hpp"
    # "zensim/**/**/*.h"
    # "zensim/**/*.inl"
    "zensim/*"
)

# print all the header files
# message(STATUS "Header files:")
# foreach(header ${HEADERS})
#     message(STATUS "  ${header}")
# endforeach()

file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS 
    "CCD/*.cu"
    # "Energy/*.cu"
    # "IPC/*.cu"
    # "PCG/*.cu"
    "UTILS/*.cu"
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
    ${CMAKE_SOURCE_DIR}/src/CCD
    ${CMAKE_SOURCE_DIR}/src/Energy
    ${CMAKE_SOURCE_DIR}/src/IPC
    ${CMAKE_SOURCE_DIR}/src/RWBuffer
    ${CMAKE_SOURCE_DIR}/src/UTILS
    ${CMAKE_SOURCE_DIR}/src/zensim
)

target_compile_features(${library_name} PRIVATE cuda_std_17)
target_compile_definitions(${library_name} PRIVATE USE_SNK SVD_DOUBLE ADAPTIVE_KAPPA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86")

target_compile_options(${library_name}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: 
        ${CMAKE_CUDA_FLAGS} 
        --use_fast_math 
        -lineinfo 
        --ptxas-options=-allow-expensive-optimizations=true 
        -Xcudafe 
        "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
        # --expt-relaxed-constexpr
    >
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: 
        --extended-lambda 
        --expt-relaxed-constexpr 
        --default-stream=per-thread
    >
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES} ${HEADERS} ${CUDA_SOURCES})