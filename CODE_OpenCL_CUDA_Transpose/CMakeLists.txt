cmake_minimum_required(VERSION 3.20)

project(HDK_DOP_GASOPENCL_CUDA)

list(APPEND CMAKE_PREFIX_PATH "/opt/hfs19.5/toolkit/cmake")

find_package(Houdini REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Eigen3 REQUIRED)

if (EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
else (EIGEN3_FOUND)
    message(FATAL_ERROR "Cannot find Eigen3")
endif (EIGEN3_FOUND)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")

set(library_name HDK_DOP_gasopencl_cuda)

add_library(${library_name} SHARED
    # main.cpp
    # main.hpp
    main_test.cpp
    main_test.hpp
    public_functions.hpp
)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCL_INCLUDE_DIRS})

target_link_libraries(${library_name} ${CUDA_LIBRARIES} ${OpenCL_LIBRARIES} Houdini)

houdini_configure_target(${library_name})
