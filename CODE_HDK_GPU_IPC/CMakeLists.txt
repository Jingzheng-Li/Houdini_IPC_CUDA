
cmake_minimum_required(VERSION 3.18)

project(CODE_HDK_GPU_IPC LANGUAGES CXX CUDA)

set(library_name HDK_GPU_IPC)

# Set CUDA architectures
set(CUDA_ARCHITECTURES "86")
set(CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
endif()

# Locate Houdini's libraries and header files
list(APPEND CMAKE_PREFIX_PATH "/opt/hfs19.5/toolkit/cmake")
find_package(Houdini REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cufft)
find_package(Eigen3 REQUIRED)
# find_package(GLEW REQUIRED)
# find_package(GLUT REQUIRED)
# find_package(OpenGL REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${HOUDINI_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
# include_directories(${GLEW_INCLUDE_DIRS})
# include_directories(${GLUT_INCLUDE_DIRS})
# include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add library with source files
file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE GIPC_HEADER CONFIGURE_DEPENDS "src/*.h" "src/*.cuh" "src/*.hpp" "src/*.inl" "src/*.inc")
add_library(${library_name} SHARED ${GIPC_SOURCE} ${GIPC_HEADER})

# Add include directories for the target
target_include_directories(${library_name} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/CCD
    ${CMAKE_SOURCE_DIR}/src/Energy
    ${CMAKE_SOURCE_DIR}/src/IPC
    ${CMAKE_SOURCE_DIR}/src/RWBuffer
    ${CMAKE_SOURCE_DIR}/src/UTILS
    ${CMAKE_SOURCE_DIR}/src/zensim
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Clear previous CUDA NVCC flags to avoid duplication
set(CUDA_NVCC_FLAGS "")
# Add C++17 standard flag
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
# Add architecture-specific code generation flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")

# Target properties
set_target_properties(${library_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 86)

# Link libraries
target_link_libraries(${library_name} PRIVATE Houdini)
target_link_libraries(${library_name} PRIVATE cusparse cublas cusolver cudart_static)
target_link_libraries(${library_name} PRIVATE Eigen3::Eigen)
# target_link_libraries(${library_name} PRIVATE GLUT::GLUT GLEW::GLEW OpenGL::GLU)

# Compile definitions
target_compile_definitions(${library_name} PRIVATE GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/")
target_compile_definitions(${library_name} PRIVATE GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/")
target_compile_definitions(${library_name} PRIVATE USE_SNK SVD_DOUBLE ADAPTIVE_KAPPA)

# Compile options
target_compile_features(${library_name} PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(${library_name}
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        ${CMAKE_CUDA_FLAGS}
        --use_fast_math
        -lineinfo
        --ptxas-options=-allow-expensive-optimizations=true
        -Xcudafe
        "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
    >
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --default-stream=per-thread
    >
)

# Configure Houdini target
houdini_configure_target(${library_name})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE} ${GIPC_HEADER})
