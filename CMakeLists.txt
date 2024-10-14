
cmake_minimum_required(VERSION 3.18)

SET(CMAKE_C_COMPILER /usr/bin/clang)
SET(CMAKE_CXX_COMPILER /usr/bin/clang++)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

set(library_name gipc)

# 项目配置
project(gpu_ipc LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

# 检查是否设置了编译类型，默认为 Release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Release or Debug)" FORCE)
endif()

# 优化编译选项
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# CUDA 架构设置，自动选择合适的架构
find_package(CUDAToolkit REQUIRED)
if (NOT DEFINED CUDA_ARCHITECTURES)
    set(CUDA_ARCHITECTURES 86)
endif()

# 查找所需库
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

# 输出配置信息
message(STATUS "Building for CUDA architectures: ${CUDA_ARCHITECTURES}")
message(STATUS "CMake Build Type: ${CMAKE_BUILD_TYPE}")

# 创建可执行文件
add_executable(${library_name})

# 设置编译标准
target_compile_features(${library_name} PRIVATE cxx_std_17 cuda_std_17)

# 手动管理源文件，避免 GLOB_RECURSE 性能开销
file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS "src/*.cu" "src/*.cpp")
file(GLOB_RECURSE GIPC_HEADER CONFIGURE_DEPENDS "src/*.h" "src/*.hpp" "src/*.cuh")

# 包含头文件目录
target_include_directories(${library_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/LBVH
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ACCD
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FEM
    ${CMAKE_CURRENT_SOURCE_DIR}/src/IPC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/UTILS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/RWBuffer
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 设置源文件
target_sources(${library_name} PRIVATE ${GIPC_SOURCE} ${GIPC_HEADER})

# 设置编译定义，一次性添加所有的宏定义
target_compile_definitions(${library_name} PRIVATE 
    GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/"
    GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/"
    DEFAULT_THREADS=256
    USE_SNK
    USE_FRICTION
    USE_DOUBLE_PRECISION
)

# 链接 CUDA 库和其他依赖库
target_link_libraries(${library_name} PRIVATE 
    cusparse 
    cublas 
    cusolver 
    cufft 
    cudart_static 
    GLUT::GLUT 
    GLEW::GLEW 
    OpenGL::GLU 
    Eigen3::Eigen
    CUDA::cuda_driver 
    CUDA::cudart
)

# 设置 CUDA 相关属性
set_target_properties(${library_name} PROPERTIES
    CUDA_EXTENSIONS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES 86
)

# 优化编译选项，使用 PRIVATE 限制作用范围
target_compile_options(${library_name} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: 
        --use_fast_math 
        -lineinfo 
        --ptxas-options=-O3
        -Xcudafe "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
        --extended-lambda 
        --expt-relaxed-constexpr 
        --default-stream=per-thread
    >
)

# 设置 source_group 来整理源代码结构
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE} ${GIPC_HEADER})

