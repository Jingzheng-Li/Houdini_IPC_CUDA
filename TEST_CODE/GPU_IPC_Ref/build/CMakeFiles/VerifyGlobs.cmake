# CMAKE generated file: DO NOT EDIT!
# Generated by CMake Version 3.22
cmake_policy(SET CMP0009 NEW)

# GIPC_SOURCE at CMakeLists.txt:110 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/*.cpp")
set(OLD_GLOB
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/gl_main.cpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/load_mesh.cpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/build/CMakeFiles/cmake.verify_globs")
endif()

# GIPC_SOURCE at CMakeLists.txt:110 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/*.cu")
set(OLD_GLOB
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/ACCD.cu"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/GIPC.cu"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/MASPreconditioner.cu"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/PCG_SOLVER.cu"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/device_fem_data.cu"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/femEnergy.cu"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/gpu_eigen_libs.cu"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/mlbvh.cu"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/build/CMakeFiles/cmake.verify_globs")
endif()

# GIPC_HANDER at CMakeLists.txt:112 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/*.cuh")
set(OLD_GLOB
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/ACCD.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/FrictionUtils.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/GIPC.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/GIPC_PDerivative.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/MASPreconditioner.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/PCG_SOLVER.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/device_fem_data.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/femEnergy.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/gpu_eigen_libs.cuh"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/mlbvh.cuh"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/build/CMakeFiles/cmake.verify_globs")
endif()

# GIPC_HANDER at CMakeLists.txt:112 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/*.h")
set(OLD_GLOB
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/cuda_tools.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/eigen_data.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/fem_parameters.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/gipc_path.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/load_mesh.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/Reflection.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/MathUtils.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/Vec.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/bit/Bits.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/matrix/Utility.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/meta/ControlFlow.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/meta/Functional.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/meta/Meta.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/meta/Relationship.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/meta/Sequence.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/Function.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/Iterator.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/Optional.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/Polymorphism.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/Property.h"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/Tuple.h"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/build/CMakeFiles/cmake.verify_globs")
endif()

# GIPC_HANDER at CMakeLists.txt:112 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/*.hpp")
set(OLD_GLOB
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/TypeAlias.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/geometry/Distance.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/geometry/SpatialQuery.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/Complex.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/VecInterface.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/matrix/Eigen.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/matrix/Givens.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/matrix/QRSVD.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/matrix/Transform.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/BuilderBase.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/Pointers.hpp"
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/types/SmallVector.hpp"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/build/CMakeFiles/cmake.verify_globs")
endif()

# GIPC_HANDER at CMakeLists.txt:112 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/*.inc")
set(OLD_GLOB
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/build/CMakeFiles/cmake.verify_globs")
endif()

# GIPC_HANDER at CMakeLists.txt:112 (file)
file(GLOB_RECURSE NEW_GLOB LIST_DIRECTORIES false "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/*.inl")
set(OLD_GLOB
  "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/GPU_IPC/zensim/math/MatrixUtils.inl"
  )
if(NOT "${NEW_GLOB}" STREQUAL "${OLD_GLOB}")
  message("-- GLOB mismatch!")
  file(TOUCH_NOCREATE "/home/jingzheng/Public/Study/HoudiniLearn/Houdini_HDK_CUDA/GPU_IPC_Ref/build/CMakeFiles/cmake.verify_globs")
endif()
