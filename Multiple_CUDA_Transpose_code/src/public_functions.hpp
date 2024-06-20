#pragma once

#include <iostream>
#include <vector>

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

#include <CE/CE_API.h>
#include <CE/CE_Context.h>
#include <CE/CE_MemoryPool.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <SIM/SIM_GeometryCopy.h>
#include <SIM/SIM_Geometry.h>
#include <SIM/SIM_OptionsUser.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_ScalarField.h>
#include <SIM/SIM_DataFilter.h>
#include <SIM/SIM_Engine.h>
#include <SIM/SIM_VectorField.h>
#include <SIM/SIM_Time.h>
#include <SIM/SIM_Solver.h>
#include <SIM/SIM_DopDescription.h>
#include <GEO/GEO_Primitive.h>
#include <GEO/GEO_PrimVDB.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_VoxelArray.h>
#include <GA/GA_Handle.h>
#include <GU/GU_Detail.h>
#include <DOP/DOP_Node.h>
#include <DOP/DOP_Engine.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#define CHECK_ERROR(cond, msg) \
    if (!(cond)) { \
        std::cerr << msg << std::endl; \
        return; \
    }

#define CHECK_ERROR_SOLVER(cond, msg) \
    if (!(cond)) { \
        std::cerr << msg << std::endl; \
        return false; \
    }


class CUDAMemoryManager {
public:

    static void initialize(const Eigen::MatrixXf& positionsMat);
    static void copyDataToCUDA(const Eigen::MatrixXf& positionsMat);
    static void copyDataFromCUDA(Eigen::MatrixXf& positionsMat);
    static void free();

public:
    static float3* cudaPositions;

};


class GeometryManager {
public:
    static void initialize(const Eigen::MatrixXf& positionsMat);
    static void free();

public:
    static Eigen::MatrixXf positions;

};

