
#pragma once

#include "BlockHessian.cuh"
#include "MASPreconditioner.cuh"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/GeometryManager.hpp"
#include "UTILS/MathUtils.cuh"


class PCGSolver {
   public:
    void CUDA_MALLOC_PCGSOLVER(const int& vertex_num,
                               const int& tetradedra_num);
    void CUDA_FREE_PCGSOLVER();

    int PCG_Process(std::unique_ptr<GeometryManager>& instance,
                    const BlockHessian& BlockHessian, Scalar3* _mvDir,
                    int vertexNum, int tetrahedraNum, Scalar IPC_dt,
                    Scalar meanVolume, Scalar threshold);

    int MASPCG_Process(std::unique_ptr<GeometryManager>& instance,
                       const BlockHessian& BlockHessian, Scalar3* _mvDir,
                       int vertexNum, int tetrahedraNum, Scalar IPC_dt,
                       Scalar meanVolume, int cpNum, Scalar threshold);

   public:
    MASPreconditioner MP;
    int PrecondType;

   public:
    // PCG data
    Scalar* cudaPCGSqueue;
    Scalar3* cudaPCGb;
    Scalar3* cudaPCGr;
    Scalar3* cudaPCGc;
    Scalar3* cudaPCGq;
    Scalar3* cudaPCGs;
    Scalar3* cudaPCGz;
    Scalar3* cudaPCGdx;
    Scalar3* cudaPCGTempDx;

    // local preconditioner
    __MATHUTILS__::Matrix3x3S* cudaPCGP;

    // friction parameters
    Scalar3* cudaPCGFilterTempVec3;
    Scalar3* cudaPCGPrecondTempVec3;
};
