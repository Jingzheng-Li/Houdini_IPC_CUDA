

#pragma once

#include "BlockHessian.cuh"
#include "LBVH.cuh"
#include "PCGSolver.cuh"

#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "MathUtils.cuh"


namespace __GPUIPC__ {

__global__ void _getBarrierEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                               const Scalar3* rest_vertexes, int4* _collisionPair,
                                               Scalar _Kappa, Scalar _dHat, int cpNum);

__global__ void _computeGroundEnergy_Reduction(Scalar* squeue, const Scalar3* vertexes,
                                               const Scalar* g_offset, const Scalar3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               Scalar dHat, Scalar Kappa, int number);

__global__ void _getFrictionEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                const Scalar3* o_vertexes,
                                                const int4* _collisionPair, int cpNum, Scalar dt,
                                                const Scalar2* distCoord,
                                                const __MATHUTILS__::Matrix3x2S* tanBasis,
                                                const Scalar* lastH, Scalar fricDHat, Scalar eps

);

__global__ void _getFrictionEnergy_gd_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                   const Scalar3* o_vertexes,
                                                   const Scalar3* _normal,
                                                   const uint32_t* _collisionPair_gd, int gpNum,
                                                   Scalar dt, const Scalar* lastH, Scalar eps

);

void calKineticGradient(Scalar3* _vertexes, Scalar3* _xTilta, Scalar3* _gradient, Scalar* _masses,
                        int numbers);

void buildGP(std::unique_ptr<GeometryManager>& instance);

void GroundCollisionDetect(std::unique_ptr<GeometryManager>& instance);

void calBarrierGradientAndHessian(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr,
                                  Scalar3* _gradient, Scalar mKappa);

void calBarrierGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient,
                        Scalar mKap);

void calFrictionHessian(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr);

void calFrictionGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient);

Scalar ground_largestFeasibleStepSize(std::unique_ptr<GeometryManager>& instance, Scalar slackness,
                                      Scalar* mqueue);

Scalar self_largestFeasibleStepSize(std::unique_ptr<GeometryManager>& instance, Scalar slackness,
                                    Scalar* mqueue, int numbers);

Scalar cfl_largestSpeed(std::unique_ptr<GeometryManager>& instance, Scalar* mqueue);

bool isIntersected(std::unique_ptr<GeometryManager>& instance,
                   std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr);

bool checkGroundIntersection(std::unique_ptr<GeometryManager>& instance);

void computeCloseGroundVal(std::unique_ptr<GeometryManager>& instance);

void computeSelfCloseVal(std::unique_ptr<GeometryManager>& instance);

bool checkCloseGroundVal(std::unique_ptr<GeometryManager>& instance);

bool checkSelfCloseVal(std::unique_ptr<GeometryManager>& instance);

Scalar2 minMaxGroundDist(std::unique_ptr<GeometryManager>& instance);

Scalar2 minMaxSelfDist(std::unique_ptr<GeometryManager>& instance);

void initKappa(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr,
               std::unique_ptr<PCGSolver>& PCG_ptr);

void suggestKappa(std::unique_ptr<GeometryManager>& instance, Scalar& kappa);

void upperBoundKappa(std::unique_ptr<GeometryManager>& instance, Scalar& kappa);

void buildFrictionSets(std::unique_ptr<GeometryManager>& instance);

};  // namespace __GPUIPC__
