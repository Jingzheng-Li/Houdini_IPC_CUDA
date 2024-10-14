
#pragma once

#include "BlockHessian.cuh"
#include "GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"

namespace __FEMENERGY__ {

__device__ __host__ void __calculateDm2D_Scalar(const Scalar3* vertexes, const uint3& index,
                                                __MATHUTILS__::Matrix2x2S& M);

__device__ __host__ void __calculateDs2D_Scalar(const Scalar3* vertexes, const uint3& index,
                                                __MATHUTILS__::Matrix3x2S& M);

__device__ __host__ void __calculateDms3D_Scalar(const Scalar3* vertexes, const uint4& index,
                                                 __MATHUTILS__::Matrix3x3S& M);

__device__ Scalar __cal_BaraffWitkinStretch_energy(const Scalar3* vertexes, const uint3& triangle,
                                                   const __MATHUTILS__::Matrix2x2S& triDmInverse,
                                                   const Scalar& area, const Scalar& stretchStiff,
                                                   const Scalar& shearhStiff);

__device__ Scalar __cal_StabbleNHK_energy_3D(const Scalar3* vertexes, const uint4& tetrahedra,
                                             const __MATHUTILS__::Matrix3x3S& DmInverse,
                                             const Scalar& volume, const Scalar& lenRate,
                                             const Scalar& volRate);

__device__ Scalar __cal_ARAP_energy_3D(const Scalar3* vertexes, const uint4& tetrahedra,
                                       const __MATHUTILS__::Matrix3x3S& DmInverse,
                                       const Scalar& volume, const Scalar& lenRate);

__device__ Scalar __cal_bending_energy(const Scalar3* vertexes, const Scalar3* rest_vertexes,
                                       const uint2& edge, const int2& adj, Scalar length,
                                       Scalar bendStiff);

__device__ __MATHUTILS__::Matrix9x12S __computePFDsPX3D_Scalar(
    const __MATHUTILS__::Matrix3x3S& InverseDm);

__device__ __MATHUTILS__::Matrix6x12S __computePFDsPX3D_6x12_Scalar(
    const __MATHUTILS__::Matrix2x2S& InverseDm);

__device__ __MATHUTILS__::Matrix6x9S __computePFDsPX3D_6x9_Scalar(
    const __MATHUTILS__::Matrix2x2S& InverseDm);

__device__ __MATHUTILS__::Matrix3x6S __computePFDsPX3D_3x6_Scalar(const Scalar& InverseDm);

__device__ __MATHUTILS__::Matrix9x12S __computePFDmPX3D_Scalar(
    const __MATHUTILS__::Matrix12x9S& PDmPx, const __MATHUTILS__::Matrix3x3S& Ds,
    const __MATHUTILS__::Matrix3x3S& DmInv);

__device__ __MATHUTILS__::Matrix6x12S __computePFDmPX3D_6x12_Scalar(
    const __MATHUTILS__::Matrix12x4S& PDmPx, const __MATHUTILS__::Matrix3x2S& Ds,
    const __MATHUTILS__::Matrix2x2S& DmInv);

__device__ __MATHUTILS__::Matrix3x6S __computePFDmPX3D_3x6_Scalar(
    const __MATHUTILS__::Vector6S& PDmPx, const Scalar3& Ds, const Scalar& DmInv);

__device__ __MATHUTILS__::Matrix6x9S __computePFDmPX3D_6x9_Scalar(
    const __MATHUTILS__::Matrix9x4S& PDmPx, const __MATHUTILS__::Matrix3x2S& Ds,
    const __MATHUTILS__::Matrix2x2S& DmInv);

__device__ __MATHUTILS__::Matrix9x12S __computePFPX3D_Scalar(
    const __MATHUTILS__::Matrix3x3S& InverseDm);

__global__ void _getFEMEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                           const uint4* tetrahedras,
                                           const __MATHUTILS__::Matrix3x3S* DmInverses,
                                           const Scalar* volume, int tetrahedraNum, Scalar lenRate,
                                           Scalar volRate);

__global__ void _computeBoundConstraintEnergy_Reduction(Scalar* squeue, const Scalar3* vertexes,
                                                       const Scalar3* targetVert,
                                                       const uint32_t* targetInd, Scalar motionRate,
                                                       Scalar rate, int number);

__global__ void _get_triangleFEMEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                    const uint3* triangles,
                                                    const __MATHUTILS__::Matrix2x2S* triDmInverses,
                                                    const Scalar* area, int trianglesNum,
                                                    Scalar stretchStiff, Scalar shearStiff);

__global__ void _getRestStableNHKEnergy_Reduction_3D(Scalar* squeue, const Scalar* volume,
                                                     int tetrahedraNum, Scalar lenRate,
                                                     Scalar volRate);

__global__ void _computeSoftConstraintEnergy_Reduction(
    Scalar* squeue, const Scalar3* vertexes, const Scalar3* softTargetPos,
    const uint32_t* softTargetIds, Scalar softStiffness, Scalar rate, int number);

__global__ void _getBendingEnergy_Reduction(Scalar* squeue, const Scalar3* vertexes,
                                            const Scalar3* rest_vertexex, const uint2* edges,
                                            const uint2* edge_adj_vertex, int edgesNum,
                                            Scalar bendStiff);

void calculate_fem_gradient(__MATHUTILS__::Matrix3x3S* DmInverses, const Scalar3* vertexes,
                            const uint4* tetrahedras, const Scalar* volume, Scalar3* gradient,
                            int tetrahedraNum, Scalar lenRate, Scalar volRate, Scalar dt);

void calculate_triangle_fem_gradient(__MATHUTILS__::Matrix2x2S* triDmInverses,
                                     const Scalar3* vertexes, const uint3* triangles,
                                     const Scalar* area, Scalar3* gradient, int triangleNum,
                                     Scalar stretchStiff, Scalar shearStiff, Scalar IPC_dt);

void calculate_fem_gradient_hessian(__MATHUTILS__::Matrix3x3S* DmInverses, const Scalar3* vertexes,
                                    const uint4* tetrahedras, __MATHUTILS__::Matrix12x12S* Hessians,
                                    const uint32_t& offset, const Scalar* volume, Scalar3* gradient,
                                    int tetrahedraNum, Scalar lenRate, Scalar volRate,
                                    Scalar IPC_dt);

void calculate_triangle_fem_gradient_hessian(__MATHUTILS__::Matrix2x2S* triDmInverses,
                                             const Scalar3* vertexes, const uint3* triangles,
                                             __MATHUTILS__::Matrix9x9S* Hessians,
                                             const uint32_t& offset, const Scalar* area,
                                             Scalar3* gradient, int triangleNum,
                                             Scalar stretchStiff, Scalar shearStiff, Scalar IPC_dt);

void calculate_bending_gradient_hessian(const Scalar3* vertexes, const Scalar3* rest_vertexes,
                                        const uint2* edges, const uint2* edges_adj_vertex,
                                        __MATHUTILS__::Matrix12x12S* Hessians, uint4* Indices,
                                        const uint32_t& offset, Scalar3* gradient, int edgeNum,
                                        Scalar bendStiff, Scalar IPC_dt);

void computeGroundGradient(std::unique_ptr<GeometryManager>& instance, BlockHessian& blockHessian,
                           Scalar3* _gradient, Scalar mKap);

void computeGroundGradientAndHessian(std::unique_ptr<GeometryManager>& instance,
                                     BlockHessian& blockHessian, Scalar3* _gradient);

void computeBoundConstraintGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient);

void computeBoundConstraintGradientAndHessian(std::unique_ptr<GeometryManager>& instance,
                                             BlockHessian& blockHessian, Scalar3* _gradient);

void computeSoftConstraintGradient(std::unique_ptr<GeometryManager>& instance,
                                           Scalar3* _gradient);

void computeSoftConstraintGradientAndHessian(std::unique_ptr<GeometryManager>& instance,
                                                     BlockHessian& blockHessian,
                                                     Scalar3* _gradient);

};  // namespace __FEMENERGY__
