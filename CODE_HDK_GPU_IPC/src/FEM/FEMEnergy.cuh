#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "UTILS/GeometryManager.hpp"

namespace FEMENERGY {

    __device__ __host__
    void __calculateDm2D_double(const double3* vertexes, const uint3& index, MATHUTILS::Matrix2x2d& M);

    __device__ __host__
    void __calculateDs2D_double(const double3* vertexes, const uint3& index, MATHUTILS::Matrix3x2d& M);
    
    __device__ __host__
    void __calculateDms3D_double(const double3* vertexes, const uint4& index, MATHUTILS::Matrix3x3d& M);
    
    __device__
    MATHUTILS::Matrix6x6d __project_BaraffWitkinStretch_H(const MATHUTILS::Matrix3x2d& F);
    
    __device__ 
    MATHUTILS::Matrix9x9d __project_StabbleNHK_H_3D(const MATHUTILS::Matrix3x3d& sigma, const MATHUTILS::Matrix3x3d& U, const MATHUTILS::Matrix3x3d& V, const double& lengthRate, const double& volumRate);
    
    __device__
    MATHUTILS::Matrix9x9d project_ARAP_H_3D(const MATHUTILS::Matrix3x3d& Sigma, const MATHUTILS::Matrix3x3d& U, const MATHUTILS::Matrix3x3d& V, const double& lengthRate);

    __device__
    MATHUTILS::Matrix3x3d computePEPF_ARAP_double(const MATHUTILS::Matrix3x3d& F, const MATHUTILS::Matrix3x3d& Sigma, const MATHUTILS::Matrix3x3d& U, const MATHUTILS::Matrix3x3d& V, const double& lengthRate);

    __device__ 
    MATHUTILS::Matrix9x9d __project_ANIOSI5_H_3D(const MATHUTILS::Matrix3x3d& F, const MATHUTILS::Matrix3x3d& sigma, const MATHUTILS::Matrix3x3d& U, const MATHUTILS::Matrix3x3d& V, const double3& fiber_direction, const double& scale, const double& contract_length);

    __device__ 
    MATHUTILS::Matrix3x3d __computePEPF_StableNHK3D_double(const MATHUTILS::Matrix3x3d& F, const MATHUTILS::Matrix3x3d& Sigma, const MATHUTILS::Matrix3x3d& U, const MATHUTILS::Matrix3x3d& V, double lengthRate, double volumRate);
    
    __device__
    MATHUTILS::Matrix3x2d __computePEPF_BaraffWitkinStretch_double(const MATHUTILS::Matrix3x2d& F, double stretchStiff, double shearStiff);
    
    __device__ 
    MATHUTILS::Matrix3x3d __computePEPF_Aniostropic3D_double(const MATHUTILS::Matrix3x3d& F, double3 fiber_direction, const double& scale, const double& contract_length);
    
    __device__
    double __cal_BaraffWitkinStretch_energy(const double3* vertexes, const uint3& triangle, const MATHUTILS::Matrix2x2d& triDmInverse, const double& area, const double& stretchStiff, const double& shearhStiff);
    
    __device__ 
    double __cal_StabbleNHK_energy_3D(const double3* vertexes, const uint4& tetrahedra, const MATHUTILS::Matrix3x3d& DmInverse, const double& volume, const double& lenRate, const double& volRate);

    __device__
    double __cal_ARAP_energy_3D(const double3* vertexes, const uint4& tetrahedra, const MATHUTILS::Matrix3x3d& DmInverse, const double& volume, const double& lenRate);

    __device__
    double __cal_bending_energy(const double3* vertexes, const double3* rest_vertexes, const uint2& edge, const uint2& adj, double length, double bendStiff);
    
    __global__
    void _calculate_bending_gradient_hessian(const double3* vertexes, const double3* rest_vertexes, const uint2* edges, const uint2* edges_adj_vertex,
        MATHUTILS::Matrix12x12d* Hessians, uint4* Indices, uint32_t offset, double3* gradient, int edgeNum, double bendStiff, double IPC_dt);

    __device__
    MATHUTILS::Matrix9x12d __computePFDsPX3D_double(const MATHUTILS::Matrix3x3d& InverseDm);

    __device__
    MATHUTILS::Matrix6x12d __computePFDsPX3D_6x12_double(const MATHUTILS::Matrix2x2d& InverseDm);

    __device__
    MATHUTILS::Matrix6x9d __computePFDsPX3D_6x9_double(const MATHUTILS::Matrix2x2d& InverseDm);

    __device__
    MATHUTILS::Matrix3x6d __computePFDsPX3D_3x6_double(const double& InverseDm);

    __device__
    MATHUTILS::Matrix9x12d __computePFDmPX3D_double(const MATHUTILS::Matrix12x9d& PDmPx, const MATHUTILS::Matrix3x3d& Ds, const MATHUTILS::Matrix3x3d& DmInv);

    __device__
    MATHUTILS::Matrix6x12d __computePFDmPX3D_6x12_double(const MATHUTILS::Matrix12x4d& PDmPx, const MATHUTILS::Matrix3x2d& Ds, const MATHUTILS::Matrix2x2d& DmInv);

    __device__
    MATHUTILS::Matrix3x6d __computePFDmPX3D_3x6_double(const MATHUTILS::Vector6& PDmPx, const double3& Ds, const double& DmInv);

    __device__
    MATHUTILS::Matrix6x9d __computePFDmPX3D_6x9_double(const MATHUTILS::Matrix9x4d& PDmPx, const MATHUTILS::Matrix3x2d& Ds, const MATHUTILS::Matrix2x2d& DmInv);

    __device__
    MATHUTILS::Matrix9x12d __computePFPX3D_double(const MATHUTILS::Matrix3x3d& InverseDm);

    __global__
    void _calculate_fem_gradient_hessian(MATHUTILS::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
        MATHUTILS::Matrix12x12d* Hessians, uint32_t offset, const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double IPC_dt);

    __global__
    void _calculate_triangle_fem_gradient_hessian(MATHUTILS::Matrix2x2d* trimInverses, const double3* vertexes, const uint3* triangles,
        MATHUTILS::Matrix9x9d* Hessians, uint32_t offset, const double* area, double3* gradient, int triangleNum, double stretchStiff, double shearhStiff, double IPC_dt);

    __global__
    void _calculate_fem_gradient(MATHUTILS::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
        const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double dt);

    __global__
    void _calculate_triangle_fem_gradient(MATHUTILS::Matrix2x2d* trimInverses, const double3* vertexes, const uint3* triangles,
        const double* area, double3* gradient, int triangleNum, double stretchStiff, double shearhStiff, double IPC_dt);

    __global__
    void _computeXTilta(int* _btype, double3* _velocities, double3* _o_vertexes, double3* _xTilta, double ipc_dt, double rate, int numbers);

    void computeXTilta(std::unique_ptr<GeometryManager>& instance, const double& rate);

} // namespace FEMENERGY

