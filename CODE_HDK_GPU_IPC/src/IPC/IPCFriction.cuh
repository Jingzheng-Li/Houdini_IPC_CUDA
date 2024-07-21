
// #pragma once

// #include "UTILS/CUDAUtils.hpp"
// #include "UTILS/MathUtils.cuh"

// namespace IPCFRICTION {

//     __device__ __host__ MATHUTILS::Matrix2x2d __M3x2_transpose_self__multiply(const MATHUTILS::Matrix3x2d& A);

//     __device__ __host__ double2 __M3x2_transpose_vec3__multiply(const MATHUTILS::Matrix3x2d& A, const double3& b);

//     __device__ __host__ double2 __M2x2_v2__multiply(const MATHUTILS::Matrix2x2d& A, const double2& b);

//     __device__ __host__ void computeTangentBasis_PT(
//         double3 v0,
//         double3 v1,
//         double3 v2,
//         double3 v3,
//         MATHUTILS::Matrix3x2d& basis);

//     __device__ __host__ void computeClosestPoint_PT(
//         double3 v0,
//         double3 v1,
//         double3 v2,
//         double3 v3,
//         double2& beta);

//     __device__ __host__ __device__ __host__ void computeRelDX_PT(
//         const double3 dx0,
//         const double3 dx1,
//         const double3 dx2,
//         const double3 dx3,
//         double beta1, double beta2,
//         double3& relDX);

//     __device__ __host__ void liftRelDXTanToMesh_PT(
//         const double2& relDXTan,
//         const MATHUTILS::Matrix3x2d& basis,
//         double beta1, double beta2,
//         MATHUTILS::Vector12& TTTDX);
        
//     __device__ __host__ void computeT_PT(
//         MATHUTILS::Matrix3x2d basis,
//         double beta1, double beta2,
//         MATHUTILS::Matrix12x2d& T);

//     __device__ __host__ void computeTangentBasis_EE(
//         const double3& v0,
//         const double3& v1,
//         const double3& v2,
//         const double3& v3,
//         MATHUTILS::Matrix3x2d& basis);

//     __device__ __host__ void computeClosestPoint_EE(
//         const double3& v0,
//         const double3& v1,
//         const double3& v2,
//         const double3& v3,
//         double2& gamma);

//     __device__ __host__ void computeRelDX_EE(
//         const double3& dx0,
//         const double3& dx1,
//         const double3& dx2,
//         const double3& dx3,
//         double gamma1, double gamma2,
//         double3& relDX);

//     __device__ __host__ void computeT_EE(
//         const MATHUTILS::Matrix3x2d& basis,
//         double gamma1, double gamma2,
//         MATHUTILS::Matrix12x2d& T);

//     __device__ __host__ void liftRelDXTanToMesh_EE(
//         const double2& relDXTan,
//         const MATHUTILS::Matrix3x2d& basis,
//         double gamma1, double gamma2,
//         MATHUTILS::Vector12& TTTDX);

//     __device__ __host__ void computeTangentBasis_PE(
//         const double3& v0,
//         const double3& v1,
//         const double3& v2,
//         MATHUTILS::Matrix3x2d& basis);

//     __device__ __host__ void computeClosestPoint_PE(
//         const double3& v0,
//         const double3& v1,
//         const double3& v2,
//         double& yita);

//     __device__ __host__ void computeRelDX_PE(
//         const double3& dx0,
//         const double3& dx1,
//         const double3& dx2,
//         double yita,
//         double3& relDX);

//     __device__ __host__ void liftRelDXTanToMesh_PE(
//         const double2& relDXTan,
//         const MATHUTILS::Matrix3x2d& basis,
//         double yita,
//         MATHUTILS::Vector9& TTTDX);

//     __device__ __host__ void computeT_PE(
//         const MATHUTILS::Matrix3x2d& basis,
//         double yita,
//         MATHUTILS::Matrix9x2d& T);

//     __device__ __host__ void computeTangentBasis_PP(
//         const double3& v0,
//         const double3& v1,
//         MATHUTILS::Matrix3x2d& basis);

//     __device__ __host__ void computeRelDX_PP(
//         const double3& dx0,
//         const double3& dx1,
//         double3& relDX);

//     __device__ __host__ void liftRelDXTanToMesh_PP(
//         const double2& relDXTan,
//         const MATHUTILS::Matrix3x2d& basis,
//         MATHUTILS::Vector6& TTTDX);

//     __device__ __host__ void computeT_PP(
//         const MATHUTILS::Matrix3x2d& basis,
//         MATHUTILS::Matrix6x2d& T);

//     // static friction clamping model
// // C0 clamping
//     __device__ __host__ void f0_SF_C0(double x2, double eps_f, double& f0);

//     __device__ __host__ void f1_SF_div_relDXNorm_C0(double eps_f, double& result);

//     __device__ __host__ void f2_SF_C0(double eps_f, double& f2);

//     // C1 clamping
//     __device__ __host__ void f0_SF_C1(double x2, double eps_f, double& f0);

//     __device__ __host__ void f1_SF_div_relDXNorm_C1(double x2, double eps_f, double& result);

//     __device__ __host__ void f2_SF_C1(double x2, double eps_f, double& f2);

//     // C2 clamping
//     __device__ __host__ void f0_SF_C2(double x2, double eps_f, double& f0);

//     __device__ __host__ void f1_SF_div_relDXNorm_C2(double x2, double eps_f, double& result);

//     __device__ __host__ void f2_SF_C2(double x2, double eps_f, double& f2);

//     // interfaces
//     __device__ __host__ void f0_SF(double relDXSqNorm, double eps_f, double& f0);

//     __device__ __host__ void f1_SF_div_relDXNorm(double relDXSqNorm, double eps_f, double& result);

//     __device__ __host__ void f2_SF(double relDXSqNorm, double eps_f, double& f2);

// }; // namespace IPCFRICTION


