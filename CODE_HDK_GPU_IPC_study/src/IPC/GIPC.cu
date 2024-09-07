#include "GIPC.cuh"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <fstream>

#include "zensim/math/Complex.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/math/matrix/Eigen.hpp"
#include "zensim/math/MathUtils.h"

#include "ACCD/ACCD.cuh"
#include "FEM/FEMEnergy.cuh"
#include "IPCFriction.cuh"
#include "GIPCPDerivative.cuh"


#define RANK 2
#define NEWF
#define MAKEPD2
#define OLDBARRIER2

namespace GPUIPC {

__global__
void _calFrictionHessian_gd(const double3* _vertexes, const double3* _o_vertexes, const double3* _normal, const uint32_t* _last_collisionPair_gd,
    MATHUTILS::Matrix3x3d* H3x3, uint32_t* D1Index, 
    int number, double dt, double eps2, double* lastH, double coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double eps = sqrt(eps2);
    int gidx = _last_collisionPair_gd[idx];
    double multiplier_vI = coef * lastH[idx];
    MATHUTILS::Matrix3x3d H_vI;

    double3 Vdiff = MATHUTILS::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 normal = *_normal;
    double3 VProj = MATHUTILS::__minus(Vdiff, MATHUTILS::__s_vec_multiply(normal, MATHUTILS::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = MATHUTILS::__squaredNorm(VProj);

    if (VProjMag2 > eps2) {
        double VProjMag = sqrt(VProjMag2);

        MATHUTILS::Matrix2x2d projH;
        MATHUTILS::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

        double eigenValues[2];
        int eigenNum = 0;
        double2 eigenVecs[2];
        MATHUTILS::__makePD2x2(VProj.x * VProj.x * -multiplier_vI / VProjMag2 / VProjMag + (multiplier_vI / VProjMag),
            VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
            VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
            VProj.z * VProj.z * -multiplier_vI / VProjMag2 / VProjMag + (multiplier_vI / VProjMag),
            eigenValues, eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                MATHUTILS::Matrix2x2d eigenMatrix = MATHUTILS::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = MATHUTILS::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = MATHUTILS::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        MATHUTILS::__set_Mat_val(H_vI, projH.m[0][0], 0, projH.m[0][1], 0, 0, 0, projH.m[1][0], 0, projH.m[1][1]);
    }
    else {
        MATHUTILS::__set_Mat_val(H_vI, (multiplier_vI / eps), 0, 0, 0, 0, 0, 0, 0, (multiplier_vI / eps));
    }

    H3x3[idx] = H_vI;
    D1Index[idx] = gidx;
}


__global__
void _calFrictionHessian(const double3* _vertexes, const double3* _o_vertexes, const int4* _last_collisionPair,
    MATHUTILS::Matrix12x12d* H12x12, MATHUTILS::Matrix9x9d* H9x9,
    MATHUTILS::Matrix6x6d* H6x6, uint4* D4Index, uint3* D3Index, uint2* D2Index, uint32_t* _cpNum,
    int number, double dt, double2* distCoord,
    MATHUTILS::Matrix3x2d* tanBasis, double eps2, double* lastH, double coef, int offset4, int offset3, int offset2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _last_collisionPair[idx];
    double eps = sqrt(eps2);
    double3 relDX3D;
    if (MMCVIDI.x >= 0) {

        IPCFRICTION::computeRelDX_EE(MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            MATHUTILS::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
            distCoord[idx].y, relDX3D);


        MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis[idx]);
        double2 relDX = MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D);
        double relDXSqNorm = MATHUTILS::__squaredNorm(relDX);
        double relDXNorm = sqrt(relDXSqNorm);
        MATHUTILS::Matrix12x2d T;
        IPCFRICTION::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        MATHUTILS::Matrix2x2d M2;
        if (relDXSqNorm > eps2) {
            MATHUTILS::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = MATHUTILS::__Mat2x2_minus(M2, MATHUTILS::__s_Mat2x2_multiply(
                MATHUTILS::__v2_vec2_toMat2x2(relDX, relDX), 1 / (relDXSqNorm * relDXNorm)));
        }
        else {
            double f1_div_relDXNorm;
            IPCFRICTION::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            double f2;
            IPCFRICTION::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {

                MATHUTILS::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = MATHUTILS::__Mat2x2_minus(M2, MATHUTILS::__s_Mat2x2_multiply(
                    MATHUTILS::__v2_vec2_toMat2x2(relDX, relDX), (f1_div_relDXNorm - f2) / relDXSqNorm));
            }
            else {
                MATHUTILS::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        MATHUTILS::Matrix2x2d projH;
        MATHUTILS::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

        double eigenValues[2];
        int eigenNum = 0;
        double2 eigenVecs[2];
        MATHUTILS::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues, eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                MATHUTILS::Matrix2x2d eigenMatrix = MATHUTILS::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = MATHUTILS::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = MATHUTILS::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        MATHUTILS::Matrix12x2d TM2 = MATHUTILS::__M12x2_M2x2_Multiply(T, projH);

        MATHUTILS::Matrix12x12d HessianBlock = MATHUTILS::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
            coef * lastH[idx]);
        int Hidx = atomicAdd(_cpNum + 4, 1);
        Hidx += offset4;
        H12x12[Hidx] = HessianBlock;
        D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
    }
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {

            MMCVIDI.x = v0I;
            IPCFRICTION::computeRelDX_PP(MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis[idx]);
            double2 relDX = MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D);
            double relDXSqNorm = MATHUTILS::__squaredNorm(relDX);
            double relDXNorm = sqrt(relDXSqNorm);
            MATHUTILS::Matrix6x2d T;
            IPCFRICTION::computeT_PP(tanBasis[idx], T);
            MATHUTILS::Matrix2x2d M2;
            if (relDXSqNorm > eps2) {
                MATHUTILS::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = MATHUTILS::__Mat2x2_minus(M2, MATHUTILS::__s_Mat2x2_multiply(
                    MATHUTILS::__v2_vec2_toMat2x2(relDX, relDX), 1 / (relDXSqNorm * relDXNorm)));
            }
            else {
                double f1_div_relDXNorm;
                IPCFRICTION::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                IPCFRICTION::f2_SF(relDXSqNorm, eps, f2);
                if (f2 != f1_div_relDXNorm && relDXSqNorm) {

                    MATHUTILS::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = MATHUTILS::__Mat2x2_minus(M2, MATHUTILS::__s_Mat2x2_multiply(
                        MATHUTILS::__v2_vec2_toMat2x2(relDX, relDX), (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else {
                    MATHUTILS::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            MATHUTILS::Matrix2x2d projH;
            MATHUTILS::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

            double eigenValues[2];
            int eigenNum = 0;
            double2 eigenVecs[2];
            MATHUTILS::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues, eigenNum, eigenVecs);
            for (int i = 0; i < eigenNum; i++) {
                if (eigenValues[i] > 0) {
                    MATHUTILS::Matrix2x2d eigenMatrix = MATHUTILS::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix = MATHUTILS::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = MATHUTILS::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            MATHUTILS::Matrix6x2d TM2 = MATHUTILS::__M6x2_M2x2_Multiply(T, projH);

            MATHUTILS::Matrix6x6d HessianBlock = MATHUTILS::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T), coef * lastH[idx]);

            int Hidx = atomicAdd(_cpNum + 2, 1);
            Hidx += offset2;
            H6x6[Hidx] = HessianBlock;
            D2Index[Hidx] = make_uint2(MMCVIDI.x, MMCVIDI.y);
        }
        else if (MMCVIDI.w < 0) {

            MMCVIDI.x = v0I;
            IPCFRICTION::computeRelDX_PE(MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x, relDX3D);

            MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis[idx]);
            double2 relDX = MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D);
            double relDXSqNorm = MATHUTILS::__squaredNorm(relDX);
            double relDXNorm = sqrt(relDXSqNorm);
            MATHUTILS::Matrix9x2d T;
            IPCFRICTION::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
            MATHUTILS::Matrix2x2d M2;
            if (relDXSqNorm > eps2) {
                MATHUTILS::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = MATHUTILS::__Mat2x2_minus(M2, MATHUTILS::__s_Mat2x2_multiply(
                    MATHUTILS::__v2_vec2_toMat2x2(relDX, relDX), 1 / (relDXSqNorm * relDXNorm)));
            }
            else {
                double f1_div_relDXNorm;
                IPCFRICTION::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                IPCFRICTION::f2_SF(relDXSqNorm, eps, f2);
                if (f2 != f1_div_relDXNorm && relDXSqNorm) {

                    MATHUTILS::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = MATHUTILS::__Mat2x2_minus(M2, MATHUTILS::__s_Mat2x2_multiply(
                        MATHUTILS::__v2_vec2_toMat2x2(relDX, relDX), (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else {
                    MATHUTILS::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            MATHUTILS::Matrix2x2d projH;
            MATHUTILS::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

            double eigenValues[2];
            int eigenNum = 0;
            double2 eigenVecs[2];
            MATHUTILS::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues, eigenNum, eigenVecs);
            for (int i = 0; i < eigenNum; i++) {
                if (eigenValues[i] > 0) {
                    MATHUTILS::Matrix2x2d eigenMatrix = MATHUTILS::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix = MATHUTILS::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = MATHUTILS::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            MATHUTILS::Matrix9x2d TM2 = MATHUTILS::__M9x2_M2x2_Multiply(T, projH);

            MATHUTILS::Matrix9x9d HessianBlock = MATHUTILS::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T), coef * lastH[idx]);
            int Hidx = atomicAdd(_cpNum + 3, 1);
            Hidx += offset3;
            H9x9[Hidx] = HessianBlock;
            D3Index[Hidx] = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
        }
        else {
            MMCVIDI.x = v0I;
            IPCFRICTION::computeRelDX_PT(MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
                distCoord[idx].y, relDX3D);


            MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis[idx]);
            double2 relDX = MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D);
            double relDXSqNorm = MATHUTILS::__squaredNorm(relDX);
            double relDXNorm = sqrt(relDXSqNorm);
            MATHUTILS::Matrix12x2d T;
            IPCFRICTION::computeT_PT(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
            MATHUTILS::Matrix2x2d M2;
            if (relDXSqNorm > eps2) {
                MATHUTILS::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = MATHUTILS::__Mat2x2_minus(M2, MATHUTILS::__s_Mat2x2_multiply(
                    MATHUTILS::__v2_vec2_toMat2x2(relDX, relDX), 1 / (relDXSqNorm * relDXNorm)));
            }
            else {
                double f1_div_relDXNorm;
                IPCFRICTION::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                double f2;
                IPCFRICTION::f2_SF(relDXSqNorm, eps, f2);
                if (f2 != f1_div_relDXNorm && relDXSqNorm) {

                    MATHUTILS::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = MATHUTILS::__Mat2x2_minus(M2, MATHUTILS::__s_Mat2x2_multiply(
                        MATHUTILS::__v2_vec2_toMat2x2(relDX, relDX), (f1_div_relDXNorm - f2) / relDXSqNorm));
                }
                else {
                    MATHUTILS::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            MATHUTILS::Matrix2x2d projH;
            MATHUTILS::__set_Mat2x2_val_column(projH, make_double2(0, 0), make_double2(0, 0));

            double eigenValues[2];
            int eigenNum = 0;
            double2 eigenVecs[2];
            MATHUTILS::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues, eigenNum, eigenVecs);
            for (int i = 0; i < eigenNum; i++) {
                if (eigenValues[i] > 0) {
                    MATHUTILS::Matrix2x2d eigenMatrix = MATHUTILS::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix = MATHUTILS::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = MATHUTILS::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            MATHUTILS::Matrix12x2d TM2 = MATHUTILS::__M12x2_M2x2_Multiply(T, projH);

            MATHUTILS::Matrix12x12d HessianBlock = MATHUTILS::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T), coef * lastH[idx]);
            int Hidx = atomicAdd(_cpNum + 4, 1);
            Hidx += offset4;
            H12x12[Hidx] = HessianBlock;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

        }
    }
}


__global__
void _calBarrierGradient(const double3* _vertexes, const double3* _rest_vertexes, const int4* _collisionPair, double3* _gradient, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
#ifdef NEWF
            double dis;
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            double d_hat_sqrt = sqrt(dHat);
            MATHUTILS::Matrix12x9d PFPxT;
            GIPCPDERIV::pFpx_ee2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], d_hat_sqrt,
                PFPxT);
            double I5 = pow(dis / d_hat_sqrt, 2);
            MATHUTILS::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8] = dis / d_hat_sqrt;
#else

            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            MATHUTILS::Matrix3x3d Ds;
            MATHUTILS::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(v0, MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double dis = MATHUTILS::__v_vec_dot(v1, normal);
            if (dis < 0) {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis = -dis;
            }

            double3 pos2 = MATHUTILS::__add(_vertexes[MMCVIDI.z], MATHUTILS::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 = MATHUTILS::__add(_vertexes[MMCVIDI.w], MATHUTILS::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = MATHUTILS::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = MATHUTILS::__minus(pos3, _vertexes[MMCVIDI.x]);

            MATHUTILS::Matrix3x3d Dm, DmInv;
            MATHUTILS::__set_Mat_val_column(Dm, u0, u1, u2);

            MATHUTILS::__Inverse(Dm, DmInv);

            MATHUTILS::Matrix3x3d F;
            MATHUTILS::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = MATHUTILS::__M_v_multiply(F, normal);
            double I5 = MATHUTILS::__squaredNorm(FxN);

            MATHUTILS::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            MATHUTILS::Matrix3x3d fnn;

            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);

            MATHUTILS::__M_Mat_multiply(F, nn, fnn);

            MATHUTILS::Vector9 tmp = MATHUTILS::__Mat3x3_to_vec9_double(fnn);

#endif

#if (RANK == 1)
            double judge = (2 * dHat * dHat * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 * dis / d_hat_sqrt;
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);
#elif (RANK == 2)
            //MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
            double judge2 = 2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5 * dis / dHat_sqrt;
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);

#elif (RANK == 3)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif

#ifdef NEWF
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply((PFPxT), flatten_pk1);
#else

            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(MATHUTILS::__Transpose9x12(PFPx), flatten_pk1);
#endif

            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }

        }
        else {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
            double I1 = c * c;
            if (I1 == 0) return;
            double dis;
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            double I2 = dis / dHat;
            dis = sqrt(dis);

            MATHUTILS::Matrix3x3d F;
            MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);

            MATHUTILS::Matrix3x3d g1, g2;

            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
            MATHUTILS::__M_Mat_multiply(F, nn, g1);
            nn = MATHUTILS::__v_vec_toMat(n2, n2);
            MATHUTILS::__M_Mat_multiply(F, nn, g2);

            MATHUTILS::Vector9 flatten_g1 = MATHUTILS::__Mat3x3_to_vec9_double(g1);
            MATHUTILS::Vector9 flatten_g2 = MATHUTILS::__Mat3x3_to_vec9_double(g2);

            MATHUTILS::Matrix12x9d PFPx;
            GIPCPDERIV::pFpx_pee(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);



#if (RANK == 1)
            double p1 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double p2 = Kappa * 2 * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif    


            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__add9(MATHUTILS::__s_vec9_multiply(flatten_g1, p1), MATHUTILS::__s_vec9_multiply(flatten_g2, p2));
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }
        }
    }
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
                double I1 = c * c;
                if (I1 == 0) return;
                double dis;
                MATHUTILS::__distancePointPoint(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis = sqrt(dis);

                MATHUTILS::Matrix3x3d F;
                MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.w]);

                MATHUTILS::Matrix3x3d g1, g2;

                MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
                MATHUTILS::__M_Mat_multiply(F, nn, g1);
                nn = MATHUTILS::__v_vec_toMat(n2, n2);
                MATHUTILS::__M_Mat_multiply(F, nn, g2);

                MATHUTILS::Vector9 flatten_g1 = MATHUTILS::__Mat3x3_to_vec9_double(g1);
                MATHUTILS::Vector9 flatten_g2 = MATHUTILS::__Mat3x3_to_vec9_double(g2);

                MATHUTILS::Matrix12x9d PFPx;
                GIPCPDERIV::pFpx_ppp(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);
#if (RANK == 1)
                double p1 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = Kappa * 2 * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif  
                MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__add9(MATHUTILS::__s_vec9_multiply(flatten_g1, p1), MATHUTILS::__s_vec9_multiply(flatten_g2, p2));
                MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }
            }
            else {
#ifdef NEWF
                double dis;
                MATHUTILS::__distancePointPoint(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis = sqrt(dis);
                double d_hat_sqrt = sqrt(dHat);
                MATHUTILS::Vector6 PFPxT;
                GIPCPDERIV::pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5 = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if (RANK == 1)


                double judge = (2 * dHat * dHat * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) / I5;
                double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 * dis / d_hat_sqrt;
                double flatten_pk1 = fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
                //if (dis*dis<1e-2*dHat)
                    //flatten_pk1 = fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/;
#elif (RANK == 2)
                //double flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;

                double judge = -(4 * dHat * dHat * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
                double judge2 = 2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5 * dis / dHat_sqrt;
                double flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
                //if (dis*dis<1e-2*dHat)
                    //flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5;

#elif (RANK == 3)                        
                double flatten_pk1 = fnn * -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5;
#elif (RANK == 4)                        
                double flatten_pk1 = fnn * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5;
#elif (RANK == 5)                        
                double flatten_pk1 = fnn * -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5;
#elif (RANK == 6)                        
                double flatten_pk1 = fnn * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5;
#endif

                MATHUTILS::Vector6 gradient_vec = MATHUTILS::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds = v0;
                double dis = MATHUTILS::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal = MATHUTILS::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec = MATHUTILS::__v_vec_cross(vec_normal, target);
                double cos = MATHUTILS::__v_vec_dot(vec_normal, target);
                MATHUTILS::Matrix3x3d rotation;
                MATHUTILS::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                MATHUTILS::Vector6 PDmPx;
                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else {
                    MATHUTILS::Matrix3x3d cross_vec;
                    MATHUTILS::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = MATHUTILS::__Mat_add(rotation, MATHUTILS::__Mat_add(cross_vec, MATHUTILS::__S_Mat_multiply(MATHUTILS::__M_Mat_multiply(cross_vec, cross_vec), 1.0 / (1 + cos))));
                }

                double3 pos0 = MATHUTILS::__add(_vertexes[v0I], MATHUTILS::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = MATHUTILS::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 = MATHUTILS::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0 = uv1 - uv0;
                double Dm = u0;//PFPx
                double DmInv = 1 / u0;

                double3 F = MATHUTILS::__s_vec_multiply(Ds, DmInv);
                double I5 = MATHUTILS::__squaredNorm(F);

                double3 tmp = F;

#if (RANK == 1)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif
                MATHUTILS::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                MATHUTILS::Vector6 gradient_vec = MATHUTILS::__M6x3_v3_multiply(MATHUTILS::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                }
            }

        }
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
                double I1 = c * c;
                if (I1 == 0) return;
                double dis;
                MATHUTILS::__distancePointEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I2 = dis / dHat;
                dis = sqrt(dis);

                MATHUTILS::Matrix3x3d F;
                MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z]);

                MATHUTILS::Matrix3x3d g1, g2;

                MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
                MATHUTILS::__M_Mat_multiply(F, nn, g1);
                nn = MATHUTILS::__v_vec_toMat(n2, n2);
                MATHUTILS::__M_Mat_multiply(F, nn, g2);

                MATHUTILS::Vector9 flatten_g1 = MATHUTILS::__Mat3x3_to_vec9_double(g1);
                MATHUTILS::Vector9 flatten_g2 = MATHUTILS::__Mat3x3_to_vec9_double(g2);

                MATHUTILS::Matrix12x9d PFPx;
                GIPCPDERIV::pFpx_ppe(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
                double p1 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = Kappa * 2 * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif  
                MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__add9(MATHUTILS::__s_vec9_multiply(flatten_g1, p1), MATHUTILS::__s_vec9_multiply(flatten_g2, p2));
                MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }
            }
            else {
#ifdef NEWF
                double dis;
                MATHUTILS::__distancePointEdge(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis = sqrt(dis);
                double d_hat_sqrt = sqrt(dHat);
                MATHUTILS::Matrix9x4d PFPxT;
                GIPCPDERIV::pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt,
                    PFPxT);
                double I5 = pow(dis / d_hat_sqrt, 2);
                MATHUTILS::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;// = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);

#if (RANK == 1)


                double judge = (2 * dHat * dHat * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) / I5;
                double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 * dis / d_hat_sqrt;
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                //if (dis*dis<1e-2*dHat)
                    //flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
                //MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

                double judge = -(4 * dHat * dHat * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
                double judge2 = 2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5 * dis / dHat_sqrt;
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
                //if (dis*dis<1e-2*dHat)
                    //flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif

                MATHUTILS::Vector9 gradient_vec = MATHUTILS::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                MATHUTILS::Matrix3x2d Ds;
                MATHUTILS::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = MATHUTILS::__v_vec_cross(triangle_normal, target);
                double cos = MATHUTILS::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]), triangle_normal));
                double dis = MATHUTILS::__v_vec_dot(MATHUTILS::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                MATHUTILS::Matrix3x3d rotation;
                MATHUTILS::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                MATHUTILS::Matrix9x4d PDmPx;

                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1; rotation.m[1][1] = -1;
                }
                else {
                    MATHUTILS::Matrix3x3d cross_vec;
                    MATHUTILS::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = MATHUTILS::__Mat_add(rotation, MATHUTILS::__Mat_add(cross_vec, MATHUTILS::__S_Mat_multiply(MATHUTILS::__M_Mat_multiply(cross_vec, cross_vec), 1.0 / (1 + cos))));
                }

                double3 pos0 = MATHUTILS::__add(_vertexes[v0I], MATHUTILS::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = MATHUTILS::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 = MATHUTILS::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 = MATHUTILS::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = MATHUTILS::__M_v_multiply(rotation, edge_normal);

                double2 uv0 = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1 = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2 = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = MATHUTILS::__minus_v2(uv1, uv0);
                double2 u1 = MATHUTILS::__minus_v2(uv2, uv0);

                MATHUTILS::Matrix2x2d Dm;

                MATHUTILS::__set_Mat2x2_val_column(Dm, u0, u1);

                MATHUTILS::Matrix2x2d DmInv;
                MATHUTILS::__Inverse2x2(Dm, DmInv);

                MATHUTILS::Matrix3x2d F = MATHUTILS::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = MATHUTILS::__M3x2_v2_multiply(F, normal);
                double I5 = MATHUTILS::__squaredNorm(FxN);

                MATHUTILS::Matrix3x2d fnn;

                MATHUTILS::Matrix2x2d nn = MATHUTILS::__v2_vec2_toMat2x2(normal, normal);

                fnn = MATHUTILS::__M3x2_M2x2_Multiply(F, nn);

                MATHUTILS::Vector6 tmp = MATHUTILS::__Mat3x2_to_vec6_double(fnn);


#if (RANK == 1)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif

                MATHUTILS::Matrix6x9d PFPx = __computePFPX3D_6x9_double(DmInv);

                MATHUTILS::Vector9 gradient_vec = MATHUTILS::__M9x6_v6_multiply(MATHUTILS::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                }
            }

        }
        else {
#ifdef NEWF
            double dis;
            MATHUTILS::__distancePointTriangle(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            double d_hat_sqrt = sqrt(dHat);
            MATHUTILS::Matrix12x9d PFPxT;
            GIPCPDERIV::pFpx_pt2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], d_hat_sqrt,
                PFPxT);
            double I5 = pow(dis / d_hat_sqrt, 2);
            MATHUTILS::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8] = dis / d_hat_sqrt;
#else
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            MATHUTILS::Matrix3x3d Ds;
            MATHUTILS::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]), MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double dis = MATHUTILS::__v_vec_dot(v0, normal);
            //if (abs(dis) > dHat_sqrt) return;
            MATHUTILS::Matrix12x9d PDmPx;
            //bool is_flip = false;

            if (dis > 0) {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                //pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }
            else {
                dis = -dis;
                //pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("dHat_sqrt = %f,   dis = %f\n", dHat_sqrt, dis);
            }

            double3 pos0 = MATHUTILS::__add(_vertexes[v0I], MATHUTILS::__s_vec_multiply(normal, dHat_sqrt - dis));


            double3 u0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], pos0);

            MATHUTILS::Matrix3x3d Dm, DmInv;
            MATHUTILS::__set_Mat_val_column(Dm, u0, u1, u2);

            MATHUTILS::__Inverse(Dm, DmInv);

            MATHUTILS::Matrix3x3d F;//, Ftest;
            MATHUTILS::__M_Mat_multiply(Ds, DmInv, F);
            //MATHUTILS::__M_Mat_multiply(Dm, DmInv, Ftest);

            double3 FxN = MATHUTILS::__M_v_multiply(F, normal);
            double I5 = MATHUTILS::__squaredNorm(FxN);

            //printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));


            MATHUTILS::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);

            MATHUTILS::Matrix3x3d fnn;

            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);

            MATHUTILS::__M_Mat_multiply(F, nn, fnn);

            MATHUTILS::Vector9 tmp = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
#endif


#if (RANK == 1)


            double judge = (2 * dHat * dHat * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) / I5;
            double judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 * dis / d_hat_sqrt;
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
            //MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            double judge = -(4 * dHat * dHat * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
            double judge2 = 2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5 * dis / dHat_sqrt;
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            //if (dis*dis<1e-2*dHat)
                //flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif

#ifdef NEWF
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(MATHUTILS::__Transpose9x12(PFPx), flatten_pk1);
#endif

            atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
            atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
            atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
        }
    }
}


__global__
void _calBarrierGradientAndHessian(const double3* _vertexes, const double3* _rest_vertexes, const int4* _collisionPair, double3* _gradient, MATHUTILS::Matrix12x12d* H12x12, MATHUTILS::Matrix9x9d* H9x9, MATHUTILS::Matrix6x6d* H6x6, uint4* D4Index, uint3* D3Index, uint2* D2Index, uint32_t* _cpNum, int* matIndex, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    int4 MMCVIDI = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    double gassThreshold = 1e-4;

    // collisionPair situation
    // [+,+,+,+] edge-edge; [-,+,+,+] point-triangle; 
    // [-,+,+,#] point-edge; [-,+,#,#] point-point
    
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {

#ifdef NEWF
            double dis;
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            double d_hat_sqrt = sqrt(dHat);
            MATHUTILS::Matrix12x9d PFPxT;
            GIPCPDERIV::pFpx_ee2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], d_hat_sqrt, PFPxT);
            double I5 = pow(dis / d_hat_sqrt, 2);
            MATHUTILS::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8] = dis / d_hat_sqrt;

            MATHUTILS::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
            q0.v[8] = 1;

            MATHUTILS::Matrix9x9d H;
#else
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            MATHUTILS::Matrix3x3d Ds;
            MATHUTILS::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(v0, MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double dis = MATHUTILS::__v_vec_dot(v1, normal);
            if (dis < 0) {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis = -dis;
            }

            double3 pos2 = MATHUTILS::__add(_vertexes[MMCVIDI.z], MATHUTILS::__s_vec_multiply(normal, dHat_sqrt - dis));
            double3 pos3 = MATHUTILS::__add(_vertexes[MMCVIDI.w], MATHUTILS::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = v0;
            double3 u1 = MATHUTILS::__minus(pos2, _vertexes[MMCVIDI.x]);
            double3 u2 = MATHUTILS::__minus(pos3, _vertexes[MMCVIDI.x]);

            MATHUTILS::Matrix3x3d Dm, DmInv;
            MATHUTILS::__set_Mat_val_column(Dm, u0, u1, u2);

            MATHUTILS::__Inverse(Dm, DmInv);

            MATHUTILS::Matrix3x3d F;
            MATHUTILS::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = MATHUTILS::__M_v_multiply(F, normal);
            double I5 = MATHUTILS::__squaredNorm(FxN);

            MATHUTILS::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);
            MATHUTILS::Matrix3x3d fnn;
            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);
            MATHUTILS::__M_Mat_multiply(F, nn, fnn);
            MATHUTILS::Vector9 tmp = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
#endif // ifdef NEWF

#if (RANK == 1)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif

#if (RANK == 1)
            double lambda0 = Kappa * (2 * dHat * dHat * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) / I5;
            if (dis * dis < gassThreshold * dHat) {
                double lambda1 = Kappa * (2 * dHat * dHat * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold) - 7 * gassThreshold * gassThreshold - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1)) / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 = -(4 * Kappa * dHat * dHat * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
            if (dis * dis < gassThreshold * dHat) {
                double lambda1 = -(4 * Kappa * dHat * dHat * (4 * gassThreshold + log(gassThreshold) - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold + gassThreshold * log(gassThreshold) * log(gassThreshold) - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 = (2 * Kappa * dHat * dHat * log(I5) * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) - 12 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12)) / I5;
#elif (RANK == 4)
            double lambda0 = -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12)) / I5;
#elif (RANK == 5)
            double lambda0 = (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 30 * I5 * log(I5) - 40 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40)) / I5;
#elif (RANK == 6)
            double lambda0 = -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30)) / I5;
#endif


#ifdef NEWF
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply((PFPxT), flatten_pk1);
            H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);
            MATHUTILS::Matrix12x12d Hessian;
            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(MATHUTILS::__Transpose9x12(PFPx), flatten_pk1);
            MATHUTILS::Vector9 q0 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
            q0 = MATHUTILS::__s_vec9_multiply(q0, 1.0 / sqrt(I5));
            MATHUTILS::Matrix9x9d H;
            MATHUTILS::__init_Mat9x9(H, 0);
            H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);
            MATHUTILS::Matrix12x9d PFPxTransPos = MATHUTILS::__Transpose9x12(PFPx);
            MATHUTILS::Matrix12x12d Hessian = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }

            int Hidx = matIndex[idx];
            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

        } // MMCVIDI.w >= 0
        else {
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1));
            double I1 = c * c;
            if (I1 == 0) return;
            double dis;
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            double I2 = dis / dHat;
            dis = sqrt(dis);

            MATHUTILS::Matrix3x3d F;
            MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);

            MATHUTILS::Matrix3x3d g1, g2;
            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
            MATHUTILS::__M_Mat_multiply(F, nn, g1);
            nn = MATHUTILS::__v_vec_toMat(n2, n2);
            MATHUTILS::__M_Mat_multiply(F, nn, g2);

            MATHUTILS::Vector9 flatten_g1 = MATHUTILS::__Mat3x3_to_vec9_double(g1);
            MATHUTILS::Vector9 flatten_g2 = MATHUTILS::__Mat3x3_to_vec9_double(g2);

            MATHUTILS::Matrix12x9d PFPx;
            GIPCPDERIV::pFpx_pee(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
            double p1 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double p2 = Kappa * 2 * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif    
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__add9(MATHUTILS::__s_vec9_multiply(flatten_g1, p1), MATHUTILS::__s_vec9_multiply(flatten_g2, p2));
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(PFPx, flatten_pk1);

            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }

#if (RANK == 1)
            double lambda10 = Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            double lambda11 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double lambda12 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 2)
            double lambda10 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            double lambda11 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double lambda12 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 4)
            double lambda10 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            double lambda11 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double lambda12 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 6)
            double lambda10 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
            double lambda11 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
            double lambda12 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#endif
            MATHUTILS::Matrix3x3d Tx, Ty, Tz;
            MATHUTILS::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
            MATHUTILS::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
            MATHUTILS::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

            MATHUTILS::Vector9 q11 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tx, g1));
            MATHUTILS::__normalized_vec9_double(q11);
            MATHUTILS::Vector9 q12 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tz, g1));
            MATHUTILS::__normalized_vec9_double(q12);

            MATHUTILS::Matrix9x9d projectedH;
            MATHUTILS::__init_Mat9x9(projectedH, 0);

            MATHUTILS::Matrix9x9d M9_temp = MATHUTILS::__v9_vec9_toMat9x9(q11, q11);
            M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, lambda11);
            projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);

            M9_temp = MATHUTILS::__v9_vec9_toMat9x9(q12, q12);
            M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, lambda12);
            projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
            double lambda20 = -Kappa * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x) * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
            double lambda20 = Kappa * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) - 2 * I2 * I2 + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambda20 = Kappa * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x) * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 12 * I2 * log(I2) - 12 * I2 * I2 + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambda20 = Kappa * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x) * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 18 * I2 * log(I2) - 30 * I2 * I2 + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30)) / (I2 * (eps_x * eps_x));
#endif       

#if (RANK == 1)
            double lambdag1g = Kappa * 4 * c * F.m[2][2] * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x));
#elif (RANK == 2)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif

            double eigenValues[2];
            int eigenNum = 0;
            double2 eigenVecs[2];
            MATHUTILS::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

            for (int i = 0;i < eigenNum;i++) {
                if (eigenValues[i] > 0) {
                    MATHUTILS::Matrix3x3d eigenMatrix;
                    MATHUTILS::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0, 0, eigenVecs[i].y);
                    MATHUTILS::Vector9 eigenMVec = MATHUTILS::__Mat3x3_to_vec9_double(eigenMatrix);
                    M9_temp = MATHUTILS::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
                    M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, eigenValues[i]);
                    projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);
                }
            }

            MATHUTILS::Matrix12x12d Hessian;
            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
            int Hidx = matIndex[idx];
            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    } // MMCVIDI.x >= 0
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y < 0) { // xyz<0
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1));
                double I1 = c * c;
                if (I1 == 0) return;

                double dis;
                MATHUTILS::__distancePointPoint(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis = sqrt(dis);

                MATHUTILS::Matrix3x3d F;
                MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.w]);

                MATHUTILS::Matrix3x3d g1, g2;

                MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
                MATHUTILS::__M_Mat_multiply(F, nn, g1);
                nn = MATHUTILS::__v_vec_toMat(n2, n2);
                MATHUTILS::__M_Mat_multiply(F, nn, g2);

                MATHUTILS::Vector9 flatten_g1 = MATHUTILS::__Mat3x3_to_vec9_double(g1);
                MATHUTILS::Vector9 flatten_g2 = MATHUTILS::__Mat3x3_to_vec9_double(g2);

                MATHUTILS::Matrix12x9d PFPx;
                GIPCPDERIV::pFpx_ppp(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
                double p1 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = Kappa * 2 * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif

                MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__add9(MATHUTILS::__s_vec9_multiply(flatten_g1, p1), MATHUTILS::__s_vec9_multiply(flatten_g2, p2));
                MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }

#if (RANK == 1)
                double lambda10 = Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
                double lambda11 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double lambda12 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
                double lambda11 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double lambda12 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
                double lambda11 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double lambda12 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
                double lambda11 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double lambda12 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#endif

                MATHUTILS::Matrix3x3d Tx, Ty, Tz;
                MATHUTILS::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                MATHUTILS::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                MATHUTILS::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                MATHUTILS::Vector9 q11 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tx, g1));
                MATHUTILS::__normalized_vec9_double(q11);
                MATHUTILS::Vector9 q12 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tz, g1));
                MATHUTILS::__normalized_vec9_double(q12);

                MATHUTILS::Matrix9x9d projectedH;
                MATHUTILS::__init_Mat9x9(projectedH, 0);

                MATHUTILS::Matrix9x9d M9_temp = MATHUTILS::__v9_vec9_toMat9x9(q11, q11);
                M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);

                M9_temp = MATHUTILS::__v9_vec9_toMat9x9(q12, q12);
                M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x) * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 = Kappa * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) - 2 * I2 * I2 + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 = Kappa * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x) * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 12 * I2 * log(I2) - 12 * I2 * I2 + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 = Kappa * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x) * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 18 * I2 * log(I2) - 30 * I2 * I2 + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30)) / (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2] * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif

                double eigenValues[2];
                int eigenNum = 0;
                double2 eigenVecs[2];
                MATHUTILS::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

                for (int i = 0;i < eigenNum;i++) {
                    if (eigenValues[i] > 0) {
                        MATHUTILS::Matrix3x3d eigenMatrix;
                        MATHUTILS::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0, 0, eigenVecs[i].y);

                        MATHUTILS::Vector9 eigenMVec = MATHUTILS::__Mat3x3_to_vec9_double(eigenMatrix);

                        M9_temp = MATHUTILS::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
                        M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, eigenValues[i]);
                        projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);
                    }
                }

                MATHUTILS::Matrix12x12d Hessian;
                MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];
                H12x12[Hidx] = Hessian;
                D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            }
            else { // 
#ifdef NEWF
                double dis;
                MATHUTILS::__distancePointPoint(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis = sqrt(dis);
                double d_hat_sqrt = sqrt(dHat);
                MATHUTILS::Vector6 PFPxT;
                GIPCPDERIV::pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                double I5 = pow(dis / d_hat_sqrt, 2);
                double fnn = dis / d_hat_sqrt;

#if (RANK == 1)
                double flatten_pk1 = fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
#elif (RANK == 2)
                double flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
#elif (RANK == 3)
                double flatten_pk1 = fnn * -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5;
#elif (RANK == 4)
                double flatten_pk1 = fnn * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5;
#elif (RANK == 5)
                double flatten_pk1 = fnn * -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5;
#elif (RANK == 6)
                double flatten_pk1 = fnn * (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5;
#endif

                MATHUTILS::Vector6 gradient_vec = MATHUTILS::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 Ds = v0;
                double dis = MATHUTILS::__norm(v0);
                //if (dis > dHat_sqrt) return;
                double3 vec_normal = MATHUTILS::__normalized(make_double3(-v0.x, -v0.y, -v0.z));
                double3 target = make_double3(0, 1, 0);
                double3 vec = MATHUTILS::__v_vec_cross(vec_normal, target);
                double cos = MATHUTILS::__v_vec_dot(vec_normal, target);
                MATHUTILS::Matrix3x3d rotation;
                MATHUTILS::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                MATHUTILS::Vector6 PDmPx;
                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else {
                    MATHUTILS::Matrix3x3d cross_vec;
                    MATHUTILS::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = MATHUTILS::__Mat_add(rotation, MATHUTILS::__Mat_add(cross_vec, MATHUTILS::__S_Mat_multiply(MATHUTILS::__M_Mat_multiply(cross_vec, cross_vec), 1.0 / (1 + cos))));
                }

                double3 pos0 = MATHUTILS::__add(_vertexes[v0I], MATHUTILS::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                double3 rotate_uv0 = MATHUTILS::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 = MATHUTILS::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                double uv0 = rotate_uv0.y;
                double uv1 = rotate_uv1.y;

                double u0 = uv1 - uv0;
                double Dm = u0;//PFPx
                double DmInv = 1 / u0;

                double3 F = MATHUTILS::__s_vec_multiply(Ds, DmInv);
                double I5 = MATHUTILS::__squaredNorm(F);

                double3 tmp = F;

#if (RANK == 1)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

#elif (RANK == 3)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
                double3 flatten_pk1 = MATHUTILS::__s_vec_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif
                MATHUTILS::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);

                MATHUTILS::Vector6 gradient_vec = MATHUTILS::__M6x3_v3_multiply(MATHUTILS::__Transpose3x6(PFPx), flatten_pk1);
#endif


                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                }

#if (RANK == 1)
                double lambda0 = Kappa * (2 * dHat * dHat * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) / I5;
            if (dis * dis < gassThreshold * dHat) {
                double lambda1 = Kappa * (2 * dHat * dHat * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold) - 7 * gassThreshold * gassThreshold - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1)) / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
                double lambda0 = -(4 * Kappa * dHat * dHat * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
                if (dis * dis < gassThreshold * dHat) {
                    double lambda1 = -(4 * Kappa * dHat * dHat * (4 * gassThreshold + log(gassThreshold) - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold + gassThreshold * log(gassThreshold) * log(gassThreshold) - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 = (2 * Kappa * dHat * dHat * log(I5) * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) - 12 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12)) / I5;
#elif (RANK == 4)
                double lambda0 = -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12)) / I5;
#elif (RANK == 5)
                double lambda0 = (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 30 * I5 * log(I5) - 40 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40)) / I5;
#elif (RANK == 6)
                double lambda0 = -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30)) / I5;
#endif


#ifdef NEWF
                double H = lambda0;
                MATHUTILS::Matrix6x6d Hessian = MATHUTILS::__s_M6x6_Multiply(MATHUTILS::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
#else
                double3 q0 = MATHUTILS::__s_vec_multiply(F, 1 / sqrt(I5));

                MATHUTILS::Matrix3x3d H = MATHUTILS::__S_Mat_multiply(MATHUTILS::__v_vec_toMat(q0, q0), lambda0);//lambda0 * q0 * q0.transpose();

                MATHUTILS::Matrix6x3d PFPxTransPos = MATHUTILS::__Transpose3x6(PFPx);
                MATHUTILS::Matrix6x6d Hessian = MATHUTILS::__M6x3_M3x6_Multiply(MATHUTILS::__M6x3_M3x3_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];
                H6x6[Hidx] = Hessian;
                D2Index[Hidx] = make_uint2(v0I, MMCVIDI.y);
            }
        }
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
                double I1 = c * c;
                if (I1 == 0) return;
                double dis;
                MATHUTILS::__distancePointEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I2 = dis / dHat;
                dis = sqrt(dis);

                MATHUTILS::Matrix3x3d F;
                MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z]);

                MATHUTILS::Matrix3x3d g1, g2;

                MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
                MATHUTILS::__M_Mat_multiply(F, nn, g1);
                nn = MATHUTILS::__v_vec_toMat(n2, n2);
                MATHUTILS::__M_Mat_multiply(F, nn, g2);

                MATHUTILS::Vector9 flatten_g1 = MATHUTILS::__Mat3x3_to_vec9_double(g1);
                MATHUTILS::Vector9 flatten_g2 = MATHUTILS::__Mat3x3_to_vec9_double(g2);

                MATHUTILS::Matrix12x9d PFPx;
                GIPCPDERIV::pFpx_ppe(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
                double p1 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = Kappa * 2 * (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double p1 = -Kappa * 2 * (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif    
                MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__add9(MATHUTILS::__s_vec9_multiply(flatten_g1, p1), MATHUTILS::__s_vec9_multiply(flatten_g2, p2));
                MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(PFPx, flatten_pk1);

                {
                    atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                    atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                    atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                    atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
                }

#if (RANK == 1)
                double lambda10 = Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
                double lambda11 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double lambda12 = Kappa * 2 * (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 2)
                double lambda10 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
                double lambda11 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double lambda12 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 4)
                double lambda10 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
                double lambda11 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double lambda12 = -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#elif (RANK == 6)
                double lambda10 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
                double lambda11 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
                double lambda12 = -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
#endif
                MATHUTILS::Matrix3x3d Tx, Ty, Tz;
                MATHUTILS::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                MATHUTILS::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                MATHUTILS::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                MATHUTILS::Vector9 q11 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tx, g1));
                MATHUTILS::__normalized_vec9_double(q11);
                MATHUTILS::Vector9 q12 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tz, g1));
                MATHUTILS::__normalized_vec9_double(q12);

                MATHUTILS::Matrix9x9d projectedH;
                MATHUTILS::__init_Mat9x9(projectedH, 0);

                MATHUTILS::Matrix9x9d M9_temp = MATHUTILS::__v9_vec9_toMat9x9(q11, q11);
                M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);

                M9_temp = MATHUTILS::__v9_vec9_toMat9x9(q12, q12);
                M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                double lambda20 = -Kappa * (2 * I1 * dHat * dHat * (I1 - 2 * eps_x) * (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1)) / (I2 * eps_x * eps_x);
#elif (RANK == 2)
                double lambda20 = Kappa * (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) * (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) - 2 * I2 * I2 + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambda20 = Kappa * (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x) * (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 12 * I2 * log(I2) - 12 * I2 * I2 + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambda20 = Kappa * (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x) * (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 18 * I2 * log(I2) - 30 * I2 * I2 + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30)) / (I2 * (eps_x * eps_x));
#endif       

#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2] * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif
                double eigenValues[2];
                int eigenNum = 0;
                double2 eigenVecs[2];
                MATHUTILS::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

                for (int i = 0;i < eigenNum;i++) {
                    if (eigenValues[i] > 0) {
                        MATHUTILS::Matrix3x3d eigenMatrix;
                        MATHUTILS::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0, 0, eigenVecs[i].y);

                        MATHUTILS::Vector9 eigenMVec = MATHUTILS::__Mat3x3_to_vec9_double(eigenMatrix);

                        M9_temp = MATHUTILS::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
                        M9_temp = MATHUTILS::__S_Mat9x9_multiply(M9_temp, eigenValues[i]);
                        projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);
                    }
                }

                MATHUTILS::Matrix12x12d Hessian;
                MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];

                H12x12[Hidx] = Hessian;
                D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            }
            else {
#ifdef NEWF
                double dis;
                MATHUTILS::__distancePointEdge(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                dis = sqrt(dis);
                double d_hat_sqrt = sqrt(dHat);
                MATHUTILS::Matrix9x4d PFPxT;
                GIPCPDERIV::pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt,
                         PFPxT);
                double I5 = pow(dis / d_hat_sqrt, 2);
                MATHUTILS::Vector4 fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;                
                MATHUTILS::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3] = 1;
                MATHUTILS::Matrix4x4d H;
                
#if (RANK == 1)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
                MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif

                MATHUTILS::Vector9 gradient_vec = MATHUTILS::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);


                MATHUTILS::Matrix3x2d Ds;
                MATHUTILS::__set_Mat3x2_val_column(Ds, v0, v1);

                double3 triangle_normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(v0, v1));
                double3 target = make_double3(0, 1, 0);

                double3 vec = MATHUTILS::__v_vec_cross(triangle_normal, target);
                double cos = MATHUTILS::__v_vec_dot(triangle_normal, target);

                double3 edge_normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]), triangle_normal));
                double dis = MATHUTILS::__v_vec_dot(MATHUTILS::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                MATHUTILS::Matrix3x3d rotation;
                MATHUTILS::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                MATHUTILS::Matrix9x4d PDmPx;

                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;rotation.m[1][1] = -1;
                }
                else {
                    MATHUTILS::Matrix3x3d cross_vec;
                    MATHUTILS::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x, 0);

                    rotation = MATHUTILS::__Mat_add(rotation, MATHUTILS::__Mat_add(cross_vec, MATHUTILS::__S_Mat_multiply(MATHUTILS::__M_Mat_multiply(cross_vec, cross_vec), 1.0 / (1 + cos))));
                }

                double3 pos0 = MATHUTILS::__add(_vertexes[v0I], MATHUTILS::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                double3 rotate_uv0 = MATHUTILS::__M_v_multiply(rotation, pos0);
                double3 rotate_uv1 = MATHUTILS::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                double3 rotate_uv2 = MATHUTILS::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                double3 rotate_normal = MATHUTILS::__M_v_multiply(rotation, edge_normal);

                double2 uv0 = make_double2(rotate_uv0.x, rotate_uv0.z);
                double2 uv1 = make_double2(rotate_uv1.x, rotate_uv1.z);
                double2 uv2 = make_double2(rotate_uv2.x, rotate_uv2.z);
                double2 normal = make_double2(rotate_normal.x, rotate_normal.z);

                double2 u0 = MATHUTILS::__minus_v2(uv1, uv0);
                double2 u1 = MATHUTILS::__minus_v2(uv2, uv0);

                MATHUTILS::Matrix2x2d Dm;

                MATHUTILS::__set_Mat2x2_val_column(Dm, u0, u1);

                MATHUTILS::Matrix2x2d DmInv;
                MATHUTILS::__Inverse2x2(Dm, DmInv);

                MATHUTILS::Matrix3x2d F = MATHUTILS::__M3x2_M2x2_Multiply(Ds, DmInv);

                double3 FxN = MATHUTILS::__M3x2_v2_multiply(F, normal);
                double I5 = MATHUTILS::__squaredNorm(FxN);

                MATHUTILS::Matrix3x2d fnn;

                MATHUTILS::Matrix2x2d nn = MATHUTILS::__v2_vec2_toMat2x2(normal, normal);

                fnn = MATHUTILS::__M3x2_M2x2_Multiply(F, nn);

                MATHUTILS::Vector6 tmp = MATHUTILS::__Mat3x2_to_vec6_double(fnn);

#if (RANK == 1)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
                MATHUTILS::Vector6 flatten_pk1 = MATHUTILS::__s_vec6_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif

                MATHUTILS::Matrix6x9d PFPx = __computePFPX3D_6x9_double(DmInv);

                MATHUTILS::Vector9 gradient_vec = MATHUTILS::__M9x6_v6_multiply(MATHUTILS::__Transpose6x9(PFPx), flatten_pk1);
#endif

                {
                    atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                    atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                    atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                    atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                    atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                    atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                    atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                    atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                    atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                }

#if (RANK == 1)
                double lambda0 = Kappa * (2 * dHat * dHat * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) / I5;
            if (dis * dis < gassThreshold * dHat) {
                double lambda1 = Kappa * (2 * dHat * dHat * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold) - 7 * gassThreshold * gassThreshold - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1)) / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
                double lambda0 = -(4 * Kappa * dHat * dHat * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
                if (dis * dis < gassThreshold * dHat) {
                    double lambda1 = -(4 * Kappa * dHat * dHat * (4 * gassThreshold + log(gassThreshold) - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold + gassThreshold * log(gassThreshold) * log(gassThreshold) - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) / gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                double lambda0 = (2 * Kappa * dHat * dHat * log(I5) * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) - 12 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12)) / I5;
#elif (RANK == 4)
                double lambda0 = -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12)) / I5;
#elif (RANK == 5)
                double lambda0 = (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 30 * I5 * log(I5) - 40 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40)) / I5;
#elif (RANK == 6)
                double lambda0 = -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30)) / I5;
#endif


#ifdef NEWF
                H = MATHUTILS::__S_Mat4x4_multiply(MATHUTILS::__v4_vec4_toMat4x4(q0, q0), lambda0);

                MATHUTILS::Matrix9x9d Hessian;
                MATHUTILS::__M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
#else

                MATHUTILS::Vector6 q0 = MATHUTILS::__Mat3x2_to_vec6_double(fnn);
                q0 = MATHUTILS::__s_vec6_multiply(q0, 1.0 / sqrt(I5));
                MATHUTILS::Matrix6x6d H;
                MATHUTILS::__init_Mat6x6(H, 0);
                H = MATHUTILS::__S_Mat6x6_multiply(MATHUTILS::__v6_vec6_toMat6x6(q0, q0), lambda0);
                MATHUTILS::Matrix9x6d PFPxTransPos = MATHUTILS::__Transpose6x9(PFPx);
                MATHUTILS::Matrix9x9d Hessian = MATHUTILS::__M9x6_M6x9_Multiply(MATHUTILS::__M9x6_M6x6_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];
                H9x9[Hidx] = Hessian;
                D3Index[Hidx] = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            }

        }
        else {
#ifdef NEWF
            double dis;
            MATHUTILS::__distancePointTriangle(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            double d_hat_sqrt = sqrt(dHat);
            MATHUTILS::Matrix12x9d PFPxT;
            GIPCPDERIV::pFpx_pt2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], d_hat_sqrt, PFPxT);
            double I5 = pow(dis / d_hat_sqrt, 2);
            MATHUTILS::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8] = dis / d_hat_sqrt;

            MATHUTILS::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
            q0.v[8] = 1;

            MATHUTILS::Matrix9x9d H;

#else
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            MATHUTILS::Matrix3x3d Ds;
            MATHUTILS::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]), MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double dis = MATHUTILS::__v_vec_dot(v0, normal);
            MATHUTILS::Matrix12x9d PDmPx;

            if (dis > 0) {
                normal = make_double3(-normal.x, -normal.y, -normal.z);
            }
            else {
                dis = -dis;
            }

            double3 pos0 = MATHUTILS::__add(_vertexes[v0I], MATHUTILS::__s_vec_multiply(normal, dHat_sqrt - dis));

            double3 u0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], pos0);
            double3 u1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], pos0);
            double3 u2 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], pos0);

            MATHUTILS::Matrix3x3d Dm, DmInv;
            MATHUTILS::__set_Mat_val_column(Dm, u0, u1, u2);

            MATHUTILS::__Inverse(Dm, DmInv);

            MATHUTILS::Matrix3x3d F;
            MATHUTILS::__M_Mat_multiply(Ds, DmInv, F);

            double3 FxN = MATHUTILS::__M_v_multiply(F, normal);
            double I5 = MATHUTILS::__squaredNorm(FxN);

            MATHUTILS::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);
            MATHUTILS::Matrix3x3d fnn;
            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);
            MATHUTILS::__M_Mat_multiply(F, nn, fnn);
            MATHUTILS::Vector9 tmp = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
#endif

#if (RANK == 1)
            double lambda0 = Kappa * (2 * dHat * dHat * (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) / I5;
            if (dis * dis < gassThreshold * dHat) {
                double lambda1 = Kappa * (2 * dHat * dHat * (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold) - 7 * gassThreshold * gassThreshold - 6 * gassThreshold * gassThreshold * log(gassThreshold) + 1)) / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            double lambda0 = -(4 * Kappa * dHat * dHat * (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
            if (dis * dis < gassThreshold * dHat) {
                double lambda1 = -(4 * Kappa * dHat * dHat * (4 * gassThreshold + log(gassThreshold) - 3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) + 6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold + gassThreshold * log(gassThreshold) * log(gassThreshold) - 7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) / gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            double lambda0 = (2 * Kappa * dHat * dHat * log(I5) * (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) - 12 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12)) / I5;
#elif (RANK == 4)
            double lambda0 = -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 12 * I5 * log(I5) - 12 * I5 * I5 + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12)) / I5;
#elif (RANK == 5)
            double lambda0 = (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 30 * I5 * log(I5) - 40 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40)) / I5;
#elif (RANK == 6)
            double lambda0 = -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) - 30 * I5 * I5 + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30)) / I5;
#endif

#if (RANK == 1)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + 2 * I5 * log(I5) - 3)) / I5);
#elif (RANK == 4)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) / I5);
#elif (RANK == 5)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, -2 * (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) / I5);
#elif (RANK == 6)
            MATHUTILS::Vector9 flatten_pk1 = MATHUTILS::__s_vec9_multiply(tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) / I5);
#endif

#ifdef NEWF
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(MATHUTILS::__Transpose9x12(PFPx), flatten_pk1);
#endif

            {
                atomicAdd(&(_gradient[v0I].x), gradient_vec.v[0]);
                atomicAdd(&(_gradient[v0I].y), gradient_vec.v[1]);
                atomicAdd(&(_gradient[v0I].z), gradient_vec.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
                atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
                atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
                atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
            }

#ifdef NEWF

            H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);
            MATHUTILS::Matrix12x12d Hessian;
            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            MATHUTILS::Vector9 q0 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
            q0 = MATHUTILS::__s_vec9_multiply(q0, 1.0 / sqrt(I5));
            MATHUTILS::Matrix9x9d H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);
            MATHUTILS::Matrix12x9d PFPxTransPos = MATHUTILS::__Transpose9x12(PFPx);
            MATHUTILS::Matrix12x12d Hessian = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];
            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

        }
    }
}



__global__
void _calFrictionGradient_gd(const double3* _vertexes,
    const double3* _o_vertexes,
    const double3* _normal,
    const uint32_t* _last_collisionPair_gd,
    double3* _gradient,
    int number,
    double dt,
    double eps2,
    double* lastH,
    double coef
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double eps = sqrt(eps2);
    double3 normal = *_normal;
    uint32_t gidx = _last_collisionPair_gd[idx];
    double3 Vdiff = MATHUTILS::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 VProj = MATHUTILS::__minus(Vdiff, MATHUTILS::__s_vec_multiply(normal, MATHUTILS::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = MATHUTILS::__squaredNorm(VProj);
    if (VProjMag2 > eps2) {
        double3 gdf = MATHUTILS::__s_vec_multiply(VProj, coef * lastH[idx] / sqrt(VProjMag2));
        /*atomicAdd(&(_gradient[gidx].x), gdf.x);
        atomicAdd(&(_gradient[gidx].y), gdf.y);
        atomicAdd(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = MATHUTILS::__add(_gradient[gidx], gdf);
    }
    else {
        double3 gdf = MATHUTILS::__s_vec_multiply(VProj, coef * lastH[idx] / eps);
        /*atomicAdd(&(_gradient[gidx].x), gdf.x);
        atomicAdd(&(_gradient[gidx].y), gdf.y);
        atomicAdd(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = MATHUTILS::__add(_gradient[gidx], gdf);
    }
}

__global__
void _calFrictionGradient(const double3* _vertexes,
    const double3* _o_vertexes,
    const int4* _last_collisionPair,
    double3* _gradient,
    int number,
    double dt,
    double2* distCoord,
    MATHUTILS::Matrix3x2d* tanBasis,
    double eps2,
    double* lastH,
    double coef
) {
    double eps = std::sqrt(eps2);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _last_collisionPair[idx];
    double3 relDX3D;
    if (MMCVIDI.x >= 0) {
        IPCFRICTION::computeRelDX_EE(MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            MATHUTILS::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
            distCoord[idx].y, relDX3D);

        MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis[idx]);
        double2 relDX = MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D);
        double relDXSqNorm = MATHUTILS::__squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = MATHUTILS::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        }
        else {
            double f1_div_relDXNorm;
            IPCFRICTION::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = MATHUTILS::__s_vec_multiply(relDX, f1_div_relDXNorm);
        }
        MATHUTILS::Vector12 TTTDX;
        IPCFRICTION::liftRelDXTanToMesh_EE(relDX, tanBasis[idx],
            distCoord[idx].x, distCoord[idx].y, TTTDX);
        TTTDX = MATHUTILS::__s_vec12_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            MMCVIDI.x = v0I;

            IPCFRICTION::computeRelDX_PP(
                MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                relDX3D);

            MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis[idx]);
            double2 relDX = MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D);
            double relDXSqNorm = MATHUTILS::__squaredNorm(relDX);
            if (relDXSqNorm > eps2) {
                relDX = MATHUTILS::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else {
                double f1_div_relDXNorm;
                IPCFRICTION::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = MATHUTILS::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }

            MATHUTILS::Vector6 TTTDX;
            IPCFRICTION::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
            TTTDX = MATHUTILS::__s_vec6_multiply(TTTDX, lastH[idx] * coef);
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            }
        }
        else if (MMCVIDI.w < 0) {
            MMCVIDI.x = v0I;
            IPCFRICTION::computeRelDX_PE(MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x, relDX3D);

            MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis[idx]);
            double2 relDX = MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D);
            double relDXSqNorm = MATHUTILS::__squaredNorm(relDX);
            if (relDXSqNorm > eps2) {
                relDX = MATHUTILS::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else {
                double f1_div_relDXNorm;
                IPCFRICTION::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = MATHUTILS::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            MATHUTILS::Vector9 TTTDX;
            IPCFRICTION::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
            TTTDX = MATHUTILS::__s_vec9_multiply(TTTDX, lastH[idx] * coef);
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
                atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
                atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
                atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            }
        }
        else {
            MMCVIDI.x = v0I;
            IPCFRICTION::computeRelDX_PT(MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
                distCoord[idx].y, relDX3D);

            MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis[idx]);
            double2 relDX = MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D);

            double relDXSqNorm = MATHUTILS::__squaredNorm(relDX);
            if (relDXSqNorm > eps2) {
                relDX = MATHUTILS::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            }
            else {
                double f1_div_relDXNorm;
                IPCFRICTION::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = MATHUTILS::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            MATHUTILS::Vector12 TTTDX;
            IPCFRICTION::liftRelDXTanToMesh_PT(relDX, tanBasis[idx],
                distCoord[idx].x, distCoord[idx].y, TTTDX);
            TTTDX = MATHUTILS::__s_vec12_multiply(TTTDX, lastH[idx] * coef);

            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }
}



__global__
void _GroundCollisionDetectIPC(const double3* vertexes, const uint32_t* surfVertIds, const double* g_offset, const double3* g_normal, uint32_t* _environment_collisionPair, uint32_t* _gpNum, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double dist = MATHUTILS::__v_vec_dot(*g_normal, vertexes[surfVertIds[idx]]) - *g_offset;
    if (dist * dist > dHat) return;

    _environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}



__global__
void _computeSelfCloseVal(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double dTol, uint32_t* _closeConstraintID, double* _closeConstraintVal, uint32_t* _close_gpNum, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = MATHUTILS::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx] = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}


__global__
void _checkGroundIntersection(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, int* _isIntersect, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = MATHUTILS::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    //printf("%f  %f\n", *g_offset, dist);
    if (dist < 0)
        *_isIntersect = -1;
}

__global__
void _reduct_double3Sqn_to_double(const double3* A, double* D, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = MATHUTILS::__squaredNorm(A[idx]);


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        //double tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }
    if (threadIdx.x == 0) {
        D[blockIdx.x] = temp;
    }

}

__global__
void _reduct_double3Dot_to_double(const double3* A, const double3* B, double* D, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = MATHUTILS::__v_vec_dot(A[idx], B[idx]);


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        //double tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }
    if (threadIdx.x == 0) {
        D[blockIdx.x] = temp;
    }

}

__global__
void _calFrictionLastH_gd(const double3* _vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _collisionPair_environment, double* lambda_lastH_gd, uint32_t* _collisionPair_last_gd, double dHat,
    double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    double3 normal = *g_normal;
    int gidx = _collisionPair_environment[idx];
    double dist = MATHUTILS::__v_vec_dot(normal, _vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    double t = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    lambda_lastH_gd[idx] = -Kappa * 2.0 * sqrt(dist2) * g_b;
    _collisionPair_last_gd[idx] = gidx;
}

__global__
void _calFrictionLastH_DistAndTan(const double3* _vertexes, const int4* _collisionPair, double* lambda_lastH, double2* distCoord, MATHUTILS::Matrix3x2d* tanBasis, int4* _collisionPair_last, double dHat,
    double Kappa, uint32_t* _cpNum_last, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _collisionPair[idx];
    double dis;
    int last_index = -1;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            IPCFRICTION::computeClosestPoint_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                _vertexes[MMCVIDI.w], distCoord[last_index]);
            IPCFRICTION::computeTangentBasis_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                _vertexes[MMCVIDI.w], tanBasis[last_index]);
        }
    }
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y >= 0) {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 2, 1);
                MATHUTILS::__distancePointPoint(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                distCoord[last_index].x = 0;
                distCoord[last_index].y = 0;
                IPCFRICTION::computeTangentBasis_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], tanBasis[last_index]);
            }

        }
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y >= 0) {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 3, 1);
                MATHUTILS::__distancePointEdge(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                IPCFRICTION::computeClosestPoint_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                    distCoord[last_index].x);
                distCoord[last_index].y = 0;
                IPCFRICTION::computeTangentBasis_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                    tanBasis[last_index]);
            }
        }
        else {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            MATHUTILS::__distancePointTriangle(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            IPCFRICTION::computeClosestPoint_PT(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                _vertexes[MMCVIDI.w], distCoord[last_index]);
            IPCFRICTION::computeTangentBasis_PT(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                _vertexes[MMCVIDI.w], tanBasis[last_index]);
        }
    }
    if (last_index >= 0) {
//        double t = dis - dHat;
//        lambda_lastH[last_index] = -Kappa * 2.0 * std::sqrt(dis) * (t * std::log(dis / dHat) * -2.0 - (t * t) / dis);
#if (RANK == 1)
        double t = dis - dHat;
        lambda_lastH[last_index] = -Kappa * 2.0 * sqrt(dis) * (t * log(dis / dHat) * -2.0 - (t * t) / dis);
#elif (RANK == 2)
        lambda_lastH[last_index] = -Kappa * 2.0 * sqrt(dis) * (log(dis / dHat) * log(dis / dHat) * (2 * dis - 2 * dHat) + (2 * log(dis / dHat) * (dis - dHat) * (dis - dHat)) / dis);
#endif
        _collisionPair_last[last_index] = _collisionPair[idx];
    }
}


void buildFrictionSets(std::unique_ptr<GeometryManager>& instance) {
    CUDAMemcpyHToDSafe(instance->cudaCPNum, Eigen::VectorXi::Zero(5));
    // CUDA_SAFE_CALL(cudaMemset(instance->cudaCPNum, 0, 5 * sizeof(uint32_t)));
    int numbers = instance->cpNum[0];
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calFrictionLastH_DistAndTan <<<blockNum, threadNum>>> (instance->cudaVertPos, instance->cudaCollisionPairs, instance->cudaLambdaLastHScalar, instance->cudaDistCoord, instance->cudaTanBasis, instance->cudaCollisonPairsLastH, instance->dHat, instance->Kappa, instance->cudaCPNum, instance->cpNum[0]);
    CUDA_SAFE_CALL(cudaMemcpy(instance->cpNumLast, instance->cudaCPNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    numbers = instance->gpNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    _calFrictionLastH_gd << <blockNum, threadNum >> > (instance->cudaVertPos, instance->cudaGroundOffset, instance->cudaGroundNormal, instance->cudaEnvCollisionPairs, instance->cudaLambdaLastHScalarGd, instance->cudaCollisonPairsLastHGd, instance->dHat, instance->Kappa, instance->gpNum);
    instance->gpNumLast = instance->gpNum;
}

void GroundCollisionDetect(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->numSurfVerts;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _GroundCollisionDetectIPC << <blockNum, threadNum >> > (instance->cudaVertPos, instance->cudaSurfVert, instance->cudaGroundOffset, instance->cudaGroundNormal, instance->cudaEnvCollisionPairs, instance->cudaGPNum, instance->dHat, numbers);

}


double reduction2Kappa(int type, const double3* A, const double3* B, double* _queue, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double)));*/
    if (type == 0) {
        //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
        _reduct_double3Dot_to_double << <blockNum, threadNum, sharedMsize >> > (A, B, _queue, numbers);
    }
    else if (type == 1) {
        _reduct_double3Sqn_to_double << <blockNum, threadNum, sharedMsize >> > (A, _queue, numbers);
    }
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::__add_reduction << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double dotValue;
    cudaMemcpy(&dotValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_queue));
    return dotValue;
}



void buildCP(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_E>& bvh_e, std::unique_ptr<LBVH_F>& bvh_f) {

    CUDA_SAFE_CALL(cudaMemset(instance->cudaCPNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(instance->cudaGPNum, 0, sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //bvh_f->Construct();
    bvh_f->SelfCollisionDetect(instance->dHat);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //bvh_e->Construct();
    bvh_e->SelfCollisionDetect(instance->dHat);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    GroundCollisionDetect(instance);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(&instance->cpNum, instance->cudaCPNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&instance->gpNum, instance->cudaGPNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /*CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));*/
}

void buildFullCP(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_E>& bvh_e, std::unique_ptr<LBVH_F>& bvh_f, const double& alpha) {

    CUDA_SAFE_CALL(cudaMemset(instance->cudaCPNum, 0, sizeof(uint32_t)));

    bvh_f->SelfCollisionFullDetect(instance->dHat, instance->cudaMoveDir, alpha);
    bvh_e->SelfCollisionFullDetect(instance->dHat, instance->cudaMoveDir, alpha);

    CUDA_SAFE_CALL(cudaMemcpy(&instance->ccdCpNum, instance->cudaCPNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
}


void buildBVH(std::unique_ptr<LBVH_E>& bvh_e, std::unique_ptr<LBVH_F>& bvh_f) {
    bvh_f->Construct();
    bvh_e->Construct();
}

void buildBVH_FULLCCD(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_E>& bvh_e, std::unique_ptr<LBVH_F>& bvh_f, const double& alpha) {
    bvh_f->ConstructFullCCD(instance->cudaMoveDir, alpha);
    bvh_e->ConstructFullCCD(instance->cudaMoveDir, alpha);
}



void calFrictionHessian(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->cpNumLast[0];
    //if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum; //

    _calFrictionHessian << < blockNum, threadNum >> > (
        instance->cudaVertPos,
        instance->cudaOriginVertPos,
        instance->cudaCollisonPairsLastH,
        instance->cudaH12x12,
        instance->cudaH9x9,
        instance->cudaH6x6,
        instance->cudaD4Index,
        instance->cudaD3Index,
        instance->cudaD2Index,
        instance->cudaCPNum,
        numbers,
        instance->IPC_dt, instance->cudaDistCoord,
        instance->cudaTanBasis,
        instance->fDhat * instance->IPC_dt * instance->IPC_dt,
        instance->cudaLambdaLastHScalar,
        instance->frictionRate,
        instance->cpNum[4],
        instance->cpNum[3],
        instance->cpNum[2]);


    numbers = instance->gpNumLast;

    std::cout << "h_gpNum_last!!!!!!!!!!!!!!" << instance->gpNumLast << std::endl;
    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaGPNum, &instance->gpNumLast, sizeof(uint32_t), cudaMemcpyHostToDevice));
    std::cout << "h_gpNum_last!!!!!!!!!2!!!!!" << instance->gpNumLast << std::endl;

    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionHessian_gd << < blockNum, threadNum >> > (
        instance->cudaVertPos,
        instance->cudaOriginVertPos,
        instance->cudaGroundNormal,
        instance->cudaCollisonPairsLastHGd,
        instance->cudaH3x3,
        instance->cudaD1Index,
        numbers,
        instance->IPC_dt,
        instance->fDhat * instance->IPC_dt * instance->IPC_dt,
        instance->cudaLambdaLastHScalarGd,
        instance->frictionRate);
}


void calBarrierGradient(std::unique_ptr<GeometryManager>& instance, double3* _gradient, double mKappa) {
    int numbers = instance->cpNum[0];
    if (numbers < 1)return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradient << <blockNum, threadNum >> > (instance->cudaVertPos, instance->cudaRestVertPos, instance->cudaCollisionPairs, _gradient, instance->dHat, mKappa, numbers);

}


void calBarrierGradientAndHessian(const double3* _vertexes, const double3* _rest_vertexes, const int4* _collisionPair, double3* _gradient, MATHUTILS::Matrix12x12d* H12x12, MATHUTILS::Matrix9x9d* H9x9, MATHUTILS::Matrix6x6d* H6x6, uint4* D4Index, uint3* D3Index, uint2* D2Index, uint32_t* _cpNum, int* matIndex, double dHat, double Kappa, uint32_t* hCpNum) {
    int numbers = hCpNum[0];
    if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradientAndHessian <<<blockNum, threadNum>>> (_vertexes, _rest_vertexes, _collisionPair, _gradient, H12x12, H9x9, H6x6, D4Index, D3Index, D2Index, _cpNum, matIndex, dHat, Kappa, numbers);
}



void calFrictionGradient(double3* _gradient, std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->cpNumLast[0];
    //if (numbers < 1)return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient << < blockNum, threadNum >> > (
        instance->cudaVertPos,
        instance->cudaOriginVertPos,
        instance->cudaCollisonPairsLastH,
        _gradient,
        numbers,
        instance->IPC_dt,
        instance->cudaDistCoord,
        instance->cudaTanBasis,
        instance->fDhat * instance->IPC_dt * instance->IPC_dt,
        instance->cudaLambdaLastHScalar,
        instance->frictionRate
        );

    numbers = instance->gpNumLast;
    //if (numbers < 1)return;
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient_gd << < blockNum, threadNum >> > (
        instance->cudaVertPos,
        instance->cudaOriginVertPos,
        instance->cudaGroundNormal,
        instance->cudaCollisonPairsLastHGd,
        _gradient,
        numbers,
        instance->IPC_dt,
        instance->fDhat * instance->IPC_dt * instance->IPC_dt,
        instance->cudaLambdaLastHScalarGd,
        instance->frictionRate
        );
}




void compute_H_b(double d, double dHat, double& H) {
    double t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

void suggestKappa(std::unique_ptr<GeometryManager>& instance, double& kappa) {
    double H_b;
    compute_H_b(1.0e-16 * instance->bboxDiagSize2, instance->dHat, H_b);
    if (instance->meanMass == 0.0) {
        kappa = instance->minKappaCoef / (4.0e-16 * instance->bboxDiagSize2 * H_b);
    }
    else {
        kappa = instance->minKappaCoef * instance->meanMass / (4.0e-16 * instance->bboxDiagSize2 * H_b);
    }
}

void upperBoundKappa(std::unique_ptr<GeometryManager>& instance, double& kappa) {
    double H_b;
    compute_H_b(1.0e-16 * instance->bboxDiagSize2, instance->dHat, H_b);
    double kappaMax = 100 * instance->minKappaCoef * instance->meanMass / (4.0e-16 * instance->bboxDiagSize2 * H_b);
    if (instance->meanMass == 0.0) {
        kappaMax = 100 * instance->minKappaCoef / (4.0e-16 * instance->bboxDiagSize2 * H_b);
    }
    if (kappa > kappaMax) {
        kappa = kappaMax;
    }
}


void initKappa(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<PCGData>& pcg_data) {

    if (instance->cpNum[0] > 0) {
        double3* _GE = instance->cudaFb;
        double3* _gc = instance->cudaTempDouble3Mem;
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, instance->numVertices * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, instance->numVertices * sizeof(double3)));
        FEMENERGY::calKineticGradient(instance->cudaVertPos, instance->cudaXTilta, _GE, instance->cudaVertMass, instance->numVertices);
        FEMENERGY::calculate_fem_gradient(instance->cudaTetDmInverses, instance->cudaVertPos, instance->cudaTetElement, instance->cudaTetVolume, _GE, instance->numTetElements, instance->lengthRate, instance->volumeRate, instance->IPC_dt);
        // FEMENERGY::calculate_triangle_fem_gradient(instance->triDmInverses, instance->cudaVertPos, instance->triangles, instance->area, _GE, triangleNum, stretchStiff, shearStiff, IPC_dt);
        // computeSoftConstraintGradient(_GE);
        // computeGroundGradient(_gc,1);
        FEMENERGY::computeSoftConstraintGradient(
            instance->cudaVertPos,
            instance->cudaTargetVertPos,
            instance->cudaTargetIndex,
            _GE,
            instance->softMotionRate,
            instance->softAnimationFullRate,
            instance->numSoftConstraints
        );
        FEMENERGY::computeGroundGradient(
            instance->cudaVertPos,
            instance->cudaGroundOffset,
            instance->cudaGroundNormal,
            instance->cudaEnvCollisionPairs,
            _gc,
            instance->cudaGPNum,
            instance->cudaH3x3,
            instance->gpNum,
            instance->dHat,
            1
        );

        calBarrierGradient(instance, _gc, 1);
        double gsum = reduction2Kappa(0, _gc, _GE, pcg_data->mc_squeue, instance->numVertices);
        double gsnorm = reduction2Kappa(1, _gc, _GE, pcg_data->mc_squeue, instance->numVertices);
        double minKappa = -gsum / gsnorm;

        if (minKappa > 0.0) {
            instance->Kappa = minKappa;
        }
        suggestKappa(instance, minKappa);
        if (instance->Kappa < minKappa) {
            instance->Kappa = minKappa;
        }
        upperBoundKappa(instance, instance->Kappa);
    }
}


bool checkGroundIntersection(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->gpNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //

    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection << <blockNum, threadNum >> > (instance->cudaVertPos, instance->cudaGroundOffset, instance->cudaGroundNormal, instance->cudaEnvCollisionPairs, _isIntersect, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if (h_isITST < 0) {
        return true;
    }
    return false;
}

bool isIntersected(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_EF>& bvh_ef) {
    if (checkGroundIntersection(instance)) {
        return true;
    }
    if (bvh_ef->checkEdgeTriIntersectionIfAny(instance->cudaVertPos, instance->dHat)) {
        return true;
    }
    return false;
}


}; // namespace GPUIPC
