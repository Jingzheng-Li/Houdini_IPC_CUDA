

#include <fstream>

#include "ACCD.cuh"
#include "FEMEnergy.cuh"
#include "GIPC.cuh"
#include "GIPC_PDerivative.cuh"
#include "GeometryManager.hpp"
#include "IPCFriction.cuh"

namespace __GPUIPC__ {

#define RANK 2
#define NEWF

__device__ Scalar __cal_Barrier_energy(const Scalar3* _vertexes, const Scalar3* _rest_vertexes,
                                       int4 MMCVIDI, Scalar _Kappa, Scalar _dHat) {
    Scalar dHat_sqrt = sqrt(_dHat);
    Scalar dHat = _dHat;
    Scalar Kappa = _Kappa;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
            Scalar dis;
            __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            Scalar I5 = dis / dHat;

            Scalar lenE = (dis - dHat);
#if (RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#endif
        } else {
            // return 0;
            MMCVIDI.w = -MMCVIDI.w - 1;
            Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            Scalar c = __MATHUTILS__::__norm(
                __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
            Scalar I1 = c * c;
            if (I1 == 0) return 0;
            Scalar dis;
            __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            Scalar I2 = dis / dHat;
            Scalar eps_x =
                __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
            Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                            -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
            Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                            (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
            Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                            (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) * log(I2) *
                            log(I2);
#elif (RANK == 6)
            Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                            (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) * log(I2) *
                            log(I2) * log(I2) * log(I2);
#endif
            if (Energy < 0) printf("I am pee\n");
            return Energy;
        }
    } else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                Scalar c = __MATHUTILS__::__norm(
                    __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
                Scalar I1 = c * c;
                if (I1 == 0) return 0;
                Scalar dis;
                __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                Scalar I2 = dis / dHat;
                Scalar eps_x = __MATHUTILS__::_compute_epx(
                    _rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.y],
                    _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
                Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                                -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                                (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                                (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) *
                                log(I2) * log(I2);
#elif (RANK == 6)
                Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                                (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) *
                                log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if (Energy < 0) printf("I am pp\n");
                return Energy;
            } else {
                Scalar dis;
                __MATHUTILS__::_d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                Scalar I5 = dis / dHat;

                Scalar lenE = (dis - dHat);
#if (RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) *
                       log(I5);
#endif
            }
        } else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                // MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                Scalar c = __MATHUTILS__::__norm(
                    __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
                Scalar I1 = c * c;
                if (I1 == 0) return 0;
                Scalar dis;
                __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                     _vertexes[MMCVIDI.z], dis);
                Scalar I2 = dis / dHat;
                Scalar eps_x = __MATHUTILS__::_compute_epx(
                    _rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w], _rest_vertexes[MMCVIDI.y],
                    _rest_vertexes[MMCVIDI.z]);
#if (RANK == 1)
                Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                                -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                                (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                                (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) *
                                log(I2) * log(I2);
#elif (RANK == 6)
                Scalar Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) *
                                (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) *
                                log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if (Energy < 0) printf("I am ppe\n");
                return Energy;
            } else {
                Scalar dis;
                __MATHUTILS__::_d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                     dis);
                Scalar I5 = dis / dHat;

                Scalar lenE = (dis - dHat);
#if (RANK == 1)
                return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
                return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
                return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
                return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) *
                       log(I5);
#endif
            }
        } else {
            Scalar dis;
            __MATHUTILS__::_d_PT(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            Scalar I5 = dis / dHat;

            Scalar lenE = (dis - dHat);
#if (RANK == 1)
            return -Kappa * lenE * lenE * log(I5);
#elif (RANK == 2)
            return Kappa * lenE * lenE * log(I5) * log(I5);
#elif (RANK == 3)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5);
#elif (RANK == 4)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 5)
            return -Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#elif (RANK == 6)
            return Kappa * lenE * lenE * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) * log(I5);
#endif
        }
    }
}


__device__ Scalar _selfConstraintVal(const Scalar3* vertexes, const int4& active) {
    Scalar val;
    if (active.x >= 0) {
        if (active.w >= 0) {
            __MATHUTILS__::_d_EE(vertexes[active.x], vertexes[active.y], vertexes[active.z],
                                 vertexes[active.w], val);
        } else {
            __MATHUTILS__::_d_EE(vertexes[active.x], vertexes[active.y], vertexes[active.z],
                                 vertexes[-active.w - 1], val);
        }
    } else {
        if (active.z < 0) {
            if (active.y < 0) {
                __MATHUTILS__::_d_PP(vertexes[-active.x - 1], vertexes[-active.y - 1], val);
            } else {
                __MATHUTILS__::_d_PP(vertexes[-active.x - 1], vertexes[active.y], val);
            }
        } else if (active.w < 0) {
            if (active.y < 0) {
                __MATHUTILS__::_d_PE(vertexes[-active.x - 1], vertexes[-active.y - 1],
                                     vertexes[active.z], val);
            } else {
                __MATHUTILS__::_d_PE(vertexes[-active.x - 1], vertexes[active.y],
                                     vertexes[active.z], val);
            }
        } else {
            __MATHUTILS__::_d_PT(vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z],
                                 vertexes[active.w], val);
        }
    }
    return val;
}

__device__ Scalar __cal_Friction_gd_energy(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                           const Scalar3* _normal, uint32_t gidx, Scalar dt,
                                           Scalar lastH, Scalar eps) {
    Scalar3 normal = *_normal;
    Scalar3 Vdiff = __MATHUTILS__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    Scalar3 VProj = __MATHUTILS__::__minus(
        Vdiff, __MATHUTILS__::__s_vec_multiply(normal, __MATHUTILS__::__v_vec_dot(Vdiff, normal)));
    Scalar VProjMag2 = __MATHUTILS__::__squaredNorm(VProj);
    if (VProjMag2 > eps * eps) {
        return lastH * (sqrt(VProjMag2) - eps * 0.5);

    } else {
        return lastH * VProjMag2 / eps * 0.5;
    }
}

__device__ Scalar __cal_Friction_energy(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                        int4 MMCVIDI, Scalar dt, Scalar2 distCoord,
                                        __MATHUTILS__::Matrix3x2S tanBasis, Scalar lastH,
                                        Scalar fricDHat, Scalar eps) {
    Scalar3 relDX3D;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
            __IPCFRICTION__::computeRelDX_EE(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord.x,
                distCoord.y, relDX3D);
        }
    } else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y >= 0) {
                __IPCFRICTION__::computeRelDX_PP(
                    __MATHUTILS__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), relDX3D);
            }
        } else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y >= 0) {
                __IPCFRICTION__::computeRelDX_PE(
                    __MATHUTILS__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                    __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                    distCoord.x, relDX3D);
            }
        } else {
            __IPCFRICTION__::computeRelDX_PT(
                __MATHUTILS__::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord.x,
                distCoord.y, relDX3D);
        }
    }
    __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis);
    Scalar relDXSqNorm =
        __MATHUTILS__::__squaredNorm(__MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D));
    if (relDXSqNorm > fricDHat) {
        return lastH * sqrt(relDXSqNorm);
    } else {
        Scalar f0;
        __IPCFRICTION__::f0_SF(relDXSqNorm, eps, f0);
        return lastH * f0;
    }
}

__global__ void _calFrictionHessian_gd(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                       const Scalar3* _normal,
                                       const uint32_t* _last_collisionPair_gd,
                                       __MATHUTILS__::Matrix3x3S* H3x3, uint32_t* D1Index,
                                       int numbers, Scalar dt, Scalar eps2, Scalar* lastH,
                                       Scalar coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar eps = sqrt(eps2);
    int gidx = _last_collisionPair_gd[idx];
    Scalar multiplier_vI = coef * lastH[idx];
    __MATHUTILS__::Matrix3x3S H_vI;

    Scalar3 Vdiff = __MATHUTILS__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    Scalar3 normal = *_normal;
    Scalar3 VProj = __MATHUTILS__::__minus(
        Vdiff, __MATHUTILS__::__s_vec_multiply(normal, __MATHUTILS__::__v_vec_dot(Vdiff, normal)));
    Scalar VProjMag2 = __MATHUTILS__::__squaredNorm(VProj);

    if (VProjMag2 > eps2) {
        Scalar VProjMag = sqrt(VProjMag2);

        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(
            VProj.x * VProj.x * -multiplier_vI / VProjMag2 / VProjMag + (multiplier_vI / VProjMag),
            VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
            VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
            VProj.z * VProj.z * -multiplier_vI / VProjMag2 / VProjMag + (multiplier_vI / VProjMag),
            eigenValues, eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::__set_Mat_val(H_vI, projH.m[0][0], 0, projH.m[0][1], 0, 0, 0, projH.m[1][0],
                                     0, projH.m[1][1]);
    } else {
        __MATHUTILS__::__set_Mat_val(H_vI, (multiplier_vI / eps), 0, 0, 0, 0, 0, 0, 0,
                                     (multiplier_vI / eps));
    }

    H3x3[idx] = H_vI;
    D1Index[idx] = gidx;
}

__global__ void _calFrictionHessian(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                    const int4* _last_collisionPair,
                                    __MATHUTILS__::Matrix12x12S* H12x12,
                                    __MATHUTILS__::Matrix9x9S* H9x9,
                                    __MATHUTILS__::Matrix6x6S* H6x6, uint4* D4Index, uint3* D3Index,
                                    uint2* D2Index, uint32_t* _cpNum, int numbers, Scalar dt,
                                    Scalar2* distCoord, __MATHUTILS__::Matrix3x2S* tanBasis,
                                    Scalar eps2, Scalar* lastH, Scalar coef, int offset4,
                                    int offset3, int offset2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _last_collisionPair[idx];
    Scalar eps = sqrt(eps2);
    Scalar3 relDX3D;
    if (MMCVIDI.x >= 0) {
        __IPCFRICTION__::computeRelDX_EE(
            __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
            distCoord[idx].y, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__squaredNorm(relDX);
        Scalar relDXNorm = sqrt(relDXSqNorm);
        __MATHUTILS__::Matrix12x2S T;
        __IPCFRICTION__::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __MATHUTILS__::Matrix2x2S M2;
        if (relDXSqNorm > eps2) {
            __MATHUTILS__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __MATHUTILS__::__Mat2x2_minus(
                M2,
                __MATHUTILS__::__s_Mat2x2_multiply(__MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                                                   1 / (relDXSqNorm * relDXNorm)));
        } else {
            Scalar f1_div_relDXNorm;
            __IPCFRICTION__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            Scalar f2;
            __IPCFRICTION__::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
            } else {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                   eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::Matrix12x2S TM2 = __MATHUTILS__::__M12x2_M2x2_Multiply(T, projH);

        __MATHUTILS__::Matrix12x12S HessianBlock =
            __MATHUTILS__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T), coef * lastH[idx]);
        int Hidx = atomicAdd(_cpNum + 4, 1);
        Hidx += offset4;
        H12x12[Hidx] = HessianBlock;
        D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
    } else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            MMCVIDI.x = v0I;
            __IPCFRICTION__::computeRelDX_PP(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), relDX3D);

            __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
            Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
            Scalar relDXSqNorm = __MATHUTILS__::__squaredNorm(relDX);
            Scalar relDXNorm = sqrt(relDXSqNorm);
            __MATHUTILS__::Matrix6x2S T;
            __IPCFRICTION__::computeT_PP(tanBasis[idx], T);
            __MATHUTILS__::Matrix2x2S M2;
            if (relDXSqNorm > eps2) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            1 / (relDXSqNorm * relDXNorm)));
            } else {
                Scalar f1_div_relDXNorm;
                __IPCFRICTION__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                Scalar f2;
                __IPCFRICTION__::f2_SF(relDXSqNorm, eps, f2);
                if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                    __MATHUTILS__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __MATHUTILS__::__Mat2x2_minus(
                        M2, __MATHUTILS__::__s_Mat2x2_multiply(
                                __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                                (f1_div_relDXNorm - f2) / relDXSqNorm));
                } else {
                    __MATHUTILS__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __MATHUTILS__::Matrix2x2S projH;
            __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

            Scalar eigenValues[2];
            int eigenNum = 0;
            Scalar2 eigenVecs[2];
            __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                       eigenNum, eigenVecs);
            for (int i = 0; i < eigenNum; i++) {
                if (eigenValues[i] > 0) {
                    __MATHUTILS__::Matrix2x2S eigenMatrix =
                        __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            __MATHUTILS__::Matrix6x2S TM2 = __MATHUTILS__::__M6x2_M2x2_Multiply(T, projH);

            __MATHUTILS__::Matrix6x6S HessianBlock =
                __MATHUTILS__::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T), coef * lastH[idx]);

            int Hidx = atomicAdd(_cpNum + 2, 1);
            Hidx += offset2;
            H6x6[Hidx] = HessianBlock;
            D2Index[Hidx] = make_uint2(MMCVIDI.x, MMCVIDI.y);
        } else if (MMCVIDI.w < 0) {
            MMCVIDI.x = v0I;
            __IPCFRICTION__::computeRelDX_PE(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x, relDX3D);

            __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
            Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
            Scalar relDXSqNorm = __MATHUTILS__::__squaredNorm(relDX);
            Scalar relDXNorm = sqrt(relDXSqNorm);
            __MATHUTILS__::Matrix9x2S T;
            __IPCFRICTION__::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
            __MATHUTILS__::Matrix2x2S M2;
            if (relDXSqNorm > eps2) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            1 / (relDXSqNorm * relDXNorm)));
            } else {
                Scalar f1_div_relDXNorm;
                __IPCFRICTION__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                Scalar f2;
                __IPCFRICTION__::f2_SF(relDXSqNorm, eps, f2);
                if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                    __MATHUTILS__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __MATHUTILS__::__Mat2x2_minus(
                        M2, __MATHUTILS__::__s_Mat2x2_multiply(
                                __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                                (f1_div_relDXNorm - f2) / relDXSqNorm));
                } else {
                    __MATHUTILS__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __MATHUTILS__::Matrix2x2S projH;
            __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

            Scalar eigenValues[2];
            int eigenNum = 0;
            Scalar2 eigenVecs[2];
            __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                       eigenNum, eigenVecs);
            for (int i = 0; i < eigenNum; i++) {
                if (eigenValues[i] > 0) {
                    __MATHUTILS__::Matrix2x2S eigenMatrix =
                        __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            __MATHUTILS__::Matrix9x2S TM2 = __MATHUTILS__::__M9x2_M2x2_Multiply(T, projH);

            __MATHUTILS__::Matrix9x9S HessianBlock =
                __MATHUTILS__::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T), coef * lastH[idx]);
            int Hidx = atomicAdd(_cpNum + 3, 1);
            Hidx += offset3;
            H9x9[Hidx] = HessianBlock;
            D3Index[Hidx] = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
        } else {
            MMCVIDI.x = v0I;
            __IPCFRICTION__::computeRelDX_PT(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x, distCoord[idx].y, relDX3D);

            __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
            Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
            Scalar relDXSqNorm = __MATHUTILS__::__squaredNorm(relDX);
            Scalar relDXNorm = sqrt(relDXSqNorm);
            __MATHUTILS__::Matrix12x2S T;
            __IPCFRICTION__::computeT_PT(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
            __MATHUTILS__::Matrix2x2S M2;
            if (relDXSqNorm > eps2) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] /= relDXNorm;
                M2.m[1][1] /= relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            1 / (relDXSqNorm * relDXNorm)));
            } else {
                Scalar f1_div_relDXNorm;
                __IPCFRICTION__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                Scalar f2;
                __IPCFRICTION__::f2_SF(relDXSqNorm, eps, f2);
                if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                    __MATHUTILS__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                    M2 = __MATHUTILS__::__Mat2x2_minus(
                        M2, __MATHUTILS__::__s_Mat2x2_multiply(
                                __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                                (f1_div_relDXNorm - f2) / relDXSqNorm));
                } else {
                    __MATHUTILS__::__set_Mat_identity(M2);
                    M2.m[0][0] *= f1_div_relDXNorm;
                    M2.m[1][1] *= f1_div_relDXNorm;
                }
            }
            __MATHUTILS__::Matrix2x2S projH;
            __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

            Scalar eigenValues[2];
            int eigenNum = 0;
            Scalar2 eigenVecs[2];
            __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                       eigenNum, eigenVecs);
            for (int i = 0; i < eigenNum; i++) {
                if (eigenValues[i] > 0) {
                    __MATHUTILS__::Matrix2x2S eigenMatrix =
                        __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                    eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                    projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
                }
            }

            __MATHUTILS__::Matrix12x2S TM2 = __MATHUTILS__::__M12x2_M2x2_Multiply(T, projH);

            __MATHUTILS__::Matrix12x12S HessianBlock = __MATHUTILS__::__s_M12x12_Multiply(
                __M12x2_M12x2T_Multiply(TM2, T), coef * lastH[idx]);
            int Hidx = atomicAdd(_cpNum + 4, 1);
            Hidx += offset4;
            H12x12[Hidx] = HessianBlock;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    }
}

__global__ void _calBarrierGradientAndHessian(
    const Scalar3* _vertexes, const Scalar3* _rest_vertexes, const int4* _collisionPair,
    Scalar3* _gradient, __MATHUTILS__::Matrix12x12S* H12x12, __MATHUTILS__::Matrix9x9S* H9x9,
    __MATHUTILS__::Matrix6x6S* H6x6, uint4* D4Index, uint3* D3Index, uint2* D2Index,
    uint32_t* _cpNum, int* matIndex, Scalar dHat, Scalar Kappa, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _collisionPair[idx];
    Scalar dHat_sqrt = sqrt(dHat);
    // Scalar dHat = dHat_sqrt * dHat_sqrt;
    // Scalar Kappa = 1;
    Scalar gassThreshold = 1e-4;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
#ifdef NEWF
            Scalar dis;
            __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            Scalar d_hat_sqrt = sqrt(dHat);
            __MATHUTILS__::Matrix12x9S PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w], d_hat_sqrt, PFPxT);
            Scalar I5 = pow(dis / d_hat_sqrt, 2);
            __MATHUTILS__::Vector9S tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] =
                0;
            tmp.v[8] = dis / d_hat_sqrt;

            __MATHUTILS__::Vector9S q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
            q0.v[8] = 1;
            // q0 = __MATHUTILS__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __MATHUTILS__::Matrix9x9S H;
            //__MATHUTILS__::__init_Mat9x9(H, 0);
#else

            Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            Scalar3 v2 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __MATHUTILS__::Matrix3x3S Ds;
            __MATHUTILS__::__set_Mat_val_column(Ds, v0, v1, v2);
            Scalar3 normal = __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(
                v0, __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            Scalar dis = __MATHUTILS__::__v_vec_dot(v1, normal);
            if (dis < 0) {
                normal = make_Scalar3(-normal.x, -normal.y, -normal.z);
                dis = -dis;
            }

            Scalar3 pos2 = __MATHUTILS__::__add(
                _vertexes[MMCVIDI.z], __MATHUTILS__::__s_vec_multiply(normal, dHat_sqrt - dis));
            Scalar3 pos3 = __MATHUTILS__::__add(
                _vertexes[MMCVIDI.w], __MATHUTILS__::__s_vec_multiply(normal, dHat_sqrt - dis));

            Scalar3 u0 = v0;
            Scalar3 u1 = __MATHUTILS__::__minus(pos2, _vertexes[MMCVIDI.x]);
            Scalar3 u2 = __MATHUTILS__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __MATHUTILS__::Matrix3x3S Dm, DmInv;
            __MATHUTILS__::__set_Mat_val_column(Dm, u0, u1, u2);

            __MATHUTILS__::__Inverse(Dm, DmInv);

            __MATHUTILS__::Matrix3x3S F;
            __MATHUTILS__::__M_Mat_multiply(Ds, DmInv, F);

            Scalar3 FxN = __MATHUTILS__::__M_v_multiply(F, normal);
            Scalar I5 = __MATHUTILS__::__squaredNorm(FxN);

            __MATHUTILS__::Matrix9x12S PFPx = __computePFDsPX3D_Scalar(DmInv);

            __MATHUTILS__::Matrix3x3S fnn;

            __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(normal, normal);

            __MATHUTILS__::__M_Mat_multiply(F, nn, fnn);

            __MATHUTILS__::Vector9S tmp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(fnn);

#endif

#if (RANK == 1)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, -2 *
                         (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                          (3 * I5 + 2 * I5 * log(I5) - 3)) /
                         I5);
#elif (RANK == 4)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                      (2 * I5 + I5 * log(I5) - 2)) /
                         I5);
#elif (RANK == 5)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, -2 *
                         (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (5 * I5 + 2 * I5 * log(I5) - 5)) /
                         I5);
#elif (RANK == 6)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) *
                      (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                         I5);
#endif

#if (RANK == 1)
            Scalar lambda0 =
                Kappa *
                (2 * dHat * dHat *
                 (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) /
                I5;
            if (dis * dis < gassThreshold * dHat) {
                Scalar lambda1 = Kappa *
                                 (2 * dHat * dHat *
                                  (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold) -
                                   7 * gassThreshold * gassThreshold -
                                   6 * gassThreshold * gassThreshold * log(gassThreshold) + 1)) /
                                 gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            Scalar lambda0 =
                -(4 * Kappa * dHat * dHat *
                  (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) -
                   2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) /
                I5;
            if (dis * dis < gassThreshold * dHat) {
                Scalar lambda1 =
                    -(4 * Kappa * dHat * dHat *
                      (4 * gassThreshold + log(gassThreshold) -
                       3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) +
                       6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold +
                       gassThreshold * log(gassThreshold) * log(gassThreshold) -
                       7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) /
                    gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            Scalar lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) *
                 (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) -
                  12 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12)) /
                I5;
#elif (RANK == 4)
            Scalar lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) *
                  (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 12 * I5 * log(I5) -
                   12 * I5 * I5 + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12)) /
                I5;
#elif (RANK == 5)
            Scalar lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) *
                 (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 30 * I5 * log(I5) -
                  40 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40)) /
                I5;
#elif (RANK == 6)
            Scalar lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                  (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) -
                   30 * I5 * I5 + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30)) /
                I5;
#endif

#ifdef NEWF
            __MATHUTILS__::Vector12S gradient_vec =
                __MATHUTILS__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
            H = __MATHUTILS__::__S_Mat9x9_multiply(__MATHUTILS__::__v9_vec9_toMat9x9(q0, q0),
                                                   lambda0);

            __MATHUTILS__::Matrix12x12S
                Hessian;  // =
                          // __MATHUTILS__::__M12x9_M9x12_Multiply(__MATHUTILS__::__M12x9_M9x9_Multiply(PFPxT,
                          // H), __MATHUTILS__::__Transpose12x9(PFPxT));
            __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(
                __MATHUTILS__::__Transpose9x12(PFPx), flatten_pk1);
            //__MATHUTILS__::Matrix3x3S Q0;

            //            __MATHUTILS__::Matrix3x3S fnn;

            //           __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(normal,
            //           normal);

            //            __MATHUTILS__::__M_Mat_multiply(F, nn, fnn);

            __MATHUTILS__::Vector9S q0 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(fnn);

            q0 = __MATHUTILS__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __MATHUTILS__::Matrix9x9S H;
            __MATHUTILS__::__init_Mat9x9(H, 0);

            H = __MATHUTILS__::__S_Mat9x9_multiply(__MATHUTILS__::__v9_vec9_toMat9x9(q0, q0),
                                                   lambda0);

            __MATHUTILS__::Matrix12x9S PFPxTransPos = __MATHUTILS__::__Transpose9x12(PFPx);
            __MATHUTILS__::Matrix12x12S Hessian = __MATHUTILS__::__M12x9_M9x12_Multiply(
                __MATHUTILS__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
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
            int Hidx = matIndex[idx];  // atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

        } else {
            // return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            Scalar c = __MATHUTILS__::__norm(
                __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
            Scalar I1 = c * c;
            if (I1 == 0) return;
            Scalar dis;
            __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            Scalar I2 = dis / dHat;
            dis = sqrt(dis);

            __MATHUTILS__::Matrix3x3S F;
            __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            Scalar3 n1 = make_Scalar3(0, 1, 0);
            Scalar3 n2 = make_Scalar3(0, 0, 1);

            Scalar eps_x =
                __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);

            __MATHUTILS__::Matrix3x3S g1, g2;

            __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
            __MATHUTILS__::__M_Mat_multiply(F, nn, g1);
            nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
            __MATHUTILS__::__M_Mat_multiply(F, nn, g2);

            __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
            __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

            __MATHUTILS__::Matrix12x9S PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
            Scalar p1 = Kappa * 2 *
                        (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                        (eps_x * eps_x);
            Scalar p2 =
                Kappa * 2 *
                (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                (I2 * eps_x * eps_x);
#elif (RANK == 2)
            Scalar p1 = -Kappa * 2 *
                        (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                        (eps_x * eps_x);
            Scalar p2 = -Kappa * 2 *
                        (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) *
                         (I2 + I2 * log(I2) - 1)) /
                        (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            Scalar p1 = -Kappa * 2 *
                        (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                        (eps_x * eps_x);
            Scalar p2 = -Kappa * 2 *
                        (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) *
                         (2 * I2 + I2 * log(I2) - 2)) /
                        (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            Scalar p1 = -Kappa * 2 *
                        (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                        (eps_x * eps_x);
            Scalar p2 = -Kappa * 2 *
                        (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) *
                         (3 * I2 + I2 * log(I2) - 3)) /
                        (I2 * (eps_x * eps_x));
#endif
            __MATHUTILS__::Vector9S flatten_pk1 =
                __MATHUTILS__::__add9(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                      __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
            __MATHUTILS__::Vector12S gradient_vec =
                __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

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
            Scalar lambda10 = Kappa *
                              (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                              (eps_x * eps_x);
            Scalar lambda11 = Kappa * 2 *
                              (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                              (eps_x * eps_x);
            Scalar lambda12 = Kappa * 2 *
                              (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                              (eps_x * eps_x);
#elif (RANK == 2)
            Scalar lambda10 =
                -Kappa *
                (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                (eps_x * eps_x);
            Scalar lambda11 =
                -Kappa *
                (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                (eps_x * eps_x);
            Scalar lambda12 =
                -Kappa *
                (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                (eps_x * eps_x);
#elif (RANK == 4)
            Scalar lambda10 =
                -Kappa *
                (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                (eps_x * eps_x);
            Scalar lambda11 =
                -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                (eps_x * eps_x);
            Scalar lambda12 =
                -Kappa * (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                (eps_x * eps_x);
#elif (RANK == 6)
            Scalar lambda10 =
                -Kappa *
                (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                (eps_x * eps_x);
            Scalar lambda11 =
                -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                (eps_x * eps_x);
            Scalar lambda12 =
                -Kappa * (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                (eps_x * eps_x);
#endif
            __MATHUTILS__::Matrix3x3S Tx, Ty, Tz;
            __MATHUTILS__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
            __MATHUTILS__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
            __MATHUTILS__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

            __MATHUTILS__::Vector9S q11 =
                __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M_Mat_multiply(Tx, g1));
            __MATHUTILS__::__normalized_vec9_Scalar(q11);
            __MATHUTILS__::Vector9S q12 =
                __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M_Mat_multiply(Tz, g1));
            __MATHUTILS__::__normalized_vec9_Scalar(q12);

            __MATHUTILS__::Matrix9x9S projectedH;
            __MATHUTILS__::__init_Mat9x9(projectedH, 0);

            __MATHUTILS__::Matrix9x9S M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q11, q11);
            M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lambda11);
            projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

            M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q12, q12);
            M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lambda12);
            projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
            Scalar lambda20 =
                -Kappa *
                (2 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                 (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1)) /
                (I2 * eps_x * eps_x);
#elif (RANK == 2)
            Scalar lambda20 =
                Kappa *
                (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                 (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) -
                  2 * I2 * I2 + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) /
                (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            Scalar lambda20 =
                Kappa *
                (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x) *
                 (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 12 * I2 * log(I2) -
                  12 * I2 * I2 + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12)) /
                (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            Scalar lambda20 =
                Kappa *
                (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x) *
                 (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 18 * I2 * log(I2) -
                  30 * I2 * I2 + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30)) /
                (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
            Scalar lambdag1g =
                Kappa * 4 * c * F.m[2][2] *
                ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                 (I2 * eps_x * eps_x));
#elif (RANK == 2)
            Scalar lambdag1g =
                -Kappa * 4 * c * F.m[2][2] *
                (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                               (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) *
                                (2 * I2 + I2 * log(I2) - 2)) /
                               (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                               (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) *
                                (3 * I2 + I2 * log(I2) - 3)) /
                               (I2 * (eps_x * eps_x));
#endif
            Scalar eigenValues[2];
            int eigenNum = 0;
            Scalar2 eigenVecs[2];
            __MATHUTILS__::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues,
                                       eigenNum, eigenVecs);

            for (int i = 0; i < eigenNum; i++) {
                if (eigenValues[i] > 0) {
                    __MATHUTILS__::Matrix3x3S eigenMatrix;
                    __MATHUTILS__::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0, 0,
                                                 eigenVecs[i].y);

                    __MATHUTILS__::Vector9S eigenMVec =
                        __MATHUTILS__::__Mat3x3_to_vec9_Scalar(eigenMatrix);

                    M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
                    M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, eigenValues[i]);
                    projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);
                }
            }

            //__MATHUTILS__::Matrix9x12S PFPxTransPos = __MATHUTILS__::__Transpose12x9(PFPx);
            __MATHUTILS__::Matrix12x12S
                Hessian;  // =
                          // __MATHUTILS__::__M12x9_M9x12_Multiply(__MATHUTILS__::__M12x9_M9x9_Multiply(PFPx,
                          // projectedH), PFPxTransPos);
            __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
            int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    } else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                Scalar c = __MATHUTILS__::__norm(
                    __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
                Scalar I1 = c * c;
                if (I1 == 0) return;
                Scalar dis;
                __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                Scalar I2 = dis / dHat;
                dis = sqrt(dis);

                __MATHUTILS__::Matrix3x3S F;
                __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                Scalar3 n1 = make_Scalar3(0, 1, 0);
                Scalar3 n2 = make_Scalar3(0, 0, 1);

                Scalar eps_x = __MATHUTILS__::_compute_epx(
                    _rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.y],
                    _rest_vertexes[MMCVIDI.w]);

                __MATHUTILS__::Matrix3x3S g1, g2;

                __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
                __MATHUTILS__::__M_Mat_multiply(F, nn, g1);
                nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
                __MATHUTILS__::__M_Mat_multiply(F, nn, g2);

                __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
                __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

                __MATHUTILS__::Matrix12x9S PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
                Scalar p1 = Kappa * 2 *
                            (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                            (eps_x * eps_x);
                Scalar p2 =
                    Kappa * 2 *
                    (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                    (I2 * eps_x * eps_x);
#elif (RANK == 2)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (I2 + I2 * log(I2) - 1)) /
                            (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (2 * I2 + I2 * log(I2) - 2)) /
                            (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (3 * I2 + I2 * log(I2) - 3)) /
                            (I2 * (eps_x * eps_x));
#endif
                __MATHUTILS__::Vector9S flatten_pk1 =
                    __MATHUTILS__::__add9(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                          __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
                __MATHUTILS__::Vector12S gradient_vec =
                    __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

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
                Scalar lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                    (eps_x * eps_x);
                Scalar lambda11 = Kappa * 2 *
                                  (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                                  (eps_x * eps_x);
                Scalar lambda12 = Kappa * 2 *
                                  (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                                  (eps_x * eps_x);
#elif (RANK == 2)
                Scalar lambda10 =
                    -Kappa *
                    (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                    (eps_x * eps_x);
                Scalar lambda11 =
                    -Kappa *
                    (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar lambda12 =
                    -Kappa *
                    (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
#elif (RANK == 4)
                Scalar lambda10 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                    (eps_x * eps_x);
                Scalar lambda11 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar lambda12 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
#elif (RANK == 6)
                Scalar lambda10 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                    (eps_x * eps_x);
                Scalar lambda11 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar lambda12 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
#endif
                __MATHUTILS__::Matrix3x3S Tx, Ty, Tz;
                __MATHUTILS__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __MATHUTILS__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __MATHUTILS__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                __MATHUTILS__::Vector9S q11 =
                    __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M_Mat_multiply(Tx, g1));
                __MATHUTILS__::__normalized_vec9_Scalar(q11);
                __MATHUTILS__::Vector9S q12 =
                    __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M_Mat_multiply(Tz, g1));
                __MATHUTILS__::__normalized_vec9_Scalar(q12);

                __MATHUTILS__::Matrix9x9S projectedH;
                __MATHUTILS__::__init_Mat9x9(projectedH, 0);

                __MATHUTILS__::Matrix9x9S M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                Scalar lambda20 =
                    -Kappa *
                    (2 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                     (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1)) /
                    (I2 * eps_x * eps_x);
#elif (RANK == 2)
                Scalar lambda20 =
                    Kappa *
                    (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                     (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) -
                      2 * I2 * I2 + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) /
                    (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                Scalar lambda20 =
                    Kappa *
                    (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x) *
                     (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 12 * I2 * log(I2) -
                      12 * I2 * I2 + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12)) /
                    (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                Scalar lambda20 =
                    Kappa *
                    (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x) *
                     (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 18 * I2 * log(I2) -
                      30 * I2 * I2 + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30)) /
                    (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
                Scalar lambdag1g =
                    Kappa * 4 * c * F.m[2][2] *
                    ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                     (I2 * eps_x * eps_x));
#elif (RANK == 2)
                Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                                   (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) *
                                    (I2 + I2 * log(I2) - 1)) /
                                   (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                                   (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) *
                                    (2 * I2 + I2 * log(I2) - 2)) /
                                   (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                                   (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) *
                                    (3 * I2 + I2 * log(I2) - 3)) /
                                   (I2 * (eps_x * eps_x));
#endif
                Scalar eigenValues[2];
                int eigenNum = 0;
                Scalar2 eigenVecs[2];
                __MATHUTILS__::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues,
                                           eigenNum, eigenVecs);

                for (int i = 0; i < eigenNum; i++) {
                    if (eigenValues[i] > 0) {
                        __MATHUTILS__::Matrix3x3S eigenMatrix;
                        __MATHUTILS__::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0,
                                                     0, eigenVecs[i].y);

                        __MATHUTILS__::Vector9S eigenMVec =
                            __MATHUTILS__::__Mat3x3_to_vec9_Scalar(eigenMatrix);

                        M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
                        M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, eigenValues[i]);
                        projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);
                    }
                }

                //__MATHUTILS__::Matrix9x12S PFPxTransPos = __MATHUTILS__::__Transpose12x9(PFPx);
                __MATHUTILS__::Matrix12x12S
                    Hessian;  // =
                              // __MATHUTILS__::__M12x9_M9x12_Multiply(__MATHUTILS__::__M12x9_M9x9_Multiply(PFPx,
                              // projectedH), PFPxTransPos);
                __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 4, 1);

                H12x12[Hidx] = Hessian;
                D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            } else {
#ifdef NEWF
                Scalar dis;
                __MATHUTILS__::_d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis = sqrt(dis);
                Scalar d_hat_sqrt = sqrt(dHat);
                __MATHUTILS__::Vector6S PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                Scalar I5 = pow(dis / d_hat_sqrt, 2);
                Scalar fnn = dis / d_hat_sqrt;

#if (RANK == 1)
                Scalar flatten_pk1 =
                    fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
#elif (RANK == 2)
                Scalar flatten_pk1 =
                    fnn * 2 *
                    (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
#elif (RANK == 3)
                Scalar flatten_pk1 = fnn * -2 *
                                     (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                                      (3 * I5 + 2 * I5 * log(I5) - 3)) /
                                     I5;
#elif (RANK == 4)
                Scalar flatten_pk1 = fnn *
                                     (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) *
                                      (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) /
                                     I5;
#elif (RANK == 5)
                Scalar flatten_pk1 = fnn * -2 *
                                     (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                                      (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) /
                                     I5;
#elif (RANK == 6)
                Scalar flatten_pk1 = fnn *
                                     (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) *
                                      log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                                     I5;
#endif

                __MATHUTILS__::Vector6S gradient_vec =
                    __MATHUTILS__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                Scalar3 Ds = v0;
                Scalar dis = __MATHUTILS__::__norm(v0);
                // if (dis > dHat_sqrt) return;
                Scalar3 vec_normal = __MATHUTILS__::__normalized(make_Scalar3(-v0.x, -v0.y, -v0.z));
                Scalar3 target = make_Scalar3(0, 1, 0);
                Scalar3 vec = __MATHUTILS__::__v_vec_cross(vec_normal, target);
                Scalar cos = __MATHUTILS__::__v_vec_dot(vec_normal, target);
                __MATHUTILS__::Matrix3x3S rotation;
                __MATHUTILS__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __MATHUTILS__::Vector6S PDmPx;
                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                } else {
                    __MATHUTILS__::Matrix3x3S cross_vec;
                    __MATHUTILS__::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x,
                                                 -vec.y, vec.x, 0);

                    rotation = __MATHUTILS__::__Mat_add(
                        rotation,
                        __MATHUTILS__::__Mat_add(
                            cross_vec, __MATHUTILS__::__S_Mat_multiply(
                                           __MATHUTILS__::__M_Mat_multiply(cross_vec, cross_vec),
                                           1.0 / (1 + cos))));
                }

                Scalar3 pos0 = __MATHUTILS__::__add(
                    _vertexes[v0I], __MATHUTILS__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                Scalar3 rotate_uv0 = __MATHUTILS__::__M_v_multiply(rotation, pos0);
                Scalar3 rotate_uv1 = __MATHUTILS__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                Scalar uv0 = rotate_uv0.y;
                Scalar uv1 = rotate_uv1.y;

                Scalar u0 = uv1 - uv0;
                Scalar Dm = u0;  // PFPx
                Scalar DmInv = 1 / u0;

                Scalar3 F = __MATHUTILS__::__s_vec_multiply(Ds, DmInv);
                Scalar I5 = __MATHUTILS__::__squaredNorm(F);

                Scalar3 tmp = F;

#if (RANK == 1)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                        I5);

#elif (RANK == 3)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                              (3 * I5 + 2 * I5 * log(I5) - 3)) /
                             I5);
#elif (RANK == 4)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (2 * I5 + I5 * log(I5) - 2)) /
                             I5);
#elif (RANK == 5)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                              (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) /
                             I5);
#elif (RANK == 6)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                          log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                             I5);
#endif
                __MATHUTILS__::Matrix3x6S PFPx = __FEMENERGY__::__computePFDsPX3D_3x6_Scalar(DmInv);

                __MATHUTILS__::Vector6S gradient_vec = __MATHUTILS__::__M6x3_v3_multiply(
                    __MATHUTILS__::__Transpose3x6(PFPx), flatten_pk1);
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
                Scalar lambda0 =
                    Kappa *
                    (2 * dHat * dHat *
                     (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) /
                    I5;
                if (dis * dis < gassThreshold * dHat) {
                    Scalar lambda1 =
                        Kappa *
                        (2 * dHat * dHat *
                         (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold) -
                          7 * gassThreshold * gassThreshold -
                          6 * gassThreshold * gassThreshold * log(gassThreshold) + 1)) /
                        gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                Scalar lambda0 =
                    -(4 * Kappa * dHat * dHat *
                      (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) -
                       2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) /
                    I5;
                if (dis * dis < gassThreshold * dHat) {
                    Scalar lambda1 =
                        -(4 * Kappa * dHat * dHat *
                          (4 * gassThreshold + log(gassThreshold) -
                           3 * gassThreshold * gassThreshold * log(gassThreshold) *
                               log(gassThreshold) +
                           6 * gassThreshold * log(gassThreshold) -
                           2 * gassThreshold * gassThreshold +
                           gassThreshold * log(gassThreshold) * log(gassThreshold) -
                           7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) /
                        gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                Scalar lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) *
                     (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) -
                      12 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12)) /
                    I5;
#elif (RANK == 4)
                Scalar lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) *
                      (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 12 * I5 * log(I5) -
                       12 * I5 * I5 + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12)) /
                    I5;
#elif (RANK == 5)
                Scalar lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) *
                     (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 30 * I5 * log(I5) -
                      40 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40)) /
                    I5;
#elif (RANK == 6)
                Scalar lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                      (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) -
                       30 * I5 * I5 + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30)) /
                    I5;
#endif

#ifdef NEWF
                Scalar H = lambda0;
                __MATHUTILS__::Matrix6x6S Hessian = __MATHUTILS__::__s_M6x6_Multiply(
                    __MATHUTILS__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
#else
                Scalar3 q0 = __MATHUTILS__::__s_vec_multiply(F, 1 / sqrt(I5));

                __MATHUTILS__::Matrix3x3S H =
                    __MATHUTILS__::__S_Mat_multiply(__MATHUTILS__::__v_vec_toMat(q0, q0),
                                                    lambda0);  // lambda0 * q0 * q0.transpose();

                __MATHUTILS__::Matrix6x3S PFPxTransPos = __MATHUTILS__::__Transpose3x6(PFPx);
                __MATHUTILS__::Matrix6x6S Hessian = __MATHUTILS__::__M6x3_M3x6_Multiply(
                    __MATHUTILS__::__M6x3_M3x3_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 2, 1);

                H6x6[Hidx] = Hessian;
                D2Index[Hidx] = make_uint2(v0I, MMCVIDI.y);
            }

        } else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                Scalar c = __MATHUTILS__::__norm(
                    __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
                Scalar I1 = c * c;
                if (I1 == 0) return;
                Scalar dis;
                __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                     _vertexes[MMCVIDI.z], dis);
                Scalar I2 = dis / dHat;
                dis = sqrt(dis);

                __MATHUTILS__::Matrix3x3S F;
                __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                Scalar3 n1 = make_Scalar3(0, 1, 0);
                Scalar3 n2 = make_Scalar3(0, 0, 1);

                Scalar eps_x = __MATHUTILS__::_compute_epx(
                    _rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w], _rest_vertexes[MMCVIDI.y],
                    _rest_vertexes[MMCVIDI.z]);

                __MATHUTILS__::Matrix3x3S g1, g2;

                __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
                __MATHUTILS__::__M_Mat_multiply(F, nn, g1);
                nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
                __MATHUTILS__::__M_Mat_multiply(F, nn, g2);

                __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
                __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

                __MATHUTILS__::Matrix12x9S PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
                Scalar p1 = Kappa * 2 *
                            (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                            (eps_x * eps_x);
                Scalar p2 =
                    Kappa * 2 *
                    (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                    (I2 * eps_x * eps_x);
#elif (RANK == 2)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (I2 + I2 * log(I2) - 1)) /
                            (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (2 * I2 + I2 * log(I2) - 2)) /
                            (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (3 * I2 + I2 * log(I2) - 3)) /
                            (I2 * (eps_x * eps_x));
#endif
                __MATHUTILS__::Vector9S flatten_pk1 =
                    __MATHUTILS__::__add9(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                          __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
                __MATHUTILS__::Vector12S gradient_vec =
                    __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

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
                Scalar lambda10 =
                    Kappa * (4 * dHat * dHat * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                    (eps_x * eps_x);
                Scalar lambda11 = Kappa * 2 *
                                  (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                                  (eps_x * eps_x);
                Scalar lambda12 = Kappa * 2 *
                                  (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                                  (eps_x * eps_x);
#elif (RANK == 2)
                Scalar lambda10 =
                    -Kappa *
                    (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                    (eps_x * eps_x);
                Scalar lambda11 =
                    -Kappa *
                    (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar lambda12 =
                    -Kappa *
                    (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
#elif (RANK == 4)
                Scalar lambda10 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 4) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                    (eps_x * eps_x);
                Scalar lambda11 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar lambda12 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
#elif (RANK == 6)
                Scalar lambda10 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 6) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) /
                    (eps_x * eps_x);
                Scalar lambda11 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar lambda12 =
                    -Kappa *
                    (4 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
#endif
                __MATHUTILS__::Matrix3x3S Tx, Ty, Tz;
                __MATHUTILS__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                __MATHUTILS__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                __MATHUTILS__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                __MATHUTILS__::Vector9S q11 =
                    __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M_Mat_multiply(Tx, g1));
                __MATHUTILS__::__normalized_vec9_Scalar(q11);
                __MATHUTILS__::Vector9S q12 =
                    __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M_Mat_multiply(Tz, g1));
                __MATHUTILS__::__normalized_vec9_Scalar(q12);

                __MATHUTILS__::Matrix9x9S projectedH;
                __MATHUTILS__::__init_Mat9x9(projectedH, 0);

                __MATHUTILS__::Matrix9x9S M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q11, q11);
                M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lambda11);
                projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

                M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q12, q12);
                M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lambda12);
                projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

#if (RANK == 1)
                Scalar lambda20 =
                    -Kappa *
                    (2 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                     (6 * I2 + 2 * I2 * log(I2) - 7 * I2 * I2 - 6 * I2 * I2 * log(I2) + 1)) /
                    (I2 * eps_x * eps_x);
#elif (RANK == 2)
                Scalar lambda20 =
                    Kappa *
                    (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                     (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) -
                      2 * I2 * I2 + I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) /
                    (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                Scalar lambda20 =
                    Kappa *
                    (4 * I1 * dHat * dHat * log(I2) * log(I2) * (I1 - 2 * eps_x) *
                     (24 * I2 + 2 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 12 * I2 * log(I2) -
                      12 * I2 * I2 + I2 * log(I2) * log(I2) - 14 * I2 * I2 * log(I2) - 12)) /
                    (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                Scalar lambda20 =
                    Kappa *
                    (4 * I1 * dHat * dHat * pow(log(I2), 4) * (I1 - 2 * eps_x) *
                     (60 * I2 + 3 * log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 18 * I2 * log(I2) -
                      30 * I2 * I2 + I2 * log(I2) * log(I2) - 21 * I2 * I2 * log(I2) - 30)) /
                    (I2 * (eps_x * eps_x));
#endif

#if (RANK == 1)
                Scalar lambdag1g =
                    Kappa * 4 * c * F.m[2][2] *
                    ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                     (I2 * eps_x * eps_x));
#elif (RANK == 2)
                Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                                   (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) *
                                    (I2 + I2 * log(I2) - 1)) /
                                   (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                                   (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) *
                                    (2 * I2 + I2 * log(I2) - 2)) /
                                   (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                                   (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) *
                                    (3 * I2 + I2 * log(I2) - 3)) /
                                   (I2 * (eps_x * eps_x));
#endif
                Scalar eigenValues[2];
                int eigenNum = 0;
                Scalar2 eigenVecs[2];
                __MATHUTILS__::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues,
                                           eigenNum, eigenVecs);

                for (int i = 0; i < eigenNum; i++) {
                    if (eigenValues[i] > 0) {
                        __MATHUTILS__::Matrix3x3S eigenMatrix;
                        __MATHUTILS__::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0,
                                                     0, eigenVecs[i].y);

                        __MATHUTILS__::Vector9S eigenMVec =
                            __MATHUTILS__::__Mat3x3_to_vec9_Scalar(eigenMatrix);

                        M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
                        M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, eigenValues[i]);
                        projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);
                    }
                }

                //__MATHUTILS__::Matrix9x12S PFPxTransPos = __MATHUTILS__::__Transpose12x9(PFPx);
                __MATHUTILS__::Matrix12x12S
                    Hessian;  // =
                              // __MATHUTILS__::__M12x9_M9x12_Multiply(__MATHUTILS__::__M12x9_M9x9_Multiply(PFPx,
                              // projectedH), PFPxTransPos);
                __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 4, 1);

                H12x12[Hidx] = Hessian;
                D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            } else {
#ifdef NEWF
                Scalar dis;
                __MATHUTILS__::_d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                     dis);
                dis = sqrt(dis);
                Scalar d_hat_sqrt = sqrt(dHat);
                __MATHUTILS__::Matrix9x4S PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt,
                         PFPxT);
                Scalar I5 = pow(dis / d_hat_sqrt, 2);
                __MATHUTILS__::Vector4S fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //__MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(fnn, 2 *
                // Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                __MATHUTILS__::Vector4S q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3] = 1;
                __MATHUTILS__::Matrix4x4S H;
                //__MATHUTILS__::__init_Mat4x4_val(H, 0);
#if (RANK == 1)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                        I5);
#elif (RANK == 3)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                              (3 * I5 + 2 * I5 * log(I5) - 3)) /
                             I5);
#elif (RANK == 4)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (2 * I5 + I5 * log(I5) - 2)) /
                             I5);
#elif (RANK == 5)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                              (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) /
                             I5);
#elif (RANK == 6)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                          log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                             I5);
#endif

                __MATHUTILS__::Vector9S gradient_vec =
                    __MATHUTILS__::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);

                __MATHUTILS__::Matrix3x2S Ds;
                __MATHUTILS__::__set_Mat3x2_val_column(Ds, v0, v1);

                Scalar3 triangle_normal =
                    __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(v0, v1));
                Scalar3 target = make_Scalar3(0, 1, 0);

                Scalar3 vec = __MATHUTILS__::__v_vec_cross(triangle_normal, target);
                Scalar cos = __MATHUTILS__::__v_vec_dot(triangle_normal, target);

                Scalar3 edge_normal = __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(
                    __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                Scalar dis = __MATHUTILS__::__v_vec_dot(
                    __MATHUTILS__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __MATHUTILS__::Matrix3x3S rotation;
                __MATHUTILS__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __MATHUTILS__::Matrix9x4S PDmPx;

                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                } else {
                    __MATHUTILS__::Matrix3x3S cross_vec;
                    __MATHUTILS__::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x,
                                                 -vec.y, vec.x, 0);

                    rotation = __MATHUTILS__::__Mat_add(
                        rotation,
                        __MATHUTILS__::__Mat_add(
                            cross_vec, __MATHUTILS__::__S_Mat_multiply(
                                           __MATHUTILS__::__M_Mat_multiply(cross_vec, cross_vec),
                                           1.0 / (1 + cos))));
                }

                Scalar3 pos0 = __MATHUTILS__::__add(
                    _vertexes[v0I], __MATHUTILS__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                Scalar3 rotate_uv0 = __MATHUTILS__::__M_v_multiply(rotation, pos0);
                Scalar3 rotate_uv1 = __MATHUTILS__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                Scalar3 rotate_uv2 = __MATHUTILS__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                Scalar3 rotate_normal = __MATHUTILS__::__M_v_multiply(rotation, edge_normal);

                Scalar2 uv0 = make_Scalar2(rotate_uv0.x, rotate_uv0.z);
                Scalar2 uv1 = make_Scalar2(rotate_uv1.x, rotate_uv1.z);
                Scalar2 uv2 = make_Scalar2(rotate_uv2.x, rotate_uv2.z);
                Scalar2 normal = make_Scalar2(rotate_normal.x, rotate_normal.z);

                Scalar2 u0 = __MATHUTILS__::__minus_v2(uv1, uv0);
                Scalar2 u1 = __MATHUTILS__::__minus_v2(uv2, uv0);

                __MATHUTILS__::Matrix2x2S Dm;

                __MATHUTILS__::__set_Mat2x2_val_column(Dm, u0, u1);

                __MATHUTILS__::Matrix2x2S DmInv;
                __MATHUTILS__::__Inverse2x2(Dm, DmInv);

                __MATHUTILS__::Matrix3x2S F = __MATHUTILS__::__M3x2_M2x2_Multiply(Ds, DmInv);

                Scalar3 FxN = __MATHUTILS__::__M3x2_v2_multiply(F, normal);
                Scalar I5 = __MATHUTILS__::__squaredNorm(FxN);

                __MATHUTILS__::Matrix3x2S fnn;

                __MATHUTILS__::Matrix2x2S nn = __MATHUTILS__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __MATHUTILS__::__M3x2_M2x2_Multiply(F, nn);

                __MATHUTILS__::Vector6S tmp = __MATHUTILS__::__Mat3x2_to_vec6_Scalar(fnn);

#if (RANK == 1)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                        I5);
#elif (RANK == 3)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                              (3 * I5 + 2 * I5 * log(I5) - 3)) /
                             I5);
#elif (RANK == 4)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (2 * I5 + I5 * log(I5) - 2)) /
                             I5);
#elif (RANK == 5)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                              (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) /
                             I5);
#elif (RANK == 6)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                          log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                             I5);
#endif

                __MATHUTILS__::Matrix6x9S PFPx = __FEMENERGY__::__computePFDsPX3D_6x9_Scalar(DmInv);

                __MATHUTILS__::Vector9S gradient_vec = __MATHUTILS__::__M9x6_v6_multiply(
                    __MATHUTILS__::__Transpose6x9(PFPx), flatten_pk1);
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
                Scalar lambda0 =
                    Kappa *
                    (2 * dHat * dHat *
                     (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) /
                    I5;
                if (dis * dis < gassThreshold * dHat) {
                    Scalar lambda1 =
                        Kappa *
                        (2 * dHat * dHat *
                         (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold) -
                          7 * gassThreshold * gassThreshold -
                          6 * gassThreshold * gassThreshold * log(gassThreshold) + 1)) /
                        gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 2)
                Scalar lambda0 =
                    -(4 * Kappa * dHat * dHat *
                      (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) -
                       2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) /
                    I5;
                if (dis * dis < gassThreshold * dHat) {
                    Scalar lambda1 =
                        -(4 * Kappa * dHat * dHat *
                          (4 * gassThreshold + log(gassThreshold) -
                           3 * gassThreshold * gassThreshold * log(gassThreshold) *
                               log(gassThreshold) +
                           6 * gassThreshold * log(gassThreshold) -
                           2 * gassThreshold * gassThreshold +
                           gassThreshold * log(gassThreshold) * log(gassThreshold) -
                           7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) /
                        gassThreshold;
                    lambda0 = lambda1;
                }
#elif (RANK == 3)
                Scalar lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) *
                     (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) -
                      12 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12)) /
                    I5;
#elif (RANK == 4)
                Scalar lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) *
                      (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 12 * I5 * log(I5) -
                       12 * I5 * I5 + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12)) /
                    I5;
#elif (RANK == 5)
                Scalar lambda0 =
                    (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) *
                     (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 30 * I5 * log(I5) -
                      40 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40)) /
                    I5;
#elif (RANK == 6)
                Scalar lambda0 =
                    -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                      (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) -
                       30 * I5 * I5 + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30)) /
                    I5;
#endif

#ifdef NEWF
                H = __MATHUTILS__::__S_Mat4x4_multiply(__MATHUTILS__::__v4_vec4_toMat4x4(q0, q0),
                                                       lambda0);

                __MATHUTILS__::Matrix9x9S
                    Hessian;  // =
                              // __MATHUTILS__::__M9x4_M4x9_Multiply(__MATHUTILS__::__M9x4_M4x4_Multiply(PFPxT,
                              // H), __MATHUTILS__::__Transpose9x4(PFPxT));
                __MATHUTILS__::__M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
#else

                __MATHUTILS__::Vector6S q0 = __MATHUTILS__::__Mat3x2_to_vec6_Scalar(fnn);

                q0 = __MATHUTILS__::__s_vec6_multiply(q0, 1.0 / sqrt(I5));

                __MATHUTILS__::Matrix6x6S H;
                __MATHUTILS__::__init_Mat6x6(H, 0);

                H = __MATHUTILS__::__S_Mat6x6_multiply(__MATHUTILS__::__v6_vec6_toMat6x6(q0, q0),
                                                       lambda0);

                __MATHUTILS__::Matrix9x6S PFPxTransPos = __MATHUTILS__::__Transpose6x9(PFPx);
                __MATHUTILS__::Matrix9x9S Hessian = __MATHUTILS__::__M9x6_M6x9_Multiply(
                    __MATHUTILS__::__M9x6_M6x6_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 3, 1);

                H9x9[Hidx] = Hessian;
                D3Index[Hidx] = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            }

        } else {
#ifdef NEWF
            Scalar dis;
            __MATHUTILS__::_d_PT(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            Scalar d_hat_sqrt = sqrt(dHat);
            __MATHUTILS__::Matrix12x9S PFPxT;
            pFpx_pt2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w], d_hat_sqrt, PFPxT);
            Scalar I5 = pow(dis / d_hat_sqrt, 2);
            __MATHUTILS__::Vector9S tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] =
                0;
            tmp.v[8] = dis / d_hat_sqrt;

            __MATHUTILS__::Vector9S q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
            q0.v[8] = 1;

            __MATHUTILS__::Matrix9x9S H;
            //__MATHUTILS__::__init_Mat9x9(H, 0);
#else
            Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            Scalar3 v2 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __MATHUTILS__::Matrix3x3S Ds;
            __MATHUTILS__::__set_Mat_val_column(Ds, v0, v1, v2);

            Scalar3 normal = __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            Scalar dis = __MATHUTILS__::__v_vec_dot(v0, normal);
            // if (abs(dis) > dHat_sqrt) return;
            __MATHUTILS__::Matrix12x9S PDmPx;
            // bool is_flip = false;

            if (dis > 0) {
                // is_flip = true;
                normal = make_Scalar3(-normal.x, -normal.y, -normal.z);
                // pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                // _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx); printf("dHat_sqrt = %f,   dis = %f\n",
                // dHat_sqrt, dis);
            } else {
                dis = -dis;
                // pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                // _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx); printf("dHat_sqrt = %f,   dis = %f\n",
                // dHat_sqrt, dis);
            }

            Scalar3 pos0 = __MATHUTILS__::__add(
                _vertexes[v0I], __MATHUTILS__::__s_vec_multiply(normal, dHat_sqrt - dis));

            Scalar3 u0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], pos0);
            Scalar3 u1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], pos0);
            Scalar3 u2 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], pos0);

            __MATHUTILS__::Matrix3x3S Dm, DmInv;
            __MATHUTILS__::__set_Mat_val_column(Dm, u0, u1, u2);

            __MATHUTILS__::__Inverse(Dm, DmInv);

            __MATHUTILS__::Matrix3x3S F;  //, Ftest;
            __MATHUTILS__::__M_Mat_multiply(Ds, DmInv, F);
            //__MATHUTILS__::__M_Mat_multiply(Dm, DmInv, Ftest);

            Scalar3 FxN = __MATHUTILS__::__M_v_multiply(F, normal);
            Scalar I5 = __MATHUTILS__::__squaredNorm(FxN);

            // printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));

            __MATHUTILS__::Matrix9x12S PFPx = __FEMENERGY__::__computePFDsPX3D_Scalar(DmInv);

            __MATHUTILS__::Matrix3x3S fnn;

            __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(normal, normal);

            __MATHUTILS__::__M_Mat_multiply(F, nn, fnn);

            __MATHUTILS__::Vector9S tmp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(fnn);
#endif
#if (RANK == 1)
            Scalar lambda0 =
                Kappa *
                (2 * dHat * dHat *
                 (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) /
                I5;
            if (dis * dis < gassThreshold * dHat) {
                Scalar lambda1 = Kappa *
                                 (2 * dHat * dHat *
                                  (6 * gassThreshold + 2 * gassThreshold * log(gassThreshold) -
                                   7 * gassThreshold * gassThreshold -
                                   6 * gassThreshold * gassThreshold * log(gassThreshold) + 1)) /
                                 gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 2)
            Scalar lambda0 =
                -(4 * Kappa * dHat * dHat *
                  (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) -
                   2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) /
                I5;
            if (dis * dis < gassThreshold * dHat) {
                Scalar lambda1 =
                    -(4 * Kappa * dHat * dHat *
                      (4 * gassThreshold + log(gassThreshold) -
                       3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) +
                       6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold +
                       gassThreshold * log(gassThreshold) * log(gassThreshold) -
                       7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) /
                    gassThreshold;
                lambda0 = lambda1;
            }
#elif (RANK == 3)
            Scalar lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) *
                 (24 * I5 + 3 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) -
                  12 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 12)) /
                I5;
#elif (RANK == 4)
            Scalar lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) *
                  (24 * I5 + 2 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 12 * I5 * log(I5) -
                   12 * I5 * I5 + I5 * log(I5) * log(I5) - 14 * I5 * I5 * log(I5) - 12)) /
                I5;
#elif (RANK == 5)
            Scalar lambda0 =
                (2 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) *
                 (80 * I5 + 5 * log(I5) - 6 * I5 * I5 * log(I5) * log(I5) + 30 * I5 * log(I5) -
                  40 * I5 * I5 + 2 * I5 * log(I5) * log(I5) - 35 * I5 * I5 * log(I5) - 40)) /
                I5;
#elif (RANK == 6)
            Scalar lambda0 =
                -(4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                  (60 * I5 + 3 * log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 18 * I5 * log(I5) -
                   30 * I5 * I5 + I5 * log(I5) * log(I5) - 21 * I5 * I5 * log(I5) - 30)) /
                I5;
#endif

#if (RANK == 1)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
#elif (RANK == 3)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, -2 *
                         (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                          (3 * I5 + 2 * I5 * log(I5) - 3)) /
                         I5);
#elif (RANK == 4)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                      (2 * I5 + I5 * log(I5) - 2)) /
                         I5);
#elif (RANK == 5)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, -2 *
                         (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (5 * I5 + 2 * I5 * log(I5) - 5)) /
                         I5);
#elif (RANK == 6)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) *
                      (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                         I5);
#endif

#ifdef NEWF
            __MATHUTILS__::Vector12S gradient_vec =
                __MATHUTILS__::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(
                __MATHUTILS__::__Transpose9x12(PFPx), flatten_pk1);
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

#ifdef NEWF

            H = __MATHUTILS__::__S_Mat9x9_multiply(__MATHUTILS__::__v9_vec9_toMat9x9(q0, q0),
                                                   lambda0);

            __MATHUTILS__::Matrix12x12S
                Hessian;  // =
                          // __MATHUTILS__::__M12x9_M9x12_Multiply(__MATHUTILS__::__M12x9_M9x9_Multiply(PFPxT,
                          // H), __MATHUTILS__::__Transpose12x9(PFPxT));
            __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            //__MATHUTILS__::Matrix3x3S Q0;

            //__MATHUTILS__::Matrix3x3S fnn;

            //__MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(normal, normal);

            //__MATHUTILS__::__M_Mat_multiply(F, nn, fnn);

            __MATHUTILS__::Vector9S q0 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(fnn);

            q0 = __MATHUTILS__::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            __MATHUTILS__::Matrix9x9S H = __MATHUTILS__::__S_Mat9x9_multiply(
                __MATHUTILS__::__v9_vec9_toMat9x9(q0, q0), lambda0);

            __MATHUTILS__::Matrix12x9S PFPxTransPos = __MATHUTILS__::__Transpose9x12(PFPx);
            __MATHUTILS__::Matrix12x12S Hessian = __MATHUTILS__::__M12x9_M9x12_Multiply(
                __MATHUTILS__::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    }
}

__global__ void _calSelfCloseVal(const Scalar3* _vertexes, const int4* _collisionPair,
                                 int4* _close_collisionPair, Scalar* _close_collisionVal,
                                 uint32_t* _close_cpNum, Scalar dTol, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _collisionPair[idx];
    Scalar dist2 = _selfConstraintVal(_vertexes, MMCVIDI);
    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx] = dist2;
    }
}

__global__ void _checkSelfCloseVal(const Scalar3* _vertexes, int* _isChange,
                                   int4* _close_collisionPair, Scalar* _close_collisionVal,
                                   int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _close_collisionPair[idx];
    Scalar dist2 = _selfConstraintVal(_vertexes, MMCVIDI);
    if (dist2 < _close_collisionVal[idx]) {
        *_isChange = 1;
    }
}

__global__ void _calFrictionGradient_gd(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                        const Scalar3* _normal,
                                        const uint32_t* _last_collisionPair_gd,
                                        Scalar3* _gradient, int numbers, Scalar dt, Scalar eps2,
                                        Scalar* lastH, Scalar coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar eps = sqrt(eps2);
    Scalar3 normal = *_normal;
    uint32_t gidx = _last_collisionPair_gd[idx];
    Scalar3 Vdiff = __MATHUTILS__::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    Scalar3 VProj = __MATHUTILS__::__minus(
        Vdiff, __MATHUTILS__::__s_vec_multiply(normal, __MATHUTILS__::__v_vec_dot(Vdiff, normal)));
    Scalar VProjMag2 = __MATHUTILS__::__squaredNorm(VProj);
    if (VProjMag2 > eps2) {
        Scalar3 gdf = __MATHUTILS__::__s_vec_multiply(VProj, coef * lastH[idx] / sqrt(VProjMag2));
        /*atomicAdd(&(_gradient[gidx].x), gdf.x);
        atomicAdd(&(_gradient[gidx].y), gdf.y);
        atomicAdd(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = __MATHUTILS__::__add(_gradient[gidx], gdf);
    } else {
        Scalar3 gdf = __MATHUTILS__::__s_vec_multiply(VProj, coef * lastH[idx] / eps);
        /*atomicAdd(&(_gradient[gidx].x), gdf.x);
        atomicAdd(&(_gradient[gidx].y), gdf.y);
        atomicAdd(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = __MATHUTILS__::__add(_gradient[gidx], gdf);
    }
}

__global__ void _calFrictionGradient(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                     const int4* _last_collisionPair, Scalar3* _gradient,
                                     int numbers, Scalar dt, Scalar2* distCoord,
                                     __MATHUTILS__::Matrix3x2S* tanBasis, Scalar eps2,
                                     Scalar* lastH, Scalar coef) {
    Scalar eps = std::sqrt(eps2);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _last_collisionPair[idx];
    Scalar3 relDX3D;
    if (MMCVIDI.x >= 0) {
        __IPCFRICTION__::computeRelDX_EE(
            __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
            distCoord[idx].y, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = __MATHUTILS__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        } else {
            Scalar f1_div_relDXNorm;
            __IPCFRICTION__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __MATHUTILS__::__s_vec_multiply(relDX, f1_div_relDXNorm);
        }
        __MATHUTILS__::Vector12S TTTDX;
        __IPCFRICTION__::liftRelDXTanToMesh_EE(relDX, tanBasis[idx], distCoord[idx].x,
                                               distCoord[idx].y, TTTDX);
        TTTDX = __MATHUTILS__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);
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
    } else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            MMCVIDI.x = v0I;

            __IPCFRICTION__::computeRelDX_PP(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), relDX3D);

            __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
            Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
            Scalar relDXSqNorm = __MATHUTILS__::__squaredNorm(relDX);
            if (relDXSqNorm > eps2) {
                relDX = __MATHUTILS__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            } else {
                Scalar f1_div_relDXNorm;
                __IPCFRICTION__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __MATHUTILS__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }

            __MATHUTILS__::Vector6S TTTDX;
            __IPCFRICTION__::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
            TTTDX = __MATHUTILS__::__s_vec6_multiply(TTTDX, lastH[idx] * coef);
            {
                atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
                atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
                atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
                atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
                atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
                atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            }
        } else if (MMCVIDI.w < 0) {
            MMCVIDI.x = v0I;
            __IPCFRICTION__::computeRelDX_PE(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                distCoord[idx].x, relDX3D);

            __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
            Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
            Scalar relDXSqNorm = __MATHUTILS__::__squaredNorm(relDX);
            if (relDXSqNorm > eps2) {
                relDX = __MATHUTILS__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            } else {
                Scalar f1_div_relDXNorm;
                __IPCFRICTION__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __MATHUTILS__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __MATHUTILS__::Vector9S TTTDX;
            __IPCFRICTION__::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
            TTTDX = __MATHUTILS__::__s_vec9_multiply(TTTDX, lastH[idx] * coef);
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
        } else {
            MMCVIDI.x = v0I;
            __IPCFRICTION__::computeRelDX_PT(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
                distCoord[idx].x, distCoord[idx].y, relDX3D);

            __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
            Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);

            Scalar relDXSqNorm = __MATHUTILS__::__squaredNorm(relDX);
            if (relDXSqNorm > eps2) {
                relDX = __MATHUTILS__::__s_vec_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
            } else {
                Scalar f1_div_relDXNorm;
                __IPCFRICTION__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
                relDX = __MATHUTILS__::__s_vec_multiply(relDX, f1_div_relDXNorm);
            }
            __MATHUTILS__::Vector12S TTTDX;
            __IPCFRICTION__::liftRelDXTanToMesh_PT(relDX, tanBasis[idx], distCoord[idx].x,
                                                   distCoord[idx].y, TTTDX);
            TTTDX = __MATHUTILS__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);

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

__global__ void _calBarrierGradient(const Scalar3* _vertexes, const Scalar3* _rest_vertexes,
                                    const int4* _collisionPair, Scalar3* _gradient,
                                    Scalar dHat, Scalar Kappa, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _collisionPair[idx];
    Scalar dHat_sqrt = sqrt(dHat);
    // Scalar dHat = dHat_sqrt * dHat_sqrt;
    // Scalar Kappa = 1;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
#ifdef NEWF
            Scalar dis;
            __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            Scalar d_hat_sqrt = sqrt(dHat);
            __MATHUTILS__::Matrix12x9S PFPxT;
            pFpx_ee2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w], d_hat_sqrt, PFPxT);
            Scalar I5 = pow(dis / d_hat_sqrt, 2);
            __MATHUTILS__::Vector9S tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] =
                0;
            tmp.v[8] = dis / d_hat_sqrt;
#else

            Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            Scalar3 v2 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            __MATHUTILS__::Matrix3x3S Ds;
            __MATHUTILS__::__set_Mat_val_column(Ds, v0, v1, v2);
            Scalar3 normal = __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(
                v0, __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            Scalar dis = __MATHUTILS__::__v_vec_dot(v1, normal);
            if (dis < 0) {
                normal = make_Scalar3(-normal.x, -normal.y, -normal.z);
                dis = -dis;
            }

            Scalar3 pos2 = __MATHUTILS__::__add(
                _vertexes[MMCVIDI.z], __MATHUTILS__::__s_vec_multiply(normal, dHat_sqrt - dis));
            Scalar3 pos3 = __MATHUTILS__::__add(
                _vertexes[MMCVIDI.w], __MATHUTILS__::__s_vec_multiply(normal, dHat_sqrt - dis));

            Scalar3 u0 = v0;
            Scalar3 u1 = __MATHUTILS__::__minus(pos2, _vertexes[MMCVIDI.x]);
            Scalar3 u2 = __MATHUTILS__::__minus(pos3, _vertexes[MMCVIDI.x]);

            __MATHUTILS__::Matrix3x3S Dm, DmInv;
            __MATHUTILS__::__set_Mat_val_column(Dm, u0, u1, u2);

            __MATHUTILS__::__Inverse(Dm, DmInv);

            __MATHUTILS__::Matrix3x3S F;
            __MATHUTILS__::__M_Mat_multiply(Ds, DmInv, F);

            Scalar3 FxN = __MATHUTILS__::__M_v_multiply(F, normal);
            Scalar I5 = __MATHUTILS__::__squaredNorm(FxN);

            __MATHUTILS__::Matrix9x12S PFPx = __computePFDsPX3D_Scalar(DmInv);

            __MATHUTILS__::Matrix3x3S fnn;

            __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(normal, normal);

            __MATHUTILS__::__M_Mat_multiply(F, nn, fnn);

            __MATHUTILS__::Vector9S tmp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(fnn);

#endif

#if (RANK == 1)
            Scalar judge = (2 * dHat * dHat *
                            (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) /
                           I5;
            Scalar judge2 =
                2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 * dis / d_hat_sqrt;
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            // if (dis*dis<1e-2*dHat)
            // flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 -
            // 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);
#elif (RANK == 2)
            //__MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, 2 * (2 *
            // Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            Scalar judge =
                -(4 * dHat * dHat *
                  (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) -
                   2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) /
                I5;
            Scalar judge2 = 2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                            I5 * dis / dHat_sqrt;
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            // if (dis*dis<1e-2*dHat)
            // flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat *
            // log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);

#elif (RANK == 3)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, -2 *
                         (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                          (3 * I5 + 2 * I5 * log(I5) - 3)) /
                         I5);
#elif (RANK == 4)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                      (2 * I5 + I5 * log(I5) - 2)) /
                         I5);
#elif (RANK == 5)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, -2 *
                         (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (5 * I5 + 2 * I5 * log(I5) - 5)) /
                         I5);
#elif (RANK == 6)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) *
                      (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                         I5);
#endif

#ifdef NEWF
            __MATHUTILS__::Vector12S gradient_vec =
                __MATHUTILS__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
#else

            __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(
                __MATHUTILS__::__Transpose9x12(PFPx), flatten_pk1);
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

        } else {
            // return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            Scalar c = __MATHUTILS__::__norm(
                __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
            Scalar I1 = c * c;
            if (I1 == 0) return;
            Scalar dis;
            __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            Scalar I2 = dis / dHat;
            dis = sqrt(dis);

            __MATHUTILS__::Matrix3x3S F;
            __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            Scalar3 n1 = make_Scalar3(0, 1, 0);
            Scalar3 n2 = make_Scalar3(0, 0, 1);

            Scalar eps_x =
                __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y],
                                            _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);

            __MATHUTILS__::Matrix3x3S g1, g2;

            __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
            __MATHUTILS__::__M_Mat_multiply(F, nn, g1);
            nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
            __MATHUTILS__::__M_Mat_multiply(F, nn, g2);

            __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
            __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

            __MATHUTILS__::Matrix12x9S PFPx;
            pFpx_pee(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
            Scalar p1 = Kappa * 2 *
                        (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                        (eps_x * eps_x);
            Scalar p2 =
                Kappa * 2 *
                (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                (I2 * eps_x * eps_x);
#elif (RANK == 2)
            Scalar p1 = -Kappa * 2 *
                        (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                        (eps_x * eps_x);
            Scalar p2 = -Kappa * 2 *
                        (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) *
                         (I2 + I2 * log(I2) - 1)) /
                        (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            Scalar p1 = -Kappa * 2 *
                        (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                        (eps_x * eps_x);
            Scalar p2 = -Kappa * 2 *
                        (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) *
                         (2 * I2 + I2 * log(I2) - 2)) /
                        (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            Scalar p1 = -Kappa * 2 *
                        (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                        (eps_x * eps_x);
            Scalar p2 = -Kappa * 2 *
                        (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) *
                         (3 * I2 + I2 * log(I2) - 3)) /
                        (I2 * (eps_x * eps_x));
#endif

            __MATHUTILS__::Vector9S flatten_pk1 =
                __MATHUTILS__::__add9(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                      __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
            __MATHUTILS__::Vector12S gradient_vec =
                __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

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
    } else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                Scalar c = __MATHUTILS__::__norm(
                    __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
                Scalar I1 = c * c;
                if (I1 == 0) return;
                Scalar dis;
                __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                Scalar I2 = dis / dHat;
                dis = sqrt(dis);

                __MATHUTILS__::Matrix3x3S F;
                __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                Scalar3 n1 = make_Scalar3(0, 1, 0);
                Scalar3 n2 = make_Scalar3(0, 0, 1);

                Scalar eps_x = __MATHUTILS__::_compute_epx(
                    _rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.y],
                    _rest_vertexes[MMCVIDI.w]);

                __MATHUTILS__::Matrix3x3S g1, g2;

                __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
                __MATHUTILS__::__M_Mat_multiply(F, nn, g1);
                nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
                __MATHUTILS__::__M_Mat_multiply(F, nn, g2);

                __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
                __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

                __MATHUTILS__::Matrix12x9S PFPx;
                pFpx_ppp(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);
#if (RANK == 1)
                Scalar p1 = Kappa * 2 *
                            (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                            (eps_x * eps_x);
                Scalar p2 =
                    Kappa * 2 *
                    (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                    (I2 * eps_x * eps_x);
#elif (RANK == 2)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (I2 + I2 * log(I2) - 1)) /
                            (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (2 * I2 + I2 * log(I2) - 2)) /
                            (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (3 * I2 + I2 * log(I2) - 3)) /
                            (I2 * (eps_x * eps_x));
#endif
                __MATHUTILS__::Vector9S flatten_pk1 =
                    __MATHUTILS__::__add9(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                          __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
                __MATHUTILS__::Vector12S gradient_vec =
                    __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

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
            } else {
#ifdef NEWF
                Scalar dis;
                __MATHUTILS__::_d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis = sqrt(dis);
                Scalar d_hat_sqrt = sqrt(dHat);
                __MATHUTILS__::Vector6S PFPxT;
                pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt, PFPxT);
                Scalar I5 = pow(dis / d_hat_sqrt, 2);
                Scalar fnn = dis / d_hat_sqrt;

#if (RANK == 1)

                Scalar judge =
                    (2 * dHat * dHat *
                     (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) /
                    I5;
                Scalar judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 *
                                dis / d_hat_sqrt;
                Scalar flatten_pk1 =
                    fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5;
                // if (dis*dis<1e-2*dHat)
                // flatten_pk1 = fnn * 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5)
                // - 1)) / I5 / (I5) /*/ (I5) / (I5)*/;
#elif (RANK == 2)
                // Scalar flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) *
                // (I5 + I5 * log(I5) - 1)) / I5;

                Scalar judge =
                    -(4 * dHat * dHat *
                      (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) -
                       2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) /
                    I5;
                Scalar judge2 = 2 *
                                (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                                I5 * dis / dHat_sqrt;
                Scalar flatten_pk1 =
                    fnn * 2 *
                    (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
                // if (dis*dis<1e-2*dHat)
                // flatten_pk1 = fnn * 2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5
                // * log(I5) - 1)) / I5/I5;

#elif (RANK == 3)
                Scalar flatten_pk1 = fnn * -2 *
                                     (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                                      (3 * I5 + 2 * I5 * log(I5) - 3)) /
                                     I5;
#elif (RANK == 4)
                Scalar flatten_pk1 = fnn *
                                     (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) *
                                      (I5 - 1) * (2 * I5 + I5 * log(I5) - 2)) /
                                     I5;
#elif (RANK == 5)
                Scalar flatten_pk1 = fnn * -2 *
                                     (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                                      (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) /
                                     I5;
#elif (RANK == 6)
                Scalar flatten_pk1 = fnn *
                                     (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) *
                                      log(I5) * log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                                     I5;
#endif

                __MATHUTILS__::Vector6S gradient_vec =
                    __MATHUTILS__::__s_vec6_multiply(PFPxT, flatten_pk1);

#else
                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                Scalar3 Ds = v0;
                Scalar dis = __MATHUTILS__::__norm(v0);
                // if (dis > dHat_sqrt) return;
                Scalar3 vec_normal = __MATHUTILS__::__normalized(make_Scalar3(-v0.x, -v0.y, -v0.z));
                Scalar3 target = make_Scalar3(0, 1, 0);
                Scalar3 vec = __MATHUTILS__::__v_vec_cross(vec_normal, target);
                Scalar cos = __MATHUTILS__::__v_vec_dot(vec_normal, target);
                __MATHUTILS__::Matrix3x3S rotation;
                __MATHUTILS__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
                __MATHUTILS__::Vector6S PDmPx;
                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                } else {
                    __MATHUTILS__::Matrix3x3S cross_vec;
                    __MATHUTILS__::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x,
                                                 -vec.y, vec.x, 0);

                    rotation = __MATHUTILS__::__Mat_add(
                        rotation,
                        __MATHUTILS__::__Mat_add(
                            cross_vec, __MATHUTILS__::__S_Mat_multiply(
                                           __MATHUTILS__::__M_Mat_multiply(cross_vec, cross_vec),
                                           1.0 / (1 + cos))));
                }

                Scalar3 pos0 = __MATHUTILS__::__add(
                    _vertexes[v0I], __MATHUTILS__::__s_vec_multiply(vec_normal, dHat_sqrt - dis));
                Scalar3 rotate_uv0 = __MATHUTILS__::__M_v_multiply(rotation, pos0);
                Scalar3 rotate_uv1 = __MATHUTILS__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);

                Scalar uv0 = rotate_uv0.y;
                Scalar uv1 = rotate_uv1.y;

                Scalar u0 = uv1 - uv0;
                Scalar Dm = u0;  // PFPx
                Scalar DmInv = 1 / u0;

                Scalar3 F = __MATHUTILS__::__s_vec_multiply(Ds, DmInv);
                Scalar I5 = __MATHUTILS__::__squaredNorm(F);

                Scalar3 tmp = F;

#if (RANK == 1)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                        I5);
#elif (RANK == 3)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                              (3 * I5 + 2 * I5 * log(I5) - 3)) /
                             I5);
#elif (RANK == 4)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (2 * I5 + I5 * log(I5) - 2)) /
                             I5);
#elif (RANK == 5)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                              (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) /
                             I5);
#elif (RANK == 6)
                Scalar3 flatten_pk1 = __MATHUTILS__::__s_vec_multiply(
                    tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                          log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                             I5);
#endif
                __MATHUTILS__::Matrix3x6S PFPx = __computePFDsPX3D_3x6_Scalar(DmInv);

                __MATHUTILS__::Vector6S gradient_vec = __MATHUTILS__::__M6x3_v3_multiply(
                    __MATHUTILS__::__Transpose3x6(PFPx), flatten_pk1);
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

        } else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.x = v0I;
                MMCVIDI.w = -MMCVIDI.w - 1;
                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                Scalar c = __MATHUTILS__::__norm(
                    __MATHUTILS__::__v_vec_cross(v0, v1)) /*/ __MATHUTILS__::__norm(v0)*/;
                Scalar I1 = c * c;
                if (I1 == 0) return;
                Scalar dis;
                __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                     _vertexes[MMCVIDI.z], dis);
                Scalar I2 = dis / dHat;
                dis = sqrt(dis);

                __MATHUTILS__::Matrix3x3S F;
                __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                Scalar3 n1 = make_Scalar3(0, 1, 0);
                Scalar3 n2 = make_Scalar3(0, 0, 1);

                Scalar eps_x = __MATHUTILS__::_compute_epx(
                    _rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w], _rest_vertexes[MMCVIDI.y],
                    _rest_vertexes[MMCVIDI.z]);

                __MATHUTILS__::Matrix3x3S g1, g2;

                __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
                __MATHUTILS__::__M_Mat_multiply(F, nn, g1);
                nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
                __MATHUTILS__::__M_Mat_multiply(F, nn, g2);

                __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
                __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

                __MATHUTILS__::Matrix12x9S PFPx;
                pFpx_ppe(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                         _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

#if (RANK == 1)
                Scalar p1 = Kappa * 2 *
                            (2 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                            (eps_x * eps_x);
                Scalar p2 =
                    Kappa * 2 *
                    (I1 * dHat * dHat * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) /
                    (I2 * eps_x * eps_x);
#elif (RANK == 2)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (I2 + I2 * log(I2) - 1)) /
                            (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * pow(log(I2), 4) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * pow(log(I2), 3) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (2 * I2 + I2 * log(I2) - 2)) /
                            (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                Scalar p1 =
                    -Kappa * 2 *
                    (2 * dHat * dHat * pow(log(I2), 6) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) /
                    (eps_x * eps_x);
                Scalar p2 = -Kappa * 2 *
                            (2 * I1 * dHat * dHat * pow(log(I2), 5) * (I1 - 2 * eps_x) * (I2 - 1) *
                             (3 * I2 + I2 * log(I2) - 3)) /
                            (I2 * (eps_x * eps_x));
#endif
                __MATHUTILS__::Vector9S flatten_pk1 =
                    __MATHUTILS__::__add9(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                          __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
                __MATHUTILS__::Vector12S gradient_vec =
                    __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

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
            } else {
#ifdef NEWF
                Scalar dis;
                __MATHUTILS__::_d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                     dis);
                dis = sqrt(dis);
                Scalar d_hat_sqrt = sqrt(dHat);
                __MATHUTILS__::Matrix9x4S PFPxT;
                pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt,
                         PFPxT);
                Scalar I5 = pow(dis / d_hat_sqrt, 2);
                __MATHUTILS__::Vector4S fnn;
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;  // = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //__MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(fnn, 2 *
                // Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);

#if (RANK == 1)

                Scalar judge =
                    (2 * dHat * dHat *
                     (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) /
                    I5;
                Scalar judge2 = 2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 *
                                dis / d_hat_sqrt;
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                // if (dis*dis<1e-2*dHat)
                // flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat *
                // (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
                //__MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(fnn, 2 * (2
                //* Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

                Scalar judge =
                    -(4 * dHat * dHat *
                      (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) -
                       2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) /
                    I5;
                Scalar judge2 = 2 *
                                (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                                I5 * dis / dHat_sqrt;
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                        I5);
                // if (dis*dis<1e-2*dHat)
                // flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(fnn, 2 * (2 * Kappa * dHat * dHat
                // * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                              (3 * I5 + 2 * I5 * log(I5) - 3)) /
                             I5);
#elif (RANK == 4)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (2 * I5 + I5 * log(I5) - 2)) /
                             I5);
#elif (RANK == 5)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                              (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) /
                             I5);
#elif (RANK == 6)
                __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(
                    fnn, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                          log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                             I5);
#endif

                __MATHUTILS__::Vector9S gradient_vec =
                    __MATHUTILS__::__M9x4_v4_multiply(PFPxT, flatten_pk1);
#else

                Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
                Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);

                __MATHUTILS__::Matrix3x2S Ds;
                __MATHUTILS__::__set_Mat3x2_val_column(Ds, v0, v1);

                Scalar3 triangle_normal =
                    __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(v0, v1));
                Scalar3 target = make_Scalar3(0, 1, 0);

                Scalar3 vec = __MATHUTILS__::__v_vec_cross(triangle_normal, target);
                Scalar cos = __MATHUTILS__::__v_vec_dot(triangle_normal, target);

                Scalar3 edge_normal = __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(
                    __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z]),
                    triangle_normal));
                Scalar dis = __MATHUTILS__::__v_vec_dot(
                    __MATHUTILS__::__minus(_vertexes[v0I], _vertexes[MMCVIDI.y]), edge_normal);

                __MATHUTILS__::Matrix3x3S rotation;
                __MATHUTILS__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                __MATHUTILS__::Matrix9x4S PDmPx;

                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                } else {
                    __MATHUTILS__::Matrix3x3S cross_vec;
                    __MATHUTILS__::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x,
                                                 -vec.y, vec.x, 0);

                    rotation = __MATHUTILS__::__Mat_add(
                        rotation,
                        __MATHUTILS__::__Mat_add(
                            cross_vec, __MATHUTILS__::__S_Mat_multiply(
                                           __MATHUTILS__::__M_Mat_multiply(cross_vec, cross_vec),
                                           1.0 / (1 + cos))));
                }

                Scalar3 pos0 = __MATHUTILS__::__add(
                    _vertexes[v0I], __MATHUTILS__::__s_vec_multiply(edge_normal, dHat_sqrt - dis));

                Scalar3 rotate_uv0 = __MATHUTILS__::__M_v_multiply(rotation, pos0);
                Scalar3 rotate_uv1 = __MATHUTILS__::__M_v_multiply(rotation, _vertexes[MMCVIDI.y]);
                Scalar3 rotate_uv2 = __MATHUTILS__::__M_v_multiply(rotation, _vertexes[MMCVIDI.z]);
                Scalar3 rotate_normal = __MATHUTILS__::__M_v_multiply(rotation, edge_normal);

                Scalar2 uv0 = make_Scalar2(rotate_uv0.x, rotate_uv0.z);
                Scalar2 uv1 = make_Scalar2(rotate_uv1.x, rotate_uv1.z);
                Scalar2 uv2 = make_Scalar2(rotate_uv2.x, rotate_uv2.z);
                Scalar2 normal = make_Scalar2(rotate_normal.x, rotate_normal.z);

                Scalar2 u0 = __MATHUTILS__::__minus_v2(uv1, uv0);
                Scalar2 u1 = __MATHUTILS__::__minus_v2(uv2, uv0);

                __MATHUTILS__::Matrix2x2S Dm;

                __MATHUTILS__::__set_Mat2x2_val_column(Dm, u0, u1);

                __MATHUTILS__::Matrix2x2S DmInv;
                __MATHUTILS__::__Inverse2x2(Dm, DmInv);

                __MATHUTILS__::Matrix3x2S F = __MATHUTILS__::__M3x2_M2x2_Multiply(Ds, DmInv);

                Scalar3 FxN = __MATHUTILS__::__M3x2_v2_multiply(F, normal);
                Scalar I5 = __MATHUTILS__::__squaredNorm(FxN);

                __MATHUTILS__::Matrix3x2S fnn;

                __MATHUTILS__::Matrix2x2S nn = __MATHUTILS__::__v2_vec2_toMat2x2(normal, normal);

                fnn = __MATHUTILS__::__M3x2_M2x2_Multiply(F, nn);

                __MATHUTILS__::Vector6S tmp = __MATHUTILS__::__Mat3x2_to_vec6_Scalar(fnn);

#if (RANK == 1)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
#elif (RANK == 2)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp,
                    2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                        I5);
#elif (RANK == 3)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                              (3 * I5 + 2 * I5 * log(I5) - 3)) /
                             I5);
#elif (RANK == 4)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (2 * I5 + I5 * log(I5) - 2)) /
                             I5);
#elif (RANK == 5)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, -2 *
                             (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                              (I5 - 1) * (5 * I5 + 2 * I5 * log(I5) - 5)) /
                             I5);
#elif (RANK == 6)
                __MATHUTILS__::Vector6S flatten_pk1 = __MATHUTILS__::__s_vec6_multiply(
                    tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) *
                          log(I5) * (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                             I5);
#endif

                __MATHUTILS__::Matrix6x9S PFPx = __computePFDsPX3D_6x9_Scalar(DmInv);

                __MATHUTILS__::Vector9S gradient_vec = __MATHUTILS__::__M9x6_v6_multiply(
                    __MATHUTILS__::__Transpose6x9(PFPx), flatten_pk1);
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

        } else {
#ifdef NEWF
            Scalar dis;
            __MATHUTILS__::_d_PT(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            Scalar d_hat_sqrt = sqrt(dHat);
            __MATHUTILS__::Matrix12x9S PFPxT;
            pFpx_pt2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                     _vertexes[MMCVIDI.w], d_hat_sqrt, PFPxT);
            Scalar I5 = pow(dis / d_hat_sqrt, 2);
            __MATHUTILS__::Vector9S tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] =
                0;
            tmp.v[8] = dis / d_hat_sqrt;
#else
            Scalar3 v0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            Scalar3 v1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            Scalar3 v2 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            __MATHUTILS__::Matrix3x3S Ds;
            __MATHUTILS__::__set_Mat_val_column(Ds, v0, v1, v2);

            Scalar3 normal = __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]),
                __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            Scalar dis = __MATHUTILS__::__v_vec_dot(v0, normal);
            // if (abs(dis) > dHat_sqrt) return;
            __MATHUTILS__::Matrix12x9S PDmPx;
            // bool is_flip = false;

            if (dis > 0) {
                // is_flip = true;
                normal = make_Scalar3(-normal.x, -normal.y, -normal.z);
                // pDmpx_pt_flip(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                // _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx); printf("dHat_sqrt = %f,   dis = %f\n",
                // dHat_sqrt, dis);
            } else {
                dis = -dis;
                // pDmpx_pt(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                // _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx); printf("dHat_sqrt = %f,   dis = %f\n",
                // dHat_sqrt, dis);
            }

            Scalar3 pos0 = __MATHUTILS__::__add(
                _vertexes[v0I], __MATHUTILS__::__s_vec_multiply(normal, dHat_sqrt - dis));

            Scalar3 u0 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.y], pos0);
            Scalar3 u1 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.z], pos0);
            Scalar3 u2 = __MATHUTILS__::__minus(_vertexes[MMCVIDI.w], pos0);

            __MATHUTILS__::Matrix3x3S Dm, DmInv;
            __MATHUTILS__::__set_Mat_val_column(Dm, u0, u1, u2);

            __MATHUTILS__::__Inverse(Dm, DmInv);

            __MATHUTILS__::Matrix3x3S F;  //, Ftest;
            __MATHUTILS__::__M_Mat_multiply(Ds, DmInv, F);
            //__MATHUTILS__::__M_Mat_multiply(Dm, DmInv, Ftest);

            Scalar3 FxN = __MATHUTILS__::__M_v_multiply(F, normal);
            Scalar I5 = __MATHUTILS__::__squaredNorm(FxN);

            // printf("I5 = %f,   dist/dHat_sqrt = %f\n", I5, (dis / dHat_sqrt)* (dis / dHat_sqrt));

            __MATHUTILS__::Matrix9x12S PFPx = __computePFDsPX3D_Scalar(DmInv);

            __MATHUTILS__::Matrix3x3S fnn;

            __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(normal, normal);

            __MATHUTILS__::__M_Mat_multiply(F, nn, fnn);

            __MATHUTILS__::Vector9S tmp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(fnn);
#endif

#if (RANK == 1)

            Scalar judge = (2 * dHat * dHat *
                            (6 * I5 + 2 * I5 * log(I5) - 7 * I5 * I5 - 6 * I5 * I5 * log(I5) + 1)) /
                           I5;
            Scalar judge2 =
                2 * (dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 * dis / d_hat_sqrt;
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
            // if (dis*dis<1e-2*dHat)
            // flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, 2 * Kappa * -(dHat * dHat * (I5 -
            // 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5 / (I5) /*/ (I5) / (I5)*/);

#elif (RANK == 2)
            //__MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, 2 * (2 *
            // Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);

            Scalar judge =
                -(4 * dHat * dHat *
                  (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) -
                   2 * I5 * I5 + I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) /
                I5;
            Scalar judge2 = 2 * (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) /
                            I5 * dis / dHat_sqrt;
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp,
                2 * (2 * Kappa * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5);
            // if (dis*dis<1e-2*dHat)
            // flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, 2 * (2 * Kappa * dHat * dHat *
            // log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5/I5);
#elif (RANK == 3)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, -2 *
                         (Kappa * dHat * dHat * log(I5) * log(I5) * (I5 - 1) *
                          (3 * I5 + 2 * I5 * log(I5) - 3)) /
                         I5);
#elif (RANK == 4)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                      (2 * I5 + I5 * log(I5) - 2)) /
                         I5);
#elif (RANK == 5)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, -2 *
                         (Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * (I5 - 1) *
                          (5 * I5 + 2 * I5 * log(I5) - 5)) /
                         I5);
#elif (RANK == 6)
            __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(
                tmp, (4 * Kappa * dHat * dHat * log(I5) * log(I5) * log(I5) * log(I5) * log(I5) *
                      (I5 - 1) * (3 * I5 + I5 * log(I5) - 3)) /
                         I5);
#endif

#ifdef NEWF
            __MATHUTILS__::Vector12S gradient_vec =
                __MATHUTILS__::__M12x9_v9_multiply(PFPxT, flatten_pk1);
#else
            __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(
                __MATHUTILS__::__Transpose9x12(PFPx), flatten_pk1);
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

__global__ void _GroundCollisionDetect(const Scalar3* vertexes, const uint32_t* surfVertIds,
                                       const Scalar* g_offset, const Scalar3* g_normal,
                                       uint32_t* _environment_collisionPair, uint32_t* _gpNum,
                                       Scalar dHat, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar dist = __MATHUTILS__::__v_vec_dot(*g_normal, vertexes[surfVertIds[idx]]) - *g_offset;
    if (dist * dist > dHat) return;

    _environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeGroundCloseVal(const Scalar3* vertexes, const Scalar* g_offset,
                                       const Scalar3* g_normal,
                                       const uint32_t* _environment_collisionPair, Scalar dTol,
                                       uint32_t* _closeConstraintID, Scalar* _closeConstraintVal,
                                       uint32_t* _close_gpNum, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;

    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx] = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

__global__ void _checkGroundCloseVal(const Scalar3* vertexes, const Scalar* g_offset,
                                     const Scalar3* g_normal, int* _isChange,
                                     uint32_t* _closeConstraintID, Scalar* _closeConstraintVal,
                                     int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _closeConstraintID[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;

    if (dist2 < _closeConstraintVal[gidx]) {
        *_isChange = 1;
    }
}

__global__ void _computeSelfCloseVal(const Scalar3* vertexes, const Scalar* g_offset,
                                     const Scalar3* g_normal,
                                     const uint32_t* _environment_collisionPair, Scalar dTol,
                                     uint32_t* _closeConstraintID, Scalar* _closeConstraintVal,
                                     uint32_t* _close_gpNum, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;

    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx] = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

__global__ void _checkGroundIntersection(const Scalar3* vertexes, const Scalar* g_offset,
                                         const Scalar3* g_normal,
                                         const uint32_t* _environment_collisionPair,
                                         int* _isIntersect, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    // printf("%f  %f\n", *g_offset, dist);
    if (dist < 0) *_isIntersect = -1;
}

__global__ void _getFrictionEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                const Scalar3* o_vertexes,
                                                const int4* _collisionPair, int cpNum, Scalar dt,
                                                const Scalar2* distCoord,
                                                const __MATHUTILS__::Matrix3x2S* tanBasis,
                                                const Scalar* lastH, Scalar fricDHat, Scalar eps

) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];
    int numbers = cpNum;
    if (idx >= numbers) return;

    Scalar temp = __cal_Friction_energy(vertexes, o_vertexes, _collisionPair[idx], dt,
                                        distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
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
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getFrictionEnergy_gd_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                   const Scalar3* o_vertexes,
                                                   const Scalar3* _normal,
                                                   const uint32_t* _collisionPair_gd, int gpNum,
                                                   Scalar dt, const Scalar* lastH, Scalar eps

) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];
    int numbers = gpNum;
    if (idx >= numbers) return;

    Scalar temp = __cal_Friction_gd_energy(vertexes, o_vertexes, _normal, _collisionPair_gd[idx],
                                           dt, lastH[idx], eps);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
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
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _reduct_min_groundTimeStep_to_Scalar(const Scalar3* vertexes,
                                                     const uint32_t* surfVertIds,
                                                     const Scalar* g_offset,
                                                     const Scalar3* g_normal,
                                                     const Scalar3* moveDir, Scalar* minStepSizes,
                                                     Scalar slackness, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;
    int svI = surfVertIds[idx];
    Scalar temp = 1.0;
    Scalar3 normal = *g_normal;
    Scalar coef = __MATHUTILS__::__v_vec_dot(normal, moveDir[svI]);
    if (coef > 0.0) {
        Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[svI]) - *g_offset;  // normal
        temp = coef / (dist * slackness);
        // printf("%f\n", temp);
    }
    /*if (blockIdx.x == 4) {
        printf("%f\n", temp);
    }
    __syncthreads();*/
    // printf("%f\n", temp);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
        // printf("warpNum %d\n", warpNum);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_max(temp, tempMin);
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
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = temp;
        // printf("%f   %d\n", temp, blockIdx.x);
    }
}

__global__ void _calKineticGradient(Scalar3* vertexes, Scalar3* xTilta, Scalar3* gradient,
                                    Scalar* masses, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return; // numbers of vertices
    Scalar3 deltaX = __MATHUTILS__::__minus(vertexes[idx], xTilta[idx]);
    gradient[idx] =
        make_Scalar3(deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    // printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__ void _reduct_min_selfTimeStep_to_Scalar(const Scalar3* vertexes,
                                                   const int4* _ccd_collisionPairs,
                                                   const Scalar3* moveDir, Scalar* minStepSizes,
                                                   Scalar slackness, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;
    Scalar temp = 1.0;
    Scalar CCDDistRatio = 1.0 - slackness;

    int4 MMCVIDI = _ccd_collisionPairs[idx];

    if (MMCVIDI.x < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;

        Scalar temp1 = __ACCD__::point_triangle_ccd(
            vertexes[MMCVIDI.x], vertexes[MMCVIDI.y], vertexes[MMCVIDI.z], vertexes[MMCVIDI.w],
            __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
            __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
            __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
            __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0);

        // Scalar temp2 = doCCDVF(vertexes[MMCVIDI.x],
        //     vertexes[MMCVIDI.y],
        //     vertexes[MMCVIDI.z],
        //     vertexes[MMCVIDI.w],
        //     __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
        //     __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
        //     __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
        //     __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), 1e-9, 0.2);

        temp = 1.0 / temp1;
    } else {
        temp =
            1.0 / __ACCD__::edge_edge_ccd(
                      vertexes[MMCVIDI.x], vertexes[MMCVIDI.y], vertexes[MMCVIDI.z],
                      vertexes[MMCVIDI.w], __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
                      __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
                      __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
                      __MATHUTILS__::__s_vec_multiply(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0);
    }

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_max(temp, tempMin);
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
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = temp;
    }
}

__global__ void _reduct_max_cfl_to_Scalar(const Scalar3* moveDir, Scalar* max_Scalar_val,
                                          uint32_t* mSVI, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;

    Scalar temp = __MATHUTILS__::__norm(moveDir[mSVI[idx]]);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_max(temp, tempMax);
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
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_max(temp, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        max_Scalar_val[blockIdx.x] = temp;
    }
}

__global__ void _reduct_Scalar3Sqn_to_Scalar(const Scalar3* A, Scalar* D, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;

    Scalar temp = __MATHUTILS__::__squaredNorm(A[idx]);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        // Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
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

__global__ void _reduct_Scalar3Dot_to_Scalar(const Scalar3* A, const Scalar3* B, Scalar* D,
                                             int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;

    Scalar temp = __MATHUTILS__::__v_vec_dot(A[idx], B[idx]);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        // Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
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

__global__ void _getBarrierEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                               const Scalar3* rest_vertexes, int4* _collisionPair,
                                               Scalar _Kappa, Scalar _dHat, int cpNum) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];
    int numbers = cpNum;
    if (idx >= numbers) return;

    Scalar temp = __cal_Barrier_energy(vertexes, rest_vertexes, _collisionPair[idx], _Kappa, _dHat);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
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
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _computeGroundEnergy_Reduction(Scalar* squeue, const Scalar3* vertexes,
                                               const Scalar* g_offset, const Scalar3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               Scalar dHat, Scalar Kappa, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;

    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;
    Scalar temp = -(dist2 - dHat) * (dist2 - dHat) * log(dist2 / dHat);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
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
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _calFrictionLastH_gd(const Scalar3* _vertexes, const Scalar* g_offset,
                                     const Scalar3* g_normal,
                                     const uint32_t* _collisionPair_environment,
                                     Scalar* lambda_lastH_gd, uint32_t* _collisionPair_last_gd,
                                     Scalar dHat, Scalar Kappa, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    Scalar3 normal = *g_normal;
    int gidx = _collisionPair_environment[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, _vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;

    Scalar t = dist2 - dHat;
    Scalar g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    lambda_lastH_gd[idx] = -Kappa * 2.0 * sqrt(dist2) * g_b;
    _collisionPair_last_gd[idx] = gidx;
}

__global__ void _calFrictionLastH_DistAndTan(const Scalar3* _vertexes,
                                             const int4* _collisionPair, Scalar* lambda_lastH,
                                             Scalar2* distCoord,
                                             __MATHUTILS__::Matrix3x2S* tanBasis,
                                             int4* _collisionPair_last, Scalar dHat, Scalar Kappa,
                                             uint32_t* _cpNum_last, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _collisionPair[idx];
    Scalar dis;
    int last_index = -1;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            __IPCFRICTION__::computeClosestPoint_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                    _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                    distCoord[last_index]);
            __IPCFRICTION__::computeTangentBasis_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                    _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                    tanBasis[last_index]);
        }
    } else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y >= 0) {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 2, 1);
                __MATHUTILS__::_d_PP(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                distCoord[last_index].x = 0;
                distCoord[last_index].y = 0;
                __IPCFRICTION__::computeTangentBasis_PP(_vertexes[v0I], _vertexes[MMCVIDI.y],
                                                        tanBasis[last_index]);
            }

        } else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y >= 0) {
                last_index = atomicAdd(_cpNum_last, 1);
                atomicAdd(_cpNum_last + 3, 1);
                __MATHUTILS__::_d_PE(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                     dis);
                __IPCFRICTION__::computeClosestPoint_PE(_vertexes[v0I], _vertexes[MMCVIDI.y],
                                                        _vertexes[MMCVIDI.z],
                                                        distCoord[last_index].x);
                distCoord[last_index].y = 0;
                __IPCFRICTION__::computeTangentBasis_PE(_vertexes[v0I], _vertexes[MMCVIDI.y],
                                                        _vertexes[MMCVIDI.z], tanBasis[last_index]);
            }
        } else {
            last_index = atomicAdd(_cpNum_last, 1);
            atomicAdd(_cpNum_last + 4, 1);
            __MATHUTILS__::_d_PT(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                 _vertexes[MMCVIDI.w], dis);
            __IPCFRICTION__::computeClosestPoint_PT(_vertexes[v0I], _vertexes[MMCVIDI.y],
                                                    _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                    distCoord[last_index]);
            __IPCFRICTION__::computeTangentBasis_PT(_vertexes[v0I], _vertexes[MMCVIDI.y],
                                                    _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                    tanBasis[last_index]);
        }
    }
    if (last_index >= 0) {
#if (RANK == 1)
        Scalar t = dis - dHat;
        lambda_lastH[last_index] =
            -Kappa * 2.0 * sqrt(dis) * (t * log(dis / dHat) * -2.0 - (t * t) / dis);
#elif (RANK == 2)
        lambda_lastH[last_index] = -Kappa * 2.0 * sqrt(dis) *
                                   (log(dis / dHat) * log(dis / dHat) * (2 * dis - 2 * dHat) +
                                    (2 * log(dis / dHat) * (dis - dHat) * (dis - dHat)) / dis);
#endif
        _collisionPair_last[last_index] = _collisionPair[idx];
    }
}

void buildFrictionSets(std::unique_ptr<GeometryManager>& instance) {
    CUDA_SAFE_CALL(cudaMemset(instance->getCudaCPNum(), 0, 5 * sizeof(uint32_t)));
    int numbers = instance->getHostCpNum(0);
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calFrictionLastH_DistAndTan<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaCollisionPairs(),
        instance->getCudaLambdaLastHScalar(), instance->getCudaDistCoord(),
        instance->getCudaTanBasis(), instance->getCudaCollisionPairsLastH(),
        instance->getHostDHat(), instance->getHostKappa(), instance->getCudaCPNum(),
        instance->getHostCpNum(0));
    CUDA_SAFE_CALL(cudaMemcpy(&instance->getHostCpNumLast(0), instance->getCudaCPNum(),
                              5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    numbers = instance->getHostGpNum();
    blockNum = (numbers + threadNum - 1) / threadNum;
    _calFrictionLastH_gd<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(),
        instance->getCudaLambdaLastHScalarGd(), instance->getCudaCollisionPairsLastHGd(),
        instance->getHostDHat(), instance->getHostKappa(), instance->getHostGpNum());
    instance->getHostGpNumLast() = instance->getHostGpNum();
}

void GroundCollisionDetect(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostNumSurfVerts();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _GroundCollisionDetect<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaSurfVert(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(),
        instance->getCudaGPNum(), instance->getHostDHat(), numbers);
}

void computeCloseGroundVal(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostGpNum();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundCloseVal<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(),
        instance->getHostDTol(), instance->getCudaCloseConstraintID(),
        instance->getCudaCloseConstraintVal(), instance->getCudaCloseGPNum(), numbers);
}

bool checkCloseGroundVal(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostCloseGpNum();
    if (numbers < 1) return false;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    int* _isChange;
    CUDAMallocSafe(_isChange, 1);
    _checkGroundCloseVal<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), _isChange, instance->getCudaCloseConstraintID(),
        instance->getCudaCloseConstraintVal(), numbers);
    int isChange = 0;
    CUDAMemcpyDToHSafe(isChange, _isChange);
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
    // return false;
}

Scalar self_largestFeasibleStepSize(std::unique_ptr<GeometryManager>& instance, Scalar slackness,
                                    Scalar* mqueue, int numbers) {
    // slackness = 0.9;
    // int numbers = instance->getHostCpNum(0);
    if (numbers < 1) return 1;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    // Scalar* _minSteps;
    // CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(Scalar)));
    // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, numbers * sizeof(AABB),
    // cudaMemcpyDeviceToDevice));
    _reduct_min_selfTimeStep_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(
        instance->getCudaVertPos(), instance->getCudaCCDCollisionPairs(),
        instance->getCudaMoveDir(), mqueue, slackness, numbers);
    //_reduct_min_Scalar3_to_Scalar << <blockNum, threadNum, sharedMsize >> > (_moveDir,
    //_tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    Scalar minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // printf("                 full ccd time step:  %f\n", 1.0 / minValue);
    // CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

Scalar cfl_largestSpeed(std::unique_ptr<GeometryManager>& instance, Scalar* mqueue) {
    int numbers = instance->getHostNumSurfVerts();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    /*Scalar* _maxV;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_maxV, numbers * sizeof(Scalar)));*/
    // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, numbers * sizeof(AABB),
    // cudaMemcpyDeviceToDevice));
    _reduct_max_cfl_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(
        instance->getCudaMoveDir(), mqueue, instance->getCudaSurfVert(), numbers);
    //_reduct_min_Scalar3_to_Scalar << <blockNum, threadNum, sharedMsize >> > (_moveDir,
    //_tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    Scalar minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // CUDA_SAFE_CALL(cudaFree(_maxV));
    return minValue;
}

Scalar reduction2Kappa(int type, const Scalar3* A, const Scalar3* B, Scalar* _queue,
                       int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    /*Scalar* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(Scalar)));*/
    if (type == 0) {
        // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, numbers * sizeof(AABB),
        // cudaMemcpyDeviceToDevice));
        _reduct_Scalar3Dot_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(A, B, _queue, numbers);
    } else if (type == 1) {
        _reduct_Scalar3Sqn_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(A, _queue, numbers);
    }
    //_reduct_min_Scalar3_to_Scalar << <blockNum, threadNum, sharedMsize >> > (_moveDir,
    //_tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        __MATHUTILS__::__add_reduction<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    Scalar dotValue;
    cudaMemcpy(&dotValue, _queue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // CUDA_SAFE_CALL(cudaFree(_queue));
    return dotValue;
}

Scalar ground_largestFeasibleStepSize(std::unique_ptr<GeometryManager>& instance, Scalar slackness,
                                      Scalar* mqueue) {
    int numbers = instance->getHostNumSurfVerts();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    _reduct_min_groundTimeStep_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(
        instance->getCudaVertPos(), instance->getCudaSurfVert(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaMoveDir(), mqueue, slackness, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::_reduct_max_Scalar<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    Scalar minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

void buildGP(std::unique_ptr<GeometryManager>& instance) {
    CUDA_SAFE_CALL(cudaMemset(instance->getCudaGPNum(), 0, sizeof(uint32_t)));
    GroundCollisionDetect(instance);
    CUDA_SAFE_CALL(cudaMemcpy(&instance->getHostGpNum(), instance->getCudaGPNum(), sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));
}

void calBarrierGradientAndHessian(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr,
                                  Scalar3* _gradient, Scalar mKappa) {
    int numbers = instance->getHostCpNum(0);
    if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradientAndHessian<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaRestVertPos(),
        instance->getCudaCollisionPairs(), _gradient, BH_ptr.cudaH12x12, BH_ptr.cudaH9x9,
        BH_ptr.cudaH6x6, BH_ptr.cudaD4Index, BH_ptr.cudaD3Index, BH_ptr.cudaD2Index,
        instance->getCudaCPNum(), instance->getCudaMatIndex(), instance->getHostDHat(), mKappa,
        numbers);
}

void calFrictionHessian(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr) {
    int numbers = instance->getHostCpNumLast(0);
    // if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //

    _calFrictionHessian<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaOriginVertPos(),
        instance->getCudaCollisionPairsLastH(), BH_ptr.cudaH12x12, BH_ptr.cudaH9x9, BH_ptr.cudaH6x6,
        BH_ptr.cudaD4Index, BH_ptr.cudaD3Index, BH_ptr.cudaD2Index, instance->getCudaCPNum(),
        numbers, instance->getHostIPCDt(), instance->getCudaDistCoord(),
        instance->getCudaTanBasis(),
        instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
        instance->getCudaLambdaLastHScalar(), instance->getHostFrictionRate(),
        instance->getHostCpNum(4), instance->getHostCpNum(3), instance->getHostCpNum(2));

    numbers = instance->getHostGpNumLast();
    CUDA_SAFE_CALL(cudaMemcpy(instance->getCudaGPNum(), &instance->getHostGpNumLast(),
                              sizeof(uint32_t), cudaMemcpyHostToDevice));
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionHessian_gd<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaOriginVertPos(),
        instance->getCudaGroundNormal(), instance->getCudaCollisionPairsLastHGd(), BH_ptr.cudaH3x3,
        BH_ptr.cudaD1Index, numbers, instance->getHostIPCDt(),
        instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
        instance->getCudaLambdaLastHScalarGd(), instance->getHostFrictionRate());
}

void computeSelfCloseVal(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostCpNum(0);
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _calSelfCloseVal<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaCollisionPairs(),
        instance->getCudaCloseMConstraintID(), instance->getCudaCloseMConstraintVal(),
        instance->getCudaCloseCPNum(), instance->getHostDTol(), numbers);
}

bool checkSelfCloseVal(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostCloseCpNum();
    if (numbers < 1) return false;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    int* _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkSelfCloseVal<<<blockNum, threadNum>>>(instance->getCudaVertPos(), _isChange,
                                                instance->getCudaCloseMConstraintID(),
                                                instance->getCudaCloseMConstraintVal(), numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

void calBarrierGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient,
                        Scalar mKappa) {
    int numbers = instance->getHostCpNum(0);
    if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradient<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaRestVertPos(),
        instance->getCudaCollisionPairs(), _gradient, instance->getHostDHat(), mKappa, numbers);
}

void calFrictionGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient) {
    int numbers = instance->getHostCpNumLast(0);
    // if (numbers < 1)return;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaOriginVertPos(),
        instance->getCudaCollisionPairsLastH(), _gradient, numbers, instance->getHostIPCDt(),
        instance->getCudaDistCoord(), instance->getCudaTanBasis(),
        instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
        instance->getCudaLambdaLastHScalar(), instance->getHostFrictionRate());

    numbers = instance->getHostGpNumLast();
    // if (numbers < 1)return;
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient_gd<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaOriginVertPos(),
        instance->getCudaGroundNormal(), instance->getCudaCollisionPairsLastHGd(), _gradient,
        numbers, instance->getHostIPCDt(),
        instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
        instance->getCudaLambdaLastHScalarGd(), instance->getHostFrictionRate());
}

////////////////////////TO DO LATER/////////////////////////////////////////

void compute_H_b(Scalar d, Scalar dHat, Scalar& H) {
    Scalar t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

void suggestKappa(std::unique_ptr<GeometryManager>& instance, Scalar& kappa) {
    Scalar H_b;
    // Scalar bboxDiagSize2 = __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(bvh_f.scene.upper,
    // bvh_f.scene.lower));
    compute_H_b(1.0e-16 * instance->getHostBboxDiagSize2(), instance->getHostDHat(), H_b);
    if (instance->getHostMeanMass() == 0.0) {
        kappa =
            instance->getHostMinKappaCoef() / (4.0e-16 * instance->getHostBboxDiagSize2() * H_b);
    } else {
        kappa = instance->getHostMinKappaCoef() * instance->getHostMeanMass() /
                (4.0e-16 * instance->getHostBboxDiagSize2() * H_b);
    }
    //    printf("bboxDiagSize2: %f\n", bboxDiagSize2);
    //    printf("H_b: %f\n", H_b);
    //    printf("sug Kappa: %f\n", kappa);
}

void upperBoundKappa(std::unique_ptr<GeometryManager>& instance, Scalar& kappa) {
    Scalar H_b;
    // Scalar bboxDiagSize2 = __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(bvh_f.scene.upper,
    // bvh_f.scene.lower));
    compute_H_b(1.0e-16 * instance->getHostBboxDiagSize2(), instance->getHostDHat(), H_b);
    Scalar kappaMax = 100 * instance->getHostMinKappaCoef() * instance->getHostMeanMass() /
                      (4.0e-16 * instance->getHostBboxDiagSize2() * H_b);
    // printf("max Kappa: %f\n", kappaMax);
    if (instance->getHostMeanMass() == 0.0) {
        kappaMax = 100 * instance->getHostMinKappaCoef() /
                   (4.0e-16 * instance->getHostBboxDiagSize2() * H_b);
    }

    if (kappa > kappaMax) {
        kappa = kappaMax;
    }
}

void calKineticGradient(Scalar3* _vertexes, Scalar3* _xTilta, Scalar3* _gradient, Scalar* _masses,
                        int numbers) {
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient<<<blockNum, threadNum>>>(_vertexes, _xTilta, _gradient, _masses, numbers);
}

void initKappa(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr,
               std::unique_ptr<PCGSolver>& PCG_ptr) {
    if (instance->getHostCpNum(0) > 0) {
        Scalar3* _GE = instance->getCudaFb();
        Scalar3* _gc = instance->getCudaTempScalar3Mem();
        // CUDA_SAFE_CALL(cudaMalloc((void**)&_gc, vertexNum * sizeof(Scalar3)));
        // CUDA_SAFE_CALL(cudaMalloc((void**)&_GE, vertexNum * sizeof(Scalar3)));
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, instance->getHostNumVertices() * sizeof(Scalar3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, instance->getHostNumVertices() * sizeof(Scalar3)));

        calKineticGradient(instance->getCudaVertPos(), instance->getCudaXTilta(), _GE,
                           instance->getCudaVertMass(), instance->getHostNumVertices());

        __FEMENERGY__::calculate_fem_gradient(
            instance->getCudaTetDmInverses(), instance->getCudaVertPos(),
            instance->getCudaTetElement(), instance->getCudaTetVolume(), _GE,
            instance->getHostNumTetElements(), instance->getHostLengthRate(),
            instance->getHostVolumeRate(), instance->getHostIPCDt());

        __FEMENERGY__::calculate_triangle_fem_gradient(
            instance->getCudaTriDmInverses(), instance->getCudaVertPos(),
            instance->getCudaTriElement(), instance->getCudaTriArea(), _GE,
            instance->getHostNumTriElements(), instance->getHostStretchStiff(),
            instance->getHostShearStiff(), instance->getHostIPCDt());

        __FEMENERGY__::computeBoundConstraintGradient(instance, _GE);

        __FEMENERGY__::computeSoftConstraintGradient(instance, _GE);

        __FEMENERGY__::computeGroundGradient(instance, BH_ptr, _gc, 1);

        calBarrierGradient(instance, _gc, 1);
        Scalar gsum =
            reduction2Kappa(0, _gc, _GE, PCG_ptr->cudaPCGSqueue, instance->getHostNumVertices());
        Scalar gsnorm =
            reduction2Kappa(1, _gc, _GE, PCG_ptr->cudaPCGSqueue, instance->getHostNumVertices());

        Scalar minKappa = -gsum / gsnorm;
        if (minKappa > 0.0) {
            instance->getHostKappa() = minKappa;
        }
        suggestKappa(instance, minKappa);
        if (instance->getHostKappa() < minKappa) {
            instance->getHostKappa() = minKappa;
        }
        upperBoundKappa(instance, instance->getHostKappa());
    }

    // printf("Kappa ====== %f\n", Kappa);
}

bool checkGroundIntersection(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostGpNum();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //

    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(), _isIntersect,
        numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if (h_isITST < 0) {
        return true;
    }
    return false;
}

bool isIntersected(std::unique_ptr<GeometryManager>& instance,
                   std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr) {
    if (checkGroundIntersection(instance)) {
        return true;
    }

    if (LBVH_CD_ptr->lbvh_ef.checkCollisionDetectTriEdge(instance->getHostDHat())) {
        return true;
    }

    return false;
}

};  // namespace __GPUIPC__
