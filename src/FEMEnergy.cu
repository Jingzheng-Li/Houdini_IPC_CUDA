
#include <stdio.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "FEMEnergy.cuh"

namespace __FEMENERGY__ {

__device__ __host__ static Eigen::Matrix3d crossMatrix(Eigen::Vector3d v) {
    Eigen::Matrix3d ret;
    ret << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return ret;
}

__device__ __host__ static Eigen::Matrix2d adjugate(Eigen::Matrix2d M) {
    Eigen::Matrix2d ret;
    ret << M(1, 1), -M(0, 1), -M(1, 0), M(0, 0);
    return ret;
}

__device__ __host__ static Scalar angle(const Eigen::Vector3d& v, const Eigen::Vector3d& w,
                                        const Eigen::Vector3d& axis,
                                        Eigen::Matrix<Scalar, 1, 9>* derivative,  // v, w
                                        Eigen::Matrix<Scalar, 9, 9>* hessian) {
    Scalar theta =
        2.0 * atan2((v.cross(w).dot(axis) / axis.norm()), v.dot(w) + v.norm() * w.norm());

    if (derivative) {
        derivative->segment<3>(0) = -axis.cross(v) / v.squaredNorm() / axis.norm();
        derivative->segment<3>(3) = axis.cross(w) / w.squaredNorm() / axis.norm();
        derivative->segment<3>(6).setZero();
    }
    if (hessian) {
        hessian->setZero();
        hessian->block<3, 3>(0, 0) +=
            2.0 * (axis.cross(v)) * v.transpose() / v.squaredNorm() / v.squaredNorm() / axis.norm();
        hessian->block<3, 3>(3, 3) += -2.0 * (axis.cross(w)) * w.transpose() / w.squaredNorm() /
                                      w.squaredNorm() / axis.norm();
        hessian->block<3, 3>(0, 0) += -crossMatrix(axis) / v.squaredNorm() / axis.norm();
        hessian->block<3, 3>(3, 3) += crossMatrix(axis) / w.squaredNorm() / axis.norm();

        Eigen::Matrix3d dahat = (Eigen::Matrix3d::Identity() / axis.norm() -
                                 axis * axis.transpose() / axis.norm() / axis.norm() / axis.norm());

        hessian->block<3, 3>(0, 6) += crossMatrix(v) * dahat / v.squaredNorm();
        hessian->block<3, 3>(3, 6) += -crossMatrix(w) * dahat / w.squaredNorm();
    }

    return theta;
}

__device__ __host__ static Scalar edgeTheta(
    const Eigen::Vector3d& q0, const Eigen::Vector3d& q1, const Eigen::Vector3d& q2,
    const Eigen::Vector3d& q3,
    Eigen::Matrix<Scalar, 1, 12>* derivative,  // edgeVertex, then edgeOppositeVertex
    Eigen::Matrix<Scalar, 12, 12>* hessian) {
    if (derivative) derivative->setZero();
    if (hessian) hessian->setZero();

    Eigen::Vector3d n0 = (q0 - q2).cross(q1 - q2);
    Eigen::Vector3d n1 = (q1 - q3).cross(q0 - q3);
    Eigen::Vector3d axis = q1 - q0;
    Eigen::Matrix<Scalar, 1, 9> angderiv;
    Eigen::Matrix<Scalar, 9, 9> anghess;

    Scalar theta =
        angle(n0, n1, axis, (derivative || hessian) ? &angderiv : NULL, hessian ? &anghess : NULL);

    if (derivative) {
        derivative->block<1, 3>(0, 0) += angderiv.block<1, 3>(0, 0) * crossMatrix(q2 - q1);
        derivative->block<1, 3>(0, 3) += angderiv.block<1, 3>(0, 0) * crossMatrix(q0 - q2);
        derivative->block<1, 3>(0, 6) += angderiv.block<1, 3>(0, 0) * crossMatrix(q1 - q0);

        derivative->block<1, 3>(0, 0) += angderiv.block<1, 3>(0, 3) * crossMatrix(q1 - q3);
        derivative->block<1, 3>(0, 3) += angderiv.block<1, 3>(0, 3) * crossMatrix(q3 - q0);
        derivative->block<1, 3>(0, 9) += angderiv.block<1, 3>(0, 3) * crossMatrix(q0 - q1);
    }

    if (hessian) {
        Eigen::Matrix3d vqm[3];
        vqm[0] = crossMatrix(q0 - q2);
        vqm[1] = crossMatrix(q1 - q0);
        vqm[2] = crossMatrix(q2 - q1);
        Eigen::Matrix3d wqm[3];
        wqm[0] = crossMatrix(q0 - q1);
        wqm[1] = crossMatrix(q1 - q3);
        wqm[2] = crossMatrix(q3 - q0);

        int vindices[3] = {3, 6, 0};
        int windices[3] = {9, 0, 3};

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                hessian->block<3, 3>(vindices[i], vindices[j]) +=
                    vqm[i].transpose() * anghess.block<3, 3>(0, 0) * vqm[j];
                hessian->block<3, 3>(vindices[i], windices[j]) +=
                    vqm[i].transpose() * anghess.block<3, 3>(0, 3) * wqm[j];
                hessian->block<3, 3>(windices[i], vindices[j]) +=
                    wqm[i].transpose() * anghess.block<3, 3>(3, 0) * vqm[j];
                hessian->block<3, 3>(windices[i], windices[j]) +=
                    wqm[i].transpose() * anghess.block<3, 3>(3, 3) * wqm[j];
            }

            hessian->block<3, 3>(vindices[i], 3) += vqm[i].transpose() * anghess.block<3, 3>(0, 6);
            hessian->block<3, 3>(3, vindices[i]) += anghess.block<3, 3>(6, 0) * vqm[i];
            hessian->block<3, 3>(vindices[i], 0) += -vqm[i].transpose() * anghess.block<3, 3>(0, 6);
            hessian->block<3, 3>(0, vindices[i]) += -anghess.block<3, 3>(6, 0) * vqm[i];

            hessian->block<3, 3>(windices[i], 3) += wqm[i].transpose() * anghess.block<3, 3>(3, 6);
            hessian->block<3, 3>(3, windices[i]) += anghess.block<3, 3>(6, 3) * wqm[i];
            hessian->block<3, 3>(windices[i], 0) += -wqm[i].transpose() * anghess.block<3, 3>(3, 6);
            hessian->block<3, 3>(0, windices[i]) += -anghess.block<3, 3>(6, 3) * wqm[i];
        }

        Eigen::Vector3d dang1 = angderiv.block<1, 3>(0, 0).transpose();
        Eigen::Vector3d dang2 = angderiv.block<1, 3>(0, 3).transpose();

        Eigen::Matrix3d dang1mat = crossMatrix(dang1);
        Eigen::Matrix3d dang2mat = crossMatrix(dang2);

        hessian->block<3, 3>(6, 3) += dang1mat;
        hessian->block<3, 3>(0, 3) -= dang1mat;
        hessian->block<3, 3>(0, 6) += dang1mat;
        hessian->block<3, 3>(3, 0) += dang1mat;
        hessian->block<3, 3>(3, 6) -= dang1mat;
        hessian->block<3, 3>(6, 0) -= dang1mat;

        hessian->block<3, 3>(9, 0) += dang2mat;
        hessian->block<3, 3>(3, 0) -= dang2mat;
        hessian->block<3, 3>(3, 9) += dang2mat;
        hessian->block<3, 3>(0, 3) += dang2mat;
        hessian->block<3, 3>(0, 9) -= dang2mat;
        hessian->block<3, 3>(9, 3) -= dang2mat;
    }

    return theta;
}

__device__ __host__ void __calculateDm2D_Scalar(const Scalar3* vertexes, const uint3& index,
                                                __MATHUTILS__::Matrix2x2S& M) {
    Scalar3 v01 = __MATHUTILS__::__minus(vertexes[index.y], vertexes[index.x]);
    Scalar3 v02 = __MATHUTILS__::__minus(vertexes[index.z], vertexes[index.x]);
    Scalar3 normal = __MATHUTILS__::__normalized(__MATHUTILS__::__v_vec_cross(v01, v02));
    Scalar3 target;
    target.x = 0;
    target.y = 0;
    target.z = 1;
    Scalar3 vec = __MATHUTILS__::__v_vec_cross(normal, target);
    Scalar cos = __MATHUTILS__::__v_vec_dot(normal, target);
    __MATHUTILS__::Matrix3x3S rotation;
    __MATHUTILS__::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

    if (cos + 1 == 0) {
        rotation.m[0][0] = -1;
        rotation.m[1][1] = -1;
    } else {
        __MATHUTILS__::Matrix3x3S cross_vec;
        __MATHUTILS__::__set_Mat_val(cross_vec, 0, -vec.z, vec.y, vec.z, 0, -vec.x, -vec.y, vec.x,
                                     0);
        rotation = __MATHUTILS__::__Mat_add(
            rotation, __MATHUTILS__::__Mat_add(
                          cross_vec, __MATHUTILS__::__S_Mat_multiply(
                                         __MATHUTILS__::__M_Mat_multiply(cross_vec, cross_vec),
                                         1.0 / (1 + cos))));
    }

    Scalar3 rotate_uv0 = __MATHUTILS__::__M_v_multiply(rotation, vertexes[index.x]);
    Scalar3 rotate_uv1 = __MATHUTILS__::__M_v_multiply(rotation, vertexes[index.y]);
    Scalar3 rotate_uv2 = __MATHUTILS__::__M_v_multiply(rotation, vertexes[index.z]);

    Scalar2 uv0 = make_Scalar2(rotate_uv0.x, rotate_uv0.y);
    Scalar2 uv1 = make_Scalar2(rotate_uv1.x, rotate_uv1.y);
    Scalar2 uv2 = make_Scalar2(rotate_uv2.x, rotate_uv2.y);

    Scalar2 u0 = __MATHUTILS__::__minus_v2(uv1, uv0);
    Scalar2 u1 = __MATHUTILS__::__minus_v2(uv2, uv0);

    __MATHUTILS__::__set_Mat2x2_val_column(M, u0, u1);
}

__device__ __host__ void __calculateDs2D_Scalar(const Scalar3* vertexes, const uint3& index,
                                                __MATHUTILS__::Matrix3x2S& M) {
    Scalar o1x = vertexes[index.y].x - vertexes[index.x].x;
    Scalar o1y = vertexes[index.y].y - vertexes[index.x].y;
    Scalar o1z = vertexes[index.y].z - vertexes[index.x].z;

    Scalar o2x = vertexes[index.z].x - vertexes[index.x].x;
    Scalar o2y = vertexes[index.z].y - vertexes[index.x].y;
    Scalar o2z = vertexes[index.z].z - vertexes[index.x].z;

    M.m[0][0] = o1x;
    M.m[0][1] = o2x;
    M.m[1][0] = o1y;
    M.m[1][1] = o2y;
    M.m[2][0] = o1z;
    M.m[2][1] = o2z;
}

__device__ __host__ void __calculateDms3D_Scalar(const Scalar3* vertexes, const uint4& index,
                                                 __MATHUTILS__::Matrix3x3S& M) {
    Scalar o1x = vertexes[index.y].x - vertexes[index.x].x;
    Scalar o1y = vertexes[index.y].y - vertexes[index.x].y;
    Scalar o1z = vertexes[index.y].z - vertexes[index.x].z;

    Scalar o2x = vertexes[index.z].x - vertexes[index.x].x;
    Scalar o2y = vertexes[index.z].y - vertexes[index.x].y;
    Scalar o2z = vertexes[index.z].z - vertexes[index.x].z;

    Scalar o3x = vertexes[index.w].x - vertexes[index.x].x;
    Scalar o3y = vertexes[index.w].y - vertexes[index.x].y;
    Scalar o3z = vertexes[index.w].z - vertexes[index.x].z;

    M.m[0][0] = o1x;
    M.m[0][1] = o2x;
    M.m[0][2] = o3x;
    M.m[1][0] = o1y;
    M.m[1][1] = o2y;
    M.m[1][2] = o3y;
    M.m[2][0] = o1z;
    M.m[2][1] = o2z;
    M.m[2][2] = o3z;
}

__device__ __MATHUTILS__::Matrix3x2S __computePEPF_BaraffWitkinStretch_Scalar(
    const __MATHUTILS__::Matrix3x2S& F, Scalar stretchStiff, Scalar shearStiff) {
    Scalar2 u, v;
    u.x = 1;
    u.y = 0;
    v.x = 0;
    v.y = 1;
    Scalar I5u = __MATHUTILS__::__squaredNorm(__M3x2_v2_multiply(F, u));
    Scalar I5v = __MATHUTILS__::__squaredNorm(__M3x2_v2_multiply(F, v));
    Scalar ucoeff = 1.0 - 1 / sqrt(I5u);
    Scalar vcoeff = 1.0 - 1 / sqrt(I5v);

    if (I5u < 1) {
        ucoeff *= 1e-2;
    }
    if (I5v < 1) {
        vcoeff *= 1e-2;
    }

    __MATHUTILS__::Matrix2x2S uu = __MATHUTILS__::__v2_vec2_toMat2x2(u, u);
    __MATHUTILS__::Matrix2x2S vv = __MATHUTILS__::__v2_vec2_toMat2x2(v, v);
    __MATHUTILS__::Matrix3x2S Fuu = __MATHUTILS__::__M3x2_M2x2_Multiply(F, uu);
    __MATHUTILS__::Matrix3x2S Fvv = __MATHUTILS__::__M3x2_M2x2_Multiply(F, vv);

    Scalar I6 = __MATHUTILS__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    __MATHUTILS__::Matrix2x2S uv = __MATHUTILS__::__v2_vec2_toMat2x2(u, v);
    __MATHUTILS__::Matrix2x2S vu = __MATHUTILS__::__v2_vec2_toMat2x2(v, u);
    __MATHUTILS__::Matrix3x2S Fuv = __MATHUTILS__::__M3x2_M2x2_Multiply(F, uv);
    __MATHUTILS__::Matrix3x2S Fvu = __MATHUTILS__::__M3x2_M2x2_Multiply(F, vu);
    Scalar utv = __MATHUTILS__::__v2_vec_multiply(u, v);

    __MATHUTILS__::Matrix3x2S PEPF_shear =
        __MATHUTILS__::__S_Mat3x2_multiply(__MATHUTILS__::__Mat3x2_add(Fuv, Fvu), 2 * (I6 - utv));
    __MATHUTILS__::Matrix3x2S PEPF_stretch =
        __MATHUTILS__::__Mat3x2_add(__MATHUTILS__::__S_Mat3x2_multiply(Fuu, 2 * ucoeff),
                                    __MATHUTILS__::__S_Mat3x2_multiply(Fvv, 2 * vcoeff));
    __MATHUTILS__::Matrix3x2S PEPF =
        __MATHUTILS__::__Mat3x2_add(__MATHUTILS__::__S_Mat3x2_multiply(PEPF_shear, shearStiff),
                                    __MATHUTILS__::__S_Mat3x2_multiply(PEPF_stretch, stretchStiff));
    return PEPF;
}

__device__ __MATHUTILS__::Matrix3x3S __computePEPF_StableNHK3D_Scalar(
    const __MATHUTILS__::Matrix3x3S& F, const __MATHUTILS__::Matrix3x3S& Sigma,
    const __MATHUTILS__::Matrix3x3S& U, const __MATHUTILS__::Matrix3x3S& V, Scalar lengthRate,
    Scalar volumRate) {
    Scalar I3 = Sigma.m[0][0] * Sigma.m[1][1] * Sigma.m[2][2];
    Scalar I2 = Sigma.m[0][0] * Sigma.m[0][0] + Sigma.m[1][1] * Sigma.m[1][1] +
                Sigma.m[2][2] * Sigma.m[2][2];

    Scalar u = lengthRate, r = volumRate;
    __MATHUTILS__::Matrix3x3S pI3pF;

    pI3pF.m[0][0] = F.m[1][1] * F.m[2][2] - F.m[1][2] * F.m[2][1];
    pI3pF.m[0][1] = F.m[1][2] * F.m[2][0] - F.m[1][0] * F.m[2][2];
    pI3pF.m[0][2] = F.m[1][0] * F.m[2][1] - F.m[1][1] * F.m[2][0];

    pI3pF.m[1][0] = F.m[2][1] * F.m[0][2] - F.m[2][2] * F.m[0][1];
    pI3pF.m[1][1] = F.m[2][2] * F.m[0][0] - F.m[2][0] * F.m[0][2];
    pI3pF.m[1][2] = F.m[2][0] * F.m[0][1] - F.m[2][1] * F.m[0][0];

    pI3pF.m[2][0] = F.m[0][1] * F.m[1][2] - F.m[1][1] * F.m[0][2];
    pI3pF.m[2][1] = F.m[0][2] * F.m[1][0] - F.m[0][0] * F.m[1][2];
    pI3pF.m[2][2] = F.m[0][0] * F.m[1][1] - F.m[0][1] * F.m[1][0];

    // printf("volRate and LenRate:  %f    %f\n", volumRate, lengthRate);

    __MATHUTILS__::Matrix3x3S PEPF, tempA, tempB;
    tempA = __MATHUTILS__::__S_Mat_multiply(F, u * (1 - 1 / (I2 + 1)));
    tempB = __MATHUTILS__::__S_Mat_multiply(pI3pF, (r * (I3 - 1 - u * 3 / (r * 4))));
    __MATHUTILS__::__Mat_add(tempA, tempB, PEPF);
    return PEPF;
}

__device__ __MATHUTILS__::Matrix3x3S computePEPF_ARAP_Scalar(const __MATHUTILS__::Matrix3x3S& F,
                                                             const __MATHUTILS__::Matrix3x3S& Sigma,
                                                             const __MATHUTILS__::Matrix3x3S& U,
                                                             const __MATHUTILS__::Matrix3x3S& V,
                                                             const Scalar& lengthRate) {
    __MATHUTILS__::Matrix3x3S R, S;

    S = __MATHUTILS__::__M_Mat_multiply(
        __MATHUTILS__::__M_Mat_multiply(V, Sigma),
        __MATHUTILS__::__Transpose3x3(V));  // V * sigma * V.transpose();
    R = __MATHUTILS__::__M_Mat_multiply(U, __MATHUTILS__::__Transpose3x3(V));
    __MATHUTILS__::Matrix3x3S g = __MATHUTILS__::__Mat3x3_minus(F, R);
    return __MATHUTILS__::__S_Mat_multiply(g, lengthRate);  // lengthRate * g;
}

__device__ __MATHUTILS__::Matrix9x9S project_ARAP_H_3D(const __MATHUTILS__::Matrix3x3S& Sigma,
                                                       const __MATHUTILS__::Matrix3x3S& U,
                                                       const __MATHUTILS__::Matrix3x3S& V,
                                                       const Scalar& lengthRate) {
    __MATHUTILS__::Matrix3x3S R, S;

    S = __MATHUTILS__::__M_Mat_multiply(
        __MATHUTILS__::__M_Mat_multiply(V, Sigma),
        __MATHUTILS__::__Transpose3x3(V));  // V * sigma * V.transpose();
    R = __MATHUTILS__::__M_Mat_multiply(U, __MATHUTILS__::__Transpose3x3(V));
    __MATHUTILS__::Matrix3x3S T0, T1, T2;

    __MATHUTILS__::__set_Mat_val(T0, 0, -1, 0, 1, 0, 0, 0, 0, 0);
    __MATHUTILS__::__set_Mat_val(T1, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __MATHUTILS__::__set_Mat_val(T2, 0, 0, 1, 0, 0, 0, -1, 0, 0);

    Scalar ml = 1 / sqrt(2.0);

    __MATHUTILS__::Matrix3x3S VTransp = __MATHUTILS__::__Transpose3x3(V);

    T0 = __MATHUTILS__::__S_Mat_multiply(
        __MATHUTILS__::__M_Mat_multiply(__MATHUTILS__::__M_Mat_multiply(U, T0), VTransp), ml);
    T1 = __MATHUTILS__::__S_Mat_multiply(
        __MATHUTILS__::__M_Mat_multiply(__MATHUTILS__::__M_Mat_multiply(U, T1), VTransp), ml);
    T2 = __MATHUTILS__::__S_Mat_multiply(
        __MATHUTILS__::__M_Mat_multiply(__MATHUTILS__::__M_Mat_multiply(U, T2), VTransp), ml);

    __MATHUTILS__::Vector9S t0, t1, t2;
    t0 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(T0);
    t1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(T1);
    t2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(T2);

    Scalar sx = Sigma.m[0][0];
    Scalar sy = Sigma.m[1][1];
    Scalar sz = Sigma.m[2][2];
    Scalar lambda0 = 2 / (sx + sy);
    Scalar lambda1 = 2 / (sz + sy);
    Scalar lambda2 = 2 / (sx + sz);

    if (sx + sy < 2) lambda0 = 1;
    if (sz + sy < 2) lambda1 = 1;
    if (sx + sz < 2) lambda2 = 1;

    __MATHUTILS__::Matrix9x9S SH, M9_temp;
    __MATHUTILS__::__identify_Mat9x9(SH);
    __MATHUTILS__::Vector9S V9_temp;

    M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(t0, t0);
    M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, -lambda0);
    SH = __MATHUTILS__::__Mat9x9_add(SH, M9_temp);

    M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(t1, t1);
    M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, -lambda1);
    SH = __MATHUTILS__::__Mat9x9_add(SH, M9_temp);

    M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(t2, t2);
    M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, -lambda2);
    SH = __MATHUTILS__::__Mat9x9_add(SH, M9_temp);

    return __MATHUTILS__::__S_Mat9x9_multiply(SH, lengthRate);
    ;
}
__device__ __MATHUTILS__::Matrix6x6S __project_BaraffWitkinStretch_H(
    const __MATHUTILS__::Matrix3x2S& F) {
    __MATHUTILS__::Matrix6x6S H;
    H.m[0][0] = H.m[0][1] = H.m[0][2] = H.m[0][3] = H.m[0][4] = H.m[0][5] = 0;
    H.m[1][0] = H.m[1][1] = H.m[1][2] = H.m[1][3] = H.m[1][4] = H.m[1][5] = 0;
    H.m[2][0] = H.m[2][1] = H.m[2][2] = H.m[2][3] = H.m[2][4] = H.m[2][5] = 0;
    H.m[3][0] = H.m[3][1] = H.m[3][2] = H.m[3][3] = H.m[3][4] = H.m[3][5] = 0;
    H.m[4][0] = H.m[4][1] = H.m[4][2] = H.m[4][3] = H.m[4][4] = H.m[4][5] = 0;
    H.m[5][0] = H.m[5][1] = H.m[5][2] = H.m[5][3] = H.m[5][4] = H.m[5][5] = 0;
    Scalar2 u, v;
    u.x = 1;
    u.y = 0;
    v.x = 0;
    v.y = 1;
    Scalar I5u = __MATHUTILS__::__squaredNorm(__M3x2_v2_multiply(F, u));
    Scalar I5v = __MATHUTILS__::__squaredNorm(__M3x2_v2_multiply(F, v));

    Scalar invSqrtI5u = 1.0 / sqrt(I5u);
    Scalar invSqrtI5v = 1.0 / sqrt(I5v);
    if (1 - invSqrtI5u > 0) H.m[0][0] = H.m[1][1] = H.m[2][2] = 2 * (1 - invSqrtI5u);
    if (1 - invSqrtI5v > 0) H.m[3][3] = H.m[4][4] = H.m[5][5] = 2 * (1 - invSqrtI5v);

    Scalar uCoeff = (1.0 - invSqrtI5u >= 0.0) ? invSqrtI5u : 1.0;
    Scalar vCoeff = (1.0 - invSqrtI5v >= 0.0) ? invSqrtI5v : 1.0;
    uCoeff *= 2;
    vCoeff *= 2;

    if (I5u < 1) {
        uCoeff *= 1e-2;
    }
    if (I5v < 1) {
        vCoeff *= 1e-2;
    }

    Scalar3 fu, fv;
    fu.x = F.m[0][0];
    fu.y = F.m[1][0];
    fu.z = F.m[2][0];
    fv.x = F.m[0][1];
    fv.y = F.m[1][1];
    fv.z = F.m[2][1];
    fu = __MATHUTILS__::__normalized(fu);
    fv = __MATHUTILS__::__normalized(fv);

    __MATHUTILS__::Matrix3x3S cfufu =
        __MATHUTILS__::__S_Mat_multiply(__MATHUTILS__::__v_vec_toMat(fu, fu), uCoeff);
    __MATHUTILS__::Matrix3x3S cfvfv =
        __MATHUTILS__::__S_Mat_multiply(__MATHUTILS__::__v_vec_toMat(fv, fv), vCoeff);
    H.m[0][0] += cfufu.m[0][0];
    H.m[0][1] += cfufu.m[0][1];
    H.m[0][2] += cfufu.m[0][2];
    H.m[1][0] += cfufu.m[1][0];
    H.m[1][1] += cfufu.m[1][1];
    H.m[1][2] += cfufu.m[1][2];
    H.m[2][0] += cfufu.m[2][0];
    H.m[2][1] += cfufu.m[2][1];
    H.m[2][2] += cfufu.m[2][2];

    H.m[3][3] += cfvfv.m[0][0];
    H.m[3][4] += cfvfv.m[0][1];
    H.m[3][5] += cfvfv.m[0][2];
    H.m[4][3] += cfvfv.m[1][0];
    H.m[4][4] += cfvfv.m[1][1];
    H.m[4][5] += cfvfv.m[1][2];
    H.m[5][3] += cfvfv.m[2][0];
    H.m[5][4] += cfvfv.m[2][1];
    H.m[5][5] += cfvfv.m[2][2];
    return H;
}

__device__ __MATHUTILS__::Matrix6x6S __project_BaraffWitkinShear_H(
    const __MATHUTILS__::Matrix3x2S& F) {
    __MATHUTILS__::Matrix6x6S H;
    H.m[0][0] = H.m[0][1] = H.m[0][2] = H.m[0][3] = H.m[0][4] = H.m[0][5] = 0;
    H.m[1][0] = H.m[1][1] = H.m[1][2] = H.m[1][3] = H.m[1][4] = H.m[1][5] = 0;
    H.m[2][0] = H.m[2][1] = H.m[2][2] = H.m[2][3] = H.m[2][4] = H.m[2][5] = 0;
    H.m[3][0] = H.m[3][1] = H.m[3][2] = H.m[3][3] = H.m[3][4] = H.m[3][5] = 0;
    H.m[4][0] = H.m[4][1] = H.m[4][2] = H.m[4][3] = H.m[4][4] = H.m[4][5] = 0;
    H.m[5][0] = H.m[5][1] = H.m[5][2] = H.m[5][3] = H.m[5][4] = H.m[5][5] = 0;
    Scalar2 u, v;
    u.x = 1;
    u.y = 0;
    v.x = 0;
    v.y = 1;
    __MATHUTILS__::Matrix6x6S H_shear;

    H_shear.m[0][0] = H_shear.m[0][1] = H_shear.m[0][2] = H_shear.m[0][3] = H_shear.m[0][4] =
        H_shear.m[0][5] = 0;
    H_shear.m[1][0] = H_shear.m[1][1] = H_shear.m[1][2] = H_shear.m[1][3] = H_shear.m[1][4] =
        H_shear.m[1][5] = 0;
    H_shear.m[2][0] = H_shear.m[2][1] = H_shear.m[2][2] = H_shear.m[2][3] = H_shear.m[2][4] =
        H_shear.m[2][5] = 0;
    H_shear.m[3][0] = H_shear.m[3][1] = H_shear.m[3][2] = H_shear.m[3][3] = H_shear.m[3][4] =
        H_shear.m[3][5] = 0;
    H_shear.m[4][0] = H_shear.m[4][1] = H_shear.m[4][2] = H_shear.m[4][3] = H_shear.m[4][4] =
        H_shear.m[4][5] = 0;
    H_shear.m[5][0] = H_shear.m[5][1] = H_shear.m[5][2] = H_shear.m[5][3] = H_shear.m[5][4] =
        H_shear.m[5][5] = 0;
    H_shear.m[3][0] = H_shear.m[4][1] = H_shear.m[5][2] = H_shear.m[0][3] = H_shear.m[1][4] =
        H_shear.m[2][5] = 1.0;
    Scalar I6 = __MATHUTILS__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    Scalar signI6 = (I6 >= 0) ? 1.0 : -1.0;

    __MATHUTILS__::Matrix2x2S uv = __MATHUTILS__::__v2_vec2_toMat2x2(u, v);
    __MATHUTILS__::Matrix2x2S vu = __MATHUTILS__::__v2_vec2_toMat2x2(v, u);
    __MATHUTILS__::Matrix3x2S Fuv = __MATHUTILS__::__M3x2_M2x2_Multiply(F, uv);
    __MATHUTILS__::Matrix3x2S Fvu = __MATHUTILS__::__M3x2_M2x2_Multiply(F, vu);
    __MATHUTILS__::Vector6S g;
    g.v[0] = Fuv.m[0][0] + Fvu.m[0][0];
    g.v[1] = Fuv.m[1][0] + Fvu.m[1][0];
    g.v[2] = Fuv.m[2][0] + Fvu.m[2][0];
    g.v[3] = Fuv.m[0][1] + Fvu.m[0][1];
    g.v[4] = Fuv.m[1][1] + Fvu.m[1][1];
    g.v[5] = Fuv.m[2][1] + Fvu.m[2][1];
    Scalar I2 = F.m[0][0] * F.m[0][0] + F.m[0][1] * F.m[0][1] + F.m[1][0] * F.m[1][0] +
                F.m[1][1] * F.m[1][1] + F.m[2][0] * F.m[2][0] + F.m[2][1] * F.m[2][1];
    Scalar lambda0 = 0.5 * (I2 + sqrt(I2 * I2 + 12 * I6 * I6));
    __MATHUTILS__::Vector6S q0 = __MATHUTILS__::__M6x6_v6_multiply(H_shear, g);
    q0 = __MATHUTILS__::__s_vec6_multiply(q0, I6);
    q0 = __MATHUTILS__::__add6(q0, __s_vec6_multiply(g, lambda0));
    __MATHUTILS__::__normalized_vec6_Scalar(q0);
    __MATHUTILS__::Matrix6x6S T;
    T.m[0][0] = T.m[0][1] = T.m[0][2] = T.m[0][3] = T.m[0][4] = T.m[0][5] = 0;
    T.m[1][0] = T.m[1][1] = T.m[1][2] = T.m[1][3] = T.m[1][4] = T.m[1][5] = 0;
    T.m[2][0] = T.m[2][1] = T.m[2][2] = T.m[2][3] = T.m[2][4] = T.m[2][5] = 0;
    T.m[3][0] = T.m[3][1] = T.m[3][2] = T.m[3][3] = T.m[3][4] = T.m[3][5] = 0;
    T.m[4][0] = T.m[4][1] = T.m[4][2] = T.m[4][3] = T.m[4][4] = T.m[4][5] = 0;
    T.m[5][0] = T.m[5][1] = T.m[5][2] = T.m[5][3] = T.m[5][4] = T.m[5][5] = 0;
    T.m[0][0] = T.m[1][1] = T.m[2][2] = T.m[3][3] = T.m[4][4] = T.m[5][5] = 1.0;
    __MATHUTILS__::__Mat_add(T, __S_Mat6x6_multiply(H_shear, signI6), T);
    T = __MATHUTILS__::__S_Mat6x6_multiply(T, 0.5);
    __MATHUTILS__::Vector6S Tq = __MATHUTILS__::__M6x6_v6_multiply(T, q0);
    Scalar normTQ = Tq.v[0] * Tq.v[0] + Tq.v[1] * Tq.v[1] + Tq.v[2] * Tq.v[2] + Tq.v[3] * Tq.v[3] +
                    Tq.v[4] * Tq.v[4] + Tq.v[5] * Tq.v[5];

    __MATHUTILS__::Matrix6x6S Tmp;
    __MATHUTILS__::__Mat_add(
        T, __MATHUTILS__::__S_Mat6x6_multiply(__v6_vec6_toMat6x6(Tq, Tq), -1.0 / normTQ), Tmp);
    Tmp = __S_Mat6x6_multiply(Tmp, fabs(I6));
    __MATHUTILS__::__Mat_add(Tmp, __S_Mat6x6_multiply(__v6_vec6_toMat6x6(q0, q0), lambda0), Tmp);
    Tmp = __S_Mat6x6_multiply(Tmp, 2);
    return Tmp;
}

__device__ __MATHUTILS__::Matrix9x9S __project_StabbleNHK_H_3D(
    const Scalar3& sigma, const __MATHUTILS__::Matrix3x3S& U, const __MATHUTILS__::Matrix3x3S& V,
    const Scalar& lengthRate, const Scalar& volumRate, __MATHUTILS__::Matrix9x9S& H) {
    Scalar sigxx = sigma.x * sigma.x;
    Scalar sigyy = sigma.y * sigma.y;
    Scalar sigzz = sigma.z * sigma.z;

    Scalar I3 = sigma.x * sigma.y * sigma.z;
    Scalar I2 = sigxx + sigyy + sigzz;
    Scalar g2 = sigxx * sigyy + sigxx * sigzz + sigyy * sigzz;

    Scalar u = lengthRate, r = volumRate;

    Scalar n = 2 * u / ((I2 + 1) * (I2 + 1) * (r * (I3 - 1) - 3 * u / 4));
    Scalar p = r / (r * (I3 - 1) - 3 * u / 4);
    Scalar c2 = -g2 * p - I2 * n;
    Scalar c1 = -(1 + 2 * I3 * p) * I2 - 6 * I3 * n + (g2 * I2 - 9 * I3 * I3) * p * n;
    Scalar c0 =
        -(2 + 3 * I3 * p) * I3 + (I2 * I2 - 4 * g2) * n + 2 * I3 * p * n * (I2 * I2 - 3 * g2);

    Scalar roots[3] = {0};
    int num_solution = 0;
    __MATHUTILS__::__NewtonSolverForCubicEquation_satbleNeohook(1, c2, c1, c0, roots, num_solution,
                                                                1e-6);

    __MATHUTILS__::Matrix3x3S D[3], M_temp[3];
    Scalar q[3];
    __MATHUTILS__::Matrix3x3S Q[9];
    Scalar lamda[9];
    Scalar Ut = u * (1 - 1 / (I2 + 1));
    Scalar alpha = 1 + 3 * u / r / 4;

    Scalar I3minuAlphaDotR = (I3 - alpha) * r;

    for (int i = 0; i < num_solution; i++) {
        Scalar alpha0 = roots[i] * (sigma.y + sigma.x * sigma.z * n + I3 * sigma.y * p) +
                        sigma.x * sigma.z + sigma.y * (sigxx - sigyy + sigzz) * n +
                        I3 * sigma.x * sigma.z * p +
                        sigma.x * (sigxx - sigyy) * sigma.z * (sigyy - sigzz) * p * n;

        Scalar alpha1 = roots[i] * (sigma.x + sigma.y * sigma.z * n + I3 * sigma.x * p) +
                        sigma.y * sigma.z - sigma.x * (sigxx - sigyy - sigzz) * n +
                        I3 * sigma.y * sigma.z * p -
                        sigma.y * (sigxx - sigyy) * sigma.z * (sigxx - sigzz) * p * n;

        Scalar alpha2 = roots[i] * roots[i] - roots[i] * (sigxx + sigyy) * (n + sigzz * p) - sigzz -
                        2 * I3 * n - 2 * I3 * sigzz * p +
                        ((sigxx - sigyy) * sigma.z) * ((sigxx - sigyy) * sigma.z) * p * n;

        Scalar normalSum = alpha0 * alpha0 + alpha1 * alpha1 + alpha2 * alpha2;

        if (normalSum == 0) {
            lamda[i] = 0;
            continue;
        }

        q[i] = 1 / sqrt(normalSum);
        __MATHUTILS__::__set_Mat_val(D[i], alpha0, 0, 0, 0, alpha1, 0, 0, 0, alpha2);

        __MATHUTILS__::__s_M_Mat_MT_multiply(U, D[i], V, q[i], Q[i]);
        lamda[i] = I3minuAlphaDotR * roots[i] + Ut;
    }

    lamda[3] = Ut + sigma.z * I3minuAlphaDotR;
    lamda[4] = Ut + sigma.x * I3minuAlphaDotR;
    lamda[5] = Ut + sigma.y * I3minuAlphaDotR;

    lamda[6] = Ut - sigma.z * I3minuAlphaDotR;
    lamda[7] = Ut - sigma.x * I3minuAlphaDotR;
    lamda[8] = Ut - sigma.y * I3minuAlphaDotR;

    __MATHUTILS__::__set_Mat_val(Q[3], 0, -1, 0, 1, 0, 0, 0, 0, 0);
    __MATHUTILS__::__set_Mat_val(Q[4], 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __MATHUTILS__::__set_Mat_val(Q[5], 0, 0, 1, 0, 0, 0, -1, 0, 0);
    __MATHUTILS__::__set_Mat_val(Q[6], 0, 1, 0, 1, 0, 0, 0, 0, 0);
    __MATHUTILS__::__set_Mat_val(Q[7], 0, 0, 0, 0, 0, 1, 0, 1, 0);
    __MATHUTILS__::__set_Mat_val(Q[8], 0, 0, 1, 0, 0, 0, 1, 0, 0);

    Scalar ml = 1 / sqrt(2.0);

    M_temp[1] = __MATHUTILS__::__Transpose3x3(V);
    for (int i = 3; i < 9; i++) {
        __MATHUTILS__::__M_Mat_multiply(U, Q[i], M_temp[0]);
        __MATHUTILS__::__M_Mat_multiply(M_temp[0], M_temp[1], M_temp[2]);
        Q[i] = __MATHUTILS__::__S_Mat_multiply(M_temp[2], ml);

        // Q[i] = __MATHUTILS__::__s_M_Mat_MT_multiply(U, Q[i], V, ml);
    }

    __MATHUTILS__::Matrix9x9S M9_temp;
    __MATHUTILS__::__init_Mat9x9(H, 0);
    __MATHUTILS__::Vector9S V9_temp;
    for (int i = 0; i < 9; i++) {
        if (lamda[i] > 0) {
            V9_temp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(Q[i]);
            M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(V9_temp, V9_temp, lamda[i]);
            H = __MATHUTILS__::__Mat9x9_add(H, M9_temp);
        }
    }
}

__device__ __MATHUTILS__::Matrix9x9S __project_ANIOSI5_H_3D(
    const __MATHUTILS__::Matrix3x3S& F, const __MATHUTILS__::Matrix3x3S& sigma,
    const __MATHUTILS__::Matrix3x3S& U, const __MATHUTILS__::Matrix3x3S& V,
    const Scalar3& fiber_direction, const Scalar& scale, const Scalar& contract_length) {
    Scalar3 direction = __MATHUTILS__::__normalized(fiber_direction);
    __MATHUTILS__::Matrix3x3S S, M_temp[3], Vtranspose;

    // S = V * sigma * V.transpose();
    __MATHUTILS__::__M_Mat_multiply(V, sigma, M_temp[0]);
    Vtranspose = __MATHUTILS__::__Transpose3x3(V);
    __MATHUTILS__::__M_Mat_multiply(M_temp[0], Vtranspose, S);
    //__S_Mat_multiply(M_temp[2], ml, Q[i]);

    Scalar3 v_temp = __MATHUTILS__::__M_v_multiply(S, direction);
    Scalar I4 =
        __MATHUTILS__::__v_vec_dot(direction, v_temp);  // direction.transpose() * S * direction;
    Scalar I5 = __MATHUTILS__::__v_vec_dot(
        v_temp,
        v_temp);  // direction.transpose() * S.transpose() * S * direction;

    __MATHUTILS__::Matrix9x9S H;
    __MATHUTILS__::__init_Mat9x9(H, 0);
    if (abs(I5) < 1e-15) return H;

    Scalar s = 0;
    if (I4 < 0) {
        s = -1;
    } else if (I4 > 0) {
        s = 1;
    }

    Scalar lamda0 = scale;
    Scalar lamda1 = scale * (1 - s * contract_length / sqrt(I5));
    Scalar lamda2 = lamda1;
    // Scalar lamda2 = lamda1;
    __MATHUTILS__::Matrix3x3S Q0, Q1, Q2, A;
    A = __MATHUTILS__::__v_vec_toMat(direction, direction);

    __MATHUTILS__::__M_Mat_multiply(F, A, M_temp[0]);
    Q0 = __MATHUTILS__::__S_Mat_multiply(M_temp[0], (1 / sqrt(I5)));
    // Q0 = (1 / sqrt(I5)) * F * A;

    __MATHUTILS__::Matrix3x3S Tx, Ty, Tz;

    __MATHUTILS__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __MATHUTILS__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
    __MATHUTILS__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

    //__Transpose3x3(V, M_temp[0]);
    Scalar3 directionM = __MATHUTILS__::__M_v_multiply(Vtranspose, direction);

    Scalar ratio = 1.f / sqrt(2.f);
    Tx = __MATHUTILS__::__S_Mat_multiply(Tx, ratio);
    Ty = __MATHUTILS__::__S_Mat_multiply(Ty, ratio);
    Tz = __MATHUTILS__::__S_Mat_multiply(Tz, ratio);

    // Q1 = U * Tx * sigma * V.transpose() * A;
    __MATHUTILS__::__M_Mat_multiply(U, Tx, M_temp[1]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[1], sigma, M_temp[2]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[2], Vtranspose, M_temp[1]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[1], A, Q1);

    // Q2 = (sigma(1, 1) * directionM[1]) * U * Tz * sigma * V.transpose() * A -
    // (sigma(2, 2) * directionM[2]) * U * Ty * sigma * V.transpose() * A;
    __MATHUTILS__::__M_Mat_multiply(U, Tz, M_temp[0]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[0], sigma, M_temp[1]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[1], Vtranspose, M_temp[2]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[2], A, M_temp[0]);
    M_temp[0] = __S_Mat_multiply(M_temp[0], (sigma.m[1][1] * directionM.y));
    __MATHUTILS__::__M_Mat_multiply(U, Ty, M_temp[1]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[1], sigma, M_temp[2]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[2], Vtranspose, M_temp[1]);
    __MATHUTILS__::__M_Mat_multiply(M_temp[1], A, M_temp[2]);
    M_temp[2] = __MATHUTILS__::__S_Mat_multiply(M_temp[2], -(sigma.m[2][2] * directionM.z));
    __MATHUTILS__::__Mat_add(M_temp[0], M_temp[2], Q2);

    // H = lamda0 * vec_Scalar(Q0) * vec_Scalar(Q0).transpose();
    __MATHUTILS__::Vector9S V9_temp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(Q0);
    __MATHUTILS__::Matrix9x9S M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
    H = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lamda0);
    // H = __Mat9x9_add(H, M9_temp[1]);
    if (lamda1 > 0) {
        // H += lamda1 * vec_Scalar(Q1) * vec_Scalar(Q1).transpose();
        // H += lamda2 * vec_Scalar(Q2) * vec_Scalar(Q2).transpose();
        V9_temp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(Q1);
        M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
        M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lamda1);
        H = __MATHUTILS__::__Mat9x9_add(H, M9_temp);

        V9_temp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(Q2);
        M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(V9_temp, V9_temp);
        M9_temp = __MATHUTILS__::__S_Mat9x9_multiply(M9_temp, lamda2);
        H = __MATHUTILS__::__Mat9x9_add(H, M9_temp);
    }

    return H;
}

__device__ __MATHUTILS__::Matrix3x3S __computePEPF_Aniostropic3D_Scalar(
    const __MATHUTILS__::Matrix3x3S& F, Scalar3 fiber_direction, const Scalar& scale,
    const Scalar& contract_length) {
    Scalar3 direction = __MATHUTILS__::__normalized(fiber_direction);
    __MATHUTILS__::Matrix3x3S U, V, S, sigma, M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);
    __MATHUTILS__::__M_Mat_multiply(V, sigma, M_Temp0);
    M_Temp1 = __MATHUTILS__::__Transpose3x3(V);
    __MATHUTILS__::__M_Mat_multiply(M_Temp0, M_Temp1, S);
    Scalar3 V_Temp0, V_Temp1;
    V_Temp0 = __MATHUTILS__::__v_M_multiply(direction, S);
    Scalar I4, I5;
    I4 = __MATHUTILS__::__v_vec_dot(V_Temp0, direction);
    V_Temp1 = __MATHUTILS__::__M_v_multiply(S, direction);
    I5 = __MATHUTILS__::__v_vec_dot(V_Temp1, V_Temp1);

    if (I4 == 0) {
        // system("pause");
    }

    Scalar s = 0;
    if (I4 < 0) {
        s = -1;
    } else if (I4 > 0) {
        s = 1;
    }

    __MATHUTILS__::Matrix3x3S PEPF;
    Scalar s_temp0 = scale * (1 - s * contract_length / sqrt(I5));
    M_Temp0 = __MATHUTILS__::__v_vec_toMat(direction, direction);
    __MATHUTILS__::__M_Mat_multiply(F, M_Temp0, M_Temp1);
    PEPF = __MATHUTILS__::__S_Mat_multiply(M_Temp1, s_temp0);
    return PEPF;
}

__device__ Scalar __cal_BaraffWitkinStretch_energy(const Scalar3* vertexes, const uint3& triangle,
                                                   const __MATHUTILS__::Matrix2x2S& triDmInverse,
                                                   const Scalar& area, const Scalar& stretchStiff,
                                                   const Scalar& shearhStiff) {
    __MATHUTILS__::Matrix3x2S Ds;
    __calculateDs2D_Scalar(vertexes, triangle, Ds);
    __MATHUTILS__::Matrix3x2S F = __MATHUTILS__::__M3x2_M2x2_Multiply(Ds, triDmInverse);

    Scalar2 u, v;
    u.x = 1;
    u.y = 0;
    v.x = 0;
    v.y = 1;
    Scalar I5u = __MATHUTILS__::__squaredNorm(__M3x2_v2_multiply(F, u));
    Scalar I5v = __MATHUTILS__::__squaredNorm(__M3x2_v2_multiply(F, v));

    Scalar ucoeff = 1;
    Scalar vcoeff = 1;
    if (I5u < 1) {
        ucoeff *= 1e-2;
    }
    if (I5v < 1) {
        vcoeff *= 1e-2;
    }

    Scalar I6 = __MATHUTILS__::__v_vec_dot(__M3x2_v2_multiply(F, u), __M3x2_v2_multiply(F, v));
    return area *
           (stretchStiff * (ucoeff * pow(sqrt(I5u) - 1, 2) + vcoeff * pow(sqrt(I5v) - 1, 2)) +
            shearhStiff * I6 * I6);
}

__device__ Scalar __cal_StabbleNHK_energy_3D(const Scalar3* vertexes, const uint4& tetrahedra,
                                             const __MATHUTILS__::Matrix3x3S& DmInverse,
                                             const Scalar& volume, const Scalar& lenRate,
                                             const Scalar& volRate) {
    __MATHUTILS__::Matrix3x3S Ds;
    __calculateDms3D_Scalar(vertexes, tetrahedra, Ds);
    __MATHUTILS__::Matrix3x3S F;
    __M_Mat_multiply(Ds, DmInverse, F);
    // printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", F.m[0][0],
    // F.m[0][1], F.m[0][2], F.m[1][0], F.m[1][1], F.m[1][2], F.m[2][0],
    // F.m[2][1], F.m[2][2]);
    __MATHUTILS__::Matrix3x3S U, V, sigma, S;
    __MATHUTILS__::Matrix3x3S M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);
    // printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", V.m[0][0],
    // V.m[0][1], V.m[0][2], V.m[1][0], V.m[1][1], V.m[1][2], V.m[2][0],
    // V.m[2][1], V.m[2][2]); printf("%f  %f  %f\n%f  %f  %f\n%f  %f
    // %f\n\n\n\n\n\n", U.m[0][0], U.m[0][1], U.m[0][2], U.m[1][0], U.m[1][1],
    // U.m[1][2], U.m[2][0], U.m[2][1], U.m[2][2]);
    __MATHUTILS__::__M_Mat_multiply(V, sigma, M_Temp0);
    M_Temp1 = __MATHUTILS__::__Transpose3x3(V);
    __MATHUTILS__::__M_Mat_multiply(M_Temp0, M_Temp1, S);

    __MATHUTILS__::__M_Mat_multiply(S, S, M_Temp0);
    Scalar I2 = __MATHUTILS__::__Mat_Trace(M_Temp0);
    Scalar I3;
    __MATHUTILS__::__Determiant(S, I3);
    // printf("%f     %f\n\n\n", I2, I3);
    return (0.5 * lenRate * (I2 - 3) +
            0.5 * volRate * (I3 - 1 - 3 * lenRate / 4 / volRate) * (I3 - 1 - 3 * lenRate / 4 / volRate) - 0.5 * lenRate * log(I2 + 1) /*- (0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate) - 0.5 * lenRate * log(4.0))*/) *
           volume;
    // printf("I2   I3   ler  volr\n", I2, I3, lenRate, volRate);
}

__device__ Scalar __cal_ARAP_energy_3D(const Scalar3* vertexes, const uint4& tetrahedra,
                                       const __MATHUTILS__::Matrix3x3S& DmInverse,
                                       const Scalar& volume, const Scalar& lenRate) {
    __MATHUTILS__::Matrix3x3S Ds;
    __calculateDms3D_Scalar(vertexes, tetrahedra, Ds);
    __MATHUTILS__::Matrix3x3S F;
    __M_Mat_multiply(Ds, DmInverse, F);
    // printf("%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n\n\n\n", F.m[0][0],
    // F.m[0][1], F.m[0][2], F.m[1][0], F.m[1][1], F.m[1][2], F.m[2][0],
    // F.m[2][1], F.m[2][2]);
    __MATHUTILS__::Matrix3x3S U, V, sigma, S, R;
    __MATHUTILS__::Matrix3x3S M_Temp0, M_Temp1;
    SVD(F, U, V, sigma);

    S = __MATHUTILS__::__M_Mat_multiply(
        __MATHUTILS__::__M_Mat_multiply(V, sigma),
        __MATHUTILS__::__Transpose3x3(V));  // V * sigma * V.transpose();
    R = __MATHUTILS__::__M_Mat_multiply(U, __MATHUTILS__::__Transpose3x3(V));
    __MATHUTILS__::Matrix3x3S g = __MATHUTILS__::__Mat3x3_minus(F, R);
    Scalar energy = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            energy += g.m[i][j] * g.m[i][j];
        }
    }
    return energy * volume * lenRate * 0.5;
    // printf("I2   I3   ler  volr\n", I2, I3, lenRate, volRate);
}

__device__ __MATHUTILS__::Matrix9x12S __computePFDsPX3D_Scalar(
    const __MATHUTILS__::Matrix3x3S& InverseDm) {
    __MATHUTILS__::Matrix9x12S matOut;
    __MATHUTILS__::__init_Mat9x12_val(matOut, 0);
    Scalar m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    Scalar p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    Scalar s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    Scalar t1 = -(m + p + s);
    Scalar t2 = -(n + q + t);
    Scalar t3 = -(o + r + u);
    matOut.m[0][0] = t1;
    matOut.m[0][3] = m;
    matOut.m[0][6] = p;
    matOut.m[0][9] = s;
    matOut.m[1][1] = t1;
    matOut.m[1][4] = m;
    matOut.m[1][7] = p;
    matOut.m[1][10] = s;
    matOut.m[2][2] = t1;
    matOut.m[2][5] = m;
    matOut.m[2][8] = p;
    matOut.m[2][11] = s;
    matOut.m[3][0] = t2;
    matOut.m[3][3] = n;
    matOut.m[3][6] = q;
    matOut.m[3][9] = t;
    matOut.m[4][1] = t2;
    matOut.m[4][4] = n;
    matOut.m[4][7] = q;
    matOut.m[4][10] = t;
    matOut.m[5][2] = t2;
    matOut.m[5][5] = n;
    matOut.m[5][8] = q;
    matOut.m[5][11] = t;
    matOut.m[6][0] = t3;
    matOut.m[6][3] = o;
    matOut.m[6][6] = r;
    matOut.m[6][9] = u;
    matOut.m[7][1] = t3;
    matOut.m[7][4] = o;
    matOut.m[7][7] = r;
    matOut.m[7][10] = u;
    matOut.m[8][2] = t3;
    matOut.m[8][5] = o;
    matOut.m[8][8] = r;
    matOut.m[8][11] = u;

    return matOut;
}

__device__ __MATHUTILS__::Matrix6x12S __computePFDsPX3D_6x12_Scalar(
    const __MATHUTILS__::Matrix2x2S& InverseDm) {
    __MATHUTILS__::Matrix6x12S matOut;
    __MATHUTILS__::__init_Mat6x12_val(matOut, 0);
    Scalar m = InverseDm.m[0][0], n = InverseDm.m[0][1];
    Scalar p = InverseDm.m[1][0], q = InverseDm.m[1][1];

    matOut.m[0][0] = -m;
    matOut.m[3][0] = -n;
    matOut.m[1][1] = -m;
    matOut.m[4][1] = -n;
    matOut.m[2][2] = -m;
    matOut.m[5][2] = -n;

    matOut.m[0][3] = -p;
    matOut.m[3][3] = -q;
    matOut.m[1][4] = -p;
    matOut.m[4][4] = -q;
    matOut.m[2][5] = -p;
    matOut.m[5][5] = -q;

    matOut.m[0][6] = p;
    matOut.m[3][6] = q;
    matOut.m[1][7] = p;
    matOut.m[4][7] = q;
    matOut.m[2][8] = p;
    matOut.m[5][8] = q;

    matOut.m[0][9] = m;
    matOut.m[3][9] = n;
    matOut.m[1][10] = m;
    matOut.m[4][10] = n;
    matOut.m[2][11] = m;
    matOut.m[5][11] = n;

    return matOut;
}

__device__ __MATHUTILS__::Matrix6x9S __computePFDsPX3D_6x9_Scalar(
    const __MATHUTILS__::Matrix2x2S& InverseDm) {
    __MATHUTILS__::Matrix6x9S matOut;
    __MATHUTILS__::__init_Mat6x9_val(matOut, 0);
    Scalar d0 = InverseDm.m[0][0], d2 = InverseDm.m[0][1];
    Scalar d1 = InverseDm.m[1][0], d3 = InverseDm.m[1][1];

    Scalar s0 = d0 + d1;
    Scalar s1 = d2 + d3;

    matOut.m[0][0] = -s0;
    matOut.m[3][0] = -s1;

    // dF / dy0
    matOut.m[1][1] = -s0;
    matOut.m[4][1] = -s1;

    // dF / dz0
    matOut.m[2][2] = -s0;
    matOut.m[5][2] = -s1;

    // dF / dx1
    matOut.m[0][3] = d0;
    matOut.m[3][3] = d2;

    // dF / dy1
    matOut.m[1][4] = d0;
    matOut.m[4][4] = d2;

    // dF / dz1
    matOut.m[2][5] = d0;
    matOut.m[5][5] = d2;

    // dF / dx2
    matOut.m[0][6] = d1;
    matOut.m[3][6] = d3;

    // dF / dy2
    matOut.m[1][7] = d1;
    matOut.m[4][7] = d3;

    // dF / dz2
    matOut.m[2][8] = d1;
    matOut.m[5][8] = d3;

    return matOut;
}

__device__ __MATHUTILS__::Matrix3x6S __computePFDsPX3D_3x6_Scalar(const Scalar& InverseDm) {
    __MATHUTILS__::Matrix3x6S matOut;
    __MATHUTILS__::__init_Mat3x6_val(matOut, 0);

    matOut.m[0][0] = -InverseDm;
    matOut.m[1][1] = -InverseDm;
    matOut.m[2][2] = -InverseDm;

    matOut.m[0][3] = InverseDm;
    matOut.m[1][4] = InverseDm;
    matOut.m[2][5] = InverseDm;

    return matOut;
}

__device__ __MATHUTILS__::Matrix9x12S __computePFDmPX3D_Scalar(
    const __MATHUTILS__::Matrix12x9S& PDmPx, const __MATHUTILS__::Matrix3x3S& Ds,
    const __MATHUTILS__::Matrix3x3S& DmInv) {
    __MATHUTILS__::Matrix9x12S DsPDminvPx;
    __MATHUTILS__::__init_Mat9x12_val(DsPDminvPx, 0);

    for (int i = 0; i < 12; i++) {
        __MATHUTILS__::Matrix3x3S PDmPxi = __MATHUTILS__::__vec9_to_Mat3x3_Scalar(PDmPx.m[i]);
        __MATHUTILS__::Matrix3x3S DsPDminvPxi;
        __MATHUTILS__::__M_Mat_multiply(
            Ds,
            __MATHUTILS__::__M_Mat_multiply(__MATHUTILS__::__M_Mat_multiply(DmInv, PDmPxi), DmInv),
            DsPDminvPxi);

        __MATHUTILS__::Vector9S tmp = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(DsPDminvPxi);

        for (int j = 0; j < 9; j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }
    }

    return DsPDminvPx;
}

__device__ __MATHUTILS__::Matrix6x12S __computePFDmPX3D_6x12_Scalar(
    const __MATHUTILS__::Matrix12x4S& PDmPx, const __MATHUTILS__::Matrix3x2S& Ds,
    const __MATHUTILS__::Matrix2x2S& DmInv) {
    __MATHUTILS__::Matrix6x12S DsPDminvPx;
    __MATHUTILS__::__init_Mat6x12_val(DsPDminvPx, 0);

    for (int i = 0; i < 12; i++) {
        __MATHUTILS__::Matrix2x2S PDmPxi = __MATHUTILS__::__vec4_to_Mat2x2_Scalar(PDmPx.m[i]);

        __MATHUTILS__::Matrix3x2S DsPDminvPxi = __MATHUTILS__::__M3x2_M2x2_Multiply(
            Ds, __MATHUTILS__::__M2x2_Mat2x2_multiply(
                    __MATHUTILS__::__M2x2_Mat2x2_multiply(DmInv, PDmPxi), DmInv));

        __MATHUTILS__::Vector6S tmp = __MATHUTILS__::__Mat3x2_to_vec6_Scalar(DsPDminvPxi);
        for (int j = 0; j < 6; j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }
    }

    return DsPDminvPx;
}

__device__ __MATHUTILS__::Matrix3x6S __computePFDmPX3D_3x6_Scalar(
    const __MATHUTILS__::Vector6S& PDmPx, const Scalar3& Ds, const Scalar& DmInv) {
    __MATHUTILS__::Matrix3x6S DsPDminvPx;
    __MATHUTILS__::__init_Mat3x6_val(DsPDminvPx, 0);

    for (int i = 0; i < 6; i++) {
        Scalar PDmPxi = PDmPx.v[i];

        Scalar3 DsPDminvPxi = __MATHUTILS__::__s_vec_multiply(Ds, ((DmInv * PDmPxi) * DmInv));
        DsPDminvPx.m[0][i] = -DsPDminvPxi.x;
        DsPDminvPx.m[1][i] = -DsPDminvPxi.y;
        DsPDminvPx.m[2][i] = -DsPDminvPxi.z;
    }

    return DsPDminvPx;
}

template <typename Scalar, int size>
__device__ void PDSNK(Eigen::Matrix<Scalar, size, size>& symMtr) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, size, size>> eigenSolver(symMtr);
    if (eigenSolver.eigenvalues()[0] >= 0.0) {
        return;
    }
    Eigen::Matrix<Scalar, size, size> D;
    D.setZero();
    int rows = size;  //((size == Eigen::Dynamic) ? symMtr.rows() : size);
    for (int i = 0; i < rows; i++) {
        if (eigenSolver.eigenvalues()[i] > 0.0) {
            D(i, i) = eigenSolver.eigenvalues()[i];
        }
    }
    symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
}

__device__ __MATHUTILS__::Matrix6x9S __computePFDmPX3D_6x9_Scalar(
    const __MATHUTILS__::Matrix9x4S& PDmPx, const __MATHUTILS__::Matrix3x2S& Ds,
    const __MATHUTILS__::Matrix2x2S& DmInv) {
    __MATHUTILS__::Matrix6x9S DsPDminvPx;
    __MATHUTILS__::__init_Mat6x9_val(DsPDminvPx, 0);

    for (int i = 0; i < 9; i++) {
        __MATHUTILS__::Matrix2x2S PDmPxi = __MATHUTILS__::__vec4_to_Mat2x2_Scalar(PDmPx.m[i]);

        __MATHUTILS__::Matrix3x2S DsPDminvPxi = __MATHUTILS__::__M3x2_M2x2_Multiply(
            Ds, __MATHUTILS__::__M2x2_Mat2x2_multiply(
                    __MATHUTILS__::__M2x2_Mat2x2_multiply(DmInv, PDmPxi), DmInv));

        __MATHUTILS__::Vector6S tmp = __MATHUTILS__::__Mat3x2_to_vec6_Scalar(DsPDminvPxi);
        for (int j = 0; j < 6; j++) {
            DsPDminvPx.m[j][i] = -tmp.v[j];
        }
    }

    return DsPDminvPx;
}

__device__ __MATHUTILS__::Matrix9x12S __computePFPX3D_Scalar(
    const __MATHUTILS__::Matrix3x3S& InverseDm) {
    __MATHUTILS__::Matrix9x12S matOut;
    __MATHUTILS__::__init_Mat9x12_val(matOut, 0);
    Scalar m = InverseDm.m[0][0], n = InverseDm.m[0][1], o = InverseDm.m[0][2];
    Scalar p = InverseDm.m[1][0], q = InverseDm.m[1][1], r = InverseDm.m[1][2];
    Scalar s = InverseDm.m[2][0], t = InverseDm.m[2][1], u = InverseDm.m[2][2];
    Scalar t1 = -(m + p + s);
    Scalar t2 = -(n + q + t);
    Scalar t3 = -(o + r + u);
    matOut.m[0][0] = t1;
    matOut.m[0][3] = m;
    matOut.m[0][6] = p;
    matOut.m[0][9] = s;
    matOut.m[1][1] = t1;
    matOut.m[1][4] = m;
    matOut.m[1][7] = p;
    matOut.m[1][10] = s;
    matOut.m[2][2] = t1;
    matOut.m[2][5] = m;
    matOut.m[2][8] = p;
    matOut.m[2][11] = s;
    matOut.m[3][0] = t2;
    matOut.m[3][3] = n;
    matOut.m[3][6] = q;
    matOut.m[3][9] = t;
    matOut.m[4][1] = t2;
    matOut.m[4][4] = n;
    matOut.m[4][7] = q;
    matOut.m[4][10] = t;
    matOut.m[5][2] = t2;
    matOut.m[5][5] = n;
    matOut.m[5][8] = q;
    matOut.m[5][11] = t;
    matOut.m[6][0] = t3;
    matOut.m[6][3] = o;
    matOut.m[6][6] = r;
    matOut.m[6][9] = u;
    matOut.m[7][1] = t3;
    matOut.m[7][4] = o;
    matOut.m[7][7] = r;
    matOut.m[7][10] = u;
    matOut.m[8][2] = t3;
    matOut.m[8][5] = o;
    matOut.m[8][8] = r;
    matOut.m[8][11] = u;
    return matOut;
}

__device__ void __project_StabbleNHK_H_3D_makePD(
    __MATHUTILS__::Matrix9x9S& H, const __MATHUTILS__::Matrix3x3S& F,
    const __MATHUTILS__::Matrix3x3S& sigma, const __MATHUTILS__::Matrix3x3S& U,
    const __MATHUTILS__::Matrix3x3S& V, const Scalar& lengthRate, const Scalar& volumRate) {
    Scalar I3 = sigma.m[0][0] * sigma.m[1][1] * sigma.m[2][2];
    Scalar Ic = sigma.m[0][0] * sigma.m[0][0] + sigma.m[1][1] * sigma.m[1][1] +
                sigma.m[2][2] * sigma.m[2][2];

    Scalar u = lengthRate, r = volumRate;

    __MATHUTILS__::Matrix9x9S H1, HJ;  //, M9_temp[2];
    __MATHUTILS__::__identify_Mat9x9(H1);
    H1 = __MATHUTILS__::__S_Mat9x9_multiply(H1, 2);
    __MATHUTILS__::Vector9S g = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(F);

    __MATHUTILS__::Vector9S gJ;
    Scalar3 gjc0 = __MATHUTILS__::__v_vec_cross(make_Scalar3(g.v[3], g.v[4], g.v[5]),
                                                make_Scalar3(g.v[6], g.v[7], g.v[8]));
    Scalar3 gjc1 = __MATHUTILS__::__v_vec_cross(make_Scalar3(g.v[6], g.v[7], g.v[8]),
                                                make_Scalar3(g.v[0], g.v[1], g.v[2]));
    Scalar3 gjc2 = __MATHUTILS__::__v_vec_cross(make_Scalar3(g.v[0], g.v[1], g.v[2]),
                                                make_Scalar3(g.v[3], g.v[4], g.v[5]));
    g = __MATHUTILS__::__s_vec9_multiply(g, 2);
    gJ.v[0] = gjc0.x;
    gJ.v[1] = gjc0.y;
    gJ.v[2] = gjc0.z;
    gJ.v[3] = gjc1.x;
    gJ.v[4] = gjc1.y;
    gJ.v[5] = gjc1.z;
    gJ.v[6] = gjc2.x;
    gJ.v[7] = gjc2.y;
    gJ.v[8] = gjc2.z;

    __MATHUTILS__::Matrix3x3S f0hat;
    __MATHUTILS__::__set_Mat_val(f0hat, 0, -F.m[2][0], F.m[1][0], F.m[2][0], 0, -F.m[0][0],
                                 -F.m[1][0], F.m[0][0], 0);

    __MATHUTILS__::Matrix3x3S f1hat;
    __MATHUTILS__::__set_Mat_val(f1hat, 0, -F.m[2][1], F.m[1][1], F.m[2][1], 0, -F.m[0][1],
                                 -F.m[1][1], F.m[0][1], 0);

    __MATHUTILS__::Matrix3x3S f2hat;
    __MATHUTILS__::__set_Mat_val(f2hat, 0, -F.m[2][2], F.m[1][2], F.m[2][2], 0, -F.m[0][2],
                                 -F.m[1][2], F.m[0][2], 0);

    HJ.m[0][0] = 0;
    HJ.m[0][1] = 0;
    HJ.m[0][2] = 0;
    HJ.m[1][0] = 0;
    HJ.m[1][1] = 0;
    HJ.m[1][2] = 0;
    HJ.m[2][0] = 0;
    HJ.m[2][1] = 0;
    HJ.m[2][2] = 0;

    HJ.m[0][3] = -f2hat.m[0][0];
    HJ.m[0][4] = -f2hat.m[0][1];
    HJ.m[0][5] = -f2hat.m[0][2];
    HJ.m[1][3] = -f2hat.m[1][0];
    HJ.m[1][4] = -f2hat.m[1][1];
    HJ.m[1][5] = -f2hat.m[1][2];
    HJ.m[2][3] = -f2hat.m[2][0];
    HJ.m[2][4] = -f2hat.m[2][1];
    HJ.m[2][5] = -f2hat.m[2][2];

    HJ.m[0][6] = f1hat.m[0][0];
    HJ.m[0][7] = f1hat.m[0][1];
    HJ.m[0][8] = f1hat.m[0][2];
    HJ.m[1][6] = f1hat.m[1][0];
    HJ.m[1][7] = f1hat.m[1][1];
    HJ.m[1][8] = f1hat.m[1][2];
    HJ.m[2][6] = f1hat.m[2][0];
    HJ.m[2][7] = f1hat.m[2][1];
    HJ.m[2][8] = f1hat.m[2][2];

    HJ.m[3][0] = f2hat.m[0][0];
    HJ.m[3][1] = f2hat.m[0][1];
    HJ.m[3][2] = f2hat.m[0][2];
    HJ.m[4][0] = f2hat.m[1][0];
    HJ.m[4][1] = f2hat.m[1][1];
    HJ.m[4][2] = f2hat.m[1][2];
    HJ.m[5][0] = f2hat.m[2][0];
    HJ.m[5][1] = f2hat.m[2][1];
    HJ.m[5][2] = f2hat.m[2][2];

    HJ.m[3][3] = 0;
    HJ.m[3][4] = 0;
    HJ.m[3][5] = 0;
    HJ.m[4][3] = 0;
    HJ.m[4][4] = 0;
    HJ.m[4][5] = 0;
    HJ.m[5][3] = 0;
    HJ.m[5][4] = 0;
    HJ.m[5][5] = 0;

    HJ.m[3][6] = -f0hat.m[0][0];
    HJ.m[3][7] = -f0hat.m[0][1];
    HJ.m[3][8] = -f0hat.m[0][2];
    HJ.m[4][6] = -f0hat.m[1][0];
    HJ.m[4][7] = -f0hat.m[1][1];
    HJ.m[4][8] = -f0hat.m[1][2];
    HJ.m[5][6] = -f0hat.m[2][0];
    HJ.m[5][7] = -f0hat.m[2][1];
    HJ.m[5][8] = -f0hat.m[2][2];

    HJ.m[6][0] = -f1hat.m[0][0];
    HJ.m[6][1] = -f1hat.m[0][1];
    HJ.m[6][2] = -f1hat.m[0][2];
    HJ.m[7][0] = -f1hat.m[1][0];
    HJ.m[7][1] = -f1hat.m[1][1];
    HJ.m[7][2] = -f1hat.m[1][2];
    HJ.m[8][0] = -f1hat.m[2][0];
    HJ.m[8][1] = -f1hat.m[2][1];
    HJ.m[8][2] = -f1hat.m[2][2];

    HJ.m[6][3] = f0hat.m[0][0];
    HJ.m[6][4] = f0hat.m[0][1];
    HJ.m[6][5] = f0hat.m[0][2];
    HJ.m[7][3] = f0hat.m[1][0];
    HJ.m[7][4] = f0hat.m[1][1];
    HJ.m[7][5] = f0hat.m[1][2];
    HJ.m[8][3] = f0hat.m[2][0];
    HJ.m[8][4] = f0hat.m[2][1];
    HJ.m[8][5] = f0hat.m[2][2];

    HJ.m[6][6] = 0;
    HJ.m[6][7] = 0;
    HJ.m[6][8] = 0;
    HJ.m[7][6] = 0;
    HJ.m[7][7] = 0;
    HJ.m[7][8] = 0;
    HJ.m[8][6] = 0;
    HJ.m[8][7] = 0;
    HJ.m[8][8] = 0;

    Scalar J = I3;
    Scalar mu = u, lambda = r;
    H = __MATHUTILS__::__Mat9x9_add(
        __MATHUTILS__::__S_Mat9x9_multiply(H1, (Ic * mu) / (2 * (Ic + 1))),
        __MATHUTILS__::__S_Mat9x9_multiply(HJ, lambda * (J - 1 - (3 * mu) / (4.0 * lambda))));

    H = __MATHUTILS__::__Mat9x9_add(
        H, __MATHUTILS__::__v9_vec9_toMat9x9(g, g, (mu / (2 * (Ic + 1) * (Ic + 1)))));

    H = __MATHUTILS__::__Mat9x9_add(H, __MATHUTILS__::__v9_vec9_toMat9x9(gJ, gJ, lambda));

    Eigen::Matrix<Scalar, 9, 9> mat9;
    for (int i = 0; i != 9; ++i)
        for (int j = 0; j != 9; ++j) mat9(i, j) = H.m[i][j];

    PDSNK<Scalar, 9>(mat9);

    for (int i = 0; i != 9; ++i)
        for (int j = 0; j != 9; ++j) H.m[i][j] = mat9(i, j);
}

__global__ void _calculate_fem_gradient_hessian(__MATHUTILS__::Matrix3x3S* DmInverses,
                                                const Scalar3* vertexes, const uint4* tetrahedras,
                                                __MATHUTILS__::Matrix12x12S* Hessians,
                                                uint32_t offset, const Scalar* volume,
                                                Scalar3* gradient, int tetrahedraNum,
                                                Scalar lenRate, Scalar volRate, Scalar IPC_dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tetrahedraNum) return;

    __MATHUTILS__::Matrix9x12S PFPX = __computePFPX3D_Scalar(DmInverses[idx]);

    __MATHUTILS__::Matrix3x3S Ds;
    __calculateDms3D_Scalar(vertexes, tetrahedras[idx], Ds);
    __MATHUTILS__::Matrix3x3S F;
    __M_Mat_multiply(Ds, DmInverses[idx], F);

    __MATHUTILS__::Matrix3x3S U, V, Sigma;
    SVD(F, U, V, Sigma);

#ifdef USE_SNK
    __MATHUTILS__::Matrix3x3S Iso_PEPF =
        __computePEPF_StableNHK3D_Scalar(F, Sigma, U, V, lenRate, volRate);
#else
    __MATHUTILS__::Matrix3x3S Iso_PEPF = computePEPF_ARAP_Scalar(F, Sigma, U, V, lenRate);
#endif

    __MATHUTILS__::Matrix3x3S PEPF = Iso_PEPF;

    __MATHUTILS__::Vector9S pepf = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(PEPF);

    __MATHUTILS__::Matrix12x9S PFPXTranspose = __MATHUTILS__::__Transpose9x12(PFPX);
    __MATHUTILS__::Vector12S f = __MATHUTILS__::__s_vec12_multiply(
        __MATHUTILS__::__M12x9_v9_multiply(PFPXTranspose, pepf), IPC_dt * IPC_dt * volume[idx]);
    // printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0],
    // f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9],
    // f.v[10], f.v[11]);

    {
        atomicAdd(&(gradient[tetrahedras[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[tetrahedras[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[tetrahedras[idx].x].z), f.v[2]);

        atomicAdd(&(gradient[tetrahedras[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[tetrahedras[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[tetrahedras[idx].y].z), f.v[5]);

        atomicAdd(&(gradient[tetrahedras[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[tetrahedras[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[tetrahedras[idx].z].z), f.v[8]);

        atomicAdd(&(gradient[tetrahedras[idx].w].x), f.v[9]);
        atomicAdd(&(gradient[tetrahedras[idx].w].y), f.v[10]);
        atomicAdd(&(gradient[tetrahedras[idx].w].z), f.v[11]);
    }

#ifdef USE_SNK
    __MATHUTILS__::Matrix9x9S Hq;
    //__project_StabbleNHK_H_3D(make_Scalar3(Sigma.m[0][0], Sigma.m[1][1],
    // Sigma.m[2][2]),
    // U, V, lenRate, volRate,Hq);

    __project_StabbleNHK_H_3D_makePD(Hq, F, Sigma, U, V, lenRate, volRate);
#else
    __MATHUTILS__::Matrix9x9S Hq = project_ARAP_H_3D(Sigma, U, V, lenRate);
#endif

    __MATHUTILS__::Matrix12x12S H;

    __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPXTranspose, Hq, H);

    Hessians[idx + offset] = __MATHUTILS__::__s_M12x12_Multiply(H, volume[idx] * IPC_dt * IPC_dt);
}

__global__ void _calculate_triangle_fem_gradient_hessian(
    __MATHUTILS__::Matrix2x2S* trimInverses, const Scalar3* vertexes, const uint3* triangles,
    __MATHUTILS__::Matrix9x9S* Hessians, uint32_t offset, const Scalar* area, Scalar3* gradient,
    int triangleNum, Scalar stretchStiff, Scalar shearhStiff, Scalar IPC_dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triangleNum) return;

    __MATHUTILS__::Matrix6x9S PFPX = __computePFDsPX3D_6x9_Scalar(trimInverses[idx]);

    __MATHUTILS__::Matrix3x2S Ds;
    __calculateDs2D_Scalar(vertexes, triangles[idx], Ds);
    __MATHUTILS__::Matrix3x2S F = __MATHUTILS__::__M3x2_M2x2_Multiply(Ds, trimInverses[idx]);

    __MATHUTILS__::Matrix3x2S PEPF =
        __computePEPF_BaraffWitkinStretch_Scalar(F, stretchStiff, shearhStiff);

    __MATHUTILS__::Vector6S pepf = __MATHUTILS__::__Mat3x2_to_vec6_Scalar(PEPF);

    __MATHUTILS__::Matrix9x6S PFPXTranspose = __MATHUTILS__::__Transpose6x9(PFPX);
    __MATHUTILS__::Vector9S f = __MATHUTILS__::__s_vec9_multiply(
        __MATHUTILS__::__M9x6_v6_multiply(PFPXTranspose, pepf), IPC_dt * IPC_dt * area[idx]);
    // printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0],
    // f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9],
    // f.v[10], f.v[11]);

    {
        atomicAdd(&(gradient[triangles[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[triangles[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[triangles[idx].x].z), f.v[2]);
        atomicAdd(&(gradient[triangles[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[triangles[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[triangles[idx].y].z), f.v[5]);
        atomicAdd(&(gradient[triangles[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[triangles[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[triangles[idx].z].z), f.v[8]);
    }

    __MATHUTILS__::Matrix6x6S Hq =
        __MATHUTILS__::__s_M6x6_Multiply(__project_BaraffWitkinStretch_H(F), stretchStiff);

    __MATHUTILS__::__Mat_add(
        Hq, __MATHUTILS__::__s_M6x6_Multiply(__project_BaraffWitkinShear_H(F), shearhStiff), Hq);

    __MATHUTILS__::Matrix9x6S M9x6_temp = __MATHUTILS__::__M9x6_M6x6_Multiply(PFPXTranspose, Hq);
    __MATHUTILS__::Matrix9x9S H = __MATHUTILS__::__M9x6_M6x9_Multiply(M9x6_temp, PFPX);
    H = __MATHUTILS__::__s_M9x9_Multiply(H, area[idx] * IPC_dt * IPC_dt);
    Hessians[idx + offset] = H;
}

__global__ void _calculate_triangle_fem_gradient(__MATHUTILS__::Matrix2x2S* trimInverses,
                                                 const Scalar3* vertexes, const uint3* triangles,
                                                 const Scalar* area, Scalar3* gradient,
                                                 int triangleNum, Scalar stretchStiff,
                                                 Scalar shearhStiff, Scalar IPC_dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triangleNum) return;

    __MATHUTILS__::Matrix6x9S PFPX = __computePFDsPX3D_6x9_Scalar(trimInverses[idx]);

    __MATHUTILS__::Matrix3x2S Ds;
    __calculateDs2D_Scalar(vertexes, triangles[idx], Ds);
    __MATHUTILS__::Matrix3x2S F = __MATHUTILS__::__M3x2_M2x2_Multiply(Ds, trimInverses[idx]);

    __MATHUTILS__::Matrix3x2S PEPF =
        __computePEPF_BaraffWitkinStretch_Scalar(F, stretchStiff, shearhStiff);

    __MATHUTILS__::Vector6S pepf = __MATHUTILS__::__Mat3x2_to_vec6_Scalar(PEPF);

    __MATHUTILS__::Matrix9x6S PFPXTranspose = __MATHUTILS__::__Transpose6x9(PFPX);
    __MATHUTILS__::Vector9S f = __MATHUTILS__::__s_vec9_multiply(
        __MATHUTILS__::__M9x6_v6_multiply(PFPXTranspose, pepf), IPC_dt * IPC_dt * area[idx]);
    // printf("%f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", f.v[0],
    // f.v[1], f.v[2], f.v[3], f.v[4], f.v[5], f.v[6], f.v[7], f.v[8], f.v[9],
    // f.v[10], f.v[11]);

    {
        atomicAdd(&(gradient[triangles[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[triangles[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[triangles[idx].x].z), f.v[2]);
        atomicAdd(&(gradient[triangles[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[triangles[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[triangles[idx].y].z), f.v[5]);
        atomicAdd(&(gradient[triangles[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[triangles[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[triangles[idx].z].z), f.v[8]);
    }
}

__global__ void _calculate_fem_gradient(__MATHUTILS__::Matrix3x3S* DmInverses,
                                        const Scalar3* vertexes, const uint4* tetrahedras,
                                        const Scalar* volume, Scalar3* gradient, int tetrahedraNum,
                                        Scalar lenRate, Scalar volRate, Scalar dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tetrahedraNum) return;

    __MATHUTILS__::Matrix9x12S PFPX = __computePFPX3D_Scalar(DmInverses[idx]);

    __MATHUTILS__::Matrix3x3S Ds;
    __calculateDms3D_Scalar(vertexes, tetrahedras[idx], Ds);
    __MATHUTILS__::Matrix3x3S F;
    __M_Mat_multiply(Ds, DmInverses[idx], F);

    __MATHUTILS__::Matrix3x3S U, V, Sigma;
    SVD(F, U, V, Sigma);
    // printf("%f %f\n\n\n", lenRate, volRate);
#ifdef USE_SNK
    __MATHUTILS__::Matrix3x3S Iso_PEPF =
        __computePEPF_StableNHK3D_Scalar(F, Sigma, U, V, lenRate, volRate);
#else
    __MATHUTILS__::Matrix3x3S Iso_PEPF = computePEPF_ARAP_Scalar(F, Sigma, U, V, lenRate);
#endif

    __MATHUTILS__::Matrix3x3S PEPF = Iso_PEPF;

    __MATHUTILS__::Vector9S pepf = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(PEPF);

    __MATHUTILS__::Matrix12x9S PFPXTranspose = __MATHUTILS__::__Transpose9x12(PFPX);

    __MATHUTILS__::Vector12S f = __MATHUTILS__::__M12x9_v9_multiply(PFPXTranspose, pepf);

    for (int i = 0; i < 12; i++) {
        f.v[i] = volume[idx] * f.v[i] * dt * dt;
    }

    {
        atomicAdd(&(gradient[tetrahedras[idx].x].x), f.v[0]);
        atomicAdd(&(gradient[tetrahedras[idx].x].y), f.v[1]);
        atomicAdd(&(gradient[tetrahedras[idx].x].z), f.v[2]);

        atomicAdd(&(gradient[tetrahedras[idx].y].x), f.v[3]);
        atomicAdd(&(gradient[tetrahedras[idx].y].y), f.v[4]);
        atomicAdd(&(gradient[tetrahedras[idx].y].z), f.v[5]);

        atomicAdd(&(gradient[tetrahedras[idx].z].x), f.v[6]);
        atomicAdd(&(gradient[tetrahedras[idx].z].y), f.v[7]);
        atomicAdd(&(gradient[tetrahedras[idx].z].z), f.v[8]);

        atomicAdd(&(gradient[tetrahedras[idx].w].x), f.v[9]);
        atomicAdd(&(gradient[tetrahedras[idx].w].y), f.v[10]);
        atomicAdd(&(gradient[tetrahedras[idx].w].z), f.v[11]);
    }
}

__global__ void _calculate_bending_gradient_hessian(
    const Scalar3* vertexes, const Scalar3* rest_vertexes, const uint2* edges,
    const uint2* edges_adj_vertex, __MATHUTILS__::Matrix12x12S* Hessians, uint4* Indices,
    uint32_t offset, Scalar3* gradient, int edgeNum, Scalar bendStiff, Scalar IPC_dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= edgeNum) return;
    uint2 edge = edges[idx];
    uint2 adj = edges_adj_vertex[idx];
    if (adj.y == -1) {
        __MATHUTILS__::Matrix12x12S Zero;
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                Zero.m[i][j] = 0;
            }
        }
        Hessians[idx + offset] = Zero;
        Indices[idx + offset] = make_uint4(0, 1, 2, 3);
        //
        return;
    }
    auto x0 = vertexes[edge.x];
    auto x1 = vertexes[edge.y];
    auto x2 = vertexes[adj.x];
    auto x3 = vertexes[adj.y];

    Eigen::Matrix<Scalar, 1, 12> grad_transpose;
    Eigen::Matrix<Scalar, 12, 12> H;

    Eigen::Matrix<Scalar, 3, 1> x0_eigen = Eigen::Matrix<Scalar, 3, 1>(x0.x, x0.y, x0.z);
    Eigen::Matrix<Scalar, 3, 1> x1_eigen = Eigen::Matrix<Scalar, 3, 1>(x1.x, x1.y, x1.z);
    Eigen::Matrix<Scalar, 3, 1> x2_eigen = Eigen::Matrix<Scalar, 3, 1>(x2.x, x2.y, x2.z);
    Eigen::Matrix<Scalar, 3, 1> x3_eigen = Eigen::Matrix<Scalar, 3, 1>(x3.x, x3.y, x3.z);
    Scalar t = edgeTheta(x0_eigen, x1_eigen, x2_eigen, x3_eigen, &grad_transpose, &H);

    auto rest_x0 = rest_vertexes[edge.x];
    auto rest_x1 = rest_vertexes[edge.y];
    auto rest_x2 = rest_vertexes[adj.x];
    auto rest_x3 = rest_vertexes[adj.y];
    Scalar length = __MATHUTILS__::__norm(__MATHUTILS__::__minus(rest_x0, rest_x1));
    Eigen::Vector3d rest_x0_eigen = Eigen::Vector3d(rest_x0.x, rest_x0.y, rest_x0.z);
    Eigen::Vector3d rest_x1_eigen = Eigen::Vector3d(rest_x1.x, rest_x1.y, rest_x1.z);
    Eigen::Vector3d rest_x2_eigen = Eigen::Vector3d(rest_x2.x, rest_x2.y, rest_x2.z);
    Eigen::Vector3d rest_x3_eigen = Eigen::Vector3d(rest_x3.x, rest_x3.y, rest_x3.z);
    Scalar rest_t =
        edgeTheta(rest_x0_eigen, rest_x1_eigen, rest_x2_eigen, rest_x3_eigen, nullptr, nullptr);

    H = 2 * ((t - rest_t) * H + grad_transpose.transpose() * grad_transpose);
    grad_transpose = 2 * (t - rest_t) * grad_transpose;
    PDSNK<Scalar, 12>(H);
    __MATHUTILS__::Vector12S f;
    for (int i = 0; i < 12; i++) {
        f.v[i] = IPC_dt * IPC_dt * length * grad_transpose(0, i) * bendStiff;
    }

    {
        atomicAdd(&(gradient[edge.x].x), f.v[0]);
        atomicAdd(&(gradient[edge.x].y), f.v[1]);
        atomicAdd(&(gradient[edge.x].z), f.v[2]);

        atomicAdd(&(gradient[edge.y].x), f.v[3]);
        atomicAdd(&(gradient[edge.y].y), f.v[4]);
        atomicAdd(&(gradient[edge.y].z), f.v[5]);

        atomicAdd(&(gradient[adj.x].x), f.v[6]);
        atomicAdd(&(gradient[adj.x].y), f.v[7]);
        atomicAdd(&(gradient[adj.x].z), f.v[8]);

        atomicAdd(&(gradient[adj.y].x), f.v[9]);
        atomicAdd(&(gradient[adj.y].y), f.v[10]);
        atomicAdd(&(gradient[adj.y].z), f.v[11]);
    }
    __MATHUTILS__::Matrix12x12S d_H;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            d_H.m[i][j] = IPC_dt * IPC_dt * length * H(i, j) * bendStiff;
        }
    }
    Hessians[idx + offset] = d_H;
    Indices[idx + offset] = make_uint4(edge.x, edge.y, adj.x, adj.y);
}

__device__ Scalar __cal_bending_energy(const Scalar3* vertexes, const Scalar3* rest_vertexes,
                                       const uint2& edge, const uint2& adj, Scalar length,
                                       Scalar bendStiff) {
    if (adj.y == -1) return 0;
    auto x0 = vertexes[edge.x];
    auto x1 = vertexes[edge.y];
    auto x2 = vertexes[adj.x];
    auto x3 = vertexes[adj.y];
    Eigen::Matrix<Scalar, 3, 1> x0_eigen = Eigen::Matrix<Scalar, 3, 1>(x0.x, x0.y, x0.z);
    Eigen::Matrix<Scalar, 3, 1> x1_eigen = Eigen::Matrix<Scalar, 3, 1>(x1.x, x1.y, x1.z);
    Eigen::Matrix<Scalar, 3, 1> x2_eigen = Eigen::Matrix<Scalar, 3, 1>(x2.x, x2.y, x2.z);
    Eigen::Matrix<Scalar, 3, 1> x3_eigen = Eigen::Matrix<Scalar, 3, 1>(x3.x, x3.y, x3.z);
    Scalar t = edgeTheta(x0_eigen, x1_eigen, x2_eigen, x3_eigen, nullptr, nullptr);
    //            cout << "t: " << t << std::endl;

    auto rest_x0 = rest_vertexes[edge.x];
    auto rest_x1 = rest_vertexes[edge.y];
    auto rest_x2 = rest_vertexes[adj.x];
    auto rest_x3 = rest_vertexes[adj.y];
    Eigen::Matrix<Scalar, 3, 1> rest_x0_eigen =
        Eigen::Matrix<Scalar, 3, 1>(rest_x0.x, rest_x0.y, rest_x0.z);
    Eigen::Matrix<Scalar, 3, 1> rest_x1_eigen =
        Eigen::Matrix<Scalar, 3, 1>(rest_x1.x, rest_x1.y, rest_x1.z);
    Eigen::Matrix<Scalar, 3, 1> rest_x2_eigen =
        Eigen::Matrix<Scalar, 3, 1>(rest_x2.x, rest_x2.y, rest_x2.z);
    Eigen::Matrix<Scalar, 3, 1> rest_x3_eigen =
        Eigen::Matrix<Scalar, 3, 1>(rest_x3.x, rest_x3.y, rest_x3.z);
    Scalar rest_t =
        edgeTheta(rest_x0_eigen, rest_x1_eigen, rest_x2_eigen, rest_x3_eigen, nullptr, nullptr);
    Scalar bend_energy = bendStiff * (t - rest_t) * (t - rest_t) * length;
    return bend_energy;
}

__global__ void _computeGroundGradientAndHessian(const Scalar3* vertexes, const Scalar* g_offset,
                                                 const Scalar3* g_normal,
                                                 const uint32_t* _environment_collisionPair,
                                                 Scalar3* gradient, uint32_t* _gpNum,
                                                 __MATHUTILS__::Matrix3x3S* H3x3, uint32_t* D1Index,
                                                 Scalar dHat, Scalar Kappa, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;

    Scalar t = dist2 - dHat;
    Scalar g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    Scalar H_b = (log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);

    // printf("H_b   dist   g_b    is  %lf  %lf  %lf\n", H_b, dist2, g_b);

    Scalar3 grad = __MATHUTILS__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        atomicAdd(&(gradient[gidx].x), grad.x);
        atomicAdd(&(gradient[gidx].y), grad.y);
        atomicAdd(&(gradient[gidx].z), grad.z);
    }

    Scalar param = 4.0 * H_b * dist2 + 2.0 * g_b;
    if (param > 0) {
        __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(normal, normal);
        __MATHUTILS__::Matrix3x3S Hpg = __MATHUTILS__::__S_Mat_multiply(nn, Kappa * param);

        int pidx = atomicAdd(_gpNum, 1);
        H3x3[pidx] = Hpg;
        D1Index[pidx] = gidx;
    }
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeBoundConstraintGradient(const Scalar3* vertexes, const Scalar3* targetVert,
                                               const uint32_t* targetInd, Scalar3* gradient,
                                               Scalar motionRate, Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    uint32_t vInd = targetInd[idx];
    Scalar x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z, a = targetVert[idx].x,
           b = targetVert[idx].y, c = targetVert[idx].z;
    // Scalar dis =
    // __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(vertexes[vInd],
    // targetVert[idx])); printf("%f\n", dis);
    Scalar d = motionRate;
    {
        atomicAdd(&(gradient[vInd].x), d * rate * rate * (x - a));
        atomicAdd(&(gradient[vInd].y), d * rate * rate * (y - b));
        atomicAdd(&(gradient[vInd].z), d * rate * rate * (z - c));
    }
}

__global__ void _computeGroundGradient(const Scalar3* vertexes, const Scalar* g_offset,
                                       const Scalar3* g_normal,
                                       const uint32_t* _environment_collisionPair,
                                       Scalar3* gradient, uint32_t* _gpNum,
                                       __MATHUTILS__::Matrix3x3S* H3x3, Scalar dHat, Scalar Kappa,
                                       int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;

    Scalar t = dist2 - dHat;
    Scalar g_b = t * std::log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    // Scalar H_b = (std::log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 /
    // (dist2 * dist2) * (t * t);
    Scalar3 grad = __MATHUTILS__::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        atomicAdd(&(gradient[gidx].x), grad.x);
        atomicAdd(&(gradient[gidx].y), grad.y);
        atomicAdd(&(gradient[gidx].z), grad.z);
    }
}

__global__ void _computeBoundConstraintGradientAndHessian(
    const Scalar3* vertexes, const Scalar3* targetVert, const uint32_t* targetInd,
    Scalar3* gradient, uint32_t* _gpNum, __MATHUTILS__::Matrix3x3S* H3x3, uint32_t* D1Index,
    Scalar motionRate, Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    uint32_t vInd = targetInd[idx];
    Scalar x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z, a = targetVert[idx].x,
           b = targetVert[idx].y, c = targetVert[idx].z;
    // Scalar dis =
    // __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(vertexes[vInd],
    // targetVert[idx])); printf("%f\n", dis);
    Scalar d = motionRate;
    {
        atomicAdd(&(gradient[vInd].x), rate * rate * d * (x - a));
        atomicAdd(&(gradient[vInd].y), rate * rate * d * (y - b));
        atomicAdd(&(gradient[vInd].z), rate * rate * d * (z - c));
    }
    __MATHUTILS__::Matrix3x3S Hpg;
    Hpg.m[0][0] = rate * rate * d;
    Hpg.m[0][1] = 0;
    Hpg.m[0][2] = 0;
    Hpg.m[1][0] = 0;
    Hpg.m[1][1] = rate * rate * d;
    Hpg.m[1][2] = 0;
    Hpg.m[2][0] = 0;
    Hpg.m[2][1] = 0;
    Hpg.m[2][2] = rate * rate * d;
    int pidx = atomicAdd(_gpNum, 1);
    H3x3[pidx] = Hpg;
    D1Index[pidx] = vInd;
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__ void _computeSoftConstraintGradientAndHessian(
    const Scalar3* vertexes, const Scalar3* softTargetPos, const uint32_t* softTargetIds,
    Scalar3* gradient, uint32_t* _gpNum, __MATHUTILS__::Matrix3x3S* H3x3, uint32_t* D1Index,
    Scalar softStiffness, Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int softidx = softTargetIds[idx];
    Scalar x = vertexes[softidx].x, y = vertexes[softidx].y, z = vertexes[softidx].z;
    Scalar a = softTargetPos[idx].x, b = softTargetPos[idx].y, c = softTargetPos[idx].z;
    Scalar dis =
        __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(vertexes[softidx], softTargetPos[idx]));

    Scalar d = softStiffness;
    {
        atomicAdd(&(gradient[softidx].x), d * rate * rate * (x - a));
        atomicAdd(&(gradient[softidx].y), d * rate * rate * (y - b));
        atomicAdd(&(gradient[softidx].z), d * rate * rate * (z - c));
    }

    __MATHUTILS__::Matrix3x3S Hpg;
    Hpg.m[0][0] = rate * rate * d;
    Hpg.m[0][1] = 0;
    Hpg.m[0][2] = 0;
    Hpg.m[1][0] = 0;
    Hpg.m[1][1] = rate * rate * d;
    Hpg.m[1][2] = 0;
    Hpg.m[2][0] = 0;
    Hpg.m[2][1] = 0;
    Hpg.m[2][2] = rate * rate * d;
    int pidx = atomicAdd(_gpNum, 1);
    H3x3[pidx] = Hpg;
    D1Index[pidx] = softidx;
}

__global__ void _computeSoftConstraintGradient(const Scalar3* vertexes,
                                                       const Scalar3* softTargetPos,
                                                       const uint32_t* softTargetIds,
                                                       Scalar3* gradient, Scalar softStiffness,
                                                       Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int softidx = softTargetIds[idx];
    Scalar x = vertexes[softidx].x, y = vertexes[softidx].y, z = vertexes[softidx].z;
    Scalar a = softTargetPos[idx].x, b = softTargetPos[idx].y, c = softTargetPos[idx].z;
    Scalar dis =
        __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(vertexes[softidx], softTargetPos[idx]));

    Scalar d = softStiffness;
    {
        atomicAdd(&(gradient[softidx].x), d * rate * rate * (x - a));
        atomicAdd(&(gradient[softidx].y), d * rate * rate * (y - b));
        atomicAdd(&(gradient[softidx].z), d * rate * rate * (z - c));
    }
}

void calculate_fem_gradient(__MATHUTILS__::Matrix3x3S* DmInverses, const Scalar3* vertexes,
                            const uint4* tetrahedras, const Scalar* volume, Scalar3* gradient,
                            int tetrahedraNum, Scalar lenRate, Scalar volRate, Scalar dt) {
    int numbers = tetrahedraNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient<<<blockNum, threadNum>>>(DmInverses, vertexes, tetrahedras, volume,
                                                     gradient, tetrahedraNum, lenRate, volRate, dt);
}

void calculate_triangle_fem_gradient(__MATHUTILS__::Matrix2x2S* triDmInverses,
                                     const Scalar3* vertexes, const uint3* triangles,
                                     const Scalar* area, Scalar3* gradient, int triangleNum,
                                     Scalar stretchStiff, Scalar shearStiff, Scalar IPC_dt) {
    int numbers = triangleNum;
    if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient<<<blockNum, threadNum>>>(triDmInverses, vertexes, triangles,
                                                              area, gradient, triangleNum,
                                                              stretchStiff, shearStiff, IPC_dt);
}

void calculate_fem_gradient_hessian(__MATHUTILS__::Matrix3x3S* DmInverses, const Scalar3* vertexes,
                                    const uint4* tetrahedras, __MATHUTILS__::Matrix12x12S* Hessians,
                                    const uint32_t& offset, const Scalar* volume, Scalar3* gradient,
                                    int tetrahedraNum, Scalar lenRate, Scalar volRate,
                                    Scalar IPC_dt) {
    int numbers = tetrahedraNum;
    if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calculate_fem_gradient_hessian<<<blockNum, threadNum>>>(
        DmInverses, vertexes, tetrahedras, Hessians, offset, volume, gradient, tetrahedraNum,
        lenRate, volRate, IPC_dt);
}

void calculate_triangle_fem_gradient_hessian(
    __MATHUTILS__::Matrix2x2S* triDmInverses, const Scalar3* vertexes, const uint3* triangles,
    __MATHUTILS__::Matrix9x9S* Hessians, const uint32_t& offset, const Scalar* area,
    Scalar3* gradient, int triangleNum, Scalar stretchStiff, Scalar shearStiff, Scalar IPC_dt) {
    int numbers = triangleNum;
    if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calculate_triangle_fem_gradient_hessian<<<blockNum, threadNum>>>(
        triDmInverses, vertexes, triangles, Hessians, offset, area, gradient, triangleNum,
        stretchStiff, shearStiff, IPC_dt);
}

void calculate_bending_gradient_hessian(const Scalar3* vertexes, const Scalar3* rest_vertexes,
                                        const uint2* edges, const uint2* edges_adj_vertex,
                                        __MATHUTILS__::Matrix12x12S* Hessians, uint4* Indices,
                                        const uint32_t& offset, Scalar3* gradient, int edgeNum,
                                        Scalar bendStiff, Scalar IPC_dt) {
    int numbers = edgeNum;
    if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calculate_bending_gradient_hessian<<<blockNum, threadNum>>>(
        vertexes, rest_vertexes, edges, edges_adj_vertex, Hessians, Indices, offset, gradient,
        edgeNum, bendStiff, IPC_dt);
}

void computeGroundGradientAndHessian(std::unique_ptr<GeometryManager>& instance,
                                     BlockHessian& blockHessian, Scalar3* _gradient) {
#ifndef USE_FRICTION
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));
#endif
    int numbers = instance->getHostGpNum();
    if (numbers < 1) {
        CUDA_SAFE_CALL(cudaMemcpy(&blockHessian.hostBHDNum, instance->getCudaGPNum(), sizeof(int),
                                  cudaMemcpyDeviceToHost));
        return;
    }
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundGradientAndHessian<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(), _gradient,
        instance->getCudaGPNum(), blockHessian.cudaH3x3, blockHessian.cudaD1Index,
        instance->getHostDHat(), instance->getHostKappa(), numbers);
    CUDA_SAFE_CALL(cudaMemcpy(&blockHessian.hostBHDNum, instance->getCudaGPNum(), sizeof(int),
                              cudaMemcpyDeviceToHost));
}

void computeGroundGradient(std::unique_ptr<GeometryManager>& instance, BlockHessian& blockHessian,
                           Scalar3* _gradient, Scalar mKappa) {
    int numbers = instance->getHostGpNum();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundGradient<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(), _gradient,
        instance->getCudaGPNum(), blockHessian.cudaH3x3, instance->getHostDHat(), mKappa, numbers);
}

void computeBoundConstraintGradientAndHessian(std::unique_ptr<GeometryManager>& instance,
                                             BlockHessian& blockHessian, Scalar3* _gradient) {
    int numbers = instance->getHostNumBoundTargets();
    if (numbers < 1) {
        CUDA_SAFE_CALL(cudaMemcpy(&blockHessian.hostBHDNum, instance->getCudaGPNum(), sizeof(int),
                                  cudaMemcpyDeviceToHost));
        return;
    }
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeBoundConstraintGradientAndHessian<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaBoundTargetVertPos(),
        instance->getCudaBoundTargetIndex(), _gradient, instance->getCudaGPNum(),
        blockHessian.cudaH3x3, blockHessian.cudaD1Index, instance->getHostSoftMotionRate(),
        instance->getHostAnimationFullRate(), instance->getHostNumBoundTargets());
    CUDA_SAFE_CALL(cudaMemcpy(&blockHessian.hostBHDNum, instance->getCudaGPNum(), sizeof(int),
                              cudaMemcpyDeviceToHost));
}

void computeBoundConstraintGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient) {
    int numbers = instance->getHostNumBoundTargets();

    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    // offset
    _computeBoundConstraintGradient<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaBoundTargetVertPos(),
        instance->getCudaBoundTargetIndex(), _gradient, instance->getHostSoftMotionRate(),
        instance->getHostAnimationFullRate(), instance->getHostNumBoundTargets());
}

void computeSoftConstraintGradientAndHessian(std::unique_ptr<GeometryManager>& instance,
                                                     BlockHessian& blockHessian,
                                                     Scalar3* _gradient) {
    int numbers = instance->getHostNumSoftTargets();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //

    _computeSoftConstraintGradientAndHessian<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaSoftTargetVertPos(),
        instance->getCudaSoftTargetIndex(), _gradient, instance->getCudaGPNum(),
        blockHessian.cudaH3x3, blockHessian.cudaD1Index, instance->getHostSoftStiffness(),
        instance->getHostAnimationFullRate(), numbers);
    CUDA_SAFE_CALL(cudaMemcpy(&blockHessian.hostBHDNum, instance->getCudaGPNum(), sizeof(int),
                              cudaMemcpyDeviceToHost));
}

void computeSoftConstraintGradient(std::unique_ptr<GeometryManager>& instance,
                                           Scalar3* _gradient) {
    int numbers = instance->getHostNumSoftTargets();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //

    _computeSoftConstraintGradient<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaSoftTargetVertPos(),
        instance->getCudaSoftTargetIndex(), _gradient, instance->getHostSoftStiffness(),
        instance->getHostAnimationFullRate(), numbers);
}

__global__ void _getFEMEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                           const uint4* tetrahedras,
                                           const __MATHUTILS__::Matrix3x3S* DmInverses,
                                           const Scalar* volume, int tetrahedraNum, Scalar lenRate,
                                           Scalar volRate) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];
    int numbers = tetrahedraNum;
    if (idx >= numbers) return;

#ifdef USE_SNK
    Scalar temp = __FEMENERGY__::__cal_StabbleNHK_energy_3D(
        vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate, volRate);
#else
    Scalar temp = __FEMENERGY__::__cal_ARAP_energy_3D(vertexes, tetrahedras[idx], DmInverses[idx],
                                                      volume[idx], lenRate);
#endif

    // printf("%f    %f\n\n\n", lenRate, volRate);
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

__global__ void _computeBoundConstraintEnergy_Reduction(Scalar* squeue, const Scalar3* vertexes,
                                                       const Scalar3* targetVert,
                                                       const uint32_t* targetInd, Scalar motionRate,
                                                       Scalar rate, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;
    uint32_t vInd = targetInd[idx];
    Scalar dis = __MATHUTILS__::__squaredNorm(__MATHUTILS__::__s_vec_multiply(
        __MATHUTILS__::__minus(vertexes[vInd], targetVert[idx]), rate));
    Scalar d = motionRate;
    Scalar temp = d * dis * 0.5;

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

__global__ void _get_triangleFEMEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                    const uint3* triangles,
                                                    const __MATHUTILS__::Matrix2x2S* triDmInverses,
                                                    const Scalar* area, int trianglesNum,
                                                    Scalar stretchStiff, Scalar shearStiff) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];
    int numbers = trianglesNum;
    if (idx >= numbers) return;

    Scalar temp = __FEMENERGY__::__cal_BaraffWitkinStretch_energy(
        vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);

    // printf("%f    %f\n\n\n", lenRate, volRate);
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

__global__ void _getRestStableNHKEnergy_Reduction_3D(Scalar* squeue, const Scalar* volume,
                                                     int tetrahedraNum, Scalar lenRate,
                                                     Scalar volRate) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];
    int numbers = tetrahedraNum;
    if (idx >= numbers) return;

    Scalar temp = ((0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate) -
                    0.5 * lenRate * log(4.0))) *
                  volume[idx];

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

__global__ void _computeSoftConstraintEnergy_Reduction(
    Scalar* squeue, const Scalar3* vertexes, const Scalar3* softTargetPos,
    const uint32_t* softTargetIds, Scalar softStiffness, Scalar rate, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep[];
    if (idx >= numbers) return;

    int softidx = softTargetIds[idx];
    Scalar x = vertexes[softidx].x, y = vertexes[softidx].y, z = vertexes[softidx].z;
    Scalar a = softTargetPos[idx].x, b = softTargetPos[idx].y, c = softTargetPos[idx].z;
    Scalar dis = __MATHUTILS__::__squaredNorm(__MATHUTILS__::__s_vec_multiply(
        __MATHUTILS__::__minus(make_Scalar3(x, y, z), make_Scalar3(a, b, c)), rate));

    Scalar d = softStiffness;
    Scalar temp = d * dis * 0.5;
    // printf("dis energy: %f, idx: %d", temp, idx);

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
        // printf("dis energy: %f, idx: %d", temp, idx);
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _getBendingEnergy_Reduction(Scalar* squeue, const Scalar3* vertexes,
                                            const Scalar3* rest_vertexex, const uint2* edges,
                                            const uint2* edge_adj_vertex, int edgesNum,
                                            Scalar bendStiff) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];
    int numbers = edgesNum;
    if (idx >= numbers) return;

    // Scalar temp = __cal_BaraffWitkinStretch_energy(vertexes, triangles[idx],
    // triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    //  Scalar temp = __cal_hc_cloth_energy(vertexes, triangles[idx],
    //  triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    uint2 adj = edge_adj_vertex[idx];
    Scalar3 rest_x0 = rest_vertexex[edges[idx].x];
    Scalar3 rest_x1 = rest_vertexex[edges[idx].y];
    Scalar length = __MATHUTILS__::__norm(__MATHUTILS__::__minus(rest_x0, rest_x1));
    Scalar temp = __FEMENERGY__::__cal_bending_energy(vertexes, rest_vertexex, edges[idx], adj,
                                                      length, bendStiff);
    // Scalar temp = 0;
    // printf("%f    %f\n\n\n", lenRate, volRate);
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

};  // namespace __FEMENERGY__
