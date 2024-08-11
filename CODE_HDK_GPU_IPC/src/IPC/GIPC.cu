#include "GIPC.cuh"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

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



__global__
void _reduct_max_double3_to_double(const double3* _double3Dim, double* _double1Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 tempMove = _double3Dim[idx];

    double temp = MATHUTILS::__m_max(MATHUTILS::__m_max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = MATHUTILS::__m_max(temp, tempMin);
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
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = MATHUTILS::__m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        _double1Dim[blockIdx.x] = temp;
    }
}

__global__
void _reduct_min_double(double* _double1Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = MATHUTILS::__m_min(temp, tempMin);
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
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = MATHUTILS::__m_min(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        _double1Dim[blockIdx.x] = temp;
    }
}

__global__
void _reduct_M_double2(double2* _double2Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double2 sdata[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double2 temp = _double2Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        double tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = MATHUTILS::__m_max(temp.x, tempMin);
        temp.y = MATHUTILS::__m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            double tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = MATHUTILS::__m_max(temp.x, tempMin);
            temp.y = MATHUTILS::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _double2Dim[blockIdx.x] = temp;
    }
}

__global__
void _reduct_max_double(double* _double1Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = _double1Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        double tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = MATHUTILS::__m_max(temp, tempMax);
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
            double tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = MATHUTILS::__m_max(temp, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _double1Dim[blockIdx.x] = temp;
    }
}

__device__
double __cal_Barrier_energy(const double3* _vertexes, const double3* _rest_vertexes, int4 MMCVIDI, double _Kappa, double _dHat) {
    double dHat_sqrt = sqrt(_dHat);
    double dHat = _dHat;
    double Kappa = _Kappa;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
            double dis;
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
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
        else {
            //return 0;
            MMCVIDI.w = -MMCVIDI.w - 1;
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
            double I1 = c * c;
            if (I1 == 0) return 0;
            double dis;
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            double I2 = dis / dHat;
            double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
            double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
            if (Energy < 0)
                printf("I am pee\n");
            return Energy;
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
                if (I1 == 0) return 0;
                double dis;
                MATHUTILS::__distancePointPoint(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.w]);
#if (RANK == 1)
                double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
                double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if (Energy < 0)
                    printf("I am pp\n");
                return Energy;
            }
            else {
                double dis;
                MATHUTILS::__distancePointPoint(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
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
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                //MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;

                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
                double I1 = c * c;
                if (I1 == 0) return 0;
                double dis;
                MATHUTILS::__distancePointEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I2 = dis / dHat;
                double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z]);
#if (RANK == 1)
                double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * -(dHat - dHat * I2) * (dHat - dHat * I2) * log(I2);
#elif (RANK == 2)
                double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
#elif (RANK == 4)
                double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) * log(I2) * log(I2);
#elif (RANK == 6)
                double Energy = Kappa * (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1) * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2) * log(I2) * log(I2) * log(I2) * log(I2);
#endif
                if (Energy < 0)
                    printf("I am ppe\n");
                return Energy;
            }
            else {
                double dis;
                MATHUTILS::__distancePointEdge(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I5 = dis / dHat;

                double lenE = (dis - dHat);
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
        else {
            double dis;
            MATHUTILS::__distancePointTriangle(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            double I5 = dis / dHat;

            double lenE = (dis - dHat);
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


__device__
bool segTriIntersect(const double3& ve0, const double3& ve1,
    const double3& vt0, const double3& vt1, const double3& vt2)
{

    //printf("check for tri and lines\n");

    MATHUTILS::Matrix3x3d coefMtr;
    double3 col0 = MATHUTILS::__minus(vt1, vt0);
    double3 col1 = MATHUTILS::__minus(vt2, vt0);
    double3 col2 = MATHUTILS::__minus(ve0, ve1);

    MATHUTILS::__set_Mat_val_column(coefMtr, col0, col1, col2);

    double3 n = MATHUTILS::__v_vec_cross(col0, col1);
    if (MATHUTILS::__v_vec_dot(n, MATHUTILS::__minus(ve0, vt0)) * MATHUTILS::__v_vec_dot(n, MATHUTILS::__minus(ve1, vt0)) > 0) {
        return false;
    }

    double det = MATHUTILS::__Determiant(coefMtr);

    if (abs(det) < 1e-20) {
        return false;
    }

    MATHUTILS::Matrix3x3d D1, D2, D3;
    double3 b = MATHUTILS::__minus(ve0, vt0);

    MATHUTILS::__set_Mat_val_column(D1, b, col1, col2);
    MATHUTILS::__set_Mat_val_column(D2, col0, b, col2);
    MATHUTILS::__set_Mat_val_column(D3, col0, col1, b);

    double uvt[3];
    uvt[0] = MATHUTILS::__Determiant(D1) / det;
    uvt[1] = MATHUTILS::__Determiant(D2) / det;
    uvt[2] = MATHUTILS::__Determiant(D3) / det;

    if (uvt[0] >= 0.0 && uvt[1] >= 0.0 && uvt[0] + uvt[1] <= 1.0 && uvt[2] >= 0.0 && uvt[2] <= 1.0) {
        return true;
    }
    else {
        return false;
    }
}

__device__ __host__
inline bool _overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept
{
    if ((rhs.m_lower.x - lhs.m_upper.x) >= gapL || (lhs.m_lower.x - rhs.m_upper.x) >= gapL) return false;
    if ((rhs.m_lower.y - lhs.m_upper.y) >= gapL || (lhs.m_lower.y - rhs.m_upper.y) >= gapL) return false;
    if ((rhs.m_lower.z - lhs.m_upper.z) >= gapL || (lhs.m_lower.z - rhs.m_upper.z) >= gapL) return false;
    return true;
}

__device__
double _selfConstraintVal(const double3* vertexes, const int4& active) {
    double val;
    if (active.x >= 0) {
        if (active.w >= 0) {
            MATHUTILS::__distanceEdgeEdge(vertexes[active.x], vertexes[active.y], vertexes[active.z], vertexes[active.w], val);
        }
        else {
            MATHUTILS::__distanceEdgeEdge(vertexes[active.x], vertexes[active.y], vertexes[active.z], vertexes[-active.w - 1], val);
        }
    }
    else {
        if (active.z < 0) {
            if (active.y < 0) {
                MATHUTILS::__distancePointPoint(vertexes[-active.x - 1], vertexes[-active.y - 1], val);
            }
            else {
                MATHUTILS::__distancePointPoint(vertexes[-active.x - 1], vertexes[active.y], val);
            }
        }
        else if (active.w < 0) {
            if (active.y < 0) {
                MATHUTILS::__distancePointEdge(vertexes[-active.x - 1], vertexes[-active.y - 1], vertexes[active.z], val);
            }
            else {
                MATHUTILS::__distancePointEdge(vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z], val);
            }
        }
        else {
            MATHUTILS::__distancePointTriangle(vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z], vertexes[active.w], val);
        }
    }
    return val;
}

__device__
double _computeInjectiveStepSize_3d(const double3* verts, const double3* mv, const uint32_t& v0, const uint32_t& v1, const uint32_t& v2, const uint32_t& v3, double ratio, double errorRate) {

    double x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4;
    double p1, p2, p3, p4, q1, q2, q3, q4, r1, r2, r3, r4;
    double a, b, c, d, t;


    x1 = verts[v0].x;
    x2 = verts[v1].x;
    x3 = verts[v2].x;
    x4 = verts[v3].x;

    y1 = verts[v0].y;
    y2 = verts[v1].y;
    y3 = verts[v2].y;
    y4 = verts[v3].y;

    z1 = verts[v0].z;
    z2 = verts[v1].z;
    z3 = verts[v2].z;
    z4 = verts[v3].z;

    int _3Fii0 = v0 * 3;
    int _3Fii1 = v1 * 3;
    int _3Fii2 = v2 * 3;
    int _3Fii3 = v3 * 3;

    p1 = -mv[v0].x;
    p2 = -mv[v1].x;
    p3 = -mv[v2].x;
    p4 = -mv[v3].x;

    q1 = -mv[v0].y;
    q2 = -mv[v1].y;
    q3 = -mv[v2].y;
    q4 = -mv[v3].y;

    r1 = -mv[v0].z;
    r2 = -mv[v1].z;
    r3 = -mv[v2].z;
    r4 = -mv[v3].z;

    a = -p1 * q2 * r3 + p1 * r2 * q3 + q1 * p2 * r3 - q1 * r2 * p3 - r1 * p2 * q3 + r1 * q2 * p3 + p1 * q2 * r4 - p1 * r2 * q4 - q1 * p2 * r4 + q1 * r2 * p4 + r1 * p2 * q4 - r1 * q2 * p4 - p1 * q3 * r4 + p1 * r3 * q4 + q1 * p3 * r4 - q1 * r3 * p4 - r1 * p3 * q4 + r1 * q3 * p4 + p2 * q3 * r4 - p2 * r3 * q4 - q2 * p3 * r4 + q2 * r3 * p4 + r2 * p3 * q4 - r2 * q3 * p4;
    b = -x1 * q2 * r3 + x1 * r2 * q3 + y1 * p2 * r3 - y1 * r2 * p3 - z1 * p2 * q3 + z1 * q2 * p3 + x2 * q1 * r3 - x2 * r1 * q3 - y2 * p1 * r3 + y2 * r1 * p3 + z2 * p1 * q3 - z2 * q1 * p3 - x3 * q1 * r2 + x3 * r1 * q2 + y3 * p1 * r2 - y3 * r1 * p2 - z3 * p1 * q2 + z3 * q1 * p2 + x1 * q2 * r4 - x1 * r2 * q4 - y1 * p2 * r4 + y1 * r2 * p4 + z1 * p2 * q4 - z1 * q2 * p4 - x2 * q1 * r4 + x2 * r1 * q4 + y2 * p1 * r4 - y2 * r1 * p4 - z2 * p1 * q4 + z2 * q1 * p4 + x4 * q1 * r2 - x4 * r1 * q2 - y4 * p1 * r2 + y4 * r1 * p2 + z4 * p1 * q2 - z4 * q1 * p2 - x1 * q3 * r4 + x1 * r3 * q4 + y1 * p3 * r4 - y1 * r3 * p4 - z1 * p3 * q4 + z1 * q3 * p4 + x3 * q1 * r4 - x3 * r1 * q4 - y3 * p1 * r4 + y3 * r1 * p4 + z3 * p1 * q4 - z3 * q1 * p4 - x4 * q1 * r3 + x4 * r1 * q3 + y4 * p1 * r3 - y4 * r1 * p3 - z4 * p1 * q3 + z4 * q1 * p3 + x2 * q3 * r4 - x2 * r3 * q4 - y2 * p3 * r4 + y2 * r3 * p4 + z2 * p3 * q4 - z2 * q3 * p4 - x3 * q2 * r4 + x3 * r2 * q4 + y3 * p2 * r4 - y3 * r2 * p4 - z3 * p2 * q4 + z3 * q2 * p4 + x4 * q2 * r3 - x4 * r2 * q3 - y4 * p2 * r3 + y4 * r2 * p3 + z4 * p2 * q3 - z4 * q2 * p3;
    c = -x1 * y2 * r3 + x1 * z2 * q3 + x1 * y3 * r2 - x1 * z3 * q2 + y1 * x2 * r3 - y1 * z2 * p3 - y1 * x3 * r2 + y1 * z3 * p2 - z1 * x2 * q3 + z1 * y2 * p3 + z1 * x3 * q2 - z1 * y3 * p2 - x2 * y3 * r1 + x2 * z3 * q1 + y2 * x3 * r1 - y2 * z3 * p1 - z2 * x3 * q1 + z2 * y3 * p1 + x1 * y2 * r4 - x1 * z2 * q4 - x1 * y4 * r2 + x1 * z4 * q2 - y1 * x2 * r4 + y1 * z2 * p4 + y1 * x4 * r2 - y1 * z4 * p2 + z1 * x2 * q4 - z1 * y2 * p4 - z1 * x4 * q2 + z1 * y4 * p2 + x2 * y4 * r1 - x2 * z4 * q1 - y2 * x4 * r1 + y2 * z4 * p1 + z2 * x4 * q1 - z2 * y4 * p1 - x1 * y3 * r4 + x1 * z3 * q4 + x1 * y4 * r3 - x1 * z4 * q3 + y1 * x3 * r4 - y1 * z3 * p4 - y1 * x4 * r3 + y1 * z4 * p3 - z1 * x3 * q4 + z1 * y3 * p4 + z1 * x4 * q3 - z1 * y4 * p3 - x3 * y4 * r1 + x3 * z4 * q1 + y3 * x4 * r1 - y3 * z4 * p1 - z3 * x4 * q1 + z3 * y4 * p1 + x2 * y3 * r4 - x2 * z3 * q4 - x2 * y4 * r3 + x2 * z4 * q3 - y2 * x3 * r4 + y2 * z3 * p4 + y2 * x4 * r3 - y2 * z4 * p3 + z2 * x3 * q4 - z2 * y3 * p4 - z2 * x4 * q3 + z2 * y4 * p3 + x3 * y4 * r2 - x3 * z4 * q2 - y3 * x4 * r2 + y3 * z4 * p2 + z3 * x4 * q2 - z3 * y4 * p2;
    d = (ratio) * (x1 * z2 * y3 - x1 * y2 * z3 + y1 * x2 * z3 - y1 * z2 * x3 - z1 * x2 * y3 + z1 * y2 * x3 + x1 * y2 * z4 - x1 * z2 * y4 - y1 * x2 * z4 + y1 * z2 * x4 + z1 * x2 * y4 - z1 * y2 * x4 - x1 * y3 * z4 + x1 * z3 * y4 + y1 * x3 * z4 - y1 * z3 * x4 - z1 * x3 * y4 + z1 * y3 * x4 + x2 * y3 * z4 - x2 * z3 * y4 - y2 * x3 * z4 + y2 * z3 * x4 + z2 * x3 * y4 - z2 * y3 * x4);


    //printf("a b c d:   %f  %f  %f  %f     %f     %f,    id0, id1, id2, id3:  %d  %d  %d  %d\n", a, b, c, d, ratio, errorRate, v0, v1, v2, v3);
    if (abs(a) <= errorRate /** errorRate*/) {
        if (abs(b) <= errorRate /** errorRate*/) {
            if (false && abs(c) <= errorRate) {
                t = 1;
            }
            else {
                t = -d / c;
            }
        }
        else {
            double desc = c * c - 4 * b * d;
            if (desc > 0) {
                t = (-c - sqrt(desc)) / (2 * b);
                if (t < 0)
                    t = (-c + sqrt(desc)) / (2 * b);
            }
            else
                t = 1;
        }
    }
    else {
        //double results[3];
        //int number = 0;
        //MATHUTILS::__NewtonSolverForCubicEquation(a, b, c, d, results, number, errorRate);

        //t = 1;
        //for (int index = 0;index < number;index++) {
        //    if (results[index] > 0 && results[index] < t) {
        //        t = results[index];
        //    }
        //}
        zs::complex<double> i(0, 1);
        zs::complex<double> delta0(b * b - 3 * a * c, 0);
        zs::complex<double> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
        zs::complex<double> C = pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
        if (abs(C) == 0.0) {
            // a corner case listed by wikipedia found by our collaborate from another project
            C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
        }

        zs::complex<double> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
        zs::complex<double> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;

        zs::complex<double> t1 = (b + C + delta0 / C) / (-3.0 * a);
        zs::complex<double> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
        zs::complex<double> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
        t = -1;
        if ((abs(imag(t1)) < errorRate /** errorRate*/) && (real(t1) > 0))
            t = real(t1);
        if ((abs(imag(t2)) < errorRate /** errorRate*/) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
            t = real(t2);
        if ((abs(imag(t3)) < errorRate /** errorRate*/) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
            t = real(t3);
    }
    if (t <= 0)
        t = 1;
    return t;
}

__device__
double __cal_Friction_gd_energy(const double3* _vertexes, const double3* _o_vertexes, const double3* _normal, uint32_t gidx,
    double dt, double lastH, double eps) {

    double3 normal = *_normal;
    double3 Vdiff = MATHUTILS::__minus(_vertexes[gidx], _o_vertexes[gidx]);
    double3 VProj = MATHUTILS::__minus(Vdiff, MATHUTILS::__s_vec_multiply(normal, MATHUTILS::__v_vec_dot(Vdiff, normal)));
    double VProjMag2 = MATHUTILS::__squaredNorm(VProj);
    if (VProjMag2 > eps * eps) {
        return lastH * (sqrt(VProjMag2) - eps * 0.5);

    }
    else {
        return lastH * VProjMag2 / eps * 0.5;
    }
}


__device__
double __cal_Friction_energy(const double3* _vertexes, const double3* _o_vertexes, int4 MMCVIDI,
    double dt, double2 distCoord, MATHUTILS::Matrix3x2d tanBasis, double lastH, double fricDHat, double eps) {
    double3 relDX3D;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
            IPCFRICTION::computeRelDX_EE(MATHUTILS::__minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord.x,
                distCoord.y, relDX3D);
        }
    }
    else {
        int v0I = -MMCVIDI.x - 1;
        if (MMCVIDI.z < 0) {
            if (MMCVIDI.y >= 0) {
                IPCFRICTION::computeRelDX_PP(MATHUTILS::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), relDX3D);
            }
        }
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y >= 0) {
                IPCFRICTION::computeRelDX_PE(MATHUTILS::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                    MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                    MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]), distCoord.x,
                    relDX3D);
            }
        }
        else {
            IPCFRICTION::computeRelDX_PT(MATHUTILS::__minus(_vertexes[v0I], _o_vertexes[v0I]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
                MATHUTILS::__minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord.x,
                distCoord.y, relDX3D);
        }
    }
    MATHUTILS::Matrix2x3d tB_T = MATHUTILS::__Transpose3x2(tanBasis);
    double relDXSqNorm = MATHUTILS::__squaredNorm(MATHUTILS::__M2x3_v3_multiply(tB_T, relDX3D));
    if (relDXSqNorm > fricDHat) {
        return lastH * sqrt(relDXSqNorm);
    }
    else {
        double f0;
        IPCFRICTION::f0_SF(relDXSqNorm, eps, f0);
        return lastH * f0;
    }
}

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

            MATHUTILS::Matrix6x6d HessianBlock = MATHUTILS::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T),
                coef * lastH[idx]);

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

            MATHUTILS::Matrix9x9d HessianBlock = MATHUTILS::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T),
                coef * lastH[idx]);
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

            MATHUTILS::Matrix12x12d HessianBlock = MATHUTILS::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T),
                coef * lastH[idx]);
            int Hidx = atomicAdd(_cpNum + 4, 1);
            Hidx += offset4;
            H12x12[Hidx] = HessianBlock;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

        }
    }
}

__global__
void _calBarrierHessian(const double3* _vertexes, const double3* _rest_vertexes, const int4* _collisionPair, MATHUTILS::Matrix12x12d* H12x12, MATHUTILS::Matrix9x9d* H9x9,
    MATHUTILS::Matrix6x6d* H6x6, uint4* D4Index, uint3* D3Index, uint2* D2Index, uint32_t* _cpNum, int* matIndex, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _collisionPair[idx];
    //double dHat = 1e-6;
    double dHat_sqrt = sqrt(dHat);

    //double Kappa = 1;
    // int4 MMCVIDI = make_int4(-4,0,-1,-2);
    // double3 _vertexes[4];
    // _vertexes[0] = make_double3(0,0,0);
    // _vertexes[1] = make_double3(2e-3,0,0);
    // _vertexes[2] = make_double3(1e-3,0,sqrt(3)*1e-3);
    // _vertexes[3] = make_double3(0e-3, 0.25*1e-3, sqrt(3)*0.0*1e-3);
    double gassThreshold = 0.0001;
    if (MMCVIDI.x >= 0) {
        if (MMCVIDI.w >= 0) {
#ifdef NEWF
            double dis;
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            dis = sqrt(dis);
            //double d_hat_sqrt = sqrt(dHat);
            MATHUTILS::Matrix12x9d PFPxT;
            GIPCPDERIV::pFpx_ee2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt,
                PFPxT);
            double I5 = pow(dis / dHat_sqrt, 2);
            MATHUTILS::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
            q0.v[8] = 1;
            //q0 = MATHUTILS::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            MATHUTILS::Matrix9x9d H;
            MATHUTILS::__init_Mat9x9(H, 0);
#else
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
            double3 v2 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
            MATHUTILS::Matrix3x3d Ds;
            MATHUTILS::__set_Mat_val_column(Ds, v0, v1, v2);
            double3 normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(v0, MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z])));
            double dis = MATHUTILS::__v_vec_dot(v1, normal);
            //MATHUTILS::Matrix12x9d PDmPx;
            if (dis < 0) {
                //is_flip = true;
                normal = make_double3(-normal.x, -normal.y, -normal.z);
                dis = -dis;
                //pDmpx_ee_flip(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
                //printf("------------ee_flip\n");
            }
            else {
                //pDmpx_ee(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PDmPx);
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
            H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);

            MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxT, H), MATHUTILS::__Transpose12x9(PFPxT));

            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else
            MATHUTILS::Matrix3x3d Q0;

            MATHUTILS::Matrix3x3d fnn;

            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);

            MATHUTILS::__M_Mat_multiply(F, nn, fnn);

            MATHUTILS::Vector9 q0 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);

            q0 = MATHUTILS::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            MATHUTILS::Matrix9x9d H;
            MATHUTILS::__init_Mat9x9(H, 0);

            H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);

            MATHUTILS::Matrix12x9d PFPxTransPos = MATHUTILS::__Transpose9x12(PFPx);
            MATHUTILS::Matrix12x12d Hessian = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif
            int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

        }
        else {
            //return;
            MMCVIDI.w = -MMCVIDI.w - 1;
            //printf("pee condition  ***************************************\n: %d  %d  %d  %d\n***************************************\n", MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
            double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
            double I1 = c * c;
            if (I1 == 0) return;
            MATHUTILS::Matrix12x9d PFPx;
            GIPCPDERIV::pFpx_pee(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

            double dis;
            MATHUTILS::__distanceEdgeEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            double I2 = dis / dHat;
            dis = sqrt(dis);

            MATHUTILS::Matrix3x3d F;
            MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
            double3 n1 = make_double3(0, 1, 0);
            double3 n2 = make_double3(0, 0, 1);

            double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);

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
            MATHUTILS::Matrix3x3d fnn;
            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
            MATHUTILS::__M_Mat_multiply(F, nn, fnn);
            MATHUTILS::Vector9 q10 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
            q10 = MATHUTILS::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

            MATHUTILS::Matrix3x3d Tx, Ty, Tz;
            MATHUTILS::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
            MATHUTILS::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
            MATHUTILS::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

            double ratio = 1.f / sqrt(2.f);
            Tx = __S_Mat_multiply(Tx, ratio);
            Ty = __S_Mat_multiply(Ty, ratio);
            Tz = __S_Mat_multiply(Tz, ratio);

            MATHUTILS::Vector9 q11 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tx, fnn));
            MATHUTILS::__normalized_vec9_double(q11);
            MATHUTILS::Vector9 q12 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tz, fnn));
            //MATHUTILS::__s_vec9_multiply(q12, c);
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
            nn = MATHUTILS::__v_vec_toMat(n2, n2);
            MATHUTILS::__M_Mat_multiply(F, nn, fnn);
            MATHUTILS::Vector9 q20 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
            q20 = MATHUTILS::__s_vec9_multiply(q20, 1.0 / sqrt(I2));



#if (RANK == 1)
            double lambdag1g = Kappa * 4 * c * F.m[2][2] * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x));
#elif (RANK == 2)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
            double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif
#ifndef MAKEPD
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
#else
            MatT2x2 F_{ lambda10, lambdag1g, lambdag1g, lambda20 };
            make_pd(F_);
            MATHUTILS::__init_Mat9x9(M9_temp,0);
            M9_temp.m[4][4] = F_(0,0);
            M9_temp.m[4][8] = F_(0,1);
            M9_temp.m[8][4] = F_(1,0);
            M9_temp.m[8][8] = F_(1,1);
            projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);
#endif

            //MATHUTILS::Matrix9x12d PFPxTransPos = MATHUTILS::__Transpose12x9(PFPx);
            MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
            int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
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
                //printf("ppp condition  ***************************************\n: %d  %d  %d  %d\n***************************************\n", MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
                double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
                double I1 = c * c;
                if (I1 == 0) return;
                MATHUTILS::Matrix12x9d PFPx;
                GIPCPDERIV::pFpx_ppp(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

                double dis;
                MATHUTILS::__distancePointPoint(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
                double I2 = dis / dHat;
                dis = sqrt(dis);

                MATHUTILS::Matrix3x3d F;
                MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.w]);

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
                MATHUTILS::Matrix3x3d fnn;
                MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
                MATHUTILS::__M_Mat_multiply(F, nn, fnn);
                MATHUTILS::Vector9 q10 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
                q10 = MATHUTILS::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

                MATHUTILS::Matrix3x3d Tx, Ty, Tz;
                MATHUTILS::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                MATHUTILS::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                MATHUTILS::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                double ratio = 1.f / sqrt(2.f);
                Tx = __S_Mat_multiply(Tx, ratio);
                Ty = __S_Mat_multiply(Ty, ratio);
                Tz = __S_Mat_multiply(Tz, ratio);

                MATHUTILS::Vector9 q11 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tx, fnn));
                MATHUTILS::__normalized_vec9_double(q11);
                MATHUTILS::Vector9 q12 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tz, fnn));
                //MATHUTILS::__s_vec9_multiply(q12, c);
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
                nn = MATHUTILS::__v_vec_toMat(n2, n2);
                MATHUTILS::__M_Mat_multiply(F, nn, fnn);
                MATHUTILS::Vector9 q20 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
                q20 = MATHUTILS::__s_vec9_multiply(q20, 1.0 / sqrt(I2));



#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2] * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif
#ifndef MAKEPD
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
#else
                MatT2x2 F_{ lambda10, lambdag1g, lambdag1g, lambda20 };
            make_pd(F_);
            MATHUTILS::__init_Mat9x9(M9_temp,0);
            M9_temp.m[4][4] = F_(0,0);
            M9_temp.m[4][8] = F_(0,1);
            M9_temp.m[8][4] = F_(1,0);
            M9_temp.m[8][8] = F_(1,1);
            projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);
#endif

                //MATHUTILS::Matrix9x12d PFPxTransPos = MATHUTILS::__Transpose12x9(PFPx);
                MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);

                H12x12[Hidx] = Hessian;
                D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

            }
            else {
#ifdef NEWF
                double dis;
                MATHUTILS::__distancePointPoint(_vertexes[v0I], _vertexes[MMCVIDI.y], dis);
                dis = sqrt(dis);
                double d_hat_sqrt = sqrt(dHat);
                MATHUTILS::Vector6 PFPxT;
                GIPCPDERIV::pFpx_pp2(_vertexes[v0I], _vertexes[MMCVIDI.y], d_hat_sqrt,
                    PFPxT);
                double I5 = pow(dis / d_hat_sqrt, 2);
                //double q0 = 1;
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
                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else {
                    //pDmpx_pp(_vertexes[v0I], _vertexes[MMCVIDI.y], dHat_sqrt, PDmPx);
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
                double Dm = u0;
                double DmInv = 1 / u0;

                double3 F = MATHUTILS::__s_vec_multiply(Ds, DmInv);
                double I5 = MATHUTILS::__squaredNorm(F);

                double3 fnn = F;

                MATHUTILS::Matrix3x6d PFPx = __computePFDsPX3D_3x6_double(DmInv);
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
                double H = lambda0;
                MATHUTILS::Matrix6x6d Hessian = MATHUTILS::__s_M6x6_Multiply(MATHUTILS::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);
#else
                double3 q0 = MATHUTILS::__s_vec_multiply(F, 1 / sqrt(I5));

                MATHUTILS::Matrix3x3d H = MATHUTILS::__S_Mat_multiply(MATHUTILS::__v_vec_toMat(q0, q0), lambda0);//lambda0 * q0 * q0.transpose();

                MATHUTILS::Matrix6x3d PFPxTransPos = MATHUTILS::__Transpose3x6(PFPx);
                MATHUTILS::Matrix6x6d Hessian = MATHUTILS::__M6x3_M3x6_Multiply(MATHUTILS::__M6x3_M3x3_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);

                H6x6[Hidx] = Hessian;
                D2Index[Hidx] = make_uint2(v0I, MMCVIDI.y);
            }

        }
        else if (MMCVIDI.w < 0) {
            if (MMCVIDI.y < 0) {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                MMCVIDI.x = v0I;
                //printf("ppe condition  ***************************************\n: %d  %d  %d  %d\n***************************************\n", MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
                double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
                double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
                double c = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v0, v1)) /*/ MATHUTILS::__norm(v0)*/;
                double I1 = c * c;
                if (I1 == 0) return;
                MATHUTILS::Matrix12x9d PFPx;
                GIPCPDERIV::pFpx_ppe(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dHat_sqrt, PFPx);

                double dis;
                MATHUTILS::__distancePointEdge(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
                double I2 = dis / dHat;
                dis = sqrt(dis);

                MATHUTILS::Matrix3x3d F;
                MATHUTILS::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis / dHat_sqrt);
                double3 n1 = make_double3(0, 1, 0);
                double3 n2 = make_double3(0, 0, 1);

                double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w], _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z]);

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
                MATHUTILS::Matrix3x3d fnn;
                MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(n1, n1);
                MATHUTILS::__M_Mat_multiply(F, nn, fnn);
                MATHUTILS::Vector9 q10 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
                q10 = MATHUTILS::__s_vec9_multiply(q10, 1.0 / sqrt(I1));

                MATHUTILS::Matrix3x3d Tx, Ty, Tz;
                MATHUTILS::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
                MATHUTILS::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
                MATHUTILS::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

                double ratio = 1.f / sqrt(2.f);
                Tx = __S_Mat_multiply(Tx, ratio);
                Ty = __S_Mat_multiply(Ty, ratio);
                Tz = __S_Mat_multiply(Tz, ratio);

                MATHUTILS::Vector9 q11 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tx, fnn));
                MATHUTILS::__normalized_vec9_double(q11);
                MATHUTILS::Vector9 q12 = MATHUTILS::__Mat3x3_to_vec9_double(MATHUTILS::__M_Mat_multiply(Tz, fnn));
                //MATHUTILS::__s_vec9_multiply(q12, c);
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
                nn = MATHUTILS::__v_vec_toMat(n2, n2);
                MATHUTILS::__M_Mat_multiply(F, nn, fnn);
                MATHUTILS::Vector9 q20 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);
                q20 = MATHUTILS::__s_vec9_multiply(q20, 1.0 / sqrt(I2));



#if (RANK == 1)
                double lambdag1g = Kappa * 4 * c * F.m[2][2] * ((2 * dHat * dHat * (I1 - eps_x) * (I2 - 1) * (I2 + 2 * I2 * log(I2) - 1)) / (I2 * eps_x * eps_x));
#elif (RANK == 2)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) / (I2 * (eps_x * eps_x));
#elif (RANK == 4)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 3) * (I1 - eps_x) * (I2 - 1) * (2 * I2 + I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));
#elif (RANK == 6)
                double lambdag1g = -Kappa * 4 * c * F.m[2][2] * (4 * dHat * dHat * pow(log(I2), 5) * (I1 - eps_x) * (I2 - 1) * (3 * I2 + I2 * log(I2) - 3)) / (I2 * (eps_x * eps_x));
#endif
#ifndef MAKEPD
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
#else
                MatT2x2 F_{ lambda10, lambdag1g, lambdag1g, lambda20 };
            make_pd(F_);
            MATHUTILS::__init_Mat9x9(M9_temp,0);
            M9_temp.m[4][4] = F_(0,0);
            M9_temp.m[4][8] = F_(0,1);
            M9_temp.m[8][4] = F_(1,0);
            M9_temp.m[8][8] = F_(1,1);
            projectedH = MATHUTILS::__Mat9x9_add(projectedH, M9_temp);
#endif
                //MATHUTILS::Matrix9x12d PFPxTransPos = MATHUTILS::__Transpose12x9(PFPx);
                MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);

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
                GIPCPDERIV::pFpx_pe2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], d_hat_sqrt, PFPxT);
                double I5 = pow(dis / d_hat_sqrt, 2);
                MATHUTILS::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3] = 1;

                MATHUTILS::Matrix4x4d H;
                //MATHUTILS::__init_Mat4x4_val(H, 0);
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

                //if (dis > dHat_sqrt) return;

                MATHUTILS::Matrix3x3d rotation;
                MATHUTILS::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);

                MATHUTILS::Matrix9x4d PDmPx;

                if (cos + 1 == 0) {
                    rotation.m[0][0] = -1;
                    rotation.m[1][1] = -1;
                }
                else {
                    //pDmpx_pe(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dHat_sqrt, PDmPx);
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

                MATHUTILS::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);
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
                H = MATHUTILS::__S_Mat4x4_multiply(MATHUTILS::__v4_vec4_toMat4x4(q0, q0), lambda0);

                MATHUTILS::Matrix9x9d Hessian;// = MATHUTILS::__M9x4_M4x9_Multiply(MATHUTILS::__M9x4_M4x4_Multiply(PFPxT, H), MATHUTILS::__Transpose9x4(PFPxT));
                __M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);
#else

                MATHUTILS::Vector6 q0 = MATHUTILS::__Mat3x2_to_vec6_double(fnn);

                q0 = MATHUTILS::__s_vec6_multiply(q0, 1.0 / sqrt(I5));

                MATHUTILS::Matrix6x6d H;
                MATHUTILS::__init_Mat6x6(H, 0);

                H = MATHUTILS::__S_Mat6x6_multiply(MATHUTILS::__v6_vec6_toMat6x6(q0, q0), lambda0);

                MATHUTILS::Matrix9x6d PFPxTransPos = MATHUTILS::__Transpose6x9(PFPx);
                MATHUTILS::Matrix9x9d Hessian = MATHUTILS::__M9x6_M6x9_Multiply(MATHUTILS::__M9x6_M6x6_Multiply(PFPxTransPos, H), PFPx);
#endif
                int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);

                H9x9[Hidx] = Hessian;
                D3Index[Hidx] = make_uint3(v0I, MMCVIDI.y, MMCVIDI.z);
            }

        }
        else {
#ifdef NEWF
            double dis;
            //printf("PT: %d %d %d %d\n", v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
            MATHUTILS::__distancePointTriangle(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
            double I5 = dis / dHat;
            dis = sqrt(dis);
            double d_hat_sqrt = sqrt(dHat);
            MATHUTILS::Matrix12x9d PFPxT;
            GIPCPDERIV::pFpx_pt2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], d_hat_sqrt,PFPxT);

            MATHUTILS::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
            q0.v[8] = 1;


#else
            double3 v0 = MATHUTILS::__minus(_vertexes[MMCVIDI.y], _vertexes[v0I]);
            double3 v1 = MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[v0I]);
            double3 v2 = MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[v0I]);

            MATHUTILS::Matrix3x3d Ds;
            MATHUTILS::__set_Mat_val_column(Ds, v0, v1, v2);

            double3 normal = MATHUTILS::__normalized(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]), MATHUTILS::__minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y])));
            double dis = MATHUTILS::__v_vec_dot(v0, normal);

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
            MATHUTILS::Matrix3x3d uu, vv, ss;
            MATHUTILS::SVD(F, uu, vv, ss);
            double values = ss.m[0][0] + ss.m[1][1] + ss.m[2][2];
            values = (values - 2) * (values - 2);
            double3 FxN = MATHUTILS::__M_v_multiply(F, normal);
            double I5 = MATHUTILS::__squaredNorm(FxN);

            MATHUTILS::Matrix9x12d PFPx = __computePFDsPX3D_double(DmInv);
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
            //printf("lamdba0:    %f\n", lambda0*1e6);
            //MATHUTILS::__v9_vec9_toMat9x9(H,q0, q0, lambda0); //MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);
            //MATHUTILS::Matrix9x9d H;
            //MATHUTILS::__init_Mat9x9(H, 0);
            //H.m[8][8] = lambda0;
            MATHUTILS::Matrix9x9d H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);//MATHUTILS::__v9_vec9_toMat9x9(q0, q0, lambda0);
            MATHUTILS::Matrix12x12d H2;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxT, H), MATHUTILS::__Transpose12x9(PFPxT));
            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H,H2);

#else

            MATHUTILS::Matrix3x3d Q0;

            MATHUTILS::Matrix3x3d fnn;

            MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);

            MATHUTILS::__M_Mat_multiply(F, nn, fnn);

            MATHUTILS::Vector9 q0 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);

            q0 = MATHUTILS::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            MATHUTILS::Matrix9x9d H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);

            MATHUTILS::Matrix12x9d PFPxTransPos = MATHUTILS::__Transpose9x12(PFPx);
            MATHUTILS::Matrix12x12d H2 = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = H2;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
        }
    }
}

__global__
void _calBarrierGradientAndHessian(const double3* _vertexes, const double3* _rest_vertexes, const const int4* _collisionPair, double3* _gradient, MATHUTILS::Matrix12x12d* H12x12, MATHUTILS::Matrix9x9d* H9x9,
                                   MATHUTILS::Matrix6x6d* H6x6, uint4* D4Index, uint3* D3Index, uint2* D2Index, uint32_t* _cpNum, int* matIndex, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _collisionPair[idx];
    double dHat_sqrt = sqrt(dHat);
    //double dHat = dHat_sqrt * dHat_sqrt;
    //double Kappa = 1;
    double gassThreshold = 1e-4;
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

            MATHUTILS::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
            q0.v[8] = 1;
            //q0 = MATHUTILS::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            MATHUTILS::Matrix9x9d H;
            //MATHUTILS::__init_Mat9x9(H, 0);
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

            MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxT, H), MATHUTILS::__Transpose12x9(PFPxT));
            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            MATHUTILS::Vector12 gradient_vec = MATHUTILS::__M12x9_v9_multiply(MATHUTILS::__Transpose9x12(PFPx), flatten_pk1);
            //MATHUTILS::Matrix3x3d Q0;

//            MATHUTILS::Matrix3x3d fnn;

 //           MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);

//            MATHUTILS::__M_Mat_multiply(F, nn, fnn);

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
            int Hidx = matIndex[idx];//atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

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

            //MATHUTILS::Matrix9x12d PFPxTransPos = MATHUTILS::__Transpose12x9(PFPx);
            MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
            int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
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

                //MATHUTILS::Matrix9x12d PFPxTransPos = MATHUTILS::__Transpose12x9(PFPx);
                MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 4, 1);

                H12x12[Hidx] = Hessian;
                D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
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
                int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 2, 1);

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

                //MATHUTILS::Matrix9x12d PFPxTransPos = MATHUTILS::__Transpose12x9(PFPx);
                MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPx, projectedH), PFPxTransPos);
                MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
                int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 4, 1);

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
                fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;// = fnn.v[3] = fnn.v[4] = 1;
                fnn.v[3] = dis / d_hat_sqrt;
                //MATHUTILS::Vector4 flatten_pk1 = MATHUTILS::__s_vec4_multiply(fnn, 2 * Kappa * -(dHat * dHat * (I5 - 1) * (I5 + 2 * I5 * log(I5) - 1)) / I5);
                MATHUTILS::Vector4 q0;
                q0.v[0] = q0.v[1] = q0.v[2] = 0;
                q0.v[3] = 1;
                MATHUTILS::Matrix4x4d H;
                //MATHUTILS::__init_Mat4x4_val(H, 0);
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

                MATHUTILS::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

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

                MATHUTILS::Matrix9x9d Hessian;// = MATHUTILS::__M9x4_M4x9_Multiply(MATHUTILS::__M9x4_M4x4_Multiply(PFPxT, H), MATHUTILS::__Transpose9x4(PFPxT));
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
                int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 3, 1);

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
            GIPCPDERIV::pFpx_pt2(_vertexes[v0I], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], d_hat_sqrt,
                     PFPxT);
            double I5 = pow(dis / d_hat_sqrt, 2);
            MATHUTILS::Vector9 tmp;
            tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
            tmp.v[8] = dis / d_hat_sqrt;

            MATHUTILS::Vector9 q0;
            q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
            q0.v[8] = 1;

            MATHUTILS::Matrix9x9d H;
            //MATHUTILS::__init_Mat9x9(H, 0);
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

            H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);

            MATHUTILS::Matrix12x12d Hessian;// = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxT, H), MATHUTILS::__Transpose12x9(PFPxT));
            MATHUTILS::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);
#else

            //MATHUTILS::Matrix3x3d Q0;

            //MATHUTILS::Matrix3x3d fnn;

            //MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);

            //MATHUTILS::__M_Mat_multiply(F, nn, fnn);

            MATHUTILS::Vector9 q0 = MATHUTILS::__Mat3x3_to_vec9_double(fnn);

            q0 = MATHUTILS::__s_vec9_multiply(q0, 1.0 / sqrt(I5));

            MATHUTILS::Matrix9x9d H = MATHUTILS::__S_Mat9x9_multiply(MATHUTILS::__v9_vec9_toMat9x9(q0, q0), lambda0);

            MATHUTILS::Matrix12x9d PFPxTransPos = MATHUTILS::__Transpose9x12(PFPx);
            MATHUTILS::Matrix12x12d Hessian = MATHUTILS::__M12x9_M9x12_Multiply(MATHUTILS::__M12x9_M9x9_Multiply(PFPxTransPos, H), PFPx);
#endif

            int Hidx = matIndex[idx];//int Hidx = atomicAdd(_cpNum + 4, 1);

            H12x12[Hidx] = Hessian;
            D4Index[Hidx] = make_uint4(v0I, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

        }


    }
}


__global__
void _calSelfCloseVal(const double3* _vertexes, const int4* _collisionPair, int4* _close_collisionPair, double* _close_collisionVal,
    uint32_t* _close_cpNum, double dTol, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _collisionPair[idx];
    double dist2 = _selfConstraintVal(_vertexes, MMCVIDI);
    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx] = dist2;
    }

}

__global__
void _checkSelfCloseVal(const double3* _vertexes, int* _isChange, int4* _close_collisionPair, double* _close_collisionVal, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    int4 MMCVIDI = _close_collisionPair[idx];
    double dist2 = _selfConstraintVal(_vertexes, MMCVIDI);
    if (dist2 < _close_collisionVal[idx]) {
        *_isChange = 1;
    }

}


__global__
void _reduct_MSelfDist(const double3* _vertexes, int4* _collisionPairs, double2* _queue, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if (idx >= number) return;
    int4 MMCVIDI = _collisionPairs[idx];
    double tempv = _selfConstraintVal(_vertexes, MMCVIDI);
    double2 temp = make_double2(1.0 / tempv, tempv);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        double tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = MATHUTILS::__m_max(temp.x, tempMin);
        temp.y = MATHUTILS::__m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            double tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = MATHUTILS::__m_max(temp.x, tempMin);
            temp.y = MATHUTILS::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _queue[blockIdx.x] = temp;
    }
}

__global__
void _calFrictionGradient_gd(const double3* _vertexes,
    const double3* _o_vertexes,
    const double3* _normal,
    const const uint32_t* _last_collisionPair_gd,
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
    const const int4* _last_collisionPair,
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
void _calBarrierGradient(const double3* _vertexes, const double3* _rest_vertexes, const const int4* _collisionPair, double3* _gradient, double dHat, double Kappa, int number) {
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

                MATHUTILS::Matrix6x9d PFPx = __computePFDsPX3D_6x9_double(DmInv);

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
void _calKineticGradient(double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    double3 deltaX = MATHUTILS::__minus(vertexes[idx], xTilta[idx]);
    //masses[idx] = 1;
    gradient[idx] = make_double3(deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    //printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__
void _calKineticEnergy(double3* vertexes, double3* xTilta, double3* gradient, double* masses, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    double3 deltaX = MATHUTILS::__minus(vertexes[idx], xTilta[idx]);
    gradient[idx] = make_double3(deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
}

__global__
void _computeSoftConstraintGradientAndHessian(const double3* vertexes, const double3* targetVert, const uint32_t* targetInd, double3* gradient, uint32_t* _gpNum, MATHUTILS::Matrix3x3d* H3x3, uint32_t* D1Index, double motionRate, double rate, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    uint32_t vInd = targetInd[idx];
    double x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z, a = targetVert[idx].x, b = targetVert[idx].y, c = targetVert[idx].z;
    //double dis = MATHUTILS::__squaredNorm(MATHUTILS::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        atomicAdd(&(gradient[vInd].x), d * rate * rate * (x - a));
        atomicAdd(&(gradient[vInd].y), d * rate * rate * (y - b));
        atomicAdd(&(gradient[vInd].z), d * rate * rate * (z - c));
    }
    MATHUTILS::Matrix3x3d Hpg;
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


__global__
void _computeSoftConstraintGradient(const double3* vertexes, const double3* targetVert, const uint32_t* targetInd, double3* gradient, double motionRate, double rate, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    uint32_t vInd = targetInd[idx];
    double x = vertexes[vInd].x, y = vertexes[vInd].y, z = vertexes[vInd].z, a = targetVert[idx].x, b = targetVert[idx].y, c = targetVert[idx].z;
    //double dis = MATHUTILS::__squaredNorm(MATHUTILS::__minus(vertexes[vInd], targetVert[idx]));
    //printf("%f\n", dis);
    double d = motionRate;
    {
        atomicAdd(&(gradient[vInd].x), d * rate * rate * (x - a));
        atomicAdd(&(gradient[vInd].y), d * rate * rate * (y - b));
        atomicAdd(&(gradient[vInd].z), d * rate * rate * (z - c));
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
void _computeGroundGradientAndHessian(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double3* gradient, uint32_t* _gpNum, MATHUTILS::Matrix3x3d* H3x3, uint32_t* D1Index, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = MATHUTILS::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    double t = dist2 - dHat;
    double g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    double H_b = (log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);

    //printf("H_b   dist   g_b    is  %lf  %lf  %lf\n", H_b, dist2, g_b);

    double3 grad = MATHUTILS::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        atomicAdd(&(gradient[gidx].x), grad.x);
        atomicAdd(&(gradient[gidx].y), grad.y);
        atomicAdd(&(gradient[gidx].z), grad.z);
    }

    double param = 4.0 * H_b * dist2 + 2.0 * g_b;
    if (param > 0) {
        MATHUTILS::Matrix3x3d nn = MATHUTILS::__v_vec_toMat(normal, normal);
        MATHUTILS::Matrix3x3d Hpg = MATHUTILS::__S_Mat_multiply(nn, Kappa * param);

        int pidx = atomicAdd(_gpNum, 1);
        H3x3[pidx] = Hpg;
        D1Index[pidx] = gidx;

    }
    //_environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}

__global__
void _computeGroundGradient(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double3* gradient, uint32_t* _gpNum, MATHUTILS::Matrix3x3d* H3x3, double dHat, double Kappa, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = MATHUTILS::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    double t = dist2 - dHat;
    double g_b = t * std::log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    //double H_b = (std::log(dist2 / dHat) * -2.0 - t * 4.0 / dist2) + 1.0 / (dist2 * dist2) * (t * t);
    double3 grad = MATHUTILS::__s_vec_multiply(normal, Kappa * g_b * 2 * dist);

    {
        atomicAdd(&(gradient[gidx].x), grad.x);
        atomicAdd(&(gradient[gidx].y), grad.y);
        atomicAdd(&(gradient[gidx].z), grad.z);
    }
}

__global__
void _computeGroundCloseVal(const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double dTol, uint32_t* _closeConstraintID, double* _closeConstraintVal, uint32_t* _close_gpNum, int number) {
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
void _checkGroundCloseVal(const double3* vertexes, const double* g_offset, const double3* g_normal, int* _isChange, uint32_t* _closeConstraintID, double* _closeConstraintVal, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _closeConstraintID[idx];
    double dist = MATHUTILS::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;

    if (dist2 < _closeConstraintVal[gidx]) {
        *_isChange = 1;
    }
}

__global__
void _reduct_MGroundDist(const double3* vertexes, const double* g_offset, const double3* g_normal, uint32_t* _environment_collisionPair, double2* _queue, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ double2 sdata[];

    if (idx >= number) return;
    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = MATHUTILS::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double tempv = dist * dist;
    double2 temp = make_double2(1.0 / tempv, tempv);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        double tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = MATHUTILS::__m_max(temp.x, tempMin);
        temp.y = MATHUTILS::__m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            double tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = MATHUTILS::__m_max(temp.x, tempMin);
            temp.y = MATHUTILS::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _queue[blockIdx.x] = temp;
    }
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
void _getFrictionEnergy_Reduction_3D(double* squeue,
    const double3* vertexes,
    const double3* o_vertexes,
    const int4* _collisionPair,
    int cpNum,
    double dt,
    const double2* distCoord,
    const MATHUTILS::Matrix3x2d* tanBasis,
    const double* lastH,
    double fricDHat,
    double eps

) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = cpNum;
    if (idx >= numbers) return;

    double temp = __cal_Friction_energy(vertexes, o_vertexes, _collisionPair[idx], dt, distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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

__global__
void _getFrictionEnergy_gd_Reduction_3D(double* squeue,
    const double3* vertexes,
    const double3* o_vertexes,
    const double3* _normal,
    const uint32_t* _collisionPair_gd,
    int gpNum,
    double dt,
    const double* lastH,
    double eps

) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = gpNum;
    if (idx >= numbers) return;

    double temp = __cal_Friction_gd_energy(vertexes, o_vertexes, _normal, _collisionPair_gd[idx], dt, lastH[idx], eps);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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

__global__
void _computeGroundEnergy_Reduction(double* squeue, const double3* vertexes, const double* g_offset, const double3* g_normal, const uint32_t* _environment_collisionPair, double dHat, double Kappa, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    double dist = MATHUTILS::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    double dist2 = dist * dist;
    double temp = -(dist2 - dHat) * (dist2 - dHat) * log(dist2 / dHat);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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

__global__
void _reduct_min_groundTimeStep_to_double(const double3* vertexes, const uint32_t* surfVertIds, const double* g_offset, const double3* g_normal, const double3* moveDir, double* minStepSizes, double slackness, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    int svI = surfVertIds[idx];
    double temp = 1.0;
    double3 normal = *g_normal;
    double coef = MATHUTILS::__v_vec_dot(normal, moveDir[svI]);
    if (coef > 0.0) {
        double dist = MATHUTILS::__v_vec_dot(normal, vertexes[svI]) - *g_offset;//normal
        temp = coef / (dist * slackness);
        //printf("%f\n", temp);
    }
    /*if (blockIdx.x == 4) {
        printf("%f\n", temp);
    }
    __syncthreads();*/
    //printf("%f\n", temp);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
        //printf("warpNum %d\n", warpNum);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = MATHUTILS::__m_max(temp, tempMin);
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
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = MATHUTILS::__m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = temp;
        //printf("%f   %d\n", temp, blockIdx.x);
    }

}


__global__
void _reduct_min_InjectiveTimeStep_to_double(const double3* vertexes, const uint4* tetrahedra, const double3* moveDir, double* minStepSizes, double slackness, double errorRate, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    double ratio = 1 - slackness;

    double temp = 1.0 / _computeInjectiveStepSize_3d(vertexes, moveDir, tetrahedra[idx].x, tetrahedra[idx].y, tetrahedra[idx].z, tetrahedra[idx].w, ratio, errorRate);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
        //printf("warpNum %d\n", warpNum);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = MATHUTILS::__m_max(temp, tempMin);
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
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = MATHUTILS::__m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = temp;
        //printf("%f   %d\n", temp, blockIdx.x);
    }

}

__global__
void _reduct_min_selfTimeStep_to_double(const double3* vertexes, const int4* _ccd_collitionPairs, const double3* moveDir, double* minStepSizes, double slackness, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    double temp = 1.0;
    double CCDDistRatio = 1.0 - slackness;

    int4 MMCVIDI = _ccd_collitionPairs[idx];

    if (MMCVIDI.x < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;

        double temp1 = ACCD::point_triangle_ccd(vertexes[MMCVIDI.x],
            vertexes[MMCVIDI.y],
            vertexes[MMCVIDI.z],
            vertexes[MMCVIDI.w],
            MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
            MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
            MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
            MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0);

        //double temp2 = doCCDVF(vertexes[MMCVIDI.x],
        //    vertexes[MMCVIDI.y],
        //    vertexes[MMCVIDI.z],
        //    vertexes[MMCVIDI.w],
        //    MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
        //    MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
        //    MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
        //    MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.w], -1), 1e-9, 0.2);

        temp = 1.0 / temp1;
    }
    else {
        temp = 1.0 / ACCD::edge_edge_ccd(vertexes[MMCVIDI.x],
            vertexes[MMCVIDI.y],
            vertexes[MMCVIDI.z],
            vertexes[MMCVIDI.w],
            MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.x], -1),
            MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.y], -1),
            MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.z], -1),
            MATHUTILS::__s_vec_multiply(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0);
    }

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = MATHUTILS::__m_max(temp, tempMin);
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
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = MATHUTILS::__m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        minStepSizes[blockIdx.x] = temp;
    }

}

__global__
void _reduct_max_cfl_to_double(const double3* moveDir, double* max_double_val, uint32_t* mSVI, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = MATHUTILS::__norm(moveDir[mSVI[idx]]);


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        double tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = MATHUTILS::__m_max(temp, tempMax);
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
            double tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = MATHUTILS::__m_max(temp, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        max_double_val[blockIdx.x] = temp;
    }

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
    double nextTp;
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
    double nextTp;
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
void _getKineticEnergy_Reduction_3D(double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = MATHUTILS::__squaredNorm(MATHUTILS::__minus(_vertexes[idx], _xTilta[idx])) * _masses[idx] * 0.5;

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
        _energy[blockIdx.x] = temp;
    }
}


__global__
void _getBendingEnergy_Reduction(double* squeue, const double3* vertexes, const double3* rest_vertexex, const uint2* edges, const uint2* edge_adj_vertex, int edgesNum, double bendStiff) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = edgesNum;
    if (idx >= numbers) return;

    //double temp = __cal_BaraffWitkinStretch_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    // double temp = __cal_hc_cloth_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);
    uint2 adj = edge_adj_vertex[idx];
    double3 rest_x0 = rest_vertexex[edges[idx].x];
    double3 rest_x1 = rest_vertexex[edges[idx].y];
    double length = MATHUTILS::__norm(MATHUTILS::__minus(rest_x0, rest_x1));
    double temp = FEMENERGY::__cal_bending_energy(vertexes, rest_vertexex, edges[idx], adj, length, bendStiff);
    //double temp = 0;
    //printf("%f    %f\n\n\n", lenRate, volRate);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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


__global__
void _getFEMEnergy_Reduction_3D(double* squeue, const double3* vertexes, const uint4* tetrahedras, const MATHUTILS::Matrix3x3d* DmInverses, const double* volume, int tetrahedraNum, double lenRate, double volRate) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = tetrahedraNum;
    if (idx >= numbers) return;

#ifdef USE_SNK
    double temp = FEMENERGY::__cal_StabbleNHK_energy_3D(vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate, volRate);
#else
    double temp = FEMENERGY::__cal_ARAP_energy_3D(vertexes, tetrahedras[idx], DmInverses[idx], volume[idx], lenRate);
#endif
    
    //printf("%f    %f\n\n\n", lenRate, volRate);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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
__global__
void _computeSoftConstraintEnergy_Reduction(double* squeue, const double3* vertexes, const double3* targetVert, const uint32_t* targetInd, double motionRate, double rate, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    uint32_t vInd = targetInd[idx];
    double dis = MATHUTILS::__squaredNorm(MATHUTILS::__s_vec_multiply(MATHUTILS::__minus(vertexes[vInd], targetVert[idx]), rate));
    double d = motionRate;
    double temp = d * dis * 0.5;

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
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
__global__
void _get_triangleFEMEnergy_Reduction_3D(double* squeue, const double3* vertexes, const uint3* triangles, const MATHUTILS::Matrix2x2d* triDmInverses, const double* area, int trianglesNum, double stretchStiff, double shearStiff) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = trianglesNum;
    if (idx >= numbers) return;

    double temp = FEMENERGY::__cal_BaraffWitkinStretch_energy(vertexes, triangles[idx], triDmInverses[idx], area[idx], stretchStiff, shearStiff);


    //printf("%f    %f\n\n\n", lenRate, volRate);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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
__global__
void _getRestStableNHKEnergy_Reduction_3D(double* squeue, const double* volume, int tetrahedraNum, double lenRate, double volRate) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = tetrahedraNum;
    if (idx >= numbers) return;

    double temp = ((0.5 * volRate * (3 * lenRate / 4 / volRate) * (3 * lenRate / 4 / volRate) - 0.5 * lenRate * log(4.0)))* volume[idx];

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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

__global__
void _getBarrierEnergy_Reduction_3D(double* squeue, const double3* vertexes, const double3* rest_vertexes, int4* _collisionPair, double _Kappa, double _dHat, int cpNum) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = cpNum;
    if (idx >= numbers) return;

    double temp = __cal_Barrier_energy(vertexes, rest_vertexes, _collisionPair[idx], _Kappa, _dHat);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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

__global__
void _getDeltaEnergy_Reduction(double* squeue, const double3* b, const double3* dx, int vertexNum) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = vertexNum;
    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);

    double temp = MATHUTILS::__v_vec_dot(b[idx], dx[idx]);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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

__global__
void __add_reduction(double* mem, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = mem[idx];

    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    double nextTp;
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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
        mem[blockIdx.x] = temp;
    }
}

__global__
void _stepForward(double3* _vertexes, double3* _vertexesTemp, double3* _moveDir, int* bType, double alpha, bool moveBoundary, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (abs(bType[idx]) != 1 || moveBoundary) {
        _vertexes[idx] = MATHUTILS::__minus(_vertexesTemp[idx], MATHUTILS::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

__global__
void _updateVelocities(double3* _vertexes, double3* _o_vertexes, double3* _velocities, int* btype, double ipc_dt, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (btype[idx] == 0) {
        _velocities[idx] = MATHUTILS::__s_vec_multiply(MATHUTILS::__minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        _o_vertexes[idx] = _vertexes[idx];
    }
    else {
        _velocities[idx] = make_double3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

__global__
void _updateBoundary(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    if ((_btype[idx]) == -1|| (_btype[idx]) == 1) {
        _vertexes[idx] = MATHUTILS::__add(_vertexes[idx], _moveDir[idx]);
    }
}

__global__
void _updateBoundary2(int* _btype, MATHUTILS::Matrix3x3d* _constraints, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    if ((_btype[idx]) == 1) {
        _btype[idx] = 0;
        MATHUTILS::__set_Mat_val(_constraints[idx], 1,0,0,0,1,0,0,0,1);
    }

}


__global__
void _updateBoundaryMoveDir(double3* _vertexes, int* _btype, double3* _moveDir, double ipc_dt, double PI, double alpha, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double massSum = 0;
    double angleX = PI / 2.5 * ipc_dt * alpha;
    MATHUTILS::Matrix3x3d rotationL, rotationR;
    MATHUTILS::__set_Mat_val(rotationL, 1, 0, 0, 0, cos(angleX), sin(angleX), 0, -sin(angleX), cos(angleX));
    MATHUTILS::__set_Mat_val(rotationR, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

    _moveDir[idx] = make_double3(0, 0, 0);
    double mvl = -1*ipc_dt*alpha;
        if ((_btype[idx]) == 1) {
        _moveDir[idx] = make_double3(mvl,0,0);//MATHUTILS::__minus(MATHUTILS::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
    }
//    if ((_btype[idx]) > 0) {
//        _moveDir[idx] = MATHUTILS::__minus(MATHUTILS::__M_v_multiply(rotationL, _vertexes[idx]), _vertexes[idx]);
//    }
//    if ((_btype[idx]) < 0) {
//        _moveDir[idx] = MATHUTILS::__minus(MATHUTILS::__M_v_multiply(rotationR, _vertexes[idx]), _vertexes[idx]);
//    }
}

__global__
void _computeXTilta(int* _btype, double3* _velocities, double3* _o_vertexes, double3* _xTilta, double ipc_dt, double rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double3 gravityDtSq = make_double3(0, 0, 0);//MATHUTILS::__s_vec_multiply(make_double3(0, -9.8, 0), ipc_dt * ipc_dt);//Vector3d(0, gravity, 0) * IPC_dt * IPC_dt;
    if (_btype[idx] == 0) {
        gravityDtSq = MATHUTILS::__s_vec_multiply(make_double3(0, -9.8, 0), ipc_dt * ipc_dt);
    }
    _xTilta[idx] = MATHUTILS::__add(_o_vertexes[idx], MATHUTILS::__add(MATHUTILS::__s_vec_multiply(_velocities[idx], ipc_dt), gravityDtSq));//(mesh.V_prev[vI] + (mesh.velocities[vI] * IPC_dt + gravityDtSq));
}


__global__
void _updateNeighborNum(unsigned int* _neighborNumInit, unsigned int* _neighborNum, const uint32_t* sortMapVertIndex, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    _neighborNum[idx] = _neighborNumInit[sortMapVertIndex[idx]];
}

__global__
void _updateNeighborList(unsigned int* _neighborListInit, unsigned int* _neighborList, unsigned int* _neighborNum, unsigned int* _neighborStart, unsigned int* _neighborStartTemp, const uint32_t* sortIndex, const uint32_t* sortMapVertIndex, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int startId = _neighborStartTemp[idx];
    int o_startId = _neighborStart[sortIndex[idx]];
    int neiNum = _neighborNum[idx];
    for (int i = 0; i < neiNum; i++) {
        _neighborList[startId + i] = sortMapVertIndex[_neighborListInit[o_startId + i]];
    }
    //_neighborStart[sortMapVertIndex[idx]] = startId;
    //_neighborNum[idx] = _neighborNum[sortMapVertIndex[idx]];
}


__global__
void _edgeTriIntersectionQuery(const int* _btype, const double3* _vertexes, const uint2* _edges, const uint3* _faces, const AABB* _edge_bvs, const Node* _edge_nodes, int* _isIntesect, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t  stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;

    uint3 face = _faces[idx];
    //idx = idx + number - 1;


    AABB _bv;

    double3 _v = _vertexes[face.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.z];
    _bv.combines(_v.x, _v.y, _v.z);

    //uint32_t self_eid = _edge_nodes[idx].element_idx;
    //double instance->bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(_edge_bvs[0].upper, _edge_bvs[0].lower));
    //printf("%f\n", instance->bboxDiagSize2);
    double gapl = 0;//sqrt(dHat);
    //double dHat = gapl * gapl;// *instance->bboxDiagSize2;
    unsigned int num_found = 0;
    do
    {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _edge_nodes[node_id].m_left_idx;
        const uint32_t R_idx = _edge_nodes[node_id].m_right_idx;

        if (_overlap(_bv, _edge_bvs[L_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[L_idx].m_element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y)) {
                    if (!(_btype[face.x] >= 2 && _btype[face.y] >= 2 && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
                        if (segTriIntersect(_vertexes[_edges[obj_idx].x], _vertexes[_edges[obj_idx].y], _vertexes[face.x], _vertexes[face.y], _vertexes[face.z])) {
                            //atomicAdd(_isIntesect, -1);
                            *_isIntesect=-1;
                            printf("tri: %d %d %d,  edge: %d  %d\n", face.x, face.y, face.z, _edges[obj_idx].x, _edges[obj_idx].y);
                            return;
                        }
                }

            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (_overlap(_bv, _edge_bvs[R_idx], gapl))
        {
            const auto obj_idx = _edge_nodes[R_idx].m_element_idx;
            if (obj_idx != 0xFFFFFFFF)
            {
                if (!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y || face.y == _edges[obj_idx].x || face.y == _edges[obj_idx].y || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y)) {
                    if (!(_btype[face.x] >= 2 && _btype[face.y] >= 2 && _btype[face.z] >= 2 && _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
                        if (segTriIntersect(_vertexes[_edges[obj_idx].x], _vertexes[_edges[obj_idx].y], _vertexes[face.x], _vertexes[face.y], _vertexes[face.z])) {
                            //atomicAdd(_isIntesect, -1);
                            *_isIntesect=-1;
                            printf("tri: %d %d %d,  edge: %d  %d\n", face.x, face.y, face.z, _edges[obj_idx].x, _edges[obj_idx].y);
                            return;
                        }
                }

            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}

__global__
void _calFrictionLastH_gd(const double3* _vertexes, const double* g_offset, const double3* g_normal, const const uint32_t* _collisionPair_environment, double* lambda_lastH_gd, uint32_t* _collisionPair_last_gd, double dHat,
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
void _calFrictionLastH_DistAndTan(const double3* _vertexes, const const int4* _collisionPair, double* lambda_lastH, double2* distCoord, MATHUTILS::Matrix3x2d* tanBasis, int4* _collisionPair_last, double dHat,
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


void GIPC::buildFrictionSets() {
    CUDA_SAFE_CALL(cudaMemset(mc_cpNum, 0, 5 * sizeof(uint32_t)));
    int numbers = h_cpNum[0];
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calFrictionLastH_DistAndTan <<<blockNum, threadNum>>> (mc_vertexes, mc_collisonPairs, lambda_lastH_scalar, distCoord, tanBasis, _collisonPairs_lastH, m_dHat, m_IPCKappa, mc_cpNum, h_cpNum[0]);
    CUDA_SAFE_CALL(cudaMemcpy(h_cpNum_last, mc_cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    numbers = h_gpNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    _calFrictionLastH_gd << <blockNum, threadNum >> > (mc_vertexes, mc_groundOffset, mc_groundNormal, mc_environment_collisionPair, lambda_lastH_scalar_gd, _collisonPairs_lastH_gd, m_dHat, m_IPCKappa, h_gpNum);
    h_gpNum_last = h_gpNum;
}

void GIPC::GroundCollisionDetect() {
    int numbers = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _GroundCollisionDetectIPC << <blockNum, threadNum >> > (mc_vertexes, mc_surfVerts, mc_groundOffset, mc_groundNormal, mc_environment_collisionPair, mc_gpNum, m_dHat, numbers);

}



void GIPC::computeSoftConstraintGradientAndHessian(double3* _gradient) {
    
    int numbers = m_softNum;
    if (numbers < 1) {
        CUDA_SAFE_CALL(cudaMemcpy(&m_BH->m_DNum, mc_gpNum, sizeof(int), cudaMemcpyDeviceToHost));
        return;
    }
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    // offset
    _computeSoftConstraintGradientAndHessian << <blockNum, threadNum >> > (mc_vertexes, mc_targetVert, mc_targetInd, _gradient, mc_gpNum, m_BH->m_H3x3, m_BH->m_D1Index, m_softMotionRate,animation_fullRate, m_softNum);
    CUDA_SAFE_CALL(cudaMemcpy(&m_BH->m_DNum, mc_gpNum, sizeof(int), cudaMemcpyDeviceToHost));
}

void GIPC::computeGroundGradientAndHessian(double3* _gradient) {
#ifndef USE_FRICTION  
    CUDA_SAFE_CALL(cudaMemset(mc_gpNum, 0, sizeof(uint32_t)));
#endif
    int numbers = h_gpNum;
    if (numbers < 1) {
        CUDA_SAFE_CALL(cudaMemcpy(&m_BH->m_DNum, mc_gpNum, sizeof(int), cudaMemcpyDeviceToHost));
        return;
    }
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _computeGroundGradientAndHessian << <blockNum, threadNum >> > (mc_vertexes, mc_groundOffset, mc_groundNormal, mc_environment_collisionPair, _gradient, mc_gpNum, m_BH->m_H3x3, m_BH->m_D1Index, m_dHat, m_IPCKappa, numbers);
    CUDA_SAFE_CALL(cudaMemcpy(&m_BH->m_DNum, mc_gpNum, sizeof(int), cudaMemcpyDeviceToHost));
}

void GIPC::computeCloseGroundVal() {
    int numbers = h_gpNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _computeGroundCloseVal << <blockNum, threadNum >> > (mc_vertexes, mc_groundOffset, mc_groundNormal, mc_environment_collisionPair, m_dTol, mc_closeConstraintID, mc_closeConstraintVal, mc_close_gpNum, numbers);

}

bool GIPC::checkCloseGroundVal() {
    int numbers = h_close_gpNum;
    if (numbers < 1) return false;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    int* _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkGroundCloseVal << <blockNum, threadNum >> > (mc_vertexes, mc_groundOffset, mc_groundNormal, _isChange, mc_closeConstraintID, mc_closeConstraintVal, numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxGroundDist() {
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);

    int numbers = h_gpNum;
    if (numbers < 1)return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist << <blockNum, threadNum, sharedMsize >> > (mc_vertexes, mc_groundOffset, mc_groundNormal, mc_environment_collisionPair, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2 << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minMaxValue;
    cudaMemcpy(&minMaxValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minMaxValue.x = 1.0 / minMaxValue.x;
    return minMaxValue;
}

void GIPC::computeGroundGradient(double3* _gradient, double mKappa) {
    int numbers = h_gpNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _computeGroundGradient << <blockNum, threadNum >> > (mc_vertexes, mc_groundOffset, mc_groundNormal, mc_environment_collisionPair, _gradient, mc_gpNum, m_BH->m_H3x3, m_dHat, mKappa, numbers);
}

void GIPC::computeSoftConstraintGradient(double3* _gradient) {
    int numbers = m_softNum;

    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    // offset
    _computeSoftConstraintGradient << <blockNum, threadNum >> > (
        mc_vertexes, 
        mc_targetVert, 
        mc_targetInd, 
        _gradient, 
        m_softMotionRate, 
        animation_fullRate, 
        m_softNum);
}

double GIPC::self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers) {
    //slackness = 0.9;
    //int numbers = h_cpNum[0];
    if (numbers < 1) return 1;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_min_selfTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (mc_vertexes, mc_ccd_collisonPairs, mc_moveDir, mqueue, slackness, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("                 full ccd time step:  %f\n", 1.0 / minValue);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::cfl_largestSpeed(double* mqueue) {
    int numbers = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _maxV;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_maxV, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_max_cfl_to_double << <blockNum, threadNum, sharedMsize >> > (mc_moveDir, mqueue, mc_surfVerts, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_maxV));
    return minValue;
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
        __add_reduction << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double dotValue;
    cudaMemcpy(&dotValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_queue));
    return dotValue;
}

double GIPC::ground_largestFeasibleStepSize(double slackness, double* mqueue) {

    int numbers = surf_vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));

    //if (h_cpNum[0] > 0) {
    //    double3* mvd = new double3[vertexNum];
    //    cudaMemcpy(mvd, _moveDir, sizeof(double3) * vertexNum, cudaMemcpyDeviceToHost);
    //    for (int i = 0;i < vertexNum;i++) {
    //        printf("%f  %f  %f\n", mvd[i].x, mvd[i].y, mvd[i].z);
    //    }
    //    delete[] mvd;
    //}
    _reduct_min_groundTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (mc_vertexes, mc_surfVerts, mc_groundOffset, mc_groundNormal, mc_moveDir, mqueue, slackness, numbers);


    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double GIPC::InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets) {

    int numbers = tetrahedraNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_InjectiveTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (mc_vertexes, tets, mc_moveDir, mqueue, slackness, errorRate, numbers);


    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("Injective Time step:   %f\n", 1.0 / minValue);
    //if (1.0 / minValue < 1) {
    //    system("pause");
    //}
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

void GIPC::buildCP() {

    CUDA_SAFE_CALL(cudaMemset(mc_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(mc_gpNum, 0, sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //m_bvh_f->Construct();
    m_bvh_f->SelfCollitionDetect(m_dHat);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //m_bvh_e->Construct();
    m_bvh_e->SelfCollitionDetect(m_dHat);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    GroundCollisionDetect();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(&h_cpNum, mc_cpNum, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&h_gpNum, mc_gpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    /*CUDA_SAFE_CALL(cudaMemset(_cpNum, 0, 5 * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemset(_gpNum, 0, sizeof(uint32_t)));*/
}

void GIPC::buildFullCP(const double& alpha) {

    CUDA_SAFE_CALL(cudaMemset(mc_cpNum, 0, sizeof(uint32_t)));

    m_bvh_f->SelfCollitionFullDetect(m_dHat, mc_moveDir, alpha);
    m_bvh_e->SelfCollitionFullDetect(m_dHat, mc_moveDir, alpha);

    CUDA_SAFE_CALL(cudaMemcpy(&h_ccd_cpNum, mc_cpNum, sizeof(uint32_t), cudaMemcpyDeviceToHost));
}


void GIPC::buildBVH() {
    m_bvh_f->Construct();
    m_bvh_e->Construct();
}

AABB* GIPC::calcuMaxSceneSize() {
    return m_bvh_f->getSceneSize();
}

void GIPC::buildBVH_FULLCCD(const double& alpha) {
    m_bvh_f->ConstructFullCCD(mc_moveDir, alpha);
    m_bvh_e->ConstructFullCCD(mc_moveDir, alpha);
}

void GIPC::calBarrierGradientAndHessian(double3* _gradient, double mKappa) {
    int numbers = h_cpNum[0];
    if (numbers < 1)return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;


    _calBarrierGradientAndHessian << <blockNum, threadNum >> > (mc_vertexes, mc_rest_vertexes, mc_collisonPairs, _gradient, m_BH->m_H12x12, m_BH->m_H9x9, m_BH->m_H6x6, m_BH->m_D4Index, m_BH->m_D3Index, m_BH->m_D2Index, mc_cpNum, mc_MatIndex, m_dHat, mKappa, numbers);


}



void GIPC::calBarrierHessian() {

    int numbers = h_cpNum[0];
    if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum; //

    _calBarrierHessian << <blockNum, threadNum >> > (mc_vertexes, mc_rest_vertexes, mc_collisonPairs, m_BH->m_H12x12, m_BH->m_H9x9, m_BH->m_H6x6, m_BH->m_D4Index, m_BH->m_D3Index, m_BH->m_D2Index, mc_cpNum, mc_MatIndex, m_dHat, m_IPCKappa, numbers);
}

void GIPC::calFrictionHessian(std::unique_ptr<GeometryManager>& instance) {
    int numbers = h_cpNum_last[0];
    //if (numbers < 1) return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum; //

    _calFrictionHessian << < blockNum, threadNum >> > (
        mc_vertexes,
        instance->cudaOriginVertPos,
        _collisonPairs_lastH,
        m_BH->m_H12x12,
        m_BH->m_H9x9,
        m_BH->m_H6x6,
        m_BH->m_D4Index,
        m_BH->m_D3Index,
        m_BH->m_D2Index,
        mc_cpNum,
        numbers,
        m_IPC_dt, distCoord,
        tanBasis,
        m_fDhat * m_IPC_dt * m_IPC_dt,
        lambda_lastH_scalar,
        m_frictionRate,
        h_cpNum[4],
        h_cpNum[3],
        h_cpNum[2]);


    numbers = h_gpNum_last;
    CUDA_SAFE_CALL(cudaMemcpy(mc_gpNum, &h_gpNum_last, sizeof(uint32_t), cudaMemcpyHostToDevice));
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionHessian_gd << < blockNum, threadNum >> > (
        mc_vertexes,
        instance->cudaOriginVertPos,
        mc_groundNormal,
        _collisonPairs_lastH_gd,
        m_BH->m_H3x3,
        m_BH->m_D1Index,
        numbers,
        m_IPC_dt,
        m_fDhat * m_IPC_dt * m_IPC_dt,
        lambda_lastH_scalar_gd,
        m_frictionRate);
}

void GIPC::computeSelfCloseVal() {
    int numbers = h_cpNum[0];
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _calSelfCloseVal << <blockNum, threadNum >> > (mc_vertexes, mc_collisonPairs, mc_closeMConstraintID, mc_closeMConstraintVal,
        mc_close_cpNum, m_dTol, numbers);
}

bool GIPC::checkSelfCloseVal() {
    int numbers = h_close_cpNum;
    if (numbers < 1) return false;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    int* _isChange;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isChange, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkSelfCloseVal << <blockNum, threadNum >> > (mc_vertexes, _isChange, mc_closeMConstraintID, mc_closeMConstraintVal, numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(_isChange));

    return (isChange == 1);
}

double2 GIPC::minMaxSelfDist() {
    int numbers = h_cpNum[0];
    if (numbers < 1)return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist << <blockNum, threadNum, sharedMsize >> > (mc_vertexes, mc_collisonPairs, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_M_double2 << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double2 minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minValue.x = 1.0 / minValue.x;
    return minValue;
}

void GIPC::calBarrierGradient(double3* _gradient, double mKappa) {
    int numbers = h_cpNum[0];
    if (numbers < 1)return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradient << <blockNum, threadNum >> > (mc_vertexes, mc_rest_vertexes, mc_collisonPairs, _gradient, m_dHat, mKappa, numbers);

}

void GIPC::calFrictionGradient(double3* _gradient, std::unique_ptr<GeometryManager>& instance) {
    int numbers = h_cpNum_last[0];
    //if (numbers < 1)return;
    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient << < blockNum, threadNum >> > (
        mc_vertexes,
        instance->cudaOriginVertPos,
        _collisonPairs_lastH,
        _gradient,
        numbers,
        m_IPC_dt,
        distCoord,
        tanBasis,
        m_fDhat * m_IPC_dt * m_IPC_dt,
        lambda_lastH_scalar,
        m_frictionRate
        );

    numbers = h_gpNum_last;
    //if (numbers < 1)return;
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient_gd << < blockNum, threadNum >> > (
        mc_vertexes,
        instance->cudaOriginVertPos,
        mc_groundNormal,
        _collisonPairs_lastH_gd,
        _gradient,
        numbers,
        m_IPC_dt,
        m_fDhat * m_IPC_dt * m_IPC_dt,
        lambda_lastH_scalar_gd,
        m_frictionRate
        );
}


void calKineticGradient(double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient << <blockNum, threadNum >> > (_vertexes, _xTilta, _gradient, _masses, numbers);
}

void calKineticEnergy(double3* _vertexes, double3* _xTilta, double3* _gradient, double* _masses, int numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient << <blockNum, threadNum >> > (_vertexes, _xTilta, _gradient, _masses, numbers);
}

void calculate_fem_gradient_hessian(MATHUTILS::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    MATHUTILS::Matrix12x12d* Hessians, const uint32_t& offset, const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double IPC_dt) {
    int numbers = tetrahedraNum;
    if (numbers < 1) return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    FEMENERGY::_calculate_fem_gradient_hessian <<<blockNum, threadNum>>> (DmInverses, vertexes, tetrahedras,
        Hessians, offset, volume, gradient, tetrahedraNum, lenRate, volRate, IPC_dt);
}

void calculate_triangle_fem_gradient_hessian(MATHUTILS::Matrix2x2d* triDmInverses, const double3* vertexes, const uint3* triangles,
    MATHUTILS::Matrix9x9d* Hessians, const uint32_t& offset, const double* area, double3* gradient, int triangleNum, double stretchStiff, double shearStiff, double IPC_dt) {
    int numbers = triangleNum;
    if (numbers < 1) return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    FEMENERGY::_calculate_triangle_fem_gradient_hessian << <blockNum, threadNum >> > (triDmInverses, vertexes, triangles,
        Hessians, offset, area, gradient, triangleNum, stretchStiff, shearStiff, IPC_dt);
}

void calculate_bending_gradient_hessian(const double3* vertexes, const double3* rest_vertexes, const uint2* edges, const uint2* edges_adj_vertex,
    MATHUTILS::Matrix12x12d* Hessians, uint4* Indices, const uint32_t& offset, double3* gradient, int edgeNum, double bendStiff, double IPC_dt) {
    int numbers = edgeNum;
    if (numbers < 1) return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    FEMENERGY::_calculate_bending_gradient_hessian << <blockNum, threadNum >> > (vertexes, rest_vertexes, edges, edges_adj_vertex,
        Hessians, Indices, offset, gradient, edgeNum, bendStiff, IPC_dt);
}

void calculate_fem_gradient(MATHUTILS::Matrix3x3d* DmInverses, const double3* vertexes, const uint4* tetrahedras,
    const double* volume, double3* gradient, int tetrahedraNum, double lenRate, double volRate, double dt) {
    int numbers = tetrahedraNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    FEMENERGY::_calculate_fem_gradient << <blockNum, threadNum >> > (DmInverses, vertexes, tetrahedras,
        volume, gradient, tetrahedraNum, lenRate, volRate, dt);
}

void calculate_triangle_fem_gradient(MATHUTILS::Matrix2x2d* triDmInverses, const double3* vertexes, const uint3* triangles,
    const double* area, double3* gradient, int triangleNum, double stretchStiff, double shearStiff, double IPC_dt) {
    int numbers = triangleNum;
    if (numbers < 1) return;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    FEMENERGY::_calculate_triangle_fem_gradient << <blockNum, threadNum >> > (triDmInverses, vertexes, triangles,
        area, gradient, triangleNum, stretchStiff, shearStiff, IPC_dt);
}

double calcMinMovement(const double3* _moveDir, double* _queue, const int& number) {

    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _tempMinMovement;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempMinMovement, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));

    _reduct_max_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _queue, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        _reduct_max_double << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_tempMinMovement));
    return minValue;
}

void stepForward(double3* _vertexes, double3* _vertexesTemp, double3* _moveDir, int* bType, double alpha, bool moveBoundary, int numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _stepForward << <blockNum, threadNum >> > (_vertexes, _vertexesTemp, _moveDir, bType, alpha, moveBoundary, numbers);
}


void updateNeighborInfo(unsigned int* _neighborList, unsigned int* d_neighborListInit, unsigned int* _neighborNum, unsigned int* _neighborNumInit, unsigned int* _neighborStart, unsigned int* _neighborStartTemp, const uint32_t* sortIndex, const uint32_t* sortMapVertIndex, const int& numbers, const int& neighborListSize) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateNeighborNum << <blockNum, threadNum >> > (_neighborNumInit, _neighborNum, sortIndex, numbers);
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(_neighborNum), thrust::device_ptr<unsigned int>(_neighborNum) + numbers, thrust::device_ptr<unsigned int>(_neighborStartTemp));
    _updateNeighborList << <blockNum, threadNum >> > (d_neighborListInit, _neighborList, _neighborNum, _neighborStart, _neighborStartTemp, sortIndex, sortMapVertIndex, numbers);
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborListInit, _neighborList, neighborListSize * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_neighborStart, _neighborStartTemp, numbers * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_neighborNumInit, _neighborNum, numbers * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

}




////////////////////////TO DO LATER/////////////////////////////////////////






void compute_H_b(double d, double dHat, double& H) {
    double t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

void GIPC::suggestKappa(double& kappa) {
    double H_b;
    //double bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(m_bvh_f->scene.upper, m_bvh_f->scene.lower));
    compute_H_b(1.0e-16 * m_bboxDiagSize2, m_dHat, H_b);

    if (m_meanMass == 0.0) {
        kappa = m_minKappaCoef / (4.0e-16 * m_bboxDiagSize2 * H_b);
    }
    else {
        kappa = m_minKappaCoef * m_meanMass / (4.0e-16 * m_bboxDiagSize2 * H_b);
    }
}

void GIPC::upperBoundKappa(double& kappa)
{
    double H_b;
    //double bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(m_bvh_f->scene.upper, m_bvh_f->scene.lower));//(maxConer - minConer).squaredNorm();
    compute_H_b(1.0e-16 * m_bboxDiagSize2, m_dHat, H_b);
    double kappaMax = 100 * m_minKappaCoef * m_meanMass / (4.0e-16 * m_bboxDiagSize2 * H_b);
    //printf("max Kappa: %f\n", kappaMax);
    if (m_meanMass == 0.0) {
        kappaMax = 100 * m_minKappaCoef / (4.0e-16 * m_bboxDiagSize2 * H_b);
    }

    if (kappa > kappaMax) {
        kappa = kappaMax;
    }
}


void GIPC::initKappa(std::unique_ptr<GeometryManager>& instance)
{

    if (h_cpNum[0] > 0) {
        double3* _GE = instance->cudaFb;
        double3* _gc = instance->cudaTempDouble3Mem;
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_gc, vertexNum * sizeof(double3)));
        //CUDA_SAFE_CALL(cudaMalloc((void**)&_GE, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, vertexNum * sizeof(double3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, vertexNum * sizeof(double3)));
        calKineticGradient(instance->cudaVertPos, instance->cudaXTilta, _GE, instance->cudaVertMass, vertexNum);
        calculate_fem_gradient(instance->cudaDmInverses, instance->cudaVertPos, instance->cudaTetElement, instance->cudaTetVolume,
            _GE, tetrahedraNum, m_lengthRate, m_volumeRate, m_IPC_dt);
        //calculate_triangle_fem_gradient(instance->triDmInverses, instance->cudaVertPos, instance->triangles, instance->area, _GE, triangleNum, stretchStiff, shearStiff, IPC_dt);
        computeSoftConstraintGradient(_GE);
        computeGroundGradient(_gc,1);
        calBarrierGradient(_gc,1);
        double gsum = reduction2Kappa(0, _gc, _GE, m_pcg_data->m_squeue, vertexNum);
        double gsnorm = reduction2Kappa(1, _gc, _GE, m_pcg_data->m_squeue, vertexNum);
        //CUDA_SAFE_CALL(cudaFree(_gc));
        //CUDA_SAFE_CALL(cudaFree(_GE));
        double minKappa = -gsum / gsnorm;
        if (minKappa > 0.0) {
            m_IPCKappa = minKappa;
        }
        suggestKappa(minKappa);
        if (m_IPCKappa < minKappa) {
            m_IPCKappa = minKappa;
        }
        upperBoundKappa(m_IPCKappa);
    }

    //printf("Kappa ====== %f\n", Kappa);
}



float GIPC::computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance) {

    calKineticGradient(instance->cudaVertPos, instance->cudaXTilta, instance->cudaFb, instance->cudaVertMass, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(mc_cpNum, 0, 5 * sizeof(uint32_t)));
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //calBarrierHessian();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    calBarrierGradientAndHessian(instance->cudaFb, m_IPCKappa);

    float time00 = 0;

    //calBarrierGradient(instance->cudaFb, Kappa);
#ifdef USE_FRICTION
    calFrictionGradient(instance->cudaFb, instance);
    calFrictionHessian(instance);
#endif

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    calculate_fem_gradient_hessian(instance->cudaDmInverses, instance->cudaVertPos, instance->cudaTetElement, m_BH->m_H12x12,
                                   h_cpNum[4] + h_cpNum_last[4], instance->cudaTetVolume,
                                   instance->cudaFb, tetrahedraNum, m_lengthRate, m_volumeRate, m_IPC_dt);


    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(m_BH->m_D4Index + h_cpNum[4] + h_cpNum_last[4], instance->cudaTetElement, tetrahedraNum * sizeof(uint4),cudaMemcpyDeviceToDevice));

    calculate_bending_gradient_hessian(instance->cudaVertPos, instance->cudaRestVertPos, instance->cudaTriEdges, instance->cudaTriEdgeAdjVertex, m_BH->m_H12x12, m_BH->m_D4Index, h_cpNum[4] + h_cpNum_last[4] + tetrahedraNum, instance->cudaFb, tri_edge_num, m_bendStiff, m_IPC_dt);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    calculate_triangle_fem_gradient_hessian(instance->cudaTriDmInverses, instance->cudaVertPos, instance->cudaTriElement, m_BH->m_H9x9,
                                            h_cpNum[3] + h_cpNum_last[3], instance->cudaTriArea, instance->cudaFb, m_triangleNum,
                                            m_stretchStiff, m_shearStiff, m_IPC_dt);


    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaMemcpy(m_BH->m_D3Index + h_cpNum[3] + h_cpNum_last[3], instance->cudaTriElement, m_triangleNum * sizeof(uint3), cudaMemcpyDeviceToDevice));

    computeGroundGradientAndHessian(instance->cudaFb);
    computeSoftConstraintGradientAndHessian(instance->cudaFb);


    return time00;
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}



double GIPC::Energy_Add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance) {
    int numbers = tetrahedraNum;

    if (type == 0 || type == 3) {
        numbers = vertexNum;
    }
    else if (type == 2) {
        numbers = h_cpNum[0];
    }
    else if (type == 4) {
        numbers = h_gpNum;
    }
    else if (type == 5) {
        numbers = h_cpNum_last[0];
    }
    else if (type == 6) {
        numbers = h_gpNum_last;
    }
    else if (type == 7 || type == 1) {
        numbers = tetrahedraNum;
    }
    else if (type == 8) {
        numbers = m_triangleNum;
    }
    else if (type == 9) {
        numbers = m_softNum;
    }
    else if (type == 10) {
        numbers = tri_edge_num;
    }
    if (numbers == 0) return 0;
    double* queue = m_pcg_data->m_squeue;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&queue, numbers * sizeof(double)));*/

    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch (type) {
    case 0:
        _getKineticEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (instance->cudaVertPos, instance->cudaXTilta, queue, instance->cudaVertMass, numbers);
        break;
    case 1:
        _getFEMEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaTetElement, instance->cudaDmInverses, instance->cudaTetVolume, numbers, m_lengthRate, m_volumeRate);
        break;
    case 2:
        _getBarrierEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaRestVertPos, mc_collisonPairs, m_IPCKappa, m_dHat, numbers);
        break;
    case 3:
        _getDeltaEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaFb, mc_moveDir, numbers);
        break;
    case 4:
        _computeGroundEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, mc_groundOffset, mc_groundNormal, mc_environment_collisionPair, m_dHat, m_IPCKappa, numbers);
        break;
    case 5:
        _getFrictionEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaOriginVertPos, _collisonPairs_lastH, numbers, m_IPC_dt, distCoord, tanBasis, lambda_lastH_scalar, m_fDhat * m_IPC_dt * m_IPC_dt, sqrt(m_fDhat) * m_IPC_dt);
        break;
    case 6:
        _getFrictionEnergy_gd_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaOriginVertPos, mc_groundNormal, _collisonPairs_lastH_gd, numbers, m_IPC_dt, lambda_lastH_scalar_gd, sqrt(m_fDhat) * m_IPC_dt);
        break;
    case 7:
        _getRestStableNHKEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaTetVolume, numbers, m_lengthRate, m_volumeRate);
        break;
    case 8:
        _get_triangleFEMEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaTriElement, instance->cudaTriDmInverses, instance->cudaTriArea, numbers, m_stretchStiff, m_shearStiff);
        break;
    case 9:
        _computeSoftConstraintEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaTargetVert, instance->cudaTargetIndex, m_softMotionRate, animation_fullRate, numbers);
        break;
    case 10:
        _getBendingEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaRestVertPos, instance->cudaTriEdges, instance->cudaTriEdgeAdjVertex, numbers, m_bendStiff);
        break;
    }
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __add_reduction << <blockNum, threadNum, sharedMsize >> > (queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    double result;
    cudaMemcpy(&result, queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(queue));
    return result;
}


double GIPC::computeEnergy(std::unique_ptr<GeometryManager>& instance) {
    double Energy = Energy_Add_Reduction_Algorithm(0, instance);

    Energy += m_IPC_dt * m_IPC_dt * Energy_Add_Reduction_Algorithm(1, instance);

    Energy += m_IPC_dt * m_IPC_dt * Energy_Add_Reduction_Algorithm(8, instance);

    Energy += m_IPC_dt * m_IPC_dt * Energy_Add_Reduction_Algorithm(10, instance);

    Energy += Energy_Add_Reduction_Algorithm(9, instance);

    Energy += Energy_Add_Reduction_Algorithm(2, instance);

    Energy += m_IPCKappa * Energy_Add_Reduction_Algorithm(4, instance);

#ifdef USE_FRICTION
    Energy += frictionRate * Energy_Add_Reduction_Algorithm(5, instance);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += frictionRate * Energy_Add_Reduction_Algorithm(6, instance);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    return Energy;
}

int GIPC::calculateMovingDirection(std::unique_ptr<GeometryManager>& instance, int cpNum, int preconditioner_type) {
    if (preconditioner_type == 0) {
        return PCGSOLVER::PCG_Process(instance, m_pcg_data, m_BH, mc_moveDir, vertexNum, tetrahedraNum, m_IPC_dt, m_meanVolumn, m_pcg_threshold);
    }
    else if (preconditioner_type == 1) {
        std::cout << "not support preconditioner type right now!" << std::endl;
        // int cgCount = MASPCG_Process(&instance, &m_pcg_data, m_BH, _moveDir, vertexNum, tetrahedraNum, IPC_dt, meanVolumn, cpNum, pcg_threshold);
        // if (cgCount == 3000) {
        //     printf("MASPCG fail, turn to PCG\n");
        //     cgCount = PCG_Process(&instance, &m_pcg_data, m_BH, _moveDir, vertexNum, tetrahedraNum, IPC_dt, meanVolumn, pcg_threshold);
        //     printf("PCG finish:  %d\n", cgCount);
        // }
    }

}


bool edgeTriIntersectionQuery(const int* _btype, const double3* _vertexes, const uint2* _edges, const uint3* _faces, const AABB* _edge_bvs, const Node* _edge_nodes, double dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));

    _edgeTriIntersectionQuery << <blockNum, threadNum >> > (_btype, _vertexes, _edges, _faces, _edge_bvs, _edge_nodes, _isIntersect, dHat, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if (h_isITST < 0) {
        return true;
    }
    return false;
}

bool GIPC::checkEdgeTriIntersectionIfAny(std::unique_ptr<GeometryManager>& instance)
{
    return edgeTriIntersectionQuery(m_bvh_e->mc_btype, instance->cudaVertPos, m_bvh_e->mc_edges, m_bvh_f->mc_faces, m_bvh_e->mc_boundVolumes, m_bvh_e->mc_nodes, m_dHat, m_bvh_f->m_face_number);
}

bool GIPC::checkGroundIntersection() {
    int numbers = h_gpNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //

    int* _isIntersect;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_isIntersect, sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection << <blockNum, threadNum >> > (mc_vertexes, mc_groundOffset, mc_groundNormal, mc_environment_collisionPair, _isIntersect, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_isIntersect));
    if (h_isITST < 0) {
        return true;
    }
    return false;

}

bool GIPC::isIntersected(std::unique_ptr<GeometryManager>& instance)
{
    if (checkGroundIntersection()) {
        return true;
    }

    if (checkEdgeTriIntersectionIfAny(instance)) {
        return true;
    }
    return false;
}

bool GIPC::lineSearch(std::unique_ptr<GeometryManager>& instance, double& alpha, const double& cfl_alpha) {

    bool stopped = false;
    //buildCP();
    double lastEnergyVal = computeEnergy(instance);
    double c1m = 0.0;
    double armijoParam = 0;
    if (armijoParam > 0.0) {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, instance);
    }

    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaTempDouble3Mem, instance->cudaVertPos, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

    stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, mc_moveDir, instance->cudaBoundaryType, alpha, false, vertexNum);

    bool rehash = true;

    buildBVH();
    //buildCP();
    //if (h_cpNum[0] > 0) system("pause");
    int numOfIntersect = 0;
    int insectNum = 0;

    bool checkInterset = true;

    while (checkInterset && isIntersected(instance)) {
        printf("type 0 intersection happened:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        numOfIntersect++;
        alpha = MATHUTILS::__m_min(cfl_alpha, alpha);
        stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, mc_moveDir, instance->cudaBoundaryType, alpha, false, vertexNum);
        buildBVH();
    }

    buildCP();
    //if (h_cpNum[0] > 0) system("pause");
    //rehash = false;

    //buildCollisionSets(mesh, sh, gd, true);
    double testingE = computeEnergy(instance);

    int numOfLineSearch = 0;
    double LFStepSize = alpha;
    //double temp_c1m = c1m;
    std::cout.precision(18);
    while ((testingE > lastEnergyVal + c1m * alpha) && alpha > 1e-3 * LFStepSize) {
        //printf("testE:    %f      lastEnergyVal:        %f         clm*alpha:    %f\n", testingE, lastEnergyVal, c1m * alpha);
        //std::cout <<numOfLineSearch<<  "   testE:    " << testingE << "      lastEnergyVal:        " << lastEnergyVal << std::endl;
        alpha /= 2.0;
        ++numOfLineSearch;

        stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, mc_moveDir, instance->cudaBoundaryType, alpha, false, vertexNum);

        buildBVH();
        buildCP();
        testingE = computeEnergy(instance);
    }

    if (numOfLineSearch > 8)
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        
    if (alpha < LFStepSize) {
        bool needRecomputeCS = false;
        while (checkInterset && isIntersected(instance)) {
            printf("type 1 intersection happened:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            numOfIntersect++;
            alpha = MATHUTILS::__m_min(cfl_alpha, alpha);
            stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, mc_moveDir, instance->cudaBoundaryType, alpha, false, vertexNum);
            buildBVH();
            needRecomputeCS = true;
        }
        if (needRecomputeCS) {
            buildCP();
        }
    }
    //printf("    lineSearch time step:  %f\n", alpha);


    return stopped;
}


void GIPC::postLineSearch(std::unique_ptr<GeometryManager>& instance, double alpha) {
    if (m_IPCKappa == 0.0) {
        initKappa(instance);
    }
    else {

        bool updateKappa = checkCloseGroundVal();
        if (!updateKappa) {
            updateKappa = checkSelfCloseVal();
        }
        if (updateKappa) {
            m_IPCKappa *= 2.0;
            upperBoundKappa(m_IPCKappa);
        }
        tempFree_closeConstraint();
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(mc_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(mc_close_gpNum, 0, sizeof(uint32_t)));

        computeCloseGroundVal();

        computeSelfCloseVal();
    }
    //printf("------------------------------------------Kappa: %f\n", Kappa);
}

void GIPC::tempMalloc_closeConstraint() {
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_closeConstraintID, h_gpNum * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_closeConstraintVal, h_gpNum * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_closeMConstraintID, h_cpNum[0] * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_closeMConstraintVal, h_cpNum[0] * sizeof(double)));
}

void GIPC::tempFree_closeConstraint() {
    CUDA_SAFE_CALL(cudaFree(mc_closeConstraintID));
    CUDA_SAFE_CALL(cudaFree(mc_closeConstraintVal));
    CUDA_SAFE_CALL(cudaFree(mc_closeMConstraintID));
    CUDA_SAFE_CALL(cudaFree(mc_closeMConstraintVal));
}

double maxCOllisionPairNum = 0;
double totalCollisionPairs = 0;
double total_Cg_count = 0;
double timemakePd = 0;
#include<vector>
#include<fstream>

std::vector<int> iterV;
int GIPC::solve_subIP(std::unique_ptr<GeometryManager>& instance, double& time0, double& time1, double& time2, double& time3, double& time4) {
    int iterCap = 10000, k = 0;

    CUDA_SAFE_CALL(cudaMemset(mc_moveDir, 0, vertexNum * sizeof(double3)));
    //m_BH->m_MALLOC_DEVICE_MEM_O(tetrahedraNum, h_cpNum + 1, h_gpNum);
    double totalTimeStep = 0;
    for (; k < iterCap; ++k) {
        totalCollisionPairs += h_cpNum[0];
        maxCOllisionPairNum = (maxCOllisionPairNum > h_cpNum[0]) ? maxCOllisionPairNum : h_cpNum[0];
        cudaEvent_t start, end0, end1 , end2, end3, end4;
        cudaEventCreate(&start);
        cudaEventCreate(&end0);
        cudaEventCreate(&end1);
        cudaEventCreate(&end2);
        cudaEventCreate(&end3);
        cudaEventCreate(&end4);

        m_BH->updateDNum(m_triangleNum, tetrahedraNum, h_cpNum + 1, h_cpNum_last + 1, tri_edge_num);

        //printf("collision num  %d\n", h_cpNum[0]);

        cudaEventRecord(start);


        timemakePd += computeGradientAndHessian(instance);


        double distToOpt_PN = calcMinMovement(mc_moveDir, m_pcg_data->m_squeue, vertexNum);

        bool gradVanish = (distToOpt_PN < sqrt(m_Newton_solver_threshold * m_Newton_solver_threshold * m_bboxDiagSize2 * m_IPC_dt * m_IPC_dt));
        if (k && gradVanish) {
            break;
        }
        cudaEventRecord(end0);
        total_Cg_count += calculateMovingDirection(instance, h_cpNum[0], m_pcg_data->m_P_type);

        cudaEventRecord(end1);
        double alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

        alpha = MATHUTILS::__m_min(alpha, ground_largestFeasibleStepSize(slackness_a, m_pcg_data->m_squeue));
        //alpha = MATHUTILS::__m_min(alpha, InjectiveStepSize(0.2, 1e-6, m_pcg_data->m_squeue, instance->cudaTetElement));
        alpha = MATHUTILS::__m_min(alpha, self_largestFeasibleStepSize(slackness_m, m_pcg_data->m_squeue, h_cpNum[0]));
        double temp_alpha = alpha;
        double alpha_CFL = alpha;

        double ccd_size = 1.0;
#ifdef USE_FRICTION
        ccd_size = 0.6;
#endif
        buildBVH_FULLCCD(temp_alpha);
        buildFullCP(temp_alpha);
        if (h_ccd_cpNum > 0) {
            double maxSpeed = cfl_largestSpeed(m_pcg_data->m_squeue);
            alpha_CFL = sqrt(m_dHat) / maxSpeed * 0.5;
            alpha = MATHUTILS::__m_min(alpha, alpha_CFL);
            if (temp_alpha > 2 * alpha_CFL) {
                /*buildBVH_FULLCCD(temp_alpha);
                buildFullCP(temp_alpha);*/
                alpha = MATHUTILS::__m_min(temp_alpha, self_largestFeasibleStepSize(slackness_m, m_pcg_data->m_squeue, h_ccd_cpNum) * ccd_size);
                alpha = MATHUTILS::__m_max(alpha, alpha_CFL);
            }
        }

        cudaEventRecord(end2);
        //printf("alpha:  %f\n", alpha);

        bool isStop = lineSearch(instance, alpha, alpha_CFL);
        cudaEventRecord(end3);
        postLineSearch(instance, alpha);
        cudaEventRecord(end4);
        //m_BH->m_FREE_DEVICE_MEM();
        //if (h_cpNum[0] > 0) return;
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        float time00, time11, time22, time33, time44;
        cudaEventElapsedTime(&time00, start, end0);
        cudaEventElapsedTime(&time11, end0, end1);
        //total_Cg_time += time1;
        cudaEventElapsedTime(&time22, end1, end2);
        cudaEventElapsedTime(&time33, end2, end3);
        cudaEventElapsedTime(&time44, end3, end4);
        time0 += time00;
        time1 += time11;
        time2 += time22;
        time3 += time33;
        time4 += time44;
        ////*cflTime = ptime;
        //printf("time0 = %f,  time1 = %f,  time2 = %f,  time3 = %f,  time4 = %f\n", time00, time11, time22, time33, time44);
        (cudaEventDestroy(start));
        (cudaEventDestroy(end0));
        (cudaEventDestroy(end1));
        (cudaEventDestroy(end2));
        (cudaEventDestroy(end3));
        (cudaEventDestroy(end4));
        totalTimeStep += alpha;

    }
    //iterV.push_back(k);
    //std::ofstream outiter("iterCount.txt");
    //for (int ii = 0;ii < iterV.size();ii++) {
    //    outiter << iterV[ii] << std::endl;
    //}
    //outiter.close();
    printf("\n\n Kappa: %f  iteration k:  %d\n\n\n", m_IPCKappa, k);
    return k;
   
}

void GIPC::updateVelocities(std::unique_ptr<GeometryManager>& instance) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateVelocities << <blockNum, threadNum >> > (instance->cudaVertPos, instance->cudaOriginVertPos, instance->cudaVertVel, instance->cudaBoundaryType, m_IPC_dt, numbers);
}

void GIPC::updateBoundary(std::unique_ptr<GeometryManager>& instance, double alpha) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateBoundary << <blockNum, threadNum >> > (instance->cudaVertPos, instance->cudaBoundaryType, mc_moveDir, alpha, numbers);
}

void GIPC::updateBoundaryMoveDir(std::unique_ptr<GeometryManager>& instance, double alpha) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateBoundaryMoveDir << <blockNum, threadNum >> > (instance->cudaVertPos, instance->cudaBoundaryType, mc_moveDir, m_IPC_dt, MATHUTILS::PI, alpha, numbers);
}

void GIPC::updateBoundary2(std::unique_ptr<GeometryManager>& instance) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateBoundary2<< <blockNum, threadNum >> > (instance->cudaBoundaryType, instance->cudaConstraints, numbers);
}






///////////////////////////////////////////
// GPU IPC 
///////////////////////////////////////////


GIPC::GIPC(std::unique_ptr<GeometryManager>& instance) 
    : m_instance(instance),
    m_scene_size(instance->AABB_SceneSize_ptr),
    m_bvh_f(instance->LBVH_F_ptr),
    m_bvh_e(instance->LBVH_E_ptr),
    m_pcg_data(instance->PCGData_ptr),
    m_BH(instance->BH_ptr) {

    mc_vertexes = instance->cudaVertPos;
    mc_rest_vertexes = instance->cudaRestVertPos;
    mc_faces = instance->cudaSurfFace;
    mc_edges = instance->cudaSurfEdge;
    mc_surfVerts = instance->cudaSurfVert;


    mc_targetVert = instance->cudaTargetVert;
    mc_targetInd = instance->cudaTargetIndex;
    mc_moveDir = instance->cudaMoveDir;
    mc_collisonPairs = instance->cudaCollisionPairs;
    mc_ccd_collisonPairs = instance->cudaCCDCollisionPairs;
    mc_cpNum = instance->cudaCPNum;
    mc_MatIndex = instance->cudaMatIndex;
    mc_close_cpNum = instance->cudaCloseCPNum;
    mc_environment_collisionPair = instance->cudaEnvCollisionPairs;
    mc_gpNum = instance->cudaGPNum;
    mc_close_gpNum = instance->cudaCloseGPNum;
    mc_groundNormal = instance->cudaGroundNormal;
    mc_groundOffset = instance->cudaGroundOffset;


    m_IPCKappa = instance->Kappa;
    m_dHat = instance->dHat;
    m_fDhat = instance->fDhat;
	m_bboxDiagSize2 = instance->bboxDiagSize2;
	m_relative_dhat = instance->relative_dhat;
    m_dTol = instance->dTol;
    m_minKappaCoef = instance->minKappaCoef;
    m_IPC_dt = instance->IPC_dt;
    m_meanMass = instance->meanMass;
    m_meanVolumn = instance->meanVolume;

    m_softNum = instance->softNum;
    m_triangleNum = instance->numTriElements;
    vertexNum = instance->numVertices;
    surf_vertexNum = instance->numSurfVerts;
    surf_edgeNum = instance->numSurfEdges;
    tri_edge_num = instance->triEdges.rows();
    surf_faceNum = instance->numSurfFaces;
    tetrahedraNum = instance->numTetElements;

    MAX_COLLITION_PAIRS_NUM = instance->MAX_COLLITION_PAIRS_NUM;
    MAX_CCD_COLLITION_PAIRS_NUM = instance->MAX_CCD_COLLITION_PAIRS_NUM;

    animation_subRate = instance->animation_subRate;
    animation_fullRate = instance->animation_fullRate;


    m_bendStiff = instance->bendStiff;
    m_density = instance->density;
    m_YoungModulus = instance->YoungModulus;
    m_PoissonRate = instance->PoissonRate;
    m_lengthRateLame = instance->lengthRateLame;
    m_volumeRateLame = instance->volumeRateLame;
    m_lengthRate = instance->lengthRate;
    m_volumeRate = instance->volumeRate;
    m_frictionRate = instance->frictionRate;
    m_clothThickness = instance->clothThickness;
    m_clothYoungModulus = instance->clothYoungModulus;
    m_stretchStiff = instance->stretchStiff;
    m_shearStiff = instance->shearStiff;
    m_clothDensity = instance->clothDensity;
    m_softMotionRate = instance->softMotionRate;
    m_Newton_solver_threshold = instance->Newton_solver_threshold;
    m_pcg_threshold = instance->pcg_threshold;

    h_cpNum_last[0] = 0;
    h_cpNum_last[1] = 0;
    h_cpNum_last[2] = 0;
    h_cpNum_last[3] = 0;
    h_cpNum_last[4] = 0;

    m_animation = instance->animation;

}


GIPC::~GIPC() { }



int totalNT = 0;
double totalTime = 0;
int total_Frames = 0;
double ttime0 = 0;
double ttime1 = 0;
double ttime2 = 0;
double ttime3 = 0;
double ttime4 = 0;
bool isRotate = false;



void GIPC::IPC_Solver() {

    CHECK_ERROR(m_instance, "not initialize m_instance");
    CHECK_ERROR(m_scene_size, "not initialize m_scene_size");
    CHECK_ERROR(m_bvh_f, "not initialize m_bvh_f");
    CHECK_ERROR(m_bvh_e, "not initialize m_bvh_e");
    CHECK_ERROR(m_pcg_data, "not initialize m_pcg_data");
    CHECK_ERROR(m_BH, "not initialize m_BH");


    //double animation_fullRate = 0;
    cudaEvent_t start, end0;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    double alpha = 1;
    cudaEventRecord(start);
//    if(isRotate&&total_Frames*IPC_dt>=2.2){
//        isRotate = false;
//        updateBoundary2(TetMesh);
//    }


    // if (isRotate)
    // {
    //     updateBoundaryMoveDir(instance, alpha);
    //     buildBVH_FULLCCD(alpha);
    //     buildFullCP(alpha);
    //     if (h_ccd_cpNum > 0) {
    //         double slackness_m = 0.8;
    //         alpha = MATHUTILS::__m_min(alpha, self_largestFeasibleStepSize(slackness_m, m_pcg_data->m_squeue, h_ccd_cpNum));
    //     }
    //     //updateBoundary(TetMesh, alpha);

    //     CUDA_SAFE_CALL(cudaMemcpy(TetMesh.temp_double3Mem, TetMesh.vertexes, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));
    //     updateBoundaryMoveDir(TetMesh, alpha);
    //     stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, 1, true, vertexNum);

    //     bool rehash = true;

    //     buildBVH();
    //     int numOfIntersect = 0;
    //     while (isIntersected(TetMesh)) {
    //         //printf("type 0 intersection happened\n");
    //         alpha /= 2.0;
    //         updateBoundaryMoveDir(TetMesh, alpha);
    //         numOfIntersect++;
    //         stepForward(TetMesh.vertexes, TetMesh.temp_double3Mem, _moveDir, TetMesh.BoundaryType, 1, true, vertexNum);
    //         buildBVH();
    //     }

    //     buildCP();
    // }


    upperBoundKappa(m_IPCKappa);
    if (m_IPCKappa < 1e-16) {
        suggestKappa(m_IPCKappa);
    }
    initKappa(m_instance);



#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar, h_cpNum[0] * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&distCoord, h_cpNum[0] * sizeof(double2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&tanBasis, h_cpNum[0] * sizeof(MATHUTILS::Matrix3x2d)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH, h_cpNum[0] * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex_last, h_cpNum[0] * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar_gd, h_gpNum * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH_gd, h_gpNum * sizeof(uint32_t)));
    buildFrictionSets();
#endif
    animation_fullRate = animation_subRate;
    int k = 0;
    double time0 = 0;
    double time1 = 0;
    double time2 = 0;
    double time3 = 0;
    double time4 = 0;

    while (true) {
    // for (int i = 0; i < 1; i++) {
        //if (h_cpNum[0] > 0) return;
        tempMalloc_closeConstraint();
        CUDA_SAFE_CALL(cudaMemset(mc_close_cpNum, 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(mc_close_gpNum, 0, sizeof(uint32_t)));

        totalNT += solve_subIP(m_instance, time0, time1, time2, time3, time4);

        double2 minMaxDist1 = minMaxGroundDist();
        double2 minMaxDist2 = minMaxSelfDist();

        double minDist = MATHUTILS::__m_min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = MATHUTILS::__m_max(minMaxDist1.y, minMaxDist2.y);
        
        bool finishMotion = animation_fullRate > 0.99 ? true : false;

        if (finishMotion) {
            if ((h_cpNum[0] + h_gpNum) > 0) {

                if (minDist < m_dTol) {
                    tempFree_closeConstraint();
                    break;
                }
                else if (maxDist < m_dHat) {
                    tempFree_closeConstraint();
                    break;
                }
                else {
                    tempFree_closeConstraint();
                }
            }
            else {
                tempFree_closeConstraint();
                break;
            }
        }
        else {
            tempFree_closeConstraint();
        }

        animation_fullRate += animation_subRate;
        
#ifdef USE_FRICTION
        CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
        CUDA_SAFE_CALL(cudaFree(distCoord));
        CUDA_SAFE_CALL(cudaFree(tanBasis));
        CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
        CUDA_SAFE_CALL(cudaFree(_MatIndex_last));

        CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
        CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));

        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar, h_cpNum[0] * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&distCoord, h_cpNum[0] * sizeof(double2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&tanBasis, h_cpNum[0] * sizeof(MATHUTILS::Matrix3x2d)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH, h_cpNum[0] * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_MatIndex_last, h_cpNum[0] * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&lambda_lastH_scalar_gd, h_gpNum * sizeof(double)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&_collisonPairs_lastH_gd, h_gpNum * sizeof(uint32_t)));
        buildFrictionSets();
#endif
    }


#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar));
    CUDA_SAFE_CALL(cudaFree(distCoord));
    CUDA_SAFE_CALL(cudaFree(tanBasis));
    CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH));
    CUDA_SAFE_CALL(cudaFree(_MatIndex_last));

    CUDA_SAFE_CALL(cudaFree(lambda_lastH_scalar_gd));
    CUDA_SAFE_CALL(cudaFree(_collisonPairs_lastH_gd));
#endif

    updateVelocities(m_instance);

    FEMENERGY::computeXTilta(m_instance, 1);
    cudaEventRecord(end0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    float tttime;
    cudaEventElapsedTime(&tttime, start, end0);
    totalTime += tttime;
    total_Frames++;
    printf("average time cost:     %f,    frame id:   %d\n", totalTime / totalNT, total_Frames);
    printf("boundary alpha: %f\n  finished a step\n", alpha);


    ttime0 += time0;
    ttime1 += time1;
    ttime2 += time2;
    ttime3 += time3;
    ttime4 += time4;

    std::ofstream outTime("timeCost.txt");

    outTime << "time0: " << ttime0 / 1000.0 << std::endl;
    outTime << "time1: " << ttime1 / 1000.0 << std::endl;
    outTime << "time2: " << ttime2 / 1000.0 << std::endl;
    outTime << "time3: " << ttime3 / 1000.0 << std::endl;
    outTime << "time4: " << ttime4 / 1000.0 << std::endl;
    outTime << "time_makePD: " << timemakePd / 1000.0 << std::endl;

    outTime << "totalTime: " << totalTime / 1000.0 << std::endl;
    outTime << "total iter: " << totalNT << std::endl;
    outTime << "frames: " << total_Frames << std::endl;
    outTime << "totalCollisionNum: " << totalCollisionPairs << std::endl;
    outTime << "averageCollision: " << totalCollisionPairs/totalNT << std::endl;
    outTime << "maxCOllisionPairNum: " << maxCOllisionPairNum << std::endl;
    outTime << "totalCgTime: " << total_Cg_count << std::endl;
    outTime.close();

}


