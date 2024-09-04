
#include "ImplicitIntergrator.cuh"
#include "FEM/FEMEnergy.cuh"
#include "ACCD/ACCD.cuh"
#include "IPC/IPCFriction.cuh"

#include "zensim/math/Complex.hpp"
#include "zensim/geometry/Distance.hpp"
#include "zensim/geometry/SpatialQuery.hpp"
#include "zensim/math/matrix/Eigen.hpp"
#include "zensim/math/MathUtils.h"


#define RANK 2


ImplicitIntegrator::ImplicitIntegrator(std::unique_ptr<GeometryManager>& instance) 
    : m_instance(instance),
    m_gipc(instance->GIPC_ptr),
    m_bvh_f(instance->LBVH_F_ptr),
    m_bvh_e(instance->LBVH_E_ptr),
    m_bvh_ef(instance->LBVH_EF_ptr),
    m_pcg_data(instance->PCGData_ptr)
    {}

ImplicitIntegrator::~ImplicitIntegrator() {};


__device__ bool device_cuda_error = false;
bool host_cuda_error = false;

__global__
void _reduct_min_groundTimeStep_to_double(const double3* vertexes, const uint32_t* surfVertIds, const double* g_offset, const double3* g_normal, const double3* moveDir, double* minStepSizes, double slackness, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    // array shared inside threads
    extern __shared__ double tep[];

    if (idx >= number) return;
    int svI = surfVertIds[idx];
    double temp = 1.0;
    double3 normal = *g_normal;
    double coef = MATHUTILS::__v_vec_dot(normal, moveDir[svI]);
    if (coef > 0.0) { // means point moving close to ground
        double dist = MATHUTILS::__v_vec_dot(normal, vertexes[svI]) - *g_offset; // projected dist along normal
        temp = coef / (dist * slackness); // after tempsubstep point will arrive ground
    }

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
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
        minStepSizes[blockIdx.x] = temp; // find minist step size for each thread block
        //printf("%f   %d\n", temp, blockIdx.x);
    }

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
void _reduct_min_selfTimeStep_to_double(const double3* vertexes, const int4* _ccd_collisionPairs, const double3* moveDir, double* minStepSizes, double slackness, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;
    double temp = 1.0;
    double CCDDistRatio = 1.0 - slackness;

    int4 MMCVIDI = _ccd_collisionPairs[idx];

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
            CHECK_ERROR_CUDA(Energy >= 0, "barrier enery less than zero pee", device_cuda_error);
            // if (Energy < 0)
            //     printf("I am pee\n");
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
                CHECK_ERROR_CUDA(Energy >= 0, "barrier enery less than zero pp", device_cuda_error);
                // if (Energy < 0)
                //     printf("I am pp\n");
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
                CHECK_ERROR_CUDA(Energy >= 0, "barrier enery less than zero ppe", device_cuda_error);
                // if (Energy < 0)
                //     printf("I am ppe\n");
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
void _getKineticEnergy_Reduction_3D(double3* _vertexes, double3* _xTilta, double* _energy, double* _masses, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= number) return;

    double temp = MATHUTILS::__squaredNorm(MATHUTILS::__minus(_vertexes[idx], _xTilta[idx])) * _masses[idx] * 0.5;

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
void _getFrictionEnergy_Reduction_3D(double* squeue, const double3* vertexes, const double3* o_vertexes, const int4* _collisionPair, int cpNum, double dt, const double2* distCoord, const MATHUTILS::Matrix3x2d* tanBasis, const double* lastH, double fricDHat, double eps) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = cpNum;
    if (idx >= numbers) return;

    double temp = __cal_Friction_energy(vertexes, o_vertexes, _collisionPair[idx], dt, distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
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
void _getFrictionEnergy_gd_Reduction_3D(double* squeue, const double3* vertexes, const double3* o_vertexes, const double3* _normal, const uint32_t* _collisionPair_gd, int gpNum, double dt, const double* lastH, double eps) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];
    int numbers = gpNum;
    if (idx >= numbers) return;

    double temp = __cal_Friction_gd_energy(vertexes, o_vertexes, _normal, _collisionPair_gd[idx], dt, lastH[idx], eps);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
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


void ImplicitIntegrator::updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number) {
    m_instance->DNum[1] = cpNums[1];
    m_instance->DNum[2] = cpNums[2] + tri_Num;
    m_instance->DNum[3] = tet_number + cpNums[3] + tri_edge_number;

#ifdef USE_FRICTION
    m_instance->DNum[1] += last_cpNums[1];
    m_instance->DNum[2] += last_cpNums[2];
    m_instance->DNum[3] += last_cpNums[3];
#endif

}

double calcMinMovement(const double3* _moveDir, double* _queue, const int& number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    MATHUTILS::_reduct_max_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _queue, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::__reduct_max_double << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);

    return minValue;
}


int ImplicitIntegrator::calculateMovingDirection(std::unique_ptr<GeometryManager>& instance, int cpNum, int preconditioner_type) {
    if (preconditioner_type == 0) {
        int cgCount = PCGSOLVER::PCG_Process(instance, m_pcg_data, instance->cudaMoveDir, instance->numVertices, instance->numTetElements, m_instance->IPC_dt, m_instance->meanVolume, m_instance->pcg_threshold);
        return cgCount;
    }
    else if (preconditioner_type == 1) {
        int cgCount = PCGSOLVER::MASPCG_Process(instance, m_pcg_data, instance->cudaMoveDir, instance->numVertices, instance->numTetElements, m_instance->IPC_dt, m_instance->meanVolume, cpNum, m_instance->pcg_threshold);
        if (cgCount == 3000) {
            printf("MASPCG fail, turn to PCG\n");
            cgCount = PCGSOLVER::PCG_Process(instance, m_pcg_data, instance->cudaMoveDir, instance->numVertices, instance->numTetElements, m_instance->IPC_dt, m_instance->meanVolume, m_instance->pcg_threshold);
            printf("PCG finish:  %d\n", cgCount);
        }
        return cgCount;
    } else {
        std::cerr << "precondtioner type should be 0/1 right now!" << std::endl;
        return 0;
    }

}


double ImplicitIntegrator::ground_largestFeasibleStepSize(double slackness, double* mqueue) {

    int numbers = m_instance->numSurfVerts;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));

    //if (instance->cpNum[0] > 0) {
    //    double3* mvd = new double3[vertexNum];
    //    cudaMemcpy(mvd, _moveDir, sizeof(double3) * vertexNum, cudaMemcpyDeviceToHost);
    //    for (int i = 0;i < vertexNum;i++) {
    //        printf("%f  %f  %f\n", mvd[i].x, mvd[i].y, mvd[i].z);
    //    }
    //    delete[] mvd;
    //}
    _reduct_min_groundTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (m_instance->cudaVertPos, m_instance->cudaSurfVert, m_instance->cudaGroundOffset, m_instance->cudaGroundNormal, m_instance->cudaMoveDir, mqueue, slackness, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::__reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_minSteps));
    return 1.0 / minValue;
}

double ImplicitIntegrator::InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets) {

    int numbers = m_instance->numTetElements;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    _reduct_min_InjectiveTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (m_instance->cudaVertPos, tets, m_instance->cudaMoveDir, mqueue, slackness, errorRate, numbers);


    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::__reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
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


double ImplicitIntegrator::self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers) {
    //slackness = 0.9;
    //int numbers = instance->cpNum[0];
    if (numbers < 1) return 1;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    //double* _minSteps;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&_minSteps, numbers * sizeof(double)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_min_selfTimeStep_to_double << <blockNum, threadNum, sharedMsize >> > (m_instance->cudaVertPos, m_instance->cudaCCDCollisionPairs, m_instance->cudaMoveDir, mqueue, slackness, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::__reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
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

double ImplicitIntegrator::cfl_largestSpeed(double* mqueue) {
    int numbers = m_instance->numSurfVerts;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    /*double* _maxV;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_maxV, numbers * sizeof(double)));*/
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_max_cfl_to_double << <blockNum, threadNum, sharedMsize >> > (m_instance->cudaMoveDir, mqueue, m_instance->cudaSurfVert, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::__reduct_max_double << <blockNum, threadNum, sharedMsize >> > (mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(_maxV));
    return minValue;
}




void ImplicitIntegrator::computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance) {

    // rhs = M * (x_tilta - (xn + dt*vn)))
    FEMENERGY::calKineticGradient(
        instance->cudaVertPos, 
        instance->cudaXTilta, 
        instance->cudaFb, 
        instance->cudaVertMass, 
        instance->numVertices);

    CUDA_SAFE_CALL(cudaMemset(instance->cudaCPNum, 0, 5 * sizeof(uint32_t)));

    // calculate barrier gradient and Hessian
    m_gipc->calBarrierGradientAndHessian(
        instance->cudaFb, 
        m_instance->Kappa);

#ifdef USE_FRICTION
    m_gipc->calFrictionGradient(instance->cudaFb, instance);
    m_gipc->calFrictionHessian(instance);
#endif

    // rhs += -dt^2 * vol * force
    // lhs += dt^2 * H12x12
    FEMENERGY::calculate_tetrahedra_fem_gradient_hessian(
        instance->cudaTetDmInverses, 
        instance->cudaVertPos, 
        instance->cudaTetElement, 
        instance->cudaH12x12,
        instance->cpNum[4] + instance->cpNumLast[4], 
        instance->cudaTetVolume,
        instance->cudaFb, 
        instance->numTetElements, 
        m_instance->lengthRate, 
        m_instance->volumeRate, 
        m_instance->IPC_dt);

    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaD4Index + instance->cpNum[4] + instance->cpNumLast[4], instance->cudaTetElement, instance->numTetElements * sizeof(uint4),cudaMemcpyDeviceToDevice));

    // rhs += -dt^2 * area * force
    // lhs += dt^2 * H9x9
    FEMENERGY::calculate_triangle_fem_gradient_hessian(
        instance->cudaTriDmInverses, 
        instance->cudaVertPos, 
        instance->cudaTriElement, 
        instance->cudaH9x9, 
        instance->cpNum[3] + instance->cpNumLast[3], 
        instance->cudaTriArea, 
        instance->cudaFb, 
        instance->numTriElements, 
        m_instance->stretchStiff, 
        m_instance->shearStiff, 
        m_instance->IPC_dt);
    
    FEMENERGY::calculate_bending_gradient_hessian(
        instance->cudaVertPos, 
        instance->cudaRestVertPos, 
        instance->cudaTriEdges, 
        instance->cudaTriEdgeAdjVertex, 
        instance->cudaH12x12, 
        instance->cudaD4Index, 
        instance->cpNum[4] + instance->cpNumLast[4] + instance->numTetElements, 
        instance->cudaFb, 
        instance->numTriEdges, 
        m_instance->bendStiff, 
        m_instance->IPC_dt);

    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaD3Index + instance->cpNum[3] + instance->cpNumLast[3], instance->cudaTriElement, instance->numTriElements * sizeof(uint3), cudaMemcpyDeviceToDevice));

    // calculate Ground gradient save in H3x3
    FEMENERGY::computeGroundGradientAndHessian(
        instance->cudaVertPos,
        instance->cudaGroundOffset,
        instance->cudaGroundNormal,
        instance->cudaEnvCollisionPairs,
        instance->cudaFb,
        instance->cudaGPNum,
        instance->cudaH3x3,
        instance->cudaD1Index,
        instance->gpNum,
        instance->DNum,
        instance->dHat,
        instance->Kappa
    );

    // calcukate Soft Constraint Gradient and Hessian
    FEMENERGY::computeSoftConstraintGradientAndHessian(
        instance->cudaVertPos,
        instance->cudaTargetVertPos,
        instance->cudaTargetIndex,
        instance->cudaGPNum,
        instance->cudaH3x3,
        instance->cudaD1Index,
        instance->cudaFb,
        instance->DNum,
        instance->softMotionRate,
        instance->softAnimationFullRate,
        instance->numSoftConstraints
    );

}



double ImplicitIntegrator::Energy_Add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->numTetElements;

    if (type == 0 || type == 3) {
        numbers = instance->numVertices;
    }
    else if (type == 2) {
        numbers = instance->cpNum[0];
    }
    else if (type == 4) {
        numbers = instance->gpNum;
    }
    else if (type == 5) {
        numbers = instance->cpNumLast[0];
    }
    else if (type == 6) {
        numbers = instance->gpNumLast;
    }
    else if (type == 7 || type == 1) {
        numbers = instance->numTetElements;
    }
    else if (type == 8) {
        numbers = instance->numTriElements;
    }
    else if (type == 9) {
        numbers = instance->numSoftConstraints;
    }
    else if (type == 10) {
        numbers = instance->numTriEdges;
    }
    if (numbers == 0) return 0;
    double* queue = m_pcg_data->mc_squeue;
    //CUDA_SAFE_CALL(cudaMalloc((void**)&queue, numbers * sizeof(double)));*/

    const unsigned int threadNum = 256;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch (type) {
    case 0:
        _getKineticEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (instance->cudaVertPos, instance->cudaXTilta, queue, instance->cudaVertMass, numbers);
        break;
    case 1:
        _getFEMEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaTetElement, instance->cudaTetDmInverses, instance->cudaTetVolume, numbers, m_instance->lengthRate, m_instance->volumeRate);
        break;
    case 2:
        _getBarrierEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaRestVertPos, instance->cudaCollisionPairs, m_instance->Kappa, m_instance->dHat, numbers);
        break;
    case 3:
        _getDeltaEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaFb, instance->cudaMoveDir, numbers);
        break;
    case 4:
        _computeGroundEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaGroundOffset, instance->cudaGroundNormal, instance->cudaEnvCollisionPairs, m_instance->dHat, m_instance->Kappa, numbers);
        break;
    case 5:
        _getFrictionEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaOriginVertPos, instance->cudaCollisonPairsLastH, numbers, m_instance->IPC_dt, instance->cudaDistCoord, instance->cudaTanBasis, instance->cudaLambdaLastHScalar, m_instance->fDhat * m_instance->IPC_dt * m_instance->IPC_dt, sqrt(m_instance->fDhat) * m_instance->IPC_dt);
        break;
    case 6:
        _getFrictionEnergy_gd_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaOriginVertPos, instance->cudaGroundNormal, instance->cudaCollisonPairsLastHGd, numbers, m_instance->IPC_dt, instance->cudaLambdaLastHScalarGd, sqrt(m_instance->fDhat) * m_instance->IPC_dt);
        break;
    case 7:
        _getRestStableNHKEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaTetVolume, numbers, m_instance->lengthRate, m_instance->volumeRate);
        break;
    case 8:
        _get_triangleFEMEnergy_Reduction_3D << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaTriElement, instance->cudaTriDmInverses, instance->cudaTriArea, numbers, m_instance->stretchStiff, m_instance->shearStiff);
        break;
    case 9:
        _computeSoftConstraintEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaTargetVertPos, instance->cudaTargetIndex, m_instance->softMotionRate, m_instance->softAnimationFullRate, numbers);
        break;
    case 10:
        _getBendingEnergy_Reduction << <blockNum, threadNum, sharedMsize >> > (queue, instance->cudaVertPos, instance->cudaRestVertPos, instance->cudaTriEdges, instance->cudaTriEdgeAdjVertex, numbers, m_instance->bendStiff);
        break;
    }
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        MATHUTILS::__add_reduction << <blockNum, threadNum, sharedMsize >> > (queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    double result;
    cudaMemcpy(&result, queue, sizeof(double), cudaMemcpyDeviceToHost);
    //CUDA_SAFE_CALL(cudaFree(queue));
    return result;
}


double ImplicitIntegrator::computeEnergy(std::unique_ptr<GeometryManager>& instance) {
    double Energy = Energy_Add_Reduction_Algorithm(0, instance);

    Energy += m_instance->IPC_dt * m_instance->IPC_dt * Energy_Add_Reduction_Algorithm(1, instance);

    Energy += m_instance->IPC_dt * m_instance->IPC_dt * Energy_Add_Reduction_Algorithm(8, instance);

    Energy += m_instance->IPC_dt * m_instance->IPC_dt * Energy_Add_Reduction_Algorithm(10, instance);

    Energy += Energy_Add_Reduction_Algorithm(9, instance);

    Energy += Energy_Add_Reduction_Algorithm(2, instance);

    Energy += m_instance->Kappa * Energy_Add_Reduction_Algorithm(4, instance);

#ifdef USE_FRICTION
    Energy += m_instance->frictionRate * Energy_Add_Reduction_Algorithm(5, instance);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Energy += m_instance->frictionRate * Energy_Add_Reduction_Algorithm(6, instance);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
#endif
    return Energy;
}


__global__
void _stepForward(double3* _vertexes, double3* _vertexesTemp, double3* _moveDir, int* bType, double alpha, bool moveBoundary, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (abs(bType[idx]) != 1 || moveBoundary) {
        _vertexes[idx] = MATHUTILS::__minus(_vertexesTemp[idx], MATHUTILS::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

void stepForward(double3* _vertexes, double3* _vertexesTemp, double3* _moveDir, int* bType, double alpha, bool moveBoundary, int numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _stepForward << <blockNum, threadNum >> > (_vertexes, _vertexesTemp, _moveDir, bType, alpha, moveBoundary, numbers);
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


void ImplicitIntegrator::updateVelocities(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->numVertices;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateVelocities << <blockNum, threadNum >> > (instance->cudaVertPos, instance->cudaOriginVertPos, instance->cudaVertVel, instance->cudaBoundaryType, m_instance->IPC_dt, numbers);
}

void ImplicitIntegrator::tempMalloc_closeConstraint() {
    CUDAMallocSafe(m_instance->cudaCloseConstraintID, m_instance->gpNum);
    CUDAMallocSafe(m_instance->cudaCloseConstraintVal, m_instance->gpNum);
    CUDAMallocSafe(m_instance->cudaCloseMConstraintID, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaCloseMConstraintVal, m_instance->cpNum[0]);
}

void ImplicitIntegrator::tempFree_closeConstraint() {
    CUDAFreeSafe(m_instance->cudaCloseConstraintID);
    CUDAFreeSafe(m_instance->cudaCloseConstraintVal);
    CUDAFreeSafe(m_instance->cudaCloseMConstraintID);
    CUDAFreeSafe(m_instance->cudaCloseMConstraintVal);
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

void ImplicitIntegrator::computeCloseGroundVal() {
    int numbers = m_instance->gpNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _computeGroundCloseVal << <blockNum, threadNum >> > (m_instance->cudaVertPos, m_instance->cudaGroundOffset, m_instance->cudaGroundNormal, m_instance->cudaEnvCollisionPairs, m_instance->dTol, m_instance->cudaCloseConstraintID, m_instance->cudaCloseConstraintVal, m_instance->cudaCloseGPNum, numbers);

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

void ImplicitIntegrator::computeSelfCloseVal() {
    int numbers = m_instance->cpNum[0];
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum; //
    _calSelfCloseVal << <blockNum, threadNum >> > (m_instance->cudaVertPos, m_instance->cudaCollisionPairs, m_instance->cudaCloseMConstraintID, m_instance->cudaCloseMConstraintVal, m_instance->cudaCloseCPNum, m_instance->dTol, numbers);
}


double2 ImplicitIntegrator::minMaxSelfDist() {
    int numbers = m_instance->cpNum[0];
    if (numbers < 1)return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist << <blockNum, threadNum, sharedMsize >> > (m_instance->cudaVertPos, m_instance->cudaCollisionPairs, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::_reduct_M_double2 << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
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

double2 ImplicitIntegrator::minMaxGroundDist() {
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes, _groundOffset, _groundNormal, _isChange, _closeConstraintID, _closeConstraintVal, numbers);

    int numbers = m_instance->gpNum;
    if (numbers < 1)return make_double2(1e32, 0);
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double2) * (threadNum >> 5);

    double2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(double2)));
    //CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist << <blockNum, threadNum, sharedMsize >> > (m_instance->cudaVertPos, m_instance->cudaGroundOffset, m_instance->cudaGroundNormal, m_instance->cudaEnvCollisionPairs, _queue, numbers);
    //_reduct_min_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::_reduct_M_double2 << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
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


void ImplicitIntegrator::lineSearch(std::unique_ptr<GeometryManager>& instance, double& alpha, const double& cfl_alpha) {

    //buildCP();
    double lastEnergyVal = computeEnergy(instance);
    double c1m = 0.0;
    double armijoParam = 0;
    if (armijoParam > 0.0) {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, instance);
    }

    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaTempDouble3Mem, instance->cudaVertPos, instance->numVertices * sizeof(double3), cudaMemcpyDeviceToDevice));

    stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, instance->cudaMoveDir, instance->cudaBoundaryType, alpha, false, instance->numVertices);

    m_gipc->buildBVH();

    // if (instance->cpNum[0] > 0) system("pause");
    int numOfIntersect = 0;
    int insectNum = 0;

    bool checkInterset = true;

    // if under all ACCD/Ground/CFL defined alpha, still intersection happens
    // then we return back to alpha/2 util find collision free alpha 
    while (checkInterset && m_gipc->isIntersected(instance)) {
        printf("type 0 intersection happened:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        numOfIntersect++;
        alpha = MATHUTILS::__m_min(cfl_alpha, alpha);
        stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, instance->cudaMoveDir, instance->cudaBoundaryType, alpha, false, instance->numVertices);
        m_gipc->buildBVH();
    }

    m_gipc->buildCP();
    //if (instance->cpNum[0] > 0) system("pause");

    //buildCollisionSets(mesh, sh, gd, true);
    double testingE = computeEnergy(instance);

    int numOfLineSearch = 0;
    double LFStepSize = alpha;
    //double temp_c1m = c1m;

    while ((testingE > lastEnergyVal + c1m * alpha) && alpha > 1e-3 * LFStepSize) {
        printf("Enery not drop down, testE:%f, lastEnergyVal:%f, clm*alpha:%f\n", testingE, lastEnergyVal, c1m * alpha);
        alpha /= 2.0;
        ++numOfLineSearch;
        stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, instance->cudaMoveDir, instance->cudaBoundaryType, alpha, false, instance->numVertices);
        m_gipc->buildBVH();
        m_gipc->buildCP();
        testingE = computeEnergy(instance);
        CHECK_ERROR_CUDA(numOfLineSearch <= 10, "energy not drops down correctly in more than ten iterations\n", host_cuda_error);
        if (numOfLineSearch > 10) return;
    }
        
    // if alpha fails down in past process, then check again will there be intersection again
    if (alpha < LFStepSize) {
        bool needRecomputeCS = false;
        while (checkInterset && m_gipc->isIntersected(instance)) {
            printf("type 1 intersection happened:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            numOfIntersect++;
            alpha = MATHUTILS::__m_min(cfl_alpha, alpha);
            stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, instance->cudaMoveDir, instance->cudaBoundaryType, alpha, false, instance->numVertices);
            m_gipc->buildBVH();
            needRecomputeCS = true;
        }
        if (needRecomputeCS) {
            m_gipc->buildCP();
        }
    }
    printf("lineSearch time step:  %f\n", alpha);

}


void ImplicitIntegrator::postLineSearch(std::unique_ptr<GeometryManager>& instance, double alpha) {
    if (m_instance->Kappa == 0.0) {
        m_gipc->initKappa(instance);
    }
    else {

        tempFree_closeConstraint();
        tempMalloc_closeConstraint();
        CUDAMemcpyHToDSafe(instance->cudaCloseCPNum, Eigen::VectorXi::Zero(1));
        CUDAMemcpyHToDSafe(instance->cudaCloseGPNum, Eigen::VectorXi::Zero(1));
        
        computeCloseGroundVal();
        computeSelfCloseVal();
    }
    //printf("------------------------------------------Kappa: %f\n", Kappa);
}





int ImplicitIntegrator::solve_subIP(std::unique_ptr<GeometryManager>& instance) {

    std::cout.precision(18);

    int iterCap = 10000, iterk = 0;
    CUDA_SAFE_CALL(cudaMemset(instance->cudaMoveDir, 0, instance->numVertices * sizeof(double3)));

    instance->totalPCGCount = 0;
    instance->totalCollisionPairs = 0;

    for (; iterk < iterCap; ++iterk) {

        instance->totalCollisionPairs += instance->cpNum[0];
        
        updateDNum(instance->numTriElements, instance->numTetElements, instance->cpNum + 1, instance->cpNumLast + 1, instance->numTriEdges);

        // calculate gradient gradx(g) and Hessian gradx^2(g)
        computeGradientAndHessian(instance);

        double distToOpt_PN = calcMinMovement(instance->cudaMoveDir, m_pcg_data->mc_squeue, instance->numVertices);
        // line search iteration stop 
        bool gradVanish = (distToOpt_PN < sqrt(instance->Newton_solver_threshold * instance->Newton_solver_threshold * instance->bboxDiagSize2 * instance->IPC_dt * instance->IPC_dt));
        if (iterk > 0 && gradVanish) {
            break;
        }

        // solve PCG with MAS Preconditioner and get instance->cudaMoveDir (i.e. dx)
        instance->totalPCGCount += calculateMovingDirection(instance, instance->cpNum[0], instance->precondType);

        double alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

        alpha = MATHUTILS::__m_min(alpha, ground_largestFeasibleStepSize(slackness_a, m_pcg_data->mc_squeue));
        // alpha = MATHUTILS::__m_min(alpha, InjectiveStepSize(0.2, 1e-6, m_pcg_data->mc_squeue, instance->cudaTetElement));
        alpha = MATHUTILS::__m_min(alpha, self_largestFeasibleStepSize(slackness_m, m_pcg_data->mc_squeue, instance->cpNum[0]));
        
        double temp_alpha = alpha;
        double alpha_CFL = alpha;

        double ccd_size = 1.0;
#ifdef USE_FRICTION
        ccd_size = 0.6;
#endif

        // build BVH tree of type ccd, get collision pairs num instance->ccdCpNum, 
        // if instance->ccdCpNum > 0, means there will be collision in temp_alpha substep
        m_gipc->buildBVH_FULLCCD(temp_alpha);
        m_gipc->buildFullCP(temp_alpha);
        if (instance->ccdCpNum > 0) {
            // obtain max velocity of moveDir
            double maxSpeed = cfl_largestSpeed(m_pcg_data->mc_squeue);
            alpha_CFL = sqrt(instance->dHat) / maxSpeed * 0.5;
            alpha = MATHUTILS::__m_min(alpha, alpha_CFL);
            if (temp_alpha > 2 * alpha_CFL) {
                alpha = MATHUTILS::__m_min(temp_alpha, self_largestFeasibleStepSize(slackness_m, m_pcg_data->mc_squeue, instance->ccdCpNum) * ccd_size);
                alpha = MATHUTILS::__m_max(alpha, alpha_CFL);
            }
        }

        //printf("alpha:  %f\n", alpha);

        lineSearch(instance, alpha, alpha_CFL);
        postLineSearch(instance, alpha);

        CUDA_SAFE_CALL(cudaDeviceSynchronize());

    }
    
    printf("\n");
    printf("Kappa: %f  iteration k:  %d \n", instance->Kappa, iterk);
    std::cout << "instance->totalPCGCount: " << instance->totalPCGCount << std::endl;
    std::cout << "instance->totalCollisionPairs: " << instance->totalCollisionPairs << std::endl;
    printf("\n");

    return iterk;
   
}








bool ImplicitIntegrator::IPC_Solver() {

    host_cuda_error = false;
    device_cuda_error = false;
    // cudaMemcpyToSymbol(cuda_error, &host_cuda_error, sizeof(bool));

    // calculate a lowerbound and upperbound of a kappa, mainly to keep stability of the system
    m_gipc->upperBoundKappa(m_instance->Kappa);
    if (m_instance->Kappa < 1e-16) {
        // init Kappa, basically only active for 1st frame, to give you a first suggest kappa value.
        m_gipc->suggestKappa(m_instance->Kappa);
    }
    m_gipc->initKappa(m_instance);


#ifdef USE_FRICTION
    CUDAMallocSafe(m_instance->cudaLambdaLastHScalar, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaDistCoord, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaTanBasis, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaCollisonPairsLastH, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaMatIndexLast, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaLambdaLastHScalarGd, m_instance->gpNum);
    CUDAMallocSafe(m_instance->cudaCollisonPairsLastHGd, m_instance->gpNum);
    m_gipc->buildFrictionSets();

#endif

    m_instance->softAnimationFullRate = m_instance->softAnimationSubRate;

    while (true) {
        tempMalloc_closeConstraint();

        CUDAMemcpyHToDSafe(m_instance->cudaCloseCPNum, Eigen::VectorXi::Zero(1));
        CUDAMemcpyHToDSafe(m_instance->cudaCloseGPNum, Eigen::VectorXi::Zero(1));

        solve_subIP(m_instance);

        double2 minMaxDist1 = minMaxGroundDist();
        double2 minMaxDist2 = minMaxSelfDist();

        double minDist = MATHUTILS::__m_min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = MATHUTILS::__m_max(minMaxDist1.y, minMaxDist2.y);
        
        bool finishMotion = m_instance->softAnimationFullRate > 0.99 ? true : false;

        if (finishMotion) {
            if ((m_instance->cpNum[0] + m_instance->gpNum) > 0) {

                if (minDist < m_instance->dTol) {
                    tempFree_closeConstraint();
                    break;
                }
                else if (maxDist < m_instance->dHat) {
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

        m_instance->softAnimationFullRate += m_instance->softAnimationSubRate;
        
#ifdef USE_FRICTION
        CUDAFreeSafe(m_instance->cudaLambdaLastHScalar);
        CUDAFreeSafe(m_instance->cudaDistCoord);
        CUDAFreeSafe(m_instance->cudaTanBasis);
        CUDAFreeSafe(m_instance->cudaCollisonPairsLastH);
        CUDAFreeSafe(m_instance->cudaMatIndexLast);
        CUDAFreeSafe(m_instance->cudaLambdaLastHScalarGd);
        CUDAFreeSafe(m_instance->cudaCollisonPairsLastHGd);

        CUDAMallocSafe(m_instance->cudaLambdaLastHScalar, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaDistCoord, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaTanBasis, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaCollisonPairsLastH, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaMatIndexLast, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaLambdaLastHScalarGd, m_instance->gpNum);
        CUDAMallocSafe(m_instance->cudaCollisonPairsLastHGd, m_instance->gpNum);
        m_gipc->buildFrictionSets();
#endif
    }


#ifdef USE_FRICTION
    CUDAFreeSafe(m_instance->cudaLambdaLastHScalar);
    CUDAFreeSafe(m_instance->cudaDistCoord);
    CUDAFreeSafe(m_instance->cudaTanBasis);
    CUDAFreeSafe(m_instance->cudaCollisonPairsLastH);
    CUDAFreeSafe(m_instance->cudaMatIndexLast);
    CUDAFreeSafe(m_instance->cudaLambdaLastHScalarGd);
    CUDAFreeSafe(m_instance->cudaCollisonPairsLastHGd);
#endif

    updateVelocities(m_instance);

    FEMENERGY::computeXTilta(
        m_instance->cudaBoundaryType,
        m_instance->cudaVertVel,
        m_instance->cudaOriginVertPos,
        m_instance->cudaXTilta,
        m_instance->IPC_dt,
        m_instance->numSIMVertPos,
        1
    );

    CUDA_SAFE_CALL(cudaDeviceSynchronize());


    bool _host_cuda_error;
    cudaMemcpyFromSymbol(&_host_cuda_error, device_cuda_error, sizeof(bool));
    return host_cuda_error || _host_cuda_error;

}


