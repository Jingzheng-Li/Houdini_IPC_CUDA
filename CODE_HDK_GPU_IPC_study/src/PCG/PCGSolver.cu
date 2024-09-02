
#include "PCGSolver.cuh"


/////////////////////////////
// update PCGData 
/////////////////////////////

PCGData::PCGData(std::unique_ptr<GeometryManager>& instance) {
    m_precondType = instance->precondType;
};

PCGData::~PCGData() {};

void PCGData::CUDA_MALLOC_PCGDATA(const int& vertexNum, const int& tetrahedraNum) {
    CUDAMallocSafe(mc_squeue, MATHUTILS::__m_max(vertexNum, tetrahedraNum));
    CUDAMallocSafe(mc_P, vertexNum);
    CUDAMallocSafe(mc_r, vertexNum);
    CUDAMallocSafe(mc_c, vertexNum);
    CUDAMallocSafe(mc_z, vertexNum);
    CUDAMallocSafe(mc_q, vertexNum);
    CUDAMallocSafe(mc_s, vertexNum);
    CUDAMallocSafe(mc_dx, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(mc_z, 0, vertexNum * sizeof(double3)));

    if (m_precondType == 1) {
        CUDAMallocSafe(mc_preconditionTempVec3, vertexNum);
        CUDAMallocSafe(mc_filterTempVec3, vertexNum);
    }

}

void PCGData::CUDA_FREE_PCGDATA() {
    CUDAFreeSafe(mc_squeue);
    CUDAFreeSafe(mc_r);
    CUDAFreeSafe(mc_r);
    CUDAFreeSafe(mc_c);
    CUDAFreeSafe(mc_z);
    CUDAFreeSafe(mc_q);
    CUDAFreeSafe(mc_s);
    CUDAFreeSafe(mc_dx);
    
    if (m_precondType == 1) {
        CUDAFreeSafe(mc_filterTempVec3);
        CUDAFreeSafe(mc_preconditionTempVec3);
    }
}


namespace PCGSOLVER {

__global__ 
void PCG_vdv_Reduction(double* squeue, const double3* a, const double3* b, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    //double3 t_b = b[idx];

    double temp = MATHUTILS::__v_vec_dot(a[idx], b[idx]);//MATHUTILS::__norm(t_b);//MATHUTILS::__mabs(t_b.x) + MATHUTILS::__mabs(t_b.y) + MATHUTILS::__mabs(t_b.z);

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
void add_reduction(double* mem, int numbers) {
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
void PCG_add_Reduction_delta0(double* squeue, const MATHUTILS::Matrix3x3d* P, const double3* b, const MATHUTILS::Matrix3x3d* constraint, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;

    //double3 t_P = P[idx];
    double3 t_b = b[idx];
    MATHUTILS::Matrix3x3d t_constraint = constraint[idx];
    /*MATHUTILS::Matrix3x3d PInverse;
    MATHUTILS::__Inverse(P[idx], PInverse);*/
    //double vx = 1 / t_P.x, vy = 1 / t_P.y, vz = 1 / t_P.z;
    double3 filter_b = MATHUTILS::__M_v_multiply(t_constraint, t_b);

    double temp = MATHUTILS::__v_vec_dot(MATHUTILS::__v_M_multiply(filter_b, P[idx]), filter_b);//filter_b.x * filter_b.x * vx + filter_b.y * filter_b.y * vy + filter_b.z * filter_b.z * vz;

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
void PCG_add_Reduction_deltaN0(double* squeue, const MATHUTILS::Matrix3x3d* P, const double3* b, double3* r, double3* c, const MATHUTILS::Matrix3x3d* constraint, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //double3 t_P = P[idx];
    /*MATHUTILS::Matrix3x3d PInverse;
    MATHUTILS::__Inverse(P[idx], PInverse);*/
    double3 t_b = b[idx];
    MATHUTILS::Matrix3x3d t_constraint = constraint[idx];
    double3 t_r = MATHUTILS::__M_v_multiply(t_constraint, MATHUTILS::__minus(t_b, r[idx]));
    double3 t_c = MATHUTILS::__M_v_multiply(P[idx], t_r);//MATHUTILS::__v_vec_multiply(t_r, make_double3(1 / t_P.x, 1 / t_P.y, 1 / t_P.z));
    t_c = MATHUTILS::__M_v_multiply(t_constraint, t_c);
    r[idx] = t_r;
    c[idx] = t_c;

    double temp = MATHUTILS::__v_vec_dot(t_r, t_c);

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
void PCG_add_Reduction_deltaN(double* squeue, double3* dx, const double3* c, double3* r, const double3* q, const MATHUTILS::Matrix3x3d* P, double3* s, double alpha, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    //double3 t_P = P[idx];
    /*MATHUTILS::Matrix3x3d PInverse;
    MATHUTILS::__Inverse(P[idx], PInverse);*/
    double3 t_c = c[idx];
    double3 t_dx = dx[idx];
    double3 t_r = r[idx];
    double3 t_q = q[idx];
    double3 t_s = s[idx];

    dx[idx] = MATHUTILS::__add(t_dx, MATHUTILS::__s_vec_multiply(t_c, alpha));
    t_r = MATHUTILS::__add(t_r, MATHUTILS::__s_vec_multiply(t_q, -alpha));
    r[idx] = t_r;
    t_s = MATHUTILS::__M_v_multiply(P[idx], t_r);//MATHUTILS::__v_vec_multiply(t_r, make_double3(1 / t_P.x, 1 / t_P.y, 1 / t_P.z));
    s[idx] = t_s;

    double temp = MATHUTILS::__v_vec_dot(t_r, t_s);

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
void PCG_add_Reduction_tempSum(double* squeue, const double3* c, double3* q, const MATHUTILS::Matrix3x3d* constraint, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 t_c = c[idx];
    double3 t_q = q[idx];
    MATHUTILS::Matrix3x3d t_constraint = constraint[idx];
    t_q = MATHUTILS::__M_v_multiply(t_constraint, t_q);
    q[idx] = t_q;

    double temp = MATHUTILS::__v_vec_dot(t_q, t_c);

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
void PCG_add_Reduction_force(double* squeue, const double3* b, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double3 t_b = b[idx];

    double temp = MATHUTILS::__norm(t_b);//MATHUTILS::__mabs(t_b.x) + MATHUTILS::__mabs(t_b.y) + MATHUTILS::__mabs(t_b.z);

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
void __PCG_FinalStep_UpdateC(const MATHUTILS::Matrix3x3d* constraints, const double3* s, double3* c, double rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double3 tempc = MATHUTILS::__add(s[idx], MATHUTILS::__s_vec_multiply(c[idx], rate));
    c[idx] = MATHUTILS::__M_v_multiply(constraints[idx], tempc);
}

__global__ 
void __PCG_constraintFilter(const MATHUTILS::Matrix3x3d* constraints, const double3* input, double3* output, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    output[idx] = MATHUTILS::__M_v_multiply(constraints[idx], input[idx]);
}

__global__ 
void __PCG_initDX(double3* dx, const double3* z, double rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    double3 tz = z[idx];
    dx[idx] = make_double3(tz.x * rate, tz.y * rate, tz.z * rate);
}


__global__ 
void __PCG_Solve_AX12_b(const MATHUTILS::Matrix12x12d* Hessians, const uint4* D4Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    MATHUTILS::Matrix12x12d H = Hessians[idx];
    MATHUTILS::Vector12 tempC, tempQ;

    tempC.v[0] = c[D4Index[idx].x].x;
    tempC.v[1] = c[D4Index[idx].x].y;
    tempC.v[2] = c[D4Index[idx].x].z;

    tempC.v[3] = c[D4Index[idx].y].x;
    tempC.v[4] = c[D4Index[idx].y].y;
    tempC.v[5] = c[D4Index[idx].y].z;

    tempC.v[6] = c[D4Index[idx].z].x;
    tempC.v[7] = c[D4Index[idx].z].y;
    tempC.v[8] = c[D4Index[idx].z].z;

    tempC.v[9] = c[D4Index[idx].w].x;
    tempC.v[10] = c[D4Index[idx].w].y;
    tempC.v[11] = c[D4Index[idx].w].z;

    tempQ = MATHUTILS::__M12x12_v12_multiply(H, tempC);

    atomicAdd(&(q[D4Index[idx].x].x), tempQ.v[0]);
    atomicAdd(&(q[D4Index[idx].x].y), tempQ.v[1]);
    atomicAdd(&(q[D4Index[idx].x].z), tempQ.v[2]);

    atomicAdd(&(q[D4Index[idx].y].x), tempQ.v[3]);
    atomicAdd(&(q[D4Index[idx].y].y), tempQ.v[4]);
    atomicAdd(&(q[D4Index[idx].y].z), tempQ.v[5]);

    atomicAdd(&(q[D4Index[idx].z].x), tempQ.v[6]);
    atomicAdd(&(q[D4Index[idx].z].y), tempQ.v[7]);
    atomicAdd(&(q[D4Index[idx].z].z), tempQ.v[8]);

    atomicAdd(&(q[D4Index[idx].w].x), tempQ.v[9]);
    atomicAdd(&(q[D4Index[idx].w].y), tempQ.v[10]);
    atomicAdd(&(q[D4Index[idx].w].z), tempQ.v[11]);
}

__global__ 
void __PCG_Solve_AX12_b1(const MATHUTILS::Matrix12x12d* Hessians, const uint4* D4Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    extern __shared__ double sData[];
    __shared__ int offset;
    int Hid = idx / 144;
    int MRid = (idx % 144) / 12;
    int MCid = (idx % 144) % 12;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 12;
    sData[threadIdx.x] = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (12 - GRtid);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 12) / 12;
    int Num;// = 12 + BRid - GRtid;
    int startId = offset + BRid * 12 - 12;
    int landidx = (threadIdx.x - offset) % 12;
    if (BRid == 0) {
        Num = offset;
        startId = 0;
        landidx = threadIdx.x;
    }
    else if (BRid * 12 + offset > blockDim.x) {
        Num = blockDim.x - offset - BRid * 12 + 12;
    }
    else {
        Num = 12;
    }

    int iter = Num;
    for (int i = 1;i < 12;i = (i << 1)) {
        if (i < Num) {
            int tempNum = iter;
            iter = ((iter + 1) >> 1);
            if (landidx < iter) {
                if (threadIdx.x + iter < blockDim.x && threadIdx.x + iter < startId + tempNum)
                    sData[threadIdx.x] += sData[threadIdx.x + iter];
            }
        }
        __syncthreads();
        //__threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0 || GRtid == 0)
        atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), sData[threadIdx.x]);
}

__global__ 
void __PCG_Solve_AXALL_b2(const MATHUTILS::Matrix12x12d* Hessians12, const MATHUTILS::Matrix9x9d* Hessians9,
    const MATHUTILS::Matrix6x6d* Hessians6, const MATHUTILS::Matrix3x3d* Hessians3, const uint4* D4Index, const uint3* D3Index,
    const uint2* D2Index, const uint32_t* D1Index, const double3* c, double3* q, int numbers4, int numbers3, int numbers2, int numbers1,
    int offset4, int offset3, int offset2) {

    if (blockIdx.x < offset4) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numbers4) return;
        __shared__ int offset;
        int Hid = idx / 144;
        int MRid = (idx % 144) / 12;
        int MCid = (idx % 144) % 12;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 12;

        double rdata = Hessians12[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (12 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 12) / 12;
        int landidx = (threadIdx.x - offset) % 12;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
        mark = __brev(mark);
        unsigned int interval = MATHUTILS::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 12; iter <<= 1) {
            double tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 && blockIdx.x < offset4 + offset3) {
        int idx = (blockIdx.x - offset4) * blockDim.x + threadIdx.x;
        if (idx >= numbers3) return;
        __shared__ int offset;
        int Hid = idx / 81;
        int MRid = (idx % 81) / 9;
        int MCid = (idx % 81) % 9;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 9;

        double rdata = Hessians9[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (9 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 9) / 9;
        int landidx = (threadIdx.x - offset) % 9;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary); // a bit-mask 
        mark = __brev(mark);
        unsigned int interval = MATHUTILS::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 9; iter <<= 1) {
            double tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            atomicAdd((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 + offset3 && blockIdx.x < offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3) * blockDim.x + threadIdx.x;
        if (idx >= numbers2) return;
        __shared__ int offset;
        int Hid = idx / 36;
        int MRid = (idx % 36) / 6;
        int MCid = (idx % 36) % 6;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 6;

        double rdata = Hessians6[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (6 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 6) / 6;
        int landidx = (threadIdx.x - offset) % 6;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
        mark = __brev(mark);
        unsigned int interval = MATHUTILS::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 6; iter <<= 1) {
            double tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary)
            atomicAdd((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
    }
    else if (blockIdx.x >= offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3 - offset2) * blockDim.x + threadIdx.x;
        if (idx >= numbers1) return;
        MATHUTILS::Matrix3x3d H = Hessians3[idx];
        double3 tempC, tempQ;

        tempC.x = c[D1Index[idx]].x;
        tempC.y = c[D1Index[idx]].y;
        tempC.z = c[D1Index[idx]].z;


        tempQ = MATHUTILS::__M_v_multiply(H, tempC);

        atomicAdd(&(q[D1Index[idx]].x), tempQ.x);
        atomicAdd(&(q[D1Index[idx]].y), tempQ.y);
        atomicAdd(&(q[D1Index[idx]].z), tempQ.z);
    }
}


__global__ 
void __PCG_Solve_AX12_b2(const MATHUTILS::Matrix12x12d* Hessians, const uint4* D4Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __shared__ int offset;
    int Hid = idx / 144;
    int MRid = (idx % 144) / 12;
    int MCid = (idx % 144) % 12;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 12;

    double rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (12 - GRtid);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 12) / 12;
    int landidx = (threadIdx.x - offset) % 12;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
    mark = __brev(mark);
    unsigned int interval = MATHUTILS::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);
    //mark = interval;
    //for (int iter = 1; iter & 0x1f; iter <<= 1) {
    //    int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
    //    mark = tmp > mark ? tmp : mark; 
    //}
    //mark = __shfl_sync(0xFFFFFFFF, mark, 0);
    //__syncthreads();

    for (int iter = 1; iter < 12; iter <<= 1) {
        double tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

}

__global__ 
void __PCG_Solve_AX12_b3(const MATHUTILS::Matrix12x12d* Hessians, const uint4* D4Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __shared__ int offset0, offset1;
    __shared__ double tempB[36];

    int Hid = idx / 144;

    int HRtid = idx % 144;

    int MRid = (HRtid) / 12;
    int MCid = (HRtid) % 12;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 12;

    if (threadIdx.x == 0) {
        offset0 = (144 - HRtid);
        offset1 = (12 - GRtid);
    }
    __syncthreads();

    int HRid = (threadIdx.x - offset0 + 144) / 144;
    int Hlandidx = (threadIdx.x - offset0) % 144;
    if (HRid == 0) {
        Hlandidx = threadIdx.x;
    }

    int BRid = (threadIdx.x - offset1 + 12) / 12;
    int landidx = (threadIdx.x - offset1) % 12;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    if (HRid > 0 && Hlandidx < 12) {
        tempB[HRid * 12 + Hlandidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
    }
    else if (HRid == 0) {
        if (offset0 <= 12) {
            tempB[HRid * 12 + Hlandidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
        }
        else if (BRid == 1) {
            tempB[HRid * 12 + landidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
        }
    }

    __syncthreads();

    int readBid = landidx;
    if (offset0 > 12 && threadIdx.x < offset1)
        readBid = landidx + (12 - offset1);
    double rdata = Hessians[Hid].m[MRid][MCid] * tempB[HRid * 12 + readBid];//(*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
    mark = __brev(mark);
    unsigned int interval = MATHUTILS::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

    for (int iter = 1; iter < 12; iter <<= 1) {
        double tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

}

__global__ 
void __PCG_AXALL_P(const MATHUTILS::Matrix12x12d* Hessians12, const MATHUTILS::Matrix9x9d* Hessians9,
    const MATHUTILS::Matrix6x6d* Hessians6, const MATHUTILS::Matrix3x3d* Hessians3,
    const uint4* D4Index, const uint3* D3Index, const uint2* D2Index, const uint32_t* D1Index,
    MATHUTILS::Matrix3x3d* P, int numbers4, int numbers3, int numbers2, int numbers1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers4 + numbers3 + numbers2 + numbers1) return;

    if (idx < numbers4) {
        int Hid = idx / 12;
        int qid = idx % 12;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians12[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else if (numbers4 <= idx && idx < numbers3 + numbers4) {
        idx -= numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians9[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else if (numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2) {
        idx -= numbers3 + numbers4;
        int Hid = idx / 6;
        int qid = idx % 6;

        int mid = (qid / 3) * 3;
        int tid = qid % 3;

        double Hval = Hessians6[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
    }
    else {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 3;
        int qid = idx % 3;
        atomicAdd(&(P[D1Index[Hid]].m[0][qid]), Hessians3[Hid].m[0][qid]);
        atomicAdd(&(P[D1Index[Hid]].m[1][qid]), Hessians3[Hid].m[1][qid]);
        atomicAdd(&(P[D1Index[Hid]].m[2][qid]), Hessians3[Hid].m[2][qid]);
    }
}

__global__ 
void __PCG_AX12_P(const MATHUTILS::Matrix12x12d* Hessians, const uint4* D4Index, MATHUTILS::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int Hid = idx / 12;
    int qid = idx % 12;

    //double Hval = Hessians[Hid].m[qid][qid];
    //atomicAdd((&(P[*(&(D4Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
    int mid = (qid / 3) * 3;
    int tid = qid % 3;

    double Hval = Hessians[Hid].m[mid][mid + tid];
    atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 1][mid + tid];
    atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 2][mid + tid];
    atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
}


__global__ 
void __PCG_Solve_AX9_b(const MATHUTILS::Matrix9x9d* Hessians, const uint3* D3Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    MATHUTILS::Matrix9x9d H = Hessians[idx];
    MATHUTILS::Vector9 tempC, tempQ;

    tempC.v[0] = c[D3Index[idx].x].x;
    tempC.v[1] = c[D3Index[idx].x].y;
    tempC.v[2] = c[D3Index[idx].x].z;

    tempC.v[3] = c[D3Index[idx].y].x;
    tempC.v[4] = c[D3Index[idx].y].y;
    tempC.v[5] = c[D3Index[idx].y].z;

    tempC.v[6] = c[D3Index[idx].z].x;
    tempC.v[7] = c[D3Index[idx].z].y;
    tempC.v[8] = c[D3Index[idx].z].z;



    tempQ = MATHUTILS::__M9x9_v9_multiply(H, tempC);

    atomicAdd(&(q[D3Index[idx].x].x), tempQ.v[0]);
    atomicAdd(&(q[D3Index[idx].x].y), tempQ.v[1]);
    atomicAdd(&(q[D3Index[idx].x].z), tempQ.v[2]);

    atomicAdd(&(q[D3Index[idx].y].x), tempQ.v[3]);
    atomicAdd(&(q[D3Index[idx].y].y), tempQ.v[4]);
    atomicAdd(&(q[D3Index[idx].y].z), tempQ.v[5]);

    atomicAdd(&(q[D3Index[idx].z].x), tempQ.v[6]);
    atomicAdd(&(q[D3Index[idx].z].y), tempQ.v[7]);
    atomicAdd(&(q[D3Index[idx].z].z), tempQ.v[8]);
}

__global__ 
void __PCG_AX9_P(const MATHUTILS::Matrix9x9d* Hessians, const uint3* D3Index, MATHUTILS::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int Hid = idx / 9;
    int qid = idx % 9;

    //double Hval = Hessians[Hid].m[qid][qid];
    //atomicAdd((&(P[*(&(D3Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
    int mid = (qid / 3) * 3;
    int tid = qid % 3;

    double Hval = Hessians[Hid].m[mid][mid + tid];
    atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 1][mid + tid];
    atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 2][mid + tid];
    atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
}


__global__ 
void __PCG_Solve_AX9_b2(const MATHUTILS::Matrix9x9d* Hessians, const uint3* D3Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    //extern __shared__ double sData[];
    __shared__ int offset;
    int Hid = idx / 81;
    int MRid = (idx % 81) / 9;
    int MCid = (idx % 81) % 9;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 9;
    //sData[threadIdx.x] = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
    //printf("landidx  %f  %d   %d   %d\n", sData[threadIdx.x], offset, 1, 1);
    double rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (9 - GRtid);// < 12 ? (12 - GRtid) : 0;
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 9) / 9;
    int landidx = (threadIdx.x - offset) % 9;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary); // a bit-mask 
    mark = __brev(mark);
    unsigned int interval = MATHUTILS::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);
    //mark = interval;
    //for (int iter = 1; iter & 0x1f; iter <<= 1) {
    //    int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
    //    mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
    //}
    //mark = __shfl_sync(0xFFFFFFFF, mark, 0);
    //__syncthreads();

    for (int iter = 1; iter < 9; iter <<= 1) {
        double tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
}

__global__ 
void __PCG_Solve_AX6_b(const MATHUTILS::Matrix6x6d* Hessians, const uint2* D2Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    MATHUTILS::Matrix6x6d H = Hessians[idx];
    MATHUTILS::Vector6 tempC, tempQ;

    tempC.v[0] = c[D2Index[idx].x].x;
    tempC.v[1] = c[D2Index[idx].x].y;
    tempC.v[2] = c[D2Index[idx].x].z;

    tempC.v[3] = c[D2Index[idx].y].x;
    tempC.v[4] = c[D2Index[idx].y].y;
    tempC.v[5] = c[D2Index[idx].y].z;



    tempQ = MATHUTILS::__M6x6_v6_multiply(H, tempC);

    atomicAdd(&(q[D2Index[idx].x].x), tempQ.v[0]);
    atomicAdd(&(q[D2Index[idx].x].y), tempQ.v[1]);
    atomicAdd(&(q[D2Index[idx].x].z), tempQ.v[2]);

    atomicAdd(&(q[D2Index[idx].y].x), tempQ.v[3]);
    atomicAdd(&(q[D2Index[idx].y].y), tempQ.v[4]);
    atomicAdd(&(q[D2Index[idx].y].z), tempQ.v[5]);
}

__global__ 
void __PCG_AX6_P(const MATHUTILS::Matrix6x6d* Hessians, const uint2* D2Index, MATHUTILS::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int Hid = idx / 6;
    int qid = idx % 6;

    //double Hval = Hessians[Hid].m[qid][qid];
    //atomicAdd((&(P[*(&(D2Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
    int mid = (qid / 3) * 3;
    int tid = qid % 3;

    double Hval = Hessians[Hid].m[mid][mid + tid];
    atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 1][mid + tid];
    atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
    Hval = Hessians[Hid].m[mid + 2][mid + tid];
    atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
}


__global__ 
void __PCG_Solve_AX6_b2(const MATHUTILS::Matrix6x6d* Hessians, const uint2* D2Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __shared__ int offset;
    int Hid = idx / 36;
    int MRid = (idx % 36) / 6;
    int MCid = (idx % 36) % 6;

    int vId = MCid / 3;
    int axisId = MCid % 3;
    int GRtid = idx % 6;

    double rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (6 - GRtid);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 6) / 6;
    int landidx = (threadIdx.x - offset) % 6;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
    mark = __brev(mark);
    unsigned int interval = MATHUTILS::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);
    //mark = interval;
    //for (int iter = 1; iter & 0x1f; iter <<= 1) {
    //    int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
    //    mark = tmp > mark ? tmp : mark; 
    //}
    //mark = __shfl_sync(0xFFFFFFFF, mark, 0);
    //__syncthreads();

    for (int iter = 1; iter < 6; iter <<= 1) {
        double tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
}

__global__ 
void __PCG_Update_Dx_R(const double3* c, double3* dx, const double3* q, double3* r, double rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    dx[idx] = MATHUTILS::__add(dx[idx], MATHUTILS::__s_vec_multiply(c[idx], rate));
    r[idx] = MATHUTILS::__add(r[idx], MATHUTILS::__s_vec_multiply(q[idx], -rate));
}


__global__ 
void __PCG_Solve_AX3_b(const MATHUTILS::Matrix3x3d* Hessians, const uint32_t* D1Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    MATHUTILS::Matrix3x3d H = Hessians[idx];
    double3 tempC, tempQ;

    tempC.x = c[D1Index[idx]].x;
    tempC.y = c[D1Index[idx]].y;
    tempC.z = c[D1Index[idx]].z;


    tempQ = MATHUTILS::__M_v_multiply(H, tempC);

    atomicAdd(&(q[D1Index[idx]].x), tempQ.x);
    atomicAdd(&(q[D1Index[idx]].y), tempQ.y);
    atomicAdd(&(q[D1Index[idx]].z), tempQ.z);
}

__global__ 
void __PCG_AX3_P(const MATHUTILS::Matrix3x3d* Hessians, const uint32_t* D1Index, MATHUTILS::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int Hid = idx / 3;
    int qid = idx % 3;

    //double Hval = Hessians[Hid].m[qid][qid];
    //*(&(P[(D1Index[Hid])].x) + qid) += Hval;
    //P[D1Index[Hid]].m[0][qid] += Hessians[Hid].m[0][qid];
    //P[D1Index[Hid]].m[1][qid] += Hessians[Hid].m[1][qid];
    //P[D1Index[Hid]].m[2][qid] += Hessians[Hid].m[2][qid];
    atomicAdd(&(P[D1Index[Hid]].m[0][qid]), Hessians[Hid].m[0][qid]);
    atomicAdd(&(P[D1Index[Hid]].m[1][qid]), Hessians[Hid].m[1][qid]);
    atomicAdd(&(P[D1Index[Hid]].m[2][qid]), Hessians[Hid].m[2][qid]);
}


__global__ 
void __PCG_Solve_AX3_b2(const MATHUTILS::Matrix3x3d* Hessians, const uint32_t* D1Index, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    __shared__ int offset;
    int Hid = idx / 9;
    int MRid = (idx % 9) / 3;
    int MCid = (idx % 9) % 3;


    int axisId = MCid % 3;
    int GRtid = idx % 3;

    double rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[(D1Index[Hid])].x) + axisId));

    if (threadIdx.x == 0) {
        offset = (3 - GRtid);
    }
    __syncthreads();

    int BRid = (threadIdx.x - offset + 3) / 3;
    int landidx = (threadIdx.x - offset) % 3;
    if (BRid == 0) {
        landidx = threadIdx.x;
    }

    int warpId = threadIdx.x & 0x1f;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
    mark = __brev(mark);
    unsigned int interval = MATHUTILS::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);
    mark = interval;
    for (int iter = 1; iter & 0x1f; iter <<= 1) {
        int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
        mark = tmp > mark ? tmp : mark;
    }
    mark = __shfl_sync(0xFFFFFFFF, mark, 0);
    __syncthreads();

    for (int iter = 1; iter <= mark; iter <<= 1) {
        double tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
        if (interval >= iter) rdata += tmp;
    }

    if (bBoundary)
        atomicAdd((&(q[(D1Index[Hid])].x) + MRid % 3), rdata);
}


__global__ 
void __PCG_Solve_AX_mass_b(const double* _masses, const double3* c, double3* q, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;


    double3 tempQ = MATHUTILS::__s_vec_multiply(c[idx], _masses[idx]);

    q[idx] = tempQ;

    //atomicAdd(&(q[idx].x), tempQ.x);
    //atomicAdd(&(q[idx].y), tempQ.y);
    //atomicAdd(&(q[idx].z), tempQ.z);
}



__global__ 
void __PCG_mass_P(const double* _masses, MATHUTILS::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    double mass = _masses[idx];
    MATHUTILS::__init_Mat3x3(P[idx], 0);
    P[idx].m[0][0] = mass;
    P[idx].m[1][1] = mass;
    P[idx].m[2][2] = mass;
}

__global__ 
void __PCG_init_P(const double* _masses, MATHUTILS::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    double s1 = P[idx].m[0][0];
    double s2 = P[idx].m[1][1];
    double s3 = P[idx].m[2][2];
    MATHUTILS::__init_Mat3x3(P[idx], 0);
    /*P[idx].m[0][0] = 1;
    P[idx].m[1][1] = 1;
    P[idx].m[2][2] = 1;*/
    P[idx].m[0][0] = 1 / s1;
    P[idx].m[1][1] = 1 / s2;
    P[idx].m[2][2] = 1 / s3;
}

__global__ 
void __PCG_inverse_P(MATHUTILS::Matrix3x3d* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    MATHUTILS::Matrix3x3d PInverse;
    MATHUTILS::__Inverse(P[idx], PInverse);

    P[idx] = PInverse;

}


double My_PCG_add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance, 
std::unique_ptr<PCGData>& pcg_data,
// PCGData* pcg_data,
int vertexNum, double alpha = 1) {

    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    switch (type) {
    case 0:
        PCG_add_Reduction_force << <blockNum, threadNum, sharedMsize >> > (pcg_data->mc_squeue, pcg_data->mc_b, numbers);
        break;
    case 1:
        PCG_add_Reduction_delta0 << <blockNum, threadNum, sharedMsize >> > (pcg_data->mc_squeue, pcg_data->mc_P, pcg_data->mc_b, instance->cudaConstraintsMat, numbers);
        break;
    case 2:
        PCG_add_Reduction_deltaN0 << <blockNum, threadNum, sharedMsize >> > (pcg_data->mc_squeue, pcg_data->mc_P, pcg_data->mc_b, pcg_data->mc_r, pcg_data->mc_c, instance->cudaConstraintsMat, numbers);
        break;
    case 3:
        PCG_add_Reduction_tempSum << <blockNum, threadNum, sharedMsize >> > (pcg_data->mc_squeue, pcg_data->mc_c, pcg_data->mc_q, instance->cudaConstraintsMat, numbers);
        break;
    case 4:
        PCG_add_Reduction_deltaN << <blockNum, threadNum, sharedMsize >> > (pcg_data->mc_squeue, pcg_data->mc_dx, pcg_data->mc_c, pcg_data->mc_r, pcg_data->mc_q, pcg_data->mc_P, pcg_data->mc_s, alpha, numbers);
        break;
    }

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        add_reduction << <blockNum, threadNum, sharedMsize >> > (pcg_data->mc_squeue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    double result;
    cudaMemcpy(&result, pcg_data->mc_squeue, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

void Solve_PCG_AX_B(const std::unique_ptr<GeometryManager>& instance, const double3* c, double3* q, const std::unique_ptr<BHessian>& BH, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Solve_AX_mass_b << <blockNum, threadNum >> > (instance->cudaVertMass, c, q, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = BH->m_DNum[3];
    if (numbers > 0) {
        //unsigned int sharedMsize = sizeof(double) * threadNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_Solve_AX12_b << <blockNum, threadNum >> > (BH->mc_H12x12, BH->mc_D4Index, c, q, numbers);
    }
    numbers = BH->m_DNum[2];
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_Solve_AX9_b << <blockNum, threadNum >> > (BH->mc_H9x9, BH->mc_D3Index, c, q, numbers);
    }
    numbers = BH->m_DNum[1];
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_Solve_AX6_b << <blockNum, threadNum >> > (BH->mc_H6x6, BH->mc_D2Index, c, q, numbers);
    }
    numbers = BH->m_DNum[0];
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_Solve_AX3_b << <blockNum, threadNum >> > (BH->mc_H3x3, BH->mc_D1Index, c, q, numbers);
    }

}

void PCG_Update_Dx_R(const double3* c, double3* dx, const double3* q, double3* r, const double& rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Update_Dx_R << <blockNum, threadNum >> > (c, dx, q, r, rate, numbers);
}


double My_PCG_General_v_v_Reduction_Algorithm(
    std::unique_ptr<GeometryManager>& instance,     
    std::unique_ptr<PCGData>& pcg_data, 
    double3* A, double3* B, int vertexNum) {

    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction << <blockNum, threadNum >> > (pcg_data->mc_squeue, A, B, numbers);


    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        add_reduction << <blockNum, threadNum, sharedMsize >> > (pcg_data->mc_squeue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    double result;
    cudaMemcpy(&result, pcg_data->mc_squeue, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

void Solve_PCG_AX_B2(const std::unique_ptr<GeometryManager>& instance, const double3* c, double3* q, const std::unique_ptr<BHessian>& BH, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Solve_AX_mass_b << <blockNum, threadNum >> > (instance->cudaVertMass, c, q, numbers);

    int offset4 = (BH->m_DNum[3] * 144 + threadNum - 1) / threadNum;
    int offset3 = (BH->m_DNum[2] * 81 + threadNum - 1) / threadNum;
    int offset2 = (BH->m_DNum[1] * 36 + threadNum - 1) / threadNum;
    int offset1 = (BH->m_DNum[0] + threadNum - 1) / threadNum;
    blockNum = offset1 + offset2 + offset3 + offset4;
    __PCG_Solve_AXALL_b2 << <blockNum, threadNum >> > (BH->mc_H12x12, BH->mc_H9x9, BH->mc_H6x6, BH->mc_H3x3, BH->mc_D4Index, BH->mc_D3Index, BH->mc_D2Index, BH->mc_D1Index, c, q, BH->m_DNum[3] * 144, BH->m_DNum[2] * 81, BH->m_DNum[1] * 36, BH->m_DNum[0], offset4, offset3, offset2);

}

void construct_P(const std::unique_ptr<GeometryManager>& instance, MATHUTILS::Matrix3x3d* P, const std::unique_ptr<BHessian>& BH, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_mass_P << <blockNum, threadNum >> > (instance->cudaVertMass, P, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = BH->m_DNum[3] * 12;
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_AX12_P << <blockNum, threadNum >> > (BH->mc_H12x12, BH->mc_D4Index, P, numbers);
    }
    numbers = BH->m_DNum[2] * 9;
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_AX9_P << <blockNum, threadNum >> > (BH->mc_H9x9, BH->mc_D3Index, P, numbers);
    }
    numbers = BH->m_DNum[1] * 6;
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_AX6_P << <blockNum, threadNum >> > (BH->mc_H6x6, BH->mc_D2Index, P, numbers);
    }
    numbers = BH->m_DNum[0] * 3;
    if (numbers > 0) {
        blockNum = (numbers + threadNum - 1) / threadNum;
        __PCG_AX3_P << <blockNum, threadNum >> > (BH->mc_H3x3, BH->mc_D1Index, P, numbers);
    }
    blockNum = (vertNum + threadNum - 1) / threadNum;
    //__PCG_inverse_P << <blockNum, threadNum >> > (P, vertNum);
    __PCG_init_P << <blockNum, threadNum >> > (instance->cudaVertMass, P, vertNum);
}

void construct_P2(
    const std::unique_ptr<GeometryManager>& instance,
    std::unique_ptr<PCGData>& pcg_data,
    const std::unique_ptr<BHessian>& BH,
    int vertNum) {

    int numbers = vertNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_mass_P <<<blockNum, threadNum >>> (instance->cudaVertMass, pcg_data->mc_P, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers = BH->m_DNum[3] * 12 + BH->m_DNum[2] * 9 + BH->m_DNum[1] * 6 + BH->m_DNum[0] * 3;
    blockNum = (numbers + threadNum - 1) / threadNum;

    __PCG_AXALL_P <<<blockNum, threadNum>>> (
        BH->mc_H12x12, 
        BH->mc_H9x9, 
        BH->mc_H6x6, 
        BH->mc_H3x3, 
        BH->mc_D4Index, 
        BH->mc_D3Index, 
        BH->mc_D2Index, 
        BH->mc_D1Index, 
        pcg_data->mc_P, 
        BH->m_DNum[3] * 12, 
        BH->m_DNum[2] * 9, 
        BH->m_DNum[1] * 6, 
        BH->m_DNum[0] * 3);

    blockNum = (vertNum + threadNum - 1) / threadNum;
    __PCG_inverse_P << <blockNum, threadNum >> > (pcg_data->mc_P, vertNum);
    //__PCG_init_P << <blockNum, threadNum >> > (instance->masses, P, vertNum);

}

void PCG_FinalStep_UpdateC(const std::unique_ptr<GeometryManager>& instance, double3* c, const double3* s, const double& rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_FinalStep_UpdateC << <blockNum, threadNum >> > (instance->cudaConstraintsMat, s, c, rate, numbers);
}

void PCG_initDX(double3* dx, const double3* z, double rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_initDX << <blockNum, threadNum >> > (dx, z, rate, numbers);
}

void PCG_constraintFilter(const std::unique_ptr<GeometryManager>& instance, const double3* input, double3* output, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_constraintFilter << <blockNum, threadNum >> > (instance->cudaConstraintsMat, input, output, numbers);
}


int PCG_Process(
    std::unique_ptr<GeometryManager>& instance, 
    std::unique_ptr<PCGData>& pcg_data, 
    const std::unique_ptr<BHessian>& BH,
    double3* _mvDir, 
    int vertexNum, 
    int tetrahedraNum, 
    double IPC_dt, 
    double meanVolumn, 
    double threshold) {

    construct_P2(instance, pcg_data, BH, vertexNum);

    double deltaN = 0;
    double delta0 = 0;
    double deltaO = 0;
    //PCG_initDX(pcg_data->dx, pcg_data->z, 0.5, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(pcg_data->mc_dx, 0x0, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(pcg_data->mc_r, 0x0, vertexNum * sizeof(double3)));
    delta0 = My_PCG_add_Reduction_Algorithm(1, instance, pcg_data, vertexNum);
    //Solve_PCG_AX_B2(instance, pcg_data->z, pcg_data->r, BH, vertexNum);
    deltaN = My_PCG_add_Reduction_Algorithm(2, instance, pcg_data, vertexNum);
    //std::cout << "gpu  delta0:   " << delta0 << "      deltaN:   " << deltaN << std::endl;
    //double errorRate = std::min(1e-8 * 0.5 * IPC_dt / std::pow(meanVolumn, 1), 1e-4);
    double errorRate = threshold/* * IPC_dt * IPC_dt*/;
    //printf("cg error Rate:   %f        meanVolumn: %f\n", errorRate, meanVolumn);
    int cgCounts = 0;
    while (cgCounts<30000 && deltaN > errorRate * delta0) {
        cgCounts++;
        //std::cout << "delta0:   " << delta0 << "      deltaN:   " << deltaN << "      iteration_counts:      " << cgCounts << std::endl;
        //CUDA_SAFE_CALL(cudaMemset(pcg_data->q, 0, vertexNum * sizeof(double3)));
        Solve_PCG_AX_B2(instance, pcg_data->mc_c, pcg_data->mc_q, BH, vertexNum);
        double tempSum = My_PCG_add_Reduction_Algorithm(3, instance, pcg_data, vertexNum);
        double alpha = deltaN / tempSum;
        deltaO = deltaN;
        //deltaN = 0;
        //CUDA_SAFE_CALL(cudaMemset(pcg_data->s, 0, vertexNum * sizeof(double3)));
        deltaN = My_PCG_add_Reduction_Algorithm(4, instance, pcg_data, vertexNum, alpha);
        double rate = deltaN / deltaO;
        PCG_FinalStep_UpdateC(instance, pcg_data->mc_c, pcg_data->mc_s, rate, vertexNum);
        //cudaDeviceSynchronize();
    }
    _mvDir = pcg_data->mc_dx;
    //CUDA_SAFE_CALL(cudaMemcpy(pcg_data->z, _mvDir, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));
    if (cgCounts == 0) {
        printf("indefinite exit\n");
        // exit(0);
    }
    return cgCounts;
}



int MASPCG_Process(
    std::unique_ptr<GeometryManager>& instance, 
    std::unique_ptr<PCGData>& pcg_data, 
    const std::unique_ptr<BHessian>& BH, 
    double3* _mvDir, 
    int vertexNum, 
    int tetrahedraNum, 
    double IPC_dt, 
    double meanVolumn, 
    int cpNum, 
    double threshold) {

    instance->MAS_ptr->setPreconditioner(BH, instance->cudaVertMass, cpNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    double deltaN = 0;
    double delta0 = 0;
    double deltaO = 0;
    //PCG_initDX(pcg_data->dx, pcg_data->z, 0.5, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(pcg_data->mc_dx, 0x0, vertexNum * sizeof(double3)));
    CUDA_SAFE_CALL(cudaMemset(pcg_data->mc_r, 0x0, vertexNum * sizeof(double3)));

    PCG_constraintFilter(instance, pcg_data->mc_b, pcg_data->mc_filterTempVec3, vertexNum);

    instance->MAS_ptr->preconditioning(pcg_data->mc_filterTempVec3, pcg_data->mc_preconditionTempVec3);
    //Solve_PCG_Preconditioning24(mesh, pcg_data->P24, pcg_data->P, pcg_data->restP, pcg_data->filterTempVec3, pcg_data->preconditionTempVec3, vertexNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    delta0 = My_PCG_General_v_v_Reduction_Algorithm(instance, pcg_data, pcg_data->mc_filterTempVec3, pcg_data->mc_preconditionTempVec3, vertexNum);

    CUDA_SAFE_CALL(cudaMemcpy(pcg_data->mc_r, pcg_data->mc_filterTempVec3, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

    PCG_constraintFilter(instance, pcg_data->mc_preconditionTempVec3, pcg_data->mc_filterTempVec3, vertexNum);

    CUDA_SAFE_CALL(cudaMemcpy(pcg_data->mc_c, pcg_data->mc_filterTempVec3, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

    deltaN = My_PCG_General_v_v_Reduction_Algorithm(instance, pcg_data, pcg_data->mc_r, pcg_data->mc_c, vertexNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //delta0 = My_PCG_add_Reduction_Algorithm(1, mesh, pcg_data, vertexNum);
    //Solve_PCG_AX_B2(mesh, pcg_data->z, pcg_data->r, BH, vertexNum);
    //deltaN = My_PCG_add_Reduction_Algorithm(2, mesh, pcg_data, vertexNum);
    //std::cout << "gpu  delta0:   " << delta0 << "      deltaN:   " << deltaN << std::endl;
    //double errorRate = std::min(1e-8 * 0.5 * IPC_dt / std::pow(meanVolumn, 1), 1e-4);
    double errorRate = threshold/* * IPC_dt * IPC_dt*/;
    //printf("cg error Rate:   %f        meanVolumn: %f\n", errorRate, meanVolumn);
    int cgCounts = 0;
    while (cgCounts<3000 && deltaN > errorRate * delta0) {

        cgCounts++;
        //std::cout << "delta0:   " << delta0 << "      deltaN:   " << deltaN << "      iteration_counts:      " << cgCounts << std::endl;
        //CUDA_SAFE_CALL(cudaMemset(pcg_data->q, 0, vertexNum * sizeof(double3)));
        Solve_PCG_AX_B2(instance, pcg_data->mc_c, pcg_data->mc_q, BH, vertexNum);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        double tempSum = My_PCG_add_Reduction_Algorithm(3, instance, pcg_data, vertexNum);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        double alpha = deltaN / tempSum;
        deltaO = deltaN;
        //deltaN = 0;
        //CUDA_SAFE_CALL(cudaMemset(pcg_data->s, 0, vertexNum * sizeof(double3)));
        //deltaN = My_PCG_add_Reduction_Algorithm(4, mesh, pcg_data, vertexNum, alpha);
        PCG_Update_Dx_R(pcg_data->mc_c, pcg_data->mc_dx, pcg_data->mc_q, pcg_data->mc_r, alpha, vertexNum);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        instance->MAS_ptr->preconditioning(pcg_data->mc_r, pcg_data->mc_s);
        //Solve_PCG_Preconditioning24(mesh, pcg_data->P24, pcg_data->P, pcg_data->restP, pcg_data->r, pcg_data->s, vertexNum);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        deltaN = My_PCG_General_v_v_Reduction_Algorithm(instance, pcg_data, pcg_data->mc_r, pcg_data->mc_s, vertexNum);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        double rate = deltaN / deltaO;
        PCG_FinalStep_UpdateC(instance, pcg_data->mc_c, pcg_data->mc_s, rate, vertexNum);
        //cudaDeviceSynchronize();
        //std::cout << "gpu  delta0:   " << delta0 << "      deltaN:   " << deltaN << std::endl;
    }
    
    _mvDir = pcg_data->mc_dx;
    //CUDA_SAFE_CALL(cudaMemcpy(pcg_data->z, _mvDir, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));
    //printf("cg counts = %d\n", cgCounts);
    if (cgCounts == 0) {
        printf("indefinite exit\n");
        //exit(0);
    }
    return cgCounts;
}




}; // namespace PCGSOLVER


