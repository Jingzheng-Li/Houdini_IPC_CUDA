
#include "PCGSolver.cuh"


__global__ void PCG_vdv_Reduction(double* squeue, const double3* a, const double3* b, int numbers) {
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


__global__ void PCG_add_Reduction_delta0(double* squeue, const MATHUTILS::Matrix3x3d* P, const double3* b, const MATHUTILS::Matrix3x3d* constraint, int numbers) {
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

__global__ void PCG_add_Reduction_deltaN0(double* squeue, const MATHUTILS::Matrix3x3d* P, const double3* b, double3* r, double3* c, const MATHUTILS::Matrix3x3d* constraint, int numbers) {
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







void PCGData::CUDA_MALLOC_PCGDATA(const int& vertexNum, const int& tetrahedraNum) {
    CUDAMallocSafe(squeue, MATHUTILS::__m_max(vertexNum, tetrahedraNum));
    CUDAMallocSafe(P, vertexNum);
    CUDAMallocSafe(r, vertexNum);
    CUDAMallocSafe(c, vertexNum);
    CUDAMallocSafe(z, vertexNum);
    CUDAMallocSafe(q, vertexNum);
    CUDAMallocSafe(s, vertexNum);
    CUDAMallocSafe(dx, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(z, 0, vertexNum * sizeof(double3)));

}


void PCGData::CUDA_FREE_PCGDATA() {
    CUDAFreeSafe(squeue);
    CUDAFreeSafe(r);
    CUDAFreeSafe(r);
    CUDAFreeSafe(c);
    CUDAFreeSafe(z);
    CUDAFreeSafe(q);
    CUDAFreeSafe(s);
    CUDAFreeSafe(dx);

}

void BHessian::CUDA_MALLOC_BHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number, const int& triangle_num, const int& tri_Edge_number) {
    CUDAMallocSafe(H12x12, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number));
    CUDAMallocSafe(H9x9, (2 * (surfEdge_number + surfvert_number) + triangle_num));
    CUDAMallocSafe(H6x6, 2 * (surfvert_number + surfEdge_number));
    CUDAMallocSafe(H3x3, 2 * surfvert_number);

    CUDAMallocSafe(D4Index, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number));
    CUDAMallocSafe(D3Index, (2 * (surfEdge_number + surfvert_number)+ triangle_num));
    CUDAMallocSafe(D2Index, 2 * (surfvert_number + surfEdge_number));
    CUDAMallocSafe(D1Index, 2 * surfvert_number);

}

void BHessian::CUDA_FREE_BHESSIAN() {
    CUDAFreeSafe(H12x12);
    CUDAFreeSafe(H9x9);
    CUDAFreeSafe(H6x6);
    CUDAFreeSafe(H3x3);
    CUDAFreeSafe(D1Index);
    CUDAFreeSafe(D2Index);
    CUDAFreeSafe(D3Index);
    CUDAFreeSafe(D4Index);    

}















// int PCGSOLVER::PCG_Process(std::unique_ptr<GeometryManager>& instance, PCGData* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn, double threshold) {
//     construct_P2(mesh, pcg_data->P, BH, vertexNum);
//     double deltaN = 0;
//     double delta0 = 0;
//     double deltaO = 0;
//     //PCG_initDX(pcg_data->dx, pcg_data->z, 0.5, vertexNum);
//     CUDA_SAFE_CALL(cudaMemset(pcg_data->dx, 0x0, vertexNum * sizeof(double3)));
//     CUDA_SAFE_CALL(cudaMemset(pcg_data->r, 0x0, vertexNum * sizeof(double3)));
//     delta0 = My_PCG_add_Reduction_Algorithm(1, mesh, pcg_data, vertexNum);
//     //Solve_PCG_AX_B2(mesh, pcg_data->z, pcg_data->r, BH, vertexNum);
//     deltaN = My_PCG_add_Reduction_Algorithm(2, mesh, pcg_data, vertexNum);
//     //std::cout << "gpu  delta0:   " << delta0 << "      deltaN:   " << deltaN << std::endl;
//     //double errorRate = std::min(1e-8 * 0.5 * IPC_dt / std::pow(meanVolumn, 1), 1e-4);
//     double errorRate = threshold/* * IPC_dt * IPC_dt*/;
//     //printf("cg error Rate:   %f        meanVolumn: %f\n", errorRate, meanVolumn);
//     int cgCounts = 0;
//     while (cgCounts<30000 && deltaN > errorRate * delta0) {
//         cgCounts++;
//         //std::cout << "delta0:   " << delta0 << "      deltaN:   " << deltaN << "      iteration_counts:      " << cgCounts << std::endl;
//         //CUDA_SAFE_CALL(cudaMemset(pcg_data->q, 0, vertexNum * sizeof(double3)));
//         Solve_PCG_AX_B2(mesh, pcg_data->c, pcg_data->q, BH, vertexNum);
//         double tempSum = My_PCG_add_Reduction_Algorithm(3, mesh, pcg_data, vertexNum);
//         double alpha = deltaN / tempSum;
//         deltaO = deltaN;
//         //deltaN = 0;
//         //CUDA_SAFE_CALL(cudaMemset(pcg_data->s, 0, vertexNum * sizeof(double3)));
//         deltaN = My_PCG_add_Reduction_Algorithm(4, mesh, pcg_data, vertexNum, alpha);
//         double rate = deltaN / deltaO;
//         PCG_FinalStep_UpdateC(mesh, pcg_data->c, pcg_data->s, rate, vertexNum);
//         //cudaDeviceSynchronize();
//     }
//     _mvDir = pcg_data->dx;
//     //CUDA_SAFE_CALL(cudaMemcpy(pcg_data->z, _mvDir, vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));
//     if (cgCounts == 0) {
//         printf("indefinite exit\n");
//         exit(0);
//     }
//     return cgCounts;
// }











