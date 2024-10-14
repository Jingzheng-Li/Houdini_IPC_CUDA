
#include "BlockHessian.cuh"




void BlockHessian::updateDNum(
    const int& tri_Num,
    const int& tri_edge_number,
    const int& tet_number,
    const uint32_t& cpNum2,
    const uint32_t& cpNum3,
    const uint32_t& cpNum4,
    const uint32_t& last_cpNum2,
    const uint32_t& last_cpNum3,
    const uint32_t& last_cpNum4
) {
    hostBHDNum[1] = cpNum2;
    hostBHDNum[2] = cpNum3 + tri_Num;
    hostBHDNum[3]= cpNum4 + tet_number + tri_edge_number;

#ifdef USE_FRICTION
    hostBHDNum[1] += last_cpNum2;
    hostBHDNum[2] += last_cpNum3;
    hostBHDNum[3] += last_cpNum4;
#endif
}



void BlockHessian::CUDA_MALLOC_BLOCKHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number, const int& triangle_num, const int& tri_Edge_number) {

    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaH12x12, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number) * sizeof(__MATHUTILS__::Matrix12x12S)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaH9x9, (2 * (surfEdge_number + surfvert_number) + triangle_num) * sizeof(__MATHUTILS__::Matrix9x9S)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaH6x6, 2 * (surfvert_number + surfEdge_number) * sizeof(__MATHUTILS__::Matrix6x6S)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaH3x3, 2 * surfvert_number * sizeof(__MATHUTILS__::Matrix3x3S)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaD4Index, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number) * sizeof(uint4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaD3Index, (2 * (surfEdge_number + surfvert_number)+ triangle_num) * sizeof(uint3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaD2Index, 2 * (surfvert_number + surfEdge_number) * sizeof(uint2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&cudaD1Index, 2 * surfvert_number * sizeof(uint32_t)));

}

void BlockHessian::CUDA_FREE_BLOCKHESSIAN() {
    CUDA_SAFE_CALL(cudaFree(cudaH12x12));
    CUDA_SAFE_CALL(cudaFree(cudaH9x9));
    CUDA_SAFE_CALL(cudaFree(cudaH6x6));
    CUDA_SAFE_CALL(cudaFree(cudaH3x3));

    CUDA_SAFE_CALL(cudaFree(cudaD4Index));
    CUDA_SAFE_CALL(cudaFree(cudaD3Index));
    CUDA_SAFE_CALL(cudaFree(cudaD2Index));
    CUDA_SAFE_CALL(cudaFree(cudaD1Index));
}


