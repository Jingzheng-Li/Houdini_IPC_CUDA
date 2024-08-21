
#include "BHessian.cuh"


/////////////////////////////
// update BHessian 
/////////////////////////////

BHessian::BHessian() {};

BHessian::~BHessian() {};

void BHessian::CUDA_MALLOC_BHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number, const int& triangle_num, const int& tri_Edge_number) {
    
    CUDAMallocSafe(mc_H12x12, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number));
    CUDAMallocSafe(mc_H9x9, (2 * (surfEdge_number + surfvert_number) + triangle_num));
    CUDAMallocSafe(mc_H6x6, 2 * (surfvert_number + surfEdge_number));
    CUDAMallocSafe(mc_H3x3, 2 * surfvert_number);
    CUDAMallocSafe(mc_D4Index, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number));
    CUDAMallocSafe(mc_D3Index, (2 * (surfEdge_number + surfvert_number)+ triangle_num));
    CUDAMallocSafe(mc_D2Index, 2 * (surfvert_number + surfEdge_number));
    CUDAMallocSafe(mc_D1Index, 2 * surfvert_number);

}

void BHessian::CUDA_FREE_BHESSIAN() {
    CUDAFreeSafe(mc_H12x12);
    CUDAFreeSafe(mc_H9x9);
    CUDAFreeSafe(mc_H6x6);
    CUDAFreeSafe(mc_H3x3);
    CUDAFreeSafe(mc_D1Index);
    CUDAFreeSafe(mc_D2Index);
    CUDAFreeSafe(mc_D3Index);
    CUDAFreeSafe(mc_D4Index);    

}


void BHessian::updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number) {
    m_DNum[1] = cpNums[1];
    m_DNum[2] = cpNums[2] + tri_Num;
    m_DNum[3] = tet_number + cpNums[3] + tri_edge_number;

#ifdef USE_FRICTION
    m_DNum[1] += last_cpNums[1];
    m_DNum[2] += last_cpNums[2];
    m_DNum[3] += last_cpNums[3];
#endif

}



