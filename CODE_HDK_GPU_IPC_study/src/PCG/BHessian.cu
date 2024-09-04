
#include "BHessian.cuh"


/////////////////////////////
// update BHessian 
/////////////////////////////

BHessian::BHessian(std::unique_ptr<GeometryManager>& instance)
: m_instance(instance)
{};

BHessian::~BHessian() {};

void BHessian::CUDA_MALLOC_BHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number, const int& triangle_num, const int& tri_Edge_number) {

    CUDAMallocSafe(m_instance->cudaH3x3, 2 * surfvert_number);
    CUDAMallocSafe(m_instance->cudaH6x6, 2 * (surfvert_number + surfEdge_number));
    CUDAMallocSafe(m_instance->cudaH9x9, (2 * (surfEdge_number + surfvert_number) + triangle_num));    
    CUDAMallocSafe(m_instance->cudaH12x12, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number));

    CUDAMallocSafe(m_instance->cudaD1Index, 2 * surfvert_number);
    CUDAMallocSafe(m_instance->cudaD2Index, 2 * (surfvert_number + surfEdge_number));
    CUDAMallocSafe(m_instance->cudaD3Index, (2 * (surfEdge_number + surfvert_number)+ triangle_num));
    CUDAMallocSafe(m_instance->cudaD4Index, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number));

}

void BHessian::CUDA_FREE_BHESSIAN() {
    CUDAFreeSafe(m_instance->cudaH3x3);
    CUDAFreeSafe(m_instance->cudaH6x6);
    CUDAFreeSafe(m_instance->cudaH9x9);
    CUDAFreeSafe(m_instance->cudaH12x12);

    CUDAFreeSafe(m_instance->cudaD1Index);
    CUDAFreeSafe(m_instance->cudaD2Index);
    CUDAFreeSafe(m_instance->cudaD3Index);
    CUDAFreeSafe(m_instance->cudaD4Index);    

}


void BHessian::updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number) {
    m_instance->DNum[1] = cpNums[1];
    m_instance->DNum[2] = cpNums[2] + tri_Num;
    m_instance->DNum[3] = tet_number + cpNums[3] + tri_edge_number;

#ifdef USE_FRICTION
    m_instance->DNum[1] += last_cpNums[1];
    m_instance->DNum[2] += last_cpNums[2];
    m_instance->DNum[3] += last_cpNums[3];
#endif

}



