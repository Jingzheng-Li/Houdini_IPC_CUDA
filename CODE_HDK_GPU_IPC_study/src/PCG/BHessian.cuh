#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "UTILS/GeometryManager.hpp"

class BHessian {
public:
    BHessian(std::unique_ptr<GeometryManager>& instance);
	~BHessian();
	void updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number);

public:

    // uint32_t* mc_D1Index;//pIndex, DpeIndex, DptIndex;
	// uint3* mc_D3Index;
	// uint4* mc_D4Index;
	// uint2* mc_D2Index;
	// MATHUTILS::Matrix12x12d* mc_H12x12;
	// MATHUTILS::Matrix3x3d* mc_H3x3;
	// MATHUTILS::Matrix6x6d* mc_H6x6;
	// MATHUTILS::Matrix9x9d* mc_H9x9;
	// uint32_t m_DNum[4];
	std::unique_ptr<GeometryManager>& m_instance;

public:
    void CUDA_MALLOC_BHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number, const int& triangle_num, const int& tri_Edge_number);
    void CUDA_FREE_BHESSIAN();

};

