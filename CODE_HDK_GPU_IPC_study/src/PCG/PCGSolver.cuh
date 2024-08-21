#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "UTILS/GeometryManager.hpp"


class BHessian {
public:
    BHessian();
	~BHessian();
	void updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number);

public:

    uint32_t* mc_D1Index;//pIndex, DpeIndex, DptIndex;
	uint3* mc_D3Index;
	uint4* mc_D4Index;
	uint2* mc_D2Index;
	MATHUTILS::Matrix12x12d* mc_H12x12;
	MATHUTILS::Matrix3x3d* mc_H3x3;
	MATHUTILS::Matrix6x6d* mc_H6x6;
	MATHUTILS::Matrix9x9d* mc_H9x9;

	uint32_t m_DNum[4];

public:
    void CUDA_MALLOC_BHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number, const int& triangle_num, const int& tri_Edge_number);
    void CUDA_FREE_BHESSIAN();

};


// class MASPreconditioner {
// public:
//     MASPreconditioner();
//     ~MASPreconditioner();
//     // Preconditioner is not avaliable

// };



class PCGData {
public:
    PCGData();
    ~PCGData();

public:
    // TODO: 全部切换成mc_*
    double* m_squeue;
	double3* m_b;
	MATHUTILS::Matrix3x3d* m_P;
	double3* m_r;
	double3* m_c;
	double3* m_q;
	double3* m_s;
	double3* m_z;
	double3* m_dx;
	double3* m_tempDx;

    int m_PrecondType;
	// double3* m_filterTempVec3;
	// double3* m_preconditionTempVec3;

public:
    void CUDA_MALLOC_PCGDATA(const int& vertexNum, const int& tetrahedraNum);
    void CUDA_FREE_PCGDATA();

};


namespace PCGSOLVER {
    
	int PCG_Process(
		std::unique_ptr<GeometryManager>& instance, 
		std::unique_ptr<PCGData>& pcg_data, 
		const std::unique_ptr<BHessian>& BH,
		double3* _mvDir, 
		int vertexNum, 
		int tetrahedraNum, 
		double IPC_dt, 
		double meanVolumn, 
		double threshold);

    // int MASPCG_Process(device_TetraData* mesh, PCG_Data* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn, int cpNum, double threshold);

};


