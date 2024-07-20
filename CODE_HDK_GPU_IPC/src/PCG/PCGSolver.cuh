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

    // TODO: 全部切换成mc_*
    uint32_t* D1Index;//pIndex, DpeIndex, DptIndex;
	uint3* D3Index;
	uint4* D4Index;
	uint2* D2Index;
	MATHUTILS::Matrix12x12d* H12x12;
	MATHUTILS::Matrix3x3d* H3x3;
	MATHUTILS::Matrix6x6d* H6x6;
	MATHUTILS::Matrix9x9d* H9x9;

	uint32_t DNum[4];

public:
    void CUDA_MALLOC_BHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number, const int& triangle_num, const int& tri_Edge_number);
    void CUDA_FREE_BHESSIAN();

};


class MASPreconditioner {
public:
    MASPreconditioner();
    ~MASPreconditioner();
    // Preconditioner is not avaliable


};



class PCGData {
public:
    PCGData();
    ~PCGData();

public:
    // TODO: 全部切换成mc_*
    double* squeue;
	double3* b;
	MATHUTILS::Matrix3x3d* P;
	double3* r;
	double3* c;
	double3* q;
	double3* s;
	double3* z;
	double3* dx;
	double3* tempDx;

	double3* filterTempVec3;
	double3* preconditionTempVec3;

    // int P_type;
    // double3* preconditionTempVec3;

public:
    void CUDA_MALLOC_PCGDATA(const int& vertexNum, const int& tetrahedraNum);
    void CUDA_FREE_PCGDATA();

};


namespace PCGSOLVER {
    int PCG_Process(std::unique_ptr<GeometryManager>& instance, PCGData* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn, double threshold);

    // int MASPCG_Process(device_TetraData* mesh, PCG_Data* pcg_data, const BHessian& BH, double3* _mvDir, int vertexNum, int tetrahedraNum, double IPC_dt, double meanVolumn, int cpNum, double threshold);
};


