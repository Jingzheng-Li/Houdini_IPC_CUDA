#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "UTILS/GeometryManager.hpp"

#include "BHessian.cuh"
#include "MASPreconditioner.cuh"


class PCGData {
public:
    PCGData(std::unique_ptr<GeometryManager>& instance);
    ~PCGData();

public:

    double* mc_squeue;
	double3* mc_b;
	MATHUTILS::Matrix3x3d* mc_P;
	double3* mc_r;
	double3* mc_c;
	double3* mc_q;
	double3* mc_s;
	double3* mc_z;
	double3* mc_dx;
	double3* mc_tempDx;

	double3* mc_filterTempVec3;
	double3* mc_preconditionTempVec3;
	
	int m_precondType;

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
		double threshold);

};


