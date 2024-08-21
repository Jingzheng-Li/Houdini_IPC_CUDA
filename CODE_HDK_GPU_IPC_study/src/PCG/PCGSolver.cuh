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

	// TODO: change m_ into mc_
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

	double3* m_filterTempVec3;
	double3* m_preconditionTempVec3;
	int m_precondType;

	MASPreconditioner MP;
	

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


