#pragma once

#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"

#include "PCG/PCGSolver.cuh"
#include "LBVH/LBVH.cuh"


class GIPC {

public:

	GIPC(std::unique_ptr<GeometryManager>& instance);
	~GIPC();

	void CUDA_FREE_GIPC();

	void buildCP();
	void buildFullCP(const double& alpha);
	void buildBVH();
	void buildBVH_FULLCCD(const double& alpha);
	void GroundCollisionDetect();
	bool isIntersected(std::unique_ptr<GeometryManager>& instance);
	bool checkGroundIntersection();	

    void calBarrierGradientAndHessian(double3* _gradient, double mKappa);
	void calBarrierGradient(double3* _gradient, double mKap);
	void calFrictionHessian(std::unique_ptr<GeometryManager>& instance);
	void calFrictionGradient(double3* _gradient, std::unique_ptr<GeometryManager>& instance);


	void initKappa(std::unique_ptr<GeometryManager>& instance);
	void suggestKappa(double& kappa);
	void upperBoundKappa(double& kappa);

	// int solve_subIP(std::unique_ptr<GeometryManager>& instance);
	void buildFrictionSets();
	bool Inverse_Physics(std::unique_ptr<GeometryManager>& instance);

	void computeInverseHessian(std::unique_ptr<GeometryManager>& instance);
	void computeGroundHessian(double3* _gradient);
	void computeInverseGradient(std::unique_ptr<GeometryManager>& instance);
	void computeFldm(double3* _deltaPos, double3* fldm);

	// bool IPC_Solver();



public:

	std::unique_ptr<GeometryManager>& m_instance;
    std::unique_ptr<LBVH_F>& m_bvh_f;
    std::unique_ptr<LBVH_E>& m_bvh_e;
	std::unique_ptr<LBVH_EF>& m_bvh_ef;
    std::unique_ptr<PCGData>& m_pcg_data;

};

