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

	void tempMalloc_closeConstraint();
	void tempFree_closeConstraint();

	void buildCP();
	void buildFullCP(const double& alpha);
	void buildBVH();
	void buildBVH_FULLCCD(const double& alpha);
	void GroundCollisionDetect();
	// bool checkEdgeTriIntersectionIfAny(std::unique_ptr<GeometryManager>& instance);
	bool isIntersected(std::unique_ptr<GeometryManager>& instance);
	bool checkGroundIntersection();

	void computeCloseGroundVal();
	void computeSelfCloseVal();
	// bool checkCloseGroundVal();
	// bool checkSelfCloseVal();
	double2 minMaxGroundDist();
	double2 minMaxSelfDist();
	

    void calBarrierGradientAndHessian(double3* _gradient, double mKappa);
	// void calBarrierHessian();
	void calBarrierGradient(double3* _gradient, double mKap);
	void calFrictionHessian(std::unique_ptr<GeometryManager>& instance);
	void calFrictionGradient(double3* _gradient, std::unique_ptr<GeometryManager>& instance);

	int calculateMovingDirection(std::unique_ptr<GeometryManager>& instance, int cpNum, int preconditioner_type = 0);
	// void computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance);
	void computeGroundGradientAndHessian(double3* _gradient);
	void computeGroundGradient(double3* _gradient, double mKap);
	void computeSoftConstraintGradientAndHessian(double3* _gradient);
	void computeSoftConstraintGradient(double3* _gradient);

	double computeEnergy(std::unique_ptr<GeometryManager>& instance);
	double Energy_Add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance);

	double ground_largestFeasibleStepSize(double slackness, double* mqueue);
	double self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers);
	double InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets);
	double cfl_largestSpeed(double* mqueue);

	void lineSearch(std::unique_ptr<GeometryManager>& instance, double& alpha, const double& cfl_alpha);
	void postLineSearch(std::unique_ptr<GeometryManager>& instance, double alpha);



	void updateVelocities(std::unique_ptr<GeometryManager>& instance);
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

