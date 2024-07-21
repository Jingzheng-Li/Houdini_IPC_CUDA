#pragma once

// #include "UTILS/GeometryManager.hpp"
// #include "UTILS/CUDAUtils.hpp"
// #include "UTILS/MathUtils.cuh"

// #include "PCG/PCGSolver.cuh"
// #include "LBVH/LBVH.cuh"

// class GIPC {
// public:
//     GIPC();
//     ~GIPC();

//     void tempMalloc_closeConstraint();
// 	void tempFree_closeConstraint();

//     void buildFullCP(const double& alpha);
//     void buildBVH_FULLCCD(const double& alpha);

//     void calBarrierGradientAndHessian(double3* _gradient, double mKappa);
// 	void calBarrierHessian();
// 	void calBarrierGradient(double3* _gradient, double mKap);
// 	void calFrictionHessian(std::unique_ptr<GeometryManager>& instance);
// 	void calFrictionGradient(double3* _gradient, std::unique_ptr<GeometryManager>& instance);

// 	int calculateMovingDirection(std::unique_ptr<GeometryManager>& instance, int cpNum, int preconditioner_type = 0);
// 	float computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance);
// 	void computeGroundGradientAndHessian(double3* _gradient);
// 	void computeGroundGradient(double3* _gradient, double mKap);
// 	void computeSoftConstraintGradientAndHessian(double3* _gradient);
// 	void computeSoftConstraintGradient(double3* _gradient);
// 	double computeEnergy(std::unique_ptr<GeometryManager>& instance);

//     double Energy_Add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance);
// 	double ground_largestFeasibleStepSize(double slackness, double* mqueue);
// 	double self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers);
// 	double InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets);
// 	double cfl_largestSpeed(double* mqueue);

// 	bool lineSearch(std::unique_ptr<GeometryManager>& instance, double& alpha, const double& cfl_alpha);
// 	void postLineSearch(std::unique_ptr<GeometryManager>& instance, double alpha);

// 	bool checkEdgeTriIntersectionIfAny(std::unique_ptr<GeometryManager>& instance);
// 	bool isIntersected(std::unique_ptr<GeometryManager>& instance);
// 	bool checkGroundIntersection();

// 	void computeCloseGroundVal();
// 	void computeSelfCloseVal();

// 	bool checkCloseGroundVal();
// 	bool checkSelfCloseVal();

//     void updateVelocities(std::unique_ptr<GeometryManager>& instance);
// 	void updateBoundary(std::unique_ptr<GeometryManager>& instance, double alpha);
// 	void updateBoundaryMoveDir(std::unique_ptr<GeometryManager>& instance, double alpha);
//     void updateBoundary2(std::unique_ptr<GeometryManager>& instance);

//     void initKappa(std::unique_ptr<GeometryManager>& instance);
// 	void suggestKappa(double& kappa);
// 	void upperBoundKappa(double& kappa);
// 	int solve_subIP(std::unique_ptr<GeometryManager>& instance, double& time0, double& time1, double& time2, double& time3, double& time4);
// 	void buildFrictionSets();

// 	bool Inverse_Physics(std::unique_ptr<GeometryManager>& instance);
// 	void computeInverseHessian(std::unique_ptr<GeometryManager>& instance);
// 	void computeGroundHessian(double3* _gradient);
// 	void computeInverseGradient(std::unique_ptr<GeometryManager>& instance);
// 	void computeFldm(double3* _deltaPos, double3* fldm);

// 	void IPC_Solver(std::unique_ptr<GeometryManager>& instance);


// public:





// public:

//     void CUDA_MALLOC_GIPC();
//     void CUDA_FREE_GIPC();

// };

