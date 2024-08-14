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

	void tempMalloc_closeConstraint();
	void tempFree_closeConstraint();

	void buildCP();
	void buildFullCP(const double& alpha);
	void buildBVH();

	AABB* calcuMaxSceneSize();

	void buildBVH_FULLCCD(const double& alpha);

	void GroundCollisionDetect();
	
    void calBarrierGradientAndHessian(double3* _gradient, double mKappa);
	void calBarrierHessian();
	void calBarrierGradient(double3* _gradient, double mKap);
	void calFrictionHessian(std::unique_ptr<GeometryManager>& instance);
	void calFrictionGradient(double3* _gradient, std::unique_ptr<GeometryManager>& instance);

	int calculateMovingDirection(std::unique_ptr<GeometryManager>& instance, int cpNum, int preconditioner_type = 0);
	float computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance);
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

	bool lineSearch(std::unique_ptr<GeometryManager>& instance, double& alpha, const double& cfl_alpha);
	void postLineSearch(std::unique_ptr<GeometryManager>& instance, double alpha);

	bool checkEdgeTriIntersectionIfAny(std::unique_ptr<GeometryManager>& instance);
	bool isIntersected(std::unique_ptr<GeometryManager>& instance);
	bool checkGroundIntersection();

	void computeCloseGroundVal();
	void computeSelfCloseVal();

	bool checkCloseGroundVal();
	bool checkSelfCloseVal();

	double2 minMaxGroundDist();
	double2 minMaxSelfDist();

	void updateVelocities(std::unique_ptr<GeometryManager>& instance);
	void updateBoundary(std::unique_ptr<GeometryManager>& instance, double alpha);
	void updateBoundaryMoveDir(std::unique_ptr<GeometryManager>& instance, double alpha);
    void updateBoundary2(std::unique_ptr<GeometryManager>& instance);

	void initKappa(std::unique_ptr<GeometryManager>& instance);
	void suggestKappa(double& kappa);
	void upperBoundKappa(double& kappa);

	int solve_subIP(std::unique_ptr<GeometryManager>& instance);
	void buildFrictionSets();
	bool Inverse_Physics(std::unique_ptr<GeometryManager>& instance);

	void computeInverseHessian(std::unique_ptr<GeometryManager>& instance);
	void computeGroundHessian(double3* _gradient);
	void computeInverseGradient(std::unique_ptr<GeometryManager>& instance);
	void computeFldm(double3* _deltaPos, double3* fldm);

	void IPC_Solver();


public:

	std::unique_ptr<GeometryManager>& m_instance;
    std::unique_ptr<LBVH_F>& m_bvh_f;
    std::unique_ptr<LBVH_E>& m_bvh_e;
    std::unique_ptr<PCGData>& m_pcg_data;
    std::unique_ptr<BHessian>& m_BH;



	double3* mc_vertexes;
	double3* mc_rest_vertexes;
	uint3* mc_faces;
	uint2* mc_edges;
	uint32_t* mc_surfVerts;


	double3* mc_targetVert;	
	uint32_t* mc_targetInd;	
	uint32_t m_softNum;
	uint32_t m_triangleNum;

	double3* mc_moveDir;

	int4* mc_collisonPairs;
	int4* mc_ccd_collisonPairs;
	uint32_t* mc_cpNum;
	uint32_t* mc_close_cpNum;
	int* mc_MatIndex;

	uint32_t* mc_environment_collisionPair;

	uint32_t* mc_closeConstraintID;
	double* mc_closeConstraintVal;
	int4* mc_closeMConstraintID;
	double* mc_closeMConstraintVal;

	uint32_t* mc_gpNum;
	uint32_t* mc_close_gpNum;

	uint32_t h_cpNum[5];
	uint32_t h_ccd_cpNum;
	uint32_t h_gpNum;
	uint32_t h_close_cpNum;
	uint32_t h_close_gpNum;
	uint32_t h_gpNum_last;
	uint32_t h_cpNum_last[5];



	double3* mc_groundNormal;
	double* mc_groundOffset;

	// for friction
	double* mc_lambda_lastH_scalar;
	double2* mc_distCoord;
	MATHUTILS::Matrix3x2d* mc_tanBasis;
	int4* mc_collisonPairs_lastH;
	int* mc_MatIndex_last;
	double* mc_lambda_lastH_scalar_gd;
	uint32_t* mc_collisonPairs_lastH_gd;


	uint32_t m_vertexNum;
	uint32_t m_surf_vertexNum;
	uint32_t m_surf_edgeNum;
	uint32_t m_tri_edge_num;
	uint32_t m_surf_faceNum;
	uint32_t m_tetrahedraNum;


	int m_MAX_COLLITION_PAIRS_NUM;
	int m_MAX_CCD_COLLITION_PAIRS_NUM;

	double m_animation_subRate;
	double m_animation_fullRate;

	bool m_isRotate;

};

