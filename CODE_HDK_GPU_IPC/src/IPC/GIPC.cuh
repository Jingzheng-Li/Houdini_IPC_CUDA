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

	int solve_subIP(std::unique_ptr<GeometryManager>& instance, double& time0, double& time1, double& time2, double& time3, double& time4);
	void buildFrictionSets();
	bool Inverse_Physics(std::unique_ptr<GeometryManager>& instance);

	void computeInverseHessian(std::unique_ptr<GeometryManager>& instance);
	void computeGroundHessian(double3* _gradient);
	void computeInverseGradient(std::unique_ptr<GeometryManager>& instance);
	void computeFldm(double3* _deltaPos, double3* fldm);

	void IPC_Solver();


public:

	std::unique_ptr<GeometryManager>& m_instance;
	std::unique_ptr<AABB>& m_scene_size;
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

	bool m_animation;
	double m_IPCKappa;
	double m_dHat;
	double m_fDhat;
	double m_bboxDiagSize2;
	double m_relative_dhat;
	double dTol;
    double minKappaCoef;
	double IPC_dt;
	double meanMass;
	double meanVolumn;
	double3* _groundNormal;
	double* _groundOffset;

	// for friction
	double* lambda_lastH_scalar;
	double2* distCoord;
	MATHUTILS::Matrix3x2d* tanBasis;
	int4* _collisonPairs_lastH;
	int* _MatIndex_last;
	double* lambda_lastH_scalar_gd;
	uint32_t* _collisonPairs_lastH_gd;


	uint32_t vertexNum;
	uint32_t surf_vertexNum;
	uint32_t surf_edgeNum;
	uint32_t tri_edge_num;
	uint32_t surf_faceNum;
	uint32_t tetrahedraNum;


	int MAX_COLLITION_PAIRS_NUM;
	int MAX_CCD_COLLITION_PAIRS_NUM;

	double animation_subRate;
	double animation_fullRate;

	double m_bendStiff;
	double m_density;
	double m_YoungModulus;
	double m_PoissonRate;
	double m_lengthRateLame;
	double m_volumeRateLame;
	double m_lengthRate;
	double m_volumeRate;
	double m_frictionRate;
	double m_clothThickness;
	double m_clothYoungModulus;
	double m_stretchStiff;
	double m_shearStiff;
	double m_clothDensity;
	double m_softMotionRate;
	double m_Newton_solver_threshold;
	double m_pcg_threshold;
	//bool USE_MAS;

};

