#pragma once

#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"

#include "PCG/PCGSolver.cuh"
#include "LBVH/LBVH.cuh"


// class GIPC {

// public:

// 	GIPC();
// 	~GIPC();

// 	void MALLOC_DEVICE_MEM();

// 	void tempMalloc_closeConstraint();
// 	void tempFree_closeConstraint();

// 	void FREE_DEVICE_MEM();
// 	void initBVH(int* _btype);
// 	void init(double m_meanMass, double m_meanVolumn, double3 minConer, double3 maxConer);

// 	void buildCP();
// 	void buildFullCP(const double& alpha);
// 	void buildBVH();

// 	AABB* calcuMaxSceneSize();

// 	void buildBVH_FULLCCD(const double& alpha);


// 	void GroundCollisionDetect();
	
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

// 	double Energy_Add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance);

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

// 	double2 minMaxGroundDist();
// 	double2 minMaxSelfDist();

// 	void updateVelocities(std::unique_ptr<GeometryManager>& instance);
// 	void updateBoundary(std::unique_ptr<GeometryManager>& instance, double alpha);
// 	void updateBoundaryMoveDir(std::unique_ptr<GeometryManager>& instance, double alpha);
//     void updateBoundary2(std::unique_ptr<GeometryManager>& instance);
// 	void computeXTilta(std::unique_ptr<GeometryManager>& instance, const double& rate);

// 	void initKappa(std::unique_ptr<GeometryManager>& instance);
// 	void suggestKappa(double& kappa);
// 	void upperBoundKappa(double& kappa);
// 	int solve_subIP(std::unique_ptr<GeometryManager>& instance, double& time0, double& time1, double& time2, double& time3, double& time4);
// 	void IPC_Solver(std::unique_ptr<GeometryManager>& instance);
// 	void sortMesh(std::unique_ptr<GeometryManager>& instance, int updateVertNum);
// 	void buildFrictionSets();


// 	bool Inverse_Physics(std::unique_ptr<GeometryManager>& instance);
// 	void computeInverseHessian(std::unique_ptr<GeometryManager>& instance);
// 	void computeGroundHessian(double3* _gradient);
// 	void computeInverseGradient(std::unique_ptr<GeometryManager>& instance);
// 	void computeFldm(double3* _deltaPos, double3* fldm);


// public:

// 	bool animation;
// 	double3* _vertexes;
// 	double3* _rest_vertexes;
// 	uint3* _faces;
// 	uint2* _edges;
// 	uint32_t* _surfVerts;


// 	double3* targetVert;	
// 	uint32_t* targetInd;	
// 	uint32_t softNum;
// 	uint32_t triangleNum;


// 	double3* _moveDir;
// 	// lbvh_f bvh_f;
// 	// lbvh_e bvh_e;

// 	PCGData pcg_data;

// 	int4* _collisonPairs;
// 	int4* _ccd_collisonPairs;
// 	uint32_t* _cpNum;
// 	int* _MatIndex;
// 	uint32_t* _close_cpNum;

// 	uint32_t* _environment_collisionPair;

// 	uint32_t* _closeConstraintID;
// 	double* _closeConstraintVal;

// 	int4* _closeMConstraintID;
// 	double* _closeMConstraintVal;

// 	uint32_t* _gpNum;
// 	uint32_t* _close_gpNum;
// 	//uint32_t* _cpNum;
// 	uint32_t h_cpNum[5];
// 	uint32_t h_ccd_cpNum;
// 	uint32_t h_gpNum;
	
// 	uint32_t h_close_cpNum;
// 	uint32_t h_close_gpNum;

// 	double IPCKappa;
// 	double dHat;
// 	double fDhat;
// 	double bboxDiagSize2;
// 	double relative_dhat;
// 	double dTol;
//     double minKappaCoef;
// 	double IPC_dt;
// 	double Step;
// 	double meanMass;
// 	double meanVolumn;
// 	double3* _groundNormal;
// 	double* _groundOffset;

// 	// for friction
// 	double* lambda_lastH_scalar;
// 	double2* distCoord;
// 	MATHUTILS::Matrix3x2d* tanBasis;
// 	int4* _collisonPairs_lastH;
// 	uint32_t h_cpNum_last[5];
// 	int* _MatIndex_last;

// 	double* lambda_lastH_scalar_gd;
// 	uint32_t* _collisonPairs_lastH_gd;
// 	uint32_t h_gpNum_last;

// 	uint32_t vertexNum;
// 	uint32_t surf_vertexNum;
// 	uint32_t surf_edgeNum;
// 	uint32_t tri_edge_num;
// 	uint32_t surf_faceNum;
// 	uint32_t tetrahedraNum;

// 	BHessian BH;
// 	AABB SceneSize;
// 	int MAX_COLLITION_PAIRS_NUM;
// 	int MAX_CCD_COLLITION_PAIRS_NUM;

// 	double RestNHEnergy;
// 	double animation_subRate;
// 	double animation_fullRate;


// 	double bendStiff;

// 	double density;
// 	double YoungModulus;
// 	double PoissonRate;
// 	double lengthRateLame;
// 	double volumeRateLame;
// 	double lengthRate;
// 	double volumeRate;
// 	double frictionRate;
// 	double clothThickness;
// 	double clothYoungModulus;
// 	double stretchStiff;
// 	double shearStiff;
// 	double clothDensity;
// 	double softMotionRate;
// 	double Newton_solver_threshold;
// 	double pcg_threshold;
// 	//bool USE_MAS;

// };

