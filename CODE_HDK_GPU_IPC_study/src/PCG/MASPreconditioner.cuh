
#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "UTILS/GeometryManager.hpp"
#include "BHessian.cuh"

class MASPreconditioner {

public:
	MASPreconditioner();
	~MASPreconditioner();
	void CUDA_MALLOC_MAS(int vertNum, int totalNeighborNum, int4* m_collisonPairs);
	void CUDA_FREE_MAS();

public:
	int neighborListSize;
	unsigned int* d_neighborList;
	unsigned int* d_neighborStart;
	unsigned int* d_neighborStartTemp;
	unsigned int* d_neighborNum;
	unsigned int* d_neighborListInit;
	unsigned int* d_neighborNumInit;

public:
	void computeNumLevels(int vertNum);

	void BuildConnectMaskL0();
	void PreparePrefixSumL0();

	void BuildLevel1();
	void BuildConnectMaskLx(int level);
	void NextLevelCluster(int level);
	void PrefixSumLx(int level);
	void ComputeNextLevel(int level);
	void AggregationKernel();

	int ReorderRealtime(int cpNum);

	void PrepareHessian(const std::unique_ptr<BHessian>& BH, const double* masses);

	void setPreconditioner(const std::unique_ptr<BHessian>& BH, const double* masses, int cpNum);

	void BuildCollisionConnection(unsigned int* connectionMsk, int* coarseTableSpace, int level, int cpNum);

	void BuildMultiLevelR(const double3* R);
	void SchwarzLocalXSym();

	void CollectFinalZ(double3* Z);

	void preconditioning(const double3* R, double3* Z);


private:
	
	int totalNodes;
	int levelnum;
	//int totalSize;
	int totalNumberClusters;
	//int bankSize;
	int2 h_clevelSize;
	int4* _collisonPairs;

	int2* d_levelSize;
	int* d_coarseSpaceTables;
	int* d_prefixOriginal;
	int* d_prefixSumOriginal;
	int* d_goingNext;
	int* d_denseLevel;
	int4* d_coarseTable;
	unsigned int* d_fineConnectMask;
	unsigned int* d_nextConnectMask;
	unsigned int* d_nextPrefix;
	unsigned int* d_nextPrefixSum;

	MATHUTILS::Matrix96x96T* d_Mat96;
    MATHUTILS::MasMatrixSymf* d_inverseMat96;
	Precision_T3* d_multiLevelR;
	Precision_T3* d_multiLevelZ;


};