
#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "UTILS/GeometryManager.hpp"

class MASPreconditioner {

public:
	MASPreconditioner();
	~MASPreconditioner();
	void CUDA_MALLOC_MAS(int vertNum, int totalNeighborNum, int4* _collisonPairs);
	void CUDA_FREE_MAS();

public:
	int neighborListSize;
	unsigned int* mc_neighborList;
	unsigned int* mc_neighborStart;
	unsigned int* mc_neighborStartTemp;
	unsigned int* mc_neighborNum;
	unsigned int* mc_neighborListInit;
	unsigned int* mc_neighborNumInit;

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

	void PrepareHessian(const std::unique_ptr<GeometryManager>& instance, const double* masses);

	void setPreconditioner(const std::unique_ptr<GeometryManager>& instance, int cpNum);

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
	int4* mc_collisonPairs;

	int2* mc_levelSize;
	int* mc_coarseSpaceTables;
	int* mc_prefixOriginal;
	int* mc_prefixSumOriginal;
	int* mc_goingNext;
	int* mc_denseLevel;
	int4* mc_coarseTable;
	unsigned int* mc_fineConnectMask;
	unsigned int* mc_nextConnectMask;
	unsigned int* mc_nextPrefix;
	unsigned int* mc_nextPrefixSum;

	MATHUTILS::Matrix96x96T* mc_Mat96;
    MATHUTILS::MasMatrixSymf* mc_inverseMat96;
	Precision_T3* mc_multiLevelR;
	Precision_T3* mc_multiLevelZ;

};