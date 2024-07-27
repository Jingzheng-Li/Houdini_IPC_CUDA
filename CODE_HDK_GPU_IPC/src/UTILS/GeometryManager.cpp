
#include "GeometryManager.hpp"

#include "LBVH/LBVH.cuh"
#include "PCG/PCGSolver.cuh"
#include "IPC/GIPC.cuh"

std::unique_ptr<GeometryManager> GeometryManager::instance = nullptr;


GeometryManager::GeometryManager() : 
    cudaVertPos(nullptr), 
    cudaVertVel(nullptr), 
    cudaVertMass(nullptr), 
    cudaTetElement(nullptr),
    cudaTriElement(nullptr),
    cudaTetVolume(nullptr),
    cudaSurfVert(nullptr), 
    cudaSurfFace(nullptr), 
    cudaSurfEdge(nullptr),
    cudaBoundaryType(nullptr),
    cudaTempBoundaryType(nullptr),
    cudaConstraints(nullptr),
    cudaRestVertPos(nullptr),
    cudaOriginVertPos(nullptr),
    cudaCollisionPairs(nullptr),
    cudaCCDCollisionPairs(nullptr),
    cudaEnvCollisionPairs(nullptr),
    cudaCPNum(nullptr),
    cudaGPNum(nullptr),
    cudaGroundNormal(nullptr),
    cudaGroundOffset(nullptr),
    cudaMatIndex(nullptr),
    cudaMortonCodeHash(nullptr),
    cudaSortIndex(nullptr),
    cudaSortMapVertIndex(nullptr),
    cudaTempDouble(nullptr),
    cudaDmInverses(nullptr),
    cudaTempMat3x3(nullptr),
    cudaH12x12(nullptr),
    cudaH9x9(nullptr),
    cudaH6x6(nullptr),
    cudaH3x3(nullptr),
    cudaD1Index(nullptr),
    cudaD3Index(nullptr),
    cudaD4Index(nullptr),
    cudaD2Index(nullptr) {}

GeometryManager::~GeometryManager() {
    freeCUDA();
}

void GeometryManager::totallyfree() {
    if (instance) {
        instance->LBVH_E_ptr->CUDA_FREE_LBVH();
        instance->LBVH_F_ptr->CUDA_FREE_LBVH();
        // instance->AABB_SceneSize_ptr->;
        // instance->PCGData_ptr->CUDA_FREE_PCGDATA();
        // instance->BH_ptr->CUDA_FREE_BHESSIAN();
        // instance->GIPC_ptr->;

        instance.reset();
    }
}

void GeometryManager::freeCUDA() {
    CUDAFreeSafe(instance->cudaVertPos);
    CUDAFreeSafe(instance->cudaVertVel);
    CUDAFreeSafe(instance->cudaVertMass);
    CUDAFreeSafe(instance->cudaTetElement);
    CUDAFreeSafe(instance->cudaTriElement);
    CUDAFreeSafe(instance->cudaTetVolume);
    CUDAFreeSafe(instance->cudaSurfVert);
    CUDAFreeSafe(instance->cudaSurfFace);
    CUDAFreeSafe(instance->cudaSurfEdge);
    CUDAFreeSafe(instance->cudaConstraints);
    CUDAFreeSafe(instance->cudaRestVertPos);
    CUDAFreeSafe(instance->cudaOriginVertPos);
    CUDAFreeSafe(instance->cudaCollisionPairs);
    CUDAFreeSafe(instance->cudaCCDCollisionPairs);
    CUDAFreeSafe(instance->cudaEnvCollisionPairs);
    CUDAFreeSafe(instance->cudaCPNum);
    CUDAFreeSafe(instance->cudaGPNum);
    CUDAFreeSafe(instance->cudaGroundNormal);
    CUDAFreeSafe(instance->cudaGroundOffset);
    CUDAFreeSafe(instance->cudaMatIndex);
    CUDAFreeSafe(instance->cudaMortonCodeHash);
    CUDAFreeSafe(instance->cudaSortIndex);
    CUDAFreeSafe(instance->cudaSortMapVertIndex);
    CUDAFreeSafe(instance->cudaTempDouble);
    CUDAFreeSafe(instance->cudaDmInverses);
    CUDAFreeSafe(instance->cudaTempMat3x3);
    CUDAFreeSafe(instance->cudaBoundaryType);
    CUDAFreeSafe(instance->cudaTempBoundaryType);
    CUDAFreeSafe(instance->cudaH12x12);
    CUDAFreeSafe(instance->cudaH9x9);
    CUDAFreeSafe(instance->cudaH6x6);
    CUDAFreeSafe(instance->cudaH3x3);
    CUDAFreeSafe(instance->cudaD1Index);
    CUDAFreeSafe(instance->cudaD3Index);
    CUDAFreeSafe(instance->cudaD4Index);
    CUDAFreeSafe(instance->cudaD2Index);
}
