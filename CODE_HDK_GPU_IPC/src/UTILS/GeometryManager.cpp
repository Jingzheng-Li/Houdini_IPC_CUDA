
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
    cudaD4Index(nullptr),
    cudaD3Index(nullptr),
    cudaD2Index(nullptr),
    cudaD1Index(nullptr),
    cudaCloseGPNum(nullptr),
    cudaCloseCPNum(nullptr),
    cudaTargetVert(nullptr),
    cudaTargetIndex(nullptr),
    cudaXTilta(nullptr),
    cudaFb(nullptr),
    cudaMoveDir(nullptr),
    cudaTriArea(nullptr),
    cudaTriEdges(nullptr),
    cudaTriEdgeAdjVertex(nullptr),
    cudaTempDouble3Mem(nullptr),
    cudaTriDmInverses(nullptr)
    {}

GeometryManager::~GeometryManager() {
    std::cout << "deconstruct GeometryManager" << std::endl;
    if (instance) {
        std::cout << "~GeometryManager freeGeometryManager" << std::endl;
        freeGeometryManager();
    }
}

void GeometryManager::freeGeometryManager() {
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    freeCUDAptr();
    freeCUDA();
    instance.reset();
}


void GeometryManager::freeCUDAptr() {
    if (instance->LBVH_E_ptr) {
        instance->LBVH_E_ptr->CUDA_FREE_LBVH();
        instance->LBVH_E_ptr.reset();
    }
    if (instance->LBVH_F_ptr) {
        instance->LBVH_F_ptr->CUDA_FREE_LBVH();
        instance->LBVH_F_ptr.reset();
    }
    if (instance->PCGData_ptr) {
        instance->PCGData_ptr->CUDA_FREE_PCGDATA();
        instance->PCGData_ptr.reset();
    }
    if (instance->BH_ptr) {
        instance->BH_ptr->CUDA_FREE_BHESSIAN();
        instance->BH_ptr.reset();
    }
    // instance->GIPC_ptr->;
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
    CUDAFreeSafe(instance->cudaD4Index);
    CUDAFreeSafe(instance->cudaD3Index);
    CUDAFreeSafe(instance->cudaD2Index);
    CUDAFreeSafe(instance->cudaD1Index);
    CUDAFreeSafe(instance->cudaCloseGPNum);
    CUDAFreeSafe(instance->cudaCloseCPNum);
    CUDAFreeSafe(instance->cudaTargetVert);
    CUDAFreeSafe(instance->cudaTargetIndex);
    CUDAFreeSafe(instance->cudaXTilta);
    CUDAFreeSafe(instance->cudaFb);
    // CUDAFreeSafe(instance->cudaMoveDir);
    CUDAFreeSafe(instance->cudaTriArea);
    CUDAFreeSafe(instance->cudaTriEdges);
    CUDAFreeSafe(instance->cudaTriEdgeAdjVertex);
    CUDAFreeSafe(instance->cudaTempDouble3Mem);
    CUDAFreeSafe(instance->cudaTriDmInverses);
    
}
