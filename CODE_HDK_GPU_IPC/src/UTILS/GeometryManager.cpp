
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
    cudaCloseGPNum(nullptr),
    cudaCloseCPNum(nullptr),
    cudaGroundNormal(nullptr),
    cudaGroundOffset(nullptr),
    cudaMatIndex(nullptr),
    cudaMortonCodeHash(nullptr),
    cudaSortIndex(nullptr),
    cudaSortMapVertIndex(nullptr),
    cudaTempDouble(nullptr),
    cudaTempDouble3Mem(nullptr),
    cudaDmInverses(nullptr),
    cudaTriDmInverses(nullptr),
    cudaTempMat3x3(nullptr),
    cudaH12x12(nullptr),
    cudaH9x9(nullptr),
    cudaH6x6(nullptr),
    cudaH3x3(nullptr),
    cudaD1Index(nullptr),
    cudaD3Index(nullptr),
    cudaD4Index(nullptr),
    cudaD2Index(nullptr),
    cudaTargetVert(nullptr),
    cudaTargetIndex(nullptr),
    cudaXTilta(nullptr),
    cudaFb(nullptr),
    cudaMoveDir(nullptr),
    cudaTriArea(nullptr),
    cudaTriEdges(nullptr),
    cudaTriEdgeAdjVertex(nullptr) {}

GeometryManager::~GeometryManager() {
    std::cout << "GeometryManager Deconstruction" << std::endl;
}

void GeometryManager::freeGeometryManager() {

    std::cout << "running freeGeometryManager" << std::endl;

    freeCUDA();
    freeCUDAptr();
    instance.reset();

}



void GeometryManager::freeCUDAptr() {
    CHECK_ERROR(instance, "freeCUDAptr instance not exist right now");

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
    if (instance->AABB_SceneSize_ptr) {
        instance->AABB_SceneSize_ptr.reset();
    }
    if (instance->BH_ptr) {
        instance->BH_ptr->CUDA_FREE_BHESSIAN();
        instance->BH_ptr.reset();
    }
    if (instance->GIPC_ptr) {
        instance->GIPC_ptr->CUDA_FREE_GIPC();
        instance->GIPC_ptr.reset();
    }

}

void GeometryManager::freeCUDA() {
    CHECK_ERROR(instance, "freeCUDA instance not exist right now");

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
    CUDAFreeSafe(instance->cudaCloseGPNum);
    CUDAFreeSafe(instance->cudaCloseCPNum);
    CUDAFreeSafe(instance->cudaGroundNormal);
    CUDAFreeSafe(instance->cudaGroundOffset);
    CUDAFreeSafe(instance->cudaMatIndex);
    CUDAFreeSafe(instance->cudaMortonCodeHash);
    CUDAFreeSafe(instance->cudaSortIndex);
    CUDAFreeSafe(instance->cudaSortMapVertIndex);
    CUDAFreeSafe(instance->cudaTempDouble);
    CUDAFreeSafe(instance->cudaTempDouble3Mem);
    CUDAFreeSafe(instance->cudaDmInverses);
    CUDAFreeSafe(instance->cudaTriDmInverses);
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
    CUDAFreeSafe(instance->cudaTargetVert);
    CUDAFreeSafe(instance->cudaTargetIndex);
    CUDAFreeSafe(instance->cudaXTilta);
    CUDAFreeSafe(instance->cudaFb);
    // CUDAFreeSafe(instance->cudaMoveDir); ??? why this is use pcg m_dx??
    CUDAFreeSafe(instance->cudaTriArea);
    CUDAFreeSafe(instance->cudaTriEdges);
    CUDAFreeSafe(instance->cudaTriEdgeAdjVertex);

}
