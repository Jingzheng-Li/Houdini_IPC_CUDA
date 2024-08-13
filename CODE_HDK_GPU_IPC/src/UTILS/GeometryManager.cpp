
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

    freeCUDA();
    freeCUDAptr();

    instance.reset();

    CHECK_ERROR(!instance, "instance not reset");
    CHECK_ERROR(!LBVH_F_ptr, "LBVH_F_ptr not reset");
    CHECK_ERROR(!LBVH_E_ptr, "LBVH_E_ptr not reset");
    CHECK_ERROR(!AABB_SceneSize_ptr, "AABB_SceneSize_ptr not reset");
    CHECK_ERROR(!PCGData_ptr, "PCGData_ptr not reset");
    CHECK_ERROR(!BH_ptr, "BH_ptr not reset");
    CHECK_ERROR(!GIPC_ptr, "GIPC_ptr not reset");

}

void GeometryManager::freeCUDAptr() {
    std::cout << "GeometryManager freeCUDAptr" << std::endl;
    CHECK_ERROR(instance, "freeCUDAptr instance not exist right now");

    if (instance->LBVH_E_ptr)
        instance->LBVH_E_ptr->CUDA_FREE_LBVH();
    if (instance->LBVH_F_ptr)
        instance->LBVH_F_ptr->CUDA_FREE_LBVH();
    if (instance->PCGData_ptr)
        instance->PCGData_ptr->CUDA_FREE_PCGDATA();
    if (instance->BH_ptr)
        instance->BH_ptr->CUDA_FREE_BHESSIAN();
    if (instance->GIPC_ptr)
        instance->GIPC_ptr->CUDA_FREE_GIPC();

}

void GeometryManager::freeCUDA() {
    std::cout << "GeometryManager freeCUDA" << std::endl;
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
    CUDAFreeSafe(instance->cudaMoveDir);
    CUDAFreeSafe(instance->cudaTriArea);
    CUDAFreeSafe(instance->cudaTriEdges);
    CUDAFreeSafe(instance->cudaTriEdgeAdjVertex);

}
