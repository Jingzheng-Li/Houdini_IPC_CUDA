
#include "GeometryManager.hpp"

#include "LBVH/LBVH.cuh"
#include "PCG/PCGSolver.cuh"
#include "IPC/GIPC.cuh"
#include "INTEGRATOR/ImplicitIntergrator.cuh"

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
    cudaConstraintsMat(nullptr),
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
    cudaTetDmInverses(nullptr),
    cudaTempMat3x3(nullptr),
    cudaCloseGPNum(nullptr),
    cudaCloseCPNum(nullptr),
    cudaTargetVertPos(nullptr),
    cudaTargetIndex(nullptr),
    cudaXTilta(nullptr),
    cudaFb(nullptr),
    cudaMoveDir(nullptr),
    cudaTriArea(nullptr),
    cudaTriEdges(nullptr),
    cudaTriEdgeAdjVertex(nullptr),
    cudaTempDouble3Mem(nullptr),
    cudaTriDmInverses(nullptr),
    cudaCloseConstraintID(nullptr),
    cudaCloseConstraintVal(nullptr),
    cudaCloseMConstraintID(nullptr),
    cudaCloseMConstraintVal(nullptr),
    cudaLambdaLastHScalar(nullptr),
    cudaDistCoord(nullptr),
    cudaTanBasis(nullptr),
    cudaCollisonPairsLastH(nullptr),
    cudaMatIndexLast(nullptr),
    cudaLambdaLastHScalarGd(nullptr),
    cudaCollisonPairsLastHGd(nullptr),
    cudaD1Index(nullptr),
    cudaD2Index(nullptr),
    cudaD3Index(nullptr),
    cudaD4Index(nullptr),
    cudaH3x3(nullptr),
    cudaH6x6(nullptr),
    cudaH9x9(nullptr),
    cudaH12x12(nullptr)
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
    if (instance->LBVH_EF_ptr) {
        instance->LBVH_EF_ptr.reset();
    }
    if (instance->MAS_ptr) {
        instance->MAS_ptr->CUDA_FREE_MAS();
        instance->MAS_ptr.reset();
    }
    if (instance->PCGData_ptr) {
        instance->PCGData_ptr->CUDA_FREE_PCGDATA();
        instance->PCGData_ptr.reset();
    }
    if (instance->Integrator_ptr) {
        instance->Integrator_ptr.reset();
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
    CUDAFreeSafe(instance->cudaConstraintsMat);
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
    CUDAFreeSafe(instance->cudaTetDmInverses);
    CUDAFreeSafe(instance->cudaTempMat3x3);
    CUDAFreeSafe(instance->cudaBoundaryType);
    CUDAFreeSafe(instance->cudaTempBoundaryType);
    CUDAFreeSafe(instance->cudaCloseGPNum);
    CUDAFreeSafe(instance->cudaCloseCPNum);
    CUDAFreeSafe(instance->cudaTargetVertPos);
    CUDAFreeSafe(instance->cudaTargetIndex);
    CUDAFreeSafe(instance->cudaXTilta);
    CUDAFreeSafe(instance->cudaFb);
    // CUDAFreeSafe(instance->cudaMoveDir);
    CUDAFreeSafe(instance->cudaTriArea);
    CUDAFreeSafe(instance->cudaTriEdges);
    CUDAFreeSafe(instance->cudaTriEdgeAdjVertex);
    CUDAFreeSafe(instance->cudaTempDouble3Mem);
    CUDAFreeSafe(instance->cudaTriDmInverses);
    CUDAFreeSafe(instance->cudaCloseConstraintID);
    CUDAFreeSafe(instance->cudaCloseConstraintVal);
    CUDAFreeSafe(instance->cudaCloseMConstraintID);
    CUDAFreeSafe(instance->cudaCloseMConstraintVal);
    CUDAFreeSafe(instance->cudaLambdaLastHScalar);
    CUDAFreeSafe(instance->cudaDistCoord);
    CUDAFreeSafe(instance->cudaTanBasis);
    CUDAFreeSafe(instance->cudaCollisonPairsLastH);
    CUDAFreeSafe(instance->cudaMatIndexLast);
    CUDAFreeSafe(instance->cudaLambdaLastHScalarGd);
    CUDAFreeSafe(instance->cudaCollisonPairsLastHGd);
    CUDAFreeSafe(instance->cudaD1Index);
    CUDAFreeSafe(instance->cudaD2Index);
    CUDAFreeSafe(instance->cudaD4Index);
    CUDAFreeSafe(instance->cudaH3x3);
    CUDAFreeSafe(instance->cudaH6x6);
    CUDAFreeSafe(instance->cudaH9x9);
    CUDAFreeSafe(instance->cudaH12x12);
    
}
