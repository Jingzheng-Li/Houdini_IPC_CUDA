
#include "GeometryManager.hpp"

std::unique_ptr<GeometryManager> GeometryManager::instance = nullptr;

GeometryManager::GeometryManager() :
    cudaVertPos(nullptr), 
    cudaVertVel(nullptr), 
    cudaVertMass(nullptr), 
    cudaTetElement(nullptr), 
    cudaTetVolume(nullptr), 
    cudaTriArea(nullptr), 
    cudaTetDmInverses(nullptr), 
    cudaTriDmInverses(nullptr), 
    cudaSurfVert(nullptr), 
    cudaSurfFace(nullptr), 
    cudaSurfEdge(nullptr), 
    cudaTriElement(nullptr), 
    cudaTriBendEdges(nullptr), 
    cudaTriBendVerts(nullptr), 
    cudaBoundTargetVertPos(nullptr), 
    cudaBoundTargetIndex(nullptr), 
    cudaSoftTargetIndex(nullptr), 
    cudaSoftTargetVertPos(nullptr), 
    cudaConstraintsMat(nullptr), 
    cudaOriginVertPos(nullptr), 
    cudaRestVertPos(nullptr), 
    cudaCollisionPairs(nullptr), 
    cudaCCDCollisionPairs(nullptr), 
    cudaEnvCollisionPairs(nullptr), 
    cudaCPNum(nullptr), 
    cudaGPNum(nullptr), 
    cudaCloseCPNum(nullptr), 
    cudaCloseGPNum(nullptr), 
    cudaGroundOffset(nullptr), 
    cudaGroundNormal(nullptr), 
    cudaXTilta(nullptr), 
    cudaFb(nullptr), 
    cudaMoveDir(nullptr), 
    cudaMortonCodeHash(nullptr), 
    cudaSortIndex(nullptr), 
    cudaSortMapVertIndex(nullptr), 
    cudaTempScalar(nullptr), 
    cudaTempScalar3Mem(nullptr), 
    cudaTempMat3x3(nullptr), 
    cudaBoundaryType(nullptr), 
    cudaTempBoundaryType(nullptr), 
    cudaCloseConstraintID(nullptr), 
    cudaCloseConstraintVal(nullptr), 
    cudaCloseMConstraintID(nullptr), 
    cudaCloseMConstraintVal(nullptr), 
    cudaD1Index(nullptr), 
    cudaD2Index(nullptr), 
    cudaD3Index(nullptr), 
    cudaD4Index(nullptr), 
    cudaH3x3(nullptr), 
    cudaH6x6(nullptr), 
    cudaH9x9(nullptr), 
    cudaH12x12(nullptr), 
    cudaLambdaLastHScalar(nullptr), 
    cudaDistCoord(nullptr), 
    cudaTanBasis(nullptr), 
    cudaCollisionPairsLastH(nullptr), 
    cudaMatIndexLast(nullptr), 
    cudaLambdaLastHScalarGd(nullptr), 
    cudaCollisionPairsLastHGd(nullptr) 
    {}



void GeometryManager::freeCUDAMem() {
    CUDAFreeSafe(instance->cudaVertPos);
    CUDAFreeSafe(instance->cudaVertVel);
    CUDAFreeSafe(instance->cudaVertMass);
    CUDAFreeSafe(instance->cudaTetElement);
    CUDAFreeSafe(instance->cudaTetVolume);
    CUDAFreeSafe(instance->cudaTriArea);
    CUDAFreeSafe(instance->cudaTetDmInverses);
    CUDAFreeSafe(instance->cudaTriDmInverses);
    CUDAFreeSafe(instance->cudaSurfVert);
    CUDAFreeSafe(instance->cudaSurfFace);
    CUDAFreeSafe(instance->cudaSurfEdge);
    CUDAFreeSafe(instance->cudaTriElement);
    CUDAFreeSafe(instance->cudaTriBendEdges);
    CUDAFreeSafe(instance->cudaTriBendVerts);
    CUDAFreeSafe(instance->cudaBoundTargetVertPos);
    CUDAFreeSafe(instance->cudaBoundTargetIndex);
    CUDAFreeSafe(instance->cudaSoftTargetIndex);
    CUDAFreeSafe(instance->cudaSoftTargetVertPos);
    CUDAFreeSafe(instance->cudaConstraintsMat);
    CUDAFreeSafe(instance->cudaOriginVertPos);
    CUDAFreeSafe(instance->cudaRestVertPos);
    CUDAFreeSafe(instance->cudaCollisionPairs);
    CUDAFreeSafe(instance->cudaCCDCollisionPairs);
    CUDAFreeSafe(instance->cudaEnvCollisionPairs);
    CUDAFreeSafe(instance->cudaCPNum);
    CUDAFreeSafe(instance->cudaGPNum);
    CUDAFreeSafe(instance->cudaCloseCPNum);
    CUDAFreeSafe(instance->cudaCloseGPNum);
    CUDAFreeSafe(instance->cudaGroundOffset);
    CUDAFreeSafe(instance->cudaGroundNormal);
    CUDAFreeSafe(instance->cudaXTilta);
    CUDAFreeSafe(instance->cudaFb);
    CUDAFreeSafe(instance->cudaMoveDir);
    CUDAFreeSafe(instance->cudaMortonCodeHash);
    CUDAFreeSafe(instance->cudaSortIndex);
    CUDAFreeSafe(instance->cudaSortMapVertIndex);
    CUDAFreeSafe(instance->cudaTempScalar);
    CUDAFreeSafe(instance->cudaTempScalar3Mem);
    CUDAFreeSafe(instance->cudaTempMat3x3);
    CUDAFreeSafe(instance->cudaBoundaryType);
    CUDAFreeSafe(instance->cudaTempBoundaryType);
    CUDAFreeSafe(instance->cudaCloseConstraintID);
    CUDAFreeSafe(instance->cudaCloseConstraintVal);
    CUDAFreeSafe(instance->cudaCloseMConstraintID);
    CUDAFreeSafe(instance->cudaCloseMConstraintVal);
    CUDAFreeSafe(instance->cudaD1Index);
    CUDAFreeSafe(instance->cudaD2Index);
    CUDAFreeSafe(instance->cudaD3Index);
    CUDAFreeSafe(instance->cudaD4Index);
    CUDAFreeSafe(instance->cudaH3x3);
    CUDAFreeSafe(instance->cudaH6x6);
    CUDAFreeSafe(instance->cudaH9x9);
    CUDAFreeSafe(instance->cudaH12x12);
    CUDAFreeSafe(instance->cudaLambdaLastHScalar);
    CUDAFreeSafe(instance->cudaDistCoord);
    CUDAFreeSafe(instance->cudaTanBasis);
    CUDAFreeSafe(instance->cudaCollisionPairsLastH);
    CUDAFreeSafe(instance->cudaMatIndexLast);
    CUDAFreeSafe(instance->cudaLambdaLastHScalarGd);
    CUDAFreeSafe(instance->cudaCollisionPairsLastHGd);
    
}



GeometryManager::~GeometryManager() {
    std::cout << "deconstruct GeometryManager" << std::endl;
    if (instance) {
        std::cout << "~GeometryManager freeGeometryManager" << std::endl;
        freeGeometryManager();
    }
}

void GeometryManager::freeGeometryManager() {
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    freeCUDAPtr();
    freeCUDAMem();
    instance.reset();    
}


void GeometryManager::freeCUDAPtr() {

}

