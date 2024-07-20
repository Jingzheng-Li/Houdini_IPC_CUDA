#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "CCD/LBVH.cuh"
#include "CUDAUtils.hpp"
#include "MathUtils.hpp"
#include "MathUtils.cuh"

class GeometryManager {
public:
    GeometryManager() : 
        cudaTetPos(nullptr), 
        cudaTetVel(nullptr), 
        cudaTetMass(nullptr), 
        cudaTetElement(nullptr),
        cudaTriElement(nullptr),
        cudaTetVolume(nullptr),
        cudaSurfVert(nullptr), 
        cudaSurfFace(nullptr), 
        cudaSurfEdge(nullptr),
        cudaBoundaryType(nullptr),
        cudaTempBoundaryType(nullptr),
        cudaConstraints(nullptr),
        cudaRestTetPos(nullptr),
        cudaOriginTetPos(nullptr),
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

    ~GeometryManager() {
        freeCUDA();
    }

    static void totallyfree() {
        if (instance) {
            instance->LBVH_E_ptr->CUDA_FREE_LBVH();
            instance->LBVH_F_ptr->CUDA_FREE_LBVH();

            instance.reset();
        }
    }

private:
    static void freeCUDA() {
        CUDAFreeSafe(instance->cudaTetPos);
        CUDAFreeSafe(instance->cudaTetVel);
        CUDAFreeSafe(instance->cudaTetMass);
        CUDAFreeSafe(instance->cudaTetElement);
        CUDAFreeSafe(instance->cudaTriElement);
        CUDAFreeSafe(instance->cudaTetVolume);
        CUDAFreeSafe(instance->cudaSurfVert);
        CUDAFreeSafe(instance->cudaSurfFace);
        CUDAFreeSafe(instance->cudaSurfEdge);
        CUDAFreeSafe(instance->cudaConstraints);
        CUDAFreeSafe(instance->cudaRestTetPos);
        CUDAFreeSafe(instance->cudaOriginTetPos);
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

    static void freeDynamicGeometry() {
        instance->tetPos.resize(0, 0);
        instance->tetVel.resize(0, 0);
    }

    static void freeAllGeometry() {
        instance->tetPos.resize(0, 0);
        instance->tetVel.resize(0, 0);
        instance->tetMass.resize(0);
        instance->surfVert.resize(0);
        instance->surfFace.resize(0, 0);
        instance->surfEdge.resize(0, 0);
    }

public:
    static std::unique_ptr<GeometryManager> instance;

    double3 minCorner;
    double3 maxCorner;

    int numVertices;
    int numElements;
    int numSurfVerts;
    int numSurfFaces;
    int numSurfEdges;

    Eigen::MatrixX3d tetPos; // numPoints * 3
    double3* cudaTetPos;

    Eigen::MatrixX3d tetVel; // numPoints * 3
    double3* cudaTetVel;

    Eigen::VectorXd tetMass; // numPoints * 1
    double* cudaTetMass;

    std::vector<MATHUTILS::Matrix3x3d> constraints;
    MATHUTILS::Matrix3x3d* cudaConstraints;

    Eigen::MatrixX4i tetElement; // numTets * 4
    uint4* cudaTetElement;

    Eigen::MatrixX3i triElement; // numTris * 3
    uint3* cudaTriElement;

    Eigen::VectorXd tetVolume; // numTets
    double* cudaTetVolume;

    Eigen::VectorXi surfVert; // num SurfPoints
    uint32_t* cudaSurfVert;

    Eigen::MatrixX3i surfFace; // num surfTriangles * 3
    uint3* cudaSurfFace;

    Eigen::MatrixX2i surfEdge; // num surfEdges * 2
    uint2* cudaSurfEdge;

    double meanMass;
    double meanVolume;

    // Eigen::MatrixXi surfEdgeAdjVert; // num Edges * 2
    // uint2* cudaSurfEdgeAdjVert;

    double3* cudaOriginTetPos;
    double3* cudaRestTetPos;

    int* cudaMatIndex;


public:

    std::unique_ptr<AABB> AABB_SceneSize_ptr;
    std::unique_ptr<LBVH_F> LBVH_F_ptr;
    std::unique_ptr<LBVH_E> LBVH_E_ptr;

    int MAX_COLLITION_PAIRS_NUM;
    int MAX_CCD_COLLITION_PAIRS_NUM;

    int4* cudaCollisionPairs;
    int4* cudaCCDCollisionPairs;
    uint32_t* cudaEnvCollisionPairs;

    uint32_t* cudaCPNum; // collision pair
    uint32_t* cudaGPNum; // ground pair

    double* cudaGroundOffset;
    double3* cudaGroundNormal;


public:
    // physics parameters
    double IPC_dt;

    
    double density;
    double YoungModulus;
    double PoissonRate;
    double lengthRateLame;
    double volumeRateLame;
    double lengthRate;
    double volumeRate;
    double frictionRate;
    double clothThickness;
    double clothYoungModulus;
    double stretchStiff;
    double shearStiff;
    double clothDensity;
    double softMotionRate;
    double Newton_solver_threshold;
    double pcg_threshold;

public:

    uint64_t* cudaMortonCodeHash;
    uint32_t* cudaSortIndex;
    uint32_t* cudaSortMapVertIndex;

    double* cudaTempDouble;
    MATHUTILS::Matrix3x3d* cudaDmInverses;
    MATHUTILS::Matrix3x3d* cudaTempMat3x3;
    // MATHUTILS::Matrix2x2d* cudaTriDmInverses;

    Eigen::VectorXi boundaryTypies;
    int* cudaBoundaryType;
    int* cudaTempBoundaryType;

public:
    double bboxDiagSize2;
    double dTol;
    double minKappaCoef;
    double dHat;
    double relative_dhat;
    double fDhat;

    
public:
    // PCG_Solver
    MATHUTILS::Matrix12x12d* cudaH12x12;
	MATHUTILS::Matrix9x9d* cudaH9x9;
	MATHUTILS::Matrix6x6d* cudaH6x6;
	MATHUTILS::Matrix3x3d* cudaH3x3;
	uint32_t* cudaD1Index;//pIndex, DpeIndex, DptIndex;
	uint3* cudaD3Index;
	uint4* cudaD4Index;
	uint2* cudaD2Index;

};
