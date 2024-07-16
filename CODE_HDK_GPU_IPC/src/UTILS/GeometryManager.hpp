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
        cudaSurfVert(nullptr), 
        cudaSurfFace(nullptr), 
        cudaSurfEdge(nullptr),
        cudaBoundaryType(nullptr),
        cudaRestTetPos(nullptr),
        cudaOriginTetPos(nullptr),
        cudaCollisionPairs(nullptr),
        cudaCCDCollisionPairs(nullptr),
        cudaCPNum(nullptr),
        cudaMatIndex(nullptr) {}

    ~GeometryManager() {
        freeCUDA();
    }

    static void totallyfree() {
        if (instance) {
            instance.reset();
        }
    }

private:
    static void freeCUDA() {
        freeCUDASafe(instance->cudaTetPos);
        freeCUDASafe(instance->cudaTetVel);
        freeCUDASafe(instance->cudaTetMass);
        freeCUDASafe(instance->cudaTetElement);
        freeCUDASafe(instance->cudaSurfVert);
        freeCUDASafe(instance->cudaSurfFace);
        freeCUDASafe(instance->cudaSurfEdge);
        freeCUDASafe(instance->cudaBoundaryType);
        freeCUDASafe(instance->cudaRestTetPos);
        freeCUDASafe(instance->cudaOriginTetPos);
        freeCUDASafe(instance->cudaCollisionPairs);
        freeCUDASafe(instance->cudaCCDCollisionPairs);
        freeCUDASafe(instance->cudaCPNum);
        freeCUDASafe(instance->cudaMatIndex);
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

    Eigen::MatrixXd tetPos; // numPoints * 3
    double3* cudaTetPos;

    Eigen::MatrixXd tetVel; // numPoints * 3
    double3* cudaTetVel;

    Eigen::VectorXd tetMass; // numPoints * 1
    double* cudaTetMass;

    std::vector<MATHUTILS::Matrix3x3d> constraints;
    MATHUTILS::Matrix3x3d* cudaConstraints;

    Eigen::MatrixXi tetElement; // numTets * 4
    uint4* cudaTetElement;

    Eigen::MatrixXi triElement;
    uint3* cudaTriElement;

    Eigen::VectorXd tetVolume; // numTets
    double* cudaTetVolume;

    Eigen::VectorXi surfVert; // num SurfPoints
    uint32_t* cudaSurfVert;

    Eigen::MatrixXi surfFace; // num Triangles * 3
    uint3* cudaSurfFace;

    Eigen::MatrixXi surfEdge; // num Edges * 2
    uint2* cudaSurfEdge;

    // Eigen::MatrixXi surfEdgeAdjVert; // num Edges * 2
    // uint2* cudaSurfEdgeAdjVert;

    double3* cudaOriginTetPos;
    double3* cudaRestTetPos;

    int4* cudaCollisionPairs;
    int4* cudaCCDCollisionPairs;
    uint32_t* cudaCPNum;
    int* cudaMatIndex;


public:
    std::unique_ptr<LBVH_F> LBVH_F_ptr;
    std::unique_ptr<LBVH_E> LBVH_E_ptr;

public:
    // physics parameters
    double IPC_dt;
    int MAX_COLLITION_PAIRS_NUM;
    int MAX_CCD_COLLITION_PAIRS_NUM;
    
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



};
