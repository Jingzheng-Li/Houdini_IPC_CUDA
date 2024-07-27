#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "CUDAUtils.hpp"
#include "MathUtils.hpp"
#include "MathUtils.cuh"

class Node;
class AABB;
class LBVH_E;
class LBVH_F;
class BHessian;
class PCGData;
class GIPC;
class BHessian;

class GeometryManager {
public:
    GeometryManager();
    ~GeometryManager();

    static void totallyfree();

private:
    static void freeCUDA();

public:

    std::unique_ptr<AABB> AABB_SceneSize_ptr;
    std::unique_ptr<LBVH_F> LBVH_F_ptr;
    std::unique_ptr<LBVH_E> LBVH_E_ptr;
    std::unique_ptr<PCGData> PCGData_ptr;
    std::unique_ptr<BHessian> BH_ptr;
    std::unique_ptr<GIPC> GIPC_ptr;

public:
    static std::unique_ptr<GeometryManager> instance;

    double3 minCorner;
    double3 maxCorner;

    int numVertices;
    int numTetElements;
    int numSurfVerts;
    int numSurfFaces;
    int numSurfEdges;

    int numTriElements;
    int numTriEdges;

    Eigen::MatrixX3d vertPos; // numPoints * 3
    std::vector<double3> vecVertPos;
    double3* cudaVertPos;

    Eigen::MatrixX3d vertVel; // numPoints * 3
    std::vector<double3> vecVertVel;
    double3* cudaVertVel;

    Eigen::VectorXd vertMass; // numPoints * 1
    double* cudaVertMass;

    std::vector<MATHUTILS::Matrix3x3d> constraints;
    MATHUTILS::Matrix3x3d* cudaConstraints;

    std::vector<MATHUTILS::Matrix3x3d> DMInverse;
    MATHUTILS::Matrix3x3d* cudaDmInverses;

    std::vector<MATHUTILS::Matrix2x2d> TriDMInverse;
    MATHUTILS::Matrix2x2d* cudaTriDmInverses;


    Eigen::MatrixX4i tetElement; // numTets * 4
    std::vector<uint4> vectetElement;
    uint4* cudaTetElement;

    Eigen::VectorXd tetVolume; // numTets
    double* cudaTetVolume;
    Eigen::VectorXd triArea; // numTris
    double* cudaTriArea;

    double meanMass;
    double meanVolume;

    Eigen::VectorXi surfVert; // num SurfPoints
    uint32_t* cudaSurfVert;

    Eigen::MatrixX3i surfFace; // num surfTriangles * 3
    uint3* cudaSurfFace;

    Eigen::MatrixX2i surfEdge; // num surfEdges * 2
    uint2* cudaSurfEdge;

    Eigen::MatrixX3i triElement; // numTris * 3
    std::vector<uint3> vectriElement;
    uint3* cudaTriElement;

    Eigen::MatrixX2i triEdges; //
    uint2* cudaTriEdges;

    Eigen::MatrixX2i triEdgeAdjVertex;
    uint2* cudaTriEdgeAdjVertex;


    double3* cudaOriginTetPos;
    double3* cudaRestTetPos;

    int* cudaMatIndex;


    int MAX_COLLITION_PAIRS_NUM;
    int MAX_CCD_COLLITION_PAIRS_NUM;

    int4* cudaCollisionPairs;
    int4* cudaCCDCollisionPairs;
    uint32_t* cudaEnvCollisionPairs;

    uint32_t* cudaCPNum; // collision pair
    uint32_t* cudaGPNum; // ground pair
    uint32_t* cudaCloseGPNum; // close ground pair
    uint32_t* cudaCloseCPNum; // close collision pair

    double* cudaGroundOffset;
    double3* cudaGroundNormal;

    double3* cudaXTilta;
    double3* cudaFb;

    double3* cudaMoveDir;

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
    double bendStiff;
    double Newton_solver_threshold;
    double pcg_threshold;

    double animation_subRate;
    double animation_fullRate;

    uint64_t* cudaMortonCodeHash;
    uint32_t* cudaSortIndex;
    uint32_t* cudaSortMapVertIndex;

    double* cudaTempDouble;
    double3* cudaTempDouble3Mem;
    MATHUTILS::Matrix3x3d* cudaTempMat3x3;

    Eigen::VectorXi boundaryTypies;
    int* cudaBoundaryType;
    int* cudaTempBoundaryType;

    double Kappa;
    double bboxDiagSize2;
    double dTol;
    double minKappaCoef;
    double dHat;
    double relative_dhat;
    double fDhat;

    MATHUTILS::Matrix12x12d* cudaH12x12;
    MATHUTILS::Matrix9x9d* cudaH9x9;
    MATHUTILS::Matrix6x6d* cudaH6x6;
    MATHUTILS::Matrix3x3d* cudaH3x3;
    uint32_t* cudaD1Index; // pIndex, DpeIndex, DptIndex;
    uint3* cudaD3Index;
    uint4* cudaD4Index;
    uint2* cudaD2Index;

    uint32_t softNum;
    double3* cudaTargetVert;
    uint32_t* cudaTargetIndex;

};

