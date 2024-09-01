#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "CUDAUtils.hpp"
#include "MathUtils.hpp"
#include "MathUtils.cuh"

struct Node;
class AABB;
class LBVH_E;
class LBVH_F;
class LBVH_EF;
class BHessian;
class MASPreconditioner;
class PCGData;
class GIPC;
class ImplicitIntegrator;

class GeometryManager {
public:
    GeometryManager();
    ~GeometryManager();
    void freeGeometryManager();

private:
    static void freeCUDA();
    static void freeCUDAptr();

public:

    std::unique_ptr<LBVH_F> LBVH_F_ptr;
    std::unique_ptr<LBVH_E> LBVH_E_ptr;
    std::unique_ptr<LBVH_EF> LBVH_EF_ptr;
    std::unique_ptr<BHessian> BH_ptr;
    std::unique_ptr<MASPreconditioner> MAS_ptr;
    std::unique_ptr<PCGData> PCGData_ptr;
    std::unique_ptr<GIPC> GIPC_ptr;
    std::unique_ptr<ImplicitIntegrator> Integrator_ptr;

public:
    static std::unique_ptr<GeometryManager> instance;

    int numVertices;
    int numTetElements;
    int numSurfVerts;
    int numSurfFaces;
    int numSurfEdges;
    int numSIMVertPos;

    int numTriElements;
    int numTriEdges;

    Eigen::MatrixX3d vertPos; // numPoints * 3
    double3* cudaVertPos;

    Eigen::MatrixX3d vertVel; // numPoints * 3
    double3* cudaVertVel;

    Eigen::VectorXd vertMass; // numPoints * 1
    double* cudaVertMass;

    Eigen::MatrixX4i tetElement; // numTets * 4
    uint4* cudaTetElement;

    Eigen::VectorXd tetVolume; // numTets
    double* cudaTetVolume;

    Eigen::VectorXd triArea; // numTris
    double* cudaTriArea;

    std::vector<MATHUTILS::Matrix3x3d> DMInverse;
    MATHUTILS::Matrix3x3d* cudaTetDmInverses;

    std::vector<MATHUTILS::Matrix2x2d> TriDMInverse;
    MATHUTILS::Matrix2x2d* cudaTriDmInverses;



    double meanMass;
    double meanVolume;

    Eigen::VectorXi surfVert; // num SurfPoints
    uint32_t* cudaSurfVert;
    Eigen::MatrixX3i surfFace; // num surfTriangles * 3
    uint3* cudaSurfFace;
    Eigen::MatrixX2i surfEdge; // num surfEdges * 2
    uint2* cudaSurfEdge;

    Eigen::MatrixX3i triElement; // numTris * 3
    // std::vector<uint3> vectriElement;
    uint3* cudaTriElement;
    Eigen::MatrixX2i triEdges; // only work for cloth bending
    uint2* cudaTriEdges;
    Eigen::MatrixX2i triEdgeAdjVertex; // only work for cloth bending
    uint2* cudaTriEdgeAdjVertex;

    Eigen::MatrixX3d collisionVertPos;
    Eigen::MatrixX3i collisionSurfFace;
    Eigen::VectorXi collisionBoundaryType;


    uint32_t numSoftConstraints;
    double3* cudaTargetVertPos;
    uint32_t* cudaTargetIndex;
    // std::vector<MATHUTILS::Matrix3x3d> constraintsMat;
    Eigen::Matrix<MATHUTILS::Matrix3x3d, Eigen::Dynamic, 1> constraintsMat;
    MATHUTILS::Matrix3x3d* cudaConstraintsMat;
    Eigen::VectorXi targetIndex;
    Eigen::MatrixX3d targetVertPos;


    double3* cudaOriginVertPos;
    double3* cudaRestVertPos;

    int* cudaMatIndex;


    int MAX_COLLITION_PAIRS_NUM;
    int MAX_CCD_COLLITION_PAIRS_NUM;

    int4* cudaCollisionPairs;
    int4* cudaCCDCollisionPairs;
    uint32_t* cudaEnvCollisionPairs;

    // cpnum[0] = all collision pairs
    // cpnum[1] = all specific cps (pp, pe...)
    // cpnum[2] = cps of pp
    // cpnum[3] = cps of pe
    // cpnum[4] = cps of pt
    uint32_t* cudaCPNum; // collision pair [5]
    uint32_t* cudaGPNum; // ground pair
    uint32_t* cudaCloseGPNum; // close ground pair
    uint32_t* cudaCloseCPNum; // close collision pair

    double* cudaGroundOffset;
    double3* cudaGroundNormal;

    double3* cudaXTilta;
    double3* cudaFb;

    double3* cudaMoveDir;

    double IPC_dt;

    int collision_detection_buff_scale;

    int precondType;
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

    double ground_left_offset;
    double ground_right_offset;
    double ground_near_offset;
    double ground_far_offset;
    double ground_bottom_offset;
    double3 gravityforce;

    bool animation;
    double animation_subRate;
    double animation_fullRate;

    uint64_t* cudaMortonCodeHash;
    uint32_t* cudaSortIndex;
    uint32_t* cudaSortMapVertIndex;

    double* cudaTempDouble;
    double3* cudaTempDouble3Mem;
    MATHUTILS::Matrix3x3d* cudaTempMat3x3;

    // boundary 0: particle into FEM/CCD
    // boundary 1: 
    // boundary 2: particle into CCD, act as dynamic geometry
    // boundary 3: particle into CCD, act as static geometry
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


    uint32_t cpNum[5];
	uint32_t ccdCpNum;
	uint32_t gpNum;
	uint32_t closeCpNum;
	uint32_t closeGpNum;
	uint32_t gpNumLast;
	uint32_t cpNumLast[5];

};

