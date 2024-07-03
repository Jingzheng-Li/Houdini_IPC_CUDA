#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include "CCD/LBVH.cuh"
#include "CUDAUtils.hpp"

class GeometryManager {
public:
    static void initializePoints(const Eigen::MatrixXd& tetPosMat, const Eigen::MatrixXd& tetVelMat, const Eigen::VectorXd& tetMassVec) {
        if (!instance) {
            instance = std::unique_ptr<GeometryManager>(new GeometryManager());
        }
        instance->tetPos = tetPosMat;
        instance->tetVel = tetVelMat;
        instance->tetMass = tetMassVec;
        initializeCUDAPoints(tetPosMat, tetVelMat, tetMassVec);
    }

    static void initializePrims(const Eigen::MatrixXi& tetIndMat) {
        if (!instance) {
            instance = std::unique_ptr<GeometryManager>(new GeometryManager());
        }
        instance->tetInd = tetIndMat;
        initializeCUDAPrims(tetIndMat);
    }

    static void initializeSurfs(const Eigen::MatrixXd& surfPosMat, const Eigen::MatrixXi& surfTriMat, const Eigen::MatrixXi& surfEdgeMat) {
        if (!instance) {
            instance = std::unique_ptr<GeometryManager>(new GeometryManager());
        }
        instance->surfPos = surfPosMat;
        instance->surfInd = surfTriMat;
        instance->surfEdge = surfEdgeMat;
        initializeCUDASurfs(surfPosMat, surfTriMat, surfEdgeMat);
    }

    // TODO: we prefer not to delete const geometry after one frame, but if instance not be reset, CUDA memory goes wrong
    static void free() {
        if (instance) {
            freeCUDA();
            freeDynamicGeometry();
            instance.reset();
        }
    }

    static void totallyfree() {
        if (instance) {
            freeCUDA();
            freeAllGeometry();
            instance.reset();
        }
    }

    static void copyPointsDataToCUDA() {
        copyToCUDASafe(instance->tetPos, instance->cudaTetPos);
        copyToCUDASafe(instance->tetVel, instance->cudaTetVel);
        copyToCUDASafe(instance->tetMass, instance->cudaTetMass);
    }

    static void copyPrimsDataToCUDA() {
        copyToCUDASafe(instance->tetInd, instance->cudaTetInd);
    }

    static void copyDetailsDataToCUDA() {
        copyToCUDASafe(instance->surfPos, instance->cudaSurfPos);
        copyToCUDASafe(instance->surfInd, instance->cudaSurfInd);
        copyToCUDASafe(instance->surfEdge, instance->cudaSurfEdge);
    }

    static void copyPointsDataFromCUDA() {
        copyFromCUDASafe(instance->tetPos, instance->cudaTetPos);
        copyFromCUDASafe(instance->tetVel, instance->cudaTetVel);
        copyFromCUDASafe(instance->tetMass, instance->cudaTetMass);
    }

    static void copyPrimsDataFromCUDA() {
        copyFromCUDASafe(instance->tetInd, instance->cudaTetInd);
    }

    static void copyDetailsDataFromCUDA() {
        copyFromCUDASafe(instance->surfPos, instance->cudaSurfPos);
        copyFromCUDASafe(instance->surfInd, instance->cudaSurfInd);
        copyFromCUDASafe(instance->surfEdge, instance->cudaSurfEdge);
    }

private:
    GeometryManager() : 
        cudaTetPos(nullptr), 
        cudaTetVel(nullptr), 
        cudaTetMass(nullptr), 
        cudaTetInd(nullptr),
        cudaSurfPos(nullptr), 
        cudaSurfInd(nullptr), 
        cudaSurfEdge(nullptr) {}

    static void initializeCUDAPoints(const Eigen::MatrixXd& tetPosMat, const Eigen::MatrixXd& tetVelMat, const Eigen::VectorXd& tetMassVec) {
        allocateCUDASafe(instance->cudaTetPos, tetPosMat.rows());
        allocateCUDASafe(instance->cudaTetVel, tetVelMat.rows());
        allocateCUDASafe(instance->cudaTetMass, tetMassVec.size());
    }

    static void initializeCUDAPrims(const Eigen::MatrixXi& tetIndMat) {
        allocateCUDASafe(instance->cudaTetInd, tetIndMat.rows());
    }

    static void initializeCUDASurfs(const Eigen::MatrixXd& surfPosMat, const Eigen::MatrixXi& surfTriMat, const Eigen::MatrixXi& surfEdgeMat) {
        allocateCUDASafe(instance->cudaSurfPos, surfPosMat.rows());
        allocateCUDASafe(instance->cudaSurfInd, surfTriMat.rows());
        allocateCUDASafe(instance->cudaSurfEdge, surfEdgeMat.rows());
    }

    static void freeCUDA() {
        freeCUDASafe(instance->cudaTetPos);
        freeCUDASafe(instance->cudaTetVel);
        freeCUDASafe(instance->cudaTetMass);
        freeCUDASafe(instance->cudaTetInd);
        freeCUDASafe(instance->cudaSurfPos);
        freeCUDASafe(instance->cudaSurfInd);
        freeCUDASafe(instance->cudaSurfEdge);
    }

    static void freeDynamicGeometry() {
        instance->tetPos.resize(0, 0);
        instance->tetVel.resize(0, 0);
    }

    static void freeAllGeometry() {
        instance->tetPos.resize(0, 0);
        instance->tetVel.resize(0, 0);
        instance->tetMass.resize(0);
        instance->surfPos.resize(0, 0);
        instance->surfInd.resize(0, 0);
        instance->surfEdge.resize(0, 0);
    }


public:
    static std::unique_ptr<GeometryManager> instance;

    Eigen::Vector3d minCorner;
    Eigen::Vector3d maxCorner;

    Eigen::MatrixXd tetPos; // numPoints * 3
    Eigen::MatrixXd tetVel; // numPoints * 3
    Eigen::VectorXd tetMass; // numPoints * 1
    Eigen::MatrixXi tetInd; // numTets * 4

    Eigen::MatrixXd surfPos; // num SurfPoints * 3
    Eigen::MatrixXi surfInd; // num Triangles * 3
    Eigen::MatrixXi surfEdge; // num Edges * 2

    double3* cudaTetPos;
    double3* cudaTetVel;
    double* cudaTetMass;
    int4* cudaTetInd;

    double3* cudaSurfPos;
    int3* cudaSurfInd;
    int2* cudaSurfEdge;

public:
    
    

};
