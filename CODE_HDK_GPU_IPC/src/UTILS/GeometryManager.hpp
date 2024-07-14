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

    GeometryManager() : 
        cudaTetPos(nullptr), 
        cudaTetVel(nullptr), 
        cudaTetMass(nullptr), 
        cudaTetInd(nullptr),
        cudaSurfPos(nullptr), 
        cudaSurfInd(nullptr), 
        cudaSurfEdge(nullptr) {}


    // TODO: we prefer not to delete const geometry after one frame, but if instance not be reset, CUDA memory goes wrong
    static void totallyfree() {
        if (instance) {
            freeCUDA();
            freeAllGeometry();
            instance.reset();
        }
    }

private:

    static void freeCUDA() {
        freeCUDASafe(instance->cudaTetPos);
        freeCUDASafe(instance->cudaTetVel);
        freeCUDASafe(instance->cudaTetMass);
        freeCUDASafe(instance->cudaTetInd);
        freeCUDASafe(instance->cudaSurfPos);
        freeCUDASafe(instance->cudaSurfInd);
        freeCUDASafe(instance->cudaSurfEdge);
        freeCUDASafe(instance->cudaBoundaryType);
        freeCUDASafe(instance->cudaRestTetPos);
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

    double3 minCorner;
    double3 maxCorner;

    Eigen::MatrixXd tetPos; // numPoints * 3
    double3* cudaTetPos;

    Eigen::MatrixXd tetVel; // numPoints * 3
    double3* cudaTetVel;

    Eigen::VectorXd tetMass; // numPoints * 1
    double* cudaTetMass;

    Eigen::MatrixXi tetInd; // numTets * 4
    uint4* cudaTetInd;

    Eigen::MatrixXd surfPos; // num SurfPoints * 3
    double3* cudaSurfPos;

    Eigen::MatrixXi surfInd; // num Triangles * 3
    uint3* cudaSurfInd;

    Eigen::MatrixXi surfEdge; // num Edges * 2
    uint2* cudaSurfEdge;

    Eigen::VectorXi boundaryTypies;
    int* cudaBoundaryType;

    double3* cudaOriginTetPos;
    double3* cudaRestTetPos;

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


};
