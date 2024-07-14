#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

// #include "CCD/LBVH.cuh"
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
    std::vector<int> boundaryTypies;


};
