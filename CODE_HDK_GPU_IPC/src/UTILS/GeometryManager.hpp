#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

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
        copyToCUDA(instance->tetPos, instance->cudaTetPos);
        copyToCUDA(instance->tetVel, instance->cudaTetVel);
        copyToCUDA(instance->tetMass, instance->cudaTetMass);
    }

    static void copyPrimsDataToCUDA() {
        copyToCUDA(instance->tetInd, instance->cudaTetInd);
    }

    static void copyDetailsDataToCUDA() {
        copyToCUDA(instance->surfPos, instance->cudaSurfPos);
        copyToCUDA(instance->surfInd, instance->cudaSurfInd);
        copyToCUDA(instance->surfEdge, instance->cudaSurfEdge);
    }

    static void copyPointsDataFromCUDA() {
        copyFromCUDA(instance->tetPos, instance->cudaTetPos);
        copyFromCUDA(instance->tetVel, instance->cudaTetVel);
        copyFromCUDA(instance->tetMass, instance->cudaTetMass);
    }

    static void copyPrimsDataFromCUDA() {
        copyFromCUDA(instance->tetInd, instance->cudaTetInd);
    }

    static void copyDetailsDataFromCUDA() {
        copyFromCUDA(instance->surfPos, instance->cudaSurfPos);
        copyFromCUDA(instance->surfInd, instance->cudaSurfInd);
        copyFromCUDA(instance->surfEdge, instance->cudaSurfEdge);
    }

private:
    GeometryManager() : cudaTetPos(nullptr), cudaTetVel(nullptr), cudaTetMass(nullptr), cudaTetInd(nullptr),
                        cudaSurfPos(nullptr), cudaSurfInd(nullptr), cudaSurfEdge(nullptr) {}

    template <typename EigenType, typename CudaType>
    static void copyToCUDA(const EigenType& data, CudaType* cudaData) {
        std::vector<CudaType> temp(data.rows());
        for (int i = 0; i < data.rows(); ++i) {
            if constexpr (std::is_same<CudaType, double3>::value) {
                temp[i] = make_double3(data(i, 0), data(i, 1), data(i, 2));
            } else if constexpr (std::is_same<CudaType, int4>::value) {
                temp[i] = make_int4(data(i, 0), data(i, 1), data(i, 2), data(i, 3));
            } else if constexpr (std::is_same<CudaType, int3>::value) {
                temp[i] = make_int3(data(i, 0), data(i, 1), data(i, 2));
            } else if constexpr (std::is_same<CudaType, int2>::value) {
                temp[i] = make_int2(data(i, 0), data(i, 1));
            } else if constexpr (std::is_same<CudaType, double>::value) {
                temp[i] = data(i);
            }
        }
        CUDA_SAFE_CALL(cudaMemcpy(cudaData, temp.data(), data.rows() * sizeof(CudaType), cudaMemcpyHostToDevice));
    }

    template <typename EigenType, typename CudaType>
    static void copyFromCUDA(EigenType& data, CudaType* cudaData) {
        std::vector<CudaType> temp(data.rows());
        CUDA_SAFE_CALL(cudaMemcpy(temp.data(), cudaData, data.rows() * sizeof(CudaType), cudaMemcpyDeviceToHost));
        for (int i = 0; i < data.rows(); ++i) {
            if constexpr (std::is_same<CudaType, double3>::value) {
                data(i, 0) = temp[i].x;
                data(i, 1) = temp[i].y;
                data(i, 2) = temp[i].z;
            } else if constexpr (std::is_same<CudaType, int4>::value) {
                data(i, 0) = temp[i].x;
                data(i, 1) = temp[i].y;
                data(i, 2) = temp[i].z;
                data(i, 3) = temp[i].w;
            } else if constexpr (std::is_same<CudaType, int3>::value) {
                data(i, 0) = temp[i].x;
                data(i, 1) = temp[i].y;
                data(i, 2) = temp[i].z;
            } else if constexpr (std::is_same<CudaType, int2>::value) {
                data(i, 0) = temp[i].x;
                data(i, 1) = temp[i].y;
            } else if constexpr (std::is_same<CudaType, double>::value) {
                data(i) = temp[i];
            }
        }
    }

    static void initializeCUDAPoints(const Eigen::MatrixXd& tetPosMat, const Eigen::MatrixXd& tetVelMat, const Eigen::VectorXd& tetMassVec) {
        allocateCUDA(instance->cudaTetPos, tetPosMat.rows());
        allocateCUDA(instance->cudaTetVel, tetVelMat.rows());
        allocateCUDA(instance->cudaTetMass, tetMassVec.size());
    }

    static void initializeCUDAPrims(const Eigen::MatrixXi& tetIndMat) {
        allocateCUDA(instance->cudaTetInd, tetIndMat.rows());
    }

    static void initializeCUDASurfs(const Eigen::MatrixXd& surfPosMat, const Eigen::MatrixXi& surfTriMat, const Eigen::MatrixXi& surfEdgeMat) {
        allocateCUDA(instance->cudaSurfPos, surfPosMat.rows());
        allocateCUDA(instance->cudaSurfInd, surfTriMat.rows());
        allocateCUDA(instance->cudaSurfEdge, surfEdgeMat.rows());
    }

    static void allocateCUDA(double3*& cudaData, int rows) {
        CUDA_SAFE_CALL(cudaMalloc((void**)&cudaData, rows * sizeof(double3)));
    }

    static void allocateCUDA(double*& cudaData, int size) {
        CUDA_SAFE_CALL(cudaMalloc((void**)&cudaData, size * sizeof(double)));
    }

    static void allocateCUDA(int4*& cudaData, int rows) {
        CUDA_SAFE_CALL(cudaMalloc((void**)&cudaData, rows * sizeof(int4)));
    }

    static void allocateCUDA(int3*& cudaData, int rows) {
        CUDA_SAFE_CALL(cudaMalloc((void**)&cudaData, rows * sizeof(int3)));
    }

    static void allocateCUDA(int2*& cudaData, int rows) {
        CUDA_SAFE_CALL(cudaMalloc((void**)&cudaData, rows * sizeof(int2)));
    }

    static void freeCUDA() {
        freeCUDA(instance->cudaTetPos);
        freeCUDA(instance->cudaTetVel);
        freeCUDA(instance->cudaTetMass);
        freeCUDA(instance->cudaTetInd);
        freeCUDA(instance->cudaSurfPos);
        freeCUDA(instance->cudaSurfInd);
        freeCUDA(instance->cudaSurfEdge);
    }

    template<typename T>
    static void freeCUDA(T*& cudaData) {
        if (cudaData) {
            CUDA_SAFE_CALL(cudaFree(cudaData));
            cudaData = nullptr;
        }
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
};
