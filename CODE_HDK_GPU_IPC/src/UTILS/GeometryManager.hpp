#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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

    static void free() {
        if (instance) {
            freeCUDA();
            freeEigen();
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

    static void copyPointsDataFromCUDA() {
        copyFromCUDA(instance->tetPos, instance->cudaTetPos);
        copyFromCUDA(instance->tetVel, instance->cudaTetVel);
        copyFromCUDA(instance->tetMass, instance->cudaTetMass);
    }

    static void copyPrimsDataFromCUDA() {
        copyFromCUDA(instance->tetInd, instance->cudaTetInd);
    }

private:
    GeometryManager() : cudaTetPos(nullptr), cudaTetVel(nullptr), cudaTetMass(nullptr), cudaTetInd(nullptr) {}

    static void copyToCUDA(const Eigen::MatrixXd& data, double3* cudaData) {
        std::vector<double3> temp(data.rows());
        for (int i = 0; i < data.rows(); ++i) {
            temp[i] = make_double3(data(i, 0), data(i, 1), data(i, 2));
        }
        cudaError_t error = cudaMemcpy(cudaData, temp.data(), data.rows() * sizeof(double3), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy data to CUDA: " << cudaGetErrorString(error) << std::endl;
        }
    }

    static void copyToCUDA(const Eigen::VectorXd& data, double* cudaData) {
        cudaError_t error = cudaMemcpy(cudaData, data.data(), data.rows() * sizeof(double), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy data to CUDA: " << cudaGetErrorString(error) << std::endl;
        }
    }

    static void copyToCUDA(const Eigen::MatrixXi& data, int4* cudaData) {
        std::vector<int4> temp(data.rows());
        for (int i = 0; i < data.rows(); ++i) {
            temp[i] = make_int4(data(i, 0), data(i, 1), data(i, 2), data(i, 3));
        }
        cudaError_t error = cudaMemcpy(cudaData, temp.data(), data.rows() * sizeof(int4), cudaMemcpyHostToDevice);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy data to CUDA: " << cudaGetErrorString(error) << std::endl;
        }
    }

    static void copyFromCUDA(Eigen::MatrixXd& data, double3* cudaData) {
        std::vector<double3> temp(data.rows());
        cudaError_t error = cudaMemcpy(temp.data(), cudaData, data.rows() * sizeof(double3), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy data from CUDA: " << cudaGetErrorString(error) << std::endl;
        }
        for (int i = 0; i < data.rows(); ++i) {
            data(i, 0) = temp[i].x;
            data(i, 1) = temp[i].y;
            data(i, 2) = temp[i].z;
        }
    }

    static void copyFromCUDA(Eigen::VectorXd& data, double* cudaData) {
        cudaError_t error = cudaMemcpy(data.data(), cudaData, data.rows() * sizeof(double), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy data from CUDA: " << cudaGetErrorString(error) << std::endl;
        }
    }

    static void copyFromCUDA(Eigen::MatrixXi& data, int4* cudaData) {
        std::vector<int4> temp(data.rows());
        cudaError_t error = cudaMemcpy(temp.data(), cudaData, data.rows() * sizeof(int4), cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) {
            std::cerr << "Failed to copy data from CUDA: " << cudaGetErrorString(error) << std::endl;
        }
        for (int i = 0; i < data.rows(); ++i) {
            data(i, 0) = temp[i].x;
            data(i, 1) = temp[i].y;
            data(i, 2) = temp[i].z;
            data(i, 3) = temp[i].w;
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

    static void allocateCUDA(double3*& cudaData, int rows) {
        cudaError_t error = cudaMalloc((void**)&cudaData, rows * sizeof(double3));
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate CUDA memory: " << cudaGetErrorString(error) << std::endl;
        }
    }

    static void allocateCUDA(double*& cudaData, int size) {
        cudaError_t error = cudaMalloc((void**)&cudaData, size * sizeof(double));
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate CUDA memory: " << cudaGetErrorString(error) << std::endl;
        }
    }

    static void allocateCUDA(int4*& cudaData, int rows) {
        cudaError_t error = cudaMalloc((void**)&cudaData, rows * sizeof(int4));
        if (error != cudaSuccess) {
            std::cerr << "Failed to allocate CUDA memory: " << cudaGetErrorString(error) << std::endl;
        }
    }

    static void freeCUDA() {
        freeCUDA(instance->cudaTetPos);
        freeCUDA(instance->cudaTetVel);
        freeCUDA(instance->cudaTetMass);
        freeCUDA(instance->cudaTetInd);
    }

    template<typename T>
    static void freeCUDA(T* cudaData) {
        if (cudaData) {
            cudaError_t error = cudaFree(cudaData);
            if (error != cudaSuccess) {
                std::cerr << "Failed to free CUDA memory: " << cudaGetErrorString(error) << std::endl;
            }
        }
    }

    static void freeEigen() {
        instance->tetPos.resize(0, 0);
        instance->tetVel.resize(0, 0);
        instance->tetMass.resize(0);
        instance->tetInd.resize(0, 0);
    }

public:
    static std::unique_ptr<GeometryManager> instance;

    Eigen::Vector3d minTCorner;
    Eigen::Vector3d maxTCorner;

    Eigen::MatrixXd tetPos; // numPoints * 3
    Eigen::MatrixXd tetVel; // numPoints * 3
    Eigen::VectorXd tetMass; // numPoints * 1
    Eigen::MatrixXi tetInd; // numTets * 4

    double3* cudaTetPos;
    double3* cudaTetVel;
    double* cudaTetMass;
    int4* cudaTetInd;
};

