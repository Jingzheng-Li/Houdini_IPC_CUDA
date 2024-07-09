
#include "public_functions.hpp"


float3* CUDAMemoryManager::cudaPositions = nullptr;
Eigen::MatrixXf GeometryManager::positions;

void CUDAMemoryManager::initialize(const Eigen::MatrixXf& positionsMat) {
    int numPoints = positionsMat.rows();
    cudaError_t error = cudaMalloc((void**)&cudaPositions, numPoints * sizeof(float3));
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate cudaPositions: " << cudaGetErrorString(error) << std::endl;
    }
}


void CUDAMemoryManager::copyDataToCUDA(const Eigen::MatrixXf& positionsMat) {
    int numPoints = positionsMat.rows();
    std::vector<float3> tempPositions(numPoints);
    for (int i = 0; i < numPoints; ++i) {
        tempPositions[i] = make_float3(positionsMat(i, 0), positionsMat(i, 1), positionsMat(i, 2));
    }
    cudaError_t error = cudaMemcpy(cudaPositions, tempPositions.data(), numPoints * sizeof(float3), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy data to cudaPositions: " << cudaGetErrorString(error) << std::endl;
    }
}

void CUDAMemoryManager::copyDataFromCUDA(Eigen::MatrixXf& positionsMat) {
    int numPoints = positionsMat.rows();
    std::vector<float3> tempPositions(numPoints);
    cudaError_t error = cudaMemcpy(tempPositions.data(), cudaPositions, numPoints * sizeof(float3), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Failed to copy data from cudaPositions: " << cudaGetErrorString(error) << std::endl;
    }
    for (int i = 0; i < numPoints; ++i) {
        positionsMat(i, 0) = tempPositions[i].x;
        positionsMat(i, 1) = tempPositions[i].y;
        positionsMat(i, 2) = tempPositions[i].z;
    }
}

void CUDAMemoryManager::free() {
    if (cudaPositions) {
        cudaError_t error = cudaFree(cudaPositions);
        if (error != cudaSuccess) {
            std::cerr << "Failed to free cudaPositions: " << cudaGetErrorString(error) << std::endl;
        } else {
            cudaPositions = nullptr;
        }
    }
}


void GeometryManager::initialize(const Eigen::MatrixXf& positionsMat) {
    if (positionsMat.cols() == 3) {
        positions = positionsMat;
    } else {
        std::cerr << "New positions matrix size does not match the initialized size." << std::endl;
    }
}

void GeometryManager::free() {
    if (positions.rows() > 0 || positions.cols() > 0) {
        positions.resize(0, 0);
    }
}




