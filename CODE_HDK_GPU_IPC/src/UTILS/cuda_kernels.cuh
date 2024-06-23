#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ void transformLissajous(double3* positions, int numPoints, float time);

void transformLissajousCUDA(double3* cudaTetPos, int numPoints, float time);

void printDeviceProperties();
