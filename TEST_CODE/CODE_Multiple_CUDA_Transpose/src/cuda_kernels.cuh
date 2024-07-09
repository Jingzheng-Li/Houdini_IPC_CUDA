#pragma once

__global__ void transformLissajous(float3* positions, int numPoints, float time);

void transformLissajousCUDA(float3* cudaPositions, int numPoints, float time);

void printDeviceProperties();
