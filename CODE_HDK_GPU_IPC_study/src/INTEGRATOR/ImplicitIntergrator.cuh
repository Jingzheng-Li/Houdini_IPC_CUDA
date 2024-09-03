
#pragma once

#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"

#include "PCG/PCGSolver.cuh"
#include "LBVH/LBVH.cuh"
#include "IPC/GIPC.cuh"

class ImplicitIntegrator {

public:

    ImplicitIntegrator(std::unique_ptr<GeometryManager>& instance);
    ~ImplicitIntegrator();


public:
    std::unique_ptr<GeometryManager>& m_instance;



};
