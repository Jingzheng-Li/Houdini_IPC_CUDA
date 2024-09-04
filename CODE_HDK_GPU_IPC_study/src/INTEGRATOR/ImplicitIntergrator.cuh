
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

    bool IPC_Solver();
    int solve_subIP(std::unique_ptr<GeometryManager>& instance);

	void computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance);

    void updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number);

public:
    std::unique_ptr<GeometryManager>& m_instance;
    std::unique_ptr<GIPC>& m_gipc;
    std::unique_ptr<LBVH_F>& m_bvh_f;
    std::unique_ptr<LBVH_E>& m_bvh_e;
	std::unique_ptr<LBVH_EF>& m_bvh_ef;
    std::unique_ptr<PCGData>& m_pcg_data;

};
