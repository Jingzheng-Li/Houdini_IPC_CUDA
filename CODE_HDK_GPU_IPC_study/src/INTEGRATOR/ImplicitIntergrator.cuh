
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

    int calculateMovingDirection(std::unique_ptr<GeometryManager>& instance, int cpNum, int preconditioner_type = 0);

    double ground_largestFeasibleStepSize(double slackness, double* mqueue);
	double self_largestFeasibleStepSize(double slackness, double* mqueue, int numbers);
	double InjectiveStepSize(double slackness, double errorRate, double* mqueue, uint4* tets);
	double cfl_largestSpeed(double* mqueue);

	double computeEnergy(std::unique_ptr<GeometryManager>& instance);
	double Energy_Add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance);
	void lineSearch(std::unique_ptr<GeometryManager>& instance, double& alpha, const double& cfl_alpha);
	void postLineSearch(std::unique_ptr<GeometryManager>& instance, double alpha);

	void tempMalloc_closeConstraint();
	void tempFree_closeConstraint();

    void updateVelocities(std::unique_ptr<GeometryManager>& instance);

    void computeCloseGroundVal();
	void computeSelfCloseVal();
    double2 minMaxGroundDist();
	double2 minMaxSelfDist();

    void updateDNum(const int& tri_Num, const int& tet_number, const uint32_t* cpNums, const uint32_t* last_cpNums, const int& tri_edge_number);



    void debug_print_Hessian_val(std::unique_ptr<GeometryManager>& instance);




public:
    std::unique_ptr<GeometryManager>& m_instance;
    std::unique_ptr<LBVH_F>& m_bvh_f;
    std::unique_ptr<LBVH_E>& m_bvh_e;
	std::unique_ptr<LBVH_EF>& m_bvh_ef;
    std::unique_ptr<PCGData>& m_pcg_data;

};
