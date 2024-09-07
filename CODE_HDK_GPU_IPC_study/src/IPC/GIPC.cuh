#pragma once

#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"

#include "PCG/PCGSolver.cuh"
#include "LBVH/LBVH.cuh"


namespace GPUIPC {

void buildCP(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_E>& bvh_e, std::unique_ptr<LBVH_F>& bvh_f);
void buildFullCP(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_E>& bvh_e, std::unique_ptr<LBVH_F>& bvh_f, const double& alpha);
void buildBVH(std::unique_ptr<LBVH_E>& bvh_e, std::unique_ptr<LBVH_F>& bvh_f);
void buildBVH_FULLCCD(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_E>& bvh_e, std::unique_ptr<LBVH_F>& bvh_f, const double& alpha);
void GroundCollisionDetect(std::unique_ptr<GeometryManager>& instance);
bool isIntersected(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_EF>& bvh_ef);
bool checkGroundIntersection(std::unique_ptr<GeometryManager>& instance);	

void calBarrierGradientAndHessian(const double3* _vertexes, const double3* _rest_vertexes, const int4* _collisionPair, double3* _gradient, MATHUTILS::Matrix12x12d* H12x12, MATHUTILS::Matrix9x9d* H9x9, MATHUTILS::Matrix6x6d* H6x6, uint4* D4Index, uint3* D3Index, uint2* D2Index, uint32_t* _cpNum, int* matIndex, double dHat, double Kappa, uint32_t* hCpNum);
void calBarrierGradient(std::unique_ptr<GeometryManager>& instance, double3* _gradient, double mKap);
void calFrictionHessian(std::unique_ptr<GeometryManager>& instance);
void calFrictionGradient(double3* _gradient, std::unique_ptr<GeometryManager>& instance);


void initKappa(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<PCGData>& pcg_data);
void suggestKappa(std::unique_ptr<GeometryManager>& instance, double& kappa);
void upperBoundKappa(std::unique_ptr<GeometryManager>& instance, double& kappa);

// int solve_subIP(std::unique_ptr<GeometryManager>& instance);
void buildFrictionSets(std::unique_ptr<GeometryManager>& instance);
bool Inverse_Physics(std::unique_ptr<GeometryManager>& instance);

void computeInverseHessian(std::unique_ptr<GeometryManager>& instance);
void computeGroundHessian(double3* _gradient);
void computeInverseGradient(std::unique_ptr<GeometryManager>& instance);
void computeFldm(double3* _deltaPos, double3* fldm);

};


