#pragma once

#include "UTILS/CUDAUtils.hpp"


struct Node {
    uint32_t parent_idx;
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t element_idx;
};


struct AABB {
    double3 upper;
    double3 lower;
    __host__ __device__ AABB();
    __host__ __device__ void combines(const double& x, const double& y, const double& z);
    __host__ __device__  void combines(const double& x, const double& y, const double& z, const double& xx, const double& yy, const double& zz);
    __host__ __device__  void combines(const AABB& aabb);
    __host__ __device__  double3 center();
};


class LBVH {
public:
    LBVH() {};
    ~LBVH() {};


};


class LBVH_F : LBVH {
public:

};

class LBVH_E : LBVH {
public:

};



