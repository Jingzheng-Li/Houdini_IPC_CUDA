#pragma once

#include <cstdint>
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
    LBVH();
    ~LBVH();
    void ALLOCATE_BVH_CUDA(const int& number);
    void FREE_BVH_CUDA();

public:
    AABB scene;
    AABB* _boundVolumes; // point to AABB
    AABB* _tempLeafBox; // point to leaf AABB
    Node* _nodes; // point to BVH Node
    int4* _collisionPair; // collision pair
    int4* _ccd_collisionPair; // ccd collision pair
    uint64_t* _MortonHash; // point to 64bit morton code
    uint32_t* _indices; // point to 32bit indices
    uint32_t* _cpNum; // point to 32bit number of collision pairs
    uint32_t* _flags;
    int* _MatIndex;
    int* _btype;
    
    double3* _vertexes; // save vertices (x,y,z)
    uint32_t vert_number; // save number of vertices

};


class LBVH_F : LBVH {
public:
    void init(int* _btype, double3* _mVerts, uint3* _mFaces, uint32_t* _mSurfVert, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& faceNum, const int& vertNum);
    double Construct();
    AABB* getSceneSize();
    double ConstructFullCCD(const double3* moveDir, const double& alpha);
    void SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha);

public:
    uint32_t face_number;
    uint3* _faces;
    uint32_t* _surfVerts;

};

class LBVH_E : LBVH {
public:
    void init(int* _btype, double3* _mVerts, double3* _rest_vertexes, uint2* _mEdges, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& edgeNum, const int& vertNum);
    double Construct();
    double ConstructFullCCD(const double3* moveDir, const double& alpha);
    void SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha);

public:
    double3* _rest_vertexes;
    uint32_t edge_number;
    uint2* _edges;

};





