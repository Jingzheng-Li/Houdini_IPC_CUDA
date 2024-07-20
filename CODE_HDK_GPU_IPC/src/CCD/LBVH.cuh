#pragma once

#include <cstdint>

#include "UTILS/CUDAUtils.hpp"


struct Node {
    uint32_t m_parent_idx;
    uint32_t m_left_idx;
    uint32_t m_right_idx;
    uint32_t m_element_idx;
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

    void GroundCollisionDetect(const double3* _vertexes, const uint32_t* _surfVerts, const double* _groundOffset, const double3* _groundNormal, uint32_t* _environment_collisionPair, uint32_t* _gpNum, double dHat, int numSurfVerts);

public:
    AABB m_scene;
    AABB* mc_boundVolumes; // point to AABB
    AABB* mc_tempLeafBox; // point to leaf AABB
    Node* mc_nodes; // point to BVH Node
    int4* mc_collisionPair; // collision pair
    int4* mc_ccd_collisionPair; // ccd collision pair
    uint64_t* mc_MortonCodeHash; // point to 64bit morton code
    uint32_t* mc_indices; // point to 32bit indices
    uint32_t* mc_cpNum; // point to 32bit number of collision pairs
    uint32_t* mc_flags;
    int* mc_MatIndex;
    int* mc_btype;
    
    double3* mc_vertexes; // save vertices (x,y,z)
    uint32_t m_vert_number; // save number of vertices

public:
    void CUDA_MALLOC_LBVH(const int& number);
    void CUDA_FREE_LBVH();

};


class LBVH_F : public LBVH {
public:
    void init(int* _btype, double3* _mVerts, uint3* _mFaces, uint32_t* _mSurfVert, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& faceNum, const int& vertNum);
    double Construct();
    AABB* getSceneSize();
    double ConstructFullCCD(const double3* moveDir, const double& alpha);
    void SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha);

public:
    uint3* mc_faces;
    uint32_t* mc_surfVerts;
    uint32_t m_face_number;

};

class LBVH_E : public LBVH {
public:
    void init(int* _btype, double3* _mVerts, double3* _rest_vertexes, uint2* _mEdges, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& edgeNum, const int& vertNum);
    double Construct();
    double ConstructFullCCD(const double3* moveDir, const double& alpha);
    void SelfCollitionDetect(double dHat);
    void SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha);

public:
    uint2* mc_edges;
    double3* mc_rest_vertexes;
    uint32_t m_edge_number;

};







