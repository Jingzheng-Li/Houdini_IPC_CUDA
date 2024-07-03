
#include "LBVH.cuh"

#include <cmath>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include "UTILS/MathUtils.cuh"

template<class T>
__device__ __host__ 
inline T __m_min(T a, T b) {
    return a < b ? a : b;
}

template <class T>
__device__ __host__ 
inline T __m_max(T a, T b) {
    return a < b ? b : a;
}

// merge two AABB boundaries
__device__ __host__ 
inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept {
    AABB merged;
    merged.upper.x = __m_max(lhs.upper.x, rhs.upper.x);
    merged.upper.y = __m_max(lhs.upper.y, rhs.upper.y);
    merged.upper.z = __m_max(lhs.upper.z, rhs.upper.z);
    merged.lower.x = __m_min(lhs.lower.x, rhs.lower.x);
    merged.lower.y = __m_min(lhs.lower.y, rhs.lower.y);
    merged.lower.z = __m_min(lhs.lower.z, rhs.lower.z);
    return merged;
}

// calculate whether AABB are overlapped
__device__ __host__ 
inline bool overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept {
    if ((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL) return false;
    if ((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL) return false;
    if ((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL) return false;
    return true;
}

// calculate the centroid of AABB boundaries
__device__ __host__ 
inline double3 centroid(const AABB& box) noexcept {
    return make_double3(
        (box.upper.x + box.lower.x) * 0.5,
        (box.upper.y + box.lower.y) * 0.5,
        (box.upper.z + box.lower.z) * 0.5
    );
}

// calculate the Morton code of a int number
__device__ __host__ 
inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// calculate the Morton code a point
__device__ __host__ 
inline std::uint32_t morton_code(double x, double y, double z, double resolution = 1024.0) noexcept {
    x = __m_min(__m_max(x * resolution, 0.0), resolution - 1.0);
    y = __m_min(__m_max(y * resolution, 0.0), resolution - 1.0);
    z = __m_min(__m_max(z * resolution, 0.0), resolution - 1.0);

    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(z));

    return (xx << 2) + (yy << 1) + zz;
}

// calculate max common upper bits
__device__ inline 
int common_upper_bits(const unsigned long long lhs, const unsigned long long rhs) noexcept {
    return __clzll(lhs ^ rhs);
}

// calculate the range of a tree node can range for leaves node
__device__ inline 
uint2 determine_range(const uint64_t* node_code, const unsigned int num_leaves, unsigned int idx) {
    if (idx == 0) {
        return make_uint2(0, num_leaves - 1);
    }

    const uint64_t self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx - 1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx + 1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    const int delta_min = __m_min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if (0 <= i_tmp && i_tmp < num_leaves) {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while (delta > delta_min) {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    int l = 0;
    int t = l_max >> 1;
    while (t > 0) {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if (delta > delta_min) {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if (d < 0) {
        unsigned int temp_jdx = jdx;
        jdx = idx;
        idx = temp_jdx;
    }
    return make_uint2(idx, jdx);
}


// calculate the split of of a BVH tree node
__device__ 
inline unsigned int find_split(const uint64_t* node_code, const unsigned int num_leaves, const unsigned int first, const unsigned int last) noexcept {
    const uint64_t first_code = node_code[first];
    const uint64_t last_code = node_code[last];
    if (first_code == last_code) {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    int split = first;
    int stride = last - first;
    do {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last) {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node) {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}


__device__ __host__
void AABB::combines(const double& x, const double&y, const double& z) {
    lower = make_double3(__m_min(lower.x, x), __m_min(lower.y, y), __m_min(lower.z, z));
    upper = make_double3(__m_max(upper.x, x), __m_max(upper.y, y), __m_max(upper.z, z));
}

__device__ __host__
void AABB::combines(const double& x, const double& y, const double& z, const double& xx, const double& yy, const double& zz) {
    lower = make_double3(__m_min(lower.x, x), __m_min(lower.y, y), __m_min(lower.z, z));
    upper = make_double3(__m_max(upper.x, xx), __m_max(upper.y, yy), __m_max(upper.z, zz));
}

__device__ __host__
void AABB::combines(const AABB& aabb) {
    lower = make_double3(__m_min(lower.x, aabb.lower.x), __m_min(lower.y, aabb.lower.y), __m_min(lower.z, aabb.lower.z));
    upper = make_double3(__m_max(upper.x, aabb.upper.x), __m_max(upper.y, aabb.upper.y), __m_max(upper.z, aabb.upper.z));
}

__device__ __host__
double3 AABB::center() {
    return make_double3((upper.x + lower.x) * 0.5, (upper.y + lower.y) * 0.5, (upper.z + lower.z) * 0.5);
}

__device__ __host__
AABB::AABB() {
    lower = make_double3(1e32, 1e32, 1e32);
    upper = make_double3(-1e32, -1e32, -1e32);
}


__device__
void _d_PP(const double3& v0, const double3& v1, double& d) {
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1));
}

__device__
void _d_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d) {
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v2, v1), __GEIGEN__::__minus(v3, v1));
    double3 test = __GEIGEN__::__minus(v0, v1);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v0, v1), b);//(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__
void _d_PE(const double3& v0, const double3& v1, const double3& v2, double& d) {
    d = __GEIGEN__::__squaredNorm(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0))) / __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v1));
}

__device__
void _d_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d) {
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v3, v2));//(v1 - v0).cross(v3 - v2);
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}


__device__
void _d_EEParallel(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d) {
    double3 b = __GEIGEN__::__v_vec_cross(__GEIGEN__::__v_vec_cross(__GEIGEN__::__minus(v1, v0), __GEIGEN__::__minus(v2, v0)), __GEIGEN__::__minus(v1, v0));
    double aTb = __GEIGEN__::__v_vec_dot(__GEIGEN__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __GEIGEN__::__squaredNorm(b);
}

__device__
double _compute_epx(const double3& v0, const double3& v1, const double3& v2, const double3& v3) {
    return 1e-3 * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v0, v1)) * __GEIGEN__::__squaredNorm(__GEIGEN__::__minus(v2, v3));
}

LBVH::LBVH() {
    std::cout << "const LBVH" << std::endl;
}

LBVH::~LBVH() {
    std::cout << "decon LBVH" << std::endl;
    FREE_BVH_CUDA();
}

void LBVH_F::init(int* _mbtype, double3* _mVerts, uint3* _mFaces, uint32_t* _mSurfVert, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& faceNum, const int& vertNum) {
    _faces = _mFaces;
    _surfVerts = _mSurfVert;
    _vertexes = _mVerts;
    _collisionPair = _mCollisonPairs;
    _ccd_collisionPair = _ccd_mCollisonPairs;
    _cpNum = _mcpNum;
    _MatIndex = _mMatIndex;
    face_number = faceNum;
    vert_number = vertNum;
    _btype = _mbtype;

}

void LBVH_E::init(int* _mbtype, double3* _mVerts, double3* _mRest_vertexes, uint2* _mEdges, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& edgeNum, const int& vertNum) {

}

double LBVH_F::Construct() {

}

double LBVH_F::ConstructFullCCD(const double3* moveDir, const double& alpha) {

}

double LBVH_E::Construct() {

}

double LBVH_E::ConstructFullCCD(const double3* moveDir, const double& alpha) {

}

void LBVH_F::SelfCollitionDetect(double dHat) {

}

void LBVH_E::SelfCollitionDetect(double dHat) {

}

void LBVH_F::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha) {

}

void LBVH_E::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha) {

}

AABB* LBVH_F::getSceneSize() {

}






///////////////////
// CUDA MALLOC
///////////////////


void LBVH::ALLOCATE_BVH_CUDA(const int& number) {
    CUDA_SAFE_CALL(cudaMalloc((void**)&_indices, (number) * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_MortonHash, (number) * sizeof(uint64_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_nodes, (2 * number - 1) * sizeof(Node)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_boundVolumes, (2 * number - 1) * sizeof(AABB)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempLeafBox, number * sizeof(AABB)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&_flags, (number - 1) * sizeof(uint32_t)));
}



void LBVH::FREE_BVH_CUDA() {
    freeCUDASafe(_indices);
    freeCUDASafe(_MortonHash);
    freeCUDASafe(_nodes);
    freeCUDASafe(_boundVolumes);
    freeCUDASafe(_flags);
    freeCUDASafe(_tempLeafBox);

}

