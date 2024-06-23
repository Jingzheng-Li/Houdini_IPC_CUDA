
#include "LBVH.cuh"

template<class T>
__device__ __host__ inline T __m_min(T a, T b) {
    return a < b ? a : b;
}

template <class T>
__device__ __host__ inline T __m_max(T a, T b) {
    return a < b ? b : a;
}

// merge two AABB boundaries
__device__ __host__ inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept {
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
__device__ __host__ inline bool overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept {
    if ((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL) return false;
    if ((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL) return false;
    if ((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL) return false;
    return true;
}

// calculate the centroid of AABB boundaries
__device__ __host__ inline double3 centroid(const AABB& box) noexcept {
    return make_double3(
        (box.upper.x + box.lower.x) * 0.5,
        (box.upper.y + box.lower.y) * 0.5,
        (box.upper.z + box.lower.z) * 0.5
    );
}

// calculate the Morton code of a int number
__device__ __host__ inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// calculate the Morton code a point
__device__ __host__ inline std::uint32_t morton_code(double x, double y, double z, double resolution = 1024.0) noexcept {
    x = __m_min(__m_max(x * resolution, 0.0), resolution - 1.0);
    y = __m_min(__m_max(y * resolution, 0.0), resolution - 1.0);
    z = __m_min(__m_max(z * resolution, 0.0), resolution - 1.0);

    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(z));

    return (xx << 2) + (yy << 1) + zz;
}

// calculate max common upper bits
__device__ inline int common_upper_bits(const unsigned long long lhs, const unsigned long long rhs) noexcept {
    return __clzll(lhs ^ rhs);
}

// calculate the range of a tree node can range for leaves node
__device__ inline uint2 determine_range(const uint64_t* node_code, const unsigned int num_leaves, unsigned int idx) {
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
__device__ inline unsigned int find_split(const uint64_t* node_code, const unsigned int num_leaves, const unsigned int first, const unsigned int last) noexcept {
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


