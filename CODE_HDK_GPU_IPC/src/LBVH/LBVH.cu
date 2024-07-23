
#include "LBVH.cuh"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

#include "UTILS/MathUtils.cuh"

// merge two AABB boundaries
__device__ __host__ 
inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept {
	AABB merged;
	merged.m_upper.x = MATHUTILS::__m_max(lhs.m_upper.x, rhs.m_upper.x);
	merged.m_upper.y = MATHUTILS::__m_max(lhs.m_upper.y, rhs.m_upper.y);
	merged.m_upper.z = MATHUTILS::__m_max(lhs.m_upper.z, rhs.m_upper.z);
	merged.m_lower.x = MATHUTILS::__m_min(lhs.m_lower.x, rhs.m_lower.x);
	merged.m_lower.y = MATHUTILS::__m_min(lhs.m_lower.y, rhs.m_lower.y);
	merged.m_lower.z = MATHUTILS::__m_min(lhs.m_lower.z, rhs.m_lower.z);
	return merged;
}

// calculate whether AABB are overlapped
__device__ __host__ 
inline bool overlap(const AABB& lhs, const AABB& rhs, const double& gapL) noexcept {
	if ((rhs.m_lower.x - lhs.m_upper.x) >= gapL || (lhs.m_lower.x - rhs.m_upper.x) >= gapL) return false;
	if ((rhs.m_lower.y - lhs.m_upper.y) >= gapL || (lhs.m_lower.y - rhs.m_upper.y) >= gapL) return false;
	if ((rhs.m_lower.z - lhs.m_upper.z) >= gapL || (lhs.m_lower.z - rhs.m_upper.z) >= gapL) return false;
	return true;
}

// calculate the centroid of AABB boundaries
__device__ __host__ 
inline double3 centroid(const AABB& box) noexcept {
	return make_double3(
		(box.m_upper.x + box.m_lower.x) * 0.5,
		(box.m_upper.y + box.m_lower.y) * 0.5,
		(box.m_upper.z + box.m_lower.z) * 0.5
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
	x = MATHUTILS::__m_min(MATHUTILS::__m_max(x * resolution, 0.0), resolution - 1.0);
	y = MATHUTILS::__m_min(MATHUTILS::__m_max(y * resolution, 0.0), resolution - 1.0);
	z = MATHUTILS::__m_min(MATHUTILS::__m_max(z * resolution, 0.0), resolution - 1.0);

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

	const int delta_min = MATHUTILS::__m_min(L_delta, R_delta);
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
	m_lower = make_double3(MATHUTILS::__m_min(m_lower.x, x), MATHUTILS::__m_min(m_lower.y, y), MATHUTILS::__m_min(m_lower.z, z));
	m_upper = make_double3(MATHUTILS::__m_max(m_upper.x, x), MATHUTILS::__m_max(m_upper.y, y), MATHUTILS::__m_max(m_upper.z, z));
}

__device__ __host__
void AABB::combines(const double& x, const double& y, const double& z, const double& xx, const double& yy, const double& zz) {
	m_lower = make_double3(MATHUTILS::__m_min(m_lower.x, x), MATHUTILS::__m_min(m_lower.y, y), MATHUTILS::__m_min(m_lower.z, z));
	m_upper = make_double3(MATHUTILS::__m_max(m_upper.x, xx), MATHUTILS::__m_max(m_upper.y, yy), MATHUTILS::__m_max(m_upper.z, zz));
}

__device__ __host__
void AABB::combines(const AABB& aabb) {
	m_lower = make_double3(MATHUTILS::__m_min(m_lower.x, aabb.m_lower.x), MATHUTILS::__m_min(m_lower.y, aabb.m_lower.y), MATHUTILS::__m_min(m_lower.z, aabb.m_lower.z));
	m_upper = make_double3(MATHUTILS::__m_max(m_upper.x, aabb.m_upper.x), MATHUTILS::__m_max(m_upper.y, aabb.m_upper.y), MATHUTILS::__m_max(m_upper.z, aabb.m_upper.z));
}

__device__ __host__
double3 AABB::center() {
	return make_double3((m_upper.x + m_lower.x) * 0.5, (m_upper.y + m_lower.y) * 0.5, (m_upper.z + m_lower.z) * 0.5);
}

__device__ __host__
AABB::AABB() {
	m_lower = make_double3(1e32, 1e32, 1e32);
	m_upper = make_double3(-1e32, -1e32, -1e32);
}


__device__
int _dType_PT(const double3& v0, const double3& v1, const double3& v2, const double3& v3) {
	double3 basis0 = MATHUTILS::__minus(v2, v1);
	double3 basis1 = MATHUTILS::__minus(v3, v1);
	double3 basis2 = MATHUTILS::__minus(v0, v1);

	const double3 nVec = MATHUTILS::__v_vec_cross(basis0, basis1);

	basis1 = MATHUTILS::__v_vec_cross(basis0, nVec);
	MATHUTILS::Matrix3x3d D, D1, D2;

	MATHUTILS::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
	MATHUTILS::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
	MATHUTILS::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

	double2 param[3];
	param[0].x = MATHUTILS::__Determiant(D1) / MATHUTILS::__Determiant(D);
	param[0].y = MATHUTILS::__Determiant(D2) / MATHUTILS::__Determiant(D);

	if (param[0].x > 0 && param[0].x < 1 && param[0].y >= 0) {
		return 3; // PE v1v2
	}
	else {
		basis0 = MATHUTILS::__minus(v3, v2);
		basis1 = MATHUTILS::__v_vec_cross(basis0, nVec);
		basis2 = MATHUTILS::__minus(v0, v2);

		MATHUTILS::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
		MATHUTILS::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
		MATHUTILS::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

		param[1].x = MATHUTILS::__Determiant(D1) / MATHUTILS::__Determiant(D);
		param[1].y = MATHUTILS::__Determiant(D2) / MATHUTILS::__Determiant(D);

		if (param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0) {
			return 4; // PE v2v3
		}
		else {
			basis0 = MATHUTILS::__minus(v1, v3);
			basis1 = MATHUTILS::__v_vec_cross(basis0, nVec);
			basis2 = MATHUTILS::__minus(v0, v3);

			MATHUTILS::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
			MATHUTILS::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
			MATHUTILS::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

			param[2].x = MATHUTILS::__Determiant(D1) / MATHUTILS::__Determiant(D);
			param[2].y = MATHUTILS::__Determiant(D2) / MATHUTILS::__Determiant(D);

			if (param[2].x > 0.0 && param[2].x < 1.0 && param[2].y >= 0.0) {
				return 5; // PE v3v1
			}
			else {
				if (param[0].x <= 0.0 && param[2].x >= 1.0) {
					return 0; // PP v1
				}
				else if (param[1].x <= 0.0 && param[0].x >= 1.0) {
					return 1; // PP v2
				}
				else if (param[2].x <= 0.0 && param[1].x >= 1.0) {
					return 2; // PP v3
				}
				else {
					return 6; // PT
				}
			}
		}
	}
}

__device__
int _dType_EE(const double3& v0, const double3& v1, const double3& v2, const double3& v3)
{
	double3 u = MATHUTILS::__minus(v1, v0);
	double3 v = MATHUTILS::__minus(v3, v2);
	double3 w = MATHUTILS::__minus(v0, v2);

	double a = MATHUTILS::__squaredNorm(u);
	double b = MATHUTILS::__v_vec_dot(u, v);
	double c = MATHUTILS::__squaredNorm(v);
	double d = MATHUTILS::__v_vec_dot(u, w);
	double e = MATHUTILS::__v_vec_dot(v, w);

	double D = a * c - b * b; // always >= 0
	double tD = D; // tc = tN / tD, default tD = D >= 0
	double sN, tN;
	int defaultCase = 8;
	sN = (b * e - c * d);
	if (sN <= 0.0) { // sc < 0 => the s=0 edge is visible
		tN = e;
		tD = c;
		defaultCase = 2;
	}
	else if (sN >= D) { // sc > 1  => the s=1 edge is visible
		tN = e + b;
		tD = c;
		defaultCase = 5;
	}
	else {
		tN = (a * e - b * d);
		if (tN > 0.0 && tN < tD && (MATHUTILS::__v_vec_dot(w, MATHUTILS::__v_vec_cross(u, v)) == 0.0 || MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(u, v)) < 1.0e-20 * a * c)) {
			if (sN < D / 2) {
				tN = e;
				tD = c;
				defaultCase = 2;
			}
			else {
				tN = e + b;
				tD = c;
				defaultCase = 5;
			}
		}
	}

	if (tN <= 0.0) { 
		if (-d <= 0.0) {
			return 0;
		}
		else if (-d >= a) {
			return 3;
		}
		else {
			return 6;
		}
	}
	else if (tN >= tD) { 
		if ((-d + b) <= 0.0) {
			return 1;
		}
		else if ((-d + b) >= a) {
			return 4;
		}
		else {
			return 7;
		}
	}

	return defaultCase;
}


__device__ 
inline bool _checkPTintersection(const double3* _vertexes, const uint32_t& id0, const uint32_t& id1, const uint32_t& id2, const uint32_t& id3, const double& dHat, uint32_t* _cpNum, int* _mInx, int4* _collisionPair, int4* _ccd_collisionPair) noexcept
{
	double3 v0 = _vertexes[id0];
	double3 v1 = _vertexes[id1];
	double3 v2 = _vertexes[id2];
	double3 v3 = _vertexes[id3];

	int dtype = _dType_PT(v0, v1, v2, v3);

	double d = 100;
	switch (dtype) {
	case 0: {
		MATHUTILS::__distancePointPoint(v0, v1, d);
		if (d < dHat) {
			//printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
			int cdp_idx = atomicAdd(_cpNum, 1);         
			_ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
			_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, -1, -1);
			_mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
		}
		break;
	}

	case 1: {
		MATHUTILS::__distancePointPoint(v0, v2, d);
		if (d < dHat) {
			//printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
			int cdp_idx = atomicAdd(_cpNum, 1);         
			_ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
			_collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, -1);
			_mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
		}
		break;
	}

	case 2: {
		MATHUTILS::__distancePointPoint(v0, v3, d);
		if (d < dHat) {
			//printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
			int cdp_idx = atomicAdd(_cpNum, 1);
			_ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
			_collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, -1);
			_mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
		}
		break;
	}

	case 3: {
		MATHUTILS::__distancePointEdge(v0, v1, v2, d);
		if (d < dHat) {
			//printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
			int cdp_idx = atomicAdd(_cpNum, 1);
			_ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
			_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, -1);
			_mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
		}
		break;
	}

	case 4: {
		MATHUTILS::__distancePointEdge(v0, v2, v3, d);
		if (d < dHat) {
			//printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
			int cdp_idx = atomicAdd(_cpNum, 1);
			_ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
			_collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, -1);
			_mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
		}
		break;
	}

	case 5: {
		MATHUTILS::__distancePointEdge(v0, v3, v1, d);
		if (d < dHat) {
			//printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
			int cdp_idx = atomicAdd(_cpNum, 1);
			_ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
			_collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, id1, -1);
			_mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
		}
		break;
	}

	case 6: {
		MATHUTILS::__distancePointTriangle(v0, v1, v2, v3, d);
		if (d < dHat) {
			//printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, d);
			int cdp_idx = atomicAdd(_cpNum, 1);
			_ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
			_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
			//printf("ccbcbcbcbbcbcbbcbcb  %d  %d  %d  %d\n", -id0 - 1, id1, id2, id3);
			_mInx[cdp_idx] = atomicAdd(_cpNum + 4, 1);
		}
		break;
	}

	default:
		break;
	}
}

__device__
inline bool _checkPTintersection_fullCCD(const double3* _vertexes, const uint32_t& id0, const uint32_t& id1, const uint32_t& id2, const uint32_t& id3, const double& dHat, uint32_t* _cpNum, int4* _ccd_collisionPair) noexcept
{
	double3 v0 = _vertexes[id0];
	double3 v1 = _vertexes[id1];
	double3 v2 = _vertexes[id2];
	double3 v3 = _vertexes[id3];

	int dtype = _dType_PT(v0, v1, v2, v3);

	double3 basis0 = MATHUTILS::__minus(v2, v1);
	double3 basis1 = MATHUTILS::__minus(v3, v1);
	double3 basis2 = MATHUTILS::__minus(v0, v1);

	const double3 nVec = MATHUTILS::__v_vec_cross(basis0, basis1);

	double sign = MATHUTILS::__v_vec_dot(nVec, basis2);

	if (dtype==6&&(sign <0)) {
		return;
	}

	_ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(-id0 - 1, id1, id2, id3);
}

__device__
inline bool _checkEEintersection(const double3* _vertexes, const double3* _rest_vertexes, const uint32_t& id0, const uint32_t& id1, const uint32_t& id2, const uint32_t& id3, const uint32_t& obj_idx, const double& dHat, uint32_t* _cpNum, int* MatIndex, int4* _collisionPair, int4* _ccd_collisionPair, int edgeNum) noexcept
{
	double3 v0 = _vertexes[id0];
	double3 v1 = _vertexes[id1];
	double3 v2 = _vertexes[id2];
	double3 v3 = _vertexes[id3];


	int dtype = _dType_EE(v0, v1, v2, v3);
	int add_e = -1;
	double d = 100.0;
	bool smooth = true;
	switch (dtype) {
	case 0: {
		MATHUTILS::__distancePointPoint(v0, v2, d);
		if (d < dHat) {

			double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v0, v1), MATHUTILS::__minus(v2, v3)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v0, v1))*/;
			double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id0], _rest_vertexes[id1], _rest_vertexes[id2], _rest_vertexes[id3]);
			add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;

			if (add_e <= -2) {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {
					_collisionPair[cdp_idx] = make_int4(-id0 - 1, -id2 - 1, -id1 - 1, -id3 - 1);
					MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
					
					break;
				}
				_collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
			}
			else {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
			}
		}
		break;
	}
	
	case 1: {
		MATHUTILS::__distancePointPoint(v0, v3, d);
		if (d < dHat) {

			double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v0, v1), MATHUTILS::__minus(v2, v3)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v0, v1))*/;
			double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id0], _rest_vertexes[id1], _rest_vertexes[id2], _rest_vertexes[id3]);
			add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;

			if (add_e <= -2) {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {
					_collisionPair[cdp_idx] = make_int4(-id0 - 1, -id3 - 1, -id1 - 1, -id2 - 1);
					MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
					break;
				}
				_collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
			}
			else {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
			}
		}
		break;
	}
		
	case 2: {
		MATHUTILS::__distancePointEdge(v0, v2, v3, d);
		if (d < dHat) {

			double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v0, v1), MATHUTILS::__minus(v2, v3)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v0, v1))*/;
			double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id0], _rest_vertexes[id1], _rest_vertexes[id2], _rest_vertexes[id3]);
			add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;


			if (add_e <= -2) {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {
					_collisionPair[cdp_idx] = make_int4(-id0 - 1, -id2 - 1, id3, -id1 - 1);
					MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
					break;
				}
				_collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
			}
			else {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
			}
		}
		break;
	}

	case 3: {
		MATHUTILS::__distancePointPoint(v1, v2, d);
		if (d < dHat) {

			double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v0, v1), MATHUTILS::__minus(v2, v3)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v0, v1))*/;
			double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id0], _rest_vertexes[id1], _rest_vertexes[id2], _rest_vertexes[id3]);
			add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;

			if (add_e <= -2) {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {
					_collisionPair[cdp_idx] = make_int4(-id1 - 1, -id2 - 1, -id0 - 1, -id3 - 1);
					MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
					break;
				}
				_collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
			}
			else {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
			}
		}
		break;
	}

	case 4: {
		MATHUTILS::__distancePointPoint(v1, v3, d);
		if (d < dHat) {

			double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v0, v1), MATHUTILS::__minus(v2, v3)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v0, v1))*/;
			double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id0], _rest_vertexes[id1], _rest_vertexes[id2], _rest_vertexes[id3]);
			add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;

			if (add_e <= -2) {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {
					_collisionPair[cdp_idx] = make_int4(-id1 - 1, -id3 - 1, -id0 - 1, -id2 - 1);
					MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
					break;
				}
				_collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
			}
			else {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
			}
		}
		break;
	}

	case 5: {
		MATHUTILS::__distancePointEdge(v1, v2, v3, d);
		if (d < dHat) {

			double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v0, v1), MATHUTILS::__minus(v2, v3)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v0, v1))*/;
			double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id0], _rest_vertexes[id1], _rest_vertexes[id2], _rest_vertexes[id3]);
			add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;

			if (add_e <= -2) {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {
					_collisionPair[cdp_idx] = make_int4(-id1 - 1, -id2 - 1, id3, -id0 - 1);
					MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
					break;
				}
				_collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
			}
			else {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
			}
		}
		break;
	}

	case 6: {
		MATHUTILS::__distancePointEdge(v2, v0, v1, d);
		if (d < dHat) {

			double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v2, v3), MATHUTILS::__minus(v0, v1)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v2, v3))*/;
			double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id2], _rest_vertexes[id3], _rest_vertexes[id0], _rest_vertexes[id1]);
			add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;


			if (add_e <= -2) {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {
					_collisionPair[cdp_idx] = make_int4(-id2 - 1, -id0 - 1, id1, -id3 - 1);
					MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
					break;
				}
				_collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
			}
			else {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
			}
		}
		break;
	}

	case 7: {
		MATHUTILS::__distancePointEdge(v3, v0, v1, d);
		if (d < dHat) {

			double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v2, v3), MATHUTILS::__minus(v0, v1)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v2, v3))*/;
			double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id2], _rest_vertexes[id3], _rest_vertexes[id0], _rest_vertexes[id1]);
			add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;


			if (add_e <= -2) {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {
					_collisionPair[cdp_idx] = make_int4(-id3 - 1, -id0 - 1, id1, -id2 - 1);
					MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
					break;
				}
				_collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
			}
			else {
				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
			}
		}
		break;
	}

	case 8: {
		MATHUTILS::__distanceEdgeEdge(v0, v1, v2, v3, d);

		double eeSqureNCross = MATHUTILS::__squaredNorm(MATHUTILS::__v_vec_cross(MATHUTILS::__minus(v0, v1), MATHUTILS::__minus(v2, v3)))/* / MATHUTILS::__squaredNorm(MATHUTILS::__minus(v0, v1))*/;
		double eps_x = MATHUTILS::__computeEdgeProductNorm(_rest_vertexes[id0], _rest_vertexes[id1], _rest_vertexes[id2], _rest_vertexes[id3]);
		add_e = (eeSqureNCross < eps_x) ? -obj_idx - 2 : -1;

		if (d < dHat) {
			if (add_e <= -2) {
				//printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nxxxxxxxxxxx\n");
				int cdp_idx = atomicAdd(_cpNum, 1);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				if (smooth) {                  
					_collisionPair[cdp_idx] = make_int4(id0, id1, id2, -id3 - 1);
					break;
				}
				_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);             
			}
			else {

				int cdp_idx = atomicAdd(_cpNum, 1);
				_ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
				MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);

			}
		}
		break;
	}

	default:
		break;
	}
}



__global__
void _reduct_max_box(AABB* _leafBoxes, int number) {
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;

	extern __shared__ AABB tep[];

	if (idx >= number) return;
	//int cfid = tid + CONFLICT_FREE_OFFSET(tid);
	AABB temp = _leafBoxes[idx];

	__threadfence();

	double xmin = temp.m_lower.x, ymin = temp.m_lower.y, zmin = temp.m_lower.z;
	double xmax = temp.m_upper.x, ymax = temp.m_upper.y, zmax = temp.m_upper.z;
	//printf("%f   %f    %f   %f   %f    %f\n", xmin, ymin, zmin, xmax, ymax, zmax);
	//printf("%f   %f    %f\n", xmax, ymax, zmax);
	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	double nextTp;
	int warpNum;
	int tidNum = 32;
	if (blockIdx.x == gridDim.x - 1) {
		warpNum = ((number - idof + 31) >> 5);
		if (warpId == warpNum - 1) {
			tidNum = number - idof - (warpNum - 1) * 32;
		}
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}
	for (int i = 1; i < tidNum; i = (i << 1)) {
		temp.combines(__shfl_down_sync(0xFFFFFFFF, xmin, i), __shfl_down_sync(0xFFFFFFFF, ymin, i), __shfl_down_sync(0xFFFFFFFF, zmin, i),
			__shfl_down_sync(0xFFFFFFFF, xmax, i), __shfl_down_sync(0xFFFFFFFF, ymax, i), __shfl_down_sync(0xFFFFFFFF, zmax, i));
		if (warpTid + i < tidNum) {
			xmin = temp.m_lower.x, ymin = temp.m_lower.y, zmin = temp.m_lower.z;
			xmax = temp.m_upper.x, ymax = temp.m_upper.y, zmax = temp.m_upper.z;
		}
	}
	if (warpTid == 0) {
		tep[warpId] = temp;
	}
	__syncthreads();
	if (threadIdx.x >= warpNum) return;
	if (warpNum > 1) {
		//	tidNum = warpNum;
		temp = tep[threadIdx.x];
		xmin = temp.m_lower.x, ymin = temp.m_lower.y, zmin = temp.m_lower.z;
		xmax = temp.m_upper.x, ymax = temp.m_upper.y, zmax = temp.m_upper.z;
		//	warpNum = ((tidNum + 31) >> 5);
		for (int i = 1; i < warpNum; i = (i << 1)) {
			temp.combines(__shfl_down_sync(0xFFFFFFFF, xmin, i), __shfl_down_sync(0xFFFFFFFF, ymin, i), __shfl_down_sync(0xFFFFFFFF, zmin, i),
				__shfl_down_sync(0xFFFFFFFF, xmax, i), __shfl_down_sync(0xFFFFFFFF, ymax, i), __shfl_down_sync(0xFFFFFFFF, zmax, i));
			if (threadIdx.x + i < warpNum) {
				xmin = temp.m_lower.x, ymin = temp.m_lower.y, zmin = temp.m_lower.z;
				xmax = temp.m_upper.x, ymax = temp.m_upper.y, zmax = temp.m_upper.z;
			}
		}
	}
	if (threadIdx.x == 0) {
		_leafBoxes[blockIdx.x] = temp;
	}
}

template <class element_type>
__global__
void _calcLeafBvs(const double3* _vertexes, const element_type* _elements, AABB* _boundVolumes, int faceNum, int type = 0) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= faceNum) return;
	AABB _bv;

	element_type _e = _elements[idx];
	double3 _v = _vertexes[_e.x];
	_bv.combines(_v.x, _v.y, _v.z);
	_v = _vertexes[_e.y];
	_bv.combines(_v.x, _v.y, _v.z);
	if (type == 0) {
		_v = _vertexes[*((uint32_t*)(&_e) + 2)];
		_bv.combines(_v.x, _v.y, _v.z);
	}
	_boundVolumes[idx] = _bv;
}

template <class element_type>
__global__
void _calcLeafBvs_ccd(const double3* _vertexes, const double3* _moveDir, double alpha, const element_type* _elements, AABB* _boundVolumes, int faceNum, int type = 0) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= faceNum) return;
	AABB _bv;

	element_type _e = _elements[idx];
	double3 _v = _vertexes[_e.x];
	double3 _mvD = _moveDir[_e.x];
	_bv.combines(_v.x, _v.y, _v.z);
	_bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);


	_v = _vertexes[_e.y];
	_mvD = _moveDir[_e.y];
	_bv.combines(_v.x, _v.y, _v.z);
	_bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
	if (type == 0) {
		_v = _vertexes[*((uint32_t*)(&_e) + 2)];
		_mvD = _moveDir[*((uint32_t*)(&_e) + 2)];
		_bv.combines(_v.x, _v.y, _v.z);
		_bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
	}
	_boundVolumes[idx] = _bv;
}

__global__
void _calcMChash(uint64_t* _MortonHash, AABB* _boundVolumes, int number) {
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= number) return;
	AABB maxBv = _boundVolumes[0];
	double3 SceneSize = make_double3(maxBv.m_upper.x - maxBv.m_lower.x, maxBv.m_upper.y - maxBv.m_lower.y, maxBv.m_upper.z - maxBv.m_lower.z);
	double3 centerP = _boundVolumes[idx + number - 1].center();
	double3 offset = make_double3(centerP.x - maxBv.m_lower.x, centerP.y - maxBv.m_lower.y, centerP.z - maxBv.m_lower.z);
	
	//printf("%d   %f     %f     %f\n", offset.x, offset.y, offset.z);
	uint64_t mc32 = morton_code(offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z);
	uint64_t mc64 = ((mc32 << 32) | idx);
	_MortonHash[idx] = mc64;
}

__global__
void _calcLeafNodes(Node* _nodes, const uint32_t* _indices, int number) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= number) return;
	if (idx < number - 1) {
		_nodes[idx].m_left_idx = 0xFFFFFFFF;
		_nodes[idx].m_right_idx = 0xFFFFFFFF;
		_nodes[idx].m_parent_idx = 0xFFFFFFFF;
		_nodes[idx].m_element_idx = 0xFFFFFFFF;
	}
	int l_idx = idx + number - 1;
	_nodes[l_idx].m_left_idx = 0xFFFFFFFF;
	_nodes[l_idx].m_right_idx = 0xFFFFFFFF;
	_nodes[l_idx].m_parent_idx = 0xFFFFFFFF;
	_nodes[l_idx].m_element_idx = _indices[idx];
}




__global__
void _calcInternalNodes(Node* _nodes, const uint64_t* _MortonHash, int number) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= number - 1) return;
	const uint2 ij = determine_range(_MortonHash, number, idx);
	const unsigned int gamma = find_split(_MortonHash, number, ij.x, ij.y);

	_nodes[idx].m_left_idx = gamma;
	_nodes[idx].m_right_idx = gamma + 1;
	if (MATHUTILS::__m_min(ij.x, ij.y) == gamma)
	{
		_nodes[idx].m_left_idx += number - 1;
	}
	if (MATHUTILS::__m_max(ij.x, ij.y) == gamma + 1)
	{
		_nodes[idx].m_right_idx += number - 1;
	}
	_nodes[_nodes[idx].m_left_idx].m_parent_idx = idx;
	_nodes[_nodes[idx].m_right_idx].m_parent_idx = idx;
}

__global__
void _calcInternalAABB(const Node* _nodes, AABB* _boundVolumes, uint32_t* flags, int number) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= number) return;
	idx = idx + number - 1;

	uint32_t parent = _nodes[idx].m_parent_idx;
	while (parent != 0xFFFFFFFF) // means idx == 0
	{
		const int old = atomicCAS(flags + parent, 0xFFFFFFFF, 0);
		if (old == 0xFFFFFFFF)
		{
			return;
		}

		const uint32_t lidx = _nodes[parent].m_left_idx;
		const uint32_t ridx = _nodes[parent].m_right_idx;

		const AABB lbox = _boundVolumes[lidx];
		const AABB rbox = _boundVolumes[ridx];
		_boundVolumes[parent] = merge(lbox, rbox);

		__threadfence();

		parent = _nodes[parent].m_parent_idx;

	}
}

__global__
void _sortBvs(const uint32_t* _indices, AABB* _boundVolumes, AABB* _temp_bvs, int number) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= number) return;
	_boundVolumes[idx] = _temp_bvs[_indices[idx]];
}

__global__
void _selfQuery_vf(const int* _btype, const double3* _vertexes, const uint3* _faces, const uint32_t* _surfVerts, const AABB* _boundVolumes, const Node* _nodes, int4* _collisionPair, int4* _ccd_collisionPair, uint32_t* _cpNum, int* MatIndex, double dHat, int number) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= number) return;

	uint32_t  stack[64];
	uint32_t* stack_ptr = stack;
	*stack_ptr++ = 0;
	  
	AABB _bv;
	idx = _surfVerts[idx];
	_bv.m_upper = _vertexes[idx];
	_bv.m_lower = _vertexes[idx];
	//double bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(_boundVolumes[0].upper, _boundVolumes[0].lower));
	//printf("%f\n", bboxDiagSize2);
	double gapl = sqrt(dHat);//0.001 * sqrt(bboxDiagSize2);
	//double dHat = gapl * gapl;// *bboxDiagSize2;
	unsigned int num_found = 0;
	do
	{
		const uint32_t node_id = *--stack_ptr;
		const uint32_t L_idx = _nodes[node_id].m_left_idx;
		const uint32_t R_idx = _nodes[node_id].m_right_idx;

		if (overlap(_bv, _boundVolumes[L_idx], gapl))
		{
			const auto obj_idx = _nodes[L_idx].m_element_idx;
			if (obj_idx != 0xFFFFFFFF)
			{
				if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
					if (!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2 && _btype[_faces[obj_idx].y] >= 2 && _btype[_faces[obj_idx].z] >= 2))
						_checkPTintersection(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair);
				}
			}
			else // the node is not a leaf.
			{
				*stack_ptr++ = L_idx;
			}
		}
		if (overlap(_bv, _boundVolumes[R_idx], gapl))
		{
			const auto obj_idx = _nodes[R_idx].m_element_idx;
			if (obj_idx != 0xFFFFFFFF)
			{
				if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
					if (!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2 && _btype[_faces[obj_idx].y] >= 2 && _btype[_faces[obj_idx].z] >= 2))
						_checkPTintersection(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair);
				}
			}
			else // the node is not a leaf.
			{
				*stack_ptr++ = R_idx;
			}
		}
	} while (stack < stack_ptr);
} 

__global__
void _selfQuery_vf_ccd(const int* _btype, const double3* _vertexes, const double3* moveDir, double alpha, const uint3* _faces, const uint32_t* _surfVerts, const AABB* _boundVolumes, const Node* _nodes, int4* _ccd_collisionPair, uint32_t* _cpNum, double dHat, int number) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= number) return;

	uint32_t  stack[64];
	uint32_t* stack_ptr = stack;
	*stack_ptr++ = 0;

	AABB _bv;
	idx = _surfVerts[idx];
	double3 current_vertex = _vertexes[idx];
	double3 mvD = moveDir[idx];
	_bv.m_upper = current_vertex;
	_bv.m_lower = current_vertex;
	_bv.combines(current_vertex.x - mvD.x * alpha, current_vertex.y - mvD.y * alpha, current_vertex.z - mvD.z * alpha);
	//double bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(_boundVolumes[0].upper, _boundVolumes[0].lower));
	//printf("%f\n", bboxDiagSize2);
	double gapl = sqrt(dHat);//0.001 * sqrt(bboxDiagSize2);
	//double dHat = gapl * gapl;// *bboxDiagSize2;
	unsigned int num_found = 0;
	do
	{
		const uint32_t node_id = *--stack_ptr;
		const uint32_t L_idx = _nodes[node_id].m_left_idx;
		const uint32_t R_idx = _nodes[node_id].m_right_idx;

		if (overlap(_bv, _boundVolumes[L_idx], gapl))
		{
			const auto obj_idx = _nodes[L_idx].m_element_idx;
			if (obj_idx != 0xFFFFFFFF)
			{
				if(!(_btype[idx]>=2&& _btype[_faces[obj_idx].x] >= 2 && _btype[_faces[obj_idx].y] >= 2 && _btype[_faces[obj_idx].z] >= 2))
					if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
						_ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(-idx - 1, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z);
						//_checkPTintersection_fullCCD(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, _ccd_collisionPair);
					}
			}
			else // the node is not a leaf.
			{
				*stack_ptr++ = L_idx;
			}
		}
		if (overlap(_bv, _boundVolumes[R_idx], gapl))
		{
			const auto obj_idx = _nodes[R_idx].m_element_idx;
			if (obj_idx != 0xFFFFFFFF)
			{
				if(!(_btype[idx]>=2&& _btype[_faces[obj_idx].x] >= 2 && _btype[_faces[obj_idx].y] >= 2 && _btype[_faces[obj_idx].z] >= 2))
					if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
						_ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(-idx - 1, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z);
						//_checkPTintersection_fullCCD(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z, dHat, _cpNum, _ccd_collisionPair);
					}
			}
			else // the node is not a leaf.
			{
				*stack_ptr++ = R_idx;
			}
		}
	} while (stack < stack_ptr);
}


__global__
void _selfQuery_ee(const int* _btype, const double3* _vertexes, const double3* _rest_vertexes, const uint2* _edges, const AABB* _boundVolumes, const Node* _nodes, int4* _collisionPair, int4* _ccd_collisionPair, uint32_t* _cpNum, int* MatIndex, double dHat, int number) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= number) return;

	uint32_t  stack[64];
	uint32_t* stack_ptr = stack;
	*stack_ptr++ = 0;

	idx = idx + number - 1;
	AABB _bv = _boundVolumes[idx];
	uint32_t self_eid = _nodes[idx].m_element_idx;
	//double bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(_boundVolumes[0].upper, _boundVolumes[0].lower));
	//printf("%f\n", bboxDiagSize2);
	double gapl = sqrt(dHat);//0.001 * sqrt(bboxDiagSize2);
	//double dHat = gapl * gapl;// *bboxDiagSize2;
	unsigned int num_found = 0;
	do
	{
		const uint32_t node_id = *--stack_ptr;
		const uint32_t L_idx = _nodes[node_id].m_left_idx;
		const uint32_t R_idx = _nodes[node_id].m_right_idx;
		
		if (overlap(_bv, _boundVolumes[L_idx], gapl))
		{
			const auto obj_idx = _nodes[L_idx].m_element_idx;
			if (obj_idx != 0xFFFFFFFF)
			{
				if (self_eid != obj_idx) {
					if (!(_edges[self_eid].x == _edges[obj_idx].x || _edges[self_eid].x == _edges[obj_idx].y || _edges[self_eid].y == _edges[obj_idx].x || _edges[self_eid].y == _edges[obj_idx].y || obj_idx < self_eid)) {
						//printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y);
						if (!(_btype[_edges[self_eid].x] >= 2 && _btype[_edges[self_eid].y]>= 2 && _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
							_checkEEintersection(_vertexes, _rest_vertexes, _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y, obj_idx, dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair, number);
					}
				}
			}
			else // the node is not a leaf.
			{
				*stack_ptr++ = L_idx;
			}
		}
		if (overlap(_bv, _boundVolumes[R_idx], gapl))
		{
			const auto obj_idx = _nodes[R_idx].m_element_idx;
			if (obj_idx != 0xFFFFFFFF)
			{
				if (self_eid != obj_idx) {
					if (!(_edges[self_eid].x == _edges[obj_idx].x || _edges[self_eid].x == _edges[obj_idx].y || _edges[self_eid].y == _edges[obj_idx].x || _edges[self_eid].y == _edges[obj_idx].y || obj_idx < self_eid)) {
						//printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y);
						if (!(_btype[_edges[self_eid].x] >= 2 && _btype[_edges[self_eid].y]>= 2 && _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
							_checkEEintersection(_vertexes, _rest_vertexes, _edges[self_eid].x, _edges[self_eid].y, _edges[obj_idx].x, _edges[obj_idx].y, obj_idx, dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair, number);
					}
				}
			}
			else // the node is not a leaf.
			{
				*stack_ptr++ = R_idx;
			}
		}
	} while (stack < stack_ptr);
}

__global__
void _selfQuery_ee_ccd(const int* _btype, const double3* _vertexes, const double3* moveDir, double alpha, const uint2* _edges, const AABB* _boundVolumes, const Node* _nodes, int4* _ccd_collisionPair, uint32_t* _cpNum, double dHat, int number) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= number) return;

	uint32_t  stack[64];
	uint32_t* stack_ptr = stack;
	*stack_ptr++ = 0;
	idx = idx + number - 1;
	AABB _bv = _boundVolumes[idx];
	uint32_t self_eid = _nodes[idx].m_element_idx;
	uint2 current_edge = _edges[self_eid];
	//double3 edge_tvert0 = MATHUTILS::__minus(_vertexes[current_edge.x], MATHUTILS::__s_vec_multiply(moveDir[current_edge.x], alpha));
	//double3 edge_tvert1 = MATHUTILS::__minus(_vertexes[current_edge.y], MATHUTILS::__s_vec_multiply(moveDir[current_edge.y], alpha));
	//_bv.combines(edge_tvert0.x, edge_tvert0.y, edge_tvert0.z);
	//_bv.combines(edge_tvert1.x, edge_tvert1.y, edge_tvert1.z);
	double gapl = sqrt(dHat);

	unsigned int num_found = 0;
	do
	{
		const uint32_t node_id = *--stack_ptr;
		const uint32_t L_idx = _nodes[node_id].m_left_idx;
		const uint32_t R_idx = _nodes[node_id].m_right_idx;

		if (overlap(_bv, _boundVolumes[L_idx], gapl))
		{
			const auto obj_idx = _nodes[L_idx].m_element_idx;
			if (obj_idx != 0xFFFFFFFF)
			{
				if (self_eid != obj_idx) {
					if (!(_btype[_edges[self_eid].x] >= 2 && _btype[_edges[self_eid].y] >= 2 && _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
						if (!(current_edge.x == _edges[obj_idx].x || current_edge.x == _edges[obj_idx].y || current_edge.y == _edges[obj_idx].x || current_edge.y == _edges[obj_idx].y || obj_idx < self_eid)) {
							_ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(current_edge.x, current_edge.y, _edges[obj_idx].x, _edges[obj_idx].y);
						}
				}
			}
			else // the node is not a leaf.
			{
				*stack_ptr++ = L_idx;
			}
		}
		if (overlap(_bv, _boundVolumes[R_idx], gapl))
		{
			const auto obj_idx = _nodes[R_idx].m_element_idx;
			if (obj_idx != 0xFFFFFFFF)
			{
				if (self_eid != obj_idx) {
					if (!(_btype[_edges[self_eid].x] >= 2 && _btype[_edges[self_eid].y] >= 2 && _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
						if (!(current_edge.x == _edges[obj_idx].x || current_edge.x == _edges[obj_idx].y || current_edge.y == _edges[obj_idx].x || current_edge.y == _edges[obj_idx].y || obj_idx < self_eid)) {
							_ccd_collisionPair[atomicAdd(_cpNum, 1)] = make_int4(current_edge.x, current_edge.y, _edges[obj_idx].x, _edges[obj_idx].y);
						}
				}
			}
			else // the node is not a leaf.
			{
				*stack_ptr++ = R_idx;
			}
		}
	} while (stack < stack_ptr);
}

///////////////////////////////////////host//////////////////////////////////////////////


AABB calcMaxBV(AABB* _leafBoxes, AABB* _tempLeafBox, const int& number) {

	int numbers = number;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;

	unsigned int sharedMsize = sizeof(AABB) * (threadNum >> 5);

	//AABB* _tempLeafBox;
	//CUDA_SAFE_CALL(cudaMalloc((void**)&_tempLeafBox, number * sizeof(AABB)));
	CUDA_SAFE_CALL(cudaMemcpy(_tempLeafBox, _leafBoxes + number - 1, number * sizeof(AABB), cudaMemcpyDeviceToDevice));
	
	_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);

	numbers = blockNum;
	blockNum = (numbers + threadNum - 1) / threadNum;

	while (numbers > 1) {
		_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
		numbers = blockNum;
		blockNum = (numbers + threadNum - 1) / threadNum;

	}
	cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
	AABB h_bv;
	cudaMemcpy(&h_bv, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToHost);
	//CUDA_SAFE_CALL(cudaFree(_tempLeafBox));
	return h_bv;
}

template <class element_type>
void calcLeafBvs(const double3* _vertexes, const element_type* _faces, AABB* _boundVolumes, const int& faceNum, const int& type) {
	int numbers = faceNum;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	_calcLeafBvs << <blockNum, threadNum >> > (_vertexes, _faces, _boundVolumes + numbers - 1, faceNum, type);
}

template <class element_type>
void calcLeafBvs_fullCCD(const double3* _vertexes, const double3* _moveDir, const double& alpha, const element_type* _faces, AABB* _boundVolumes, const int& faceNum, const int& type) {
	int numbers = faceNum;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	_calcLeafBvs_ccd << <blockNum, threadNum >> > (_vertexes, _moveDir, alpha, _faces, _boundVolumes + numbers - 1, faceNum, type);
}

void calcMChash(uint64_t* _MortonHash, AABB* _boundVolumes, int number) {
	int numbers = number;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	_calcMChash << <blockNum, threadNum >> > (_MortonHash, _boundVolumes, number);
}

void calcLeafNodes(Node* _nodes, const uint32_t* _indices, int number) {
	int numbers = number;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	_calcLeafNodes << <blockNum, threadNum >> > (_nodes, _indices, number);
}

void calcInternalNodes(Node* _nodes, const uint64_t* _MortonHash, int number) {
	int numbers = number;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	_calcInternalNodes << <blockNum, threadNum >> > (_nodes, _MortonHash, number);
}

void calcInternalAABB(const Node* _nodes, AABB* _boundVolumes, uint32_t* flags, int number) {
	int numbers = number;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	//uint32_t* flags;
	//CUDA_SAFE_CALL(cudaMalloc((void**)&flags, (numbers-1) * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMemset(flags, 0xFFFFFFFF, sizeof(uint32_t) * (numbers - 1)));
	_calcInternalAABB << <blockNum, threadNum >> > (_nodes, _boundVolumes, flags, numbers);
	//CUDA_SAFE_CALL(cudaFree(flags));

}

void sortBvs(const uint32_t* _indices, AABB* _boundVolumes, AABB* _temp_bvs, int number) {
	int numbers = number;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	//AABB* _temp_bvs = _tempLeafBox;
   // CUDA_SAFE_CALL(cudaMalloc((void**)&_temp_bvs, (number) * sizeof(AABB)));
	cudaMemcpy(_temp_bvs, _boundVolumes + number - 1, sizeof(AABB) * number, cudaMemcpyDeviceToDevice);
	_sortBvs << <blockNum, threadNum >> > (_indices, _boundVolumes + number - 1, _temp_bvs, number);
	//CUDA_SAFE_CALL(cudaFree(_temp_bvs));
}


void selfQuery_ee(const int* _btype, const double3* _vertexes, const double3* _rest_vertexes, const uint2* _edges, const AABB* _boundVolumes, const Node* _nodes, int4* _collisonPairs, int4* _ccd_collisonPairs, uint32_t* _cpNum, int* MatIndex, double dHat, int number) {
	int numbers = number;
	const unsigned int threadNum = 256;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	
	_selfQuery_ee << <blockNum, threadNum >> > (_btype, _vertexes, _rest_vertexes, _edges, _boundVolumes, _nodes, _collisonPairs, _ccd_collisonPairs, _cpNum, MatIndex, dHat, numbers);
}

void fullCCDselfQuery_ee(const int* _btype, const double3* _vertexes, const double3* moveDir, const double& alpha, const uint2* _edges, const AABB* _boundVolumes, const Node* _nodes, int4* _ccd_collisonPairs, uint32_t* _cpNum, double dHat, int number) {
	int numbers = number;
	const unsigned int threadNum = 256;
	int blockNum = (numbers + threadNum - 1) / threadNum;

	_selfQuery_ee_ccd << <blockNum, threadNum >> > (_btype, _vertexes, moveDir, alpha, _edges, _boundVolumes, _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers);
}

void selfQuery_vf(const int* _btype, const double3* _vertexes, const uint3* _faces, const uint32_t* _surfVerts, const AABB* _boundVolumes, const Node* _nodes, int4* _collisonPairs, int4* _ccd_collisonPairs, uint32_t* _cpNum, int* MatIndex, double dHat, int number) {
	int numbers = number;
	const unsigned int threadNum = 256;
	int blockNum = (numbers + threadNum - 1) / threadNum;

	_selfQuery_vf << <blockNum, threadNum >> > (_btype, _vertexes, _faces, _surfVerts, _boundVolumes, _nodes, _collisonPairs, _ccd_collisonPairs, _cpNum, MatIndex, dHat, numbers);
}

void fullCCDselfQuery_vf(const int* _btype, const double3* _vertexes, const double3* moveDir, const double& alpha, const uint3* _faces, const uint32_t* _surfVerts, const AABB* _boundVolumes, const Node* _nodes, int4* _ccd_collisonPairs, uint32_t* _cpNum, double dHat, int number) {
	int numbers = number;
	const unsigned int threadNum = 256;
	int blockNum = (numbers + threadNum - 1) / threadNum;

	_selfQuery_vf_ccd << <blockNum, threadNum >> > (_btype, _vertexes, moveDir, alpha, _faces, _surfVerts, _boundVolumes, _nodes, _ccd_collisonPairs, _cpNum, dHat, numbers);
}







///////////////////////////////
// Construct LBVH
///////////////////////////////

LBVH::LBVH() {
	std::cout << "construct LBVH" << std::endl;
}

LBVH::~LBVH() {
	std::cout << "deconstruct LBVH" << std::endl;
}


void LBVH_F::init(int* _mbtype, double3* _mVerts, uint3* _mFaces, uint32_t* _mSurfVert, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& faceNum, const int& vertNum) {
	mc_faces = _mFaces;
	mc_surfVerts = _mSurfVert;
	mc_vertexes = _mVerts;
	mc_collisionPair = _mCollisonPairs;
	mc_ccd_collisionPair = _ccd_mCollisonPairs;
	mc_cpNum = _mcpNum;
	mc_MatIndex = _mMatIndex;
	m_face_number = faceNum;
	m_vert_number = vertNum;
	mc_btype = _mbtype;
	// CUDA_MALLOC_LBVH(m_face_number);
}

void LBVH_E::init(int* _mbtype, double3* _mVerts, double3* _mRest_vertexes, uint2* _mEdges, int4* _mCollisonPairs, int4* _ccd_mCollisonPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& edgeNum, const int& vertNum) {
	mc_rest_vertexes = _mRest_vertexes;
	mc_edges = _mEdges;
	mc_vertexes = _mVerts;
	mc_cpNum = _mcpNum;
	mc_collisionPair = _mCollisonPairs;
	mc_ccd_collisionPair = _ccd_mCollisonPairs;
	mc_MatIndex = _mMatIndex;
	m_edge_number = edgeNum;
	m_vert_number = vertNum;
	mc_btype = _mbtype;
	// CUDA_MALLOC_LBVH(m_edge_number);
}

double LBVH_F::Construct() {
	calcLeafBvs(mc_vertexes, mc_faces, mc_boundVolumes, m_face_number, 0);
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	m_scene = calcMaxBV(mc_boundVolumes, mc_tempLeafBox, m_face_number);
	calcMChash(mc_MortonCodeHash, mc_boundVolumes, m_face_number);
	thrust::sequence(thrust::device_ptr<uint32_t>(mc_indices), thrust::device_ptr<uint32_t>(mc_indices) + m_face_number);
	thrust::sort_by_key(thrust::device_ptr<uint64_t>(mc_MortonCodeHash), thrust::device_ptr<uint64_t>(mc_MortonCodeHash) + m_face_number, thrust::device_ptr<uint32_t>(mc_indices));
	sortBvs(mc_indices, mc_boundVolumes, mc_tempLeafBox, m_face_number);
	calcLeafNodes(mc_nodes, mc_indices, m_face_number);
	calcInternalNodes(mc_nodes, mc_MortonCodeHash, m_face_number);
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	calcInternalAABB(mc_nodes, mc_boundVolumes, mc_flags, m_face_number);
	return 0;//time0 + time1 + time2;

}

double LBVH_F::ConstructFullCCD(const double3* moveDir, const double& alpha) {
	calcLeafBvs_fullCCD(mc_vertexes, moveDir, alpha, mc_faces, mc_boundVolumes, m_face_number, 0);
	m_scene = calcMaxBV(mc_boundVolumes, mc_tempLeafBox, m_face_number);
	calcMChash(mc_MortonCodeHash, mc_boundVolumes, m_face_number);
	thrust::sequence(thrust::device_ptr<uint32_t>(mc_indices), thrust::device_ptr<uint32_t>(mc_indices) + m_face_number);

	thrust::sort_by_key(thrust::device_ptr<uint64_t>(mc_MortonCodeHash), thrust::device_ptr<uint64_t>(mc_MortonCodeHash) + m_face_number, thrust::device_ptr<uint32_t>(mc_indices));
	sortBvs(mc_indices, mc_boundVolumes, mc_tempLeafBox, m_face_number);

	calcLeafNodes(mc_nodes, mc_indices, m_face_number);

	calcInternalNodes(mc_nodes, mc_MortonCodeHash, m_face_number);
	calcInternalAABB(mc_nodes, mc_boundVolumes, mc_flags, m_face_number);

	return 0;
}

double LBVH_E::Construct() {

	/*cudaEvent_t start, end0, end1, end2;
	cudaEventCreate(&start);
	cudaEventCreate(&end0);
	cudaEventCreate(&end1);
	cudaEventCreate(&end2);

	cudaEventRecord(start);*/
	calcLeafBvs(mc_vertexes, mc_edges, mc_boundVolumes, m_edge_number, 1);
	m_scene = calcMaxBV(mc_boundVolumes, mc_tempLeafBox, m_edge_number);
	calcMChash(mc_MortonCodeHash, mc_boundVolumes, m_edge_number);
	thrust::sequence(thrust::device_ptr<uint32_t>(mc_indices), thrust::device_ptr<uint32_t>(mc_indices) + m_edge_number);
	//cudaEventRecord(end0);

	thrust::sort_by_key(thrust::device_ptr<uint64_t>(mc_MortonCodeHash), thrust::device_ptr<uint64_t>(mc_MortonCodeHash) + m_edge_number, thrust::device_ptr<uint32_t>(mc_indices));
	sortBvs(mc_indices, mc_boundVolumes, mc_tempLeafBox, m_edge_number);

	//cudaEventRecord(end1);

	calcLeafNodes(mc_nodes, mc_indices, m_edge_number);

	calcInternalNodes(mc_nodes, mc_MortonCodeHash, m_edge_number);
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	calcInternalAABB(mc_nodes, mc_boundVolumes, mc_flags, m_edge_number);
	//selfQuery(_vertexes, _edges, _boundVolumes, _nodes, _collisionPair, _cpNum, edge_number);
	//cudaEventRecord(end2);
	//CUDA_SAFE_CALL(cudaDeviceSynchronize());
	/*float time0 = 0, time1 = 0, time2 = 0;
	cudaEventElapsedTime(&time0, start, end0);
	cudaEventElapsedTime(&time1, end0, end1);
	cudaEventElapsedTime(&time2, end1, end2);
	(cudaEventDestroy(start));
	(cudaEventDestroy(end0));
	(cudaEventDestroy(end1));
	(cudaEventDestroy(end2));*/
	//std::cout << "sort time: " << time1 << std::endl;
	return 0;//time0 + time1 + time2;
	//std::cout << "generation done: " << time0 + time1 + time2 << std::endl;
}

double LBVH_E::ConstructFullCCD(const double3* moveDir, const double& alpha) {
	calcLeafBvs_fullCCD(mc_vertexes, moveDir, alpha, mc_edges, mc_boundVolumes, m_edge_number, 1);
	m_scene = calcMaxBV(mc_boundVolumes, mc_tempLeafBox, m_edge_number);
	calcMChash(mc_MortonCodeHash, mc_boundVolumes, m_edge_number);
	thrust::sequence(thrust::device_ptr<uint32_t>(mc_indices), thrust::device_ptr<uint32_t>(mc_indices) + m_edge_number);

	thrust::sort_by_key(thrust::device_ptr<uint64_t>(mc_MortonCodeHash), thrust::device_ptr<uint64_t>(mc_MortonCodeHash) + m_edge_number, thrust::device_ptr<uint32_t>(mc_indices));
	sortBvs(mc_indices, mc_boundVolumes, mc_tempLeafBox, m_edge_number);

	calcLeafNodes(mc_nodes, mc_indices, m_edge_number);

	calcInternalNodes(mc_nodes, mc_MortonCodeHash, m_edge_number);

	calcInternalAABB(mc_nodes, mc_boundVolumes, mc_flags, m_edge_number);

	return 0;
}

void LBVH_F::SelfCollitionDetect(double dHat) {

	selfQuery_vf(mc_btype, mc_vertexes, mc_faces, mc_surfVerts, mc_boundVolumes, mc_nodes, mc_collisionPair, mc_ccd_collisionPair, mc_cpNum, mc_MatIndex, dHat, m_vert_number);

}

void LBVH_E::SelfCollitionDetect(double dHat) {

	selfQuery_ee(mc_btype, mc_vertexes, mc_rest_vertexes, mc_edges, mc_boundVolumes, mc_nodes, mc_collisionPair, mc_ccd_collisionPair, mc_cpNum, mc_MatIndex, dHat, m_edge_number);

}

void LBVH_F::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha) {

	fullCCDselfQuery_vf(mc_btype, mc_vertexes, moveDir, alpha, mc_faces, mc_surfVerts, mc_boundVolumes, mc_nodes, mc_ccd_collisionPair, mc_cpNum, dHat, m_vert_number);

}

void LBVH_E::SelfCollitionFullDetect(double dHat, const double3* moveDir, const double& alpha) {

	fullCCDselfQuery_ee(mc_btype, mc_vertexes, moveDir, alpha, mc_edges, mc_boundVolumes, mc_nodes, mc_ccd_collisionPair, mc_cpNum, dHat, m_edge_number);

}

AABB* LBVH_F::getSceneSize() {
	calcLeafBvs(mc_vertexes, mc_faces, mc_boundVolumes, m_face_number, 0);
	calcMaxBV(mc_boundVolumes, mc_tempLeafBox, m_face_number);
	return mc_boundVolumes;
}


__global__
void _GroundCollisionDetect(const double3* vertexes, const uint32_t* surfVertIds, const double* g_offset, const double3* g_normal, uint32_t* _environment_collisionPair, uint32_t* _gpNum, double dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    double dist = MATHUTILS::__v_vec_dot(*g_normal, vertexes[surfVertIds[idx]]) - *g_offset;
    if (dist * dist > dHat) return;

    _environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];

}

void LBVH::GroundCollisionDetect(const double3* _vertexes, const uint32_t* _surfVerts, const double* _groundOffset, const double3* _groundNormal, uint32_t* _environment_collisionPair, uint32_t* _gpNum, double dHat, int numSurfVerts) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numSurfVerts + threadNum - 1) / threadNum; //
    _GroundCollisionDetect <<<blockNum, threadNum >>> (_vertexes, _surfVerts, _groundOffset, _groundNormal, _environment_collisionPair, _gpNum, dHat, numSurfVerts);
}



///////////////////
// CUDA MALLOC
///////////////////


void LBVH::CUDA_MALLOC_LBVH(const int& number) {
	// number will be leafnodes, number-1 will be internal nodes
	CUDAMallocSafe(mc_indices, number);
	CUDAMallocSafe(mc_MortonCodeHash, number);
	CUDAMallocSafe(mc_nodes, 2 * number - 1);
	CUDAMallocSafe(mc_boundVolumes, 2 * number - 1);
	CUDAMallocSafe(mc_tempLeafBox, number);
	CUDAMallocSafe(mc_flags, number - 1);
}

void LBVH::CUDA_FREE_LBVH() {


	CUDAFreeSafe(mc_indices);
	CUDAFreeSafe(mc_MortonCodeHash);
	CUDAFreeSafe(mc_nodes);
	CUDAFreeSafe(mc_boundVolumes);
	CUDAFreeSafe(mc_tempLeafBox);
	CUDAFreeSafe(mc_flags);
}


