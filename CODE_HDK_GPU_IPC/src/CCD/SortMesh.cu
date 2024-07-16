
#include "SortMesh.cuh"
#include "UTILS/MathUtils.cuh"
#include "LBVH.cuh"


#include <cmath>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>


/////////////////////////////////
// SORT MESH
/////////////////////////////////

__device__ __host__ 
inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}


__device__ __host__
inline uint32_t hash_code(int type, double x, double y, double z, double resolution = 1024) noexcept
{
	x = MATHUTILS::__m_min(MATHUTILS::__m_max(x * resolution, 0.0), resolution - 1.0);
	y = MATHUTILS::__m_min(MATHUTILS::__m_max(y * resolution, 0.0), resolution - 1.0);
	z = MATHUTILS::__m_min(MATHUTILS::__m_max(z * resolution, 0.0), resolution - 1.0);

	if (type == -1) {
		const uint32_t xx = expand_bits(static_cast<uint32_t>(x));
		const uint32_t yy = expand_bits(static_cast<uint32_t>(y));
		const uint32_t zz = expand_bits(static_cast<uint32_t>(z));
		std::uint32_t mchash = ((xx << 2) + (yy << 1) + zz);

		return mchash;
	}
	else if (type == 0) {
		return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(y)) * 1024) + static_cast<uint32_t>(x);
	}
	else if (type == 1) {
		return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(z)) * 1024) + static_cast<uint32_t>(x);
	}
	else if (type == 2) {
		return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(z)) * 1024) + static_cast<uint32_t>(y);
	}
	else if (type == 3) {
		return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(x)) * 1024) + static_cast<uint32_t>(y);
	}
	else if (type == 4) {
		return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(x)) * 1024) + static_cast<uint32_t>(z);
	}
	else {
		return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(y)) * 1024) + static_cast<uint32_t>(z);
	}
	//std::uint32_t mchash = (((static_cast<std::uint32_t>(z) * 1024) + static_cast<std::uint32_t>(y)) * 1024) + static_cast<std::uint32_t>(x);//((xx << 2) + (yy << 1) + zz);
	//return mchash;
}

__global__
void _calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number) {
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= number) return;
	double3 SceneSize = make_double3((*_MaxBv).upper.x - (*_MaxBv).lower.x, (*_MaxBv).upper.y - (*_MaxBv).lower.y, (*_MaxBv).upper.z - (*_MaxBv).lower.z);
	double3 centerP = _vertexes[idx];
	double3 offset = make_double3(centerP.x - (*_MaxBv).lower.x, centerP.y - (*_MaxBv).lower.y, centerP.z - (*_MaxBv).lower.z);
	int type = -1;
	if (type >= 0) {
		if (SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z) {
			type = 0;
		}
		else if (SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y) {
			type = 1;
		}
		else if (SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x) {
			type = 2;
		}
		else if (SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z) {
			type = 3;
		}
		else if (SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y) {
			type = 4;
		}
		else {
			type = 5;
		}
	}

	//printf("minSize %f     %f     %f\n", SceneSize.x, SceneSize.y, SceneSize.z);
	uint64_t mc32 = hash_code(type, offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z);
	uint64_t mc64 = ((mc32 << 32) | idx);
	//printf("morton code %lld\n", mc64);
	_MChash[idx] = mc64;
}


__global__
void _updateTopology(uint4* tets, uint3* tris, const uint32_t* sortMapVertIndex, int traNumber, int triNumber) {
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < traNumber) {

		tets[idx].x = sortMapVertIndex[tets[idx].x];
		tets[idx].y = sortMapVertIndex[tets[idx].y];
		tets[idx].z = sortMapVertIndex[tets[idx].z];
		tets[idx].w = sortMapVertIndex[tets[idx].w];
	}
	if (idx < triNumber) {
		tris[idx].x = sortMapVertIndex[tris[idx].x];
		tris[idx].y = sortMapVertIndex[tris[idx].y];
		tris[idx].z = sortMapVertIndex[tris[idx].z];
	}
}

__global__
void _updateVertexes(double3* o_vertexes, const double3* _vertexes, double* tempM, const double* mass, MATHUTILS::Matrix3x3d* tempCons, int* tempBtype, const MATHUTILS::Matrix3x3d* cons, const int* bType, const uint32_t* sortIndex, uint32_t* sortMapIndex, int number) {
	uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= number) return;

	CHECK_INDEX_BOUNDS(sortIndex[idx], number);

	o_vertexes[idx] = _vertexes[sortIndex[idx]];
	tempM[idx] = mass[sortIndex[idx]];
	tempCons[idx] = cons[sortIndex[idx]];
	sortMapIndex[sortIndex[idx]] = idx;
	tempBtype[idx] = bType[sortIndex[idx]];

	// printf("original idx: %d new idx: %d\n", sortIndex[idx], idx);
}

__global__
void _updateEdges(uint32_t* sortIndex, uint2* _edges, int _offset_num, int numbers) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numbers) return;
	if (_edges[idx].x < _offset_num) {
		_edges[idx].x = sortIndex[_edges[idx].x];
	}
	else {
		_edges[idx].x = _edges[idx].x;
	}
	if (_edges[idx].y < _offset_num) {
		_edges[idx].y = sortIndex[_edges[idx].y];
	}
	else {
		_edges[idx].y = _edges[idx].y;
	}
}

__global__
void _updateTriEdges_adjVerts(uint32_t* sortIndex, uint2* _edges, uint2* _adj_verts, int _offset_num, int numbers) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numbers) return;
	if (_edges[idx].x < _offset_num) {
		_edges[idx].x = sortIndex[_edges[idx].x];
	}
	else {
		_edges[idx].x = _edges[idx].x;
	}
	if (_edges[idx].y < _offset_num) {
		_edges[idx].y = sortIndex[_edges[idx].y];
	}
	else {
		_edges[idx].y = _edges[idx].y;
	}


	if (_adj_verts[idx].x < _offset_num) {
		_adj_verts[idx].x = sortIndex[_adj_verts[idx].x];
	}
	else {
		_adj_verts[idx].x = _adj_verts[idx].x;
	}
	if (_adj_verts[idx].y < _offset_num) {
		_adj_verts[idx].y = sortIndex[_adj_verts[idx].y];
	}
	else {
		_adj_verts[idx].y = _adj_verts[idx].y;
	}
}

__global__
void _updateSurfaces(uint32_t* sortIndex, uint3* _faces, int _offset_num, int numbers) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numbers) return;
	if (_faces[idx].x < _offset_num) {
		_faces[idx].x = sortIndex[_faces[idx].x];
	}
	else {
		_faces[idx].x = _faces[idx].x;
	}
	if (_faces[idx].y < _offset_num) {
		_faces[idx].y = sortIndex[_faces[idx].y];
	}else{
		_faces[idx].y = _faces[idx].y;
	}
	if (_faces[idx].z < _offset_num) {
		_faces[idx].z = sortIndex[_faces[idx].z];
	}
	else {
		_faces[idx].z = _faces[idx].z;
	}
	//printf("sorted face: %d  %d  %d\n", _faces[idx].x, _faces[idx].y, _faces[idx].z);
}

__global__
void _updateSurfVerts(uint32_t* sortIndex, uint32_t* _sVerts, int _offset_num, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (_sVerts[idx] < _offset_num) {
        _sVerts[idx] = sortIndex[_sVerts[idx]];
    }
    else {
        _sVerts[idx] = _sVerts[idx];
    }
}

// caculate morton code of each tet vertex pos
void calcVertMChash(uint64_t* _MChash, const double3* _vertexes, const AABB* _MaxBv, int number) {
	int numbers = number;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	_calcVertMChash <<<blockNum, threadNum>>> (_MChash, _vertexes, _MaxBv, number);
}

void updateVertexes(double3* o_vertexes, const double3* _vertexes, double* tempM, const double* mass, MATHUTILS::Matrix3x3d* tempCons, int* tempBtype, const MATHUTILS::Matrix3x3d* cons, const int* bType, const uint32_t* sortIndex, uint32_t* sortMapIndex, int number) {
	int numbers = number;
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	_updateVertexes <<<blockNum, threadNum>>> (o_vertexes, _vertexes, tempM, mass, tempCons, tempBtype, cons, bType, sortIndex, sortMapIndex, numbers);
}


void updateTopology(uint4* tets, uint3* tris, const uint32_t* sortMapVertIndex, int traNumber, int triNumber) {
	int numbers = MATHUTILS::__m_max(traNumber, triNumber);
	const unsigned int threadNum = default_threads;
	int blockNum = (numbers + threadNum - 1) / threadNum;
	_updateTopology <<<blockNum, threadNum >>> (tets, tris, sortMapVertIndex, traNumber, triNumber);
}


void sortGeometry(std::unique_ptr<GeometryManager>& instance, const AABB* _MaxBv, const int& vertex_num, const int& tetradedra_num, const int& triangle_num) {

    calcVertMChash(instance->cudaMortonCodeHash, instance->cudaTetPos, _MaxBv, vertex_num);
    
	// generate cudaSortIndex from 0 to vertex_num - 1
	thrust::sequence(thrust::device_ptr<uint32_t>(instance->cudaSortIndex), thrust::device_ptr<uint32_t>(instance->cudaSortIndex) + vertex_num);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(instance->cudaMortonCodeHash), thrust::device_ptr<uint64_t>(instance->cudaMortonCodeHash) + vertex_num, thrust::device_ptr<uint32_t>(instance->cudaSortIndex));

    updateVertexes(instance->cudaOriginTetPos, instance->cudaTetPos, instance->cudaTempDouble, instance->cudaTetMass, instance->cudaTempMat3x3, instance->cudaTempBoundaryType, instance->cudaConstraints, instance->cudaBoundaryType, instance->cudaSortIndex, instance->cudaSortMapVertIndex, vertex_num);

    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaTetPos, instance->cudaOriginTetPos, vertex_num * sizeof(double3), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaTetMass, instance->cudaTempDouble, vertex_num * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaConstraints, instance->cudaTempMat3x3, vertex_num * sizeof(MATHUTILS::Matrix3x3d), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaBoundaryType, instance->cudaTempBoundaryType, vertex_num * sizeof(int), cudaMemcpyDeviceToDevice));

    updateTopology(instance->cudaTetElement, instance->cudaTriElement, instance->cudaSortMapVertIndex, tetradedra_num, triangle_num);

}


void updateSurfaces(uint32_t* sortIndex, uint3* _faces, const int& offset_num, const int& numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateSurfaces <<<blockNum, threadNum >>> (sortIndex, _faces, offset_num, numbers);
}

void updateSurfaceEdges(uint32_t* sortIndex, uint2* _edges, const int& offset_num, const int& numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateEdges <<<blockNum, threadNum >>> (sortIndex, _edges, offset_num, numbers);
}

void updateTriEdges_adjVerts(uint32_t* sortIndex, uint2* _tri_edges, uint2* _adj_verts, const int& offset_num, const int& numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateTriEdges_adjVerts <<<blockNum, threadNum >>> (sortIndex, _tri_edges, _adj_verts, offset_num, numbers);
}

void updateSurfaceVerts(uint32_t* sortIndex, uint32_t* _sVerts, const int& offset_num, const int& numbers) {
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateSurfVerts << <blockNum, threadNum >> > (sortIndex, _sVerts, offset_num, numbers);
}

AABB* calcuMaxSceneSize(std::unique_ptr<LBVH_F>& LBVH_F_ptr) {
	return LBVH_F_ptr->getSceneSize();
}

void SortMesh::sortMesh(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_F>& LBVH_F_ptr) {
	int triangleNum = 0;
	int triangleedgeNum = 0;

	int numVerts = instance->tetPos.rows();
	int numTetEles = instance->tetElement.rows();
	int numSurfVerts = instance->surfVert.rows();
	int numSurfFaces = instance->surfFace.rows();
	int numSurfEdges = instance->surfEdge.rows();

    sortGeometry(instance, calcuMaxSceneSize(LBVH_F_ptr), numVerts, numTetEles, triangleNum);
    CUDA_KERNEL_CHECK();

    updateSurfaces(instance->cudaSortMapVertIndex, instance->cudaSurfFace, numVerts, numSurfFaces);
	CUDA_KERNEL_CHECK();

    updateSurfaceEdges(instance->cudaSortMapVertIndex, instance->cudaSurfEdge, numVerts, numSurfEdges);
	CUDA_KERNEL_CHECK();

    // updateTriEdges_adjVerts(instance->cudaSortMapVertIndex, TetMesh.tri_edges, TetMesh.tri_edge_adj_vertex, numVerts, 0);
	// CUDA_KERNEL_CHECK();

    updateSurfaceVerts(instance->cudaSortMapVertIndex, instance->cudaSurfVert, numVerts, numSurfVerts);
	CUDA_KERNEL_CHECK();


	//////////////////////
	// check the result //
	//////////////////////
	Eigen::MatrixXi tempsurfface(numSurfFaces, 3);
	Eigen::MatrixXi tempsurfedge(numSurfEdges, 2);
	Eigen::VectorXi tempsurfvert(numSurfVerts);
	copyFromCUDASafe(tempsurfface, instance->cudaSurfFace);
	copyFromCUDASafe(tempsurfedge, instance->cudaSurfEdge);
	copyFromCUDASafe(tempsurfvert, instance->cudaSurfVert);
	std::cout << "gettempsurfface~" << tempsurfface.row(0) << std::endl;
	std::cout << "gettempsurfedge~" << tempsurfedge.row(0) << std::endl;
	std::cout << "gettempsurfvert~" << tempsurfvert(0) << std::endl;

	Eigen::VectorXi tempsortmapvertindex(numVerts);
	copyFromCUDASafe(tempsortmapvertindex, instance->cudaSortMapVertIndex);
	std::cout << "getcudaSortMapVertIndex~" << tempsortmapvertindex(0) << " " << tempsortmapvertindex(100) << " " << tempsortmapvertindex(200) << std::endl;

}

