
#pragma once

#include <vector>
#include <string>
#include <sstream>

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"


#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/locate.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

typedef CGAL::Simple_cartesian<Scalar> CGAL_K;
typedef CGAL_K::Point_3 CGAL_Point_3;
typedef CGAL_K::Vector_3 CGAL_Vector_3;
typedef CGAL_K::Triangle_3 CGAL_Triangle_3;
typedef CGAL::Surface_mesh<CGAL_Point_3> CGAL_Mesh;
typedef CGAL_Mesh::Face_index CGAL_Face_index;
typedef CGAL_Mesh::Vertex_index CGAL_Vertex_index;
typedef CGAL::AABB_face_graph_triangle_primitive<CGAL_Mesh> CGAL_Primitive;
typedef CGAL::AABB_traits<CGAL_K, CGAL_Primitive> CGAL_AABB_traits;
typedef CGAL::AABB_tree<CGAL_AABB_traits> CGAL_Tree;
typedef CGAL_Tree::Point_and_primitive_id CGAL_Point_and_primitive_id;


struct SIMMesh {

	CGAL_Mesh CGAL_tetmesh;
	CGAL_Mesh CGAL_clothmesh;
	CGAL_Mesh CGAL_bodymesh;
	CGAL_Mesh CGAL_surfmesh;

	std::vector<Scalar3> totalverts;
	std::vector<uint3> totalfaces;
	std::vector<Scalar3> clothverts;
	std::vector<uint3> clothfaces;
	std::vector<Scalar3> bodyverts;
	std::vector<uint3> bodyfaces;

}; // struct objMesh


namespace LOADMESH {

void readObj_CGAL(const std::string& filename, CGAL_Mesh& mesh);
void writeObj_CGAL(const std::string& filename, const CGAL_Mesh& mesh);

void convertCGALVertsToVector(const CGAL_Mesh& mesh, std::vector<Scalar3>& verts);
void convertCGALFacesToVector(const CGAL_Mesh& mesh, std::vector<uint3>& faces);

// used for triangle bending
void extractTriBendEdgesFaces_CGAL(const CGAL_Mesh& mesh, std::vector<uint2>& triBendEdges, std::vector<uint2>& triBendVerts);

} // namespace LOADMESH




class LoadObjMesh {

public:


public:

	LoadObjMesh();

	std::vector<Scalar> volume;
	std::vector<Scalar> area;
	std::vector<Scalar> masses;

	Scalar meanMass;
	Scalar meanVolume;
	std::vector<Scalar3> vertexes;
	std::vector<Scalar3> velocities;

	std::vector<uint4> tetrahedras;
	std::vector<uint3> triangles;

	std::vector<int> boundaryTypies;
	std::vector<uint32_t> targetIndex;

	std::vector<__MATHUTILS__::Matrix3x3S> DMInverse;
	std::vector<__MATHUTILS__::Matrix2x2S> triDMInverse;
	std::vector<__MATHUTILS__::Matrix3x3S> constraints;

	std::vector<Scalar3> targetPos;
	std::vector<uint2> triBendEdges; // 一条边的两个顶点索引
	std::vector<uint2> triBendVerts; // 一条边的两个面第三个点的索引

	std::vector<uint32_t> surfId2TetId;
	std::vector<uint3> surfFaces;
	std::vector<uint32_t> surfVerts;
	std::vector<uint2> surfEdges;

	std::vector<std::vector<unsigned int>> vertNeighbors;
	std::vector<unsigned int> neighborList;
	std::vector<unsigned int> neighborStart;
	std::vector<unsigned int> neighborNum;

	int numVertices;
	int numTetElements;
	int numTriElements;
	int numBoundTargets;
	int vertexOffset;


	int getVertNeighbors();

	bool load_tetrahedraMesh(const std::string& filename, Scalar scale, Scalar3 position_offset);
	bool load_triMesh(const std::string& filename, Scalar scale, Scalar3 transform, int boundaryType);
	
	bool load_animation(const std::string& filename, Scalar scale, Scalar3 transform);
	bool load_soft_animation(const std::string& filename, Scalar scale, Scalar3 transform, std::vector<Scalar3>& softTargetPos);
	
	void getTetSurface();

	void getSurface();
	


};
