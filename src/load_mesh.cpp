

#include <iostream>
#include <unordered_map>
#include <fstream>
#include <set>
#include <queue>
#include <map>
#include<iostream>
#include <cfloat>
#include <cstring>

#include "load_mesh.hpp"
#include "Eigen/Eigen"



void LOADMESH::readObj_CGAL(const std::string& filename, CGAL_Mesh& mesh) {
    if (!CGAL::IO::read_polygon_mesh(filename, mesh)) {
        std::cerr << "Error: failed to read obj file" << filename << std::endl;
    }
}

void LOADMESH::writeObj_CGAL(const std::string& filename, const CGAL_Mesh& mesh) {
    if (!CGAL::IO::write_polygon_mesh(filename, mesh)) {
        std::cerr << "Error: failed to write obj file" << filename << std::endl;
    }
}

void LOADMESH::convertCGALVertsToVector(const CGAL_Mesh& mesh, std::vector<Scalar3>& verts) {
    verts.clear();
    for (CGAL_Vertex_index v : mesh.vertices()) {
        CGAL_Point_3 p = mesh.point(v);
		verts.emplace_back(make_Scalar3(p.x(), p.y(), p.z()));
    }
}

void LOADMESH::convertCGALFacesToVector(const CGAL_Mesh& mesh, std::vector<uint3>& faces) {
    faces.clear();
    for (CGAL_Face_index f : mesh.faces()) {
        std::vector<unsigned int> faceVertices;
        for (CGAL_Vertex_index v : vertices_around_face(mesh.halfedge(f), mesh)) {
            faceVertices.push_back(v.idx());
        }
        if (faceVertices.size() == 3) {
            faces.emplace_back(make_uint3(faceVertices[0], faceVertices[1], faceVertices[2]));
        }
    }
}


void LOADMESH::extractTriBendEdgesFaces_CGAL(const CGAL_Mesh& mesh, std::vector<uint2>& triBendEdges, std::vector<uint2>& triBendVerts) {
    triBendEdges.clear();
    triBendVerts.clear();
    triBendEdges.reserve(mesh.number_of_edges());
    triBendVerts.reserve(mesh.number_of_edges());

    auto find_third_vertex = [&](const CGAL_Face_index& face, const CGAL_Vertex_index& src, const CGAL_Vertex_index& tgt) -> int {
        if (face == mesh.null_face()) return -1;
        auto he = mesh.halfedge(face);
        do {
            auto v = mesh.target(he);
            if (v != src && v != tgt) {
                return static_cast<int>(v);
            }
            he = mesh.next(he);
        } while (he != mesh.halfedge(face));
        return -1;
    };

    for (auto edge : mesh.edges()) {
        auto halfedge1 = mesh.halfedge(edge, 0);
        auto halfedge2 = mesh.halfedge(edge, 1);

        bool is_border1 = mesh.is_border(halfedge1);
        bool is_border2 = mesh.is_border(halfedge2);

        if (!is_border1 && !is_border2) {
            auto source = mesh.source(halfedge1);
            auto target = mesh.target(halfedge1);
            unsigned int source_id = static_cast<unsigned int>(source);
            unsigned int target_id = static_cast<unsigned int>(target);

            if (source_id > target_id) std::swap(source_id, target_id);
            triBendEdges.emplace_back(make_uint2(source_id, target_id));

            auto face1 = mesh.face(halfedge1);
            auto face2 = mesh.face(halfedge2);
            int third_vertex1 = find_third_vertex(face1, source, target);
            int third_vertex2 = find_third_vertex(face2, source, target);

            triBendVerts.emplace_back(make_uint2(third_vertex1, third_vertex2));
        }
    }

    size_t n = triBendEdges.size();
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) -> bool {
        if (triBendEdges[a].x != triBendEdges[b].x)
            return triBendEdges[a].x < triBendEdges[b].x;
        return triBendEdges[a].y < triBendEdges[b].y;
    });

    std::vector<uint2> sortedTriBendEdges;
    std::vector<uint2> sortedTriBendVerts;
    sortedTriBendEdges.reserve(n);
    sortedTriBendVerts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        size_t idx = indices[i];
        sortedTriBendEdges.emplace_back(triBendEdges[idx]);
        sortedTriBendVerts.emplace_back(triBendVerts[idx]);
    }

    triBendEdges = std::move(sortedTriBendEdges);
    triBendVerts = std::move(sortedTriBendVerts);
}

















































class Triangle {

public:
	uint64_t key[3];

	Triangle(const uint64_t* p_key)
	{
		key[0] = p_key[0];
		key[1] = p_key[1];
		key[2] = p_key[2];
	}
	Triangle(uint64_t key0, uint64_t key1, uint64_t key2)
	{
		key[0] = key0;
		key[1] = key1;
		key[2] = key2;
	}

	uint64_t operator[](int i) const
	{
		//assert(0 <= i && i <= 2);
		return key[i];
	}

	bool operator<(const Triangle& right) const
	{
		if (key[0] < right.key[0]) {
			return true;
		}
		else if (key[0] == right.key[0]) {
			if (key[1] < right.key[1]) {
				return true;
			}
			else if (key[1] == right.key[1]) {
				if (key[2] < right.key[2]) {
					return true;
				}
			}
		}
		return false;
	}

	bool operator==(const Triangle& right) const
	{
		return key[0] == right[0] && key[1] == right[1] && key[2] == right[2];
	}
};

void split(std::string str, std::vector<std::string>& v, std::string spacer)
{
	int pos1, pos2;
	int len = spacer.length();
	pos1 = 0;
	pos2 = str.find(spacer);
	while (pos2 != std::string::npos)
	{
		v.push_back(str.substr(pos1, pos2 - pos1));
		pos1 = pos2 + len;
		pos2 = str.find(spacer, pos1);
	}
	if (pos1 != str.length())
		v.push_back(str.substr(pos1));
}


bool LoadObjMesh::load_triMesh(const std::string& filename, Scalar scale, Scalar3 transform, int mboundaryType) {
	std::ifstream ifs(filename);
	if (!ifs) {

		fprintf(stderr, "unable to read file %s\n", filename.c_str());
		ifs.close();
		exit(-1);
		return false;
	}
	char buffer[1024];
	std::string line = "";
	int nodeNumber = 0;
	int elementNumber = 0;
	Scalar x, y, z;

	Scalar xmin = DBL_MAX, ymin = DBL_MAX, zmin = DBL_MAX;
	Scalar xmax = -DBL_MAX, ymax = -DBL_MAX, zmax = -DBL_MAX;

	while (getline(ifs, line)) {
		std::string key = line.substr(0, 2);
		if (key.length() <= 1) continue;
		std::stringstream ss(line.substr(2));
		if (key == "v ") {
			ss >> x >> y >> z;
			Scalar3 vertex = make_Scalar3(scale * x + transform.x, scale * y + transform.y, scale * z + transform.z);
			vertexes.push_back(vertex);
			numVertices++;
			__MATHUTILS__::Matrix3x3S constraint;
			__MATHUTILS__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);


            if (mboundaryType == 3) {
                masses.push_back(1);
                __MATHUTILS__::__set_Mat_val(constraint, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            }else {
                masses.push_back(0);
                __MATHUTILS__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
            }

            constraints.push_back(constraint);
			int boundaryType = mboundaryType;
			
			if (mboundaryType == 2) {
				int tid = numVertices - 1;
				targetIndex.push_back(tid);
				// std::cout << "targetindex tid: " << tid << std::endl;
			}

			boundaryTypies.push_back(boundaryType);

			Scalar3 velocity = make_Scalar3(0, 0, 0);
			velocities.push_back(velocity);
			
			// if (xmin > vertex.x) xmin = vertex.x;
			// if (ymin > vertex.y) ymin = vertex.y;
			// if (zmin > vertex.z) zmin = vertex.z;
			// if (xmax < vertex.x) xmax = vertex.x;
			// if (ymax < vertex.y) ymax = vertex.y;
			// if (zmax < vertex.z) zmax = vertex.z;
		}

		// else if (key == "vn") {
		// 	ss >> x >> y >> z;
		// 	Scalar3 normal = make_Scalar3(x, y, z);
		// 	//normals.push_back(normal);
		// }
		else if (key == "f ") {
			if (line.length() >= 1024) {
				printf("[WARN]: skip line due to exceed max buffer length (1024).\n");
				continue;
			}


			std::vector<std::string> fs;

			{
				std::string buf;
				std::stringstream ss(line);
				std::vector<std::string> tokens;
				while (ss >> buf)
					tokens.push_back(buf);

				for (size_t index = 3; index < tokens.size(); index += 1) {
					fs.push_back("f " + tokens[1] + " " + tokens[index - 1] + " " + tokens[index]);
				}
			}

			int uv0, uv1, uv2;

			for (const auto& f : fs) {
				memset(buffer, 0, sizeof(char) * 1024);
				std::copy(f.begin(), f.end(), buffer);

				uint3 faceVertIndex;
				uint3 faceNormalIndex;

				if (sscanf(buffer, "f %d/%d/%d %d/%d/%d %d/%d/%d", &faceVertIndex.x, &uv0, &faceNormalIndex.x,
					&faceVertIndex.y, &uv1, &faceNormalIndex.y,
					&faceVertIndex.z, &uv2, &faceNormalIndex.z) == 9) {

					faceVertIndex.x -= (1 - vertexOffset);
					faceVertIndex.y -= (1 - vertexOffset);
					faceVertIndex.z -= (1 - vertexOffset);
					//triangles.push_back(faceVertIndex);
					//facenormals.push_back(faceNormalIndex);
				}
				else if (sscanf(buffer, "f %d %d %d", &faceVertIndex.x,
					&faceVertIndex.y,
					&faceVertIndex.z) == 3) {
					faceVertIndex.x -= (1 - vertexOffset);
					faceVertIndex.y -= (1 - vertexOffset);
					faceVertIndex.z -= (1 - vertexOffset);
					//triangles.push_back(faceVertIndex);
				}
				else if (sscanf(buffer, "f %d/%d %d/%d %d/%d", &faceVertIndex.x, &uv0,
					&faceVertIndex.y, &uv1,
					&faceVertIndex.z, &uv2) == 6) {
					faceVertIndex.x -= (1 - vertexOffset);
					faceVertIndex.y -= (1 - vertexOffset);
					faceVertIndex.z -= (1 - vertexOffset);
					//triangles.push_back(faceVertIndex);
				}
				if (mboundaryType >= 2) {
					surfFaces.push_back(faceVertIndex);
				}
				else {
					surfFaces.push_back(faceVertIndex);
					triangles.push_back(faceVertIndex);
				}
			}
		}
	}


	numTriElements = triangles.size();
	vertexOffset += numVertices;
	numBoundTargets = targetIndex.size();


	// std::set<std::pair<int, int>> edge_set;
	// std::map<std::pair<int, int>, std::vector<int>> edge_map;
	// std::vector<Eigen::Vector2i> my_edges;

	// for (auto tri : triangles) {
	// 	auto x = tri.x;
	// 	auto y = tri.y;
	// 	auto z = tri.z;
	// 	if (x < y) {
	// 		edge_set.insert(std::make_pair(x, y));
	// 		edge_map[std::make_pair(x, y)].emplace_back(z);
	// 	}
	// 	else {
	// 		edge_set.insert(std::make_pair(y, x));
	// 		edge_map[std::make_pair(y, x)].emplace_back(z);
	// 	}

	// 	if (y < z) {
	// 		edge_set.insert(std::make_pair(y, z));
	// 		edge_map[std::make_pair(y, z)].emplace_back(x);
	// 	}
	// 	else {
	// 		edge_set.insert(std::make_pair(z, y));
	// 		edge_map[std::make_pair(z, y)].emplace_back(x);
	// 	}

	// 	if (x < z) {
	// 		edge_set.insert(std::make_pair(x, z));
	// 		edge_map[std::make_pair(x, z)].emplace_back(y);
	// 	}
	// 	else {
	// 		edge_set.insert(std::make_pair(z, x));
	// 		edge_map[std::make_pair(z, x)].emplace_back(y);
	// 	}
	// }
	// std::vector<std::vector<int>> temp_edges_adj_points;
	// for (auto p : edge_set) {
	// 	if (edge_map[p].size() != 2)continue;
	// 	my_edges.emplace_back(p.first, p.second);
	// 	temp_edges_adj_points.emplace_back(edge_map[std::make_pair(p.first, p.second)]);
	// }
	// if (mboundaryType == 0) {
	// 	for (int i = 0; i < temp_edges_adj_points.size(); i++) {
	// 		triBendEdges.emplace_back(make_uint2(my_edges[i].x(), my_edges[i].y()));
	// 		if (temp_edges_adj_points[i].size() == 2)
	// 			triBendVerts.emplace_back(make_int2(temp_edges_adj_points[i][0], temp_edges_adj_points[i][1]));
	// 		else
	// 			triBendVerts.emplace_back(make_int2(temp_edges_adj_points[i][0], -1));
	// 	}
	// }

	std::cout << "filename in loadmesh: " << filename << std::endl;
	std::cout << "triedges size: " << triBendEdges.size() << std::endl;
	std::cout << "triBendVerts size: " << triBendVerts.size() << std::endl;

	return true;
}




bool LoadObjMesh::load_animation(const std::string& filename, Scalar scale, Scalar3 transform) {
	std::ifstream ifs(filename);
	if (!ifs) {

		fprintf(stderr, "unable to read file %s\n", filename.c_str());
		ifs.close();
		exit(-1);
		return false;
	}
	char buffer[1024];
	std::string line = "";
	Scalar x, y, z;
	targetPos.clear();
	while (getline(ifs, line)) {
		std::string key = line.substr(0, 2);
		if (key.length() <= 1) continue;
		std::stringstream ss(line.substr(2));
        //printf(line.c_str());
        //printf("\n");
		if (key == "v ") {

			ss >> x >> y >> z;
			Scalar3 vertex = make_Scalar3(scale * x + transform.x, scale * y + transform.y, scale * z + transform.z);
			targetPos.push_back(vertex);
		}
	}
    return true;
}


LoadObjMesh::LoadObjMesh() {
	numBoundTargets = 0;
	vertexOffset = 0;
	numVertices = 0;
	numTetElements = 0;
	numTriElements = 0;
	// minConer = make_Scalar3(0, 0, 0);
	// maxConer = make_Scalar3(0, 0, 0);
}


bool LoadObjMesh::load_tetrahedraMesh(const std::string& filename, Scalar scale, Scalar3 position_offset) {

	std::ifstream ifs(filename);
	if (!ifs) {

		fprintf(stderr, "unable to read file %s\n", filename.c_str());
		ifs.close();
		exit(-1);
		return false;
	}

	Scalar x, y, z;
	int index0, index1, index2, index3;
	std::string line = "";
	int nodeNumber = 0;
	int elementNumber = 0;
	while (getline(ifs, line)) {
		if (line.length() <= 1) continue;
        if (line.substr(1, 5) == "Nodes") {
			getline(ifs, line);
			nodeNumber = atoi(line.c_str());
			numVertices += nodeNumber;

			// Scalar xmin = DBL_MAX, ymin = DBL_MAX, zmin = DBL_MAX;
			// Scalar xmax = -DBL_MAX, ymax = -DBL_MAX, zmax = -DBL_MAX;
			for (int i = 0; i < nodeNumber; i++) {
				getline(ifs, line);
				std::vector<std::string> nodePos;
				std::string spacer = " ";
				split(line, nodePos, spacer);
				x = atof(nodePos[1].c_str());
				y = atof(nodePos[2].c_str());
				z = atof(nodePos[3].c_str());
				Scalar3 d_velocity = make_Scalar3(0, 0, 0);
				Scalar3 vertex = make_Scalar3(scale * x - position_offset.x, scale * y - position_offset.y, scale * z - position_offset.z);
				//Matrix3d Constraint; Constraint.setIdentity();
				//Vector3d force = Vector3d(0, 0, 0);
				Scalar3 velocity = make_Scalar3(0, 0, 0);
				Scalar3 d_pos = make_Scalar3(0, 0, 0);
				Scalar mass = 0;
				int boundaryType = 0;
				boundaryTypies.push_back(boundaryType);
				vertexes.push_back(vertex);
				//forces.push_back(force);
				velocities.push_back(velocity);
				//Constraints.push_back(Constraint);
				//d_velocities.push_back(d_velocity);
				masses.push_back(mass);
				//isDelete.push_back(false);
				//d_positions.push_back(d_pos);
				//externalForce.push_back(Vector3d(0, 0, 0));


				__MATHUTILS__::Matrix3x3S constraint;
				__MATHUTILS__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);

				constraints.push_back(constraint);

				// if (xmin > vertex.x) xmin = vertex.x;
				// if (ymin > vertex.y) ymin = vertex.y;
				// if (zmin > vertex.z) zmin = vertex.z;
				// if (xmax < vertex.x) xmax = vertex.x;
				// if (ymax < vertex.y) ymax = vertex.y;
				// if (zmax < vertex.z) zmax = vertex.z;
			}
			// minTConer = make_Scalar3(xmin, ymin, zmin);
			// maxTConer = make_Scalar3(xmax, ymax, zmax);
		}

        if (line.substr(1, 8) == "Elements") {
			getline(ifs, line);
			elementNumber = atoi(line.c_str());
			numTetElements += elementNumber;
			for (int i = 0; i < elementNumber; i++) {
				getline(ifs, line);

				std::vector<std::string> elementIndexex;
				std::string spacer = " ";
				split(line, elementIndexex, spacer);
				index0 = atoi(elementIndexex[3].c_str()) - 1;
				index1 = atoi(elementIndexex[4].c_str()) - 1;
				index2 = atoi(elementIndexex[5].c_str()) - 1;
				index3 = atoi(elementIndexex[6].c_str()) - 1;

				uint4 tetrahedra;
				tetrahedra.x = index0 + vertexOffset;
				tetrahedra.y = index1 + vertexOffset;
				tetrahedra.z = index2 + vertexOffset;
				tetrahedra.w = index3 + vertexOffset;
				tetrahedras.push_back(tetrahedra);

			}
			break;
		}
	}
	ifs.close();

	// Scalar boxTVolum = (maxTConer.x - minTConer.x) * (maxTConer.y - minTConer.y) * (maxTConer.z - minTConer.z);
	// Scalar boxVolum = (maxConer.x - minConer.x) * (maxConer.y - minConer.y) * (maxConer.z - minConer.z);

	// if (boxTVolum > boxVolum) {
	// 	maxConer = maxTConer;
	// 	minConer = minTConer;
	// }
	
	//V_prev = vertexes;
	vertexOffset = numVertices;


	return true;
}


int LoadObjMesh::getVertNeighbors() {
	vertNeighbors.resize(numVertices);
	std::set<std::pair<uint64_t, uint64_t>> SFEdges_set;
	for (int i = 0; i < numTetElements; i++) {
		const auto& edgI4 = tetrahedras[i];
		uint64_t edgI[4] = { edgI4.x,  edgI4.y ,edgI4.z ,edgI4.w };
		if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[0], edgI[1])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[1], edgI[0])) == SFEdges_set.end()) {
			SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[0], edgI[1]));
		}
		if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[0], edgI[2])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[2], edgI[0])) == SFEdges_set.end()) {
			SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[0], edgI[2]));
		}
		if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[0], edgI[3])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[3], edgI[0])) == SFEdges_set.end()) {
			SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[0], edgI[3]));
		}
		if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[1], edgI[2])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[2], edgI[1])) == SFEdges_set.end()) {
			SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[1], edgI[2]));
		}
		if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[1], edgI[3])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[3], edgI[1])) == SFEdges_set.end()) {
			SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[1], edgI[3]));
		}
		if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[2], edgI[3])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[3], edgI[2])) == SFEdges_set.end()) {
			SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[2], edgI[3]));
		}
	}

	for (const auto& cTri : triangles) {
		for (int i = 0; i < 3; i++) {
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y, cTri.x)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x, cTri.y)) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.x, cTri.y));
			}
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z, cTri.y)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y, cTri.z)) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.y, cTri.z));
			}
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x, cTri.z)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z, cTri.x)) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.z, cTri.x));
			}
		}
	}


	for (const auto& edgI : SFEdges_set) {
		vertNeighbors[edgI.first].push_back(edgI.second);
		vertNeighbors[edgI.second].push_back(edgI.first);
	}
	//neighborStart.push_back(0);
	neighborNum.resize(numVertices);
	int offset = 0;
	for (int i = 0; i < numVertices; i++) {
		for (int j = 0; j < vertNeighbors[i].size(); j++) {
			neighborList.push_back(vertNeighbors[i][j]);
		}

		neighborStart.push_back(offset);

		offset += vertNeighbors[i].size();
		neighborNum[i] = vertNeighbors[i].size();
	}

	return neighborStart[numVertices - 1] + neighborNum[numVertices - 1];
}

void LoadObjMesh::getTetSurface() {
	uint64_t length = numVertices;
	auto triangle_hash = [&](const Triangle& tri) {
		return length * (length * tri[0] + tri[1]) + tri[2];
	};

	//std::vector<Vector4i> surface;
	std::unordered_map<Triangle, uint64_t, decltype(triangle_hash)> tri2Tet(4 * numTetElements, triangle_hash);
	for (int i = 0;i < numTetElements;i++) {

		const auto& triI4 = tetrahedras[i];
		uint64_t triI[4] = { triI4.x,  triI4.y ,triI4.z ,triI4.w };
		for (int j = 0;j < 4;j++) {
			const Triangle& triVInd = Triangle(triI[j % 4], triI[(1 + j) % 4], triI[(2 + j) % 4]);
			if (tri2Tet.find(Triangle(triVInd[0], triVInd[1], triVInd[2])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = numTetElements + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[0], triVInd[2], triVInd[1])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[0], triVInd[2], triVInd[1])] = numTetElements + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[1], triVInd[0], triVInd[2])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[1], triVInd[0], triVInd[2])] = numTetElements + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[1], triVInd[2], triVInd[0])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[1], triVInd[2], triVInd[0])] = numTetElements + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[2], triVInd[0], triVInd[1])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[2], triVInd[0], triVInd[1])] = numTetElements + 1;
			}
			else if (tri2Tet.find(Triangle(triVInd[2], triVInd[1], triVInd[0])) != tri2Tet.end()) {
				tri2Tet[Triangle(triVInd[2], triVInd[1], triVInd[0])] = numTetElements + 1;
			}
			else {
				tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = i;
			}
		}
	}

	for (const auto& triI : tri2Tet) {
		const uint64_t& tetId = triI.second;
		const Triangle& triVInd = triI.first;
		if (tetId < numTetElements) {
			Scalar3 vec1 = __MATHUTILS__::__minus(vertexes[triVInd[1]], vertexes[triVInd[0]]);
			Scalar3 vec2 = __MATHUTILS__::__minus(vertexes[triVInd[2]], vertexes[triVInd[0]]);
			int id3 = 0;

			if (tetrahedras[tetId].x != triVInd[0]
				&& tetrahedras[tetId].x != triVInd[1]
				&& tetrahedras[tetId].x != triVInd[2]) {
				id3 = tetrahedras[tetId].x;
			}
			else if (tetrahedras[tetId].y != triVInd[0]
				&& tetrahedras[tetId].y != triVInd[1]
				&& tetrahedras[tetId].y != triVInd[2]) {
				id3 = tetrahedras[tetId].y;
			}
			else if (tetrahedras[tetId].z != triVInd[0]
				&& tetrahedras[tetId].z != triVInd[1]
				&& tetrahedras[tetId].z != triVInd[2]) {
				id3 = tetrahedras[tetId].z;
			}
			else if (tetrahedras[tetId].w != triVInd[0]
				&& tetrahedras[tetId].w != triVInd[1]
				&& tetrahedras[tetId].w != triVInd[2]) {
				id3 = tetrahedras[tetId].w;
			}


			Scalar3 vec3 = __MATHUTILS__::__minus(vertexes[id3], vertexes[triVInd[0]]);
			Scalar3 n = __MATHUTILS__::__v_vec_cross(vec1, vec2);
			if (__MATHUTILS__::__v_vec_dot(n, vec3) > 0) {
				surfId2TetId.push_back(tetId);
				surfFaces.push_back(make_uint3(triVInd[0], triVInd[2], triVInd[1]));
			}
			else {
				surfId2TetId.push_back(tetId);
				surfFaces.push_back(make_uint3(triVInd[0], triVInd[1], triVInd[2]));
			}
		}
	}



}

void LoadObjMesh::getSurface() {

	std::vector<bool> flag(numVertices, false);
	for (const auto& cTri : surfFaces) {

		if (!flag[cTri.x]) {
			surfVerts.push_back(cTri.x);
			flag[cTri.x] = true;
		}
		if (!flag[cTri.y]) {
			surfVerts.push_back(cTri.y);
			flag[cTri.y] = true;
		}
		if (!flag[cTri.z]) {
			surfVerts.push_back(cTri.z);
			flag[cTri.z] = true;
		}

	}

	std::set<std::pair<uint64_t, uint64_t>> SFEdges_set;
	for (const auto& cTri : surfFaces) {
		for (int i = 0;i < 3;i++) {
			for (int i = 0;i < 3;i++) {
				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y, cTri.x)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x, cTri.y)) == SFEdges_set.end()) {
					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.x, cTri.y));
				}
				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z, cTri.y)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y, cTri.z)) == SFEdges_set.end()) {
					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.y, cTri.z));
				}
				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x, cTri.z)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z, cTri.x)) == SFEdges_set.end()) {
					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.z, cTri.x));
				}
			}
		}
	}


	std::vector<std::pair<uint64_t, uint64_t>> tempEdge = std::vector<std::pair<uint64_t, uint64_t>>(SFEdges_set.begin(), SFEdges_set.end());
	for (int i = 0;i < tempEdge.size();i++) {
		surfEdges.push_back(make_uint2(tempEdge[i].first, tempEdge[i].second));
	}
}


bool LoadObjMesh::load_soft_animation(const std::string& filename, Scalar scale, Scalar3 transform, std::vector<Scalar3>& softTargetPos) {
	std::ifstream ifs(filename);
	if (!ifs) {

		fprintf(stderr, "unable to read file %s\n", filename.c_str());
		ifs.close();
		exit(-1);
		return false;
	}
	char buffer[1024];
	std::string line = "";
	Scalar x, y, z;
	softTargetPos.clear();
	while (getline(ifs, line)) {
		std::string key = line.substr(0, 2);
		if (key.length() <= 1) continue;
		std::stringstream ss(line.substr(2));
        //printf(line.c_str());
        //printf("\n");
		if (key == "v ") {
			ss >> x >> y >> z;
			Scalar3 vertex = make_Scalar3(scale * x + transform.x, scale * y + transform.y, scale * z + transform.z);
			// std::cout << "targetPos load_mesh: " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
			softTargetPos.push_back(vertex);
		}
	}
	
	ifs.close();
    return true;
}



