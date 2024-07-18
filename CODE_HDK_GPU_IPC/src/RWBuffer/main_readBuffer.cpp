
#include "main_readBuffer.hpp"

#include "UTILS/GeometryManager.hpp"
#include "CCD/SortMesh.cuh"
#include "CCD/LBVH.cuh"

namespace FIRSTFRAME {
	static bool hou_initialized = false;
	static int collision_detection_buff_scale = 1;

};

const SIM_DopDescription* GAS_Read_Buffer::getDopDescription() {
	static PRM_Template templateList[] = {
		PRM_Template()
	};

	static SIM_DopDescription dopDescription(
		true,
		"gas_read_buffer", // internal name of the dop
		"Gas Read Buffer", // label of the dop
		"GasReadBuffer", // template list for generating the dop
		classname(),
		templateList);

	setGasDescription(dopDescription);

	return &dopDescription;
}

GAS_Read_Buffer::GAS_Read_Buffer(const SIM_DataFactory* factory) : BaseClass(factory) {}


GAS_Read_Buffer::~GAS_Read_Buffer() {
	FIRSTFRAME::hou_initialized = false;
	GeometryManager::totallyfree();
}

bool GAS_Read_Buffer::solveGasSubclass(SIM_Engine& engine,
										SIM_Object* object,
										SIM_Time time,
										SIM_Time timestep) {

	const SIM_Geometry *geo = object->getGeometry();
	CHECK_ERROR_SOLVER(geo != NULL, "Failed to get readBuffer geometry object")
	GU_DetailHandleAutoReadLock readlock(geo->getGeometry());
	CHECK_ERROR_SOLVER(readlock.isValid(), "Failed to get readBuffer geometry detail");
	const GU_Detail *gdp = readlock.getGdp();
	CHECK_ERROR_SOLVER(!gdp->isEmpty(), "readBuffer Geometry is empty");
	auto &instance = GeometryManager::instance;

	if (!FIRSTFRAME::hou_initialized) {
		
		PRINT_BLUE("running_initialized");

		if (!instance) {
			instance = std::unique_ptr<GeometryManager>(new GeometryManager());
		}
		CHECK_ERROR_SOLVER(instance, "geometry instance not initialize");

		transferPTAttribTOCUDA(geo, gdp);
		transferPRIMAttribTOCUDA(geo, gdp);
		transferDTAttribTOCUDA(geo, gdp);
		transferOtherTOCUDA();


		loadSIMParams();
		initSIMFEM();
		initSIMBVH();
		initSIMIPC();

		CUDA_SAFE_CALL(cudaMemcpy(instance->cudaRestTetPos, instance->cudaOriginTetPos, instance->tetPos.rows() * sizeof(double3), cudaMemcpyDeviceToDevice));

		buildSIMBVH();


		FIRSTFRAME::hou_initialized = true;
	}

	return true;
}


void GAS_Read_Buffer::transferPTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferPTAttribTOCUDA geoinstance not initialized");
	int num_points = gdp->getNumPoints();

	// position velocity mass
	auto &tetpos = instance->tetPos;
	auto &tetvel = instance->tetVel;
	auto &tetmass = instance->tetMass;
	tetpos.resize(num_points, Eigen::NoChange);
	tetvel.resize(num_points, Eigen::NoChange);
	tetmass.resize(num_points);

	auto &cons = instance->constraints;
	cons.resize(num_points);

	// boundary type
	auto &boundarytype = instance->boundaryTypies;
	boundarytype.resize(num_points);

	GA_ROHandleV3D velHandle(gdp, GA_ATTRIB_POINT, "v");
	GA_ROHandleD massHandle(gdp, GA_ATTRIB_POINT, "mass");
	CHECK_ERROR(velHandle.isValid() && massHandle.isValid(), "Failed to get velocity and mass attributes");

	double xmin = DBL_MAX, ymin = DBL_MAX, zmin = DBL_MAX;
	double xmax = -DBL_MAX, ymax = -DBL_MAX, zmax = -DBL_MAX;

	GA_Offset ptoff;
	GA_FOR_ALL_PTOFF(gdp, ptoff) {
		UT_Vector3D pos3 = gdp->getPos3D(ptoff);
		UT_Vector3D vel3 = velHandle.get(ptoff);
		tetpos.row(ptoff) << pos3.x(), pos3.y(), pos3.z();
		tetvel.row(ptoff) << vel3.x(), vel3.y(), vel3.z();
		tetmass(ptoff) = massHandle.get(ptoff);

		MATHUTILS::Matrix3x3d constraint;
		MATHUTILS::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
		cons[ptoff] = constraint;

		if (xmin > pos3.x()) xmin = pos3.x();
		if (ymin > pos3.y()) ymin = pos3.y();
		if (zmin > pos3.z()) zmin = pos3.z();
		if (xmax < pos3.x()) xmax = pos3.x();
		if (ymax < pos3.y()) ymax = pos3.y();
		if (zmax < pos3.z()) zmax = pos3.z();

		boundarytype(ptoff) = 0;

	}
	CHECK_ERROR(ptoff==num_points, "Failed to get all points");

	CUDAMallocSafe(instance->cudaTetPos, num_points);
	CUDAMallocSafe(instance->cudaTetVel, num_points);
	CUDAMallocSafe(instance->cudaTetMass, num_points);
	copyToCUDASafe(instance->tetPos, instance->cudaTetPos);
	copyToCUDASafe(instance->tetVel, instance->cudaTetVel);
	copyToCUDASafe(instance->tetMass, instance->cudaTetMass);

	CUDAMallocSafe(instance->cudaBoundaryType, num_points);
	copyToCUDASafe(instance->boundaryTypies, instance->cudaBoundaryType);

	CUDAMallocSafe(instance->cudaOriginTetPos, num_points);
	copyToCUDASafe(instance->tetPos, instance->cudaOriginTetPos);
	CUDAMallocSafe(instance->cudaRestTetPos, num_points);
	copyToCUDASafe(instance->tetPos, instance->cudaRestTetPos);

	CUDAMallocSafe(instance->cudaConstraints, num_points);
	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaConstraints, instance->constraints.data(), num_points * sizeof(MATHUTILS::Matrix3x3d), cudaMemcpyHostToDevice));


	instance->minCorner = make_double3(xmin, ymin, zmin);
	instance->maxCorner = make_double3(xmax, ymax, zmax);

}


void GAS_Read_Buffer::transferPRIMAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferPRIMAttribTOCUDA geoinstance not initialized");

	if (GeometryManager::instance->tetElement.rows() == gdp->getNumPrimitives()) return;

	auto &tetEle = instance->tetElement;
	tetEle.resize(gdp->getNumPrimitives(), Eigen::NoChange);


	GA_Offset primoff;
	GA_FOR_ALL_PRIMOFF(gdp, primoff) {
		const GA_Primitive* prim = gdp->getPrimitive(primoff);
		for (int i = 0; i < prim->getVertexCount(); ++i) {
			GA_Offset vtxoff = prim->getVertexOffset(i);
			GA_Offset ptoff = gdp->vertexPoint(vtxoff);
			tetEle(primoff, i) = static_cast<int>(gdp->pointIndex(ptoff));
		}
	}
	CHECK_ERROR(primoff==gdp->getNumPrimitives(), "Failed to get all primitives");

	CUDAMallocSafe(instance->cudaTetElement, tetEle.rows());
	copyToCUDASafe(instance->tetElement, instance->cudaTetElement);

}


void GAS_Read_Buffer::transferDTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferDTAttribTOCUDA geoinstance not initialized");

	// Read surf_verts attribute
	GA_ROHandleIA surfVertHandle(gdp, GA_ATTRIB_DETAIL, "surf_verts");
	CHECK_ERROR(surfVertHandle.isValid(), "Failed to get surf_verts attribute");
	UT_IntArray surf_verts;
	surfVertHandle.get(0, surf_verts);
	int numSurfVerts = surf_verts.size();
	Eigen::VectorXi surfVert(numSurfVerts, 3);
	for (int i = 0; i < numSurfVerts; ++i) {
		surfVert[i] = surf_verts[i];
	}

	// Read surf_edges attribute
	GA_ROHandleIA surfEdgeHandle(gdp, GA_ATTRIB_DETAIL, "surf_edges");
	CHECK_ERROR(surfEdgeHandle.isValid(), "Failed to get surf_edges attribute");
	UT_IntArray surf_edges;
	surfEdgeHandle.get(0, surf_edges);
	int numSurfEdges = surf_edges.size() / 2;
	Eigen::MatrixXi surfEdge(numSurfEdges, 2);
	for (int i = 0; i < numSurfEdges; ++i) {
		surfEdge(i, 0) = surf_edges[i * 2];
		surfEdge(i, 1) = surf_edges[i * 2 + 1];
	}

	// Read surf_faces attribute
	GA_ROHandleIA surfFaceHandle(gdp, GA_ATTRIB_DETAIL, "surf_faces");
	CHECK_ERROR(surfFaceHandle.isValid(), "Failed to get surf_faces attribute");
	UT_IntArray surf_faces;
	surfFaceHandle.get(0, surf_faces);
	int numSurfFaces = surf_faces.size() / 3;
	Eigen::MatrixXi surfFace(numSurfFaces, 3);
	for (int i = 0; i < numSurfFaces; ++i) {
		surfFace(i, 0) = surf_faces[i * 3];
		surfFace(i, 1) = surf_faces[i * 3 + 1];
		surfFace(i, 2) = surf_faces[i * 3 + 2];
	}

	instance->surfVert = surfVert;
	instance->surfFace = surfFace;
	instance->surfEdge = surfEdge;
	CUDAMallocSafe(instance->cudaSurfVert, surfVert.rows());
	CUDAMallocSafe(instance->cudaSurfFace, surfFace.rows());
	CUDAMallocSafe(instance->cudaSurfEdge, surfEdge.rows());
	copyToCUDASafe(instance->surfVert, instance->cudaSurfVert);
	copyToCUDASafe(instance->surfFace, instance->cudaSurfFace);
	copyToCUDASafe(instance->surfEdge, instance->cudaSurfEdge);

}


void GAS_Read_Buffer::transferOtherTOCUDA() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferOtherTOCUDA geoinstance not initialized");

	int numVerts = instance->tetPos.rows();
	int numElems = instance->tetElement.rows();
	int maxNumbers = numVerts > numElems ? numVerts : numElems; 
	CUDAMallocSafe(instance->cudaMortonCodeHash, maxNumbers);
	CUDAMallocSafe(instance->cudaSortIndex, maxNumbers);
	CUDAMallocSafe(instance->cudaSortMapVertIndex, numVerts);
	CUDAMallocSafe(instance->cudaDmInverses, numElems);
	
	CUDAMallocSafe(instance->cudaTempBoundaryType, numVerts);
	CUDAMallocSafe(instance->cudaTempDouble, maxNumbers);
	CUDAMallocSafe(instance->cudaTempMat3x3, maxNumbers);

	int triangle_num = 0;
	CUDAMallocSafe(instance->cudaTriElement, triangle_num);

	//TODO: DmInverses
	// copyToCUDASafe(instance->cudaDmInverses, );
	// copyToCUDASafe(instance->cudaTetVolume, );


	std::cout << "numVerts~~" << numVerts << std::endl;
	std::cout << "numElems~~" << numElems << std::endl;
	std::cout << "maxnumbers~~" << maxNumbers << std::endl;



}



void GAS_Read_Buffer::loadSIMParams() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "loadSIMParams geoinstance not initialized");

	instance->density = 1e3;
	instance->YoungModulus = 1e5;
	instance->PoissonRate = 0.49;
	instance->lengthRateLame = instance->YoungModulus / (2 * (1 + instance->PoissonRate));
	instance->volumeRateLame = instance->YoungModulus * instance->PoissonRate / ((1 + instance->PoissonRate) * (1 - 2 * instance->PoissonRate));
	instance->lengthRate = 4 * instance->lengthRateLame / 3;
	instance->volumeRate = instance->volumeRateLame + 5 * instance->lengthRateLame / 6;
	instance->frictionRate = 0.4;
	instance->clothThickness = 1e-3;
	instance->clothYoungModulus = 1e6;
	instance->stretchStiff = instance->clothYoungModulus / (2 * (1 + instance->PoissonRate));
	instance->shearStiff = instance->stretchStiff * 0.05;
	instance->clothDensity = 2e2;
	instance->softMotionRate = 1e0;
	// instance->bendStiff = 3e-4;
	instance->Newton_solver_threshold = 1e-1;
	instance->pcg_threshold = 1e-3;
	instance->IPC_dt = 1e-2;
	// instance->relative_dhat = 1e-3;
	// instance->bendStiff = instance->clothYoungModulus * pow(instance->clothThickness, 3) / (24 * (1 - instance->PoissonRate * instance->PoissonRate));
	instance->shearStiff = 0.03 * instance->stretchStiff;

	instance->MAX_CCD_COLLITION_PAIRS_NUM = 1 * FIRSTFRAME::collision_detection_buff_scale * (((double)(instance->surfFace.rows() * 15 + instance->surfEdge.rows() * 10)) * std::max((instance->IPC_dt / 0.01), 2.0));
	instance->MAX_COLLITION_PAIRS_NUM = (instance->surfVert.rows() * 3 + instance->surfEdge.rows() * 2) * 3 * FIRSTFRAME::collision_detection_buff_scale;

	std::cout << "arrived here ccd~" << instance->MAX_CCD_COLLITION_PAIRS_NUM << std::endl;
	std::cout << "arrived here collision~" << instance->MAX_COLLITION_PAIRS_NUM << std::endl;

}



void GAS_Read_Buffer::initSIMFEM() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "loadSIMParams geoinstance not initialized");

	double sumMass = 0;
	double sumVolume = 0;

	auto &tetpos = instance->tetPos;
	auto &tetele = instance->tetElement;
	for (int i = 0; i < instance->tetElement.rows(); i++) {
		int idx0 = tetele(i, 0);
        int idx1 = tetele(i, 1);
        int idx2 = tetele(i, 2);
        int idx3 = tetele(i, 3);
        const double3 &v0 = make_double3(tetpos(idx0, 0), tetpos(idx0, 1), tetpos(idx0, 2));
        const double3 &v1 = make_double3(tetpos(idx1, 0), tetpos(idx1, 1), tetpos(idx1, 2));
        const double3 &v2 = make_double3(tetpos(idx2, 0), tetpos(idx2, 1), tetpos(idx2, 2));
        const double3 &v3 = make_double3(tetpos(idx3, 0), tetpos(idx3, 1), tetpos(idx3, 2));
		double vlm = MATHUTILS::__calculateVolume(v0, v1, v2, v3);
	}


}

void GAS_Read_Buffer::initSIMBVH() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "initSIMBVH geoinstance not initialized");

	if (!instance->LBVH_F_ptr) {
		instance->LBVH_F_ptr = std::make_unique<LBVH_F>();
	}
	if (!instance->LBVH_E_ptr) {
		instance->LBVH_E_ptr = std::make_unique<LBVH_E>();
	}

	CUDAMallocSafe(instance->cudaCollisionPairs, instance->MAX_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaCCDCollisionPairs, instance->MAX_CCD_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaMatIndex, instance->MAX_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaCPNum, 5); // TODO: why 5?


	instance->LBVH_E_ptr->init(
		instance->cudaBoundaryType,
		instance->cudaTetPos,
		instance->cudaRestTetPos,
		instance->cudaSurfEdge,
		instance->cudaCollisionPairs,
		instance->cudaCCDCollisionPairs,
		instance->cudaCPNum,
		instance->cudaMatIndex,
		instance->surfEdge.rows(),
		instance->surfVert.rows()
	);

	instance->LBVH_F_ptr->init(
		instance->cudaBoundaryType, 
		instance->cudaTetPos,
		instance->cudaSurfFace,
		instance->cudaSurfVert,
		instance->cudaCollisionPairs,
		instance->cudaCCDCollisionPairs, 
		instance->cudaCPNum, 
		instance->cudaMatIndex, 
		instance->surfFace.rows(), 
		instance->surfVert.rows()
	);

	// calcuate Morton Code and sort MC together with face index
	SortMesh::sortMesh(instance, instance->LBVH_F_ptr);

}

void GAS_Read_Buffer::initSIMIPC() {

}


void GAS_Read_Buffer::buildSIMBVH() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "buildSIMBVH geoinstance not initialized");

	instance->LBVH_F_ptr->Construct();
	instance->LBVH_E_ptr->Construct();


}
