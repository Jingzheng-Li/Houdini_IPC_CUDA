
#include "main_readBuffer.hpp"

#include "UTILS/GeometryManager.hpp"
#include "LBVH/SortMesh.cuh"

#include "LBVH/LBVH.cuh"
#include "PCG/PCGSolver.cuh"
#include "FEMEnergy.cuh"

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

	cudaError_t cudaStatus = cudaSetDevice(0);
	CHECK_ERROR_SOLVER(cudaStatus==cudaSuccess, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

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
		buildSIMBVH();
		initSIMIPC();
		buildSIMCP();

		FIRSTFRAME::hou_initialized = true;
	}

	return true;
}


void GAS_Read_Buffer::transferPTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferPTAttribTOCUDA geoinstance not initialized");
	int num_points = gdp->getNumPoints();

	// position velocity mass
	auto &tetpos = instance->vertPos;
	auto &tetvel = instance->vertVel;
	auto &tetmass = instance->vertMass;
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
	int ptidx = 0;
	GA_FOR_ALL_PTOFF(gdp, ptoff) {
		UT_Vector3D pos3 = gdp->getPos3D(ptoff);
		UT_Vector3D vel3 = velHandle.get(ptoff);
		tetpos.row(ptidx) << pos3.x(), pos3.y(), pos3.z();
		tetvel.row(ptidx) << vel3.x(), vel3.y(), vel3.z();
		tetmass(ptidx) = massHandle.get(ptoff);

		MATHUTILS::Matrix3x3d constraint;
		MATHUTILS::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
		cons[ptidx] = constraint;

		if (xmin > pos3.x()) xmin = pos3.x();
		if (ymin > pos3.y()) ymin = pos3.y();
		if (zmin > pos3.z()) zmin = pos3.z();
		if (xmax < pos3.x()) xmax = pos3.x();
		if (ymax < pos3.y()) ymax = pos3.y();
		if (zmax < pos3.z()) zmax = pos3.z();

		boundarytype(ptidx) = 0;
		ptidx++;

	}
	CHECK_ERROR(ptidx==num_points, "Failed to get all points");

	CUDAMallocSafe(instance->cudaVertPos, num_points);
	CUDAMallocSafe(instance->cudaVertVel, num_points);
	CUDAMallocSafe(instance->cudaVertMass, num_points);
	CUDAMemcpyHToDSafe(instance->vertPos, instance->cudaVertPos);
	CUDAMemcpyHToDSafe(instance->vertVel, instance->cudaVertVel);
	CUDAMemcpyHToDSafe(instance->vertMass, instance->cudaVertMass);

	CUDAMallocSafe(instance->cudaBoundaryType, num_points);
	CUDAMemcpyHToDSafe(instance->boundaryTypies, instance->cudaBoundaryType);

	CUDAMallocSafe(instance->cudaOriginTetPos, num_points);
	CUDAMemcpyHToDSafe(instance->vertPos, instance->cudaOriginTetPos);
	CUDAMallocSafe(instance->cudaRestTetPos, num_points);
	CUDAMemcpyHToDSafe(instance->vertPos, instance->cudaRestTetPos);

	CUDAMallocSafe(instance->cudaConstraints, num_points);
	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaConstraints, instance->constraints.data(), num_points * sizeof(MATHUTILS::Matrix3x3d), cudaMemcpyHostToDevice));

	instance->minCorner = make_double3(xmin, ymin, zmin);
	instance->maxCorner = make_double3(xmax, ymax, zmax);

	instance->vecVertPos = MATHUTILS::__convertEigenToVector<double3>(instance->vertPos);
	instance->vecVertVel = MATHUTILS::__convertEigenToVector<double3>(instance->vertVel);
	CHECK_ERROR(instance->vecVertPos.size()==instance->vertPos.rows(), "not init vecVertPos correctly");
	CHECK_ERROR(instance->vecVertVel.size()==instance->vertVel.rows(), "not init vecVertVel correctly");


}


void GAS_Read_Buffer::transferPRIMAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferPRIMAttribTOCUDA geoinstance not initialized");

	auto &tetEle = instance->tetElement;
	tetEle.resize(gdp->getNumPrimitives(), Eigen::NoChange);

	GA_Offset primoff;
	int primidx = 0;
	GA_FOR_ALL_PRIMOFF(gdp, primoff) {
		const GA_Primitive* prim = gdp->getPrimitive(primoff);
		for (int i = 0; i < prim->getVertexCount(); ++i) {
			GA_Offset vtxoff = prim->getVertexOffset(i);
			GA_Offset ptoff = gdp->vertexPoint(vtxoff);
			tetEle(primidx, i) = static_cast<int>(gdp->pointIndex(ptoff));
		}
		primidx++;
	}
	CHECK_ERROR(primidx==gdp->getNumPrimitives(), "Failed to get all primitives");

	CUDAMallocSafe(instance->cudaTetElement, tetEle.rows());
	CUDAMemcpyHToDSafe(instance->tetElement, instance->cudaTetElement);

	instance->vectetElement = MATHUTILS::__convertEigenToUintVector<uint4>(instance->tetElement);
	CHECK_ERROR(instance->vectetElement.size()==instance->tetElement.rows(), "not init tetElement correctly");

	instance->tetVolume.resize(tetEle.rows());
	for (int i = 0; i < tetEle.rows(); i++) {
		instance->tetVolume[i] = MATHUTILS::__calculateVolume(instance->vecVertPos.data(), instance->vectetElement[i]);
	}
	CUDAMallocSafe(instance->cudaTetVolume, instance->tetVolume.rows());
	CUDAMemcpyHToDSafe(instance->tetVolume, instance->cudaTetVolume);

}


void GAS_Read_Buffer::transferDTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferDTAttribTOCUDA geoinstance not initialized");

	Eigen::VectorXi &surfverts = instance->surfVert;
	Eigen::MatrixX3i &surffaces = instance->surfFace;
	Eigen::MatrixX2i &surfedges = instance->surfEdge;
	MATHUTILS::__getSurface(surfverts, surffaces, surfedges, instance->vertPos, instance->tetElement);

	CUDAMallocSafe(instance->cudaSurfVert, surfverts.rows());
	CUDAMallocSafe(instance->cudaSurfFace, surffaces.rows());
	CUDAMallocSafe(instance->cudaSurfEdge, surfedges.rows());
	CUDAMemcpyHToDSafe(instance->surfVert, instance->cudaSurfVert);
	CUDAMemcpyHToDSafe(instance->surfFace, instance->cudaSurfFace);
	CUDAMemcpyHToDSafe(instance->surfEdge, instance->cudaSurfEdge);
	instance->numSurfVerts = surfverts.rows();
	instance->numSurfEdges = surfedges.rows();
	instance->numSurfFaces = surffaces.rows();

}


void GAS_Read_Buffer::transferOtherTOCUDA() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferOtherTOCUDA geoinstance not initialized");

	instance->numVertices = instance->vertPos.rows();
	instance->numElements = instance->tetElement.rows();
	int &numVerts = instance->numVertices;
	int &numElems = instance->numElements;
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

	instance->IPC_dt = 0.01;
	instance->MAX_CCD_COLLITION_PAIRS_NUM = 1 * FIRSTFRAME::collision_detection_buff_scale * (((double)(instance->surfFace.rows() * 15 + instance->surfEdge.rows() * 10)) * std::max((instance->IPC_dt / 0.01), 2.0));
	instance->MAX_COLLITION_PAIRS_NUM = (instance->surfVert.rows() * 3 + instance->surfEdge.rows() * 2) * 3 * FIRSTFRAME::collision_detection_buff_scale;

	CHECK_ERROR(instance->MAX_CCD_COLLITION_PAIRS_NUM > 0, "MAX_CCD_COLLITION_PAIRS_NUM is 0, this is incorrect");
	CHECK_ERROR(instance->MAX_COLLITION_PAIRS_NUM > 0, "MAX_COLLITION_PAIRS_NUM is 0, this is incorrect");

	CUDAMallocSafe(instance->cudaCollisionPairs, instance->MAX_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaCCDCollisionPairs, instance->MAX_CCD_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaMatIndex, instance->MAX_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaCPNum, 5);
	CUDAMallocSafe(instance->cudaGPNum, 1);
	CUDAMallocSafe(instance->cudaEnvCollisionPairs, numVerts);
	CUDAMallocSafe(instance->cudaGroundNormal, 5);
	CUDAMallocSafe(instance->cudaGroundOffset, 5);
	double h_offset[5] = {-1, -1, 1, -1, 1};
	double3 H_normal[5];
	H_normal[0] = make_double3(0, 1, 0);
    H_normal[1] = make_double3(1, 0, 0);
    H_normal[2] = make_double3(-1, 0, 0);
    H_normal[3] = make_double3(0, 0, 1);
    H_normal[4] = make_double3(0, 0, -1);
	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaGroundOffset, &h_offset, 5 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaGroundNormal, &H_normal, 5 * sizeof(double3), cudaMemcpyHostToDevice));

	CUDAMallocSafe(instance->cudaXTilta, instance->numVertices);
	CUDAMallocSafe(instance->cudaFb, instance->numVertices);

	CUDAMallocSafe(instance->cudaTempDouble3Mem, instance->numVertices);


	instance->softNum = 0;
	instance->numTriElements = 0;
	instance->numTriEdges = 0;
	CUDAMallocSafe(instance->cudaTargetVert, instance->softNum);
	CUDAMallocSafe(instance->cudaTargetIndex, instance->softNum);
	CUDAMallocSafe(instance->cudaTriDmInverses, instance->numTriElements);
	CUDAMallocSafe(instance->cudaTriArea, instance->numTriElements);
	CUDAMallocSafe(instance->cudaTriEdgeAdjVertex, instance->numTriEdges);
	CUDAMallocSafe(instance->cudaTriEdges, instance->numTriEdges);


	CUDAMallocSafe(instance->cudaCloseCPNum, 1);
	CUDAMallocSafe(instance->cudaCloseGPNum, 1);
	CUDA_SAFE_CALL(cudaMemset(instance->cudaCloseCPNum, 0, sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMemset(instance->cudaCloseGPNum, 0, sizeof(uint32_t)));




	std::cout << "numVerts~~" << numVerts << std::endl;
	std::cout << "numElems~~" << numElems << std::endl;
	std::cout << "maxnumbers~~" << maxNumbers << std::endl;
	std::cout << "numSurfVerts~~" << instance->surfVert.rows() << std::endl;
	std::cout << "numSurfEdges~~" << instance->surfEdge.rows() << std::endl;
	std::cout << "numSurfFaces~~" << instance->surfFace.rows() << std::endl;
	std::cout << "MAX_CCD_COLLITION_PAIRS_NUM~" << instance->MAX_CCD_COLLITION_PAIRS_NUM << std::endl;
	std::cout << "MAX_COLLITION_PAIRS_NUM~" << instance->MAX_COLLITION_PAIRS_NUM << std::endl;

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
	instance->bendStiff = 3e-4;
	instance->Newton_solver_threshold = 1e-1;
	instance->pcg_threshold = 1e-3;
	instance->relative_dhat = 1e-3;
	instance->bendStiff = instance->clothYoungModulus * pow(instance->clothThickness, 3) / (24 * (1 - instance->PoissonRate * instance->PoissonRate));
	instance->shearStiff = 0.03 * instance->stretchStiff;

}



void GAS_Read_Buffer::initSIMFEM() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "initSIMFEM geoinstance not initialized");

	/////////////////////////////
	// init meanMass & meanVolume
	/////////////////////////////
	double sumMass = 0;
	double sumVolume = 0;
	for (int i = 0; i < instance->numVertices; i++) {
		double mass = instance->vertMass[i];
		sumMass += mass;
	}
	for (int i = 0; i < instance->numElements; i++) {
		double vlm = instance->tetVolume[i];
		sumVolume += vlm;
	}
	
	instance->meanMass = sumMass / instance->numVertices;
	instance->meanVolume = sumVolume / instance->numVertices;
	printf("meanMass: %f\n", instance->meanMass);
	printf("meanVolum: %f\n", instance->meanVolume);

	/////////////////////////////
	// init meanMass & meanVolume
	/////////////////////////////
	double angleX = -MATHUTILS::PI / 4, angleY = -MATHUTILS::PI / 4, angleZ = MATHUTILS::PI / 2;
	MATHUTILS::Matrix3x3d rotation, rotationZ, rotationY, rotationX, eigenTest;
	MATHUTILS::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
	MATHUTILS::__set_Mat_val(rotationZ, cos(angleZ), -sin(angleZ), 0, sin(angleZ), cos(angleZ), 0, 0, 0, 1);
	MATHUTILS::__set_Mat_val(rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0, sin(angleY), 0, cos(angleY));
	MATHUTILS::__set_Mat_val(rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

	for (int i = 0; i < instance->numElements; i++) {
		MATHUTILS::Matrix3x3d DM;
		MATHUTILS::Matrix3x3d DM_inverse;
		FEMENERGY::__calculateDms3D_double(instance->vecVertPos.data(), instance->vectetElement[i], DM);
		MATHUTILS::__Inverse(DM, DM_inverse);
		instance->DMInverse.push_back(DM_inverse);
	}
	std::cout << "DMInverse0: " 
		<< instance->DMInverse[0].m[0][0] << " "
		<< instance->DMInverse[0].m[0][1] << " "
		<< instance->DMInverse[0].m[0][2] << " "
		<< std::endl;


	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaDmInverses, instance->DMInverse.data(), instance->numElements * sizeof(MATHUTILS::Matrix3x3d), cudaMemcpyHostToDevice));


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

	instance->LBVH_E_ptr->init(
		instance->cudaBoundaryType,
		instance->cudaVertPos,
		instance->cudaRestTetPos,
		instance->cudaSurfEdge,
		instance->cudaCollisionPairs,
		instance->cudaCCDCollisionPairs,
		instance->cudaCPNum,
		instance->cudaMatIndex,
		instance->numSurfEdges,
		instance->numSurfVerts
	);
	instance->LBVH_E_ptr->CUDA_MALLOC_LBVH(instance->numSurfEdges);

	instance->LBVH_F_ptr->init(
		instance->cudaBoundaryType, 
		instance->cudaVertPos,
		instance->cudaSurfFace,
		instance->cudaSurfVert,
		instance->cudaCollisionPairs,
		instance->cudaCCDCollisionPairs, 
		instance->cudaCPNum, 
		instance->cudaMatIndex, 
		instance->numSurfFaces,
		instance->numSurfVerts
	);
	instance->LBVH_F_ptr->CUDA_MALLOC_LBVH(instance->numSurfFaces);

	// calcuate Morton Code and sort MC together with face index
	SortMesh::sortMesh(instance, instance->LBVH_F_ptr);

}




void GAS_Read_Buffer::buildSIMBVH() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "buildSIMBVH geoinstance not initialized");

	instance->LBVH_F_ptr->Construct();
	instance->LBVH_E_ptr->Construct();

}


void GAS_Read_Buffer::initSIMIPC() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "initSIMIPC geoinstance not initialized");

	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaRestTetPos, instance->cudaOriginTetPos, instance->numVertices * sizeof(double3), cudaMemcpyDeviceToDevice));

	if (!instance->AABB_SceneSize_ptr) {
		instance->AABB_SceneSize_ptr = std::make_unique<AABB>();
	}
	*(instance->AABB_SceneSize_ptr) = instance->LBVH_F_ptr->m_scene;
	auto &AABBScene = instance->AABB_SceneSize_ptr;
	double3 &upper = AABBScene->m_upper;
	double3 &lower = AABBScene->m_lower;
	CHECK_ERROR((upper.x >= lower.x) && (upper.y >= lower.y) && (upper.z >= lower.z), "AABB maybe error, please check again");
    std::cout << "SceneSize upper/lower: ~~" << upper.x << " " << lower.x << std::endl;

	instance->bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(AABBScene->m_upper, AABBScene->m_lower));
	instance->dTol = 1e-18 * instance->bboxDiagSize2;
	instance->minKappaCoef = 1e11;
	instance->dHat = instance->relative_dhat * instance->relative_dhat * instance->bboxDiagSize2;
	instance->fDhat = 1e-6 * instance->bboxDiagSize2;
	
}

void GAS_Read_Buffer::buildSIMCP() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "buildSIMCP geoinstance not initialized");

	CUDA_SAFE_CALL(cudaMemset(instance->cudaCPNum, 0, 5 * sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMemset(instance->cudaGPNum, 0, sizeof(uint32_t)));

	instance->LBVH_F_ptr->SelfCollitionDetect(instance->dHat);
	instance->LBVH_E_ptr->SelfCollitionDetect(instance->dHat);

	instance->LBVH_F_ptr->GroundCollisionDetect(
		instance->cudaVertPos, 
		instance->cudaSurfVert,
		instance->cudaGroundOffset,
		instance->cudaGroundNormal,
		instance->cudaEnvCollisionPairs,
		instance->cudaGPNum,
		instance->dHat,
		instance->surfVert.rows());


	if (!instance->PCGData_ptr) {
		instance->PCGData_ptr = std::make_unique<PCGData>();
	}

	instance->PCGData_ptr->CUDA_MALLOC_PCGDATA(instance->numVertices, instance->numElements);

	instance->PCGData_ptr->m_P_type = 0;
	instance->PCGData_ptr->m_b = instance->cudaFb;
	instance->cudaMoveDir = instance->PCGData_ptr->m_dx;

	double motion_rate = 1.0;
	instance->animation_subRate = 1.0 / motion_rate;
	FEMENERGY::computeXTilta(instance, 1);

	std::vector<AABB> boundVolumes(2 * instance->numSurfEdges - 1);
	std::vector<Node> Nodes(2 * instance->numSurfEdges - 1);
	CUDA_SAFE_CALL(cudaMemcpy(&boundVolumes[0], instance->LBVH_E_ptr->mc_boundVolumes, (2 * instance->numSurfEdges - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&Nodes[0], instance->LBVH_E_ptr->mc_nodes, (2 * instance->numSurfEdges - 1) * sizeof(Node), cudaMemcpyDeviceToHost));

	if (instance->BH_ptr) {
		instance->BH_ptr = std::make_unique<BHessian>();
	}
	instance->BH_ptr->CUDA_MALLOC_BHESSIAN(instance->numElements, instance->numSurfVerts, instance->numSurfFaces, instance->numSurfEdges, instance->numTriElements, instance->numTriEdges);

	std::cout << "BH Malloc Nums~~ " << instance->numElements << " " << instance->numSurfVerts << " " << instance->numSurfFaces << " " << instance->numSurfEdges << " " << instance->numTriElements << " " << instance->numTriEdges << std::endl;



}
