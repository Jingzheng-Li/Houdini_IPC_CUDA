
#include "main_readBuffer.hpp"

#include "UTILS/GeometryManager.hpp"
#include "LBVH/SortMesh.cuh"

#include "LBVH/LBVH.cuh"
#include "PCG/PCGSolver.cuh"
#include "FEMEnergy.cuh"
#include "IPC/GIPC.cuh"

namespace FIRSTFRAME {
	static bool hou_initialized = false;
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
    if (GeometryManager::instance) {
        // GeometryManager::instance->freeGeometryManager();
    }
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

		loadSIMParams();

		transferPTAttribTOCUDA(geo, gdp);
		transferPRIMAttribTOCUDA(geo, gdp);
		transferDTAttribTOCUDA(geo, gdp);
		transferOtherTOCUDA();

		initSIMFEM();
		initSIMBVH();
		initSIMIPC();

		FIRSTFRAME::hou_initialized = true;
	}

	return true;
}


void GAS_Read_Buffer::transferPTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferPTAttribTOCUDA geoinstance not initialized");
	int num_points = gdp->getNumPoints();

	// position velocity mass
	auto &vertpos = instance->vertPos;
	auto &vertvel = instance->vertVel;
	auto &vertmass = instance->vertMass;
	vertpos.resize(num_points, Eigen::NoChange);
	vertvel.resize(num_points, Eigen::NoChange);
	vertmass.resize(num_points);
	instance->numVertices = num_points;

	auto &cons = instance->constraints;
	cons.resize(num_points);

	// boundary type
	auto &boundarytype = instance->boundaryTypies;
	boundarytype.resize(num_points);
	boundarytype.fill(0);
	instance->softNum = 0;


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
		vertpos.row(ptidx) << pos3.x(), pos3.y(), pos3.z();
		vertvel.row(ptidx) << vel3.x(), vel3.y(), vel3.z();
		vertmass(ptidx) = massHandle.get(ptoff);

		// TODO: boundary type should add 0/1/2/3... figure out what is going on here
		MATHUTILS::Matrix3x3d constraint;
		MATHUTILS::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
		cons[ptidx] = constraint;

		if (xmin > pos3.x()) xmin = pos3.x();
		if (ymin > pos3.y()) ymin = pos3.y();
		if (zmin > pos3.z()) zmin = pos3.z();
		if (xmax < pos3.x()) xmax = pos3.x();
		if (ymax < pos3.y()) ymax = pos3.y();
		if (zmax < pos3.z()) zmax = pos3.z();

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

	CUDAMallocSafe(instance->cudaOriginVertPos, num_points);
	CUDAMemcpyHToDSafe(instance->vertPos, instance->cudaOriginVertPos);
	CUDAMallocSafe(instance->cudaRestVertPos, num_points);
	CUDAMemcpyHToDSafe(instance->vertPos, instance->cudaRestVertPos);

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
	auto &triEle = instance->triElement;
	// Eigen::Matrix4i totalPrim;
	// totalPrim.resize(gdp->getNumPrimitives(), Eigen::NoChange);
	
	int numTris = 0;
	int numTets = 0;
	GA_Offset primoff;
	int primidx = 0;
	GA_FOR_ALL_PRIMOFF(gdp, primoff) {
		const GA_Primitive* prim = gdp->getPrimitive(primoff);
		if (prim->getVertexCount() == 4) {
			// update tet elements
			tetEle.conservativeResize(numTets + 1, Eigen::NoChange);
			for (int i = 0; i < prim->getVertexCount(); ++i) {
				GA_Offset vtxoff = prim->getVertexOffset(i);
				GA_Offset ptoff = gdp->vertexPoint(vtxoff);
				tetEle(numTets, i) = static_cast<int>(gdp->pointIndex(ptoff));
			}
			numTets++;

		} else if (prim->getVertexCount() == 3) {
			// update tri elements
			triEle.conservativeResize(numTris + 1, Eigen::NoChange);
			for (int i = 0; i < prim->getVertexCount(); ++i) {
				GA_Offset vtxoff = prim->getVertexOffset(i);
				GA_Offset ptoff = gdp->vertexPoint(vtxoff);
				triEle(numTris, i) = static_cast<int>(gdp->pointIndex(ptoff));
			}
			numTris++;

		} else {
			std::cerr << "not support this type of prim right now" << std::endl;
			return;
		}
		primidx++;
	}

	CHECK_ERROR(primidx==gdp->getNumPrimitives(), "Failed to get all primitives");
	CHECK_ERROR(numTets + numTris==gdp->getNumPrimitives(), "Failed to get all tets and tris");

	instance->numTetElements = numTets;
	instance->numTriElements = numTris;

	CUDAMallocSafe(instance->cudaTetElement, numTets);
	CUDAMemcpyHToDSafe(tetEle, instance->cudaTetElement);
	CUDAMallocSafe(instance->cudaTriElement, numTris);
	CUDAMemcpyHToDSafe(triEle, instance->cudaTriElement);


	instance->tetVolume.resize(numTets);
	instance->triArea.resize(numTris);
	instance->vectetElement = MATHUTILS::__convertEigenToUintVector<uint4>(tetEle);
	instance->vectriElement = MATHUTILS::__convertEigenToUintVector<uint3>(triEle);
	CHECK_ERROR(instance->vectetElement.size()==instance->numTetElements, "not init tetElement correctly");
	CHECK_ERROR(instance->vectriElement.size()==instance->numTriElements, "not init triElement correctly");
	for (int i = 0; i < numTets; i++) {
		instance->tetVolume[i] = MATHUTILS::__calculateVolume(instance->vecVertPos.data(), instance->vectetElement[i]);
	}
	for (int i = 0; i < numTris; i++) {
		instance->triArea[i] = MATHUTILS::__calculateArea(instance->vecVertPos.data(), instance->vectriElement[i]) * instance->clothThickness;
	}

	CUDAMallocSafe(instance->cudaTetVolume, numTets);
	CUDAMemcpyHToDSafe(instance->tetVolume, instance->cudaTetVolume);
	CUDAMallocSafe(instance->cudaTriArea, numTris);
	CUDAMemcpyHToDSafe(instance->triArea, instance->cudaTriArea);

}


void GAS_Read_Buffer::transferDTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferDTAttribTOCUDA geoinstance not initialized");

	Eigen::VectorXi &surfverts = instance->surfVert;
	Eigen::MatrixX3i &surffaces = instance->surfFace;
	Eigen::MatrixX2i &surfedges = instance->surfEdge;

	// get triangle surface
	MATHUTILS::__getTriSurface(instance->triElement, instance->triEdges, instance->triEdgeAdjVertex);
	instance->numTriEdges = instance->triEdges.rows();

	CUDAMallocSafe(instance->cudaTargetVert, instance->softNum);
	CUDAMallocSafe(instance->cudaTargetIndex, instance->softNum);

	CUDAMallocSafe(instance->cudaTriDmInverses, instance->numTriElements);

	CUDAMallocSafe(instance->cudaTriArea, instance->numTriElements);
	CUDAMallocSafe(instance->cudaTriEdges, instance->numTriEdges);
	CUDAMallocSafe(instance->cudaTriEdgeAdjVertex, instance->numTriEdges);
	CUDAMemcpyHToDSafe(instance->triArea, instance->cudaTriArea);
	CUDAMemcpyHToDSafe(instance->triEdges, instance->cudaTriEdges);
	CUDAMemcpyHToDSafe(instance->triEdgeAdjVertex, instance->cudaTriEdgeAdjVertex);


	// get tetrahedra surface
	MATHUTILS::__getTetSurface(surfverts, surffaces, surfedges, instance->vertPos, instance->tetElement, instance->triElement);

	CUDAMallocSafe(instance->cudaDmInverses, instance->numTetElements);

	CUDAMallocSafe(instance->cudaSurfVert, surfverts.rows());
	CUDAMallocSafe(instance->cudaSurfFace, surffaces.rows());
	CUDAMallocSafe(instance->cudaSurfEdge, surfedges.rows());
	CUDAMemcpyHToDSafe(instance->surfVert, instance->cudaSurfVert);
	CUDAMemcpyHToDSafe(instance->surfFace, instance->cudaSurfFace);
	CUDAMemcpyHToDSafe(instance->surfEdge, instance->cudaSurfEdge);
	instance->numSurfVerts = surfverts.rows();
	instance->numSurfFaces = surffaces.rows();
	instance->numSurfEdges = surfedges.rows();


}


void GAS_Read_Buffer::transferOtherTOCUDA() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferOtherTOCUDA geoinstance not initialized");

	int &numVerts = instance->numVertices;
	int &numTriElems = instance->numTriElements;
	int &numTetElems = instance->numTetElements;
	int maxNumbers = numVerts > numTetElems ? numVerts : numTetElems; 
	CUDAMallocSafe(instance->cudaMortonCodeHash, maxNumbers);
	CUDAMallocSafe(instance->cudaSortIndex, maxNumbers);
	CUDAMallocSafe(instance->cudaSortMapVertIndex, numVerts);
	
	CUDAMallocSafe(instance->cudaTempBoundaryType, numVerts);
	CUDAMallocSafe(instance->cudaTempDouble, maxNumbers);
	CUDAMallocSafe(instance->cudaTempMat3x3, maxNumbers);

	instance->IPC_dt = 0.01;
	instance->MAX_CCD_COLLITION_PAIRS_NUM = 1 * instance->collision_detection_buff_scale * (((double)(instance->surfFace.rows() * 15 + instance->surfEdge.rows() * 10)) * std::max((instance->IPC_dt / 0.01), 2.0));
	instance->MAX_COLLITION_PAIRS_NUM = (instance->surfVert.rows() * 3 + instance->surfEdge.rows() * 2) * 3 * instance->collision_detection_buff_scale;

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


	CUDAMallocSafe(instance->cudaCloseCPNum, 1);
	CUDAMallocSafe(instance->cudaCloseGPNum, 1);
	CUDA_SAFE_CALL(cudaMemset(instance->cudaCloseCPNum, 0, sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMemset(instance->cudaCloseGPNum, 0, sizeof(uint32_t)));


	CHECK_ERROR(instance->vertPos.rows() == instance->numVertices, "numVerts not match with Eigen");
	CHECK_ERROR(instance->tetElement.rows() == instance->numTetElements, "numTetEles not match with Eigen");
	CHECK_ERROR(instance->triElement.rows() == instance->numTriElements, "numTriEles not match with Eigen");
	CHECK_ERROR(instance->triEdges.rows() == instance->numTriEdges, "numTriEdges not match with Eigen")
	CHECK_ERROR(instance->surfVert.rows() == instance->numSurfVerts, "numSurfVerts not match with Eigen");
	CHECK_ERROR(instance->surfEdge.rows() == instance->numSurfEdges, "numSurfEdges not match with Eigen");
	CHECK_ERROR(instance->surfFace.rows() == instance->numSurfFaces, "numSurfFaces not match with Eigen");

	std::cout << "numVerts~~" << numVerts << std::endl;
	std::cout << "numTriElems~~" << numTriElems << std::endl;
	std::cout << "numTetElems~~" << numTetElems << std::endl;
	std::cout << "maxnumbers~~" << maxNumbers << std::endl;
	std::cout << "numSurfVerts~~" << instance->surfVert.rows() << std::endl;
	std::cout << "numSurfEdges~~" << instance->surfEdge.rows() << std::endl;
	std::cout << "numSurfFaces~~" << instance->surfFace.rows() << std::endl;
	std::cout << "MAX_CCD_COLLITION_PAIRS_NUM~" << instance->MAX_CCD_COLLITION_PAIRS_NUM << std::endl;
	std::cout << "MAX_COLLITION_PAIRS_NUM~" << instance->MAX_COLLITION_PAIRS_NUM << std::endl;
	std::cout << "numTriEdges~~" << instance->triEdges.rows() << std::endl;
	std::cout << "masslast~~" << instance->vertMass.row(numVerts-1) << std::endl;
	std::cout << "min coner~~" << instance->minCorner.x << " " 
								<< instance->minCorner.y << " " 
								<< instance->minCorner.z << " " << std::endl;
	std::cout << "max coner~~" << instance->maxCorner.x << " " 
								<< instance->maxCorner.y << " " 
								<< instance->maxCorner.z << " " << std::endl;
	std::cout << "numMasses: " << instance->vertMass.rows() << std::endl;
	std::cout << "numVolume: " << instance->tetVolume.rows() << std::endl;
	std::cout << "numAreas: " << instance->triArea.rows() << std::endl;

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

	instance->animation = false;
	instance->collision_detection_buff_scale = 1;
	
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
	for (int i = 0; i < instance->numTetElements; i++) {
		double vlm = instance->tetVolume[i];
		sumVolume += vlm;
	}
	for (int i = 0; i < instance->numTriElements; i++) {
		double vlm = instance->triArea[i];
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
	MATHUTILS::Matrix3x3d rotation, rotationZ, rotationY, rotationX;
	MATHUTILS::__set_Mat_val(rotation, 1, 0, 0, 0, 1, 0, 0, 0, 1);
	MATHUTILS::__set_Mat_val(rotationZ, cos(angleZ), -sin(angleZ), 0, sin(angleZ), cos(angleZ), 0, 0, 0, 1);
	MATHUTILS::__set_Mat_val(rotationY, cos(angleY), 0, -sin(angleY), 0, 1, 0, sin(angleY), 0, cos(angleY));
	MATHUTILS::__set_Mat_val(rotationX, 1, 0, 0, 0, cos(angleX), -sin(angleX), 0, sin(angleX), cos(angleX));

	for (int i = 0; i < instance->numTetElements; i++) {
		MATHUTILS::Matrix3x3d DM;
		MATHUTILS::Matrix3x3d DM_inverse;
		FEMENERGY::__calculateDms3D_double(instance->vecVertPos.data(), instance->vectetElement[i], DM);
		MATHUTILS::__Inverse(DM, DM_inverse);
		instance->DMInverse.push_back(DM_inverse);
	}


	for (int i = 0; i < instance->numTriElements; i++) {
		MATHUTILS::Matrix2x2d DM;
		MATHUTILS::Matrix2x2d DM_inverse;
		FEMENERGY::__calculateDm2D_double(instance->vecVertPos.data(), instance->vectriElement[i], DM);
		MATHUTILS::__Inverse2x2(DM, DM_inverse);
		instance->TriDMInverse.push_back(DM_inverse);
	}


	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaDmInverses, instance->DMInverse.data(), instance->numTetElements * sizeof(MATHUTILS::Matrix3x3d), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaTriDmInverses, instance->TriDMInverse.data(), instance->numTriElements * sizeof(MATHUTILS::Matrix2x2d), cudaMemcpyHostToDevice));
	

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
		instance->cudaRestVertPos,
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

	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaRestVertPos, instance->cudaOriginVertPos, instance->numVertices * sizeof(double3), cudaMemcpyDeviceToDevice));

	// buildBVH()
	instance->LBVH_F_ptr->Construct();
	instance->LBVH_E_ptr->Construct();

}























// 这里要大的修正一下 把ipc指针也初始化在这里！！
// 之前的函数应该都没有什么大问题 后面的函数倒是需要大的修改一下啊!!!!!


void GAS_Read_Buffer::initSIMIPC() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "initSIMIPC geoinstance not initialized");

	// ipc.init()
	AABB AABBScene = instance->LBVH_F_ptr->m_scene;
	CHECK_ERROR((AABBScene.m_upper.x >= AABBScene.m_lower.x) && (AABBScene.m_upper.y >= AABBScene.m_lower.y) && (AABBScene.m_upper.z >= AABBScene.m_lower.z), "AABB maybe error, please check again");
	std::cout << "SceneSize upper/lower: ~~" << AABBScene.m_upper.x << " " << AABBScene.m_lower.x << std::endl;

	instance->bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(AABBScene.m_upper, AABBScene.m_lower));
	instance->dTol = 1e-18 * instance->bboxDiagSize2;
	instance->minKappaCoef = 1e11;
	instance->dHat = instance->relative_dhat * instance->relative_dhat * instance->bboxDiagSize2;
	instance->fDhat = 1e-6 * instance->bboxDiagSize2;

	// init BH_ptr
	if (instance->BH_ptr) {
		instance->BH_ptr = std::make_unique<BHessian>();
	}
	instance->BH_ptr->CUDA_MALLOC_BHESSIAN(instance->numTetElements, instance->numSurfVerts, instance->numSurfFaces, instance->numSurfEdges, instance->numTriElements, instance->numTriEdges);


	// init PCGData_ptr
	if (!instance->PCGData_ptr) {
		instance->PCGData_ptr = std::make_unique<PCGData>();
	}
	instance->PCGData_ptr->CUDA_MALLOC_PCGDATA(instance->numVertices, instance->numTetElements);
	instance->PCGData_ptr->m_P_type = 0;
	instance->PCGData_ptr->m_b = instance->cudaFb;
	instance->cudaMoveDir = instance->PCGData_ptr->m_dx;










	// 这个地方非常乱 一定要非常非常小心

	double motion_rate = 1.0;
	instance->animation_subRate = 1.0 / motion_rate;

    if (!instance->GIPC_ptr) {
        instance->GIPC_ptr = std::make_unique<GIPC>(instance);
    }
	instance->GIPC_ptr->buildCP();
	// personal intuition, maybe not correct
	// instance->GIPC_ptr->h_close_gpNum = instance->GIPC_ptr->h_gpNum;



	FEMENERGY::computeXTilta(instance, 1);

	std::vector<AABB> boundVolumes(2 * instance->numSurfEdges - 1);
	std::vector<Node> Nodes(2 * instance->numSurfEdges - 1);
	CUDA_SAFE_CALL(cudaMemcpy(&boundVolumes[0], instance->LBVH_E_ptr->mc_boundVolumes, (2 * instance->numSurfEdges - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&Nodes[0], instance->LBVH_E_ptr->mc_nodes, (2 * instance->numSurfEdges - 1) * sizeof(Node), cudaMemcpyDeviceToHost));


}
