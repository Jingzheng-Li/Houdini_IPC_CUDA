
#include "main_readBuffer.hpp"

#include "UTILS/GeometryManager.hpp"
#include "LBVH/SortMesh.cuh"

#include "LBVH/LBVH.cuh"
#include "PCG/BHessian.cuh"
#include "PCG/MASPreconditioner.cuh"
#include "PCG/PCGSolver.cuh"
#include "FEMEnergy.cuh"
#include "IPC/GIPC.cuh"
#include "INTEGRATOR/ImplicitIntergrator.cuh"

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
        GeometryManager::instance->freeGeometryManager();
    }
}

bool GAS_Read_Buffer::solveGasSubclass(SIM_Engine& engine,
										SIM_Object* object,
										SIM_Time time,
										SIM_Time timestep) {

	cudaError_t cudaStatus = cudaSetDevice(0);
	CHECK_ERROR_SOLVER(cudaStatus==cudaSuccess, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	auto &instance = GeometryManager::instance;

	if (!FIRSTFRAME::hou_initialized) {
		
		PRINT_BLUE("running_initialized");

		if (!instance) {
			instance = std::unique_ptr<GeometryManager>(new GeometryManager());
		}
		CHECK_ERROR_SOLVER(instance, "geometry instance not initialize");

		loadSIMParamsFromHoudini();
		loadSIMGeometryFromHoudini(object);
		loadCollisionGeometryFromHoudini(object);

		transferDetailAttribTOCUDA();
		transferOtherAttribTOCUDA();

		initSIMFEM();
		initSIMBVH();
		initSIMPCG();
		initSIMIPC();

		FEMENERGY::computeXTilta(instance, 1); // get a initial guess

		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		debugSIM();

		FIRSTFRAME::hou_initialized = true;
	}

	return true;
}



void GAS_Read_Buffer::loadSIMParamsFromHoudini() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "loadSIMParamsFromHoudini geoinstance not initialized");

	instance->IPC_dt = 0.01;
	instance->precondType = 1;

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
	instance->shearStiff = instance->stretchStiff * 0.03;
	// instance->stretchStiff = 0.0;
	// instance->shearStiff = 0.0;

	instance->clothDensity = 2e2;
	instance->softMotionRate = 1e0;
	instance->Newton_solver_threshold = 1e-1;
	instance->pcg_threshold = 1e-3;
	instance->relative_dhat = 1e-3;
	// instance->bendStiff = instance->clothYoungModulus * pow(instance->clothThickness, 3) / (24 * (1 - instance->PoissonRate * instance->PoissonRate));
	instance->bendStiff = 1e-3; // TODO: bound is extremely small in the previous expression, find a correct expression

	instance->animation = false;
	instance->collision_detection_buff_scale = 1;
	double motion_rate = 1.0;
	instance->animation_fullRate = 0.0;
	instance->animation_subRate = 1.0 / motion_rate;

	instance->ground_bottom_offset = 0.0;
	instance->ground_left_offset = -2.0;
	instance->ground_right_offset = 2.0;
	instance->ground_near_offset = -2.0;
	instance->ground_far_offset = 2.0;

	instance->gravityforce = make_double3(0.0, -9.8, 0.0);
	
}


void GAS_Read_Buffer::loadSIMGeometryFromHoudini(SIM_Object* object) {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "loadSIMGeometryFromHoudini geoinstance not initialized");

	const SIM_Geometry *geo = object->getGeometry();
	CHECK_ERROR(geo != NULL, "Failed to get readBuffer geometry object")
	GU_DetailHandleAutoReadLock readlock(geo->getGeometry());
	CHECK_ERROR(readlock.isValid(), "Failed to get readBuffer geometry detail");
	const GU_Detail *gdp = readlock.getGdp();
	CHECK_ERROR(!gdp->isEmpty(), "readBuffer Geometry is empty");

	// position velocity mass
	auto &vertpos = instance->vertPos;
	auto &vertvel = instance->vertVel;
	auto &vertmass = instance->vertMass;
	vertpos.resize(gdp->getNumPoints(), Eigen::NoChange);
	vertvel.resize(gdp->getNumPoints(), Eigen::NoChange);
	vertmass.resize(gdp->getNumPoints());
	vertpos.setZero();
	vertvel.setZero();
	vertmass.setZero();
	instance->numVertices = gdp->getNumPoints();
	instance->boundaryTypies.conservativeResize(gdp->getNumPoints());
	instance->boundaryTypies.setZero();

	GA_ROHandleV3D velHandle(gdp, GA_ATTRIB_POINT, "v");
	GA_ROHandleD massHandle(gdp, GA_ATTRIB_POINT, "mass");
	CHECK_ERROR(velHandle.isValid() && massHandle.isValid(), "Failed to get velocity and mass attributes");

	// TODO 不行 还不能这么简单 还得分一下type=2和type=3的两种情况 关键是type=1还不知道是什么意思 也得再看一下 type=2就得

	{ // obtain sim geometry points
		GA_Offset ptoff;
		int ptidx = 0;
		GA_FOR_ALL_PTOFF(gdp, ptoff) {
			UT_Vector3D pos3 = gdp->getPos3D(ptoff);
			UT_Vector3D vel3 = velHandle.get(ptoff);
			vertpos.row(ptidx) << pos3.x(), pos3.y(), pos3.z();
			vertvel.row(ptidx) << vel3.x(), vel3.y(), vel3.z();
			vertmass(ptidx) = massHandle.get(ptoff);
			instance->boundaryTypies(ptidx) = 0;

			ptidx++;
		}
		CHECK_ERROR(ptidx==gdp->getNumPoints(), "Failed to get all soft geometry points");
	}

	{ // obtain sim geometry primitives
		auto &tetEle = instance->tetElement;
		auto &triEle = instance->triElement;
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
				std::cerr << "\033[1;31m" << "sim geometry only support tri and tet right now" << "\033[0m" << std::endl;
				return;
			}
			primidx++;
		}
		CHECK_ERROR(primidx==gdp->getNumPrimitives(), "Failed to get all primitives");
		CHECK_ERROR(numTets + numTris==gdp->getNumPrimitives(), "Failed to get all tets and tris");

	}

}


void GAS_Read_Buffer::loadCollisionGeometryFromHoudini(SIM_Object* object) {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "loadCollisionGeometryFromHoudini geoinstance not initialized");

	SIM_ConstObjectArray colaffectors;
	object->getConstAffectors(colaffectors, "SIM_RelationshipCollide");
    CHECK_ERROR(colaffectors.entries() == 2, "houdini scene static object is not correct, we only allow one static object exists");
	
	// colaffector(0) will be our collsion geometry
	const SIM_Geometry* collidegeo = SIM_DATA_GETCONST(*colaffectors(0), SIM_GEOMETRY_DATANAME, SIM_Geometry);
	CHECK_ERROR(collidegeo != nullptr, "collidegeo is nullptr right now");
	GU_DetailHandleAutoReadLock readlock(collidegeo->getGeometry());
	const GU_Detail *collidegdp = readlock.getGdp();
	CHECK_ERROR(!collidegdp->isEmpty(), "not get any collision objects gdp");

	instance->collisionVertPos.resize(collidegdp->getNumPoints(), Eigen::NoChange);
	instance->collisionVertPos.setZero();
	instance->collisionBoundaryType.resize(collidegdp->getNumPoints());
	instance->collisionBoundaryType.setZero();
	GA_ROHandleV3D initPosHandle(collidegdp, GA_ATTRIB_POINT, "initP");
	GA_ROHandleI btypeHandle(collidegdp, GA_ATTRIB_POINT, "boundaryType");
	CHECK_ERROR(initPosHandle.isValid(), "Failed to get collision objects initP");
	CHECK_ERROR(btypeHandle.isValid(), "Failed to get collision objects boundaryType");

	{ // obtain collision geometry points
		GA_Offset ptoff;
		int ptidx = 0;
		GA_FOR_ALL_PTOFF(collidegdp, ptoff) {
			UT_Vector3D initP3 = initPosHandle.get(ptoff);
			instance->collisionVertPos.row(ptidx) << initP3.x(), initP3.y(), initP3.z();
			instance->collisionBoundaryType.row(ptidx) << btypeHandle.get(ptoff);
			ptidx++;
		}
		CHECK_ERROR(ptidx==collidegdp->getNumPoints(), "Failed to get all collision points");
	}


	{ // obtain collision geometry primitives
		int primidx = 0;
		GA_Offset primoff;
		instance->collisionSurfFace.resize(collidegdp->getNumPrimitives(), Eigen::NoChange);
		GA_FOR_ALL_PRIMOFF(collidegdp, primoff) {
			const GA_Primitive* prim = collidegdp->getPrimitive(primoff);
			if (prim->getVertexCount() == 3) {
				for (int i = 0; i < prim->getVertexCount(); ++i) {
					GA_Offset vtxoff = prim->getVertexOffset(i);
					GA_Offset ptoff = collidegdp->vertexPoint(vtxoff);
					instance->collisionSurfFace(primidx, i) = static_cast<int>(collidegdp->pointIndex(ptoff));
				}
				primidx++;
			} else {
				std::cerr << "\033[1;31m" << "collision geometry only support tri right now" << "\033[0m" << std::endl;
				return;
			}
		}
		CHECK_ERROR(primidx==collidegdp->getNumPrimitives(), "Failed to get all primitives");
	}
}



void GAS_Read_Buffer::transferDetailAttribTOCUDA() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferDetailAttribTOCUDA geoinstance not initialized");

	// update surfvert/surfface/edge/boundary/constraint
	
	// add collisionVertPos to total vertPos
	instance->numSIMVertPos = instance->vertPos.rows();
	instance->numVertices = instance->vertPos.rows() + instance->collisionVertPos.rows();
	instance->vertPos.conservativeResize(instance->numVertices, Eigen::NoChange);
	instance->vertPos.bottomRows(instance->collisionVertPos.rows()) = instance->collisionVertPos;
	CHECK_ERROR(instance->vertPos.rows()==instance->numVertices, "vertPos number not match with numVertices");
	instance->vertVel.conservativeResize(instance->numVertices, Eigen::NoChange);
	instance->vertVel.bottomRows(instance->collisionVertPos.rows()).setZero();
	instance->vertMass.conservativeResize(instance->numVertices);
	instance->vertMass.bottomRows(instance->collisionVertPos.rows()).setZero();
	CHECK_ERROR(instance->vertVel.rows()==instance->numVertices && instance->vertMass.rows()==instance->numVertices, "vertvel and vertmass size somehow not equal to total num vertexes");

	instance->numTriElements = instance->triElement.rows();
	instance->numTetElements = instance->tetElement.rows();
	// get simulation triangle edges (for cloth bending only)

	

	// get 1.all triangles 2.tetrahedrao outer surface 3.collision surface as surfaces
	MATHUTILS::__getTriSurface(instance->triElement, instance->surfFace);
	MATHUTILS::__getTetSurface(instance->tetElement, instance->vertPos, instance->surfFace);
	MATHUTILS::__getColSurface(instance->numSIMVertPos, instance->collisionSurfFace, instance->surfFace);
	instance->numSurfFaces = instance->surfFace.rows();

	// add collisionSurfFace to total surfFace
	MATHUTILS::__getSurfaceVertsAndEdges(instance->surfFace, instance->vertPos, instance->surfVert, instance->surfEdge);
	
	instance->numSoftConstraints = instance->collisionBoundaryType.rows();
	instance->boundaryTypies.conservativeResize(instance->vertPos.rows());
	instance->boundaryTypies.bottomRows(instance->collisionVertPos.rows()) = instance->collisionBoundaryType;


	CUDAMallocSafe(instance->cudaVertPos, instance->numVertices);
	CUDAMallocSafe(instance->cudaVertVel, instance->numVertices);
	CUDAMallocSafe(instance->cudaVertMass, instance->numVertices);
	CUDAMallocSafe(instance->cudaOriginVertPos, instance->numVertices);
	CUDAMallocSafe(instance->cudaRestVertPos, instance->numVertices);
	CUDAMallocSafe(instance->cudaTetElement, instance->numTetElements);
	CUDAMallocSafe(instance->cudaTriElement, instance->numTriElements);

	CUDAMemcpyHToDSafe(instance->cudaVertPos, instance->vertPos);
	CUDAMemcpyHToDSafe(instance->cudaVertVel, instance->vertVel);
	CUDAMemcpyHToDSafe(instance->cudaVertMass ,instance->vertMass);
	CUDAMemcpyHToDSafe(instance->cudaOriginVertPos, instance->vertPos);
	CUDAMemcpyHToDSafe(instance->cudaRestVertPos, instance->vertPos);
	CUDAMemcpyHToDSafe(instance->cudaTetElement, instance->tetElement);
	CUDAMemcpyHToDSafe(instance->cudaTriElement, instance->triElement);




	instance->tetVolume.resize(instance->numTetElements);
	instance->triArea.resize(instance->numTriElements);
	std::vector<double3> vecVertPos = MATHUTILS::__convertEigenToVector<double3>(instance->vertPos);
	std::vector<double3> vecVertVel = MATHUTILS::__convertEigenToVector<double3>(instance->vertVel);
	std::vector<uint4> vectetElement = MATHUTILS::__convertEigenToUintVector<uint4>(instance->tetElement);
	std::vector<uint3> vectriElement = MATHUTILS::__convertEigenToUintVector<uint3>(instance->triElement);
	CHECK_ERROR(vecVertPos.size()==instance->vertPos.rows(), "not init vecVertPos correctly");
	CHECK_ERROR(vecVertVel.size()==instance->vertVel.rows(), "not init vecVertVel correctly");
	CHECK_ERROR(vectetElement.size()==instance->numTetElements, "not init tetElement correctly");
	CHECK_ERROR(vectriElement.size()==instance->numTriElements, "not init triElement correctly");
	for (int i = 0; i < instance->numTetElements; i++) {
		instance->tetVolume[i] = MATHUTILS::__calculateVolume(vecVertPos.data(), vectetElement[i]);
	}
	for (int i = 0; i < instance->numTriElements; i++) {
		instance->triArea[i] = MATHUTILS::__calculateArea(vecVertPos.data(), vectriElement[i]) * instance->clothThickness;
	}

	// init meanMass & meanVolume
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

	CUDAMallocSafe(instance->cudaTetVolume, instance->numTetElements);
	CUDAMemcpyHToDSafe(instance->cudaTetVolume, instance->tetVolume);
	CUDAMallocSafe(instance->cudaTriArea, instance->numTriElements);
	CUDAMemcpyHToDSafe(instance->cudaTriArea, instance->triArea);



	instance->targetIndex.resize(instance->numSoftConstraints);
	instance->targetVertPos.resize(instance->numSoftConstraints, Eigen::NoChange);
	MATHUTILS::Matrix3x3d IdentityMat;
	MATHUTILS::Matrix3x3d zeroMat;
	MATHUTILS::__set_Mat_val(IdentityMat, 1, 0, 0, 0, 1, 0, 0, 0, 1);
	MATHUTILS::__set_Mat_val(zeroMat, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	instance->constraintsMat.resize(instance->numVertices);
	instance->constraintsMat.setConstant(IdentityMat);
	// if btype is 0/1/2 constraint set as identity matrix, 3 set as zero matrix
	for (int i = 0; i < instance->numSoftConstraints; i++) {
		instance->targetIndex[i] = i + instance->numSIMVertPos;
		instance->targetVertPos.row(i) = instance->vertPos.row(i + instance->numSIMVertPos);
		if (instance->boundaryTypies[i] == 3)
			instance->constraintsMat[i + instance->numSIMVertPos] = zeroMat;
	}
	CUDAMallocSafe(instance->cudaTargetIndex, instance->numSoftConstraints);
	CUDAMemcpyHToDSafe(instance->cudaTargetIndex, instance->targetIndex);
	CUDAMallocSafe(instance->cudaTargetVertPos, instance->numSoftConstraints);
	CUDAMemcpyHToDSafe(instance->cudaTargetVertPos, instance->targetVertPos);
	CUDAMallocSafe(instance->cudaConstraintsMat, instance->numVertices);
	CUDAMemcpyHToDSafe(instance->cudaConstraintsMat, instance->constraintsMat);



	MATHUTILS::__getTriEdges(instance->triElement, instance->triEdges, instance->triEdgeAdjVertex);
	instance->numTriEdges = instance->triEdges.rows();
	CUDAMallocSafe(instance->cudaTriEdges, instance->numTriEdges);
	CUDAMallocSafe(instance->cudaTriEdgeAdjVertex, instance->numTriEdges);
	CUDAMemcpyHToDSafe(instance->cudaTriEdges, instance->triEdges);
	CUDAMemcpyHToDSafe(instance->cudaTriEdgeAdjVertex, instance->triEdgeAdjVertex);


	instance->numSurfFaces = instance->surfFace.rows();
	instance->numSurfVerts = instance->surfVert.rows();
	instance->numSurfEdges = instance->surfEdge.rows();
	CUDAMallocSafe(instance->cudaSurfFace, instance->surfFace.rows());
	CUDAMallocSafe(instance->cudaSurfVert, instance->surfVert.rows());
	CUDAMallocSafe(instance->cudaSurfEdge, instance->surfEdge.rows());
	CUDAMemcpyHToDSafe(instance->cudaSurfFace, instance->surfFace);
	CUDAMemcpyHToDSafe(instance->cudaSurfVert, instance->surfVert);
	CUDAMemcpyHToDSafe(instance->cudaSurfEdge, instance->surfEdge);


	CUDAMallocSafe(instance->cudaBoundaryType, instance->numVertices);
	CUDAMemcpyHToDSafe(instance->cudaBoundaryType, instance->boundaryTypies);

}




void GAS_Read_Buffer::transferOtherAttribTOCUDA() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "transferOtherAttribTOCUDA geoinstance not initialized");

	int maxSIMGeoNumbers = std::max(instance->numVertices, instance->numTetElements);
	CUDAMallocSafe(instance->cudaMortonCodeHash, maxSIMGeoNumbers);
	CUDAMallocSafe(instance->cudaSortIndex, maxSIMGeoNumbers);
	CUDAMallocSafe(instance->cudaSortMapVertIndex, instance->numVertices);
	CUDAMallocSafe(instance->cudaTempDouble, maxSIMGeoNumbers);
	CUDAMallocSafe(instance->cudaTempMat3x3, maxSIMGeoNumbers);

	instance->MAX_CCD_COLLITION_PAIRS_NUM = 1 * instance->collision_detection_buff_scale * (((double)(instance->surfFace.rows() * 15 + instance->surfEdge.rows() * 10)) * std::max((instance->IPC_dt / 0.01), 2.0));
	instance->MAX_COLLITION_PAIRS_NUM = (instance->surfVert.rows() * 3 + instance->surfEdge.rows() * 2) * 3 * instance->collision_detection_buff_scale;
	CHECK_ERROR(instance->MAX_CCD_COLLITION_PAIRS_NUM > 0, "MAX_CCD_COLLITION_PAIRS_NUM is 0, this is incorrect");
	CHECK_ERROR(instance->MAX_COLLITION_PAIRS_NUM > 0, "MAX_COLLITION_PAIRS_NUM is 0, this is incorrect");

	CUDAMallocSafe(instance->cudaCollisionPairs, instance->MAX_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaCCDCollisionPairs, instance->MAX_CCD_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaMatIndex, instance->MAX_COLLITION_PAIRS_NUM);
	CUDAMallocSafe(instance->cudaCPNum, 5);
	CUDAMallocSafe(instance->cudaGPNum, 1);
	CUDAMallocSafe(instance->cudaEnvCollisionPairs, instance->numVertices);

	// update ground collisions
	CUDAMallocSafe(instance->cudaGroundNormal, 5);
	CUDAMallocSafe(instance->cudaGroundOffset, 5);
	// bottom left right near far
	double ground_offset[5] = {
		instance->ground_bottom_offset, 
		instance->ground_left_offset, 
		instance->ground_right_offset, 
		instance->ground_near_offset,
		instance->ground_far_offset
	};
	double3 ground_normal[5];
	ground_normal[0] = make_double3(0, 1, 0);
    ground_normal[1] = make_double3(1, 0, 0);
    ground_normal[2] = make_double3(-1, 0, 0);
    ground_normal[3] = make_double3(0, 0, 1);
    ground_normal[4] = make_double3(0, 0, -1);
	CUDA_SAFE_CALL(cudaMemcpy(instance->cudaGroundOffset, &ground_offset, 5 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaGroundNormal, &ground_normal, 5 * sizeof(double3), cudaMemcpyHostToDevice));


	CUDAMallocSafe(instance->cudaXTilta, instance->numVertices);
	CUDAMallocSafe(instance->cudaFb, instance->numVertices);
	CUDAMallocSafe(instance->cudaTempDouble3Mem, instance->numVertices);

	CUDAMallocSafe(instance->cudaCloseCPNum, 1);
	CUDAMallocSafe(instance->cudaCloseGPNum, 1);
	CUDAMemcpyHToDSafe(instance->cudaCloseCPNum, Eigen::VectorXi::Zero(1));
	CUDAMemcpyHToDSafe(instance->cudaCloseGPNum, Eigen::VectorXi::Zero(1));


	CHECK_ERROR(instance->vertPos.rows() == instance->numVertices, "numVerts not match with Eigen");
	CHECK_ERROR(instance->tetElement.rows() == instance->numTetElements, "numTetEles not match with Eigen");
	CHECK_ERROR(instance->triElement.rows() == instance->numTriElements, "numTriEles not match with Eigen");
	CHECK_ERROR(instance->triEdges.rows() == instance->numTriEdges, "numTriEdges not match with Eigen")
	CHECK_ERROR(instance->surfVert.rows() == instance->numSurfVerts, "numSurfVerts not match with Eigen");
	CHECK_ERROR(instance->surfEdge.rows() == instance->numSurfEdges, "numSurfEdges not match with Eigen");
	CHECK_ERROR(instance->surfFace.rows() == instance->numSurfFaces, "numSurfFaces not match with Eigen");

}


void GAS_Read_Buffer::initSIMFEM() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "initSIMFEM geoinstance not initialized");

	instance->tetDMInverse.resize(instance->numTetElements);
	instance->triDMInverse.resize(instance->numTriElements);
	CUDAMallocSafe(instance->cudaTriDmInverses, instance->numTriElements);
	CUDAMallocSafe(instance->cudaTetDmInverses, instance->numTetElements);

	std::vector<double3> vecVertPos = MATHUTILS::__convertEigenToVector<double3>(instance->vertPos);
	std::vector<uint4> vectetElement = MATHUTILS::__convertEigenToUintVector<uint4>(instance->tetElement);
	std::vector<uint3> vectriElement = MATHUTILS::__convertEigenToUintVector<uint3>(instance->triElement);

	for (int i = 0; i < instance->numTetElements; i++) {
		MATHUTILS::Matrix3x3d DM;
		MATHUTILS::Matrix3x3d DM_inverse;
		FEMENERGY::__calculateDms3D_double(vecVertPos.data(), vectetElement[i], DM);
		MATHUTILS::__Inverse(DM, DM_inverse);
		instance->tetDMInverse.row(i) << DM_inverse;
	}

	for (int i = 0; i < instance->numTriElements; i++) {
		MATHUTILS::Matrix2x2d DM;
		MATHUTILS::Matrix2x2d DM_inverse;
		FEMENERGY::__calculateDm2D_double(vecVertPos.data(), vectriElement[i], DM);
		MATHUTILS::__Inverse2x2(DM, DM_inverse);
		instance->triDMInverse.row(i) << DM_inverse;
	}

	CUDAMemcpyHToDSafe(instance->cudaTetDmInverses, instance->tetDMInverse);
	CUDAMemcpyHToDSafe(instance->cudaTriDmInverses, instance->triDMInverse);
	
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
	if (!instance->LBVH_EF_ptr) {
		instance->LBVH_EF_ptr = std::make_unique<LBVH_EF>();
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
	SortMesh::sortMesh(instance, instance->LBVH_F_ptr->getSceneSize());

	// get sortmapvertinex from cuda memory, used in writebuffer
	instance->sortMapVertIndex.resize(instance->numVertices);
	CUDAMemcpyDToHSafe(instance->sortMapVertIndex, instance->cudaSortMapVertIndex);

	// build LBVH
	instance->LBVH_F_ptr->Construct();
	instance->LBVH_E_ptr->Construct();

	std::vector<AABB> boundVolumes(2 * instance->numSurfEdges - 1);
	std::vector<Node> Nodes(2 * instance->numSurfEdges - 1);
	CUDA_SAFE_CALL(cudaMemcpy(&boundVolumes[0], instance->LBVH_E_ptr->mc_boundVolumes, (2 * instance->numSurfEdges - 1) * sizeof(AABB), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&Nodes[0], instance->LBVH_E_ptr->mc_nodes, (2 * instance->numSurfEdges - 1) * sizeof(Node), cudaMemcpyDeviceToHost));


	AABB AABBScene = instance->LBVH_F_ptr->m_scene;
	instance->bboxDiagSize2 = MATHUTILS::__squaredNorm(MATHUTILS::__minus(AABBScene.m_upper, AABBScene.m_lower));
	instance->dTol = 1e-18 * instance->bboxDiagSize2;
	instance->minKappaCoef = 1e11;
	instance->dHat = instance->relative_dhat * instance->relative_dhat * instance->bboxDiagSize2;
	instance->fDhat = 1e-6 * instance->bboxDiagSize2;
	CHECK_ERROR((AABBScene.m_upper.x >= AABBScene.m_lower.x) && 
				(AABBScene.m_upper.y >= AABBScene.m_lower.y) && 
				(AABBScene.m_upper.z >= AABBScene.m_lower.z), 
				"AABB maybe error, please check again");

}

void GAS_Read_Buffer::initSIMPCG() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "initSIMIPC geoinstance not initialized");

	// init BH_ptr
	if (!instance->BH_ptr) {
		instance->BH_ptr = std::make_unique<BHessian>();
	}
	instance->BH_ptr->CUDA_MALLOC_BHESSIAN(instance->numTetElements, instance->numSurfVerts, instance->numSurfFaces, instance->numSurfEdges, instance->numTriElements, instance->numTriEdges);

	// init PCGData_ptr
	if (!instance->PCGData_ptr) {
		instance->PCGData_ptr = std::make_unique<PCGData>(instance);
	}
	instance->PCGData_ptr->CUDA_MALLOC_PCGDATA(instance->numVertices, instance->numTetElements);
	instance->PCGData_ptr->mc_b = instance->cudaFb;
	instance->cudaMoveDir = instance->PCGData_ptr->mc_dx; // cudaMovedir will be mc_dx


	// init MAS preconditioner
	CHECK_ERROR(instance->precondType == 0 || instance->precondType == 1, "conjugate gradient preconditioner only support MAS right now");
	if (instance->precondType == 1) {
		if (!instance->MAS_ptr) {
			instance->MAS_ptr = std::make_unique<MASPreconditioner>();
		}
	}

	if (instance->precondType == 1) {
		std::vector<unsigned int> neighborList;
		std::vector<unsigned int> neighborStart;
		std::vector<unsigned int> neighborNum;
		int neighborListSize = MATHUTILS::__getVertNeighbours(instance->numVertices, instance->tetElement, instance->triElement, neighborList, neighborStart, neighborNum);
		instance->MAS_ptr->CUDA_MALLOC_MAS(instance->numVertices, neighborListSize, instance->cudaCollisionPairs);
		instance->MAS_ptr->neighborListSize = neighborListSize;
		CUDA_SAFE_CALL(cudaMemcpy(instance->MAS_ptr->mc_neighborListInit, neighborList.data(), neighborListSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(instance->MAS_ptr->mc_neighborStart, neighborStart.data(), instance->numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(instance->MAS_ptr->mc_neighborNumInit, neighborNum.data(), instance->numVertices * sizeof(unsigned int), cudaMemcpyHostToDevice));
	}

	if (instance->precondType == 1) {
		SortMesh::sortPreconditioner(instance);
	}
}


void GAS_Read_Buffer::initSIMIPC() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "initSIMIPC geoinstance not initialized");


    if (!instance->GIPC_ptr) {
        instance->GIPC_ptr = std::make_unique<GIPC>(instance);
    }
	instance->GIPC_ptr->buildCP();

	if (!instance->Integrator_ptr) {
		instance->Integrator_ptr = std::make_unique<ImplicitIntegrator>(instance);
	}

}


void GAS_Read_Buffer::debugSIM() {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "initSIMIPC geoinstance not initialized");

	std::cout << "numVerts~~" << instance->numVertices << std::endl;
	std::cout << "numTriElems~~" << instance->numTriElements << std::endl;
	std::cout << "numTetElems~~" << instance->numTetElements << std::endl;
	int maxSIMGeoNumbers = instance->numVertices > instance->numTriElements ? instance->numVertices : instance->numTriElements; 
	std::cout << "maxSIMGeoNumbers~~~~" << maxSIMGeoNumbers << std::endl;
	std::cout << "numSurfVerts~~" << instance->surfVert.rows() << std::endl;
	std::cout << "numSurfEdges~~" << instance->surfEdge.rows() << std::endl;
	std::cout << "numSurfFaces~~" << instance->surfFace.rows() << std::endl;
	std::cout << "MAX_CCD_COLLITION_PAIRS_NUM~" << instance->MAX_CCD_COLLITION_PAIRS_NUM << std::endl;
	std::cout << "MAX_COLLITION_PAIRS_NUM~" << instance->MAX_COLLITION_PAIRS_NUM << std::endl;
	std::cout << "numTriEdges~~" << instance->triEdges.rows() << std::endl;
	std::cout << "masslast~~" << instance->vertMass.row(instance->numVertices-1) << std::endl;
	std::cout << "numMasses: " << instance->vertMass.rows() << std::endl;
	std::cout << "numVolume: " << instance->tetVolume.rows() << std::endl;
	std::cout << "numAreas: " << instance->triArea.rows() << std::endl;
	printf("meanMass: %f\n", instance->meanMass);
	printf("meanVolum: %f\n", instance->meanVolume);

	double totalvertpos = 0.0;
	for(int i = 0; i < instance->numVertices; i++) {
		totalvertpos += instance->vertPos.row(i).x();
	}
	totalvertpos /= instance->numVertices;
	std::cout << "totalvertpos~~~~~~~~~~" << totalvertpos << std::endl;

}
