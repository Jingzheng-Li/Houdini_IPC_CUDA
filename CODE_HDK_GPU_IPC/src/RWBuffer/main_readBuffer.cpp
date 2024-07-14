
#include "main_readBuffer.hpp"

#include "UTILS/GeometryManager.hpp"
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

        loadSIMParams();
        initSIMBVH();

        FIRSTFRAME::hou_initialized = true;
    }

    return true;
}


void GAS_Read_Buffer::transferPTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

    auto &instance = GeometryManager::instance;
    CHECK_ERROR(instance, "PT geoinstance not initialized");
    int num_points = gdp->getNumPoints();

    // position velocity mass
    auto &tetpos = instance->tetPos;
    auto &tetvel = instance->tetVel;
    auto &tetmass = instance->tetMass;
    tetpos.resize(num_points, 3);
    tetvel.resize(num_points, 3);
    tetmass.resize(num_points);

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


    instance->minCorner = make_double3(xmin, ymin, zmin);
    instance->maxCorner = make_double3(xmax, ymax, zmax);


}


void GAS_Read_Buffer::transferPRIMAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

    auto &instance = GeometryManager::instance;
    CHECK_ERROR(instance, "PRIM geoinstance not initialized");

    if (GeometryManager::instance->tetInd.rows() == gdp->getNumPrimitives()) return;

    auto &tetind = instance->tetInd;
    tetind.resize(gdp->getNumPrimitives(), 4);


    GA_Offset primoff;
    GA_FOR_ALL_PRIMOFF(gdp, primoff) {
        const GA_Primitive* prim = gdp->getPrimitive(primoff);
        for (int i = 0; i < prim->getVertexCount(); ++i) {
            GA_Offset vtxoff = prim->getVertexOffset(i);
            GA_Offset ptoff = gdp->vertexPoint(vtxoff);
            tetind(primoff, i) = static_cast<int>(gdp->pointIndex(ptoff));
        }
    }
    CHECK_ERROR(primoff==gdp->getNumPrimitives(), "Failed to get all primitives");

    CUDAMallocSafe(instance->cudaTetInd, tetind.rows());
    copyToCUDASafe(instance->tetInd, instance->cudaTetInd);

}


void GAS_Read_Buffer::transferDTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

    auto &instance = GeometryManager::instance;
    CHECK_ERROR(instance, "DT geoinstance not initialized");

    GA_ROHandleDA surfposHandle(gdp, GA_ATTRIB_DETAIL, "surf_positions");
    CHECK_ERROR(surfposHandle.isValid(), "Failed to get surf_positions attribute");
    UT_DoubleArray surf_positions;
    surfposHandle.get(0, surf_positions);
    int numSurfPts = surf_positions.size() / 3;
    Eigen::MatrixXd surfPos(numSurfPts, 3);
    for (int i = 0; i < numSurfPts; ++i) {
        surfPos(i, 0) = surf_positions[i * 3];
        surfPos(i, 1) = surf_positions[i * 3 + 1];
        surfPos(i, 2) = surf_positions[i * 3 + 2];
    }

    GA_ROHandleIA surfedgeHandle(gdp, GA_ATTRIB_DETAIL, "surf_edges");
    CHECK_ERROR(surfedgeHandle.isValid(), "Failed to get surf_edges attribute");
    UT_IntArray surf_edges;
    surfedgeHandle.get(0, surf_edges);
    int numEdges = surf_edges.size() / 2;
    Eigen::MatrixXi surfEdge(numEdges, 2);
    for (int i = 0; i < numEdges; ++i) {
        surfEdge(i, 0) = surf_edges[i * 2];
        surfEdge(i, 1) = surf_edges[i * 2 + 1];
    }

    GA_ROHandleIA surfIndHandle(gdp, GA_ATTRIB_DETAIL, "surf_triangles");
    CHECK_ERROR(surfIndHandle.isValid(), "Failed to get surf_triangles attribute");
    UT_IntArray surf_triangles;
    surfIndHandle.get(0, surf_triangles);
    int numSurfTri = surf_triangles.size() / 3;
    Eigen::MatrixXi surfInd(numSurfTri, 3);
    for (int i = 0; i < numSurfTri; ++i) {
        surfInd(i, 0) = surf_triangles[i * 3];
        surfInd(i, 1) = surf_triangles[i * 3 + 1];
        surfInd(i, 2) = surf_triangles[i * 3 + 2];
    }

    instance->surfPos = surfPos;
    instance->surfInd = surfInd;
    instance->surfEdge = surfEdge;
    CUDAMallocSafe(instance->cudaSurfPos, surfPos.rows());
    CUDAMallocSafe(instance->cudaSurfInd, surfInd.rows());
    CUDAMallocSafe(instance->cudaSurfEdge, surfEdge.rows());
    copyToCUDASafe(instance->surfPos, instance->cudaSurfPos);
    copyToCUDASafe(instance->surfInd, instance->cudaSurfInd);
    copyToCUDASafe(instance->surfEdge, instance->cudaSurfEdge);

}


void GAS_Read_Buffer::loadSIMParams() {
    auto &instance = GeometryManager::instance;
    
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

    instance->MAX_CCD_COLLITION_PAIRS_NUM = 1 * FIRSTFRAME::collision_detection_buff_scale * (((double)(instance->surfInd.rows() * 15 + instance->surfEdge.rows() * 10)) * std::max((instance->IPC_dt / 0.01), 2.0));
    instance->MAX_COLLITION_PAIRS_NUM = (instance->surfPos.rows() * 3 + instance->surfEdge.rows() * 2) * 3 * FIRSTFRAME::collision_detection_buff_scale;

    std::cout << "arrived here ccd~" << instance->MAX_CCD_COLLITION_PAIRS_NUM << std::endl;
    std::cout << "arrived here collision~" << instance->MAX_COLLITION_PAIRS_NUM << std::endl;

}

void GAS_Read_Buffer::initSIMBVH() {
    auto &instance = GeometryManager::instance;
    if (!instance->LBVH_F_ptr) {
        instance->LBVH_F_ptr = std::make_unique<LBVH_F>();
    }
    if (!instance->LBVH_E_ptr) {
        instance->LBVH_E_ptr = std::make_unique<LBVH_E>();
    }

    instance->LBVH_F_ptr->init(
        instance->cudaBoundaryType, 
        instance->cudaTetPos,
        instance->cudaSurfInd,
        instance->surverts,
        instance->collisionpairs,
        instance->collidionspair,
        instance->_cpNum, 
        instance->_MatIndex, 
        instance->surface_Num, 
        instance->surf_vertexNum);




// _btype, _vertexes, _faces, _surfVerts, _collisonPairs, _ccd_collisonPairs, _cpNum, _MatIndex, surface_Num, surf_vertexNum

}