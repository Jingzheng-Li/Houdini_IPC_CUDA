
#include "main_readBuffer.hpp"

#include "UTILS/GeometryManager.hpp"

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

    transferPTAttribTOCUDA(geo, gdp);
    transferPRIMAttribTOCUDA(geo, gdp);
    transferDTAttribTOCUDA(geo, gdp);

    return true;
}


void GAS_Read_Buffer::transferPTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

    Eigen::MatrixXd tetpos(gdp->getNumPoints(), 3);
    Eigen::MatrixXd tetvel(gdp->getNumPoints(), 3);
    Eigen::VectorXd tetmass(gdp->getNumPoints());

    GA_ROHandleV3D velHandle(gdp, GA_ATTRIB_POINT, "v");
    GA_ROHandleD massHandle(gdp, GA_ATTRIB_POINT, "mass");
    CHECK_ERROR(velHandle.isValid() && massHandle.isValid(), "Failed to get velocity and mass attributes");

    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        UT_Vector3D pos3 = gdp->getPos3D(ptoff);
        UT_Vector3D vel3 = velHandle.get(ptoff);
        tetpos.row(ptoff) << pos3.x(), pos3.y(), pos3.z();
        tetvel.row(ptoff) << vel3.x(), vel3.y(), vel3.z();
        tetmass(ptoff) = massHandle.get(ptoff);
    }
    CHECK_ERROR(ptoff==gdp->getNumPoints(), "Failed to get all points");

    GeometryManager::initializePoints(tetpos, tetvel, tetmass);
    GeometryManager::copyPointsDataToCUDA();

}


void GAS_Read_Buffer::transferPRIMAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

    if (GeometryManager::instance->tetInd.rows() == gdp->getNumPrimitives()) return;

    Eigen::MatrixXi tetInd(gdp->getNumPrimitives(), 4);
    GA_Offset primoff;
    GA_FOR_ALL_PRIMOFF(gdp, primoff) {
        const GA_Primitive* prim = gdp->getPrimitive(primoff);
        for (int i = 0; i < prim->getVertexCount(); ++i) {
                GA_Offset vtxoff = prim->getVertexOffset(i);
                GA_Offset ptoff = gdp->vertexPoint(vtxoff);
                tetInd(primoff, i) = static_cast<int>(gdp->pointIndex(ptoff));
        }
    }
    CHECK_ERROR(primoff==gdp->getNumPrimitives(), "Failed to get all primitives");

    GeometryManager::initializePrims(tetInd);
    GeometryManager::copyPrimsDataToCUDA();

}


void GAS_Read_Buffer::transferDTAttribTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

    // TODO: surf info also be delete after each frame
    if (GeometryManager::instance->surfEdge.rows() > 0 && 
        GeometryManager::instance->surfInd.rows() > 0 &&
        GeometryManager::instance->surfPos.rows() > 0) return;


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

    GeometryManager::initializeSurfs(surfPos, surfInd, surfEdge);
    GeometryManager::copyDetailsDataToCUDA();

}