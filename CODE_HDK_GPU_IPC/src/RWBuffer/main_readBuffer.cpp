
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
    GeometryManager::free();
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