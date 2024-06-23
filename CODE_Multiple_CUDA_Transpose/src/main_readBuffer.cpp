
#include "main_readBuffer.hpp"

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
    CUDAMemoryManager::free();
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

    transferPositionsTOCUDA(geo, gdp);

    return true;
}


void GAS_Read_Buffer::transferPositionsTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp) {

    Eigen::MatrixXf positions(gdp->getNumPoints(), 3);

    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        UT_Vector3 pos3 = gdp->getPos3(ptoff);
        positions.row(ptoff) << pos3.x(), pos3.y(), pos3.z();
    }

    GeometryManager::initialize(positions);
    CUDAMemoryManager::initialize(GeometryManager::positions);
    CUDAMemoryManager::copyDataToCUDA(GeometryManager::positions);

}
