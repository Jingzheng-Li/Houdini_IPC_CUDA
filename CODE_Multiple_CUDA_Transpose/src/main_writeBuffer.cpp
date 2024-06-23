
#include "main_writeBuffer.hpp"

const SIM_DopDescription* GAS_Write_Buffer::getDopDescription() {
    static PRM_Template templateList[] = {
        PRM_Template()
    };

    static SIM_DopDescription dopDescription(
        true,
        "gas_write_buffer", // internal name of the dop
        "Gas Write Buffer", // label of the dop
        "GasWriteBuffer", // template list for generating the dop
        classname(),
        templateList);

    setGasDescription(dopDescription);

    return &dopDescription;
}

GAS_Write_Buffer::GAS_Write_Buffer(const SIM_DataFactory* factory) : BaseClass(factory) {}

GAS_Write_Buffer::~GAS_Write_Buffer() {
    CUDAMemoryManager::free();
    GeometryManager::free();
}

bool GAS_Write_Buffer::solveGasSubclass(SIM_Engine& engine,
                                        SIM_Object* object,
                                        SIM_Time time,
                                        SIM_Time timestep) {

    SIM_GeometryCopy *newgeo = SIM_DATA_CREATE(*object, "Geometry", SIM_GeometryCopy, SIM_DATA_RETURN_EXISTING | SIM_DATA_ADOPT_EXISTING_ON_DELETE);
    CHECK_ERROR_SOLVER(newgeo!=NULL, "Failed to create writeBuffer GeometryCopy object");
    GU_DetailHandleAutoWriteLock writelock(newgeo->getOwnGeometry());
    CHECK_ERROR_SOLVER(writelock.isValid(), "Failed to get writeBuffer geometry detail");
    GU_Detail *gdp = writelock.getGdp();
    CHECK_ERROR_SOLVER(!gdp->isEmpty(), "writeBuffer Geometry is empty");

    transferPositionsTOHoudini(newgeo, gdp);

    return true;
}

void GAS_Write_Buffer::transferPositionsTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp) {
    
    CUDAMemoryManager::copyDataFromCUDA(GeometryManager::positions);
    Eigen::MatrixXf &positions = GeometryManager::positions;
    CHECK_ERROR((positions.rows()==gdp->getNumPoints()), "num particles not match");

    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        gdp->setPos3(ptoff, UT_Vector3(positions(ptoff, 0), positions(ptoff, 1), positions(ptoff, 2)));
    }

    CUDAMemoryManager::free();
    GeometryManager::free();
}



