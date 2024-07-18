#include "main_writeBuffer.hpp"
#include "UTILS/GeometryManager.hpp"

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
    GeometryManager::totallyfree();
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

    transferPTAttribTOHoudini(newgeo, gdp);

    return true;
}

void GAS_Write_Buffer::transferPTAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp) {

    auto &instance = GeometryManager::instance;
    CHECK_ERROR(instance, "PT geoinstance to Houdini not initialized");

    copyFromCUDASafe(instance->tetPos, instance->cudaTetPos);
    copyFromCUDASafe(instance->tetVel, instance->cudaTetVel);

    Eigen::MatrixX3d &tetpos = GeometryManager::instance->tetPos;
    Eigen::MatrixX3d &tetvel = GeometryManager::instance->tetVel;
    CHECK_ERROR((tetpos.rows() == gdp->getNumPoints()), "Number of particles does not match");
    CHECK_ERROR((tetvel.rows() == gdp->getNumPoints()), "Number of velocities does not match");

    GA_RWHandleV3 velHandle(gdp, GA_ATTRIB_POINT, "v");
    GA_RWHandleF massHandle(gdp, GA_ATTRIB_POINT, "mass");
    CHECK_ERROR((velHandle.isValid() || massHandle.isValid()), "Failed to get velocity or mass attribute handle");

    GA_Offset ptoff;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        gdp->setPos3(ptoff, UT_Vector3(tetpos(ptoff, 0), tetpos(ptoff, 1), tetpos(ptoff, 2)));
        velHandle.set(ptoff, UT_Vector3(tetvel(ptoff, 0), tetvel(ptoff, 1), tetvel(ptoff, 2)));
    }

}
