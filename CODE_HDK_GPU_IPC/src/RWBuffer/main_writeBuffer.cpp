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
    GeometryManager::freeCUDAptr();
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

    Eigen::Matrix<uint32_t, Eigen::Dynamic, 1> sortmapvertindex;
    sortmapvertindex.resize(instance->numVertices);
    CUDAMemcpyDToHSafe(sortmapvertindex, instance->cudaSortMapVertIndex);
    CUDAMemcpyDToHSafe(instance->vertPos, instance->cudaVertPos);
    CUDAMemcpyDToHSafe(instance->vertVel, instance->cudaVertVel);

    Eigen::MatrixX3d &tetpos = GeometryManager::instance->vertPos;
    Eigen::MatrixX3d &tetvel = GeometryManager::instance->vertVel;
    CHECK_ERROR((tetpos.rows() == gdp->getNumPoints()), "Number of particles does not match");
    CHECK_ERROR((tetvel.rows() == gdp->getNumPoints()), "Number of velocities does not match");


    GA_RWHandleV3 velHandle(gdp, GA_ATTRIB_POINT, "v");
    GA_RWHandleF massHandle(gdp, GA_ATTRIB_POINT, "mass");
    CHECK_ERROR((velHandle.isValid() || massHandle.isValid()), "Failed to get velocity or mass attribute handle");

    GA_Offset ptoff;
    int ptidx = 0;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        int sortidx = sortmapvertindex(ptidx);
        gdp->setPos3(ptoff, UT_Vector3(tetpos(sortidx, 0), tetpos(sortidx, 1), tetpos(sortidx, 2)));
        // TODO: 写回速度需要小心一些 注意一下tetvel有没有速度变换的行为
        // velHandle.set(sortidx, UT_Vector3(tetvel(sortidx, 0), tetvel(sortidx, 1), tetvel(sortidx, 2)));
        ptidx++;
    }
    CHECK_ERROR(ptidx == instance->numVertices, "num vertices not match with writeback");

}
