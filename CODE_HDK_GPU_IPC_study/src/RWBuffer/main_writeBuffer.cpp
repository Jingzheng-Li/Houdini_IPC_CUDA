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
    if (GeometryManager::instance) {
        GeometryManager::instance->freeGeometryManager();
    }
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

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    return true;
}

void GAS_Write_Buffer::transferPTAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp) {

    auto &instance = GeometryManager::instance;
    CHECK_ERROR(instance, "PT geoinstance to Houdini not initialized");

    CUDAMemcpyDToHSafe(instance->vertPos, instance->cudaVertPos);
    CUDAMemcpyDToHSafe(instance->vertVel, instance->cudaVertVel);
    CHECK_ERROR(instance->numSIMVertPos == gdp->getNumPoints(), "Number of writebuffer particles does not match");
    CHECK_ERROR(instance->sortMapVertIndex.rows() == instance->numVertices, "Number of writebuffer sortMapVertIndex not match");


    GA_RWHandleV3 velHandle(gdp, GA_ATTRIB_POINT, "v");
    CHECK_ERROR(velHandle.isValid(), "Failed to get velocity or mass attribute handle");

    GA_Offset ptoff;
    int ptidx = 0;
    GA_FOR_ALL_PTOFF(gdp, ptoff) {
        int sortidx = instance->sortMapVertIndex(ptidx);
        gdp->setPos3(ptoff, UT_Vector3(
            instance->vertPos(sortidx, 0), 
            instance->vertPos(sortidx, 1), 
            instance->vertPos(sortidx, 2)));

        velHandle.set(ptoff, UT_Vector3(
            instance->vertVel(ptidx, 0),
            instance->vertVel(ptidx, 1),
            instance->vertVel(ptidx, 2)
        ));

        ptidx++;
        if (ptidx >= instance->numSIMVertPos) return; // we won't update collision mesh
    }
    CHECK_ERROR(ptidx == instance->numSIMVertPos, "num vertices not match with writeback");

}
