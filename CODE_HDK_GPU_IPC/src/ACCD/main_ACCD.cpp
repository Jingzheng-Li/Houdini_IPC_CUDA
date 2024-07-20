
#include "main_ACCD.hpp"

#include "UTILS/GeometryManager.hpp"

// const SIM_DopDescription* GAS_CUDA_ACCD::getDopDescription() {
//     static PRM_Template templateList[] = {
//         PRM_Template()
//     };

//     static SIM_DopDescription dopDescription(
//         true,
//         "GAS_CUDA_ACCD", // internal name of the dop
//         "Gas CUDA ACCD", // label of the dop
//         "GasCUDAACCD", // template list for generating the dop
//         classname(),
//         templateList);

//     setGasDescription(dopDescription);

//     return &dopDescription;
// }

// GAS_CUDA_ACCD::GAS_CUDA_ACCD(const SIM_DataFactory* factory) : BaseClass(factory) {}


// GAS_CUDA_ACCD::~GAS_CUDA_ACCD() {
//     GeometryManager::totallyfree();
// }

// bool GAS_CUDA_ACCD::solveGasSubclass(SIM_Engine& engine,
//                                         SIM_Object* object,
//                                         SIM_Time time,
//                                         SIM_Time timestep) {

// 	// const SIM_Geometry *geo = object->getGeometry();
//     // CHECK_ERROR_SOLVER(geo != NULL, "Failed to get ACCD geometry object")
//     // GU_DetailHandleAutoReadLock readlock(geo->getGeometry());
//     // CHECK_ERROR_SOLVER(readlock.isValid(), "Failed to get ACCD geometry detail");
//     // const GU_Detail *gdp = readlock.getGdp();
//     // CHECK_ERROR_SOLVER(!gdp->isEmpty(), "readBuffer Geometry is empty");

    

//     return true;
// }

