
#include "main_LBVH.hpp"
#include "LBVH.cuh"
#include "UTILS/GeometryManager.hpp"

const SIM_DopDescription* GAS_CUDA_LBVH::getDopDescription() {
    static PRM_Template templateList[] = {
        PRM_Template()
    };

    static SIM_DopDescription dopDescription(
        true,
        "GAS_CUDA_LBVH", // internal name of the dop
        "Gas CUDA LBVH", // label of the dop
        "GasCUDALBVH", // template list for generating the dop
        classname(),
        templateList);

    setGasDescription(dopDescription);

    return &dopDescription;
}

GAS_CUDA_LBVH::GAS_CUDA_LBVH(const SIM_DataFactory* factory) : BaseClass(factory) {}

GAS_CUDA_LBVH::~GAS_CUDA_LBVH() {
    GeometryManager::free();
}

bool GAS_CUDA_LBVH::solveGasSubclass(SIM_Engine& engine,
                                               SIM_Object* object,
                                               SIM_Time time,
                                               SIM_Time timestep) {
    
    buildBVH();
    buildBVH_FULLCCD();


    return true;
}

void GAS_CUDA_LBVH::buildBVH() {

}

void GAS_CUDA_LBVH::buildBVH_FULLCCD() {

}

