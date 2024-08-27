
#include "main_Intergrator.hpp"
#include "ImplicitIntergrator.cuh"

const SIM_DopDescription* GAS_CUDA_Intergrator::getDopDescription() {
    static PRM_Template templateList[] = {
        PRM_Template()
    };

    static SIM_DopDescription dopDescription(
        true,
        "gas_cuda_intergrator", // internal name of the dop
        "Gas CUDA_Intergrator", // label of the dop
        "GasCUDAIntergrator", // template list for generating the dop
        classname(),
        templateList);

    setGasDescription(dopDescription);

    return &dopDescription;
}

GAS_CUDA_Intergrator::GAS_CUDA_Intergrator(const SIM_DataFactory* factory) : BaseClass(factory) {}

GAS_CUDA_Intergrator::~GAS_CUDA_Intergrator() {
    if (GeometryManager::instance) {
        GeometryManager::instance->freeGeometryManager();
    }
}

bool GAS_CUDA_Intergrator::solveGasSubclass(SIM_Engine& engine,
                                        SIM_Object* object,
                                        SIM_Time time,
                                        SIM_Time timestep) {

    IPC_Solver();

    return true;
}

void GAS_CUDA_Intergrator::IPC_Solver() {
    auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "GAS_CUDA_Intergrator IPC_Solver geoinstance not initialized");
    CHECK_ERROR(instance->Integrator_ptr, "GAS_CUDA_Intergrator IPC_Solver Integrator_ptr not initialized");

    instance->Integrator_ptr->IPC_Solver();

}
