
#include "main_Intergrator.hpp"

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


    std::cout << "time ~~~~~~~" << time << std::endl;

    std::cout << "run intergrator here~~~~~~~~" << std::endl;

    return true;
}
