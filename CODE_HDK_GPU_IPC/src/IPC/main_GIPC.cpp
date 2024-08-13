// 等下还是依照libwetcloth把东西都写到一个节点里面，等到运行没有问题了，在拆分出来

#include "main_GIPC.hpp"

#include "UTILS/GeometryManager.hpp"

const SIM_DopDescription* GAS_CUDA_GIPC::getDopDescription() {
    static PRM_Template templateList[] = {
        PRM_Template()
    };

    static SIM_DopDescription dopDescription(
        true,
        "GAS_CUDA_GIPC", // internal name of the dop
        "GAS CUDA GIPC", // label of the dop
        "GasCUDAGIPC", // template list for generating the dop
        classname(),
        templateList);

    setGasDescription(dopDescription);

    return &dopDescription;
}

GAS_CUDA_GIPC::GAS_CUDA_GIPC(const SIM_DataFactory* factory) : BaseClass(factory) {}

GAS_CUDA_GIPC::~GAS_CUDA_GIPC() {
    if (GeometryManager::instance) {
        // GeometryManager::instance->freeGeometryManager();
    }
}

bool GAS_CUDA_GIPC::solveGasSubclass(SIM_Engine& engine,
                                    SIM_Object* object,
                                    SIM_Time time,
                                    SIM_Time timestep) {

    IPC_Solver();

    return true;
}

void GAS_CUDA_GIPC::IPC_Solver() {
    auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "IPC_Solver geoinstance not initialized");

    if (!instance->GIPC_ptr) {
        instance->GIPC_ptr = std::make_unique<GIPC>(instance);
    }
    CHECK_ERROR(instance->GIPC_ptr, "IPC_Solver GIPC_ptr not initialized");

    instance->GIPC_ptr->IPC_Solver();
    
}

