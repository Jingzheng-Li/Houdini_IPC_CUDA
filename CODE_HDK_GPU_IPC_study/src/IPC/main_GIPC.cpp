// 等下还是依照libwetcloth把东西都写到一个节点里面，等到运行没有问题了，在拆分出来

#include "main_GIPC.hpp"

#include "UTILS/GeometryManager.hpp"

const SIM_DopDescription* GAS_CUDA_GIPC::getDopDescription() {
    static PRM_Template templateList[] = {
        PRM_Template()
    };

    static SIM_DopDescription dopDescription(
        true,
        "gas_cuda_gipc", // internal name of the dop
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
        GeometryManager::instance->freeGeometryManager();
    }
}

bool GAS_CUDA_GIPC::solveGasSubclass(SIM_Engine& engine,
                                    SIM_Object* object,
                                    SIM_Time time,
                                    SIM_Time timestep) {


    // gas_IPC_Solver();

    return true;
}

// void GAS_CUDA_GIPC::gas_IPC_Solver() {
//     auto &instance = GeometryManager::instance;
// 	CHECK_ERROR(instance, "gas_IPC_Solver geoinstance not initialized");
//     CHECK_ERROR(instance->GIPC_ptr, "gas_IPC_Solver GIPC_ptr not initialized");
//     CHECK_ERROR(instance->LBVH_E_ptr, "not initialize m_bvh_f");
//     CHECK_ERROR(instance->LBVH_F_ptr, "not initialize m_bvh_e");
//     CHECK_ERROR(instance->LBVH_EF_ptr, "not initialize m_bvh_ef");
//     CHECK_ERROR(instance->PCGData_ptr, "not initialize m_pcg_data");

//     for (int i = 0; i < instance->IPC_substep; i++) {
//         bool cuda_error = instance->GIPC_ptr->IPC_Solver();
//         CHECK_ERROR(!cuda_error, "IPC_Solver meet some errors, please check what happens");
//     }
    
// }

