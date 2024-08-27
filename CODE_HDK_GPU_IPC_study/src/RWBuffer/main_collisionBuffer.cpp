
#include "main_collisionBuffer.hpp"

const SIM_DopDescription* GAS_Collision_Buffer::getDopDescription() {
    static PRM_Template templateList[] = {
        PRM_Template()
    };

    static SIM_DopDescription dopDescription(
        true,
        "gas_collision_buffer", // internal name of the dop
        "Gas Collision Buffer", // label of the dop
        "GasCollisionBuffer", // template list for generating the dop
        classname(),
        templateList);

    setGasDescription(dopDescription);

    return &dopDescription;
}

GAS_Collision_Buffer::GAS_Collision_Buffer(const SIM_DataFactory* factory) : BaseClass(factory) {}

GAS_Collision_Buffer::~GAS_Collision_Buffer() {
    if (GeometryManager::instance) {
        GeometryManager::instance->freeGeometryManager();
    }
}

bool GAS_Collision_Buffer::solveGasSubclass(SIM_Engine& engine,
                                        SIM_Object* object,
                                        SIM_Time time,
                                        SIM_Time timestep) {


    std::cout << "time ~~~~~~~" << time << std::endl;

    std::cout << "run collision here~~~~~~~~" << std::endl;

    return true;
}
