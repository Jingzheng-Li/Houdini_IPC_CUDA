#include "main_transformLissajous.hpp"

#include "UTILS/GeometryManager.hpp"
// #include "UTILS/CUDAMemoryManager.hpp"

const SIM_DopDescription* GAS_Transform_Lissajous::getDopDescription() {
    static PRM_Template templateList[] = {
        PRM_Template()
    };

    static SIM_DopDescription dopDescription(
        true,
        "gas_transform_Lissajous", // internal name of the dop
        "Gas Transform Lissajous", // label of the dop
        "GasTransformLissajous", // template list for generating the dop
        classname(),
        templateList);

    setGasDescription(dopDescription);

    return &dopDescription;
}

GAS_Transform_Lissajous::GAS_Transform_Lissajous(const SIM_DataFactory* factory) : BaseClass(factory) {}

GAS_Transform_Lissajous::~GAS_Transform_Lissajous() {
    GeometryManager::free();
}

bool GAS_Transform_Lissajous::solveGasSubclass(SIM_Engine& engine,
                                               SIM_Object* object,
                                               SIM_Time time,
                                               SIM_Time timestep) {

    int numPoints = GeometryManager::instance->positions.rows();

    std::cout << "transform numpoints" << numPoints << std::endl;

    transformPositions(time, numPoints);

    return true;
}

void GAS_Transform_Lissajous::transformPositions(SIM_Time time, int numPoints) {
    // double3* cudaTetPos = CUDAMemoryManager::instance->cudaTetPos;
    double3* cudaTetPos = GeometryManager::instance->cudaTetPos;

    transformLissajousCUDA(cudaTetPos, numPoints, time);
}
