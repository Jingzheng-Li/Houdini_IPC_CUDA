
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



    return true;
}


// void GAS_CUDA_Intergrator::transferDynamicCollisionToCUDA(SIM_Object* object) {

// 	auto &instance = GeometryManager::instance;
// 	CHECK_ERROR(instance, "loadSIMParams geoinstance not initialized");

//     SIM_ConstObjectArray colaffectors;
// 	object->getConstAffectors(colaffectors, "SIM_RelationshipCollide");
//     instance->collisionSurfFace.setZero();
//     int num_colobjects = 0;
    
//     // assume that we have multiple "static objects" in Dop network
//     for (exint colaffector = 0; colaffector < colaffectors.entries(); ++colaffector) {
// 		if (colaffectors(colaffector) == object) continue;
// 		const SIM_Geometry* colgeo = SIM_DATA_GETCONST(*colaffectors(colaffector), SIM_GEOMETRY_DATANAME, SIM_Geometry);
// 		if (colgeo == nullptr) continue;

//         GU_DetailHandleAutoReadLock readlock(colgeo->getGeometry());
//         const GU_Detail *colgdp = readlock.getGdp();
//         CHECK_ERROR(!colgdp->isEmpty(), "not get any collision objects gdp");

//         GA_Offset ptoff;
//         int ptidx = 0;
//         GA_FOR_ALL_PTOFF(colgdp, ptoff) {
//             UT_Vector3D pos3 = colgdp->getPos3D(ptoff);
//             instance->collisionSurfFace.row(ptidx) << pos3.x(), pos3.y(), pos3.z();
//             ptidx++;
//         }
//         CHECK_ERROR(ptidx==colgdp->getNumPoints(), "Failed to get all collision points");
//         num_colobjects += ptidx;
//     }
//     CHECK_ERROR(num_colobjects==instance->collisionSurfFace.rows(), "new collision objects have somehow changed it topology");

//     MatrixX3
//     std::cout << "collisionVerts~~~~~~~~" <<  << " " << collisionVerts.row(0).x() << " " << collisionVerts.row(0).y() << " " << collisionVerts.row(0).z() << std::endl;

// }

