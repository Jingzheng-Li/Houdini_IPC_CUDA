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

    SIM_ConstObjectArray affs;
	object->getConstAffectors(affs, "SIM_RelationshipCollide");
    Eigen::MatrixX3d collisionVerts;
    Eigen::MatrixX3d collisionVels;
    Eigen::MatrixX3i collisionElems;
    for (exint afi = 0; afi < affs.entries(); ++afi) {
		const SIM_Object* aff = affs(afi);
		if (aff == object) continue;
		const SIM_Geometry*affgeo = SIM_DATA_GETCONST(*aff, SIM_GEOMETRY_DATANAME, SIM_Geometry);
		if (affgeo == nullptr) continue;

        GU_DetailHandleAutoReadLock readlock(affgeo->getGeometry());
        const GU_Detail *gdp = readlock.getGdp();
        CHECK_ERROR_SOLVER(!gdp->isEmpty(), "not get any collision objects gdp");
        collisionVerts.conservativeResize(collisionVerts.rows() + gdp->getNumPoints(), Eigen::NoChange);
        collisionVels.conservativeResize(collisionVels.rows() + gdp->getNumPoints(), Eigen::NoChange);
        collisionElems.conservativeResize(collisionElems.rows() + gdp->getNumPrimitives(), Eigen::NoChange);
        
        GA_ROHandleV3D collisionVelHandle(gdp, GA_ATTRIB_POINT, "v");
        CHECK_ERROR_SOLVER(collisionVelHandle.isValid(), "Failed to get collision velocity");
        GA_Offset ptoff;
        int ptidx = 0;
        GA_FOR_ALL_PTOFF(gdp, ptoff) {
            UT_Vector3D pos3 = gdp->getPos3D(ptoff);
            UT_Vector3D vel3 = collisionVelHandle.get(ptoff);
            collisionVerts.row(ptidx) << pos3.x(), pos3.y(), pos3.z();
            collisionVels.row(ptidx) << vel3.x(), vel3.y(), vel3.z();
            ptidx++;
        }
        CHECK_ERROR_SOLVER(ptidx==gdp->getNumPoints(), "Failed to get all collision points");
    
    }
    std::cout << "collisionVerts~~~~~~~~" << collisionVerts.row(0).x() << " " << collisionVerts.row(0).y() << " " << collisionVerts.row(0).z() << std::endl;
    std::cout << "collisionVels~~~~~~~~" << collisionVels.row(0).x() << " " << collisionVels.row(0).y() << " " << collisionVels.row(0).z() << std::endl;

    
    // IPC_Solver();

    return true;
}

void GAS_CUDA_GIPC::IPC_Solver() {
    // auto &instance = GeometryManager::instance;
	// CHECK_ERROR(instance, "IPC_Solver geoinstance not initialized");
    // CHECK_ERROR(instance->GIPC_ptr, "IPC_Solver GIPC_ptr not initialized");


    // instance->GIPC_ptr->IPC_Solver();
    
}

