
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

    transferDynamicCollisionToCUDA(object);

    return true;
}


void GAS_CUDA_Intergrator::transferDynamicCollisionToCUDA(SIM_Object* object) {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "loadSIMParams geoinstance not initialized");
    CHECK_ERROR(instance->collisionSurfVert.rows()!=0, "collision objects not initialized");

    SIM_ConstObjectArray colaffectors;
	object->getConstAffectors(colaffectors, "SIM_RelationshipCollide");
    CHECK_ERROR(colaffectors.entries() == 2, "houdini scene static object is not correct, we only allow one static object exists");
	
	// colaffector(0) will be our collsion geometry
	const SIM_Geometry* collidegeo = SIM_DATA_GETCONST(*colaffectors(0), SIM_GEOMETRY_DATANAME, SIM_Geometry);
	CHECK_ERROR(collidegeo != nullptr, "collidegeo is nullptr right now");
	GU_DetailHandleAutoReadLock readlock(collidegeo->getGeometry());
	const GU_Detail *collidegdp = readlock.getGdp();
	CHECK_ERROR(!collidegdp->isEmpty(), "not get any collision objects gdp");

    CHECK_ERROR(collidegdp->getNumPoints()==instance->collisionSurfVert.rows(), "new collide geo topology has changed somehow");
	instance->collisionSurfVert.setZero();

	GA_Offset ptoff;
	int ptidx = 0;
	GA_FOR_ALL_PTOFF(collidegdp, ptoff) {
        UT_Vector3D pos3 = collidegdp->getPos3D(ptoff);
        instance->collisionSurfVert.row(ptidx) << pos3.x(), pos3.y(), pos3.z();
		ptidx++;
	}
	CHECK_ERROR(ptidx==collidegdp->getNumPoints(), "Failed to get all collision points");

    // std::cout << "colliefsf " << instance->collisionSurfVert.row(instance->collisionSurfVert.rows()-1) << std::endl;

}

