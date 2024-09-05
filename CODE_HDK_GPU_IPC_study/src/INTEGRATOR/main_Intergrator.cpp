
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

    debugPrint();

    gas_IPC_Solver();

    return true;
}


void GAS_CUDA_Intergrator::transferDynamicCollisionToCUDA(SIM_Object* object) {
	auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "loadSIMParams geoinstance not initialized");
    CHECK_ERROR(instance->collisionVertPos.rows()!=0, "collision objects not initialized");

    SIM_ConstObjectArray colaffectors;
	object->getConstAffectors(colaffectors, "SIM_RelationshipCollide");
    CHECK_ERROR(colaffectors.entries() == 2, "houdini scene static object is not correct, we only allow one static object exists");
	
	// colaffector(0) will be our collsion geometry
	const SIM_Geometry* collidegeo = SIM_DATA_GETCONST(*colaffectors(0), SIM_GEOMETRY_DATANAME, SIM_Geometry);
	CHECK_ERROR(collidegeo != nullptr, "collidegeo is nullptr right now");
	GU_DetailHandleAutoReadLock readlock(collidegeo->getGeometry());
	const GU_Detail *collidegdp = readlock.getGdp();
	CHECK_ERROR(!collidegdp->isEmpty(), "not get any collision objects gdp");

    CHECK_ERROR(collidegdp->getNumPoints()==instance->collisionVertPos.rows(), "new collide geo topology has changed somehow");
	instance->collisionVertPos.setZero();

	GA_Offset ptoff;
	int ptidx = 0;
	GA_FOR_ALL_PTOFF(collidegdp, ptoff) {
        UT_Vector3D pos3 = collidegdp->getPos3D(ptoff);
        instance->collisionVertPos.row(ptidx) << pos3.x(), pos3.y(), pos3.z();
		ptidx++;
	}
	CHECK_ERROR(ptidx==collidegdp->getNumPoints(), "Failed to get all collision points");

    CUDAMemcpyHToDSafe(instance->cudaTargetVertPos, instance->collisionVertPos);

}


void GAS_CUDA_Intergrator::gas_IPC_Solver() {
    auto &instance = GeometryManager::instance;
	CHECK_ERROR(instance, "gas_IPC_Solver geoinstance not initialized");
    CHECK_ERROR(instance->LBVH_E_ptr, "not initialize m_bvh_f");
    CHECK_ERROR(instance->LBVH_F_ptr, "not initialize m_bvh_e");
    CHECK_ERROR(instance->LBVH_EF_ptr, "not initialize m_bvh_ef");
    CHECK_ERROR(instance->PCGData_ptr, "not initialize m_pcg_data");
    CHECK_ERROR(instance->Integrator_ptr, "not initialize Integrator_ptr");

    for (int i = 0; i < instance->IPC_substep; i++) {
        bool cuda_error = instance->Integrator_ptr->IPC_Solver();
        CHECK_ERROR(!cuda_error, "IPC_Solver meet some errors, please check what happens");
    }
    
}


double3 add_double3(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

void GAS_CUDA_Intergrator::debugPrint() {
    auto &instance = GeometryManager::instance;

    // std::cout << "IPC_substep: " << instance->IPC_substep << std::endl;
    // std::cout << "IPC_fps: " << instance->IPC_fps << std::endl;
    // std::cout << "IPC_dt: " << instance->IPC_dt << std::endl;
    // std::cout << "precondType: " << instance->precondType << std::endl;

    // // Printing physical properties
    // std::cout << "density: " << instance->density << std::endl;
    // std::cout << "YoungModulus: " << instance->YoungModulus << std::endl;
    // std::cout << "PoissonRate: " << instance->PoissonRate << std::endl;
    // std::cout << "lengthRateLame: " << instance->lengthRateLame << std::endl;
    // std::cout << "volumeRateLame: " << instance->volumeRateLame << std::endl;
    // std::cout << "lengthRate: " << instance->lengthRate << std::endl;
    // std::cout << "volumeRate: " << instance->volumeRate << std::endl;

    // // Printing cloth parameters
    // std::cout << "frictionRate: " << instance->frictionRate << std::endl;
    // std::cout << "clothDensity: " << instance->clothDensity << std::endl;
    // std::cout << "clothThickness: " << instance->clothThickness << std::endl;
    // std::cout << "clothYoungModulus: " << instance->clothYoungModulus << std::endl;
    // std::cout << "stretchStiff: " << instance->stretchStiff << std::endl;
    // std::cout << "shearStiff: " << instance->shearStiff << std::endl;
    // std::cout << "bendStiff: " << instance->bendStiff << std::endl;

    // // Printing collision and solver parameters
    // std::cout << "relative_dhat: " << instance->relative_dhat << std::endl;
    // std::cout << "Newton_solver_threshold: " << instance->Newton_solver_threshold << std::endl;
    // std::cout << "pcg_threshold: " << instance->pcg_threshold << std::endl;
    // std::cout << "collision_detection_buff_scale: " << instance->collision_detection_buff_scale << std::endl;

    // // Printing motion and animation rates
    // std::cout << "softMotionRate: " << instance->softMotionRate << std::endl;
    // std::cout << "softAnimationSubRate: " << instance->softAnimationSubRate << std::endl;
    // std::cout << "softAnimationFullRate: " << instance->softAnimationFullRate << std::endl;

    // // Printing ground offsets
    // std::cout << "ground_bottom_offset: " << instance->ground_bottom_offset << std::endl;
    // std::cout << "ground_left_offset: " << instance->ground_left_offset << std::endl;
    // std::cout << "ground_right_offset: " << instance->ground_right_offset << std::endl;
    // std::cout << "ground_near_offset: " << instance->ground_near_offset << std::endl;
    // std::cout << "ground_far_offset: " << instance->ground_far_offset << std::endl;

    // // Printing forces
    // std::cout << "gravityforce: (" << instance->gravityforce.x << ", " 
    //           << instance->gravityforce.y << ", " << instance->gravityforce.z << ")" << std::endl;
    // std::cout << "windforce: (" << instance->windforce.x << ", " 
    //           << instance->windforce.y << ", " << instance->windforce.z << ")" << std::endl;

    // std::cout << "numSoftConstraints: " << instance->numSoftConstraints << std::endl;





    // std::vector<double3> new_originvertpos(instance->numVertices);
    // CUDA_SAFE_CALL(cudaMemcpy(new_originvertpos.data(), instance->cudaOriginVertPos, instance->numVertices * sizeof(double3), cudaMemcpyDeviceToHost));
    // double3 sum_originvertpos = std::accumulate(new_originvertpos.begin(), new_originvertpos.end(), make_double3(0.0, 0.0, 0.0), add_double3);
    // std::cout << "sum_originvertpos! " << sum_originvertpos.x << " " << sum_originvertpos.y << " " << sum_originvertpos.z << std::endl;
    // for (auto& overtpos : new_originvertpos) {
    //     std::cout << overtpos.x << " " << overtpos.y << " " << overtpos.z << std::endl;
    // }



    // matindex


}



















