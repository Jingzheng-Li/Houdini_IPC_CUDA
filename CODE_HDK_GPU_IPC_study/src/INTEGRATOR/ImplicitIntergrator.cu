
#include "ImplicitIntergrator.cuh"
#include "FEM/FEMEnergy.cuh"

// TODO: split computeXTilta here

ImplicitIntegrator::ImplicitIntegrator(std::unique_ptr<GeometryManager>& instance) 
    : m_instance(instance),
    m_gipc(instance->GIPC_ptr),
    m_bvh_f(instance->LBVH_F_ptr),
    m_bvh_e(instance->LBVH_E_ptr),
    m_bvh_ef(instance->LBVH_EF_ptr),
    m_pcg_data(instance->PCGData_ptr),
    m_BH(instance->BH_ptr) {}

ImplicitIntegrator::~ImplicitIntegrator() {};


// __device__ bool cuda_error = false;




double calcMinMovement(const double3* _moveDir, double* _queue, const int& number) {
    int numbers = number;
    const unsigned int threadNum = default_threads;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);

    MATHUTILS::_reduct_max_double3_to_double << <blockNum, threadNum, sharedMsize >> > (_moveDir, _queue, numbers);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        MATHUTILS::__reduct_max_double << <blockNum, threadNum, sharedMsize >> > (_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;

    }
    //cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    double minValue;
    cudaMemcpy(&minValue, _queue, sizeof(double), cudaMemcpyDeviceToHost);

    return minValue;
}




void ImplicitIntegrator::computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance) {

    // rhs = M * (x_tilta - (xn + dt*vn)))
    FEMENERGY::calKineticGradient(
        instance->cudaVertPos, 
        instance->cudaXTilta, 
        instance->cudaFb, 
        instance->cudaVertMass, 
        instance->numVertices);

    CUDA_SAFE_CALL(cudaMemset(instance->cudaCPNum, 0, 5 * sizeof(uint32_t)));

    // calculate barrier gradient and Hessian
    m_gipc->calBarrierGradientAndHessian(
        instance->cudaFb, 
        m_instance->Kappa);

#ifdef USE_FRICTION
    m_gipc->calFrictionGradient(instance->cudaFb, instance);
    m_gipc->calFrictionHessian(instance);
#endif

    // rhs += -dt^2 * vol * force
    // lhs += dt^2 * H12x12
    FEMENERGY::calculate_tetrahedra_fem_gradient_hessian(
        instance->cudaTetDmInverses, 
        instance->cudaVertPos, 
        instance->cudaTetElement, 
        instance->cudaH12x12,
        instance->cpNum[4] + instance->cpNumLast[4], 
        instance->cudaTetVolume,
        instance->cudaFb, 
        instance->numTetElements, 
        m_instance->lengthRate, 
        m_instance->volumeRate, 
        m_instance->IPC_dt);

    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaD4Index + instance->cpNum[4] + instance->cpNumLast[4], instance->cudaTetElement, instance->numTetElements * sizeof(uint4),cudaMemcpyDeviceToDevice));

    // rhs += -dt^2 * area * force
    // lhs += dt^2 * H9x9
    FEMENERGY::calculate_triangle_fem_gradient_hessian(
        instance->cudaTriDmInverses, 
        instance->cudaVertPos, 
        instance->cudaTriElement, 
        instance->cudaH9x9, 
        instance->cpNum[3] + instance->cpNumLast[3], 
        instance->cudaTriArea, 
        instance->cudaFb, 
        instance->numTriElements, 
        m_instance->stretchStiff, 
        m_instance->shearStiff, 
        m_instance->IPC_dt);
    
    FEMENERGY::calculate_bending_gradient_hessian(
        instance->cudaVertPos, 
        instance->cudaRestVertPos, 
        instance->cudaTriEdges, 
        instance->cudaTriEdgeAdjVertex, 
        instance->cudaH12x12, 
        instance->cudaD4Index, 
        instance->cpNum[4] + instance->cpNumLast[4] + instance->numTetElements, 
        instance->cudaFb, 
        instance->numTriEdges, 
        m_instance->bendStiff, 
        m_instance->IPC_dt);

    CUDA_SAFE_CALL(cudaMemcpy(instance->cudaD3Index + instance->cpNum[3] + instance->cpNumLast[3], instance->cudaTriElement, instance->numTriElements * sizeof(uint3), cudaMemcpyDeviceToDevice));

    // calculate Ground gradient save in H3x3
    m_gipc->computeGroundGradientAndHessian(instance->cudaFb);

    // calcukate Soft Constraint Gradient and Hessian
    m_gipc->computeSoftConstraintGradientAndHessian(instance->cudaFb);

}




int ImplicitIntegrator::solve_subIP(std::unique_ptr<GeometryManager>& instance) {

    std::cout.precision(18);

    int iterCap = 10000, iterk = 0;
    CUDA_SAFE_CALL(cudaMemset(instance->cudaMoveDir, 0, instance->numVertices * sizeof(double3)));

    instance->totalPCGCount = 0;
    instance->totalCollisionPairs = 0;

    for (; iterk < iterCap; ++iterk) {

        instance->totalCollisionPairs += instance->cpNum[0];
        
        m_BH->updateDNum(instance->numTriElements, instance->numTetElements, instance->cpNum + 1, instance->cpNumLast + 1, instance->numTriEdges);

        // calculate gradient gradx(g) and Hessian gradx^2(g)
        computeGradientAndHessian(instance);

        double distToOpt_PN = calcMinMovement(instance->cudaMoveDir, m_pcg_data->mc_squeue, instance->numVertices);
        // line search iteration stop 
        bool gradVanish = (distToOpt_PN < sqrt(instance->Newton_solver_threshold * instance->Newton_solver_threshold * instance->bboxDiagSize2 * instance->IPC_dt * instance->IPC_dt));
        if (iterk > 0 && gradVanish) {
            break;
        }

        // solve PCG with MAS Preconditioner and get instance->cudaMoveDir (i.e. dx)
        instance->totalPCGCount += m_gipc->calculateMovingDirection(instance, instance->cpNum[0], instance->precondType);

        double alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

        alpha = MATHUTILS::__m_min(alpha, m_gipc->ground_largestFeasibleStepSize(slackness_a, m_pcg_data->mc_squeue));
        // alpha = MATHUTILS::__m_min(alpha, InjectiveStepSize(0.2, 1e-6, m_pcg_data->mc_squeue, instance->cudaTetElement));
        alpha = MATHUTILS::__m_min(alpha, m_gipc->self_largestFeasibleStepSize(slackness_m, m_pcg_data->mc_squeue, instance->cpNum[0]));
        
        double temp_alpha = alpha;
        double alpha_CFL = alpha;

        double ccd_size = 1.0;
#ifdef USE_FRICTION
        ccd_size = 0.6;
#endif

        // build BVH tree of type ccd, get collision pairs num instance->ccdCpNum, 
        // if instance->ccdCpNum > 0, means there will be collision in temp_alpha substep
        m_gipc->buildBVH_FULLCCD(temp_alpha);
        m_gipc->buildFullCP(temp_alpha);
        if (instance->ccdCpNum > 0) {
            // obtain max velocity of moveDir
            double maxSpeed = m_gipc->cfl_largestSpeed(m_pcg_data->mc_squeue);
            alpha_CFL = sqrt(instance->dHat) / maxSpeed * 0.5;
            alpha = MATHUTILS::__m_min(alpha, alpha_CFL);
            if (temp_alpha > 2 * alpha_CFL) {
                alpha = MATHUTILS::__m_min(temp_alpha, m_gipc->self_largestFeasibleStepSize(slackness_m, m_pcg_data->mc_squeue, instance->ccdCpNum) * ccd_size);
                alpha = MATHUTILS::__m_max(alpha, alpha_CFL);
            }
        }

        //printf("alpha:  %f\n", alpha);

        m_gipc->lineSearch(instance, alpha, alpha_CFL);
        m_gipc->postLineSearch(instance, alpha);

        CUDA_SAFE_CALL(cudaDeviceSynchronize());

    }
    
    printf("\n");
    printf("Kappa: %f  iteration k:  %d \n", instance->Kappa, iterk);
    std::cout << "instance->totalPCGCount: " << instance->totalPCGCount << std::endl;
    std::cout << "instance->totalCollisionPairs: " << instance->totalCollisionPairs << std::endl;
    printf("\n");

    return iterk;
   
}








bool ImplicitIntegrator::IPC_Solver() {

    // bool host_cuda_error = false;
    // cudaMemcpyToSymbol(cuda_error, &host_cuda_error, sizeof(bool));

    // calculate a lowerbound and upperbound of a kappa, mainly to keep stability of the system
    m_gipc->upperBoundKappa(m_instance->Kappa);
    if (m_instance->Kappa < 1e-16) {
        // init Kappa, basically only active for 1st frame, to give you a first suggest kappa value.
        m_gipc->suggestKappa(m_instance->Kappa);
    }
    m_gipc->initKappa(m_instance);


#ifdef USE_FRICTION
    CUDAMallocSafe(m_instance->cudaLambdaLastHScalar, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaDistCoord, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaTanBasis, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaCollisonPairsLastH, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaMatIndexLast, m_instance->cpNum[0]);
    CUDAMallocSafe(m_instance->cudaLambdaLastHScalarGd, m_instance->gpNum);
    CUDAMallocSafe(m_instance->cudaCollisonPairsLastHGd, m_instance->gpNum);
    m_gipc->buildFrictionSets();

#endif

    m_instance->animation_fullRate = m_instance->animation_subRate;

    while (true) {
        m_gipc->tempMalloc_closeConstraint();

        CUDAMemcpyHToDSafe(m_instance->cudaCloseCPNum, Eigen::VectorXi::Zero(1));
        CUDAMemcpyHToDSafe(m_instance->cudaCloseGPNum, Eigen::VectorXi::Zero(1));

        solve_subIP(m_instance);

        double2 minMaxDist1 = m_gipc->minMaxGroundDist();
        double2 minMaxDist2 = m_gipc->minMaxSelfDist();

        double minDist = MATHUTILS::__m_min(minMaxDist1.x, minMaxDist2.x);
        double maxDist = MATHUTILS::__m_max(minMaxDist1.y, minMaxDist2.y);
        
        bool finishMotion = m_instance->animation_fullRate > 0.99 ? true : false;

        if (finishMotion) {
            if ((m_instance->cpNum[0] + m_instance->gpNum) > 0) {

                if (minDist < m_instance->dTol) {
                    m_gipc->tempFree_closeConstraint();
                    break;
                }
                else if (maxDist < m_instance->dHat) {
                    m_gipc->tempFree_closeConstraint();
                    break;
                }
                else {
                    m_gipc->tempFree_closeConstraint();
                }
            }
            else {
                m_gipc->tempFree_closeConstraint();
                break;
            }
        }
        else {
            m_gipc->tempFree_closeConstraint();
        }

        m_instance->animation_fullRate += m_instance->animation_subRate;
        
#ifdef USE_FRICTION
        CUDAFreeSafe(m_instance->cudaLambdaLastHScalar);
        CUDAFreeSafe(m_instance->cudaDistCoord);
        CUDAFreeSafe(m_instance->cudaTanBasis);
        CUDAFreeSafe(m_instance->cudaCollisonPairsLastH);
        CUDAFreeSafe(m_instance->cudaMatIndexLast);
        CUDAFreeSafe(m_instance->cudaLambdaLastHScalarGd);
        CUDAFreeSafe(m_instance->cudaCollisonPairsLastHGd);

        CUDAMallocSafe(m_instance->cudaLambdaLastHScalar, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaDistCoord, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaTanBasis, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaCollisonPairsLastH, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaMatIndexLast, m_instance->cpNum[0]);
        CUDAMallocSafe(m_instance->cudaLambdaLastHScalarGd, m_instance->gpNum);
        CUDAMallocSafe(m_instance->cudaCollisonPairsLastHGd, m_instance->gpNum);
        m_gipc->buildFrictionSets();
#endif
    }


#ifdef USE_FRICTION
    CUDAFreeSafe(m_instance->cudaLambdaLastHScalar);
    CUDAFreeSafe(m_instance->cudaDistCoord);
    CUDAFreeSafe(m_instance->cudaTanBasis);
    CUDAFreeSafe(m_instance->cudaCollisonPairsLastH);
    CUDAFreeSafe(m_instance->cudaMatIndexLast);
    CUDAFreeSafe(m_instance->cudaLambdaLastHScalarGd);
    CUDAFreeSafe(m_instance->cudaCollisonPairsLastHGd);
#endif

    m_gipc->updateVelocities(m_instance);

    FEMENERGY::computeXTilta(m_instance, 1);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // cudaMemcpyFromSymbol(&host_cuda_error, cuda_error, sizeof(bool));

    // return host_cuda_error;

    return false;
}











