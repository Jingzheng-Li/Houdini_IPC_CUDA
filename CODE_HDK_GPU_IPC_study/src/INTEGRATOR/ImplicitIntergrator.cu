
#include "ImplicitIntergrator.cuh"

// TODO: split computeXTilta here

ImplicitIntegrator::ImplicitIntegrator(std::unique_ptr<GeometryManager>& instance) 
    : m_instance(instance) {}

ImplicitIntegrator::~ImplicitIntegrator() {};


// void ImplicitIntegrator::IPC_Solver() {

//     CHECK_ERROR(m_instance, "not initialize m_instance");
//     CHECK_ERROR(m_bvh_f, "not initialize m_bvh_f");
//     CHECK_ERROR(m_bvh_e, "not initialize m_bvh_e");
//     CHECK_ERROR(m_pcg_data, "not initialize m_pcg_data");
//     CHECK_ERROR(m_BH, "not initialize m_BH");


//     double alpha = 1;


//     if (m_isRotate) {
//         updateBoundaryMoveDir(m_instance, alpha);
//         buildBVH_FULLCCD(alpha);
//         buildFullCP(alpha);
//         if (h_ccd_cpNum > 0) {
//             double slackness_m = 0.8;
//             alpha = MATHUTILS::__m_min(alpha, self_largestFeasibleStepSize(slackness_m, m_pcg_data->mc_squeue, h_ccd_cpNum));
//         }
//         updateBoundary(m_instance, alpha);
//         CUDA_SAFE_CALL(cudaMemcpy(
//             m_instance->cudaTempDouble3Mem,
//             m_instance->cudaVertPos,
//             m_vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

//         updateBoundaryMoveDir(m_instance, alpha);
//         stepForward(m_instance->cudaVertPos, m_instance->cudaTempDouble3Mem, m_instance->cudaMoveDir, m_instance->cudaBoundaryType, 1, true, m_vertexNum);
//         buildBVH();
//         int numOfIntersect = 0;
//         while (isIntersected(m_instance)) {
//             //printf("type 0 intersection happened\n");
//             alpha /= 2.0;
//             updateBoundaryMoveDir(m_instance, alpha);
//             numOfIntersect++;
//             stepForward(m_instance->cudaVertPos, m_instance->cudaTempDouble3Mem, m_instance->cudaMoveDir, m_instance->cudaBoundaryType, 1, true, m_vertexNum);
//             buildBVH();
//         }
//         buildCP();
//     }

//     // calculate a lowerbound and upperbound of a kappa, mainly to keep stability of the system
//     upperBoundKappa(m_instance->Kappa);
//     if (m_instance->Kappa < 1e-16) {
//         // init Kappa, basically only active for 1st frame, to give you a first suggest kappa value.
//         suggestKappa(m_instance->Kappa);
//     }
//     initKappa(m_instance);


// #ifdef USE_FRICTION
//     CUDA_SAFE_CALL(cudaMalloc((void**)&mc_lambda_lastH_scalar, h_cpNum[0] * sizeof(double)));
//     CUDA_SAFE_CALL(cudaMalloc((void**)&mc_distCoord, h_cpNum[0] * sizeof(double2)));
//     CUDA_SAFE_CALL(cudaMalloc((void**)&mc_tanBasis, h_cpNum[0] * sizeof(MATHUTILS::Matrix3x2d)));
//     CUDA_SAFE_CALL(cudaMalloc((void**)&mc_collisonPairs_lastH, h_cpNum[0] * sizeof(int4)));
//     CUDA_SAFE_CALL(cudaMalloc((void**)&mc_MatIndex_last, h_cpNum[0] * sizeof(int)));
//     CUDA_SAFE_CALL(cudaMalloc((void**)&mc_lambda_lastH_scalar_gd, h_gpNum * sizeof(double)));
//     CUDA_SAFE_CALL(cudaMalloc((void**)&mc_collisonPairs_lastH_gd, h_gpNum * sizeof(uint32_t)));
//     buildFrictionSets();
// #endif

//     m_animation_fullRate = m_animation_subRate;

//     while (true) {
//         tempMalloc_closeConstraint();

//         CUDA_SAFE_CALL(cudaMemset(mc_close_cpNum, 0, sizeof(uint32_t)));
//         CUDA_SAFE_CALL(cudaMemset(mc_close_gpNum, 0, sizeof(uint32_t)));

//         solve_subIP(m_instance);

//         double2 minMaxDist1 = minMaxGroundDist();
//         double2 minMaxDist2 = minMaxSelfDist();

//         double minDist = MATHUTILS::__m_min(minMaxDist1.x, minMaxDist2.x);
//         double maxDist = MATHUTILS::__m_max(minMaxDist1.y, minMaxDist2.y);
        
//         bool finishMotion = m_animation_fullRate > 0.99 ? true : false;

//         if (finishMotion) {
//             if ((h_cpNum[0] + h_gpNum) > 0) {

//                 if (minDist < m_instance->dTol) {
//                     tempFree_closeConstraint();
//                     break;
//                 }
//                 else if (maxDist < m_instance->dHat) {
//                     tempFree_closeConstraint();
//                     break;
//                 }
//                 else {
//                     tempFree_closeConstraint();
//                 }
//             }
//             else {
//                 tempFree_closeConstraint();
//                 break;
//             }
//         }
//         else {
//             tempFree_closeConstraint();
//         }

//         m_animation_fullRate += m_animation_subRate;
        
// #ifdef USE_FRICTION
//         CUDA_SAFE_CALL(cudaFree(mc_lambda_lastH_scalar));
//         CUDA_SAFE_CALL(cudaFree(mc_distCoord));
//         CUDA_SAFE_CALL(cudaFree(mc_tanBasis));
//         CUDA_SAFE_CALL(cudaFree(mc_collisonPairs_lastH));
//         CUDA_SAFE_CALL(cudaFree(mc_MatIndex_last));
//         CUDA_SAFE_CALL(cudaFree(mc_lambda_lastH_scalar_gd));
//         CUDA_SAFE_CALL(cudaFree(mc_collisonPairs_lastH_gd));

//         CUDA_SAFE_CALL(cudaMalloc((void**)&mc_lambda_lastH_scalar, h_cpNum[0] * sizeof(double)));
//         CUDA_SAFE_CALL(cudaMalloc((void**)&mc_distCoord, h_cpNum[0] * sizeof(double2)));
//         CUDA_SAFE_CALL(cudaMalloc((void**)&mc_tanBasis, h_cpNum[0] * sizeof(MATHUTILS::Matrix3x2d)));
//         CUDA_SAFE_CALL(cudaMalloc((void**)&mc_collisonPairs_lastH, h_cpNum[0] * sizeof(int4)));
//         CUDA_SAFE_CALL(cudaMalloc((void**)&mc_MatIndex_last, h_cpNum[0] * sizeof(int)));
//         CUDA_SAFE_CALL(cudaMalloc((void**)&mc_lambda_lastH_scalar_gd, h_gpNum * sizeof(double)));
//         CUDA_SAFE_CALL(cudaMalloc((void**)&mc_collisonPairs_lastH_gd, h_gpNum * sizeof(uint32_t)));
//         buildFrictionSets();
// #endif
//     }


// #ifdef USE_FRICTION
//     CUDA_SAFE_CALL(cudaFree(mc_lambda_lastH_scalar));
//     CUDA_SAFE_CALL(cudaFree(mc_distCoord));
//     CUDA_SAFE_CALL(cudaFree(mc_tanBasis));
//     CUDA_SAFE_CALL(cudaFree(mc_collisonPairs_lastH));
//     CUDA_SAFE_CALL(cudaFree(mc_MatIndex_last));
//     CUDA_SAFE_CALL(cudaFree(mc_lambda_lastH_scalar_gd));
//     CUDA_SAFE_CALL(cudaFree(mc_collisonPairs_lastH_gd));
// #endif

//     updateVelocities(m_instance);

//     FEMENERGY::computeXTilta(m_instance, 1);

//     CUDA_SAFE_CALL(cudaDeviceSynchronize());


// }



// int GIPC::solve_subIP(std::unique_ptr<GeometryManager>& instance) {

//     int iterCap = 10000, iterk = 0;
//     CUDA_SAFE_CALL(cudaMemset(mc_moveDir, 0, m_vertexNum * sizeof(double3)));

//     m_total_Cg_count = 0;
//     m_totalCollisionPairs = 0;

//     for (; iterk < iterCap; ++iterk) {

//         m_totalCollisionPairs += h_cpNum[0];
        
//         m_BH->updateDNum(m_triangleNum, m_tetrahedraNum, h_cpNum + 1, h_cpNum_last + 1, m_tri_edge_num);

//         // calculate gradient gradx(g) and Hessian gradx^2(g)
//         computeGradientAndHessian(instance);

//         double distToOpt_PN = calcMinMovement(mc_moveDir, m_pcg_data->mc_squeue, m_vertexNum);
//         // line search iteration stop 
//         bool gradVanish = (distToOpt_PN < sqrt(instance->Newton_solver_threshold * instance->Newton_solver_threshold * instance->bboxDiagSize2 * instance->IPC_dt * instance->IPC_dt));
//         if (iterk > 0 && gradVanish) {
//             break;
//         }

//         // solve PCG with MAS Preconditioner and get mc_moveDir (i.e. dx)
//         m_total_Cg_count += calculateMovingDirection(instance, h_cpNum[0], instance->precondType);

//         double alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

//         alpha = MATHUTILS::__m_min(alpha, ground_largestFeasibleStepSize(slackness_a, m_pcg_data->mc_squeue));
//         // alpha = MATHUTILS::__m_min(alpha, InjectiveStepSize(0.2, 1e-6, m_pcg_data->mc_squeue, instance->cudaTetElement));
//         alpha = MATHUTILS::__m_min(alpha, self_largestFeasibleStepSize(slackness_m, m_pcg_data->mc_squeue, h_cpNum[0]));
        
//         double temp_alpha = alpha;
//         double alpha_CFL = alpha;

//         double ccd_size = 1.0;
// #ifdef USE_FRICTION
//         ccd_size = 0.6;
// #endif

//         // build BVH tree of type ccd, get collision pairs num h_ccd_cpNum, 
//         // if h_ccd_cpNum > 0, means there will be collision in temp_alpha substep
//         buildBVH_FULLCCD(temp_alpha);
//         buildFullCP(temp_alpha);
//         if (h_ccd_cpNum > 0) {
//             // obtain max velocity of moveDir
//             double maxSpeed = cfl_largestSpeed(m_pcg_data->mc_squeue);
//             alpha_CFL = sqrt(instance->dHat) / maxSpeed * 0.5;
//             alpha = MATHUTILS::__m_min(alpha, alpha_CFL);
//             if (temp_alpha > 2 * alpha_CFL) {
//                 alpha = MATHUTILS::__m_min(temp_alpha, self_largestFeasibleStepSize(slackness_m, m_pcg_data->mc_squeue, h_ccd_cpNum) * ccd_size);
//                 alpha = MATHUTILS::__m_max(alpha, alpha_CFL);
//             }
//         }

//         //printf("alpha:  %f\n", alpha);

//         lineSearch(instance, alpha, alpha_CFL);
//         postLineSearch(instance, alpha);

//         CUDA_SAFE_CALL(cudaDeviceSynchronize());

//     }
    
//     printf("\n");
//     printf("Kappa: %f  iteration k:  %d \n", instance->Kappa, iterk);
//     std::cout << "m_total_Cg_count: " << m_total_Cg_count << std::endl;
//     std::cout << "m_totalCollisionPairs: " << m_totalCollisionPairs << std::endl;
//     printf("\n");

//     return iterk;
   
// }


// void GIPC::lineSearch(std::unique_ptr<GeometryManager>& instance, double& alpha, const double& cfl_alpha) {

//     //buildCP();
//     double lastEnergyVal = computeEnergy(instance);
//     double c1m = 0.0;
//     double armijoParam = 0;
//     if (armijoParam > 0.0) {
//         c1m += armijoParam * Energy_Add_Reduction_Algorithm(3, instance);
//     }

//     CUDA_SAFE_CALL(cudaMemcpy(instance->cudaTempDouble3Mem, instance->cudaVertPos, m_vertexNum * sizeof(double3), cudaMemcpyDeviceToDevice));

//     stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, mc_moveDir, instance->cudaBoundaryType, alpha, false, m_vertexNum);

//     buildBVH();

//     // if (h_cpNum[0] > 0) system("pause");
//     int numOfIntersect = 0;
//     int insectNum = 0;

//     bool checkInterset = true;
//     // if under all ACCD/Ground/CFL defined alpha, still intersection happens
//     // then we return back to alpha/2 util find collision free alpha 
//     while (checkInterset && isIntersected(instance)) {
//         printf("type 0 intersection happened:  %d\n", insectNum);
//         insectNum++;
//         alpha /= 2.0;
//         numOfIntersect++;
//         alpha = MATHUTILS::__m_min(cfl_alpha, alpha);
//         stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, mc_moveDir, instance->cudaBoundaryType, alpha, false, m_vertexNum);
//         buildBVH();
//     }

//     buildCP();
//     //if (h_cpNum[0] > 0) system("pause");

//     //buildCollisionSets(mesh, sh, gd, true);
//     double testingE = computeEnergy(instance);

//     int numOfLineSearch = 0;
//     double LFStepSize = alpha;
//     //double temp_c1m = c1m;
//     std::cout.precision(18);
//     while ((testingE > lastEnergyVal + c1m * alpha) && alpha > 1e-3 * LFStepSize) {
//         printf("Enery not drop down, testE:%f, lastEnergyVal:%f, clm*alpha:%f\n", testingE, lastEnergyVal, c1m * alpha);
//         alpha /= 2.0;
//         ++numOfLineSearch;
//         stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, mc_moveDir, instance->cudaBoundaryType, alpha, false, m_vertexNum);
//         buildBVH();
//         buildCP();
//         testingE = computeEnergy(instance);
//     }
//     if (numOfLineSearch > 8) {
//         printf("!!!!!!!!!!!!! energy raise for %d times of numOfLineSearch\n", numOfLineSearch);
//     }
        
//     // if alpha fails down in past process, then check again will there be intersection again
//     if (alpha < LFStepSize) {
//         bool needRecomputeCS = false;
//         while (checkInterset && isIntersected(instance)) {
//             printf("type 1 intersection happened:  %d\n", insectNum);
//             insectNum++;
//             alpha /= 2.0;
//             numOfIntersect++;
//             alpha = MATHUTILS::__m_min(cfl_alpha, alpha);
//             stepForward(instance->cudaVertPos, instance->cudaTempDouble3Mem, mc_moveDir, instance->cudaBoundaryType, alpha, false, m_vertexNum);
//             buildBVH();
//             needRecomputeCS = true;
//         }
//         if (needRecomputeCS) {
//             buildCP();
//         }
//     }
//     printf("lineSearch time step:  %f\n", alpha);

// }


// void GIPC::postLineSearch(std::unique_ptr<GeometryManager>& instance, double alpha) {
//     if (m_instance->Kappa == 0.0) {
//         initKappa(instance);
//     }
//     else {

//         bool updateKappa = checkCloseGroundVal();
//         if (!updateKappa) {
//             updateKappa = checkSelfCloseVal();
//         }
//         if (updateKappa) {
//             m_instance->Kappa *= 2.0;
//             upperBoundKappa(m_instance->Kappa);
//         }
//         tempFree_closeConstraint();
//         tempMalloc_closeConstraint();
//         CUDA_SAFE_CALL(cudaMemset(mc_close_cpNum, 0, sizeof(uint32_t)));
//         CUDA_SAFE_CALL(cudaMemset(mc_close_gpNum, 0, sizeof(uint32_t)));

//         computeCloseGroundVal();

//         computeSelfCloseVal();
//     }
//     //printf("------------------------------------------Kappa: %f\n", Kappa);
// }


