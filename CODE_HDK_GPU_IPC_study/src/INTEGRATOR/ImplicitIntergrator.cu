
#include "ImplicitIntergrator.cuh"
#include "FEM/FEMEnergy.cuh"

// TODO: split computeXTilta here

ImplicitIntegrator::ImplicitIntegrator(std::unique_ptr<GeometryManager>& instance) 
    : m_instance(instance) {}

ImplicitIntegrator::~ImplicitIntegrator() {};


// __device__ bool cuda_error = false;

// bool GIPC::IPC_Solver() {

//     bool host_cuda_error = false;
//     cudaMemcpyToSymbol(cuda_error, &host_cuda_error, sizeof(bool));

//     // calculate a lowerbound and upperbound of a kappa, mainly to keep stability of the system
//     upperBoundKappa(m_instance->Kappa);
//     if (m_instance->Kappa < 1e-16) {
//         // init Kappa, basically only active for 1st frame, to give you a first suggest kappa value.
//         suggestKappa(m_instance->Kappa);
//     }
//     initKappa(m_instance);


// #ifdef USE_FRICTION
//     CUDAMallocSafe(m_instance->cudaLambdaLastHScalar, m_instance->cpNum[0]);
//     CUDAMallocSafe(m_instance->cudaDistCoord, m_instance->cpNum[0]);
//     CUDAMallocSafe(m_instance->cudaTanBasis, m_instance->cpNum[0]);
//     CUDAMallocSafe(m_instance->cudaCollisonPairsLastH, m_instance->cpNum[0]);
//     CUDAMallocSafe(m_instance->cudaMatIndexLast, m_instance->cpNum[0]);
//     CUDAMallocSafe(m_instance->cudaLambdaLastHScalarGd, m_instance->gpNum);
//     CUDAMallocSafe(m_instance->cudaCollisonPairsLastHGd, m_instance->gpNum);
//     buildFrictionSets();

// #endif

//     m_instance->animation_fullRate = m_instance->animation_subRate;

//     while (true) {
//         tempMalloc_closeConstraint();

//         CUDAMemcpyHToDSafe(m_instance->cudaCloseCPNum, Eigen::VectorXi::Zero(1));
//         CUDAMemcpyHToDSafe(m_instance->cudaCloseGPNum, Eigen::VectorXi::Zero(1));

//         solve_subIP(m_instance);

//         double2 minMaxDist1 = minMaxGroundDist();
//         double2 minMaxDist2 = minMaxSelfDist();

//         double minDist = MATHUTILS::__m_min(minMaxDist1.x, minMaxDist2.x);
//         double maxDist = MATHUTILS::__m_max(minMaxDist1.y, minMaxDist2.y);
        
//         bool finishMotion = m_instance->animation_fullRate > 0.99 ? true : false;

//         if (finishMotion) {
//             if ((m_instance->cpNum[0] + m_instance->gpNum) > 0) {

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

//         m_instance->animation_fullRate += m_instance->animation_subRate;
        
// #ifdef USE_FRICTION
//         CUDAFreeSafe(m_instance->cudaLambdaLastHScalar);
//         CUDAFreeSafe(m_instance->cudaDistCoord);
//         CUDAFreeSafe(m_instance->cudaTanBasis);
//         CUDAFreeSafe(m_instance->cudaCollisonPairsLastH);
//         CUDAFreeSafe(m_instance->cudaMatIndexLast);
//         CUDAFreeSafe(m_instance->cudaLambdaLastHScalarGd);
//         CUDAFreeSafe(m_instance->cudaCollisonPairsLastHGd);

//         CUDAMallocSafe(m_instance->cudaLambdaLastHScalar, m_instance->cpNum[0]);
//         CUDAMallocSafe(m_instance->cudaDistCoord, m_instance->cpNum[0]);
//         CUDAMallocSafe(m_instance->cudaTanBasis, m_instance->cpNum[0]);
//         CUDAMallocSafe(m_instance->cudaCollisonPairsLastH, m_instance->cpNum[0]);
//         CUDAMallocSafe(m_instance->cudaMatIndexLast, m_instance->cpNum[0]);
//         CUDAMallocSafe(m_instance->cudaLambdaLastHScalarGd, m_instance->gpNum);
//         CUDAMallocSafe(m_instance->cudaCollisonPairsLastHGd, m_instance->gpNum);
//         buildFrictionSets();
// #endif
//     }


// #ifdef USE_FRICTION
//     CUDAFreeSafe(m_instance->cudaLambdaLastHScalar);
//     CUDAFreeSafe(m_instance->cudaDistCoord);
//     CUDAFreeSafe(m_instance->cudaTanBasis);
//     CUDAFreeSafe(m_instance->cudaCollisonPairsLastH);
//     CUDAFreeSafe(m_instance->cudaMatIndexLast);
//     CUDAFreeSafe(m_instance->cudaLambdaLastHScalarGd);
//     CUDAFreeSafe(m_instance->cudaCollisonPairsLastHGd);
// #endif

//     updateVelocities(m_instance);

//     FEMENERGY::computeXTilta(m_instance, 1);

//     CUDA_SAFE_CALL(cudaDeviceSynchronize());

//     cudaMemcpyFromSymbol(&host_cuda_error, cuda_error, sizeof(bool));

//     return host_cuda_error;

// }











