
#include <fstream>

#include "FEMEnergy.cuh"
#include "GIPC.cuh"
#include "GeometryManager.hpp"
#include "ImplicitIntegrator.cuh"

namespace __INTEGRATOR__ {

void tempMalloc_closeConstraint(std::unique_ptr<GeometryManager>& instance) {
    // CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaCloseConstraintID(),
    //                           instance->getHostGpNum() * sizeof(uint32_t)));
    // CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaCloseConstraintVal(),
    //                           instance->getHostGpNum() * sizeof(Scalar)));
    // CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaCloseMConstraintID(),
    //                           instance->getHostCpNum(0) * sizeof(int4)));
    // CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaCloseMConstraintVal(),
    //                           instance->getHostCpNum(0) * sizeof(Scalar)));

    CUDAMallocSafe(instance->getCudaCloseConstraintID(), instance->getHostGpNum());
    CUDAMallocSafe(instance->getCudaCloseConstraintVal(), instance->getHostGpNum());
    CUDAMallocSafe(instance->getCudaCloseMConstraintID(), instance->getHostCpNum(0));
    CUDAMallocSafe(instance->getCudaCloseMConstraintVal(), instance->getHostCpNum(0));

}

void tempFree_closeConstraint(std::unique_ptr<GeometryManager>& instance) {
    // CUDA_SAFE_CALL(cudaFree(instance->getCudaCloseConstraintID()));
    // CUDA_SAFE_CALL(cudaFree(instance->getCudaCloseConstraintVal()));
    // CUDA_SAFE_CALL(cudaFree(instance->getCudaCloseMConstraintID()));
    // CUDA_SAFE_CALL(cudaFree(instance->getCudaCloseMConstraintVal()));

    CUDAFreeSafe(instance->getCudaCloseConstraintID());
    CUDAFreeSafe(instance->getCudaCloseConstraintVal());
    CUDAFreeSafe(instance->getCudaCloseMConstraintID());
    CUDAFreeSafe(instance->getCudaCloseMConstraintVal());

}

__global__ void _computeXTilta(int* _btype, Scalar3* _velocities, Scalar3* _o_vertexes,
                               Scalar3* _xTilta, Scalar ipc_dt, Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    Scalar3 gravityDtSq =
        make_Scalar3(0, 0, 0);  //__MATHUTILS__::__s_vec_multiply(make_Scalar3(0, -9.8, 0),
                                // ipc_dt * ipc_dt);//Vector3d(0, gravity, 0) *
                                // instance->getHostIPCDt() * instance->getHostIPCDt();
    if (_btype[idx] == 0) {
        gravityDtSq = __MATHUTILS__::__s_vec_multiply(make_Scalar3(0, -9.8, 0), ipc_dt * ipc_dt);
    }
    _xTilta[idx] = __MATHUTILS__::__add(
        _o_vertexes[idx],
        __MATHUTILS__::__add(__MATHUTILS__::__s_vec_multiply(_velocities[idx], ipc_dt),
                             gravityDtSq));  //(mesh.V_prev[vI] + (mesh.velocities[vI] *
                                             // instance->getHostIPCDt() + gravityDtSq));
}

__global__ void _updateVelocities(Scalar3* _vertexes, Scalar3* _o_vertexes, Scalar3* _velocities,
                                  int* btype, Scalar ipc_dt, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (btype[idx] == 0) {
        _velocities[idx] = __MATHUTILS__::__s_vec_multiply(
            __MATHUTILS__::__minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        _o_vertexes[idx] = _vertexes[idx];
    } else {
        _velocities[idx] = make_Scalar3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

__global__ void _changeBoundarytoSIMPoint(int* _btype, __MATHUTILS__::Matrix3x3S* _constraints,
                                          int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    if ((_btype[idx]) == 1) {
        _btype[idx] = 0;
        __MATHUTILS__::__set_Mat_val(_constraints[idx], 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}

__global__ void _getKineticEnergy_Reduction_3D(Scalar3* _vertexes, Scalar3* _xTilta,
                                               Scalar* _energy, Scalar* _masses, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= number) return;

    Scalar temp =
        __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(_vertexes[idx], _xTilta[idx])) *
        _masses[idx] * 0.5;

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }
    if (threadIdx.x == 0) {
        _energy[blockIdx.x] = temp;
    }
}

__global__ void _stepForward(Scalar3* _vertexes, Scalar3* _vertexesTemp, Scalar3* _moveDir,
                             int* bType, Scalar alpha, bool moveBoundary, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (abs(bType[idx]) != 1 || moveBoundary) {
        _vertexes[idx] = __MATHUTILS__::__minus(
            _vertexesTemp[idx], __MATHUTILS__::__s_vec_multiply(_moveDir[idx], alpha));
    }
}

__global__ void _getDeltaEnergy_Reduction(Scalar* squeue, const Scalar3* b, const Scalar3* dx,
                                          int vertexNum) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];
    int numbers = vertexNum;
    if (idx >= numbers) return;
    // int cfid = tid + CONFLICT_FREE_OFFSET(tid);

    Scalar temp = __MATHUTILS__::__v_vec_dot(b[idx], dx[idx]);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _reduct_MGroundDist(const Scalar3* vertexes, const Scalar* g_offset,
                                    const Scalar3* g_normal, uint32_t* _environment_collisionPair,
                                    Scalar2* _queue, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar2 sdata[];

    if (idx >= number) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__v_vec_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar tempv = dist * dist;
    Scalar2 temp = make_Scalar2(1.0 / tempv, tempv);

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
        temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
            temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _queue[blockIdx.x] = temp;
    }
}



__device__ Scalar _selfConstraintVal_integrator(const Scalar3* vertexes, const int4& active) {
    Scalar val;
    if (active.x >= 0) {
        if (active.w >= 0) {
            __MATHUTILS__::_d_EE(vertexes[active.x], vertexes[active.y], vertexes[active.z],
                                 vertexes[active.w], val);
        } else {
            __MATHUTILS__::_d_EE(vertexes[active.x], vertexes[active.y], vertexes[active.z],
                                 vertexes[-active.w - 1], val);
        }
    } else {
        if (active.z < 0) {
            if (active.y < 0) {
                __MATHUTILS__::_d_PP(vertexes[-active.x - 1], vertexes[-active.y - 1], val);
            } else {
                __MATHUTILS__::_d_PP(vertexes[-active.x - 1], vertexes[active.y], val);
            }
        } else if (active.w < 0) {
            if (active.y < 0) {
                __MATHUTILS__::_d_PE(vertexes[-active.x - 1], vertexes[-active.y - 1],
                                     vertexes[active.z], val);
            } else {
                __MATHUTILS__::_d_PE(vertexes[-active.x - 1], vertexes[active.y],
                                     vertexes[active.z], val);
            }
        } else {
            __MATHUTILS__::_d_PT(vertexes[-active.x - 1], vertexes[active.y], vertexes[active.z],
                                 vertexes[active.w], val);
        }
    }
    return val;
}

__global__ void _reduct_MSelfDist(const Scalar3* _vertexes, int4* _collisionPairs, Scalar2* _queue,
                                  int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar2 sdata[];

    if (idx >= number) return;
    int4 MMCVIDI = _collisionPairs[idx];
    Scalar tempv = _selfConstraintVal_integrator(_vertexes, MMCVIDI);
    Scalar2 temp = make_Scalar2(1.0 / tempv, tempv);
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
        temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
            temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _queue[blockIdx.x] = temp;
    }
}

Scalar2 minMaxGroundDist(std::unique_ptr<GeometryManager>& instance) {
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes,
    //_groundOffset, _groundNormal, _isChange, _closeConstraintID,
    //_closeConstraintVal, numbers);

    int numbers = instance->getHostGpNum();
    if (numbers < 1) return make_Scalar2(1e32, 0);
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar2) * (threadNum >> 5);

    Scalar2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(Scalar2)));
    // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number *
    // sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist<<<blockNum, threadNum, sharedMsize>>>(
        instance->getCudaVertPos(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(), _queue, numbers);
    //_reduct_min_Scalar3_to_Scalar << <blockNum, threadNum, sharedMsize >> >
    //(_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> >
        //(_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB),
    // cudaMemcpyDeviceToDevice);
    Scalar2 minMaxValue;
    cudaMemcpy(&minMaxValue, _queue, sizeof(Scalar2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minMaxValue.x = 1.0 / minMaxValue.x;
    return minMaxValue;
}

Scalar calcMinMovement(const Scalar3* _moveDir, Scalar* _queue, const int& number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    /*Scalar* _tempMinMovement;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_tempMinMovement, numbers *
    sizeof(Scalar)));*/
    // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number *
    // sizeof(AABB), cudaMemcpyDeviceToDevice));

    __MATHUTILS__::_reduct_max_Scalar3_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(_moveDir, _queue, numbers);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> >
        //(_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB),
    // cudaMemcpyDeviceToDevice);
    Scalar minValue;
    cudaMemcpy(&minValue, _queue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // CUDA_SAFE_CALL(cudaFree(_tempMinMovement));
    return minValue;
}

void stepForward(Scalar3* _vertexes, Scalar3* _vertexesTemp, Scalar3* _moveDir, int* bType,
                 Scalar alpha, bool moveBoundary, int numbers) {
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _stepForward<<<blockNum, threadNum>>>(_vertexes, _vertexesTemp, _moveDir, bType, alpha,
                                          moveBoundary, numbers);
}

void computeXTilta(std::unique_ptr<GeometryManager>& instance, const Scalar& rate) {
    int numbers = instance->getHostNumVertices();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _computeXTilta<<<blockNum, threadNum>>>(
        instance->getCudaBoundaryType(), instance->getCudaVertVel(),
        instance->getCudaOriginVertPos(), instance->getCudaXTilta(), instance->getHostIPCDt(), rate,
        numbers);
}

void updateVelocities(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostNumVertices();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateVelocities<<<blockNum, threadNum>>>(
        instance->getCudaVertPos(), instance->getCudaOriginVertPos(), instance->getCudaVertVel(),
        instance->getCudaBoundaryType(), instance->getHostIPCDt(), numbers);
}

void changeBoundarytoSIMPoint(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostNumVertices();
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _changeBoundarytoSIMPoint<<<blockNum, threadNum>>>(instance->getCudaBoundaryType(),
                                                       instance->getCudaConstraintsMat(), numbers);
}

int calculateMovingDirection(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr,
                             std::unique_ptr<PCGSolver>& PCG_ptr, int cpNum,
                             int preconditioner_type) {
    if (!preconditioner_type) {
        int cgCount = PCG_ptr->PCG_Process(
            instance, BH_ptr, instance->getCudaMoveDir(), instance->getHostNumVertices(),
            instance->getHostNumTetElements(), instance->getHostIPCDt(),
            instance->getHostMeanVolume(), instance->getHostPCGThreshold());
        return cgCount;
    } else if (preconditioner_type == 1) {
        int cgCount = PCG_ptr->MASPCG_Process(
            instance, BH_ptr, instance->getCudaMoveDir(), instance->getHostNumVertices(),
            instance->getHostNumTetElements(), instance->getHostIPCDt(),
            instance->getHostMeanVolume(), cpNum, instance->getHostPCGThreshold());
        if (cgCount == 3000) {
            printf("MASPCG fail, turn to PCG\n");
            cgCount = PCG_ptr->PCG_Process(
                instance, BH_ptr, instance->getCudaMoveDir(), instance->getHostNumVertices(),
                instance->getHostNumTetElements(), instance->getHostIPCDt(),
                instance->getHostMeanVolume(), instance->getHostPCGThreshold());
            printf("PCG finish:  %d\n", cgCount);
        }
        return cgCount;
    }
}

float computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr) {
    __GPUIPC__::calKineticGradient(instance->getCudaVertPos(), instance->getCudaXTilta(),
                                   instance->getCudaFb(), instance->getCudaVertMass(),
                                   instance->getHostNumVertices());

    CUDA_SAFE_CALL(cudaMemset(instance->getCudaCPNum(), 0, 5 * sizeof(uint32_t)));

    __GPUIPC__::calBarrierGradientAndHessian(instance, BH_ptr, instance->getCudaFb(),
                                             instance->getHostKappa());

    float time00 = 0;

#ifdef USE_FRICTION
    __GPUIPC__::calFrictionGradient(instance, instance->getCudaFb());
    __GPUIPC__::calFrictionHessian(instance, BH_ptr);
#endif

    __FEMENERGY__::calculate_fem_gradient_hessian(
        instance->getCudaTetDmInverses(), instance->getCudaVertPos(), instance->getCudaTetElement(),
        BH_ptr.cudaH12x12, instance->getHostCpNum(4) + instance->getHostCpNumLast(4),
        instance->getCudaTetVolume(), instance->getCudaFb(), instance->getHostNumTetElements(),
        instance->getHostLengthRate(), instance->getHostVolumeRate(), instance->getHostIPCDt());

    CUDAMemcpyDToDSafe(
        BH_ptr.cudaD4Index + instance->getHostCpNum(4) + instance->getHostCpNumLast(4),
        instance->getCudaTetElement(), instance->getHostNumTetElements());

    __FEMENERGY__::calculate_bending_gradient_hessian(
        instance->getCudaVertPos(), instance->getCudaRestVertPos(), instance->getCudaTriBendEdges(),
        instance->getCudaTriBendVerts(), BH_ptr.cudaH12x12, BH_ptr.cudaD4Index,
        instance->getHostCpNum(4) + instance->getHostCpNumLast(4) +
            instance->getHostNumTetElements(),
        instance->getCudaFb(), instance->getHostNumTriBendEdges(), instance->getHostBendStiff(),
        instance->getHostIPCDt());

    __FEMENERGY__::calculate_triangle_fem_gradient_hessian(
        instance->getCudaTriDmInverses(), instance->getCudaVertPos(), instance->getCudaTriElement(),
        BH_ptr.cudaH9x9, instance->getHostCpNum(3) + instance->getHostCpNumLast(3),
        instance->getCudaTriArea(), instance->getCudaFb(), instance->getHostNumTriElements(),
        instance->getHostStretchStiff(), instance->getHostShearStiff(), instance->getHostIPCDt());

    CUDAMemcpyDToDSafe(
        BH_ptr.cudaD3Index + instance->getHostCpNum(3) + instance->getHostCpNumLast(3),
        instance->getCudaTriElement(), instance->getHostNumTriElements());

    __FEMENERGY__::computeGroundGradientAndHessian(instance, BH_ptr, instance->getCudaFb());

    __FEMENERGY__::computeBoundConstraintGradientAndHessian(instance, BH_ptr, instance->getCudaFb());

    __FEMENERGY__::computeSoftConstraintGradientAndHessian(instance, BH_ptr, instance->getCudaFb());

    return time00;
}

Scalar Energy_Add_Reduction_Algorithm(std::unique_ptr<GeometryManager>& instance,
                                      std::unique_ptr<PCGSolver>& PCG_ptr, int type) {
    int numbers = instance->getHostNumTetElements();

    if (type == 0 || type == 3) {
        numbers = instance->getHostNumVertices();
    } else if (type == 2) {
        numbers = instance->getHostCpNum(0);
    } else if (type == 4) {
        numbers = instance->getHostGpNum();
    } else if (type == 5) {
        numbers = instance->getHostCpNumLast(0);
    } else if (type == 6) {
        numbers = instance->getHostGpNumLast();
    } else if (type == 7 || type == 1) {
        numbers = instance->getHostNumTetElements();
    } else if (type == 8) {
        numbers = instance->getHostNumTriElements();
    } else if (type == 9) {
        numbers = instance->getHostNumBoundTargets();
    } else if (type == 10) {
        numbers = instance->getHostNumTriBendEdges();
    } else if (type == 11) {
        numbers = instance->getHostNumSoftTargets();
    }
    if (numbers == 0) return 0;

    Scalar* queue = PCG_ptr->cudaPCGSqueue;
    // CUDA_SAFE_CALL(cudaMalloc((void**)&queue, numbers * sizeof(Scalar)));*/

    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);
    switch (type) {
        case 0:
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                instance->getCudaVertPos(), instance->getCudaXTilta(), queue,
                instance->getCudaVertMass(), numbers);
            break;
        case 1:
            __FEMENERGY__::_getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaVertPos(), instance->getCudaTetElement(),
                instance->getCudaTetDmInverses(), instance->getCudaTetVolume(), numbers,
                instance->getHostLengthRate(), instance->getHostVolumeRate());
            break;
        case 2:
            __GPUIPC__::_getBarrierEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaVertPos(), instance->getCudaRestVertPos(),
                instance->getCudaCollisionPairs(), instance->getHostKappa(),
                instance->getHostDHat(), numbers);
            break;
        case 3:
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaFb(), instance->getCudaMoveDir(), numbers);
            break;
        case 4:
            __GPUIPC__::_computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaVertPos(), instance->getCudaGroundOffset(),
                instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(),
                instance->getHostDHat(), instance->getHostKappa(), numbers);
            break;
        case 5:
            __GPUIPC__::_getFrictionEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaVertPos(), instance->getCudaOriginVertPos(),
                instance->getCudaCollisionPairsLastH(), numbers, instance->getHostIPCDt(),
                instance->getCudaDistCoord(), instance->getCudaTanBasis(),
                instance->getCudaLambdaLastHScalar(),
                instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
                sqrt(instance->getHostFDHat()) * instance->getHostIPCDt());
            break;
        case 6:
            __GPUIPC__::_getFrictionEnergy_gd_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaVertPos(), instance->getCudaOriginVertPos(),
                instance->getCudaGroundNormal(), instance->getCudaCollisionPairsLastHGd(), numbers,
                instance->getHostIPCDt(), instance->getCudaLambdaLastHScalarGd(),
                sqrt(instance->getHostFDHat()) * instance->getHostIPCDt());
            break;
        case 7:
            __FEMENERGY__::
                _getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                    queue, instance->getCudaTetVolume(), numbers, instance->getHostLengthRate(),
                    instance->getHostVolumeRate());
            break;
        case 8:
            __FEMENERGY__::
                _get_triangleFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                    queue, instance->getCudaVertPos(), instance->getCudaTriElement(),
                    instance->getCudaTriDmInverses(), instance->getCudaTriArea(), numbers,
                    instance->getHostStretchStiff(), instance->getHostShearStiff());
            break;
        case 9:
            __FEMENERGY__::
                _computeBoundConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                    queue, instance->getCudaVertPos(), instance->getCudaBoundTargetVertPos(),
                    instance->getCudaBoundTargetIndex(), instance->getHostSoftMotionRate(),
                    instance->getHostAnimationFullRate(), numbers);
            break;
        case 10:
            __FEMENERGY__::_getBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaVertPos(), instance->getCudaRestVertPos(),
                instance->getCudaTriBendEdges(), instance->getCudaTriBendVerts(), numbers,
                instance->getHostBendStiff());
            break;
        case 11:
            __FEMENERGY__::_computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum,
                                                                             sharedMsize>>>(
                queue, instance->getCudaVertPos(), instance->getCudaSoftTargetVertPos(),
                instance->getCudaSoftTargetIndex(), instance->getHostSoftStiffness(),
                instance->getHostAnimationFullRate(), numbers);
            break;
    }

    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::__add_reduction<<<blockNum, threadNum, sharedMsize>>>(queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar result;
    cudaMemcpy(&result, queue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // CUDA_SAFE_CALL(cudaFree(queue));
    return result;
}

Scalar computeEnergy(std::unique_ptr<GeometryManager>& instance,
                     std::unique_ptr<PCGSolver>& PCG_ptr) {
    Scalar Energy = Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 0);

    Energy += instance->getHostIPCDt() * instance->getHostIPCDt() *
              Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 1);

    Energy += instance->getHostIPCDt() * instance->getHostIPCDt() *
              Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 8);

    Energy += instance->getHostIPCDt() * instance->getHostIPCDt() *
              Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 10);

    Energy += Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 9);

    Energy += Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 11);

    Energy += Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 2);

    Energy += instance->getHostKappa() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 4);

#ifdef USE_FRICTION
    Energy +=
        instance->getHostFrictionRate() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 5);

    Energy +=
        instance->getHostFrictionRate() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 6);

#endif

    return Energy;
}

bool lineSearch(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<PCGSolver>& PCG_ptr,
                std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr, Scalar& alpha,
                const Scalar& cfl_alpha) {
    bool stopped = false;
    // buildCP();
    Scalar lastEnergyVal = computeEnergy(instance, PCG_ptr);
    Scalar c1m = 0.0;
    Scalar armijoParam = 0;
    if (armijoParam > 0.0) {
        c1m += armijoParam * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 3);
    }

    CUDA_SAFE_CALL(cudaMemcpy(instance->getCudaTempScalar3Mem(), instance->getCudaVertPos(),
                              instance->getHostNumVertices() * sizeof(Scalar3),
                              cudaMemcpyDeviceToDevice));

    stepForward(instance->getCudaVertPos(), instance->getCudaTempScalar3Mem(),
                instance->getCudaMoveDir(), instance->getCudaBoundaryType(), alpha, false,
                instance->getHostNumVertices());

    bool rehash = true;

    LBVH_CD_ptr->buildBVH(instance);
    // buildCP();
    // if (instance->getHostCpNum(0) > 0) system("pause");
    int numOfIntersect = 0;
    int insectNum = 0;

    bool checkInterset = true;

    while (checkInterset && __GPUIPC__::isIntersected(instance, LBVH_CD_ptr)) {
        // printf("type 0 intersection happened:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        numOfIntersect++;
        alpha = __MATHUTILS__::__m_min(cfl_alpha, alpha);
        stepForward(instance->getCudaVertPos(), instance->getCudaTempScalar3Mem(),
                    instance->getCudaMoveDir(), instance->getCudaBoundaryType(), alpha, false,
                    instance->getHostNumVertices());
        LBVH_CD_ptr->buildBVH(instance);
    }

    LBVH_CD_ptr->buildCP(instance);
    __GPUIPC__::buildGP(instance);
    // if (instance->getHostCpNum(0) > 0) system("pause");
    // rehash = false;

    // buildCollisionSets(mesh, sh, gd, true);
    Scalar testingE = computeEnergy(instance, PCG_ptr);

    int numOfLineSearch = 0;
    Scalar LFStepSize = alpha;
    // Scalar temp_c1m = c1m;
    std::cout.precision(18);
    // std::cout << "testE:    " << testingE << "      lastEnergyVal:        "
    // << abs(lastEnergyVal- RestNHEnergy) << std::endl;
    while ((testingE > lastEnergyVal + c1m * alpha) && alpha > 1e-3 * LFStepSize) {
        // printf("testE:    %f      lastEnergyVal:        %f         clm*alpha:
        // %f\n", testingE, lastEnergyVal, c1m * alpha); std::cout
        // <<numOfLineSearch<<  "   testE:    " << testingE << " lastEnergyVal:
        // " << lastEnergyVal << std::endl;
        alpha /= 2.0;
        ++numOfLineSearch;

        stepForward(instance->getCudaVertPos(), instance->getCudaTempScalar3Mem(),
                    instance->getCudaMoveDir(), instance->getCudaBoundaryType(), alpha, false,
                    instance->getHostNumVertices());

        LBVH_CD_ptr->buildBVH(instance);
        LBVH_CD_ptr->buildCP(instance);
        __GPUIPC__::buildGP(instance);
        testingE = computeEnergy(instance, PCG_ptr);
    }
    if (numOfLineSearch > 8)
        printf(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            "!!!!!!!!!!!!!\n");
    if (alpha < LFStepSize) {
        bool needRecomputeCS = false;
        while (checkInterset && __GPUIPC__::isIntersected(instance, LBVH_CD_ptr)) {
            printf("type 1 intersection happened:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            numOfIntersect++;
            alpha = __MATHUTILS__::__m_min(cfl_alpha, alpha);
            stepForward(instance->getCudaVertPos(), instance->getCudaTempScalar3Mem(),
                        instance->getCudaMoveDir(), instance->getCudaBoundaryType(), alpha, false,
                        instance->getHostNumVertices());
            LBVH_CD_ptr->buildBVH(instance);
            needRecomputeCS = true;
        }
        if (needRecomputeCS) {
            LBVH_CD_ptr->buildCP(instance);
            __GPUIPC__::buildGP(instance);
        }
    }

    return stopped;
}

void postLineSearch(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr,
                    std::unique_ptr<PCGSolver>& PCG_ptr, Scalar alpha) {
    if (instance->getHostKappa() == 0.0) {
        __GPUIPC__::initKappa(instance, BH_ptr, PCG_ptr);
    } else {
        bool updateKappa = __GPUIPC__::checkCloseGroundVal(instance);
        if (!updateKappa) {
            updateKappa = __GPUIPC__::checkSelfCloseVal(instance);
        }
        if (updateKappa) {
            instance->getHostKappa() *= 2.0;
            __GPUIPC__::upperBoundKappa(instance, instance->getHostKappa());
        }
        tempFree_closeConstraint(instance);
        tempMalloc_closeConstraint(instance);
        CUDA_SAFE_CALL(cudaMemset(instance->getCudaCloseCPNum(), 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(instance->getCudaCloseGPNum(), 0, sizeof(uint32_t)));

        __GPUIPC__::computeCloseGroundVal(instance);
        __GPUIPC__::computeSelfCloseVal(instance);
    }
    // printf("------------------------------------------Kappa: %f\n",
    // instance->getHostKappa());
}

Scalar maxCOllisionPairNum = 0;
Scalar totalCollisionPairs = 0;
Scalar total_Cg_count = 0;
Scalar timemakePd = 0;
#include <fstream>
#include <vector>
std::vector<int> iterV;

int solve_subIP(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr,
                std::unique_ptr<PCGSolver>& PCG_ptr,
                std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr, Scalar& time0, Scalar& time1,
                Scalar& time2, Scalar& time3, Scalar& time4) {
    int iterCap = 10000, k = 0;

    CUDA_SAFE_CALL(cudaMemset(instance->getCudaMoveDir(), 0,
                              instance->getHostNumVertices() * sizeof(Scalar3)));

    Scalar totalTimeStep = 0;
    for (; k < iterCap; ++k) {
        totalCollisionPairs += instance->getHostCpNum(0);
        maxCOllisionPairNum = (maxCOllisionPairNum > instance->getHostCpNum(0))
                                  ? maxCOllisionPairNum
                                  : instance->getHostCpNum(0);
        cudaEvent_t start, end0, end1, end2, end3, end4;
        cudaEventCreate(&start);
        cudaEventCreate(&end0);
        cudaEventCreate(&end1);
        cudaEventCreate(&end2);
        cudaEventCreate(&end3);
        cudaEventCreate(&end4);

        BH_ptr.updateDNum(instance->getHostNumTriElements(),  // tri_Num
                          instance->getHostNumTriBendEdges(),
                          instance->getHostNumTetElements(),  // tet_number
                          instance->getHostCpNum(2),          // cpNum1
                          instance->getHostCpNum(3),          // cpNum2
                          instance->getHostCpNum(4),          // cpNum3
                          instance->getHostCpNumLast(2),      // last_cpNum1
                          instance->getHostCpNumLast(3),      // last_cpNum2
                          instance->getHostCpNumLast(4)       // last_cpNum3
        );

        // printf("collision num  %d\n", instance->getHostCpNum(0));

        cudaEventRecord(start);

        timemakePd += computeGradientAndHessian(instance, BH_ptr);

        Scalar distToOpt_PN = calcMinMovement(instance->getCudaMoveDir(), PCG_ptr->cudaPCGSqueue,
                                              instance->getHostNumVertices());

        bool gradVanish =
            (distToOpt_PN <
             sqrt(instance->getHostNewtonSolverThreshold() *
                  instance->getHostNewtonSolverThreshold() * instance->getHostBboxDiagSize2() *
                  instance->getHostIPCDt() * instance->getHostIPCDt()));
        if (k && gradVanish) {
            break;
        }
        cudaEventRecord(end0);
        total_Cg_count += calculateMovingDirection(instance, BH_ptr, PCG_ptr,
                                                   instance->getHostCpNum(0), PCG_ptr->PrecondType);
        cudaEventRecord(end1);
        Scalar alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;

        alpha = __MATHUTILS__::__m_min(alpha, __GPUIPC__::ground_largestFeasibleStepSize(
                                                  instance, slackness_a, PCG_ptr->cudaPCGSqueue));
        alpha = __MATHUTILS__::__m_min(
            alpha, __GPUIPC__::self_largestFeasibleStepSize(
                       instance, slackness_m, PCG_ptr->cudaPCGSqueue, instance->getHostCpNum(0)));

        Scalar temp_alpha = alpha;
        Scalar alpha_CFL = alpha;

        Scalar ccd_size = 1.0;
#ifdef USE_FRICTION
        ccd_size = 0.6;
#endif

        std::cout << "alpha before linesearch: " << alpha << std::endl;

        LBVH_CD_ptr->buildBVH_FULLCCD(instance, temp_alpha);
        LBVH_CD_ptr->buildFullCP(instance, temp_alpha);
        if (instance->getHostCcdCpNum() > 0) {
            Scalar maxSpeed = __GPUIPC__::cfl_largestSpeed(instance, PCG_ptr->cudaPCGSqueue);
            alpha_CFL = sqrt(instance->getHostDHat()) / maxSpeed * 0.5;
            alpha = __MATHUTILS__::__m_min(alpha, alpha_CFL);
            if (temp_alpha > 2 * alpha_CFL) {
                /*buildBVH_FULLCCD(temp_alpha);
                buildFullCP(temp_alpha);*/
                alpha = __MATHUTILS__::__m_min(temp_alpha,
                                               __GPUIPC__::self_largestFeasibleStepSize(
                                                   instance, slackness_m, PCG_ptr->cudaPCGSqueue,
                                                   instance->getHostCcdCpNum()) *
                                                   ccd_size);
                alpha = __MATHUTILS__::__m_max(alpha, alpha_CFL);
            }
        }

        cudaEventRecord(end2);
        // printf("alpha:  %f\n", alpha);

        bool isStop = lineSearch(instance, PCG_ptr, LBVH_CD_ptr, alpha, alpha_CFL);
        cudaEventRecord(end3);
        std::cout << "alpha after linesearch: " << alpha << std::endl;

        postLineSearch(instance, BH_ptr, PCG_ptr, alpha);
        std::cout << "alpha after postlinesearch: " << alpha << std::endl;

        cudaEventRecord(end4);

        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        float time00, time11, time22, time33, time44;
        cudaEventElapsedTime(&time00, start, end0);
        cudaEventElapsedTime(&time11, end0, end1);
        // total_Cg_time += time1;
        cudaEventElapsedTime(&time22, end1, end2);
        cudaEventElapsedTime(&time33, end2, end3);
        cudaEventElapsedTime(&time44, end3, end4);
        time0 += time00;
        time1 += time11;
        time2 += time22;
        time3 += time33;
        time4 += time44;

        (cudaEventDestroy(start));
        (cudaEventDestroy(end0));
        (cudaEventDestroy(end1));
        (cudaEventDestroy(end2));
        (cudaEventDestroy(end3));
        (cudaEventDestroy(end4));
        totalTimeStep += alpha;
    }

    printf(
        "\n\n      Kappa: %f                               iteration k:  "
        "%d\n\n\n",
        instance->getHostKappa(), k);
    return k;
}

Scalar2 minMaxSelfDist(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostCpNum(0);
    if (numbers < 1) return make_Scalar2(1e32, 0);
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar2) * (threadNum >> 5);

    Scalar2* _queue;
    CUDA_SAFE_CALL(cudaMalloc((void**)&_queue, numbers * sizeof(Scalar2)));
    // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number *
    // sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist<<<blockNum, threadNum, sharedMsize>>>(
        instance->getCudaVertPos(), instance->getCudaCollisionPairs(), _queue, numbers);
    //_reduct_min_Scalar3_to_Scalar << <blockNum, threadNum, sharedMsize >> >
    //(_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> >
        //(_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB),
    // cudaMemcpyDeviceToDevice);
    Scalar2 minValue;
    cudaMemcpy(&minValue, _queue, sizeof(Scalar2), cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaFree(_queue));
    minValue.x = 1.0 / minValue.x;
    return minValue;
}

int totalNT = 0;
int total_Frames = 0;
Scalar totalTime = 0;
Scalar ttime0 = 0;
Scalar ttime1 = 0;
Scalar ttime2 = 0;
Scalar ttime3 = 0;
Scalar ttime4 = 0;

void IPC_Solver(std::unique_ptr<GeometryManager>& instance, BlockHessian& BH_ptr,
                std::unique_ptr<PCGSolver>& PCG_ptr,
                std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr) {
    // Scalar instance->getHostAnimationFullRate() = 0;
    cudaEvent_t start, end0;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    Scalar alpha = 1;
    cudaEventRecord(start);

    __GPUIPC__::upperBoundKappa(instance, instance->getHostKappa());
    if (instance->getHostKappa() < 1e-16) {
        __GPUIPC__::suggestKappa(instance, instance->getHostKappa());
    }
    __GPUIPC__::initKappa(instance, BH_ptr, PCG_ptr);

#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaLambdaLastHScalar(),
                              instance->getHostCpNum(0) * sizeof(Scalar)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaDistCoord(),
                              instance->getHostCpNum(0) * sizeof(Scalar2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaTanBasis(),
                              instance->getHostCpNum(0) * sizeof(__MATHUTILS__::Matrix3x2S)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaCollisionPairsLastH(),
                              instance->getHostCpNum(0) * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaMatIndexLast(),
                              instance->getHostCpNum(0) * sizeof(int)));

    CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaLambdaLastHScalarGd(),
                              instance->getHostGpNum() * sizeof(Scalar)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaCollisionPairsLastHGd(),
                              instance->getHostGpNum() * sizeof(uint32_t)));
    __GPUIPC__::buildFrictionSets(instance);
#endif

    instance->getHostAnimationFullRate() = instance->getHostAnimationSubRate();
    int k = 0;
    Scalar time0 = 0;
    Scalar time1 = 0;
    Scalar time2 = 0;
    Scalar time3 = 0;
    Scalar time4 = 0;

    while (true) {
        // if (instance->getHostCpNum(0) > 0) return;
        tempMalloc_closeConstraint(instance);
        CUDA_SAFE_CALL(cudaMemset(instance->getCudaCloseCPNum(), 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(instance->getCudaCloseGPNum(), 0, sizeof(uint32_t)));

        totalNT +=
            solve_subIP(instance, BH_ptr, PCG_ptr, LBVH_CD_ptr, time0, time1, time2, time3, time4);

        Scalar2 minMaxDist1 = minMaxGroundDist(instance);
        Scalar2 minMaxDist2 = minMaxSelfDist(instance);

        Scalar minDist = __MATHUTILS__::__m_min(minMaxDist1.x, minMaxDist2.x);
        Scalar maxDist = __MATHUTILS__::__m_max(minMaxDist1.y, minMaxDist2.y);

        bool finishMotion = instance->getHostAnimationFullRate() > 0.99 ? true : false;
        // std::cout << "minDist:  " << minDist << "       maxDist:  " <<
        // maxDist << std::endl; std::cout << "dTol:  " << dTol << "       1e-6
        // * bboxDiagSize2:  " << 1e-6 * bboxDiagSize2 << std::endl;
        if (finishMotion) {
            if ((instance->getHostCpNum(0) + instance->getHostGpNum()) > 0) {
                if (minDist < instance->getHostDTol()) {
                    tempFree_closeConstraint(instance);
                    break;
                } else if (maxDist < instance->getHostDHat()) {
                    tempFree_closeConstraint(instance);
                    break;
                } else {
                    tempFree_closeConstraint(instance);
                }
            } else {
                tempFree_closeConstraint(instance);
                break;
            }
        } else {
            tempFree_closeConstraint(instance);
        }

        instance->getHostAnimationFullRate() += instance->getHostAnimationSubRate();

#ifdef USE_FRICTION
        CUDA_SAFE_CALL(cudaFree(instance->getCudaLambdaLastHScalar()));
        CUDA_SAFE_CALL(cudaFree(instance->getCudaDistCoord()));
        CUDA_SAFE_CALL(cudaFree(instance->getCudaTanBasis()));
        CUDA_SAFE_CALL(cudaFree(instance->getCudaCollisionPairsLastH()));
        CUDA_SAFE_CALL(cudaFree(instance->getCudaMatIndexLast()));

        CUDA_SAFE_CALL(cudaFree(instance->getCudaLambdaLastHScalarGd()));
        CUDA_SAFE_CALL(cudaFree(instance->getCudaCollisionPairsLastHGd()));

        CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaLambdaLastHScalar(),
                                  instance->getHostCpNum(0) * sizeof(Scalar)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaDistCoord(),
                                  instance->getHostCpNum(0) * sizeof(Scalar2)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaTanBasis(),
                                  instance->getHostCpNum(0) * sizeof(__MATHUTILS__::Matrix3x2S)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaCollisionPairsLastH(),
                                  instance->getHostCpNum(0) * sizeof(int4)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaMatIndexLast(),
                                  instance->getHostCpNum(0) * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaLambdaLastHScalarGd(),
                                  instance->getHostGpNum() * sizeof(Scalar)));
        CUDA_SAFE_CALL(cudaMalloc((void**)&instance->getCudaCollisionPairsLastHGd(),
                                  instance->getHostGpNum() * sizeof(uint32_t)));
        __GPUIPC__::buildFrictionSets(instance);
#endif
    }

#ifdef USE_FRICTION
    CUDA_SAFE_CALL(cudaFree(instance->getCudaLambdaLastHScalar()));
    CUDA_SAFE_CALL(cudaFree(instance->getCudaDistCoord()));
    CUDA_SAFE_CALL(cudaFree(instance->getCudaTanBasis()));
    CUDA_SAFE_CALL(cudaFree(instance->getCudaCollisionPairsLastH()));
    CUDA_SAFE_CALL(cudaFree(instance->getCudaMatIndexLast()));

    CUDA_SAFE_CALL(cudaFree(instance->getCudaLambdaLastHScalarGd()));
    CUDA_SAFE_CALL(cudaFree(instance->getCudaCollisionPairsLastHGd()));
#endif

    updateVelocities(instance);

    computeXTilta(instance, 1);
    cudaEventRecord(end0);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    float tttime;
    cudaEventElapsedTime(&tttime, start, end0);
    totalTime += tttime;
    total_Frames++;
    printf("average time cost:     %f,    frame id:   %d\n", totalTime / totalNT, total_Frames);
    printf("boundary alpha: %f\n  finished a step\n", alpha);

    ttime0 += time0;
    ttime1 += time1;
    ttime2 += time2;
    ttime3 += time3;
    ttime4 += time4;

    std::ofstream outTime("timeCost.txt");

    outTime << "time0: " << ttime0 / 1000.0 << std::endl;
    outTime << "time1: " << ttime1 / 1000.0 << std::endl;
    outTime << "time2: " << ttime2 / 1000.0 << std::endl;
    outTime << "time3: " << ttime3 / 1000.0 << std::endl;
    outTime << "time4: " << ttime4 / 1000.0 << std::endl;
    outTime << "time_makePD: " << timemakePd / 1000.0 << std::endl;

    outTime << "totalTime: " << totalTime / 1000.0 << std::endl;
    outTime << "total iter: " << totalNT << std::endl;
    outTime << "frames: " << total_Frames << std::endl;
    outTime << "totalCollisionNum: " << totalCollisionPairs << std::endl;
    outTime << "averageCollision: " << totalCollisionPairs / totalNT << std::endl;
    outTime << "maxCOllisionPairNum: " << maxCOllisionPairNum << std::endl;
    outTime << "totalCgTime: " << total_Cg_count << std::endl;
    outTime.close();

    std::vector<Scalar3> temppos(instance->getHostNumVertices());
    std::vector<Scalar3> tempvel(instance->getHostNumVertices());
    CUDA_SAFE_CALL(cudaMemcpy(temppos.data(), instance->getCudaVertPos(),
                              instance->getHostNumVertices() * sizeof(Scalar3),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(tempvel.data(), instance->getCudaVertVel(),
                              instance->getHostNumVertices() * sizeof(Scalar3),
                              cudaMemcpyDeviceToHost));
    Scalar sumpos = 0, sumvel = 0;
    for (int i = 0; i < instance->getHostNumVertices(); i++) {
        sumpos += temppos[i].x + temppos[i].y + temppos[i].z;
        sumvel += tempvel[i].x + tempvel[i].y + tempvel[i].z;
    }
    std::cout << "sumpos, sumvel: " << sumpos << " " << sumvel << std::endl;
}

};  // namespace __INTEGRATOR__
