

#include "MASPreconditioner.cuh"
#include "device_launch_parameters.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <vector>
#include <bitset>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "cooperative_groups.h"
using namespace cooperative_groups;

template <class F>
__device__ __host__ inline F __mm_min(F a, F b)
{
    return a > b ? b : a;
}


template <class F>
__device__ __host__ inline F __mm_max(F a, F b)
{
    return a > b ? a : b;
}

#define BANKSIZE 32
#define DEFAULT_BLOCKSIZE 256
#define DEFAULT_WARPNUM 8
__global__ void _buildCML0(const unsigned int* _neighborStart,
                           unsigned int*       _neighborNum,
                           unsigned int*       _neighborList,
                           unsigned int*       _fineConnectedMsk,
                           int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int          warpId      = idx / BANKSIZE;
    int          laneId      = idx % BANKSIZE;
    int          numNeighbor = _neighborNum[idx];
    unsigned int connectMsk  = (1U << laneId);
    int          nk          = 0;
    int          startId     = _neighborStart[idx];
    for(int i = 0; i < numNeighbor; i++)
    {
        int vIdConnected     = _neighborList[startId + i];
        int warpIdxConnected = vIdConnected / BANKSIZE;
        if(warpId == warpIdxConnected)
        {
            unsigned int laneIdxConnected = vIdConnected % BANKSIZE;
            connectMsk |= (1U << laneIdxConnected);
        }
        else
        {
            _neighborList[startId + nk] = vIdConnected;
            nk++;
        }
    }
    _neighborNum[idx]      = nk;
    _fineConnectedMsk[idx] = connectMsk;
}

__device__ unsigned int _LanemaskLt(int laneIdx)
{
    return (1U << laneIdx) - 1;
}

__global__ void _preparePrefixSumL0(int* _prefixOriginal, unsigned int* _fineConnectedMsk, int vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int          warpId      = idx / BANKSIZE;
    int          localWarpId = threadIdx.x / BANKSIZE;
    int          laneId      = idx % BANKSIZE;
    unsigned int connectMsk  = _fineConnectedMsk[idx];
    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    cacheMask[threadIdx.x] = connectMsk;
    unsigned int visited   = (1U << laneId);
    while(connectMsk != -1)
    {
        unsigned int todo = visited ^ connectMsk;

        if(!todo)
            break;

        unsigned int nextVist = __ffs(todo) - 1;
        visited |= (1U << nextVist);
        connectMsk |= cacheMask[nextVist + localWarpId * BANKSIZE];  //__shfl_sync(0xFFFFFFFF, cacheMask, nextVist);//?????!!!!!
    }

    _fineConnectedMsk[idx] = connectMsk;

    unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        //prefixSum[warpId]++;
        atomicAdd(prefixSum + localWarpId, 1);
    }

    if(laneId == 0)
    {
        _prefixOriginal[warpId] = prefixSum[localWarpId];
    }
}

__global__ void _buildLevel1(int2*               _levelSize,
                             int*                _coarseSpaceTable,
                             int*                _goingNext,
                             const unsigned int* _fineConnectedMsk,
                             const int*          _prefixSumOriginal,
                             const int*          _prefixOriginal,
                             int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int warpId      = idx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = idx % BANKSIZE;

    __shared__ unsigned int electedMask[BANKSIZE];
    __shared__ unsigned int lanePrefix[BANKSIZE * BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    if(idx == vertNum - 1)
    {
        _levelSize[1].x = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
        _levelSize[1].y = (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }

    unsigned int connMsk = _fineConnectedMsk[idx];

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }

    //unsigned int lanePrefix2 = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    //lanePrefix2 += _prefixSumOriginal[warpId];

    //unsigned int elected_lane = __ffs(connMsk) - 1;
    //unsigned int theLanePrefix = __shfl_sync(0xFFFFFFFF, lanePrefix2, elected_lane);

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _prefixSumOriginal[warpId];

    unsigned int elected_lane = __ffs(connMsk) - 1;
    unsigned int theLanePrefix = lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xFFFFFFFF, lanePrefix, elected_lane);


    _coarseSpaceTable[idx + 0 * vertNum] = theLanePrefix;
    _goingNext[idx] = theLanePrefix + (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
}

__global__ void _buildConnectMaskLx(const unsigned int* _neighborStart,
                                    unsigned int*       _neighborNum,
                                    unsigned int*       _neighborList,
                                    int*                _coarseSpaceTable,
                                    unsigned int*       _nextConnectedMsk,
                                    const unsigned int* _fineConnectedMsk,
                                    int                 level,
                                    int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int warpId      = idx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = idx % BANKSIZE;

    unsigned int prefixMsk = _fineConnectedMsk[idx];
    unsigned int connMsk   = 0;
    unsigned int coarseIdx = _coarseSpaceTable[(level - 1) * vertNum + idx];
    int          kn        = _neighborNum[idx];
    int          nk        = 0;
    int          startId   = _neighborStart[idx];
    for(int i = 0; i < kn; i++)
    {
        unsigned int connect = _neighborList[startId + i];
        unsigned int coarseConnect = _coarseSpaceTable[(level - 1) * vertNum + connect];

        if(coarseIdx / BANKSIZE == coarseConnect / BANKSIZE)
        {
            unsigned int off = coarseConnect % BANKSIZE;
            connMsk |= (1U << off);
        }
        else
        {
            _neighborList[startId + nk] = connect;
            nk++;
        }
    }

    _neighborNum[idx] = nk;

    __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
    cacheMsk[threadIdx.x] = 0;

    if(__popc(prefixMsk) == BANKSIZE)
    {
        atomicOr(cacheMsk + localWarpId * BANKSIZE, connMsk);
        connMsk = cacheMsk[localWarpId * BANKSIZE];
        //if (laneId == 0) {
        //  cacheMsk[localWarpId] = 0;
        //}
    }
    else
    {
        unsigned int electedLane = __ffs(prefixMsk) - 1;
        if(connMsk)
        {
            atomicOr(cacheMsk + localWarpId * BANKSIZE + electedLane, connMsk);
        }
        connMsk = cacheMsk[localWarpId * BANKSIZE + electedLane];
    }

    unsigned int electedPrefix = __popc(prefixMsk & _LanemaskLt(laneId));

    if(connMsk && electedPrefix == 0)
    {
        atomicOr(_nextConnectedMsk + coarseIdx, connMsk);
    }
}

__global__ void _nextLevelCluster(unsigned int* _nextConnectedMsk, unsigned int* _nextPrefix, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int            warpId      = idx / BANKSIZE;
    int            localWarpId = threadIdx.x / BANKSIZE;
    int            laneId      = idx % BANKSIZE;
    __shared__ int prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    unsigned int connMsk = (1U << laneId);

    connMsk |= _nextConnectedMsk[idx];

    //unsigned int cachedMsk = connMsk;

    __shared__ unsigned int cachedMsk[DEFAULT_BLOCKSIZE];
    cachedMsk[threadIdx.x] = connMsk;
    unsigned int visited   = (1U << laneId);

    while(true)
    {
        unsigned int todo = visited ^ connMsk;

        if(!todo)
            break;

        unsigned int nextVisit = __ffs(todo) - 1;

        visited |= (1U << nextVisit);

        connMsk |= cachedMsk[nextVisit + localWarpId * BANKSIZE];  //__shfl_sync(0xFFFFFFFF, cachedMsk, nextVisit);
    }

    _nextConnectedMsk[idx] = connMsk;

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicAdd(prefixSum + localWarpId, 1);
    }

    if(laneId == 0)
        _nextPrefix[warpId] = prefixSum[localWarpId];
}

__global__ void _prefixSumLx(int2*         _levelSize,
                             unsigned int* _nextPrefix,
                             unsigned int* _nextPrefixSum,
                             unsigned int* _nextConnectMsk,
                             int*          _goingNext,
                             int           level,
                             int           levelBegin,
                             int           number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int warpId      = idx / BANKSIZE;
    int localWarpId = threadIdx.x / BANKSIZE;
    int laneId      = idx % BANKSIZE;

    __shared__ unsigned int electedMask[BANKSIZE];
    __shared__ unsigned int lanePrefix[BANKSIZE * BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }

    if(idx == number - 1)
    {
        _levelSize[level + 1].x = _nextPrefixSum[warpId] + _nextPrefix[warpId];
        _levelSize[level + 1].y = levelBegin + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    }

    unsigned int connMsk = _nextConnectMsk[idx];

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _nextPrefixSum[warpId];

    unsigned int elected_lane = __ffs(connMsk) - 1;
    unsigned int theLanePrefix = lanePrefix[elected_lane + BANKSIZE * localWarpId];  //__shfl_sync(0xFFFFFFFF, lanePrefix, elected_lane);

    _nextConnectMsk[idx] = theLanePrefix;
    _goingNext[idx + levelBegin] =
        theLanePrefix + levelBegin + (number + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
}

__global__ void _computeNextLevel(int*          _coarseSpaceTable,
                                  unsigned int* _nextConnectMsk,
                                  int           level,
                                  int           number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int next = _coarseSpaceTable[(level - 1) * number + idx];
    _coarseSpaceTable[(level)*number + idx] = _nextConnectMsk[next];
}

__global__ void _aggregationKernel(
    int* _denseLevel, int4* _coarseTable, int* _goingNext, int levelNum, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int currentId = idx;
    int aggLevel  = levelNum - 1;
    //__shared__ int4 ctable[DEFAULT_BLOCKSIZE];
    int4 ctable;
    for(int l = 0; l < levelNum - 1; l++)
    {
        int next = _goingNext[currentId];

        //int next0 = __shfl_sync(0xFFFFFFFF, next, 0);
        ////printf("%d   %d   %d    %d\n", next, next0, l,  idx);
        //if (next == next0) {
        //  aggLevel = __mm_min(l, aggLevel);
        //}

        currentId          = next;
        *(&(ctable.x) + l) = next;
    }

    _denseLevel[idx] = aggLevel;

    //printf("%d   %d\n", aggLevel, idx);

    _coarseTable[idx] = ctable;
}


__global__ void _prepareHessian(const MATHUTILS::Matrix12x12d* Hessians12,
                                const MATHUTILS::Matrix9x9d*   Hessians9,
                                const MATHUTILS::Matrix6x6d*   Hessians6,
                                const MATHUTILS::Matrix3x3d*   Hessians3,
                                const uint4*                    D4Index,
                                const uint3*                    D3Index,
                                const uint2*                    D2Index,
                                const uint32_t*                 D1Index,
                                MATHUTILS::Matrix96x96T*       P96,
                                int                             numbers4,
                                int                             numbers3,
                                int                             numbers2,
                                int                             numbers1,
                                int*                            _goingNext,
                                int                             levelNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers4 + numbers3 + numbers2 + numbers1)
        return;

    if(idx < numbers4)
    {
        int Hid  = idx / 144;
        int qid  = idx % 144;
        int qrid = qid / 12;
        int qcid = qid % 12;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D4Index[Hid].x);
        int   vertCid  = *(nodeInex + vcid);
        int   vertRid  = *(nodeInex + vrid);

        //int cha = vertCid - vertRid;

        int         roffset = qrid % 3;
        int         coffset = qcid % 3;
        Precision_T Hval    = Hessians12[Hid].m[qrid][qcid];

        int cPid  = vertCid / 32;
        int level = 0;
        while(vertCid / 32 != vertRid / 32 && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / 32;
        }
        if(level >= levelNum)
        {
            return;
        }
        //int cPid = vertCid / 32;

        atomicAdd(&(P96[cPid].m[(vertRid % 32) * 3 + roffset][(vertCid % 32) * 3 + coffset]),
                  Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / 32;
            if(vertCid / 32 == vertRid / 32)
            {

                atomicAdd(&(P96[cPid].m[(vertRid % 32) * 3 + roffset][(vertCid % 32) * 3 + coffset]),
                          Hval);
            }
        }
    }
    else if(numbers4 <= idx && idx < numbers3 + numbers4)
    {
        idx -= numbers4;
        int Hid = idx / 81;
        int qid = idx % 81;

        int qrid = qid / 9;
        int qcid = qid % 9;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D3Index[Hid].x);
        int   vertCid  = *(nodeInex + vcid);
        int   vertRid  = *(nodeInex + vrid);
        //int Pid = vertCid / 12;
        //int cha = vertCid - vertRid;

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians9[Hid].m[qrid][qcid];

        int cPid  = vertCid / 32;
        int level = 0;
        while(vertCid / 32 != vertRid / 32 && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / 32;
        }
        if(level >= levelNum)
        {
            return;
        }
        atomicAdd(&(P96[cPid].m[(vertRid % 32) * 3 + roffset][(vertCid % 32) * 3 + coffset]),
                  Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / 32;
            if(vertCid / 32 == vertRid / 32)
            {

                atomicAdd(&(P96[cPid].m[(vertRid % 32) * 3 + roffset][(vertCid % 32) * 3 + coffset]),
                          Hval);
            }
        }
    }
    else if(numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2)
    {
        idx -= numbers3 + numbers4;
        int Hid = idx / 36;
        int qid = idx % 36;

        int qrid = qid / 6;
        int qcid = qid % 6;

        int vcid = qcid / 3;
        int vrid = qrid / 3;

        auto* nodeInex = &(D2Index[Hid].x);

        int vertCid = *(nodeInex + vcid);
        int vertRid = *(nodeInex + vrid);
        //int Pid = vertCid / 12;
        int cha = vertCid - vertRid;

        int roffset = qrid % 3;
        int coffset = qcid % 3;

        Precision_T Hval = Hessians6[Hid].m[qrid][qcid];

        int cPid  = vertCid / 32;
        int level = 0;
        while(vertCid / 32 != vertRid / 32 && level < levelNum)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / 32;
        }
        if(level >= levelNum)
        {
            return;
        }
        atomicAdd(&(P96[cPid].m[(vertRid % 32) * 3 + roffset][(vertCid % 32) * 3 + coffset]),
                  Hval);

        while(level < levelNum - 1)
        {
            level++;
            vertCid = _goingNext[vertCid];
            vertRid = _goingNext[vertRid];
            cPid    = vertCid / 32;
            if(vertCid / 32 == vertRid / 32)
            {

                atomicAdd(&(P96[cPid].m[(vertRid % 32) * 3 + roffset][(vertCid % 32) * 3 + coffset]),
                          Hval);
            }
        }
    }
    else
    {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;

        int qrid = qid / 3;
        int qcid = qid % 3;

        int nodeIndex = D1Index[Hid];

        Precision_T Hval = Hessians3[Hid].m[qrid][qcid];

        int cPid  = nodeIndex / 32;
        int Pod   = nodeIndex % 32;
        int level = 0;


        atomicAdd(&(P96[cPid].m[Pod * 3 + qrid][Pod * 3 + qcid]), Hval);

        while(level < levelNum - 1)
        {
            level++;
            nodeIndex = _goingNext[nodeIndex];
            Pod       = nodeIndex % 32;
            cPid      = nodeIndex / 32;
            atomicAdd(&(P96[cPid].m[Pod * 3 + qrid][Pod * 3 + qcid]), Hval);
        }
    }
}

__global__ void __setMassMat_P96(const double*             _masses,
                                 const int*                _goingNext,
                                 MATHUTILS::Matrix96x96T* _Mat96,
                                 int                       levelNum,
                                 int                       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int warpId = idx / 32;
    int laneId = idx % 32;

    Precision_T mass = _masses[idx];

    int Pid = idx / 32;
    int Pod = idx % 32;

    _Mat96[Pid].m[Pod * 3][Pod * 3]         = mass;
    _Mat96[Pid].m[Pod * 3 + 1][Pod * 3 + 1] = mass;
    _Mat96[Pid].m[Pod * 3 + 2][Pod * 3 + 2] = mass;

    int level = 0;

    while(level < levelNum - 1)
    {
        level++;
        idx = _goingNext[idx];
        Pid = idx / 32;
        Pod = idx % 32;
        atomicAdd(&(_Mat96[Pid].m[Pod * 3][Pod * 3]), mass);
        atomicAdd(&(_Mat96[Pid].m[Pod * 3 + 1][Pod * 3 + 1]), mass);
        atomicAdd(&(_Mat96[Pid].m[Pod * 3 + 2][Pod * 3 + 2]), mass);
    }
}


__global__ void __inverse2_P96x96(MATHUTILS::Matrix96x96T*  PMas,
                                  MATHUTILS::MasMatrixSymf* invP96,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId = idx / (BANKSIZE * 3);
    int i     = idx % (BANKSIZE * 3);
    //int localMatId = threadIdx.x / 96;
    int                    block_matId = threadIdx.x / (BANKSIZE * 3);
    __shared__ Precision_T colm[32 / BANKSIZE][BANKSIZE * 3];
    //invPMas[matId].m[j][i] = 1;
    if(PMas[matId].m[i][i] == 0)
    {
        PMas[matId].m[i][i] = 1;
    }

    __syncthreads();
    __threadfence();

    int         j = 0;
    Precision_T rt;

    while(j < (BANKSIZE * 3))
    {
        __syncthreads();
        __threadfence();

        rt = PMas[matId].m[j][j];

        colm[block_matId][i] = PMas[matId].m[i][j];

        __syncthreads();
        __threadfence();
        if(i == j)
        {

            PMas[matId].m[i][j] = 1;
        }
        else
        {
            PMas[matId].m[i][j] = 0;
        }
        __syncthreads();
        __threadfence();

        PMas[matId].m[j][i] /= rt;

        __syncthreads();
        __threadfence();
        for(int k = 0; k < (BANKSIZE * 3); k++)
        {
            if(k != j)
            {
                Precision_T rate = -colm[block_matId][k];
                __syncthreads();
                __threadfence();

                PMas[matId].m[k][i] += rate * PMas[matId].m[j][i];
            }
        }

        j++;
    }
    __syncthreads();
    __threadfence();
    if(i % 3 < 2)
        PMas[matId].m[i + 1][i] = PMas[matId].m[i][i + 1];
    else
        PMas[matId].m[i][i - 2] = PMas[matId].m[i - 2][i];
    __syncthreads();
    __threadfence();


    for(int j = 0; j < (BANKSIZE * 3); j++)
    {
        //PMas[matId].m[j][i] = sPMas[block_matId][j][i];
        int rowId = j / 3;
        int colId = i / 3;
        int index = 0;
        if(colId >= rowId)
        {
            index = BANKSIZE * rowId - rowId * (rowId + 1) / 2 + colId;
            invP96[matId].M[index].m[j % 3][i % 3] = PMas[matId].m[j][i];
        }
    }
}

__global__ void __inverse3_P96x96(MATHUTILS::Matrix96x96T*  P96,
                                  MATHUTILS::Matrix96x96MT* invP96,
                                  int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId = idx / 96;
    int i     = idx % 96;
    //int localMatId = threadIdx.x / 96;

    for(int j = 0; j < 96; j++)
    {
        if(i == j)
        {
            invP96[matId].m[j][i] = 1;
            if(P96[matId].m[j][i] == 0)
            {
                P96[matId].m[j][i] = 1;
            }
        }
        else
        {
            invP96[matId].m[j][i] = 0;
        }
    }
    __syncthreads();
    __threadfence();
    int         j  = 0;
    Precision_T rt = P96[matId].m[0][0];
    __syncthreads();
    __threadfence();
    while(/*loopId[localMatId]*/ j < 96)
    {
        if(i <= j)
            invP96[matId].m[j][i] /= rt;
        if(i > j)
            P96[matId].m[j][i] /= rt;

        __syncthreads();
        __threadfence();
        for(int k = 0; k < 96; k++)
        {
            if(k != j)
            {
                Precision_T rate = -P96[matId].m[k][j];
                __syncthreads();
                __threadfence();
                if(i <= j)
                    invP96[matId].m[k][i] += rate * invP96[matId].m[j][i];
                if(i > j)
                    P96[matId].m[k][i] += rate * P96[matId].m[j][i];
            }
        }

        __syncthreads();
        __threadfence();
        j++;
        rt = P96[matId].m[j][j];
    }
}


//__global__ void __inverse2_P96x96(MATHUTILS::Matrix96x96d* P96, MATHUTILS::Matrix96x96T* invP96, int numbers) {
//  int idx = blockIdx.x * blockDim.x + threadIdx.x;
//  if (idx >= numbers) return;
//
//  int matId = idx / 96;
//  int i = idx % 96;
//  //int localMatId = threadIdx.x / 96;
//
//  for (int j = 0; j < 96; j++)
//  {
//      if (i == j) {
//          invP96[matId].m[j][i] = 1;
//          if (P96[matId].m[j][i] == 0) {
//              P96[matId].m[j][i] = 1;
//          }
//      }
//      else {
//          invP96[matId].m[j][i] = 0;
//      }
//  }
//  __syncthreads();
//  //__shared__ int loopId[3];
//  //__shared__ double tempRate[3];
//
//  //if (i == 0) {
//  //  loopId[localMatId] = 0;
//  //  tempRate[localMatId] = P96[matId].m[0][0];
//  //}
//  int j = 0;
//  Precision_T rt = P96[matId].m[0][0];
//  __syncthreads();
//  while (/*loopId[localMatId]*/j < 96) {
//
//      //const int j = loopId[localMatId];
//      //const double rt = tempRate;//tempRate[localMatId];
//      if (i >= j) {
//          P96[matId].m[j][i] /= rt;
//      }
//      if (i <= j) {
//          invP96[matId].m[j][i] /= rt;
//      }
//      __syncthreads();
//      Precision_T rate = -P96[matId].m[i][j];
//      for (int k = 0; k < 96; k++) {
//          if (i != j) {
//
//              //__syncthreads();
//              if (k <= i) {
//                  invP96[matId].m[i][k] += rate * invP96[matId].m[j][k];
//              }
//              if (k >= j) {
//                  P96[matId].m[i][k] += rate * P96[matId].m[j][k];
//              }
//          }
//      }
//
//      __syncthreads();
//      //if (i == 0) {
//      //  loopId[localMatId]++;
//      //  tempRate[localMatId] = P96[matId].m[j + 1][j + 1];
//      //}
//      j++;
//      rt = P96[matId].m[j][j];
//      //__syncthreads();
//  }
//}


__global__ void __warp_inverse_P96x96(MATHUTILS::Matrix96x96MT* P96,
                                      MATHUTILS::Matrix96x96T*  invP96,
                                      int                        numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    int matId = idx / 32;
    int i     = idx % 32;

    for(int j = 0; j < 96; j++)
    {
        for(int k = 0; k < 3; k++)
        {
            int cid = (i + 32 * k);
            if(cid == j)
            {
                invP96[matId].m[j][cid] = 1;
                if(P96[matId].m[j][cid] == 0)
                {
                    P96[matId].m[j][cid] = 1;
                }
            }
            else
            {
                invP96[matId].m[j][cid] = 0;
            }
        }
    }

    int         j  = 0;
    Precision_T rt = P96[matId].m[j][j];
    while(j < 96)
    {

        for(int t = 0; t < 3; t++)
        {
            int cid = i + t * 32;
            if(cid >= j)
            {
                P96[matId].m[j][cid] /= rt;
            }
            invP96[matId].m[j][cid] /= rt;
        }

        for(int k = 0; k < 96; k++)
        {
            if(k != j)
            {
                Precision_T rate = -P96[matId].m[k][j];
                for(int t = 0; t < 3; t++)
                {
                    int cid = i + t * 32;
                    invP96[matId].m[k][cid] += rate * invP96[matId].m[j][cid];
                    if(cid >= j)
                    {
                        P96[matId].m[k][cid] += rate * P96[matId].m[j][cid];
                    }
                }
            }
        }

        j++;
        rt = P96[matId].m[j][j];
    }
}

__global__ void __buildMultiLevelR_optimized(const double3* _R,
                                             Precision_T3*  _multiLR,
                                             int*           _goingNext,
                                             unsigned int*  _fineConnectMsk,
                                             int            levelNum,
                                             int            numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    Precision_T3 r;
    r.x = _R[idx].x;
    r.y = _R[idx].y;
    r.z = _R[idx].z;

    int laneId      = threadIdx.x % 32;
    int localWarpId = threadIdx.x / 32;
    int level       = 0;
    _multiLR[idx]   = r;

    __shared__ Precision_TM c_sumResidual[DEFAULT_BLOCKSIZE * 3];

    unsigned int connectMsk = _fineConnectMsk[idx];
    if(connectMsk == -1)
    {
        for(int iter = 1; iter < 32; iter <<= 1)
        {
            r.x += __shfl_down_sync(0xFFFFFFFF, r.x, iter);
            r.y += __shfl_down_sync(0xFFFFFFFF, r.y, iter);
            r.z += __shfl_down_sync(0xFFFFFFFF, r.z, iter);
        }
        //int level = 0;

        if(laneId == 0)
        {
            while(level < levelNum - 1)
            {
                level++;
                idx = _goingNext[idx];
                atomicAdd((&((_multiLR + idx)->x)), r.x);
                atomicAdd((&((_multiLR + idx)->x) + 1), r.y);
                atomicAdd((&((_multiLR + idx)->x) + 2), r.z);
            }
        }
        return;
    }
    else
    {
        int elected_lane = __ffs(connectMsk) - 1;

        c_sumResidual[threadIdx.x]                         = 0;
        c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]     = 0;
        c_sumResidual[threadIdx.x + 2 * DEFAULT_BLOCKSIZE] = 0;
        atomicAdd(c_sumResidual + localWarpId * 32 + elected_lane, r.x);
        atomicAdd(c_sumResidual + localWarpId * 32 + elected_lane + DEFAULT_BLOCKSIZE, r.y);
        atomicAdd(c_sumResidual + localWarpId * 32 + elected_lane + 2 * DEFAULT_BLOCKSIZE,
                  r.z);

        unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
        if(electedPrefix == 0)
        {
            while(level < levelNum - 1)
            {
                level++;
                idx = _goingNext[idx];
                atomicAdd((&((_multiLR + idx)->x)), c_sumResidual[threadIdx.x]);
                atomicAdd((&((_multiLR + idx)->x) + 1),
                          c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]);
                atomicAdd((&((_multiLR + idx)->x) + 2),
                          c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE * 2]);
            }
        }
    }
}

__global__ void __buildMultiLevelR(
    const double3* _R, Precision_T3* _multiLR, int* _goingNext, int levelNum, int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    Precision_T3 r;
    r.x = _R[idx].x;
    r.y = _R[idx].y;
    r.z = _R[idx].z;

    int level     = 0;
    _multiLR[idx] = r;
    while(level < levelNum - 1)
    {
        level++;
        idx = _goingNext[idx];
        atomicAdd((&((_multiLR + idx)->x)), r.x);
        atomicAdd((&((_multiLR + idx)->x) + 1), r.y);
        atomicAdd((&((_multiLR + idx)->x) + 2), r.z);
    }
}

__global__ void __collectFinalZ(double3*            _Z,
                                const Precision_T3* mc_multiLevelZ,
                                const int4*         _coarseTable,
                                int                 levelnum,
                                int                 number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    Precision_T3 cz;  // = mc_multiLevelZ[idx];
    cz.x          = mc_multiLevelZ[idx].x;
    cz.y          = mc_multiLevelZ[idx].y;
    cz.z          = mc_multiLevelZ[idx].z;
    int4 table    = _coarseTable[idx];
    int* tablePtr = &(table.x);
    for(int i = 1; i < __mm_min(levelnum, 4); i++)
    {
        int now = *(tablePtr + i - 1);
        cz.x += mc_multiLevelZ[now].x;
        cz.y += mc_multiLevelZ[now].y;
        cz.z += mc_multiLevelZ[now].z;
    }

    _Z[idx].x = cz.x;
    _Z[idx].y = cz.y;
    _Z[idx].z = cz.z;
}

__global__ void _schwarzLocalXSym0(const MATHUTILS::Matrix96x96MT* P96,
                                   const Precision_T3*              mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    namespace cg = ::cooperative_groups;
    int idx      = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    auto tile = cg::tiled_partition<32>(cg::this_thread_block());

    int tileNo = idx / 32;
    int Hid    = tileNo / 96;
    int MRid   = tileNo % 96;

    int  vrid   = Hid * 32 + MRid / 3;
    auto laneid = tile.thread_rank();

    Precision_TM sum      = 0.;
    auto         get_vcid = [Hid](int cid) { return Hid * 32 + cid / 3; };
    sum += P96[Hid].m[MRid][laneid] * (*(&(mR[get_vcid(laneid)].x) + laneid % 3));
    laneid += 32;
    sum += P96[Hid].m[MRid][laneid] * (*(&(mR[get_vcid(laneid)].x) + laneid % 3));
    laneid += 32;
    sum += P96[Hid].m[MRid][laneid] * (*(&(mR[get_vcid(laneid)].x) + laneid % 3));

    auto val = cg::reduce(tile, sum, cg::plus<Precision_TM>());
    if(tile.thread_rank() == 0)
        *(&(mZ[vrid].x) + MRid % 3) += val;
}

__global__ void _schwarzLocalXSym(const MATHUTILS::Matrix96x96MT* P96,
                                  const Precision_T3*              mR,
                                  Precision_T3*                    mZ,
                                  int                              number)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = 96 * 96;

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / 96;
    int MCid = (idx % hessianSize) % 96;

    int vrid = Hid * 32 + MRid / 3;
    int vcid = Hid * 32 + MCid / 3;

    //int vId = MCid / 3;
    int axisId = MCid % 3;
    //int GRtid = idx % 96;

    Precision_TM rdata = P96[Hid].m[MRid][MCid] * (*(&(mR[vcid].x) + axisId));

    int warpId = threadIdx.x & 0x1f;

    unsigned int interval = 32;

    for(int iter = 1; iter < 32; iter <<= 1)
    {
        rdata += __shfl_down_sync(0xFFFFFFFF, rdata, iter);
    }

    if(!warpId)
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
}

__global__ void _schwarzLocalXSym3(const MATHUTILS::MasMatrixSymf* Pred,
                                   const Precision_T3*              mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (BANKSIZE * 3) * (BANKSIZE);

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / (BANKSIZE);
    int MCid = (idx % hessianSize) % (BANKSIZE);

    int vrid = Hid * BANKSIZE + MRid / 3;
    int vcid = Hid * BANKSIZE + MCid;

    int r3id = MRid % 3;

    int          lvrid = vrid % BANKSIZE;
    int          lvcid = vcid % BANKSIZE;
    Precision_TM rdata = 0;

    __shared__ Precision_T3 smR[BANKSIZE];

    if(threadIdx.x < BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid];
    }
    __syncthreads();

    if(lvcid >= lvrid)
    {
        int index = BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index].m[r3id][0] * smR[lvcid].x
                + Pred[Hid].M[index].m[r3id][1] * smR[lvcid].y
                + Pred[Hid].M[index].m[r3id][2] * smR[lvcid].z;
    }
    else
    {
        int index = BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index].m[0][r3id] * smR[lvcid].x
                + Pred[Hid].M[index].m[1][r3id] * smR[lvcid].y
                + Pred[Hid].M[index].m[2][r3id] * smR[lvcid].z;
    }
    //__syncthreads();
    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark     = __ballot_sync(0xFFFFFFFF, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    unsigned int interval = __mm_min(__clz(mark << (warpId + 1)), 31 - warpId);

    int maxSize = __mm_min(32, BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        Precision_TM tmpx = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
        if(interval >= iter)
        {

            rdata += tmpx;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
    }
}

__global__ void _buildCollisionConnection(unsigned int*     _pConnect,
                                          const int*        _pCoarseSpaceTable,
                                          const const int4* _collisionPair,
                                          int               level,
                                          int               vertNum,
                                          int               number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4 MMCVIDI              = _collisionPair[idx];
    int* collisionPairStartId = &(MMCVIDI.x);
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w < 0)
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
        }
        int cpVertNum = 4;
        int cpVid[4];
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                cpVid[i] = _pCoarseSpaceTable[collisionPairStartId[i] + (level - 1) * vertNum];
        }
        else
        {
            for(int i = 0; i < 4; i++)
                cpVid[i] = collisionPairStartId[i];
        }

        unsigned int connMsk[4] = {0};

        for(int i = 0; i < 4; i++)
        {
            for(int j = i + 1; j < 4; j++)
            {
                unsigned int myId = cpVid[i];
                unsigned int otId = cpVid[j];

                if(myId == otId)
                {
                    continue;
                }
                if(myId / BANKSIZE == otId / BANKSIZE)
                {
                    connMsk[i] |= (1U << (otId % BANKSIZE));
                    connMsk[j] |= (1U << (myId % BANKSIZE));
                }
            }
        }

        for(int i = 0; i < 4; i++)
            atomicOr(_pConnect + cpVid[i], connMsk[i]);
    }
    else
    {
        int v0I   = -MMCVIDI.x - 1;
        MMCVIDI.x = v0I;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;

                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        cpVid[i] =
                            _pCoarseSpaceTable[collisionPairStartId[i] + (level - 1) * vertNum];
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        cpVid[i] = collisionPairStartId[i];
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                for(int i = 0; i < 4; i++)
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
            }
            else
            {
                int cpVertNum = 2;
                int cpVid[2];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        cpVid[i] =
                            _pCoarseSpaceTable[collisionPairStartId[i] + (level - 1) * vertNum];
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        cpVid[i] = collisionPairStartId[i];
                }

                unsigned int connMsk[2] = {0};

                for(int i = 0; i < 2; i++)
                {
                    for(int j = i + 1; j < 2; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                for(int i = 0; i < 2; i++)
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;

                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        cpVid[i] =
                            _pCoarseSpaceTable[collisionPairStartId[i] + (level - 1) * vertNum];
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        cpVid[i] = collisionPairStartId[i];
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                for(int i = 0; i < 4; i++)
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
            }
            else
            {
                int cpVertNum = 3;
                int cpVid[3];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        cpVid[i] =
                            _pCoarseSpaceTable[collisionPairStartId[i] + (level - 1) * vertNum];
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        cpVid[i] = collisionPairStartId[i];
                }

                unsigned int connMsk[3] = {0};

                for(int i = 0; i < 3; i++)
                {
                    for(int j = i + 1; j < 3; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId)
                        {
                            continue;
                        }
                        if(myId / BANKSIZE == otId / BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % BANKSIZE));
                            connMsk[j] |= (1U << (myId % BANKSIZE));
                        }
                    }
                }

                for(int i = 0; i < 3; i++)
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
            }
        }
        else
        {
            int cpVertNum = 4;
            int cpVid[4];
            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    cpVid[i] =
                        _pCoarseSpaceTable[collisionPairStartId[i] + (level - 1) * vertNum];
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    cpVid[i] = collisionPairStartId[i];
            }

            unsigned int connMsk[4] = {0};

            for(int i = 0; i < 4; i++)
            {
                for(int j = i + 1; j < 4; j++)
                {
                    unsigned int myId = cpVid[i];
                    unsigned int otId = cpVid[j];

                    if(myId == otId)
                    {
                        continue;
                    }
                    if(myId / BANKSIZE == otId / BANKSIZE)
                    {
                        connMsk[i] |= (1U << (otId % BANKSIZE));
                        connMsk[j] |= (1U << (myId % BANKSIZE));
                    }
                }
            }

            for(int i = 0; i < 4; i++)
                atomicOr(_pConnect + cpVid[i], connMsk[i]);
        }
    }
}


void MASPreconditioner::BuildConnectMaskL0()
{
    int number    = totalNodes;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _buildCML0<<<numBlocks, blockSize>>>(
        mc_neighborStart, mc_neighborNum, mc_neighborList, mc_fineConnectMask, number);
}

void MASPreconditioner::PreparePrefixSumL0()
{
    int number    = totalNodes;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _preparePrefixSumL0<<<numBlocks, blockSize>>>(mc_prefixOriginal, mc_fineConnectMask, number);
}

void MASPreconditioner::BuildLevel1()
{
    int number    = totalNodes;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    //exclusive(mc_prefixOriginal, mc_prefixSumOriginal); wait to do;
    int warpNum = (number + 31) / 32;
    thrust::exclusive_scan(thrust::device_ptr<int>(mc_prefixOriginal),
                           thrust::device_ptr<int>(mc_prefixOriginal) + warpNum,
                           thrust::device_ptr<int>(mc_prefixSumOriginal));
    _buildLevel1<<<numBlocks, blockSize>>>(mc_levelSize,
                                           mc_coarseSpaceTables,
                                           mc_goingNext,
                                           mc_fineConnectMask,
                                           mc_prefixSumOriginal,
                                           mc_prefixOriginal,
                                           number);
}

void MASPreconditioner::BuildConnectMaskLx(int level)
{
    int number    = totalNodes;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _buildConnectMaskLx<<<numBlocks, blockSize>>>(mc_neighborStart,
                                                  mc_neighborNum,
                                                  mc_neighborList,
                                                  mc_coarseSpaceTables,
                                                  mc_nextConnectMask,
                                                  mc_fineConnectMask,
                                                  level,
                                                  number);
}

void MASPreconditioner::NextLevelCluster(int level)
{
    int number    = h_clevelSize.x;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _nextLevelCluster<<<numBlocks, blockSize>>>(mc_nextConnectMask, mc_nextPrefix, number);
}

void MASPreconditioner::ComputeNextLevel(int level)
{
    int number    = totalNodes;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _computeNextLevel<<<numBlocks, blockSize>>>(
        mc_coarseSpaceTables, mc_nextConnectMask, level, number);
}

void MASPreconditioner::PrefixSumLx(int level)
{
    int number     = h_clevelSize.x;
    int levelBegin = h_clevelSize.y;
    int blockSize  = BANKSIZE * BANKSIZE;
    int numBlocks  = (number + blockSize - 1) / blockSize;

    int warpNum = (number + 31) / 32;
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(mc_nextPrefix),
                           thrust::device_ptr<unsigned int>(mc_nextPrefix) + warpNum,
                           thrust::device_ptr<unsigned int>(mc_nextPrefixSum));

    _prefixSumLx<<<numBlocks, blockSize>>>(
        mc_levelSize, mc_nextPrefix, mc_nextPrefixSum, mc_nextConnectMask, mc_goingNext, level, levelBegin, number);
}

void MASPreconditioner::AggregationKernel()
{
    int number    = totalNodes;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _aggregationKernel<<<numBlocks, blockSize>>>(
        mc_denseLevel, mc_coarseTable, mc_goingNext, levelnum, number);
}


void MASPreconditioner::computeNumLevels(int vertNum)
{
    int totalSz = 0;
    int nLevel  = 1;
    int levelSz = (vertNum + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
    totalSz += levelSz;

    while(levelSz > BANKSIZE)
    {
        levelSz /= BANKSIZE;

        nLevel++;
        levelSz = (levelSz + BANKSIZE - 1) / BANKSIZE * BANKSIZE;
        totalSz += levelSz;
    }

    levelnum = nLevel;
    printf("level num:  %d\n", levelnum);
    //totalSize = totalSz * SizeRatio;
    totalNodes = vertNum;
}

void MASPreconditioner::BuildCollisionConnection(unsigned int* connectionMsk,
                                                 int*          coarseTableSpace,
                                                 int           level,
                                                 int           cpNum)
{
    int number    = cpNum;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _buildCollisionConnection<<<numBlocks, blockSize>>>(
        connectionMsk, coarseTableSpace, mc_collisonPairs, level, totalNodes, number);
}

int MASPreconditioner::ReorderRealtime(int cpNum)
{
    CUDA_SAFE_CALL(cudaMemset(mc_levelSize, 0, levelnum * sizeof(int2)));


    BuildConnectMaskL0();


    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(cpNum)
        BuildCollisionConnection(mc_fineConnectMask, nullptr, -1, cpNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    PreparePrefixSumL0();

    //vector<unsigned int> h_fineCMsk(totalSize);
    //CUDA_SAFE_CALL(cudaMemcpy(h_fineCMsk.data(), mc_prefixOriginal, totalNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    //for (int i = 0; i < totalNodes; i++) {
    //  /*char s[40];
    //  itoa(h_fineCMsk[i], s, 2);
    //  printf("%s\n", s);*/
    //  //cout << bitset<sizeof(h_fineCMsk[i]) * 8>(h_fineCMsk[i]) << endl;
    //  cout << h_fineCMsk[i] << endl;
    //}

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    BuildLevel1();

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for(int level = 1; level < levelnum; level++)
    {
        CUDA_SAFE_CALL(cudaMemset(mc_nextConnectMask, 0, totalNodes * sizeof(int)));

        BuildConnectMaskLx(level);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        if(cpNum)
            BuildCollisionConnection(mc_nextConnectMask, mc_coarseSpaceTables, level, cpNum);


        CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize, mc_levelSize + level, sizeof(int2), cudaMemcpyDeviceToHost));

        //cout << "hello:    " << h_clevelSize.x << endl;

        NextLevelCluster(level);


        //vector<unsigned int> h_fineCMsk(totalSize);
        //CUDA_SAFE_CALL(cudaMemcpy(h_fineCMsk.data(), mc_nextPrefix, totalNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));

        //for (int i = 0; i < totalNodes; i++) {
        //  /*char s[40];
        //  itoa(h_fineCMsk[i], s, 2);
        //  printf("%s\n", s);*/
        //  //cout << bitset<sizeof(h_fineCMsk[i]) * 8>(h_fineCMsk[i]) << endl;
        //  cout << h_fineCMsk[i] << endl;
        //}


        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        PrefixSumLx(level);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        ComputeNextLevel(level);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize, mc_levelSize + levelnum, sizeof(int2), cudaMemcpyDeviceToHost));

    totalNumberClusters = h_clevelSize.y;

    AggregationKernel();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    return totalNumberClusters;

    //vector<unsigned int> h_fineCMsk(totalNumberClusters);
    //CUDA_SAFE_CALL(cudaMemcpy(h_fineCMsk.data(), mc_goingNext, totalNumberClusters * sizeof(unsigned int), cudaMemcpyDeviceToHost));


    //for (int i = 0; i < totalNumberClusters; i++) {
    //  /*char s[40];
    //  itoa(h_fineCMsk[i], s, 2);
    //  printf("%s\n", s);*/
    //  //cout << bitset<sizeof(h_fineCMsk[i]) * 8>(h_fineCMsk[i]) << endl;
    //  cout << i << "    " << h_fineCMsk[i] << endl;
    //}
}

//#include <fstream>

void MASPreconditioner::PrepareHessian(const std::unique_ptr<GeometryManager>& instance, const double* masses)
{
    //cudaEvent_t start, end0, end1, end2;
    //cudaEventCreate(&start);
    //cudaEventCreate(&end0);
    //cudaEventCreate(&end1);
    //cudaEventCreate(&end2);


    int number = totalNodes;

    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //cout << totalSize / 32 << endl;
    //cudaEventRecord(start);
    __setMassMat_P96<<<numBlocks, blockSize>>>(masses, mc_goingNext, mc_Mat96, levelnum, totalNodes);

    //cudaEventRecord(end0);


    number = instance->DNum[3] * 144 + instance->DNum[2] * 81 + instance->DNum[1] * 36 + instance->DNum[0] * 9;
    numBlocks = (number + blockSize - 1) / blockSize;

    _prepareHessian<<<numBlocks, blockSize>>>(instance->cudaH12x12,
                                              instance->cudaH9x9,
                                              instance->cudaH6x6,
                                              instance->cudaH3x3,
                                              instance->cudaD4Index,
                                              instance->cudaD3Index,
                                              instance->cudaD2Index,
                                              instance->cudaD1Index,
                                              mc_Mat96,
                                              instance->DNum[3] * 144,
                                              instance->DNum[2] * 81,
                                              instance->DNum[1] * 36,
                                              instance->DNum[0] * 9,
                                              mc_goingNext,
                                              levelnum);

    //cudaEventRecord(end1);

    blockSize = 96;
    number    = totalNumberClusters * 3;
    numBlocks = (number + blockSize - 1) / blockSize;
    __inverse2_P96x96<<<numBlocks, blockSize>>>(mc_Mat96, mc_inverseMat96, number);

    //cudaEventRecord(end2);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //float time0, time1, time2, time3, time4;
    //cudaEventElapsedTime(&time0, start, end0);
    //cudaEventElapsedTime(&time1, end0, end1);
    //cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\ntime0 = %f,  time1 = %f,  time1 = %f\n\n", time0, time1, time2);

    //(cudaEventDestroy(start));
    //(cudaEventDestroy(end0));
    //(cudaEventDestroy(end1));
    //(cudaEventDestroy(end2));
}

void MASPreconditioner::BuildMultiLevelR(const double3* R)
{
    int number = totalNodes;

    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //__buildMultiLevelR << <numBlocks, blockSize >> > (R, mc_multiLevelR, mc_goingNext, levelnum, number);
    __buildMultiLevelR_optimized<<<numBlocks, blockSize>>>(
        R, mc_multiLevelR, mc_goingNext, mc_fineConnectMask, levelnum, number);
    //vector<double3> h_r(totalSize);
    //CUDA_SAFE_CALL(cudaMemcpy(h_r.data(), R, totalNodes * sizeof(double3), cudaMemcpyDeviceToHost));

    //for (int i = 0; i < totalSize; i++) {

    //  cout << h_r[i].x << " " << h_r[i].y << " " << h_r[i].z << endl;
    //  //cout << h_fineCMsk[i] << endl;
    //}
}

void MASPreconditioner::SchwarzLocalXSym()
{
    int number    = totalNumberClusters * BANKSIZE * 3;
    int blockSize = BANKSIZE * BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, mc_multiLevelR, mc_multiLevelZ, number);
    _schwarzLocalXSym3<<<numBlocks, blockSize>>>(mc_inverseMat96, mc_multiLevelR, mc_multiLevelZ, number);
}

void MASPreconditioner::CollectFinalZ(double3* Z)
{
    int number = totalNodes;

    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    __collectFinalZ<<<numBlocks, blockSize>>>(Z, mc_multiLevelZ, mc_coarseTable, levelnum, number);

}

void MASPreconditioner::setPreconditioner(const std::unique_ptr<GeometryManager>& instance, int cpNum) {

    CUDA_SAFE_CALL(cudaMemcpy(mc_neighborList,
                              mc_neighborListInit,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(ipc.instance->MAS_ptr->mc_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mc_neighborNum,
                              mc_neighborNumInit,
                              totalNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));


    //cudaEventRecord(start);

    ReorderRealtime(cpNum);

    //cudaEventRecord(end0);

    CUDA_SAFE_CALL(cudaMemset(
        mc_Mat96, 0, totalNumberClusters / BANKSIZE * sizeof(MATHUTILS::Matrix96x96T)));

    PrepareHessian(instance, instance->cudaVertMass);

    //cudaEventRecord(end1);
}

void MASPreconditioner::preconditioning(const double3* R, double3* Z)
{
    CUDA_SAFE_CALL(cudaMemset(mc_multiLevelR + totalNodes,
                              0,
                              (totalNumberClusters - totalNodes) * sizeof(Precision_T3)));
    CUDA_SAFE_CALL(cudaMemset(mc_multiLevelZ, 0, (totalNumberClusters) * sizeof(Precision_T3)));

    //cudaEvent_t start, end0, end1, end2;
    //cudaEventCreate(&start);
    //cudaEventCreate(&end0);
    //cudaEventCreate(&end1);
    //cudaEventCreate(&end2);

    //cudaEventRecord(start);
    BuildMultiLevelR(R);
    //cudaEventRecord(end0);
    SchwarzLocalXSym();
    //cudaEventRecord(end1);
    CollectFinalZ(Z);
    //cudaEventRecord(end2);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //float time0, time1, time2, time3, time4;
    //cudaEventElapsedTime(&time0, start, end0);
    //cudaEventElapsedTime(&time1, end0, end1);
    //cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\npreconditioning  time0 = %f,  time1 = %f,  time1 = %f\n\n", time0, time1, time2);

    //(cudaEventDestroy(start));
    //(cudaEventDestroy(end0));
    //(cudaEventDestroy(end1));
    //(cudaEventDestroy(end2));
}




MASPreconditioner::MASPreconditioner() {}

MASPreconditioner::~MASPreconditioner() {}

void MASPreconditioner::CUDA_MALLOC_MAS(int vertNum, int totalNeighborNum, int4* _collisonPairs) {
    //bankSize = 32;
    computeNumLevels(vertNum);
    mc_collisonPairs = _collisonPairs;

    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_denseLevel, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_coarseTable, vertNum * sizeof(int4)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_coarseSpaceTables, vertNum * levelnum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_levelSize, (levelnum + 1) * sizeof(int2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_goingNext, vertNum * levelnum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_prefixOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_nextPrefix, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_nextPrefixSum, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_prefixSumOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_fineConnectMask, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_nextConnectMask, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_neighborList, totalNeighborNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_neighborStart, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_neighborStartTemp, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_neighborNum, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_neighborListInit, totalNeighborNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_neighborNumInit, vertNum * sizeof(int)));

    int totalCluster = ReorderRealtime(0) * 1.05;

    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_Mat96, totalCluster / BANKSIZE * sizeof(MATHUTILS::Matrix96x96T)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_inverseMat96, totalCluster / BANKSIZE * sizeof(MATHUTILS::MasMatrixSymf)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_multiLevelR, totalCluster * sizeof(Precision_T3)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&mc_multiLevelZ, totalCluster * sizeof(Precision_T3)));
}

void MASPreconditioner::CUDA_FREE_MAS() {

    CUDA_SAFE_CALL(cudaFree(mc_denseLevel));
    CUDA_SAFE_CALL(cudaFree(mc_coarseSpaceTables));
    CUDA_SAFE_CALL(cudaFree(mc_levelSize));
    CUDA_SAFE_CALL(cudaFree(mc_goingNext));
    CUDA_SAFE_CALL(cudaFree(mc_prefixOriginal));
    CUDA_SAFE_CALL(cudaFree(mc_nextPrefix));
    CUDA_SAFE_CALL(cudaFree(mc_nextPrefixSum));
    CUDA_SAFE_CALL(cudaFree(mc_prefixSumOriginal));
    CUDA_SAFE_CALL(cudaFree(mc_fineConnectMask));
    CUDA_SAFE_CALL(cudaFree(mc_nextConnectMask));
    CUDA_SAFE_CALL(cudaFree(mc_neighborList));
    CUDA_SAFE_CALL(cudaFree(mc_neighborListInit));
    CUDA_SAFE_CALL(cudaFree(mc_neighborStart));
    CUDA_SAFE_CALL(cudaFree(mc_neighborStartTemp));
    CUDA_SAFE_CALL(cudaFree(mc_neighborNum));
    CUDA_SAFE_CALL(cudaFree(mc_neighborNumInit));
    CUDA_SAFE_CALL(cudaFree(mc_Mat96));
    CUDA_SAFE_CALL(cudaFree(mc_inverseMat96));
    CUDA_SAFE_CALL(cudaFree(mc_multiLevelR));
    CUDA_SAFE_CALL(cudaFree(mc_multiLevelZ));
}
