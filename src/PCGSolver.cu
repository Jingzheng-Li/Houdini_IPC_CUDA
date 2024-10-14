
#include "PCGSolver.cuh"

// __global__ void __PCG_Solve_AX12_b2(const __MATHUTILS__::Matrix12x12S* Hessians, const uint4*
// D4Index, const Scalar3* c,
//                                     Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     __shared__ int offset;
//     int Hid = idx / 144;
//     int MRid = (idx % 144) / 12;
//     int MCid = (idx % 144) % 12;

//     int vId = MCid / 3;
//     int axisId = MCid % 3;
//     int GRtid = idx % 12;

//     Scalar rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) +
//     axisId));

//     if (threadIdx.x == 0) {
//         offset = (12 - GRtid);
//     }
//     __syncthreads();

//     int BRid = (threadIdx.x - offset + 12) / 12;
//     int landidx = (threadIdx.x - offset) % 12;
//     if (BRid == 0) {
//         landidx = threadIdx.x;
//     }

//     int warpId = threadIdx.x & 0x1f;
//     bool bBoundary = (landidx == 0) || (warpId == 0);

//     unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
//     mark = __brev(mark);
//     unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);
//     // mark = interval;
//     // for (int iter = 1; iter & 0x1f; iter <<= 1) {
//     //     int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
//     //     mark = tmp > mark ? tmp : mark;
//     // }
//     // mark = __shfl_sync(0xFFFFFFFF, mark, 0);
//     //__syncthreads();

//     for (int iter = 1; iter < 12; iter <<= 1) {
//         Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
//         if (interval >= iter) rdata += tmp;
//     }

//     if (bBoundary) atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
// }

// __global__ void __PCG_Solve_AX12_b3(const __MATHUTILS__::Matrix12x12S* Hessians, const uint4*
// D4Index, const Scalar3* c,
//                                     Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     __shared__ int offset0, offset1;
//     __shared__ Scalar tempB[36];

//     int Hid = idx / 144;

//     int HRtid = idx % 144;

//     int MRid = (HRtid) / 12;
//     int MCid = (HRtid) % 12;

//     int vId = MCid / 3;
//     int axisId = MCid % 3;
//     int GRtid = idx % 12;

//     if (threadIdx.x == 0) {
//         offset0 = (144 - HRtid);
//         offset1 = (12 - GRtid);
//     }
//     __syncthreads();

//     int HRid = (threadIdx.x - offset0 + 144) / 144;
//     int Hlandidx = (threadIdx.x - offset0) % 144;
//     if (HRid == 0) {
//         Hlandidx = threadIdx.x;
//     }

//     int BRid = (threadIdx.x - offset1 + 12) / 12;
//     int landidx = (threadIdx.x - offset1) % 12;
//     if (BRid == 0) {
//         landidx = threadIdx.x;
//     }

//     if (HRid > 0 && Hlandidx < 12) {
//         tempB[HRid * 12 + Hlandidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
//     } else if (HRid == 0) {
//         if (offset0 <= 12) {
//             tempB[HRid * 12 + Hlandidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
//         } else if (BRid == 1) {
//             tempB[HRid * 12 + landidx] = (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));
//         }
//     }

//     __syncthreads();

//     int readBid = landidx;
//     if (offset0 > 12 && threadIdx.x < offset1) readBid = landidx + (12 - offset1);
//     Scalar rdata =
//         Hessians[Hid].m[MRid][MCid] * tempB[HRid * 12 + readBid];  //(*(&(c[*(&(D4Index[Hid].x) +
//         vId)].x) + axisId));

//     int warpId = threadIdx.x & 0x1f;
//     bool bBoundary = (landidx == 0) || (warpId == 0);

//     unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
//     mark = __brev(mark);
//     unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

//     for (int iter = 1; iter < 12; iter <<= 1) {
//         Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
//         if (interval >= iter) rdata += tmp;
//     }

//     if (bBoundary) atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
// }

// __global__ void __PCG_Solve_AX9_b(const __MATHUTILS__::Matrix9x9S* Hessians, const uint3*
// D3Index, const Scalar3* c,
//                                   Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     __MATHUTILS__::Matrix9x9S H = Hessians[idx];
//     __MATHUTILS__::Vector9S tempC, tempQ;

//     tempC.v[0] = c[D3Index[idx].x].x;
//     tempC.v[1] = c[D3Index[idx].x].y;
//     tempC.v[2] = c[D3Index[idx].x].z;

//     tempC.v[3] = c[D3Index[idx].y].x;
//     tempC.v[4] = c[D3Index[idx].y].y;
//     tempC.v[5] = c[D3Index[idx].y].z;

//     tempC.v[6] = c[D3Index[idx].z].x;
//     tempC.v[7] = c[D3Index[idx].z].y;
//     tempC.v[8] = c[D3Index[idx].z].z;

//     tempQ = __MATHUTILS__::__M9x9_v9_multiply(H, tempC);

//     atomicAdd(&(q[D3Index[idx].x].x), tempQ.v[0]);
//     atomicAdd(&(q[D3Index[idx].x].y), tempQ.v[1]);
//     atomicAdd(&(q[D3Index[idx].x].z), tempQ.v[2]);

//     atomicAdd(&(q[D3Index[idx].y].x), tempQ.v[3]);
//     atomicAdd(&(q[D3Index[idx].y].y), tempQ.v[4]);
//     atomicAdd(&(q[D3Index[idx].y].z), tempQ.v[5]);

//     atomicAdd(&(q[D3Index[idx].z].x), tempQ.v[6]);
//     atomicAdd(&(q[D3Index[idx].z].y), tempQ.v[7]);
//     atomicAdd(&(q[D3Index[idx].z].z), tempQ.v[8]);
// }

// __global__ void __PCG_Solve_AX9_b2(const __MATHUTILS__::Matrix9x9S* Hessians, const uint3*
// D3Index, const Scalar3* c,
//                                    Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     // extern __shared__ Scalar sData[];
//     __shared__ int offset;
//     int Hid = idx / 81;
//     int MRid = (idx % 81) / 9;
//     int MCid = (idx % 81) % 9;

//     int vId = MCid / 3;
//     int axisId = MCid % 3;
//     int GRtid = idx % 9;
//     // sData[threadIdx.x] = Hessians[Hid].m[MRid][MCid] *
//     // (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId)); printf("landidx  %f  %d
//     // %d   %d\n", sData[threadIdx.x], offset, 1, 1);
//     Scalar rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) +
//     axisId));

//     if (threadIdx.x == 0) {
//         offset = (9 - GRtid);  // < 12 ? (12 - GRtid) : 0;
//     }
//     __syncthreads();

//     int BRid = (threadIdx.x - offset + 9) / 9;
//     int landidx = (threadIdx.x - offset) % 9;
//     if (BRid == 0) {
//         landidx = threadIdx.x;
//     }

//     int warpId = threadIdx.x & 0x1f;
//     bool bBoundary = (landidx == 0) || (warpId == 0);

//     unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);  // a bit-mask
//     mark = __brev(mark);
//     unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);
//     // mark = interval;
//     // for (int iter = 1; iter & 0x1f; iter <<= 1) {
//     //     int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
//     //     mark = tmp > mark ? tmp : mark; /*if (tmp > mark) mark = tmp;*/
//     // }
//     // mark = __shfl_sync(0xFFFFFFFF, mark, 0);
//     //__syncthreads();

//     for (int iter = 1; iter < 9; iter <<= 1) {
//         Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
//         if (interval >= iter) rdata += tmp;
//     }

//     if (bBoundary) atomicAdd((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
// }

// __global__ void __PCG_Solve_AX6_b(const __MATHUTILS__::Matrix6x6S* Hessians, const uint2*
// D2Index, const Scalar3* c,
//                                   Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     __MATHUTILS__::Matrix6x6S H = Hessians[idx];
//     __MATHUTILS__::Vector6S tempC, tempQ;

//     tempC.v[0] = c[D2Index[idx].x].x;
//     tempC.v[1] = c[D2Index[idx].x].y;
//     tempC.v[2] = c[D2Index[idx].x].z;

//     tempC.v[3] = c[D2Index[idx].y].x;
//     tempC.v[4] = c[D2Index[idx].y].y;
//     tempC.v[5] = c[D2Index[idx].y].z;

//     tempQ = __MATHUTILS__::__M6x6_v6_multiply(H, tempC);

//     atomicAdd(&(q[D2Index[idx].x].x), tempQ.v[0]);
//     atomicAdd(&(q[D2Index[idx].x].y), tempQ.v[1]);
//     atomicAdd(&(q[D2Index[idx].x].z), tempQ.v[2]);

//     atomicAdd(&(q[D2Index[idx].y].x), tempQ.v[3]);
//     atomicAdd(&(q[D2Index[idx].y].y), tempQ.v[4]);
//     atomicAdd(&(q[D2Index[idx].y].z), tempQ.v[5]);
// }

// __global__ void __PCG_Solve_AX6_b2(const __MATHUTILS__::Matrix6x6S* Hessians, const uint2*
// D2Index, const Scalar3* c,
//                                    Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     __shared__ int offset;
//     int Hid = idx / 36;
//     int MRid = (idx % 36) / 6;
//     int MCid = (idx % 36) % 6;

//     int vId = MCid / 3;
//     int axisId = MCid % 3;
//     int GRtid = idx % 6;

//     Scalar rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) +
//     axisId));

//     if (threadIdx.x == 0) {
//         offset = (6 - GRtid);
//     }
//     __syncthreads();

//     int BRid = (threadIdx.x - offset + 6) / 6;
//     int landidx = (threadIdx.x - offset) % 6;
//     if (BRid == 0) {
//         landidx = threadIdx.x;
//     }

//     int warpId = threadIdx.x & 0x1f;
//     bool bBoundary = (landidx == 0) || (warpId == 0);

//     unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
//     mark = __brev(mark);
//     unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);
//     // mark = interval;
//     // for (int iter = 1; iter & 0x1f; iter <<= 1) {
//     //     int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
//     //     mark = tmp > mark ? tmp : mark;
//     // }
//     // mark = __shfl_sync(0xFFFFFFFF, mark, 0);
//     //__syncthreads();

//     for (int iter = 1; iter < 6; iter <<= 1) {
//         Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
//         if (interval >= iter) rdata += tmp;
//     }

//     if (bBoundary) atomicAdd((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);
// }

// __global__ void __PCG_Solve_AX3_b(const __MATHUTILS__::Matrix3x3S* Hessians, const uint32_t*
// D1Index, const Scalar3* c,
//                                   Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     __MATHUTILS__::Matrix3x3S H = Hessians[idx];
//     Scalar3 tempC, tempQ;

//     tempC.x = c[D1Index[idx]].x;
//     tempC.y = c[D1Index[idx]].y;
//     tempC.z = c[D1Index[idx]].z;

//     tempQ = __MATHUTILS__::__M_v_multiply(H, tempC);

//     atomicAdd(&(q[D1Index[idx]].x), tempQ.x);
//     atomicAdd(&(q[D1Index[idx]].y), tempQ.y);
//     atomicAdd(&(q[D1Index[idx]].z), tempQ.z);
// }

// __global__ void __PCG_Solve_AX12_b1(const __MATHUTILS__::Matrix12x12S* Hessians, const uint4*
// D4Index, const Scalar3* c,
//                                     Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     extern __shared__ Scalar sData[];
//     __shared__ int offset;
//     int Hid = idx / 144;
//     int MRid = (idx % 144) / 12;
//     int MCid = (idx % 144) % 12;

//     int vId = MCid / 3;
//     int axisId = MCid % 3;
//     int GRtid = idx % 12;
//     sData[threadIdx.x] = Hessians[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) +
//     axisId));

//     if (threadIdx.x == 0) {
//         offset = (12 - GRtid);
//     }
//     __syncthreads();

//     int BRid = (threadIdx.x - offset + 12) / 12;
//     int Num;  // = 12 + BRid - GRtid;
//     int startId = offset + BRid * 12 - 12;
//     int landidx = (threadIdx.x - offset) % 12;
//     if (BRid == 0) {
//         Num = offset;
//         startId = 0;
//         landidx = threadIdx.x;
//     } else if (BRid * 12 + offset > blockDim.x) {
//         Num = blockDim.x - offset - BRid * 12 + 12;
//     } else {
//         Num = 12;
//     }

//     int iter = Num;
//     for (int i = 1; i < 12; i = (i << 1)) {
//         if (i < Num) {
//             int tempNum = iter;
//             iter = ((iter + 1) >> 1);
//             if (landidx < iter) {
//                 if (threadIdx.x + iter < blockDim.x && threadIdx.x + iter < startId + tempNum)
//                     sData[threadIdx.x] += sData[threadIdx.x + iter];
//             }
//         }
//         __syncthreads();
//         //__threadfence();
//     }
//     __syncthreads();
//     if (threadIdx.x == 0 || GRtid == 0)
//         atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), sData[threadIdx.x]);
// }

// __global__ void __PCG_Solve_AX12_b(const __MATHUTILS__::Matrix12x12S* Hessians, const uint4*
// D4Index, const Scalar3* c,
//                                    Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     __MATHUTILS__::Matrix12x12S H = Hessians[idx];
//     __MATHUTILS__::Vector12S tempC, tempQ;

//     tempC.v[0] = c[D4Index[idx].x].x;
//     tempC.v[1] = c[D4Index[idx].x].y;
//     tempC.v[2] = c[D4Index[idx].x].z;

//     tempC.v[3] = c[D4Index[idx].y].x;
//     tempC.v[4] = c[D4Index[idx].y].y;
//     tempC.v[5] = c[D4Index[idx].y].z;

//     tempC.v[6] = c[D4Index[idx].z].x;
//     tempC.v[7] = c[D4Index[idx].z].y;
//     tempC.v[8] = c[D4Index[idx].z].z;

//     tempC.v[9] = c[D4Index[idx].w].x;
//     tempC.v[10] = c[D4Index[idx].w].y;
//     tempC.v[11] = c[D4Index[idx].w].z;

//     tempQ = __MATHUTILS__::__M12x12_v12_multiply(H, tempC);

//     atomicAdd(&(q[D4Index[idx].x].x), tempQ.v[0]);
//     atomicAdd(&(q[D4Index[idx].x].y), tempQ.v[1]);
//     atomicAdd(&(q[D4Index[idx].x].z), tempQ.v[2]);

//     atomicAdd(&(q[D4Index[idx].y].x), tempQ.v[3]);
//     atomicAdd(&(q[D4Index[idx].y].y), tempQ.v[4]);
//     atomicAdd(&(q[D4Index[idx].y].z), tempQ.v[5]);

//     atomicAdd(&(q[D4Index[idx].z].x), tempQ.v[6]);
//     atomicAdd(&(q[D4Index[idx].z].y), tempQ.v[7]);
//     atomicAdd(&(q[D4Index[idx].z].z), tempQ.v[8]);

//     atomicAdd(&(q[D4Index[idx].w].x), tempQ.v[9]);
//     atomicAdd(&(q[D4Index[idx].w].y), tempQ.v[10]);
//     atomicAdd(&(q[D4Index[idx].w].z), tempQ.v[11]);
// }

// __global__ void __PCG_Solve_AX3_b2(const __MATHUTILS__::Matrix3x3S* Hessians, const uint32_t*
// D1Index, const Scalar3* c,
//                                    Scalar3* q, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     __shared__ int offset;
//     int Hid = idx / 9;
//     int MRid = (idx % 9) / 3;
//     int MCid = (idx % 9) % 3;

//     int axisId = MCid % 3;
//     int GRtid = idx % 3;

//     Scalar rdata = Hessians[Hid].m[MRid][MCid] * (*(&(c[(D1Index[Hid])].x) + axisId));

//     if (threadIdx.x == 0) {
//         offset = (3 - GRtid);
//     }
//     __syncthreads();

//     int BRid = (threadIdx.x - offset + 3) / 3;
//     int landidx = (threadIdx.x - offset) % 3;
//     if (BRid == 0) {
//         landidx = threadIdx.x;
//     }

//     int warpId = threadIdx.x & 0x1f;
//     bool bBoundary = (landidx == 0) || (warpId == 0);

//     unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
//     mark = __brev(mark);
//     unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);
//     mark = interval;
//     for (int iter = 1; iter & 0x1f; iter <<= 1) {
//         int tmp = __shfl_down_sync(0xFFFFFFFF, mark, iter);
//         mark = tmp > mark ? tmp : mark;
//     }
//     mark = __shfl_sync(0xFFFFFFFF, mark, 0);
//     __syncthreads();

//     for (int iter = 1; iter <= mark; iter <<= 1) {
//         Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
//         if (interval >= iter) rdata += tmp;
//     }

//     if (bBoundary) atomicAdd((&(q[(D1Index[Hid])].x) + MRid % 3), rdata);
// }


// __global__ void __PCG_init_P(const Scalar* _masses, __MATHUTILS__::Matrix3x3S* P, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;
    
//     Scalar s1 = P[idx].m[0][0];
//     Scalar s2 = P[idx].m[1][1];
//     Scalar s3 = P[idx].m[2][2];
//     __MATHUTILS__::__init_Mat3x3(P[idx], 0);
//     P[idx].m[0][0] = 1 / s1;
//     P[idx].m[1][1] = 1 / s2;
//     P[idx].m[2][2] = 1 / s3;
// }

// __global__ void __PCG_AX12_P(const __MATHUTILS__::Matrix12x12S* Hessians, const uint4* D4Index,
//                              __MATHUTILS__::Matrix3x3S* P, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     int Hid = idx / 12;
//     int qid = idx % 12;

//     // Scalar Hval = Hessians[Hid].m[qid][qid];
//     // atomicAdd((&(P[*(&(D4Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
//     int mid = (qid / 3) * 3;
//     int tid = qid % 3;

//     Scalar Hval = Hessians[Hid].m[mid][mid + tid];
//     atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
//     Hval = Hessians[Hid].m[mid + 1][mid + tid];
//     atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
//     Hval = Hessians[Hid].m[mid + 2][mid + tid];
//     atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
// }

// __global__ void __PCG_AX9_P(const __MATHUTILS__::Matrix9x9S* Hessians, const uint3* D3Index,
//                             __MATHUTILS__::Matrix3x3S* P, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     int Hid = idx / 9;
//     int qid = idx % 9;

//     // Scalar Hval = Hessians[Hid].m[qid][qid];
//     // atomicAdd((&(P[*(&(D3Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
//     int mid = (qid / 3) * 3;
//     int tid = qid % 3;

//     Scalar Hval = Hessians[Hid].m[mid][mid + tid];
//     atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
//     Hval = Hessians[Hid].m[mid + 1][mid + tid];
//     atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
//     Hval = Hessians[Hid].m[mid + 2][mid + tid];
//     atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
// }

// __global__ void __PCG_AX6_P(const __MATHUTILS__::Matrix6x6S* Hessians, const uint2* D2Index,
//                             __MATHUTILS__::Matrix3x3S* P, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     int Hid = idx / 6;
//     int qid = idx % 6;

//     // Scalar Hval = Hessians[Hid].m[qid][qid];
//     // atomicAdd((&(P[*(&(D2Index[Hid].x) + qid / 3)].x) + qid % 3), Hval);
//     int mid = (qid / 3) * 3;
//     int tid = qid % 3;

//     Scalar Hval = Hessians[Hid].m[mid][mid + tid];
//     atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
//     Hval = Hessians[Hid].m[mid + 1][mid + tid];
//     atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
//     Hval = Hessians[Hid].m[mid + 2][mid + tid];
//     atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);
// }

// __global__ void __PCG_AX3_P(const __MATHUTILS__::Matrix3x3S* Hessians, const uint32_t* D1Index,
//                             __MATHUTILS__::Matrix3x3S* P, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;

//     int Hid = idx / 3;
//     int qid = idx % 3;

//     // Scalar Hval = Hessians[Hid].m[qid][qid];
//     //*(&(P[(D1Index[Hid])].x) + qid) += Hval;
//     // P[D1Index[Hid]].m[0][qid] += Hessians[Hid].m[0][qid];
//     // P[D1Index[Hid]].m[1][qid] += Hessians[Hid].m[1][qid];
//     // P[D1Index[Hid]].m[2][qid] += Hessians[Hid].m[2][qid];
//     atomicAdd(&(P[D1Index[Hid]].m[0][qid]), Hessians[Hid].m[0][qid]);
//     atomicAdd(&(P[D1Index[Hid]].m[1][qid]), Hessians[Hid].m[1][qid]);
//     atomicAdd(&(P[D1Index[Hid]].m[2][qid]), Hessians[Hid].m[2][qid]);
// }

// __global__ void __PCG_initDX(Scalar3* dx, const Scalar3* z, Scalar rate, int numbers) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= numbers) return;
//     Scalar3 tz = z[idx];
//     dx[idx] = make_Scalar3(tz.x * rate, tz.y * rate, tz.z * rate);
// }

// void PCG_initDX(Scalar3* dx, const Scalar3* z, Scalar rate, int vertexNum) {
//     int numbers = vertexNum;
//     const unsigned int threadNum = DEFAULT_THREADS;
//     int blockNum = (numbers + threadNum - 1) / threadNum;
//     __PCG_initDX<<<blockNum, threadNum>>>(dx, z, rate, numbers);
// }

// void construct_P(const std::unique_ptr<GeometryManager>& instance, __MATHUTILS__::Matrix3x3S* P,
//                  const BlockHessian& BlockHessian, int vertNum) {
//     int numbers = vertNum;
//     const unsigned int threadNum = DEFAULT_THREADS;
//     int blockNum = (numbers + threadNum - 1) / threadNum;
//     __PCG_mass_P<<<blockNum, threadNum>>>(instance->getCudaVertMass(), P, numbers);

//     numbers = BlockHessian.hostBHDNum[3] * 12;
//     if (numbers > 0) {
//         blockNum = (numbers + threadNum - 1) / threadNum;
//         __PCG_AX12_P<<<blockNum, threadNum>>>(BlockHessian.cudaH12x12, BlockHessian.cudaD4Index, P, numbers);
//     }
//     numbers = BlockHessian.hostBHDNum[2] * 9;
//     if (numbers > 0) {
//         blockNum = (numbers + threadNum - 1) / threadNum;
//         __PCG_AX9_P<<<blockNum, threadNum>>>(BlockHessian.cudaH9x9, BlockHessian.cudaD3Index, P, numbers);
//     }
//     numbers = BlockHessian.hostBHDNum[1] * 6;
//     if (numbers > 0) {
//         blockNum = (numbers + threadNum - 1) / threadNum;
//         __PCG_AX6_P<<<blockNum, threadNum>>>(BlockHessian.cudaH6x6, BlockHessian.cudaD2Index, P, numbers);
//     }
//     numbers = BlockHessian.hostBHDNum[0] * 3;
//     if (numbers > 0) {
//         blockNum = (numbers + threadNum - 1) / threadNum;
//         __PCG_AX3_P<<<blockNum, threadNum>>>(BlockHessian.cudaH3x3, BlockHessian.cudaD1Index, P, numbers);
//     }
//     blockNum = (vertNum + threadNum - 1) / threadNum;
//     //__PCG_inverse_P << <blockNum, threadNum >> > (P, vertNum);
//     __PCG_init_P<<<blockNum, threadNum>>>(instance->getCudaVertMass(), P, vertNum);
// }


// void Solve_PCG_AX_B(const std::unique_ptr<GeometryManager>& instance, const Scalar3* c, Scalar3*
// q,
//                     const BlockHessian& BlockHessian, int vertNum) {
//     int numbers = vertNum;
//     const unsigned int threadNum = DEFAULT_THREADS;
//     int blockNum = (numbers + threadNum - 1) / threadNum;
//     __PCG_Solve_AX_mass_b<<<blockNum, threadNum>>>(instance->getCudaVertMass(), c, q, numbers);
//     // CUDA_SAFE_CALL(cudaDeviceSynchronize());
//     numbers = BlockHessian.hostBHDNum[3];
//     if (numbers > 0) {
//         // unsigned int sharedMsize = sizeof(Scalar) * threadNum;
//         blockNum = (numbers + threadNum - 1) / threadNum;
//         __PCG_Solve_AX12_b<<<blockNum, threadNum>>>(BlockHessian.cudaH12x12,
//         BlockHessian.cudaD4Index, c, q, numbers);
//     }
//     numbers = BlockHessian.hostBHDNum[2];
//     if (numbers > 0) {
//         blockNum = (numbers + threadNum - 1) / threadNum;
//         __PCG_Solve_AX9_b<<<blockNum, threadNum>>>(BlockHessian.cudaH9x9,
//         BlockHessian.cudaD3Index, c, q, numbers);
//     }
//     numbers = BlockHessian.hostBHDNum[1];
//     if (numbers > 0) {
//         blockNum = (numbers + threadNum - 1) / threadNum;
//         __PCG_Solve_AX6_b<<<blockNum, threadNum>>>(BlockHessian.cudaH6x6,
//         BlockHessian.cudaD2Index, c, q, numbers);
//     }
//     numbers = BlockHessian.hostBHDNum[0];
//     if (numbers > 0) {
//         blockNum = (numbers + threadNum - 1) / threadNum;
//         __PCG_Solve_AX3_b<<<blockNum, threadNum>>>(BlockHessian.cudaH3x3,
//         BlockHessian.cudaD1Index, c, q, numbers);
//     }
// }




















__global__ void __PCG_AXALL_P(const __MATHUTILS__::Matrix12x12S* Hessians12,
                              const __MATHUTILS__::Matrix9x9S* Hessians9,
                              const __MATHUTILS__::Matrix6x6S* Hessians6,
                              const __MATHUTILS__::Matrix3x3S* Hessians3, 
                              const uint4* D4Index, const uint3* D3Index, 
                              const uint2* D2Index, const uint32_t* D1Index,
                              __MATHUTILS__::Matrix3x3S* P, 
                              int numbers4, int numbers3,
                              int numbers2, int numbers1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers4 + numbers3 + numbers2 + numbers1) return;

    if (idx < numbers4) {
        int Hid = idx / 12;
        int qid = idx % 12;
        int mid = (qid / 3) * 3;
        int tid = qid % 3;
        Scalar Hval;
        Hval = Hessians12[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);

    } else if (numbers4 <= idx && idx < numbers3 + numbers4) {
        idx -= numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;
        int mid = (qid / 3) * 3;
        int tid = qid % 3;
        Scalar Hval;
        Hval = Hessians9[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);

    } else if (numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2) {
        idx -= numbers3 + numbers4;
        int Hid = idx / 6;
        int qid = idx % 6;
        int mid = (qid / 3) * 3;
        int tid = qid % 3;
        Scalar Hval;
        Hval = Hessians6[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);

    } else {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 3;
        int qid = idx % 3;
        atomicAdd(&(P[D1Index[Hid]].m[0][qid]), Hessians3[Hid].m[0][qid]);
        atomicAdd(&(P[D1Index[Hid]].m[1][qid]), Hessians3[Hid].m[1][qid]);
        atomicAdd(&(P[D1Index[Hid]].m[2][qid]), Hessians3[Hid].m[2][qid]);
    }
}

__global__ void PCG_add_Reduction_delta0(Scalar* squeue, const __MATHUTILS__::Matrix3x3S* P,
                                         const Scalar3* b,
                                         const __MATHUTILS__::Matrix3x3S* constraint, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep[];
    if (idx >= numbers) return;

    // delta0 = rT @ A @ r
    Scalar3 t_b = b[idx];
    __MATHUTILS__::Matrix3x3S t_constraint = constraint[idx];
    Scalar3 filter_b = __MATHUTILS__::__M_v_multiply(t_constraint, t_b);
    Scalar temp = __MATHUTILS__::__v_vec_dot(__MATHUTILS__::__v_M_multiply(filter_b, P[idx]), filter_b);
    // add reduction to squeue
    __MATHUTILS__::perform_reduction(temp, tep, squeue, numbers, idof, blockDim.x, gridDim.x, blockIdx.x);

}

__global__ void PCG_add_Reduction_deltaN0(Scalar* squeue, const __MATHUTILS__::Matrix3x3S* P,
                                          const Scalar3* b, Scalar3* r, Scalar3* c,
                                          const __MATHUTILS__::Matrix3x3S* constraint,
                                          int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep[];
    if (idx >= numbers) return;

    // deltaN = rT @ c
    Scalar3 t_b = b[idx];
    __MATHUTILS__::Matrix3x3S t_constraint = constraint[idx];
    Scalar3 t_r = __MATHUTILS__::__M_v_multiply(t_constraint, __MATHUTILS__::__minus(t_b, r[idx]));
    Scalar3 t_c = __MATHUTILS__::__M_v_multiply(P[idx], t_r);
    t_c = __MATHUTILS__::__M_v_multiply(t_constraint, t_c);
    r[idx] = t_r;
    c[idx] = t_c;
    Scalar temp = __MATHUTILS__::__v_vec_dot(t_r, t_c);

    __MATHUTILS__::perform_reduction(temp, tep, squeue, numbers, idof, blockDim.x, gridDim.x, blockIdx.x);
}

__global__ void PCG_add_Reduction_deltaN(Scalar* squeue, Scalar3* dx, const Scalar3* c, Scalar3* r,
                                         const Scalar3* q, const __MATHUTILS__::Matrix3x3S* P,
                                         Scalar3* s, Scalar alpha, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep[];
    if (idx >= numbers) return;

    // deltaN = rT @ s
    Scalar3 t_c = c[idx];
    Scalar3 t_dx = dx[idx];
    Scalar3 t_r = r[idx];
    Scalar3 t_q = q[idx];
    Scalar3 t_s = s[idx];
    dx[idx] = __MATHUTILS__::__add(t_dx, __MATHUTILS__::__s_vec_multiply(t_c, alpha));
    t_r = __MATHUTILS__::__add(t_r, __MATHUTILS__::__s_vec_multiply(t_q, -alpha));
    r[idx] = t_r;
    t_s = __MATHUTILS__::__M_v_multiply(P[idx], t_r);
    s[idx] = t_s;
    Scalar temp = __MATHUTILS__::__v_vec_dot(t_r, t_s);

    __MATHUTILS__::perform_reduction(temp, tep, squeue, numbers, idof, blockDim.x, gridDim.x, blockIdx.x);

}

__global__ void PCG_add_Reduction_tempSum(Scalar* squeue, const Scalar3* c, Scalar3* q,
                                          const __MATHUTILS__::Matrix3x3S* constraint,
                                          int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep[];
    if (idx >= numbers) return;

    // tempSum = qT @ c
    Scalar3 t_c = c[idx];
    Scalar3 t_q = q[idx];
    __MATHUTILS__::Matrix3x3S t_constraint = constraint[idx];
    t_q = __MATHUTILS__::__M_v_multiply(t_constraint, t_q);
    q[idx] = t_q;
    Scalar temp = __MATHUTILS__::__v_vec_dot(t_q, t_c);

    __MATHUTILS__::perform_reduction(temp, tep, squeue, numbers, idof, blockDim.x, gridDim.x, blockIdx.x);

}

__global__ void PCG_add_Reduction_force(Scalar* squeue, const Scalar3* b, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep[];
    if (idx >= numbers) return;

    // calculate ||b|| norm as init residual
    Scalar3 t_b = b[idx];
    Scalar temp = __MATHUTILS__::__norm(t_b);

    __MATHUTILS__::perform_reduction(temp, tep, squeue, numbers, idof, blockDim.x, gridDim.x, blockIdx.x);
}

__global__ void __PCG_FinalStep_UpdateC(const __MATHUTILS__::Matrix3x3S* constraints,
                                        const Scalar3* s, Scalar3* c, Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    // update search diretion c
    Scalar3 tempc = __MATHUTILS__::__add(s[idx], __MATHUTILS__::__s_vec_multiply(c[idx], rate));
    c[idx] = __MATHUTILS__::__M_v_multiply(constraints[idx], tempc);
}



__global__ void __PCG_Solve_AXALL_b2(
    const __MATHUTILS__::Matrix12x12S* Hessians12, const __MATHUTILS__::Matrix9x9S* Hessians9,
    const __MATHUTILS__::Matrix6x6S* Hessians6, const __MATHUTILS__::Matrix3x3S* Hessians3,
    const uint4* D4Index, const uint3* D3Index, const uint2* D2Index, const uint32_t* D1Index,
    const Scalar3* c, Scalar3* q, int numbers4, int numbers3, int numbers2, int numbers1,
    int offset4, int offset3, int offset2) {

    // calculate A = M + dt^2 * K, q = A @ c
    if (blockIdx.x < offset4) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numbers4) return;
        __shared__ int offset;
        int Hid = idx / 144;
        int MRid = (idx % 144) / 12;
        int MCid = (idx % 144) % 12;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 12;

        Scalar rdata =
            Hessians12[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (12 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 12) / 12;
        int landidx = (threadIdx.x - offset) % 12;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 12; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } else if (blockIdx.x >= offset4 && blockIdx.x < offset4 + offset3) {
        int idx = (blockIdx.x - offset4) * blockDim.x + threadIdx.x;
        if (idx >= numbers3) return;
        __shared__ int offset;
        int Hid = idx / 81;
        int MRid = (idx % 81) / 9;
        int MCid = (idx % 81) % 9;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 9;

        Scalar rdata =
            Hessians9[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (9 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 9) / 9;
        int landidx = (threadIdx.x - offset) % 9;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);  // a bit-mask
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 9; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } else if (blockIdx.x >= offset4 + offset3 && blockIdx.x < offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3) * blockDim.x + threadIdx.x;
        if (idx >= numbers2) return;
        __shared__ int offset;
        int Hid = idx / 36;
        int MRid = (idx % 36) / 6;
        int MCid = (idx % 36) % 6;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 6;

        Scalar rdata =
            Hessians6[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (6 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 6) / 6;
        int landidx = (threadIdx.x - offset) % 6;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 6; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } else if (blockIdx.x >= offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3 - offset2) * blockDim.x + threadIdx.x;
        if (idx >= numbers1) return;
        __MATHUTILS__::Matrix3x3S H = Hessians3[idx];
        Scalar3 tempC, tempQ;

        tempC.x = c[D1Index[idx]].x;
        tempC.y = c[D1Index[idx]].y;
        tempC.z = c[D1Index[idx]].z;

        tempQ = __MATHUTILS__::__M_v_multiply(H, tempC);

        atomicAdd(&(q[D1Index[idx]].x), tempQ.x);
        atomicAdd(&(q[D1Index[idx]].y), tempQ.y);
        atomicAdd(&(q[D1Index[idx]].z), tempQ.z);
    }
}

__global__ void __PCG_Solve_AX_mass_b(const Scalar* _masses, const Scalar3* c, Scalar3* q,
                                      int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    // q = M @ c
    Scalar3 tempQ = __MATHUTILS__::__s_vec_multiply(c[idx], _masses[idx]);
    q[idx] = tempQ;
}


__global__ void __PCG_mass_P(const Scalar* _masses, __MATHUTILS__::Matrix3x3S* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    // P = M
    __MATHUTILS__::__init_Mat3x3(P[idx], 0);
    Scalar mass = _masses[idx];
    P[idx].m[0][0] = mass;
    P[idx].m[1][1] = mass;
    P[idx].m[2][2] = mass;
}


__global__ void __PCG_inverse_P(__MATHUTILS__::Matrix3x3S* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    // P = P^-1
    __MATHUTILS__::Matrix3x3S PInverse;
    __MATHUTILS__::__Inverse(P[idx], PInverse);
    P[idx] = PInverse;
}

Scalar My_PCG_add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance,
                                      Scalar* _PCGSqueue, Scalar3* _PCGb, Scalar3* _PCGr,
                                      Scalar3* _PCGc, Scalar3* _PCGq, Scalar3* _PCGs,
                                      Scalar3* _PCGz, Scalar3* _PCGdx,
                                      __MATHUTILS__::Matrix3x3S* _PCGP, int vertexNum,
                                      Scalar alpha = 1) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);
    switch (type) {
        case 0:
            PCG_add_Reduction_force<<<blockNum, threadNum, sharedMsize>>>(_PCGSqueue, _PCGb,
                                                                          numbers);
            break;
        case 1:
            PCG_add_Reduction_delta0<<<blockNum, threadNum, sharedMsize>>>(
                _PCGSqueue, _PCGP, _PCGb, instance->getCudaConstraintsMat(),
                numbers);  // compute delta_0
            break;
        case 2:
            PCG_add_Reduction_deltaN0<<<blockNum, threadNum, sharedMsize>>>(
                _PCGSqueue, _PCGP, _PCGb, _PCGr, _PCGc, instance->getCudaConstraintsMat(),
                numbers);  // compute delta_N
            break;
        case 3:
            PCG_add_Reduction_tempSum<<<blockNum, threadNum, sharedMsize>>>(
                _PCGSqueue, _PCGc, _PCGq, instance->getCudaConstraintsMat(),
                numbers);  // compute tempSum
            break;
        case 4:
            PCG_add_Reduction_deltaN<<<blockNum, threadNum, sharedMsize>>>(
                _PCGSqueue, _PCGdx, _PCGc, _PCGr, _PCGq, _PCGP, _PCGs, alpha,
                numbers);  // update delta_N and delta_x
            break;
    }

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::__add_reduction<<<blockNum, threadNum, sharedMsize>>>(_PCGSqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar result;
    cudaMemcpy(&result, _PCGSqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    return result;
}

void Solve_PCG_AX_B2(const std::unique_ptr<GeometryManager>& instance, const Scalar3* c, Scalar3* q,
                     const BlockHessian& BlockHessian, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    // q = M @ c
    __PCG_Solve_AX_mass_b<<<blockNum, threadNum>>>(instance->getCudaVertMass(), c, q, numbers);

    // q += K @ c
    int offset4 = (BlockHessian.hostBHDNum[3] * 144 + threadNum - 1) / threadNum;
    int offset3 = (BlockHessian.hostBHDNum[2] * 81 + threadNum - 1) / threadNum;
    int offset2 = (BlockHessian.hostBHDNum[1] * 36 + threadNum - 1) / threadNum;
    int offset1 = (BlockHessian.hostBHDNum[0] + threadNum - 1) / threadNum;
    blockNum = offset1 + offset2 + offset3 + offset4;
    __PCG_Solve_AXALL_b2<<<blockNum, threadNum>>>(
        BlockHessian.cudaH12x12, BlockHessian.cudaH9x9, BlockHessian.cudaH6x6,
        BlockHessian.cudaH3x3, BlockHessian.cudaD4Index, BlockHessian.cudaD3Index,
        BlockHessian.cudaD2Index, BlockHessian.cudaD1Index, c, q, BlockHessian.hostBHDNum[3] * 144,
        BlockHessian.hostBHDNum[2] * 81, BlockHessian.hostBHDNum[1] * 36,
        BlockHessian.hostBHDNum[0], offset4, offset3, offset2);
}

void construct_P2(const std::unique_ptr<GeometryManager>& instance, __MATHUTILS__::Matrix3x3S* P,
                  const BlockHessian& BlockHessian, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    // init diagonal matrix Precond P and init mass P = M
    __PCG_mass_P<<<blockNum, threadNum>>>(instance->getCudaVertMass(), P, numbers);

    numbers = BlockHessian.hostBHDNum[3] * 12 + BlockHessian.hostBHDNum[2] * 9 +
              BlockHessian.hostBHDNum[1] * 6 + BlockHessian.hostBHDNum[0] * 3;
    blockNum = (numbers + threadNum - 1) / threadNum;
    // P = M+h^2*K
    __PCG_AXALL_P<<<blockNum, threadNum>>>(
        BlockHessian.cudaH12x12, BlockHessian.cudaH9x9, BlockHessian.cudaH6x6,
        BlockHessian.cudaH3x3, BlockHessian.cudaD4Index, BlockHessian.cudaD3Index,
        BlockHessian.cudaD2Index, BlockHessian.cudaD1Index, P, BlockHessian.hostBHDNum[3] * 12,
        BlockHessian.hostBHDNum[2] * 9, BlockHessian.hostBHDNum[1] * 6,
        BlockHessian.hostBHDNum[0] * 3);

    // P = P^-1 = (M+h^2*K)^-1
    blockNum = (vertNum + threadNum - 1) / threadNum;
    __PCG_inverse_P<<<blockNum, threadNum>>>(P, vertNum);

}

void PCG_FinalStep_UpdateC(const std::unique_ptr<GeometryManager>& instance, Scalar3* c,
                           const Scalar3* s, const Scalar& rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_FinalStep_UpdateC<<<blockNum, threadNum>>>(instance->getCudaConstraintsMat(), s, c, rate,
                                                     numbers);
}


int PCGSolver::PCG_Process(std::unique_ptr<GeometryManager>& instance,
                           const BlockHessian& BlockHessian, Scalar3* _mvDir, int vertexNum,
                           int tetrahedraNum, Scalar IPC_dt, Scalar meanVolume, Scalar threshold) {
    
    cudaPCGdx = instance->getCudaMoveDir();
    cudaPCGb = instance->getCudaFb();

    // calculate preconditioner P
    construct_P2(instance, cudaPCGP, BlockHessian, vertexNum);

    // calculate delta0=r^T@z deltaN=r^T@c 
    Scalar deltaN = 0;
    Scalar delta0 = 0;
    Scalar deltaO = 0;
    CUDA_SAFE_CALL(cudaMemset(cudaPCGdx, 0x0, vertexNum * sizeof(Scalar3)));
    CUDA_SAFE_CALL(cudaMemset(cudaPCGr, 0x0, vertexNum * sizeof(Scalar3)));
    delta0 = My_PCG_add_Reduction_Algorithm(1, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
                                            cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
                                            cudaPCGP, vertexNum);
    deltaN = My_PCG_add_Reduction_Algorithm(2, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
                                            cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
                                            cudaPCGP, vertexNum);

    Scalar errorRate = threshold; // threadhold = 1e-3

    int cgCounts = 0;
    while (cgCounts < 30000 && deltaN > errorRate * delta0) {
        cgCounts++;
        // q = A@c
        Solve_PCG_AX_B2(instance, cudaPCGc, cudaPCGq, BlockHessian, vertexNum);
        // tempsum = p^T @ K @ p
        Scalar tempSum = My_PCG_add_Reduction_Algorithm(3, instance, cudaPCGSqueue, cudaPCGb,
                                                        cudaPCGr, cudaPCGc, cudaPCGq, cudaPCGs,
                                                        cudaPCGz, cudaPCGdx, cudaPCGP, vertexNum);
        Scalar alpha = deltaN / tempSum; // alpha = deltaN / tempSum
        deltaO = deltaN;
        deltaN = My_PCG_add_Reduction_Algorithm(4, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
                                                cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
                                                cudaPCGP, vertexNum, alpha); // update deltax deltaN
        Scalar rate = deltaN / deltaO;
        PCG_FinalStep_UpdateC(instance, cudaPCGc, cudaPCGs, rate, vertexNum); // update search direction c
    }
    _mvDir = cudaPCGdx;
    printf("cg counts = %d\n", cgCounts);
    if (cgCounts == 0) {
        printf("indefinite exit\n");
        // exit(0);
    }
    return cgCounts;
}


// int PCGSolver::PCG_Process(std::unique_ptr<GeometryManager>& instance,
//                            const BlockHessian& BlockHessian, Scalar3* _mvDir, int vertexNum,
//                            int tetrahedraNum, Scalar IPC_dt, Scalar meanVolume, Scalar threshold) {
    
//     cudaPCGdx = instance->getCudaMoveDir();
//     cudaPCGb = instance->getCudaFb();

//     construct_P2(instance, cudaPCGP, BlockHessian, vertexNum);
//     Scalar deltaN = 0;
//     Scalar delta0 = 0;
//     Scalar deltaO = 0;
//     CUDA_SAFE_CALL(cudaMemset(cudaPCGdx, 0x0, vertexNum * sizeof(Scalar3)));
//     CUDA_SAFE_CALL(cudaMemset(cudaPCGr, 0x0, vertexNum * sizeof(Scalar3)));
//     delta0 = My_PCG_add_Reduction_Algorithm(1, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
//                                             cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
//                                             cudaPCGP, vertexNum);
//     // Solve_PCG_AX_B2(instance, z, r, BlockHessian, vertexNum);
//     deltaN = My_PCG_add_Reduction_Algorithm(2, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
//                                             cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
//                                             cudaPCGP, vertexNum);
//     // std::cout << "gpu  delta0:   " << delta0 << "      deltaN:   " << deltaN
//     // << std::endl; Scalar errorRate = std::min(1e-8 * 0.5 * IPC_dt /
//     // std::pow(meanVolume, 1), 1e-4);
//     Scalar errorRate = threshold /* * IPC_dt * IPC_dt*/;
//     // printf("cg error Rate:   %f        meanVolume: %f\n", errorRate,
//     // meanVolume);
//     int cgCounts = 0;
//     while (cgCounts < 30000 && deltaN > errorRate * delta0) {
//         cgCounts++;
//         // std::cout << "delta0:   " << delta0 << "      deltaN:   " << deltaN
//         // << "      iteration_counts:      " << cgCounts << std::endl;
//         // CUDA_SAFE_CALL(cudaMemset(q, 0, vertexNum * sizeof(Scalar3)));
//         Solve_PCG_AX_B2(instance, cudaPCGc, cudaPCGq, BlockHessian, vertexNum);
//         Scalar tempSum = My_PCG_add_Reduction_Algorithm(3, instance, cudaPCGSqueue, cudaPCGb,
//                                                         cudaPCGr, cudaPCGc, cudaPCGq, cudaPCGs,
//                                                         cudaPCGz, cudaPCGdx, cudaPCGP, vertexNum);
//         Scalar alpha = deltaN / tempSum;
//         deltaO = deltaN;
//         // deltaN = 0;
//         // CUDA_SAFE_CALL(cudaMemset(s, 0, vertexNum * sizeof(Scalar3)));
//         deltaN = My_PCG_add_Reduction_Algorithm(4, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
//                                                 cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
//                                                 cudaPCGP, vertexNum, alpha);
//         Scalar rate = deltaN / deltaO;
//         PCG_FinalStep_UpdateC(instance, cudaPCGc, cudaPCGs, rate, vertexNum);
//         // cudaDeviceSynchronize();
//     }
//     _mvDir = cudaPCGdx;
//     // CUDA_SAFE_CALL(cudaMemcpy(z, _mvDir, vertexNum * sizeof(Scalar3),
//     // cudaMemcpyDeviceToDevice));
//     printf("cg counts = %d\n", cgCounts);
//     if (cgCounts == 0) {
//         printf("indefinite exit\n");
//         // exit(0);
//     }
//     return cgCounts;
// }

















////////////////////////////////////////////////////////
// MAS Preconditioner PCG
////////////////////////////////////////////////////////

__global__ void PCG_vdv_Reduction(Scalar* squeue, const Scalar3* a, const Scalar3* b, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep[];
    if (idx >= numbers) return;
    // int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    // Scalar3 t_b = b[idx];

    Scalar temp = __MATHUTILS__::__v_vec_dot(
        a[idx],
        b[idx]);  //__MATHUTILS__::__norm(t_b);//__MATHUTILS__::__mabs(t_b.x) +
                  //__MATHUTILS__::__mabs(t_b.y) + __MATHUTILS__::__mabs(t_b.z);

    __MATHUTILS__::perform_reduction(temp, tep, squeue, numbers, idof, blockDim.x, gridDim.x, blockIdx.x);

}

Scalar My_PCG_General_v_v_Reduction_Algorithm(std::unique_ptr<GeometryManager>& instance,
                                              Scalar* _PCGSqueue, Scalar3* A, Scalar3* B,
                                              int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum>>>(_PCGSqueue, A, B, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::__add_reduction<<<blockNum, threadNum, sharedMsize>>>(_PCGSqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar result;
    cudaMemcpy(&result, _PCGSqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    return result;
}

__global__ void __PCG_Update_Dx_R(const Scalar3* c, Scalar3* dx, const Scalar3* q, Scalar3* r,
                                  Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    dx[idx] = __MATHUTILS__::__add(dx[idx], __MATHUTILS__::__s_vec_multiply(c[idx], rate));
    r[idx] = __MATHUTILS__::__add(r[idx], __MATHUTILS__::__s_vec_multiply(q[idx], -rate));
}

void PCG_Update_Dx_R(const Scalar3* c, Scalar3* dx, const Scalar3* q, Scalar3* r,
                     const Scalar& rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Update_Dx_R<<<blockNum, threadNum>>>(c, dx, q, r, rate, numbers);
}

__global__ void __PCG_constraintFilter(const __MATHUTILS__::Matrix3x3S* constraints,
                                       const Scalar3* input, Scalar3* output, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    output[idx] = __MATHUTILS__::__M_v_multiply(constraints[idx], input[idx]);
}

void PCG_constraintFilter(const std::unique_ptr<GeometryManager>& instance, const Scalar3* input,
                          Scalar3* output, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_constraintFilter<<<blockNum, threadNum>>>(instance->getCudaConstraintsMat(), input,
                                                    output, numbers);
}

int PCGSolver::MASPCG_Process(std::unique_ptr<GeometryManager>& instance,
                              const BlockHessian& BlockHessian, Scalar3* _mvDir, int vertexNum,
                              int tetrahedraNum, Scalar IPC_dt, Scalar meanVolume, int cpNum,
                              Scalar threshold) {
    cudaPCGdx = instance->getCudaMoveDir();
    cudaPCGb = instance->getCudaFb();

    MP.setPreconditioner(BlockHessian, instance->getCudaVertMass(), cpNum);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    Scalar deltaN = 0;
    Scalar delta0 = 0;
    Scalar deltaO = 0;
    // PCG_initDX(dx, z, 0.5, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(cudaPCGdx, 0x0, vertexNum * sizeof(Scalar3)));
    CUDA_SAFE_CALL(cudaMemset(cudaPCGr, 0x0, vertexNum * sizeof(Scalar3)));

    PCG_constraintFilter(instance, cudaPCGb, cudaPCGFilterTempVec3, vertexNum);

    MP.preconditioning(cudaPCGFilterTempVec3, cudaPCGPrecondTempVec3);
    // Solve_PCG_Preconditioning24(instance, P24, P, restP, filterTempVec3,
    // preconditionTempVec3, vertexNum);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    delta0 = My_PCG_General_v_v_Reduction_Algorithm(instance, cudaPCGSqueue, cudaPCGFilterTempVec3,
                                                    cudaPCGPrecondTempVec3, vertexNum);

    CUDA_SAFE_CALL(cudaMemcpy(cudaPCGr, cudaPCGFilterTempVec3, vertexNum * sizeof(Scalar3),
                              cudaMemcpyDeviceToDevice));

    PCG_constraintFilter(instance, cudaPCGPrecondTempVec3, cudaPCGFilterTempVec3, vertexNum);

    CUDA_SAFE_CALL(cudaMemcpy(cudaPCGc, cudaPCGFilterTempVec3, vertexNum * sizeof(Scalar3),
                              cudaMemcpyDeviceToDevice));

    deltaN = My_PCG_General_v_v_Reduction_Algorithm(instance, cudaPCGSqueue, cudaPCGr, cudaPCGc,
                                                    vertexNum);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // delta0 = My_PCG_add_Reduction_Algorithm(1, instance, instance,
    // vertexNum); Solve_PCG_AX_B2(instance, z, r, BlockHessian, vertexNum);
    // deltaN = My_PCG_add_Reduction_Algorithm(2, instance, instance,
    // vertexNum); std::cout << "gpu  delta0:   " << delta0 << "      deltaN: "
    // << deltaN << std::endl; Scalar errorRate = min(1e-8 * 0.5 * IPC_dt /
    // pow(meanVolume, 1), 1e-4);
    Scalar errorRate = threshold /* * IPC_dt * IPC_dt*/;
    // printf("cg error Rate:   %f        meanVolume: %f\n", errorRate,
    // meanVolume);
    int cgCounts = 0;
    while (cgCounts < 3000 && deltaN > errorRate * delta0) {
        cgCounts++;
        // std::cout << "delta0:   " << delta0 << "      deltaN:   " << deltaN
        // << "      iteration_counts:      " << cgCounts << std::endl;
        // CUDA_SAFE_CALL(cudaMemset(q, 0, vertexNum * sizeof(Scalar3)));
        Solve_PCG_AX_B2(instance, cudaPCGc, cudaPCGq, BlockHessian, vertexNum);
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
        Scalar tempSum = My_PCG_add_Reduction_Algorithm(3, instance, cudaPCGSqueue, cudaPCGb,
                                                        cudaPCGr, cudaPCGc, cudaPCGq, cudaPCGs,
                                                        cudaPCGz, cudaPCGdx, cudaPCGP, vertexNum);
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
        Scalar alpha = deltaN / tempSum;
        deltaO = deltaN;
        // deltaN = 0;
        // CUDA_SAFE_CALL(cudaMemset(s, 0, vertexNum * sizeof(Scalar3)));
        // deltaN = My_PCG_add_Reduction_Algorithm(4, instance, instance,
        // vertexNum, alpha);
        PCG_Update_Dx_R(cudaPCGc, cudaPCGdx, cudaPCGq, cudaPCGr, alpha, vertexNum);
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
        MP.preconditioning(cudaPCGr, cudaPCGs);
        // Solve_PCG_Preconditioning24(instance, P24, P, restP, r, s,
        // vertexNum); CUDA_SAFE_CALL(cudaDeviceSynchronize());
        deltaN = My_PCG_General_v_v_Reduction_Algorithm(instance, cudaPCGSqueue, cudaPCGr, cudaPCGs,
                                                        vertexNum);
        // CUDA_SAFE_CALL(cudaDeviceSynchronize());
        Scalar rate = deltaN / deltaO;
        PCG_FinalStep_UpdateC(instance, cudaPCGc, cudaPCGs, rate, vertexNum);
        // cudaDeviceSynchronize();
        // std::cout << "gpu  delta0:   " << delta0 << "      deltaN:   " <<
        // deltaN << std::endl;
    }

    _mvDir = cudaPCGdx;
    // CUDA_SAFE_CALL(cudaMemcpy(z, _mvDir, vertexNum * sizeof(Scalar3),
    // cudaMemcpyDeviceToDevice)); printf("cg counts = %d\n", cgCounts);
    if (cgCounts == 0) {
        printf("indefinite exit\n");
        // exit(0);
    }
    return cgCounts;
}

void PCGSolver::CUDA_MALLOC_PCGSOLVER(const int& vertexNum, const int& tetrahedraNum) {
    CUDAMallocSafe(cudaPCGSqueue, __MATHUTILS__::__m_max(vertexNum, tetrahedraNum));
    CUDAMallocSafe(cudaPCGb, vertexNum);
    CUDAMallocSafe(cudaPCGP, vertexNum);
    CUDAMallocSafe(cudaPCGr, vertexNum);
    CUDAMallocSafe(cudaPCGc, vertexNum);
    CUDAMallocSafe(cudaPCGq, vertexNum);
    CUDAMallocSafe(cudaPCGs, vertexNum);
    CUDAMallocSafe(cudaPCGz, vertexNum);
    CUDAMallocSafe(cudaPCGdx, vertexNum);
    CUDAMallocSafe(cudaPCGTempDx, vertexNum);
    CUDAMallocSafe(cudaPCGPrecondTempVec3, vertexNum);
    CUDAMallocSafe(cudaPCGFilterTempVec3, vertexNum);
}

void PCGSolver::CUDA_FREE_PCGSOLVER() {
    CUDAFreeSafe(cudaPCGSqueue);
    CUDAFreeSafe(cudaPCGb);
    CUDAFreeSafe(cudaPCGP);
    CUDAFreeSafe(cudaPCGr);
    CUDAFreeSafe(cudaPCGc);
    CUDAFreeSafe(cudaPCGq);
    CUDAFreeSafe(cudaPCGs);
    CUDAFreeSafe(cudaPCGz);
    CUDAFreeSafe(cudaPCGdx);
    CUDAFreeSafe(cudaPCGTempDx);
    CUDAFreeSafe(cudaPCGFilterTempVec3);
    CUDAFreeSafe(cudaPCGPrecondTempVec3);

    if (PrecondType == 1) {
        MP.CUDA_MALLOC_MAS_PRECONDITIONER();
    }
}
