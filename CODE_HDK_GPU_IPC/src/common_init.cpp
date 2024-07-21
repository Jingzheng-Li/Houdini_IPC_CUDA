
#include <UT/UT_DSOVersion.h>

#include "RWBuffer/main_readBuffer.hpp"
#include "RWBuffer/main_writeBuffer.hpp"
#include "LBVH/main_LBVH.hpp"
#include "ACCD/main_ACCD.hpp"
#include "IPC/main_GIPC.hpp"

#include "UTILS/GeometryManager.hpp"



void initializeSIM(void *) {
    IMPLEMENT_DATAFACTORY(GAS_Read_Buffer);
    IMPLEMENT_DATAFACTORY(GAS_Write_Buffer);
    // IMPLEMENT_DATAFACTORY(GAS_CUDA_LBVH);
    // IMPLEMENT_DATAFACTORY(GAS_CUDA_ACCD);
    IMPLEMENT_DATAFACTORY(GAS_CUDA_GIPC);
}