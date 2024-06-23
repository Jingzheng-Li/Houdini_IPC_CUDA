
#include <UT/UT_DSOVersion.h>

#include "RWBuffer/main_readBuffer.hpp"
#include "RWBuffer/main_writeBuffer.hpp"

#include "UTILS/GeometryManager.hpp"

std::unique_ptr<GeometryManager> GeometryManager::instance = nullptr;


void initializeSIM(void *) {
    IMPLEMENT_DATAFACTORY(GAS_Read_Buffer);
    IMPLEMENT_DATAFACTORY(GAS_Write_Buffer);

}