
#include "main_readBuffer.hpp"
#include "main_writeBuffer.hpp"
#include "main_transformLissajous.hpp"

#include <UT/UT_DSOVersion.h>

void initializeSIM(void *) {
    IMPLEMENT_DATAFACTORY(GAS_Read_Buffer);
    IMPLEMENT_DATAFACTORY(GAS_Write_Buffer);
    IMPLEMENT_DATAFACTORY(GAS_Transform_Lissajous);

}