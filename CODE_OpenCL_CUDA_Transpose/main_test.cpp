
#include "main_test.hpp"

#include <SIM/SIM_GeometryCopy.h>
#include <SIM/SIM_Geometry.h>
#include <SIM/SIM_OptionsUser.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_ScalarField.h>
#include <SIM/SIM_DataFilter.h>
#include <SIM/SIM_Engine.h>
#include <SIM/SIM_VectorField.h>
#include <SIM/SIM_Time.h>
#include <SIM/SIM_Solver.h>
#include <SIM/SIM_DopDescription.h>
#include <GEO/GEO_Primitive.h>
#include <GEO/GEO_PrimVDB.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_VoxelArray.h>
#include <UT/UT_DSOVersion.h>
#include <GA/GA_Handle.h>
#include <GU/GU_Detail.h>
#include <DOP/DOP_Node.h>
#include <DOP/DOP_Engine.h>


void initializeSIM(void *) {
    IMPLEMENT_DATAFACTORY(GAS_Push_Buffer);
}

const SIM_DopDescription* GAS_Push_Buffer::getDopDescription() {
    static PRM_Template templateList[] = {
        PRM_Template()
    };

    static SIM_DopDescription dopDescription(
        true,
        "gas_push_buffer", // internal name of the dop
        "Gas Push Buffer", // label of the dop
        "GasPushBuffer", // template list for generating the dop
        classname(),
        templateList);

    setGasDescription(dopDescription);

    return &dopDescription;
}

GAS_Push_Buffer::GAS_Push_Buffer(const SIM_DataFactory* factory) : BaseClass(factory) {}

GAS_Push_Buffer::~GAS_Push_Buffer() {}


bool GAS_Push_Buffer::solveGasSubclass(SIM_Engine& engine,
                                        SIM_Object* object,
                                        SIM_Time time,
                                        SIM_Time timestep) {

	const SIM_Geometry *geo = object->getGeometry();
    GU_DetailHandleAutoReadLock readlock(geo->getGeometry());
    const GU_Detail *gdp = readlock.getGdp();

    // accessOpenCLBuffer(geo, gdp, static_cast<float>(time));

    return true;
}

