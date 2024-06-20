#pragma once

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "public_functions.hpp"

class GAS_Push_Buffer : public GAS_SubSolver {

public:

protected:

    explicit GAS_Push_Buffer(const SIM_DataFactory* factory);
    virtual ~GAS_Push_Buffer() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_Push_Buffer,
                        GAS_SubSolver,
                        "gas push buffer",
                        getDopDescription());

};

