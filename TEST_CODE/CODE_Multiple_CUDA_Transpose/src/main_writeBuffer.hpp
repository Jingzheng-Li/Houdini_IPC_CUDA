#pragma once

#include "public_functions.hpp"

class GAS_Write_Buffer : public GAS_SubSolver {

public:

protected:

    explicit GAS_Write_Buffer(const SIM_DataFactory* factory);
    virtual ~GAS_Write_Buffer() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

    void transferPositionsTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp);

private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_Write_Buffer,
                        GAS_SubSolver,
                        "gas write buffer",
                        getDopDescription());

};

