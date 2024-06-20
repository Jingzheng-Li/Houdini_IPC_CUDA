#pragma once

#include "public_functions.hpp"

class GAS_Read_Buffer : public GAS_SubSolver {

public:

protected:

    explicit GAS_Read_Buffer(const SIM_DataFactory* factory);
    virtual ~GAS_Read_Buffer() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

    void transferPositionsTOCUDA(const SIM_Geometry *geo, const GU_Detail *gdp);


private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_Read_Buffer,
                        GAS_SubSolver,
                        "gas read buffer",
                        getDopDescription());

};

