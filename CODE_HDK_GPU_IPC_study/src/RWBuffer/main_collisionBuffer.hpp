
#pragma once

#include "UTILS/HoudiniUtils.hpp"
#include "UTILS/GeometryManager.hpp"

class GAS_Collision_Buffer : public GAS_SubSolver {

public:

protected:

    explicit GAS_Collision_Buffer(const SIM_DataFactory* factory);
    virtual ~GAS_Collision_Buffer() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

    void transferPTAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp);
    void transferDTAttribTOHoudini(SIM_GeometryCopy *geo, GU_Detail *gdp);


private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_Collision_Buffer,
                        GAS_SubSolver,
                        "gas collision buffer",
                        getDopDescription());

};
