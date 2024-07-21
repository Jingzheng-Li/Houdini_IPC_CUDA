
#pragma once

#include "UTILS/HoudiniUtils.hpp"

class GAS_CUDA_LBVH : public GAS_SubSolver {

public:

protected:

    explicit GAS_CUDA_LBVH(const SIM_DataFactory* factory);
    virtual ~GAS_CUDA_LBVH() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

protected:


private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_CUDA_LBVH,
                        GAS_SubSolver,
                        "gas cuda lbvh",
                        getDopDescription());

};