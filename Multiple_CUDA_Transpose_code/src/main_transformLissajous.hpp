#pragma once

#include "public_functions.hpp"
#include "cuda_kernels.cuh"

class GAS_Transform_Lissajous : public GAS_SubSolver {

public:

protected:

    explicit GAS_Transform_Lissajous(const SIM_DataFactory* factory);
    virtual ~GAS_Transform_Lissajous() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

    void transformPositions(SIM_Time time, int numPoints);

private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_Transform_Lissajous,
                        GAS_SubSolver,
                        "gas transform Lissajous",
                        getDopDescription());

};
