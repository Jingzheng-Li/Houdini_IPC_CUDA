#pragma once

#include "UTILS/HoudiniUtils.hpp"
#include "GIPC.cuh"

class GAS_CUDA_GIPC : public GAS_SubSolver {

public:

protected:

    explicit GAS_CUDA_GIPC(const SIM_DataFactory* factory);
    virtual ~GAS_CUDA_GIPC() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

protected:
    void gas_IPC_Solver();

private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_CUDA_GIPC,
                        GAS_SubSolver,
                        "gas cuda gipc",
                        getDopDescription());

};