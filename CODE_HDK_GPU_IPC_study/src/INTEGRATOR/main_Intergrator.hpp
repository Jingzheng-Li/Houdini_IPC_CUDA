#pragma once

#include "UTILS/HoudiniUtils.hpp"
#include "UTILS/GeometryManager.hpp"
#include "ImplicitIntergrator.cuh"

class GAS_CUDA_Intergrator : public GAS_SubSolver {

public:

protected:

    explicit GAS_CUDA_Intergrator(const SIM_DataFactory* factory);
    virtual ~GAS_CUDA_Intergrator() override;

    bool solveGasSubclass(SIM_Engine& engine,
                        SIM_Object* object,
                        SIM_Time time,
                        SIM_Time timestep) override;

    void transferDynamicCollisionToCUDA(SIM_Object* object);

protected:
    void gas_IPC_Solver();

private:

    static const SIM_DopDescription* getDopDescription();

    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(GAS_CUDA_Intergrator,
                        GAS_SubSolver,
                        "gas cuda intergrator",
                        getDopDescription());





protected:
    void debugPrint();


};
