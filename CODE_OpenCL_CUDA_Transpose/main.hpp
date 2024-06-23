// #pragma once

// #include <iostream>
// #include <vector>
// #include <GAS/GAS_SubSolver.h>
// #include <GAS/GAS_Utils.h>

// #include <CL/cl.h>
// #include <CL/cl_gl.h>
// #include <CE/CE_API.h>
// #include <CE/CE_Context.h>
// #include <CE/CE_MemoryPool.h>

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// #include <cuda_gl_interop.h>


// class GAS_OpenCL_Lissajous : public GAS_SubSolver {

// public:

// protected:

//     explicit GAS_OpenCL_Lissajous(const SIM_DataFactory* factory);
//     virtual ~GAS_OpenCL_Lissajous() override;

//     bool solveGasSubclass(SIM_Engine& engine,
//                         SIM_Object* object,
//                         SIM_Time time,
//                         SIM_Time timestep) override;

// private:

//     // void accessOpenCLBuffer(const SIM_Geometry *geo, const GU_Detail *gdp, float time);
//     // void updatePositionCUDA(cl::Buffer clBuffer, GA_Size numEntries);

//     static const SIM_DopDescription* getDopDescription();

//     DECLARE_STANDARD_GETCASTTOTYPE();
//     DECLARE_DATAFACTORY(GAS_OpenCL_Lissajous,
//                         GAS_SubSolver,
//                         "gas opencl Lissajous",
//                         getDopDescription());

// };












#pragma once

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

/// A simple field manipulation class that will add fields
/// together.  
class SIM_GasAdd : public GAS_SubSolver
{
public:
    /// These macros are used to create the accessors
    /// getFieldDstName and getFieldSrcName functions we'll use
    /// to access our data options.
    GET_DATA_FUNC_S(GAS_NAME_FIELDDEST, FieldDstName);
    GET_DATA_FUNC_S(GAS_NAME_FIELDSOURCE, FieldSrcName);
protected:
    explicit             SIM_GasAdd(const SIM_DataFactory *factory);
                        ~SIM_GasAdd() override;
    /// Used to determine if the field is complicated enough to justify
    /// the overhead of multithreading.
    bool                 shouldMultiThread(const SIM_RawField *field) const 
                         { return field->field()->numTiles() > 1; }
    /// The overloaded callback that GAS_SubSolver will invoke to
    /// perform our actual computation.  We are giving a single object
    /// at a time to work on.
    bool                 solveGasSubclass(SIM_Engine &engine,
                                SIM_Object *obj,
                                SIM_Time time,
                                SIM_Time timestep) override;
    /// Add two raw fields together.  Use UT_ThreadedAlgorithm's macros
    /// to define the addFields method that will invoke addFieldPartial()
    /// on each worker thread.
    THREADED_METHOD2(SIM_GasAdd, shouldMultiThread(dst),
                     addFields,
                     SIM_RawField *, dst,
                     const SIM_RawField *, src);
    void         addFieldsPartial(SIM_RawField *dst, const SIM_RawField *src, const UT_JobInfo &info);
    
private:
    /// We define this to be a DOP_Auto node which means we do not
    /// need to implement a DOP_Node derivative for this data.  Instead,
    /// this description is used to define the interface.
    static const SIM_DopDescription     *getDopDescription();
    /// These macros are necessary to bind our node to the factory and
    /// ensure useful constants like BaseClass are defined.
    DECLARE_STANDARD_GETCASTTOTYPE();
    DECLARE_DATAFACTORY(SIM_GasAdd,
                        GAS_SubSolver,
                        "Gas Add",
                        getDopDescription());
};
