
// // 结论：可以从Houdini的OpenCL中获取GPU buffer,但是缺少从OpenCL到CUDA的映射。
// // 所以最好不要混合GAS OpenCL和CUDA共同使用。
// // 后续：如果要做GPU IPC,可以直接搭建场景，然后传输到CUDA中每个节点自己编写。



// #include "main.hpp"

// #include <SIM/SIM_GeometryCopy.h>
// #include <SIM/SIM_Geometry.h>
// #include <SIM/SIM_OptionsUser.h>
// #include <SIM/SIM_Object.h>
// #include <SIM/SIM_ScalarField.h>
// #include <SIM/SIM_DataFilter.h>
// #include <SIM/SIM_Engine.h>
// #include <SIM/SIM_VectorField.h>
// #include <SIM/SIM_Time.h>
// #include <SIM/SIM_Solver.h>
// #include <SIM/SIM_DopDescription.h>
// #include <GEO/GEO_Primitive.h>
// #include <GEO/GEO_PrimVDB.h>
// #include <UT/UT_Interrupt.h>
// #include <UT/UT_VoxelArray.h>
// #include <GA/GA_Handle.h>
// #include <GU/GU_Detail.h>
// #include <DOP/DOP_Node.h>
// #include <DOP/DOP_Engine.h>




// void initializeSIM(void *) {
//     IMPLEMENT_DATAFACTORY(GAS_OpenCL_Lissajous);
// }

// const SIM_DopDescription* GAS_OpenCL_Lissajous::getDopDescription() {
//     static PRM_Template templateList[] = {
//         PRM_Template()
//     };

//     static SIM_DopDescription dopDescription(
//         true,
//         "gas_opencl_lissajous", // internal name of the dop
//         "Gas OpenCL Lissajous", // label of the dop
//         "GasOpenCLLissajous", // template list for generating the dop
//         classname(),
//         templateList);

//     setGasDescription(dopDescription);

//     return &dopDescription;
// }

// GAS_OpenCL_Lissajous::GAS_OpenCL_Lissajous(const SIM_DataFactory* factory) : BaseClass(factory) {}

// GAS_OpenCL_Lissajous::~GAS_OpenCL_Lissajous() {}

// bool GAS_OpenCL_Lissajous::solveGasSubclass(SIM_Engine& engine,
//                                             SIM_Object* object,
//                                             SIM_Time time,
//                                             SIM_Time timestep) {

// 	const SIM_Geometry *geo = object->getGeometry();
//     GU_DetailHandleAutoReadLock readlock(geo->getGeometry());
//     const GU_Detail *gdp = readlock.getGdp();

//     accessOpenCLBuffer(geo, gdp, static_cast<float>(time));

//     return true;
// }












// // void GAS_OpenCL_Lissajous::accessOpenCLBuffer(const SIM_Geometry* geo, const GU_Detail* gdp, float time) {
// //     int tuplesize;
// //     GA_CEAttribute* ceAttrib = geo->getReadableCEAttribute(GA_ATTRIB_POINT, "P", GA_STORECLASS_FLOAT, tuplesize, false, false);
// //     CHECK_ERROR(ceAttrib && ceAttrib->isValid(), "Failed to get OpenCL buffer for attribute 'P'");

// //     CE_Context *ceContext = CE_Context::getContext();
// //     CHECK_ERROR(ceContext && ceContext->isValid(), "Invalid CE_Context.");

// //     cl::Buffer clBuffer = ceAttrib->buffer();
// //     CHECK_ERROR(clBuffer(), "Invalid OpenCL buffer.");

// //     GA_Size numEntries = ceAttrib->entries();

// //     updatePositionCUDA(clBuffer, numEntries);
// // }




// // __global__ void updatePositionsKernel(float3* positions, int numEntries) {
// //     int idx = blockIdx.x * blockDim.x + threadIdx.x;
// //     if (idx < numEntries) {
// //         positions[idx].x += 1.0f;
// //     }
// // }




// // // 创建共享的OpenCL和CUDA上下文
// // cl_context createSharedContext(cl_device_id device, cudaDeviceProp &deviceProp) {
// //     // 创建OpenCL上下文属性
// //     cl_context_properties contextProperties[] = {
// //         CL_GL_CONTEXT_KHR, (cl_context_properties)glfwGetCurrentContext(),
// //         CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
// //         CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
// //         0
// //     };

// //     // 创建OpenCL上下文
// //     cl_int err;
// //     cl_context context = clCreateContext(contextProperties, 1, &device, NULL, NULL, &err);
// //     CHECK_ERROR(err == CL_SUCCESS, "Failed to create OpenCL context");

// //     // 设置CUDA设备
// //     cudaError_t cudaStatus = cudaGLSetGLDevice(deviceProp.pciBusID);
// //     CHECK_ERROR(cudaStatus == cudaSuccess, "Failed to set CUDA device");

// //     return context;
// // }

// // cudaGraphicsResource_t registerBufferWithCuda(cl_mem buffer) {
// //     cudaGraphicsResource_t cudaResource;
// //     cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&cudaResource, buffer, cudaGraphicsMapFlagsWriteDiscard);
// //     CHECK_ERROR(cudaStatus == cudaSuccess, "Failed to register buffer with CUDA");

// //     return cudaResource;
// // }


// // float* getCudaPointerFromOpenCLBuffer(cudaGraphicsResource_t cudaResource) {
// //     cudaError_t cudaStatus;
// //     float* cudaPtr;
// //     size_t numBytes;

// //     cudaStatus = cudaGraphicsMapResources(1, &cudaResource, 0);
// //     CHECK_ERROR(cudaStatus == cudaSuccess, "Failed to map CUDA resource");

// //     cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&cudaPtr, &numBytes, cudaResource);
// //     CHECK_ERROR(cudaStatus == cudaSuccess, "Failed to get CUDA pointer from resource");

// //     cudaStatus = cudaGraphicsUnmapResources(1, &cudaResource, 0);
// //     CHECK_ERROR(cudaStatus == cudaSuccess, "Failed to unmap CUDA resource");

// //     return cudaPtr;
// // }



// // void GAS_OpenCL_Lissajous::updatePositionCUDA(cl::Buffer clBuffer, GA_Size numEntries) {
// //     // CUdeviceptr cuBuffer;
// //     // size_t bufferSize;

// //     // clGetMemObjectInfo(clBuffer(), CL_MEM_SIZE, &bufferSize, nullptr);
// //     // clGetMemObjectInfo(clBuffer(), CL_MEM_CUDA_PTR, &cuBuffer, nullptr);

// //     // float3* d_positions = reinterpret_cast<float3*>(cuBuffer);

// //     // int threadsPerBlock = 256;
// //     // int blocksPerGrid = (numEntries + threadsPerBlock - 1) / threadsPerBlock;

// //     // updatePositionsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, numEntries);

// //     // cudaDeviceSynchronize();

// // }




#include "main.hpp"
#include <UT/UT_DSOVersion.h>
#include <UT/UT_Interrupt.h>
#include <PRM/PRM_Include.h>
#include <SIM/SIM_PRMShared.h>
#include <SIM/SIM_DopDescription.h>
#include <SIM/SIM_FieldSampler.h>
#include <SIM/SIM_ScalarField.h>
#include <SIM/SIM_VectorField.h>
#include <SIM/SIM_MatrixField.h>
#include <SIM/SIM_Object.h>
#include <GAS/GAS_SubSolver.h>

///
/// This is the hook that Houdini grabs from the dll to link in
/// this.  As such, it merely has to implement the data factory
/// for this node.
///
void
initializeSIM(void *)
{
    IMPLEMENT_DATAFACTORY(SIM_GasAdd);
}
/// Standard constructor, note that BaseClass was crated by the
/// DECLARE_DATAFACTORY and provides an easy way to chain through
/// the class hierarchy.
SIM_GasAdd::SIM_GasAdd(const SIM_DataFactory *factory)
    : BaseClass(factory)
{
}
SIM_GasAdd::~SIM_GasAdd()
{
}
/// Used to automatically populate the node which will represent
/// this data type.
const SIM_DopDescription *
SIM_GasAdd::getDopDescription()
{
    static PRM_Name     theDstFieldName(GAS_NAME_FIELDDEST, "Dest Field");
    static PRM_Name     theSrcFieldName(GAS_NAME_FIELDSOURCE, "Source Field");
    static PRM_Template          theTemplates[] = {
        PRM_Template(PRM_STRING, 1, &theDstFieldName),
        PRM_Template(PRM_STRING, 1, &theSrcFieldName),
        PRM_Template()
    };
    static SIM_DopDescription    theDopDescription(
            true,               // Should we make a DOP?
            "hdk_gasadd",       // Internal name of the DOP.
            "Gas Add",          // Label of the DOP
            "Solver",           // Default data name
            classname(),        // The type of this DOP, usually the class.
            theTemplates);      // Template list for generating the DOP
    return &theDopDescription;
}
bool
SIM_GasAdd::solveGasSubclass(SIM_Engine &engine,
                        SIM_Object *obj,
                        SIM_Time time,
                        SIM_Time timestep)
{
    SIM_ScalarField     *srcscalar, *dstscalar;
    SIM_VectorField     *srcvector, *dstvector;
    SIM_MatrixField     *srcmatrix, *dstmatrix;
    SIM_DataArray        src, dst;
    int                  i, j, k;
    getMatchingData(src, obj, GAS_NAME_FIELDSOURCE);
    getMatchingData(dst, obj, GAS_NAME_FIELDDEST);
    // Now for each pair of source and dst fields, we want to add
    // src to dst.  We want to support scalar, vector, and matrix fields,
    // but only compatible operations.  We can determine what type we
    // have via casting.
    for (i = 0; i < dst.entries(); i++)
    {
        // Check to see if we exceeded our src list.
        if (i >= src.entries())
        {
            addError(obj, SIM_MESSAGE, "Fewer source fields than destination fields.", UT_ERROR_WARNING);
            break;
        }
        // Try each casting option.
        dstscalar = SIM_DATA_CAST(dst(i), SIM_ScalarField);
        srcscalar = SIM_DATA_CAST(src(i), SIM_ScalarField);
        dstvector = SIM_DATA_CAST(dst(i), SIM_VectorField);
        srcvector = SIM_DATA_CAST(src(i), SIM_VectorField);
        dstmatrix = SIM_DATA_CAST(dst(i), SIM_MatrixField);
        srcmatrix = SIM_DATA_CAST(src(i), SIM_MatrixField);
        if (dstscalar && srcscalar)
        {
            addFields(dstscalar->getField(), srcscalar->getField());
        }
        if (dstvector && srcvector)
        {
            for (j = 0; j < 3; j++)
                addFields(dstvector->getField(j), srcvector->getField(j));
        }
        if (dstmatrix && srcmatrix)
        {
            for (j = 0; j < 3; j++)
                for (k = 0; k < 3; k++)
                    addFields(dstmatrix->getField(j, k), srcmatrix->getField(j, k));
        }
        // Make sure we are flagged as dirty
        if (dstscalar)
            dstscalar->pubHandleModification();
        if (dstvector)
            dstvector->pubHandleModification();
        if (dstmatrix)
            dstmatrix->pubHandleModification();
    }
    // Successful cook
    return true;
}
void
SIM_GasAdd::addFieldsPartial(SIM_RawField *dst, const SIM_RawField *src, const UT_JobInfo &info)
{
    UT_VoxelArrayIteratorF      vit;
    UT_Interrupt                *boss = UTgetInterrupt();
    // Initialize our iterator to run over our destination field.
    vit.setArray(dst->fieldNC());
    // When we complete each tile the tile is tested to see if it can be
    // compressed, ie, is now constant.  If so, it is compressed.
    vit.setCompressOnExit(true);
    // Restrict our iterator only over part of the range.  Using the
    // info parameters means each thread gets its own subregion.
    vit.setPartialRange(info.job(), info.numJobs());
    // Create a sampler for the source field.
    SIM_ScalarFieldSampler srcsampler(dst, src);
    float srcval;
    // Visit every voxel of the destination array.
    for (vit.rewind(); !vit.atEnd(); vit.advance())
    {
        if (vit.isStartOfTile())
        {
            if (boss->opInterrupt())
                break;
            // Check if both source and destination tiles are constant.
            if (vit.isTileConstant() &&
                srcsampler.isTileConstant(vit, srcval))
            {
                // If both are constant, we can process the whole tile at
                // once. We call skipToEndOfTile() here so that the loop's
                // call to advance() will move us to the next tile.
                vit.getTile()->makeConstant( vit.getValue() + srcval );
                vit.skipToEndOfTile();
                continue;
            }
        }
        // Write out the sum of the two fields. Instead of using the
        // iterator, we could also have built a UT_VoxelRWProbeF.
        float srcval = srcsampler.getValue(vit);
        vit.setValue( vit.getValue() + srcval );
    }
}


