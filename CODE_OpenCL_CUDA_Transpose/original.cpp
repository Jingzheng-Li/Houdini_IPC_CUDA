// void GAS_OpenCL_Lissajous::accessOpenCLBuffer(const SIM_Geometry* geo, const GU_Detail* gdp, float time) {
//     int tuplesize;
//     GA_CEAttribute* ceAttrib = geo->getReadableCEAttribute(GA_ATTRIB_POINT, "P", GA_STORECLASS_FLOAT, tuplesize, false, false);
//     CHECK_ERROR(ceAttrib && ceAttrib->isValid(), "Failed to get OpenCL buffer for attribute 'P'");

//     CE_Context *ceContext = CE_Context::getContext();
//     CHECK_ERROR(ceContext && ceContext->isValid(), "Invalid CE_Context.");

//     cl::Buffer clBuffer = ceAttrib->buffer();
//     CHECK_ERROR(clBuffer(), "Invalid OpenCL buffer.");

//     GA_Size numEntries = ceAttrib->entries();

//     cl_ulong bufferSize;
//     clBuffer.getInfo(CL_MEM_SIZE, &bufferSize);
//     std::vector<vecfloat3> positions(numEntries);
//     size_t positionsSize = numEntries * sizeof(vecfloat3);
//     CHECK_ERROR(positionsSize == bufferSize, "Position size and buffer size not match");

//     try {
//         ceContext->readBuffer(clBuffer, bufferSize, reinterpret_cast<float*>(positions.data()));

//         for (auto& pos : positions) {
//             pos.x += 1.0f; // Example modification
//         }

//         ceContext->writeBuffer(clBuffer, bufferSize, reinterpret_cast<const float*>(positions.data()));
//     } catch (const cl::Error& err) {
//         std::cerr << "OpenCL error: " << err.what() << ", " << err.err() << std::endl;
//     } catch (const std::exception& ex) {
//         std::cerr << "Standard exception: " << ex.what() << std::endl;
//     } catch (...) {
//         std::cerr << "Unknown exception occurred." << std::endl;
//     }
// }



// struct vecfloat3 {
//     float x, y, z;
// };









// void GAS_OpenCL_Lissajous::accessOpenCLBuffer(const SIM_Geometry* geo, const GU_Detail* gdp, float time) {
//     int tuplesize;
//     GA_CEAttribute* ceAttrib = geo->getReadableCEAttribute(GA_ATTRIB_POINT, "P", GA_STORECLASS_FLOAT, tuplesize, false, false);
//     CHECK_ERROR(ceAttrib && ceAttrib->isValid(), "Failed to get OpenCL buffer for attribute 'P'");

//     CE_Context *ceContext = CE_Context::getContext();
//     CHECK_ERROR(ceContext && ceContext->isValid(), "Invalid CE_Context.");

//     cl::Buffer clBuffer = ceAttrib->buffer();
//     CHECK_ERROR(clBuffer(), "Invalid OpenCL buffer.");

//     GA_Size numEntries = ceAttrib->entries();

//     cl_ulong bufferSize;
//     clBuffer.getInfo(CL_MEM_SIZE, &bufferSize);
//     size_t positionsSize = numEntries * sizeof(vecfloat3);
//     CHECK_ERROR(positionsSize == bufferSize, "Position size and buffer size not match");

//     // 创建一个临时的 OpenCL 缓冲区
//     cl::Buffer tempBuffer = cl::Buffer(ceContext->context(), CL_MEM_READ_WRITE, bufferSize);

//     // 在 GPU 内部复制数据
//     ceContext->commandQueue().enqueueCopyBuffer(clBuffer, tempBuffer, 0, 0, bufferSize);

//     // 调用 CUDA 内核进行计算
//     updatePositionCUDA(tempBuffer, numEntries);

//     // 将结果复制回原始缓冲区
//     ceContext->commandQueue().enqueueCopyBuffer(tempBuffer, clBuffer, 0, 0, bufferSize);
// }

// void GAS_OpenCL_Lissajous::updatePositionCUDA(cl::Buffer clBuffer, GA_Size numEntries) {
//     CUdeviceptr cuBuffer;
//     size_t bufferSize;

//     clGetMemObjectInfo(clBuffer(), CL_MEM_SIZE, &bufferSize, nullptr);
//     clGetMemObjectInfo(clBuffer(), CL_MEM_CUDA_PTR, &cuBuffer, nullptr);

//     vecfloat3* d_positions = reinterpret_cast<vecfloat3*>(cuBuffer);

//     int threadsPerBlock = 256;
//     int blocksPerGrid = (numEntries + threadsPerBlock - 1) / threadsPerBlock;

//     updatePositionsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_positions, numEntries);
//     cudaDeviceSynchronize();
// }

