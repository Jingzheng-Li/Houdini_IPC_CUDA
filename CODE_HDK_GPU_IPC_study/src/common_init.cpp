
/*
TODO:
1. 解决cuda内存报错的问题
2. 整理一下代码，把主函数分离出来
3. 构造各个节点 
4. 检查tet关键字变量的分配是否正确
5. 可以把GIPC里面的FEM计算代码都放到FEMEnergy中
6. 把Integrator拆分出来 放到integrator中 最后的subsolverIP就从integrator中走 GIPC只提供对应的碰撞Hessian
7. 解决空中形态就错误了的问题 可能能量有问题？？把碰撞能量项都先去掉 大概率是_calculate_triangle_fem_gradient_hessian能量错误
8. 提供单边碰撞检测的方法
9. BoundaryType能够决定是否是attach pin或soft 仔细看看stepForward的代码部分 有一个moveBoundary
10. 计算Hessian没有加上M？是否是错误的？
11. 明天如果看完IPC时间够的话 可以把IPC的edgetri放到LBVH中 groundcollision也放过去 然后linesearch放到INTEGRATOR中 这样拆分到GIPC只有2k行
*/




#include <UT/UT_DSOVersion.h>

#include "RWBuffer/main_readBuffer.hpp"
#include "RWBuffer/main_writeBuffer.hpp"
#include "LBVH/main_LBVH.hpp"
#include "ACCD/main_ACCD.hpp"
#include "IPC/main_GIPC.hpp"

#include "UTILS/GeometryManager.hpp"



void initializeSIM(void *) {
    IMPLEMENT_DATAFACTORY(GAS_Read_Buffer);
    IMPLEMENT_DATAFACTORY(GAS_Write_Buffer);
    // IMPLEMENT_DATAFACTORY(GAS_CUDA_LBVH);
    // IMPLEMENT_DATAFACTORY(GAS_CUDA_ACCD);
    IMPLEMENT_DATAFACTORY(GAS_CUDA_GIPC);
}