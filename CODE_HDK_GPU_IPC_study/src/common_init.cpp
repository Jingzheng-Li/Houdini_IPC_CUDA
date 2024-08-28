
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

/*
IPC Method Conclusion:
1. Construction of BVH tree: At the beginning of each frame, a BVH tree is constructed to quickly detect possible collisions. This tree is based on a triangle mesh and is used to speed up the subsequent collision detection process.

2. ACCD and generation of collision pairs: Through the time-based continuous collision detection algorithm (ACCD), the motion trajectory of the object can be detected in the time domain and the edge pairs or faces that may collide can be identified. These potential collision pairs are recorded and used in subsequent energy calculations.

3. Energy modeling: For each pair of geometric objects that may collide (such as point-edge, edge-edge, or point-face), the distance between them is calculated and expressed as a form of energy. This energy gradient and Hessian are used to guide the implicit solution of the equations of motion.

4. Implicit solution and direction of motion: In this step, the solver calculates the direction of motion (dx) for each vertex by solving the implicit equation of motion that includes an energy term.

5. Line Search and step size α selection: The purpose of line search is to find a suitable step size α so that movement within this step size will not cause any new collisions or intersections. By continuously reducing the value of α, it is ensured that there will be no geometric intersections in each small time step.

6. Iterate until convergence: The entire process will be iterative, gradually updating the vertex positions until the convergence conditions are met. This means that in every frame of motion, the IPC algorithm ensures that objects remain collision-free.

Conclusion: At the beginning of each frame, we start from a collision-free state. First, we build a BVH tree to store all the surface triangle meshes in the CCD. Next, the ACCD algorithm is used to find pairs that may collide, and the distances of these collision pairs are modeled as energy terms and added to the equation of motion. Then, the equation of motion is implicitly solved to calculate the direction of movement (dx) of each vertex. Next, a suitable step size α is found through the line search method to ensure that no interpenetration will occur within this step size. The vertex position moves along the dx direction by the step size of α. We repeat this process until the frame converges. Therefore, during the entire iteration process of each frame, it is always ensured that no interpenetration will occur within each α step size.
*/



#include <UT/UT_DSOVersion.h>

#include "RWBuffer/main_readBuffer.hpp"
#include "RWBuffer/main_writeBuffer.hpp"
#include "LBVH/main_LBVH.hpp"
#include "ACCD/main_ACCD.hpp"
#include "IPC/main_GIPC.hpp"
#include "INTEGRATOR/main_Intergrator.hpp"

#include "UTILS/GeometryManager.hpp"



void initializeSIM(void *) {
    IMPLEMENT_DATAFACTORY(GAS_Read_Buffer);
    IMPLEMENT_DATAFACTORY(GAS_Write_Buffer);
    // IMPLEMENT_DATAFACTORY(GAS_CUDA_LBVH);
    // IMPLEMENT_DATAFACTORY(GAS_CUDA_ACCD);
    IMPLEMENT_DATAFACTORY(GAS_CUDA_GIPC);
    IMPLEMENT_DATAFACTORY(GAS_CUDA_Intergrator);
}