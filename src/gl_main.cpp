
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "GL/glew.h"
#include "GL/freeglut.h"

#include "FEMEnergy.cuh"
#include "GIPC.cuh"
#include "GeometryManager.hpp"
#include "ImplicitIntegrator.cuh"
#include "LBVH.cuh"
#include "SortMesh.cuh"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "device_launch_parameters.h"
#include "load_mesh.hpp"

namespace gipc {
constexpr auto assets_dir() { return std::string_view{GIPC_ASSETS_DIR}; }
constexpr auto output_dir() { return std::string_view{GIPC_OUTPUT_DIR}; }
}  // namespace gipc

// Global Variables
std::unique_ptr<LBVHCollisionDetector> LBVH_CD_ptr;
std::unique_ptr<PCGSolver> PCG_ptr;
BlockHessian BH_ptr;

auto& instance = GeometryManager::instance;

// Simulation Parameters
int collision_detection_buff_scale = 1;
Scalar motion_rate = 1;

// Mesh Data
SIMMesh simMesh;
LoadObjMesh objMesh;
std::vector<Node> nodes;
std::vector<AABB> bvs;
std::vector<std::string> obj_pathes;
int initPath = 0;

// Rendering and Interaction
int step = 0;
int frameId = 0;
int surfNumId = 0;
float xRot = 0.0f, yRot = 0.0f;
float xTrans = 0.0f, yTrans = 0.0f, zTrans = 0.0f;
int ox = 0, oy = 0;
int buttonState = 0;
float window_width = 1000.0f, window_height = 1000.0f;
bool saveSurface = false;
bool change = false;
bool drawbvh = false;
bool drawSurface = true;
bool stop = true;


// Offsets
int clothFaceOffset = 0;
int bodyVertOffset = 0;

// Animation Flag
bool animation = false;



void saveSurfaceMesh(const std::string& path) {
    std::stringstream ss;
    ss << path;
    ss.fill('0');
    ss.width(5);
    ss << (surfNumId++) / 1;  // Adjust divisor as needed
    ss << ".obj";
    std::string file_path = ss.str();

    std::ofstream outSurf(file_path);
    if (!outSurf.is_open()) return;

    std::map<int, int> meshToSurf;
    for (size_t i = 0; i < objMesh.surfVerts.size(); ++i) {
        const auto& pos = objMesh.vertexes[objMesh.surfVerts[i]];
        outSurf << "v " << pos.x << " " << pos.y << " " << pos.z << "\n";
        meshToSurf[objMesh.surfVerts[i]] = static_cast<int>(i);
    }

    for (const auto& tri : objMesh.surfFaces) {
        outSurf << "f " << meshToSurf[tri.x] + 1 << " " << meshToSurf[tri.y] + 1 << " "
                << meshToSurf[tri.z] + 1 << "\n";
    }
    outSurf.close();
}



// GLUT Callback Functions
void idle_func() { glutPostRedisplay(); }

void reshape_func(GLint width, GLint height) {
    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<float>(width) / height, 0.1f, 500.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.0f);
}

void keyboard_func(unsigned char key, int x, int y) {
    switch (key) {
        case 'w':
            zTrans += 0.01f;
            break;
        case 's':
            zTrans -= 0.01f;
            break;
        case 'a':
            xTrans += 0.01f;
            break;
        case 'd':
            xTrans -= 0.01f;
            break;
        case 'q':
            yTrans -= 0.01f;
            break;
        case 'e':
            yTrans += 0.01f;
            break;
        case '9':
            saveSurface = !saveSurface;
            break;
        case 'k':
            drawSurface = !drawSurface;
            break;
        case 'f':
            drawbvh = !drawbvh;
            break;
        case ' ':
            stop = !stop;
            break;
        default:
            break;
    }
    glutPostRedisplay();
}

void special_keyboard_func(int key, int x, int y) { glutPostRedisplay(); }

void mouse_func(int button, int state, int x, int y) {
    buttonState = (state == GLUT_DOWN) ? 1 : 0;
    ox = x;
    oy = y;
    glutPostRedisplay();
}

void motion_func(int x, int y) {
    float dx = static_cast<float>(x - ox);
    float dy = static_cast<float>(y - oy);

    if (buttonState == 1) {
        xRot += dy / 5.0f;
        yRot += dx / 5.0f;
    }

    ox = x;
    oy = y;
    glutPostRedisplay();
}

void SpecialKey(GLint key, GLint x, GLint y) {
    if (key == GLUT_KEY_DOWN) {
        change = true;
        initPath = (initPath - 1 < 0) ? static_cast<int>(obj_pathes.size()) - 1 : initPath - 1;
    }
    if (key == GLUT_KEY_UP) {
        change = true;
        initPath = (initPath + 1 == static_cast<int>(obj_pathes.size())) ? 0 : initPath + 1;
    }
    glutPostRedisplay();
}

void draw_box3D(float ox, float oy, float oz, float width, float height, float length,
                int boxType = 0) {
    glLineWidth(boxType == 1 ? 1.5f : 0.5f);
    glColor3f(boxType == 1 ? 0.8f : 0.8f, boxType == 1 ? 0.8f : 0.8f, boxType == 1 ? 0.8f : 0.1f);

    glBegin(GL_LINES);
    // Bottom face
    glVertex3f(ox, oy, oz);
    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy + height, oz);
    glVertex3f(ox, oy, oz);
    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox, oy + height, oz);
    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox, oy + height, oz);
    // Top face
    glVertex3f(ox + width, oy, oz);
    glVertex3f(ox + width, oy, oz + length);
    glVertex3f(ox, oy, oz + length);
    glVertex3f(ox + width, oy, oz + length);
    glVertex3f(ox + width, oy + height, oz);
    glVertex3f(ox + width, oy + height, oz + length);
    glVertex3f(ox + width, oy + height, oz + length);
    glVertex3f(ox + width, oy, oz + length);
    glVertex3f(ox, oy + height, oz + length);
    glVertex3f(ox + width, oy + height, oz + length);
    glEnd();
}


void draw_mesh3D() {
    glEnable(GL_DEPTH_TEST);
    glLineWidth(1.5f);
    glColor3f(0.9f, 0.1f, 0.1f);
    const std::vector<uint3>& surf = objMesh.surfFaces;

    glBegin(GL_TRIANGLES);
    for (const auto& face : surf) {
        glVertex3f(objMesh.vertexes[face.x].x, objMesh.vertexes[face.x].y,
                   objMesh.vertexes[face.x].z);
        glVertex3f(objMesh.vertexes[face.y].x, objMesh.vertexes[face.y].y,
                   objMesh.vertexes[face.y].z);
        glVertex3f(objMesh.vertexes[face.z].x, objMesh.vertexes[face.z].y,
                   objMesh.vertexes[face.z].z);
    }
    glEnd();

    glColor3f(0.9f, 0.9f, 0.9f);
    glLineWidth(0.1f);
    glBegin(GL_LINES);
    for (const auto& edge : objMesh.surfEdges) {
        glVertex3f(objMesh.vertexes[edge.x].x, objMesh.vertexes[edge.x].y,
                   objMesh.vertexes[edge.x].z);
        glVertex3f(objMesh.vertexes[edge.y].x, objMesh.vertexes[edge.y].y,
                   objMesh.vertexes[edge.y].z);
    }
    glEnd();
}

void draw_bvh() {
    for (const auto& bv : bvs) {
        float ox = static_cast<float>(bv.lower.x);
        float oy = static_cast<float>(bv.lower.y);
        float oz = static_cast<float>(bv.lower.z);
        float bwidth = static_cast<float>(bv.upper.x - bv.lower.x);
        float bheight = static_cast<float>(bv.upper.y - bv.lower.y);
        float blength = static_cast<float>(bv.upper.z - bv.lower.z);
        draw_box3D(ox, oy, oz, bwidth, bheight, blength);
    }
}

void draw_Scene3D() {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glTranslatef(xTrans, yTrans, zTrans);
    glRotatef(xRot, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot, 0.0f, 1.0f, 0.0f);

    draw_box3D(-1.0f, -1.0f, -1.0f, 2.0f, 2.0f, 2.0f, 1);
    if (drawSurface) draw_mesh3D();
    if (drawbvh) draw_bvh();

    glPopMatrix();
    glutSwapBuffers();
}




// Initialization Functions
void LoadSettings() {
    // Global settings
    instance->getHostDensity() = 1e3;
    instance->getHostYoungModulus() = 1e5;
    instance->getHostPoissonRate() = 0.49;
    instance->getHostLengthRateLame() =
        instance->getHostYoungModulus() / (2 * (1 + instance->getHostPoissonRate()));
    instance->getHostVolumeRateLame() =
        instance->getHostYoungModulus() * instance->getHostPoissonRate() /
        ((1 + instance->getHostPoissonRate()) * (1 - 2 * instance->getHostPoissonRate()));
    instance->getHostLengthRate() = 4 * instance->getHostLengthRateLame() / 3;
    instance->getHostVolumeRate() =
        instance->getHostVolumeRateLame() + 5 * instance->getHostLengthRateLame() / 6;
    instance->getHostFrictionRate() = 0.4;
    instance->getHostClothThickness() = 1e-3;
    instance->getHostClothYoungModulus() = 1e6;
    instance->getHostStretchStiff() =
        instance->getHostClothYoungModulus() / (2 * (1 + instance->getHostPoissonRate()));
    instance->getHostShearStiff() = instance->getHostStretchStiff() * 0.05;
    instance->getHostClothDensity() = 2e2;
    instance->getHostSoftMotionRate() = 1.0;
    instance->getHostBendStiff() = 3e-4;
    instance->getHostNewtonSolverThreshold() = 1e-1;
    instance->getHostPCGThreshold() = 1e-3;
    instance->getHostIPCDt() = 1e-2;
    instance->getHostRelativeDHat() = 1e-3;
    instance->getHostBendStiff() = instance->getHostClothYoungModulus() *
                                   std::pow(instance->getHostClothThickness(), 3) /
                                   (24 * (1 - std::pow(instance->getHostPoissonRate(), 2)));
    instance->getHostShearStiff() = 0.03 * instance->getHostStretchStiff();
    instance->getHostSoftStiffness() = 0.5;
}

void initFEM(LoadObjMesh& mesh) {
    Scalar massSum = 0;
    Scalar volumeSum = 0;

    for (int i = 0; i < mesh.numTetElements; i++) {
        __MATHUTILS__::Matrix3x3S DM;
        __FEMENERGY__::__calculateDms3D_Scalar(mesh.vertexes.data(), mesh.tetrahedras[i], DM);
        __MATHUTILS__::Matrix3x3S DMInverse;
        __MATHUTILS__::__Inverse(DM, DMInverse);

        Scalar vlm = __MATHUTILS__::calculateVolume(mesh.vertexes.data(), mesh.tetrahedras[i]);
        mesh.volume.push_back(vlm);
        mesh.masses[mesh.tetrahedras[i].x] += vlm * instance->getHostDensity() / 4;
        mesh.masses[mesh.tetrahedras[i].y] += vlm * instance->getHostDensity() / 4;
        mesh.masses[mesh.tetrahedras[i].z] += vlm * instance->getHostDensity() / 4;
        mesh.masses[mesh.tetrahedras[i].w] += vlm * instance->getHostDensity() / 4;

        massSum += vlm * instance->getHostDensity();
        volumeSum += vlm;
        mesh.DMInverse.push_back(DMInverse);
    }

    for (int i = 0; i < mesh.numTriElements; i++) {
        __MATHUTILS__::Matrix2x2S DM;
        __FEMENERGY__::__calculateDm2D_Scalar(mesh.vertexes.data(), mesh.triangles[i], DM);
        __MATHUTILS__::Matrix2x2S DMInverse;
        __MATHUTILS__::__Inverse2x2(DM, DMInverse);

        Scalar area = __MATHUTILS__::calculateArea(mesh.vertexes.data(), mesh.triangles[i]);
        area *= instance->getHostClothThickness();
        mesh.area.push_back(area);
        mesh.masses[mesh.triangles[i].x] += instance->getHostClothDensity() * area / 3;
        mesh.masses[mesh.triangles[i].y] += instance->getHostClothDensity() * area / 3;
        mesh.masses[mesh.triangles[i].z] += instance->getHostClothDensity() * area / 3;

        massSum += area * instance->getHostClothDensity();
        volumeSum += area;
        mesh.triDMInverse.push_back(DMInverse);
    }

    mesh.meanMass = massSum / mesh.numVertices;
    mesh.meanVolume = volumeSum / mesh.numVertices;
}

void initScene1() {
    auto assets_dir = std::string{gipc::assets_dir()};

    LOADMESH::readObj_CGAL(assets_dir + "tshirt/Tshirt_total.obj", simMesh.CGAL_clothmesh);
    LOADMESH::readObj_CGAL(assets_dir + "tshirt/body_1.obj", simMesh.CGAL_bodymesh);
    LOADMESH::extractTriBendEdgesFaces_CGAL(simMesh.CGAL_clothmesh, objMesh.triBendEdges, objMesh.triBendVerts);


    // objMesh.numVertices = simMesh.CGAL_clothmesh.number_of_vertices() + simMesh.CGAL_bodymesh.number_of_vertices();
    LOADMESH::convertCGALVertsToVector(simMesh.CGAL_clothmesh, simMesh.clothverts);
    LOADMESH::convertCGALFacesToVector(simMesh.CGAL_clothmesh, simMesh.clothfaces);
    LOADMESH::convertCGALVertsToVector(simMesh.CGAL_bodymesh, simMesh.bodyverts);
    LOADMESH::convertCGALFacesToVector(simMesh.CGAL_bodymesh, simMesh.bodyfaces);
    simMesh.totalverts.reserve(simMesh.clothverts.size() + simMesh.bodyverts.size());
    simMesh.totalverts.insert(simMesh.totalverts.end(), simMesh.clothverts.begin(), simMesh.clothverts.end());
    simMesh.totalverts.insert(simMesh.totalverts.end(), simMesh.bodyverts.begin(), simMesh.bodyverts.end());
    simMesh.totalfaces.reserve(simMesh.clothfaces.size() + simMesh.bodyfaces.size());
    simMesh.totalfaces.insert(simMesh.totalfaces.end(), simMesh.clothfaces.begin(), simMesh.clothfaces.end());
    simMesh.totalfaces.insert(simMesh.totalfaces.end(), simMesh.bodyfaces.begin(), simMesh.bodyfaces.end());



    // objMesh.numVertices = simMesh.totalverts.size();
    // objMesh.numTriElements = simMesh.clothfaces.size();
    // objMesh.numBoundTargets = simMesh.bodyfaces.size();
    // for (int i = 0; i < objMesh.numVertices; i++) {
    //     objMesh.vertexes.push_back(simMesh.totalverts[i]);
    //     objMesh.masses.push_back(0);
    //     objMesh.velocities.push_back(make_Scalar3(0, 0, 0));

    //     __MATHUTILS__::Matrix3x3S constraint;
    //     __MATHUTILS__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
    //     objMesh.constraints.push_back(constraint);

    //     if (i < simMesh.clothverts.size()) {
    //         objMesh.boundaryTypies.push_back(0); // set cloth boundary as 0
    //         objMesh.triangles.push_back(simMesh.totalfaces[i]);
    //         objMesh.surfFaces.push_back(simMesh.totalfaces[i]);
    //     } else {
    //         objMesh.boundaryTypies.push_back(2); // set body boundary as 2
    //         objMesh.surfFaces.push_back(simMesh.totalfaces[i]);
    //         objMesh.targetIndex.push_back(i);
    //         std::cout << "targetindex tid: " << i << std::endl;
    //     }
    // }














    objMesh.load_triMesh(assets_dir + "tshirt/Tshirt_total.obj", 1.0, make_Scalar3(0, 0.0, 0), 0);
    objMesh.load_triMesh(assets_dir + "tshirt/body_1.obj", 1.0, make_Scalar3(0, 0, 0), 2);







    objMesh.getSurface();
    

    objMesh.load_animation(assets_dir + "tshirt/body_1.obj", 1.0, make_Scalar3(0, 0, 0));

    instance->getHostSoftTargetIdsBeforeSort() = {
        138,  140,  143,  147,  149,  236,  238,  239,  241,  251,  253,  258,  260,
        262,  497,  498,  499,  501,  502,  504,  507,  509,  511,  513,  515,  517,
        519,  526,  529,  533,  535,  536,  538,  539,  544,  548,  549,  551,  553,
        1633, 1635, 1637, 1639, 1640, 1691, 1694, 1697, 1700, 1702, 1704, 1706, 1707,
        1709, 1712, 1714, 1716, 1718, 1719, 1721, 1723, 1724, 1726, 1728, 1730, 1732,
        1734, 1736, 1738, 1742, 1746, 1748, 1749, 1751, 1754, 1755, 1757};

    instance->getHostSoftTargetIdsAfterSort() = {
        1154, 1197, 1148, 1157, 1055, 1049, 1010, 1005, 1006, 423,  589,  583,  585,
        633,  4144, 4153, 4112, 4111, 4099, 4039, 4032, 3965, 3961, 3615, 3613, 3470,
        3467, 3465, 3462, 3418, 3416, 3414, 3413, 3412, 3410, 640,  638,  637,  636,
        5329, 5334, 5291, 5298, 5377, 5384, 5387, 5783, 5785, 5513, 5538, 5525, 5528,
        5645, 5593, 5605, 5598, 5573, 5579, 2648, 2645, 2634, 2629, 2569, 2556, 2545,
        2543, 2530, 2522, 2334, 2317, 2311, 2302, 2258, 2243, 2238, 1203};

    objMesh.load_soft_animation(assets_dir + "tshirt/cloth_targetpos_1.obj", 1.0,
                                make_Scalar3(0, 0, 0), instance->getHostSoftTargetPos());

    instance->getHostNumSoftTargets() = instance->getHostSoftTargetIdsAfterSort().size();

    initFEM(objMesh);

    instance->getHostNumVertices() = objMesh.numVertices;
    instance->getHostNumTetElements() = objMesh.numTetElements;
    instance->getHostNumTriElements() = objMesh.numTriElements;
    instance->getHostNumTriBendEdges() = objMesh.triBendEdges.size();
    instance->getHostNumBoundTargets() = objMesh.numBoundTargets;
    instance->getHostNumSurfVerts() = objMesh.surfVerts.size();
    instance->getHostNumSurfFaces() = objMesh.surfFaces.size();
    instance->getHostNumSurfEdges() = objMesh.surfEdges.size();
    instance->getHostMaxCCDCollisionPairsNum() =
        1 * collision_detection_buff_scale *
        (((Scalar)(instance->getHostNumSurfFaces() * 15 + instance->getHostNumSurfEdges() * 10)) *
         std::max((instance->getHostIPCDt() / 0.01), 2.0));
    instance->getHostMaxCollisionPairsNum() =
        (instance->getHostNumSurfVerts() * 3 + instance->getHostNumSurfEdges() * 2) * 3 *
        collision_detection_buff_scale;
    instance->getHostMaxTetTriMortonCodeNum() =
        std::max(instance->getHostNumVertices(), instance->getHostNumTetElements());
    // instance->getHostPrecondType() = 1;
    instance->getHostPrecondType() = 0;

    PCG_ptr->PrecondType = instance->getHostPrecondType();

    CUDAMallocSafe(instance->getCudaVertPos(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaOriginVertPos(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaVertVel(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaRestVertPos(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaTempScalar3Mem(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaXTilta(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaFb(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaTetElement(), instance->getHostNumTetElements());
    CUDAMallocSafe(instance->getCudaTriBendEdges(), instance->getHostNumTriBendEdges());
    CUDAMallocSafe(instance->getCudaTriBendVerts(), instance->getHostNumTriBendEdges());
    CUDAMallocSafe(instance->getCudaTetVolume(), instance->getHostNumTetElements());
    CUDAMallocSafe(instance->getCudaVertMass(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaMortonCodeHash(), instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(instance->getCudaSortIndex(), instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(instance->getCudaBoundaryType(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaTempBoundaryType(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaSortMapVertIndex(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaTetDmInverses(), instance->getHostNumTetElements());
    CUDAMallocSafe(instance->getCudaConstraintsMat(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaTempScalar(), instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(instance->getCudaTempMat3x3(), instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(instance->getCudaBoundTargetIndex(), instance->getHostNumBoundTargets());
    CUDAMallocSafe(instance->getCudaBoundTargetVertPos(), instance->getHostNumBoundTargets());
    CUDAMallocSafe(instance->getCudaTriDmInverses(), instance->getHostNumTriElements());
    CUDAMallocSafe(instance->getCudaTriArea(), instance->getHostNumTriElements());
    CUDAMallocSafe(instance->getCudaTriElement(), instance->getHostNumTriElements());

    CUDAMemcpyHToDSafe(instance->getCudaVertPos(), objMesh.vertexes);
    CUDAMemcpyHToDSafe(instance->getCudaVertVel(), objMesh.velocities);
    CUDAMemcpyHToDSafe(instance->getCudaVertMass(), objMesh.masses);
    CUDAMemcpyHToDSafe(instance->getCudaOriginVertPos(), objMesh.vertexes);
    CUDAMemcpyHToDSafe(instance->getCudaTriElement(), objMesh.triangles);
    CUDAMemcpyHToDSafe(instance->getCudaTetElement(), objMesh.tetrahedras);
    CUDAMemcpyHToDSafe(instance->getCudaTriArea(), objMesh.area);
    CUDAMemcpyHToDSafe(instance->getCudaTetVolume(), objMesh.volume);
    CUDAMemcpyHToDSafe(instance->getCudaTriDmInverses(), objMesh.triDMInverse);
    CUDAMemcpyHToDSafe(instance->getCudaTetDmInverses(), objMesh.DMInverse);
    CUDAMemcpyHToDSafe(instance->getCudaTriBendEdges(), objMesh.triBendEdges);
    CUDAMemcpyHToDSafe(instance->getCudaTriBendVerts(), objMesh.triBendVerts);
    CUDAMemcpyHToDSafe(instance->getCudaConstraintsMat(), objMesh.constraints);
    CUDAMemcpyHToDSafe(instance->getCudaBoundaryType(), objMesh.boundaryTypies);
    CUDAMemcpyHToDSafe(instance->getCudaBoundTargetIndex(), objMesh.targetIndex);
    CUDAMemcpyHToDSafe(instance->getCudaBoundTargetVertPos(), objMesh.targetPos);

    CUDAMallocSafe(instance->getCudaSoftTargetIndex(), instance->getHostNumSoftTargets());
    CUDAMallocSafe(instance->getCudaSoftTargetVertPos(), instance->getHostNumSoftTargets());
    CUDAMemcpyHToDSafe(instance->getCudaSoftTargetIndex(),
                       instance->getHostSoftTargetIdsAfterSort());
    CUDAMemcpyHToDSafe(instance->getCudaSoftTargetVertPos(), instance->getHostSoftTargetPos());

    CUDAMallocSafe(instance->getCudaMoveDir(), instance->getHostNumVertices());
    CUDAMallocSafe(instance->getCudaMatIndex(), instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(instance->getCudaCollisionPairs(), instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(instance->getCudaCCDCollisionPairs(),
                   instance->getHostMaxCCDCollisionPairsNum());
    CUDAMallocSafe(instance->getCudaEnvCollisionPairs(), instance->getHostNumSurfVerts());
    CUDAMallocSafe(instance->getCudaCPNum(), 5);         // 固定 5 个元素
    CUDAMallocSafe(instance->getCudaGPNum(), 1);         // 单个元素
    CUDAMallocSafe(instance->getCudaGroundNormal(), 5);  // 固定 5 个元素
    CUDAMallocSafe(instance->getCudaGroundOffset(), 5);  // 固定 5 个元素

    // 将 Host 端数据复制到 Device 端
    std::vector<Scalar> h_offset = {-1, -1, 1, -1, 1};
    std::vector<Scalar3> H_normal = {make_Scalar3(0, 1, 0), make_Scalar3(1, 0, 0),
                                     make_Scalar3(-1, 0, 0), make_Scalar3(0, 0, 1),
                                     make_Scalar3(0, 0, -1)};
    CUDAMemcpyHToDSafe(instance->getCudaGroundOffset(), h_offset);
    CUDAMemcpyHToDSafe(instance->getCudaGroundNormal(), H_normal);

    // 分配 Surf 数据
    CUDAMallocSafe(instance->getCudaSurfFace(), instance->getHostNumSurfFaces());
    CUDAMallocSafe(instance->getCudaSurfEdge(), instance->getHostNumSurfEdges());
    CUDAMallocSafe(instance->getCudaSurfVert(), instance->getHostNumSurfVerts());

    CUDAMemcpyHToDSafe(instance->getCudaSurfFace(), objMesh.surfFaces);
    CUDAMemcpyHToDSafe(instance->getCudaSurfEdge(), objMesh.surfEdges);
    CUDAMemcpyHToDSafe(instance->getCudaSurfVert(), objMesh.surfVerts);

    // 分配 CloseCP 和 CloseGP
    CUDAMallocSafe(instance->getCudaCloseCPNum(), 1);  // 单个元素
    CUDAMallocSafe(instance->getCudaCloseGPNum(), 1);  // 单个元素

    PCG_ptr->CUDA_MALLOC_PCGSOLVER(instance->getHostNumVertices(),
                                   instance->getHostNumTetElements());
    BH_ptr.CUDA_MALLOC_BLOCKHESSIAN(
        instance->getHostNumTetElements(), instance->getHostNumSurfVerts(),
        instance->getHostNumSurfFaces(), instance->getHostNumSurfEdges(),
        instance->getHostNumTriElements(), instance->getHostNumTriBendEdges());

    LBVH_CD_ptr->initBVH(instance, instance->getCudaBoundaryType());

    if (PCG_ptr->PrecondType) {
        int neighborListSize = objMesh.getVertNeighbors();
        PCG_ptr->MP.CUDA_FREE_MAS_PRECONDITIONER(instance->getHostNumVertices(), neighborListSize,
                                                 instance->getCudaCollisionPairs());

        PCG_ptr->MP.hostMASNeighborListSize = neighborListSize;

        CUDAMemcpyHToDSafe(PCG_ptr->MP.cudaNeighborListInit, objMesh.neighborList);
        CUDAMemcpyHToDSafe(PCG_ptr->MP.cudaNeighborStart, objMesh.neighborStart);
        CUDAMemcpyHToDSafe(PCG_ptr->MP.cudaNeighborNumInit, objMesh.neighborNum);
    }

    animation = true;
    bodyVertOffset = 6164;

    AABB* bvs_AABB = LBVH_CD_ptr->lbvh_f.getSceneSize();
    std::vector<Scalar3> vec_upper_bvs(1);
    std::vector<Scalar3> vec_lower_bvs(1);
    CUDAMemcpyDToHSafe(vec_upper_bvs, &bvs_AABB->upper);
    CUDAMemcpyDToHSafe(vec_lower_bvs, &bvs_AABB->lower);
    Scalar3 _upper_bvs = vec_upper_bvs[0];
    Scalar3 _lower_bvs = vec_lower_bvs[0];

    if (animation) {
        __SORTMESH__::sortMesh(instance->getCudaVertPos(), instance->getCudaMortonCodeHash(),
                               instance->getCudaSortIndex(), instance->getCudaSortMapVertIndex(),
                               instance->getCudaOriginVertPos(), instance->getCudaTempScalar(),
                               instance->getCudaVertMass(), instance->getCudaTempMat3x3(),
                               instance->getCudaConstraintsMat(), instance->getCudaBoundaryType(),
                               instance->getCudaTempBoundaryType(), instance->getCudaTetElement(),
                               instance->getCudaTriElement(), instance->getCudaSurfFace(),
                               instance->getCudaSurfEdge(), instance->getCudaTriBendEdges(),
                               instance->getCudaTriBendVerts(), instance->getCudaSurfVert(),
                               instance->getHostNumTetElements(), instance->getHostNumTriElements(),
                               instance->getHostNumSurfVerts(), instance->getHostNumSurfFaces(),
                               instance->getHostNumSurfEdges(), instance->getHostNumTriBendEdges(),
                               _upper_bvs, _lower_bvs, bodyVertOffset);

        if (instance->getHostPrecondType() == 1) {
            __SORTMESH__::sortPreconditioner(
                PCG_ptr->MP.cudaNeighborList, PCG_ptr->MP.cudaNeighborListInit,
                PCG_ptr->MP.cudaNeighborNum, PCG_ptr->MP.cudaNeighborNumInit,
                PCG_ptr->MP.cudaNeighborStart, PCG_ptr->MP.cudaNeighborStartTemp,
                instance->getCudaSortIndex(), instance->getCudaSortMapVertIndex(),
                PCG_ptr->MP.hostMASNeighborListSize, bodyVertOffset);
        }

        CUDA_SAFE_CALL(cudaMemcpy(objMesh.vertexes.data(), instance->getCudaVertPos(),
                                  instance->getHostNumVertices() * sizeof(Scalar3),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(objMesh.surfFaces.data(), instance->getCudaSurfFace(),
                                  instance->getHostNumSurfFaces() * sizeof(uint3),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(objMesh.surfEdges.data(), instance->getCudaSurfEdge(),
                                  instance->getHostNumSurfEdges() * sizeof(uint2),
                                  cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(objMesh.surfVerts.data(), instance->getCudaSurfVert(),
                                  instance->getHostNumSurfVerts() * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));

    }

    CUDA_SAFE_CALL(cudaMemcpy(instance->getCudaRestVertPos(), instance->getCudaOriginVertPos(),
                              instance->getHostNumVertices() * sizeof(Scalar3),
                              cudaMemcpyDeviceToDevice));

    LBVH_CD_ptr->buildBVH(instance);

    // __GPUIPC__::init(instance, objMesh.meanMass, objMesh.meanVolume);

    instance->getHostBboxDiagSize2() = __MATHUTILS__::__squaredNorm(
        __MATHUTILS__::__minus(LBVH_CD_ptr->lbvh_f.scene.upper, LBVH_CD_ptr->lbvh_f.scene.lower));
    instance->getHostDTol() = 1e-18 * instance->getHostBboxDiagSize2();
    instance->getHostMinKappaCoef() = 1e11;
    instance->getHostMeanMass() = objMesh.meanMass;
    instance->getHostMeanVolume() = objMesh.meanVolume;
    instance->getHostDHat() = instance->getHostRelativeDHat() * instance->getHostRelativeDHat() *
                              instance->getHostBboxDiagSize2();
    instance->getHostFDHat() = 1e-6 * instance->getHostBboxDiagSize2();

    instance->getHostCpNumLast(0) = 0;
    instance->getHostCpNumLast(1) = 0;
    instance->getHostCpNumLast(2) = 0;
    instance->getHostCpNumLast(3) = 0;
    instance->getHostCpNumLast(4) = 0;

    // printf("bboxDiagSize2: %f\n", instance->getHostBboxDiagSize2());
    // printf("maxConer: %f  %f   %f           minCorner: %f  %f   %f\n", objMesh.maxConer.x,
    // objMesh.maxConer.y, objMesh.maxConer.z, objMesh.minConer.x, objMesh.minConer.y,
    // objMesh.minConer.z);

    LBVH_CD_ptr->buildCP(instance);
    __GPUIPC__::buildGP(instance);
    // PCG_ptr->cudaPCGb = instance->getCudaFb();
    // instance->getCudaMoveDir() = PCG_ptr->cudaPCGdx;

    instance->getHostAnimationSubRate() = 1.0 / motion_rate;

    __INTEGRATOR__::computeXTilta(instance, 1);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    ///////////////////////////////////////////////////////////////////////////////////

    // view whether intersection, useful
    if (__GPUIPC__::isIntersected(instance, LBVH_CD_ptr)) {
        printf("init intersection\n");
    }

    // view BVH
    bvs.resize(2 * instance->getHostNumSurfEdges() - 1);
    nodes.resize(2 * instance->getHostNumSurfEdges() - 1);
    CUDA_SAFE_CALL(cudaMemcpy(&bvs[0], LBVH_CD_ptr->lbvh_e._bvs,
                              (2 * instance->getHostNumSurfEdges() - 1) * sizeof(AABB),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&nodes[0], LBVH_CD_ptr->lbvh_e._nodes,
                              (2 * instance->getHostNumSurfEdges() - 1) * sizeof(Node),
                              cudaMemcpyDeviceToHost));

}


// Initialization Function
void init(void) {
    Init_CUDA();

    if (!instance) {
        instance = std::make_unique<GeometryManager>();
    }
    CHECK_ERROR(instance, "init instance not initialize");

    if (!LBVH_CD_ptr) {
        LBVH_CD_ptr = std::make_unique<LBVHCollisionDetector>();
    }

    if (!PCG_ptr) {
        PCG_ptr = std::make_unique<PCGSolver>();
    }

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cerr << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    LoadSettings();
    initScene1();

    glViewport(0, 0, static_cast<int>(window_width), static_cast<int>(window_height));
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, window_width / window_height, 10.1f, 500.0f);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -3.0f);
}


// Display Function
void display(void) {
    auto& instance = GeometryManager::instance;
    CHECK_ERROR(instance, "PCGSolver::Malloc_DEVICE_MEM geoinstance not initialized");

    draw_Scene3D();

    if (stop) return;

    // Run IPC Solver
    __INTEGRATOR__::IPC_Solver(instance, BH_ptr, PCG_ptr, LBVH_CD_ptr);

    // Update Animation Frame
    std::string assets_dir = std::string{gipc::assets_dir()};
    frameId++;
    objMesh.load_soft_animation(
        assets_dir + "tshirt/cloth_targetpos_" + std::to_string(frameId) + ".obj", 1.0,
        make_Scalar3(0, 0, 0.0), instance->getHostSoftTargetPos());
    objMesh.load_animation(assets_dir + "tshirt/body_" + std::to_string(frameId) + ".obj", 1.0,
                           make_Scalar3(0, 0, 0.0));
    CUDA_SAFE_CALL(cudaMemcpy(instance->getCudaBoundTargetVertPos(), objMesh.targetPos.data(),
                              objMesh.numBoundTargets * sizeof(Scalar3), cudaMemcpyHostToDevice));

    // Update BVH Data
    CUDA_SAFE_CALL(cudaMemcpy(bvs.data(), LBVH_CD_ptr->lbvh_e._bvs, bvs.size() * sizeof(AABB),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(nodes.data(), LBVH_CD_ptr->lbvh_e._nodes, nodes.size() * sizeof(Node),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(objMesh.vertexes.data(), instance->getCudaVertPos(),
                              instance->getHostNumVertices() * sizeof(Scalar3),
                              cudaMemcpyDeviceToHost));

    step++;

    // Handle Surface Saving
    if (saveSurface) {
        auto assets_dir = std::string{gipc::assets_dir()};
        saveSurfaceMesh(assets_dir + "saveSurface_constraint1/surf_");
    }

    // Exit after first step (remove or modify as needed)
    if (step > 0) {
        exit(EXIT_SUCCESS);
    }
}







// Main Function
int main(int argc, char** argv) {
    glutInit(&argc, argv);

    // Enable multisampling for better rendering quality
    glutSetOption(GLUT_MULTISAMPLE, 16);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);

    glutInitWindowSize(static_cast<int>(window_width), static_cast<int>(window_height));
    glutInitWindowPosition(0, 0);
    glutCreateWindow("FEM");

    init();

    // OpenGL Settings
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glHint(GL_MULTISAMPLE_FILTER_HINT_NV, GL_NICEST);

    // Register GLUT Callback Functions
    glutDisplayFunc(display);
    glutReshapeFunc(reshape_func);
    glutKeyboardFunc(keyboard_func);
    glutSpecialFunc(SpecialKey);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutIdleFunc(idle_func);

    glutMainLoop();

    return 0;
}
