#pragma once

#include "UTILS/GeometryManager.hpp"
#include "UTILS/MathUtils.cuh"

namespace SortMesh {
    
    void sortMesh(std::unique_ptr<GeometryManager>& instance, AABB* LBVHSceneSize);

    void sortPreconditioner(std::unique_ptr<GeometryManager>& instance);

};
