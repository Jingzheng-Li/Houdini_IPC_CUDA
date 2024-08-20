#pragma once

#include "UTILS/GeometryManager.hpp"

namespace SortMesh {
    
    void sortMesh(std::unique_ptr<GeometryManager>& instance, AABB* LBVHSceneSize);

};
