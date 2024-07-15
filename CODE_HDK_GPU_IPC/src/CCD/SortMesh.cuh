#pragma once

#include "UTILS/GeometryManager.hpp"

namespace SortMesh {
    void sortMesh(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVH_F>& LBVH_F_ptr);
};
