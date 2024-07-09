#pragma once

#include <vector>
#include <iostream>


#define CHECK_ERROR(cond, msg) \
    if (!(cond)) { \
        std::cerr << msg << std::endl; \
        return; \
    }