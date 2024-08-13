#pragma once

#include <iostream>

#include <GAS/GAS_SubSolver.h>
#include <GAS/GAS_Utils.h>

#include <SIM/SIM_GeometryCopy.h>
#include <SIM/SIM_Geometry.h>
#include <SIM/SIM_OptionsUser.h>
#include <SIM/SIM_Object.h>
#include <SIM/SIM_ScalarField.h>
#include <SIM/SIM_DataFilter.h>
#include <SIM/SIM_Engine.h>
#include <SIM/SIM_VectorField.h>
#include <SIM/SIM_Time.h>
#include <SIM/SIM_Solver.h>
#include <SIM/SIM_DopDescription.h>
#include <GEO/GEO_Primitive.h>
#include <GEO/GEO_PrimVDB.h>
#include <UT/UT_Interrupt.h>
#include <UT/UT_VoxelArray.h>
#include <GA/GA_Handle.h>
#include <GU/GU_Detail.h>
#include <DOP/DOP_Node.h>
#include <DOP/DOP_Engine.h>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

#define CHECK_ERROR(correctcond, msg) \
    if (!(correctcond)) { \
        std::cerr << msg << std::endl; \
        return; \
    }

#define CHECK_ERROR_SOLVER(correctcond, msg) \
    if (!(correctcond)) { \
        std::cerr << msg << std::endl; \
        return false; \
    }

#define BLUE_TEXT "\033[34m"
#define RESET_TEXT "\033[0m"
#define PRINT_BLUE(msg) \
    std::cout << BLUE_TEXT << msg << RESET_TEXT << std::endl;

