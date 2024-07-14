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

#define CHECK_ERROR(cond, msg) \
    if (!(cond)) { \
        std::cerr << msg << std::endl; \
        return; \
    }

#define CHECK_ERROR_SOLVER(cond, msg) \
    if (!(cond)) { \
        std::cerr << msg << std::endl; \
        return false; \
    }

#define BLUE_TEXT "\033[34m"
#define RESET_TEXT "\033[0m"
#define PRINT_BLUE(msg) \
    std::cout << BLUE_TEXT << msg << RESET_TEXT << std::endl;

inline bool writeNodesToFile(const std::string &filename, const Eigen::MatrixXd &tetpos) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return false;
    }

    outFile << "$Nodes\n";
    outFile << tetpos.rows() << "\n";

    for (int i = 0; i < tetpos.rows(); ++i) {
        outFile << i + 1 << " " 
                << std::fixed << std::setprecision(6) 
                << tetpos(i, 0) << " " 
                << tetpos(i, 1) << " " 
                << tetpos(i, 2) << "\n";
    }
    outFile << "$EndNodes\n";

    outFile.close();
    return true;
}

inline bool writeElementsToFile(const std::string &filename, const Eigen::MatrixXi &tetInd) {
    std::ofstream outFile(filename, std::ios_base::app);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open the output file." << std::endl;
        return false;
    }

    outFile << "$Elements\n";
    outFile << tetInd.rows() << "\n";

    for (int i = 0; i < tetInd.rows(); ++i) {
        outFile << i + 1 << " 4 0 "
                << tetInd(i, 0) + 1 << " " 
                << tetInd(i, 1) + 1 << " " 
                << tetInd(i, 2) + 1 << " " 
                << tetInd(i, 3) + 1 << "\n";
    }
    outFile << "$EndElements\n";

    outFile.close();
    return true;
}
