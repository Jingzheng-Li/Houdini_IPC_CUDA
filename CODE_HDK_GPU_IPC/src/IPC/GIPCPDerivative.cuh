
// #pragma once

// #include "UTILS/CUDAUtils.hpp"
// #include "UTILS/MathUtils.cuh"

// namespace GIPCPDERIV {


// void pFpx_peeV0(double d, double x11, double x12, double x13, double x21,
//                 double x22, double x23, double x31, double x32, double x33,
//                 double x41, double x42, double x43, double result[12][9]);

// void pFpx_ppeV0(double d, double x11, double x12, double x13, double x21,
//                 double x22, double x23, double x31, double x32, double x33,
//                 double x41, double x42, double x43, double result[12][9]);

// void pFpx_pppV0(double d, double x11, double x12, double x13, double x21,
//                 double x22, double x23, double x31, double x32, double x33,
//                 double x41, double x42, double x43, double result[12][9]);

// void pFpx_peeV1(double d, double x11, double x12, double x13, double x21,
//                 double x22, double x23, double x31, double x32, double x33,
//                 double x41, double x42, double x43, double result[12][9]);

// void pFpx_pppV1(double d, double x11, double x12, double x13, double x21,
//                 double x22, double x23, double x31, double x32, double x33,
//                 double x41, double x42, double x43, double result[12][9]);

// void pFpx_ppeV1(double d, double x11, double x12, double x13, double x21,
//                 double x22, double x23, double x31, double x32, double x33,
//                 double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pFpx_pppV0V1(double d, double x11, double x12, double x13, double x21,
//                   double x22, double x23, double x31, double x32, double x33,
//                   double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pFpx_ppeV0V1(double d, double x11, double x12, double x13, double x21,
//                   double x22, double x23, double x31, double x32, double x33,
//                   double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pFpx_peeV0V1(double d, double x11, double x12, double x13, double x21,
//                   double x22, double x23, double x31, double x32, double x33,
//                   double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pFpx_ppe(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pFpx);

// __host__ __device__
// void pFpx_ppp(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pFpx);

// __host__ __device__
// void pFpx_pee(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pFpx);

// __host__ __device__
// void pFpx_ee2(double d, double x11, double x12, double x13, double x21,
//               double x22, double x23, double x31, double x32, double x33,
//               double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pFpx_ee2(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pFpx);

// __host__ __device__
// void pFpx_pp2(double d, double x11, double x12, double x13, double x21,
//               double x22, double x23, double result[6]);

// __host__ __device__
// void pFpx_pp2(const double3& x0, const double3& x1, double d_hatSqrt, MATHUTILS::Vector6& pFpx);


// __host__ __device__
// void pFpx_pt2(double d, double x11, double x12, double x13, double x21,
//               double x22, double x23, double x31, double x32, double x33,
//               double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pFpx_pe2(double d, double x11, double x12, double x13, double x21,
//               double x22, double x23, double x31, double x32, double x33,
//               double result[9][4]);

// __host__ __device__
// void pFpx_pe2(const double3& x0, const double3& x1, const double3& x2, double d_hatSqrt, MATHUTILS::Matrix9x4d& pFpx);

// __host__ __device__
// void pFpx_pt2(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pFpx);

// __host__ __device__
// void pDmpx_ee_flip(double d, double x11, double x12, double x13, double x21,
//                    double x22, double x23, double x31, double x32, double x33,
//                    double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pDmpx_ee_flip(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pDmpx);

// __host__ __device__
// void pDmpx_ee(double d, double x11, double x12, double x13, double x21,
//               double x22, double x23, double x31, double x32, double x33,
//               double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pDmpx_ee(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pDmpx);

// __host__ __device__
// void pDmpx_pee_reflect(double d, double x11, double x12, double x13, double x21,
//                        double x22, double x23, double x31, double x32,
//                        double x33, double x41, double x42, double x43,
//                        double result[12][4]);

// __host__ __device__
// void pDmpx_pee_reflect(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x4d& pDmpx);

// __host__ __device__
// void ft_2(const double ct[270], double result[12][4]);

// __host__ __device__
// void pDmpx_pee(double d, double x11, double x12, double x13, double x21,
//                double x22, double x23, double x31, double x32, double x33,
//                double x41, double x42, double x43, double result[12][4]);

// __host__ __device__
// void pDmpx_pee(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x4d& pDmpx);

// __host__ __device__
// void pDmpx_pt(double d, double x11, double x12, double x13, double x21,
//               double x22, double x23, double x31, double x32, double x33,
//               double x41, double x42, double x43, double result[12][9]);
              
// __host__ __device__
// void pDmpx_pt(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pDmpx);

// __host__ __device__
// void pDmpx_pt_flip(double d, double x11, double x12, double x13, double x21,
//                    double x22, double x23, double x31, double x32, double x33,
//                    double x41, double x42, double x43, double result[12][9]);

// __host__ __device__
// void pDmpx_pt_flip(const double3& x0, const double3& x1, const double3& x2, const double3& x3, double d_hatSqrt, MATHUTILS::Matrix12x9d& pDmpx);

// __host__ __device__
// void pDmpx_pe_reflect(double d, double x11, double x12, double x13, double x21,
//                       double x22, double x23, double x31, double x32,
//                       double x33, double result[9][4]);

// __host__ __device__
// void pDmpx_pe_reflect(const double3& x0,
//                       const double3& x1,
//                       const double3& x2,
//                       double d_hatSqrt, MATHUTILS::Matrix9x4d& pDmpx);

// __host__ __device__
// void pDmpx_pe(double d, double x11, double x12, double x13, double x21,
//               double x22, double x23, double x31, double x32, double x33,
//               double result[9][4]);

// __host__ __device__
// void pDmpx_pe(const double3& x0,
//               const double3& x1,
//               const double3& x2,
//               double d_hatSqrt, MATHUTILS::Matrix9x4d& pDmpx);

// __host__ __device__
// void pDmpx_pp_reflect(double d, double x11, double x12, double x13, double x21,
//                       double x22, double x23, double result[6]);
// __host__ __device__
// void pDmpx_pp_reflect(const double3& x0,
//                       const double3& x1,
//                       double d_hatSqrt, MATHUTILS::Vector6& pDmpx);

// __host__ __device__
// void pDmpx_pp(double d, double x11, double x12, double x13, double x21,
//               double x22, double x23, double result[6]);

// __host__ __device__
// void pDmpx_pp(const double3& x0,
//               const double3& x1,
//               double d_hatSqrt, MATHUTILS::Vector6& pDmpx);


// }; // GIPCPDERIV
