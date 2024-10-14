
#pragma once

#include <cmath>
#include <vector_functions.h>

#ifdef USE_DOUBLE_PRECISION
    using Scalar = double;
    using Scalar2 = double2;
    using Scalar3 = double3;
    using Scalar4 = double4;
    __host__ __device__ inline Scalar2 make_Scalar2(double x, double y) {
        return make_double2(x, y);
    }
    __host__ __device__ inline Scalar3 make_Scalar3(double x, double y, double z) {
        return make_double3(x, y, z);
    }
    __host__ __device__ inline Scalar4 make_Scalar4(double x, double y, double z, double w) {
        return make_double4(x, y, z, w);
    }
#else
    using Scalar = float;
    using Scalar2 = float2;
    using Scalar3 = float3;
    using Scalar4 = float4;
    __host__ __device__ inline Scalar2 make_Scalar2(float x, float y) {
        return make_float2(x, y);
    }
    __host__ __device__ inline Scalar3 make_Scalar3(float x, float y, float z) {
        return make_float3(x, y, z);
    }
    __host__ __device__ inline Scalar4 make_Scalar4(float x, float y, float z, float w) {
        return make_float4(x, y, z, w);
    }
#endif


namespace __MATHUTILS__ {

struct Vector4S {
    Scalar v[4];
};

struct Vector6S {
    Scalar v[6];
};

struct Vector9S {
    Scalar v[9];
};

struct Vector12S {
    Scalar v[12];
};

struct Matrix2x2S {
    Scalar m[2][2];
};

struct Matrix3x3S {
    Scalar m[3][3];
};

struct Matrix4x4S {
    Scalar m[4][4];
};

struct Matrix6x6S {
    Scalar m[6][6];
};

struct Matrix9x9S {
    Scalar m[9][9];
};

struct Matrix3x6S {
    Scalar m[3][6];
};

struct Matrix6x3S {
    Scalar m[6][3];
};

struct Matrix3x2S {
    Scalar m[3][2];
};

struct Matrix12x12S {
    Scalar m[12][12];
};

struct Matrix24x24S {
    Scalar m[24][24];
};

struct Matrix36x36S {
    Scalar m[36][36];
};

struct Matrix96x96S {
    Scalar m[96][96];
};

struct Matrix9x2S {
    Scalar m[9][2];
};

struct Matrix6x2S {
    Scalar m[6][2];
};

struct Matrix12x2S {
    Scalar m[12][2];
};

struct Matrix9x12S {
    Scalar m[9][12];
};

struct Matrix12x9S {
    Scalar m[12][9];
};

struct Matrix12x6S {
    Scalar m[12][6];
};

struct Matrix6x12S {
    Scalar m[6][12];
};

struct Matrix12x4S {
    Scalar m[12][4];
};

struct Matrix9x4S {
    Scalar m[9][4];
};

struct Matrix4x9S {
    Scalar m[4][9];
};

struct Matrix6x9S {
    Scalar m[6][9];
};

struct Matrix9x6S {
    Scalar m[9][6];
};

struct Matrix2x3S {
    Scalar m[2][3];
};

struct MasMatrixSym {
    Matrix3x3S M[32 * (32 + 1) / 2];
};


}  // namespace __MATHUTILS__


namespace __MATHUTILS__ {

__device__ __host__ Scalar __PI();

__device__ __host__ void __init_Mat3x3(Matrix3x3S& M, const Scalar& val);

__device__ __host__ void __init_Mat6x6(Matrix6x6S& M, const Scalar& val);

__device__ __host__ void __init_Mat9x9(Matrix9x9S& M, const Scalar& val);

__device__ __host__ void __identify_Mat3x3(Matrix3x3S& M);

__device__ __host__ void __identify_Mat6x6(Matrix6x6S& M);

__device__ __host__ void __identify_Mat9x9(Matrix9x9S& M);

__device__ __host__ Scalar __mabs(const Scalar& a);

__device__ __host__ Scalar __norm(const Scalar3& n);

__device__ __host__ Scalar3 __s_vec_multiply(const Scalar3& a, Scalar b);

__device__ __host__ Scalar2 __s_vec_multiply(const Scalar2& a, Scalar b);

__device__ __host__ Scalar3 __normalized(Scalar3 n);

__device__ __host__ Scalar3 __add(Scalar3 a, Scalar3 b);

__device__ __host__ Vector9S __add9(const Vector9S& a, const Vector9S& b);

__device__ __host__ Vector6S __add6(const Vector6S& a, const Vector6S& b);

__device__ __host__ Scalar3 __minus(Scalar3 a, Scalar3 b);

__device__ __host__ Scalar2 __minus_v2(Scalar2 a, Scalar2 b);

__device__ __host__ Scalar3 __v_vec_multiply(Scalar3 a, Scalar3 b);

__device__ __host__ Scalar __v2_vec_multiply(Scalar2 a, Scalar2 b);

__device__ __host__ Scalar __squaredNorm(Scalar3 a);

__device__ __host__ Scalar __squaredNorm(Scalar2 a);

__device__ __host__ void __M_Mat_multiply(const Matrix3x3S& A,
                                          const Matrix3x3S& B,
                                          Matrix3x3S& output);

__device__ __host__ Matrix3x3S __M_Mat_multiply(const Matrix3x3S& A,
                                                const Matrix3x3S& B);

__device__ __host__ Matrix2x2S __M2x2_Mat2x2_multiply(const Matrix2x2S& A,
                                                      const Matrix2x2S& B);

__device__ __host__ Scalar __Mat_Trace(const Matrix3x3S& A);

__device__ __host__ Scalar3 __v_M_multiply(const Scalar3& n,
                                           const Matrix3x3S& A);

__device__ __host__ Scalar3 __M_v_multiply(const Matrix3x3S& A,
                                           const Scalar3& n);

__device__ __host__ Scalar3 __M3x2_v2_multiply(const Matrix3x2S& A,
                                               const Scalar2& n);

__device__ __host__ Matrix3x2S __S_Mat3x2_multiply(const Matrix3x2S& A,
                                                   const Scalar& b);

__device__ __host__ Matrix3x2S __Mat3x2_add(const Matrix3x2S& A,
                                            const Matrix3x2S& B);

__device__ __host__ Vector12S __M12x9_v9_multiply(const Matrix12x9S& A,
                                                 const Vector9S& n);

__device__ __host__ Vector12S __M12x6_v6_multiply(const Matrix12x6S& A,
                                                 const Vector6S& n);

__device__ __host__ Vector6S __M6x3_v3_multiply(const Matrix6x3S& A,
                                               const Scalar3& n);

__device__ __host__ Scalar2 __M2x3_v3_multiply(const Matrix2x3S& A,
                                               const Scalar3& n);

__device__ __host__ Vector9S __M9x6_v6_multiply(const Matrix9x6S& A,
                                               const Vector6S& n);

__device__ __host__ Vector12S __M12x12_v12_multiply(const Matrix12x12S& A,
                                                   const Vector12S& n);

__device__ __host__ Vector9S __M9x9_v9_multiply(const Matrix9x9S& A,
                                               const Vector9S& n);

__device__ __host__ Vector6S __M6x6_v6_multiply(const Matrix6x6S& A,
                                               const Vector6S& n);

__device__ __host__ Matrix9x9S __S_Mat9x9_multiply(const Matrix9x9S& A,
                                                   const Scalar& B);

__device__ __host__ Matrix6x6S __S_Mat6x6_multiply(const Matrix6x6S& A,
                                                   const Scalar& B);

__device__ __host__ Scalar __v_vec_dot(const Scalar3& a, const Scalar3& b);

__device__ __host__ Scalar3 __v_vec_cross(Scalar3 a, Scalar3 b);

__device__ __host__ Matrix3x3S __v_vec_toMat(Scalar3 a, Scalar3 b);

__device__ __host__ Matrix2x2S __v2_vec2_toMat2x2(Scalar2 a, Scalar2 b);

__device__ __host__ Matrix2x2S __s_Mat2x2_multiply(Matrix2x2S A, Scalar b);

__device__ __host__ Matrix2x2S __Mat2x2_minus(Matrix2x2S A, Matrix2x2S B);

__device__ __host__ Matrix3x3S __Mat3x3_minus(Matrix3x3S A, Matrix3x3S B);

__device__ __host__ Matrix9x9S __v9_vec9_toMat9x9(const Vector9S& a,
                                                  const Vector9S& b,
                                                  const Scalar& coe = 1);

__device__ __host__ Matrix6x6S __v6_vec6_toMat6x6(Vector6S a, Vector6S b);

__device__ __host__ Vector9S __s_vec9_multiply(Vector9S a, Scalar b);

__device__ __host__ Vector12S __s_vec12_multiply(Vector12S a, Scalar b);

__device__ __host__ Vector6S __s_vec6_multiply(Vector6S a, Scalar b);

__device__ __host__ void __Mat_add(const Matrix3x3S& A, const Matrix3x3S& B,
                                   Matrix3x3S& output);

__device__ __host__ void __Mat_add(const Matrix6x6S& A, const Matrix6x6S& B,
                                   Matrix6x6S& output);

__device__ __host__ Matrix3x3S __Mat_add(const Matrix3x3S& A,
                                         const Matrix3x3S& B);

__device__ __host__ Matrix2x2S __Mat2x2_add(const Matrix2x2S& A,
                                            const Matrix2x2S& B);

__device__ __host__ Matrix9x9S __Mat9x9_add(const Matrix9x9S& A,
                                            const Matrix9x9S& B);

__device__ __host__ Matrix9x12S __Mat9x12_add(const Matrix9x12S& A,
                                              const Matrix9x12S& B);

__device__ __host__ Matrix6x12S __Mat6x12_add(const Matrix6x12S& A,
                                              const Matrix6x12S& B);

__device__ __host__ Matrix6x9S __Mat6x9_add(const Matrix6x9S& A,
                                            const Matrix6x9S& B);

__device__ __host__ Matrix3x6S __Mat3x6_add(const Matrix3x6S& A,
                                            const Matrix3x6S& B);

__device__ __host__ void __set_Mat_identity(Matrix2x2S& M);

__device__ __host__ void __set_Mat_val(Matrix3x3S& M, const Scalar& a00,
                                       const Scalar& a01, const Scalar& a02,
                                       const Scalar& a10, const Scalar& a11,
                                       const Scalar& a12, const Scalar& a20,
                                       const Scalar& a21, const Scalar& a22);

__device__ __host__ void __set_Mat_val_row(Matrix3x3S& M, const Scalar3& row0,
                                           const Scalar3& row1,
                                           const Scalar3& row2);

__device__ __host__ void __set_Mat_val_column(Matrix3x3S& M,
                                              const Scalar3& col0,
                                              const Scalar3& col1,
                                              const Scalar3& col2);

__device__ __host__ void __set_Mat3x2_val_column(Matrix3x2S& M,
                                                 const Scalar3& col0,
                                                 const Scalar3& col1);

__device__ __host__ void __set_Mat2x2_val_column(Matrix2x2S& M,
                                                 const Scalar2& col0,
                                                 const Scalar2& col1);

__device__ __host__ void __init_Mat9x12_val(Matrix9x12S& M, const Scalar& val);

__device__ __host__ void __init_Mat6x12_val(Matrix6x12S& M, const Scalar& val);

__device__ __host__ void __init_Mat6x9_val(Matrix6x9S& M, const Scalar& val);

__device__ __host__ void __init_Mat3x6_val(Matrix3x6S& M, const Scalar& val);

__device__ __host__ Matrix3x3S __S_Mat_multiply(const Matrix3x3S& A,
                                                const Scalar& B);

__device__ __host__ Matrix3x3S __Transpose3x3(Matrix3x3S input);

__device__ __host__ Matrix12x9S __Transpose9x12(const Matrix9x12S& input);

__device__ __host__ Matrix2x3S __Transpose3x2(const Matrix3x2S& input);

__device__ __host__ Matrix9x12S __Transpose12x9(const Matrix12x9S& input);

__device__ __host__ Matrix12x6S __Transpose6x12(const Matrix6x12S& input);

__device__ __host__ Matrix9x6S __Transpose6x9(const Matrix6x9S& input);

__device__ __host__ Matrix6x3S __Transpose3x6(const Matrix3x6S& input);

__device__ __host__ Matrix12x9S __M12x9_M9x9_Multiply(const Matrix12x9S& A,
                                                      const Matrix9x9S& B);

__device__ __host__ Matrix12x6S __M12x6_M6x6_Multiply(const Matrix12x6S& A,
                                                      const Matrix6x6S& B);

__device__ __host__ Matrix9x6S __M9x6_M6x6_Multiply(const Matrix9x6S& A,
                                                    const Matrix6x6S& B);

__device__ __host__ Matrix6x3S __M6x3_M3x3_Multiply(const Matrix6x3S& A,
                                                    const Matrix3x3S& B);

__device__ __host__ Matrix3x2S __M3x2_M2x2_Multiply(const Matrix3x2S& A,
                                                    const Matrix2x2S& B);

__device__ __host__ Matrix12x12S __M12x9_M9x12_Multiply(const Matrix12x9S& A,
                                                        const Matrix9x12S& B);

__device__ __host__ Matrix12x2S __M12x2_M2x2_Multiply(const Matrix12x2S& A,
                                                      const Matrix2x2S& B);

__device__ __host__ Matrix9x2S __M9x2_M2x2_Multiply(const Matrix9x2S& A,
                                                    const Matrix2x2S& B);

__device__ __host__ Matrix6x2S __M6x2_M2x2_Multiply(const Matrix6x2S& A,
                                                    const Matrix2x2S& B);

__device__ __host__ Matrix12x12S __M12x2_M12x2T_Multiply(const Matrix12x2S& A,
                                                         const Matrix12x2S& B);

__device__ __host__ Matrix9x9S __M9x2_M9x2T_Multiply(const Matrix9x2S& A,
                                                     const Matrix9x2S& B);

__device__ __host__ Matrix6x6S __M6x2_M6x2T_Multiply(const Matrix6x2S& A,
                                                     const Matrix6x2S& B);

__device__ __host__ Matrix12x12S __M12x6_M6x12_Multiply(const Matrix12x6S& A,
                                                        const Matrix6x12S& B);

__device__ __host__ Matrix9x9S __M9x6_M6x9_Multiply(const Matrix9x6S& A,
                                                    const Matrix6x9S& B);

__device__ __host__ Matrix6x6S __M6x3_M3x6_Multiply(const Matrix6x3S& A,
                                                    const Matrix3x6S& B);

__device__ __host__ Matrix12x12S __s_M12x12_Multiply(const Matrix12x12S& A,
                                                     const Scalar& B);

__device__ __host__ Matrix9x9S __s_M9x9_Multiply(const Matrix9x9S& A,
                                                 const Scalar& B);

__device__ __host__ Matrix6x6S __s_M6x6_Multiply(const Matrix6x6S& A,
                                                 const Scalar& B);

__device__ __host__ void __Determiant(const Matrix3x3S& input,
                                      Scalar& determinant);

__device__ __host__ Scalar __Determiant(const Matrix3x3S& input);

__device__ __host__ void __Inverse(const Matrix3x3S& input, Matrix3x3S& output);

__device__ __host__ void __Inverse2x2(const Matrix2x2S& input,
                                      Matrix2x2S& output);

__device__ __host__ Scalar __f(const Scalar& x, const Scalar& a,
                               const Scalar& b, const Scalar& c,
                               const Scalar& d);

__device__ __host__ Scalar __df(const Scalar& x, const Scalar& a,
                                const Scalar& b, const Scalar& c);

__device__ __host__ void __NewtonSolverForCubicEquation(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS = 1e-6);

__device__ __host__ void __NewtonSolverForCubicEquation_satbleNeohook(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS = 1e-6);

__device__ __host__ void __SolverForCubicEquation(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS = 1e-6);

__device__ __host__ Vector9S __Mat3x3_to_vec9_Scalar(const Matrix3x3S& F);

__device__ __host__ void __normalized_vec9_Scalar(Vector9S& v9);

__device__ __host__ void __normalized_vec6_Scalar(Vector6S& v6);

__device__ __host__ Vector6S __Mat3x2_to_vec6_Scalar(const Matrix3x2S& F);

__device__ __host__ Matrix3x3S __vec9_to_Mat3x3_Scalar(const Scalar vec9[9]);

__device__ __host__ Matrix2x2S __vec4_to_Mat2x2_Scalar(const Scalar vec4[4]);

__device__ void SVD(const Matrix3x3S& F, Matrix3x3S& Uout, Matrix3x3S& Vout,
                    Matrix3x3S& Sigma);

__device__ __host__ void __makePD2x2(const Scalar& a00, const Scalar& a01,
                                     const Scalar& a10, const Scalar& a11,
                                     Scalar eigenValues[2], int& num,
                                     Scalar2 eigenVectors[2],
                                     Scalar eps = 1e-32);

__device__ __host__ void __M12x9_S9x9_MT9x12_Multiply(const Matrix12x9S& A,
                                                      const Matrix9x9S& B,
                                                      Matrix12x12S& output);

__device__ __host__ void __M9x4_S4x4_MT4x9_Multiply(const Matrix9x4S& A,
                                                    const Matrix4x4S& B,
                                                    Matrix9x9S& output);

__device__ __host__ Vector4S __s_vec4_multiply(Vector4S a, Scalar b);

__device__ __host__ Vector9S __M9x4_v4_multiply(const Matrix9x4S& A,
                                               const Vector4S& n);

__device__ __host__ Matrix4x4S __S_Mat4x4_multiply(const Matrix4x4S& A,
                                                   const Scalar& B);

__device__ __host__ Matrix4x4S __v4_vec4_toMat4x4(Vector4S a, Vector4S b);

__device__ __host__ void __s_M_Mat_MT_multiply(const Matrix3x3S& A,
                                               const Matrix3x3S& B,
                                               const Matrix3x3S& C,
                                               const Scalar& coe,
                                               Matrix3x3S& output);

template<class T>
inline __device__ __host__ T __m_min(T a, T b) {
    return a < b ? a : b;
}

template <class T>
inline __device__ __host__ T __m_max(T a, T b) {
    return a < b ? b : a;
}


__device__
void _d_PP(const Scalar3& v0, const Scalar3& v1, Scalar& d);

__device__
void _d_PT(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d);

__device__
void _d_PE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, Scalar& d);

__device__
void _d_EE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d);

__device__
void _d_EEParallel(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d);

__device__
Scalar _compute_epx(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3);

__device__
Scalar _compute_epx_cp(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3);

__device__ __host__
Scalar calculateVolume(const Scalar3* vertexes, const uint4& index);

__device__ __host__
Scalar calculateArea(const Scalar3* vertexes, const uint3& index);

}  // namespace __MATHUTILS__



namespace __MATHUTILS__ {

__device__ void perform_reduction(
    Scalar temp, Scalar* tep, Scalar* squeue, int numbers, int idof, int blockDimX, int gridDimX, int blockIdX
);

__global__ void _reduct_max_Scalar(Scalar* _Scalar1Dim, int number);

__global__ void _reduct_min_Scalar(Scalar* _Scalar1Dim, int number);

__global__ void __add_reduction(Scalar* mem, int numbers);

__global__ void _reduct_max_Scalar2(Scalar2* _Scalar2Dim, int number);

__global__ void _reduct_max_Scalar3_to_Scalar(const Scalar3* _Scalar3Dim, Scalar* _Scalar1Dim, int number);

}  // namespace __MATHUTILS__

