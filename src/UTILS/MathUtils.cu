
#include "MathUtils.cuh"



namespace __MATHUTILS__ {

__device__ __host__ Scalar __PI() {
    return 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899;
}

__device__ __host__ void __init_Mat3x3(Matrix3x3S& M, const Scalar& val) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat6x6(Matrix6x6S& M, const Scalar& val) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat9x9(Matrix9x9S& M, const Scalar& val) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __identify_Mat3x3(Matrix3x3S& M) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                M.m[i][j] = 1;
            } else {
                M.m[i][j] = 0;
            }
        }
    }
}

__device__ __host__ void __identify_Mat6x6(Matrix6x6S& M) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                M.m[i][j] = 1;
            } else {
                M.m[i][j] = 0;
            }
        }
    }
}

__device__ __host__ void __identify_Mat9x9(Matrix9x9S& M) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (i == j) {
                M.m[i][j] = 1;
            } else {
                M.m[i][j] = 0;
            }
        }
    }
}

__device__ __host__ Scalar __mabs(const Scalar& a) { return a > 0 ? a : -a; }

__device__ __host__ Scalar __norm(const Scalar3& n) {
    return sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
}

__device__ __host__ Scalar3 __s_vec_multiply(const Scalar3& a, Scalar b) {
    return make_Scalar3(a.x * b, a.y * b, a.z * b);
}

__device__ __host__ Scalar2 __s_vec_multiply(const Scalar2& a, Scalar b) {
    return make_Scalar2(a.x * b, a.y * b);
}

__device__ __host__ Scalar3 __normalized(Scalar3 n) {
    Scalar norm = __norm(n);
    norm = 1 / norm;
    return __s_vec_multiply(n, norm);
}

__device__ __host__ Scalar3 __add(Scalar3 a, Scalar3 b) {
    return make_Scalar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ Vector9S __add9(const Vector9S& a, const Vector9S& b) {
    Vector9S V;
    for (int i = 0; i < 9; i++) {
        V.v[i] = a.v[i] + b.v[i];
    }
    return V;
}

__device__ __host__ Vector6S __add6(const Vector6S& a, const Vector6S& b) {
    Vector6S V;
    for (int i = 0; i < 6; i++) {
        V.v[i] = a.v[i] + b.v[i];
    }
    return V;
}

__device__ __host__ Scalar3 __minus(Scalar3 a, Scalar3 b) {
    return make_Scalar3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ Scalar2 __minus_v2(Scalar2 a, Scalar2 b) {
    return make_Scalar2(a.x - b.x, a.y - b.y);
}

__device__ __host__ Scalar3 __v_vec_multiply(Scalar3 a, Scalar3 b) {
    return make_Scalar3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ Scalar __v2_vec_multiply(Scalar2 a, Scalar2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ __host__ Scalar __squaredNorm(Scalar3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ __host__ Scalar __squaredNorm(Scalar2 a) {
    return a.x * a.x + a.y * a.y;
}

__device__ __host__ void __M_Mat_multiply(const Matrix3x3S& A,
                                          const Matrix3x3S& B,
                                          Matrix3x3S& output) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
}

__device__ __host__ Matrix3x3S __M_Mat_multiply(const Matrix3x3S& A,
                                                const Matrix3x3S& B) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix2x2S __M2x2_Mat2x2_multiply(const Matrix2x2S& A,
                                                      const Matrix2x2S& B) {
    Matrix2x2S output;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Scalar __Mat_Trace(const Matrix3x3S& A) {
    return A.m[0][0] + A.m[1][1] + A.m[2][2];
}

__device__ __host__ Scalar3 __v_M_multiply(const Scalar3& n,
                                           const Matrix3x3S& A) {
    Scalar x = A.m[0][0] * n.x + A.m[1][0] * n.y + A.m[2][0] * n.z;
    Scalar y = A.m[0][1] * n.x + A.m[1][1] * n.y + A.m[2][1] * n.z;
    Scalar z = A.m[0][2] * n.x + A.m[1][2] * n.y + A.m[2][2] * n.z;
    return make_Scalar3(x, y, z);
}

__device__ __host__ Scalar3 __M_v_multiply(const Matrix3x3S& A,
                                           const Scalar3& n) {
    Scalar x = A.m[0][0] * n.x + A.m[0][1] * n.y + A.m[0][2] * n.z;
    Scalar y = A.m[1][0] * n.x + A.m[1][1] * n.y + A.m[1][2] * n.z;
    Scalar z = A.m[2][0] * n.x + A.m[2][1] * n.y + A.m[2][2] * n.z;
    return make_Scalar3(x, y, z);
}

__device__ __host__ Scalar3 __M3x2_v2_multiply(const Matrix3x2S& A,
                                               const Scalar2& n) {
    Scalar x = A.m[0][0] * n.x + A.m[0][1] * n.y;  // +A.m[0][2] * n.z;
    Scalar y = A.m[1][0] * n.x + A.m[1][1] * n.y;  // +A.m[1][2] * n.z;
    Scalar z = A.m[2][0] * n.x + A.m[2][1] * n.y;  // +A.m[2][2] * n.z;
    return make_Scalar3(x, y, z);
}

__device__ __host__ Matrix3x2S __Mat3x2_add(const Matrix3x2S& A,
                                            const Matrix3x2S& B) {
    Matrix3x2S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[i][j] = A.m[i][j] + B.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix3x2S __S_Mat3x2_multiply(const Matrix3x2S& A,
                                                   const Scalar& b) {
    Matrix3x2S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[i][j] = A.m[i][j] * b;
        }
    }
    return output;
}

__device__ __host__ Vector12S __M12x9_v9_multiply(const Matrix12x9S& A,
                                                 const Vector9S& n) {
    Vector12S v12;
    for (int i = 0; i < 12; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 9; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v12.v[i] = temp;
    }
    return v12;
}

__device__ __host__ Vector12S __M12x6_v6_multiply(const Matrix12x6S& A,
                                                 const Vector6S& n) {
    Vector12S v12;
    for (int i = 0; i < 12; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 6; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v12.v[i] = temp;
    }
    return v12;
}

__device__ __host__ Vector6S __M6x3_v3_multiply(const Matrix6x3S& A,
                                               const Scalar3& n) {
    Vector6S v6;
    for (int i = 0; i < 6; i++) {
        Scalar temp = A.m[i][0] * n.x;
        temp += A.m[i][1] * n.y;
        temp += A.m[i][2] * n.z;

        v6.v[i] = temp;
    }
    return v6;
}

__device__ __host__ Scalar2 __M2x3_v3_multiply(const Matrix2x3S& A,
                                               const Scalar3& n) {
    Scalar2 output;
    output.x = A.m[0][0] * n.x + A.m[0][1] * n.y + A.m[0][2] * n.z;
    output.y = A.m[1][0] * n.x + A.m[1][1] * n.y + A.m[1][2] * n.z;
    return output;
}

__device__ __host__ Vector9S __M9x6_v6_multiply(const Matrix9x6S& A,
                                               const Vector6S& n) {
    Vector9S v9;
    for (int i = 0; i < 9; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 6; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v9.v[i] = temp;
    }
    return v9;
}

__device__ __host__ Vector12S __M12x12_v12_multiply(const Matrix12x12S& A,
                                                   const Vector12S& n) {
    Vector12S v12;
    for (int i = 0; i < 12; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 12; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v12.v[i] = temp;
    }
    return v12;
}

__device__ __host__ Vector9S __M9x9_v9_multiply(const Matrix9x9S& A,
                                               const Vector9S& n) {
    Vector9S v9;
    for (int i = 0; i < 9; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 9; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v9.v[i] = temp;
    }
    return v9;
}

__device__ __host__ Vector6S __M6x6_v6_multiply(const Matrix6x6S& A,
                                               const Vector6S& n) {
    Vector6S v6;
    for (int i = 0; i < 6; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 6; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v6.v[i] = temp;
    }
    return v6;
}

__device__ __host__ Matrix9x9S __S_Mat9x9_multiply(const Matrix9x9S& A,
                                                   const Scalar& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            output.m[i][j] = A.m[i][j] * B;
        }
    }
    return output;
}

__device__ __host__ Matrix6x6S __S_Mat6x6_multiply(const Matrix6x6S& A,
                                                   const Scalar& B) {
    Matrix6x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            output.m[i][j] = A.m[i][j] * B;
        }
    }
    return output;
}

__device__ __host__ Scalar __v_vec_dot(const Scalar3& a, const Scalar3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ Scalar3 __v_vec_cross(Scalar3 a, Scalar3 b) {
    return make_Scalar3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__device__ __host__ Matrix3x3S __v_vec_toMat(Scalar3 a, Scalar3 b) {
    Matrix3x3S M;
    M.m[0][0] = a.x * b.x;
    M.m[0][1] = a.x * b.y;
    M.m[0][2] = a.x * b.z;
    M.m[1][0] = a.y * b.x;
    M.m[1][1] = a.y * b.y;
    M.m[1][2] = a.y * b.z;
    M.m[2][0] = a.z * b.x;
    M.m[2][1] = a.z * b.y;
    M.m[2][2] = a.z * b.z;
    return M;
}

__device__ __host__ Matrix2x2S __v2_vec2_toMat2x2(Scalar2 a, Scalar2 b) {
    Matrix2x2S M;
    M.m[0][0] = a.x * b.x;
    M.m[0][1] = a.x * b.y;
    M.m[1][0] = a.y * b.x;
    M.m[1][1] = a.y * b.y;
    return M;
}

__device__ __host__ Matrix2x2S __s_Mat2x2_multiply(Matrix2x2S A, Scalar b) {
    Matrix2x2S output;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[i][j] = A.m[i][j] * b;
        }
    }
    return output;
}

__device__ __host__ Matrix2x2S __Mat2x2_minus(Matrix2x2S A, Matrix2x2S B) {
    Matrix2x2S output;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[i][j] = A.m[i][j] - B.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix3x3S __Mat3x3_minus(Matrix3x3S A, Matrix3x3S B) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            output.m[i][j] = A.m[i][j] - B.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix9x9S __v9_vec9_toMat9x9(const Vector9S& a,
                                                  const Vector9S& b,
                                                  const Scalar& coe) {
    Matrix9x9S M;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            M.m[i][j] = a.v[i] * b.v[j] * coe;
        }
    }
    return M;
}

__device__ __host__ Matrix6x6S __v6_vec6_toMat6x6(Vector6S a, Vector6S b) {
    Matrix6x6S M;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            M.m[i][j] = a.v[i] * b.v[j];
        }
    }
    return M;
}

__device__ __host__ Vector9S __s_vec9_multiply(Vector9S a, Scalar b) {
    Vector9S V;
    for (int i = 0; i < 9; i++) V.v[i] = a.v[i] * b;
    return V;
}

__device__ __host__ Vector12S __s_vec12_multiply(Vector12S a, Scalar b) {
    Vector12S V;
    for (int i = 0; i < 12; i++) V.v[i] = a.v[i] * b;
    return V;
}

__device__ __host__ Vector6S __s_vec6_multiply(Vector6S a, Scalar b) {
    Vector6S V;
    for (int i = 0; i < 6; i++) V.v[i] = a.v[i] * b;
    return V;
}

__device__ __host__ void __Mat_add(const Matrix3x3S& A, const Matrix3x3S& B,
                                   Matrix3x3S& output) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
}

__device__ __host__ void __Mat_add(const Matrix6x6S& A, const Matrix6x6S& B,
                                   Matrix6x6S& output) {
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
}

__device__ __host__ Matrix3x3S __Mat_add(const Matrix3x3S& A,
                                         const Matrix3x3S& B) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix2x2S __Mat2x2_add(const Matrix2x2S& A,
                                            const Matrix2x2S& B) {
    Matrix2x2S output;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix9x9S __Mat9x9_add(const Matrix9x9S& A,
                                            const Matrix9x9S& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix9x12S __Mat9x12_add(const Matrix9x12S& A,
                                              const Matrix9x12S& B) {
    Matrix9x12S output;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 12; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix6x12S __Mat6x12_add(const Matrix6x12S& A,
                                              const Matrix6x12S& B) {
    Matrix6x12S output;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 12; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix6x9S __Mat6x9_add(const Matrix6x9S& A,
                                            const Matrix6x9S& B) {
    Matrix6x9S output;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 9; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix3x6S __Mat3x6_add(const Matrix3x6S& A,
                                            const Matrix3x6S& B) {
    Matrix3x6S output;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 6; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ void __set_Mat_identity(Matrix2x2S& M) {
    M.m[0][0] = 1;
    M.m[1][0] = 0;
    M.m[0][1] = 0;
    M.m[1][1] = 1;
}

__device__ __host__ void __set_Mat_val(Matrix3x3S& M, const Scalar& a00,
                                       const Scalar& a01, const Scalar& a02,
                                       const Scalar& a10, const Scalar& a11,
                                       const Scalar& a12, const Scalar& a20,
                                       const Scalar& a21, const Scalar& a22) {
    M.m[0][0] = a00;
    M.m[0][1] = a01;
    M.m[0][2] = a02;
    M.m[1][0] = a10;
    M.m[1][1] = a11;
    M.m[1][2] = a12;
    M.m[2][0] = a20;
    M.m[2][1] = a21;
    M.m[2][2] = a22;
}

__device__ __host__ void __set_Mat_val_row(Matrix3x3S& M, const Scalar3& row0,
                                           const Scalar3& row1,
                                           const Scalar3& row2) {
    M.m[0][0] = row0.x;
    M.m[0][1] = row0.y;
    M.m[0][2] = row0.z;
    M.m[1][0] = row1.x;
    M.m[1][1] = row1.y;
    M.m[1][2] = row1.z;
    M.m[2][0] = row2.x;
    M.m[2][1] = row2.y;
    M.m[2][2] = row2.z;
}

__device__ __host__ void __set_Mat_val_column(Matrix3x3S& M,
                                              const Scalar3& col0,
                                              const Scalar3& col1,
                                              const Scalar3& col2) {
    M.m[0][0] = col0.x;
    M.m[0][1] = col1.x;
    M.m[0][2] = col2.x;
    M.m[1][0] = col0.y;
    M.m[1][1] = col1.y;
    M.m[1][2] = col2.y;
    M.m[2][0] = col0.z;
    M.m[2][1] = col1.z;
    M.m[2][2] = col2.z;
}

__device__ __host__ void __set_Mat3x2_val_column(Matrix3x2S& M,
                                                 const Scalar3& col0,
                                                 const Scalar3& col1) {
    M.m[0][0] = col0.x;
    M.m[0][1] = col1.x;
    M.m[1][0] = col0.y;
    M.m[1][1] = col1.y;
    M.m[2][0] = col0.z;
    M.m[2][1] = col1.z;
}

__device__ __host__ void __set_Mat2x2_val_column(Matrix2x2S& M,
                                                 const Scalar2& col0,
                                                 const Scalar2& col1) {
    M.m[0][0] = col0.x;
    M.m[0][1] = col1.x;
    M.m[1][0] = col0.y;
    M.m[1][1] = col1.y;
}

__device__ __host__ void __init_Mat9x12_val(Matrix9x12S& M, const Scalar& val) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 12; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat6x12_val(Matrix6x12S& M, const Scalar& val) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 12; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat6x9_val(Matrix6x9S& M, const Scalar& val) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 9; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat3x6_val(Matrix3x6S& M, const Scalar& val) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ Matrix3x3S __S_Mat_multiply(const Matrix3x3S& A,
                                                const Scalar& B) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) output.m[i][j] = A.m[i][j] * B;
    return output;
}

__device__ __host__ Matrix3x3S __Transpose3x3(Matrix3x3S input) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            output.m[i][j] = input.m[j][i];
        }
    }
    return output;
    // output.m[0][0] = input.m[0][0];output.m[0][1] =
    // input.m[1][0];output.m[0][2] = input.m[2][0]; output.m[1][0] =
    // input.m[0][1];output.m[1][1] = input.m[1][1];output.m[1][2] =
    // input.m[2][1]; output.m[2][0] = input.m[0][2];output.m[2][1] =
    // input.m[1][2];output.m[2][2] = input.m[2][2];
}

__device__ __host__ Matrix12x9S __Transpose9x12(const Matrix9x12S& input) {
    Matrix12x9S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 12; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix2x3S __Transpose3x2(const Matrix3x2S& input) {
    Matrix2x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix9x12S __Transpose12x9(const Matrix12x9S& input) {
    Matrix9x12S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 9; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix12x6S __Transpose6x12(const Matrix6x12S& input) {
    Matrix12x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 12; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix9x6S __Transpose6x9(const Matrix6x9S& input) {
    Matrix9x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 9; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix6x3S __Transpose3x6(const Matrix3x6S& input) {
    Matrix6x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix12x9S __M12x9_M9x9_Multiply(const Matrix12x9S& A,
                                                      const Matrix9x9S& B) {
    Matrix12x9S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 9; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x6S __M12x6_M6x6_Multiply(const Matrix12x6S& A,
                                                      const Matrix6x6S& B) {
    Matrix12x6S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 6; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 6; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix9x6S __M9x6_M6x6_Multiply(const Matrix9x6S& A,
                                                    const Matrix6x6S& B) {
    Matrix9x6S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 6; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 6; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix6x3S __M6x3_M3x3_Multiply(const Matrix6x3S& A,
                                                    const Matrix3x3S& B) {
    Matrix6x3S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix3x2S __M3x2_M2x2_Multiply(const Matrix3x2S& A,
                                                    const Matrix2x2S& B) {
    Matrix3x2S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x12S __M12x9_M9x12_Multiply(const Matrix12x9S& A,
                                                        const Matrix9x12S& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 9; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x2S __M12x2_M2x2_Multiply(const Matrix12x2S& A,
                                                      const Matrix2x2S& B) {
    Matrix12x2S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix9x2S __M9x2_M2x2_Multiply(const Matrix9x2S& A,
                                                    const Matrix2x2S& B) {
    Matrix9x2S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix6x2S __M6x2_M2x2_Multiply(const Matrix6x2S& A,
                                                    const Matrix2x2S& B) {
    Matrix6x2S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x12S __M12x2_M12x2T_Multiply(const Matrix12x2S& A,
                                                         const Matrix12x2S& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix9x9S __M9x2_M9x2T_Multiply(const Matrix9x2S& A,
                                                     const Matrix9x2S& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix6x6S __M6x2_M6x2T_Multiply(const Matrix6x2S& A,
                                                     const Matrix6x2S& B) {
    Matrix6x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x12S __M12x6_M6x12_Multiply(const Matrix12x6S& A,
                                                        const Matrix6x12S& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 6; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix9x9S __M9x6_M6x9_Multiply(const Matrix9x6S& A,
                                                    const Matrix6x9S& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 6; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix6x6S __M6x3_M3x6_Multiply(const Matrix6x3S& A,
                                                    const Matrix3x6S& B) {
    Matrix6x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x12S __s_M12x12_Multiply(const Matrix12x12S& A,
                                                     const Scalar& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++) output.m[i][j] = A.m[i][j] * B;
    return output;
}

__device__ __host__ Matrix9x9S __s_M9x9_Multiply(const Matrix9x9S& A,
                                                 const Scalar& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++) output.m[i][j] = A.m[i][j] * B;
    return output;
}

__device__ __host__ Matrix6x6S __s_M6x6_Multiply(const Matrix6x6S& A,
                                                 const Scalar& B) {
    Matrix6x6S output;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) output.m[i][j] = A.m[i][j] * B;
    return output;
}

__device__ __host__ void __Determiant(const Matrix3x3S& input,
                                      Scalar& determinant) {
    determinant = input.m[0][0] * input.m[1][1] * input.m[2][2] +
                  input.m[1][0] * input.m[2][1] * input.m[0][2] +
                  input.m[2][0] * input.m[0][1] * input.m[1][2] -
                  input.m[2][0] * input.m[1][1] * input.m[0][2] -
                  input.m[0][0] * input.m[1][2] * input.m[2][1] -
                  input.m[0][1] * input.m[1][0] * input.m[2][2];
}

__device__ __host__ Scalar __Determiant(const Matrix3x3S& input) {
    return input.m[0][0] * input.m[1][1] * input.m[2][2] +
           input.m[1][0] * input.m[2][1] * input.m[0][2] +
           input.m[2][0] * input.m[0][1] * input.m[1][2] -
           input.m[2][0] * input.m[1][1] * input.m[0][2] -
           input.m[0][0] * input.m[1][2] * input.m[2][1] -
           input.m[0][1] * input.m[1][0] * input.m[2][2];
}

__device__ __host__ void __Inverse(const Matrix3x3S& input,
                                   Matrix3x3S& result) {
    Scalar eps = 1e-15;
    const int dim = 3;
    Scalar mat[dim][dim * 2];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < 2 * dim; j++) {
            if (j < dim) {
                mat[i][j] = input.m[i][j];  //[i, j];
            } else {
                mat[i][j] = j - dim == i ? 1 : 0;
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        if (abs(mat[i][i]) < eps) {
            int j;
            for (j = i + 1; j < dim; j++) {
                if (abs(mat[j][i]) > eps) break;
            }
            if (j == dim) return;
            for (int r = i; r < 2 * dim; r++) {
                mat[i][r] += mat[j][r];
            }
        }
        Scalar ep = mat[i][i];
        for (int r = i; r < 2 * dim; r++) {
            mat[i][r] /= ep;
        }

        for (int j = i + 1; j < dim; j++) {
            Scalar e = -1 * (mat[j][i] / mat[i][i]);
            for (int r = i; r < 2 * dim; r++) {
                mat[j][r] += e * mat[i][r];
            }
        }
    }

    for (int i = dim - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            Scalar e = -1 * (mat[j][i] / mat[i][i]);
            for (int r = i; r < 2 * dim; r++) {
                mat[j][r] += e * mat[i][r];
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        for (int r = dim; r < 2 * dim; r++) {
            result.m[i][r - dim] = mat[i][r];
        }
    }
}

__device__ __host__ void __Inverse2x2(const Matrix2x2S& input,
                                      Matrix2x2S& result) {
    Scalar eps = 1e-15;
    const int dim = 2;
    Scalar mat[dim][dim * 2];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < 2 * dim; j++) {
            if (j < dim) {
                mat[i][j] = input.m[i][j];  //[i, j];
            } else {
                mat[i][j] = j - dim == i ? 1 : 0;
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        if (abs(mat[i][i]) < eps) {
            int j;
            for (j = i + 1; j < dim; j++) {
                if (abs(mat[j][i]) > eps) break;
            }
            if (j == dim) return;
            for (int r = i; r < 2 * dim; r++) {
                mat[i][r] += mat[j][r];
            }
        }
        Scalar ep = mat[i][i];
        for (int r = i; r < 2 * dim; r++) {
            mat[i][r] /= ep;
        }

        for (int j = i + 1; j < dim; j++) {
            Scalar e = -1 * (mat[j][i] / mat[i][i]);
            for (int r = i; r < 2 * dim; r++) {
                mat[j][r] += e * mat[i][r];
            }
        }
    }

    for (int i = dim - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            Scalar e = -1 * (mat[j][i] / mat[i][i]);
            for (int r = i; r < 2 * dim; r++) {
                mat[j][r] += e * mat[i][r];
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        for (int r = dim; r < 2 * dim; r++) {
            result.m[i][r - dim] = mat[i][r];
        }
    }
}

__device__ __host__ Scalar __f(const Scalar& x, const Scalar& a,
                               const Scalar& b, const Scalar& c,
                               const Scalar& d) {
    Scalar f = a * x * x * x + b * x * x + c * x + d;
    return f;
}

__device__ __host__ Scalar __df(const Scalar& x, const Scalar& a,
                                const Scalar& b, const Scalar& c) {
    Scalar df = 3 * a * x * x + 2 * b * x + c;
    return df;
}

__device__ __host__ void __NewtonSolverForCubicEquation(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS) {
    // Scalar EPS = 1e-6;
    Scalar DX = 0;
    // Scalar results[3];
    num_solutions = 0;
    Scalar specialPoint = -b / a / 3;
    Scalar pos[2];
    int solves = 1;
    Scalar delta = 4 * b * b - 12 * a * c;
    if (delta > 0) {
        pos[0] = (sqrt(delta) - 2 * b) / 6 / a;
        pos[1] = (-sqrt(delta) - 2 * b) / 6 / a;
        Scalar v1 = __f(pos[0], a, b, c, d);
        Scalar v2 = __f(pos[1], a, b, c, d);
        if (__mabs(v1) < EPS * EPS) {
            v1 = 0;
        }
        if (__mabs(v2) < EPS * EPS) {
            v2 = 0;
        }
        Scalar sign = v1 * v2;
        DX = (pos[0] - pos[1]);
        if (sign <= 0) {
            solves = 3;
        } else if (sign > 0) {
            if ((a < 0 && __f(pos[0], a, b, c, d) > 0) ||
                (a > 0 && __f(pos[0], a, b, c, d) < 0)) {
                DX = -DX;
            }
        }
    } else if (delta == 0) {
        if (__mabs(__f(specialPoint, a, b, c, d)) < EPS * EPS) {
            for (int i = 0; i < 3; i++) {
                Scalar tempReuslt = specialPoint;
                results[num_solutions] = tempReuslt;
                num_solutions++;
            }
            return;
        }
        if (a > 0) {
            if (__f(specialPoint, a, b, c, d) > 0) {
                DX = 1;
            } else if (__f(specialPoint, a, b, c, d) < 0) {
                DX = -1;
            }
        } else if (a < 0) {
            if (__f(specialPoint, a, b, c, d) > 0) {
                DX = -1;
            } else if (__f(specialPoint, a, b, c, d) < 0) {
                DX = 1;
            }
        }
    }

    Scalar start = specialPoint - DX;
    Scalar x0 = start;
    // Scalar result[3];

    for (int i = 0; i < solves; i++) {
        Scalar x1 = 0;
        int itCount = 0;
        do {
            if (itCount) x0 = x1;

            x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
            itCount++;

        } while (__mabs(x1 - x0) > EPS && itCount < 100000);
        results[num_solutions] = (x1);
        num_solutions++;
        start = start + DX;
        x0 = start;
    }
}

__device__ __host__ void __NewtonSolverForCubicEquation_satbleNeohook(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS) {
    Scalar DX = 0;
    num_solutions = 0;
    Scalar specialPoint = -b / 3;
    Scalar pos[2];
    int solves = 1;
    Scalar delta = 4 * b * b - 12 * c;
    Scalar sign = -1;
    if (delta > 0) {
        pos[0] = (sqrt(delta) - 2 * b) / 6;
        pos[1] = (-sqrt(delta) - 2 * b) / 6;
        Scalar v1 = __f(pos[0], a, b, c, d);
        Scalar v2 = __f(pos[1], a, b, c, d);
        DX = (pos[0] - pos[1]);
        if ((v1) >= 0) {
            v1 = 0;
            results[1] = pos[0] /* + 1e-15*/;
            results[2] = pos[0] /* - 1e-15*/;
        }
        if ((v2) <= 0) {
            v2 = 0;
            results[1] = pos[1] /* + 1e-15*/;
            results[2] = pos[1] /* - 1e-15*/;
        }
        sign = v1 * v2;

        if (sign < 0) {
            solves = 3;
        } else {
            if ((v2 <= 0)) {
                DX = -DX;
            }
        }
    } else {
        results[0] = specialPoint;
        results[1] = specialPoint;
        results[2] = specialPoint;
        num_solutions = 3;
        return;
    }

    Scalar start = specialPoint - DX;
    Scalar x0 = start;
    // Scalar result[3];

    for (int i = 0; i < solves; i++) {
        Scalar x1 = 0;
        int itCount = 0;
        do {
            if (itCount) x0 = x1;

            x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
            itCount++;

        } while (__mabs(x1 - x0) > EPS && itCount < 100000);
        results[num_solutions] = (x1);
        num_solutions++;
        start = start + DX;
        x0 = start;
    }
    // printf("%f   %f    %f    %f    %f\n", specialPoint, DX, results[0],
    // results[1], results[2]);
    num_solutions = 3;
}

__device__ __host__ void __SolverForCubicEquation(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS) {
    Scalar A = b * b - 3 * a * c;
    Scalar B = b * c - 9 * a * d;
    Scalar C = c * c - 3 * b * d;
    Scalar delta = B * B - 4 * A * C;
    num_solutions = 0;
    if (abs(A) < EPS * EPS && abs(B) < EPS * EPS) {
        results[0] = -b / 3.0 / a;
        results[1] = results[0];
        results[2] = results[0];
        num_solutions = 3;
    } else if (abs(delta) <= EPS * EPS) {
        Scalar K = B / A;
        results[0] = -b / a + K;
        results[1] = -K / 2.0;
        results[2] = results[1];
        num_solutions = 3;
    } else if (delta < -EPS * EPS) {
        Scalar T = (2 * A * b - 3 * a * B) / (2 * A * sqrt(A));
        Scalar theta = acos(T);
        results[0] = (-b - 2 * sqrt(A) * cos(theta / 3.0)) / (3 * a);
        results[1] =
            (-b + sqrt(A) * (cos(theta / 3.0) + sqrt(3.0) * sin(theta / 3.0))) /
            (3 * a);
        results[2] =
            (-b + sqrt(A) * (cos(theta / 3.0) - sqrt(3.0) * sin(theta / 3.0))) /
            (3 * a);
        num_solutions = 3;
    } else if (delta > EPS * EPS) {
        Scalar Y1 = A * b + 3 * a * (-B + sqrt(delta)) / 2;
        Scalar Y2 = A * b + 3 * a * (-B - sqrt(delta)) / 2;
        results[0] = -b - cbrt(Y1) - cbrt(Y2);
        num_solutions = 1;
    }
}

__device__ __host__ Vector9S __Mat3x3_to_vec9_Scalar(const Matrix3x3S& F) {
    Vector9S result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.v[i * 3 + j] = F.m[j][i];
        }
    }
    return result;
}

__device__ __host__ void __normalized_vec9_Scalar(Vector9S& v9) {
    Scalar length = 0;
    for (int i = 0; i < 9; i++) {
        length += v9.v[i] * v9.v[i];
    }
    length = 1.0 / sqrt(length);
    for (int i = 0; i < 9; i++) {
        v9.v[i] = v9.v[i] * length;
    }
}

__device__ __host__ void __normalized_vec6_Scalar(Vector6S& v6) {
    Scalar length = 0;
    for (int i = 0; i < 6; i++) {
        length += v6.v[i] * v6.v[i];
    }
    length = 1.0 / sqrt(length);
    for (int i = 0; i < 6; i++) {
        v6.v[i] = v6.v[i] * length;
    }
}

__device__ __host__ Vector6S __Mat3x2_to_vec6_Scalar(const Matrix3x2S& F) {
    Vector6S result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            result.v[i * 3 + j] = F.m[j][i];
        }
    }
    return result;
}

__device__ __host__ Matrix3x3S __vec9_to_Mat3x3_Scalar(const Scalar vec9[9]) {
    Matrix3x3S mat;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat.m[j][i] = vec9[i * 3 + j];
        }
    }
    return mat;
}

__device__ __host__ Matrix2x2S __vec4_to_Mat2x2_Scalar(const Scalar vec4[4]) {
    Matrix2x2S mat;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            mat.m[j][i] = vec4[i * 2 + j];
        }
    }
    return mat;
}


__device__ void SVD(const Matrix3x3S& F, Matrix3x3S& Uout, Matrix3x3S& Vout, Matrix3x3S& Sigma) {
    // 1. QR分解的Givens旋转
    auto GivensRotation = [](Scalar a, Scalar b, Scalar& c, Scalar& s) {
        Scalar r = sqrt(a * a + b * b);
        c = a / r;
        s = -b / r;
    };

    // 2. 对F进行QR分解
    Matrix3x3S R = F;
    Matrix3x3S Q;
    __identify_Mat3x3(Q);
    // Matrix3x3S Q = Matrix3x3S::Identity();

    // 3. 使用Givens旋转清除下三角元素
    for (int j = 0; j < 3; ++j) {
        for (int i = 2; i > j; --i) {
            Scalar c, s;
            GivensRotation(R.m[i - 1][j], R.m[i][j], c, s);

            // 更新R矩阵
            for (int k = 0; k < 3; ++k) {
                Scalar temp1 = c * R.m[i - 1][k] - s * R.m[i][k];
                Scalar temp2 = s * R.m[i - 1][k] + c * R.m[i][k];
                R.m[i - 1][k] = temp1;
                R.m[i][k] = temp2;
            }

            // 更新Q矩阵
            for (int k = 0; k < 3; ++k) {
                Scalar temp1 = c * Q.m[k][i - 1] - s * Q.m[k][i];
                Scalar temp2 = s * Q.m[k][i - 1] + c * Q.m[k][i];
                Q.m[k][i - 1] = temp1;
                Q.m[k][i] = temp2;
            }
        }
    }

    // 4. 将QR分解的Q作为U矩阵
    Uout = Q;

    // 5. 现在，我们通过QR分解的R矩阵对V矩阵和Sigma进行处理
    // 假设R矩阵的对角线元素是奇异值
    Scalar singularValues[3] = {fabs(R.m[0][0]), fabs(R.m[1][1]), fabs(R.m[2][2])};

    // 对奇异值排序
    for (int i = 0; i < 3; ++i) {
        Sigma.m[i][i] = singularValues[i];
    }

    // 6. 假设V矩阵是单位矩阵（根据QR分解的性质）
    // Vout = Matrix3x3S::Identity();
    __identify_Mat3x3(Vout);

    // 7. 手动将Sigma对角化并确保符号处理（根据你的需求来处理不同的符号情况）
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Sigma.m[i][j] = (i == j) ? singularValues[i] : 0.0;
        }
    }
}


// #include "zensim/math/bit/Bits.h"
// #include "zensim/math/matrix/QRSVD.hpp"
// #include "zensim/math/Complex.hpp"
// #include "zensim/math/matrix/Eigen.hpp"
// #include "zensim/math/MathUtils.h"
// #include "zensim/geometry/Distance.hpp"
// #include "zensim/geometry/SpatialQuery.hpp"
// __device__ void SVD(const Matrix3x3S& F, Matrix3x3S& Uout, Matrix3x3S& Vout,
//                     Matrix3x3S& Sigma) {
//     using matview = zs::vec_view<Scalar, zs::integer_seq<int, 3, 3>>;
//     using cmatview = zs::vec_view<const Scalar, zs::integer_seq<int, 3, 3>>;
//     using vec3 = zs::vec<Scalar, 3>;
//     cmatview F_{(const Scalar*)F.m};
//     matview UU{(Scalar*)Uout.m}, VV{(Scalar*)Vout.m};
//     vec3 SS{};
//     zs::tie(UU, SS, VV) = zs::math::qr_svd(F_);
//     for (int i = 0; i != 3; ++i)
//         for (int j = 0; j != 3; ++j) {
//             Uout.m[i][j] = UU(i, j);
//             Vout.m[i][j] = VV(i, j);
//             Sigma.m[i][j] = (i != j ? 0. : SS[i]);
//         }
// }


__device__ __host__ void __makePD2x2(const Scalar& a00, const Scalar& a01,
                                     const Scalar& a10, const Scalar& a11,
                                     Scalar eigenValues[2], int& num,
                                     Scalar2 eigenVectors[2], Scalar eps) {
    Scalar b = -(a00 + a11), c = a00 * a11 - a10 * a01;
    Scalar existEv = b * b - 4 * c;
    if ((a01) == 0 || (a10) == 0) {
        if (a00 > 0) {
            eigenValues[num] = a00;
            eigenVectors[num].x = 1;
            eigenVectors[num].y = 0;
            num++;
        }
        if (a11 > 0) {
            eigenValues[num] = a11;
            eigenVectors[num].x = 0;
            eigenVectors[num].y = 1;
            num++;
        }
    } else {
        if (existEv > 0) {
            num = 2;
            eigenValues[0] = (-b - sqrt(existEv)) / 2;
            eigenVectors[0].x = 1;
            eigenVectors[0].y = (eigenValues[0] - a00) / a01;
            Scalar length = sqrt(eigenVectors[0].x * eigenVectors[0].x +
                                 eigenVectors[0].y * eigenVectors[0].y);
            // eigenValues[0] *= length;
            eigenVectors[0].x /= length;
            eigenVectors[0].y /= length;

            eigenValues[1] = (-b + sqrt(existEv)) / 2;
            eigenVectors[1].x = 1;
            eigenVectors[1].y = (eigenValues[1] - a00) / a01;
            length = sqrt(eigenVectors[1].x * eigenVectors[1].x +
                          eigenVectors[1].y * eigenVectors[1].y);
            // eigenValues[1] *= length;
            eigenVectors[1].x /= length;
            eigenVectors[1].y /= length;
        } else if (existEv == 0) {
            num = 1;
            eigenValues[0] = (-b - sqrt(existEv)) / 2;
            eigenVectors[0].x = 1;
            eigenVectors[0].y = (eigenValues[0] - a00) / a01;
            Scalar length = sqrt(eigenVectors[0].x * eigenVectors[0].x +
                                 eigenVectors[0].y * eigenVectors[0].y);
            // eigenValues[0] *= length;
            eigenVectors[0].x /= length;
            eigenVectors[0].y /= length;
        } else {
            num = 0;
        }
    }
}

__device__ __host__ void __M9x4_S4x4_MT4x9_Multiply(const Matrix9x4S& A,
                                                    const Matrix4x4S& B,
                                                    Matrix9x9S& output) {
    // Matrix12x12S output;
    Vector4S tempM;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 4; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 4; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            tempM.v[j] = temp;
        }

        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 4; k++) {
                temp += A.m[j][k] * tempM.v[k];
            }
            output.m[i][j] = temp;
        }
    }
    // return output;
}

__device__ __host__ void __M12x9_S9x9_MT9x12_Multiply(const Matrix12x9S& A,
                                                      const Matrix9x9S& B,
                                                      Matrix12x12S& output) {
    // Matrix12x12S output;
    Vector9S tempM;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 9; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            tempM.v[j] = temp;
        }

        for (int j = 0; j < 12; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 9; k++) {
                temp += A.m[j][k] * tempM.v[k];
            }
            output.m[i][j] = temp;
        }
    }
    // return output;
}

__device__ __host__ Vector4S __s_vec4_multiply(Vector4S a, Scalar b) {
    Vector4S V;
    for (int i = 0; i < 4; i++) V.v[i] = a.v[i] * b;
    return V;
}

__device__ __host__ Vector9S __M9x4_v4_multiply(const Matrix9x4S& A,
                                               const Vector4S& n) {
    Vector9S v9;
    for (int i = 0; i < 9; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 4; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v9.v[i] = temp;
    }
    return v9;
}

__device__ __host__ Matrix4x4S __S_Mat4x4_multiply(const Matrix4x4S& A,
                                                   const Scalar& B) {
    Matrix4x4S output;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            output.m[i][j] = A.m[i][j] * B;
        }
    }
    return output;
}

__device__ __host__ Matrix4x4S __v4_vec4_toMat4x4(Vector4S a, Vector4S b) {
    Matrix4x4S M;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            M.m[i][j] = a.v[i] * b.v[j];
        }
    }
    return M;
}

__device__ __host__ void __s_M_Mat_MT_multiply(const Matrix3x3S& A,
                                               const Matrix3x3S& B,
                                               const Matrix3x3S& C,
                                               const Scalar& coe,
                                               Matrix3x3S& output) {
    Scalar tvec3[3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            // output.m[i][j] = temp;
            tvec3[j] = temp;
        }

        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += C.m[j][k] * tvec3[k];
            }
            output.m[i][j] = temp * coe;
            // tvec3[j] = temp;
        }
    }
}


__device__
void _d_PP(const Scalar3& v0, const Scalar3& v1, Scalar& d)
{
    d = __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(v0, v1));
}

__device__
void _d_PT(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d)
{
    Scalar3 b = __MATHUTILS__::__v_vec_cross(__MATHUTILS__::__minus(v2, v1), __MATHUTILS__::__minus(v3, v1));
    Scalar3 test = __MATHUTILS__::__minus(v0, v1);
    Scalar aTb = __MATHUTILS__::__v_vec_dot(__MATHUTILS__::__minus(v0, v1), b);//(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    d = aTb * aTb / __MATHUTILS__::__squaredNorm(b);
}

__device__
void _d_PE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, Scalar& d)
{
    d = __MATHUTILS__::__squaredNorm(__MATHUTILS__::__v_vec_cross(__MATHUTILS__::__minus(v1, v0), __MATHUTILS__::__minus(v2, v0))) / __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(v2, v1));
}

__device__
void _d_EE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d)
{
    Scalar3 b = __MATHUTILS__::__v_vec_cross(__MATHUTILS__::__minus(v1, v0), __MATHUTILS__::__minus(v3, v2));//(v1 - v0).cross(v3 - v2);
    Scalar aTb = __MATHUTILS__::__v_vec_dot(__MATHUTILS__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __MATHUTILS__::__squaredNorm(b);
}


__device__
void _d_EEParallel(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d)
{
    Scalar3 b = __MATHUTILS__::__v_vec_cross(__MATHUTILS__::__v_vec_cross(__MATHUTILS__::__minus(v1, v0), __MATHUTILS__::__minus(v2, v0)), __MATHUTILS__::__minus(v1, v0));
    Scalar aTb = __MATHUTILS__::__v_vec_dot(__MATHUTILS__::__minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __MATHUTILS__::__squaredNorm(b);
}

__device__
Scalar _compute_epx(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3) {
    return 1e-3 * __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(v0, v1)) * __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(v2, v3));
}

__device__
Scalar _compute_epx_cp(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3) {
    return 1e-3 * __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(v0, v1)) * __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(v2, v3));
}

__device__ __host__
Scalar calculateVolume(const Scalar3* vertexes, const uint4& index) {
    int id0 = 0;
    int id1 = 1;
    int id2 = 2;
    int id3 = 3;
    Scalar o1x = vertexes[index.y].x - vertexes[index.x].x;
    Scalar o1y = vertexes[index.y].y - vertexes[index.x].y;
    Scalar o1z = vertexes[index.y].z - vertexes[index.x].z;
    Scalar3 OA = make_Scalar3(o1x, o1y, o1z);

    Scalar o2x = vertexes[index.z].x - vertexes[index.x].x;
    Scalar o2y = vertexes[index.z].y - vertexes[index.x].y;
    Scalar o2z = vertexes[index.z].z - vertexes[index.x].z;
    Scalar3 OB = make_Scalar3(o2x, o2y, o2z);

    Scalar o3x = vertexes[index.w].x - vertexes[index.x].x;
    Scalar o3y = vertexes[index.w].y - vertexes[index.x].y;
    Scalar o3z = vertexes[index.w].z - vertexes[index.x].z;
    Scalar3 OC = make_Scalar3(o3x, o3y, o3z);

    Scalar3 heightDir = __MATHUTILS__::__v_vec_cross(OA, OB);  // OA.cross(OB);
    Scalar bottomArea = __MATHUTILS__::__norm(heightDir);      // heightDir.norm();
    heightDir = __MATHUTILS__::__normalized(heightDir);

    Scalar volum = bottomArea * __MATHUTILS__::__v_vec_dot(heightDir, OC) / 6;
    return volum > 0 ? volum : -volum;
}

__device__ __host__
Scalar calculateArea(const Scalar3* vertexes, const uint3& index) {
    Scalar3 v10 = __MATHUTILS__::__minus(vertexes[index.y], vertexes[index.x]);
    Scalar3 v20 = __MATHUTILS__::__minus(vertexes[index.z], vertexes[index.x]);
    Scalar area = __MATHUTILS__::__norm(__MATHUTILS__::__v_vec_cross(v10, v20));
    return 0.5 * area;
}

}  // namespace __MATHUTILS__



// reduction
namespace __MATHUTILS__ {

__device__ void perform_reduction(
    Scalar temp,           // 每个线程计算得到的临时值
    Scalar* tep,           // 共享内存数组，用于存储每个 warp 的中间结果
    Scalar* squeue,        // 存储每个 block 的最终归约结果
    int numbers,           // 总的元素数量
    int idof,              // 每个 block 的起始索引偏移量
    int blockDimX,         // 每个 block 的线程数
    int gridDimX,          // grid 的 block 数量
    int blockIdX            // 当前 block 的索引
) {

    /////////////////////////////////////////////////
    // Block (blockDim.x = 64 threads)
    // +-------------------------------------------------------+
    // | Thread 0 | Thread 1 | ... | Thread 31 | Thread 32 | ... | Thread 63 |
    // |  temp0   |  temp1   | ... |  temp31  |  temp32  | ... |  temp63  |
    // +-------------------------------------------------------+
    /////////////////////////////////////////////////
    // Warp 0 (Threads 0-31):
    // +---------+---------+---------+---------+---------+---------+---------+---------+
    // | temp0   | temp1   | temp2   | ...     | temp30  | temp31  |
    // +---------+---------+---------+---------+---------+---------+---------+---------+
    // After warp-level reduction:
    // Thread 0 holds the sum of temp0 to temp31
    // .......
    // Warp 1 (Threads 32-63):
    // +---------+---------+---------+---------+---------+---------+---------+---------+
    // | temp32  | temp33  | temp34  | ...     | temp62  | temp63  |
    // +---------+---------+---------+---------+---------+---------+---------+---------+
    // After warp-level reduction:
    // Thread 32 holds the sum of temp32 to temp63
    /////////////////////////////////////////////////
    // Shared Memory (`tep`):
    // +----------+----------+
    // | tep0     | tep1     |
    // +----------+----------+
    // | sum_warp0| sum_warp1|
    // +----------+----------+
    /////////////////////////////////////////////////
    // Block-level reduction:
    // sum_warp0 + sum_warp1 = total_sum_block
    // squeue[blockIdx.x] = total_sum_block
    /////////////////////////////////////////////////

    // 计算当前线程在 warp 内的 ID 和 warp 的 ID
    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);

    // 计算当前 block 中的 warp 数量
    int warpNum;
    if (blockIdX == gridDimX - 1) {
        // 最后一个 block 可能包含不完整的 warp
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = (blockDimX >> 5);
    }

    // Warp-level 归约
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }

    // 将每个 warp 的部分和存储到共享内存中
    if (warpTid == 0) {
        tep[warpId] = temp;
    }

    // 同步所有线程，确保共享内存中的数据已被写入
    __syncthreads();

    // 仅保留有效的 warp 数量，其他线程退出
    if (threadIdx.x >= warpNum) return;

    // 如果有多个 warp，进行 block-level 归约
    if (warpNum > 1) {
        // 读取共享内存中的 warp-level 部分和
        temp = tep[threadIdx.x];
        // 再次进行归约
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }

    // 仅线程 0 写入最终的归约结果到 squeue
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}


__global__ void _reduct_max_Scalar(Scalar* _Scalar1Dim, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep[];
    if (idx >= numbers) return;
    // int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    Scalar temp = _Scalar1Dim[idx];

    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_max(temp, tempMax);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_max(temp, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _Scalar1Dim[blockIdx.x] = temp;
    }
}


__global__ void _reduct_min_Scalar(Scalar* _Scalar1Dim, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;
    // int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    Scalar temp = _Scalar1Dim[idx];

    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_min(temp, tempMin);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_min(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        _Scalar1Dim[blockIdx.x] = temp;
    }
}


__global__ void __add_reduction(Scalar* mem, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;
    // int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    Scalar temp = mem[idx];

    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }
    if (threadIdx.x == 0) {
        mem[blockIdx.x] = temp;
    }
}


__global__ void _reduct_max_Scalar2(Scalar2* _Scalar2Dim, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar2 sdata[];

    if (idx >= numbers) return;
    // int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    Scalar2 temp = _Scalar2Dim[idx];

    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
        temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        sdata[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = sdata[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
            temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _Scalar2Dim[blockIdx.x] = temp;
    }
}

__global__ void _reduct_max_Scalar3_to_Scalar(const Scalar3* _Scalar3Dim, Scalar* _Scalar1Dim,
                                              int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep[];

    if (idx >= numbers) return;
    // int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    Scalar3 tempMove = _Scalar3Dim[idx];

    Scalar temp = __MATHUTILS__::__m_max(__MATHUTILS__::__m_max(abs(tempMove.x), abs(tempMove.y)),
                                         abs(tempMove.z));

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_max(temp, tempMin);
    }
    if (warpTid == 0) {
        tep[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        _Scalar1Dim[blockIdx.x] = temp;
    }
}

};
