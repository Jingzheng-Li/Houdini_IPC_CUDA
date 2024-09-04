

#define _USE_MATH_DEFINES 
#include <cmath>

#include "MathUtils.cuh"
#include "zensim/math/matrix/QRSVD.hpp"
#include "zensim/math/bit/Bits.h"

#include <vector>
#include <set>
#include <iostream>
#include <map>
// #include <utility>


namespace MATHUTILS {


	__device__ __host__ void __init_Mat3x3(Matrix3x3d& M, const double& val) {
		for (int i = 0;i < 3;i++) {
			for (int j = 0;j < 3;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat6x6(Matrix6x6d& M, const double& val) {
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 6;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat9x9(Matrix9x9d& M, const double& val) {
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 9;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __identify_Mat9x9(Matrix9x9d& M) {
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 9;j++) {
				if (i == j) {
					M.m[i][j] = 1;
				}
				else {
					M.m[i][j] = 0;
				}
			}
		}
	}

	__device__ __host__ double __mabs(const double& a) {
		return a > 0 ? a : -a;
	}

	__device__ __host__ double __norm(const double3& n) {
		return sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
	}

	__device__ __host__ double3 __s_vec_multiply(const double3& a, double b) {
		return make_double3(a.x * b, a.y * b, a.z * b);
	}

	__device__ __host__ double2 __s_vec_multiply(const double2& a, double b) {
		return make_double2(a.x * b, a.y * b);
	}

	__device__ __host__ double3 __normalized(double3 n) {
		double norm = __norm(n);
		norm = 1 / norm;
		return __s_vec_multiply(n, norm);
	}

	__device__ __host__ double3 __add(double3 a, double3 b) {
		return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
	}

	__device__ __host__ Vector9 __add9(const Vector9& a, const Vector9& b) {
		Vector9 V;
		for (int i = 0;i < 9;i++) {
			V.v[i] = a.v[i] + b.v[i];
		}
		return V;
	}

	__device__ __host__ Vector6 __add6(const Vector6& a, const Vector6& b) {
		Vector6 V;
		for (int i = 0;i < 6;i++) {
			V.v[i] = a.v[i] + b.v[i];
		}
		return V;
	}

	__device__ __host__ double3 __minus(double3 a, double3 b) {
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}

	__device__ __host__ double2 __minus_v2(double2 a, double2 b) {
		return make_double2(a.x - b.x, a.y - b.y);
	}

	__device__ __host__ double3 __v_vec_multiply(double3 a, double3 b) {
		return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
	}

	__device__ __host__ double __v2_vec_multiply(double2 a, double2 b) {
		return a.x * b.x + a.y * b.y;
	}

	__device__ __host__ double __squaredNorm(double3 a) {
		return a.x * a.x + a.y * a.y + a.z * a.z;
	}

	__device__ __host__ double __squaredNorm(double2 a)
	{
		return  a.x * a.x + a.y * a.y;
	}

	__device__ __host__ void __M_Mat_multiply(const Matrix3x3d& A, const Matrix3x3d& B, Matrix3x3d& output) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				double temp = 0;
				for (int k = 0; k < 3; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
	}


	__device__ __host__ Matrix3x3d __M_Mat_multiply(const Matrix3x3d& A, const Matrix3x3d& B) {
		Matrix3x3d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				double temp = 0;
				for (int k = 0; k < 3; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix2x2d __M2x2_Mat2x2_multiply(const Matrix2x2d& A, const Matrix2x2d& B) {
		Matrix2x2d output;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ double __Mat_Trace(const Matrix3x3d& A) {
		return A.m[0][0] + A.m[1][1] + A.m[2][2];
	}

	__device__ __host__ double3 __v_M_multiply(const double3& n, const Matrix3x3d& A) {
		double x = A.m[0][0] * n.x + A.m[1][0] * n.y + A.m[2][0] * n.z;
		double y = A.m[0][1] * n.x + A.m[1][1] * n.y + A.m[2][1] * n.z;
		double z = A.m[0][2] * n.x + A.m[1][2] * n.y + A.m[2][2] * n.z;
		return make_double3(x, y, z);
	}

	__device__ __host__ double3 __M_v_multiply(const Matrix3x3d& A, const double3& n) {
		double x = A.m[0][0] * n.x + A.m[0][1] * n.y + A.m[0][2] * n.z;
		double y = A.m[1][0] * n.x + A.m[1][1] * n.y + A.m[1][2] * n.z;
		double z = A.m[2][0] * n.x + A.m[2][1] * n.y + A.m[2][2] * n.z;
		return make_double3(x, y, z);
	}

	__device__ __host__ double3 __M3x2_v2_multiply(const Matrix3x2d& A, const double2& n) {
		double x = A.m[0][0] * n.x + A.m[0][1] * n.y;// +A.m[0][2] * n.z;
		double y = A.m[1][0] * n.x + A.m[1][1] * n.y;// +A.m[1][2] * n.z;
		double z = A.m[2][0] * n.x + A.m[2][1] * n.y;// +A.m[2][2] * n.z;
		return make_double3(x, y, z);
	}

	__device__ __host__ Matrix3x2d __Mat3x2_add(const Matrix3x2d& A, const Matrix3x2d& B) {
		Matrix3x2d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				output.m[i][j] = A.m[i][j] + B.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix3x2d __S_Mat3x2_multiply(const Matrix3x2d& A, const double& b) {
		Matrix3x2d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				output.m[i][j] = A.m[i][j] * b;
			}
		}
		return output;
	}

	__device__ __host__ Vector12 __M12x9_v9_multiply(const Matrix12x9d& A, const Vector9& n) {
		Vector12 v12;
		for (int i = 0;i < 12;i++) {
			double temp = 0;
			for (int j = 0;j < 9;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v12.v[i] = temp;
		}
		return v12;
	}

	__device__ __host__ Vector12 __M12x6_v6_multiply(const Matrix12x6d& A, const Vector6& n) {
		Vector12 v12;
		for (int i = 0;i < 12;i++) {
			double temp = 0;
			for (int j = 0;j < 6;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v12.v[i] = temp;
		}
		return v12;
	}

	__device__ __host__ Vector6 __M6x3_v3_multiply(const Matrix6x3d& A, const double3& n) {
		Vector6 v6;
		for (int i = 0;i < 6;i++) {

			double temp = A.m[i][0] * n.x;
			temp += A.m[i][1] * n.y;
			temp += A.m[i][2] * n.z;

			v6.v[i] = temp;
		}
		return v6;
	}

	__device__ __host__ double2 __M2x3_v3_multiply(const Matrix2x3d& A, const double3& n) {
		double2 output;
		output.x = A.m[0][0] * n.x + A.m[0][1] * n.y + A.m[0][2] * n.z;
		output.y = A.m[1][0] * n.x + A.m[1][1] * n.y + A.m[1][2] * n.z;
		return output;
	}

	__device__ __host__ Vector9 __M9x6_v6_multiply(const Matrix9x6d& A, const Vector6& n) {
		Vector9 v9;
		for (int i = 0;i < 9;i++) {
			double temp = 0;
			for (int j = 0;j < 6;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v9.v[i] = temp;
		}
		return v9;
	}

	__device__ __host__ Vector12 __M12x12_v12_multiply(const Matrix12x12d& A, const Vector12& n) {
		Vector12 v12;
		for (int i = 0;i < 12;i++) {
			double temp = 0;
			for (int j = 0;j < 12;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v12.v[i] = temp;
		}
		return v12;
	}

	__device__ __host__ Vector9 __M9x9_v9_multiply(const Matrix9x9d& A, const Vector9& n) {
		Vector9 v9;
		for (int i = 0;i < 9;i++) {
			double temp = 0;
			for (int j = 0;j < 9;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v9.v[i] = temp;
		}
		return v9;
	}

	__device__ __host__ Vector6 __M6x6_v6_multiply(const Matrix6x6d& A, const Vector6& n) {
		Vector6 v6;
		for (int i = 0;i < 6;i++) {
			double temp = 0;
			for (int j = 0;j < 6;j++) {
				temp += A.m[i][j] * n.v[j];
			}
			v6.v[i] = temp;
		}
		return v6;
	}

	__device__ __host__ Matrix9x9d __S_Mat9x9_multiply(const Matrix9x9d& A, const double& B) {
		Matrix9x9d output;
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 9;j++) {
				output.m[i][j] = A.m[i][j] * B;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x6d __S_Mat6x6_multiply(const Matrix6x6d& A, const double& B) {
		Matrix6x6d output;
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 6;j++) {
				output.m[i][j] = A.m[i][j] * B;
			}
		}
		return output;
	}

	__device__ __host__ double __v_vec_dot(const double3& a, const double3& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__ __host__ double3 __v_vec_cross(double3 a, double3 b) {
		return make_double3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
	}

	__device__ __host__ Matrix3x3d __v_vec_toMat(double3 a, double3 b) {
		Matrix3x3d M;
		M.m[0][0] = a.x * b.x;M.m[0][1] = a.x * b.y;M.m[0][2] = a.x * b.z;
		M.m[1][0] = a.y * b.x;M.m[1][1] = a.y * b.y;M.m[1][2] = a.y * b.z;
		M.m[2][0] = a.z * b.x;M.m[2][1] = a.z * b.y;M.m[2][2] = a.z * b.z;
		return M;
	}

	__device__ __host__ Matrix2x2d __v2_vec2_toMat2x2(double2 a, double2 b) {
		Matrix2x2d M;
		M.m[0][0] = a.x * b.x;M.m[0][1] = a.x * b.y;
		M.m[1][0] = a.y * b.x;M.m[1][1] = a.y * b.y;
		return M;
	}

	__device__ __host__ Matrix2x2d __s_Mat2x2_multiply(Matrix2x2d A, double b)
	{
		Matrix2x2d output;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				output.m[i][j] = A.m[i][j] * b;
			}
		}
		return output;
	}

	__device__ __host__ Matrix2x2d __Mat2x2_minus(Matrix2x2d A, Matrix2x2d B)
	{
		Matrix2x2d output;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				output.m[i][j] = A.m[i][j] - B.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix3x3d __Mat3x3_minus(Matrix3x3d A, Matrix3x3d B)
	{
		Matrix3x3d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				output.m[i][j] = A.m[i][j] - B.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x9d __v9_vec9_toMat9x9(const Vector9& a, const Vector9& b, const double& coe) {
		Matrix9x9d M;
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 9;j++) {
				M.m[i][j] = a.v[i] * b.v[j]*coe;
			}
		}
		return M;
	}

	__device__ __host__ Matrix6x6d __v6_vec6_toMat6x6(Vector6 a, Vector6 b) {
		Matrix6x6d M;
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 6;j++) {
				M.m[i][j] = a.v[i] * b.v[j];
			}
		}
		return M;
	}

	__device__ __host__ Vector9 __s_vec9_multiply(Vector9 a, double b) {
		Vector9 V;
		for (int i = 0;i < 9;i++)
			V.v[i] = a.v[i] * b;
		return V;
	}

	__device__ __host__ Vector12 __s_vec12_multiply(Vector12 a, double b) {
		Vector12 V;
		for (int i = 0;i < 12;i++)
			V.v[i] = a.v[i] * b;
		return V;
	}

	__device__ __host__ Vector6 __s_vec6_multiply(Vector6 a, double b) {
		Vector6 V;
		for (int i = 0;i < 6;i++)
			V.v[i] = a.v[i] * b;
		return V;
	}

	__device__ __host__ void __Mat_add(const Matrix3x3d& A, const Matrix3x3d& B, Matrix3x3d& output) {
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
	}


	__device__ __host__ void __Mat_add(const Matrix6x6d& A, const Matrix6x6d& B, Matrix6x6d& output) {
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
	}

	__device__ __host__ Matrix3x3d __Mat_add(const Matrix3x3d& A, const Matrix3x3d& B) {
		Matrix3x3d output;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix2x2d __Mat2x2_add(const Matrix2x2d& A, const Matrix2x2d& B) {
		Matrix2x2d output;
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix9x9d __Mat9x9_add(const Matrix9x9d& A, const Matrix9x9d& B) {
		Matrix9x9d output;
		for (int i = 0; i < 9; i++)
			for (int j = 0; j < 9; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix9x12d __Mat9x12_add(const Matrix9x12d& A, const Matrix9x12d& B) {
		Matrix9x12d output;
		for (int i = 0; i < 9; i++)
			for (int j = 0; j < 12; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix6x12d __Mat6x12_add(const Matrix6x12d& A, const Matrix6x12d& B) {
		Matrix6x12d output;
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 12; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix6x9d __Mat6x9_add(const Matrix6x9d& A, const Matrix6x9d& B) {
		Matrix6x9d output;
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 9; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ Matrix3x6d __Mat3x6_add(const Matrix3x6d& A, const Matrix3x6d& B) {
		Matrix3x6d output;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 6; j++)
				output.m[i][j] = A.m[i][j] + B.m[i][j];
		return output;
	}

	__device__ __host__ void __set_Mat_identity(Matrix2x2d& M) {
		M.m[0][0] = 1;
		M.m[1][0] = 0;
		M.m[0][1] = 0;
		M.m[1][1] = 1;
	}

	__device__ __host__ void __set_Mat_val(Matrix3x3d& M, const double& a00, const double& a01, const double& a02,
		const double& a10, const double& a11, const double& a12,
		const double& a20, const double& a21, const double& a22) {
		M.m[0][0] = a00;M.m[0][1] = a01;M.m[0][2] = a02;
		M.m[1][0] = a10;M.m[1][1] = a11;M.m[1][2] = a12;
		M.m[2][0] = a20;M.m[2][1] = a21;M.m[2][2] = a22;
	}

	__device__ __host__ void __set_Mat_val_row(Matrix3x3d& M, const double3& row0, const double3& row1, const double3& row2) {
		M.m[0][0] = row0.x;M.m[0][1] = row0.y;M.m[0][2] = row0.z;
		M.m[1][0] = row1.x;M.m[1][1] = row1.y;M.m[1][2] = row1.z;
		M.m[2][0] = row2.x;M.m[2][1] = row2.y;M.m[2][2] = row2.z;
	}

	__device__ __host__ void __set_Mat_val_column(Matrix3x3d& M, const double3& col0, const double3& col1, const double3& col2) {
		M.m[0][0] = col0.x;M.m[0][1] = col1.x;M.m[0][2] = col2.x;
		M.m[1][0] = col0.y;M.m[1][1] = col1.y;M.m[1][2] = col2.y;
		M.m[2][0] = col0.z;M.m[2][1] = col1.z;M.m[2][2] = col2.z;
	}

	__device__ __host__ void __set_Mat3x2_val_column(Matrix3x2d& M, const double3& col0, const double3& col1) {
		M.m[0][0] = col0.x;M.m[0][1] = col1.x;
		M.m[1][0] = col0.y;M.m[1][1] = col1.y;
		M.m[2][0] = col0.z;M.m[2][1] = col1.z;
	}

	__device__ __host__ void __set_Mat2x2_val_column(Matrix2x2d& M, const double2& col0, const double2& col1) {
		M.m[0][0] = col0.x;M.m[0][1] = col1.x;
		M.m[1][0] = col0.y;M.m[1][1] = col1.y;
	}

	__device__ __host__ void __init_Mat9x12_val(Matrix9x12d& M, const double& val) {
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 12;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat6x12_val(Matrix6x12d& M, const double& val) {
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 12;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat6x9_val(Matrix6x9d& M, const double& val) {
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 9;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ void __init_Mat3x6_val(Matrix3x6d& M, const double& val) {
		for (int i = 0;i < 3;i++) {
			for (int j = 0;j < 6;j++) {
				M.m[i][j] = val;
			}
		}
	}

	__device__ __host__ Matrix3x3d __S_Mat_multiply(const Matrix3x3d& A, const double& B) {
		Matrix3x3d output;
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				output.m[i][j] = A.m[i][j] * B;
		return output;
	}

	__device__ __host__ Matrix3x3d __Transpose3x3(Matrix3x3d input) {
		Matrix3x3d output;
		for (int i = 0;i < 3;i++) {
			for (int j = 0;j < 3;j++) {
				output.m[i][j] = input.m[j][i];
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x9d __Transpose9x12(const Matrix9x12d& input) {
		Matrix12x9d output;
		for (int i = 0;i < 9;i++) {
			for (int j = 0;j < 12;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix2x3d __Transpose3x2(const Matrix3x2d& input) {
		Matrix2x3d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x12d __Transpose12x9(const Matrix12x9d& input) {
		Matrix9x12d output;
		for (int i = 0;i < 12;i++) {
			for (int j = 0;j < 9;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x6d __Transpose6x12(const Matrix6x12d& input) {
		Matrix12x6d output;
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 12;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x6d __Transpose6x9(const Matrix6x9d& input) {
		Matrix9x6d output;
		for (int i = 0;i < 6;i++) {
			for (int j = 0;j < 9;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x3d __Transpose3x6(const Matrix3x6d& input) {
		Matrix6x3d output;
		for (int i = 0;i < 3;i++) {
			for (int j = 0;j < 6;j++) {
				output.m[j][i] = input.m[i][j];
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x9d __M12x9_M9x9_Multiply(const Matrix12x9d& A, const Matrix9x9d& B) {
		Matrix12x9d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 9; j++) {
				double temp = 0;
				for (int k = 0; k < 9; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x6d __M12x6_M6x6_Multiply(const Matrix12x6d& A, const Matrix6x6d& B) {
		Matrix12x6d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 6; j++) {
				double temp = 0;
				for (int k = 0; k < 6; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x6d __M9x6_M6x6_Multiply(const Matrix9x6d& A, const Matrix6x6d& B) {
		Matrix9x6d output;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 6; j++) {
				double temp = 0;
				for (int k = 0; k < 6; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x3d __M6x3_M3x3_Multiply(const Matrix6x3d& A, const Matrix3x3d& B) {
		Matrix6x3d output;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 3; j++) {
				double temp = 0;
				for (int k = 0; k < 3; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix3x2d __M3x2_M2x2_Multiply(const Matrix3x2d& A, const Matrix2x2d& B) {
		Matrix3x2d output;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];

				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x12d __M12x9_M9x12_Multiply(const Matrix12x9d& A, const Matrix9x12d& B) {
		Matrix12x12d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				double temp = 0;
				for (int k = 0; k < 9; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x2d __M12x2_M2x2_Multiply(const Matrix12x2d& A, const Matrix2x2d& B) {
		Matrix12x2d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x2d __M9x2_M2x2_Multiply(const Matrix9x2d& A, const Matrix2x2d& B)
	{
		Matrix9x2d output;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x2d __M6x2_M2x2_Multiply(const Matrix6x2d& A, const Matrix2x2d& B)
	{
		Matrix6x2d output;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 2; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x12d __M12x2_M12x2T_Multiply(const Matrix12x2d& A, const Matrix12x2d& B) {
		Matrix12x12d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[j][k];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x9d __M9x2_M9x2T_Multiply(const Matrix9x2d& A, const Matrix9x2d& B)
	{
		Matrix9x9d output;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[j][k];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x6d __M6x2_M6x2T_Multiply(const Matrix6x2d& A, const Matrix6x2d& B)
	{
		Matrix6x6d output;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				double temp = 0;
				for (int k = 0; k < 2; k++) {
					temp += A.m[i][k] * B.m[j][k];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x12d __M12x6_M6x12_Multiply(const Matrix12x6d& A, const Matrix6x12d& B) {
		Matrix12x12d output;
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				double temp = 0;
				for (int k = 0; k < 6; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix9x9d __M9x6_M6x9_Multiply(const Matrix9x6d& A, const Matrix6x9d& B) {
		Matrix9x9d output;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				double temp = 0;
				for (int k = 0; k < 6; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix6x6d __M6x3_M3x6_Multiply(const Matrix6x3d& A, const Matrix3x6d& B) {
		Matrix6x6d output;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 6; j++) {
				double temp = 0;
				for (int k = 0; k < 3; k++) {
					temp += A.m[i][k] * B.m[k][j];
				}
				output.m[i][j] = temp;
			}
		}
		return output;
	}

	__device__ __host__ Matrix12x12d __s_M12x12_Multiply(const Matrix12x12d& A, const double& B) {
		Matrix12x12d output;
		for (int i = 0; i < 12; i++)
			for (int j = 0; j < 12; j++)
				output.m[i][j] = A.m[i][j] * B;
		return output;
	}

	__device__ __host__ Matrix9x9d __s_M9x9_Multiply(const Matrix9x9d& A, const double& B)
	{
		Matrix9x9d output;
		for (int i = 0; i < 9; i++)
			for (int j = 0; j < 9; j++)
				output.m[i][j] = A.m[i][j] * B;
		return output;
	}

	__device__ __host__ Matrix6x6d __s_M6x6_Multiply(const Matrix6x6d& A, const double& B)
	{
		Matrix6x6d output;
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 6; j++)
				output.m[i][j] = A.m[i][j] * B;
		return output;
	}

	__device__ __host__ void __Determiant(const Matrix3x3d& input, double& determinant) {
		determinant = input.m[0][0] * input.m[1][1] * input.m[2][2] +
			input.m[1][0] * input.m[2][1] * input.m[0][2] +
			input.m[2][0] * input.m[0][1] * input.m[1][2] -
			input.m[2][0] * input.m[1][1] * input.m[0][2] -
			input.m[0][0] * input.m[1][2] * input.m[2][1] -
			input.m[0][1] * input.m[1][0] * input.m[2][2];
	}

	__device__ __host__ double __Determiant(const Matrix3x3d& input) {
		return input.m[0][0] * input.m[1][1] * input.m[2][2] +
			input.m[1][0] * input.m[2][1] * input.m[0][2] +
			input.m[2][0] * input.m[0][1] * input.m[1][2] -
			input.m[2][0] * input.m[1][1] * input.m[0][2] -
			input.m[0][0] * input.m[1][2] * input.m[2][1] -
			input.m[0][1] * input.m[1][0] * input.m[2][2];
	}

	__device__ __host__ void __Inverse(const Matrix3x3d& input, Matrix3x3d& result) {
		double eps = 1e-15;
		const int dim = 3;
		double mat[dim][dim * 2];
		for (int i = 0;i < dim; i++)
		{
			for (int j = 0;j < 2 * dim; j++)
			{
				if (j < dim) {
					mat[i][j] = input.m[i][j];//[i, j];
				}
				else {
					mat[i][j] = j - dim == i ? 1 : 0;
				}
			}
		}

		for (int i = 0;i < dim; i++)
		{
			if (abs(mat[i][i]) < eps)
			{
				int j;
				for (j = i + 1; j < dim; j++) {
					if (abs(mat[j][i]) > eps) break;
				}
				if (j == dim) return;
				for (int r = i; r < 2 * dim; r++) {
					mat[i][r] += mat[j][r];
				}
			}
			double ep = mat[i][i];
			for (int r = i; r < 2 * dim; r++) {
				mat[i][r] /= ep;
			}

			for (int j = i + 1; j < dim; j++)
			{
				double e = -1 * (mat[j][i] / mat[i][i]);
				for (int r = i; r < 2 * dim; r++)
				{
					mat[j][r] += e * mat[i][r];
				}
			}
		}

		for (int i = dim - 1; i >= 0; i--)
		{
			for (int j = i - 1; j >= 0; j--)
			{
				double e = -1 * (mat[j][i] / mat[i][i]);
				for (int r = i; r < 2 * dim; r++)
				{
					mat[j][r] += e * mat[i][r];
				}
			}
		}


		for (int i = 0;i < dim; i++)
		{
			for (int r = dim; r < 2 * dim; r++)
			{
				result.m[i][r - dim] = mat[i][r];
			}
		}
	}

	__device__ __host__ void __Inverse2x2(const Matrix2x2d& input, Matrix2x2d& result) {
		double eps = 1e-15;
		const int dim = 2;
		double mat[dim][dim * 2];
		for (int i = 0;i < dim; i++)
		{
			for (int j = 0;j < 2 * dim; j++)
			{
				if (j < dim)
				{
					mat[i][j] = input.m[i][j];//[i, j];
				}
				else
				{
					mat[i][j] = j - dim == i ? 1 : 0;
				}
			}
		}

		for (int i = 0;i < dim; i++)
		{
			if (abs(mat[i][i]) < eps)
			{
				int j;
				for (j = i + 1; j < dim; j++)
				{
					if (abs(mat[j][i]) > eps) break;
				}
				if (j == dim) return;
				for (int r = i; r < 2 * dim; r++)
				{
					mat[i][r] += mat[j][r];
				}
			}
			double ep = mat[i][i];
			for (int r = i; r < 2 * dim; r++)
			{
				mat[i][r] /= ep;
			}

			for (int j = i + 1; j < dim; j++)
			{
				double e = -1 * (mat[j][i] / mat[i][i]);
				for (int r = i; r < 2 * dim; r++)
				{
					mat[j][r] += e * mat[i][r];
				}
			}
		}

		for (int i = dim - 1; i >= 0; i--)
		{
			for (int j = i - 1; j >= 0; j--)
			{
				double e = -1 * (mat[j][i] / mat[i][i]);
				for (int r = i; r < 2 * dim; r++)
				{
					mat[j][r] += e * mat[i][r];
				}
			}
		}


		for (int i = 0;i < dim; i++)
		{
			for (int r = dim; r < 2 * dim; r++)
			{
				result.m[i][r - dim] = mat[i][r];
			}
		}
	}

	__device__ __host__ double __f(const double& x, const double& a, const double& b, const double& c, const double& d) {
		double f = a * x * x * x + b * x * x + c * x + d;
		return f;
	}

	__device__ __host__ double __df(const double& x, const double& a, const double& b, const double& c) {
		double df = 3 * a * x * x + 2 * b * x + c;
		return df;
	}

	__device__ __host__ void __NewtonSolverForCubicEquation(const double& a, const double& b, const double& c, const double& d, double* results, int& num_solutions, double EPS)
	{
		//double EPS = 1e-6;
		double DX = 0;
		//double results[3];
		num_solutions = 0;
		double specialPoint = -b / a / 3;
		double pos[2];
		int solves = 1;
		double delta = 4 * b * b - 12 * a * c;
		if (delta > 0) {
			pos[0] = (sqrt(delta) - 2 * b) / 6 / a;
			pos[1] = (-sqrt(delta) - 2 * b) / 6 / a;
			double v1 = __f(pos[0], a, b, c, d);
			double v2 = __f(pos[1], a, b, c, d);
			if (__mabs(v1) < EPS * EPS) {
				v1 = 0;
			}
			if (__mabs(v2) < EPS * EPS) {
				v2 = 0;
			}
			double sign = v1 * v2;
			DX = (pos[0] - pos[1]);
			if (sign <= 0) {
				solves = 3;
			}
			else if (sign > 0) {
				if ((a < 0 && __f(pos[0], a, b, c, d) > 0) || (a > 0 && __f(pos[0], a, b, c, d) < 0)) {
					DX = -DX;
				}
			}
		}
		else if (delta == 0) {
			if (__mabs(__f(specialPoint, a, b, c, d)) < EPS * EPS) {
				for (int i = 0; i < 3; i++) {
					double tempReuslt = specialPoint;
					results[num_solutions] = tempReuslt;
					num_solutions++;
				}
				return;
			}
			if (a > 0) {
				if (__f(specialPoint, a, b, c, d) > 0) {
					DX = 1;
				}
				else if (__f(specialPoint, a, b, c, d) < 0) {
					DX = -1;
				}
			}
			else if (a < 0) {
				if (__f(specialPoint, a, b, c, d) > 0) {
					DX = -1;
				}
				else if (__f(specialPoint, a, b, c, d) < 0) {
					DX = 1;
				}
			}

		}

		double start = specialPoint - DX;
		double x0 = start;
		// double result[3];

		for (int i = 0; i < solves; i++) {
			double x1 = 0;
			int itCount = 0;
			do
			{
				if (itCount)
					x0 = x1;

				x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
				itCount++;

			} while (__mabs(x1 - x0) > EPS && itCount < 100000);
			results[num_solutions] = (x1);
			num_solutions++;
			start = start + DX;
			x0 = start;
		}
	}


	__device__ __host__ void __NewtonSolverForCubicEquation_satbleNeohook(const double& a, const double& b, const double& c, const double& d, double* results, int& num_solutions, double EPS)
	{
		double DX = 0;
		num_solutions = 0;
		double specialPoint = -b / 3;
		double pos[2];
		int solves = 1;
		double delta = 4 * b * b - 12 * c;
		double sign = -1;
		if (delta > 0) {
			pos[0] = (sqrt(delta) - 2 * b) / 6;
			pos[1] = (-sqrt(delta) - 2 * b) / 6;
			double v1 = __f(pos[0], a, b, c, d);
			double v2 = __f(pos[1], a, b, c, d);
			DX = (pos[0] - pos[1]);
			if ((v1) >= 0) {
				v1 = 0;
				results[1] = pos[0]/* + 1e-15*/;
				results[2] = pos[0]/* - 1e-15*/;
			}
			if ((v2) <= 0) {
				v2 = 0;
				results[1] = pos[1]/* + 1e-15*/;
				results[2] = pos[1]/* - 1e-15*/;
			}
			sign = v1 * v2;

			if (sign < 0) {
				solves = 3;
			}
			else {
				if ((v2 <= 0)) {
					DX = -DX;
				}
			}
		}
		else {
			results[0] = specialPoint;
			results[1] = specialPoint;
			results[2] = specialPoint;
			num_solutions = 3;
			return;
		}

		double start = specialPoint - DX;
		double x0 = start;
		//double result[3];

		for (int i = 0; i < solves; i++) {
			double x1 = 0;
			int itCount = 0;
			do
			{
				if (itCount)
					x0 = x1;

				x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
				itCount++;

			} while (__mabs(x1 - x0) > EPS && itCount < 100000);
			results[num_solutions] = (x1);
			num_solutions++;
			start = start + DX;
			x0 = start;
		}
		//printf("%f   %f    %f    %f    %f\n", specialPoint, DX, results[0], results[1], results[2]);
		num_solutions = 3;
	}

	__device__ __host__ void __SolverForCubicEquation(const double& a, const double& b, const double& c, const double& d, double* results, int& num_solutions, double EPS) {
		double A = b * b - 3 * a * c;
		double B = b * c - 9 * a * d;
		double C = c * c - 3 * b * d;
		double delta = B * B - 4 * A * C;
		num_solutions = 0;
		if (abs(A) < EPS * EPS && abs(B) < EPS * EPS) {
			results[0] = -b / 3.0 / a;
			results[1] = results[0];
			results[2] = results[0];
			num_solutions = 3;
		}
		else if (abs(delta) <= EPS * EPS) {
			double K = B / A;
			results[0] = -b / a + K;
			results[1] = -K / 2.0;
			results[2] = results[1];
			num_solutions = 3;
		}
		else if (delta < -EPS * EPS) {
			double T = (2 * A * b - 3 * a * B) / (2 * A * sqrt(A));
			double theta = acos(T);
			results[0] = (-b - 2 * sqrt(A) * cos(theta / 3.0)) / (3 * a);
			results[1] = (-b + sqrt(A) * (cos(theta / 3.0) + sqrt(3.0) * sin(theta / 3.0))) / (3 * a);
			results[2] = (-b + sqrt(A) * (cos(theta / 3.0) - sqrt(3.0) * sin(theta / 3.0))) / (3 * a);
			num_solutions = 3;
		}
		else if (delta > EPS * EPS) {
			double Y1 = A * b + 3 * a * (-B + sqrt(delta)) / 2;
			double Y2 = A * b + 3 * a * (-B - sqrt(delta)) / 2;

			results[0] = -b - cbrt(Y1) - cbrt(Y2);
			num_solutions = 1;
		}
	}

	__device__ __host__ Vector9 __Mat3x3_to_vec9_double(const Matrix3x3d& F) {

		Vector9 result;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				result.v[i * 3 + j] = F.m[j][i];
			}
		}
		return result;
	}

	__device__ __host__ void __normalized_vec9_double(Vector9& v9) {

		double length = 0;
		for (int i = 0;i < 9;i++) {
			length += v9.v[i] * v9.v[i];
		}
		length = 1.0 / sqrt(length);
		for (int i = 0;i < 9;i++) {
			v9.v[i] = v9.v[i] * length;
		}
	}

	__device__ __host__ void __normalized_vec6_double(Vector6& v6) {

		double length = 0;
		for (int i = 0;i < 6;i++) {
			length += v6.v[i] * v6.v[i];
		}
		length = 1.0 / sqrt(length);
		for (int i = 0;i < 6;i++) {
			v6.v[i] = v6.v[i] * length;
		}
	}

	__device__ __host__ Vector6 __Mat3x2_to_vec6_double(const Matrix3x2d& F) {

		Vector6 result;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 3; j++) {
				result.v[i * 3 + j] = F.m[j][i];
			}
		}
		return result;
	}

	__device__ __host__ Matrix3x3d __vec9_to_Mat3x3_double(const double vec9[9]) {
		Matrix3x3d mat;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				mat.m[j][i] = vec9[i * 3 + j];
			}
		}
		return mat;
	}

	__device__ __host__ Matrix2x2d __vec4_to_Mat2x2_double(const double vec4[4]) {
		Matrix2x2d mat;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				mat.m[j][i] = vec4[i * 2 + j];
			}
		}
		return mat;
	}

	__device__ __host__ void SVD(const Matrix3x3d& F, Matrix3x3d& Uout, Matrix3x3d& Vout, Matrix3x3d& Sigma) {
		using matview = zs::vec_view<double, zs::integer_seq<int, 3, 3>>;
		using cmatview = zs::vec_view<const double, zs::integer_seq<int, 3, 3>>;
		using vec3 = zs::vec<double, 3>;
		cmatview F_{ (const double*)F.m };
		matview UU{ (double*)Uout.m }, VV{ (double*)Vout.m };
		vec3 SS{};
		zs::tie(UU, SS, VV) = zs::math::qr_svd(F_);
		for (int i = 0; i != 3; ++i)
			for (int j = 0; j != 3; ++j) {
				Uout.m[i][j] = UU(i, j);
				Vout.m[i][j] = VV(i, j);
				Sigma.m[i][j] = (i != j ? 0. : SS[i]);
			}
	}

	__device__ __host__ void __makePD2x2(const double& a00, const double& a01, const double& a10, const double& a11, double eigenValues[2], int& num, double2 eigenVectors[2], double eps) {
		double b = -(a00 + a11), c = a00 * a11 - a10 * a01;
		double existEv = b * b - 4 * c;
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
		}
		else {
			if (existEv > 0) {
				num = 2;
				eigenValues[0] = (-b - sqrt(existEv)) / 2;
				eigenVectors[0].x = 1;
				eigenVectors[0].y = (eigenValues[0] - a00) / a01;
				double length = sqrt(eigenVectors[0].x * eigenVectors[0].x + eigenVectors[0].y * eigenVectors[0].y);
				//eigenValues[0] *= length;
				eigenVectors[0].x /= length;
				eigenVectors[0].y /= length;

				eigenValues[1] = (-b + sqrt(existEv)) / 2;
				eigenVectors[1].x = 1;
				eigenVectors[1].y = (eigenValues[1] - a00) / a01;
				length = sqrt(eigenVectors[1].x * eigenVectors[1].x + eigenVectors[1].y * eigenVectors[1].y);
				//eigenValues[1] *= length;
				eigenVectors[1].x /= length;
				eigenVectors[1].y /= length;
			}
			else if (existEv == 0) {
				num = 1;
				eigenValues[0] = (-b - sqrt(existEv)) / 2;
				eigenVectors[0].x = 1;
				eigenVectors[0].y = (eigenValues[0] - a00) / a01;
				double length = sqrt(eigenVectors[0].x * eigenVectors[0].x + eigenVectors[0].y * eigenVectors[0].y);
				//eigenValues[0] *= length;
				eigenVectors[0].x /= length;
				eigenVectors[0].y /= length;
			}
			else {
				num = 0;
			}
		}
	}


    __device__ __host__ void __M9x4_S4x4_MT4x9_Multiply(const Matrix9x4d& A, const Matrix4x4d& B, Matrix9x9d& output) {
        //Matrix12x12d output;
        Vector4 tempM;
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 4; j++) {
                double temp = 0;
                for (int k = 0; k < 4; k++) {
                    temp += A.m[i][k] * B.m[j][k];
                }
                tempM.v[j] = temp;
            }

            for (int j = 0; j < 9; j++) {
                double temp = 0;
                for (int k = 0; k < 4; k++) {
                    temp += A.m[j][k] * tempM.v[k];
                }
                output.m[i][j] = temp;
            }
        }
        //return output;
    }


    __device__ __host__ void __M12x9_S9x9_MT9x12_Multiply(const Matrix12x9d& A, const Matrix9x9d& B, Matrix12x12d& output) {
        //Matrix12x12d output = A^T @ B @ A
        Vector9 tempM;
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 9; j++) {
                double temp = 0;
                for (int k = 0; k < 9; k++) {
                    temp += A.m[i][k] * B.m[j][k];
                }
                tempM.v[j] = temp; // tempM = A^T @ B
            }

            for (int j = 0; j < 12; j++) {
                double temp = 0;
                for (int k = 0; k < 9; k++) {
                    temp += A.m[j][k] * tempM.v[k];
                }
                output.m[i][j] = temp; // tempM @ A
            }
        }
    }

    __device__ __host__ Vector4 __s_vec4_multiply(Vector4 a, double b) {
        Vector4 V;
        for (int i = 0; i < 4; i++)
            V.v[i] = a.v[i] * b;
        return V;
    }

    __device__ __host__ Vector9 __M9x4_v4_multiply(const Matrix9x4d& A, const Vector4& n) {
        Vector9 v9;
        for (int i = 0; i < 9; i++) {
            double temp = 0;
            for (int j = 0; j < 4; j++) {
                temp += A.m[i][j] * n.v[j];
            }
            v9.v[i] = temp;
        }
        return v9;
    }

    __device__ __host__ Matrix4x4d __S_Mat4x4_multiply(const Matrix4x4d& A, const double& B)
    {
        Matrix4x4d output;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                output.m[i][j] = A.m[i][j] * B;
            }
        }
        return output;
    }

    __device__ __host__ Matrix4x4d __v4_vec4_toMat4x4(Vector4 a, Vector4 b)
    {
        Matrix4x4d M;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                M.m[i][j] = a.v[i] * b.v[j];
            }
        }
        return M;
    }

    __device__ __host__ void __s_M_Mat_MT_multiply(const Matrix3x3d& A, const Matrix3x3d& B, const Matrix3x3d& C, const double& coe, Matrix3x3d& output) {
        double tvec3[3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                double temp = 0;
                for (int k = 0; k < 3; k++) {
                    temp += A.m[i][k] * B.m[k][j];
                }
                //output.m[i][j] = temp;
                tvec3[j] = temp;
            }

            for (int j = 0; j < 3; j++) {
                double temp = 0;
                for (int k = 0; k < 3; k++) {
                    temp += C.m[j][k] * tvec3[k];
                }
                output.m[i][j] = temp*coe;
                //tvec3[j] = temp;
            }
        }
    }

	// calcuate square distance between point to point
	__device__ __host__ 
	void __distancePointPoint(const double3& v0, const double3& v1, double& d) {
		d = __squaredNorm(__minus(v0, v1));
	}

	// calcuate square distance between point to triangle
	__device__ __host__ 
	void __distancePointTriangle(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d) {
		double3 b = __v_vec_cross(__minus(v2, v1), __minus(v3, v1));
		double aTb = __v_vec_dot(__minus(v0, v1), b);
		d = aTb * aTb / __squaredNorm(b);
	}

	// calculate square distance between point to edge
	__device__ __host__ 
	void __distancePointEdge(const double3& v0, const double3& v1, const double3& v2, double& d) {
		d = __squaredNorm(__v_vec_cross(__minus(v1, v0), __minus(v2, v0))) / __squaredNorm(__minus(v2, v1));
	}

	// calculate square distance between edge to edge
	__device__ __host__ 
	void __distanceEdgeEdge(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d) {
		double3 b = __v_vec_cross(__minus(v1, v0), __minus(v3, v2));//(v1 - v0).cross(v3 - v2);
		double aTb = __v_vec_dot(__minus(v2, v0), b);//(v2 - v0).dot(b);
		d = aTb * aTb / __squaredNorm(b);
	}

	// calculate square distance between edge to edge which two are parallel
	__device__ __host__ 
	void __distanceEdgeEdgeParallel(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double& d) {
		double3 b = __v_vec_cross(__v_vec_cross(__minus(v1, v0), __minus(v2, v0)), __minus(v1, v0));
		double aTb = __v_vec_dot(__minus(v2, v0), b);//(v2 - v0).dot(b);
		d = aTb * aTb / __squaredNorm(b);
	}

	__device__ __host__
	double __computeEdgeProductNorm(const double3& v0, const double3& v1, const double3& v2, const double3& v3) {
		return 1e-3 * __squaredNorm(__minus(v0, v1)) * __squaredNorm(__minus(v2, v3));
	}

	__device__ __host__ 
	double __calculateVolume(const double3* vertexes, const uint4& index) {

		double o1x = vertexes[index.y].x - vertexes[index.x].x;
		double o1y = vertexes[index.y].y - vertexes[index.x].y;
		double o1z = vertexes[index.y].z - vertexes[index.x].z;
		double3 OA = make_double3(o1x, o1y, o1z);

		double o2x = vertexes[index.z].x - vertexes[index.x].x;
		double o2y = vertexes[index.z].y - vertexes[index.x].y;
		double o2z = vertexes[index.z].z - vertexes[index.x].z;
		double3 OB = make_double3(o2x, o2y, o2z);

		double o3x = vertexes[index.w].x - vertexes[index.x].x;
		double o3y = vertexes[index.w].y - vertexes[index.x].y;
		double o3z = vertexes[index.w].z - vertexes[index.x].z;
		double3 OC = make_double3(o3x, o3y, o3z);

		double3 heightDir = MATHUTILS::__v_vec_cross(OA, OB);//OA.cross(OB);
		double bottomArea = MATHUTILS::__norm(heightDir);//heightDir.norm();
		heightDir = MATHUTILS::__normalized(heightDir);

		double volum = bottomArea * MATHUTILS::__v_vec_dot(heightDir, OC) / 6;
		return volum > 0 ? volum : -volum;
	}

	__device__ __host__ 
	double __calculateArea(const double3* vertexes, const uint3& index) {
		double3 v10 = MATHUTILS::__minus(vertexes[index.y], vertexes[index.x]);
		double3 v20 = MATHUTILS::__minus(vertexes[index.z], vertexes[index.x]);
		double area = MATHUTILS::__norm(MATHUTILS::__v_vec_cross(v10, v20));
		return 0.5 * area;
	}


	void __getTriEdges(Eigen::MatrixX3i& triElems, Eigen::MatrixX2i& tri_edges, Eigen::MatrixX2i& tri_edges_adj_points) {
		std::set<std::pair<int, int>> edge_set;
		std::map<std::pair<int, int>, std::vector<int>> edge_map;
		std::vector<Eigen::Vector2i> my_edges;

		for (int i = 0; i < triElems.rows(); i++) {
			auto tri = triElems.row(i);
			auto x = tri.x();
			auto y = tri.y();
			auto z = tri.z();

			if (x < y) {
				edge_set.insert(std::make_pair(x, y));
				edge_map[std::make_pair(x, y)].emplace_back(z);
			}
			else {
				edge_set.insert(std::make_pair(y, x));
				edge_map[std::make_pair(y, x)].emplace_back(z);
			}

			if (y < z) {
				edge_set.insert(std::make_pair(y, z));
				edge_map[std::make_pair(y, z)].emplace_back(x);
			}
			else {
				edge_set.insert(std::make_pair(z, y));
				edge_map[std::make_pair(z, y)].emplace_back(x);
			}

			if (x < z) {
				edge_set.insert(std::make_pair(x, z));
				edge_map[std::make_pair(x, z)].emplace_back(y);
			}
			else {
				edge_set.insert(std::make_pair(z, x));
				edge_map[std::make_pair(z, x)].emplace_back(y);
			}
		}

		std::vector<std::vector<int>> temp_edges_adj_points;
		for (auto p : edge_set) {
			if (edge_map[p].size() != 2)continue;
			my_edges.emplace_back(p.first, p.second);
			temp_edges_adj_points.emplace_back(edge_map[std::make_pair(p.first, p.second)]);
		}

		tri_edges.resize(my_edges.size(), Eigen::NoChange);
		tri_edges_adj_points.resize(temp_edges_adj_points.size(), Eigen::NoChange);
		
		for (int i = 0; i < my_edges.size(); i++) {
			tri_edges(i, 0) = my_edges[i].x();
			tri_edges(i, 1) = my_edges[i].y();
			if (temp_edges_adj_points[i].size() == 2) {
				tri_edges_adj_points(i, 0) = temp_edges_adj_points[i][0];
				tri_edges_adj_points(i, 1) = temp_edges_adj_points[i][1];
			} else {
				tri_edges_adj_points(i, 0) = temp_edges_adj_points[i][0];
				tri_edges_adj_points(i, 1) = -1;
			}
		}

	}


	void __getTriSurface(Eigen::MatrixX3i& triangles, Eigen::MatrixX3i& surfFaces) {
		int originalRows = surfFaces.rows();
		surfFaces.conservativeResize(originalRows + triangles.rows(), Eigen::NoChange);
		surfFaces.bottomRows(triangles.rows()) = triangles;
	}

	void __getTetSurface(Eigen::MatrixX4i& tetrahedras, Eigen::MatrixX3d& vertexes, Eigen::MatrixX3i& surfFaces) {
		uint64_t length = vertexes.rows();
		uint64_t tetrahedraNum = tetrahedras.rows();

		auto triangle_hash = [&](const Triangle& tri) {
			return length * (length * tri[0] + tri[1]) + tri[2];
		};

		std::unordered_map<Triangle, uint64_t, decltype(triangle_hash)> tri2Tet(4 * tetrahedraNum, triangle_hash);
		for (int i = 0; i < tetrahedraNum; i++) {
			const auto& triI4 = tetrahedras.row(i);
			uint64_t triI[4] = { triI4.x(),  triI4.y() ,triI4.z() ,triI4.w() };
			for (int j = 0;j < 4;j++) {
				const Triangle& triVInd = Triangle(triI[j % 4], triI[(1 + j) % 4], triI[(2 + j) % 4]);
				if (tri2Tet.find(Triangle(triVInd[0], triVInd[1], triVInd[2])) != tri2Tet.end()) {
					tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = tetrahedraNum + 1;
				}
				else if (tri2Tet.find(Triangle(triVInd[0], triVInd[2], triVInd[1])) != tri2Tet.end()) {
					tri2Tet[Triangle(triVInd[0], triVInd[2], triVInd[1])] = tetrahedraNum + 1;
				}
				else if (tri2Tet.find(Triangle(triVInd[1], triVInd[0], triVInd[2])) != tri2Tet.end()) {
					tri2Tet[Triangle(triVInd[1], triVInd[0], triVInd[2])] = tetrahedraNum + 1;
				}
				else if (tri2Tet.find(Triangle(triVInd[1], triVInd[2], triVInd[0])) != tri2Tet.end()) {
					tri2Tet[Triangle(triVInd[1], triVInd[2], triVInd[0])] = tetrahedraNum + 1;
				}
				else if (tri2Tet.find(Triangle(triVInd[2], triVInd[0], triVInd[1])) != tri2Tet.end()) {
					tri2Tet[Triangle(triVInd[2], triVInd[0], triVInd[1])] = tetrahedraNum + 1;
				}
				else if (tri2Tet.find(Triangle(triVInd[2], triVInd[1], triVInd[0])) != tri2Tet.end()) {
					tri2Tet[Triangle(triVInd[2], triVInd[1], triVInd[0])] = tetrahedraNum + 1;
				}
				else {
					tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = i;
				}
			}
		}

		for (const auto& triI : tri2Tet) {
			const uint64_t& tetId = triI.second;
			const Triangle& triVInd = triI.first;
			if (tetId < tetrahedraNum) {
				double3 tempvec0 = make_double3(vertexes.row(triVInd[0]).x(), vertexes.row(triVInd[0]).y(), vertexes.row(triVInd[0]).z());
				double3 tempvec1 = make_double3(vertexes.row(triVInd[1]).x(), vertexes.row(triVInd[1]).y(), vertexes.row(triVInd[1]).z());
				double3 tempvec2 = make_double3(vertexes.row(triVInd[2]).x(), vertexes.row(triVInd[2]).y(), vertexes.row(triVInd[2]).z());

				double3 vec1 = MATHUTILS::__minus(tempvec0, tempvec1);
				double3 vec2 = MATHUTILS::__minus(tempvec0, tempvec2);
				int id3 = 0;

				if (tetrahedras.row(tetId).x() != triVInd[0]
					&& tetrahedras.row(tetId).x() != triVInd[1]
					&& tetrahedras.row(tetId).x() != triVInd[2]) {
					id3 = tetrahedras.row(tetId).x();
				}
				else if (tetrahedras.row(tetId).y() != triVInd[0]
					&& tetrahedras.row(tetId).y() != triVInd[1]
					&& tetrahedras.row(tetId).y() != triVInd[2]) {
					id3 = tetrahedras.row(tetId).y();
				}
				else if (tetrahedras.row(tetId).z() != triVInd[0]
					&& tetrahedras.row(tetId).z() != triVInd[1]
					&& tetrahedras.row(tetId).z() != triVInd[2]) {
					id3 = tetrahedras.row(tetId).z();
				}
				else if (tetrahedras.row(tetId).w() != triVInd[0]
					&& tetrahedras.row(tetId).w() != triVInd[1]
					&& tetrahedras.row(tetId).w() != triVInd[2]) {
					id3 = tetrahedras.row(tetId).w();
				}

				double3 tempvec3 = make_double3(vertexes.row(id3).x(), vertexes.row(id3).y(), vertexes.row(id3).z());

				double3 vec3 = MATHUTILS::__minus(tempvec3, tempvec0);
				double3 n = MATHUTILS::__v_vec_cross(vec1, vec2);
				if (MATHUTILS::__v_vec_dot(n, vec3) > 0) {
					// surfId2TetId.push_back(tetId);
					surfFaces.conservativeResize(surfFaces.rows() + 1, Eigen::NoChange);
					surfFaces.row(surfFaces.rows() - 1) = Eigen::Vector3i(triVInd[0], triVInd[2], triVInd[1]);
				}
				else {
					// surfId2TetId.push_back(tetId);
					surfFaces.conservativeResize(surfFaces.rows() + 1, Eigen::NoChange);
                	surfFaces.row(surfFaces.rows() - 1) = Eigen::Vector3i(triVInd[0], triVInd[1], triVInd[2]);
				}
			}
		}

	}

	void __getColSurface(int numsimverts, Eigen::MatrixX3i& colsurface, Eigen::MatrixX3i& surfFaces) {
		colsurface.array() += numsimverts;
		int originalRows = surfFaces.rows();
		surfFaces.conservativeResize(originalRows + colsurface.rows(), Eigen::NoChange);
		surfFaces.bottomRows(colsurface.rows()) = colsurface;
	}

	void __getSurfaceVertsAndEdges(Eigen::MatrixX3i& surfFaces, Eigen::MatrixX3d& vertexes,Eigen::VectorXi& surfVerts, Eigen::MatrixX2i& surfEdges) {
		uint64_t length = vertexes.rows();
		std::vector<bool> flag(length, false);
		for (int i = 0; i < surfFaces.rows(); ++i) {
			const auto& cTri = surfFaces.row(i);

			if (!flag[cTri[0]]) {
				surfVerts.conservativeResize(surfVerts.size() + 1);
				surfVerts(surfVerts.size() - 1) = cTri[0];
				flag[cTri[0]] = true;
			}
			if (!flag[cTri[1]]) {
				surfVerts.conservativeResize(surfVerts.size() + 1);
				surfVerts(surfVerts.size() - 1) = cTri[1];
				flag[cTri[1]] = true;
			}
			if (!flag[cTri[2]]) {
				surfVerts.conservativeResize(surfVerts.size() + 1);
				surfVerts(surfVerts.size() - 1) = cTri[2];
				flag[cTri[2]] = true;
			}
		}

		std::set<std::pair<uint64_t, uint64_t>> SFEdges_set;
		for (int i = 0; i < surfFaces.rows(); ++i) {
			const auto& cTri = surfFaces.row(i);
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.x())) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.y())) == SFEdges_set.end()) {
						SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.y()));
					}
					if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.y())) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.z())) == SFEdges_set.end()) {
						SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.z()));
					}
					if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.z())) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.x())) == SFEdges_set.end()) {
						SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.x()));
					}
				}
			}
		}

		std::vector<std::pair<uint64_t, uint64_t>> tempEdge(SFEdges_set.begin(), SFEdges_set.end());
		for (const auto& edge : tempEdge) {
			surfEdges.conservativeResize(surfEdges.rows() + 1, Eigen::NoChange);
			surfEdges.row(surfEdges.rows() - 1) = Eigen::Vector2i(edge.first, edge.second);
		}
	}

	// void __getTetSurfaceFull(Eigen::VectorXi& surfVerts, Eigen::MatrixX3i& surfFaces, Eigen::MatrixX2i& surfEdges, Eigen::MatrixX3d& vertexes, Eigen::MatrixX4i& tetrahedras, Eigen::MatrixX3i& triangles) {
	// 	uint64_t length = vertexes.rows();
	// 	uint64_t tetrahedraNum = tetrahedras.rows();
	// 	uint64_t triangleNum = triangles.rows();

	// 	auto triangle_hash = [&](const Triangle& tri) {
	// 		return length * (length * tri[0] + tri[1]) + tri[2];
	// 	};

	// 	std::unordered_map<Triangle, uint64_t, decltype(triangle_hash)> tri2Tet(4 * tetrahedraNum, triangle_hash);
	// 	for (int i = 0; i < tetrahedraNum; i++) {
	// 		const auto& triI4 = tetrahedras.row(i);
	// 		uint64_t triI[4] = { triI4.x(),  triI4.y() ,triI4.z() ,triI4.w() };
	// 		for (int j = 0;j < 4;j++) {
	// 			const Triangle& triVInd = Triangle(triI[j % 4], triI[(1 + j) % 4], triI[(2 + j) % 4]);
	// 			if (tri2Tet.find(Triangle(triVInd[0], triVInd[1], triVInd[2])) != tri2Tet.end()) {
	// 				tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = tetrahedraNum + 1;
	// 			}
	// 			else if (tri2Tet.find(Triangle(triVInd[0], triVInd[2], triVInd[1])) != tri2Tet.end()) {
	// 				tri2Tet[Triangle(triVInd[0], triVInd[2], triVInd[1])] = tetrahedraNum + 1;
	// 			}
	// 			else if (tri2Tet.find(Triangle(triVInd[1], triVInd[0], triVInd[2])) != tri2Tet.end()) {
	// 				tri2Tet[Triangle(triVInd[1], triVInd[0], triVInd[2])] = tetrahedraNum + 1;
	// 			}
	// 			else if (tri2Tet.find(Triangle(triVInd[1], triVInd[2], triVInd[0])) != tri2Tet.end()) {
	// 				tri2Tet[Triangle(triVInd[1], triVInd[2], triVInd[0])] = tetrahedraNum + 1;
	// 			}
	// 			else if (tri2Tet.find(Triangle(triVInd[2], triVInd[0], triVInd[1])) != tri2Tet.end()) {
	// 				tri2Tet[Triangle(triVInd[2], triVInd[0], triVInd[1])] = tetrahedraNum + 1;
	// 			}
	// 			else if (tri2Tet.find(Triangle(triVInd[2], triVInd[1], triVInd[0])) != tri2Tet.end()) {
	// 				tri2Tet[Triangle(triVInd[2], triVInd[1], triVInd[0])] = tetrahedraNum + 1;
	// 			}
	// 			else {
	// 				tri2Tet[Triangle(triVInd[0], triVInd[1], triVInd[2])] = i;
	// 			}
	// 		}
	// 	}

	// 	for (const auto& triI : tri2Tet) {
	// 		const uint64_t& tetId = triI.second;
	// 		const Triangle& triVInd = triI.first;
	// 		if (tetId < tetrahedraNum) {
	// 			double3 tempvec0 = make_double3(vertexes.row(triVInd[0]).x(), vertexes.row(triVInd[0]).y(), vertexes.row(triVInd[0]).z());
	// 			double3 tempvec1 = make_double3(vertexes.row(triVInd[1]).x(), vertexes.row(triVInd[1]).y(), vertexes.row(triVInd[1]).z());
	// 			double3 tempvec2 = make_double3(vertexes.row(triVInd[2]).x(), vertexes.row(triVInd[2]).y(), vertexes.row(triVInd[2]).z());

	// 			double3 vec1 = MATHUTILS::__minus(tempvec0, tempvec1);
	// 			double3 vec2 = MATHUTILS::__minus(tempvec0, tempvec2);
	// 			int id3 = 0;

	// 			if (tetrahedras.row(tetId).x() != triVInd[0]
	// 				&& tetrahedras.row(tetId).x() != triVInd[1]
	// 				&& tetrahedras.row(tetId).x() != triVInd[2]) {
	// 				id3 = tetrahedras.row(tetId).x();
	// 			}
	// 			else if (tetrahedras.row(tetId).y() != triVInd[0]
	// 				&& tetrahedras.row(tetId).y() != triVInd[1]
	// 				&& tetrahedras.row(tetId).y() != triVInd[2]) {
	// 				id3 = tetrahedras.row(tetId).y();
	// 			}
	// 			else if (tetrahedras.row(tetId).z() != triVInd[0]
	// 				&& tetrahedras.row(tetId).z() != triVInd[1]
	// 				&& tetrahedras.row(tetId).z() != triVInd[2]) {
	// 				id3 = tetrahedras.row(tetId).z();
	// 			}
	// 			else if (tetrahedras.row(tetId).w() != triVInd[0]
	// 				&& tetrahedras.row(tetId).w() != triVInd[1]
	// 				&& tetrahedras.row(tetId).w() != triVInd[2]) {
	// 				id3 = tetrahedras.row(tetId).w();
	// 			}

	// 			double3 tempvec3 = make_double3(vertexes.row(id3).x(), vertexes.row(id3).y(), vertexes.row(id3).z());

	// 			double3 vec3 = MATHUTILS::__minus(tempvec3, tempvec0);
	// 			double3 n = MATHUTILS::__v_vec_cross(vec1, vec2);
	// 			if (MATHUTILS::__v_vec_dot(n, vec3) > 0) {
	// 				// surfId2TetId.push_back(tetId);
	// 				surfFaces.conservativeResize(surfFaces.rows() + 1, Eigen::NoChange);
	// 				surfFaces.row(surfFaces.rows() - 1) = Eigen::Vector3i(triVInd[0], triVInd[2], triVInd[1]);
	// 			}
	// 			else {
	// 				// surfId2TetId.push_back(tetId);
	// 				surfFaces.conservativeResize(surfFaces.rows() + 1, Eigen::NoChange);
    //             	surfFaces.row(surfFaces.rows() - 1) = Eigen::Vector3i(triVInd[0], triVInd[1], triVInd[2]);
	// 			}
	// 		}
	// 	}

	// 	for (int i = 0; i < triangleNum; i++) {
	// 		surfFaces.conservativeResize(surfFaces.rows() + 1, Eigen::NoChange);
	// 		Eigen::Vector3i tri = triangles.row(i);
	// 		surfFaces.row(surfFaces.rows() - 1) = tri;
	// 	}







	// 	std::vector<bool> flag(length, false);
	// 	for (int i = 0; i < surfFaces.rows(); ++i) {
	// 		const auto& cTri = surfFaces.row(i);

	// 		if (!flag[cTri[0]]) {
	// 			surfVerts.conservativeResize(surfVerts.size() + 1);
	// 			surfVerts(surfVerts.size() - 1) = cTri[0];
	// 			flag[cTri[0]] = true;
	// 		}
	// 		if (!flag[cTri[1]]) {
	// 			surfVerts.conservativeResize(surfVerts.size() + 1);
	// 			surfVerts(surfVerts.size() - 1) = cTri[1];
	// 			flag[cTri[1]] = true;
	// 		}
	// 		if (!flag[cTri[2]]) {
	// 			surfVerts.conservativeResize(surfVerts.size() + 1);
	// 			surfVerts(surfVerts.size() - 1) = cTri[2];
	// 			flag[cTri[2]] = true;
	// 		}
	// 	}

	// 	std::set<std::pair<uint64_t, uint64_t>> SFEdges_set;
	// 	for (int i = 0; i < surfFaces.rows(); ++i) {
	// 		const auto& cTri = surfFaces.row(i);
	// 		for (int j = 0; j < 3; j++) {
	// 			for (int k = 0; k < 3; k++) {
	// 				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.x())) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.y())) == SFEdges_set.end()) {
	// 					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.y()));
	// 				}
	// 				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.y())) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.z())) == SFEdges_set.end()) {
	// 					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.z()));
	// 				}
	// 				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.z())) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.x())) == SFEdges_set.end()) {
	// 					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.x()));
	// 				}
	// 			}
	// 		}
	// 	}

	// 	std::vector<std::pair<uint64_t, uint64_t>> tempEdge(SFEdges_set.begin(), SFEdges_set.end());
	// 	for (const auto& edge : tempEdge) {
	// 		surfEdges.conservativeResize(surfEdges.rows() + 1, Eigen::NoChange);
	// 		surfEdges.row(surfEdges.rows() - 1) = Eigen::Vector2i(edge.first, edge.second);
	// 	}

	// }


	int __getVertNeighbours(int vertexNum, Eigen::MatrixX4i& tetrahedras, Eigen::MatrixX3i& triangles, std::vector<unsigned int>& neighborList, std::vector<unsigned int>& neighborStart, std::vector<unsigned int>& neighborNum) {

		std::vector<std::vector<unsigned int>> vertNeighbors;

		vertNeighbors.resize(vertexNum);
		std::set<std::pair<uint64_t, uint64_t>> SFEdges_set;
		for (int i = 0; i < tetrahedras.rows(); i++) {
			uint64_t edgI[4] = { tetrahedras(i, 0), tetrahedras(i, 1), tetrahedras(i, 2), tetrahedras(i, 3) };
			if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[0], edgI[1])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[1], edgI[0])) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[0], edgI[1]));
			}
			if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[0], edgI[2])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[2], edgI[0])) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[0], edgI[2]));
			}
			if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[0], edgI[3])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[3], edgI[0])) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[0], edgI[3]));
			}
			if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[1], edgI[2])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[2], edgI[1])) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[1], edgI[2]));
			}
			if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[1], edgI[3])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[3], edgI[1])) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[1], edgI[3]));
			}
			if (SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[2], edgI[3])) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint64_t, uint64_t>(edgI[3], edgI[2])) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint64_t, uint64_t>(edgI[2], edgI[3]));
			}
		}


		for (int i = 0; i < triangles.rows(); i++) {
			const auto& cTri = triangles.row(i);
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.x())) == SFEdges_set.end() &&
				SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.y())) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.y()));
			}
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.y())) == SFEdges_set.end() &&
				SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.z())) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.y(), cTri.z()));
			}
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x(), cTri.z())) == SFEdges_set.end() &&
				SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.x())) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.z(), cTri.x()));
			}
		}

		for (const auto& edgI : SFEdges_set) {
			vertNeighbors[edgI.first].push_back(edgI.second);
			vertNeighbors[edgI.second].push_back(edgI.first);
		}
		//neighborStart.push_back(0);
		neighborNum.resize(vertexNum);
		int offset = 0;
		for (int i = 0; i < vertexNum; i++) {
			for (int j = 0; j < vertNeighbors[i].size(); j++) {
				neighborList.push_back(vertNeighbors[i][j]);
			}
			neighborStart.push_back(offset);
			offset += vertNeighbors[i].size();
			neighborNum[i] = vertNeighbors[i].size();
		}

		return neighborStart[vertexNum - 1] + neighborNum[vertexNum - 1];
	}

}


namespace MATHUTILS {

__global__
void __reduct_max_double(double* _double1Dim, int number) {
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;

	extern __shared__ double tep[];

	if (idx >= number) return;
	//int cfid = tid + CONFLICT_FREE_OFFSET(tid);
	double temp = _double1Dim[idx];

	__threadfence();

	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	int warpNum;
	//int tidNum = 32;
	if (blockIdx.x == gridDim.x - 1) {
		//tidNum = numbers - idof;
		warpNum = ((number - idof + 31) >> 5);
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}
	for (int i = 1; i < 32; i = (i << 1)) {
		double tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
		temp = MATHUTILS::__m_max(temp, tempMax);
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
			double tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
			temp = MATHUTILS::__m_max(temp, tempMax);
		}
	}
	if (threadIdx.x == 0) {
		_double1Dim[blockIdx.x] = temp;
	}
}


__global__
void _reduct_min_double(double* _double1Dim, int number) {
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;

	extern __shared__ double tep[];

	if (idx >= number) return;
	//int cfid = tid + CONFLICT_FREE_OFFSET(tid);
	double temp = _double1Dim[idx];

	__threadfence();


	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	int warpNum;
	//int tidNum = 32;
	if (blockIdx.x == gridDim.x - 1) {
		//tidNum = numbers - idof;
		warpNum = ((number - idof + 31) >> 5);
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}
	for (int i = 1; i < 32; i = (i << 1)) {
		double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
		temp = MATHUTILS::__m_min(temp, tempMin);
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
			double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
			temp = MATHUTILS::__m_min(temp, tempMin);
		}
	}
	if (threadIdx.x == 0) {
		_double1Dim[blockIdx.x] = temp;
	}
}


__global__
void _reduct_max_double3_to_double(const double3* _double3Dim, double* _double1Dim, int number) {
	int idof = blockIdx.x * blockDim.x;
	int idx = threadIdx.x + idof;
	extern __shared__ double tep[];
	if (idx >= number) return;

	//int cfid = tid + CONFLICT_FREE_OFFSET(tid);
	double3 tempMove = _double3Dim[idx];

	double temp = MATHUTILS::__m_max(MATHUTILS::__m_max(abs(tempMove.x), abs(tempMove.y)), abs(tempMove.z));

	int warpTid = threadIdx.x % 32;
	int warpId = (threadIdx.x >> 5);
	int warpNum;
	//int tidNum = 32;
	if (blockIdx.x == gridDim.x - 1) {
		//tidNum = numbers - idof;
		warpNum = ((number - idof + 31) >> 5);
	}
	else {
		warpNum = ((blockDim.x) >> 5);
	}
	for (int i = 1; i < 32; i = (i << 1)) {
		double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
		temp = MATHUTILS::__m_max(temp, tempMin);
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
			double tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
			temp = MATHUTILS::__m_max(temp, tempMin);
		}
	}
	if (threadIdx.x == 0) {
		_double1Dim[blockIdx.x] = temp;
	}
}

__global__
void __add_reduction(double* mem, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double tep[];

    if (idx >= numbers) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double temp = mem[idx];

    __threadfence();

    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else {
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


__global__
void _reduct_M_double2(double2* _double2Dim, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ double2 sdata[];

    if (idx >= number) return;
    //int cfid = tid + CONFLICT_FREE_OFFSET(tid);
    double2 temp = _double2Dim[idx];

    __threadfence();


    int warpTid = threadIdx.x % 32;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    //int tidNum = 32;
    if (blockIdx.x == gridDim.x - 1) {
        //tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    }
    else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < 32; i = (i << 1)) {
        double tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        double tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = MATHUTILS::__m_max(temp.x, tempMin);
        temp.y = MATHUTILS::__m_max(temp.y, tempMax);
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
            double tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            double tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = MATHUTILS::__m_max(temp.x, tempMin);
            temp.y = MATHUTILS::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _double2Dim[blockIdx.x] = temp;
    }
}


}

