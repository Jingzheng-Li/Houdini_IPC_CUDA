#pragma once

namespace MATHUTILS {
	const static double PI = 3.141592653589793238462643383279502884197169399;

	struct Vector4 {
		double v[4];
	};

	struct Vector6 {
		double v[6];
	};

	struct Vector9 {
		double v[9];
	};

	struct Vector12 {
		double v[12];
	};

	struct Matrix3x3d {
		double m[3][3];
	};

	struct Matrix4x4d {
		double m[4][4];
	};

	struct Matrix3x6d {
		double m[3][6];
	};

	struct Matrix6x3d {
		double m[6][3];
	};

	struct Matrix2x2d {
		double m[2][2];
	};

	struct Matrix3x2d {
		double m[3][2];
	};

	struct Matrix9x9d {
		double m[9][9];
	};

	struct Matrix12x12d {
		double m[12][12];
	};

	struct Matrix24x24d {
		double m[24][24];
	};

	struct Matrix36x36d {
		double m[36][36];
	};

	struct Matrix96x96d {
		double m[96][96];
	};

	struct Matrix96x96T {
		double m[96][96];
	};


	struct Matrix3x3f
    {
        float m[3][3];
    };

    struct MasMatrixSymf
    {
        Matrix3x3f M[32 * (32 + 1) / 2];
    };

	struct Matrix96x96MT {
		float m[96][96];
	};

	struct Matrix9x2d {
		double m[9][2];
	};

	struct Matrix6x2d {
		double m[6][2];
	};

	struct Matrix12x2d {
		double m[12][2];
	};

	struct Matrix9x12d {
		double m[9][12];
	};

	struct Matrix12x9d {
		double m[12][9];
	};

	struct Matrix12x6d {
		double m[12][6];
	};

	struct Matrix6x12d {
		double m[6][12];
	};

	struct Matrix6x6d {
		double m[6][6];
	};

	struct Matrix12x4d {
		double m[12][4];
	};

	struct Matrix9x4d {
		double m[9][4];
	};

	struct Matrix4x9d {
		double m[4][9];
	};

	struct Matrix6x9d {
		double m[6][9];
	};

	struct Matrix9x6d {
		double m[9][6];
	};

	struct Matrix2x3d {
		double m[2][3];
	};


}


