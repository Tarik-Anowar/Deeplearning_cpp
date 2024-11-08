#include "linalgebra.h"
#include <cmath>
#include <iostream>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Matrix LinearAlgebra::dot(const Matrix& a, const Matrix& b) {
    int rowA = a.size();
    int colA = a[0].size();
    int rowB = b.size();
    int colB = b[0].size();
    if (colA != rowB) {
        throw std::invalid_argument("The matrices are not compatible for multiplication");
    }
    Matrix result(rowA, Vector(colB, 0.0));
    for (int i = 0; i < rowA; ++i) {
        for (int j = 0; j < colB; ++j) {
            for (int k = 0; k < colA; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

Matrix LinearAlgebra::transpose(const Matrix& m) {
    int row = m.size();
    int col = m[0].size();
    Matrix result(col, Vector(row, 0.0));
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result[j][i] = m[i][j];
        }
    }
    return result;
}

Matrix LinearAlgebra::elementWiseMul(const Matrix& a, const Matrix& b) {
    int row = a.size();
    int col = a[0].size();
    if (row != b.size() || col != b[0].size()) {
        throw std::invalid_argument("The matrices must have the same dimensions for element-wise multiplication");
    }
    Matrix result(row, Vector(col, 0.0));
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}

Matrix LinearAlgebra::elementWiseAdd(const Matrix& a, const Matrix& b) {
    int row = a.size();
    int col = a[0].size();
    if (row != b.size() || col != b[0].size()) {
        throw std::invalid_argument("The matrices must have the same dimensions for element-wise addition");
    }
    Matrix result(row, Vector(col, 0.0));
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    return result;
}

Matrix LinearAlgebra::scalarMul(const Matrix& m, double scalar) {
    int row = m.size();
    int col = m[0].size();
    Matrix result(row, Vector(col, 0.0));
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            result[i][j] = m[i][j] * scalar;
        }
    }
    return result;
}
