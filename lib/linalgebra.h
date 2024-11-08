#ifndef LINEAR_ALGRBRA_H
#define LINEAR_ALGRBRA_H

#include <vector>
#include <iostream>

using Matrix = std::vector<std::vector<double>>;

class LinearAlgebra {
public:
    int rows, cols;
    static Matrix dot(const Matrix& a, const Matrix& b);
    static Matrix transpose(const Matrix& m);
    static Matrix elementWiseMul(const Matrix& a, const Matrix& b);
    static Matrix elementWiseAdd(const Matrix& a, const Matrix& b);
    static Matrix scalarMul(const Matrix& m, double scalar);

};

#endif
