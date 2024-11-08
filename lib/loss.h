// Loss.h

#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include<vector>
#include<iostream>
#include<omp.h>
#include<cmath>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

namespace Loss {
    double mean_squared_error(const Vector& y_true, const Vector& y_pred);
    Vector mean_squared_error_derivative(const Vector& y_true,const Vector& y_pred);

    double categorical_cross_entropy(const Vector& y_true, const Vector& y_pred);
    Vector categorical_cross_entropy_derivative(const Vector& y_true, const Vector& y_pred);
}

#endif 
