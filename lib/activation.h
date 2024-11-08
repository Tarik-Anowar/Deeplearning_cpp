
#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>

namespace Activation {
    double relu(double x);
    double relu_derivative(double x);
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    double tanh(double x);
    double tanh_derivative(double x);
}

#endif
