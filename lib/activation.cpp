
#include "activation.h"
#include <cmath>

namespace Activation {

    double relu(double x) {
        return x > 0 ? x : 0;
    }

    double relu_derivative(double x) {
        return x > 0 ? 1 : 0;
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x) {
        double sigmoid_x = sigmoid(x);
        return sigmoid_x * (1 - sigmoid_x);
    }

    double tanh(double x) {
        return std::tanh(x);
    }

    double tanh_derivative(double x) {
        double tanh_x = tanh(x);
        return 1 - tanh_x * tanh_x;
    }

}
