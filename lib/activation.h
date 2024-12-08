#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>

namespace Activation {
    std::vector<double> relu(const std::vector<double>& x);
    std::vector<double> relu_derivative(const std::vector<double>& x);
    std::vector<double> sigmoid(const std::vector<double>& x);
    std::vector<double> sigmoid_derivative(const std::vector<double>& x);
    std::vector<double> tanh(const std::vector<double>& x);
    std::vector<double> tanh_derivative(const std::vector<double>& x);
    std::vector<double> softmax(const std::vector<double>& x);
    std::vector<double> softmax_derivative(const std::vector<double>& x);
}

#endif
