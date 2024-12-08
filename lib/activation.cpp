#include "activation.h"
#include <cmath>
#include <vector>
#include<algorithm>

namespace Activation {

    std::vector<double> relu(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = x[i] > 0 ? x[i] : 0;
        }
        return result;
    }

    std::vector<double> relu_derivative(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = x[i] > 0 ? 1 : 0;
        }
        return result;
    }

    std::vector<double> sigmoid(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = 1.0 / (1.0 + std::exp(-x[i]));
        }
        return result;
    }

    std::vector<double> sigmoid_derivative(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            double sigmoid_x = sigmoid({x[i]})[0];  // Apply sigmoid to each element
            result[i] = sigmoid_x * (1 - sigmoid_x);
        }
        return result;
    }

    std::vector<double> tanh(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::tanh(x[i]);
        }
        return result;
    }

    std::vector<double> tanh_derivative(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            double tanh_x = tanh({x[i]})[0];  
            result[i] = 1 - tanh_x * tanh_x;
        }
        return result;
    }

    std::vector<double> softmax(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        double max_val = *std::max_element(x.begin(), x.end());
        double sum_exp = 0.0;

        for (size_t i = 0; i < x.size(); ++i) {
            sum_exp += std::exp(x[i] - max_val); 
        }

        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val) / sum_exp;
        }

        return result;
    }

    std::vector<double> softmax_derivative(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        std::vector<double> softmax_vals = softmax(x);

        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = softmax_vals[i] * (1 - softmax_vals[i]); 
        }
        return result;
    }

}
