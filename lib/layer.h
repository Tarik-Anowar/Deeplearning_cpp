#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include<omp.h>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class Layer
{

public:
    int num_neurons;
    Matrix W;
    std::vector<double> b;
    std::function<double(double)> activation_fn;
    std::function<double(double)> activation_fn_derivative;

    Layer(int num_neurons, int input_size, std::function<double(double)> activation_fn, std::function<double(double)> activation_fn_derivative);
    Vector forward(const Vector &input);
};

#endif