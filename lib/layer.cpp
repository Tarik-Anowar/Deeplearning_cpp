#include "layer.h"

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Layer::Layer(int num_neurons, int input_size, std::function<double(double)> activation_fn, std::function<double(double)> activation_deriv)
    : num_neurons(num_neurons), activation_fn(activation_fn), activation_fn_derivative(activation_deriv) {

    std::srand(static_cast<unsigned int>(std::time(0)));
    W.resize(num_neurons, Vector(input_size));
    b.resize(num_neurons);

    #pragma omp parallel for
    for (int i = 0; i < num_neurons; ++i) {
        for (int j = 0; j < input_size; ++j) {
            W[i][j] = (std::rand() % 1000) / 1000.0; // Random value between 0 and 1
        }
        b[i] = (std::rand() % 1000) / 1000.0; // Random bias
    }
};


Vector Layer::forward(const Vector &input)
{
    Vector output(num_neurons);
    #pragma omp parallel for
    for (int i = 0; i < num_neurons; ++i)
    {
        double sum = std::inner_product(W[i].begin(), W[i].end(), input.begin(), 0.0) + b[i];
        output[i] = activation_fn(sum);
    }

    return output;
}