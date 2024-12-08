#include "layer.h"
#include <random>

// Constructor: Initialize weights and biases
Layer::Layer(int num_neurons, int input_size,
             std::function<std::vector<double>(const std::vector<double> &)> activation,
             std::function<std::vector<double>(const std::vector<double> &)> activation_derivative)
    : num_neurons(num_neurons), input_size(input_size),
      activation(activation), activation_derivative(activation_derivative)
{
    // Random initialization of weights and biases
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    weights.resize(num_neurons, Vector(input_size));
    biases.resize(num_neurons);

    for (int i = 0; i < num_neurons; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights[i][j] = dis(gen);
        }
        biases[i] = dis(gen);
    }
}

// Utility: Compute dot product
double Layer::dot_product(const Vector &a, const Vector &b)
{
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

void Layer::forward(const Vector &input, ComputationGraph &graph)
{
    graph.add_node(
        // Forward pass function
        [this](const Vector &input)
        {
            Vector linear_output(num_neurons);
            for (int i = 0; i < num_neurons; ++i)
            {
                linear_output[i] = dot_product(input, weights[i]) + biases[i];
            }

            return activation(linear_output);
        },

        // Backward pass function
        [this](const Vector &input, const Vector &downstream_gradient) -> std::tuple<Matrix, Vector, Vector>
        {
            Vector linear_output(num_neurons);
            Vector activation_grad(num_neurons);

            // Compute linear outputs and activation gradients
            for (int i = 0; i < num_neurons; ++i)
            {
                linear_output[i] = dot_product(input, weights[i]) + biases[i];
            }
            activation_grad = activation_derivative(linear_output);

            // Precompute common factor: activation_grad[i] * downstream_gradient[i]
            Vector grad_factor(num_neurons);
            for (int i = 0; i < num_neurons; ++i)
            {
                grad_factor[i] = activation_grad[i] * downstream_gradient[i];
            }

            // Compute input gradient
            Vector input_gradient(input.size(), 0.0);
            for (int i = 0; i < num_neurons; ++i)
            {
                for (size_t j = 0; j < input.size(); ++j)
                {
                    input_gradient[j] += weights[i][j] * grad_factor[i];
                }
            }

            // Compute weight gradients
            Matrix weight_gradient(num_neurons, Vector(input.size(), 0.0));
            for (int i = 0; i < num_neurons; ++i)
            {
                for (size_t j = 0; j < input.size(); ++j)
                {
                    weight_gradient[i][j] = input[j] * grad_factor[i];
                }
            }

            // Compute bias gradients
            Vector bias_gradient(num_neurons);
            for (int i = 0; i < num_neurons; ++i)
            {
                bias_gradient[i] = grad_factor[i];
            }

            return {weight_gradient, bias_gradient, input_gradient};
        });
}

Matrix &Layer::get_weights()
{
    return weights;
}

Vector &Layer::get_biases()
{
    return biases;
}
