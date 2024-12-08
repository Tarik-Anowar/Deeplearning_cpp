#include "optimizer.h"
#include "layer.h"
#include <cmath>
#include <iostream>

AdamOptimizer::AdamOptimizer(double learning_rate,
                             double beta1,
                             double beta2,
                             double epsilon)
    : learning_rate(learning_rate),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon),
      timestep(0) {}

void AdamOptimizer::initialize_momentum(Layer &layer)
{
    m.resize(layer.get_weights().size(), Vector(layer.get_weights()[0].size(), 0));
    v.resize(layer.get_weights().size(), Vector(layer.get_weights()[0].size(), 0));
    
    m_bias.resize(layer.get_biases().size(), 0);
    v_bias.resize(layer.get_biases().size(), 0);
}

void AdamOptimizer::update_weights(std::vector<Layer>& layers, const std::vector<std::pair<Matrix, Vector>>& gradients)
{
    timestep++;

    for (size_t l = 0; l < layers.size(); ++l)
    {
        Layer& layer = layers[l];
        const auto& gradient = gradients[l];

        if (m.empty()) {
            initialize_momentum(layer);
        }
        Matrix& weights = layer.get_weights();
        const Matrix& weight_grad = gradient.first;
        Vector& biases = layer.get_biases();
        const Vector& bias_grad = gradient.second;
        // Check if dimensions match for weights and their gradients
        if (weights.size() != weight_grad.size() || weights[0].size() != weight_grad[0].size()) {
            std::cerr << "Weight dimensions do not match gradient dimensions for layer " << l << std::endl;
            std::cerr << "Weight dimensions: " << weights.size() << " x " << (weights.empty() ? 0 : weights[0].size()) << std::endl;
            std::cerr << "Gradient dimensions: " << weight_grad.size() << " x " << (weight_grad.empty() ? 0 : weight_grad[0].size()) << std::endl;
            throw std::runtime_error("Weight dimensions do not match gradient dimensions.");
        }

        // Check if dimensions match for biases and their gradients
        if (biases.size() != bias_grad.size()) {
            std::cerr << "Bias dimensions do not match gradient dimensions for layer " << l << std::endl;
            std::cerr << "Bias dimensions: " << biases.size() << std::endl;
            std::cerr << "Gradient dimensions: " << bias_grad.size() << std::endl;
            throw std::runtime_error("Bias dimensions do not match gradient dimensions.");
        }

        for (size_t i = 0; i < layer.get_weights().size(); ++i)
        {
            for (size_t j = 0; j < layer.get_weights()[i].size(); ++j)
            {
                m[i][j] = beta1 * m[i][j] + (1 - beta1) * gradient.first[i][j];
                v[i][j] = beta2 * v[i][j] + (1 - beta2) * std::pow(gradient.first[i][j], 2);

                double m_hat = m[i][j] / (1 - std::pow(beta1, timestep));
                double v_hat = v[i][j] / (1 - std::pow(beta2, timestep));

                layer.get_weights()[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }

        for (size_t i = 0; i < layer.get_biases().size(); ++i)
        {
            m_bias[i] = beta1 * m_bias[i] + (1 - beta1) * gradient.second[i];
            v_bias[i] = beta2 * v_bias[i] + (1 - beta2) * std::pow(gradient.second[i], 2);

            double m_hat_bias = m_bias[i] / (1 - std::pow(beta1, timestep));
            double v_hat_bias = v_bias[i] / (1 - std::pow(beta2, timestep));

            layer.get_biases()[i] -= learning_rate * m_hat_bias / (std::sqrt(v_hat_bias) + epsilon);
        }
    }
}

// SGD Implementation
SGD::SGD(double learning_rate) : learning_rate(learning_rate) {}
void SGD::update_weights(std::vector<Layer>& layers, const std::vector<std::pair<Matrix, Vector>>& gradients)
{
    if (layers.size() != gradients.size()) {
        throw std::runtime_error("Number of layers does not match the number of gradient pairs.");
    }

    for (size_t l = 0; l < layers.size(); ++l)
    {
        Layer& layer = layers[l];
        const auto& gradient = gradients[l];

        Matrix& weights = layer.get_weights();
        const Matrix& weight_grad = gradient.first;
        Vector& biases = layer.get_biases();
        const Vector& bias_grad = gradient.second;

        // Check if dimensions match for weights and their gradients
        if (weights.size() != weight_grad.size() || weights[0].size() != weight_grad[0].size()) {
            std::cerr << "Weight dimensions do not match gradient dimensions for layer " << l << std::endl;
            std::cerr << "Weight dimensions: " << weights.size() << " x " << (weights.empty() ? 0 : weights[0].size()) << std::endl;
            std::cerr << "Gradient dimensions: " << weight_grad.size() << " x " << (weight_grad.empty() ? 0 : weight_grad[0].size()) << std::endl;
            throw std::runtime_error("Weight dimensions do not match gradient dimensions.");
        }

        // Check if dimensions match for biases and their gradients
        if (biases.size() != bias_grad.size()) {
            std::cerr << "Bias dimensions do not match gradient dimensions for layer " << l << std::endl;
            std::cerr << "Bias dimensions: " << biases.size() << std::endl;
            std::cerr << "Gradient dimensions: " << bias_grad.size() << std::endl;
            throw std::runtime_error("Bias dimensions do not match gradient dimensions.");
        }

        for (size_t i = 0; i < weights.size(); ++i)
        {
            for (size_t j = 0; j < weights[i].size(); ++j)
            {
                weights[i][j] -= learning_rate * weight_grad[i][j];
            }
        }

        for (size_t i = 0; i < biases.size(); ++i)
        {
            biases[i] -= learning_rate * bias_grad[i];
        }
    }
}
