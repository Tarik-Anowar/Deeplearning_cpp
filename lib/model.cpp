#include "model.h"
#include "loss.h"
#include <iostream>
#include <algorithm>

Model::Model() {}

void Model::add_layer(int num_neurons, int input_size,
                      std::function<Vector(const Vector&)> activation,
                      std::function<Vector(const Vector&)> activation_derivative)
{
    layers.emplace_back(num_neurons, input_size, activation, activation_derivative);
}

void Model::compile(Optimizer *optimizer,
                    std::function<double(const Vector &, const Vector &)> loss_fn,
                    std::function<Vector(const Vector &, const Vector &)> loss_derivative_fn)
{
    this->optimizer = optimizer;
    this->loss_fn = loss_fn;
    this->loss_derivative_fn = loss_derivative_fn;
}

void Model::fit(const Matrix &X, const Matrix &y, int epochs, int batch_size)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double total_loss = 0.0;

        for (size_t i = 0; i < X.size(); ++i)
        {
            graph = ComputationGraph();

            for (auto &layer : layers)
            {
                layer.forward({0}, graph);
            }
            Vector y_pred = graph.forward_pass(X[i]);
            // if ((epoch+1) % 10 == 0)
            // {
            //     std::cout << "ypred = ";
            //     for (auto a : y_pred)
            //     {
            //         std::cout << a << " ";
            //     }
            //     std::cout << std::endl;
            //     std::cout << "ytrue = ";
            //     for (auto a : y[i])
            //     {
            //         std::cout << a << " ";
            //     }
            //     std::cout << std::endl;
            // }
            if (y_pred.size() != y[i].size())
            {
                std::cout << "y[i].size = " << y[i].size() << ", y_pred.size = " << y_pred.size() << std::endl;
                y_pred.resize(y[i].size(), 0);
            }

            double loss = loss_fn(y[i], y_pred);
            total_loss += loss;

            graph.backward_pass(loss_derivative_fn(y[i], y_pred));

            std::vector<std::pair<Matrix, Vector>> gradients;
            for (size_t j = 0; j < layers.size(); ++j)
            {
                gradients.push_back(graph.nodes[j].backward_gradient);
            }

            optimizer->update_weights(layers, gradients);
            
        }

        std::cout << "Epoch " << epoch + 1 << " - Loss: " << total_loss / X.size() << std::endl;
    }
}

Vector Model::predict(const Vector &X)
{
    Vector y_pred = graph.forward_pass(X);
    double max_value = *std::max_element(y_pred.begin(), y_pred.end());

    Vector binary_vec(y_pred.size(), 0);
    for (size_t i = 0; i < y_pred.size(); ++i)
    {
        if (y_pred[i] == max_value)
        {
            binary_vec[i] = 1;
        }
    }

    return binary_vec;
}
double Model::evaluate(const Matrix &y_true, const Matrix &y_pred)
{
    if (y_true.size() != y_pred.size())
    {
        std::cerr << "Mismatch in number of samples between true labels and predicted values!" << std::endl;
        return -1;
    }

    int correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i)
    {
        if (std::equal(y_true[i].begin(), y_true[i].end(), y_pred[i].begin()))
        {
            correct++;
        }
    }
    double accuracy = static_cast<double>(correct) / y_true.size();

    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    return accuracy; 
}
