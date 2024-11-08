#include "model.h"
#include <iostream>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

Model::Model() {}

void Model::add_layer(int num_neurons, int input_size, std::function<double(double)> activation,
                      std::function<double(double)> activation_derivative)
{
    layers.emplace_back(num_neurons, input_size, activation, activation_derivative);
}

void Model::compile(Optimizer *opt,
                    std::function<double(const Vector &, const Vector &)> loss,
                    std::function<Vector(const Vector &, const Vector &)> loss_deriv)
{
    optimizer = opt;
    loss_fn = loss;
    loss_derivative_fn = loss_deriv;
}

void Model::fit(const Matrix &X, const Matrix &y, int epochs, int batch_size)
{
    #pragma omp parallel for
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double total_loss = 0.0;
        for (size_t i = 0; i < X.size(); i++)
        {
            Matrix activations = {X[i]};

            for (auto &layer : layers)
            {
                activations.push_back(layer.forward(activations.back()));
            }

            double loss = loss_fn(y[i], activations.back());
            total_loss += loss;

            std::vector<std::pair<Matrix, Vector>> gradients = Backpropagation::compute_gradients(y[i], activations.back(), layers, activations);

            for (size_t j = 0; j < layers.size(); j++)
            {
                optimizer->update_weights(layers[j].W, layers[j].b, gradients[j].first, gradients[j].second);
            }
        }
        std::cout << "Epoch " << epoch + 1 << " - Loss: " << total_loss / X.size() << std::endl;
    }
}

Vector Model::predict(const Vector &X)
{
    Vector input = X;

    for (auto &layer : layers)
    {
        input = layer.forward(input);
    }

    return input;
}
double Model::evaluate(const Matrix &X_test, const Matrix &y_test)
{
    int correct_predictions = 0;

    for (size_t i = 0; i < X_test.size(); i++)
    {
        Vector output = predict(X_test[i]);

        int predicted_class = std::distance(output.begin(), std::max_element(output.begin(), output.end()));

        int actual_class = std::distance(y_test[i].begin(), std::max_element(y_test[i].begin(), y_test[i].end()));

        if (predicted_class == actual_class)
        {
            correct_predictions++;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / X_test.size();
    std::cout << "Evaluation Accuracy: " << accuracy * 100 << "%" << std::endl;

    return accuracy;
}

