#include "backpropagation.h"

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

namespace Backpropagation
{

    std::pair<Matrix, Vector> compute_layer_gradients(
        const Vector &layer_error,
        const Vector &previous_activation,
        const Layer &layer)
    {

        int num_neurons = layer.num_neurons;
        int input_size = previous_activation.size();

        Matrix grad_weights(num_neurons, Vector(input_size));
        Vector grad_biases(num_neurons);

        #pragma omp parallel for
        for (size_t i = 0; i < num_neurons; ++i)
        {
            for (size_t j = 0; j < input_size; ++j)
            {
                grad_weights[i][j] = layer_error[i] * previous_activation[j];
            }
            grad_biases[i] = layer_error[i];
        }

        return {grad_weights, grad_biases};
    }

    std::vector<std::pair<Matrix, Vector>> compute_gradients(
        const Vector &y_true,
        const Vector &y_pred,
        const std::vector<Layer> &layers,
        const Matrix &activations)
    {
        Vector output_error = Loss::mean_squared_error_derivative(y_true, y_pred);

        std::vector<std::pair<Matrix, Vector>> gradients;

        // Parallelize the loop over layers
        #pragma omp parallel for
        for (int l = layers.size() - 1; l >= 0; l--)
        {
            Vector layer_error(layers[l].num_neurons);

            // Parallelize the loop over neurons in the layer
            #pragma omp parallel for
            for (size_t i = 0; i < layers[l].num_neurons; i++)
            {
                layer_error[i] = output_error[i] * layers[l].activation_fn_derivative(activations[l][i]);
            }

            auto layer_gradient = compute_layer_gradients(layer_error, activations[l], layers[l]);
            gradients.push_back(layer_gradient);

            if (l > 0)
            {
                Vector new_error(layers[l - 1].num_neurons, 0.0);

                #pragma omp parallel for
                for (size_t i = 0; i < layers[l].num_neurons; i++)
                {
                    for (size_t j = 0; j < layers[l - 1].num_neurons; j++)
                    {
                        new_error[j] += layer_error[i] * layers[l].W[i][j];
                    }
                }
                output_error = new_error;
            }
        }

        std::reverse(gradients.begin(), gradients.end());
        return gradients;
    }

}
