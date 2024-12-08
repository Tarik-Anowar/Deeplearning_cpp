#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include "computation_graph.h"
#include "layer.h"

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Base class for optimizers
class Optimizer
{
public:
    virtual ~Optimizer() = default;
    virtual void update_weights(std::vector<Layer>& layers, const std::vector<std::pair<Matrix, Vector>>& gradients) = 0;
};

class AdamOptimizer : public Optimizer {
private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int timestep;

    std::vector<std::vector<double>> m; // First moment for weights
    std::vector<std::vector<double>> v; // Second moment for weights
    std::vector<double> m_bias;         // First moment for biases
    std::vector<double> v_bias;         // Second moment for biases

public:
    AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon);
    void initialize_momentum(Layer &layer);
    void update_weights(std::vector<Layer>& layers, const std::vector<std::pair<Matrix, Vector>>& gradients) override;
};

class SGD : public Optimizer {
private:
    double learning_rate;

public:
    SGD(double learning_rate = 0.001);
    void update_weights(std::vector<Layer>& layers, const std::vector<std::pair<Matrix, Vector>>& gradients) override;
};

#endif
