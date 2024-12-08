#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <functional>
#include "layer.h"
#include "computation_graph.h"
#include "optimizer.h"

class Model {
public:
    std::vector<Layer> layers;
    Optimizer* optimizer;
    ComputationGraph graph;

    std::function<double(const Vector&, const Vector&)> loss_fn;
    std::function<Vector(const Vector&, const Vector&)> loss_derivative_fn;

    Model();

    void add_layer(int num_neurons, int input_size,
                   std::function<Vector(const Vector&)> activation,
                   std::function<Vector(const Vector&)> activation_derivative);

    void compile(Optimizer* optimizer, 
                 std::function<double(const Vector&, const Vector&)> loss_fn,
                 std::function<Vector(const Vector&, const Vector&)> loss_derivative_fn);
    
    void fit(const Matrix& X, const Matrix& y, int epochs, int batch_size);
    Vector predict(const Vector& X);
    double evaluate(const Matrix& y_true, const Matrix& y_pred);
};

#endif
