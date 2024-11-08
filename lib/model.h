#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "optimizer.h"
#include "backpropagation.h"
#include <vector>
#include <functional>
#include <iostream>
#include<omp.h>


using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class Model {
public:
    std::vector<Layer> layers;
    Optimizer* optimizer;
    std::function<double(const Vector&, const Vector&)> loss_fn;
    std::function<Vector(const Vector&, const Vector&)> loss_derivative_fn;

    Model();

    void add_layer(int num_neurons, int input_size,
                   std::function<double(double)> activation,
                   std::function<double(double)> activation_derivative);
    
    void compile(Optimizer* optimizer, 
                 std::function<double(const Vector&, const Vector&)> loss_fn,
                 std::function<Vector(const Vector&, const Vector&)> loss_derivative_fn);
    
    void fit(const Matrix& X, const Matrix& y, int epochs, int batch_size);

    Vector predict(const Vector& X);
    double evaluate(const Matrix &X_test, const Matrix &y_test);
};

#endif
