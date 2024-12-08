#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <functional>
#include "computation_graph.h"

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

class Layer {
public:
    int num_neurons;  
    int input_size;   
    Matrix weights;   
    Vector biases;    
    std::function<Vector(const Vector&)> activation;            
    std::function<Vector(const Vector&)> activation_derivative; 

    // Constructor
    Layer(int num_neurons, int input_size, 
          std::function<Vector(const Vector&)> activation,
          std::function<Vector(const Vector&)> activation_derivative);

    void forward(const Vector& input, ComputationGraph& graph);

    Matrix& get_weights();

    Vector& get_biases();

private:
    double dot_product(const Vector& a, const Vector& b);
};

#endif
