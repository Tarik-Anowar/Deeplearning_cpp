#ifndef NODE_H
#define NODE_H

#include <functional>
#include <vector>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

class Node {
public:
    std::function<Vector(const Vector&)> forward_fn;  // Forward pass function
    std::function<std::tuple<Matrix, Vector,Vector>(const Vector&, const Vector&)> backward_fn;  // Backward pass function

    Vector forward_input;  // Input to the node during forward pass
    Vector forward_output;  // Output of the node after forward pass
    std::pair<Matrix, Vector> backward_gradient;  // Gradients for weights and biases

    Node(std::function<Vector(const Vector&)> fwd_fn, 
         std::function<std::tuple<Matrix,Vector,Vector>(const Vector&, const Vector&)> bwd_fn)
        : forward_fn(fwd_fn), backward_fn(bwd_fn) {}

    Vector forward(const Vector& input);

    Vector backward(const Vector& downstream_gradient);
};

#endif
