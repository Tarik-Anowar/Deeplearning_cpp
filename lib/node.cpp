#include "node.h"

Vector Node::forward(const Vector& input) {
    forward_input = input; 
    forward_output = forward_fn(forward_input);
    return forward_output;
}

Vector Node::backward(const Vector& downstream_gradient) {
    auto [weight_grad, bias_grad, input_grad] = backward_fn(forward_input, downstream_gradient);
    backward_gradient = {weight_grad, bias_grad}; 
    return input_grad; 
}
