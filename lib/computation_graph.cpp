#include "computation_graph.h"

void ComputationGraph::add_node(const std::function<Vector(const Vector&)>& forward_fn,
                                const std::function<std::tuple<Matrix, Vector,Vector>(const Vector&, const Vector&)>& backward_fn) {
    nodes.emplace_back(forward_fn, backward_fn);
}

Vector ComputationGraph::forward_pass(const Vector& input) {
    auto X= input;
    for (auto& node : nodes) {
        auto output = node.forward(X);  
        X = output;
    }
    return X;
}

void ComputationGraph::backward_pass(const Vector& final_gradient) {
    Vector input_gradient = final_gradient;
    for (int i = nodes.size() - 1; i >= 0; --i) {
        auto temp = nodes[i].backward(input_gradient);
        input_gradient = temp;
    }
}