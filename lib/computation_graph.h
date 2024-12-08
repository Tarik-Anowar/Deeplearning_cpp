#ifndef COMPUTATIONGRAPH_H
#define COMPUTATIONGRAPH_H

#include <vector>
#include <functional>
#include "node.h"

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class ComputationGraph {
public:
    std::vector<Node> nodes;

    void add_node(const std::function<Vector(const Vector&)>& forward_fn,
                                const std::function<std::tuple<Matrix, Vector,Vector>(const Vector&, const Vector&)>& backward_fn);

    Vector forward_pass(const Vector& input);

    void backward_pass(const Vector& final_gradient);
};

#endif
