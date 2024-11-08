// Backpropagation.h

#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "layer.h"
#include "loss.h"
#include <vector>
#include <iostream>
#include <omp.h>
#include <algorithm>


using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

namespace Backpropagation {
    std::vector<std::pair<Matrix,Vector>> compute_gradients(
        const Vector& y_true,
        const Vector & y_pred,
        const std::vector<Layer>& layers,
        const Matrix& activations);
}

#endif 

