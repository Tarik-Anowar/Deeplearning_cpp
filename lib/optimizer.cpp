#include "optimizer.h"
#include <cmath>

AdamOptimizer::AdamOptimizer(double lr, double b1, double b2, double eps)
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0)
{}

void AdamOptimizer::initialize_moments(const Matrix &W, const Vector &b) {
    mW = Matrix(W.size(), Vector(W[0].size(), 0.0));
    vW = Matrix(W.size(), Vector(W[0].size(), 0.0));
    mb = Vector(b.size(), 0.0);
    vb = Vector(b.size(), 0.0);
}

void AdamOptimizer::update_weights(
    Matrix &W,
    Vector &b,
    const Matrix &grad_W,
    const Vector &grad_b
) {
    // Initialize moment vectors if they are empty
    if (mW.empty() || mb.empty()) {
        initialize_moments(W, b);
    }

    t++;

    for (size_t i = 0; i < W.size(); i++) {
        for (size_t j = 0; j < W[0].size(); j++) {
            mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * grad_W[i][j];
            vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * grad_W[i][j] * grad_W[i][j];

            double mW_hat = mW[i][j] / (1 - std::pow(beta1, t));
            double vW_hat = vW[i][j] / (1 - std::pow(beta2, t));

            W[i][j] -= learning_rate * mW_hat / (std::sqrt(vW_hat) + epsilon);
        }
    }

    for (size_t i = 0; i < b.size(); i++) {
        mb[i] = beta1 * mb[i] + (1 - beta1) * grad_b[i];
        vb[i] = beta2 * vb[i] + (1 - beta2) * grad_b[i] * grad_b[i];

        double mb_hat = mb[i] / (1 - std::pow(beta1, t));
        double vb_hat = vb[i] / (1 - std::pow(beta2, t));

        b[i] -= learning_rate * mb_hat / (std::sqrt(vb_hat) + epsilon);
    }
}
