#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

class Optimizer
{
public:
    virtual void update_weights(Matrix &W,
                                Vector &b,
                                const Matrix &grad_W,
                                const Vector &grad_b) = 0;
};

class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(double lr, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);

    void update_weights(Matrix &W,
                        Vector &b,
                        const Matrix &grad_W,
                        const Vector &grad_b) override;

private:
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int t; 

    Matrix mW; 
    Matrix vW; 
    Vector mb;  
    Vector vb;

    void initialize_moments(const Matrix &W, const Vector &b);
};

#endif
