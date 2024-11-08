#include "loss.h"


using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;


namespace Loss {
    double mean_squared_error(const Vector& y_true, const Vector& y_pred) {

        double sum = 0.0;
        #pragma omp parallel for
        for (size_t i = 0; i < y_true.size(); i++) {
            sum += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i]);
        }
        return sum / static_cast<double>(y_true.size());
    }

    Vector mean_squared_error_derivative(const Vector& y_true, const Vector& y_pred) {
        Vector derivatives(y_true.size());
        #pragma omp parallel for
        for (size_t i = 0; i < y_true.size(); i++) {
            derivatives[i] = 2 * (y_true[i] - y_pred[i]) / y_true.size();
        }
        return derivatives;
    }

    double categorical_cross_entropy(const Vector& y_true, const Vector& y_pred){
        double loss = 0.0;
        for(size_t i = 0; i<y_true.size();i++){
            if(y_pred[i]>0){
                loss -= y_true[i]*std::log(y_pred[i]);
            }
        }
        return loss;
    }
    Vector categorical_cross_entropy_derivative(const Vector& y_true, const Vector& y_pred){
        Vector derivative(y_true.size());
        for(size_t i =0; i<y_true.size();i++){
            if(y_pred[i]>0){
                derivative[i]+=-y_true[i]/y_pred[i];
            }
        }
        return derivative;
    }


}

