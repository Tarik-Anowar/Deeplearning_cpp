#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include "lib/model.h"
#include "lib/optimizer.h"
#include "lib/activation.h"
#include "lib/loss.h"

using namespace Activation;
using namespace Loss;

Matrix read_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    Matrix data;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;
        
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        
        data.push_back(row);
    }

    return data;
}

void print_shape(const Matrix& matrix, const std::string& name) {
    std::cout << name << " Shape: " << matrix.size() << " x " << (matrix.empty() ? 0 : matrix[0].size()) << std::endl;
}

int main() {
    std::cout << "Loading training data..." << std::endl;
    Matrix X_train = read_csv("X_train.csv");
    Matrix y_train = read_csv("y_train.csv");
    Matrix X_test = read_csv("X_test.csv");
    Matrix y_test = read_csv("y_test.csv");

    std::cout << "Data loaded successfully!" << std::endl;

    print_shape(X_train, "X_train");
    print_shape(y_train, "y_train");
    print_shape(X_test, "X_test");
    print_shape(y_test, "y_test");

    Model model;

    model.add_layer(784, 128, relu, relu_derivative);
    model.add_layer(128, 10, sigmoid, sigmoid_derivative);


    Optimizer* optimizer = new AdamOptimizer(0.01, 0.9, 0.999, 1e-8);
    model.compile(optimizer, categorical_cross_entropy, categorical_cross_entropy_derivative);

    std::cout << "Training the model..." << std::endl;
    model.fit(X_train, y_train, 10, 32);  


    
    Matrix y_pred;

    std::cout << "Testing predictions..." << std::endl;
    for (const auto& sample : X_test) {
        Vector prediction = model.predict({sample});
        y_pred.push_back(prediction);
    }

    double score = model.evaluate(y_test,y_pred);
    std::cout<<"score  = "<<score<<std::endl;

    delete optimizer;
    return 0;
}
