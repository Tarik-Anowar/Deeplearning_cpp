#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <memory>
#include "lib/model.h"
#include "lib/optimizer.h"
#include "lib/activation.h"
#include "lib/loss.h"

using namespace Activation;
using namespace Loss;

using Matrix = std::vector<std::vector<double>>; 
using Vector = std::vector<double>;

Matrix read_csv(const std::string &filename)
{
    std::ifstream file(filename);
    std::string line;
    Matrix data;

    // Check if the file is open
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data; // Return empty matrix if file can't be opened
    }

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        // Read each value separated by commas in the row
        while (std::getline(ss, value, ','))
        {
            try
            {
                row.push_back(std::stod(value)); // Convert string to double
            }
            catch (const std::invalid_argument &e)
            {
                std::cerr << "Invalid data in CSV file: " << value << std::endl;
            }
        }

        // Add the row to the data matrix if it is not empty
        if (!row.empty())
        {
            data.push_back(row);
        }
    }

    return data;
}

void print_shape(const Matrix &matrix, const std::string &name)
{
    std::cout << name << " Shape: " << matrix.size() << " x "
              << (matrix.empty() ? 0 : matrix[0].size()) << std::endl;
}

int main()
{
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

    model.add_layer(64, 4, relu, relu_derivative);    // Input: 4, Output: 16 (Hidden Layer 1)
    model.add_layer(3, 64, softmax, softmax_derivative); // Input: 8, Output: 3 (Output Layer for 3-class classification)

    // Optimizer *optimizer = new AdamOptimizer(0.001, 0.9, 0.999, 1e-8);
    Optimizer *optimizer = new SGD(0.001);
    // Compile the model with the optimizer and loss function
    model.compile(optimizer, categorical_cross_entropy, categorical_cross_entropy_derivative);

    // Train the model (50 epochs, batch size = 32)
    std::cout << "Training the model..." << std::endl;
    model.fit(X_train, y_train, 100, 32);

    Matrix y_pred;

    std::cout << "Making predictions on test data..." << std::endl;
    for (const auto &sample : X_test)
    {
        Vector prediction = model.predict(sample); // Get prediction for each test sample
        y_pred.push_back(prediction);              // Store prediction
    }

    double score = model.evaluate(y_test, y_pred); 
    std::cout << "Model score: " << score << std::endl;
    return 0;
}
