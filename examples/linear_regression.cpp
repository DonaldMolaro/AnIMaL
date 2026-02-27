#include <iomanip>
#include <iostream>
#include <string>

#include "animal/dataset.hpp"
#include "animal/regression.hpp"

int main(int argc, char** argv) {
    using namespace animal;

    const std::string data_path =
        (argc > 1) ? std::string(argv[1]) : std::string("data/linear_regression.csv");
    const Dataset dataset = load_xy_csv(data_path, 1);

    LinearRegression model;
    model.fit(dataset.x, dataset.y, 6000, 0.01);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Linear Regression\n";
    std::cout << "Dataset: " << data_path << "\n";
    std::cout << "Learned slope: " << model.coefficients()(0, 0) << "\n";
    std::cout << "Learned intercept: " << model.intercept() << "\n";
    std::cout << "Training MSE: " << model.mse(dataset.x, dataset.y) << "\n";

    Matrix x_test = {{13.0}};
    Matrix y_pred = model.predict(x_test);
    std::cout << "Prediction for x=13: " << y_pred(0, 0) << "\n";

    return 0;
}
