#include <iomanip>
#include <iostream>
#include <string>

#include "animal/dataset.hpp"
#include "animal/regression.hpp"

int main(int argc, char** argv) {
    using namespace animal;

    const std::string data_path =
        (argc > 1) ? std::string(argv[1]) : std::string("data/polynomial_regression.csv");
    const Dataset dataset = load_xy_csv(data_path, 1);

    PolynomialRegression model(2);
    model.fit(dataset.x, dataset.y, 12000, 0.03);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Polynomial Regression (degree=2)\n";
    std::cout << "Dataset: " << data_path << "\n";
    std::cout << "w1 (x): " << model.coefficients()(0, 0) << "\n";
    std::cout << "w2 (x^2): " << model.coefficients()(1, 0) << "\n";
    std::cout << "intercept: " << model.intercept() << "\n";
    std::cout << "Training MSE: " << model.mse(dataset.x, dataset.y) << "\n";

    Matrix x_test = {{2.5}};
    Matrix y_pred = model.predict(x_test);
    std::cout << "Prediction for x=2.5: " << y_pred(0, 0) << "\n";

    return 0;
}
