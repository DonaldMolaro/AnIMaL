#include <cmath>
#include <vector>

#include "animal/regression.hpp"
#include "test_common.hpp"
#include "test_registry.hpp"

namespace {

void test_regression_training() {
    animal::LinearRegression lin;

    bool threw = false;
    try {
        (void)lin.predict(animal::Matrix{{1.0}});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    animal::test::require(threw, "predict before fit should throw");

    animal::Matrix x(20, 1);
    animal::Matrix y(20, 1);
    for (std::size_t i = 0; i < 20; ++i) {
        const double xv = static_cast<double>(i + 1);
        x(i, 0) = xv;
        y(i, 0) = 2.0 * xv + 1.0;
    }

    lin.fit(x, y, 6000, 0.01);
    animal::test::require(lin.is_fitted(), "linear model should be fitted");
    animal::test::require(std::fabs(lin.coefficients()(0, 0) - 2.0) < 0.05, "linear coefficient not close");
    animal::test::require(std::fabs(lin.intercept() - 1.0) < 0.2, "linear intercept not close");
    animal::test::require(lin.mse(x, y) < 0.01, "linear mse too high");

    animal::PolynomialRegression poly(2);
    animal::Matrix px(17, 1);
    animal::Matrix py(17, 1);
    for (int i = -8; i <= 8; ++i) {
        const std::size_t r = static_cast<std::size_t>(i + 8);
        const double xv = static_cast<double>(i) / 2.0;
        px(r, 0) = xv;
        py(r, 0) = 0.5 * xv * xv - 1.5 * xv + 2.0;
    }

    poly.fit(px, py, 18000, 0.02);
    animal::test::require(poly.is_fitted(), "poly model should be fitted");
    animal::test::require(std::fabs(poly.coefficients()(0, 0) + 1.5) < 0.05, "poly x coeff not close");
    animal::test::require(std::fabs(poly.coefficients()(1, 0) - 0.5) < 0.05, "poly x^2 coeff not close");
    animal::test::require(std::fabs(poly.intercept() - 2.0) < 0.1, "poly intercept not close");
    animal::test::require(poly.mse(px, py) < 1e-4, "poly mse too high");

    bool bad_degree = false;
    try {
        animal::PolynomialRegression invalid(0);
        (void)invalid;
    } catch (const std::runtime_error&) {
        bad_degree = true;
    }
    animal::test::require(bad_degree, "invalid polynomial degree should throw");
}

void test_regression_r_squared() {
    animal::LinearRegression lin;
    animal::Matrix x(20, 1);
    animal::Matrix y(20, 1);
    for (std::size_t i = 0; i < 20; ++i) {
        const double xv = static_cast<double>(i + 1);
        x(i, 0) = xv;
        y(i, 0) = 2.0 * xv + 1.0;
    }

    lin.fit(x, y, 6000, 0.01);
    const double r2 = lin.r_squared(x, y);
    animal::test::require(r2 > 0.99, "linear r_squared should be close to 1.0");
}

void test_regression_fit_history() {
    animal::LinearRegression lin;
    animal::Matrix x(10, 1);
    animal::Matrix y(10, 1);
    for (std::size_t i = 0; i < 10; ++i) {
        x(i, 0) = static_cast<double>(i);
        y(i, 0) = 3.0 * static_cast<double>(i) + 2.0;
    }

    lin.fit(x, y, 100, 0.01);
    const auto& history = lin.fit_history();
    animal::test::require(history.size() == 100, "fit_history should have 100 entries");
    animal::test::require(history.back() < history.front(), "fit_history should show decreasing loss");
}

void test_poly_r_squared() {
    animal::PolynomialRegression poly(2);
    animal::Matrix px(17, 1);
    animal::Matrix py(17, 1);
    for (int i = -8; i <= 8; ++i) {
        const std::size_t r = static_cast<std::size_t>(i + 8);
        const double xv = static_cast<double>(i) / 2.0;
        px(r, 0) = xv;
        py(r, 0) = 0.5 * xv * xv - 1.5 * xv + 2.0;
    }

    poly.fit(px, py, 18000, 0.02);
    const double r2 = poly.r_squared(px, py);
    animal::test::require(r2 > 0.999, "poly r_squared should be close to 1.0");

    const auto& history = poly.fit_history();
    animal::test::require(!history.empty(), "poly fit_history should not be empty");
}

}  // namespace

void register_regression_tests(std::vector<animal::test::TestCase>& tests) {
    tests.push_back({"regression_training", test_regression_training});
    tests.push_back({"regression_r_squared", test_regression_r_squared});
    tests.push_back({"regression_fit_history", test_regression_fit_history});
    tests.push_back({"poly_r_squared", test_poly_r_squared});
}
